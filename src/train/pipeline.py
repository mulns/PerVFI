import os.path as osp

import accelerate
import torch
from accelerate.logging import get_logger
from torch.optim import AdamW, lr_scheduler

from flow_estimators import build_flow_estimator
from generators import build_generator_arch
from loss import Loss

logger = get_logger("base")


def get_z(heat: float, img_size: tuple, batch: int, device: str):
    def calc_z_shapes(img_size, n_levels):
        h, w = img_size
        z_shapes = []
        channel = 3

        for _ in range(n_levels - 1):
            h //= 2
            w //= 2
            channel *= 2
            z_shapes.append((channel, h, w))
        h //= 2
        w //= 2
        z_shapes.append((channel * 4, h, w))
        return z_shapes

    z_list = []
    z_shapes = calc_z_shapes(img_size, 3)
    for z in z_shapes:
        z_new = torch.randn(batch, *z, device=device) * heat
        z_list.append(z_new)
    return z_list


class Pipeline_train:
    def __init__(self, model_version: str, acc: accelerate.Accelerator, **configs):
        ###### configs ######
        self.check_configs(configs)
        steps = configs.pop("steps", None)
        max_lr = configs.pop("max_lr", None)
        resume = configs.pop("resume", None)
        loss = configs.pop("loss", "nll").lower()

        ###### init acc, netG, loss, optim, lr ######
        self.acc = acc
        self.loss_type = loss
        model = build_generator_arch(model_version)
        trainable_params = [x for x in model.parameters() if x.requires_grad]
        optimG = AdamW(trainable_params)
        lrG = lr_scheduler.OneCycleLR(
            optimizer=optimG,
            max_lr=max_lr,
            total_steps=steps,
            anneal_strategy="cos",
        )

        self.netG, self.optimG, self.lrG, self.loss = self.acc.prepare(
            model, optimG, lrG, Loss(loss)
        )
        # self.netG = torch.compile(self.netG)

        ###### resume training ######
        if isinstance(resume, str) and osp.exists(resume):
            logger.info(f"Resumed from checkpoint: {resume}")
            self.acc.load_state(resume)

        ###### logging ######
        logger.info(
            "#Trainable params: {:,d}".format(
                sum([x.numel() for x in trainable_params])
            ),
            main_process_only=True,
        )

        logger.info(
            "#Total Params: {:,d}".format(
                sum(x.numel() for x in self.netG.parameters())
            ),
            main_process_only=True,
        )

    def check_configs(self, conf):
        # check the configurations
        assert "steps" in conf.keys()
        assert "max_lr" in conf.keys()
        if "resume" not in conf.keys():
            self.acc.print("Train from scratch, without resuming!")
        if "loss" not in conf.keys():
            self.acc.print("Use NLL loss only by default!")

    def train_one_iter(self, inputs):
        # input two reference frames and the intermediate frame, update the params
        self.netG.train()
        self.optimG.zero_grad()

        ############## calculate loss and grads ##############
        img0, img1, fflow, bflow, gt = inputs
        cond = [img0, img1, fflow, bflow]

        if "nll" not in self.loss_type.lower():  # Without NLL Loss
            pred, smasks = self.netG(inps=cond)
            with self.acc.autocast():
                loss, metric = self.loss(pred, gt)

        elif len(self.loss_type.split("+")) == 1:  # Only use NLL loss
            nll, _, smasks = self.netG(gt=gt, inps=cond, code="encode")
            loss, metric = nll.mean(), {"nll": nll.mean().detach().float()}

        else:  # Use NLL and aux_loss for training
            # zs = get_z(0.4, img0.shape[-2:], img0.shape[0], img0.device)
            nll, pred, smasks = self.netG(gt=gt, inps=cond, code="encode_decode")
            if not torch.isnan(pred).any():
                with self.acc.autocast():
                    loss, metric = self.loss(pred, gt)
            else:
                logger.info("NAN value during decoding...", main_process_only=True)
                loss, metric = 0.0, {}
            loss += nll.mean()
            metric.update({"nll": nll.mean().detach().float()})

        ############## backward propagation ##############
        self.acc.backward(loss)
        if self.acc.sync_gradients:
            self.acc.clip_grad_norm_(self.netG.parameters(), 0.1)
        self.optimG.step()
        self.lrG.step()

        metric.update({"all": loss.mean().detach().float()})
        return metric, smasks

    @torch.no_grad()
    def validate(self, inputs):
        self.netG.eval()
        img0, img1, fflow, bflow, gt = inputs
        cond = [img0, img1, fflow, bflow]
        if "nll" not in self.loss_type.lower():  # Without NLL Loss
            pred_random, _ = self.netG(inps=cond)
            pred_best = pred_random
        else:
            zs = get_z(0.3, img0.shape[-2:], img0.shape[0], img0.device)
            pred_random, _ = self.netG(zs=zs, inps=cond, code="decode")
            _, pred_best, _ = self.netG(gt=gt, inps=cond, code="encode_decode")

        pred_random, pred_best = [
            torch.clamp(x, 0, 1) for x in [pred_random, pred_best]
        ]
        return pred_random, pred_best

    def save_model(self, save_dir, name):
        self.acc.wait_for_everyone()
        unwrapped = self.acc.unwrap_model(self.netG)
        self.acc.save(unwrapped.state_dict(), osp.join(save_dir, name))


class Pipeline_infer:
    def __init__(self, flownet: str, generator: str, model_file: str):
        self.flownet, self.compute_flow = build_flow_estimator(flownet)
        self.flownet.to("cuda").eval()

        self.netG = build_generator_arch(generator)
        state_dict = {
            k.replace("module.", ""): v for k, v in torch.load(model_file).items()
        }
        self.netG.load_state_dict(state_dict)
        self.netG.to("cuda").eval()

    @torch.no_grad()
    def inference_rand_noise(self, img0, img1, heat=0.7, time=0.5, flows=None):
        zs = get_z(heat, img0.shape[-2:], img0.shape[0], img0.device)
        fflow, bflow = flows if flows is not None else self.compute_flow(img0, img1)

        conds = [img0, img1, fflow, bflow]
        pred, _ = self.netG(zs=zs, inps=conds, time=time, code="decode")
        return torch.clamp(pred, 0.0, 1.0)

    @torch.no_grad()
    def inference_best_noise(self, img0, img1, gt, time=0.5, flows=None):
        fflow, bflow = flows if flows is not None else self.compute_flow(img0, img1)
        conds = [img0, img1, fflow, bflow]
        _, pred, _ = self.netG(gt=gt, inps=conds, code="encode_decode", time=time)
        return torch.clamp(pred, 0.0, 1.0)

    @torch.no_grad()
    def inference_spec_noise(self, img0, img1, zs: list, time=0.5, flows=None):
        fflow, bflow = flows if flows is not None else self.compute_flow(img0, img1)
        conds = [img0, img1, fflow, bflow]
        pred, _ = self.netG(zs=zs, inps=conds, code="decode", time=time)
        return torch.clamp(pred, 0.0, 1.0)

    @torch.no_grad()
    def generate_masks(self, img0, img1, time=0.5):
        fflow, bflow = self.compute_flow(img0, img1)
        img0, img1 = [self.netG.normalize(x) for x in [img0, img1]]
        conds = [img0, img1, fflow, bflow]
        outs = self.netG.featurePyramid(img0, img1, [fflow, bflow], time)
        bmasks = outs[1]
        _, smasks = self.netG.get_cond(conds, time)
        return bmasks, smasks
