import argparse
import os
import os.path as osp
import random
import shutil
import sys
import time
import warnings
from distutils.util import strtobool
import torch.distributed as dist


import accelerate
import numpy as np
import torch
from accelerate.logging import get_logger
from easydict import EasyDict
from lpips import LPIPS
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from data.dataloader import OFVimeoTriplet
from pipeline import Pipeline_train
from utils import util
from utils.image_utils import calc_psnr, save_tensor

warnings.filterwarnings("ignore")


def train(ppl, train_data, val_data, args):
    ppl: Pipeline_train = ppl

    ############ configs ############
    step = args.start_step
    total_steps = args.total_steps
    save_interval = args.logger_cfg.save_interval
    eval_interval = args.logger_cfg.eval_interval
    pretrained = osp.join(EXP_DIR, "pre-trained")
    lpips_model = ACC.prepare(LPIPS(net="alex"))
    best_val_lpips = 100
    best_val_psnr = 0

    ############ logger ############
    if ACC.is_local_main_process:
        writer = SummaryWriter(LOG_DIR + "/train")
        writer_val = SummaryWriter(LOG_DIR + "/validate")
    else:
        writer = None
        writer_val = None

    ############ begin training ############
    times = []
    while step <= total_steps:
        for batch, _ in train_data:
            # forward & backward
            start_time = time.time()
            metric, smasks = ppl.train_one_iter(batch)
            duration = time.time() - start_time
            times.append(duration)

            def eta(t_iter):
                return (t_iter * (total_steps - step)) / 3600

            ACC.wait_for_everyone()
            # print message during training
            if ACC.is_local_main_process and (step <= 25 or step % 100 == 0):
                for k, v in metric.items():
                    writer.add_scalar(k, v, step)
                for i, m in enumerate(smasks):
                    if not torch.isnan(m).any():
                        writer.add_histogram("softbinary_mask_LV%d" % i, m, step)
                writer.flush()
                avg_time = sum(times) / len(times)
                message = "<iter:{:6d}/{}, eta:{:<3.1f}h>".format(
                    step, total_steps, eta(avg_time)
                )
                for k, v in metric.items():
                    message += f"<{k:s}:{v:3.3f}>"
                logger.info(message, main_process_only=True)
                # save checkpoints
                if step % save_interval == 0:
                    ACC.save_state()
                times = []

            # evaluation
            if step % eval_interval == 0:
                psnr, lpips = evaluate(ppl, val_data, step, writer_val, lpips_model)
                ACC.wait_for_everyone()
                if lpips < best_val_lpips:
                    best_val_lpips = lpips
                    ppl.save_model(pretrained, "best-lpips.pth")
                    logger.info(
                        "New best LPIPS: {}".format(best_val_lpips),
                        main_process_only=True,
                    )
                if psnr > best_val_psnr:
                    best_val_psnr = psnr
                    ppl.save_model(pretrained, "best-psnr.pth")
                    logger.info(
                        "New best PSNR: {}".format(best_val_psnr),
                        main_process_only=True,
                    )

            step += 1


@torch.no_grad()
def evaluate(ppl, val_data, step, writer_val, lpips_model=None):
    psnr_list = []
    lpips_list = []
    best_psnr_list = []
    best_lpips_list = []
    white_list = [
        "00023/0003",
        "00023/0656",
        "00023/0664",
        "00041/0148",
        "00041/0152",
        "00041/0160",
    ]
    for batch, index in tqdm(val_data, disable=not ACC.is_local_main_process):
        pred_random, pred_best = ppl.validate(batch)
        gt = batch[-1]
        # pred_random, pred_best, gt, = ACC.gather_for_metrics(
        #     [pred_random, pred_best, batch[-1]]
        # )
        with ACC.autocast():
            psnr_list.append(calc_psnr(pred_random, gt).flatten())
            best_psnr_list.append(calc_psnr(pred_best, gt).flatten())
        lpips_list.append(lpips_model(pred_random, gt, normalize=True).flatten())
        best_lpips_list.append(lpips_model(pred_best, gt, normalize=True).flatten())
        for j in range(gt.shape[0]):
            this_idx = index[j]
            if this_idx in white_list:  # save samples for visualizaiton
                save_dir = osp.join(EXP_DIR, f"val-images/{this_idx:s}")
                os.makedirs(save_dir, exist_ok=True)
                save_tensor(
                    torch.nan_to_num(pred_random[j], 0, 1, 0),
                    osp.join(save_dir, f"pred-rand-{step:06d}.png"),
                )
                save_tensor(
                    torch.nan_to_num(pred_best[j], 0, 1, 0),
                    osp.join(save_dir, f"pred-best-{step:06d}.png"),
                )
                if not osp.exists(osp.join(save_dir, "gt.png")):
                    save_tensor(batch[-1][j], osp.join(save_dir, "gt.png"))
                # remove redundant imgs
                saved = [x for x in os.listdir(save_dir) if "pred-rand" in x]
                if len(saved) >= 52 and ACC.is_local_main_process:
                    print(f"saved images: {len(saved)}")
                    for x in saved:
                        step_ = int(x.split("-")[-1].split(".")[0])
                        if not (step_ % 4000 == 0):
                            os.remove(osp.join(save_dir, x))
                            os.remove(osp.join(save_dir, x.replace("rand", "best")))
    psnr = torch.cat(psnr_list)
    lpips = torch.cat(lpips_list)
    psnr_best = torch.cat(best_psnr_list)
    lpips_best = torch.cat(best_lpips_list)
    psnr, lpips, psnr_best, lpips_best = [
        x.mean().data
        for x in ACC.gather_for_metrics([psnr, lpips, psnr_best, lpips_best])
    ]

    if ACC.is_local_main_process:
        writer_val.add_scalar("rand-PSNR", psnr, step)
        writer_val.add_scalar("rand-LPIPS", lpips, step)
        writer_val.add_scalar("best-PSNR", psnr_best, step)
        writer_val.add_scalar("best-LPIPS", lpips_best, step)
        logger.info(
            f"==> validate @ step: {step:05d}; psnr: {psnr:.2f} | {psnr_best:.2f}; lpips: {lpips:.4f} : {lpips_best:.4f}<==",
            main_process_only=True,
        )

    return psnr, lpips


@accelerate.state.PartialState().on_local_main_process
def init_exp_env():
    def prompt(query):
        sys.stdout.write("%s [y/n]:" % query)
        val = input()

        try:
            ret = strtobool(val)
        except ValueError:
            sys.stdout("please answer with y/n")
            return prompt(query)
        return ret

    # process the path
    # init train log dir, model and tf dir
    if os.path.exists(EXP_DIR) and (not RESUME):
        while True:
            if (
                prompt(
                    "Would you like to re-write the existing experimental saving dir?"
                )
                == True
            ):
                shutil.rmtree(EXP_DIR)
                break
            else:
                print("Exit the program. Please assign another expriment name!")
                exit()

    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(osp.join(EXP_DIR, "pre-trained"), exist_ok=True)
    os.makedirs(os.path.join(EXP_DIR, "val-images"), exist_ok=True)

    # set logger file
    util.setup_logger(
        "base", os.path.join(EXP_DIR, "runtime.log"), screen=True, tofile=True
    )

    # init cuda env
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run(args):
    # **********************************************************#
    # => init the dataset
    batch_size = args.dataset_cfg.get("batch_size")
    logger.info("batch size in total: %d" % batch_size, main_process_only=True)
    nb_data_worker = args.dataset_cfg.get("nb_data_worker", 8)
    data_root = args.dataset_cfg["data_root"]
    of_method = args.dataset_cfg["of_method"]  # optical flow dir
    dataset_train = OFVimeoTriplet("train", data_root=data_root, of_method=of_method)
    dataset_val = OFVimeoTriplet("valid", data_root=data_root, of_method=of_method[0])
    train_data = DataLoader(
        dataset_train,
        batch_size=batch_size,
        num_workers=nb_data_worker,
        pin_memory=True,
        drop_last=True,
    )
    val_data = DataLoader(
        dataset_val,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    train_data, val_data = ACC.prepare(train_data, val_data)

    # **********************************************************#
    # => init the training Pipeline
    total_epochs = args.optimizer_cfg.epochs
    total_steps = total_epochs * len(train_data)
    args.total_steps = total_steps
    ppl = Pipeline_train(
        args.model_cfg.version,
        acc=ACC,
        steps=total_steps,
        max_lr=args.optimizer_cfg.max_lr,
        loss=args.optimizer_cfg.loss,
        resume=RESUME,
    )
    logger.info("Start the training task ...", main_process_only=True)

    if RESUME:
        train_data = ACC.skip_first_batches(
            train_data, args.start_step % len(train_data)
        )
    train(ppl, train_data, val_data, args)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    # **********************************************************#
    # => parse args and init the training environment global variable
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", "-c", type=str, default="./configs/pvfi-nll.yml")
    parser.add_argument("--resume", "-r", type=str, default=False)
    args_ = parser.parse_args()
    args = util.parse_options(args_.conf)
    args = EasyDict(dict(args))

    RESUME = args_.resume
    EXP_DIR = osp.join(args.logger_cfg.root, args.exp_name)
    LOG_DIR = osp.join(EXP_DIR, "tensorboard")

    init_exp_env()
    resume_flag = 0
    if RESUME:
        if RESUME == "latest":
            all_ckpts = sorted(
                os.listdir(osp.join(EXP_DIR, "checkpoints")),
                key=lambda x: int(x.split("_")[-1]),
            )
            RESUME = osp.join(EXP_DIR, "checkpoints", all_ckpts[-1])
        assert osp.exists(RESUME), "Resuming directory do not exist"
        resume_flag = int(
            osp.splitext(osp.basename(RESUME))[0].replace("checkpoint_", "")
        )  # state is saved every 100 steps
        args.start_step = resume_flag * args.logger_cfg.save_interval
    else:
        args.start_step = 1

    project_config = accelerate.utils.ProjectConfiguration(
        project_dir=EXP_DIR,
        automatic_checkpoint_naming=True,
        total_limit=9,
        iteration=resume_flag + 1,
    )
    ACC = accelerate.Accelerator(
        device_placement=True,
        project_config=project_config,
        split_batches=True,
    )
    DEVICE = ACC.device
    logger = get_logger("base")
    run(args)
