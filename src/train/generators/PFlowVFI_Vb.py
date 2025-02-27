"""PerVFI: Soft-binary Blending for Photo-realistic Video Frame Interpolation

"""
import accelerate
import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from torchvision.ops import DeformConv2d

from . import thops
from .msfusion import MultiscaleFuse
from .normalizing_flow import *
from .softsplatnet import Basic, Encode, Softmetric
from .softsplatnet.softsplat import softsplat


def resize(x, size: tuple, scale: bool):
    H, W = x.shape[-2:]
    h, w = size
    scale_ = h / H
    x_ = F.interpolate(x, size, mode="bilinear", align_corners=False)
    if scale:
        return x_ * scale_
    return x_


def binary_hole(flow):
    n, _, h, w = flow.shape
    mask = softsplat(
        tenIn=torch.ones((n, 1, h, w), device=flow.device),
        tenFlow=flow,
        tenMetric=None,
        strMode="avg",
    )
    ones = torch.ones_like(mask, device=mask.device)
    zeros = torch.zeros_like(mask, device=mask.device)
    out = torch.where(mask <= 0.5, ones, zeros)
    return out


def warp_pyramid(features: list, metric, flow):
    outputs = []
    masks = []
    for lv in range(3):
        fea = features[lv]
        if lv != 0:
            h, w = fea.shape[-2:]
            metric = resize(metric, (h, w), scale=False)
            flow = resize(flow, (h, w), scale=True)
        outputs.append(softsplat(fea, flow, metric.neg().clip(-20.0, 20.0), "soft"))
        masks.append(binary_hole(flow))
    return outputs, masks


class FeaturePyramid(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.netEncode = Encode()
        self.netSoftmetric = Softmetric()
        ckpt = "checkpoints/network-lf.pytorch"
        state_dict = {
            k.replace("moduleSynthesis.", "").replace("module", "net"): v
            for k, v in torch.load(ckpt).items()
            if "moduleSynthesis" in k
        }
        self.load_state_dict(state_dict, strict=False)

    def forward(
        self,
        tenOne,
        tenTwo=None,
        tenFlows: list[Tensor] = None,
        time: float = 0.5,
    ):
        x1s = self.netEncode(tenOne)
        if tenTwo is None:  # just encode
            return x1s
        F12, F21 = tenFlows
        x2s = self.netEncode(tenTwo)
        m1t = self.netSoftmetric(x1s, x2s, F12) * 2 * time
        F1t = time * F12
        m2t = self.netSoftmetric(x2s, x1s, F21) * 2 * (1 - time)
        F2t = (1 - time) * F21
        Ft2 = -1 * softsplat(F2t, F2t, m2t.neg().clip(-20.0, 20.0), "soft")
        x1s, bmasks = warp_pyramid(x1s, m1t, F1t)
        return list(zip(x1s, x2s)), bmasks, Ft2


class SoftBinary(torch.nn.Module):
    def __init__(self, cin, dilate_size=7) -> None:
        super().__init__()
        channel = 64
        reduction = 8
        self.conv1 = torch.nn.Sequential(
            *[
                torch.nn.Conv2d(1, channel, dilate_size, 1, padding="same", bias=False),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(channel, channel, 3, 1, padding="same", bias=False),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(channel, channel, 1, 1, padding="same", bias=False),
            ]
        )
        self.att = torch.nn.Conv2d(cin * 2, channel, 3, 1, padding="same")
        self.avg = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channel, channel // reduction, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(channel // reduction, channel, bias=False),
            torch.nn.Sigmoid(),
        )
        self.conv2 = torch.nn.Conv2d(channel, 1, 1, 1, padding="same", bias=False)

    def forward(self, bmask, feaL, feaR):  # N,1,H,W
        m_fea = self.conv1(bmask)
        x = self.att(torch.cat([feaL, feaR], dim=1))
        b, c, _, _ = x.size()
        x = self.avg(x).view(b, c)
        x = self.fc(x).view(b, c, 1, 1)
        x = m_fea * x.expand_as(x)
        x = self.conv2(x)

        x = torch.tanh(torch.abs(x))
        rand_bias = (torch.rand_like(x, device=x.device) - 0.5) / 100.0
        if self.training:
            return x + rand_bias
        else:
            return x


class DCNPack(torch.nn.Module):
    def __init__(self, cin, groups, dksize):
        super().__init__()
        cout = groups * 3 * dksize**2
        self.conv_offset = torch.nn.Conv2d(cin, cout, 3, 1, 1)
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()
        self.dconv = DeformConv2d(cin, cin, dksize, padding=dksize // 2)

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger.info(f"Offset abs mean is {offset_absmean}, larger than 50.")

        return self.dconv(x, offset, mask)


class DeformableAlign(torch.nn.Module):
    def __init__(self):
        super().__init__()
        channels = [35, 64, 96]
        self.offset_conv1 = torch.nn.ModuleDict()
        self.offset_conv2 = torch.nn.ModuleDict()
        self.offset_conv3 = torch.nn.ModuleDict()
        self.deform_conv = torch.nn.ModuleDict()
        self.feat_conv = torch.nn.ModuleDict()
        self.merge_conv1 = torch.nn.ModuleDict()
        self.merge_conv2 = torch.nn.ModuleDict()
        # Pyramids
        for i in range(2, -1, -1):
            level = f"l{i}"
            c = channels[i]
            # compute offsets
            self.offset_conv1[level] = torch.nn.Conv2d(c * 2 + 3, c, 3, 1, 1)
            if i == 2:
                self.offset_conv2[level] = torch.nn.Conv2d(c, c, 3, 1, 1)
            else:
                self.offset_conv2[level] = torch.nn.Conv2d(
                    c + channels[i + 1], c, 3, 1, 1
                )
                self.offset_conv3[level] = torch.nn.Conv2d(c, c, 3, 1, 1)
            # apply deform conv
            if i == 0:
                self.deform_conv[level] = DCNPack(c, 7, 3)
            else:
                self.deform_conv[level] = DCNPack(c, 8, 3)
            self.merge_conv1[level] = torch.nn.Conv2d(c + c + 1, c, 3, 1, 1)
            if i < 2:
                self.feat_conv[level] = torch.nn.Conv2d(c + channels[i + 1], c, 3, 1, 1)
                self.merge_conv2[level] = torch.nn.Conv2d(
                    c + channels[i + 1], c, 3, 1, 1
                )

        self.upsample = torch.nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, features, bmasks, Ft2):
        outs = []

        for i in range(2, -1, -1):
            level = f"l{i}"
            feaL, feaR = features[i]
            bmask = bmasks[i]
            flow = resize(Ft2, bmask.shape[2:], scale=True)
            offset = torch.cat([feaL, feaR, bmask, flow], dim=1)
            offset = self.lrelu(self.offset_conv1[level](offset))
            if i == 2:
                offset = self.lrelu(self.offset_conv2[level](offset))
            else:
                offset = self.lrelu(
                    self.offset_conv2[level](
                        torch.cat([offset, upsampled_offset], dim=1)
                    )
                )
                offset = self.lrelu(self.offset_conv3[level](offset))

            warped_feaR = self.deform_conv[level](feaR, offset)

            if i < 2:
                warped_feaR = self.feat_conv[level](
                    torch.cat([warped_feaR, upsampled_feaR], dim=1)
                )

            merged_feat = self.merge_conv1[level](
                torch.cat([feaL, warped_feaR, bmask], dim=1)
            )
            if i < 2:
                merged_feat = self.merge_conv2[level](
                    torch.cat([merged_feat, upsampled_merged_feat], dim=1)
                )
            outs.append(merged_feat)

            if i > 0:  # upsample offset and features
                warped_feaR = self.lrelu(warped_feaR)
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feaR = self.upsample(warped_feaR)
                upsampled_merged_feat = self.upsample(merged_feat)

        return outs


class AttentionMerge(torch.nn.Module):
    def __init__(self, dilate_size=7):
        super().__init__()
        self.softbinary = torch.nn.ModuleDict()
        channels = [35, 64, 96]
        for i in range(2, -1, -1):
            level = f"{i}"
            c = channels[i]
            self.softbinary[level] = SoftBinary(c, dilate_size)

    def forward(self, feaL, feaR, bmask):
        outs = []
        soft_masks = []
        for i in range(2, -1, -1):
            level = f"{i}"
            sm = self.softbinary[level](bmask[i], feaL[i], feaR[i])
            soft_masks.append(sm)
            x = feaL[i] * (1 - sm) + feaR[i] * sm
            outs.append(x)
        return outs, soft_masks


class Decoder(torch.nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.conv3 = Basic("conv-relu-conv", [cin[2], 128, cin[1]], True)
        self.conv2 = Basic("conv-relu-conv", [cin[1] * 2, 256, cin[0]], True)
        self.conv1 = Basic("conv-relu-conv", [cin[0] * 2, 96, 128], True)
        self.tail = torch.nn.Conv2d(128 // 4, 3, 3, 1, padding="same")
        self.up = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        logger.info(
            f"Parameter of decoder: {sum(p.numel() for p in self.parameters())}"
        )

    def forward(self, xs):
        lv1, lv2, lv3 = xs
        lv3 = self.conv3(lv3)
        lv3 = self.up(lv3)
        lv2 = self.conv2(torch.cat([lv3, lv2], dim=1))
        lv2 = self.up(lv2)
        lv1 = self.conv1(torch.cat([lv2, lv1], dim=1))
        lv1 = unsqueeze2d(lv1, factor=2)
        return self.tail(lv1)


class Network(torch.torch.nn.Module):
    def __init__(self, dilate_size=9):
        super().__init__()
        cond_c = [35, 64, 96]
        self.featurePyramid = FeaturePyramid()
        self.deformableAlign = DeformableAlign()
        self.attentionMerge = AttentionMerge(dilate_size=dilate_size)
        self.multiscaleFuse = MultiscaleFuse(cond_c)
        self.generator = Decoder(cond_c)
        # self.condFLownet = CondFlowNet(cond_c, with_bn=False, train_1x1=True)

    def get_cond(self, inps: list, time: float = 0.5):
        tenOne, tenTwo, fflow, bflow = inps
        with accelerate.Accelerator().autocast():
            feas, bmasks, Ft2 = self.featurePyramid(
                tenOne, tenTwo, [fflow, bflow], time
            )
            feaR = self.deformableAlign(feas, bmasks, Ft2)[::-1]
            feaL = [feas[i][0] for i in range(3)]
            feas, smasks = self.attentionMerge(feaL, feaR, bmasks)
            # feas = [F.interpolate(x, scale_factor=0.5, mode="bilinear") for x in feas]
            feas = self.multiscaleFuse(feas[::-1])  # downscale by 2
        return feas, smasks

    def normalize(self, x, reverse=False):
        # x in [0, 1]
        if not reverse:
            return x * 2 - 1
        else:
            return (x + 1) / 2

    def forward(self, inps=[], time=0.5):
        img0, img1 = [self.normalize(x) for x in inps[:2]]
        cond = [img0, img1] + inps[-2:]

        conds, smasks = self.get_cond(cond, time=time)
        with accelerate.Accelerator().autocast():
            pred = self.generator(conds)
        pred = self.normalize(pred, reverse=True)
        return pred, smasks

    @classmethod
    def demo(cls):
        import cv2

        from utils.image_utils import readFlowKITTI

        model = cls().cuda()

        def totensor(x):
            return torch.from_numpy(x).permute(2, 0, 1).cuda()[None] / 255.0

        def save(x, pth):
            x = torch.clamp(x, 0.0, 1.0)
            x = x[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0
            x = np.round(x).astype("uint8")
            cv2.imwrite(pth, x[..., ::-1])

        im1 = "data/datasets/vimeo_triplet/sequences/00025/0001/im1.png"
        im2 = "data/datasets/vimeo_triplet/sequences/00025/0001/im3.png"
        lab = "data/datasets/vimeo_triplet/sequences/00025/0001/im2.png"
        fflow = "data/datasets/vimeo_triplet/optical_flows/gmflow/00025/0001/fflow.png"
        bflow = "data/datasets/vimeo_triplet/optical_flows/gmflow/00025/0001/bflow.png"
        im1 = totensor(cv2.imread(im1)[..., ::-1].copy())
        im2 = totensor(cv2.imread(im2)[..., ::-1].copy())
        lab = totensor(cv2.imread(lab)[..., ::-1].copy())
        fflow = torch.from_numpy(readFlowKITTI(fflow)).permute(2, 0, 1).cuda()[None]
        bflow = torch.from_numpy(readFlowKITTI(bflow)).permute(2, 0, 1).cuda()[None]

        inps = [im1, im2, fflow, bflow]
        with torch.no_grad():
            out, _ = model(inps=inps)
        print(torch.max(out), torch.min(out))
        print(torch.max(lab), torch.min(lab))
        mse = torch.mean((out - lab) ** 2)
        print("MSE: ", mse.item())
        # save(lab, "lab.png")
        # save(out, "out.png")
