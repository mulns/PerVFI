import cv2
import numpy as np
import torch


def writeFlowKITTI(filename, uv):
    uv = 128.0 * uv + 2**15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])


def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    flow = flow[:, :, ::-1].astype(np.float32)
    flow = flow[:, :, :2]
    flow = (flow - 2**15) / 128.0
    return flow


def save_tensor(ten, pth):
    # ten: [C H W] on GPU
    im = torch.clamp(ten * 255.0, 0, 255).cpu().byte().permute(1, 2, 0).numpy()
    cv2.imwrite(pth, im[..., ::-1])


def read_tensor(pth, device):
    im = cv2.imread(pth)[..., ::-1].copy()
    ten = torch.from_numpy(im).permute(2, 0, 1)[None].float().to(device)
    return ten / 255.0


def calc_psnr(pred, gt, data_range=1.0, size_average=False):
    diff = (pred - gt).div(data_range)
    mse = diff.pow(2).mean(dim=(-3, -2, -1))
    psnr = -10 * torch.log10(mse + 1e-8)
    if size_average:
        return torch.mean(psnr)
    else:
        return psnr


def backwarp(tenIn, tenFlow):
    tenHor = (
        torch.linspace(
            start=-1.0,
            end=1.0,
            steps=tenFlow.shape[3],
            dtype=tenFlow.dtype,
            device=tenFlow.device,
        )
        .view(1, 1, 1, -1)
        .repeat(1, 1, tenFlow.shape[2], 1)
    )
    tenVer = (
        torch.linspace(
            start=-1.0,
            end=1.0,
            steps=tenFlow.shape[2],
            dtype=tenFlow.dtype,
            device=tenFlow.device,
        )
        .view(1, 1, -1, 1)
        .repeat(1, 1, 1, tenFlow.shape[3])
    )

    backwarp_tenGrid = torch.cat([tenHor, tenVer], 1).cuda()

    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenIn.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenIn.shape[2] - 1.0) / 2.0),
        ],
        1,
    )

    return torch.nn.functional.grid_sample(
        input=tenIn,
        grid=(backwarp_tenGrid + tenFlow).permute(0, 2, 3, 1),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


class InputPadder:
    """Pads images such that dimensions are divisible by factor"""

    def __init__(self, size, divide=8, mode="center"):
        self.ht, self.wd = size[-2:]
        pad_ht = (((self.ht // divide) + 1) * divide - self.ht) % divide
        pad_wd = (((self.wd // divide) + 1) * divide - self.wd) % divide
        if mode == "center":
            self._pad = [
                pad_wd // 2,
                pad_wd - pad_wd // 2,
                pad_ht // 2,
                pad_ht - pad_ht // 2,
            ]
        else:
            self._pad = [0, pad_wd, 0, pad_ht]

    def _pad_(self, x):
        return torch.nn.functional.pad(x, self._pad, mode="constant")

    def pad(self, *inputs):
        return [self._pad_(x) for x in inputs]

    def _unpad_(self, x):
        return x[
            ...,
            self._pad[2] : self.ht + self._pad[2],
            self._pad[0] : self.wd + self._pad[0],
        ]

    def unpad(self, *inputs):
        return [self._unpad_(x) for x in inputs]
