import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Blank(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return 0.0


class EPE(nn.Module):
    def __init__(self):
        super(EPE, self).__init__()

    def forward(self, flow, gt, loss_mask):
        loss_map = (flow - gt.detach()) ** 2
        loss_map = (loss_map.sum(1, True) + 1e-6) ** 0.5
        if loss_mask is not None:
            return loss_map * loss_mask
        else:
            return loss_map


class Ternary(nn.Module):
    def __init__(self):
        super(Ternary, self).__init__()
        patch_size = 7
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float().cuda()

    def transform(self, img):
        patches = F.conv2d(img, self.w, padding=3, bias=None)
        transf = patches - img
        transf_norm = transf / torch.sqrt(0.81 + transf**2)
        return transf_norm

    def rgb2gray(self, rgb):
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def hamming(self, t1, t2):
        dist = (t1 - t2) ** 2
        dist_norm = torch.mean(dist / (0.1 + dist), 1, True)
        return dist_norm

    def valid_mask(self, t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def forward(self, img0, img1):
        img0 = self.transform(self.rgb2gray(img0))
        img1 = self.transform(self.rgb2gray(img1))
        return self.hamming(img0, img1) * self.valid_mask(img0, 1)


class SOBEL(nn.Module):
    def __init__(self):
        super(SOBEL, self).__init__()
        self.kernelX = torch.tensor(
            [
                [1, 0, -1],
                [2, 0, -2],
                [1, 0, -1],
            ]
        ).float()
        self.kernelY = self.kernelX.clone().T
        self.kernelX = self.kernelX.unsqueeze(0).unsqueeze(0).cuda()
        self.kernelY = self.kernelY.unsqueeze(0).unsqueeze(0).cuda()

    def forward(self, pred, gt):
        N, C, H, W = pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3]
        img_stack = torch.cat(
            [pred.reshape(N * C, 1, H, W), gt.reshape(N * C, 1, H, W)], 0
        )
        sobel_stack_x = F.conv2d(img_stack, self.kernelX, padding=1)
        sobel_stack_y = F.conv2d(img_stack, self.kernelY, padding=1)
        pred_X, gt_X = sobel_stack_x[: N * C], sobel_stack_x[N * C :]
        pred_Y, gt_Y = sobel_stack_y[: N * C], sobel_stack_y[N * C :]

        L1X, L1Y = torch.abs(pred_X - gt_X), torch.abs(pred_Y - gt_Y)
        loss = L1X + L1Y
        return loss


class LapLoss(torch.nn.Module):
    @staticmethod
    def gauss_kernel(size=5, channels=3):
        kernel = torch.tensor(
            [
                [1.0, 4.0, 6.0, 4.0, 1],
                [4.0, 16.0, 24.0, 16.0, 4.0],
                [6.0, 24.0, 36.0, 24.0, 6.0],
                [4.0, 16.0, 24.0, 16.0, 4.0],
                [1.0, 4.0, 6.0, 4.0, 1.0],
            ]
        )
        kernel /= 256.0
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel
        return kernel

    @staticmethod
    def laplacian_pyramid(img, kernel, max_levels=3):
        def downsample(x):
            return x[:, :, ::2, ::2]

        def upsample(x):
            cc = torch.cat(
                [
                    x,
                    torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).to(
                        x.device
                    ),
                ],
                dim=3,
            )
            cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
            cc = cc.permute(0, 1, 3, 2)
            cc = torch.cat(
                [
                    cc,
                    torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2).to(
                        x.device
                    ),
                ],
                dim=3,
            )
            cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
            x_up = cc.permute(0, 1, 3, 2)
            return conv_gauss(
                x_up, 4 * LapLoss.gauss_kernel(channels=x.shape[1]).to(x.device)
            )

        def conv_gauss(img, kernel):
            img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode="reflect")
            out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
            return out

        current = img
        pyr = []
        for level in range(max_levels):
            filtered = conv_gauss(current, kernel)
            down = downsample(filtered)
            up = upsample(down)
            diff = current - up
            pyr.append(diff)
            current = down
        return pyr

    def __init__(self, max_levels=5, channels=3):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = LapLoss.gauss_kernel(channels=channels)

    def forward(self, input, target):
        pyr_input = LapLoss.laplacian_pyramid(
            img=input,
            kernel=self.gauss_kernel.to(input.device),
            max_levels=self.max_levels,
        )
        pyr_target = LapLoss.laplacian_pyramid(
            img=target,
            kernel=self.gauss_kernel.to(target.device),
            max_levels=self.max_levels,
        )
        loss = sum(F.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))

        return loss / self.max_levels


class Charbonnier(nn.Module):
    def __init__(self, epsilon=0.001):
        super(Charbonnier, self).__init__()
        self.epsilon = epsilon

    def forward(self, output, gt):
        return torch.mean(torch.sqrt((output - gt) ** 2 + self.epsilon**2))


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, rank=0):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        pretrained = True

        class MeanShift(nn.Conv2d):
            def __init__(self, data_mean, data_std, data_range=1, norm=True):
                c = len(data_mean)
                super(MeanShift, self).__init__(c, c, kernel_size=1)
                std = torch.Tensor(data_std)
                self.weight.data = torch.eye(c).view(c, c, 1, 1)
                if norm:
                    self.weight.data.div_(std.view(c, 1, 1, 1))
                    self.bias.data = -1 * data_range * torch.Tensor(data_mean)
                    self.bias.data.div_(std)
                else:
                    self.weight.data.mul_(std.view(c, 1, 1, 1))
                    self.bias.data = data_range * torch.Tensor(data_mean)
                self.requires_grad = False

        self.vgg_pretrained_features = models.vgg19(pretrained=pretrained).features
        self.normalize = MeanShift(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True
        ).cuda()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X, Y, indices=None):
        X = self.normalize(X)
        Y = self.normalize(Y)
        indices = [2, 7, 12, 21, 30]
        weights = [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10 / 1.5]
        k = 0
        loss = 0
        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            Y = self.vgg_pretrained_features[i](Y)
            if (i + 1) in indices:
                loss += weights[k] * (X - Y.detach()).abs().mean() * 0.1
                k += 1
        return loss


class Loss(nn.modules.loss._Loss):
    LOSSES = dict(
        l1=nn.L1Loss,
        l2=nn.MSELoss,
        char=Charbonnier,
        per=VGGPerceptualLoss,
        lap=LapLoss,
        nll=Blank,
    )

    def __init__(self, str_loss):
        super(Loss, self).__init__()

        self.loss = []
        self.loss_module = nn.ModuleList()

        for loss in str_loss.split("+"):
            loss_function = None
            if len(loss.split("*")) == 1:
                weight, loss_type = "1", loss
            else:
                weight, loss_type = loss.split("*")
            loss_function = self.LOSSES[loss_type]()

            self.loss.append(
                {"type": loss_type, "weight": float(weight), "function": loss_function}
            )

        for l in self.loss:
            if l["function"] is not None:
                # logger.info('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l["function"])

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, output, gt):
        losses = []
        metric = {}
        for l in self.loss:
            loss = l["function"](output, gt)
            effective_loss = l["weight"] * loss
            losses.append(effective_loss)
            loss = loss.mean().detach().data if not isinstance(loss, float) else loss
            metric[l["type"]] = loss
        loss_sum = sum(losses).mean()
        return loss_sum, metric
