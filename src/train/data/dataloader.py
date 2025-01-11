import os
import random
from glob import glob

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.image_utils import readFlowKITTI


class SnuFilm(Dataset):
    def __init__(self, data_root, data_type="extreme", batch_size=16):
        self.batch_size = batch_size
        self.data_root = data_root
        self.data_type = data_type
        assert data_type in ["easy", "medium", "hard", "extreme"]
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        if self.data_type == "easy":
            easy_file = os.path.join(self.data_root, "eval_modes/test-easy.txt")
            with open(easy_file, "r") as f:
                self.meta_data = f.read().splitlines()
        if self.data_type == "medium":
            medium_file = os.path.join(self.data_root, "eval_modes/test-medium.txt")
            with open(medium_file, "r") as f:
                self.meta_data = f.read().splitlines()
        if self.data_type == "hard":
            hard_file = os.path.join(self.data_root, "eval_modes/test-hard.txt")
            with open(hard_file, "r") as f:
                self.meta_data = f.read().splitlines()
        if self.data_type == "extreme":
            extreme_file = os.path.join(self.data_root, "eval_modes/test-extreme.txt")
            with open(extreme_file, "r") as f:
                self.meta_data = f.read().splitlines()

    def getimg(self, index):
        imgpath = self.meta_data[index]
        imgpaths = imgpath.split()

        # Load images
        img0 = cv2.imread(os.path.join(self.data_root, imgpaths[0]))
        gt = cv2.imread(os.path.join(self.data_root, imgpaths[1]))
        img1 = cv2.imread(os.path.join(self.data_root, imgpaths[2]))

        return img0, gt, img1

    def __getitem__(self, index):
        img0, gt, img1 = self.getimg(index)
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return torch.cat((img0, img1, gt), 0)


class UCF101(Dataset):
    def __init__(self, data_root, batch_size=16):
        self.batch_size = batch_size
        self.data_root = data_root
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        triplet_dirs = glob(os.path.join(self.data_root, "*"))
        self.meta_data = triplet_dirs

    def getimg(self, index):
        imgpath = self.meta_data[index]
        imgpaths = [
            os.path.join(imgpath, "frame_00.png"),
            os.path.join(imgpath, "frame_01_gt.png"),
            os.path.join(imgpath, "frame_02.png"),
        ]

        # Load images
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        return img0, gt, img1

    def __getitem__(self, index):
        img0, gt, img1 = self.getimg(index)
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return torch.cat((img0, img1, gt), 0)


class VimeoTriplet(Dataset):
    def __init__(self, data_name, data_root, nb_sample=None):
        self.data_root = data_root
        self.image_root = os.path.join(self.data_root, "sequences")
        self.training = data_name == "train"
        self.crop_size = (256, 256)
        train_fn = os.path.join(self.data_root, "tri_trainlist.txt")
        test_fn = os.path.join(self.data_root, "tri_testlist.txt")
        if self.training:
            with open(train_fn, "r") as f:
                self.datalist = f.read().splitlines()
        else:
            with open(test_fn, "r") as f:
                self.datalist = f.read().splitlines()
            if nb_sample is not None:
                self.datalist = self.datalist[:nb_sample]
        if self.datalist[-1] == "":
            self.datalist.pop(-1)

        if self.training:
            self.transforms = transforms.Compose(
                [
                    transforms.RandomCrop((256, 256)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        imgpath = os.path.join(self.image_root, self.datalist[index])
        imgpaths = [imgpath + "/im1.png", imgpath + "/im2.png", imgpath + "/im3.png"]

        # Load images
        img0 = Image.open(imgpaths[0])
        img1 = Image.open(imgpaths[1])
        img2 = Image.open(imgpaths[2])

        # Data augmentation
        if self.training:
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            img0 = self.transforms(img0)
            torch.random.manual_seed(seed)
            img1 = self.transforms(img1)
            torch.random.manual_seed(seed)
            img2 = self.transforms(img2)
            # Random Temporal Flip
            if random.random() >= 0.5:
                img0, img2 = img2, img0
                imgpaths[0], imgpaths[2] = imgpaths[2], imgpaths[0]
        else:
            img0 = self.transforms(img0)
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)

        return torch.cat([img0, img2, img1], dim=0), self.datalist[index]

    def __len__(self):
        return len(self.datalist)


class VimeoSeptuplet(Dataset):
    # TODO: fix like vimeo triplet.
    def __init__(self, opt, is_training=True):
        self.data_root = opt["dataroot_video"]
        self.image_root = os.path.join(self.data_root, "sequences")
        self.training = is_training
        self.crop_size = opt["crop_size"] or 256
        self.interv = opt["interv"] or 1
        train_fn = os.path.join(self.data_root, "sep_trainlist.txt")
        test_fn = os.path.join(self.data_root, "sep_testlist.txt")
        if self.training:
            with open(train_fn, "r") as f:
                self.datalist = f.read().splitlines()
        else:
            with open(test_fn, "r") as f:
                self.datalist = f.read().splitlines()[: opt["nb_sample"]]
            if opt.get("nb_sample"):
                self.datalist = self.datalist[: opt["nb_sample"]]
        if self.training:
            self.transforms = transforms.Compose(
                [
                    transforms.RandomCrop(self.crop_size),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transforms = transforms.Compose(
                [transforms.CenterCrop(self.crop_size), transforms.ToTensor()]
            )

    def __getitem__(self, index):
        imgdir = os.path.join(self.image_root, self.datalist[index])
        imgpaths = sorted(list(glob.glob(os.path.join(imgdir, "*.png"))))
        assert len(imgpaths) == 7, "septuplet should include 7 images"
        indices = [3 - self.interv, 3, 3 + self.interv]

        # Load images
        img0 = cv2.imread(imgpaths[indices[0]])[:, :, ::-1].copy()
        img1 = cv2.imread(imgpaths[indices[1]])[:, :, ::-1].copy()
        img2 = cv2.imread(imgpaths[indices[2]])[:, :, ::-1].copy()

        # Data augmentation
        if self.training:
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            img0 = self.transforms(img0)
            torch.random.manual_seed(seed)
            img1 = self.transforms(img1)
            torch.random.manual_seed(seed)
            img2 = self.transforms(img2)
            # Random Temporal Flip
            if random.random() >= 0.5:
                img0, img2 = img2, img0
                imgpaths[0], imgpaths[2] = imgpaths[2], imgpaths[0]
        else:
            img0 = self.transforms(img0)
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)

        inps = [img0, img2]
        labs = [img1]
        return dict(inp=inps, lab=labs)

    def __len__(self):
        return len(self.datalist)


class OFVimeoTriplet(Dataset):
    """Vimeo dataset with optical flows."""

    def __init__(self, data_name, data_root, of_method, nb_sample=None):
        self.data_root = data_root
        self.image_root = os.path.join(self.data_root, "sequences")
        self.of_root = os.path.join(self.data_root, "optical_flows")
        self.of_method = [of_method] if isinstance(of_method, str) else of_method
        self.training = data_name == "train"
        self.crop_size = (256, 256)
        train_fn = os.path.join(self.data_root, "tri_trainlist.txt")
        test_fn = os.path.join(self.data_root, "tri_testlist.txt")
        if self.training:
            with open(train_fn, "r") as f:
                self.datalist = f.read().splitlines()
        else:
            with open(test_fn, "r") as f:
                self.datalist = f.read().splitlines()
            if nb_sample is not None:
                self.datalist = self.datalist[:nb_sample]
        if self.datalist[-1] == "":
            self.datalist.pop(-1)

        if self.training:
            self.transforms = transforms.Compose(
                [
                    transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transforms = transforms.ToTensor()

    def __getitem__(self, index):
        imgpath = os.path.join(self.image_root, self.datalist[index])
        if self.training and len(self.of_method) > 1:
            of_method = random.choice(self.of_method)
        else:
            of_method = self.of_method[0]
        flopath = os.path.join(self.of_root, of_method, self.datalist[index])
        imgpaths = [imgpath + "/im1.png", imgpath + "/im2.png", imgpath + "/im3.png"]
        flowpaths = [flopath + "/fflow.png", flopath + "/bflow.png"]

        # Load images & optical flows
        img0 = Image.open(imgpaths[0])
        img1 = Image.open(imgpaths[1])
        img2 = Image.open(imgpaths[2])
        fflow = readFlowKITTI(flowpaths[0])
        bflow = readFlowKITTI(flowpaths[1])

        # Data augmentation
        if self.training:
            h, w = fflow.shape[:2]
            seed = torch.random.seed()
            inps = [img0, img1, img2]
            for i in range(3):
                torch.random.manual_seed(seed)
                inps[i] = self.transforms(inps[i])
            rand_h, rand_w = random.randint(0, h - 256), random.randint(0, w - 256)
            img0, img1, img2 = [
                x[..., rand_h : rand_h + 256, rand_w : rand_w + 256] for x in inps
            ]
            fflow, bflow = [
                torch.from_numpy(x[rand_h : rand_h + 256, rand_w : rand_w + 256, ...])
                .permute(2, 0, 1)
                .float()
                for x in [fflow, bflow]
            ]
            # Random Temporal Flip
            if random.random() >= 0.5:
                img0, img2 = img2, img0
                fflow, bflow = bflow, fflow
        else:
            img0 = self.transforms(img0)
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)
            fflow, bflow = [
                torch.from_numpy(x).permute(2, 0, 1).float() for x in [fflow, bflow]
            ]

        return [img0, img2, fflow, bflow, img1], self.datalist[index]

    def __len__(self):
        return len(self.datalist)


class X_Test(Dataset):
    def make_2D_dataset_X_Test(test_data_path, multiple, t_step_size):
        """make [I0,I1,It,t,scene_folder]"""
        """ 1D (accumulated) """
        testPath = []
        t = np.linspace((1 / multiple), (1 - (1 / multiple)), (multiple - 1))
        for type_folder in sorted(
            glob(os.path.join(test_data_path, "*", ""))
        ):  # [type1,type2,type3,...]
            for scene_folder in sorted(
                glob(os.path.join(type_folder, "*", ""))
            ):  # [scene1,scene2,..]
                frame_folder = sorted(
                    glob(scene_folder + "*.png")
                )  # 32 multiple, ['00000.png',...,'00032.png']
                for idx in range(0, len(frame_folder), t_step_size):  # 0,32,64,...
                    if idx == len(frame_folder) - 1:
                        break
                    for mul in range(multiple - 1):
                        I0I1It_paths = []
                        I0I1It_paths.append(frame_folder[idx])  # I0 (fix)
                        I0I1It_paths.append(frame_folder[idx + t_step_size])  # I1 (fix)
                        I0I1It_paths.append(
                            frame_folder[
                                idx + int((t_step_size // multiple) * (mul + 1))
                            ]
                        )  # It
                        I0I1It_paths.append(t[mul])
                        I0I1It_paths.append(
                            scene_folder.split(os.path.join(test_data_path, ""))[-1]
                        )  # type1/scene1
                        testPath.append(I0I1It_paths)
        return testPath

    def frames_loader_test(I0I1It_Path):
        frames = []
        for path in I0I1It_Path:
            frame = cv2.imread(path)
            frames.append(frame)
        (ih, iw, c) = frame.shape
        frames = np.stack(frames, axis=0)  # (T, H, W, 3)
        """ np2Tensor [-1,1] normalized """
        frames = X_Test.RGBframes_np2Tensor(frames)

        return frames

    def RGBframes_np2Tensor(imgIn, channel=3):
        ## input : T, H, W, C
        if channel == 1:
            # rgb --> Y (gray)
            imgIn = (
                np.sum(
                    imgIn * np.reshape([65.481, 128.553, 24.966], [1, 1, 1, 3]) / 255.0,
                    axis=3,
                    keepdims=True,
                )
                + 16.0
            )

        # to Tensor
        ts = (3, 0, 1, 2)  ############# dimension order should be [C, T, H, W]
        imgIn = torch.Tensor(imgIn.transpose(ts).astype(float)).mul_(1.0)

        return imgIn

    def __init__(self, test_data_path, multiple):
        self.test_data_path = test_data_path
        self.multiple = multiple
        self.testPath = X_Test.make_2D_dataset_X_Test(
            self.test_data_path, multiple, t_step_size=32
        )

        self.nIterations = len(self.testPath)

        # Raise error if no images found in test_data_path.
        if len(self.testPath) == 0:
            raise (
                RuntimeError(
                    "Found 0 files in subfolders of: " + self.test_data_path + "\n"
                )
            )

    def __getitem__(self, idx):
        I0, I1, It, t_value, scene_name = self.testPath[idx]

        I0I1It_Path = [I0, I1, It]
        frames = X_Test.frames_loader_test(I0I1It_Path)
        # including "np2Tensor [-1,1] normalized"

        I0_path = I0.split(os.sep)[-1]
        I1_path = I1.split(os.sep)[-1]
        It_path = It.split(os.sep)[-1]

        return (
            frames,
            np.expand_dims(np.array(t_value, dtype=np.float32), 0),
            scene_name,
            [It_path, I0_path, I1_path],
        )

    def __len__(self):
        return self.nIterations


if __name__ == "__main__":
    pass
