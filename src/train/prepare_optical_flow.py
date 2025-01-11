import argparse
import os
import os.path as osp
import warnings
from multiprocessing import Pool, Queue

import cv2
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from flow_estimators import build_flow_estimator
from utils.flow_viz import flow_to_image
from utils.image_utils import readFlowKITTI, writeFlowKITTI

warnings.filterwarnings("ignore")


def clear_write_buffer(save_root, write_buffer):
    while True:
        item = write_buffer.get()  # fflow, bflow, idx
        if item is None:
            break
        fflows, bflows, idx = item
        B = fflows.shape[0]
        for i in range(B):
            fflow, bflow, id = fflows[i], bflows[i], idx[i]
            save_dir = osp.join(save_root, id)
            os.makedirs(save_dir, exist_ok=True)
            writeFlowKITTI(
                osp.join(save_dir, "fflow.png"), fflow.cpu().permute(1, 2, 0).numpy()
            )
            writeFlowKITTI(
                osp.join(save_dir, "bflow.png"), bflow.cpu().permute(1, 2, 0).numpy()
            )


class VimeoTriplet(Dataset):
    def __init__(self, data_root, number=None):
        self.image_root = os.path.join(data_root, "sequences")
        train_fn = os.path.join(data_root, "tri_trainlist.txt")
        test_fn = os.path.join(data_root, "tri_testlist.txt")
        with open(train_fn, "r") as f:
            train_ds = f.read().splitlines()
        with open(test_fn, "r") as f:
            test_ds = f.read().splitlines()
        # self.datalist = test_ds[1000:]
        self.datalist = train_ds + test_ds
        if self.datalist[-1] == "":
            self.datalist.pop(-1)
        if number is not None:
            self.datalist = self.datalist[:number]
        self.transforms = transforms.ToTensor()

    def __getitem__(self, index):
        imgpath = os.path.join(self.image_root, self.datalist[index])
        imgpaths = [imgpath + "/im1.png", imgpath + "/im2.png", imgpath + "/im3.png"]
        imgs = [self.transforms(Image.open(x)) for x in imgpaths]
        return imgs, self.datalist[index]

    def __len__(self):
        return len(self.datalist)


def main(args):
    NUM_THREAD = 16
    source = "data/datasets/vimeo_triplet"
    method = args.method
    target = osp.join(source, "optical_flows", method)
    dataset = VimeoTriplet(source)
    dataloader = DataLoader(
        dataset, batch_size=64, num_workers=4, pin_memory=True, drop_last=False
    )
    write_buffer = Queue()
    p = Pool(NUM_THREAD, clear_write_buffer, (target, write_buffer))
    _, infer_func = build_flow_estimator(method, "cuda")
    for data, idx in tqdm(dataloader):
        fflows, bflows = infer_func(data[0], data[2])
        write_buffer.put((fflows, bflows, idx))
    for _ in range(NUM_THREAD):
        write_buffer.put(None)
    write_buffer.close()
    write_buffer.join_thread()
    p.close()
    p.join()

    # # visualize
    # count = 0
    # for data, idx in dataloader:
    #     B = data[0].shape[0]
    #     for i in range(B):
    #         fflow = readFlowKITTI(osp.join(target, idx[i], "fflow.png"))
    #         bflow = readFlowKITTI(osp.join(target, idx[i], "bflow.png"))
    #         fflow = flow_to_image(fflow)
    #         bflow = flow_to_image(bflow)
    #         save_dir = osp.join("./optial_flow_samples", method, idx[i])
    #         os.makedirs(save_dir, exist_ok=True)
    #         cv2.imwrite(save_dir + "/fflow.png", fflow[..., ::-1])
    #         cv2.imwrite(save_dir + "/bflow.png", bflow[..., ::-1])
    #         count += 1

    #         if count >= 100:
    #             return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", "-m", type=str, choices=["raft", "gmflow"], default="raft"
    )
    parser.add_argument("--root", "-r", type=str, default="data/datasets/vimeo_triplet")
    args = parser.parse_args()
    main(args)
