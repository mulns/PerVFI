import _thread
import os
import os.path as osp
import subprocess
import time
from queue import Queue

import cv2
import numpy as np
import torch.utils.data as data
from torchvision.transforms import functional as TF

FFMPEG_HEAD = [
    "ffmpeg",
    "-hide_banner",
    "-loglevel",
    "error",
    "-y",
    "-pattern_type",
    "glob",
]


class IOBuffer:
    def __init__(self, rescale="1080P", inp_num=2):
        self.writer = Queue(maxsize=100)
        self.reader = Queue(maxsize=100)
        self.rescale = rescale
        self.inp_num = inp_num

    def build_read_buffer(self, inpSeq):
        dataGen = Tools.toPairs(inpSeq, self.inp_num)
        try:
            for inp in dataGen:
                pair = []
                for frame in inp:
                    frame = cv2.imread(frame, cv2.IMREAD_UNCHANGED)[..., ::-1]
                    frame = Tools.resize(frame.copy(), self.rescale)
                    pair.append(frame)
                pair = Tools.toTensor(pair)
                self.reader.put(pair)
        except:
            pass
        self.reader.put(None)

    def clear_write_buffer(self, out_dir):
        cnt = 0
        while True:
            item = self.writer.get()
            if item is None:
                break
            fname = osp.join(out_dir, f"{cnt:0>7d}.png")
            cv2.imwrite(fname, item[:, :, ::-1])
            cnt += 1

    def start(self, inpSeq, save_dir):
        _thread.start_new_thread(self.build_read_buffer, (inpSeq,))
        _thread.start_new_thread(self.clear_write_buffer, (save_dir,))

    def stop(self):
        while not self.writer.empty():
            time.sleep(0.1)


class Tools:
    @staticmethod
    def sample_sequence(xs, interval=2):
        # Downsample video frames temporally
        # X2 # 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1
        # X4 # 1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1
        # X8 # 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1
        assert interval in [2, 4, 8]
        assert (len(xs) - 1) % interval == 0
        out = []
        for i in range(0, len(xs), interval):
            out.append(xs[i])
        return out

    @staticmethod
    def toPairs(xs, num=2):
        # inp: [1,2,3,4,5]
        # num=2: [(1,2),(2,3),(3,4),(4,5)]
        # num=4: [(1,1,2,3),(1,2,3,4),(2,3,4,5),(3,4,5,5)]
        if num == 2:
            return list(zip(xs[:-1], xs[1:]))
        elif num == 4:
            xs = [xs[0]] + xs + [xs[-1]]
            return list(zip(xs[:-3], xs[1:-2], xs[2:-1], xs[3:]))
        else:
            raise NotImplementedError("only support 2 / 4 inputs")

    @staticmethod
    def insert_outs(inps, preds):
        """inps: List of images; preds: List of List of frames"""
        # inps: [1 1 1 1], preds: [[x x x],[y y y],[u u u]]
        # out: 1 x x x 1 y y y 1 u u u 1
        assert len(inps) - 1 == len(preds)
        A, B = inps, preds
        C = [val for pair in zip(A, B) for val in pair]  # insertion
        D = [
            ele if not isinstance(item, list) else ele
            for item in C
            for ele in (item if isinstance(item, list) else [item])
        ]  # flatten
        return D.append(A[-1])

    @staticmethod
    def resize(x, rescale=None):
        assert rescale in ["1080P", "2K", "4K", None]
        if rescale == "1080P":
            return cv2.resize(x, dsize=(1920, 1080), interpolation=cv2.INTER_AREA)
        elif rescale == "2K":
            return cv2.resize(x, dsize=(2048, 1080), interpolation=cv2.INTER_AREA)
        elif rescale == "4K":
            return x[540:-540, 1024:-1024, :]
        else:
            h, w, _ = x.shape
            return x[: h // 2 * 2, : w // 2 * 2, ...]

    @staticmethod
    def toTensor(x):
        # x: List of numpy array / A numpy array
        # out: List of torch tensor / A torch tensor
        if isinstance(x, (list, tuple)):
            return list(map(lambda x: TF.to_tensor(x)[None], x))
        return TF.to_tensor(x)[None]

    @staticmethod
    def toArray(x):
        # x: List of torch tensor / A torch tensor
        # out: List of numpy array / A numpy array
        if isinstance(x, (list, tuple)):
            return [np.array(TF.to_pil_image(y[0])) for y in x]
        return np.array(TF.to_pil_image(x[0]))

    @staticmethod
    def video2frames(video, output_glob):
        command = ["-i", video, "-pix_fmt", "rgba", output_glob]
        subprocess.run(FFMPEG_HEAD + command)

    @staticmethod
    def frames2rawvideo(inp_glob, output_pth):
        params = ["-framerate", "10", "-i", inp_glob, "-f", "rawvideo"]
        out = ["-pix_fmt", "yuv420p", output_pth]
        subprocess.run(FFMPEG_HEAD + params + out)

    @staticmethod
    def frames2mp4(inp_glob, output_pth, fps=10):
        params = ["-framerate", str(fps), "-i", inp_glob, "-c:v", "libx264"]
        out = ["-pix_fmt", "yuv420p", output_pth]
        subprocess.run(FFMPEG_HEAD + params + out)


class _BaseDST(data.Dataset):
    def __init__(self, root) -> None:
        self.triplets = []

    def transform(self, inps):
        return list(map(lambda x: TF.to_tensor(x)[None], inps))

    def imread(self, inps):
        return list(
            map(lambda x: cv2.imread(x, cv2.IMREAD_UNCHANGED)[..., ::-1].copy(), inps)
        )

    def __getitem__(self, index):
        tri = self.triplets[index]

        return self.transform(self.imread(tri))

    def __len__(self):
        return len(self.triplets)


class MiddelBury(_BaseDST):
    def __init__(self, root):
        self.triplets = []
        for x in os.listdir(f"{root}/other-data"):
            f0 = f"{root}/other-data/{x}/frame10.png"
            f1 = f"{root}/other-gt-interp/{x}/frame10i11.png"
            f2 = f"{root}/other-data/{x}/frame11.png"
            self.triplets.append([f0, f1, f2])


class Vimeo90K_test(_BaseDST):
    def __init__(self, root):
        self.triplets = []
        with open(f"{root}/tri_testlist.txt", "r") as f:
            samples = f.read().splitlines()

        for sample in [x for x in samples if x != ""]:
            f1 = f"{root}/target/{sample}/im1.png"
            f2 = f"{root}/target/{sample}/im2.png"
            f3 = f"{root}/target/{sample}/im3.png"
            self.triplets.append([f1, f2, f3])


class UCF101_test(_BaseDST):
    def __init__(self, root):
        self.triplets = []
        samples = os.listdir(root)
        for sample in samples:
            f0 = f"{root}/{sample}/frame_00.png"
            f1 = f"{root}/{sample}/frame_01_gt.png"
            f2 = f"{root}/{sample}/frame_02.png"
            self.triplets.append([f0, f1, f2])
