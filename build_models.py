import warnings

from torch.nn import functional as F

warnings.simplefilter("ignore", UserWarning)


def build_model(name, device="cuda"):
    if "pervfi-vb" in name.lower():
        from models.pipeline import Pipeline_infer

        ckpt = "checkpoints/PerVFI/vb.pth"
        # e.g., RAFT+PerVFI-vb
        ofnet = name.split("+")[0]
        ofnet = None if ofnet == "none" else ofnet

        model = Pipeline_infer(ofnet, "vb", model_file=ckpt)
    elif "pervfi" in name.lower():
        from models.pipeline import Pipeline_infer

        ckpt = "checkpoints/PerVFI/v00.pth"
        # e.g., RAFT+PerVFI
        ofnet = name.split("+")[0]
        ofnet = None if ofnet == "none" else ofnet

        model = Pipeline_infer(ofnet, "v00", model_file=ckpt)

    else:
        raise ValueError("model name not supported")

    def infer(I1, I2, time=0.5):
        divide = 8
        _, _, H, W = I1.size()
        H_padding = (divide - H % divide) % divide
        W_padding = (divide - W % divide) % divide
        I1, I2 = [
            F.pad(x, (0, W_padding, 0, H_padding), "constant", 0.0) for x in [I1, I2]
        ]
        pred = model.inference_rand_noise(I1, I2, heat=0.3, time=time)
        return pred[..., :H, :W]

    return model.to(device), infer
