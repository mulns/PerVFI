import torch

from utils.image_utils import InputPadder


def build_flow_estimator(name, device="cuda"):
    if name.lower() == "raft":
        import argparse

        from .raft.raft import RAFT

        args = argparse.Namespace(
            mixed_precision=True, alternate_corr=False, small=False
        )
        model = RAFT(args)
        ckpt = "checkpoints/raft-sintel.pth"
        model.load_state_dict(
            {k.replace("module.", ""): v for k, v in torch.load(ckpt).items()}
        )
        model.to(device).eval()

        @torch.no_grad()
        def infer(I1, I2):
            I1 = I1.to(device) * 255.0
            I2 = I2.to(device) * 255.0
            padder = InputPadder(I1.shape, 8)
            I1, I2 = padder.pad(I1, I2)
            _, fflow = model(I1, I2, test_mode=True, iters=20)
            _, bflow = model(I2, I1, test_mode=True, iters=20)
            return padder.unpad(fflow, bflow)

    if name.lower() == "gmflow":
        from .gmflow.gmflow import GMFlow

        model = GMFlow(
            feature_channels=128,
            num_scales=1,
            upsample_factor=8,
            num_head=1,
            attention_type="swin",
            ffn_dim_expansion=4,
            num_transformer_layers=6,
        )
        ckpt = "checkpoints/gmflow_sintel-0c07dcb3.pth"
        model.load_state_dict(torch.load(ckpt)["model"])
        model.to(device).eval()

        @torch.no_grad()
        def infer(I1, I2):
            I1 = I1.to(device) * 255.0
            I2 = I2.to(device) * 255.0
            padder = InputPadder(I1.shape, 8)
            I1, I2 = padder.pad(I1, I2)
            results_dict = model(
                I1,
                I2,
                attn_splits_list=[2],
                corr_radius_list=[-1],
                prop_radius_list=[-1],
                pred_bidir_flow=True,
            )
            flow_pr = results_dict["flow_preds"][-1]
            fflow, bflow = flow_pr.chunk(2)
            fflow, bflow = padder.unpad(fflow, bflow)
            return fflow, bflow

    return model, infer
