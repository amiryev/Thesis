
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List
from functools import partial

from xray_encoder import XrayEncoder
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import OrderedDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from position_estimator import PositionEstimator         
from dataset import PoseDataset            
from ddp import setup, cleanup, spawn         
from utils import PositionLoss                   

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Distributed training for PositionEstimator")
    p.add_argument("--index", type=int, default=4, help="Patient / scan index")
    p.add_argument("--steps", type=int, default=5, help="Grid steps per axis")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--log-every", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fig-dir", type=Path, default=Path("xray_depthor/figures/pose_estimator"))
    p.add_argument("--ckpt-dir", type=Path, default=Path("models"))
    return p.parse_args()


def train(rank: int, world_size: int, args: argparse.Namespace) -> None:
    setup(rank, world_size)

    torch.manual_seed(args.seed + rank)
    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")

    # Prepare dirs
    fig_dir = args.fig_dir / str(args.index)
    fig_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.ckpt_dir / f"position_estimator_{args.index}.pth"

    # Model
    ckpt = torch.load("models/xray_encoder.pth", map_location=device, weights_only=True)
    state_dict = ckpt["model_sd"]

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # remove "module." prefix
        new_state_dict[name] = v

    encoder = XrayEncoder(
        device=device,
        size=128,
        patch_size=32,
    ).to(device)

    encoder.load_state_dict(new_state_dict, strict=True)

    model = PositionEstimator(
        encoder=encoder,
        dicom_file=f"datasets/CT/{args.index}.nii.gz",
        crm_path=f"datasets/CRM/{args.index}.png",
    ).to(device)

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    loss_fn = PositionLoss().to(device)

    start_epoch, best = 0, float("inf")

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict):
            model.load_state_dict(ckpt["model_sd"])
            opt.load_state_dict(ckpt["optim_sd"])
            start_epoch = ckpt["epoch"] + 1
            best = ckpt["best_loss"]
            print(f"Resuming training of patient {args.index}") if rank == 0 else None
    else:
        ckpts = sorted(
            args.ckpt_dir.glob("position_estimator_*.pth"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if ckpts:
            latest = ckpts[0]
            if rank == 0:
                print(f"Initializing encoder and heads from {latest.name}")
            ckpt = torch.load(latest, map_location=device)

            if isinstance(ckpt, dict) and "model_sd" in ckpt:
                ckpt = ckpt["model_sd"]
            else:
                ckpt = ckpt.state_dict()

            # keep only encoder + heads
            allowed = ["module.encoder", "module.translation_head", "module.rotation_head"]
            filtered_sd = {k: v for k, v in ckpt.items() if any(k.startswith(a) for a in allowed)}

            # load selectively
            missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
            if rank == 0:
                print(f"Loaded keys: {list(filtered_sd.keys())[:5]}... "
                    f"({len(filtered_sd)} total)")
                if missing:
                    print("Missing keys (left random init):", missing[:5], "...")
                if unexpected:
                    print("Unexpected keys (ignored):", unexpected[:5], "...")
        else:
            if rank == 0:
                print("No checkpoints found. Using random initialization.")

    full_ds = PoseDataset(model.module.ct, steps=args.steps, min_intersections=500, device="cpu")
    sampler = DistributedSampler(full_ds, world_size, rank, shuffle=True, drop_last=True)
    loader = DataLoader(full_ds, batch_size=args.batch_size, sampler=sampler)

    for epoch in range(start_epoch, args.epochs):
        torch.manual_seed(args.seed + epoch)
        sampler.set_epoch(epoch)

        losses: List[float] = []
        model.train()
        
        if rank == 0:
            loader = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", ncols=100)

        for step, (rot_gt, trans_gt) in enumerate(loader, 1):
            rot_gt, trans_gt = rot_gt.to(device, non_blocking=True), trans_gt.to(device, non_blocking=True)
            vec_gt = torch.cat([rot_gt, trans_gt], 1)

            with torch.no_grad():
                proj_gt = model.module.project(vec_gt) 

            opt.zero_grad(set_to_none=True)
            proj_pred, vec_pred = model(proj_gt)
            loss = loss_fn(vec_pred, vec_gt)
            loss.backward()
            opt.step()
            losses.append(loss.item())

            if rank == 0:
                loader.set_postfix(loss=np.mean(losses))

            if rank == 0 and step % args.log_every == 0:
                gt, pr = proj_gt[0,0].cpu().numpy(), proj_pred[0,0].detach().cpu().numpy()
                vec_gt_np = vec_gt[0].detach().cpu().numpy()
                vec_pr_np = vec_pred[0].detach().cpu().numpy()

                fig = plt.figure(figsize=(12, 8))

                # --- Projection images ---
                ax1 = plt.subplot(2, 2, 1)
                ax1.set_title("GT Projection")
                ax1.imshow(gt, cmap="gray")
                ax1.axis("off")

                ax2 = plt.subplot(2, 2, 2)
                ax2.set_title("PR Projection")
                ax2.imshow(pr, cmap="gray")
                ax2.axis("off")

                # --- Rotations ---
                ax3 = plt.subplot(2, 2, 3)
                idx = np.arange(3)
                width = 0.35
                ax3.bar(idx - width/2, vec_gt_np[:3], width, label="GT")
                ax3.bar(idx + width/2, vec_pr_np[:3], width, label="Pred")
                ax3.set_xticks(idx)
                ax3.set_xticklabels(["yaw", "pitch", "roll"])
                ax3.set_title("Rotation (radians)")
                ax3.legend()

                # --- Translations ---
                ax4 = plt.subplot(2, 2, 4)
                idx = np.arange(3)
                ax4.bar(idx - width/2, vec_gt_np[3:], width, label="GT")
                ax4.bar(idx + width/2, vec_pr_np[3:], width, label="Pred")
                ax4.set_xticks(idx)
                ax4.set_xticklabels(["dx", "dy", "dz"])
                ax4.set_title("Translation (units)")
                ax4.legend()

                plt.tight_layout()
                plt.savefig(fig_dir / f"{epoch}_{step}.png")
                plt.close()

        epoch_loss = float(np.mean(losses))
        if rank == 0:
            if epoch_loss < best:
                best = epoch_loss
                torch.save({"epoch": epoch, "model_sd": model.state_dict(), "optim_sd": opt.state_dict(), "best_loss": best}, ckpt_path)

    cleanup()

def main() -> None:
    args = parse_args()
    world_size = torch.cuda.device_count() or 1
    print(world_size)
    spawn(partial(train, args=args), world_size)


if __name__ == "__main__":
    main()
