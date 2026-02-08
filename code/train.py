
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List
from functools import partial

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from trash.mamba_position import MambaPosition         
from dataset import PoseDataset            
from ddp import setup, cleanup, spawn         
from utils import PositionLoss                   

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Distributed training for MambaPosition")
    p.add_argument("--index", type=int, default=4, help="Patient / scan index")
    p.add_argument("--steps", type=int, default=5, help="Grid steps per axis")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4)
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
    ckpt_path = args.ckpt_dir / f"mamba_position_{args.index}.pth"

    # Model
    model = MambaPosition(
        dicom_file=f"datasets/CT/{args.index}.nii.gz",
        crm_path=f"datasets/CRM/{args.index}.png",
        device=device,
    ).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = PositionLoss()

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
            args.ckpt_dir.glob("mamba_position_*.pth"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if ckpts:
            latest = ckpts[0]
            print(f"Initializing weights from {latest.name}") if rank == 0 else None
            ckpt = torch.load(latest, map_location=device)
            if isinstance(ckpt, dict) and "model_sd" in ckpt:
                model.load_state_dict(ckpt["model_sd"])
            else:
                model.module.load_state_dict(ckpt.state_dict())
        else:
            print(f"No checkpoints found. Using random initialization.") if rank == 0 else None

    full_ds = PoseDataset(model, steps=args.steps, device="cpu")
    for epoch in range(start_epoch, args.epochs):
        torch.manual_seed(args.seed + epoch)
        sampler = DistributedSampler(full_ds, world_size, rank, shuffle=True, drop_last=True)
        sampler.set_epoch(epoch)
        loader = DataLoader(full_ds, batch_size=args.batch_size, sampler=sampler, pin_memory=True)

        losses: List[float] = []
        model.train()
        
        if rank == 0:
            loader = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", ncols=100)

        for b, (rot_gt, trans_gt) in enumerate(loader, 1):
            rot_gt, trans_gt = rot_gt.to(device, non_blocking=True), trans_gt.to(device, non_blocking=True)
            vec_gt = torch.cat([rot_gt, trans_gt], 1)

            with torch.no_grad():
                proj_gt = model.module.project(vec_gt) 

            opt.zero_grad(set_to_none=True)
            proj_pred, vec_pred, _ = model(proj_gt)
            loss = loss_fn(vec_pred, vec_gt)
            loss.backward()
            opt.step()
            losses.append(loss.item())

            if rank == 0:
                loader.set_postfix(loss=np.mean(losses))

            # if rank == 0 and b % 100 == 0:
            #     gt, pr = proj_gt[0,0].cpu().numpy(), proj_pred[0,0].detach().cpu().numpy()
            #     plt.figure(figsize=(5,10))
            #     for i, (t, im) in enumerate([("GT", gt), ("PR", pr)], 1):
            #         plt.subplot(2,1,i); plt.title(t); plt.imshow(im, cmap="gray"); plt.axis("off")
            #     plt.tight_layout(); plt.savefig(fig_dir / "debug.png"); plt.close()

        epoch_loss = float(np.mean(losses))
        if rank == 0:
            if epoch_loss < best:
                best = epoch_loss
                torch.save({"epoch": epoch, "model_sd": model.state_dict(), "optim_sd": opt.state_dict(), "best_loss": best}, ckpt_path)

    cleanup()

def main() -> None:
    args = parse_args()
    world_size = torch.cuda.device_count() or 1
    spawn(partial(train, args=args), world_size)


if __name__ == "__main__":
    main()
