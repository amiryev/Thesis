from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
from functools import partial

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import matplotlib.pyplot as plt

from ddp import setup, cleanup, spawn
from xray_encoder import XrayEncoder
from dataset import MultiPatientDRRDataset  # <-- use the new dataset
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Distributed masked-patch reconstruction training")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--patch-size", type=int, default=32)
    p.add_argument("--mask-ratio", type=float, default=0.6)
    p.add_argument("--steps", type=int, default=5, help="grid steps per axis for valid-pose prefilter")
    p.add_argument("--min-intersections", type=int, default=750)
    p.add_argument("--samples-per-epoch", type=int, default=100000)
    p.add_argument("--ckpt-dir", type=Path, default=Path("models"))
    p.add_argument("--ct-dir", type=Path, default=Path("Datasets/CT"))
    p.add_argument("--vis-dir", type=Path, default=Path("xray_depthor/figures/encoder"))
    return p.parse_args()


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, pixel_mask: torch.Tensor) -> torch.Tensor:
    diff = torch.abs(pred - target) * pixel_mask
    denom = pixel_mask.sum().clamp_min(1.0)
    return diff.sum() / denom


def train(rank: int, world_size: int, args: argparse.Namespace) -> None:
    setup(rank, world_size)
    torch.manual_seed(args.seed + rank)

    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.ckpt_dir / "xray_encoder.pth"

    # Dataset across multiple patients
    ds = MultiPatientDRRDataset(
        ct_dir=args.ct_dir,
        device=device,
        size=args.size,
        steps=args.steps,
        min_intersections=args.min_intersections,
        samples_per_epoch=args.samples_per_epoch,
    )

    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler)

    # Model
    model = XrayEncoder(
        device=device,
        size=args.size,
        patch_size=args.patch_size,
    ).to(device)
    model = DDP(model, device_ids=[rank] if device.type == "cuda" else None, find_unused_parameters=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    start_epoch, best = 0, float("inf")

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and "model_sd" in ckpt:
            model.load_state_dict(ckpt["model_sd"])
            if "optim_sd" in ckpt:
                opt.load_state_dict(ckpt["optim_sd"])
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            best = float(ckpt.get("best_loss", float("inf")))
            if rank == 0:
                print(f"Resuming masked recon from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        mask_ratio = 0.6
        if rank == 0:
            iter_loader = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", ncols=100)
        else:
            iter_loader = loader

        model.train()
        losses: List[float] = []

        for step, imgs in enumerate(iter_loader):
            imgs = imgs.to(device, non_blocking=True)  # (B,1,H,W)

            opt.zero_grad(set_to_none=True)
            recon, pixel_mask, _ = model(imgs, mask_ratio=mask_ratio)
            loss = masked_l1_loss(recon, imgs, pixel_mask)
            loss.backward()
            opt.step()

            losses.append(loss.item())
            if rank == 0:
                iter_loader.set_postfix(loss=np.mean(losses))

            if rank == 0 and step % 1000 == 0:  # adjust frequency
                with torch.no_grad():
                    inp = imgs[0,0].detach().cpu()    # (H,W)
                    rec = recon[0,0].detach().cpu()   # (H,W)
                    msk = pixel_mask[0,0].detach().cpu()  # (H,W)

                    fig, axs = plt.subplots(1,3, figsize=(9,3))
                    axs[0].imshow(inp, cmap="gray")
                    axs[0].set_title("Input"); axs[0].axis("off")
                    axs[1].imshow(rec, cmap="gray")
                    axs[1].set_title("Recon"); axs[1].axis("off")
                    axs[2].imshow(inp, cmap="gray")
                    axs[2].imshow(msk, cmap="jet", alpha=0.4)  # overlay mask
                    axs[2].set_title("Mask overlay"); axs[2].axis("off")

                    plt.tight_layout()
                    out_path = args.vis_dir / f"epoch{epoch:03d}_step{step:05d}.png"
                    plt.savefig(out_path)
                    plt.close(fig)


        epoch_loss = float(np.mean(losses))
        if rank == 0 and epoch_loss < best:
            best = epoch_loss
            torch.save(
                {"epoch": epoch, "model_sd": model.state_dict(), "optim_sd": opt.state_dict(), "best_loss": best},
                ckpt_path,
            )

    cleanup()


def main() -> None:
    args = parse_args()
    world_size = torch.cuda.device_count() or 1
    spawn(partial(train, args=args), world_size)


if __name__ == "__main__":
    main()
