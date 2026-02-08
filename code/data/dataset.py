import math
import itertools
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchio import Subject

from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.renderers import _get_alphas, _get_alpha_minmax
from diffdrr.pose import convert

from config import default as config

def _normalize_projection(projection: torch.Tensor) -> torch.Tensor:
    """
    projection: (1,H,W) float
    returns:   (1,H,W) in [0,1], inverted like your original code
    """
    mn = projection.amin(dim=(-2, -1), keepdim=True)
    mx = projection.amax(dim=(-2, -1), keepdim=True)
    eps = 1e-8
    proj = 1.0 - (projection - mn) / (mx - mn + eps)
    return torch.clamp(proj, 0.0, 1.0)

class PoseDataset(Dataset):
    def __init__(self,
                ct,
                device="cpu",
                steps=5,
                min_intersections=750,
                size: int = config.IMAGE_SIZE,
                sdd: float = config.SDD,
                delx: float = config.DELX):
                
        super().__init__()
        self.ct = ct
        self.device = device
        self.size = size
        self.sdd = sdd
        self.delx = delx
        self.steps = steps
        self.min_intersections = min_intersections
        
        self.drr = DRR(ct, sdd=self.sdd, height=self.size, delx=self.delx)

        self.rot_grid = torch.linspace(-math.pi / 3, math.pi / 3, steps)
        self.dx_grid = torch.linspace(-35, 35, steps)
        self.dy_grid = torch.linspace(850, 950, steps)
        self.dz_grid = torch.linspace(-50, 50, steps)

        self.all_indices = list(itertools.product(range(steps), repeat=6))
        self.filter_samples(self.min_intersections)

    def __len__(self):
        return len(self.all_indices)

    def filter_samples(self, min_intersections, batch_size=64):
        valid_indices = []
        total = len(self.all_indices)

        with torch.no_grad():
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                batch = self.all_indices[start:end]

                rotations = torch.tensor([
                    [self.rot_grid[i[0]], self.rot_grid[i[1]], self.rot_grid[i[2]]]
                    for i in batch
                ], device=self.device)

                translations = torch.tensor([
                    [self.dx_grid[i[3]], self.dy_grid[i[4]], self.dz_grid[i[5]]]
                    for i in batch
                ], device=self.device)

                poses = convert(rotations, translations, parameterization="euler_angles", convention="ZXY")

                source, target = self.drr.detector(poses, None)
                target = target.mean(dim=1, keepdim=True)

                source = self.drr.affine_inverse(source)
                target = self.drr.affine_inverse(target)

                dims = self.drr.renderer.dims(self.drr.subject.density.data.squeeze())

                alphas = _get_alphas(
                    source,
                    target,
                    dims,
                    self.drr.renderer.eps,
                    False
                )

                alphamin, alphamax = _get_alpha_minmax(source, target, dims, self.drr.renderer.eps)
                good_idxs = (alphamin <= alphas) & (alphas <= alphamax)
                lengths = good_idxs.squeeze(1).sum(dim=-1)
                keep_idx = torch.nonzero(lengths > min_intersections, as_tuple=False).squeeze(-1)
                valid_indices.extend([batch[i] for i in keep_idx.tolist()])

            self.all_indices = valid_indices

    def __getitem__(self, idx):
        index = self.all_indices[idx]

        rotation = torch.tensor([
            self.rot_grid[index[0]],
            self.rot_grid[index[1]],
            self.rot_grid[index[2]],
        ], device=self.device)

        translation = torch.tensor([
            self.dx_grid[index[3]],
            self.dy_grid[index[4]],
            self.dz_grid[index[5]],
        ], device=self.device)

        return rotation, translation


class MultiPatientDRRDataset(Dataset):
    """
    Samples random DRR projections from *multiple* CTs for masked-image reconstruction.
    """
    def __init__(
        self,
        ct_dir: Path,
        device: torch.device = torch.device("cpu"),
        size: int = config.IMAGE_SIZE,
        sdd: float = config.SDD,
        delx: float = config.DELX,
        steps: int = 5,
        min_intersections: int = 750,
        samples_per_epoch: int = 10000,
        seed: int = 42,
    ):
        super().__init__()
        self.ct_dir = Path(ct_dir)
        self.device = device
        self.size = size
        self.sdd = sdd
        self.delx = delx
        self.steps = steps
        self.min_intersections = min_intersections
        self.samples_per_epoch = samples_per_epoch
        self.rng = np.random.default_rng(seed)

        ct_files_all = sorted(list(self.ct_dir.glob("*.nii*")))
        self.entries: List[Tuple[Subject, DRR, Dict[str, torch.Tensor], List[Tuple[int, ...]]]] = []

        for ct_path in ct_files_all:
            subj = read(str(ct_path))
            drr = DRR(subj, sdd=self.sdd, height=self.size, delx=self.delx)
            drr.to(self.device)

            dims = drr.renderer.dims(drr.subject.density.data.squeeze())
            if len(dims) != 3:
                print(f"[Skip] Patient {ct_path} has invalid CT shape: {dims}")
                continue
            
            grids = self._build_grids(self.steps)
            valid_indices = self._filter_valid_indices(drr, grids, batch_size=128, min_intersections=self.min_intersections)
            if len(valid_indices) == 0:
                continue

            self.entries.append((subj, drr, grids, valid_indices))

    def __len__(self):
        return self.samples_per_epoch

    @staticmethod
    def _build_grids(steps: int) -> Dict[str, torch.Tensor]:
        yaw_grid = torch.linspace(-math.pi / 3, math.pi / 3, steps)
        pitch_grid = torch.linspace(-math.pi / 3, math.pi / 3, steps)
        roll_grid = torch.linspace(-math.pi / 2, math.pi / 2, steps)
        dx_grid = torch.linspace(-35, 35, steps)
        dy_grid = torch.linspace(850, 950, steps)
        dz_grid = torch.linspace(-50, 50, steps)
        return {
            "rot": torch.zeros(steps), # Dummy key, actual values used in index_to_pose from specific grids
            "yaw": yaw_grid,
            "pitch": pitch_grid,
            "roll": roll_grid,
            "dx": dx_grid,
            "dy": dy_grid,
            "dz": dz_grid,
            "all_indices": list(itertools.product(range(steps), repeat=6)),
        }

    @torch.no_grad()
    def _filter_valid_indices(
        self,
        drr: DRR,
        grids: Dict[str, torch.Tensor],
        batch_size: int = 128,
        min_intersections: int = 750,
    ) -> List[Tuple[int, ...]]:
        valid_indices: List[Tuple[int, ...]] = []
        all_idx = grids["all_indices"]
        # Unpack individual grids
        yaw_grid, pitch_grid, roll_grid = grids["yaw"], grids["pitch"], grids["roll"]
        dxg, dyg, dzg = grids["dx"], grids["dy"], grids["dz"]
        
        total = len(all_idx)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = all_idx[start:end]

            rotations = torch.tensor(
                [[yaw_grid[i[0]], pitch_grid[i[1]], roll_grid[i[2]]] for i in batch],
                device=self.device,
            )
            translations = torch.tensor(
                [[dxg[i[3]], dyg[i[4]], dzg[i[5]]] for i in batch],
                device=self.device,
            )

            poses = convert(rotations, translations, parameterization="euler_angles", convention="ZXY")

            source, target = drr.detector(poses, None)
            target = target.mean(dim=1, keepdim=True)

            source = drr.affine_inverse(source)
            target = drr.affine_inverse(target)

            dims = drr.renderer.dims(drr.subject.density.data.squeeze())
            alphas = _get_alphas(source, target, dims, drr.renderer.eps, False)
            alphamin, alphamax = _get_alpha_minmax(source, target, dims, drr.renderer.eps)

            good = (alphamin <= alphas) & (alphas <= alphamax)
            lengths = good.squeeze(1).sum(dim=-1)
            keep_idx = torch.nonzero(lengths > min_intersections, as_tuple=False).squeeze(-1)

            valid_indices.extend([batch[i] for i in keep_idx.tolist()])

        return valid_indices

    @torch.no_grad()
    def _index_to_pose(
        self,
        grids: Dict[str, torch.Tensor],
        index_tuple: Tuple[int, ...],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        yaw_grid, pitch_grid, roll_grid = grids["yaw"], grids["pitch"], grids["roll"]
        dxg, dyg, dzg = grids["dx"], grids["dy"], grids["dz"]
        
        rotation = torch.tensor(
            [yaw_grid[index_tuple[0]], pitch_grid[index_tuple[1]], roll_grid[index_tuple[2]]],
            device=device,
        ).unsqueeze(0)  # (1,3)
        translation = torch.tensor(
            [dxg[index_tuple[3]], dyg[index_tuple[4]], dzg[index_tuple[5]]],
            device=device,
        ).unsqueeze(0)  # (1,3)
        return rotation, translation

    def __getitem__(self, idx: int) -> torch.Tensor:
        p_idx = self.rng.integers(0, len(self.entries))
        subj, drr, grids, valid_indices = self.entries[p_idx]

        i_idx = self.rng.integers(0, len(valid_indices))
        pose_idx = valid_indices[i_idx]

        rot, trans = self._index_to_pose(grids, pose_idx, self.device)

        proj = drr(rot, trans, parameterization="euler_angles", convention="ZXY").squeeze(0)
        proj = _normalize_projection(proj)

        return proj  
