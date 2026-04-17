import os
import json
import math
import itertools
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
import random

import numpy as np
from PIL import Image

import torch
import torchvision.io as io
from torch.utils.data import Dataset
from torchio import Subject
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.renderers import _get_alphas, _get_alpha_minmax
from diffdrr.pose import convert

from src.utils import config as config

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
        data_dir: Path,
        device: torch.device = torch.device("cpu"),
        size: int = config.IMAGE_SIZE,
        sdd: float = config.SDD,
        delx: float = config.DELX,
        steps: int = 5,
        min_intersections: int = 750,
        samples_per_epoch: int = 10000,
        seed: int = 42,
        patient_ids: tuple = (1, 11),
        return_pose: bool = False
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.device = device
        self.size = size
        self.sdd = sdd
        self.delx = delx
        self.steps = steps
        self.min_intersections = min_intersections
        self.samples_per_epoch = samples_per_epoch
        self.rng = np.random.default_rng(seed)
        self.return_pose = return_pose

        index_json = os.path.join(self.data_dir, "data_index.json")
        ct_files_all = self.load_ct_subset(index_json, patient_ids[0], patient_ids[1])
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

    def load_ct_subset(self, index_file, patient_start, patient_end):
        """
        Load CT file paths for a subset of patients.

        Parameters
        ----------
        index_file : str 
            Path to dataset index JSON.
        patient_start : int
            First patient ID (inclusive).
        patient_end : int
            Last patient ID (inclusive).

        Returns
        -------
        list
            List of CT file paths.
        """

        with open(index_file, "r") as f:
            index = json.load(f)

        ct_list = []

        for entry in index["entries"]:
            pid = int(entry["id"])

            if patient_start <= pid <= patient_end:
                curr_ct_path = os.path.join(self.data_dir, entry["ct"])
                ct_list.append(curr_ct_path)

        return sorted(ct_list)

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
        
        rotation = torch.tensor([yaw_grid[index_tuple[0]], pitch_grid[index_tuple[1]], roll_grid[index_tuple[2]]], device=device,).unsqueeze(0)  # (1,3)
        translation = torch.tensor([dxg[index_tuple[3]], dyg[index_tuple[4]], dzg[index_tuple[5]]], device=device).unsqueeze(0)  # (1,3)
        return rotation, translation

    def __getitem__(self, idx: int):
        p_idx = self.rng.integers(0, len(self.entries))
        subj, drr, grids, valid_indices = self.entries[p_idx]

        i_idx = self.rng.integers(0, len(valid_indices))
        pose_idx = valid_indices[i_idx]

        rot, trans = self._index_to_pose(grids, pose_idx, self.device)

        proj = drr(rot, trans, parameterization="euler_angles", convention="ZXY").squeeze(0)
        proj = _normalize_projection(proj)

        if self.return_pose:
            pose = torch.cat([rot.squeeze(0), trans.squeeze(0)])
            return proj, pose

        return proj  

class DRRMetadataDataset(Dataset):
    def __init__(self, root_dir, transform=None, repeats: int = 0, return_pose: bool = False, valid_ids = None):
        if repeats > 0:
            assert callable(transform), "Transform must be a callable function/object when repeats > 0."

        # Dataset parameters
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.return_pose = return_pose
        self.repeats = repeats

        # Load the master index to find patient folders
        master_path = os.path.join(root_dir, 'master_index.json')
        with open(master_path, 'r') as f:
            patients = json.load(f)
            
        # Map every image to its specific metadata
        patients = patients['entries'] if 'entries' in patients else patients
        for p in patients:
            p_id = int(p['id'])
            if (valid_ids is not None) and (p_id not in valid_ids):
                continue
            p_folder = p['DRR'] if 'DRR' in p else p['folder']
            p_dir = os.path.join(root_dir, p_folder)
            
            # Load the specific patient's metadata
            metadata_path = os.path.join(p_dir, 'metadata.json')
            with open(metadata_path, 'r') as f:
                p_metadata = json.load(f)

            # Link image path to its pose/labels
            for img_name, info in p_metadata.items():
                self.samples.append({
                    'path': os.path.join(p_dir, img_name),
                    'pose': torch.tensor(info['pose'], dtype=torch.float32),
                    'is_centered': int(info['is_centered']),
                    'orientation': info['orientation'],
                    'id': int(p_id)
                })

    def use_repeats(self, image, pose):
        # Create list of image tensors (Original + N Transforms), each image is (1, H, W)
        image = image.unsqueeze(0)  # (1, 1, H, W)
        # images_list = [image] + [self.transform(image) for _ in range(self.repeats - 1)] # (N, 1, 1, H, W)
        
        # Stack images into one tensor: shape (N, 1, H, W)
        # images = torch.stack(images_list).squeeze(1)

        image_augmented = image.repeat_interleave(self.repeats - 1, dim=0) # (N-1, 1, H, W)
        image_augmented = self.transform(image_augmented)
        images = torch.cat([image, image_augmented], dim=0)  # (N, 1, H, W)

        # Duplicate pose to match number of images, shape (N, 6)
        poses = pose.repeat(self.repeats, 1)
        
        return images, poses

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load Image
        image = Image.open(sample['path']).convert('L')
        image = TF.to_tensor(image)
        # image = io.read_image(sample['path'], mode=io.ImageReadMode.GRAY)

        # Convert to float and normalize to [0, 1]
        # image = image.float() / 255.0
        # image = T.ToTensor()(image)

        if self.transform:
            image = self.transform(image)

        # Return image and the pose (common target for DRR training)
        if self.return_pose:
            return image, sample['pose'], sample

        return image
    
    # def __getitem__(self, idx):
    #     sample = self.samples[idx]
        
    #     # Load Image
    #     image = Image.open(sample['path']).convert('L')
    #     image = TF.to_tensor(image)
    #     # image = io.read_image(sample['path'], mode=io.ImageReadMode.GRAY)

    #     # Convert to float and normalize to [0, 1]
    #     # image = torch.tensor(image.float() / 255.0)

    #     # Load poses from dict
    #     pose = torch.tensor(sample['pose'], dtype=torch.float32)

    #     if self.repeats > 0:
    #         image, pose = self.use_repeats(image, pose)
    #     elif self.transform is not None:
    #         image = self.transform(image)

    #     # Return image and the pose (common target for DRR training)
    #     if self.return_pose:
    #         return image, pose, sample

    #     return image

class PairedDRRMetadataDataset(DRRMetadataDataset):
    """
    Dataset for loading paired DRRs (anchor + positive) from the same patient.

    Inherits from DRRMetadataDataset and overrides __getitem__ to return
    pairs instead of single samples, suitable for similarity-based training.

    Args:
        min_pose_diff (float): Minimum Euclidean distance between anchor and positive
                               poses to avoid trivial similarity pairs.
    """

    def __init__(self, root_dir, transform=None, return_pose: bool = False, min_pose_diff: float = 0.0):
        # Initialize parent class
        super().__init__(root_dir, transform, return_pose)
        self.min_pose_diff = min_pose_diff

        # Build patient-to-sample index for fast positive sampling
        self.patient_to_indices = defaultdict(list)
        for i, sample in enumerate(self.samples):
            self.patient_to_indices[sample["id"]].append(i)

    def __getitem__(self, idx):
        """
        Returns a pair of DRRs (anchor and positive) from the same patient.

        Args:
            idx (int): Index of the anchor DRR

        Returns:
            tuple: (anchor_img, positive_img) or
                   (anchor_img, positive_img, anchor_pose, positive_pose)
        """
        anchor_sample = self.samples[idx]
        anchor_id = anchor_sample["id"]

        # ---- Sample a positive image from same patient ----
        candidate_indices = self.patient_to_indices[anchor_id]

        if len(candidate_indices) > 1:
            pos_idx = idx
            # Ensure positive is different and meets min_pose_diff
            while pos_idx == idx or (
                self.min_pose_diff > 0.0 and
                torch.norm(self.samples[pos_idx]["pose"] - anchor_sample["pose"]) < self.min_pose_diff
            ):
                pos_idx = random.choice(candidate_indices)
        else:
            # Edge case: only one sample for this patient
            pos_idx = idx

        positive_sample = self.samples[pos_idx]

        # ---- Load images using parent class functionality ----
        def load_image(sample):
            img = io.read_image(sample["path"], mode=io.ImageReadMode.GRAY)
            img = img.float() / 255.0  # normalize to [0,1]
            if self.transform:
                img = self.transform(img)
            return img

        anchor_img = load_image(anchor_sample)
        positive_img = load_image(positive_sample)

        if self.return_pose:
            return (
                anchor_img,
                positive_img,
                anchor_sample["pose"],
                positive_sample["pose"],
            )

        return anchor_img, positive_img
    

class RepeatDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, repeats):
        self.dataset = dataset
        self.repeats = repeats

    def __len__(self):
        return len(self.dataset) * self.repeats

    def __getitem__(self, idx):
        real_idx = idx % len(self.dataset)
        return self.dataset[real_idx]