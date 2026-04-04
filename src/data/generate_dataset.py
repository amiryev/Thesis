import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import math
import argparse
import random
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image

import tempfile
import nibabel as nib
import nibabel.orientations as nio

from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.pose import convert
from diffdrr.renderers import _get_alphas, _get_alpha_minmax

from src.utils import config
from src.core.layers import Sobel

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def normalize_and_save(projection: torch.Tensor, path: Path):
    """Inverts and normalizes to [0, 255] for standard X-ray appearance."""
    proj = projection.detach().cpu().numpy().squeeze()
    mn, mx = proj.min(), proj.max()
    proj = 255.0 * (1.0 - (proj - mn) / (mx - mn + 1e-8))
    Image.fromarray(proj.astype(np.uint8)).save(path)

# --------------------------------------------------
# Generator Class
# --------------------------------------------------

class DRRDataGenerator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(args.index_file, "r") as f:
            self.index_data = json.load(f)
        os.makedirs(args.output_dir, exist_ok=True)
        random.seed(42)
        torch.manual_seed(42)

    def check_local_variance(p_norm: torch.tensor, patch_size=16, threshold=0.005):
        # Unfold image into patches
        patches = p_norm.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
        # Calculate variance per patch
        patch_vars = patches.reshape(-1, patch_size * patch_size).var(dim=1)
        # If more than 60% of patches are 'flat' blobs, discard
        flat_patches = (patch_vars < threshold).float().mean()
        return flat_patches > 0.60
    
    # def is_off_center(p_norm : torch.tensor, corner_size=64):
    #     # h, w = p_norm.shape
    #     corners = [
    #         p_norm[:corner_size, :corner_size],    # TL
    #         p_norm[:corner_size, -corner_size:],   # TR
    #         p_norm[-corner_size:, :corner_size],   # BL
    #         p_norm[-corner_size:, -corner_size:]   # BR
    #     ]
    #     # If more than 1 corner is mostly empty air, discard
    #     empty_corners = sum([(c > 0.95).float().mean() > 0.5 for c in corners])
    #     return empty_corners > 1

    # def is_too_smooth(p_norm, threshold=0.005):
    #         # Define the Laplacian kernel once
    #     laplacian_kernel = torch.tensor([
    #         [0,  1, 0],
    #         [1, -4, 1],
    #         [0,  1, 0]
    #     ], dtype=torch.float32).reshape(1, 1, 3, 3)

    #     # p_norm is [H, W], scaled 0-1
    #     img = p_norm[None, None, ...]
        
    #     # Calculate the Laplacian (second derivative)
    #     lap = F.conv2d(img, laplacian_kernel.to(img.device), padding=1)
        
    #     # Measure the variance of the Laplacian
    #     # A "smudgy" image will have a very low variance here
    #     edge_var = lap.var()
        
    #     return edge_var < threshold

    @torch.no_grad()
    def sample_valid_poses(self, drr, n_targets, deltas=[40,15,40]):
        """
        Samples poses using Gaussian distribution.
        Since center_volume=True was used, anatomy is at (0,0,0).
        """
        valid_rots = []
        valid_trans = []
        
        pbar = tqdm(total=n_targets, desc="Sampling Valid Poses", leave=False)
        
        sobel = Sobel().to(self.device)

        while len(valid_rots) < n_targets:
            batch_size = 512
        
            # Rotations: Normal distribution (std dev ~15-20 degrees)
            # Centered at 0 because orientation="AP" provides a standard baseline
            # Sample rotations: Normal(0, 1) multiplied by our specific STDs
            # angle_stds = torch.tensor([0.4, 0.25, 0.1], device=self.device)
            angle_stds = torch.tensor([0.3, 0.3, 0.3], device=self.device)
            rots = torch.randn(batch_size, 3, device=self.device) * angle_stds
            # rots = torch.zeros((batch_size, 3), device=self.device)
            
            # Translations: Normal distribution centered at (0, 0, 0)
            # because center_volume=True re-zeroes the world coordinates.
            # trans = torch.randn(batch_size, 3, device=self.device) * dz # dz std dev
            trans_stds = torch.tensor(deltas, device=self.device)
            trans = torch.randn(batch_size, 3, device=self.device) * trans_stds
            # trans = torch.zeros((batch_size, 3), device=self.device)

            # dy is the 'Depth' / Source-to-Object distance.
            # We want the anatomy (at 0) to be between the source and detector.
            trans[:, 1] += 650
            # trans[:, 1] += (self.args.sdd / 2.0)

            for r, t in zip(rots, trans):
                # --- FILTER A: Intensity Variance ---
                # If the image is just a grey blob, std() will be very low.
                proj = drr(r.unsqueeze(0), t.unsqueeze(0), parameterization="euler_angles", convention="ZXY")
                
                proj = (proj - proj.min()) / (proj.max() - proj.min())
                
                s = sobel(proj)
                s = (s - s.min()) / (s.max() - s.min())
                edge_energy = (s > 0.4).float().mean()
                white_ratio = (proj > 0.8).float().mean()

                # 3. ROI CENTERING GUARD (The "Missing Target" Fix)
                # We check the center 50% of the image. 
                # If the center is significantly lighter than the whole, the bone is off-screen.
                p_norm = proj.squeeze()
                h, w = p_norm.shape
                center_roi = p_norm[h//4:3*h//4, w//4:3*w//4]
                if center_roi.mean() > p_norm.mean() + 0.1: 
                    # This means the center is "emptier" (brighter) than the rest
                    continue

                # 5. SYMMETRY CHECK (Prevents the "Glancing Blow" from one side)
                # Horizontal Symmetry (Left vs Right)
                left_mean = p_norm[:, :w//4].mean()
                right_mean = p_norm[:, 3*w//4:].mean()
                # Vertical Symmetry (Top vs Bottom)
                top_mean = p_norm[:h//4, :].mean()
                bottom_mean = p_norm[3*h//4:, :].mean()
                # Bias Check: If any one side is > 0.35 brighter than its opposite,
                # it's a "near-miss" or a "glancing blow" projection.
                if (abs(left_mean - right_mean) > 0.35) or (abs(top_mean - bottom_mean) > 0.35):
                    continue

                # # Stage 1: Centering (Check corners for air)
                # if self.is_off_center(p_norm):
                #     continue

                # # Unfold image into patches
                # patch_size=16
                # threshold=0.005
                # patches = p_norm.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
                # # Calculate variance per patch
                # patch_vars = patches.reshape(-1, patch_size * patch_size).var(dim=1)
                # # If more than 60% of patches are 'flat' blobs, discard
                # flat_patches = (patch_vars < threshold).float().mean()
                # if flat_patches > 0.60:
                #     continue

                # # Stage 3: Information Density (Check for blobs)
                # if self.check_local_variance(p_norm):
                #     continue

                # --- STEP F: SEGMENTED VARIANCE ---
                # To ensure the whole image isn't a blob, check variance in the 4 corners
                # If a corner is a "dead zone" (variance near 0), it might be a bad crop
                corners = [
                    p_norm[:h//4, :w//4],      # Top-Left
                    p_norm[:h//4, 3*w//4:],    # Top-Right
                    p_norm[3*h//4:, :w//4],    # Bottom-Left
                    p_norm[3*h//4:, 3*w//4:]   # Bottom-Right
                ]

                if any(c.var() < 0.001 for c in corners):
                    # This prevents images where a large portion is just 'dead' grey/white
                    continue

                # empty_corners = sum([(c > 0.95).float().mean() > 0.5 for c in corners])
                # if empty_corners > 1:
                #     continue

                if (proj.squeeze().std() < 0.12) or (edge_energy < 0.01) or (white_ratio > 0.05): # Adjust based on your normalized range
                    continue
                if len(valid_rots) < n_targets:
                    valid_rots.append(r.cpu())
                    valid_trans.append(t.cpu())
                    pbar.update(1)
                else:
                    break
        pbar.close()
        return torch.stack(valid_rots), torch.stack(valid_trans)

    def run(self):
        master_registry = []
        for entry in tqdm(self.index_data["entries"], desc="Patients"):
            pid = entry["id"]
            ct_path = Path(self.args.data_root) / entry["ct"]
            patient_dir = Path(self.args.output_dir) / f"patient_{pid}"
            patient_dir.mkdir(parents=True, exist_ok=True)
            
            # APPLYING YOUR REQUESTED CHANGES HERE:
            subj = read(volume=str(ct_path), orientation="AP", center_volume=True)
            
            drr = DRR(subj, sdd=self.args.sdd, height=self.args.size, delx=self.args.delx).to(self.device)
            
            # Sample centered, valid poses
            rots, trans = self.sample_valid_poses(drr, self.args.samples_per_patient)
            
            patient_metadata = {}
            for i in tqdm(range(len(rots)), desc=f"Rendering P_{pid}", leave=False):
                r, t = rots[i].unsqueeze(0).to(self.device), trans[i].unsqueeze(0).to(self.device)
                
                img_name = f"drr_{i:05d}.png"
                proj = drr(r, t, parameterization="euler_angles", convention="ZXY")
                normalize_and_save(proj, patient_dir / img_name)
                
                # Metadata is now simplified since (0,0,0) IS the center
                patient_metadata[img_name] = {
                    "pose": r.squeeze().tolist() + t.squeeze().tolist(),
                    "is_centered": True,
                    "orientation": "AP"
                }

            with open(patient_dir / "metadata.json", "w") as f:
                json.dump(patient_metadata, f, indent=4)
            master_registry.append({"id": pid, "folder": f"patient_{pid}", "samples": len(rots)})

        # Check if 'master_index.json' exists. If it does append the new data, otherwise create a new file.
        master_path = os.path.join(Path(self.args.output_dir), 'master_index.json')
        # Load existing
        if master_path.exists():
            with open(master_path, "r") as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    existing = []
        else:
            existing = []

        # Merge
        existing.extend(master_registry)

        # Save
        with open(master_path, "w") as f:
            json.dump(existing, f, indent=4)

    def run_nifti(self):
        """
        Traverse all nested subfolders under data_root.
        Each subfolder is expected to contain exactly one .nii.gz file.
        Each CT is reoriented to RAS and temporarily saved before DRR generation.
        """
        master_registry = []
        patient_id = 1

        # recursively find all nifti files
        nifti_files = sorted(Path(self.args.data_root).rglob("*.nii.gz"))
        nifti_files[0]

        if len(nifti_files) == 0:
            raise RuntimeError(f"No .nii.gz files found in {self.args.data_root}")

        for ct_path in tqdm(nifti_files, desc="Patients"):
            patient_dir = Path(self.args.output_dir) / f"patient_{patient_id:04d}"
            patient_dir.mkdir(parents=True, exist_ok=True)

            # -----------------------------
            # Load + convert to RAS
            # -----------------------------
            img = nib.load(str(ct_path))

            original_orientation = nio.ornt2axcodes(nio.io_orientation(img.affine))

            img_ras = nib.as_closest_canonical(img)

            # temporary file for diffdrr
            tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
            tmp_path = Path(tmp.name)

            nib.save(img_ras, str(tmp_path))

            # -----------------------------
            # DRR pipeline
            # -----------------------------
            subj = read(volume=str(tmp_path), orientation="AP", center_volume=True)

            drr = DRR(subj, sdd=self.args.sdd, height=self.args.size, delx=self.args.delx).to(self.device)

            rots, trans = self.sample_valid_poses(drr, self.args.samples_per_patient)

            patient_metadata = {}

            for i in tqdm(range(len(rots)), desc=f"Rendering P_{patient_id}", leave=False):
                r = rots[i].unsqueeze(0).to(self.device)
                t = trans[i].unsqueeze(0).to(self.device)

                img_name = f"drr_{i:05d}.png"

                proj = drr(r, t, parameterization="euler_angles", convention="ZXY")

                normalize_and_save(proj, patient_dir / img_name)

                patient_metadata[img_name] = {
                    "pose": r.squeeze().tolist() + t.squeeze().tolist(),
                    "is_centered": True,
                    "orientation": "AP"
                }

            with open(patient_dir / "metadata.json", "w") as f:
                json.dump(patient_metadata, f, indent=4)

            master_registry.append({
                "id": patient_id,
                "folder": f"patient_{patient_id:04d}",
                "samples": len(rots),
                "ct_path": str(ct_path)
            })

            # remove temporary file
            tmp_path.unlink(missing_ok=True)

            patient_id += 1

        master_path = Path(self.args.output_dir) / "master_index.json"

        if master_path.exists():
            with open(master_path, "r") as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    existing = []
        else:
            existing = []

        existing.extend(master_registry)

        with open(master_path, "w") as f:
            json.dump(existing, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-render DRR Dataset")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--index_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--samples_per_patient", type=int, default=5000, help="Max images per patient")
    parser.add_argument("--size", type=int, default=config.IMAGE_SIZE)
    parser.add_argument("--sdd", type=float, default=config.SDD)
    parser.add_argument("--delx", type=float, default=config.DELX)
    parser.add_argument("--nifti", action="store_true", help="Use .nii.gz files as input instead of DICOM.")
    args = parser.parse_args()
    if args.nifti:
        DRRDataGenerator(args).run_nifti()
    else:
        DRRDataGenerator(args).run()