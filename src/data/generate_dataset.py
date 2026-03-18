import os
import json
import math
import argparse
import itertools
import random
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image

from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.pose import convert
from diffdrr.renderers import _get_alphas, _get_alpha_minmax

from src.utils import config

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def normalize_and_save(projection: torch.Tensor, path: Path):
    """Normalizes to [0, 255] and saves as PNG."""
    proj = projection.detach().cpu().numpy().squeeze()
    mn, mx = proj.min(), proj.max()
    # Invert and normalize (standard for X-ray/DRR visualization)
    proj = 255.0 * (1.0 - (proj - mn) / (mx - mn + 1e-8))
    img = Image.fromarray(proj.astype(np.uint8))
    img.save(path)

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
        # Set seed for reproducibility
        random.seed(42)
        torch.manual_seed(42)

    def build_grid(self, steps):
        """Standardizes the pose search space."""
        return {
            "yaw": torch.linspace(-math.pi / 3, math.pi / 3, steps),
            "pitch": torch.linspace(-math.pi / 3, math.pi / 3, steps),
            "roll": torch.linspace(-math.pi / 2, math.pi / 2, steps),
            "dx": torch.linspace(-35, 35, steps),
            "dy": torch.linspace(300, 600, steps),
            "dz": torch.linspace(-50, 50, steps),
        }

    @torch.no_grad()
    def get_valid_indices(self, drr, grid, min_intersections):
        """Only returns indices where the DRR actually hits the CT volume."""
        all_idx = list(itertools.product(range(self.args.steps), repeat=6))
        valid = []
        
        batch_size = 256  # Increased for faster GPU utilization
        for i in range(0, len(all_idx), batch_size):
            batch = all_idx[i : i + batch_size]
            
            rots = torch.tensor([[grid["yaw"][b[0]], grid["pitch"][b[1]], grid["roll"][b[2]]] for b in batch], device=self.device)
            trans = torch.tensor([[grid["dx"][b[3]], grid["dy"][b[4]], grid["dz"][b[5]]] for b in batch], device=self.device)
            
            poses = convert(rots, trans, parameterization="euler_angles", convention="ZXY")
            source, target = drr.detector(poses, None)
            target = target.mean(dim=1, keepdim=True)
            
            source = drr.affine_inverse(source)
            target = drr.affine_inverse(target)
            
            dims = drr.renderer.dims(drr.subject.density.data.squeeze())
            alphas = _get_alphas(source, target, dims, drr.renderer.eps, False)
            alphamin, alphamax = _get_alpha_minmax(source, target, dims, drr.renderer.eps)

            lengths = ((alphamin <= alphas) & (alphas <= alphamax)).squeeze(1).sum(dim=-1)
            keep = torch.nonzero(lengths > min_intersections).squeeze(-1).tolist()
            
            for k in keep:
                valid.append(batch[k])
        return valid

    def sample_random_poses(self, drr, num_samples):
        center_world = self.get_volume_center(drr)
        
        # Sample rotations (Normal distribution around 0)
        rots = torch.randn(num_samples, 3) * 0.2 # Std dev in radians
        
        # Sample translations (Normal distribution around the volume center)
        trans = torch.randn(num_samples, 3) * 20.0 # Std dev in mm
        trans += center_world 
        
        return rots, trans

    def run(self):
        grid = self.build_grid(self.args.steps)
        master_registry = []

        for entry in tqdm(self.index_data["entries"], desc="Patients"):
            pid = entry["id"]
            ct_path = Path(self.args.data_root) / entry["ct"]
            
            patient_dir = Path(self.args.output_dir) / f"patient_{pid}"
            patient_dir.mkdir(parents=True, exist_ok=True)
            
            # subj = read(str(ct_path))
            subj = read(volume=str(ct_path), orientation="AP", center_volume=True)
            
            drr = DRR(subj, sdd=self.args.sdd, height=self.args.size, delx=self.args.delx).to(self.device)
            
            # 1. Get all valid poses
            valid_indices = self.get_valid_indices(drr, grid, self.args.min_intersections)
            
            # 2. Apply Random Sampling
            if len(valid_indices) > self.args.samples_per_patient:
                print(f"Patient {pid}: Sampling {self.args.samples_per_patient} from {len(valid_indices)} valid poses.")
                valid_indices = random.sample(valid_indices, self.args.samples_per_patient)
            else:
                print(f"Patient {pid}: Found {len(valid_indices)} valid poses (below limit).")
            
            patient_metadata = {}

            # 3. Render and Save
            for idx, pose_idx in enumerate(tqdm(valid_indices, desc=f"Rendering P_{pid}", leave=False)):
                yaw, pitch, roll = grid["yaw"][pose_idx[0]], grid["pitch"][pose_idx[1]], grid["roll"][pose_idx[2]]
                dx, dy, dz = grid["dx"][pose_idx[3]], grid["dy"][pose_idx[4]], grid["dz"][pose_idx[5]]
                
                rot = torch.tensor([[yaw, pitch, roll]], device=self.device)
                trans = torch.tensor([[dx, dy, dz]], device=self.device)
                
                img_name = f"drr_{idx:05d}.png"
                img_path = patient_dir / img_name
                
                proj = drr(rot, trans, parameterization="euler_angles", convention="ZXY")
                normalize_and_save(proj, img_path)
                
                patient_metadata[img_name] = {
                    "pose": [yaw.item(), pitch.item(), roll.item(), dx.item(), dy.item(), dz.item()],
                    "params": {"sdd": self.args.sdd, "delx": self.args.delx, "size": self.args.size}
                }

            with open(patient_dir / "metadata.json", "w") as f:
                json.dump(patient_metadata, f, indent=4)
            
            master_registry.append({"id": pid, "folder": f"patient_{pid}", "num_samples": len(valid_indices)})

        with open(Path(self.args.output_dir) / "master_index.json", "w") as f:
            json.dump(master_registry, f, indent=4)

# --------------------------------------------------
# CLI
# --------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-render DRR Dataset")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--index_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--steps", type=int, default=8, help="Grid search density")
    parser.add_argument("--samples_per_patient", type=int, default=2000, help="Max images per patient")
    parser.add_argument("--size", type=int, default=config.IMAGE_SIZE)
    parser.add_argument("--sdd", type=float, default=config.SDD)
    parser.add_argument("--delx", type=float, default=config.DELX)
    parser.add_argument("--min_intersections", type=int, default=750)
    
    args = parser.parse_args()
    generator = DRRDataGenerator(args).run()