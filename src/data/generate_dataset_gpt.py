import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import json
import argparse
import random
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
import cv2 

import tempfile
import nibabel as nib
import nibabel.orientations as nio

from diffdrr.data import read
from diffdrr.drr import DRR

from src.core.layers import Sobel
from src.utils import config


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def normalize_and_save(projection: torch.Tensor, path: Path, save: bool = True):
    proj = projection.detach().cpu().numpy().squeeze()
    mn, mx = proj.min(), proj.max()
    denom = max(mx - mn, 1e-6)
    
    
    if save:
        proj = 255.0 * (1.0 - (proj - mn) / denom)
        Image.fromarray(proj.astype(np.uint8)).save(path)
    else:
        proj = ((proj - mn) / denom)
        # proj = (1.0 - (proj - mn) / denom)
        return torch.tensor(proj).unsqueeze(0)

def hu_windowing(vol, low=-1000, high=2000):
    # for bone-focused DRR use low=200, high=1800
    vol = np.clip(vol, low, high)  # air floor, keep bone/metal
    vol = vol - low
    # return np.clip(vol, 0, None)
    return vol.astype(np.float32)

def load_volume(ct_path):
    img = nib.load(str(ct_path))
    img_ras = nib.as_closest_canonical(img)
    # img_ras = img
    vol = img_ras.get_fdata()

    vol = hu_windowing(vol, -200, 1500)
    new_affine = img_ras.affine

    return nib.Nifti1Image(vol.astype(np.float32), new_affine)

def rotation_distance(r1, r2):
    """
    r1, r2: (3,) Euler angles in radians (ZXY)
    returns angle difference in degrees
    """
    return torch.norm(r1 - r2).item() * (180.0 / torch.pi)


def translation_distance(t1, t2, depth_axis=1):
    """
    Split translation into in-plane and depth
    """
    axes = [0,1,2]
    axes.remove(depth_axis)

    in_plane_dist = torch.norm(t1[axes] - t2[axes]).item()
    depth_dist = abs(t1[depth_axis] - t2[depth_axis]).item()

    return in_plane_dist, depth_dist

# --------------------------------------------------
# Pose Sampler Class
# --------------------------------------------------

class CarmPoseSampler:
    """
    Models a real C-arm with anatomically valid pose constraints.
    
    Coordinate frame (diffDRR, orientation='AP'):
      X = Left/Right
      Y = Anterior/Posterior  (source starts here)
      Z = Superior/Inferior
      
      rx = CRA/CAU tilt   (rotates source in Y-Z plane)
      ry = in-plane roll  (nearly always ~0)
      rz = LAO/RAO        (rotates source in X-Y plane)
    """

    # -----------------------------------------------------------
    # Hard mechanical limits (radians)
    # -----------------------------------------------------------
    RAO_LAO_MAX   = np.deg2rad(45)   # rz
    CRA_CAU_MAX   = np.deg2rad(30)   # rx
    ROLL_MAX      = np.deg2rad(5)    # ry
    TRANS_XZ_MAX  = 30.0             # mm, isocenter shift
    SOURCE_DIST   = 500.0            # mm (SDD=1000 → source at 500 from iso)

    def __init__(self, device="cuda"):
        self.device = device

    # -----------------------------------------------------------
    # Sampling modes
    # -----------------------------------------------------------

    def sample_ap_zone(self, n=1):
        """Standard AP: pedicle screws, midline — most common intraop view."""
        rx = self._gauss(0,        np.deg2rad(8),  self.CRA_CAU_MAX, n)
        ry = self._gauss(0,        np.deg2rad(2),  self.ROLL_MAX,    n)
        rz = self._gauss(0,        np.deg2rad(10), np.deg2rad(20),   n)
        return self._make_pose(rx, ry, rz, n)

    def sample_oblique_zone(self, n=1):
        """Oblique: pedicle entry point, facet joints — 15-45° RAO/LAO."""
        sign = torch.randint(0, 2, (n,)) * 2 - 1   # randomly L or R
        rx = self._gauss(0,       np.deg2rad(8),  self.CRA_CAU_MAX,  n)
        ry = self._gauss(0,       np.deg2rad(2),  self.ROLL_MAX,     n)
        rz_mag = self._uniform(np.deg2rad(15), self.RAO_LAO_MAX, n)
        rz = sign.float().to(self.device) * rz_mag
        return self._make_pose(rx, ry, rz, n)

    def sample_lat_zone(self, n=1):
        """Lateral: implant depth, cage placement — source near Z-axis."""
        rx = self._gauss(np.deg2rad(90), np.deg2rad(8), np.deg2rad(105), n,
                         low=np.deg2rad(75))
        ry = self._gauss(0, np.deg2rad(2), self.ROLL_MAX, n)
        rz = self._gauss(0, np.deg2rad(5), np.deg2rad(15), n)
        return self._make_pose(rx, ry, rz, n)

    def sample_mixed(self, n=1):
        """Realistic intraop distribution — weighted by clinical frequency."""
        ap_n  = int(n * 0.50)   # 50% AP zone
        obl_n = int(n * 0.30)   # 30% oblique
        lat_n = n - ap_n - obl_n  # 20% lateral

        r_ap,  t_ap  = self.sample_ap_zone(ap_n)
        r_obl, t_obl = self.sample_oblique_zone(obl_n)
        r_lat, t_lat = self.sample_lat_zone(lat_n)

        r = torch.cat([r_ap, r_obl, r_lat], dim=0)
        t = torch.cat([t_ap, t_obl, t_lat], dim=0)

        # shuffle so patients don't have all AP then all oblique
        idx = torch.randperm(n)
        return r[idx], t[idx]

    # -----------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------

    def _make_pose(self, rx, ry, rz, n):
        r = torch.stack([rx, ry, rz], dim=1)           # (n, 3)

        tx = self._gauss(0, 10, self.TRANS_XZ_MAX, n)  # small isocenter shifts
        ty = torch.full((n,), self.SOURCE_DIST,device=self.device)              # fixed source distance
        tz = self._gauss(0, 8, self.TRANS_XZ_MAX, n)

        t = torch.stack([tx, ty, tz], dim=1)            # (n, 3)
        return r, t

    def _gauss(self, mean, std, clip, n, low=None):
        low = low if low is not None else -clip
        x = torch.randn(n, device=self.device) * std + mean
        return x.clamp(low, clip)

    def _uniform(self, low, high, n):
        return torch.rand(n, device=self.device) * (high - low) + low

# --------------------------------------------------
# Generator Class
# --------------------------------------------------

class DRRDataGenerator:
    def __init__(self, args):
        self.args = args
        self.device = config.DEVICE
        # self.base_depth = self.args.base_depth
        self.base_depth = config.SDD / 2
        self.batch_size = self.args.batch_size

        os.makedirs(args.output_dir, exist_ok=True)

        random.seed(42)
        torch.manual_seed(42)

        self.sampler = CarmPoseSampler(device=self.device)

    # --------------------------------------------------
    # Filtering
    # --------------------------------------------------

    def is_too_similar_per_axis(self, r, t, existing_rots, existing_trans, rot_thresh = None, trans_thresh = None):
        if rot_thresh is None:
            val = torch.deg2rad(torch.tensor(0.5))
            rot_thresh = torch.tensor([val, val, val])
        if rot_thresh is None:
            val = 2.0
            trans_thresh = torch.tensor([val, val])
            trans_thresh = trans_thresh[[0,2]]

        for r2, t2 in zip(existing_rots, existing_trans):

            rot_diff = torch.abs(r - r2)  # (3,)
            trans_diff = torch.abs(t - t2)  # (3,)
            trans_diff = trans_diff[[0,2]]

            # Check if ALL axes are within threshold
            try:
                if torch.all(rot_diff < rot_thresh) and torch.all(trans_diff < trans_thresh):
                    return True  # too similar
            except:
                return True
    
        return False

    def is_too_similar(r, t, existing_rots, existing_trans, depth_axis=1,
                    rot_thresh=5.0,
                    trans_thresh=8.0,
                    depth_thresh=20.0):

        for r2, t2 in zip(existing_rots, existing_trans):

            rot_diff = rotation_distance(r, r2)

            if rot_diff > rot_thresh:
                continue  # clearly different

            in_plane, depth = translation_distance(t, t2, depth_axis)

            if (in_plane < trans_thresh) and (depth < depth_thresh):
                return True

        return False

    def check_local_variance(self, p_norm, patch_size=16, threshold=0.005):
        patches = p_norm.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
        patch_vars = patches.reshape(-1, patch_size * patch_size).var(dim=1)
        flat_ratio = (patch_vars < threshold).float().mean()
        return flat_ratio > 0.60

    def is_valid_projection(self, proj, sobel, verbose=False):
        p_norm_thr = 0.12
        white_ratio_thr = 0.05
        edge_energy_thr = 0.005

        proj = normalize_and_save(proj, None, False)
        # print(proj.shape)

        s = sobel(proj.to(self.device))
        s = (s - s.min()) / (s.max() - s.min() + 1e-6)

        edge_energy = (s > 0.4).float().mean()
        white_ratio = (proj > 0.8).float().mean()

        p_norm = proj.squeeze()
        h, w = p_norm.shape

        # # Center ROI check
        center_roi = p_norm[h//4:3*h//4, w//4:3*w//4]
        # if center_roi.mean() > p_norm.mean() + 0.1:
        #     return False

        # Corner variance
        corners = [
            p_norm[:h//4, :w//4],
            p_norm[:h//4, 3*w//4:],
            p_norm[3*h//4:, :w//4],
            p_norm[3*h//4:, 3*w//4:]
        ]

        center_diff = (center_roi.mean() - p_norm.mean()).item()
        corner_vars = [c.var().item() for c in corners]
        std_val     = p_norm.std().item()
        reasons = []
        if center_diff > 0.1:          reasons.append(f"center_roi  diff={center_diff:.3f} > 0.10")
        if any(v < 0.001 for v in corner_vars): reasons.append(f"corner_var  min={min(corner_vars):.4f} < 0.001")
        if std_val      < p_norm_thr:        reasons.append(f"std         {std_val:.3f} < {p_norm_thr}")
        if edge_energy  < edge_energy_thr:        reasons.append(f"edge_energy {edge_energy:.4f} < {edge_energy_thr}")
        if white_ratio  > white_ratio_thr:        reasons.append(f"white_ratio {white_ratio:.3f} > {white_ratio_thr}")

        if reasons and verbose:
            print(f"  std={std_val:.3f}  edge={edge_energy:.4f}  "
                f"white={white_ratio:.3f}  center_diff={center_diff:.3f}  "
                f"corner_min={min(corner_vars):.4f}")
            for r in reasons:
                print(f"    ✗ REJECTED: {r}")

        if any(c.var() < 0.001 for c in corners):
            return False

        # Global checks
        # if (p_norm.std() < 0.12) or (edge_energy < 0.01) or (white_ratio > 0.05):
        if (p_norm.std() < p_norm_thr) or (edge_energy < edge_energy_thr) or (white_ratio > white_ratio_thr):
            return False

        # Optional (kept but not forced)
        # if self.check_local_variance(p_norm):
        #     return False

        return True

    # --------------------------------------------------
    # Pose Sampling (Method 1: Mixture Gaussian)
    # --------------------------------------------------

    def sample_pose_mixture(self):
        # mode = random.random()
        mode = 1

        if mode < 0.7:
            angle_std = torch.tensor([0.15, 0.15, 0.15], device=self.device)
            trans_std = torch.tensor([20, 8, 20], device=self.device)
        elif mode < 0.9:
            angle_std = torch.tensor([0.25, 0.25, 0.25], device=self.device)
            trans_std = torch.tensor([30, 12, 30], device=self.device)
        else:
            angle_std = torch.tensor([0.3, 0.3, 0.3], device=self.device)
            trans_std = torch.tensor([40, 15, 40], device=self.device)

        r = torch.randn(self.batch_size, 3, device=self.device) * angle_std
        t = torch.randn(self.batch_size, 3, device=self.device) * trans_std

        t[:, 1] += self.base_depth  # e.g. 650mm

        return r, t

    # --------------------------------------------------
    # Pose Sampling (Method 2: Uniform grid-ish)
    # --------------------------------------------------

    def sample_pose_uniform(self):
        r = (torch.rand(1, 3, device=self.device) - 0.5) * 2 * 0.3
        t = (torch.rand(1, 3, device=self.device) - 0.5)

        t *= torch.tensor([40, 15, 40], device=self.device)
        t[:, 1] += self.args.base_depth

        return r, t

    # --------------------------------------------------
    # Pose sampler wrapper
    # --------------------------------------------------

    def sample_valid_poses(self, drr, n_targets):
        valid_rots = []
        valid_trans = []
        num_similar = 0

        sobel = Sobel().to(self.device)

        max_attempts = n_targets * 1000
        attempts = 0

        pbar = tqdm(total=n_targets, desc="Sampling Poses", leave=False)

        while len(valid_rots) < n_targets and attempts < max_attempts:
            attempts += 1
            if self.args.sampling_method == "orbit":
                self.sample_pose_camOrb()
            if self.args.sampling_method == "mixture":
                r, t = self.sample_pose_mixture()
            elif self.args.sampling_method == "uniform":
                r, t = self.sample_pose_uniform()
            else:
                r, t = self.sampler.sample_mixed(n=10)

            for r1, t1 in zip(r, t):
                r_cpu = r1.detach().cpu()
                t_cpu = t1.detach().cpu()

                if self.is_too_similar_per_axis(r_cpu, t_cpu, valid_rots, valid_trans):
                    num_similar += 1
                    continue

            proj = drr(r, t, parameterization="euler_angles", convention="ZXY")

            if self.is_valid_projection(proj, sobel):
                valid_rots.append(r.squeeze().cpu())
                valid_trans.append(t.squeeze().cpu())
                pbar.update(1)

        pbar.close()

        if num_similar > 0:
            print(f"{num_similar} similar poses filtered out")

        if len(valid_rots) < n_targets:
            print(f"Warning: only {len(valid_rots)} valid samples found")

        return torch.stack(valid_rots), torch.stack(valid_trans)

    # --------------------------------------------------
    # Pose sampler using a camera-orbit strategy (from GPT)
    # --------------------------------------------------

    def sample_pose_camOrb(self):
        """
        Sample a batch of poses.

        The camera is positioned on a spherical arc around the spine and oriented
        to look at the spine center.

        Returns:
            rot (torch.Tensor): Euler angles (ZXY convention), shape (B, 3), radians.
            trans (torch.Tensor): Translations (camera position), shape (B, 3), mm.
        """
        B = self.batch_size
        device = self.device

        # Sample orbital (theta) and tilt (phi) angles
        theta = torch.randn(B, device=device) * 20 * (torch.pi / 180)
        phi   = torch.randn(B, device=device) * 10 * (torch.pi / 180)

        # Sample distance from spine
        r = self.base_depth + torch.randn(B, device=device) * 15

        # Convert spherical coordinates to Cartesian (camera position)
        x = r * torch.cos(phi) * torch.sin(theta)
        y = r * torch.cos(phi) * torch.cos(theta)  # depth axis
        z = r * torch.sin(phi)

        trans = torch.stack([x, y, z], dim=1)

        # Compute forward direction (camera → spine)
        target = torch.tensor([0, self.base_depth, 0], device=device)
        forward = target - trans
        forward = forward / torch.norm(forward, dim=1, keepdim=True)

        # Build orthonormal basis (right, up, forward)
        up = torch.tensor([0, 0, 1], device=device).expand_as(forward)

        right = torch.cross(up, forward, dim=1)
        right = right / torch.norm(right, dim=1, keepdim=True)

        up = torch.cross(forward, right, dim=1)

        # Rotation matrix from basis vectors
        R = torch.stack([right, up, forward], dim=2)

        # Convert rotation matrix to Euler angles (ZXY)
        rot = self.rotation_matrix_to_euler_zxy(R)

        return rot, trans

    # --------------------------------------------------
    # Main NIfTI pipeline
    # --------------------------------------------------

    def run_nifti(self):
        master_registry = []
        patient_id = self.args.index
        # patient_id = 7

        nifti_files = sorted(Path(self.args.data_root).rglob("*.nii.gz"))
        # import re
        # patient_ids = [int(re.search(r"patient_(\d+)", f.parent.name).group(1)) for f in nifti_files]

        for idx, ct_path in enumerate(tqdm(nifti_files, desc="Patients")):
            # patient_id = patient_ids[idx]
            patient_dir = Path(self.args.output_dir) / f"patient_{patient_id:02d}/DRR_new"
            patient_dir.mkdir(parents=True, exist_ok=True)

            img = nib.load(str(ct_path))
            img_ras = nib.as_closest_canonical(img)
            # img_ras = img
            vol = img_ras.get_fdata()

            vol = hu_windowing(vol, -200, 1500)
            new_affine = img_ras.affine

            vol = nib.Nifti1Image(vol.astype(np.float32), new_affine)

            tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
            tmp_path = Path(tmp.name)

            try:
                # nib.save(img_ras, str(tmp_path))
                nib.save(vol, str(tmp_path))

                subj = read(volume=str(tmp_path), orientation="AP", center_volume=True)

                # Multi-DELX option
                delx = random.choice(self.args.delx_values)

                drr = DRR(subj, sdd=self.args.sdd, height=self.args.size, delx=delx).to(self.device)

                # # 1. Render the 3 canonical views and save them side by side
                # for name, (rx, ry, rz) in [("AP", (0,0,0)), ("LAT", (np.pi/2,0,0)), ("OBL", (0,0.785,0))]:
                #     r = torch.tensor([[rx, ry, rz]], dtype=torch.float32).to(self.device)
                #     t = torch.tensor([[0, 500, 0]], dtype=torch.float32).to(self.device)
                #     proj = drr(r, t, parameterization="euler_angles", convention="ZXY")
                #     normalize_and_save(proj, patient_dir / f"canonical_{name}.png")

                rots, trans = self.sample_valid_poses(drr, self.args.samples_per_patient)
                # rots, trans = self.sampler.sample_mixed(n=self.args.samples_per_patient)
                # rots, trans = self.sample_pose_mixture()

                patient_metadata = {}

                for i in tqdm(range(len(rots)), desc=f"P_{patient_id}", leave=False):
                    r = rots[i].unsqueeze(0).to(self.device)
                    t = trans[i].unsqueeze(0).to(self.device)
                    img_name = f"drr_{i:05d}.png"

                    proj = drr(r, t, parameterization="euler_angles", convention="ZXY")

                    normalize_and_save(proj, patient_dir / img_name)

                    patient_metadata[img_name] = {
                        "pose": r.squeeze().tolist() + t.squeeze().tolist(),
                        "is_centered": True,
                        "orientation": "AP",
                        "sdd": self.args.sdd,
                        "delx": delx
                    }

                with open(patient_dir / "metadata.json", "w") as f:
                    json.dump(patient_metadata, f, indent=4)

                master_registry.append({
                    "id": patient_id,
                    "folder": f"patient_{patient_id:03d}",
                    "samples": len(rots),
                    "ct_path": str(ct_path)
                })

                patient_id += 1

            finally:
                tmp_path.unlink(missing_ok=True)

        master_path = Path(self.args.output_dir) / "master_index.json"

        if master_path.exists():
            with open(master_path, "r") as f:
                try:
                    existing = json.load(f)
                except:
                    existing = []
        else:
            existing = []

        existing.extend(master_registry)

        with open(master_path, "w") as f:
            json.dump(existing, f, indent=4)


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--samples_per_patient", type=int, default=5000)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--index", type=int, default=1)

    parser.add_argument("--sdd", type=float, default=1000.0)

    # 🔥 Multi DELX support
    # parser.add_argument("--delx_values", nargs="+", type=float, default=[config.DELX])
    parser.add_argument("--delx_values", nargs="+", type=float, default=[])

    parser.add_argument("--base_depth", type=float, default=650.0)

    parser.add_argument("--sampling_method", type=str, default="mixture", choices=["mixture", "uniform", "orbit"])

    args = parser.parse_args()

    DRRDataGenerator(args).run_nifti()