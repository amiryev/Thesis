import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from diffdrr.drr import DRR
from diffdrr.data import read
from diffdrr.pose import euler_angles_to_matrix, rotation_6d_to_matrix

from src.core.pose_regressor import PoseRegressor
from src.core.layers import Sobel
from src.utils import config, image_processing
from src.utils.loss import mGNCCLoss, compute_geodesic_distance


class PoseGenerator(nn.Module):
    """
    Lightweight MLP to generate 6-DoF pose updates from a latent vector.
    """
    def __init__(self, latent_dim=32, hidden_dim=64, out_dim=9):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, z):
        return self.net(z)

class PoseRegressorOptimizer(nn.Module):
    def __init__(self, drr=None, latent_dim=32, hidden_dim=16, max_translation=10.0, max_rotation=0.1):
        """
        Pose Regressor Optimizer optimizing a small neural network to output 
        pose updates from a fixed random latent vector, starting from an initial
        pose predicted by a PoseRegressor.
        """
        super().__init__()
        if drr is None:         
            subject = read(volume=str(self.ct_volume_path), orientation="AP", center_volume=True)
            self.drr = DRR(subject, sdd=config.SDD, height=config.IMAGE_SIZE, delx=config.DELX).to(self.device)
        else:
            self.drr = drr
        
        # Instantiate Pose Regressor locally
        self.pose_regressor = PoseRegressor()
        self.pose_regressor.eval()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Scaling limits (6 DoF: 3 rotation, 3 translation)
        self.max_translation = max_translation
        self.max_rotation = max_rotation

        self.mgncc = mGNCCLoss()
        self.kernel = 1
        self.carm = None

    def update_carm(self, carm):
        """
        Update the current target CRM image for this optimization run.
        """
        self.carm = carm

    def split_prediction(self, pose, parameterization=None):
        if pose.shape[1] == 6:
            rot = pose[:, :3]
            trans = pose[:, 3:]
            projection = self.drr(rot, trans, parameterization="euler_angles", convention="ZXY") if parameterization is not None else None
        elif pose.shape[1] == 9:
            rot = pose[:, :6]
            trans = pose[:, 6:]
            projection = self.drr(rot, trans, parameterization=parameterization) if parameterization is not None else None

        return rot, trans, projection

    def render_drr(self, pose, parameterization="euler_angles"):
        """
        Render a generic DRR from the pose and normalize similarly to PositionEstimator.
        """

        if parameterization is "euler_angles":
            rot = pose[:, :3]
            trans = pose[:, 3:]
            projection = self.drr(rot, trans, parameterization="euler_angles", convention="ZXY")
        else:
            rot = pose[:, :6]
            trans = pose[:, 6:]            
            projection = self.drr(rot, trans, parameterization=parameterization)
        mn = projection.amin(dim=(-2, -1), keepdim=True) 
        mx = projection.amax(dim=(-2, -1), keepdim=True) 
        projection = 1 - (projection - mn) / (mx - mn)
        return projection

    def compute_loss(self, gt_pose, pred_pose):
            if gt_pose is None:
                return None
            
            euler_gt = gt_pose[:, :3]
            trans_gt = gt_pose[:, 3:]

            rot_6d_pred, trans_pred, _ = self.split_prediction(pred_pose)

            rot_matrix_gt = euler_angles_to_matrix(euler_gt, convention="ZXY")

            rot_matrix_pred = rotation_6d_to_matrix(rot_6d_pred)

            loss_rot = compute_geodesic_distance(rot_matrix_pred, rot_matrix_gt).mean()
            
            trans_scale = torch.tensor([40, 50, 40], device=config.DEVICE)
            loss_trans_by_axis = F.smooth_l1_loss(trans_pred, trans_gt, reduction='none')  / trans_scale
            loss_trans = loss_trans_by_axis.mean()

            trans_weight = trans_scale = torch.tensor([1.0, 0.2, 1.0], device=config.DEVICE)
            loss = loss_rot + (trans_weight * loss_trans_by_axis).mean()

            return loss.item(), loss_rot.item(), loss_trans.item()

    def forward(
        self,
        model = None,
        gt_pose = None,
        input_vec = None,
        lr: float = 1e-3,
        iters: int = 250,
        patience: int = 25,
        min_delta: float = 1e-3,
        iterative: bool = True,
        verbose: bool = True,
    ):
        if self.carm is None:
            raise RuntimeError("C-arm not set. Call update_carm() first.")

        # Initial prediction using the PoseRegressor
        with torch.no_grad():
            rotation_6d, translation = self.pose_regressor(self.carm)
            
            # Convert 6D Rotation to Matrix to Euler (ZXY)
            rotation_matrix = image_processing.rotation_6d_to_matrix(rotation_6d)
            euler_angles = image_processing.matrix_to_euler_angles(rotation_matrix, convention="ZXY")
            
            # Form final 6-element pose expected by render_drr
            # initial_pose = torch.cat([euler_angles, translation], dim=-1)
            initial_pose = torch.cat([rotation_6d, translation], dim=-1)
            
            projection = self.render_drr(initial_pose, parameterization="rotation_6d")
            
        device = initial_pose.device
        B = initial_pose.shape[0]

        # Initialize Pose Generator
        if model is None:
            model = PoseGenerator(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, maximize=True)
        
        # Fixed random latent vector
        if input_vec is None:
            z = torch.randn(B, self.latent_dim, device=device)
        else:
            z = input_vec.clone()
        # Constraint: explicitly constant
        z.requires_grad = False
        
        if model.out_dim == 6:
            max_range = torch.tensor([
                self.max_rotation, self.max_rotation, self.max_rotation,
                self.max_translation, self.max_translation, self.max_translation
            ], device=device).unsqueeze(0)  # Shape: (1, 6)
        elif model.out_dim == 9:
                max_range = torch.tensor([
                self.max_rotation, self.max_rotation, self.max_rotation, self.max_rotation, self.max_rotation, self.max_rotation,
                self.max_translation, self.max_translation, self.max_translation
            ], device=device).unsqueeze(0)  # Shape: (1, 9)

        best_pose = initial_pose.detach().clone()
        best_gain = torch.full((B,), -float('inf'), device=device)
        best_projection = projection.clone().detach()
        best_model = copy.deepcopy(model.state_dict())

        no_improve  = 0
        init_gain = 0.0
        
        init_results = []
        final_results = []

        current_pose = initial_pose.detach().clone()

        for step in range(iters):
            optimizer.zero_grad()
            
            # Predict delta pose from the fixed latent vector
            raw_output = model(z)
            
            # Scale output with tanh (acts as clipping/clamping)
            delta_pose = max_range * torch.tanh(raw_output)
            
            if iterative:
                optimize_pose = current_pose + delta_pose
            else:
                optimize_pose = initial_pose.detach() + delta_pose
                
            # Render DRR using local logic
            optimize_projection = self.render_drr(optimize_pose, parameterization="rotation_6d")
            
            # Optimization objective: Maximize Gain (mGNCC)
            gain = self.mgncc(optimize_projection, self.carm, kernel=self.kernel)
            
            if step == 0:
                init_gain = gain.max().item()
                init_results = [init_gain, initial_pose.clone(), 0.0, 0.0]

            # Update weights Using Adam
            gain.sum().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            losses = self.compute_loss(gt_pose=gt_pose, pred_pose=optimize_pose)

            if iterative:
                # Prepare for next iteration
                current_pose = optimize_pose.detach()

            # Tracking best
            improved = torch.logical_or(torch.isneginf(best_gain), gain > best_gain + min_delta * best_gain.abs())
            if improved.any():
                no_improve = 0
                best_projection[improved] = optimize_projection[improved].detach().clone()
                best_pose[improved] = optimize_pose[improved].detach().clone()
                best_gain[improved] = gain[improved].detach().clone()
            else:
                no_improve += 1

            if verbose:
                print(f"[{step:03d}] losses={losses} | gain={gain.max():.5f} | best_max={best_gain.max():.5f} | no_improve={no_improve}")
                
            if no_improve >= patience or step == iters - 1 or torch.isnan(gain).any():
                model.load_state_dict(best_model)
                if verbose:
                    print(f"Stop at step {step} (patience={patience})")
                break

        final_results = [best_gain.max().item(), best_pose.clone(), 0.0, 0.0]

        return best_pose, best_projection, step-patience, init_results, final_results




class LBFGSOptimizer(nn.Module):
    def __init__(self, ct_path, loss:callable=mGNCCLoss(), rot_dim:int=3, lr=(1e-3, 5e-3), weight_decay=0.0, scales:int=3, device=config.DEVICE):
        """
        Refines an initial pose using gradient-based optimization
        to maximize mGNCC similarity between DRR and C-arm image.

        Args:
            drr: Differentiable DRR renderer
            pose_regressor: pretrained model giving initial pose
            max_translation: clamp for translation updates
            max_rotation: clamp for rotation updates (radians)
            lambda_reg: regularization weight
            device: torch device
        """
        super().__init__()
        self.device = device
        self.criterion = loss
        self.rot_dim = rot_dim
        self.lr = lr
        self.weight_decay = weight_decay

        self.subject = read(volume=str(ct_path), orientation="AP", center_volume=True)

        # coarse → fine pyramid
        self.scales = [2**(s-1) for s in range(scales, 0, -1)]

    # --------------------------------------------------
    # Utility functions
    # --------------------------------------------------
    def normalize(self, img, invert=True):
        """Min-max normalize safely"""
        eps = 1e-6
        mn = img.amin(dim=(-2, -1), keepdim=True)
        mx = img.amax(dim=(-2, -1), keepdim=True)
        if invert:
            out = 1.0 - (img - mn) / (mx - mn + eps)
        else:
            out = (img - mn) / (mx - mn + eps)
        return out

    def resize(self, img, size):
        """Simple blur via avg pooling"""
        if size == img.shape[1]:
            return img
        return F.interpolate(img, size=(size, size), mode='bilinear', align_corners=False)

    def render_drr(self, pose, drr=None, img_size=config.IMAGE_SIZE):
        """Render DRR from pose"""
        rot = pose[:, :3]
        trans = pose[:, 3:]

        if drr is None:
            delx = 305 / img_size
            print(f"Rendered image size is {img_size}")
            drr = DRR(self.subject, sdd=config.SDD, height=img_size, delx=delx).to(self.device)
        proj = drr(rot, trans, parameterization="euler_angles", convention="ZXY")

        return self.normalize(proj).to(self.device)

    # --------------------------------------------------
    # Main Optimization
    # --------------------------------------------------
    def forward(self, carm, initial_pose, kernel=1.0, iters_per_scale=100, patience=15, verbose=True):
        """
        Args:
            carm: input DRR / C-arm image (B=1)
        Returns:
            best_pose: optimized pose (1, 6)
            best_gain: final similarity score
        """
        min_delta = 1e-4
        B, C, H, W = carm.shape
        history = {}

        # ----------------------------------
        # Optimization variable
        # ----------------------------------
        delta_pose = nn.Parameter(torch.zeros_like(initial_pose), requires_grad=True)
        # delta_rot = nn.Parameter(torch.zeros_like(initial_pose[:, :self.rot_dim]), requires_grad=True)
        # delta_trans = nn.Parameter(torch.zeros_like(initial_pose[:, self.rot_dim:]), requires_grad=True)
        # torch.nn.init.kaiming_normal_(delta_pose)

        # optimizer = torch.optim.LBFGS([
        #     {"params": delta_rot,   "lr": self.lr[0], "weight_decay": self.weight_decay},
        #     {"params": delta_trans, "lr": self.lr[1], "weight_decay": self.weight_decay},
        # ])
        optimizer = torch.optim.LBFGS([delta_pose], lr=self.lr[0])

        # # Pose limits
        # max_range = torch.tensor([
        #     self.max_rotation, self.max_rotation, self.max_rotation,
        #     self.max_translation, self.max_translation, self.max_translation
        # ], device=self.device).unsqueeze(0)


        # Tracking
        best_gain = -float("inf")
        best_pose = initial_pose.clone()
        no_improve = 0

        initial_pose = initial_pose.detach()

        # ----------------------------------
        # Multi-scale optimization
        # ----------------------------------
        for scale in self.scales:
            img_size = H // scale
            drr = DRR(self.subject, sdd=config.SDD, height=img_size, delx=config.DELX).to(self.device)

            carm_s = self.resize(carm, img_size).detach()

            if verbose:
                print(f"\n--- Scale {scale} ---")

            for step in range(iters_per_scale):
                # print(fr"$\delta$ pose: {delta_pose}")
                def closure():
                    optimizer.zero_grad()

                    pose = initial_pose + delta_pose   # ❗ no detach

                    proj = self.render_drr(pose, drr)
                    proj = proj * kernel
                    proj_s = self.resize(proj, img_size)

                    gain = self.criterion(proj_s, carm_s)
                    loss = 1 - gain.mean()

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_([delta_pose], 1.0)

                    return loss
                
                loss = optimizer.step(closure=closure)

                # Safety check
                if not torch.isfinite(loss):
                    print("Numerical instability detected. Stopping.")
                    break

                with torch.no_grad():
                    pose = initial_pose + delta_pose
                    proj = self.render_drr(pose, drr)
                    proj_s = self.resize(proj, img_size)
                    gain = self.criterion(proj_s, carm_s)
                gain_val = gain.item()
                print("grad norm:", delta_pose.grad.norm().item())

                # Track best
                if gain_val > best_gain + min_delta:
                    best_gain = gain_val
                    best_pose = pose.detach().clone()
                    no_improve = 0
                else:
                    no_improve += 1

                # Store history
                history.setdefault("gain",       []).append(gain_val)
                history.setdefault("delta_pose", []).append(delta_pose.detach().clone().cpu().tolist())
                history.setdefault("estimated_pose", []).append(pose.detach().clone().cpu().tolist())

                if verbose:
                    print(
                        f"[{step:03d}] "
                        f"gain={gain_val:.5f} "
                        f"best={best_gain:.5f} "
                        f"no_improve={no_improve}"
                    )

                # Early stopping
                if no_improve >= patience:
                    if verbose:
                        print("Early stopping triggered")
                    break

        return best_pose, best_gain, history


class AdamOptimizer(nn.Module):
    def __init__(self, ct_path, loss:callable=mGNCCLoss(), rot_dim:int=3, lr=(1e-3, 5e-3), weight_decay=0.0, scales:int=3, device=config.DEVICE):
        """
        Refines an initial pose using gradient-based optimization
        to maximize mGNCC similarity between DRR and C-arm image.

        Args:
            drr: Differentiable DRR renderer
            pose_regressor: pretrained model giving initial pose
            max_translation: clamp for translation updates
            max_rotation: clamp for rotation updates (radians)
            lambda_reg: regularization weight
            device: torch device
        """
        super().__init__()
        self.device = device
        self.criterion = loss
        self.rot_dim = rot_dim
        self.lr = lr
        self.weight_decay = weight_decay

        self.subject = read(volume=str(ct_path), orientation="AP", center_volume=True)

        # coarse → fine pyramid
        self.scales = [2**(s-1) for s in range(scales, 0, -1)]

    # --------------------------------------------------
    # Utility functions
    # --------------------------------------------------
    def normalize(self, img, invert=True):
        """Min-max normalize safely"""
        eps = 1e-6
        mn = img.amin(dim=(-2, -1), keepdim=True)
        mx = img.amax(dim=(-2, -1), keepdim=True)
        if invert:
            out = 1.0 - (img - mn) / (mx - mn + eps)
        else:
            out = (img - mn) / (mx - mn + eps)
        return out

    def resize(self, img, size):
        """Simple blur via avg pooling"""
        if size == img.shape[1]:
            return img
        return F.interpolate(img, size=(size, size), mode='bilinear', align_corners=False)

    def render_drr(self, pose, drr=None, img_size=config.IMAGE_SIZE):
        """Render DRR from pose"""
        rot = pose[:, :3]
        trans = pose[:, 3:]

        if drr is None:
            delx = 305 / img_size
            print(f"Rendered image size is {img_size}")
            drr = DRR(self.subject, sdd=config.SDD, height=img_size, delx=delx).to(self.device)
        proj = drr(rot, trans, parameterization="euler_angles", convention="ZXY")

        return self.normalize(proj).to(self.device)

    def find_closest_depth(self, carm, pose:torch.tensor, max_dist=100, size=10):
        img_size = carm.shape[2]
        depth = pose[0, -2]
        depth_list = torch.linspace(depth - max_dist, depth + max_dist, size)
        poses = pose.repeat(size, 1)
        poses[:, -2] = depth_list

        carm_s = self.resize(carm, img_size).detach()

        
        drrs = self.render_drr(poses, img_size=img_size)
        carms = torch.cat(size * [carm_s])
        # carms = carm_s.repeat(size)
        gain = self.criterion(drrs, carms)
        index = torch.argmax(gain)
        print(pose)
        print(gain)
        print(poses[index])
        return poses[index].unsqueeze(0)
    
    # --------------------------------------------------
    # Main Optimization
    # --------------------------------------------------
    def forward(self, carm, initial_pose, kernel=1.0, iters_per_scale=100, patience=15, verbose=True):
        """
        Args:
            carm: input DRR / C-arm image (B=1)
        Returns:
            best_pose: optimized pose (1, 6)
            best_gain: final similarity score
        """
        min_delta = 1e-4
        B, C, H, W = carm.shape
        history = {}
        # trans_scale = 100
        trans_scale = torch.tensor([10, 200, 10], device=self.device, requires_grad=False)

        # ----------------------------------
        # Optimization variable
        # ----------------------------------
        # delta_pose = nn.Parameter(torch.zeros_like(initial_pose), requires_grad=True)
        delta_rot = nn.Parameter(torch.zeros_like(initial_pose[:, :self.rot_dim]), requires_grad=True)
        delta_trans = nn.Parameter(torch.zeros_like(initial_pose[:, self.rot_dim:]), requires_grad=True)
        # delta_trans = nn.Parameter(torch.zeros((1,2), device=self.device), requires_grad=True)
        # torch.nn.init.kaiming_normal_(delta_pose)

        optimizer = torch.optim.AdamW([
            {"params": delta_rot,   "lr": self.lr[0], "weight_decay": self.weight_decay},
            {"params": delta_trans, "lr": self.lr[1], "weight_decay": self.weight_decay},
        ])


        # # Pose limits
        # max_range = torch.tensor([
        #     self.max_rotation, self.max_rotation, self.max_rotation,
        #     self.max_translation, self.max_translation, self.max_translation
        # ], device=self.device).unsqueeze(0)

        # Estimate best depth
        # initial_pose = self.find_closest_depth(carm, initial_pose)

        # Tracking
        best_gain = -float("inf")
        best_pose = initial_pose.clone()
        no_improve = 0

        initial_pose = initial_pose.detach()

        # ----------------------------------
        # Multi-scale optimization
        # ----------------------------------
        for scale in self.scales:
            no_improve = 0
            img_size = H // scale
            delx = 305 / img_size
            print(f"Rendered image size is {H}")
            drr = DRR(self.subject, sdd=config.SDD, height=img_size, delx=delx).to(self.device)

            carm_s = self.resize(carm, img_size).detach()
            if not type(kernel) is float:
                import torchvision.transforms.functional as TF
                kernel = TF.resize(kernel, img_size).detach()

            # Tracking
            best_gain = -float("inf")
            best_pose = initial_pose.clone()
            no_improve = 0

            if verbose:
                print(f"\n--- Scale {scale} ---")
                pose = initial_pose.clone()
                pose[:, :3] += delta_rot
                pose[:, 3:] += delta_trans * trans_scale
                print(f"Current pose: {pose} ---")

            for step in range(iters_per_scale):
                optimizer.zero_grad()
                
                rot_scale = 0.3     # ~28°
                trans_scale = 30.0  # mm

                # rot_update = rot_scale * torch.tanh(delta_rot)
                # trans_update = trans_scale * torch.tanh(delta_trans)
                pose = initial_pose.clone()
                pose[:, :3] += delta_rot
                pose[:, 3:] += delta_trans * trans_scale
                # pose[:, :3] += rot_update
                # pose[:, 3:] += trans_update
                # pose[:, 0] += trans_update[:, 0]
                # pose[:, 2] += trans_update[:, 1]

                proj = self.render_drr(pose, drr)
                # import torchvision.transforms.functional as TF
                # proj = TF.gaussian_blur(proj, 7, 1.0)
                proj_s = proj * kernel
                # proj_s = self.resize(proj, img_size)

                gain = self.criterion(proj_s, carm_s)
                loss = 1 - gain.mean()

                # Safety check
                if not torch.isfinite(loss):
                    print("Numerical instability detected. Stopping.")
                    break

                loss.backward()

                torch.nn.utils.clip_grad_norm_([delta_rot, delta_trans], 1.0)
            
                optimizer.step()

                gain_val = gain.item()
                # print("rot grad:", delta_rot.grad.norm().item(), "trans grad:", delta_trans.grad.norm().item())

                # Track best
                if gain_val > best_gain + min_delta:
                    best_gain = gain_val
                    best_pose = pose.detach().clone()
                    no_improve = 0
                else:
                    no_improve += 1

                # Store history
                history.setdefault("gain",       []).append(gain_val)
                history.setdefault("delta_pose", []).append(torch.cat([delta_rot, delta_trans * trans_scale], dim=-1).detach().clone().cpu().tolist())
                history.setdefault("estimated_pose", []).append(pose.detach().clone().cpu().tolist())

                if verbose:
                    print(
                        f"[{step:03d}] "
                        f"gain={gain_val:.5f} "
                        f"best={best_gain:.5f} "
                        f"no_improve={no_improve}"
                    )

                # Early stopping
                if no_improve >= patience:
                    if verbose:
                        print("Early stopping triggered")
                    break

        return best_pose, best_gain, history


class Optimizer(nn.Module):
    def __init__(self, ct_path, loss:callable=mGNCCLoss(), lr=(1e-3, 5e-3), weight_decay=0.0, scales:int=3, device=config.DEVICE):
        """
        Refines an initial pose using gradient-based optimization
        to maximize mGNCC similarity between DRR and C-arm image.

        Args:
            drr: Differentiable DRR renderer
            pose_regressor: pretrained model giving initial pose
            max_translation: clamp for translation updates
            max_rotation: clamp for rotation updates (radians)
            lambda_reg: regularization weight
            device: torch device
        """
        super().__init__()
        self.device = device
        self.criterion = loss
        self.lr = lr
        self.weight_decay = weight_decay

        self.subject = read(volume=str(ct_path), orientation="AP", center_volume=True)
        self.trans_scale = torch.tensor([30, 500, 30], device=self.device, requires_grad=False)

        # coarse → fine pyramid
        self.scales = [2**(s-1) for s in range(scales, 0, -1)]

    # --------------------------------------------------
    # Utility functions
    # --------------------------------------------------
    def normalize(self, img, invert=True):
        """Min-max normalize safely"""
        eps = 1e-6
        mn = img.amin(dim=(-2, -1), keepdim=True)
        mx = img.amax(dim=(-2, -1), keepdim=True)
        if invert:
            out = 1.0 - (img - mn) / (mx - mn + eps)
        else:
            out = (img - mn) / (mx - mn + eps)
        return out

    def resize(self, img, new_size, sigma: float=0.25):
        """Simple blur via avg pooling"""
        img_size = img.shape[-1]
        if new_size == img_size:
            return img
        scale = (new_size // img_size) if (new_size > img_size) else (img_size // new_size)
        out = TF.gaussian_blur(img, kernel_size=7, sigma=(sigma * scale))
        return TF.resize(out, new_size)

    def render_drr(self, pose, drr=None, img_size=config.IMAGE_SIZE):
        """Render DRR from pose"""
        rot = pose[:, :3]
        trans = pose[:, 3:]

        if drr is None:
            delx = 305 / img_size
            print(f"Rendered image size is {img_size}")
            drr = DRR(self.subject, sdd=config.SDD, height=img_size, delx=delx).to(self.device)
        proj = drr(rot, trans, parameterization="euler_angles", convention="ZXY")

        return self.normalize(proj).to(self.device)

    def find_closest_depth(self, carm, pose:torch.tensor, max_dist=100, size=10):
        img_size = carm.shape[2]
        depth = pose[0, -2]
        depth_list = torch.linspace(depth - max_dist, depth + max_dist, size)
        poses = pose.repeat(size, 1)
        poses[:, -2] = depth_list

        carm_s = self.resize(carm, img_size).detach()

        
        drrs = self.render_drr(poses, img_size=img_size)
        carms = torch.cat(size * [carm_s])
        # carms = carm_s.repeat(size)
        gain = self.criterion(drrs, carms)
        index = torch.argmax(gain)
        print(pose)
        print(gain)
        print(poses[index])
        return poses[index].unsqueeze(0)

    def add_delta_pose(self, pose, delta_rot, delta_trans):

        pose[:, :3] += delta_rot
        pose[:, 3:] += delta_trans * self.trans_scale
        return pose


    def closure(self, optimizer, carm, pose, kernel=1.0):
        optimizer.zero_grad()

        img_size = carm.shape[-1]
        proj = self.render_drr(pose)
        proj = proj * kernel
        proj_s = self.resize(proj, img_size)

        gain = self.criterion(proj_s, carm)
        loss = 1 - gain.mean()

        loss.backward()

        return loss
    
    # --------------------------------------------------
    # Main Optimization
    # --------------------------------------------------
    def forward(self, carm, initial_pose, kernel=1.0, iters_per_scale=100, patience=15, lbfgs:bool=False, verbose=True):
        """
        Args:
            carm: input DRR / C-arm image (B=1)
        Returns:
            best_pose: optimized pose (1, 6)
            best_gain: final similarity score
        """
        # General parameters
        min_delta = 1e-4
        carm_size = carm.shape[-1]
        history = {}
        
        # ----------------------------------
        # Optimization variable
        # ----------------------------------
        if lbfgs:
            delta_pose = nn.Parameter(torch.zeros_like(initial_pose), requires_grad=True)
            optimizer = torch.optim.LBFGS([delta_pose], lr=self.lr[0])
        else:
            delta_rot = nn.Parameter(torch.zeros_like(initial_pose[:, :-3]), requires_grad=True)
            delta_trans = nn.Parameter(torch.zeros_like(initial_pose[:, -3:]), requires_grad=True)
            # delta_trans = nn.Parameter(torch.zeros((1,2), device=self.device), requires_grad=True)
            # torch.nn.init.kaiming_normal_(delta_pose)
            optimizer = torch.optim.AdamW([
                {"params": delta_rot,   "lr": self.lr[0], "weight_decay": self.weight_decay},
                {"params": delta_trans, "lr": self.lr[1], "weight_decay": self.weight_decay},
            ])


        # # Pose limits
        # max_range = torch.tensor([
        #     self.max_rotation, self.max_rotation, self.max_rotation,
        #     self.max_translation, self.max_translation, self.max_translation
        # ], device=self.device).unsqueeze(0)

        # Estimate best depth
        # initial_pose = self.find_closest_depth(carm, initial_pose)

        initial_pose = initial_pose.detach()
        pose = self.add_delta_pose(initial_pose.clone(), delta_rot, delta_trans)

        # ----------------------------------
        # Multi-scale optimization
        # ----------------------------------
        for scale in self.scales:
            # Reset Optimization Parameters
            best_gain = -float("inf")
            best_pose = initial_pose.clone()
            no_improve = 0
            img_size = carm_size // scale
            delx = 305 / img_size
            drr = DRR(self.subject, sdd=config.SDD, height=img_size, delx=delx).to(self.device)

            # Resize C-arm and kernel
            carm_s = self.resize(carm, img_size).detach()
            if not type(kernel) is float:
                kernel = TF.resize(kernel, img_size).detach()

            if verbose:
                print(f"\n--- Scale {scale} ---")
                print(f"Current pose: {pose} ---")

            for step in range(iters_per_scale):
                optimizer.zero_grad()
                
                # rot_scale = 0.3     # ~28°
                # trans_scale = 30.0  # mm

                # rot_update = rot_scale * torch.tanh(delta_rot)
                # trans_update = trans_scale * torch.tanh(delta_trans)
                pose = self.add_delta_pose(initial_pose.clone(), delta_rot, delta_trans)

                proj = self.render_drr(pose, drr)
                # proj = TF.gaussian_blur(proj, 7, 1.0)
                proj_s = proj * kernel
            
                gain = self.criterion(proj_s, carm_s, None)
                loss = 1 - gain.mean()

                # Safety check
                if not torch.isfinite(loss):
                    print("Numerical instability detected. Stopping.")
                    break

                loss.backward()

                torch.nn.utils.clip_grad_norm_([delta_rot, delta_trans], 1.0)
            
                optimizer.step()

                gain_val = gain.item()
                # print("rot grad:", delta_rot.grad.norm().item(), "trans grad:", delta_trans.grad.norm().item())

                # Track best
                if gain_val > best_gain + min_delta:
                    best_gain = gain_val
                    best_pose = pose.detach().clone()
                    no_improve = 0
                else:
                    no_improve += 1

                # Store history
                history.setdefault("gain",       []).append(gain_val)
                history.setdefault("delta_pose", []).append(torch.cat([delta_rot, delta_trans * self.trans_scale], dim=-1).detach().clone().cpu().tolist())
                history.setdefault("estimated_pose", []).append(pose.detach().clone().cpu().tolist())

                if verbose:
                    print(
                        f"[{step:03d}] "
                        f"gain={gain_val:.5f} "
                        f"best={best_gain:.5f} "
                        f"no_improve={no_improve}"
                    )

                # Early stopping
                if no_improve >= patience:
                    if verbose:
                        print("Early stopping triggered")
                    break

        return best_pose, best_gain, history

