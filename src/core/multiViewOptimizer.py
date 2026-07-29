from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from diffdrr.drr import DRR
from diffdrr.data import read
from diffdrr.pose import euler_angles_to_matrix, rotation_6d_to_matrix, matrix_to_euler_angles, make_matrix

from src.utils import config
from src.utils.loss import mGNCCLoss, compute_geodesic_distance
from src.utils.pose_utils import apply_delta, compose_poses, compute_relative_transform, clone_pose


@dataclass
class OptimizationResult:
    """
    Result returned by both single-view and multi-view optimization.

    Attributes
    ----------
    R : Tensor
        Optimized rotation matrix of shape (B, 3, 3).

    t : Tensor
        Optimized translation of shape (B, 3).

    objective : float
        Final optimization objective.

        * SingleViewOpt:
            Highest similarity (mGNCC).

        * MultiViewOptimizer:
            Lowest joint loss.

    history : dict
        Complete optimization history.

    per_view_scores : Optional[List[float]]
        Final similarity of every optimized view.

        Single-view optimization leaves this as None.
    """

    R: torch.Tensor
    t: torch.Tensor

    objective: float
    history: dict

    per_view_scores: Optional[List[float]] = None

    @property
    def similarity(self) -> float:
        """
        Alias used by single-view optimization.
        """
        return self.objective

    @property
    def loss(self) -> float:
        """
        Alias used by multi-view optimization.
        """
        return self.objective


class SingleViewOpt(nn.Module):
    """
    Single-view gradient-based 2D/3D registration.

    Unlike the original Optimizer, this implementation optimizes a pose stored as (R, t)

    where
        R : (B,3,3) rotation matrix
        t : (B,3) translation (mm)

    Optimization is performed over small axis-angle and translation updates
        delta_rot   : axis-angle (rad)
        delta_trans : translation (mm)

    and converted into the current pose using
        R_current, t_current = apply_delta(...)

    The renderer itself still expects Euler angles, therefore the conversion from rotation matrix -> Euler happens only inside render_drr().
    """

    def __init__(self, ct_path, loss: callable = mGNCCLoss(), lr=(1e-3, 1), weight_decay=0.0, scales: int = 3, device=config.DEVICE):
        super().__init__()

        self.device = device
        self.criterion = loss
        self.lr = lr
        self.weight_decay = weight_decay

        self.subject = read(volume=str(ct_path), orientation="AP", center_volume=True)

        # coarse -> fine pyramid
        self.scales = [2 ** (s - 1) for s in range(scales, 0, -1)]

    # ------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------

    def normalize(self, img: torch.Tensor, invert: bool = True):
        """
        Min-max normalize a DRR.
        """
        eps = 1e-6
        mn = img.amin(dim=(-2, -1), keepdim=True)
        mx = img.amax(dim=(-2, -1), keepdim=True)
        if invert:
            img = 1.0 - (img - mn) / (mx - mn + eps)
        else:
            img = (img - mn) / (mx - mn + eps)
        return img

    def resize(self, img: torch.Tensor, new_size: int, sigma: float = 0.25):
        """
        Resize image while applying a small Gaussian blur.

        Used for the coarse-to-fine optimization pyramid.
        """
        current_size = img.shape[-1]
        if current_size == new_size:
            return img

        if new_size > current_size:
            scale = new_size // current_size
        else:
            scale = current_size // new_size

        img = TF.gaussian_blur(img, kernel_size=7, sigma=sigma * scale)
        img = TF.resize(img, new_size)
        return img

    # ------------------------------------------------------------------
    # DRR rendering
    # ------------------------------------------------------------------

    def _create_renderer(self, img_size: int) -> DRR:
        delx = 305 / img_size
        return DRR(self.subject, sdd=config.SDD, height=img_size, delx=delx).to(self.device)

    def render_drr(self, R: torch.Tensor, t: torch.Tensor, drr=None, img_size=config.IMAGE_SIZE):
        """
        Render a DRR from a rotation matrix and translation.

        Parameters
        ----------
        R: (B,3,3) rotation matrix
        t: (B,3) translation (mm)
        drr: Optional renderer for the current pyramid scale.
        img_size: Only used if a renderer must be created.
        """

        if drr is None:
            delx = 305 / img_size
            drr = DRR(self.subject, sdd=config.SDD, height=img_size, delx=delx).to(self.device)

        # ----------------------------------------------------------
        # DiffDRR currently renders from Euler angles.
        # The optimizer never works in Euler representation;
        # this conversion is only performed here.
        # ----------------------------------------------------------

        rot = matrix_to_euler_angles(R, convention="ZXY")
        proj = drr(rot, t, parameterization="euler_angles", convention="ZXY")
        # homo_matrix = make_matrix(R, t)
        # proj = drr(homo_matrix, parameterization="matrix")
        return self.normalize(proj).to(self.device)

    # ------------------------------------------------------------------
    # Main optimization
    # ------------------------------------------------------------------

    def _optimize_scale_single(self, carm_s: torch.Tensor, kernel_s, drr: DRR, R_base: torch.Tensor, t_base: torch.Tensor, delta_rot: nn.Parameter, delta_trans: nn.Parameter,
                                optimizer: torch.optim.Optimizer, patience: int, iters_per_scale: int, history: dict, verbose: bool) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Run one pyramid level of the single-view optimization.

        Parameters
        ----------
        carm_s: Target C-arm image resized to the current pyramid level.
        kernel_s: Kernel resized to the current pyramid level (or scalar).
        drr: Renderer for the current pyramid level.
        R_base: Base rotation for this scale.
        t_base: Base translation for this scale.
        delta_rot: Learnable axis-angle update.
        delta_trans: Learnable translation update.
        optimizer: AdamW optimizer.
        patience: Early stopping patience.
        iters_per_scale: Maximum iterations at this scale.
        history: Shared optimization history.
        verbose: Print optimization progress.

        Returns
        -------
        best_R: Best rotation found for this scale.
        best_t: Best translation found for this scale.
        best_similarity: Highest similarity achieved at this scale.
        """

        min_delta = 1e-4

        best_R, best_t = clone_pose(R_base, t_base)
        best_similarity = -float("inf")

        no_improve = 0

        for step in range(iters_per_scale):

            optimizer.zero_grad()

            # --------------------------------------------------
            # Current pose
            # --------------------------------------------------

            R_current, t_current = apply_delta(R_base, t_base, delta_rot, delta_trans)

            # --------------------------------------------------
            # Render DRR
            # --------------------------------------------------

            proj = self.render_drr(R_current, t_current, drr=drr)
            proj = proj * kernel_s

            # --------------------------------------------------
            # Similarity / loss
            # --------------------------------------------------

            similarity = self.criterion(proj, carm_s, None)
            loss = 1.0 - similarity.mean()

            if not torch.isfinite(loss):
                print("Numerical instability detected.")
                break

            # --------------------------------------------------
            # Backpropagation
            # --------------------------------------------------

            loss.backward()
            torch.nn.utils.clip_grad_norm_([delta_rot, delta_trans], max_norm=1.0)
            optimizer.step()
            similarity_val = similarity.item()

            # --------------------------------------------------
            # Best solution
            # --------------------------------------------------

            if similarity_val > best_similarity + min_delta:
                best_similarity = similarity_val
                best_R = R_current.detach().clone()
                best_t = t_current.detach().clone()
                no_improve = 0
            else:
                no_improve += 1

            # --------------------------------------------------
            # History
            # --------------------------------------------------

            history["similarity"].append(similarity_val)
            history["rotation"].append(R_current.detach().cpu().tolist())
            history["translation"].append(t_current.detach().cpu().tolist())
            history["delta_rot"].append(delta_rot.detach().cpu().tolist())
            history["delta_trans"].append(delta_trans.detach().cpu().tolist())

            # --------------------------------------------------
            # Verbose logging
            # --------------------------------------------------

            if verbose:
                rot_norm = torch.linalg.norm(delta_rot).item()
                trans_norm = torch.linalg.norm(delta_trans).item()
                print(
                    f"[{step:03d}] "
                    f"similarity={similarity_val:.5f} "
                    f"best={best_similarity:.5f} "
                    f"rot_update={rot_norm:.5f} "
                    f"trans_update={trans_norm:.3f} mm "
                    f"no_improve={no_improve}"
                )

            # --------------------------------------------------
            # Early stopping
            # --------------------------------------------------

            if no_improve >= patience:
                if verbose:
                    print("Early stopping triggered.")
                break

        return best_R, best_t, best_similarity

    def forward(self, carm, R_init, t_init, kernel=1.0, iters_per_scale=100, patience=15, verbose=True) -> OptimizationResult:
        """
        Parameters
        ----------
        carm : Tensor
            Input fluoroscopy / DRR.
        R_init : Tensor
            Initial rotation matrix (B, 3, 3).
        t_init : Tensor
            Initial translation (B, 3).

        Returns
        -------
        best_R : Tensor
            Optimized rotation.
        best_t : Tensor
            Optimized translation.
        best_similarity : float
            Highest similarity achieved.
        history : dict
            Optimization history.
        """

        carm_size = carm.shape[-1]

        # ----------------------------------------------------------
        # History
        # ----------------------------------------------------------

        history = {
            "similarity": [],
            "rotation": [],
            "translation": [],
            "delta_rot": [],
            "delta_trans": [],
        }

        # ----------------------------------------------------------
        # Base pose
        # ----------------------------------------------------------

        R_base, t_base = clone_pose(R_init.detach(), t_init.detach())

        # ----------------------------------------------------------
        # Optimization variables
        # ----------------------------------------------------------

        delta_rot = nn.Parameter(torch.zeros_like(t_base), requires_grad=True)
        delta_trans = nn.Parameter(torch.zeros_like(t_base), requires_grad=True)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": delta_rot,
                    "lr": self.lr[0],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": delta_trans,
                    "lr": self.lr[1],
                    "weight_decay": self.weight_decay,
                },
            ]
        )

        # ----------------------------------------------------------
        # Initial best pose
        # ----------------------------------------------------------

        best_R, best_t = clone_pose(R_base, t_base)
        best_similarity = -float("inf")

        # ----------------------------------------------------------
        # Multi-resolution optimization
        # ----------------------------------------------------------

        for scale in self.scales:
            img_size = carm_size // scale
            drr = self._create_renderer(img_size)
            carm_s = self.resize(carm, img_size).detach()

            if isinstance(kernel, float):
                kernel_s = kernel
            else:
                kernel_s = self.resize(kernel, img_size).detach()

            if verbose:
                print(f"\n--- Scale {scale} | img_size={img_size} ---")

            best_R, best_t, best_similarity = self._optimize_scale_single(
                carm_s=carm_s,
                kernel_s=kernel_s,
                drr=drr,
                R_base=R_base,
                t_base=t_base,
                delta_rot=delta_rot,
                delta_trans=delta_trans,
                optimizer=optimizer,
                patience=patience,
                iters_per_scale=iters_per_scale,
                history=history,
                verbose=verbose,
            )

            # ------------------------------------------------------
            # Scale transition
            # ------------------------------------------------------

            R_base, t_base = clone_pose(best_R.detach(), best_t.detach())

            with torch.no_grad():
                delta_rot.zero_()
                delta_trans.zero_()

        result = OptimizationResult(R=best_R, t=best_t, objective=best_similarity, history=history)
        return result


@dataclass
class ViewData:
    """
    Container holding all information associated with a single C-arm view.

    Attributes
    ----------
    image : Tensor
        (1, 1, H, W) normalized C-arm (or synthetic DRR) image.

    R_init : Tensor
        (1, 3, 3) initial rotation matrix.

    t_init : Tensor
        (1, 3) initial translation in millimeters.

    R_delta_1i : Tensor
        (1, 3, 3) fixed relative rotation from the reference view (view 0) to this view.
        For the reference view this should be the identity matrix.

    t_delta_1i : Tensor
        (1, 3) fixed relative translation from the reference view (view 0) to this view.
        For the reference view this should be zeros.

    weight : float
        Relative contribution of this view to the joint objective.
    """

    image: torch.Tensor

    R_init: torch.Tensor
    t_init: torch.Tensor

    R_delta_1i: torch.Tensor
    t_delta_1i: torch.Tensor

    weight: float = 1.0

    @property
    def is_reference(self) -> bool:
        """
        Returns True if this ViewData corresponds to the reference view.

        This is determined by checking whether the stored relative transform
        is identity (rotation) and zero (translation).
        """
        identity = torch.eye(3, dtype=self.R_delta_1i.dtype, device=self.R_delta_1i.device).unsqueeze(0)

        return (torch.allclose(self.R_delta_1i, identity) and torch.allclose(self.t_delta_1i, torch.zeros_like(self.t_delta_1i)))


class MultiViewObjective:
    """
    Computes the joint NCC loss across multiple views in a fully vectorized
    manner.

    The optimization variable is a single (delta_rot, delta_trans) pair
    applied to the reference view. All remaining view poses are derived
    deterministically using their fixed relative transforms.

    All poses are rendered simultaneously using a batched DiffDRR call,
    followed by a batched NCC computation.
    """

    def __init__(self, views: List[ViewData], renderer_factory: Callable[[int], DRR], criterion: nn.Module, resize_fn: Callable, normalize_fn: Callable):
        """
        Parameters
        ----------
        views: List of all views. views[0] is the reference.
        renderer_factory: Callable taking img_size and returning a DRR renderer.
        criterion: Similarity metric (mGNCCLoss).
        resize_fn: SingleViewOpt.resize.
        normalize_fn: SingleViewOpt.normalize.
        """

        if len(views) == 0:
            raise ValueError("MultiViewObjective requires at least one view.")

        self.renderer_factory = renderer_factory
        self.criterion = criterion
        self.resize_fn = resize_fn
        self.normalize_fn = normalize_fn

        self.num_views = len(views)

        # ----------------------------------------------------------
        # Fixed tensors
        # ----------------------------------------------------------

        self.images_batch = torch.cat([v.image for v in views], dim=0)

        self.weights = torch.tensor([v.weight for v in views], dtype=self.images_batch.dtype, device=self.images_batch.device)

        self.R1_init = views[0].R_init
        self.t1_init = views[0].t_init

        if self.num_views > 1:
            self.R_deltas = torch.cat([v.R_delta_1i for v in views[1:]], dim=0)

            self.t_deltas = torch.cat([v.t_delta_1i for v in views[1:]], dim=0)
        else:
            self.R_deltas = None
            self.t_deltas = None

        # ----------------------------------------------------------
        # Renderer cache
        # One renderer is created per image resolution and reused for
        # every optimization iteration at that pyramid level.
        # ----------------------------------------------------------

        self._renderer_cache = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_renderer(self, img_size: int) -> DRR:
        """
        Returns a cached renderer for the requested image resolution.

        A renderer is created only the first time a particular image size
        is requested, matching the behavior of SingleViewOpt.
        """

        if img_size not in self._renderer_cache:
            self._renderer_cache[img_size] = self.renderer_factory(img_size)

        return self._renderer_cache[img_size]

    def _render_batch(self, renderer: DRR, R: torch.Tensor, t: torch.Tensor, img_size: int) -> torch.Tensor:
        """
        Render a batch of DRRs.

        Parameters
        ----------
        renderer: Cached renderer for the current pyramid level.
        R: Rotation matrices of shape (N, 3, 3).
        t: Translations of shape (N, 3).
        img_size: Current pyramid level resolution.

        Returns
        -------
        Tensor: Normalized DRRs of shape (N, 1, H, W).
        """

        # H = make_matrix(R, t)
        # projs = renderer(H, parameterization="matrix")
        rot = matrix_to_euler_angles(R, convention="ZXY")
        projs = renderer(rot, t, parameterization="euler_angles", convention="ZXY")

        projs = self.normalize_fn(projs)
        projs = self.resize_fn(projs, img_size)

        return projs

    def set_reference_pose(self, R: torch.Tensor, t: torch.Tensor) -> None:
        """
        Update the reference pose used as the base initialization for the
        next pyramid level.

        This mirrors the coarse-to-fine strategy used by SingleViewOpt,
        where the best pose from one scale becomes the initial pose for
        the following scale.
        """
        self.R1_init = R.detach().clone()
        self.t1_init = t.detach().clone()

    def __call__(self, delta_rot: torch.Tensor, delta_trans: torch.Tensor, img_size: int) -> Tuple[torch.Tensor, List[float]]:
        """
        Evaluate the joint multi-view objective.

        Parameters
        ----------
        delta_rot: (1,3) axis-angle update applied to the reference pose.
        delta_trans: (1,3) translation update (mm) applied to the reference pose.
        img_size: Current pyramid-level image size.

        Returns
        -------
        joint_loss: Differentiable scalar loss.
        per_view_ncc: Detached NCC value for every view.
        """

        renderer = self._get_renderer(img_size)

        # ----------------------------------------------------------
        # Update reference pose
        # ----------------------------------------------------------

        R1_cur, t1_cur = apply_delta(self.R1_init, self.t1_init, delta_rot, delta_trans)

        # ----------------------------------------------------------
        # Compute all dependent poses simultaneously
        # ----------------------------------------------------------

        if self.num_views > 1:
            R1_exp = R1_cur.expand(self.num_views - 1, -1, -1)
            t1_exp = t1_cur.expand(self.num_views - 1, -1)

            R_dep, t_dep = compose_poses(R1_exp, t1_exp, self.R_deltas, self.t_deltas)

            R_all = torch.cat([R1_cur, R_dep], dim=0)
            t_all = torch.cat([t1_cur, t_dep], dim=0)

        else:
            R_all = R1_cur
            t_all = t1_cur

        # ----------------------------------------------------------
        # Render all DRRs simultaneously
        # ----------------------------------------------------------

        projs = self._render_batch(renderer, R_all, t_all, img_size)

        # ----------------------------------------------------------
        # Resize target images
        # ----------------------------------------------------------

        images = self.resize_fn(self.images_batch, img_size)

        # ----------------------------------------------------------
        # Compute batched NCC
        # ----------------------------------------------------------

        similarities = self.criterion(projs, images, None)

        # Expected shape:
        # similarities : (N,)
        #
        # If mGNCCLoss instead returns a scalar, this is the only section
        # that should be modified.
        assert similarities.ndim >= 1 and similarities.shape[0] == self.num_views, (
        f"criterion must return per-sample similarities of shape ({self.num_views},), "
        f"got {tuple(similarities.shape)}. "
        f"Check that mGNCCLoss / NCC does not reduce across the batch dimension."
        )   

        similarities = similarities.reshape(-1)

        # ----------------------------------------------------------
        # Weighted joint loss
        # ----------------------------------------------------------

        weights = self.weights.to(device=similarities.device, dtype=similarities.dtype)
        joint_loss = (weights * (1.0 - similarities)).sum()
        return joint_loss, similarities.detach().cpu().tolist()
    

class MultiViewOptimizer(SingleViewOpt):
    """
    Extends SingleViewOpt to joint multi-view optimization.

    The optimization variable is a single (delta_rot, delta_trans) pair
    applied to the reference pose (view 0). All remaining poses are
    deterministically derived from the fixed relative transforms.

    The inherited forward() performs standard single-view optimization.
    Use forward_multiview() for joint optimization.
    """

    def __init__(self, ct_path, loss: callable = mGNCCLoss(), lr=(1e-3, 1), weight_decay=0.0, scales: int = 3, device=config.DEVICE):
        super().__init__(ct_path=ct_path, loss=loss, lr=lr, weight_decay=weight_decay, scales=scales, device=device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def clear_renderer_cache(self) -> None:
        """
        Delete all cached DRR renderers and release GPU memory.
        Call at scale transitions to avoid holding multiple resolution
        renderers in GPU memory simultaneously.
        """
        for renderer in self._renderer_cache.values():
            del renderer
        self._renderer_cache.clear()
        torch.cuda.empty_cache()

    def _check_view_consistency(self, views: List[ViewData]) -> None:
        """
        Verify that each supplied relative transform is geometrically
        consistent with the supplied initial poses.

        This is purely a diagnostic and is executed once before
        optimization begins.
        """

        if len(views) <= 1:
            return

        print("\n========== Multi-view Consistency Check ==========")

        ref = views[0]

        for idx, view in enumerate(views[1:], start=1):
            R_pred, t_pred = compose_poses(ref.R_init, ref.t_init, view.R_delta_1i, view.t_delta_1i)
            rot_err = compute_geodesic_distance(R_pred, view.R_init)

            if torch.is_tensor(rot_err):
                rot_err = rot_err.item()

            trans_err = torch.linalg.norm(t_pred - view.t_init).item()

            print(
                f"View {idx}: "
                f"rot={rot_err:.2f}°  "
                f"trans={trans_err:.2f} mm"
            )

            if rot_err > torch.deg2rad(torch.tensor(5.0)) or trans_err > 10.0:
                print(
                    "  WARNING: Large discrepancy detected. "
                    "Check the supplied relative transform or "
                    "the initial ML pose."
                )

        print("=================================================\n")


    def _create_objective(self, views: List[ViewData]) -> MultiViewObjective:
        """
        Construct the joint optimization objective.
        """
        return MultiViewObjective(views=views, renderer_factory=self._create_renderer, criterion=self.criterion, resize_fn=self.resize, normalize_fn=self.normalize)
    
    def _optimize_scale(self, objective: MultiViewObjective, optimizer: torch.optim.Optimizer, delta_rot: nn.Parameter, delta_trans: nn.Parameter, img_size: int,
                        patience: int, iters_per_scale: int, history: dict, verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Optimize one pyramid level.

        Parameters
        ----------
        objective: Multi-view objective.
        optimizer: AdamW optimizer.
        delta_rot: Learnable axis-angle update.
        delta_trans: Learnable translation update.
        img_size: Current pyramid-level image size.
        patience: Early stopping patience.
        iters_per_scale: Maximum optimization iterations.
        history: Global optimization history.

        Returns
        -------
        best_R: Best reference rotation found for this scale.
        best_t: Best reference translation found for this scale.
        best_loss: Lowest joint loss achieved during this scale.
        """

        best_loss = float("inf")
        best_R = objective.R1_init.detach().clone()
        best_t = objective.t1_init.detach().clone()
        best_per_view_ncc: List[float] = []

        no_improve = 0
        min_delta = 1e-4

        for step in range(iters_per_scale):

            optimizer.zero_grad()

            # --------------------------------------------------
            # Evaluate current objective
            # --------------------------------------------------

            joint_loss, per_view_ncc = objective(delta_rot, delta_trans, img_size)

            # Safety check
            if not torch.isfinite(joint_loss):
                print("Numerical instability detected. Stopping optimization.")
                break

            # --------------------------------------------------
            # Backpropagation
            # --------------------------------------------------

            joint_loss.backward()

            torch.nn.utils.clip_grad_norm_([delta_rot, delta_trans], max_norm=1.0)
            
            # Compute the current pose before the parameter update.
            R_cur, t_cur = apply_delta(objective.R1_init, objective.t1_init, delta_rot, delta_trans)

            optimizer.step()
            loss_val = joint_loss.item()

            # --------------------------------------------------
            # Track best solution
            # --------------------------------------------------

            if loss_val < (best_loss - min_delta):
                best_loss = loss_val
                best_R = R_cur.detach().clone()
                best_t = t_cur.detach().clone()
                best_per_view_ncc = per_view_ncc
                no_improve = 0
            else:
                no_improve += 1

            # --------------------------------------------------
            # Store optimization history
            # --------------------------------------------------

            history["loss"].append(loss_val)
            history["per_view_ncc"].append(per_view_ncc)
            history["delta_rot"].append(delta_rot.detach().cpu().clone().tolist())
            history["delta_trans"].append(delta_trans.detach().cpu().clone().tolist())

            # --------------------------------------------------
            # Verbose logging
            # --------------------------------------------------

            if verbose:

                msg = (
                    f"[{step:03d}] "
                    f"loss={loss_val:.6f} "
                )

                for view_idx, ncc in enumerate(per_view_ncc):
                    msg += f"v{view_idx}={ncc:.5f} "

                msg += (
                    f"best={best_loss:.6f} "
                    f"no_improve={no_improve}"
                )

                print(msg)

            # --------------------------------------------------
            # Early stopping
            # --------------------------------------------------

            if no_improve >= patience:

                if verbose:
                    print("Early stopping triggered.")

                break

        return best_R, best_t, best_loss, best_per_view_ncc

    def forward_multiview(self, views: List[ViewData], iters_per_scale: int = 100, patience: int = 15, verbose: bool = True) -> OptimizationResult:
        """
        Joint multi-view optimization.

        Parameters
        ----------
        views: List of ViewData objects. views[0] is the reference view.

        Returns
        -------
        best_R: Optimized reference rotation matrix.
        best_t: Optimized reference translation.
        best_loss: Lowest joint loss achieved during optimization.
        history: Optimization history.
        """

        if len(views) == 0:
            raise ValueError("At least one view must be provided.")

        # ----------------------------------------------------------
        # Verify consistency of supplied relative transforms
        # ----------------------------------------------------------

        self._check_view_consistency(views)

        # ----------------------------------------------------------
        # Construct optimization objective
        # ----------------------------------------------------------

        objective = self._create_objective(views)

        # ----------------------------------------------------------
        # Optimization variables
        # ----------------------------------------------------------

        rot_initialization = torch.tensor([[1e-2, 1e-2, 1e-2]], device=self.device)
        delta_rot = nn.Parameter(torch.zeros((1, 3), device=self.device), requires_grad=True)
        delta_trans = nn.Parameter(torch.zeros((1, 3), device=self.device), requires_grad=True)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": delta_rot,
                    "lr": self.lr[0],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": delta_trans,
                    "lr": self.lr[1],
                    "weight_decay": self.weight_decay,
                },
            ]
        )

        # ----------------------------------------------------------
        # Optimization history
        # ----------------------------------------------------------

        history = {
            "loss": [],
            "per_view_ncc": [],
            "delta_rot": [],
            "delta_trans": [],
        }

        # ----------------------------------------------------------
        # Multi-resolution optimization
        # ----------------------------------------------------------

        best_R = views[0].R_init.detach().clone()
        best_t = views[0].t_init.detach().clone()
        best_loss = float("inf")

        for scale in self.scales:
            # Get img_size from actual input
            img_size = views[0].image.shape[-1] // scale

            objective.set_reference_pose(best_R, best_t)

            delta_rot.data = rot_initialization.clone()
            delta_trans.data.zero_()

            if verbose:
                print(f"\n--- Scale {scale} ---")
                print(f"Image size : {img_size}")
                print(f"Reference pose initialized from previous scale")

            scale_best_R, scale_best_t, scale_best_loss, scale_best_per_view_ncc = self._optimize_scale(
                objective=objective,
                optimizer=optimizer,
                delta_rot=delta_rot,
                delta_trans=delta_trans,
                img_size=img_size,
                patience=patience,
                iters_per_scale=iters_per_scale,
                history=history,
                verbose=verbose,
            )

            # self.clear_renderer_cache()

            if scale_best_loss < best_loss:
                best_R = scale_best_R
                best_t = scale_best_t
                best_loss = scale_best_loss
                best_per_view_ncc = scale_best_per_view_ncc

        result = OptimizationResult(R=best_R, t=best_t, objective=best_loss, history=history, per_view_scores=best_per_view_ncc)

        return result
    