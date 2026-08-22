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

    # Joint optimization only
    R_all: Optional[torch.Tensor] = None        # (N, 3, 3) all corrected absolute poses
    t_all: Optional[torch.Tensor] = None        # (N, 3)
    R_rel_all: Optional[torch.Tensor] = None    # (N-1, 3, 3) corrected relative transforms
    t_rel_all: Optional[torch.Tensor] = None    # (N-1, 3)

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
    

class JointMultiViewObjective:
    """
    Computes the joint optimization objective for multi-view registration.

    Unlike MultiViewObjective, which optimizes only the reference pose and
    derives all remaining poses from fixed relative transforms, this objective
    simultaneously optimizes:

        1. The absolute pose of every view.
        2. The relative transform between the reference view and every
           non-reference view.

    A consistency loss penalizes disagreement between the independently
    optimized absolute poses and the poses predicted from the optimized
    reference pose together with the optimized relative transforms.

    All DRRs are rendered simultaneously using a batched DiffDRR call.
    The predicted poses are used only for the consistency loss and are
    never rendered.
    """

    def __init__(self, views: List[ViewData], renderer_factory: Callable[[int], DRR], criterion: nn.Module, resize_fn: Callable, normalize_fn: Callable, lambda_cons: float = 0.1):
        """
        Parameters
        ----------
        views : list[ViewData]
            List of all views. views[0] is the reference view.
        renderer_factory : Callable[[int], DRR]
            Callable taking an image size and returning a DiffDRR renderer.
        criterion : nn.Module
            Similarity metric (e.g., mGNCCLoss).
        resize_fn : Callable
            Image resize function (typically SingleViewOpt.resize).
        normalize_fn : Callable
            DRR normalization function (typically SingleViewOpt.normalize).
        lambda_cons : float, optional
            Weight of the geometric consistency loss relative to the NCC
            objective.
        """

        if len(views) < 2:
            raise ValueError(
                "JointMultiViewObjective requires at least two views. "
                "For a single view use MultiViewObjective or SingleViewOpt."
            )

        if not views[0].is_reference:
            raise ValueError(
                "views[0] must be the reference view."
            )

        self.renderer_factory = renderer_factory
        self.criterion = criterion
        self.resize_fn = resize_fn
        self.normalize_fn = normalize_fn
        self.lambda_cons = lambda_cons

        self.num_views = len(views)

        # ----------------------------------------------------------
        # Fixed image tensors
        # ----------------------------------------------------------

        self.images_batch = torch.cat([v.image for v in views], dim=0)
        self.weights = torch.tensor([v.weight for v in views], dtype=self.images_batch.dtype, device=self.images_batch.device)

        # ----------------------------------------------------------
        # Initial absolute poses
        # ----------------------------------------------------------

        self.R_all_init = torch.cat([v.R_init for v in views], dim=0)
        self.t_all_init = torch.cat([v.t_init for v in views], dim=0)

        # ----------------------------------------------------------
        # Initial relative transforms
        # ----------------------------------------------------------

        self.R_rel_init = torch.cat([v.R_delta_1i for v in views[1:]], dim=0)
        self.t_rel_init = torch.cat([v.t_delta_1i for v in views[1:]], dim=0)

        # ----------------------------------------------------------
        # Renderer cache
        # ----------------------------------------------------------

        self._renderer_cache = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_renderer(self, img_size: int) -> DRR:
        """
        Returns a cached renderer for the requested image resolution.

        Parameters
        ----------
        img_size : int
            Current pyramid-level image size.

        Returns
        -------
        DRR
            Cached renderer corresponding to the requested resolution.
        """

        if img_size not in self._renderer_cache:
            self._renderer_cache[img_size] = self.renderer_factory(img_size)

        return self._renderer_cache[img_size]

    def _render_batch(self, renderer: DRR, R: torch.Tensor, t: torch.Tensor, img_size: int) -> torch.Tensor:
        """
        Render a batch of DRRs from absolute poses.

        The predicted poses used for the consistency loss are intentionally
        never rendered. Only the independently corrected absolute poses are
        passed to DiffDRR.

        Parameters
        ----------
        renderer : DRR
            Cached renderer for the current pyramid level.
        R : torch.Tensor
            Rotation matrices of shape (N, 3, 3).
        t : torch.Tensor
            Translations of shape (N, 3).
        img_size : int
            Current pyramid-level image size.

        Returns
        -------
        torch.Tensor
            Normalized DRRs of shape (N, 1, H, W).
        """
        rot = matrix_to_euler_angles(R, convention="ZXY")

        projs = renderer(rot, t, parameterization="euler_angles", convention="ZXY")

        projs = self.normalize_fn(projs)
        # projs = self.resize_fn(projs, img_size)

        return projs
    
    def _apply_pose_updates(self, pose_rot_updates: torch.Tensor, pose_trans_updates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply absolute pose corrections to all views simultaneously.

        Parameters
        ----------
        pose_rot_updates : torch.Tensor
            Axis-angle pose updates of shape (N, 3).
        pose_trans_updates : torch.Tensor
            Translation updates of shape (N, 3).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Corrected absolute rotations and translations with shapes
            (N, 3, 3) and (N, 3), respectively.
        """

        R_all_cur, t_all_cur = apply_delta(self.R_all_init, self.t_all_init, pose_rot_updates, pose_trans_updates)

        assert (R_all_cur.shape == (self.num_views, 3, 3)), (
            f"Expected corrected rotations to have shape "
            f"({self.num_views}, 3, 3), "
            f"got {tuple(R_all_cur.shape)}."
        )

        assert (t_all_cur.shape == (self.num_views, 3)), (
            f"Expected corrected translations to have shape "
            f"({self.num_views}, 3), "
            f"got {tuple(t_all_cur.shape)}."
        )

        return R_all_cur, t_all_cur

    def _apply_relative_updates(self, relative_rot_updates: torch.Tensor, relative_trans_updates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply relative transform corrections to all non-reference views.

        Parameters
        ----------
        relative_rot_updates : torch.Tensor
            Axis-angle corrections for the relative transforms with shape
            (N-1, 3).
        relative_trans_updates : torch.Tensor
            Translation corrections for the relative transforms with shape
            (N-1, 3).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Corrected relative rotations and translations with shapes
            (N-1, 3, 3) and (N-1, 3), respectively.
        """

        R_rel_cur, t_rel_cur = apply_delta(self.R_rel_init, self.t_rel_init, relative_rot_updates, relative_trans_updates)

        expected_views = self.num_views - 1

        assert (R_rel_cur.shape == (expected_views, 3, 3)), (
            f"Expected corrected relative rotations to have shape "
            f"({expected_views}, 3, 3), "
            f"got {tuple(R_rel_cur.shape)}."
        )

        assert (t_rel_cur.shape == (expected_views, 3)), (
            f"Expected corrected relative translations to have shape "
            f"({expected_views}, 3), "
            f"got {tuple(t_rel_cur.shape)}."
        )

        return R_rel_cur, t_rel_cur

    def _compute_predicted_poses(self, R1_cur: torch.Tensor, t1_cur: torch.Tensor, R_rel_cur: torch.Tensor, t_rel_cur: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the non-reference poses from the corrected reference pose and
        corrected relative transforms.

        Parameters
        ----------
        R1_cur : torch.Tensor
            Corrected reference rotation of shape (1, 3, 3).
        t1_cur : torch.Tensor
            Corrected reference translation of shape (1, 3).
        R_rel_cur : torch.Tensor
            Corrected relative rotations of shape (N-1, 3, 3).
        t_rel_cur : torch.Tensor
            Corrected relative translations of shape (N-1, 3).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Predicted non-reference rotations and translations with shapes
            (N-1, 3, 3) and (N-1, 3), respectively.
        """

        expected_views = self.num_views - 1

        assert (R1_cur.shape == (1, 3, 3)), (
            f"Expected reference rotation to have shape (1, 3, 3), "
            f"got {tuple(R1_cur.shape)}."
        )

        assert (t1_cur.shape == (1, 3)), (
            f"Expected reference translation to have shape (1, 3), "
            f"got {tuple(t1_cur.shape)}."
        )

        # Keep the batch dimension by using [0:1] rather than [0].
        # This allows the reference pose to be expanded without reshaping.
        R1_exp = R1_cur.expand(expected_views, -1, -1)
        t1_exp = t1_cur.expand(expected_views, -1)

        R_pred, t_pred = compose_poses(R1_exp, t1_exp, R_rel_cur, t_rel_cur)

        assert (R_pred.shape == (expected_views, 3, 3)), (
            f"Expected predicted rotations to have shape "
            f"({expected_views}, 3, 3), "
            f"got {tuple(R_pred.shape)}."
        )

        assert (t_pred.shape == (expected_views, 3)), (
            f"Expected predicted translations to have shape "
            f"({expected_views}, 3), "
            f"got {tuple(t_pred.shape)}."
        )

        return R_pred, t_pred

    def set_all_poses(self, R_all: torch.Tensor, t_all: torch.Tensor) -> None:
        """
        Update the absolute poses used as the initialization for the next
        pyramid level.

        The optimized absolute poses are propagated between pyramid levels,
        mirroring the coarse-to-fine strategy used by SingleViewOpt.

        The corresponding relative transforms are recomputed from the updated
        absolute poses rather than propagated directly. This guarantees that
        the relative transforms remain exactly consistent with the propagated
        absolute poses at the start of every pyramid level.

        Parameters
        ----------
        R_all : torch.Tensor
            Absolute rotations of shape (N, 3, 3).
        t_all : torch.Tensor
            Absolute translations of shape (N, 3).
        """

        assert (R_all.shape == (self.num_views, 3, 3)), (
            f"Expected rotations to have shape "
            f"({self.num_views}, 3, 3), "
            f"got {tuple(R_all.shape)}."
        )

        assert (t_all.shape == (self.num_views, 3)), (
            f"Expected translations to have shape "
            f"({self.num_views}, 3), "
            f"got {tuple(t_all.shape)}."
        )

        self.R_all_init = R_all.detach().clone()
        self.t_all_init = t_all.detach().clone()

        # Recompute the relative transforms from the propagated absolute
        # poses to keep both representations exactly consistent between
        # pyramid levels.
        R_ref = self.R_all_init[0:1]
        t_ref = self.t_all_init[0:1]

        R_dep = self.R_all_init[1:]
        t_dep = self.t_all_init[1:]

        self.R_rel_init, self.t_rel_init = compute_relative_transform(R_ref.expand(self.num_views - 1, -1, -1), t_ref.expand(self.num_views - 1, -1), R_dep, t_dep)

    def _compute_consistency_loss(self, R_dep_cur: torch.Tensor, t_dep_cur: torch.Tensor, R_pred: torch.Tensor, t_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute the geometric consistency loss between the independently
        optimized absolute poses and the poses predicted from the optimized
        reference pose together with the optimized relative transforms.

        The consistency loss is computed only for the non-reference views.

        Parameters
        ----------
        R_dep_cur : torch.Tensor
            Corrected absolute rotations of the non-reference views with shape
            (N-1, 3, 3).
        t_dep_cur : torch.Tensor
            Corrected absolute translations of the non-reference views with
            shape (N-1, 3).
        R_pred : torch.Tensor
            Predicted rotations obtained from composing the corrected
            reference pose with the corrected relative transforms.
            Shape: (N-1, 3, 3).
        t_pred : torch.Tensor
            Predicted translations obtained from composing the corrected
            reference pose with the corrected relative transforms.
            Shape: (N-1, 3).

        Returns
        -------
        torch.Tensor
            Scalar consistency loss consisting of the summed geodesic
            rotation error (radians) and translation error (mm) over all
            non-reference views.
        """

        expected_views = self.num_views - 1

        assert (R_dep_cur.shape == (expected_views, 3, 3)), (
            f"Expected corrected absolute rotations to have shape "
            f"({expected_views}, 3, 3), "
            f"got {tuple(R_dep_cur.shape)}."
        )

        assert (t_dep_cur.shape == (expected_views, 3)), (
            f"Expected corrected absolute translations to have shape "
            f"({expected_views}, 3), "
            f"got {tuple(t_dep_cur.shape)}."
        )

        assert (R_pred.shape == (expected_views, 3, 3)), (
            f"Expected predicted rotations to have shape "
            f"({expected_views}, 3, 3), "
            f"got {tuple(R_pred.shape)}."
        )

        assert (t_pred.shape == (expected_views, 3)), (
            f"Expected predicted translations to have shape "
            f"({expected_views}, 3), "
            f"got {tuple(t_pred.shape)}."
        )

        # ----------------------------------------------------------
        # Rotation consistency (geodesic distance in radians)
        # ----------------------------------------------------------

        rot_error = compute_geodesic_distance(R_pred, R_dep_cur)

        assert (rot_error.ndim == 1 and rot_error.shape[0] == expected_views), (
            f"compute_geodesic_distance() must return a tensor of shape "
            f"({expected_views},), "
            f"got {tuple(rot_error.shape)}."
        )

        rot_loss = rot_error.sum()

        # ----------------------------------------------------------
        # Translation consistency (Euclidean distance in millimeters)
        # ----------------------------------------------------------

        trans_error = torch.linalg.norm(t_pred - t_dep_cur, dim=-1)

        assert (trans_error.ndim == 1 and trans_error.shape[0] == expected_views), (
            f"Translation error must have shape "
            f"({expected_views},), "
            f"got {tuple(trans_error.shape)}."
        )

        trans_loss = trans_error.sum()

        # ----------------------------------------------------------
        # Total consistency loss
        # ----------------------------------------------------------
        lamda = (2 / config.SDD)
        consistency_loss = rot_loss + lamda*trans_loss

        assert consistency_loss.ndim == 0, (
            f"Consistency loss must be a scalar tensor, "
            f"got shape {tuple(consistency_loss.shape)}."
        )

        return consistency_loss    

    def __call__(self, pose_rot_updates: torch.Tensor, pose_trans_updates: torch.Tensor, relative_rot_updates: torch.Tensor, relative_trans_updates: torch.Tensor, img_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[float]]:
        """
        Evaluate the joint multi-view objective.

        Parameters
        ----------
        pose_rot_updates : torch.Tensor
            Axis-angle updates applied to the absolute pose of every view.
            Shape: (N, 3).
        pose_trans_updates : torch.Tensor
            Translation updates applied to the absolute pose of every view.
            Shape: (N, 3).
        relative_rot_updates : torch.Tensor
            Axis-angle updates applied to the relative transforms between the
            reference view and each non-reference view.
            Shape: (N-1, 3).
        relative_trans_updates : torch.Tensor
            Translation updates applied to the relative transforms between the
            reference view and each non-reference view.
            Shape: (N-1, 3).
        img_size : int
            Current pyramid-level image size.

        Returns
        -------
        total_loss : torch.Tensor
            Total differentiable objective.

        ncc_loss : torch.Tensor
            Weighted NCC loss.

        consistency_loss : torch.Tensor
            Geometric consistency loss.

        per_view_ncc : List[float]
            Detached NCC value for every view.
        """

        renderer = self._get_renderer(img_size)

        # ----------------------------------------------------------
        # Apply absolute pose updates
        # ----------------------------------------------------------

        R_all_cur, t_all_cur = self._apply_pose_updates(pose_rot_updates, pose_trans_updates)

        # ----------------------------------------------------------
        # Apply relative transform updates
        # ----------------------------------------------------------

        R_rel_cur, t_rel_cur = self._apply_relative_updates(relative_rot_updates, relative_trans_updates)

        # ----------------------------------------------------------
        # Predict the non-reference poses from the corrected reference pose and corrected relative transforms.
        # The [0:1] slice preserves the batch dimension (1, ...) rather than removing it as [0] would.
        # ----------------------------------------------------------

        R_pred, t_pred = self._compute_predicted_poses(R_all_cur[0:1], t_all_cur[0:1], R_rel_cur, t_rel_cur)

        # ----------------------------------------------------------
        # Render corrected absolute poses only.
        # The predicted poses are used exclusively for the consistency loss and are intentionally never rendered.
        # ----------------------------------------------------------

        projs = self._render_batch(renderer, R_all_cur, t_all_cur, img_size)

        # ----------------------------------------------------------
        # Resize target images
        # ----------------------------------------------------------

        images = self.resize_fn(self.images_batch, img_size)

        # ----------------------------------------------------------
        # Compute batched NCC
        # ----------------------------------------------------------

        similarities = self.criterion(projs, images, None)

        assert (similarities.ndim >= 1 and similarities.shape[0] == self.num_views), (
            f"criterion must return per-sample similarities of shape "
            f"({self.num_views},), "
            f"got {tuple(similarities.shape)}. "
            f"Check that mGNCCLoss / NCC does not reduce across the batch "
            f"dimension."
        )

        similarities = similarities.reshape(-1)

        # ----------------------------------------------------------
        # Weighted NCC loss
        # ----------------------------------------------------------

        weights = self.weights.to(device=similarities.device, dtype=similarities.dtype)

        ncc_loss = (weights * (1.0 - similarities)).sum()

        assert ncc_loss.ndim == 0, (
            f"NCC loss must be a scalar tensor, "
            f"got shape {tuple(ncc_loss.shape)}."
        )

        # ----------------------------------------------------------
        # Geometric consistency loss
        # ----------------------------------------------------------

        consistency_loss = self._compute_consistency_loss(R_all_cur[1:], t_all_cur[1:], R_pred, t_pred)

        # ----------------------------------------------------------
        # Final objective
        # ----------------------------------------------------------

        total_loss = ncc_loss + self.lambda_cons * consistency_loss

        assert total_loss.ndim == 0, (
            f"Total loss must be a scalar tensor, "
            f"got shape {tuple(total_loss.shape)}."
        )

        return total_loss, ncc_loss, consistency_loss, similarities.detach().cpu().tolist()


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
        self.ROT_INIT_VALUE = 1e-2

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

    # ------------------------------------------------------------------
    # Multi View Optimization
    # ------------------------------------------------------------------

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

        rot_initialization = torch.tensor([[self.ROT_INIT_VALUE, self.ROT_INIT_VALUE, self.ROT_INIT_VALUE]], device=self.device)
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

    # ------------------------------------------------------------------
    # Joint Multi View Optimization
    # ------------------------------------------------------------------

    def _optimize_scale_joint(self, objective: JointMultiViewObjective, optimizer: torch.optim.Optimizer, pose_rot_updates: nn.Parameter, pose_trans_updates: nn.Parameter, 
                              relative_rot_updates: nn.Parameter, relative_trans_updates: nn.Parameter, img_size: int, patience: int, iters_per_scale: int, history: dict, verbose: bool = True
                                ) -> Tuple[torch.Tensor, torch.Tensor, float, List[float]]:
        """
        Optimize a single image pyramid level for the joint multi-view objective.

        This method jointly optimizes the absolute pose updates for every view and
        the relative transform updates between the reference view and each
        non-reference view. The best absolute poses found during this pyramid level
        are returned. Relative transforms are intentionally not returned because
        they are recomputed from the propagated absolute poses by
        JointMultiViewObjective.set_all_poses() at the next pyramid level.

        Parameters
        ----------
        objective : JointMultiViewObjective
            Joint objective evaluated during optimization.
        optimizer : torch.optim.Optimizer
            Optimizer operating on the four update parameter tensors.
        pose_rot_updates : nn.Parameter
            Absolute rotation updates for all views (N, 3).
        pose_trans_updates : nn.Parameter
            Absolute translation updates for all views (N, 3).
        relative_rot_updates : nn.Parameter
            Relative rotation updates for the non-reference views (N-1, 3).
        relative_trans_updates : nn.Parameter
            Relative translation updates for the non-reference views (N-1, 3).
        img_size : int
            Current pyramid-level image resolution.
        patience : int
            Early stopping patience.
        iters_per_scale : int
            Maximum optimization iterations.
        history : dict
            Optimization history dictionary.
        verbose : bool, default=True
            Whether to print optimization progress.

        Returns
        -------
        best_R_all : torch.Tensor
            Best absolute rotations found during this pyramid level.
        best_t_all : torch.Tensor
            Best absolute translations found during this pyramid level.
        best_loss : float
            Lowest total objective value achieved.
        best_per_view_ncc : List[float]
            Per-view NCC values corresponding to the best solution.
        """

        # ----------------------------------------------------------
        # Initialization
        # ----------------------------------------------------------

        best_loss = float("inf")
        best_R_all = objective.R_all_init.detach().clone()
        best_t_all = objective.t_all_init.detach().clone()
        best_per_view_ncc: List[float] = []

        no_improve = 0
        min_delta = 1e-4

        # ----------------------------------------------------------
        # Optimization loop
        # ----------------------------------------------------------

        for iteration in range(iters_per_scale):

            optimizer.zero_grad()

            # ------------------------------------------------------
            # Evaluate objective
            # ------------------------------------------------------

            total_loss, ncc_loss, consistency_loss, per_view_ncc = objective(pose_rot_updates, pose_trans_updates, relative_rot_updates, relative_trans_updates, img_size)

            # ------------------------------------------------------
            # Numerical stability check
            # ------------------------------------------------------

            if not torch.isfinite(total_loss):
                if verbose:
                    print(
                        f"Encountered non-finite loss "
                        f"({total_loss.item():.6f}). "
                        f"Stopping optimization."
                    )
                break

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([pose_rot_updates, pose_trans_updates, relative_rot_updates, relative_trans_updates], max_norm=1.0)

            # ------------------------------------------------------
            # Snapshot current corrected absolute poses before the optimizer step so they correspond exactly to the loss
            # that was just evaluated.
            # ------------------------------------------------------

            R_all_cur, t_all_cur = apply_delta(objective.R_all_init, objective.t_all_init, pose_rot_updates, pose_trans_updates)

            optimizer.step()

            # ------------------------------------------------------
            # Scalar values
            # ------------------------------------------------------

            loss_val = float(total_loss.detach())
            ncc_val = float(ncc_loss.detach())
            cons_val = float(consistency_loss.detach())

            # ------------------------------------------------------
            # Best solution tracking
            # ------------------------------------------------------

            if loss_val < (best_loss - min_delta):
                best_loss = loss_val

                best_R_all = R_all_cur.detach().clone()
                best_t_all = t_all_cur.detach().clone()

                best_per_view_ncc = per_view_ncc.copy()

                no_improve = 0

            else:
                no_improve += 1

            # ------------------------------------------------------
            # History
            # ------------------------------------------------------

            history["total_loss"].append(loss_val)
            history["ncc_loss"].append(ncc_val)
            history["consistency_loss"].append(cons_val)
            history["per_view_ncc"].append(per_view_ncc)
            history["pose_rot_updates"].append(pose_rot_updates.detach().cpu().clone().tolist())
            history["pose_trans_updates"].append(pose_trans_updates.detach().cpu().clone().tolist())
            history["relative_rot_updates"].append(relative_rot_updates.detach().cpu().clone().tolist())
            history["relative_trans_updates"].append(relative_trans_updates.detach().cpu().clone().tolist())
            # history["running_best_loss"].append(best_loss)

            # ------------------------------------------------------
            # Verbose logging
            # ------------------------------------------------------

            if verbose:
                msg = (
                    f"[{iteration:03d}] "
                    f"total={loss_val:.6f} "
                    f"ncc={ncc_val:.6f} "
                    f"cons={cons_val:.6f} "
                )

                for view_idx, ncc in enumerate(per_view_ncc):
                    msg += f"v{view_idx}={ncc:.5f} "

                msg += (
                    f"best={best_loss:.6f} "
                    f"no_improve={no_improve}"
                )

                print(msg)

            # ------------------------------------------------------
            # Early stopping
            # ------------------------------------------------------

            if no_improve >= patience:
                if verbose:
                    print(
                        f"Early stopping after {iteration + 1} iterations "
                        f"(patience={patience})."
                    )
                break

        # ----------------------------------------------------------
        # Return best solution from this pyramid level
        # ----------------------------------------------------------

        return best_R_all, best_t_all, best_loss, best_per_view_ncc

    def forward_joint(self, views: List[ViewData], iters_per_scale: int = 100, patience: int = 15, lambda_cons: float = 0.1, verbose: bool = True) -> OptimizationResult:
        """
        Perform joint multi-view registration by simultaneously optimizing the
        absolute pose of every view together with the relative transforms between
        the reference view and each non-reference view.

        Unlike ``forward_multiview``, which optimizes only the reference pose while
        keeping the relative transforms fixed, this method optimizes two redundant
        representations of the scene geometry:

        1. The absolute pose of every view.
        2. The relative transform from the reference view to every non-reference
        view.

        A geometric consistency loss couples these two representations throughout
        optimization, encouraging the optimized relative transforms to agree with
        the optimized absolute poses while simultaneously maximizing image
        similarity.

        Parameters
        ----------
        views : List[ViewData]
            List of views. ``views[0]`` must be the reference view.
        iters_per_scale : int, default=100
            Maximum optimization iterations at each pyramid level.
        patience : int, default=15
            Early stopping patience for each pyramid level.
        lambda_cons : float, default=0.1
            Weight applied to the geometric consistency loss.
        verbose : bool, default=True
            Whether to print optimization progress.

        Returns
        -------
        OptimizationResult
            Registration result containing the optimized reference pose together
            with the optimized absolute and relative poses of all views.
        """

        # ----------------------------------------------------------
        # Input validation
        # ----------------------------------------------------------

        if len(views) < 2:
            raise ValueError(
                "forward_joint requires at least two views. "
                "Use SingleViewOpt or forward_multiview for single-view "
                "registration."
            )

        if not views[0].is_reference:
            raise ValueError(
                "views[0] must be the reference view. "
                "Check the order of the supplied views."
            )

        self._check_view_consistency(views)

        # ----------------------------------------------------------
        # Construct joint objective
        # ----------------------------------------------------------

        objective = JointMultiViewObjective(views=views, renderer_factory=self._create_renderer, criterion=self.criterion, resize_fn=self.resize, 
                                            normalize_fn=self.normalize, lambda_cons=lambda_cons)

        N = len(views)

        # ----------------------------------------------------------
        # Optimization parameters
        #
        # Each pyramid level starts by optimizing small updates around
        # the current initialization stored inside the objective.
        # ----------------------------------------------------------

        pose_rot_updates = nn.Parameter(torch.full((N, 3), self.ROT_INIT_VALUE, device=self.device), requires_grad=True)
        pose_trans_updates = nn.Parameter(torch.zeros((N, 3), device=self.device), requires_grad=True)
        relative_rot_updates = nn.Parameter(torch.full((N - 1, 3), self.ROT_INIT_VALUE, device=self.device), requires_grad=True)
        relative_trans_updates = nn.Parameter(torch.zeros((N - 1, 3), device=self.device), requires_grad=True)

        # ----------------------------------------------------------
        # Optimizer
        # ----------------------------------------------------------

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": pose_rot_updates,
                    "lr": self.lr[0],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": pose_trans_updates,
                    "lr": self.lr[1],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": relative_rot_updates,
                    "lr": self.lr[0],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": relative_trans_updates,
                    "lr": self.lr[1],
                    "weight_decay": self.weight_decay,
                },
            ]
        )

        # ----------------------------------------------------------
        # Optimization history
        # ----------------------------------------------------------

        history = {
            "total_loss": [],
            "ncc_loss": [],
            "consistency_loss": [],
            "per_view_ncc": [],
            "pose_rot_updates": [],
            "pose_trans_updates": [],
            "relative_rot_updates": [],
            "relative_trans_updates": [],
            "running_best_loss": [],
        }

        # ----------------------------------------------------------
        # Global best solution
        # ----------------------------------------------------------

        best_R_all = torch.cat([view.R_init for view in views], dim=0).detach().clone()
        best_t_all = torch.cat([view.t_init for view in views], dim=0).detach().clone()

        assert best_R_all.shape == (N, 3, 3), (
            f"Expected stacked rotations to have shape ({N}, 3, 3), "
            f"got {tuple(best_R_all.shape)}."
        )

        assert best_t_all.shape == (N, 3), (
            f"Expected stacked translations to have shape ({N}, 3), "
            f"got {tuple(best_t_all.shape)}."
        )

        best_loss = float("inf")
        best_per_view_ncc: List[float] = []

        # ----------------------------------------------------------
        # Multi-resolution optimization
        # ----------------------------------------------------------

        for scale in self.scales:
            img_size = views[0].image.shape[-1] // scale

            # ------------------------------------------------------
            # Propagate the best absolute poses from the previous pyramid level.
            #
            # set_all_poses() also recomputes the corresponding relative transforms from the propagated absolute poses,
            # so the relative transforms do not need to be explicitly carried between pyramid levels.
            # ------------------------------------------------------

            objective.set_all_poses(best_R_all, best_t_all)

            # ------------------------------------------------------
            # Reset optimization variables.
            # Every pyramid level optimizes small corrections around the initialization stored inside the objective.
            # ------------------------------------------------------

            with torch.no_grad():
                pose_rot_updates.copy_(torch.full_like(pose_rot_updates, self.ROT_INIT_VALUE))
                pose_trans_updates.zero_()
                relative_rot_updates.copy_(torch.full_like(relative_rot_updates, self.ROT_INIT_VALUE))
                relative_trans_updates.zero_()

            # ------------------------------------------------------
            # Verbose logging
            # ------------------------------------------------------

            if verbose:
                ref_rot = matrix_to_euler_angles(best_R_all[0:1], convention="ZXY").detach().cpu().numpy()[0]
                ref_trans = best_t_all[0].detach().cpu().numpy()

                print(
                    f"\nScale={scale:.3f} "
                    f"(image size={img_size})"
                )
                print(
                    "Reference pose: "
                    f"rot=[{ref_rot[0]:.2f}, {ref_rot[1]:.2f}, {ref_rot[2]:.2f}] "
                    f"trans=[{ref_trans[0]:.2f}, {ref_trans[1]:.2f}, {ref_trans[2]:.2f}]"
                )

            # ------------------------------------------------------
            # Optimize current pyramid level
            # ------------------------------------------------------

            scale_best_R_all, scale_best_t_all, scale_best_loss, scale_best_per_view_ncc = self._optimize_scale_joint(
                objective=objective, optimizer=optimizer, pose_rot_updates=pose_rot_updates, pose_trans_updates=pose_trans_updates, relative_rot_updates=relative_rot_updates,
                relative_trans_updates=relative_trans_updates, img_size=img_size, patience=patience, iters_per_scale=iters_per_scale, history=history, verbose=verbose)

            # ------------------------------------------------------
            # Update global best solution
            # ------------------------------------------------------

            if scale_best_loss < (best_loss - 1e-4):
                best_R_all = scale_best_R_all
                best_t_all = scale_best_t_all
                best_loss = scale_best_loss
                best_per_view_ncc = scale_best_per_view_ncc

            # ------------------------------------------------------
            # Release cached renderers from this pyramid level.
            # A new renderer will be created automatically for the
            # next image resolution if needed.
            # ------------------------------------------------------

            # objective._renderer_cache.clear()
            optimizer.state.clear()

        # ----------------------------------------------------------
        # Compute final relative transforms
        # ----------------------------------------------------------

        R_ref_final = best_R_all[0:1].expand(N - 1, -1, -1)
        t_ref_final = best_t_all[0:1].expand(N - 1, -1)
        R_rel_final, t_rel_final = compute_relative_transform(R_ref_final, t_ref_final, best_R_all[1:], best_t_all[1:])

        # ----------------------------------------------------------
        # Return optimization result
        #
        # The reference pose is returned using [0:1] rather than [0]
        # to preserve the (1, 3, 3) and (1, 3) batch dimensions used
        # throughout the optimizer.
        # ----------------------------------------------------------

        out_result = OptimizationResult(R=best_R_all[0:1], t=best_t_all[0:1], objective=best_loss, history=history, per_view_scores=best_per_view_ncc, R_all=best_R_all, 
                                    t_all=best_t_all, R_rel_all=R_rel_final, t_rel_all=t_rel_final)
        
        return out_result
        
