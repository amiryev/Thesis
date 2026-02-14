import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d

from src.core.layers import Sobel
from src.utils import image_processing
from src.utils import loss as loss_utils

class PoseOptimizer(nn.Module):
    def __init__(self, position_estimator):
        super().__init__()
        
        self.position_estimator = position_estimator
        self.position_estimator.eval() # Ensure estimator is in eval mode

        self.encoder = self.position_estimator.encoder
        # We need a copy of the encoder for reference features (orig_encoder) mechanism
        # However, the original code essentially fine-tunes the encoder's *feature extraction*
        # relative to the CRM.
        self.orig_encoder = copy.deepcopy(self.encoder)

        self.encoder.train()
        self.orig_encoder.eval()

        # Freeze everything initially
        for p in self.position_estimator.parameters():  
            p.requires_grad_(False) 
        for p in self.orig_encoder.parameters():  
            p.requires_grad_(False) 
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # Unfreeze specific parts of the encoder for optimization
        # In the original code: self.encoder.encoder.parameters() (The VGG features)
        for p in self.encoder.encoder.parameters():
            p.requires_grad_(True)
            
        # Hook for masking gradients on the very first conv layer of VGG 
        # (Original: mask[:, 0, :, :] = 1.0)
        # The logic was: keep only the gradient for the first channel (input image), 
        # effectively ignoring Sobel channels during this specific backprop?
        mask = torch.zeros_like(self.encoder.encoder[0].weight)
        mask[:, 0, :, :] = 1.0
        self.encoder.encoder[0].weight.register_hook(lambda grad: grad * mask.to(grad.device))

        self.optimizer = torch.optim.SGD(
                [
                    {"params": self.encoder.encoder.parameters(), "lr": 0.5e-3},
                ],
                maximize=True,
            )
        
        self.sobel = Sobel()
        self.register_buffer('kernel', self.position_estimator.kernel)

        self.crm = None
    
    def update_crm(self, crm):
        # Apply Gaussian blur to CRM as preprocessing
        self.crm = image_processing.gaussian_blur_tensor(crm, 5, 1.0)

    def gain(self, projection, crm, scales=(128, 64, 32, 16, 8), weights=None):
        projection_gradients = self.sobel(projection) * self.kernel
        crm_gradients = self.sobel(crm) * self.kernel

        # B = projection_gradients.size(0)
        n_levels = len(scales)

        if weights is None:
            weights = torch.ones(n_levels, device=projection.device)

        gain_fn = MultiscaleNormalizedCrossCorrelation2d(patch_sizes=scales, patch_weights=weights)

        mncc_gain =  gain_fn(projection_gradients, crm_gradients)
        return mncc_gain

    def forward(
        self,
        iters: int = 250,
        patience: int = 25,
        min_delta: float = 1e-3,
        verbose: bool = True,
    ):  
        if self.crm is None:
            raise RuntimeError("CRM not set. Call update_crm() first.")

        # Initial prediction
        with torch.no_grad():
            projection, pose = self.position_estimator(self.crm)

        B = pose.shape[0]

        best_pose = pose.detach().clone()
        best_gain       = torch.full((B,), -float('inf'), device=pose.device)
        best_projection = torch.zeros((B, 1, 128, 128), device=pose.device)
        best_encoder = copy.deepcopy(self.encoder.encoder.state_dict())

        no_improve  = 0   
        init_gain = 0.0
        
        init_results = [] # To be populated
        final_results = []

        for step in range(iters):        
            self.optimizer.zero_grad()
            
            # Forward pass through the "trainable" encoder to get features of CRM
            # Note: We pass kernel to mask inputs
            crm_features = self.encoder.encode(self.crm, kernel=self.kernel)
            
            # Predict pose/projection from CRM features
            projection, crm_pose  = self.position_estimator(crm_features, feat=True)
            
            if step == 0:
                best_projection = projection.clone().detach()
                
                # Get baseline features from original (frozen) encoder on the projected image
                with torch.no_grad():
                    proj_features = self.orig_encoder.encode(best_projection, kernel=self.kernel)
                    # For logging mostly
                    # _, drr_pose = self.position_estimator(proj_features, feat=True)
                
                init_position = crm_pose.clone()
                init_features_mse_loss = F.mse_loss(crm_features, proj_features).item()
                init_features_cos_sim = F.cosine_similarity(crm_features, proj_features).mean().item()
            
            # Optimization objective: Maximize Gain (MNCC between projection and CRM)
            gain = self.gain(projection, self.crm)

            if step == 0:
                init_gain = gain.item()
                init_results = [init_gain, init_position, init_features_mse_loss, init_features_cos_sim]

            gain.sum().backward()
            self.optimizer.step()

            # Tracking best
            improved = torch.logical_or(torch.isneginf(best_gain), gain > best_gain + min_delta * best_gain.abs())
            if improved:
                no_improve  = 0
                best_projection = projection.detach().clone()
                best_pose   = crm_pose.detach().clone()
                best_gain   = gain.detach().clone()
                best_encoder = copy.deepcopy(self.encoder.encoder.state_dict())
            else:
                no_improve += 1

            if verbose and step % 10 == 0:
                msg = (
                    f"[{step:03d}]  "
                    f"gain={gain.max():.5f}  "
                    f"best_max={best_gain.max():.5f}  "
                    f"no_improve={no_improve}  "
                )
                print(msg)
            
            # Early stopping
            if no_improve >= patience or step == iters-1 or torch.isnan(gain):
                # Restore best model state
                self.encoder.encoder.load_state_dict(best_encoder)
                
                # Final metrics computation
                with torch.no_grad():
                    proj_features = self.orig_encoder.encode(best_projection, kernel=self.kernel)
                    crm_features = self.encoder.encode(self.crm, kernel=self.kernel)
                    _, crm_pose  = self.position_estimator(crm_features, feat=True)

                position = crm_pose.clone()
                features_mse_loss = F.mse_loss(crm_features, proj_features).item()
                features_cos_sim = F.cosine_similarity(crm_features, proj_features).mean().item()
                
                if verbose:
                    print(f"Stop at step {step} (patience={patience})")
                    print(f"features loss:{features_mse_loss}, feature sim:{features_cos_sim}")
                break
        
        final_results = [best_gain.item(), position, features_mse_loss, features_cos_sim]

        return best_pose, best_projection, step-patience, init_results, final_results
