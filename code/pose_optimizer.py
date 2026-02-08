import copy
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torchvision.transforms.functional import gaussian_blur

from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d
from utils import PositionLoss                   
from MambaGlue.mambaglue import MambaGlue
from MambaGlue.mambaglue import SuperPoint

class PoseOptimizer(nn.Module):
    def __init__(self, net):
        super().__init__()
        
        self.net = net
        self.device = self.net.device
        self.sobel = self.net.sobel

        self.mambaglue = MambaGlue()
        self.superpoint = SuperPoint()
        self.delx = self.net.delx
        
        self.dict = {
                "image0": {
                    "keypoints": None,
                    "keypoint_scores": None,
                    "descriptors": None,
                    "image": None
                },
                "image1": {
                    "keypoints": None,
                    "keypoint_scores": None,
                    "descriptors": None,
                    "image": None
                }
            }

        pose = None

        self.drr = self.net.drr

        mask = torch.zeros_like(self.net.encoder[0].weight)
        mask[:, 0, :, :] = 1.0

        self.net.encoder[0].weight.register_hook(lambda grad: grad * mask.to(grad.device))

        self.register_buffer('kernel', self.net.kernel)

        self.crm = None
        self.encoded_crm = None

        self.loss_optimizer = torch.optim.SGD(
            [
                {"params": self.net.encoder.parameters(), "lr": 1e-4},
            ],
        )
    
        self.position_loss = PositionLoss()

    def update_crm(self, crm):
        self.crm = gaussian_blur(crm, 5, 1.0)
        # self.crm = crm

        self.dict['image0']['image'] = self.crm.clone()
        self.dict['image0']['keypoints'] = None

        data = self.superpoint(self.dict['image0'])

        data = self.filter_keypoints(data)
        

        for key in data.keys():
            self.dict['image0'][key] = data[key].detach()

    def gain(self, projection, crm, scales=(128, 64, 32, 16, 8), weights=None):
        # kp0    = self.dict['image0']['keypoints']
        # desc0  = self.dict['image0']['descriptors']
        # score0 = self.dict['image0']['keypoint_scores']
        
        # sp_input = {'image': projection,
        #             'keypoints': kp0,
        #             'keypoint_scores': score0}

        # sp_out   = self.superpoint(sp_input)
        # sp_out   = self.net.filter_keypoints(sp_out)
        
        # desc1    = sp_out['descriptors']
        
        # cos_scores = F.cosine_similarity(desc0, desc1, dim=2)      
        # cos_gain   = 1.0 - cos_scores.mean(dim=1)
        
        # encoded_projection = self.superpoint.encode(projection * self.kernel)
        encoded_projection = self.sobel(projection) * self.kernel
        encoded_crm = self.sobel(crm) * self.kernel

        B = encoded_projection.size(0)
        n_levels = len(scales)


        if weights is None:
            weights = torch.ones(n_levels, device=self.device)

        
        # per_level = []
        # for i, s in enumerate(scales):
        #     ps = F.interpolate(encoded_projection, size=(s, s), mode="bilinear",
        #                     align_corners=False)
        #     sc = F.interpolate(encoded_crm, size=(s, s), mode="bilinear",
        #                     align_corners=False)

        #     lvl_gain = F.mse_gain(ps, sc, reduction='none').view(B, -1).mean(1)
        #     per_level.append(weights[i] * lvl_gain.unsqueeze(0))

        # multiscale_gain = torch.stack(per_level, dim=1).sum(1)

        gain_fn = MultiscaleNormalizedCrossCorrelation2d(patch_sizes=scales, patch_weights=weights)

        mncc_gain =  gain_fn(encoded_projection, encoded_crm)

        total_gain = mncc_gain
        return total_gain

    def forward(
        self,
        iters: int = 250,
        patience: int = 25,
        min_delta: float = 1e-3,
        verbose: bool = True,
    ):  

        projection, pose, _ = self.net(self.crm)

        B = pose.shape[0]

        best_pose = pose.detach().clone()
        best_gain       = torch.full((B,), -float('inf'), device=pose.device)
        best_projection = torch.zeros((B, 1, 128, 128), device=pose.device)

        no_improve  = 0   
        init_gain = 0.0

        orig_net = copy.deepcopy(self.net)
        orig_net.eval()
        device = next(self.net.parameters()).device
        orig_net.to(device)

        for p in orig_net.parameters():  
            p.requires_grad_(False) 


        gain_optimizer = torch.optim.SGD(
            [
                {"params": self.net.encoder.parameters(), "lr": 1e-4},
            ],
            maximize=True,
        )
        
        # new_pose = self.update_pose(projection, pose)

        # rot = new_pose[:, :3].clone()
        # rot.requires_grad = True

        # trans = new_pose[:, 3:].clone()
        # trans.requires_grad = True

        # gain_optimizer = torch.optim.Adam(
        #     [
        #         {"params": rot, "lr": 0.05},
        #         {"params": trans, "lr": 0.75},
        #     ],

        #     maximize=True,
        # )

        for step in range(iters):        
            gain_optimizer.zero_grad()
            # pose = torch.cat([rot, trans], dim=-1)
            # projection = self.net.project(pose)
            projection, crm_pose, crm_features = self.net(self.crm)

            if step == 0:
                best_projection = projection.clone().detach()
                with torch.no_grad():
                    _, drr_pose, proj_features = orig_net(best_projection)
                init_position_loss = self.position_loss(drr_pose, crm_pose).item()
                init_features_mse_loss = F.mse_loss(crm_features, proj_features).item()
                init_features_cos_sim = F.cosine_similarity(crm_features, proj_features).mean().item()

            # with torch.no_grad():
            #         proj_features, drr_pose, proj_features = orig_net(best_projection)

            gain = self.gain(projection, self.crm)

            if step == 0:
                init_gain = gain.item()

            gain.sum().backward()

            gain_optimizer.step()


            improved = torch.logical_or(torch.isneginf(best_gain), gain > best_gain + min_delta * best_gain.abs())
            if improved:
                no_improve  = 0
                best_projection = projection.detach().clone()
                best_pose   = pose.detach().clone()
                best_gain   = gain.detach().clone()
                best_encoder = copy.deepcopy(self.net.encoder.state_dict())

            else:
                no_improve += 1

            if verbose:
                msg = (
                    f"[{step:03d}]  "
                    f"gain={gain.max():.5f}  "
                    f"best_max={best_gain.max():.5f}  "
                    f"no_improve(max)={no_improve}  "
                )
                print(msg)
            
            if no_improve >= patience or step == iters-1 or torch.isnan(gain):
                self.net.encoder.load_state_dict(best_encoder)
                with torch.no_grad():
                    _, drr_pose, proj_features = orig_net(best_projection)
                    projection, crm_pose, crm_features = self.net(self.crm)
                position_loss = self.position_loss(drr_pose, crm_pose).item()
                features_mse_loss = F.mse_loss(crm_features, proj_features).item()
                features_cos_sim = F.cosine_similarity(crm_features, proj_features).mean().item()
                if verbose:
                    print(f"Early stop at step {step} (patience={patience})")
                    print(f"position loss:{position_loss}, features loss:{features_mse_loss}, feature sim:{features_cos_sim}")
                break
        
        init_results = [init_gain, init_position_loss, init_features_mse_loss, init_features_cos_sim]
        final_results = [best_gain.item(), position_loss, features_mse_loss, features_cos_sim]

        return best_pose, best_projection, step-patience, init_results, final_results
    
    @torch.no_grad
    def update_pose(self, projection, cur_pose):
        pts1, pts0 = self.find_matches(projection)

        H, inliers = self.rigid_2d_ransac(pts1, pts0,
                                n_iter=1000,
                                inlier_thresh=5,  
                                min_inliers=8)
        flow = pts0[inliers] - pts1[inliers]
        median_flow = flow.median(dim=0).values if flow.shape[0] > 0 else [0.0, 0.0]
        median_dx, median_dy = median_flow        
        flow_mm = torch.tensor([median_dx * self.delx,
                                median_dy * self.delx], device=median_flow.device)

        r11, r21          = H[:, 0, 0], H[:, 1, 0]

        dyaw   = torch.atan2(r21, r11)  
        new_pose = cur_pose.clone()
        new_pose[:, 2] += dyaw
        new_pose[:, 3] += flow_mm[0]
        new_pose[:, 5] += flow_mm[1]
        # return new_pose, flow_mm, pts1, pts0, inliers, H
        return new_pose

    def filter_keypoints(self, data, pad_val=float('nan')):
        keypoints = data["keypoints"]          
        scores    = data["keypoint_scores"]   
        desc      = data["descriptors"]        
        B, N, _   = keypoints.shape
        H, W      = self.kernel.shape[-2:]

        kernel_valid = self.kernel

        xy_int = keypoints.round().long().clamp(min=0)
        x      = xy_int[..., 0].clamp(max=W - 1)
        y      = xy_int[..., 1].clamp(max=H - 1)

        flat_mask = kernel_valid.view(-1, H * W) 
        flat_idx  = y * W + x                

        if flat_mask.size(0) == 1:                
            flat_mask = flat_mask.expand(B, -1)

        keep_mask  = torch.gather(flat_mask, 1, flat_idx)    

        counts = keep_mask.sum(dim=1)              
        if (counts == 0).any():
            raise ValueError("No valid keypoints left in at least one batch")

        M      = int(counts.max())                

        order      = keep_mask.float().sort(dim=1, descending=True).indices   
        sel_idx    = order[:, :M]                                 

        kp_sel     = keypoints.gather(1, sel_idx.unsqueeze(-1)
                                            .expand(-1, -1, 2))     
        sc_sel     = scores.gather(1, sel_idx)                       

        C          = desc.size(2)
        desc_sel   = desc.gather(1, sel_idx.unsqueeze(-1)
                                        .expand(-1, -1, C))        

        valid_M_sc    = (torch.arange(M, device=keypoints.device)
                        .unsqueeze(0) < counts.unsqueeze(1))         
        valid_M_kp = valid_M_sc.unsqueeze(-1).expand(-1, -1, 2)        
        valid_M_desc = valid_M_sc.unsqueeze(-1).expand(-1, -1, C)           

        pad_kp     = torch.full_like(kp_sel,   pad_val)
        pad_sc     = torch.full_like(sc_sel,   pad_val)
        pad_desc   = torch.full_like(desc_sel, pad_val)

        out_kp     = torch.where(valid_M_kp, kp_sel,   pad_kp)
        out_sc     = torch.where(valid_M_sc,     sc_sel,  pad_sc)
        out_desc   = torch.where(valid_M_desc,  desc_sel, pad_desc)

        return {
            "keypoints"       : out_kp,         
            "keypoint_scores" : out_sc,         
            "descriptors"     : out_desc,      
        }


    @torch.no_grad
    def find_matches(self, projection, min_matches=8, pad_val=float('nan')):
        device = self.device
        self.dict["image1"]["image"] = projection
        data = self.superpoint(self.dict["image1"])
        data = self.filter_keypoints(data)
        for key in data:
            self.dict['image1'][key] = data[key]

        results = self.mambaglue(self.dict)
        k0, k1 = self.dict["image0"]["keypoints"], self.dict["image1"]["keypoints"]
        matches0 = results["matches0"]
        mask = matches0 != -1
        B, N = matches0.shape
        counts = mask.sum(dim=1)
        # if (counts < min_matches).any():
        #     raise ValueError("too few matches")
        M = int(counts.max())
        order     = mask.float().sort(dim=1, descending=True).indices                       
        selected_idx   = order[:, :M]                                                    

        k0_selected    = k0.gather(1, selected_idx.unsqueeze(-1).expand(-1, -1, 2))               

        match_idx = matches0.gather(1, selected_idx).clamp(min=0)                            
        k1_selected    = k1.gather(1, match_idx.unsqueeze(-1).expand(-1, -1, 2))             

        valid_M   = (torch.arange(M, device=device).unsqueeze(0) < counts.unsqueeze(1)) 
        valid_M   = valid_M.unsqueeze(-1).expand(-1, -1, 2)                             

        pad_tensor = torch.full_like(k0_selected, pad_val)
        pts0 = torch.where(valid_M, k0_selected, pad_tensor) 
        pts1 = torch.where(valid_M, k1_selected, pad_tensor)

        return pts1, pts0

    @torch.no_grad
    def compute_transformation(self, pts1: torch.Tensor, pts0: torch.Tensor
    ):
        """
        pts1 → pts0   (B,2) each
        Returns R (2×2) and t (2,)
        """
        c0 = pts0.mean(dim=1, keepdim=True)
        c1 = pts1.mean(dim=1, keepdim=True)
        X0, X1 = pts0 - c0, pts1 - c1

        H   = X1.transpose(1, 2) @ X0                 
        U, _, Vt = torch.linalg.svd(H)                
        R = Vt.transpose(1, 2) @ U.transpose(1, 2)   

        detR = torch.det(R)                          
        refl  = detR < 0
        if refl.any():
            Vt[refl, -1, :] *= -1                    
            R = Vt.transpose(1, 2) @ U.transpose(1, 2)

        t = (c0.squeeze(1) - (R @ c1.transpose(1, 2)).squeeze(-1))  
        H = torch.zeros((R.size(0), 3, 3), device=R.device, dtype=R.dtype)
        H[:, :2, :2] = R
        H[:, :2,  2] = t
        H[:,  2,  2] = 1.0
        return H
        

    def rigid_2d_ransac(
        self,
        pts1: torch.Tensor,        
        pts0: torch.Tensor,        
        *,
        n_iter: int      = 500,
        inlier_thresh: float = 2,      
        min_inliers: int  = 8,
):
        B, N, _ = pts0.shape
        device, dtype = pts0.device, pts0.dtype
        valid = ~(torch.isnan(pts0).any(-1) | torch.isnan(pts1).any(-1))  # (B,M)

        best_err = torch.full((B, N), float("inf"), device=device, dtype=dtype)    # (B,M)
        best_inliers = torch.zeros(B, device=device, dtype=torch.long)
        best_mask    = torch.zeros_like(valid)
        best_H       = torch.eye(3, device=device, dtype=dtype).repeat(B, 1, 1)

        if (valid.sum(dim=1) < min_inliers).any():
            return best_H, best_mask

        weight = valid.float()
        for _ in range(n_iter):
            idx = torch.multinomial(weight, min_inliers, replacement=False)                              # (B,4)
            p1 = torch.gather(pts1, 1, idx.unsqueeze(-1).expand(-1, -1, 2))
            p0 = torch.gather(pts0, 1, idx.unsqueeze(-1).expand(-1, -1, 2))
            H = self.compute_transformation(
                p1,
                p0,
            )
            pts1_h = torch.cat([pts1, torch.ones_like(pts1[..., :1])], dim=-1)
            pts1_transformed_h = torch.bmm(pts1_h, H.transpose(1, 2))
            pts1_hat = pts1_transformed_h[..., :2] / pts1_transformed_h[..., 2:].clamp(min=1e-8)
            err = torch.linalg.norm(pts1_hat - pts0, dim=2)        # reprojection error
            err[~valid] = float("inf")
            inliers   = err < inlier_thresh                                      # (B,M)
            inl_count = inliers.sum(1)                                           # (B,)
            is_above_thres = inl_count >= min_inliers
            total_err = torch.where(inliers, err, torch.zeros_like(err)).sum(1)
            prev_total_err = torch.where(best_mask, best_err, torch.zeros_like(best_err)).sum(1)

            if not is_above_thres.any():    
                continue
            
            better = (
            ((inl_count > best_inliers) & is_above_thres) |
            ((inl_count == best_inliers) & (total_err < prev_total_err))
        )


            best_inliers[better] = inl_count[better]
            best_H[better]   = H[better]
            best_err[better] = err[better]
            best_mask[better] = inliers[better]

            if (best_inliers.float() > 0.9 * valid.sum(1)).all():
                break

        best_mask = torch.zeros_like(valid)
        _, topk_idx = torch.topk(best_err, best_inliers.min().clamp_min(min_inliers).item(),
                                largest=False, dim=1)
        best_mask.scatter_(1, topk_idx, True)
        return best_H, best_mask
