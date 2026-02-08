import os
import sys
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.io as io
from torchio import Subject

from diffdrr.drr import DRR
from diffdrr.data import read
from diffdrr.pose import convert

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

class PositionEstimator(nn.Module):
    def __init__(self, encoder, dicom_file, crm_path, sdd=1020.0, delx=1.0, size=128, num_candidates=None):
        super().__init__()

        self._init_args = (encoder, dicom_file, crm_path, num_candidates)

        self.encoder = encoder

        self.size = size

        self.ct = self.load_ct(dicom_file)
        self.kernel = self.load_kernel(crm_path)
        self.sdd = sdd
        
        self.delx = delx

        self.drr = DRR(
        self.ct,     
        sdd=self.sdd, 
        height=self.size, 
        delx=self.delx,
    )

        self.encoded_shape = (512, 4, 4)

        self.rotation_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=2, stride=1, padding=0),  # (B,256,3,3)
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=2, stride=1, padding=0),  # (B,128,2,2)
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=2, stride=1, padding=0),   # (B,64,1,1)
            nn.ReLU(),
            nn.Flatten(),                                             # (B,64)
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

        self.translation_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=2, stride=1, padding=0),  # (B,256,3,3)
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=2, stride=1, padding=0),  # (B,128,2,2)
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=2, stride=1, padding=0),   # (B,64,1,1)
            nn.ReLU(),
            nn.Flatten(),                                             # (B,64)
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x, feat=False):
        # yaw, pitch, roll = self.pose_net(self.latent).squeeze()
        ## yaw - rotate left (positive) right (negative) like a door, pitch - rotate outward (positive) inward (negative) like a window, roll - spin clockwise (positive) anti-clockwise (left)
        ## dx - move right (positive) left (negative), dy - outside (positive) inside (negative), dz - move down (positive) up (negative) 

        if not feat:
            x = self.encoder.encode(x, kernel=self.kernel)

        rot = self.rotation_head(x)
        trans = self.translation_head(x)
        pose = torch.cat([rot, trans], dim=-1)
        projection_pred = self.project(pose)
        return projection_pred, pose
            
    @torch.no_grad()
    def ct_slices(self, pose, res=10, out_size: tuple[int, int] = (128, 128)):
        rotation, translation = pose[:, :3], pose[:, 3:]
        pose = convert(rotation, translation, parameterization="euler_angles", convention="ZXY").to(pose.device)
        source, target = self.drr.detector(pose, None)

        vol = self.ct['volume'].data.to(dtype=torch.float32, device=pose.device).unsqueeze(0)  
        
        dtype, device = vol.dtype, vol.device
        shape   = torch.tensor(vol.shape[-3:], device=device)
        
        target = self.drr.affine_inverse(target)
        source = self.drr.affine_inverse(source)

        target = target.view(128, 128, 3)
        center = target.mean(dim=(0, 1))
        

        n = (source - center).squeeze()
        n /= (n.norm(p=2) + 1e-8)

        below = center < 0
        above = center >= shape


        k_lower = torch.where(below & (n > 0),
                            (-center) / (n * res + 1e-8),      
                            torch.zeros_like(center))

        k_upper = torch.where(above & (n < 0),
                            (shape - 1 - center) / (n * res + 1e-8),  
                            torch.zeros_like(center))

        k = torch.ceil(torch.max(torch.max(k_lower, k_upper))).item()


        target += k * res * n
        center += k * res * n
        
        steps = torch.where(
            n > 1e-6,
            (source - center) / (n * res + 1e-6),
            torch.where(
                n < 1e-6,
                center / (-n * res + 1e-6),
                torch.full_like(n, float('inf'))
            )
                    ).squeeze()

                    
        num_slices = max(int(steps[1].item()), 1) 
        offsets = torch.linspace(0, res * (num_slices - 1), num_slices, device=vol.device)
        planes = target.unsqueeze(0) + offsets[:, None, None, None] * n
    
        grid = planes[..., [2, 1, 0]] 
        grid = grid / (shape[[2, 1, 0]] - 1) * 2 - 1  
        grid = grid.unsqueeze(1) 

        sampled = F.grid_sample(
            vol.expand(num_slices, -1, -1, -1, -1), 
            grid, 
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        ).squeeze(1).squeeze(1)

        return torch.flip(sampled, dims=[0])

    def project(self, pose):
        rot = pose[:, :3]
        trans = pose[:, 3:]
        projection = self.drr(rot, trans, parameterization="euler_angles", convention="ZXY")
        mn = projection.amin(dim=(-2, -1), keepdim=True) 
        mx = projection.amax(dim=(-2, -1), keepdim=True) 
        projection = 1 - (projection - mn) / (mx - mn)
        return projection

    def load_ct(
        self,
        dicom_file: str,
    ) -> Subject:
        subject = read(dicom_file)
        return subject

    @torch.no_grad()
    def load_kernel(self, crm_path, border=3):
        crm = (io.read_image(crm_path).float().to("cuda") / 255.0).unsqueeze(0)
        crm = F.interpolate(crm, size=(self.size, self.size), mode='bilinear', align_corners=False)
        kernel = (crm != 0).float()

        if border > 0:
            kernel_in = F.max_pool2d(
                1.0 - kernel.float(),         
                kernel_size=2 * border + 1,
                stride=1,
                padding=border,
            )
            kernel_valid = (kernel_in == 0)       
        else:
            kernel_valid = kernel.bool()      

        return kernel_valid.detach()

    @torch.no_grad()
    def load_crm(self, crm_path, flip=True):
        crm = (io.read_image(crm_path).float().to("cuda") / 255.0).unsqueeze(0)
        if flip:
            crm = crm.flip(dims=[3])
        crm = F.interpolate(crm, size=(self.size, self.size), mode='bilinear', align_corners=False)
        return crm

