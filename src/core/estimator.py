import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.io as io
from torchio import Subject

from diffdrr.drr import DRR
from diffdrr.data import read

from src.utils import config as config

class PositionEstimator(nn.Module):
    def __init__(self,
                 encoder, 
                 dicom_file, 
                 crm_path, 
                 sdd=config.SDD, 
                 delx=config.DELX, 
                 size=config.IMAGE_SIZE,
                 use_kernel: bool = False
                 ):
        super().__init__()

        self.encoder = encoder
        self.size = size
        self.sdd = sdd
        self.delx = delx

        self.ct = self.load_ct(dicom_file)
        if use_kernel:
            self.kernel = self.load_kernel(crm_path)
        else:
            self.kernel = 1
        
        self.drr = DRR(
            self.ct,     
            sdd=self.sdd, 
            height=self.size, 
            delx=self.delx,
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.size, self.size, device=config.DEVICE)
            feat = self.encoder.encode(dummy)
            C, H, W = feat.shape[1:]  # skip batch dim

        self.rotation_head = nn.Sequential(
            nn.Conv2d(C, C//2, kernel_size=3, stride=2, padding=1),  # (B,C/2,H/2,W/2)
            nn.LeakyReLU(),
            nn.Conv2d(C//2, C//4, kernel_size=3, stride=2, padding=1), # (B,C/4,H/4,W/4)
            nn.LeakyReLU(),
            nn.Conv2d(C//4, C//8, kernel_size=3, stride=2, padding=1), # (B,C/8,H/8,W/8)
            nn.LeakyReLU(),
            nn.Flatten(),                                             
            nn.Linear(C//8, C//16),
            nn.LeakyReLU(),
            nn.Linear(C//16, 3),
        )

        self.translation_head = nn.Sequential(
            nn.Conv2d(C, C//2, kernel_size=3, stride=2, padding=1),  # (B,C/2,H/2,W/2)
            nn.LeakyReLU(),
            nn.Conv2d(C//2, C//4, kernel_size=3, stride=2, padding=1), # (B,C/4,H/4,W/4)
            nn.LeakyReLU(),
            nn.Conv2d(C//4, C//8, kernel_size=3, stride=2, padding=1), # (B,C/8,H/8,W/8)
            nn.LeakyReLU(),
            nn.Flatten(),                                             
            nn.Linear(C//8, C//16),
            nn.LeakyReLU(),
            nn.Linear(C//16, 3),
        )

    def forward(self, x, feat=False, return_proj=False):
        # x is (B, 1, H, W) input image (CRM masked)
        
        if not feat:
            x = self.encoder.encode(x, kernel=self.kernel)

        rot = self.rotation_head(x)
        trans = self.translation_head(x)
        pose = torch.cat([rot, trans], dim=-1)
        projection_pred = None
        if return_proj:
            projection_pred = self.project(pose)

        return projection_pred, pose
   
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
        ct_path: str,
    ) -> Subject:
        # subject = read(ct_path)
        subject = read(volume=str(ct_path), orientation="AP", center_volume=True)
        return subject

    @torch.no_grad()
    def load_kernel(self, crm_path, border=3):
        # Load the CRM mask
        crm = (io.read_image(crm_path).float().to(config.DEVICE) / 255.0).unsqueeze(0)
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
        crm = (io.read_image(crm_path).float().to(config.DEVICE) / 255.0).unsqueeze(0)
        if flip:
            crm = crm.flip(dims=[3])
        crm = F.interpolate(crm, size=(self.size, self.size), mode='bilinear', align_corners=False)
        return crm
