import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionLoss(nn.Module):
    def __init__(self, scales=(0.01, 0.001, 0.01)):
        super().__init__()
        self.register_buffer("scales", torch.tensor(scales).float())  # (3,)

    def forward(self, vector_pred, vector_gt):
        # Normalize translations
        if vector_gt.device != vector_pred.device:
            vector_gt = vector_gt.to(vector_pred.device)

        vec_pred_norm = vector_pred.clone()
        vec_gt_norm = vector_gt.clone()

        vec_pred_norm[:, 3:] = vector_pred[:, 3:] * self.scales
        vec_gt_norm[:, 3:] = vector_gt[:, 3:] * self.scales

        # Losses
        rot_loss = F.mse_loss(vec_pred_norm[:, :3], vec_gt_norm[:, :3])
        trans_loss = F.mse_loss(vec_pred_norm[:, 3:], vec_gt_norm[:, 3:])

        total_loss = rot_loss + trans_loss

        return total_loss
