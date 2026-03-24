import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from collections import OrderedDict
import torch
from torch import nn
import numpy as np
import torchvision.transforms as T
from PIL import Image
import torch
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
from torchvision.transforms.functional import gaussian_blur

from old_code.pose_optimizer_alt import PoseOptimizer
from old_code.position_estimator import PositionEstimator
from old_code.xray_encoder import XrayEncoder
from old_code.dataset import PoseDataset
from old_code.utils import apply_noise_brightness_contrast, PositionLoss

index = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = '/mnt/storage/users/amiry//git/Thesis/models'
data_dir = '/mnt/storage/users/amiry/git/Thesis/datasets'


num_trials = 1
# dicom_file = f"datasets/CT/{index}.nii.gz"
crm_path = f"{data_dir}/CRM/{index}.png"
output_dir = f"xray_depthor/figures/{index}"
os.makedirs(output_dir, exist_ok=True)

# ckpt = torch.load("models/masked_recon_6.pth", map_location=device, weights_only=True)
# state_dict = ckpt["model_sd"]

# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k.replace("module.", "")  # remove "module." prefix
#     new_state_dict[name] = v

encoder = XrayEncoder(
    device=device,
    size=128,
    patch_size=32,
).to(device)

# encoder.load_state_dict(new_state_dict, strict=True)
ckpt = torch.load(f"{model_dir}/position_estimator_{index}.pth", map_location=device, weights_only=True)
state_dict = ckpt["model_sd"]

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "")  # remove "module." prefix
    new_state_dict[name] = v

position_estimator = PositionEstimator(
    encoder=encoder,
    dicom_file=f"{data_dir}/CT/{index}.nii.gz",
    crm_path=f"{data_dir}/CRM/{index}.png",
).to(device)

position_estimator.load_state_dict(new_state_dict, strict=True)

pose_optimizer = PoseOptimizer(position_estimator=position_estimator).to(device)

crm = position_estimator.load_crm(crm_path)
init_gains = []
final_gains = []

init_feat_mse_losses = []
final_feat_mse_losses = []

init_feat_cos_similiarities = []
final_feat_cos_similiarities = []

init_position_losses = []
final_position_losses = []

steps = []

poses = PoseDataset(position_estimator.ct, steps=5, min_intersections=500, device="cpu")

# loss = PositionLoss()
loss = PositionLoss().to(device)

num_iterations = 10
for j_iter in range(num_iterations):
    print(f'iter {j_iter}')
    i = torch.randint(len(poses), (1,)).item()
    rotation, translation = poses[i]
    # gt_pose = torch.cat([rotation, translation], dim=-1).to(device)
    gt_pose = torch.cat([rotation, translation], dim=-1).unsqueeze(0).to(device)
    new_crm = position_estimator.project(gt_pose)
    # new_crm = apply_noise_brightness_contrast(new_crm, noise_std=0.015, brightness=0.25, contrast=0.55) * (crm > 0.0)
    # angle_deg = random.uniform(-0, 0)
    # s = random.uniform(1.0, 1.1)
    # new_crm = TF.rotate(crm, angle_deg, interpolation=TF.InterpolationMode.BILINEAR)
    pose_optimizer.update_crm(new_crm)
    # new_crm = gaussian_blur(new_crm, 5, 1.0)
    
    with torch.no_grad():
        projection_pred, pose_pred = position_estimator(new_crm)

    best_pose, projection_optimized, step, init_results, final_results = pose_optimizer()
    
    init_gain, init_position, init_features_mse_loss, init_features_cos_sim = init_results
    final_gain, final_position, final_features_mse_loss, final_features_cos_sim = final_results

    steps.append(step)

    init_gains.append(init_gain)
    final_gains.append(final_gain)

    init_feat_mse_losses.append(init_features_mse_loss)
    final_feat_mse_losses.append(final_features_mse_loss)

    init_feat_cos_similiarities.append(init_features_cos_sim)
    final_feat_cos_similiarities.append(final_features_cos_sim)

    init_position_losses.append(loss(init_position, gt_pose).item())
    final_position_losses.append(loss(final_position, gt_pose).item())


    cols = 3
    plt.figure(figsize=(12, 6))
    iterations = 1
    if j_iter % 10 == 0:
        for i in range(iterations):
            # ---------- raw images ----------
            gt_img       = new_crm[i].squeeze().cpu().numpy()
            initial_img  = (projection_pred      * position_estimator.kernel)[i].squeeze().cpu().numpy()
            optimized_img= (projection_optimized * position_estimator.kernel)[i].squeeze().cpu().numpy()

            # ---------- Sobel × kernel ----------
            with torch.no_grad():
                sobel_gt      = (pose_optimizer.sobel(new_crm[i:i+1])                 * position_estimator.kernel).squeeze().cpu().numpy()
                sobel_init    = (pose_optimizer.sobel(projection_pred[i:i+1])     * position_estimator.kernel).squeeze().cpu().numpy()
                sobel_opt     = (pose_optimizer.sobel(projection_optimized[i:i+1])* position_estimator.kernel).squeeze().cpu().numpy()

            # ---- row indices ----
            row_img = 2 * i + 1          # first row (images)
            row_sob = row_img + 1        # second row (Sobel maps)

            # ---- plot images ----
            imgs   = [gt_img,  initial_img,  optimized_img]
            titles = [f"C{i+1}: GT", f"Initial: {init_gain:.5f}", f"Optimised: {final_gain:.5f}"]
            for j, (im, t) in enumerate(zip(imgs, titles)):
                plt.subplot(iterations * 2, cols, (row_img - 1) * cols + j + 1)
                plt.imshow(im, cmap="gray"); plt.title(t); plt.axis("off")

            # ---- plot Sobel maps ----
            sobels = [sobel_gt, sobel_init, sobel_opt]
            sob_titles = ["GT Sobel", "Init Sobel", "Opt Sobel"]
            for j, (im, t) in enumerate(zip(sobels, sob_titles)):
                plt.subplot(iterations * 2, cols, (row_sob - 1) * cols + j + 1)
                plt.imshow(im, cmap="gray"); plt.title(t); plt.axis("off")
            

        plt.tight_layout()
        plt.savefig(f"{output_dir}/predicted_projection_{index}_{j_iter}.png", bbox_inches='tight')
        plt.close()

    # ct_slices = position_estimator.ct_slices(best_pose[0].unsqueeze(0))

    # ct_slices =[s.cpu().numpy() for s in ct_slices]
    # n = len(ct_slices)
    # cols = min(n, 8)           
    # rows = (n + cols - 1) // cols

    # fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

    # axes = axes.flatten() if n > 1 else [axes]

    # def window_image(img, level=40, width=400):
    #     lower = level - width / 2
    #     upper = level + width / 2
    #     windowed = np.clip((img - lower) / (upper - lower), 0, 1)
    #     return windowed

    # for i in range(len(ct_slices)):
    #     img = window_image(ct_slices[i], level=40, width=400)
    #     axes[i].imshow(img, cmap='gray')
    #     axes[i].set_title(f"Slice {i}")
    #     axes[i].axis('off')

    # for i in range(len(ct_slices), len(axes)):
    #     axes[i].axis('off')

    # plt.tight_layout()

    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/ct_slices_{index}.png", bbox_inches='tight')
    # plt.close()

plt.figure(figsize=(10, 4))
plt.plot(steps)
plt.title("Steps")
plt.xlabel("Index")
plt.ylabel("Step Value")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/steps.png", bbox_inches='tight')

plt.figure(figsize=(10, 4))
plt.plot(init_gains, label='Initial Gain')
plt.plot(final_gains, label='Final Gain')
plt.axhline(y=2.0, color='red', linestyle='--', linewidth=1.5)
plt.title("Multiscale Gradient Normalized Cross Correlation ")
plt.xlabel("Index")
plt.ylabel("Gain Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/gains.png", bbox_inches='tight')


plt.figure(figsize=(10, 4))
plt.plot(init_position_losses, label='Final Position Loss')
plt.plot(final_position_losses, label='Initial Position Loss')
plt.title("Position Loss")
plt.xlabel("Index")
plt.ylabel("Loss Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/position_losses.png", bbox_inches='tight')

plt.figure(figsize=(10, 4))
plt.plot(init_feat_mse_losses, label='Final Features MSE Loss')
plt.plot(final_feat_mse_losses, label='Initial Features MSE Loss')
plt.title("Feature MSE Loss")
plt.xlabel("Index")
plt.ylabel("Loss Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/features_mse_losses.png", bbox_inches='tight')

plt.figure(figsize=(10, 4))
plt.plot(init_feat_cos_similiarities, label='Final Features Cosine Similarity')
plt.plot(final_feat_cos_similiarities, label='Initial Features Cosine Similarity')
plt.title("Feature Cosine Similarity")
plt.xlabel("Index")
plt.ylabel("Similarity Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/features_cosine_similarities.png", bbox_inches='tight')