import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import OrderedDict

from src.utils import config as config
from src.core.encoder import XrayEncoder
from src.core.estimator import PositionEstimator
from src.core.registration import PoseOptimizer
from src.data.dataset import PoseDataset
from src.utils import image_processing
from src.utils.loss import PositionLoss

def run_inference(
    patient_index: int = 3, 
    crm_image_path: str = None, 
    ct_volume_path: str = None,
    output_base_dir: str = None
):
    # Setup paths
    if crm_image_path is None:
        crm_image_path = str(config.CRM_DIR / f"{patient_index}.png")
    if ct_volume_path is None:
        ct_volume_path = str(config.CT_DIR / f"{patient_index}.nii.gz")
    if output_base_dir is None:
        output_base_dir = str(config.OUTPUT_DIR / f"{patient_index}")
    
    os.makedirs(output_base_dir, exist_ok=True)
    device = config.DEVICE
    print(f"Running inference for Patient {patient_index} on {device}")

    # 1. Initialize Models
    print("Loading models...")
    encoder = XrayEncoder(
        device=device,
        size=config.IMAGE_SIZE,
        patch_size=config.PATCH_SIZE,
    ).to(device)

    # Load Encoder Weights
    # Note: Adjust path to where your checkpoints are stored
    encoder_ckpt_path = "models/masked_recon_6.pth" # Placeholder path, adjust as needed
    # If you have weights to load:
    # ckpt = torch.load(encoder_ckpt_path, map_location=device)
    # encoder.load_state_dict(ckpt["model_sd"], strict=False) 

    position_estimator = PositionEstimator(
        encoder=encoder,
        dicom_file=ct_volume_path,
        crm_path=crm_image_path,
    ).to(device)

    # Load Estimator Weights
    estimator_ckpt_path = f"models/position_estimator_{patient_index}.pth" # Original path structure
    if os.path.exists(estimator_ckpt_path):
        print(f"Loading checkpoint from {estimator_ckpt_path}")
        ckpt = torch.load(estimator_ckpt_path, map_location=device, weights_only=True)
        state_dict = ckpt["model_sd"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        position_estimator.load_state_dict(new_state_dict, strict=True)
    else:
        print(f"Warning: Checkpoint {estimator_ckpt_path} not found. Running with random/init weights.")

    # 2. Setup Optimizer and Data
    pose_optimizer = PoseOptimizer(position_estimator=position_estimator).to(device)
    loss_fn = PositionLoss().to(device)
    
    # Load CRM image
    crm = position_estimator.load_crm(crm_image_path)

    # Initialize logging lists
    init_gains, final_gains = [], []
    init_feat_mse, final_feat_mse = [], []
    init_feat_sim, final_feat_sim = [], []
    init_pos_loss, final_pos_loss = [], []
    steps_log = []

    # Dataset for generating ground truth poses to test against
    # Note: In a real inference scenario, we might not have a "GT pose" to compare against, rather we are finding it.
    # But sticking to the original run_alt.py logic which iterates and simulates projections.
    poses = PoseDataset(position_estimator.ct, steps=5, min_intersections=500, device="cpu")

    # 3. Main Loop
    num_iterations = 5 # Reduced for demonstration, original was 1000
    print(f"Starting loop for {num_iterations} iterations...")

    for j_iter in range(num_iterations):
        print(f'-- Iteration {j_iter} --')
        
        # Simulate input: Get random GT pose and project it
        i = torch.randint(len(poses), (1,)).item()
        rotation, translation = poses[i]
        gt_pose = torch.cat([rotation, translation], dim=-1).unsqueeze(0).to(device)
        
        # Generate synthetic 'observed' CRM from GT pose
        new_crm = position_estimator.project(gt_pose)
        
        # Apply noise/augs to make it realistic (mimic real CRM)
        new_crm = image_processing.apply_noise_brightness_contrast(
            new_crm, noise_std=0.015, brightness=0.25, contrast=0.55
        ) * (crm > 0.0) # Mask with original CRM shape

        # Run Optimization
        pose_optimizer.update_crm(new_crm)
        
        # Predict Initial
        with torch.no_grad():
            projection_pred, pose_pred = position_estimator(pose_optimizer.crm)

        # Optimize
        best_pose, projection_optimized, step, init_results, final_results = pose_optimizer()

        # Unpack results
        init_gain, init_position, init_features_mse_loss, init_features_cos_sim = init_results
        final_gain, final_position, final_features_mse_loss, final_features_cos_sim = final_results

        # Log metrics
        steps_log.append(step)
        init_gains.append(init_gain)
        final_gains.append(final_gain)
        init_feat_mse.append(init_features_mse_loss)
        final_feat_mse.append(final_features_mse_loss)
        init_feat_sim.append(init_features_cos_sim)
        final_feat_sim.append(final_features_cos_sim)
        
        init_pos_loss.append(loss_fn(init_position, gt_pose).item())
        final_pos_loss.append(loss_fn(final_position, gt_pose).item())

        # Visualization (Every 10 iters or last one)
        if j_iter % 10 == 0 or j_iter == num_iterations - 1:
            save_visualization(
                j_iter, output_base_dir, new_crm, projection_pred, projection_optimized,
                pose_optimizer, position_estimator, init_gain, final_gain, gt_pose
            )

    # Save summary plots
    save_summary_plots(output_base_dir, steps_log, init_gains, final_gains, 
                      init_pos_loss, final_pos_loss, init_feat_mse, final_feat_mse,
                      init_feat_sim, final_feat_sim)
    print("Done.")

def save_visualization(idx, output_dir, gt_crm, init_proj, opt_proj, optimizer, estimator, init_gain, final_gain, gt_pose):
    # Similar plotting logic to original
    cols = 3
    plt.figure(figsize=(12, 6))
    
    # 1. Images
    gt_img = gt_crm[0].squeeze().cpu().numpy()
    init_img = (init_proj * estimator.kernel)[0].squeeze().cpu().numpy()
    opt_img = (opt_proj * estimator.kernel)[0].squeeze().cpu().numpy()
    
    # 2. Sobel Maps
    with torch.no_grad():
        sobel_gt = (optimizer.sobel(gt_crm) * estimator.kernel)[0].squeeze().cpu().numpy()
        sobel_init = (optimizer.sobel(init_proj) * estimator.kernel)[0].squeeze().cpu().numpy()
        sobel_opt = (optimizer.sobel(opt_proj) * estimator.kernel)[0].squeeze().cpu().numpy()

    imgs = [gt_img, init_img, opt_img]
    titles = ["GT", f"Init: {init_gain:.3f}", f"Opt: {final_gain:.3f}"]
    
    for j, (im, t) in enumerate(zip(imgs, titles)):
        plt.subplot(2, cols, j + 1)
        plt.imshow(im, cmap="gray"); plt.title(t); plt.axis("off")

    sobels = [sobel_gt, sobel_init, sobel_opt]
    stitles = ["GT Sobel", "Init Sobel", "Opt Sobel"]
    for j, (im, t) in enumerate(zip(sobels, stitles)):
        plt.subplot(2, cols, cols + j + 1)
        plt.imshow(im, cmap="gray"); plt.title(t); plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/pred_{idx}.png", bbox_inches='tight')
    plt.close()

def save_summary_plots(output_dir, steps, i_gains, f_gains, i_ploss, f_ploss, i_fmse, f_fmse, i_fsim, f_fsim):
    # Helper to plot simple lines
    def plot_metric(data_list, title, ylabel, filename, legends=None):
        plt.figure(figsize=(10, 4))
        if isinstance(data_list, list) and isinstance(data_list[0], list): # List of lists
             for data, leg in zip(data_list, legends):
                 plt.plot(data, label=leg)
             plt.legend()
        else:
             plt.plot(data_list)
        plt.title(title); plt.xlabel("Iter"); plt.ylabel(ylabel); plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{filename}")
        plt.close()

    plot_metric(steps, "Steps to Converge", "Steps", "steps.png")
    plot_metric([i_gains, f_gains], "MNCC Gain", "Gain", "gains.png", ["Init", "Final"])
    plot_metric([i_ploss, f_ploss], "Position Loss", "MSE", "pos_loss.png", ["Init", "Final"])
    plot_metric([i_fmse, f_fmse], "Feature MSE", "MSE", "feat_mse.png", ["Init", "Final"])
    plot_metric([i_fsim, f_fsim], "Feature Cos Sim", "Sim", "feat_sim.png", ["Init", "Final"])

if __name__ == "__main__":
    run_inference()
