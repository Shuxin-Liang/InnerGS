import os
import math
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import gc
import sys
from collections import deque
import time
from datetime import datetime

# Set CUDA memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.7"

# Add CUDA module directory to Python path
sys.path.append('raster/diff-gaussian-rasterization')

def save_image(tensor, path):
    """Save tensor as image file"""
    if tensor.dim() == 3 and tensor.shape[0] == 3:  # [C, H, W]
        tensor = tensor.permute(1, 2, 0)  # 转换为 [H, W, C]
    
    # Ensure values are in [0, 1] range
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy array and save
    import numpy as np
    img_array = (tensor.detach().cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(img_array).save(path)

from gaussian3d_with_densification import Gaussian3D
try:
    # Import low-level C++ binding functions
    from diff_gaussian_rasterization import rasterize_gaussians, GaussianRasterizationSettings
except ImportError:
    print("*"*80)
    print("ERROR: CUDA rasterizer is not available. Please make sure you have compiled it successfully.")
    print("Go to 'raster/diff-gaussian-rasterization' and run 'python setup.py build_ext --inplace'")
    print("*"*80)
    exit(1)

# Axis direction mapping constants
AXIS_MAPPING = {
    'x': 0,      # X-axis slice: YZ plane
    'y': 1,      # Y-axis slice: XZ plane  
    'z': 2,      # Z-axis slice: XY plane
    'axial': 2,  # Axial = Z-axis
    'sagittal': 0,  # Sagittal = X-axis
    'coronal': 1   # Coronal = Y-axis
}

def get_axis_index(axis_name):
    """Get axis index"""
    return AXIS_MAPPING.get(axis_name.lower(), 2)  # Default to Z-axis

def get_camera_position(slice_position, axis_name, total_slices):
    """Calculate camera position based on axis direction"""
    # Normalize slice position to [-1, 1] range
    normalized_pos = (slice_position / (total_slices - 1)) * 2.0 - 1.0
    
    axis_idx = get_axis_index(axis_name)
    
    if axis_idx == 0:  # X-axis slice
        return torch.tensor([normalized_pos, 0.0, 0.0])
    elif axis_idx == 1:  # Y-axis slice
        return torch.tensor([0.0, normalized_pos, 0.0])
    else:  # Z-axis slice
        return torch.tensor([0.0, 0.0, normalized_pos])

def get_axis_image_size(data_root, axis):
    """Auto-detect image size for specified axis"""
    import os
    from PIL import Image
    
    axis_path = os.path.join(data_root, axis)
    if not os.path.exists(axis_path):
        return None
        
    # Get first image file
    image_files = [f for f in os.listdir(axis_path) if f.endswith('.png')]
    if not image_files:
        return None
        
    # Read image dimensions
    img_path = os.path.join(axis_path, image_files[0])
    img = Image.open(img_path)
    width, height = img.size
    return (height, width)  # Return (height, width) format

# SSIM loss implementation
def ssim_loss(img1, img2, window_size=11, sigma=1.5, size_average=True):
    """
    Calculate SSIM loss between two images
    Args:
        img1, img2: [C, H, W] tensors
        window_size: size of the Gaussian window
        sigma: standard deviation of the Gaussian window
        size_average: whether to average the loss
    Returns:
        SSIM loss (1 - SSIM)
    """
    def _gaussian_window(window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def _create_window(window_size, channel):
        _1D_window = _gaussian_window(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    # Add batch dimension if needed
    if len(img1.shape) == 3:
        img1 = img1.unsqueeze(0)
    if len(img2.shape) == 3:
        img2 = img2.unsqueeze(0)
    
    (_, channel, _, _) = img1.size()
    window = _create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    ssim_value = _ssim(img1, img2, window, window_size, channel, size_average)
    return 1 - ssim_value  # Return SSIM loss (1 - SSIM)

# Helper functions from original script
def get_gaussian_state(gaussians):
    """Extract state dict from Gaussian model without saving file"""
    return {
        'mu': gaussians.mu.data.clone(),
        'log_scale': gaussians.log_scale.data.clone(), 
        'color': gaussians.color.data.clone(),
        'opacity': gaussians.opacity.data.clone(),
        'quaternion': gaussians.quaternion.data.clone(),
        'xyz_gradient_accum': gaussians.xyz_gradient_accum.clone(),
        'denom': gaussians.denom.clone(),
        'max_radii2D': gaussians.max_radii2D.clone(),
    }

def save_gaussian_state(gaussians, path):
    """Save Gaussian parameters to file"""
    state = get_gaussian_state(gaussians)
    torch.save(state, path)
    return state


def save_gaussian_state_minimal(gaussians, path):
    """Minimal save function - move to CPU to reduce GPU memory usage"""
    state = {
        'mu': gaussians.mu.data.cpu(),
        'log_scale': gaussians.log_scale.data.cpu(),
        'color': gaussians.color.data.cpu(),
        'opacity': gaussians.opacity.data.cpu(),
        'quaternion': gaussians.quaternion.data.cpu(),
        'xyz_gradient_accum': gaussians.xyz_gradient_accum.cpu(),
        'denom': gaussians.denom.cpu(),
        'max_radii2D': gaussians.max_radii2D.cpu(),
    }
    torch.save(state, path)
    # Immediate cleanup
    del state
    gc.collect()
    return None


def load_gaussian_state(state, device):
    """Create new Gaussian model from state"""
    num_points = state['mu'].shape[0]
    new_gaussians = Gaussian3D(num_points=num_points).to(device)
    new_gaussians.mu.data = state['mu'].to(device)
    new_gaussians.log_scale.data = state['log_scale'].to(device)
    new_gaussians.color.data = state['color'].to(device)
    new_gaussians.opacity.data = state['opacity'].to(device)
    new_gaussians.quaternion.data = state['quaternion'].to(device)
    new_gaussians.xyz_gradient_accum = state['xyz_gradient_accum'].to(device)
    new_gaussians.denom = state['denom'].to(device)
    new_gaussians.max_radii2D = state['max_radii2D'].to(device)
    return new_gaussians


def create_optimizer(gaussians):
    """Create new optimizer - aligned with PyTorch version"""
    return torch.optim.Adam([
        {'params': gaussians.mu, 'lr': 6e-2, 'name': 'mu'},
        {'params': gaussians.log_scale, 'lr': 8e-2, 'name': 'log_scale'},
        {'params': gaussians.quaternion, 'lr': 4e-2, 'name': 'quaternion'},
        {'params': gaussians.color, 'lr': 4e-2, 'name': 'color'},
        {'params': gaussians.opacity, 'lr': 6e-2, 'name': 'opacity'},
    ])


def train_loop_cuda(data_root='slices_sphere_rgb', 
                    axes=('axial',),
                    image_size=(207, 127),  # Modify to actual image size (height, width)
                    num_iters=301,
                    l1_weight=0.8,
                    ssim_weight=0.4,
                    brightness_weight=0.1,  # New: brightness consistency loss weight
                    skip_twentieth_slices=False,
                    resume_from=None):  # New: path to resume from checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"L1: {l1_weight}, SSIM: {ssim_weight}")
    
    # Auto-detect image size for each axis
    axis_image_sizes = {}
    for axis in axes:
        detected_size = get_axis_image_size(data_root, axis)
        if detected_size:
            axis_image_sizes[axis] = detected_size
            print(f"{axis}: {detected_size}")
        else:
            # Use default size if detection fails
            axis_image_sizes[axis] = image_size
            print(f"{axis}: {image_size} (default)")
    
    # Initialize Gaussian model
    start_step = 0
    gaussians = None

    optimizer_state = None
    scheduler_state = None

    if resume_from and os.path.exists(resume_from):
        print(f"Resume from: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        
        # Smart loading logic: compatible with full checkpoints and pure state dicts
        if 'gaussian_state' in checkpoint:
            gaussians = load_gaussian_state(checkpoint['gaussian_state'], device)
            start_step = checkpoint.get('step', 0)
            optimizer_state = checkpoint.get('optimizer_state')
            scheduler_state = checkpoint.get('scheduler_state')
        else:
            gaussians = load_gaussian_state(checkpoint, device)
            start_step = 0  # For fine-tuning, steps can continue from 0 or previous steps
    
    if gaussians is None:
        gaussians = Gaussian3D(grid_res=40).to(device)
    
    optimizer = create_optimizer(gaussians)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    # Restore optimizer and scheduler states if they exist
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
    if scheduler_state:
        scheduler.load_state_dict(scheduler_state)

    # Train/test split settings
    if skip_twentieth_slices:
        print("Skip slices: 7,27,47,67... (test set)")

    # Training loop
    print(f"\nStarting training for {num_iters} steps...")
    
    # Initialize logging variables
    start_time = time.time()
    loss_history = []
    
    # Create log file
    log_filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    print(f"Log file: {log_filename}")
    
    with open(log_filename, 'w', encoding='utf-8') as log_file:
        log_file.write(f"Training Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Data: {data_root}, Axes: {axes}\n")
        log_file.write(f"Iterations: {num_iters}, L1 weight: {l1_weight}, SSIM weight: {ssim_weight}\n")
        log_file.write(f"Skip twentieth slices: {skip_twentieth_slices}\n")
        if resume_from:
            log_file.write(f"Resumed from: {resume_from}\n")
        log_file.write("="*80 + "\n")
        log_file.write("Step | Time(s) | Loss | L1 | SSIM\n")
        log_file.write("-"*50 + "\n")
    
    for step in range(start_step, num_iters):
        optimizer.zero_grad()
        losses = []
        l1_losses = []
        ssim_losses = []
        all_radii_in_step = []



        for axis in axes:
            axis_idx = get_axis_index(axis)
            dir_path = os.path.join(data_root, axis)
            if not os.path.exists(dir_path): continue
            
            # Create corresponding transform for current axis
            current_image_size = axis_image_sizes[axis]
            transform = transforms.Compose([transforms.Resize(current_image_size), transforms.ToTensor()])
            
            image_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.png')],
                                 key=lambda f: int(f.split('_')[-1].split('.')[0]))
            num_slices = len(image_files)
            if num_slices == 0: continue

            # Count slice information
            total_slices_available = len(image_files)
            training_slices = []
            skipped_slices = []
            
            for img_file in image_files:
                try:
                    idx = int(img_file.split('_')[-1].split('.')[0])
                    
                    # Logic to skip slices 7,27,47,67... (add 7 to original base)
                    if skip_twentieth_slices and idx % 20 == 7:
                        skipped_slices.append(idx)
                        continue  # Skip this slice
                    
                    training_slices.append(idx)
                    
                    # Calculate camera position based on axis direction
                    cam_pos = get_camera_position(idx, axis, num_slices).to(device)
                    dummy_matrix = torch.eye(4, device=device)

                    # Prepare parameters
                    means3D = gaussians.mu
                    opacities = gaussians.get_opacity
                    scales = gaussians.get_scaling
                    rotations = F.normalize(gaussians.quaternion, dim=1)
                    colors = torch.clamp(gaussians.color, 0, 1)

                    # Create GaussianRasterizationSettings
                    raster_settings = GaussianRasterizationSettings(
                        image_height=current_image_size[0],
                        image_width=current_image_size[1],
                        tanfovx=1.0,
                        tanfovy=1.0,
                        bg=torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32),
                        scale_modifier=1.0,
                        viewmatrix=dummy_matrix,
                        projmatrix=dummy_matrix,
                        sh_degree=0,
                        campos=cam_pos,
                        prefiltered=False,
                        debug=False,
                        antialiasing=False,
                        axis=axis_idx
                    )

                    # Call rasterize_gaussians
                    rendered_image, radii, _ = rasterize_gaussians(
                        means3D,
                        torch.empty(0, 3, device=device),  # means2D (not used)
                        torch.empty(0, device=device),     # sh
                        colors,                             # colors_precomp
                        opacities,
                        scales,
                        rotations,
                        torch.empty(0, device=device),     # cov3Ds_precomp
                        raster_settings
                    )
                    
                    # Fix: detach radii to avoid computation graph accumulation
                    all_radii_in_step.append(radii.detach())

                    img_path = os.path.join(dir_path, img_file)
                    # Ensure image is RGB format (3 channels)
                    pil_img = Image.open(img_path).convert('RGB')
                    gt_img = transform(pil_img).to(device).float()
                    
                    # Calculate L1 loss
                    l1_loss = F.l1_loss(rendered_image, gt_img)
                    
                    # Calculate SSIM loss
                    ssim_loss_value = ssim_loss(rendered_image, gt_img)
                    
                    # Calculate brightness consistency loss
                    #brightness_loss = torch.abs(rendered_image.mean() - gt_img.mean())
                    
                    # Combined loss
                    combined_loss = l1_weight * l1_loss + ssim_weight * ssim_loss_value
                    
                    if not torch.isnan(combined_loss):
                        losses.append(combined_loss)
                        # 修复：detach individual losses for monitoring
                        l1_losses.append(l1_loss.detach())
                        ssim_losses.append(ssim_loss_value.detach())
                        
                except Exception as e:
                    print(f"  Error processing {axis}/{img_file}: {e}")
                    continue
            


        if not losses:
            print("  No valid losses. Skipping step.")
            continue
            
        total_loss = torch.stack(losses).mean()
        # Directly use scalar to calculate average
        avg_l1_loss = sum(l1_losses) / len(l1_losses) if l1_losses else 0.0
        avg_ssim_loss = sum(ssim_losses) / len(ssim_losses) if ssim_losses else 0.0
        
        total_loss.backward()

        optimizer.step()
        scheduler.step(total_loss.detach())

        with torch.no_grad():
            # Clamp log_scale to prevent explosion
            gaussians.log_scale.data.clamp_(-10, 10) 
            
            # Normalize quaternions to prevent them from becoming zero vectors
            gaussians.quaternion.data = F.normalize(gaussians.quaternion.data, p=2, dim=1)

            gaussians.mu.data = torch.clamp(gaussians.mu.data, -1, 1)
            gaussians.color.data = torch.clamp(gaussians.color.data, 0, 1)

        # More frequent GPU memory cleanup (every 50 steps) to prevent fragmentation
        if step % 50 == 0 and step > 0:
            torch.cuda.empty_cache()
            gc.collect()  # Python garbage collection

        # Densification logic has been removed for fine-tuning.

        # Record loss history
        current_loss = total_loss.item()
        loss_history.append(current_loss)
        
        # Output single-line log every 20 steps
        if step % 20 == 0:
            current_time = time.time()
            elapsed_time = current_time - start_time
            avg_time_per_step = elapsed_time / (step - start_step + 1) if step > start_step else 0
            
            # Calculate average loss of recent 20 steps
            recent_losses = loss_history[-20:] if len(loss_history) >= 20 else loss_history
            avg_recent_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0
            
            # Estimate remaining time
            eta_minutes = 0
            if avg_time_per_step > 0:
                remaining_steps = num_iters - step - 1
                eta_seconds = remaining_steps * avg_time_per_step
                eta_minutes = eta_seconds / 60
            
            # Single-line output format
            log_line = f"{step:4d} | {elapsed_time:7.1f} | {current_loss:.6f} | {avg_l1_loss.item():.6f} | {avg_ssim_loss.item():.6f}"
            print(log_line)
            
            # Write to log file
            with open(log_filename, 'a', encoding='utf-8') as log_file:
                log_file.write(log_line + "\n")
                log_file.flush()  # Ensure real-time writing

        # Save checkpoint every 100 steps
        if (step + 1) % 100 == 0:
            checkpoint_path = f'checkpoint_step_{step+1}.pth'
            checkpoint = {
                'step': step + 1,
                'gaussian_state': get_gaussian_state(gaussians),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'loss': total_loss.item()
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        # Save final model and output images
        if step == num_iters - 1:
            total_time = time.time() - start_time
            print(f"\nTraining completed")
            print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
            print(f"Final loss: {current_loss:.6f}")
            
            # Write final statistics to log
            with open(log_filename, 'a', encoding='utf-8') as log_file:
                log_file.write("-"*50 + "\n")
                log_file.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"Total time: {total_time:.1f}s\n")
                log_file.write(f"Final loss: {current_loss:.6f}\n")
            
            print("\nSaving final model...")
            save_gaussian_state(gaussians, 'gaussians_cuda_final.pth')
            print("Final Gaussians saved to 'gaussians_cuda_final.pth'")
            
            # Render and save final images for all axes
            print("Rendering final images...")
            with torch.no_grad():
                for axis in axes:
                    output_axis_dir = os.path.join("output_images", axis)
                    os.makedirs(output_axis_dir, exist_ok=True)
                    
                    image_files_path = os.path.join(data_root, axis)
                    if not os.path.exists(image_files_path):
                        continue
                    
                    image_files = sorted([f for f in os.listdir(image_files_path) if f.endswith('.png')])
                    num_slices = len(image_files)
                    if num_slices == 0:
                        continue

                    current_image_size = axis_image_sizes.get(axis, image_size)
                    axis_idx = get_axis_index(axis)  # Get axis index

                    print(f"  Rendering final images for {axis} axis...")
                    for i, img_file in enumerate(image_files):
                        if i % 10 == 0:
                            print(f"    Rendering slice {i+1}/{len(image_files)}")

                        slice_idx = int(img_file.split('_')[-1].split('.')[0])
                        cam_pos = get_camera_position(slice_idx, axis, num_slices).to(device)

                        H, W = current_image_size
                        
                        # Key: use exactly the same camera settings as during training
                        dummy_matrix = torch.eye(4, device=device)
                        
                        # Create raster_settings exactly consistent with training
                        raster_settings = GaussianRasterizationSettings(
                            image_height=H,
                            image_width=W,
                            tanfovx=1.0,  # Same as training
                            tanfovy=1.0,  # Same as training
                            bg=torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32),
                            scale_modifier=1.0,
                            viewmatrix=dummy_matrix,  # Same as training
                            projmatrix=dummy_matrix,  # Same as training
                            sh_degree=0,
                            campos=cam_pos,  # Same calculation as training
                            prefiltered=False,
                            debug=False,
                            antialiasing=False,
                            axis=axis_idx  # Same as training, tell renderer which axis
                        )

                        # Prepare rendering parameters (same as training)
                        means3D = gaussians.mu
                        opacities = gaussians.get_opacity
                        scales = gaussians.get_scaling
                        rotations = F.normalize(gaussians.quaternion, dim=1)
                        colors = torch.clamp(gaussians.color, 0, 1)

                        # Render (same calling method as training)
                        rendered_image, radii, _ = rasterize_gaussians(
                            means3D,
                            torch.empty(0, 3, device=device),  # means2D (not used)
                            torch.empty(0, device=device),     # sh
                            colors,                             # colors_precomp
                            opacities,
                            scales,
                            rotations,
                            torch.empty(0, device=device),     # cov3Ds_precomp
                            raster_settings
                        )
                        
                        output_image_path = os.path.join(output_axis_dir, f"rendered_{axis}_{slice_idx:04d}.png")
                        save_image(rendered_image, output_image_path)

            print("Final rendered images saved.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='2D Gaussian Splatting multi-axis training script (with SSIM loss)')
    parser.add_argument('--skip_twentieth_slices', action='store_true', 
                       help='Skip slices (7,27,47,67...) as test set')
    parser.add_argument('--data_root', type=str, default='slices_sphere_rgb',
                       help='Training data root directory (default: slices_sphere_rgb)')
    parser.add_argument('--axes', type=str, nargs='+', default=['axial'],
                       help='Axes to train (supports: axial, sagittal, coronal, x, y, z)')
    parser.add_argument('--image_size', type=int, nargs=2, default=[141, 207],
                       help='Image size height width (default: 141 207)')
    parser.add_argument('--num_iters', type=int, default=2001,
                       help='Training iterations (default: 2001)')
    parser.add_argument('--l1_weight', type=float, default=0.8,
                       help='L1 loss weight (default: 0.8)')
    parser.add_argument('--ssim_weight', type=float, default=0.4,
                       help='SSIM loss weight (default: 0.4)')
    parser.add_argument('--brightness_weight', type=float, default=0.1,
                       help='Brightness consistency loss weight (default: 0.1)')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Resume training from specified checkpoint file')
    
    args = parser.parse_args()
    
    print("=== 2D Gaussian Splatting Multi-axis Training (with SSIM) ===")
    print(f"Training parameters:")
    print(f"  Data directory: {args.data_root}")
    print(f"  Axes: {args.axes}")
    print(f"  Image size: {args.image_size}")
    print(f"  Training iterations: {args.num_iters}")
    print(f"  L1 loss weight: {args.l1_weight}")
    print(f"  SSIM loss weight: {args.ssim_weight}")
    print(f"  Brightness consistency loss weight: {args.brightness_weight}")
    print(f"  Skip slices (7,27,47,67...): {'Yes' if args.skip_twentieth_slices else 'No'}")
    print(f"  Resume from checkpoint: {'Yes' if args.resume_from else 'No'}")
    if args.resume_from:
        print(f"  Resume checkpoint: {args.resume_from}")
    print()
    
    # Validate axis parameters
    valid_axes = []
    for axis in args.axes:
        if axis.lower() in AXIS_MAPPING:
            valid_axes.append(axis.lower())
        else:
            print(f"⚠️  Warning: Invalid axis name '{axis}', ignored")
    
    if not valid_axes:
        print("❌ Error: No valid axis parameters")
        print(f"Supported axes: {list(AXIS_MAPPING.keys())}")
        exit(1)
    
    train_loop_cuda(
        data_root=args.data_root,
        axes=tuple(valid_axes),
        image_size=tuple(args.image_size),
        num_iters=args.num_iters,
        l1_weight=args.l1_weight,
        ssim_weight=args.ssim_weight,
        brightness_weight=args.brightness_weight,
        skip_twentieth_slices=args.skip_twentieth_slices,
        resume_from=args.resume_from
    ) 