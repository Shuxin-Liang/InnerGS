import os
import math
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import gc
import sys

# Add the directory containing CUDA module to Python path
sys.path.append('raster/diff-gaussian-rasterization')

from gaussian3d_with_densification import Gaussian3D
try:
    from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
except ImportError:
    print("*"*80)
    print("ERROR: CUDA rasterizer is not available. Please make sure you have compiled it successfully.")
    print("Go to 'raster/diff-gaussian-rasterization' and run 'python setup.py build_ext --inplace'")
    print("*"*80)
    exit(1)


def gaussian(window_size, sigma):
    """Create Gaussian kernel"""
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    """Create Gaussian window"""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, size_average=True, val_range=None):
    """Calculate SSIM loss"""
    # If input is 3D (C,H,W), add batch dimension
    if len(img1.shape) == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    # Value range
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window_size is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel).to(img1.device)
    
    # Create window
    window = create_window(window_size, channel).to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    return ret


def save_gaussian_state(gaussians, path):
    """Save Gaussian parameters to file"""
    state = {
        'mu': gaussians.mu.data.clone(),
        'log_scale': gaussians.log_scale.data.clone(), 
        'color': gaussians.color.data.clone(),
        'opacity': gaussians.opacity.data.clone(),
        'quaternion': gaussians.quaternion.data.clone(),
        'xyz_gradient_accum': gaussians.xyz_gradient_accum.clone(),
        'denom': gaussians.denom.clone(),
        'max_radii2D': gaussians.max_radii2D.clone(),
    }
    torch.save(state, path)
    return state


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
        {'params': gaussians.mu, 'lr': 3e-2, 'name': 'mu'},
        {'params': gaussians.log_scale, 'lr': 4e-2, 'name': 'log_scale'},
        {'params': gaussians.quaternion, 'lr': 2e-2, 'name': 'quaternion'},
        {'params': gaussians.color, 'lr': 2e-2, 'name': 'color'},
        {'params': gaussians.opacity, 'lr': 3e-2, 'name': 'opacity'},
    ])


def densify_and_prune(gaussians, device, densify_grad_threshold, min_opacity, max_screen_size, scene_extent, densify_scale_factor=1.6):
    """
    Perform complete densification and pruning operations.
    Returns a new Gaussian model state dictionary and a flag indicating whether changes occurred.
    """
    with torch.no_grad():
        num_points_before = gaussians.mu.shape[0]
        
        # Collect parameters into dictionary for easier processing
        params = {
            'mu': gaussians.mu,
            'log_scale': gaussians.log_scale,
            'quaternion': gaussians.quaternion,
            'color': gaussians.color,
            'opacity': gaussians.opacity
        }

        # Calculate gradient norm
        grads = gaussians.xyz_gradient_accum / gaussians.denom.clamp(min=1)
        grads[grads.isnan()] = 0.0
        grad_norm = torch.norm(grads, dim=-1)

        # Identify points that need densification
        high_grad_mask = grad_norm >= densify_grad_threshold

        # --- Clone logic ---
        # Identify small points with high gradients
        scales = gaussians.get_scaling
        small_point_mask = scales.max(dim=1).values <= 0.01 * scene_extent  # Heuristic threshold
        clone_mask = torch.logical_and(small_point_mask, high_grad_mask)
        
        if clone_mask.any():
            new_params_clone = {k: v.data[clone_mask] for k, v in params.items()}
            new_params_clone['log_scale'] = new_params_clone['log_scale'].clamp_max(math.log(0.01 * scene_extent))
        else:
            new_params_clone = {}

        # --- Split logic ---
        # Identify large points with high gradients
        large_point_mask = scales.max(dim=1).values > 0.01 * scene_extent
        split_mask = torch.logical_and(large_point_mask, high_grad_mask)

        if split_mask.any():
            num_splits = split_mask.sum().item()
            split_scales = scales[split_mask]
            split_quats = gaussians.quaternion.data[split_mask]
            
            # Create two smaller Gaussians
            new_scales = split_scales / densify_scale_factor
            
            # Calculate maximum variance direction
            R = torch.zeros((num_splits, 3, 3), device=device)
            q = F.normalize(split_quats, p=2, dim=1)
            R[:, 0, 0] = 1. - 2. * (q[:, 2]**2 + q[:, 3]**2)
            R[:, 0, 1] = 2. * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3])
            R[:, 0, 2] = 2. * (q[:, 1] * q[:, 3] + q[:, 0] * q[:, 2])
            R[:, 1, 0] = 2. * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
            R[:, 1, 1] = 1. - 2. * (q[:, 1]**2 + q[:, 3]**2)
            R[:, 1, 2] = 2. * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1])
            R[:, 2, 0] = 2. * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
            R[:, 2, 1] = 2. * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1])
            R[:, 2, 2] = 1. - 2. * (q[:, 1]**2 + q[:, 2]**2)
            
            # Apply scaling
            scaled_R = R * new_scales.unsqueeze(-1)
            
            # Move along maximum variance direction
            max_var_dir = scaled_R[torch.arange(num_splits), :, torch.argmax(new_scales, dim=1)]
            
            split_mu = gaussians.mu.data[split_mask]
            new_mu1 = split_mu + max_var_dir
            new_mu2 = split_mu - max_var_dir

            # Create parameters for two split Gaussians
            new_params_split = {k: torch.cat([v.data[split_mask], v.data[split_mask]], dim=0) for k, v in params.items()}
            new_params_split['mu'] = torch.cat([new_mu1, new_mu2], dim=0)
            new_params_split['log_scale'] = torch.log(torch.cat([new_scales, new_scales], dim=0))
        else:
            new_params_split = {}
            
        # --- Pruning logic ---
        # Prune transparent or oversized points
        prune_mask = (gaussians.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = gaussians.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_vs)

        # Keep points that are not split or pruned
        keep_mask = torch.logical_and(~split_mask, ~prune_mask)
        
        current_params = {k: v.data[keep_mask] for k, v in params.items()}
        current_extra = {
            'xyz_gradient_accum': gaussians.xyz_gradient_accum[keep_mask],
            'denom': gaussians.denom[keep_mask],
            'max_radii2D': gaussians.max_radii2D[keep_mask],
        }

        # Merge all
        all_params = {}
        for k in current_params.keys():
            parts = [current_params[k]]
            if k in new_params_clone: parts.append(new_params_clone[k])
            if k in new_params_split: parts.append(new_params_split[k])
            all_params[k] = torch.cat(parts, dim=0)

        num_points_after = all_params['mu'].shape[0]
        
        # Correctly create dictionary for extra parameters
        all_extra = {}
        for k in current_extra.keys():
            clone_size = len(new_params_clone['mu']) if 'mu' in new_params_clone and new_params_clone['mu'].numel() > 0 else 0
            split_size = len(new_params_split['mu']) if 'mu' in new_params_split and new_params_split['mu'].numel() > 0 else 0
            
            if k == 'xyz_gradient_accum':
                clone_extra = torch.zeros((clone_size, 3), device=device)
                split_extra = torch.zeros((split_size, 3), device=device)
            elif k == 'denom':
                clone_extra = torch.zeros((clone_size, 1), device=device)
                split_extra = torch.zeros((split_size, 1), device=device)
            else:  # max_radii2D
                clone_extra = torch.zeros(clone_size, device=device)
                split_extra = torch.zeros(split_size, device=device)
            
            all_extra[k] = torch.cat([current_extra[k], clone_extra, split_extra], dim=0)

        # Combine into final state
        final_state = {**all_params, **all_extra}
        
        changed = num_points_after != num_points_before
        if changed:
            print(f"  Densified/Pruned: {num_points_before} -> {num_points_after} points.")

        return final_state, changed


def train_loop_cuda(data_root='slices_sphere_rgb', 
                    axes=('axial',),
                    image_size=(1024, 1024),  # Adjust to more reasonable resolution
                    num_iters=501,  # Increase iterations for densification
                    lambda_l1=0.8,
                    lambda_ssim=0.2): # Slightly reduce SSIM weight, emphasize L1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loss weights: L1={lambda_l1}, SSIM={lambda_ssim}")

    # Densification parameters - aligned with PyTorch version
    densify_from_iter = 50  # Adjust when to start densification
    densify_until_iter = 250 # Adjust when to end densification
    densification_interval = 50 # Adjust densification frequency
    densify_grad_threshold = 0.0002
    opacity_reset_interval = 50 # Adjust opacity reset frequency
    min_opacity = 0.005 # Lower minimum opacity threshold
    max_screen_size = 20 # Moderately relax screen size limit
    scene_extent = 1.0
    
    gaussians = Gaussian3D(grid_res=20).to(device)
    optimizer = create_optimizer(gaussians)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    # Initialize gradient accumulators
    gaussians.xyz_gradient_accum = torch.zeros_like(gaussians.mu.data)
    gaussians.denom = torch.zeros(gaussians.mu.shape[0], 1, device=device)
    gaussians.max_radii2D = torch.zeros(gaussians.mu.shape[0], device=device)
    
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])

    all_radii_in_step = [] # Initialize outside the loop

    for step in range(num_iters):
        # --- Densification logic ---
        if step >= densify_from_iter and step < densify_until_iter:
            if step % densification_interval == 0:
                with torch.no_grad():
                    # Ensure max_radii2D has correct shape
                    if all_radii_in_step:
                        # Stack all radii into [num_renders, num_points] shape
                        radii_tensor = torch.stack(all_radii_in_step, dim=0)
                        # Take maximum radius for each point across all renders
                        gaussians.max_radii2D = radii_tensor.max(dim=0).values
                    else:
                        gaussians.max_radii2D = torch.zeros(gaussians.mu.shape[0], device=device)
                
                new_state, changed = densify_and_prune(
                    gaussians, device, densify_grad_threshold, min_opacity, 
                    max_screen_size, scene_extent
                )
                
                if changed:
                    gaussians = load_gaussian_state(new_state, device)
                    optimizer = create_optimizer(gaussians)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
                
                # Reset gradient accumulators after densification
                gaussians.xyz_gradient_accum.zero_()
                gaussians.denom.zero_()

        optimizer.zero_grad(set_to_none=True)

        # --- Opacity reset ---
        if step > 0 and step % opacity_reset_interval == 0:
            with torch.no_grad():
                gaussians.opacity.data.fill_(0.01)

        losses = []
        l1_losses = []
        ssim_losses = []
        all_radii_in_step = [] # Reset for current iteration

        print(f"\nStep {step} (Points: {gaussians.get_xyz.shape[0]}):")

        for axis in axes:
            dir_path = os.path.join(data_root, axis)
            if not os.path.exists(dir_path): continue
            
            image_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.png')],
                                 key=lambda f: int(f.split('_')[-1].split('.')[0]))
            num_slices = len(image_files)
            if num_slices == 0: continue

            for img_file in image_files:
                try:
                    idx = int(img_file.split('_')[-1].split('.')[0])
                    slice_pos_z = (idx / (num_slices - 1)) * 2.0 - 1.0
                    
                    cam_pos = torch.tensor([0.0, 0.0, slice_pos_z], device=device)
                    dummy_matrix = torch.eye(4, device=device)

                    raster_settings = GaussianRasterizationSettings(
                        image_height=image_size[0], image_width=image_size[1],
                        tanfovx=1.0, tanfovy=1.0,
                        bg=torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32),
                        scale_modifier=1.0, viewmatrix=dummy_matrix, projmatrix=dummy_matrix,
                        sh_degree=0, campos=cam_pos, prefiltered=False, debug=False, antialiasing=False
                    )
                    rasterizer = GaussianRasterizer(raster_settings)

                    means3D = gaussians.mu
                    opacities = gaussians.get_opacity
                    scales = gaussians.get_scaling
                    rotations = F.normalize(gaussians.quaternion, dim=1)
                    colors = torch.clamp(gaussians.color, 0, 1)
                    
                    rendered_image, radii, _ = rasterizer(
                        means3D=means3D, means2D=torch.zeros_like(means3D, requires_grad=False),
                        opacities=opacities, shs=None, colors_precomp=colors,
                        scales=scales, rotations=rotations, cov3D_precomp=None
                    )
                    all_radii_in_step.append(radii)

                    img_path = os.path.join(dir_path, img_file)
                    # Ensure image is RGB format (3 channels)
                    pil_img = Image.open(img_path).convert('RGB')
                    gt_img = transform(pil_img).to(device).float()
                    
                    # Calculate L1 loss
                    l1_loss = F.l1_loss(rendered_image, gt_img)
                    
                    # Calculate SSIM loss (1-SSIM because higher SSIM is better)
                    ssim_loss = 1 - ssim(rendered_image, gt_img)
                    
                    # Combine losses
                    loss = lambda_l1 * l1_loss + lambda_ssim * ssim_loss
                    
                    if not torch.isnan(loss):
                        losses.append(loss)
                        l1_losses.append(l1_loss)
                        ssim_losses.append(ssim_loss)
                except Exception as e:
                    print(f"  Error processing {axis}/{img_file}: {e}")
                    continue

        if not losses:
            print("  No valid losses. Skipping step.")
            continue
            
        total_loss = torch.stack(losses).mean()
        avg_l1_loss = torch.stack(l1_losses).mean() if l1_losses else torch.tensor(0.0)
        avg_ssim_loss = torch.stack(ssim_losses).mean() if ssim_losses else torch.tensor(0.0)
        
        total_loss.backward()

        # --- Gradient accumulation and clipping ---
        with torch.no_grad():
            if step < densify_until_iter:
                # Accumulate gradients for densification
                # Ensure gradients exist
                if gaussians.mu.grad is not None:
                    gaussians.xyz_gradient_accum += gaussians.mu.grad.data
                    gaussians.denom += 1
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(gaussians.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step(total_loss.detach())

        with torch.no_grad():
            # Clamp log_scale to prevent explosion
            gaussians.log_scale.data.clamp_(-10, 10) 
            
            # Normalize quaternions to prevent them from becoming zero vectors
            gaussians.quaternion.data = F.normalize(gaussians.quaternion.data, p=2, dim=1)

            gaussians.mu.data = torch.clamp(gaussians.mu.data, -1, 1)
            gaussians.color.data = torch.clamp(gaussians.color.data, 0, 1)

        print(f"  Total Loss: {total_loss.item():.6f} | L1: {avg_l1_loss.item():.6f} | SSIM: {avg_ssim_loss.item():.6f}")

        # Save final model and output images
        if step == num_iters - 1:
            print("\nTraining finished. Saving final images...")
            save_gaussian_state(gaussians, 'gaussians_cuda_final.pth')
            print("Final Gaussians saved to 'gaussians_cuda_final.pth'")
            
            # Output rendered images from last step
            with torch.no_grad():
                for axis in axes:
                    dir_path = os.path.join(data_root, axis)
                    if not os.path.exists(dir_path): continue
                    
                    image_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.png')],
                                         key=lambda f: int(f.split('_')[-1].split('.')[0]))
                    num_slices = len(image_files)
                    if num_slices == 0: continue

                    for img_idx, img_file in enumerate(image_files):
                        try:
                            idx = int(img_file.split('_')[-1].split('.')[0])
                            slice_pos_z = (idx / (num_slices - 1)) * 2.0 - 1.0
                            
                            cam_pos = torch.tensor([0.0, 0.0, slice_pos_z], device=device)
                            dummy_matrix = torch.eye(4, device=device)

                            raster_settings = GaussianRasterizationSettings(
                                image_height=image_size[0], image_width=image_size[1],
                                tanfovx=1.0, tanfovy=1.0,
                                bg=torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32),
                                scale_modifier=1.0, viewmatrix=dummy_matrix, projmatrix=dummy_matrix,
                                sh_degree=0, campos=cam_pos, prefiltered=False, debug=False, antialiasing=False
                            )
                            rasterizer = GaussianRasterizer(raster_settings)

                            means3D = gaussians.mu
                            opacities = gaussians.get_opacity
                            scales = gaussians.get_scaling
                            rotations = F.normalize(gaussians.quaternion, dim=1)
                            colors = torch.clamp(gaussians.color, 0, 1)
                            
                            rendered_image, _, _ = rasterizer(
                                means3D=means3D, means2D=torch.zeros_like(means3D, requires_grad=False),
                                opacities=opacities, shs=None, colors_precomp=colors,
                                scales=scales, rotations=rotations, cov3D_precomp=None
                            )
                            
                            # Convert tensor to PIL image and save
                            rendered_np = rendered_image.permute(1, 2, 0).cpu().numpy()
                            rendered_np = (rendered_np * 255).astype('uint8')
                            rendered_pil = Image.fromarray(rendered_np)
                            
                            output_filename = f"final_rendered_{axis}_{idx:02d}.png"
                            rendered_pil.save(output_filename)
                            print(f"  Saved rendered image: {output_filename}")
                            
                        except Exception as e:
                            print(f"  Error saving final image {axis}/{img_file}: {e}")
                            continue
            
            print("Final rendered images saved.")


if __name__ == "__main__":
    train_loop_cuda() 