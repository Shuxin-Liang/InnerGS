import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Gaussian3D(nn.Module):
    def __init__(self, num_points=None, grid_res=None):
        super().__init__()
        if grid_res is not None:
            num_points = grid_res ** 3
            coords = torch.linspace(-20, 20, grid_res)
            grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing='ij')
            points = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)
            

            points = points + (torch.rand_like(points) - 0.5) * 1.0 / grid_res
            
            self.mu = nn.Parameter(points)
            print(f"Created uniform grid with {num_points} points (grid_res={grid_res})")
        elif num_points is not None:
            self.mu = nn.Parameter(torch.rand(num_points, 3) * 40 - 20)
            print(f"Created random distribution with {num_points} points")
        else:
            raise ValueError("Either num_points or grid_res must be provided")


        self.log_scale = nn.Parameter(torch.ones(num_points, 3) * math.log(0.02))
        self.color = nn.Parameter(torch.ones(num_points, 3) * 0.8)
        self.opacity = nn.Parameter(torch.ones(num_points, 1) * 0.2)
        

        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(num_points, 1)
        noise = 0.05 * torch.randn_like(identity_quat)
        self.quaternion = nn.Parameter(F.normalize(identity_quat + noise, dim=1))
        

        self.register_buffer("xyz_gradient_accum", torch.zeros((num_points, 1)))
        self.register_buffer("denom", torch.zeros((num_points, 1)))
        self.register_buffer("max_radii2D", torch.zeros((num_points,)))
        

        self.percent_dense = 0.01
        
    @property
    def get_scaling(self):
        return torch.exp(self.log_scale)
        
    @property 
    def get_opacity(self):
        return torch.sigmoid(self.opacity)
        
    def get_covariance(self):

        q = F.normalize(self.quaternion, dim=1)
        

        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.zeros((q.shape[0], 3, 3), device=q.device)
        R[:, 0, 0] = 1 - 2 * (qy**2 + qz**2)
        R[:, 0, 1] = 2 * (qx*qy - qz*qw)
        R[:, 0, 2] = 2 * (qx*qz + qy*qw)
        R[:, 1, 0] = 2 * (qx*qy + qz*qw)
        R[:, 1, 1] = 1 - 2 * (qx**2 + qz**2)
        R[:, 1, 2] = 2 * (qy*qz - qx*qw)
        R[:, 2, 0] = 2 * (qx*qz - qy*qw)
        R[:, 2, 1] = 2 * (qy*qz + qx*qw)
        R[:, 2, 2] = 1 - 2 * (qx**2 + qy**2)
        
        scale = self.get_scaling
        S = torch.diag_embed(scale)
        cov = R @ S @ R.transpose(1, 2)
        return cov
        
    @property
    def get_xyz(self):
        return self.mu
        
    def reset_opacity(self):

        opacities_new = torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)

        with torch.no_grad():
            self.opacity.data = torch.logit(opacities_new.clamp(1e-7, 1-1e-7))
    
    def add_densification_stats(self, viewspace_point_tensor, update_filter):

        if viewspace_point_tensor.grad is not None:
            self.xyz_gradient_accum[update_filter] += torch.norm(
                viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
            )
            self.denom[update_filter] += 1
    
    def prune_points(self, mask):

        valid_points_mask = ~mask
        

        with torch.no_grad():
            self.mu = nn.Parameter(self.mu.data[valid_points_mask])
            self.log_scale = nn.Parameter(self.log_scale.data[valid_points_mask])
            self.color = nn.Parameter(self.color.data[valid_points_mask])
            self.opacity = nn.Parameter(self.opacity.data[valid_points_mask])
            self.quaternion = nn.Parameter(self.quaternion.data[valid_points_mask])
            

            self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
            self.denom = self.denom[valid_points_mask]
            self.max_radii2D = self.max_radii2D[valid_points_mask]
    
    def densify_and_clone(self, grads, grad_threshold, scene_extent):


        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        )
        
        if not selected_pts_mask.any():
            return
        

        new_mu = self.mu[selected_pts_mask].clone()
        new_log_scale = self.log_scale[selected_pts_mask].clone()
        new_color = self.color[selected_pts_mask].clone()
        new_opacity = self.opacity[selected_pts_mask].clone()
        new_quaternion = self.quaternion[selected_pts_mask].clone()
        

        self._add_new_points(new_mu, new_log_scale, new_color, new_opacity, new_quaternion)
    
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):

        n_init_points = self.get_xyz.shape[0]
        

        padded_grad = torch.zeros((n_init_points), device=grads.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        

        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )
        
        if not selected_pts_mask.any():
            return
        

        selected_scaling = self.get_scaling[selected_pts_mask]
        selected_xyz = self.get_xyz[selected_pts_mask]
        selected_quaternion = self.quaternion[selected_pts_mask]
        

        stds = selected_scaling.repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=stds.device)
        samples = torch.normal(mean=means, std=stds)
        

        q = F.normalize(selected_quaternion, dim=1)
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.zeros((q.shape[0], 3, 3), device=q.device)
        R[:, 0, 0] = 1 - 2 * (qy**2 + qz**2)
        R[:, 0, 1] = 2 * (qx*qy - qz*qw)
        R[:, 0, 2] = 2 * (qx*qz + qy*qw)
        R[:, 1, 0] = 2 * (qx*qy + qz*qw)
        R[:, 1, 1] = 1 - 2 * (qx**2 + qz**2)
        R[:, 1, 2] = 2 * (qy*qz - qx*qw)
        R[:, 2, 0] = 2 * (qx*qz - qy*qw)
        R[:, 2, 1] = 2 * (qy*qz + qx*qw)
        R[:, 2, 2] = 1 - 2 * (qx**2 + qy**2)
        
        rots = R.repeat(N, 1, 1)
        

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + selected_xyz.repeat(N, 1)
        

        new_log_scale = torch.log(selected_scaling.repeat(N, 1) / (0.8 * N))
        new_quaternion = selected_quaternion.repeat(N, 1)
        new_color = self.color[selected_pts_mask].repeat(N, 1)
        new_opacity = self.opacity[selected_pts_mask].repeat(N, 1)
        

        self._add_new_points(new_xyz, new_log_scale, new_color, new_opacity, new_quaternion)
        

        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(N * selected_pts_mask.sum(), device=selected_pts_mask.device, dtype=bool)
        ))
        self.prune_points(prune_filter)
    
    def _add_new_points(self, new_mu, new_log_scale, new_color, new_opacity, new_quaternion):

        num_new = new_mu.shape[0]
        
        with torch.no_grad():

            self.mu = nn.Parameter(torch.cat([self.mu.data, new_mu], dim=0))
            self.log_scale = nn.Parameter(torch.cat([self.log_scale.data, new_log_scale], dim=0))
            self.color = nn.Parameter(torch.cat([self.color.data, new_color], dim=0))
            self.opacity = nn.Parameter(torch.cat([self.opacity.data, new_opacity], dim=0))
            self.quaternion = nn.Parameter(torch.cat([self.quaternion.data, new_quaternion], dim=0))
            

            new_xyz_grad = torch.zeros((num_new, 1), device=self.xyz_gradient_accum.device)
            new_denom = torch.zeros((num_new, 1), device=self.denom.device)
            new_max_radii = torch.zeros((num_new,), device=self.max_radii2D.device)
            
            self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, new_xyz_grad], dim=0)
            self.denom = torch.cat([self.denom, new_denom], dim=0)
            self.max_radii2D = torch.cat([self.max_radii2D, new_max_radii], dim=0)
    
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):


        grads = self.xyz_gradient_accum / (self.denom + 1e-7)
        grads[grads.isnan()] = 0.0
        

        if radii is not None:

            num_gaussians = self.max_radii2D.shape[0]
            if radii.shape[0] > num_gaussians:

                radii = radii[:num_gaussians]
            elif radii.shape[0] < num_gaussians:

                padded_radii = torch.zeros_like(self.max_radii2D)
                padded_radii[:radii.shape[0]] = radii
                radii = padded_radii
            
            valid_radii = radii > 0
            if valid_radii.any():
                self.max_radii2D[valid_radii] = torch.max(
                    self.max_radii2D[valid_radii], 
                    radii[valid_radii]
                )
        

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        
        if max_screen_size is not None:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        
        self.prune_points(prune_mask)
        

        self.xyz_gradient_accum.zero_()
        self.denom.zero_()
        self.max_radii2D.zero_()
        
        print(f"Densification: now {self.mu.shape[0]} points")
    
    def get_parameter_groups(self, lr_scales=None):

        if lr_scales is None:
            lr_scales = {
                'mu': 6e-2,
                'log_scale': 8e-2,
                'quaternion': 4e-2,
                'color': 4e-2,
                'opacity': 6e-2
            }
        
        return [
            {'params': [self.mu], 'lr': lr_scales['mu'], 'name': 'mu'},
            {'params': [self.log_scale], 'lr': lr_scales['log_scale'], 'name': 'log_scale'},
            {'params': [self.quaternion], 'lr': lr_scales['quaternion'], 'name': 'quaternion'},
            {'params': [self.color], 'lr': lr_scales['color'], 'name': 'color'},
            {'params': [self.opacity], 'lr': lr_scales['opacity'], 'name': 'opacity'},
        ] 