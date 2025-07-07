import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Gaussian3D(nn.Module):
    def __init__(self, num_points=None, grid_res=None):
        super().__init__()
        if grid_res is not None:
            num_points = grid_res ** 3
            coords = torch.linspace(-1, 1, grid_res)
            grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing='ij')
            points = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)
            
            # Add small random perturbation
            points = points + (torch.rand_like(points) - 0.5) * 0.05 / grid_res
            
            self.mu = nn.Parameter(points)
            print(f"Created uniform grid with {num_points} points (grid_res={grid_res})")
        elif num_points is not None:
            self.mu = nn.Parameter(torch.rand(num_points, 3) * 2 - 1)
            print(f"Created random distribution with {num_points} points")
        else:
            raise ValueError("Either num_points or grid_res must be provided")

        # Gaussian parameters
        self.log_scale = nn.Parameter(torch.ones(num_points, 3) * math.log(0.02))
        self.color = nn.Parameter(torch.ones(num_points, 3) * 0.8)
        self.opacity = nn.Parameter(torch.ones(num_points, 1) * 0.2)
        
        # Quaternion rotation
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(num_points, 1)
        noise = 0.05 * torch.randn_like(identity_quat)
        self.quaternion = nn.Parameter(F.normalize(identity_quat + noise, dim=1))
        
        # Densification-related statistics
        self.register_buffer("xyz_gradient_accum", torch.zeros((num_points, 3)))
        self.register_buffer("denom", torch.zeros((num_points, 1)))
        self.register_buffer("max_radii2D", torch.zeros((num_points,)))
        
        # Densification parameters
        self.percent_dense = 0.01
        
    @property
    def get_scaling(self):
        return torch.exp(self.log_scale)
        
    @property 
    def get_opacity(self):
        return torch.sigmoid(self.opacity)
        
    def get_covariance(self):
        """Covariance = R S R^T"""
        q = F.normalize(self.quaternion, dim=1)
        
        # Convert quaternion to rotation matrix
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
        """Reset opacity"""
        opacities_new = torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self.opacity = optimizable_tensors["opacity"] 