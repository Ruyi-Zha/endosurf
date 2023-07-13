import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from tqdm import tqdm

import sys
sys.path.append("./")
from src.renderer.encoder import get_encoder
from src.renderer.utils import run_fn_split, extract_geometry, sample_pdf, get_sphere_intersection, build_mlp_idr, build_mlp_nerf


class EndoSurfRenderer(nn.Module):
    """EndoSurf renderer.
    """
    
    def __init__(self, render_cfg, net_cfg, device="cuda"):
        super().__init__()
        
        self.render_cfg = render_cfg
        self.device = device
        self.dtype = torch.get_default_dtype()
        
        # Network
        self.net_cfg = net_cfg
        model = EndoSurfNet(net_cfg)
        self.model = model.to(self.device)
        
        # Prams
        self.anneal_end = self.render_cfg["anneal_end"]
        self.n_samples = self.render_cfg["n_samples"]
        self.perturb = self.render_cfg["perturb"]
        self.n_importance = self.render_cfg["n_importance"]
        self.important_begin_iter = self.render_cfg["important_begin_iter"]
        self.up_sample_steps = self.render_cfg["up_sample_steps"]
        self.net_chunk = self.render_cfg["net_chunk"]
    
    def get_train_params(self):
        """Get training parameters.
        """
        return self.model.get_train_params()
    
    def load_checkpoint(self, ckpt):
        """Load checkpoint.
        """
        self.model.load_checkpoints(ckpt)
        
    def save_checkpoint(self):
        """Output checkpoints for saving.
        """
        return self.model.save_checkpoint()
        
    def forward(self, rays, **kwargs):
        """Forward function.
        """
        ret = self.render_rays(rays, **kwargs)
        return ret
        
    def render_rays(self, rays, iter_step=0, perturb_overwrite=None, eval=False, **kwargs):
        """Render rays.
        """
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[..., :3], rays[..., 3:6]
        time = rays[..., 8]
        rays_d_z = rays_d / (rays_d[..., 2:] + 1e-6)
        
        near, far, _ = get_sphere_intersection(rays_o, rays_d)
        cos_anneal_ratio = self.get_cos_anneal_ratio(iter_step)
        
        sample_dist = 2.0 / self.n_samples
        
        n_samples = self.n_samples
        perturb = self.perturb
        if perturb_overwrite is not None:
            perturb = perturb_overwrite
        
        t_vals = torch.linspace(0.0, 1.0, self.n_samples, device=self.device)
        z_vals = near + (far - near) * t_vals[None, :]
        if perturb:
            t_rand = (torch.rand([n_rays, 1], device=self.device) - 0.5)
            z_vals = z_vals + t_rand * sample_dist
        
        # Upsample
        if iter_step >= self.important_begin_iter and self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d_z[:, None, :] * z_vals[..., :, None]
                t = time[..., None, None].expand(n_rays, n_samples, 1)
                pts = pts.reshape(-1, 3)
                t = t.reshape(-1, 1)
                # Deform
                sdf = self.model.get_sdf_from_observed_space(pts, t)
                sdf = sdf.reshape(n_rays, n_samples)
                
                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  time,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))
                    
                n_samples = self.n_samples + self.n_importance
            
        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    time, 
                                    z_vals,
                                    sample_dist,
                                    cos_anneal_ratio=cos_anneal_ratio,
                                    eval=eval)
        
        
        
        return {
            "color_map": ret_fine["color_map"],
            "depth_map": ret_fine["depth_map"],
            "gradients_o": ret_fine["gradients_o"],
            "gradient_o_error": ret_fine["gradient_o_error"],
            "weights": ret_fine["weights"],
            "weight_max": torch.max(ret_fine["weights"], dim=-1, keepdim=True)[0],
            "cdf": ret_fine["cdf"],
            "s_val": ret_fine["s_val"].reshape(n_rays, n_samples).mean(dim=-1, keepdim=True),
        }
        
    def render_core(self,
                    rays_o,
                    rays_d,
                    time,
                    z_vals,
                    sample_dist,
                    cos_anneal_ratio=0.0,
                    eval=False):
        """Core rener function.
        """
        n_rays, n_samples = z_vals.shape
        rays_d_z = rays_d / (rays_d[..., 2:] + 1e-6)
        
        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.tensor([sample_dist], device=self.device).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5
        
        # Section midpoints
        pts = rays_o[:, None, :] + rays_d_z[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs_o = rays_d[:, None, :].expand(pts.shape) # view in observation space
        t = time[:, None, None].expand(n_rays, n_samples, 1)
        
        pts = pts.reshape(-1, 3)
        dirs_o = dirs_o.reshape(-1, 3)
        t = t.reshape(-1, 1)
    
        # Deform
        raw = run_fn_split(self.model.forward, torch.cat([pts, dirs_o, t], -1), self.net_chunk)
        sdf = raw[...,:1]
        sampled_color = raw[...,1:4].reshape(n_rays, n_samples, 3)
        gradients_o = run_fn_split(lambda x: self.model.get_sdf_grad_from_observed_space(x[...,:3], x[...,3:4]), 
                                   torch.cat([pts, t], -1), self.net_chunk)
        
        inv_s = self.model.deviation_network(torch.zeros([1, 3], device=self.device))[:, :1].clip(1e-6, 1e6)
        inv_s = inv_s.expand(n_rays * n_samples, 1)
        
        true_cos = (dirs_o * gradients_o).sum(-1, keepdim=True) # observation

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
        
        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-6) / (c + 1e-6)).reshape(n_rays, n_samples).clip(0.0, 1.0)
        
        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(n_rays, n_samples)
        relax_inside_sphere = (pts_norm < 1.2).to(self.dtype).detach()

        weights = alpha * torch.cumprod(torch.cat([torch.ones([n_rays, 1], device=self.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)
        
        # depth map and color
        depth_map = torch.sum(weights * mid_z_vals, -1, keepdim=True)
        color = (sampled_color * weights[:, :, None]).sum(dim=1)

        # Eikonal loss, observation + canonical
        gradient_o_error = (torch.linalg.norm(gradients_o.reshape(n_rays, n_samples, 3), ord=2, dim=-1) - 1.0) ** 2
        relax_inside_sphere_sum = relax_inside_sphere.sum() + 1e-6
        gradient_o_error = (relax_inside_sphere * gradient_o_error).sum() / relax_inside_sphere_sum    

        return {
            "color_map": color,
            "depth_map": depth_map,
            "gradients_o": gradients_o.reshape(n_rays, n_samples, 3),
            "gradient_o_error": gradient_o_error,
            "cdf": c.reshape(n_rays, n_samples),
            "weights": weights,
            "s_val": 1.0 / inv_s,
        }
    
    def get_cos_anneal_ratio(self, iter_step):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, iter_step / self.anneal_end])
        
    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        n_rays, n_samples = z_vals.shape
        rays_d_z = rays_d / (rays_d[..., 2:] + 1e-6) 
        pts = rays_o[:, None, :] + rays_d_z[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(n_rays, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-6)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([n_rays, 1]).to(self.device), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-6) / (prev_cdf + 1e-6)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([n_rays, 1]).to(self.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples
    
    def cat_z_vals(self, rays_o, rays_d, time, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            rays_d_z = rays_d / (rays_d[..., 2:] + 1e-6)
            pts = rays_o[:, None, :] + rays_d_z[:, None, :] * new_z_vals[..., :, None]
            t = time[:, None, None].expand(pts.shape[0], pts.shape[1], 1)
            pts = pts.reshape(-1, 3)
            t = t.reshape(-1, 1)
            # Deform
            new_sdf = self.model.get_sdf_from_observed_space(pts, t).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf
    
    def errorondepth(self, rays, d_gt, mask, iter_step=0):
        """Compute error on depth
        """
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[..., :3], rays[..., 3:6]
        near, far, time = rays[..., 6], rays[..., 7], rays[..., 8]
        rays_d_z = rays_d / (rays_d[..., 2:] + 1e-6)
        
        pts = rays_o + rays_d_z * d_gt
        dirs_o = rays_d
        pts = pts.reshape(-1, 3)
        dirs_o = dirs_o.reshape(-1, 3)
        t = time.reshape(-1, 1)
        
        sdf = self.model.get_sdf_from_observed_space(pts, t)
        gradient_o = self.model.get_sdf_grad_from_observed_space(pts, t)
        
        true_cos = (rays_d * gradient_o).sum(-1, keepdim=True)
        relu_cos = F.relu(true_cos)
        pts = pts.detach()
        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True)
        # Denoise. not use: out of mask or sphere
        inside_masksphere = (pts_norm < 1.0).to(self.dtype) * mask # inside_sphere * mask
        sdf = inside_masksphere * sdf
        inside_masksphere_sum = inside_masksphere.sum() + 1e-6
        sdf_error = F.l1_loss(sdf, torch.zeros_like(sdf), reduction='sum') / inside_masksphere_sum
        angle_error = F.l1_loss(relu_cos, torch.zeros_like(relu_cos), reduction='sum') / inside_masksphere_sum

        return sdf_error, angle_error, inside_masksphere
    
    def surface_neighbour_error(self, rays, mask, iter_step=0, neighbour_rad=0.05):
        rays_o, rays_d = rays[..., :3], rays[..., 3:6]
        near, far, time = rays[..., 6], rays[..., 7], rays[..., 8]
        rays_d_z = rays_d / (rays_d[..., 2:] + 1e-6)
        with torch.no_grad():
            d_i = self.ray_marching(rays, max_points=self.net_chunk)
            
        # Get mask for predicted depth
        mask_valid = (d_i.abs() != np.inf) & (d_i != 0) & (mask == 1)
        mask_valid = mask_valid[..., 0]
        if mask_valid.any():
            p_surf = rays_o[mask_valid] + d_i[mask_valid] * rays_d_z[mask_valid]
            p_surf_neig = p_surf + (torch.rand_like(p_surf) - 0.5) * neighbour_rad
            n_rays = p_surf.shape[0]
            pp = torch.cat([p_surf, p_surf_neig], 0)
            time_pp = torch.cat([time[mask_valid], time[mask_valid]], 0).unsqueeze(-1)
            g = run_fn_split(lambda x: self.model.get_sdf_grad_from_observed_space(x[...,:3], x[...,3:4]), 
                                    torch.cat([pp, time_pp], -1), self.net_chunk)
            normal = g / (torch.linalg.norm(g, ord=2, dim=-1, keepdim=True) + 1e-10)
            diff_norm = torch.abs(normal[:n_rays] - normal[n_rays:])
            diff = diff_norm.mean()
            return diff
        else:
            return 0.

    def ray_marching(self, rays, tau=0.0, n_steps=[128, 129],
                     n_secant_steps=8, max_points=64000):
        """ Performs ray marching to detect surface points.
        The function returns the surface points as well as d_i of the formula
            ray(d_i) = ray0 + d_i * ray_direction
        which hit the surface points. In addition, masks are returned for
        illegal values.
        """
        n_steps = torch.randint(n_steps[0], n_steps[1], (1,)).item()
        
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[..., :3], rays[..., 3:6]
        time = rays[..., 8:]
        near, far, _ = get_sphere_intersection(rays_o, rays_d)
        
        t_vals = torch.linspace(0., 1., steps=n_steps, device=self.device)
        d_proposal = near * (1. - t_vals) + far * t_vals
        
        rays_d_z = rays_d / (rays_d[...,2:] + 1e-6)  # Normalize such that z is 1.
        p_proposal = rays_o[..., None, :].expand([n_rays, n_steps, 3]) + \
            d_proposal[..., :, None].expand([n_rays,n_steps, 1]) * \
                rays_d_z[..., None, :].expand([n_rays, n_steps, 3])
        
        t = time[..., None].expand(n_rays, n_steps, 1)
        
        # Evaluate all proposal points
        pts = p_proposal.reshape(-1, 3)
        t_pts = t.reshape(-1, 1)
        val = []
        with torch.no_grad():
            for pts_batch, t_batch in zip(pts.split(max_points), t_pts.split(max_points)):
                v = self.model.get_sdf_from_observed_space(pts_batch, t_batch)
                val.append(v)
            val = torch.cat(val, 0).view(n_rays, n_steps) - tau
        
        val = -val
        # Create mask for valid points where the first point is not occupied
        mask_0_not_occupied = val[:, 0] < 0
        
        # Calculate if sign change occurred and concat 1 (no sign change) in
        # last dimension
        sign_matrix = torch.cat([torch.sign(val[:, :-1] * val[:, 1:]),
                                 torch.ones(n_rays, 1, device=self.device)], -1)
        cost_matrix = sign_matrix * torch.arange(
            n_steps, 0, -1).float().to(self.device)
        
        # Get first sign change and mask for values where a.) a sign changed
        # occurred and b.) no a neg to pos sign change occurred (meaning from
        # inside surface to outside)
        values, indices = torch.min(cost_matrix, -1)
        mask_sign_change = values < 0
        mask_neg_to_pos = val[torch.arange(n_rays), indices] < 0
        
        # Define mask where a valid depth value is found
        mask = mask_sign_change & mask_neg_to_pos & mask_0_not_occupied 
        
        # Get depth values and function values for the interval
        # to which we want to apply the Secant method
        d_low = d_proposal[torch.arange(n_rays), indices][mask]
        f_low = val[torch.arange(n_rays), indices][mask]
        indices = torch.clamp(indices + 1, max=n_steps-1)
        d_high = d_proposal[torch.arange(n_rays), indices][mask]
        f_high = val[torch.arange(n_rays), indices][mask]
        
        rays_masked = rays[mask]
        
        # Apply surface depth refinement step (e.g. Secant method)
        d_pred = 0
        if len(rays_masked) != 0:
            d_pred = self.secant(f_low, f_high, d_low, d_high, n_secant_steps, rays_masked, tau, max_points)
        
        # for sanity
        d_pred_out = torch.ones(n_rays).to(self.device)
        d_pred_out[mask] = d_pred
        d_pred_out[mask == 0] = np.inf
        d_pred_out[mask_0_not_occupied == 0] = 0
        return d_pred_out.unsqueeze(-1)
    
    def secant(self, f_low, f_high, d_low, d_high, n_secant_steps, rays, tau, max_points):
        """Runs the secant method for interval [d_low, d_high].
        """
        rays_o, rays_d = rays[..., :3], rays[..., 3:6]
        time = rays[..., 8:]
        rays_d_z = rays_d / rays_d[...,2:]  # Normalize such that z is 1.
        d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
        for i in range(n_secant_steps):
            p_mid = rays_o + d_pred.unsqueeze(-1) * rays_d_z
            t = time
            with torch.no_grad():
                val = []
                for pts_batch, t_batch in zip(p_mid.split(max_points), t.split(max_points)):
                    v = self.model.get_sdf_from_observed_space(pts_batch, t_batch)
                    val.append(v)
                val = torch.cat(val, 0)
            f_mid = val[..., 0] - tau
            ind_low = f_mid < 0
            ind_low = ind_low
            if ind_low.sum() > 0:
                d_low[ind_low] = d_pred[ind_low]
                f_low[ind_low] = f_mid[ind_low]
            if (ind_low == 0).sum() > 0:
                d_high[ind_low == 0] = d_pred[ind_low == 0]
                f_high[ind_low == 0] = f_mid[ind_low == 0]
            
            d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
        return d_pred
    
    def renderondepth(self,
                      rays,
                      depth):
        """Surface rendering: render rgb based on a given depth.
        """
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[..., :3], rays[..., 3:6]
        time = rays[..., 8]
        rays_d_z = rays_d / (rays_d[..., 2:] + 1e-6)
        near, far, _ = get_sphere_intersection(rays_o, rays_d)
        
        valid_idx = torch.bitwise_and(depth[..., 0] > 0, depth[..., 0] != np.inf)
        d_out = depth.clone()
        d_out[depth == np.inf] = far[depth == np.inf]
        if valid_idx.sum() == 0:
            return torch.zeros((n_rays, 3), device=self.device), \
                torch.zeros((n_rays, 3), device=self.device), \
                d_out
                            
        depth_valid = depth[valid_idx]
        rays_o_valid = rays_o[valid_idx]
        rays_d_valid = rays_d[valid_idx]
        rays_d_z_valid = rays_d_z[valid_idx]
        t_valid = time[valid_idx]
        
        pts = rays_o_valid + rays_d_z_valid * depth_valid
        dirs_o = rays_d_valid
        t = t_valid.unsqueeze(-1)
        with torch.no_grad():
            raw = run_fn_split(self.model.forward, torch.cat([pts, dirs_o, t], -1), self.net_chunk)
            gradients_o = run_fn_split(lambda x: self.model.get_sdf_grad_from_observed_space(x[...,:3], x[...,3:4]), 
                                   torch.cat([pts, t], -1), self.net_chunk)
            color = raw[...,1:4]
        color_out = torch.zeros((n_rays, 3), device=self.device)
        gradients_out = torch.zeros((n_rays, 3), device=self.device)
        color_out[valid_idx] = color
        gradients_out[valid_idx] = gradients_o
        return color_out, gradients_out, d_out
    
    def extract_observation_geometry(self, t, bound_min, bound_max, resolution, threshold=0.0, net_chunk=80000, cpu=True):
        """Extract geometry by marching cubes.
        """
        query_func = lambda pts: self.model.get_sdf_from_observed_space(pts, t)
        split_query_func = lambda pts: run_fn_split(query_func, pts, net_chunk, cpu=cpu)
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=split_query_func,
                                device=self.device)
    
    def renderonpts(self, pts, dirs, ts, net_chunk=80000, cpu=True):
        """Render color given point, dir and t. (surface rendering)
        pts: 
        """
        sh = list(pts.shape[:-1])
        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)
        if len(ts.shape) == 1:
            ts = ts[None, :].expand(pts.shape[0], 1)
        raw = run_fn_split(lambda x: self.model(x), torch.cat([pts, dirs, ts], -1), net_chunk)
        color = raw[...,1:]  # [n_rays, n_samples, 3]
        gradient_o = run_fn_split(lambda x: self.model.get_sdf_grad_from_observed_space(x[...,:3], x[...,3:4]), 
                                   torch.cat([pts, ts], -1), net_chunk, cpu=cpu)
        if cpu:
            normal = gradient_o / (np.linalg.norm(gradient_o, ord=2, axis=-1, keepdims=True) + 1e-10)
        else:
            normal = gradient_o / (torch.linalg.norm(gradient_o, ord=2, dim=-1, keepdim=True) + 1e-10)
        color = color.reshape(*sh, 3)
        normal = normal.reshape(*sh, 3)
        return color, normal
        
    
class EndoSurfNet(nn.Module):
    def __init__(self, net_cfg) -> None:
        """EndoSurf network.
        """
        super().__init__()
        self.bound = net_cfg["bound"]
        self.use_deform = net_cfg["use_deform"]

        # Initialize network
        if self.use_deform:
            self.deform_network = DeformNetwork(bound=self.bound, **net_cfg["deform_network"])
        self.sdf_network = SDFNetwork(bound=self.bound, **net_cfg["sdf_network"])
        self.color_network = ColorNetwork(bound=self.bound, **net_cfg["color_network"])
        self.deviation_network = SingleVarianceNetwork(**net_cfg["deviation_network"])

    def get_train_params(self):
        """Get training parameters.
        """
        train_params = {}
        if self.use_deform:
            train_params["deform_network"] = list(self.deform_network.parameters())
        train_params["sdf_network"] = list(self.sdf_network.parameters())
        train_params["color_network"] = list(self.color_network.parameters())
        train_params["deviation_network"] = list(self.deviation_network.parameters())
        return train_params
    
    def load_checkpoints(self, ckpt):
        """Load checkpoints.
        """
        if self.use_deform:
            self.deform_network.load_state_dict(ckpt["deform_network"])
        self.sdf_network.load_state_dict(ckpt["sdf_network"])
        self.color_network.load_state_dict(ckpt["color_network"])
        self.deviation_network.load_state_dict(ckpt["deviation_network"])

    def save_checkpoint(self):
        """Output checkpoint.
        """
        ckpt = {}
        if self.use_deform:
            ckpt["deform_network"] = self.deform_network.state_dict()
        ckpt["sdf_network"] = self.sdf_network.state_dict()
        ckpt["color_network"] = self.color_network.state_dict()
        ckpt["deviation_network"] = self.deviation_network.state_dict()
        return ckpt

    def get_sdf_from_observed_space(self, x, t):
        """Get sdf from observed space.
        """
        if self.use_deform:
            deform = self.deform_network(x, t)
            x_c = x + deform
        else:
            x_c = x
        sdf = self.sdf_network.sdf(x_c)
        return sdf
    
    def get_sdf_grad_from_observed_space(self, x, t):
        """Get sdf gradients from observed space.
        """
        with torch.enable_grad():
            x.requires_grad_(True)
            if self.use_deform:
                deform = self.deform_network(x, t)
                x_c = x + deform
            else:
                x_c = x
            y = self.sdf_network.sdf(x_c)
            # Gradient on observation
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradient_o = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
        return gradient_o
    
    def get_sdf_grad_from_canonical_space(self, x):
        """Get sdf gradients from canonical space.
        x: input is canonibcal position.
        """
        with torch.enable_grad():
            x.requires_grad_(True)
            y = self.sdf_network.sdf(x)
            # Gradient on observation
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradient_o = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
        return gradient_o
    
    def get_deform_grad_from_observed_space(self, x, t):
        """Get deform gradient from the obervsed space.
        """
        with torch.enable_grad():
            x.requires_grad_(True)
            if self.use_deform:
                deform = self.deform_network(x, t)
                x_c = x + deform
            else:
                x_c = x
            # Jacobian on pts
            y_0 = x_c[:, 0]
            y_1 = x_c[:, 1]
            y_2 = x_c[:, 2]
            d_output = torch.ones_like(y_0, requires_grad=False, device=y_0.device)
            grad_0 = torch.autograd.grad(
                outputs=y_0,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0].unsqueeze(1)
            grad_1 = torch.autograd.grad(
                outputs=y_1,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0].unsqueeze(1)
            grad_2 = torch.autograd.grad(
                outputs=y_2,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0].unsqueeze(1)
            gradient_pts =  torch.cat([grad_0, grad_1, grad_2], 1) # (n_rays, dim_out, dim_in) 
        return gradient_pts
    
    def forward(self, inputs):
        """ Network forward function.
        inputs: [x, d, t]
        x: [N, 3], in [-bound, bound]
        d: [N, 3], nomalized in unit sphere.
        t: [N, 1], in [0, 1]
        """
        x, d, t = inputs[..., :3], inputs[..., 3:6], inputs[..., 6:]
        
        # Deform
        if self.use_deform:
            deform = self.deform_network(x, t)
            x_c = x + deform
        else:
            x_c = x
        
        # SDF
        h = self.sdf_network(x_c)
        sdf = h[..., :1]
        geo_feat = h[...,1:]
    
        gradients_c = self.get_sdf_grad_from_canonical_space(x_c)  # conanical normal
    
        pts_jacobian = self.get_deform_grad_from_observed_space(x, t)
        d_c = torch.bmm(pts_jacobian, d.unsqueeze(-1)).squeeze(-1)
        d_c = d_c / (torch.linalg.norm(d_c, ord=2, dim=-1, keepdim=True) + 1e-10) # canonical direction
        color = self.color_network(x_c, gradients_c, d_c, geo_feat)
        out  = torch.cat([sdf, color], -1)
    
        return out
    

class DeformNetwork(nn.Module):
    """Deform network.
    """
    def __init__(self,
                 enc_time_cfg=dict(enc_type="frequency",multires=6,input_dim=1),
                 enc_pos_cfg=dict(enc_type="frequency",multires=6,input_dim=3),
                 n_layers=9,
                 hidden_dim=256,
                 skips=[5],
                 out_dim=3,
                 bound=1.0,
                 **kwargs,
                 ):
        super().__init__()
        self.bound = bound
        self.enc_fn_pos, in_dim_pos = get_encoder(**enc_pos_cfg)
        self.enc_fn_time, in_dim_time = get_encoder(**enc_time_cfg)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.out_dim = out_dim
        self.net = build_mlp_idr(self.n_layers,
                                 self.hidden_dim,
                                 in_dim_pos + in_dim_time,
                                 out_dim,
                                 self.skips,
                                 bias=True,
                                 geometric_init=False,
                                 weight_norm=True,
                                 inside_outside=False)
        self.activation = nn.ReLU()
    
    def forward(self, x, t):
        """Run deform network."""
        if len(t.shape) == 1:
            t = t[None, :].expand(x.shape[0], 1)
        x_enc = self.enc_fn_pos(x, bound=self.bound)
        t_enc = self.enc_fn_time(t, bound=self.bound)
        xt_enc = torch.cat([x_enc, t_enc], -1)
        deform = xt_enc.clone()
        for l in range(self.n_layers):
            if l in self.skips:
                deform = torch.cat([deform, xt_enc], -1) / np.sqrt(2)
            deform = self.net[l](deform)
            if l != self.n_layers - 1:
                deform = self.activation(deform)
        return deform


class SDFNetwork(nn.Module):
    """SDF network.
    """
    def __init__(self,
                 enc_pos_cfg=dict(enc_type="frequency",multires=6,input_dim=3),
                 n_layers=9,
                 hidden_dim=256,
                 skips=[5],
                 out_dim=257,
                 bound=1.0,
                 geometric_init=True,
                 geometric_init_bias=0.8,
                 **kwargs,
                 ):
        super().__init__()
        self.bound = bound
        self.enc_fn_pos, in_dim_pos = get_encoder(**enc_pos_cfg)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.out_dim = out_dim
        self.net = build_mlp_nerf(self.n_layers,
                                 self.hidden_dim,
                                 in_dim_pos,
                                 out_dim,
                                 self.skips,
                                 bias=geometric_init_bias,
                                 geometric_init=geometric_init,
                                 weight_norm=True,
                                 inside_outside=False)
        self.activation = nn.Softplus(beta=100)
    
    def forward(self, x):
        """Run sdf network.
        """
        x_enc = self.enc_fn_pos(x, bound=self.bound)
        h = x_enc.clone()
        for l in range(self.n_layers):
            if l in self.skips:
                h = torch.cat([h, x_enc], -1) / np.sqrt(2)
            h = self.net[l](h)
            if l != self.n_layers - 1:
                h = self.activation(h)
        sdf = h[..., :1]
        geo_feat = h[...,1:]
        return torch.cat([sdf, geo_feat], -1)
    
    def sdf(self, x):
        """Get sdf only.
        """
        return self.forward(x)[...,:1]
    

class ColorNetwork(nn.Module):
    """Color network.
    """
    def __init__(self,
                 enc_dir_cfg=dict(enc_type="frequency",multires=4,input_dim=3),
                 enc_pos_cfg=dict(enc_type="frequency",multires=10,input_dim=3),
                 n_layers=5,
                 hidden_dim=256,
                 skips=[],
                 out_dim=3,
                 feat_dim=256,
                 bound=1.0,
                 **kwargs,
                 ):
        super().__init__()
        self.bound = bound
        self.enc_fn_pos, in_dim_pos = get_encoder(**enc_pos_cfg)
        self.enc_fn_dir, in_dim_dir = get_encoder(**enc_dir_cfg)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.out_dim = out_dim
        self.feat_dim = feat_dim
        self.net = build_mlp_nerf(self.n_layers,
                                  self.hidden_dim,
                                  in_dim_pos + 3 + in_dim_dir + feat_dim,
                                  out_dim,
                                  self.skips,
                                  bias=True,
                                  geometric_init=False,
                                  weight_norm=True,
                                  inside_outside=False)
        self.activation = nn.ReLU()
    
    def forward(self, x, n, d, geo_feat):
        """Run color network.
        """
        x_enc = self.enc_fn_pos(x, bound=self.bound)
        d_enc = self.enc_fn_dir(d, bound=self.bound)
        input = torch.cat([x_enc, n, d_enc, geo_feat], -1)
        h = input.clone()
        for l in range(self.n_layers):
            if l in self.skips:
                h = torch.cat([h, input], -1) / np.sqrt(2)
            h = self.net[l](h)
            if l != self.n_layers - 1:
                h = self.activation(h)
        out = torch.sigmoid(h)
        return out
    

class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super().__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        device = x.device
        return torch.ones([len(x), 1]).to(device) * torch.exp(self.variance * 10.0).to(device)