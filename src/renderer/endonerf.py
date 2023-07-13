import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append("./")
from src.renderer.encoder import get_encoder
from src.renderer.utils import run_fn_split, extract_geometry


class EndoNeRFRenderer(nn.Module):
    """EndoNeRF renderer.
    """
    def __init__(self, render_cfg, net_cfg, device="cuda"):
        super().__init__()
        
        self.render_cfg = render_cfg.copy()
        self.device = device
        
        # Network
        self.model = DNeRFNet(**net_cfg).to(device)
        self.train_params = self.model.parameters()
        self.n_samples = render_cfg["n_samples"]
        self.n_importance = render_cfg["n_importance"]
        self.perturb = render_cfg["perturb"]
        self.use_depth_sampling = render_cfg["use_depth_sampling"]
        self.net_chunk = render_cfg["net_chunk"]
        
    def get_train_params(self):
        """Get training parameters.
        """
        train_params = []
        train_params += list(self.model.parameters())
        return train_params
    
    def load_checkpoint(self, ckpt):
        """Load checkpoint.
        """
        self.model.load_state_dict(ckpt)
        
    def save_checkpoint(self):
        """Output checkpoints for saving.
        """
        ckpt = {
            "network": self.model.state_dict(),
        }
        return ckpt
        
    def forward(self, rays, **kwargs):
        """Forward function.
        """
        ret = self.render_rays(rays, **kwargs)
        return ret
    
    def render_rays(self, rays, iter_step, eval=False):
        """Render rays.
        """
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[..., :3], rays[..., 3:6]
        near_or_mean, far_or_std, time = rays[..., 6:7], rays[..., 7:8], rays[..., 8:]
        rays_d_z = rays_d / (rays_d[..., 2:] + 1e-5)
        
        # Sample points
        if self.use_depth_sampling:
            mean = near_or_mean.expand([n_rays, self.n_samples])
            std = far_or_std.expand([n_rays, self.n_samples])
            z_vals, _ = torch.sort(torch.normal(mean, std), dim=1)
        else:
            t_vals = torch.linspace(0., 1., steps=self.n_samples, device=self.device)
            near_or_mean * (1.-t_vals) + far_or_std * (t_vals)
            if self.perturb:
                # get intervals between samples
                mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
                upper = torch.cat([mids, z_vals[...,-1:]], -1)
                lower = torch.cat([z_vals[...,:1], mids], -1)
                # stratified samples in those intervals
                t_rand = torch.rand(z_vals.shape, device=self.device)
                z_vals = lower + (upper - lower) * t_rand
        n_samples = self.n_samples
        
        if self.n_importance > 0:
            with torch.no_grad():
                t = time.view(-1, 1, 1).expand([n_rays, self.n_samples, 1])
                pts = rays_o[..., None, :].expand([n_rays, self.n_samples, 3]) + \
                    rays_d_z[..., None, :].expand([n_rays, self.n_samples, 3]) * \
                         z_vals[..., :, None].expand([n_rays, self.n_samples, 1])
                dirs_o = rays_d[..., None, :].expand([n_rays, self.n_samples, 3])
                inputs = torch.cat([pts, dirs_o, t], dim=-1)
                raw = run_fn_split(lambda x: self.model(x, eval=eval), inputs, self.net_chunk)
                _, _, weights = self.raw2outputs(raw, z_vals, rays_d)
            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = self.importance_sampling_ray(z_vals_mid, weights[...,1:-1], self.n_importance, det=self.perturb)
            z_samples = z_samples.detach()
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            n_samples = z_vals.shape[1]
            
        t = time.view(-1, 1, 1).expand([n_rays, n_samples, 1])
        pts = rays_o[..., None, :].expand([n_rays, n_samples, 3]) + \
            rays_d_z[..., None, :].expand([n_rays, n_samples, 3]) * \
                    z_vals[..., :, None].expand([n_rays, n_samples, 1])
        dirs_o = rays_d[..., None, :].expand([n_rays, n_samples, 3])
        inputs = torch.cat([pts, dirs_o, t], dim=-1)
        raw = run_fn_split(lambda x: self.model(x, eval=eval), inputs, self.net_chunk)
        if eval:
            rgb_map, depth_map, weights, normal_map = self.raw2outputs(raw, z_vals, rays_d, eval=eval)
            ret = {
                "color_map": rgb_map,
                "depth_map": depth_map,
                "normal_map": normal_map,
            }
        else:
            rgb_map, depth_map, weights = self.raw2outputs(raw, z_vals, rays_d, eval=eval)
            
            ret = {
                "color_map": rgb_map,
                "depth_map": depth_map,
            }
        
        return ret
        
    def importance_sampling_ray(self, bins, weights, N_samples, det=False, pytest=False):
        # Get pdf
        weights = weights + 1e-5 # prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

        # Take uniform samples
        if det:
            u = torch.linspace(0., 1., steps=N_samples).to(self.device)
            u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_samples]).to(self.device)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            new_shape = list(cdf.shape[:-1]) + [N_samples]
            if det:
                u = np.linspace(0., 1., N_samples)
                u = np.broadcast_to(u, new_shape)
            else:
                u = np.random.rand(*new_shape)
            u = torch.Tensor(u)

        # Invert CDF
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)

        below = torch.max(torch.zeros_like(inds-1), inds-1)
        above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

        # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = (cdf_g[...,1]-cdf_g[...,0])
        denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
        t = (u-cdf_g[...,0])/denom
        samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

        return samples
                
    def raw2outputs(self, raw, z_vals, rays_d, eval=False):
        """Render to rgb and others.
        """
        device = raw.device
        raw2alpha = lambda raw, dists: 1.-torch.exp(-raw*dists)
        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = torch.cat([dists, torch.tensor([1e10], device=device).expand(dists[...,:1].shape)], -1)  # [n_rays, n_samples]
        
        dists = dists * torch.norm(rays_d[...,:1,:], dim=-1)
        
        rgb = raw[...,:3]  # [n_rays, n_samples, 3]
        density = raw[...,3]
        alpha = raw2alpha(density, dists)  # [n_rays, n_samples]
        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        rgb_map = torch.sum(weights[...,None] * rgb, 1)  # [N_rays, 3]
        
        depth_map = torch.sum(weights * z_vals * torch.norm(rays_d[...,None,:], dim=-1), 1)
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / (torch.sum(weights, 1)  + 1e-6))
        depth_map = 1.0 / (disp_map + 1e-6)
        depth_map = depth_map.unsqueeze(-1)
        
        if eval:
            gradient_o = raw[...,4:7]
            gradient_o = gradient_o / (torch.linalg.norm(gradient_o, ord=2, dim=-1, keepdim=True) + 1e-10)
            normal = torch.sum(weights[...,None] * gradient_o, 1)
            return rgb_map, depth_map, weights, normal
        else:
            return rgb_map, depth_map, weights
    
    def renderondepth(self, rays, depth, mask, eval=True):
        """Surface rendering: render rgb based on a given depth.
        """
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[..., :3], rays[..., 3:6]
        time = rays[..., 8]
        rays_d_z = rays_d / (rays_d[..., 2:] + 1e-5)
        mask = mask[:,0] == 1
        depth_valid = depth[mask]
        rays_o_valid = rays_o[mask]
        rays_d_valid = rays_d[mask]
        rays_d_z_valid = rays_d_z[mask]
        t_valid = time[mask]
        
        pts = rays_o_valid + rays_d_z_valid * depth_valid
        dirs_o = rays_d_valid
        t = t_valid.unsqueeze(-1)
        inputs = torch.cat([pts, dirs_o, t], -1)
        with torch.no_grad():
            raw = run_fn_split(lambda x: self.model(x, eval=eval), inputs, self.net_chunk)
        color = raw[...,:3]  # [n_rays, n_samples, 3]
        gradient_o = raw[...,4:7]
        normal = gradient_o / (torch.linalg.norm(gradient_o, ord=2, dim=-1, keepdim=True) + 1e-10)
        color_out = torch.zeros((n_rays, 3), device=self.device)
        normal_out = torch.zeros((n_rays, 3), device=self.device)
        color_out[mask] = color
        normal_out[mask] = normal
        return color_out, normal_out
    
    def extract_observation_geometry(self, t, bound_min, bound_max, resolution, threshold=0.0, net_chunk=80000, cpu=True):
        """Extract geometry by marching cubes.
        """
        query_func = lambda pts: self.model.get_density_from_observed_space(pts, t)
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
        raw = run_fn_split(lambda x: self.model(x, eval=eval), torch.cat([pts, dirs, ts], -1), net_chunk, cpu=cpu)
        color = raw[...,:3]  # [n_rays, n_samples, 3]
        gradient_o = raw[...,4:7]
        if cpu:
            normal = gradient_o / (np.linalg.norm(gradient_o, ord=2, axis=-1, keepdims=True) + 1e-5)
        else:
            normal = gradient_o / (torch.linalg.norm(gradient_o, ord=2, dim=-1, keepdim=True) + 1e-10)
        color = color.reshape(*sh, 3)
        normal = normal.reshape(*sh, 3)
        return color, normal

        
class DNeRFNet(nn.Module):
    def __init__(self,
                 enc_pos_density_cfg=dict(enc_type="frequency",multires=10,input_dim=3),
                 enc_dir_color_cfg=dict(enc_type="frequency",multires=4,input_dim=3),
                 enc_time_deform_cfg=dict(enc_type="frequency",multires=10,input_dim=1),
                 enc_pos_deform_cfg=dict(enc_type="frequency",multires=10,input_dim=3),
                 net_deform_cfg=dict(n_layers=9,hidden_dim=256,skips=[5]),
                 net_density_cfg=dict(n_layers=9,hidden_dim=256,skips=[5]),
                 net_color_cfg=dict(n_layers=9,hidden_dim=128,skips=[]),
                 geo_feat_dim=256,
                 bound=1.5,
                 raw_noise_std=1.0,
                 use_deform=True,
                 **kwargs,
                 ):
        super().__init__()
        self.bound = bound
        self.raw_noise_std = raw_noise_std
        self.use_deform = use_deform
        
        ##### Initialize network #####
        # Deformation network
        if self.use_deform:
            self.enc_fn_pos_deform, in_dim_pos_deform = get_encoder(**enc_pos_deform_cfg)
            self.enc_fn_time, in_dim_time = get_encoder(**enc_time_deform_cfg)
            self.n_layers_deform = net_deform_cfg["n_layers"]
            hidden_dim_deform = net_deform_cfg["hidden_dim"]
            self.skips_deform = net_deform_cfg["skips"]
            self.net_deform = self.build_mlp(self.n_layers_deform,
                                            hidden_dim_deform,
                                            in_dim_pos_deform + in_dim_time,
                                            3,
                                            self.skips_deform,
                                            True)
        
        # Density network
        self.enc_fn_pos_density, in_dim_pos_density = get_encoder(**enc_pos_density_cfg)
        self.n_layers_density = net_density_cfg["n_layers"]
        hidden_dim_density = net_density_cfg["hidden_dim"]
        self.skips_density = net_density_cfg["skips"]
        self.net_density = self.build_mlp(self.n_layers_density,
                                         hidden_dim_density,
                                         in_dim_pos_density,
                                         1 + geo_feat_dim,
                                         self.skips_density,
                                         True)
        
        # Color network
        self.enc_fn_dir_color, in_dim_dir_color = get_encoder(**enc_dir_color_cfg)
        self.n_layers_color = net_color_cfg["n_layers"]
        hidden_dim_color= net_color_cfg["hidden_dim"]
        self.skips_color = net_color_cfg["skips"]
        self.net_color = self.build_mlp(self.n_layers_color,
                                        hidden_dim_color,
                                        in_dim_dir_color + geo_feat_dim,
                                        3,
                                        self.skips_color,
                                        True)
    
    def build_mlp(self, n_layers, hidden_dim, in_dim, out_dim, skips=[], bias=True):
        """Build mlp. (NeRF version)
        """
        net = []
        for l in range(n_layers):
            if l == 0:
                dim0 = in_dim
            elif l in skips:
                dim0 = hidden_dim + in_dim
            else:
                dim0 = hidden_dim
            if l == n_layers - 1:
                dim1 = out_dim
            else:
                dim1 = hidden_dim
            net.append(nn.Linear(dim0, dim1, bias=bias))
        return nn.ModuleList(net)
    
    def run_deform(self, x, t):
        """Run deform network.
        """
        if len(t.shape) == 1:
            t = t[None, :].expand(x.shape[0], 1)
        x_enc_deform = self.enc_fn_pos_deform(x, bound=self.bound)
        t_enc_deform = self.enc_fn_time(t, bound=self.bound)
        xt_enc_deform = torch.cat([x_enc_deform, t_enc_deform], dim=-1)
        deform = xt_enc_deform.clone()
        for l in range(self.n_layers_deform):
            if l in self.skips_deform:
                deform = torch.cat([deform, xt_enc_deform], dim=-1)
            deform = self.net_deform[l](deform)
            if l != self.n_layers_deform - 1:
                deform = F.relu(deform, inplace=True)
        return deform
    
    def run_density(self, x, raw_noise_std=0.0):
        """Run density network.
        """
        x_enc = self.enc_fn_pos_density(x, bound=self.bound)
        h = x_enc.clone()
        for l in range(self.n_layers_density):
            if l in self.skips_density:
                h = torch.cat([h, x_enc], dim=-1)
            h = self.net_density[l](h)
            if l != self.n_layers_density - 1:
                h = F.relu(h, inplace=True)
        density = h[..., :1]
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(density.shape, device=density.device) * self.raw_noise_std
        density = F.relu(density + noise)
        geo_feat = h[...,1:]
        return torch.cat([density, geo_feat], dim=-1)
    
    def run_color(self, d, geo_feat):
        """Run color network.
        """
        # Color
        d_enc = self.enc_fn_dir_color(d, bound=self.bound)
        h = torch.cat([d_enc, geo_feat], dim=-1)
        for l in range(self.n_layers_color):
            if l in self.skips_color:
                h = torch.cat([h, d_enc], dim=-1)
            h = self.net_color[l](h)
            if l != self.n_layers_color - 1:
                h = F.relu(h, inplace=True)
        rgb = torch.sigmoid(h)
        return rgb
    
    def forward(self, inputs, eval=False):
        """ Network forward function.
        inputs: [x, d, t]
        x: [N, 3], in [-bound, bound]
        d: [N, 3], nomalized in [-1, 1]
        t: [N, 1], in [0, 1]
        """
        x, d, t = inputs[..., :3], inputs[..., 3:6], inputs[..., 6:]

        # Deform
        if self.use_deform:
            deform = self.run_deform(x, t)
            x_c = x + deform
        else:
            x_c = x
        
        # Density
        if eval:
            raw_noise_std = 0.
        else:
            raw_noise_std = self.raw_noise_std
        h = self.run_density(x_c, raw_noise_std)
        density = h[..., :1]
        geo_feat = h[...,1:]
        
        # Color
        rgb = self.run_color(d, geo_feat)
        
        # Normal
        if eval:
            # add - because density is increaseing (0 for outside and positive infinitive for inside).
            # Instead, sdf is decreaseing (positive for outside and negative for inside). 
            gradient_o = -self.get_density_grad_from_observed_space(x, t)
            out = torch.cat([rgb, density, gradient_o], dim=-1)
        else:
            out = torch.cat([rgb, density], dim=-1)
        
        return out
    
    def get_density_grad_from_observed_space(self, x, t):
        """Get density gradients from observed space.
        """
        with torch.enable_grad():
            x.requires_grad_(True)
            if self.use_deform:
                deform = self.run_deform(x, t)
                x_c = x + deform
            else:
                x_c = x
            y = self.run_density(x_c)[...,:1]
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
    
    def get_density_from_observed_space(self, x, t):
        """Getn density from observed space.
        """
        # Deform
        if self.use_deform:
            deform = self.run_deform(x, t)
            x_c = x + deform
        else:
            x_c = x
        
        # Density
        if eval:
            raw_noise_std = 0.
        else:
            raw_noise_std = self.raw_noise_std
        h = self.run_density(x_c, raw_noise_std)
        density = h[..., :1]
        return density