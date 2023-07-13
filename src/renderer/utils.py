import torch
import mcubes
import numpy as np
import torch.nn as nn

import sys
sys.path.append("./")
from src.trainer.utils import tensor2array


def build_mlp_nerf(n_layers,
                  hidden_dim,
                  in_dim,
                  out_dim,
                  skips=[],
                  bias=0.5,
                  geometric_init=False,
                  weight_norm=True,
                  inside_outside=False):
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
        if isinstance(bias, bool):
            lin = nn.Linear(dim0, dim1, bias=bias)
        else:
            lin = nn.Linear(dim0, dim1, bias=True)
            if geometric_init:
                if l == n_layers - 1:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dim0), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dim0), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(dim1))
                elif l in skips:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(dim1))
                    torch.nn.init.constant_(lin.weight[:, -(in_dim - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(dim1))
        if weight_norm:
            lin = nn.utils.weight_norm(lin)
        net.append(lin)
    return nn.ModuleList(net)


def build_mlp_idr(n_layers,
                  hidden_dim,
                  in_dim,
                  out_dim,
                  skips=[],
                  bias=0.5,
                  geometric_init=False,
                  weight_norm=True,
                  inside_outside=False):
    """MLP from IDR."""
    net = []
    for l in range(n_layers):
        if l == 0:
            dim0 = in_dim
        else:
            dim0 = hidden_dim
        if l == n_layers - 1:
            dim1 = out_dim
        elif l+1 in skips:
            dim1 = hidden_dim - in_dim
        else:
            dim1 = hidden_dim
        if isinstance(bias, bool):
            lin = nn.Linear(dim0, dim1, bias=bias)
        else:
            lin = nn.Linear(dim0, dim1, bias=True)
            if geometric_init:
                if l == n_layers - 1:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dim0), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dim0), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(dim1))
                elif l in skips:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(dim1))
                    torch.nn.init.constant_(lin.weight[:, -(in_dim - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(dim1))
        if weight_norm:
            lin = nn.utils.weight_norm(lin)
        net.append(lin)
    return nn.ModuleList(net)

def run_fn_split(fn, inputs, chunk, cpu=False,**kwargs):
    """Run a function by spliting inputs into batches.
    input: nxm (n: number of inputs, m: input dimension)
    """
    val = []
    for inputs_split in torch.split(inputs, chunk, 0):
        if cpu:
            val.append(tensor2array(fn(inputs_split, **kwargs)))
        else:
            val.append(fn(inputs_split, **kwargs))
    if cpu:
        val = np.concatenate(val, 0)
    else:
        val = torch.cat(val, 0)
    return val


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func, device):
    u = extract_fields(bound_min, bound_max, resolution, query_func, device)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()
    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def extract_fields(bound_min, bound_max, resolution, query_func, device):
    N = 128 # 64. Change it when memory is insufficient!
    X = torch.linspace(bound_min[0], bound_max[0], resolution, device=device).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution, device=device).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution, device=device).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs))
                    if torch.is_tensor(val):
                        val = tensor2array(val)
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
                    del xx, yy, zz
    return u


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is built upon NeRF
    # Get pdf
    device = weights.device
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]).to(device), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1) # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def get_sphere_intersection(rays_o, rays_d, r=1.0, add_dim=True):
    """Get ray intersection of sphere.
    """
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(rays_d * rays_o, dim=-1) / torch.sum(rays_d * rays_d, dim=-1)
    p = rays_o + d1.unsqueeze(-1) * rays_d

    tmp = r * r - torch.sum(p * p, dim=-1)
    mask_intersect = tmp > 0.0
    d2 = torch.sqrt(torch.clamp(tmp, min=0.0)) / torch.norm(rays_d, dim=-1)
    near = torch.clamp(d1 - d2, min=0.0)
    far = d1 + d2
    if add_dim:
        near = near[..., None]
        far =far[..., None]
        mask_intersect = mask_intersect[..., None]
    return near, far, mask_intersect