import os
import os.path as osp
import json
import yaml
import wandb
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import cv2
import numpy as np
import torch.nn as nn
import scipy
import kornia
import math
import lpips
import open3d as o3d
import imageio.v2 as iio
from scipy.spatial.transform import Rotation as R
import copy
import warnings


##### Config #####
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f, Loader=yaml.Loader)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v
            
            
##### Summary writer #####
class CustomSummaryWritter(object):
    """Customized summary writer. Support wandb or tensorboard.
    """
    def __init__(self, writer_cfg, cfg):
        self.writer_type = writer_cfg["type"]
        proj_name = writer_cfg["proj_name"]
        exp_name = writer_cfg["exp_name"]
        exp_dir = writer_cfg["exp_dir"]
        writer_save_dir = osp.join(exp_dir, "logs")
        os.makedirs(writer_save_dir, exist_ok=True)
        if self.writer_type == "tensorboard":
            self.writer = SummaryWriter(writer_save_dir)
            self.writer.add_text('config', self.dict2string(cfg), global_step=0)
        elif self.writer_type == "wandb":
            entity = writer_cfg["entity"]
            resume = cfg["train"]["resume"]
            wandb.init(project=proj_name, entity=entity, name=exp_name,
                       config=cfg, dir=writer_save_dir, resume=resume)
        else:
            raise NotImplementedError("Unknown summary writter!")
        
    def add_scalar(self, tag, scalar_value, global_step):
        """Add scalar infromation.
        """
        if torch.is_tensor(scalar_value):
            scalar_value = scalar_value.item()
        if self.writer_type == "tensorboard":
            self.writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step)
        elif self.writer_type == "wandb":
            wandb.log({tag:scalar_value}, step=global_step)
    
    def add_rgb(self, tag, img, global_step):
        """Add rgb in the format of HWC.
        """
        if torch.is_tensor(img):
            img = tensor2array(img)
        if self.writer_type == "tensorboard":
            self.writer.add_image(tag, img, global_step, dataformats="HWC")
        elif self.writer_type == "wandb":
            wandb.log({tag: wandb.Image(img)}, step=global_step)
            
    def add_video(self, tag, video, global_step, fps=10):
        """Add video
        video: [t,w,h,3]
        """
        if self.writer_type == "tensorboard":
            if not torch.is_tensor(video):
                video = torch.from_numpy(video)
            video = video.permute((0,3,1,2))
            video = video[None, ...]
            self.writer.add_video(tag, video, global_step, fps=fps)
        elif self.writer_type == "wandb":
            if torch.is_tensor(video):
                video = tensor2array(video)
            video = np.transpose(video, axes=[0,3,1,2])
            wandb.log({tag: wandb.Video(video, fps=fps, format="gif")}, step=global_step)
            
    def add_mesh(self, tag, vertices, global_step, colors=None, faces=None):
        """Add mesh.
        """
        if self.writer_type == "tensorboard":
            self.writer.add_mesh(tag, vertices, colors=colors, faces=faces, global_step=global_step)
        elif self.writer_type == "wandb":
            raise NotImplementedError()
        
    
    @staticmethod
    def dict2string(hp):
        """
        Transfer dictionary to string.
        """
        json_hp = json.dumps(hp, indent=2)
        return "".join("\t" + line for line in json_hp.splitlines(True))
    
    
def tensor2array(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    else:
        return tensor

def array2tensor(array, device="cuda", dtype=torch.float32):
    return torch.tensor(array, dtype=dtype, device=device)
    

##### Visualization

MAX_WINDOW_WIDTH = 6000


def to8b(x):
    if torch.is_tensor(x):
        x = tensor2array(x)
    return (255.*np.clip(x,0,1)).astype(np.uint8)


def add_text_to_img(img,
                    text,
                    font=cv2.FONT_HERSHEY_SIMPLEX,
                    font_color=(255,0,0),
                    org=(10,50),
                    font_scale=2,
                    thickness=4,
                    line_type=cv2.LINE_AA,
                    ):
    if torch.is_tensor(img):
        img = tensor2array(img)
    if not np.issubdtype(img.dtype, np.uint8):
        img = to8b(img)
    img_show = cv2.putText(copy.deepcopy(img), text, org, font, font_scale, 
                           font_color, thickness, line_type)
    return img_show
    
    
def gen_normal(normal_stack, pose, n_frames, W, H, revert=False, filter=None):
    if isinstance(normal_stack, list): 
        normal_img = np.concatenate(normal_stack, axis=0).reshape(n_frames, -1, 3)
    elif torch.is_tensor(normal_stack):
        normal_img = tensor2array(normal_stack).reshape(n_frames, -1, 3)
    else:
        normal_img = normal_stack.reshape(n_frames, -1, 3)
    normal_img = normal_img / (np.linalg.norm(normal_img, axis=-1, keepdims=True) + 1e-10)
    rot = np.linalg.inv(pose[:, :3, :3].detach().cpu().numpy())
    normal_img = np.matmul(rot[:, None, :, :], normal_img[:, :, :, None]).reshape([n_frames, H, W, 3])
    # We set coordiante same as camera, therefore z-normal is always false, leading to no blue color. Revert it for better visualization.
    if revert:
        normal_img = - normal_img
    if filter is not None:
        normal_filtered = []
        for i in range(normal_img.shape[0]):
            normal_filtered.append(cv2.bilateralFilter(normal_img[i], filter[0], filter[1], filter[2]))
        normal_img = np.stack(normal_filtered, axis=0)
    normal_img_show = np.uint8((normal_img * 128 + 128).clip(0, 255))
    return normal_img, normal_img_show


def gen_rgb(rgb_stack, n_frames, W, H):
    """Generate rgb.
    """
    if isinstance(rgb_stack, list): 
        img_fine = np.concatenate(rgb_stack, axis=0).reshape([n_frames, H, W, 3])
    elif torch.is_tensor(rgb_stack):
        img_fine = tensor2array(rgb_stack)
    else:
        img_fine = rgb_stack
    if len(img_fine.shape) == 3:
        img_fine = np.stack([img_fine, img_fine, img_fine], -1)
    img_fine_show = np.uint8((img_fine * 256).clip(0, 255))
    return img_fine, img_fine_show


def gen_depth(depth_stack, n_frames, W, H, depth_max=None, filter=None):
    """Generate depth.
    """
    if isinstance(depth_stack, list):  
        depth_img = np.concatenate(depth_stack, axis=0)
        depth_img = depth_img.reshape([n_frames, H, W, 1])
    elif torch.is_tensor(depth_stack):  # GT
        depth_img = tensor2array(depth_stack)
    else:
        depth_img = depth_stack
    if len(depth_img.shape) == 3:
        depth_img = depth_img[..., None]
    if depth_max is None:
        depth_max = depth_img.max()
    if filter is not None:
        depth_filterd = []
        for i in range(depth_img.shape[0]):
            d_filter = cv2.medianBlur(depth_img[i], 3)
            d_filter = cv2.bilateralFilter(d_filter, filter[0], filter[1], filter[2])
            depth_filterd.append(d_filter)
        depth_img = np.stack(depth_filterd, axis=0)[..., None]
    depth_img_show = np.uint8(255. - np.clip(depth_img/depth_max, a_max=1, a_min=0) * 255.)
    depth_img_show = np.concatenate([depth_img_show, depth_img_show, depth_img_show], -1)
    return depth_img, depth_img_show


def gen_pcd(rgb, depth, K, pose, depth_scale=1., depth_truc=3., depth_filter=None):
    """Generate point cloud.
    """
    if torch.is_tensor(rgb):
        rgb = tensor2array(rgb)
    if torch.is_tensor(depth):
        depth = tensor2array(depth)
    if torch.is_tensor(K):
        K = tensor2array(K)
    if torch.is_tensor(pose):
        pose = tensor2array(pose)
    if depth_filter is not None:
         depth = cv2.bilateralFilter(depth, depth_filter[0], depth_filter[1], depth_filter[2])
    h, w = rgb.shape[:-1]
    rgb_im = o3d.geometry.Image(rgb.astype(np.uint8))
    depth_im = o3d.geometry.Image(depth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_im, 
        depth_im, 
        depth_scale=depth_scale,
        depth_trunc=depth_truc/depth_scale,
        convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(w, h, K[:3, :3]),
        pose,
        project_valid_depth_only=True,
    )
    return pcd


def vis_pcd(vis, pcd, virtual_cam_dir=None, render_option_dir=None):
    """Visualize point cloud with open3d.
    """
    vis.reset_view_point(True)
    vis.clear_geometries()
    vis.add_geometry(pcd)
    if render_option_dir is not None:
        vis.get_render_option().load_from_json(render_option_dir)
    if virtual_cam_dir is not None:
        parameters = o3d.io.read_pinhole_camera_parameters(virtual_cam_dir)
        vis.get_view_control().convert_from_pinhole_camera_parameters(parameters)
    vis.poll_events()
    vis.update_renderer()
    o3d_screenshot_mat = to8b(np.asarray(vis.capture_screen_float_buffer()))
    return o3d_screenshot_mat


def vis_mesh(vis, mesh, virtual_cam_dir=None, render_option_dir=None):
    """Visualize mesh with open3d.
    """
    vis.clear_geometries()
    mesh.compute_vertex_normals()
    vis.add_geometry(mesh)
    if virtual_cam_dir is not None:
        parameters = o3d.io.read_pinhole_camera_parameters(virtual_cam_dir)
        vis.get_view_control().convert_from_pinhole_camera_parameters(parameters)
    if render_option_dir is not None:
        vis.get_render_option().load_from_json(render_option_dir)
    vis.poll_events()
    vis.update_renderer()
    o3d_screenshot_mat = to8b(np.asarray(vis.capture_screen_float_buffer()))
    return o3d_screenshot_mat


def gen_normal_from_depth(rays, depths, device, mask=None):
    """Generate normal map from depth map.
    """
    if not torch.is_tensor(depths):
        depths = torch.tensor(depths, device=device)

    rays_o, rays_d = rays[...,:3], rays[...,3:6]

    pts = rays_o + rays_d * depths

    u = pts[:,1:-1,:-2,:] - pts[:,1:-1,1:-1,:]
    v = pts[:,:-2,1:-1,:] - pts[:,1:-1,1:-1,:]
    n = torch.cross(u,v)

    n = n / (torch.linalg.norm(n, ord=2, dim=-1, keepdim=True) + 1e-10)
    n = -tensor2array(n)
    n_pad = np.zeros([*list(depths.shape[:-1]), 3])
    n_pad[:,1:-1,1:-1,:] = n 
    n_show = np.uint8((n_pad * 128 + 128).clip(0, 255))
    if mask is not None:
        n_show = n_show * mask
    return n, n_show



##### Metics #####
def cal_psnr(a, b, mask):
    """Compute psnr.
    """
    if torch.is_tensor(a):
        a = tensor2array(a)
    if torch.is_tensor(b):
        b = tensor2array(b)
    if torch.is_tensor(mask):
        mask = tensor2array(mask)
    if len(mask.shape) == len(a.shape) - 1:
        mask = mask[..., None]
    mask_sum = np.sum(mask) + 1e-10
    psnr =  20.0 * np.log10(1.0 / (((a - b)**2 * mask).sum() / (mask_sum * 3.0))**0.5)
    return psnr
    
    
def cal_rmse(a, b, mask):
    """Compute rmse.
    """
    if torch.is_tensor(a):
        a = tensor2array(a)
    if torch.is_tensor(b):
        b = tensor2array(b)
    if torch.is_tensor(mask):
        mask = tensor2array(mask)
    if len(mask.shape) == len(a.shape) - 1:
        mask = mask[..., None]
    mask_sum = np.sum(mask) + 1e-10
    rmse = (((a - b)**2 * mask).sum() / (mask_sum))**0.5
    return rmse


# structural similarity index
class SSIM(object):
    """
    borrowed from https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py
    """
    def gaussian(self, w_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - w_size//2)**2/float(2*sigma**2)) for x in range(w_size)])
        return gauss/gauss.sum()

    def create_window(self, w_size, channel=1):
        _1D_window = self.gaussian(w_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, w_size, w_size).contiguous()
        return window

    def __call__(self, y_pred, y_true, w_size=11, size_average=True, full=False):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            w_size : int, default 11
            size_average : boolean, default True
            full : boolean, default False
        return ssim, larger the better
        """
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        if torch.max(y_pred) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(y_pred) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val

        padd = 0
        (_, channel, height, width) = y_pred.size()
        window = self.create_window(w_size, channel=channel).to(y_pred.device)

        mu1 = F.conv2d(y_pred, window, padding=padd, groups=channel)
        mu2 = F.conv2d(y_true, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(y_pred * y_pred, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(y_true * y_true, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(y_pred * y_true, window, padding=padd, groups=channel) - mu1_mu2

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

        if full:
            return ret, cs
        return ret
    
    
ssim = SSIM()
def cal_ssim(a, b, mask, device="cuda"):
    """Compute ssim.
    a, b: [batch, H, W, 3]"""
    if not torch.is_tensor(a):
        a = array2tensor(a, device)
    if not torch.is_tensor(b):
        b = array2tensor(b, device)
    if not torch.is_tensor(mask):
        mask = array2tensor(mask, device)
    a = a * mask
    b = b * mask
    a = a.permute(0,3,1,2)
    b = b.permute(0,3,1,2)
    return ssim(a, b)

# Learned Perceptual Image Patch Similarity
class LPIPS(object):
    """
    borrowed from https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py
    """
    def __init__(self, device="cuda"):
        self.model = lpips.LPIPS(net='vgg').to(device)

    def __call__(self, y_pred, y_true, normalized=True):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            normalized : change [0,1] => [-1,1] (default by LPIPS)
        return LPIPS, smaller the better
        """
        if normalized:
            y_pred = y_pred * 2.0 - 1.0
            y_true = y_true * 2.0 - 1.0
        error =  self.model.forward(y_pred, y_true)
        return torch.mean(error)
    
    
lpips = LPIPS()
def cal_lpips(a, b, mask, device="cuda", batch=2):
    """Compute lpips.
    a, b: [batch, H, W, 3]"""
    if not torch.is_tensor(a):
        a = array2tensor(a, device)
    if not torch.is_tensor(b):
        b = array2tensor(b, device)
    if not torch.is_tensor(mask):
        mask = array2tensor(mask, device)
    a = a * mask
    b = b * mask
    a = a.permute(0,3,1,2)
    b = b.permute(0,3,1,2)
    lpips_all = []
    for a_split, b_split in zip(a.split(split_size=batch, dim=0), b.split(split_size=batch, dim=0)):
        out = lpips(a_split, b_split)
        lpips_all.append(out)
    lpips_all = torch.stack(lpips_all)
    lpips_mean = lpips_all.mean()
    return lpips_mean
    

def ssim_loss_fn(X, Y, mask=None, data_range=1.0, win_size=11, win_sigma=1.5, K=(0.01, 0.03)):
    r"""Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images of shape [b, c, h, w]
        Y (torch.Tensor): images of shape [b, c, h, w]
        mask (torch.Tensor): [b, 1, h, w]
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
    Returns:
        torch.Tensor: per pixel ssim results (same size as input images X, Y)
    """
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    if not X.type() == Y.type():
        raise ValueError("Input images should have the same dtype.")

    if len(X.shape) != 4:
        raise ValueError(f"Input images should be 4-d tensors, but got {X.shape}")

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    win = _fspecial_gauss_1d(win_size, win_sigma)
    win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map
    ssim_map = ssim_map.mean(dim=1, keepdim=True)

    if mask is not None:
        ### pad ssim_map to original size
        ssim_map = F.pad(
            ssim_map, (win_size // 2, win_size // 2, win_size // 2, win_size // 2), mode="constant", value=1.0
        )

        mask = kornia.morphology.erosion(mask.float(), torch.ones(win_size, win_size).float().to(mask.device)) > 0.5
        # ic(ssim_map.shape, mask.shape)
        ssim_map = ssim_map[mask]

    return 1.0 - ssim_map.mean()


def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    r"""Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out