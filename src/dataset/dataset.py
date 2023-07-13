import os
import os.path as osp
import torch
import sys
import pickle
import imageio.v2 as iio
import numpy as np
import cv2
from tqdm import tqdm
import open3d as o3d

import sys
sys.path.append("./")
from src.trainer.utils import gen_pcd, to8b


virtual_cam_endonerf_dir = "src/dataset/vis_cfg/virtual_cam_endonerf.json"
virtual_cam_scared_dir = "src/dataset/vis_cfg/virtual_cam_scared2019.json"
render_option_endonerf_dir = "src/dataset/vis_cfg/render_option.json"


class Dataset(object):
    """Dataset laoder.
    """
    def __init__(self,
                 dset_cfg,
                 device = "cuda",
                 ):
        info_dir = dset_cfg["info_dir"]
        assert osp.exists(info_dir), f"Info file {info_dir} does not exists! Preprocess the dataset first!"
        with open(info_dir, "rb") as handle:
            info = pickle.load(handle)
            
        self.dset_cfg = dset_cfg
        self.device = device
        self.dset_name = info["dset_name"]
        self.scene_name = info["scene_name"]
        
        tqdm.write(f"[Load data] dataset: {self.dset_name}, scene: {self.scene_name}")
        
        # Intrinsics
        self.n_frames = info["n_frames"]
        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        scale_mat = info["scale_mat"]
        self.depth_scale = info["depth_norm_scale"]
        # world_mat is a projection matrix from world to image ( K[R|t] )
        world_mat = info["world_mat"]
        self.intrinsics = []
        self.poses = []  # camera to world
        self.w, self.h = info["wh"][0], info["wh"][1]
        for i_frame in range(self.n_frames):
            P = world_mat[i_frame] @ scale_mat
            P = P[:3, :4]
            K, pose = self._load_K_Rt_from_P(None, P)
            self.intrinsics.append(self._array2tensor(K, self.device))
            self.poses.append(self._array2tensor(pose))
        self.intrinsics = torch.stack(self.intrinsics, 0)  # [n_frames, 4, 4]
        self.poses = torch.stack(self.poses, 0)  # [n_frames, 4, 4]
        bounds = self._array2tensor(info["bounds"] / self.depth_scale, self.device)  # Scale bounds
        self.bbox_minmax = info["bbox_minmax"]
        if self.dset_name == "scared2019":
            bbox_minmax = np.swapaxes(np.vstack([self.bbox_minmax[:,:,0].min(0), self.bbox_minmax[:,:,1].max(0)]), 0, 1)
            self.bbox_minmax = np.tile(bbox_minmax[None,...], (self.bbox_minmax.shape[0], 1, 1))
        
        # Read colors, depths, and masks
        self.colors = self._load_imgs(info["color"], "color", self.device) # [n_frames, 4, 4]
        depth_type = info["depth_type"]
        if depth_type == "depth":
            self.depths = self._load_imgs(info["depth"], depth_type, self.device)
        elif depth_type == "disp":
            self.depths = self._load_imgs(info["depth"], depth_type, self.device, disp_const=info["disp_const"])
        else:
            assert NotImplementedError("Unknown depth type!")
        self.depths = self.depths / self.depth_scale
        self.near = np.percentile(self._tensor2array(self.depths), 3.0)
        self.far = np.percentile(self._tensor2array(self.depths), 99.5)
        self.depth_masks = torch.bitwise_and(self.depths > self.near, self.depths <self.far) * 1.0
        
        mask_type = info["mask_type"]
        if mask_type != None:
            self.color_masks = self._load_imgs(info["mask"], mask_type, self.device)
        else:
            self.color_masks = torch.ones_like(self.depth_masks)
        self.masks = self.depth_masks * self.color_masks
        
        # Rays
        rays = self.get_rays(self.intrinsics, self.poses, self.w, self.h)
        self.bds = bounds[:, None, None, :].expand(self.n_frames, self.h, self.w, 2)
        
        normalize_time = dset_cfg["normalize_time"]
        if normalize_time:
            ts = torch.linspace(0., 1., self.n_frames, device=self.device)
        else:
            ts = torch.arange(self.n_frames, device=self.device)
        self.ts = ts[:, None, None, None].expand(self.n_frames, self.h, self.w, 1)
        self.rays = torch.cat([rays, self.bds, self.ts], -1)  # [n_frames, w, h, 9]
        
        # Dataloader
        self.list_train = info["list_train"]
        self.list_test = info["list_test"]
        self.n_train = len(self.list_train)
        self.n_test = len(self.list_test)
        
        # Mask guided ray sampling
        self.ray_importance_maps = self._ray_sampling_importance_from_masks(self.masks)
        
        # Load visualization config
        if self.dset_name == "endonerf":
            self.vcam = virtual_cam_endonerf_dir
            self.render_option = render_option_endonerf_dir
        elif self.dset_name == "scared2019":
            self.vcam = virtual_cam_scared_dir
            self.render_option = render_option_endonerf_dir
        
        tqdm.write("[Load data] complete!")
        
    def get_train_batch_data_by_index(self, id_train=None, ray_batch=1024, mask_guided_ray_sampling=True):
        """Get training samples for one iteration. One image per iteration.
        """
        if id_train == None:
            id_train = np.random.choice(self.list_train)
        else:
            assert id_train in self.list_train, f"ID {id_train} is not in training list!"
        colors = self.colors[id_train]
        rays = self.rays[id_train]
        depths = self.depths[id_train]
        masks = self.masks[id_train]
        color_masks = self.color_masks[id_train]
        depth_masks = self.depth_masks[id_train]
        # Ray sampling
        coords = torch.stack(torch.meshgrid(torch.linspace(0, self.h-1, self.h),
                                            torch.linspace(0, self.w-1, self.w), indexing="ij"), -1)  # (H, W, 2)
        coords = torch.reshape(coords, [-1,2])
        coords = coords[color_masks[..., 0].flatten() == 1., ...]  # Filter out pxiels out of color mask. We don't use color+depth mask in case depth mask is too sparse.
        if mask_guided_ray_sampling:  # Mask guided ray sampling
            ray_importance_map = self.ray_importance_maps[id_train]
            select_inds = self._importance_sampling_coords(
                ray_importance_map[coords[:, 0].long(), coords[:, 1].long(), 0].unsqueeze(0),
                ray_batch, device=self.device)
            select_inds = torch.max(torch.zeros_like(select_inds), select_inds)
            select_inds = torch.min((coords.shape[0] - 1) * torch.ones_like(select_inds), select_inds)
            select_inds = select_inds.squeeze(0)
        else:  # Uniform sampling
            select_inds = np.random.choice(coords.shape[0], size=[ray_batch], replace=False)
        
        select_coords = coords[select_inds].long()  # (ray_batch, 2)
        rays_out =rays[select_coords[:, 0], select_coords[:, 1], :]  # (ray_batch, 9)
        colors_out = colors[select_coords[:, 0], select_coords[:, 1], :]  # (ray_batch, 3)
        depths_out = depths[select_coords[:, 0], select_coords[:, 1], :] # (ray_batch, 1)
        masks_out = masks[select_coords[:, 0], select_coords[:, 1], :] # (ray_batch, 1)
        color_masks_out = color_masks[select_coords[:, 0], select_coords[:, 1], :] # (ray_batch, 1)
        depth_masks_out = depth_masks[select_coords[:, 0], select_coords[:, 1], :] # (ray_batch, 1)
        out = {
            "color": colors_out,
            "rays": rays_out,
            "depth": depths_out,
            "mask": masks_out,
            "color_mask": color_masks_out,
            "depth_mask": depth_masks_out,
        }
        return out
    
    def get_frame_data_by_index(self, id):
        """Get frame data by index or indices.
        """
        colors = self.colors[id]
        rays = self.rays[id]
        depths = self.depths[id]
        masks = self.masks[id]
        color_masks = self.color_masks[id]
        depth_masks = self.depth_masks[id]
        out = {
            "color": colors,
            "rays": rays,
            "depth": depths,
            "mask": masks,
            "color_mask": color_masks,
            "depth_mask": depth_masks,
        }
        return out
    
    def vis_dataset(self):
        """Visualize dataset.
        """
        cameras = []
        bboxes_o3d = []
        pcds = o3d.geometry.PointCloud()
        for i in range(self.n_frames):
            pcd = gen_pcd(
                to8b(self.colors[i]), self.depths[i], self.intrinsics[i],
                np.linalg.inv(self._tensor2array(self.poses[i])), 1., self.far)
            pcd = pcd.random_down_sample(0.1)
            pcds = pcds + pcd
            camera = o3d.geometry.LineSet.create_camera_visualization(
                self.w, self.h,
                self._tensor2array(self.intrinsics[i][:3, :3]),
                np.linalg.inv(self._tensor2array(self.poses[i])), scale=1.0)
            bbox_o3d = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
                o3d.geometry.AxisAlignedBoundingBox(min_bound=self.bbox_minmax[i,:,:1],
                                                     max_bound=self.bbox_minmax[i,:,1:]))
            if i == 0:
                camera.paint_uniform_color(np.array([[1.],[0.],[0.]]))
                bbox_o3d.paint_uniform_color(np.array([[1.], [0.], [0.]]))
            elif i == self.n_frames - 1:
                camera.paint_uniform_color(np.array([[0.],[1.],[0.]]))
                bbox_o3d.paint_uniform_color(np.array([[0.], [1.], [0.]]))
            else:
                camera.paint_uniform_color(np.array([[0.],[0.],[1.]]))
                bbox_o3d.paint_uniform_color(np.array([[0.], [0.], [1.]]))
            cameras.append(camera)
            bboxes_o3d.append(bbox_o3d)
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0.0, 0.0, 0.0]))
        sphere = o3d.geometry.LineSet.create_from_triangle_mesh(o3d.geometry.TriangleMesh.create_sphere(radius=1.0))
        o3d.visualization.draw_geometries([pcds, coord, sphere, bbox_o3d]+cameras+bboxes_o3d)
    
    def get_rays(self, intrinsics, poses, w, h):
        """Get rays of all frames.
        """
        rays = []
        intrinsics_inv = torch.inverse(intrinsics)  # [n_images, 4, 4]
        n_frames = intrinsics.shape[0]
        for i_frame in range(n_frames):
            pixels_x, pixels_y = torch.meshgrid(torch.linspace(0, w-1, w, device=self.device),
                                                torch.linspace(0, h-1, h, device=self.device),
                                                indexing="ij")
            pixels_x = pixels_x.t()
            pixels_y = pixels_y.t()
            p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # H, W, 3
            rays_d = torch.matmul(intrinsics_inv[i_frame, None, None, :3, :3], p[:, :, :, None]).squeeze()
            rays_d = rays_d / torch.linalg.norm(rays_d, ord=2, dim=-1, keepdim=True)  # Normalize ray direction
            rays_d = torch.matmul(poses[i_frame, None, None, :3, :3], rays_d[:, :, :, None]).squeeze()
            rays_o = poses[i_frame, None, None, :3, 3].expand(rays_d.shape)  # H, W, 3
            rays.append(torch.cat([rays_o, rays_d], -1))
        rays = torch.stack(rays, 0)
        return rays    
    
    @staticmethod
    def _importance_sampling_coords(weights, N_samples, det=False, device="cuda"):
        """Sample coordianates based on importance map.
        """
        # Get pdf
        
        weights = weights + 1e-5 # prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)

        # Take uniform samples
        if det:
            u = torch.linspace(0., 1., steps=N_samples, device=device)
            u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)
            
        # Invert CDF
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)

        return inds  
    
    @staticmethod
    def _ray_sampling_importance_from_masks(masks):
        """Generate importance maps based on masks.
        """
        freq = (1.0 - masks).sum(0)
        p = freq / torch.sqrt((torch.pow(freq, 2)).sum())
        return masks * (1.0 + p)
    
    @staticmethod
    def _array2tensor(array, device="cuda", dtype=torch.float32):
        return torch.tensor(array, dtype=dtype, device=device)
    
    @staticmethod
    def _tensor2array(tensor):
        return tensor.detach().cpu().numpy()
    
    @staticmethod
    def _load_imgs(img_list, img_type, device="cuda", **kwargs):
        """Load images.
        """
        assert img_type in ["color", "depth", "disp", "mask", "mask_invert"]
        
        def imread(f):
            if f.endswith('png'):
                return iio.imread(f, ignoregamma=True)
            else:
                return iio.imread(f)
            
        imgs = []
        for i_img, img_file in enumerate(img_list):
            if img_type == "color":
                img = torch.tensor(imread(img_file)[...,:3]/255, device=device, dtype=torch.float32)
            elif img_type == "depth":
                img = torch.tensor(imread(img_file) * 1.0, device=device, dtype=torch.float32)
                img = img[..., None]
            elif img_type == "disp":
                disp_const = kwargs["disp_const"][i_img]
                disp = torch.tensor(imread(img_file) * 1.0, device=device, dtype=torch.float32)
                img = torch.zeros_like(disp)
                mask = disp != 0
                img[mask] = disp_const / disp[mask]
                img = img[..., None]
            elif img_type =="mask":
                img = torch.tensor(imread(img_file) / 255.0, device=device, dtype=torch.float32)
                img = img[..., None]
            elif img_type =="mask_invert":
                img = torch.tensor(1.0 - imread(img_file) / 255.0, device=device, dtype=torch.float32)
                img = img[..., None]
            imgs.append(img)
        imgs = torch.stack(imgs, dim=0)
        return imgs
    
    @staticmethod
    def _load_K_Rt_from_P(filename, P=None):
        # This function is borrowed from IDR: https://github.com/lioryariv/idr
        if P is None:
            lines = open(filename).read().splitlines()
            if len(lines) == 4:
                lines = lines[1:]
            lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
            P = np.asarray(lines).astype(np.float32).squeeze()

        out = cv2.decomposeProjectionMatrix(P)
        K = out[0]
        R = out[1]
        t = out[2]

        K = K / K[2, 2]
        intrinsics = np.eye(4)
        intrinsics[:3, :3] = K

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.transpose()
        pose[:3, 3] = (t[:3] / t[3])[:, 0]

        return intrinsics, pose

