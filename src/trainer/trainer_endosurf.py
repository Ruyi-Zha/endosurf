"""Trainer for EndoSurf.
"""
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import os.path as osp
import copy
from shutil import copyfile
from tqdm import tqdm, trange
import torch
import numpy as np
import imageio.v2 as iio
import open3d as o3d
import cv2
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
import trimesh
import argparse

import sys
sys.path.append("./")

from src.trainer.trainer_basic import Trainer
from src.renderer.endosurf import EndoSurfRenderer
from src.trainer.utils import *


class EndoSurfTrainer(Trainer):
    def __init__(self, cfg_dir, mode="train"):
        super().__init__(cfg_dir, mode)
        # Training
        self.dtype = torch.get_default_dtype()
        
    def init_exp(self):
        """Initialize experiment setting.
        """
        exp_cfg = self.cfg["exp"].copy()
        self.proj_name = exp_cfg["project_name"]
        dset_name = self.dset.dset_name
        scene_name = self.dset.scene_name
        self.exp_name = f"{exp_cfg['exp_name']}-{dset_name}-{scene_name}"
        self.exp_dir = osp.join(exp_cfg["exp_dir"], self.proj_name, self.exp_name)
        self.ckpt_dir = os.path.join(self.exp_dir, "ckpt.tar")
        self.ckpt_dir_backup = os.path.join(self.exp_dir, "ckpt_backup.tar")
        os.makedirs(self.exp_dir, exist_ok=True)
        
    def init_renderer(self):
        """Initialize renderer.
        """
        self.render_cfg = self.cfg["render"].copy()
        net_cfg = self.cfg["net"].copy()
        self.renderer = EndoSurfRenderer(self.render_cfg, net_cfg, self.device)
    
    def init_train(self):
        """Initialzie training.
        """
        self.train_cfg = self.cfg["train"].copy()
        self.n_iter = self.train_cfg["n_iter"]
        self.resume = self.train_cfg["resume"]
    
    def init_optimizer(self):
        """Initialize optimizer.
        """
        self.optim_cfg = self.train_cfg["optim"].copy()
        lr_init = self.optim_cfg["lr"]
        train_params = self.renderer.get_train_params()
        grad_vars = []
        for key in train_params.keys():
            grad_vars += train_params[key]
        self.optimizer = {
            "optimizer": torch.optim.Adam(params=grad_vars, lr=lr_init),
        }
        self.lr_init = {
            "optimizer": lr_init,
        }

    def load_checkpoint(self):
        """Load checkpoint.
        """
        ckpt = torch.load(self.ckpt_dir)
        self.step_start = ckpt["n_iter"] + 1
        self.renderer.load_checkpoint(ckpt)
        for optim_key in self.optimizer.keys():
            self.optimizer[optim_key].load_state_dict(ckpt[optim_key])
        
    def save_checkpoint(self, global_step):
        """Save checkpoint.
        """
        ckpt = self.renderer.save_checkpoint()
        ckpt["n_iter"] = global_step
        for optim_key in self.optimizer.keys():
            ckpt[optim_key] = self.optimizer[optim_key].state_dict()
        torch.save(ckpt, self.ckpt_dir)
        
    def train_step(self, global_step):
        """
        Training step.
        """
        for optim_key in self.optimizer.keys():
            self.optimizer[optim_key].zero_grad()
        loss = self.compute_loss(global_step)
        loss.backward()
        for optim_key in self.optimizer.keys():
            self.optimizer[optim_key].step()
        return loss.item()
    
    def compute_loss(self, global_step):
        """
        Compute loss.
        """
        # Params
        color_loss_weight = self.train_cfg["color_loss_weight"]
        depth_loss_weight = self.train_cfg["depth_loss_weight"]
        sdf_loss_weight = self.train_cfg["sdf_loss_weight"]
        angle_loss_weight = self.train_cfg["angle_loss_weight"]
        eikonal_loss_weight = self.train_cfg["eikonal_loss_weight"]
        surf_neig_loss_weight = self.train_cfg["surf_neig_loss_weight"]
        surf_neig_rad = self.train_cfg["surf_neig_rad"]
        ray_batch = self.train_cfg["ray_batch"]
        mask_guided_ray_sampling = self.train_cfg["mask_guided_ray_sampling"]

        data = self.dset.get_train_batch_data_by_index(ray_batch=ray_batch, 
                                                       mask_guided_ray_sampling=mask_guided_ray_sampling)
        rays = data["rays"]
        color_gt = data["color"]
        depth_gt = data["depth"]
        mask_gt = data["mask"]
        color_mask_gt = data["color_mask"]

        # Render
        ret = self.renderer(rays, iter_step=global_step)
        
        # Color loss
        color_pred = ret["color_map"]
        color_error = (color_pred - color_gt) * color_mask_gt
        color_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction="sum") / (color_mask_gt.sum() + 1e-10)
        color_psnr = cal_psnr(color_pred, color_gt, color_mask_gt)
        
        # SDF and angle loss
        sdf_loss, angle_loss, valid_depth_region =\
                    self.renderer.errorondepth(rays, 
                                            d_gt=depth_gt,
                                            mask=mask_gt,
                                            iter_step=global_step)
        
        # Depth loss
        depth_pred = ret["depth_map"]
        depth_error = (depth_pred - depth_gt) * valid_depth_region * mask_gt
        depth_loss = F.l1_loss(depth_error, torch.zeros_like(depth_error), reduction="sum") \
                                / ((valid_depth_region * mask_gt).sum() + 1e-10)
        
        # Eikonal loss
        eikonal_loss = ret["gradient_o_error"]
        
        # Surface neighbour loss
        surf_neig_loss = self.renderer.surface_neighbour_error(rays=rays, mask=mask_gt, iter_step=global_step, neighbour_rad=surf_neig_rad)
        
        loss = color_loss * color_loss_weight + \
            depth_loss * depth_loss_weight + \
            sdf_loss * sdf_loss_weight + \
            angle_loss * angle_loss_weight + \
            eikonal_loss * eikonal_loss_weight + \
            surf_neig_loss_weight * surf_neig_loss
            
        # Logging
        self.writer.add_scalar("train/loss_color", color_loss, global_step)
        self.writer.add_scalar("train/psnr_color", color_psnr, global_step)
        self.writer.add_scalar("train/loss_sdf", sdf_loss, global_step)
        self.writer.add_scalar("train/loss_angle", angle_loss, global_step)
        self.writer.add_scalar("train/loss_depth", depth_loss, global_step)
        self.writer.add_scalar("train/loss_eikonal", eikonal_loss, global_step)
        self.writer.add_scalar("train/loss_surf_neig", surf_neig_loss, global_step)
        self.writer.add_scalar("train/loss_total", loss, global_step)
        
        s_val = ret["s_val"]
        cdf_fine = ret["cdf"]
        weight_max = ret["weight_max"]
        self.writer.add_scalar("train/s_val", s_val.mean(), global_step)
        self.writer.add_scalar("train/cdf", (cdf_fine[:, :1] * mask_gt).sum() / (mask_gt.sum() + 1e-10), global_step)
        self.writer.add_scalar("train/weight_max", (weight_max * mask_gt).sum() / (mask_gt.sum() + 1e-10), global_step)
        
        return loss
    
    def update_learning_rate(self, global_step):
        """Update learning rate.
        """
        warm_up_end = self.optim_cfg["warm_up_end"]
        learning_rate_alpha = self.optim_cfg["lr_alpha"]
        
        for optim_key in self.optimizer.keys():
            optimizer = self.optimizer[optim_key]
            lr_init = self.lr_init[optim_key]
            
            if global_step < warm_up_end:
                learning_factor = global_step / warm_up_end
            else:
                alpha = learning_rate_alpha
                progress = (global_step - warm_up_end) / (self.n_iter - warm_up_end)
                learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
            
            current_learning_rate = lr_init * learning_factor
            for g in optimizer.param_groups:
                g["lr"] = current_learning_rate
            self.writer.add_scalar(f"train/lr_{optim_key}", current_learning_rate, global_step)        

    def eval(self, global_step):
        """Evaluation during training.
        """
        # Params
        ray_chunk = self.train_cfg["eval"]["ray_chunk"]
        depth_scale = self.dset.depth_scale
        list_test = self.dset.list_test
        list_test = list_test[:1]

        data = self.dset.get_frame_data_by_index(list_test)
        rays_in = data["rays"]
        rgb_gt = data["color"]
        depth_gt = data["depth"]
        mask_gt = data["mask"]
        color_mask_gt = data["color_mask"]

        n_frames, H, W, ray_dim = rays_in.shape
        rays = rays_in.reshape(-1, ray_dim).split(ray_chunk)
        
        out_rgb_vr = []  # Volume rendering
        out_depth_vr = []
        out_normal_vr = []
        important_begin_iter = self.render_cfg["important_begin_iter"]
        for rays_split in tqdm(rays, desc="EVAL|Render", leave=False):
            # Render with volume rendering
            render_out = self.renderer(rays_split, iter_step=global_step)      
            if global_step >= important_begin_iter:
                n_samples = self.renderer.n_samples + self.renderer.n_importance
            else:
                n_samples = self.renderer.n_samples
            normals = render_out["gradients_o"] * render_out["weights"][:, :n_samples, None]
            normals = normals.sum(dim=1)
            out_rgb_vr.append(tensor2array(render_out["color_map"]))
            out_depth_vr.append(tensor2array(render_out["depth_map"]))
            out_normal_vr.append(tensor2array(normals))
            del render_out
        
        depth_max = self.dset.far
        pose = self.dset.poses[list_test]
        out_rgb_vr, out_rgb_vr_show = gen_rgb(out_rgb_vr, n_frames, W, H)
        out_depth_vr, out_depth_vr_show = gen_depth(out_depth_vr, n_frames, W, H, depth_max)
        out_normal_vr, out_normal_vr_show = gen_normal(out_normal_vr, pose, n_frames, W, H)
        # out_normal_vr, out_normal_vr_show = gen_normal_from_depth(rays_in, out_depth_vr, self.device)

        # Show
        eval_save_dir = osp.join(self.exp_dir, "eval", f"iter_{global_step:08d}")
        os.makedirs(eval_save_dir, exist_ok=True)
        id_show = np.random.randint(n_frames)
        _, depth_gt_show = gen_depth(depth_gt, n_frames, W, H, depth_max)
        for i_frame in trange(n_frames, desc="EVAL|Save results", leave=False):
            out_show = np.hstack([add_text_to_img(rgb_gt[i_frame], "rgb_gt"),
                                  add_text_to_img(out_rgb_vr_show[i_frame], "rgb_pred"),
                                  add_text_to_img(depth_gt_show[i_frame], "depth_gt"),
                                  add_text_to_img(out_depth_vr_show[i_frame], "depth_pred"),
                                  add_text_to_img(out_normal_vr_show[i_frame], "normal_pred")])
            iio.imwrite(osp.join(eval_save_dir, f"eval_{i_frame:03d}.png"), out_show)
            if i_frame == id_show:
                self.writer.add_rgb("eval/results", out_show, global_step)
        
        # Logging
        psnr_rgb_vr = cal_psnr(rgb_gt, out_rgb_vr, color_mask_gt)
        ssim_rgb_vr = cal_ssim(rgb_gt, out_rgb_vr, color_mask_gt, self.device)
        lpips_rgb_vr = cal_lpips(rgb_gt, out_rgb_vr, color_mask_gt, self.device)
        rmse_d_vr = cal_rmse(depth_gt*depth_scale, out_depth_vr*depth_scale, mask_gt)
        self.writer.add_scalar("eval/psnr_rgb_vr", psnr_rgb_vr, global_step)
        self.writer.add_scalar("eval/ssim_rgb_vr", ssim_rgb_vr, global_step)
        self.writer.add_scalar("eval/lpips_rgb_vr", lpips_rgb_vr, global_step)
        self.writer.add_scalar("eval/rmse_d_vr", rmse_d_vr, global_step)
        stats_out = {
            "psnr_rgb_vr": psnr_rgb_vr,
            "ssim_rgb_vr": ssim_rgb_vr,
            "lpips_rgb_vr": lpips_rgb_vr,
            "rmse_d_vr": rmse_d_vr,
        }
        with open(osp.join(eval_save_dir, "stats_out.txt"), "w") as f: 
            for key, value in stats_out.items(): 
                f.write("%s: %f\n" % (key, value))
                
        tqdm.write(f"EVAL|iter:{global_step}/{self.n_iter}|Complete!")
    
    def demo(self, global_step, test_mode=False, visualize=True, demo_2d=True, demo_3d=True):
        """Demo.
        """
        # Params
        ray_batch_size = self.cfg["demo"]["ray_batch"]
        net_chunk = self.cfg["demo"]["net_chunk"]
        fps = self.cfg["demo"]["fps"]
        depth_scale = self.dset.depth_scale
        depth_max = self.dset.far
        vcam = self.dset.vcam
        render_option = self.dset.render_option
        if demo_2d:
            tqdm.write(f"DEMO|Render RGBD images")
        if demo_3d:
            mesh_resolution = self.cfg["demo"]["marching_cubes_resolution"]
            thresh = self.cfg["demo"]["marching_cubes_thresh"]
            render_view_point = self.dset.poses[:,:3,3].mean(0)
            tqdm.write(f"DEMO|Eextract meshes wtih resoltuion {mesh_resolution} and threshold {thresh}")

        if test_mode:
            list_all = self.dset.list_test
            tqdm.write(f"DEMO|Use testset with {len(list_all)} frames")
        else:
            list_all = list(np.arange(self.dset.n_frames))
            tqdm.write(f"DEMO|Use all data with {len(list_all)} frames")
        # list_all = list_all[:1]
        data = self.dset.get_frame_data_by_index(list_all)
        rays_in = data["rays"]
        rgb_gt = data["color"]
        depth_gt = data["depth"]
        mask_gt = data["mask"]
        color_mask_gt = data["color_mask"]
        Ks = self.dset.intrinsics[list_all]
        poses = self.dset.poses[list_all]

        n_frames, H, W, ray_dim = rays_in.shape
        rays = rays_in.reshape(-1, ray_dim).split(ray_batch_size)
        
        # View synthesis
        if demo_2d:
            demo_2d_save_dir = osp.join(
                self.exp_dir, "demo", f"iter_{global_step:08d}",
                f"{'test' if test_mode else 'all'}_2d")
            os.makedirs(demo_2d_save_dir, exist_ok=True)

            out_rgb_vr = []  # Volume rendering
            out_normal_vr = []
            out_depth_vr = []
            important_begin_iter = self.render_cfg["important_begin_iter"]
            for rays_split in tqdm(rays, desc="DEMO|Render 2D images", leave=False):
                # Render with volume rendering
                render_out = self.renderer(rays_split, iter_step=global_step)      
                out_rgb_vr.append(tensor2array(render_out["color_map"]))
                if global_step >= important_begin_iter:
                    n_samples = self.renderer.n_samples + self.renderer.n_importance
                else:
                    n_samples = self.renderer.n_samples
                normals = render_out["gradients_o"] * render_out["weights"][:, :n_samples, None]
                normals = normals.sum(dim=1)
                out_normal_vr.append(tensor2array(normals))
                out_depth_vr.append(tensor2array(render_out["depth_map"]))
                del render_out
            
            out_rgb_vr, out_rgb_vr_show = gen_rgb(out_rgb_vr, n_frames, W, H)
            out_depth_vr, out_depth_vr_show = gen_depth(out_depth_vr, n_frames, W, H, depth_max)
            out_normal_vr, out_normal_vr_show = gen_normal(out_normal_vr, poses, n_frames, W, H)
            # out_normal_vr, out_normal_vr_show = gen_normal_from_depth(rays_in, out_depth_vr, self.device)
            
            # Logs
            psnr_rgb_vr = cal_psnr(rgb_gt, out_rgb_vr, color_mask_gt)
            ssim_rgb_vr = cal_ssim(rgb_gt, out_rgb_vr, color_mask_gt, self.device)
            lpips_rgb_vr = cal_lpips(rgb_gt, out_rgb_vr, color_mask_gt, self.device)
            rmse_d_vr = cal_rmse(depth_gt*depth_scale, out_depth_vr*depth_scale, mask_gt)
            stats_out = {
                "psnr_rgb_vr": psnr_rgb_vr,
                "ssim_rgb_vr": ssim_rgb_vr,
                "lpips_rgb_vr": lpips_rgb_vr,
                "rmse_d_vr": rmse_d_vr,
            }
            with open(osp.join(demo_2d_save_dir, "stats_out.txt"), "w") as f: 
                for key, value in stats_out.items(): 
                    f.write("%s: %f\n" % (key, value))
            tqdm.write("DEMO|".join([f"{key}: {stats_out[key]}\n" for key in stats_out.keys()]))

            if visualize:
                # Show
                _, depth_gt_show = gen_depth(depth_gt, n_frames, W, H, depth_max)
                imgs_show = [] 
                for i_frame in trange(n_frames, desc="DEMO|Save 2D results", leave=False):
                    out_show = np.hstack([add_text_to_img(rgb_gt[i_frame], "rgb_gt"),
                                  add_text_to_img(out_rgb_vr_show[i_frame], "rgb_pred"),
                                  add_text_to_img(depth_gt_show[i_frame], "depth_gt"),
                                  add_text_to_img(out_depth_vr_show[i_frame], "depth_pred"),
                                  add_text_to_img(out_normal_vr_show[i_frame], "normal_pred")])
                    if out_show.shape[1]> MAX_WINDOW_WIDTH:
                        out_show = cv2.resize(out_show, (MAX_WINDOW_WIDTH, int(MAX_WINDOW_WIDTH * out_show.shape[0]/out_show.shape[1])))
                    imgs_show.append(out_show)
                    iio.imwrite(osp.join(demo_2d_save_dir, f"{i_frame:03d}_all.png"), out_show)
                    iio.imwrite(osp.join(demo_2d_save_dir, f"{i_frame:03d}_rgb_gt.png"), to8b(rgb_gt[i_frame]))
                    iio.imwrite(osp.join(demo_2d_save_dir, f"{i_frame:03d}_rgb_vr.png"), out_rgb_vr_show[i_frame])
                    iio.imwrite(osp.join(demo_2d_save_dir, f"{i_frame:03d}_depth_gt.png"), depth_gt_show[i_frame])
                    iio.imwrite(osp.join(demo_2d_save_dir, f"{i_frame:03d}_depth_vr.png"), out_depth_vr_show[i_frame])
                    iio.imwrite(osp.join(demo_2d_save_dir, f"{i_frame:03d}_normal_vr.png"), out_normal_vr_show[i_frame])

                # Generate video
                tqdm.write("DEMO|Generate rendering video...")
                videowriter = cv2.VideoWriter(osp.join(demo_2d_save_dir, "demo.mp4"),
                                            cv2.VideoWriter_fourcc(*"mp4v"),
                                            fps, [out_show.shape[1], out_show.shape[0]])
                for frame in imgs_show:
                    videowriter.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                videowriter.release()
                
                # Generate gif
                tqdm.write("DEMO|Generate rendering gif...")
                with iio.get_writer(osp.join(demo_2d_save_dir, "demo.gif"),mode="I", duration=1/fps) as writer:
                    for frame in imgs_show:
                        writer.append_data(frame)        

        # Marching cubes
        if demo_3d:
            demo_3d_save_dir = osp.join(
                self.exp_dir, "demo", f"iter_{global_step:08d}",
                f"{'test' if test_mode else 'all'}_3d_thresh_{thresh}_res_{mesh_resolution}")
            os.makedirs(demo_3d_save_dir, exist_ok=True)
            
            pcds_gt = []
            meshes_geometry = []
            meshes_with_color = []
            meshes_with_normal = []
            geo_errs = []
            for i_frame in trange(n_frames, desc="DEMO|Extract 3D meshes", leave=False):
                # Get frame pcd from rgbd
                pcd_gt = gen_pcd(to8b(rgb_gt[i_frame]), depth_gt[i_frame], Ks[i_frame],
                                np.linalg.inv(tensor2array(poses[i_frame])), 1, depth_max)
                pcds_gt.append(pcd_gt)

                # Marching cube
                bound_min = torch.tensor(self.dset.bbox_minmax[i_frame, :,0], dtype=self.dtype, device=self.device) * 1.2  # Make bounding box slightly bigger
                bound_max = torch.tensor(self.dset.bbox_minmax[i_frame, :,1], dtype=self.dtype, device=self.device) * 1.2
                t = torch.unique(rays_in[i_frame,...,-1])
                assert len(t) == 1
                vertices, triangles = self.renderer.extract_observation_geometry(
                    t,
                    bound_min,
                    bound_max,
                    resolution=mesh_resolution,
                    threshold=thresh,
                    net_chunk=net_chunk)
                assert len(vertices) != 0, "Failed to find surface! Please tune threshold."
                mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                                o3d.utility.Vector3iVector(triangles))
                mesh = mesh.remove_degenerate_triangles()
                mesh = mesh.remove_duplicated_triangles()

                # Remove unconnected components
                triangle_clusters, cluster_n_triangles, cluster_area = (
                    mesh.cluster_connected_triangles())
                triangle_clusters = np.asarray(triangle_clusters)
                cluster_n_triangles = np.asarray(cluster_n_triangles)
                triangles_to_remove = cluster_n_triangles[triangle_clusters] < cluster_n_triangles[triangle_clusters].max()*0.9
                mesh.remove_triangles_by_mask(triangles_to_remove)
                del vertices, triangles

                # Generate color
                vert = array2tensor(np.asarray(mesh.vertices), self.device)
                vert_dir = vert - render_view_point[None,:]
                vert_dir = vert_dir / torch.linalg.norm(vert_dir, ord=2, dim=-1, keepdim=True) 
                vert_color, _ = self.renderer.renderonpts(vert, vert_dir, t, net_chunk)
                vert_color = tensor2array(vert_color).clip(0., 1.)

                # Colorize mesh
                mesh_with_color = copy.deepcopy(mesh)
                mesh_with_color.vertex_colors = o3d.utility.Vector3dVector(vert_color)
                mesh_with_normal = copy.deepcopy(mesh)
                mesh_with_normal.compute_vertex_normals()
                vert_normal_c = (-np.asarray(mesh_with_normal.vertex_normals) * 0.5 + 0.5).clip(0., 1.)
                mesh_with_normal.vertex_colors = o3d.utility.Vector3dVector(vert_normal_c)
                o3d.io.write_triangle_mesh(os.path.join(demo_3d_save_dir, f"{i_frame:03d}_geometry.ply"), mesh)
                o3d.io.write_triangle_mesh(os.path.join(demo_3d_save_dir, f"{i_frame:03d}_color.ply"), mesh_with_color)
                o3d.io.write_triangle_mesh(os.path.join(demo_3d_save_dir, f"{i_frame:03d}_normal.ply"), mesh_with_normal)
                o3d.io.write_point_cloud(os.path.join(demo_3d_save_dir, f"{i_frame:03d}_gt.ply"), pcd_gt)
                meshes_geometry.append(mesh)
                meshes_with_color.append(mesh_with_color)
                meshes_with_normal.append(mesh_with_normal)

                # Computer geometric error
                geo_err = np.mean(pcd_gt.compute_point_cloud_distance(o3d.geometry.PointCloud(points=mesh.vertices))) * depth_scale
                geo_errs.append(geo_err)
            
            # Save metrics
            geo_err_mean = np.mean(geo_errs)
            with open(osp.join(demo_3d_save_dir, "stats_out.txt"), "w") as f: 
                f.write("mean: %f\n" % geo_err_mean)
                for key, value in enumerate(geo_errs): 
                    f.write("%s: %f\n" % (key, value))
            tqdm.write(f"DEMO|Geometric error: {geo_err_mean}")

            if visualize:
                meshes_show = {}
                for meshes, mesh_type in zip([pcds_gt, meshes_geometry, meshes_with_color, meshes_with_normal], ["reference", "geometry", "color", "normal"]):
                    meshes_show[mesh_type] = []
                    vis = o3d.visualization.Visualizer()
                    vis.create_window(width=H, height=H)
                    for i_frame in trange(n_frames, desc=f"DEMO|Visualize 3D {mesh_type}", leave=False):
                        if mesh_type == "reference":
                            mesh_show = vis_pcd(vis, meshes[i_frame], vcam, render_option)
                        else:
                            mesh_show = vis_mesh(vis, meshes[i_frame], vcam, render_option)
                        iio.imwrite(osp.join(demo_3d_save_dir, f"{i_frame:03d}_{mesh_type}.png"), mesh_show)
                        meshes_show[mesh_type].append(mesh_show)
                    vis.close()

                imgs_show = []
                for i_frame in range(n_frames):
                    img_show = np.hstack([add_text_to_img( meshes_show[i][i_frame], i) for i in meshes_show.keys()])
                    imgs_show.append(img_show)

                # Generate video
                tqdm.write("DEMO|Generate mesh video...")
                videowriter = cv2.VideoWriter(osp.join(demo_3d_save_dir, f"demo.mp4"),
                                            cv2.VideoWriter_fourcc(*"mp4v"),
                                            fps, [img_show.shape[1], img_show.shape[0]])
                for frame in imgs_show:
                    videowriter.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                videowriter.release()
                
                # Generate gif
                tqdm.write("DEMO|Generate mesh gif...")
                with iio.get_writer(osp.join(demo_3d_save_dir, f"demo.gif"),mode="I", duration=1/fps) as writer:
                    for frame in imgs_show:
                        writer.append_data(frame)
        
        # Show all
        if demo_3d and demo_2d and visualize:
            demo_final_save_dir = osp.join(
                self.exp_dir, "demo", f"iter_{global_step:08d}",
                f"{'test' if test_mode else 'all'}_final")
            os.makedirs(demo_final_save_dir, exist_ok=True)

            imgs_show = []
            for i_frame in trange(n_frames, desc="DEMO|Generate final demo", leave=False):
                img_show = np.hstack([
                    add_text_to_img(rgb_gt[i_frame], "Reference"),
                    add_text_to_img(out_rgb_vr_show[i_frame], "RGB"),
                    add_text_to_img(out_depth_vr_show[i_frame], "Depth"),
                    add_text_to_img(out_normal_vr_show[i_frame], "Normal"),
                    add_text_to_img(meshes_show["geometry"][i_frame], "Mesh"),
                    add_text_to_img(meshes_show["color"][i_frame], "Texture"),
                    add_text_to_img(meshes_show["normal"][i_frame], "Normal"),
                ])
                iio.imwrite(osp.join(demo_final_save_dir, f"{i_frame:03d}.png"), img_show)
                imgs_show.append(img_show)
            
            # Generate video
            tqdm.write("DEMO|Generate final video...")
            videowriter = cv2.VideoWriter(osp.join(demo_final_save_dir, f"demo.mp4"),
                                        cv2.VideoWriter_fourcc(*"mp4v"),
                                        fps, [img_show.shape[1], img_show.shape[0]])
            for frame in imgs_show:
                videowriter.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            videowriter.release()
            
            # Generate gif
            tqdm.write("DEMO|Generate final gif...")
            with iio.get_writer(osp.join(demo_final_save_dir, f"demo.gif"),mode="I", duration=1/fps) as writer:
                for frame in imgs_show:
                    writer.append_data(frame)

        tqdm.write("DEMO|Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/endosurf/baseline/base_cut.yml",
                        type=str, help="config file path")
    parser.add_argument("--mode", default="test_3d", type=str,
                        help="mode for train/test/test_2d/test_3d/demo/demo_2d/demo_3d")
    args = parser.parse_args()
    
    torch.set_default_dtype(torch.float32)
    
    mode = args.mode
    trainer = EndoSurfTrainer(args.cfg, mode)
    if mode == "train":
        trainer.start()
    else:
        trainer.renderer.eval()
        with torch.no_grad():
            if mode == "test":
                trainer.demo(trainer.step_start-1, test_mode=True, visualize=True, demo_2d=True, demo_3d=True)
            elif mode == "test_2d":
                trainer.demo(trainer.step_start-1, test_mode=True, visualize=True, demo_2d=True, demo_3d=False)
            elif mode == "test_3d":
                trainer.demo(trainer.step_start-1, test_mode=True, visualize=True, demo_2d=False, demo_3d=True)
            elif mode == "demo":
                trainer.demo(trainer.step_start-1, test_mode=False, visualize=True, demo_2d=True, demo_3d=True)
            elif mode == "demo_2d":
                trainer.demo(trainer.step_start-1, test_mode=False, visualize=True, demo_2d=True, demo_3d=False)
            elif mode == "demo_3d":
                trainer.demo(trainer.step_start-1, test_mode=False, visualize=True, demo_2d=False, demo_3d=True)
            else:
                raise NotImplementedError("Unknown mode!")
        