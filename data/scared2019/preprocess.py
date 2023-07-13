import os
import os.path as osp
import numpy as np
import imageio.v2 as iio
from tqdm import tqdm
import pickle
import open3d as o3d
import cv2
import argparse
import json
from tqdm import trange
import imageio.v2 as iio
import cv2
import matplotlib.pyplot as plt
import yaml
import random


random.seed(0)
np.random.seed(0)


def create_scared_info(dset_dir,
                       info_dir,
                       scale_factor=1,
                       object_scale_in_sphere=0.6,
                       skip_every=2, 
                       test_every=8, 
                       disp_type="disparity_pred", 
                       show=False):
    """Create a pickle file to store everything about the dataset.
    Note: we use left images only.
    """
    pad = np.array([0, 0, 0])  #mm
    depth_far_thresh = 300.
    depth_near_thresh = 30.
    crop_width = 100
    
    scene_name = osp.basename(dset_dir)
    
    calibs_dir = osp.join(dset_dir, "data", "frame_data")
    rgbs_dir = osp.join(dset_dir, "data", "left")
    disps_dir = osp.join(dset_dir, "data", disp_type)
    reproj_dir = osp.join(dset_dir, "data", "reprojection_data")
    frame_ids = sorted([id[:-5] for id in os.listdir(calibs_dir)])
    frame_ids = frame_ids[::skip_every]
    n_frames = len(frame_ids)
    # n_frames = 10
    
    disp_save_dir =  osp.join(dset_dir, "data_processed", f"{disp_type}_scale_{scale_factor}")
    rgb_save_dir = osp.join(dset_dir, "data_processed", f"rgb_scale_{scale_factor}")
    mask_save_dir = osp.join(dset_dir, "data_processed", f"mask_scale_{scale_factor}")
    os.makedirs(disp_save_dir, exist_ok=True)
    os.makedirs(rgb_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)
    
    rgbs = []
    disps = []
    depths = []
    world_mat = []
    camera_mat = []
    pose_mat = []
    bds = []
    rgbs_dir_list = []
    masks_dir_list = []
    disps_dir_list = []
    disp_consts = []
    for i_frame in trange(n_frames, desc="Process frames"):
        frame_id = frame_ids[i_frame]
        
        # Read intrinsics and poses
        with open(osp.join(calibs_dir, f"{frame_id}.json"), "r") as f:
            calib_dict = json.load(f)
        K = np.eye(4)
        K[:3, :3] = np.array(calib_dict["camera-calibration"]["KL"])
        if scale_factor != 1:
            K = np.diag([1/scale_factor, 1/scale_factor, 1, 1]) @ K
        c2w = np.linalg.inv(np.array(calib_dict["camera-pose"]))
        if i_frame == 0:
            c2w0 = c2w
        c2w = np.linalg.inv(c2w0) @ c2w
        w2c = np.linalg.inv(c2w)
        P = K @ w2c
        world_mat.append(P)
        camera_mat.append(K)
        pose_mat.append(c2w)   
        
        # Read images
        rgb_dir = osp.join(rgbs_dir, f"{frame_id}.png")
        disp_dir = osp.join(disps_dir, f"{frame_id}.tiff")
        rgb = iio.imread(rgb_dir)
        disp = iio.imread(disp_dir).astype(np.float32)  
        h, w = disp.shape
        if scale_factor != 1:
            w, h = int(w/scale_factor), int(h/scale_factor)
            rgb = cv2.resize(rgb, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            disp = cv2.resize(disp, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
        with open(osp.join(reproj_dir, f"{frame_id}.json"), "r") as json_file:
            Q = np.array(json.load(json_file)["reprojection-matrix"])
        fl = Q[2,3]
        bl =  1 / Q[3,2]
        disp_const = fl * bl
        mask_valid = disp != 0        
        depth = np.zeros_like(disp)
        depth[mask_valid] = disp_const / disp[mask_valid]
        depth[depth>depth_far_thresh] = 0
        depth[depth<depth_near_thresh] = 0
        
        # Mask
        depth_mask = (depth != 0).astype(float)
        disp[depth_mask==0] == 0.
        kernel = np.ones((int(w/128), int(w/128)),np.uint8)
        color_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, kernel)
        if disp_type == "disparity_pred":
            mask_crop = np.ones_like(disp)
            mask_crop[crop_width:-crop_width, crop_width:-crop_width] = 0
            depth[mask_crop == 1] = 0

        bound = np.array([depth[depth!=0].min(), depth[depth!=0].max()])

        rgb_save_path = osp.join(rgb_save_dir, f"{frame_id}.png")
        disp_save_path = osp.join(disp_save_dir, f"{frame_id}.tiff")
        mask_save_path = osp.join(mask_save_dir, f"{frame_id}.png")
        iio.imwrite(rgb_save_path, rgb)
        iio.imwrite(disp_save_path, disp)
        iio.imwrite(mask_save_path, (color_mask*255).astype(np.uint8))
        
        disp_consts.append(disp_const)
        rgbs.append(rgb)
        disps.append(disp)
        depths.append(depth)
        bds.append(bound)
        rgbs_dir_list.append(rgb_save_path)
        disps_dir_list.append(disp_save_path)
        masks_dir_list.append(mask_save_path)
    
    world_mat = np.stack(world_mat, 0)
    camera_mat = np.stack(camera_mat, 0)
    pose_mat = np.stack(pose_mat, 0)
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    depths = np.stack(depths, 0)
    bds = np.stack(bds, 0)
    inf_depth = bds.max()
    
    # Generate point cloud
    bboxes = []
    pcds = o3d.geometry.PointCloud()
    for i_frame in trange(n_frames, desc="Generate point clouds"):
        pose = pose_mat[i_frame]
        K = camera_mat[i_frame][:3,:3]
        rgb_im = o3d.geometry.Image(rgbs[i_frame])
        depth_im = o3d.geometry.Image(depths[i_frame])
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_im, depth_im,
                                                                        depth_scale=1.,
                                                                        depth_trunc=inf_depth,
                                                                        convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, 
            o3d.camera.PinholeCameraIntrinsic(w, h, K),
            np.linalg.inv(pose),
            project_valid_depth_only=True,
        )

        pcd = pcd.random_down_sample(0.1)
        pcd, _ = pcd.remove_radius_outlier(nb_points=5,
                                     radius=np.asarray(pcd.compute_nearest_neighbor_distance()).mean() * 10.)
        bbox = pcd.get_axis_aligned_bounding_box()
        bboxes.append(bbox)
        pcds = pcds + pcd

    # Generate scale matrix based on point cloud
    print("Compute scale matrix...")
    pts = np.asarray(pcds.points)
    bbox_all = pcds.get_axis_aligned_bounding_box()
    bbox_center = bbox_all.get_center()
    
    radius = np.linalg.norm(pts - bbox_center, ord=2, axis=-1).max() / object_scale_in_sphere
    scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
    scale_mat[:3, 3] = bbox_center
    
    bbox_all.scale(1/radius, np.zeros((3,1)))
    bbox_all.translate(-bbox_center[:, None]/radius)
    
    pad_norm = pad / radius
    bboxes_new = []
    bboxes_minmax = []
    for bbox in bboxes:
        bbox.scale(1/radius, np.zeros((3,1)))
        bbox.translate(-bbox_center[:, None]/radius)
        bbox_minmax = np.stack([bbox.get_min_bound()-pad_norm,
                                 bbox.get_max_bound()+pad_norm], -1)
        bboxes_new.append(o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox_minmax[:,:1], max_bound=bbox_minmax[:,1:]))
        bboxes_minmax.append(bbox_minmax)
    bboxes_minmax = np.stack(bboxes_minmax, 0)
    
    # Split train and test
    list_train = [i for i in range(n_frames) if (i-1) % test_every != 0]
    list_test = [i for i in range(n_frames) if (i-1) % test_every == 0]
    
    # Generate info
    info = {
        "dset_name": "scared2019",
        "scene_name": f"{scene_name}_{disp_type}",
        "world_mat": world_mat,
        "camera_mat": camera_mat,
        "pose_mat": pose_mat,
        "wh": [w, h],
        "n_frames": n_frames,
        "color": rgbs_dir_list,
        "depth": disps_dir_list,
        "depth_type": "disp",
        "disp_const": disp_consts,
        "mask": masks_dir_list,
        "scale_mat": scale_mat,
        "bounds": bds,
        "list_train": list_train,
        "list_test": list_test,
        "bbox_minmax": bboxes_minmax,
        "mask_type":  "mask",
        "depth_norm_scale": radius,
    }
    
    info_dir = osp.join(info_dir, f"{scene_name}_{disp_type}.pkl")
    os.makedirs(osp.dirname(info_dir), exist_ok=True)
    with open(info_dir, "wb") as handle:
        pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"info data saved in {info_dir}!")
        
    if show:
        # Show
        cameras = []
        bboxes_o3d = []
        for i_frame in trange(n_frames, desc="show"):
            P = world_mat[i_frame] @ scale_mat
            P = P[:3, :4]
            K, pose = load_K_Rt_from_P(None, P)
            K = K[:3, :3]
            # camera
            camera = o3d.geometry.LineSet.create_camera_visualization(w, h, K, np.linalg.inv(pose), scale=1.0)
            bbox_o3d = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bboxes_new[i_frame])
            if i_frame == 0:
                camera.paint_uniform_color(np.array([[1.],[0.],[0.]]))
                bbox_o3d.paint_uniform_color(np.array([[1.], [0.], [0.]]))
            elif i_frame == n_frames - 1:
                camera.paint_uniform_color(np.array([[0.],[1.],[0.]]))
                bbox_o3d.paint_uniform_color(np.array([[0.], [1.], [0.]]))
            else:
                camera.paint_uniform_color(np.array([[0.],[0.],[1.]]))
                bbox_o3d.paint_uniform_color(np.array([[0.], [0.], [1.]]))
            cameras.append(camera)
            bboxes_o3d.append(bbox_o3d)
        pcds.scale(1/radius, np.zeros((3,1)))
        pcds.translate(-bbox_center[:, None]/radius)
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0.0, 0.0, 0.0]))
        sphere = o3d.geometry.LineSet.create_from_triangle_mesh(o3d.geometry.TriangleMesh.create_sphere(radius=1.0))
        o3d.visualization.draw_geometries([pcds, coord, sphere, bbox_o3d]+cameras+bboxes_o3d)


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
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

        
def to8b(x):
    return (255.*np.clip(x,0,1)).astype(np.uint8)


def gen_pcd(rgb, depth, K, pose, depth_scale=1., depth_truc=3., depth_filter=None):
    """Generate point cloud.
    """
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
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_dir", default="data/scared2019/dataset/dataset_1_keyframe_1",
                        type=str, help="dataset path")
    parser.add_argument("--info_dir", default="data/data_info/scared2019/",
                        type=str, help="Output information pickle file path")
    parser.add_argument("--scale_factor", default=1,
                        type=int, help="Scale images.")
    parser.add_argument("--disp_type", default="disparity",
                        type=str, help="Which disparity to use (disparity or disparity_pred)")
    parser.add_argument("--object_scale_in_sphere", default=0.8,
                        type=float, help="Make the object slightly smaller than the unit sphere")
    parser.add_argument("--skip_every", default=2,
                        type=int, help="Skip every N image to save space.")
    parser.add_argument("--test_every", default=8,
                        type=int, help="Treat every N image as test")
    parser.add_argument("--show", default=False, action="store_true", help="If show results (monitor required)")
    args = parser.parse_args()
    
    create_scared_info(
        dset_dir=args.dset_dir,
        info_dir=args.info_dir,
        scale_factor=args.scale_factor,
        object_scale_in_sphere=args.object_scale_in_sphere,
        skip_every=args.skip_every,
        test_every=args.test_every,
        disp_type=args.disp_type,
        show=args.show)
    

if __name__ == "__main__":
    main()