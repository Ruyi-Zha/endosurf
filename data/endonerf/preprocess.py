import os
import os.path as osp
import numpy as np
import imageio.v2 as iio
from tqdm import tqdm
import pickle
import open3d as o3d
import cv2
import argparse
from tqdm import trange
import random


random.seed(0)
np.random.seed(0)


def create_endonerf_info(dset_dir, info_dir, test_every=8, object_scale_in_sphere=0.6, show=False):
    """Create a pickle file to store everything about the dataset.
    """
    pad = np.array([-5, -5, 10])  #mm
    scene_name = osp.basename(dset_dir)
    
    # Pose 3x5 [R|t|hwf]
    poses_arr = np.load(os.path.join(dset_dir, "poses_bounds.npy"))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])  # C2W
    bds = poses_arr[:, -2:].transpose([1,0])
    n_frames = poses.shape[-1]
    poses = poses.transpose((2,0,1))
    bds = bds.transpose((1,0))
    world_mat = []
    camera_mat = []
    pose_mat = []
    h, w = int(poses[0, 0, 4]), int(poses[0, 1, 4])
    for i_frame in range(n_frames):
        pose = poses[i_frame]
        c2w = np.vstack([pose[:, :4], np.array([[0,0,0,1]])])
        w2c = np.linalg.inv(c2w)
        h, w, f = int(pose[0, 4]), int(pose[1, 4]), pose[2, 4]
        K = np.array([[f, 0, (w-1)*0.5, 0], [0, f, (h-1)*0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        P = K @ w2c
        world_mat.append(P)
        camera_mat.append(K)
        pose_mat.append(c2w)   
    world_mat = np.stack(world_mat, 0)
    camera_mat = np.stack(camera_mat, 0)
    pose_mat = np.stack(pose_mat, 0)
    
    # Depth, color, mask
    check_fn = lambda f: f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    colors, colors_dir = load_imgs(osp.join(dset_dir, "images"), "color", n_frames, check_fn)
    depths, depths_dir = load_imgs(osp.join(dset_dir, "depth"), "depth", n_frames, check_fn)
    masks, masks_dir = load_imgs(osp.join(dset_dir, "masks"), "mask_invert", n_frames, check_fn)
    depths = depths
    depths[masks==0] = 0
    close_depth = np.percentile(depths[depths!=0], 3.0)
    inf_depth = np.percentile(depths[depths!=0], 99.9)
    depths[depths>inf_depth] = 0
    depths[np.bitwise_and(depths<close_depth, depths!=0)] = 0
    
    # Generate point cloud
    bboxes = []
    pcds = o3d.geometry.PointCloud()
    for i_frame in trange(n_frames, desc="Generate point clouds"):
        pose = pose_mat[i_frame]
        K = camera_mat[i_frame][:3,:3]
        rgb_im = o3d.geometry.Image(to8b(colors[i_frame]))
        depth_im = o3d.geometry.Image(depths[i_frame])
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_im, depth_im,
                                                                        depth_scale=1., depth_trunc=inf_depth,
                                                                        convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, 
            o3d.camera.PinholeCameraIntrinsic(w, h, K),
            np.linalg.inv(pose),
            project_valid_depth_only=True,
        )
        pcd = pcd.random_down_sample(0.005)
        pcd, _ = pcd.remove_radius_outlier(nb_points=5,
                                     radius=np.asarray(pcd.compute_nearest_neighbor_distance()).mean() * 20.)
        bbox = pcd.get_axis_aligned_bounding_box()
        bboxes.append(bbox)
    
        pcds = pcds + pcd

    pcds, _ = pcds.remove_radius_outlier(nb_points=5,
                                     radius=np.asarray(pcds.compute_nearest_neighbor_distance()).mean() * 20.)

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
        "dset_name": "endonerf",
        "scene_name": scene_name,
        "world_mat": world_mat,
        "camera_mat": camera_mat,
        "pose_mat": pose_mat,
        "wh": [w, h],
        "n_frames": n_frames,
        "color": colors_dir,
        "depth": depths_dir,
        "depth_type": "depth",
        "mask": masks_dir,
        "scale_mat": scale_mat,
        "bounds": bds,
        "list_train": list_train,
        "list_test": list_test,
        "bbox_minmax": bboxes_minmax,
        "mask_type": "mask_invert",
        "depth_norm_scale": radius,
    }

    info_dir = osp.join(info_dir, f"{scene_name}.pkl")
    os.makedirs(osp.dirname(info_dir), exist_ok=True)
    with open(info_dir, "wb") as handle:
        pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"info data saved in {info_dir}!")
        
    if show:
        # Show
        cameras = []
        bboxes_o3d = []
        for i_frame in range(n_frames):
            P = world_mat[i_frame] @ scale_mat
            P = P[:3, :4]
            K, pose = load_K_Rt_from_P(None, P)
            K = K[:3, :3]
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


def load_imgs(img_dir, img_type, num_imgs=None, check_fn=lambda x: True):
        """Load images (color, rgb, mask, mask_invert).
        """
        assert img_type in ["color", "depth", "mask", "mask_invert"]
        img_files = [osp.join(img_dir, f) for f in sorted(os.listdir(img_dir)) if check_fn(f)]
        if num_imgs is not None:
            assert len(img_files) == num_imgs, f"Mismatch between {img_type} and poses in {img_dir}."
        
        def imread(f):
            if f.endswith("png"):
                return iio.imread(f, ignoregamma=True)
            else:
                return iio.imread(f)
        
        imgs = []
        for img_file in img_files:
            if img_type == "color":
                img = imread(img_file)[...,:3].astype(np.float32) / 255.
            elif img_type == "depth":
                img = imread(img_file).astype(np.float32)
                img = img[..., None]
            elif img_type =="mask":
                img = imread(img_file).astype(np.float32) / 255.
                img = img[..., None]
            elif img_type =="mask_invert":
                img = 1.0 - imread(img_file).astype(np.float32) / 255.
                img = img[..., None]
            imgs.append(img)
        imgs = np.stack(imgs, axis=0)
        return imgs, img_files
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_dir", default="data/endonerf/dataset/pulling_soft_tissues",
                        type=str, help="dataset path")
    parser.add_argument("--info_dir", default="data/data_info/endonerf/",
                        type=str, help="Output information pickle file path")
    parser.add_argument("--test_every", default=8,
                        type=int, help="Treat every N image as test")
    parser.add_argument("--object_scale_in_sphere", default=0.8,
                        type=float, help="Make the object slightly smaller than the unit sphere")
    parser.add_argument("--show", default=False, action="store_true", help="If show results (monitor required)")
    args = parser.parse_args()
    
    create_endonerf_info(args.dset_dir, args.info_dir, args.test_every, args.object_scale_in_sphere, args.show)
    

if __name__ == "__main__":
    main()