# Download trained checkpoints
You can download checkpoints of all experiments [here](https://drive.google.com/drive/folders/135F5Y4O6svitZvtIEsNCuyNC2V7EuhKa?usp=sharing). Then merge the downloaded `logs` file with the existing one.

The overall log directory should be organized in the following structure.

``` sh
├── logs
│   ├── README.md
│   ├── endonerf
│   │   ├── base-endonerf-cutting_tissues_twice
│   │   │   ├── ckpt.tar
│   │   ├── base-endonerf-pulling_soft_tissues
│   │   │   ├── ckpt.tar
│   │   ├── base-scared2019-dataset_1_keyframe_1_disparity
│   │   │   ├── ckpt.tar
│   │   ├── base-scared2019-dataset_2_keyframe_1_disparity
│   │   │   ├── ckpt.tar
│   │   ├── base-scared2019-dataset_3_keyframe_1_disparity
│   │   │   ├── ckpt.tar
│   │   ├── base-scared2019-dataset_6_keyframe_1_disparity
│   │   │   ├── ckpt.tar
│   │   ├── base-scared2019-dataset_7_keyframe_1_disparity
│   │   │   ├── ckpt.tar
│   ├── endosurf
│   │   ├── ablation_no_angle_loss-endonerf-pulling_soft_tissues
│   │   │   ├── ckpt.tar
│   │   ├── ablation_no_eikonal_loss-endonerf-pulling_soft_tissues
│   │   │   ├── ckpt.tar
│   │   ├── ablation_no_sdf_loss-endonerf-pulling_soft_tissues
│   │   │   ├── ckpt.tar
│   │   ├── ablation_no_surf_neig_loss-endonerf-pulling_soft_tissues
│   │   │   ├── ckpt.tar
│   │   ├── base-endonerf-cutting_tissues_twice
│   │   │   ├── ckpt.tar
│   │   ├── base-endonerf-pulling_soft_tissues
│   │   │   ├── ckpt.tar
│   │   ├── base-scared2019-dataset_1_keyframe_1_disparity
│   │   │   ├── ckpt.tar
│   │   ├── base-scared2019-dataset_2_keyframe_1_disparity
│   │   │   ├── ckpt.tar
│   │   ├── base-scared2019-dataset_3_keyframe_1_disparity
│   │   │   ├── ckpt.tar
│   │   ├── base-scared2019-dataset_6_keyframe_1_disparity
│   │   │   ├── ckpt.tar
│   │   ├── base-scared2019-dataset_7_keyframe_1_disparity
│   │   │   ├── ckpt.tar
```