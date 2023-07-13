# Train EndoSurf
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endosurf.py --cfg configs/endosurf/baseline/base_pull.yml --mode train
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endosurf.py --cfg configs/endosurf/baseline/base_cut.yml --mode train
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endosurf.py --cfg configs/endosurf/baseline/base_d1k1.yml --mode train
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endosurf.py --cfg configs/endosurf/baseline/base_d2k1.yml --mode train
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endosurf.py --cfg configs/endosurf/baseline/base_d3k1.yml --mode train
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endosurf.py --cfg configs/endosurf/baseline/base_d6k1.yml --mode train
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endosurf.py --cfg configs/endosurf/baseline/base_d7k1.yml --mode train

# Test EndoSurf
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endosurf.py --cfg configs/endosurf/baseline/base_pull.yml --mode test
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endosurf.py --cfg configs/endosurf/baseline/base_cut.yml --mode test
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endosurf.py --cfg configs/endosurf/baseline/base_d1k1.yml --mode test
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endosurf.py --cfg configs/endosurf/baseline/base_d2k1.yml --mode test
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endosurf.py --cfg configs/endosurf/baseline/base_d3k1.yml --mode test
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endosurf.py --cfg configs/endosurf/baseline/base_d6k1.yml --mode test
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endosurf.py --cfg configs/endosurf/baseline/base_d7k1.yml --mode test

# Demo EndoSurf
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endosurf.py --cfg configs/endosurf/baseline/base_pull.yml --mode demo
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endosurf.py --cfg configs/endosurf/baseline/base_cut.yml --mode demo
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endosurf.py --cfg configs/endosurf/baseline/base_d1k1.yml --mode demo
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endosurf.py --cfg configs/endosurf/baseline/base_d2k1.yml --mode demo
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endosurf.py --cfg configs/endosurf/baseline/base_d3k1.yml --mode demo
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endosurf.py --cfg configs/endosurf/baseline/base_d6k1.yml --mode demo
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endosurf.py --cfg configs/endosurf/baseline/base_d7k1.yml --mode demo

# Train baseline EndoNeRF
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endonerf.py --cfg configs/endonerf/baseline/base_pull.yml --mode train
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endonerf.py --cfg configs/endonerf/baseline/base_cut.yml --mode train
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endonerf.py --cfg configs/endonerf/baseline/base_d1k1.yml --mode train
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endonerf.py --cfg configs/endonerf/baseline/base_d2k1.yml --mode train
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endonerf.py --cfg configs/endonerf/baseline/base_d3k1.yml --mode train
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endonerf.py --cfg configs/endonerf/baseline/base_d6k1.yml --mode train
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endonerf.py --cfg configs/endonerf/baseline/base_d7k1.yml --mode train

# Test baseline EndoNeRF
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endonerf.py --cfg configs/endonerf/baseline/base_pull.yml --mode test
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endonerf.py --cfg configs/endonerf/baseline/base_cut.yml --mode test
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endonerf.py --cfg configs/endonerf/baseline/base_d1k1.yml --mode test
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endonerf.py --cfg configs/endonerf/baseline/base_d2k1.yml --mode test
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endonerf.py --cfg configs/endonerf/baseline/base_d3k1.yml --mode test
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endonerf.py --cfg configs/endonerf/baseline/base_d6k1.yml --mode test
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endonerf.py --cfg configs/endonerf/baseline/base_d7k1.yml --mode test

# Demo baseline EndoNeRF
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endonerf.py --cfg configs/endonerf/baseline/base_pull.yml --mode demo
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endonerf.py --cfg configs/endonerf/baseline/base_cut.yml --mode demo
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endonerf.py --cfg configs/endonerf/baseline/base_d1k1.yml --mode demo
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endonerf.py --cfg configs/endonerf/baseline/base_d2k1.yml --mode demo
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endonerf.py --cfg configs/endonerf/baseline/base_d3k1.yml --mode demo
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endonerf.py --cfg configs/endonerf/baseline/base_d6k1.yml --mode demo
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endonerf.py --cfg configs/endonerf/baseline/base_d7k1.yml --mode demo

# Ablation study
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endosurf.py --cfg configs/endosurf/ablation/no_angle_loss.yml --mode train
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endosurf.py --cfg configs/endosurf/ablation/no_eikonal_loss.yml --mode train
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endosurf.py --cfg configs/endosurf/ablation/no_sdf_loss.yml --mode train
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endosurf.py --cfg configs/endosurf/ablation/no_surf_neig_loss.yml --mode test
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endosurf.py --cfg configs/endosurf/ablation/no_angle_loss.yml --mode test
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endosurf.py --cfg configs/endosurf/ablation/no_eikonal_loss.yml --mode test
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endosurf.py --cfg configs/endosurf/ablation/no_sdf_loss.yml --mode test
CUDA_VISIBLE_DEVICES=0 python src/trainer/trainer_endosurf.py --cfg configs/endosurf/ablation/no_surf_neig_loss.yml --mode test

