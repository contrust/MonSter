env_create:
	bash scripts/env_create.sh
pip_install:
	bash scripts/pip_install.sh
checkpoints_kitti_download:
	bash scripts/checkpoints_kitti_download.sh
checkpoints_mix_all_download:
	bash scripts/checkpoints_mix_all_download.sh
datasets_kitti_download:
	bash scripts/datasets_kitti_download.sh
datasets_whu_download:
	bash scripts/datasets_whu_download.sh
datasets_whu_train:
	bash scripts/datasets_whu_train.sh
datasets_kitti_evaluate:
	bash scripts/datasets_kitti_evaluate.sh
datasets_whu_evaluate:
	bash scripts/datasets_whu_evaluate.sh