#!/bin/sh

SBATCH_CPU=""
SBATCH_CPU="-p hiperf --gres=gpu:4"

sbatch -n1 \
--cpus-per-task=10 \
--mem=100000 \
$SBATCH_CPU \
-t 12:00:00 \
--job-name=monster_datasets_us3d_train \
--output=./logs/monster_datasets_us3d_train_%j.log \
--error=./logs/monster_datasets_us3d_train_%j.err \
\
--wrap="PATH=/home/s0214/.conda/envs/monster/bin:\$PATH CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 --num_machines=1 train_us3d.py"