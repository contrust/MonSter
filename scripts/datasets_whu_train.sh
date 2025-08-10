#!/bin/sh

CPU_TRAIN=false

SBATCH_CPU=""
if [ $CPU_TRAIN == false ]
then
  SBATCH_CPU="-p hiperf --gres=gpu:4"
else
  SBATCH_CPU="-p tesla"
fi

sbatch -n1 \
--cpus-per-task=10 \
--mem=100000 \
$SBATCH_CPU \
-t 12:00:00 \
--job-name=datasets_whu_train \
--output=./logs/datasets_whu_train_%j.log \
--error=./logs/datasets_whu_train_%j.err \
\
--wrap="PATH=/home/s0214/.conda/envs/monster/bin:\$PATH CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 --num_machines=1 --num_processes_per_machine=4 python train_whu.py"