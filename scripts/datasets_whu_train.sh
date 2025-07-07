#!/bin/sh

# Initialize conda without conda init
eval "$(conda shell.bash hook)"
conda activate monster
export PYTHONPATH=$(pwd)

CPU_TRAIN=false

SBATCH_CPU=""
PYTHON_CPU=""
if [ $CPU_TRAIN == false ]
then
  SBATCH_CPU="-p hiperf --gres=gpu:4"
else
  PYTHON_CPU="--cpu"
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
--wrap="CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch train_whu.py"