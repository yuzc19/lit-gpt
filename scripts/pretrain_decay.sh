#!/bin/bash
#SBATCH --partition=general          
#SBATCH --job-name=pretrain_decay_rand
#SBATCH --gres=gpu:A6000:8                
#SBATCH --output=pretrain_decay_%J.out
#SBATCH --error=pretrain_decay_%J.err
#SBATCH --cpus-per-task=16
#SBATCH --time=3:00:00
#SBATCH --mem=200G
#SBATCH --mail-type=BEGIN,END,FAIL  # Send email at job start, end, and failure
#SBATCH --mail-user=emilyx@andrew.cmu.edu  # Replace with your email address


# Usage
# cd Lightning-Pretrain
# conda activate myenv
# sbatch scripts/pretrain_decay.sh

export NCCL_P2P_DISABLE=1

method=random
resume=10000

python3 -m litgpt pretrain \
  --model_name pythia-1b \
  --tokenizer_dir checkpoints/EleutherAI/pythia-1b \
  --data FineWeb \
  --data_path /data/datasets/hf_cache/data/fineweb/sample-350BT/train/0 \
  --train.save_interval 200 \
  --train.micro_batch_size 8 \
  --train.max_tokens 104_857_600_00 \
  --train.resume_steps $resume \
  --train.decay true \
  --train.log_interval 10 \
  --eval.interval 50 \
  --exp_name pythia_1b_step10000_decay_random \
  --out_dir /data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/$method \
  --seed 1337 \
  --devices 8


# /data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/random

# Cores per node: 16
# CPU Utilized: 16:25:40
# CPU Efficiency: 50.10% of 1-08:47:28 core-walltime
# Job Wall-clock time: 02:02:58
# Memory Utilized: 29.15 GB