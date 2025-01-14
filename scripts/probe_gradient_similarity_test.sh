#!/bin/bash
#SBATCH --partition=general          
#SBATCH --job-name=probe_gradient_similarity_full
#SBATCH --gres=gpu:A6000:8                
#SBATCH --output=probe_gradient_similarity_full_%J.out
#SBATCH --error=probe_gradient_similarity_full_%J.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=500G
#SBATCH --mail-type=BEGIN,END,FAIL  # Send email at job start, end, and failure
#SBATCH --mail-user=emilyx@andrew.cmu.edu  # Replace with your email address

# description: calculates gradient cosine similar for all parameters in the model

# Usage
# cd Lightning-Pretrain
# conda activate myenv
# sbatch scripts/probe_gradient_similarity_full.sh


CUDA_VISIBLE_DEVICES=0 python -m litgpt.probe_gradient_similarity \
      --model_name pythia-1b \
      --train_data_dir /data/datasets/hf_cache/data/fineweb/sample-350BT/train/0 \
      --resume /data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/lit_model.pth \
      --out_dir /data/datasets/hf_cache/test \
      --rank 0 \
      --devices 1

#test probe gradients
CUDA_VISIBLE_DEVICES=0 python -m litgpt.probe_gradients \
      --model_name pythia-1b \
      --train_data_dir /data/datasets/hf_cache/data/fineweb/sample-350BT/train/0 \
      --resume /data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/lit_model.pth \
      --out_dir /data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/grad \
      --rank 0 \
      --devices 1
