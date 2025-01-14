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

gpu_index=0
for s in {0..7}; do
    echo "Running Step"
    echo $s
    CUDA_VISIBLE_DEVICES=$gpu_index PYTHONUNBUFFERED=1 python -m litgpt.probe_gradient_similarity \
      --model_name pythia-1b \
      --train_data_dir /data/datasets/hf_cache/data/fineweb/sample-350BT/train/0 \
      --resume /data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/lit_model.pth \
      --out_dir /data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/less_full \
      --rank $s \
      --devices 1 > log_job_s${s}_gpu${gpu_index}.out 2>&1 &
    ((gpu_index=(gpu_index+1)%8))
done

wait
echo "0-7 done"

gpu_index=0
for s in {8..15}; do
    echo "Running Step"
    echo $s
    CUDA_VISIBLE_DEVICES=$gpu_index PYTHONUNBUFFERED=1 python -m litgpt.probe_gradient_similarity \
      --model_name pythia-1b \
      --train_data_dir /data/datasets/hf_cache/data/fineweb/sample-350BT/train/0 \
      --resume /data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/lit_model.pth \
      --out_dir /data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/less_full \
      --rank $s \
      --devices 1 > log_job_s${s}_gpu${gpu_index}.out 2>&1 &
    ((gpu_index=(gpu_index+1)%8))
done

wait
echo "8-15 done"

gpu_index=0
for s in {16..23}; do
    echo "Running Step"
    echo $s
    CUDA_VISIBLE_DEVICES=$gpu_index PYTHONUNBUFFERED=1 python -m litgpt.probe_gradient_similarity \
      --model_name pythia-1b \
      --train_data_dir /data/datasets/hf_cache/data/fineweb/sample-350BT/train/0 \
      --resume /data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/lit_model.pth \
      --out_dir /data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/less_full \
      --rank $s \
      --devices 1 > log_job_s${s}_gpu${gpu_index}.out 2>&1 &
    ((gpu_index=(gpu_index+1)%8))
done

wait
echo "16-23 done"

gpu_index=0
for s in {24..31}; do
    echo "Running Step"
    echo $s
    CUDA_VISIBLE_DEVICES=$gpu_index PYTHONUNBUFFERED=1 python -m litgpt.probe_gradient_similarity \
      --model_name pythia-1b \
      --train_data_dir /data/datasets/hf_cache/data/fineweb/sample-350BT/train/0 \
      --resume /data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/lit_model.pth \
      --out_dir /data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/less_full \
      --rank $s \
      --devices 1 > log_job_s${s}_gpu${gpu_index}.out 2>&1 &
    ((gpu_index=(gpu_index+1)%8))
done

wait
echo "24-31 done"




#testing
# CUDA_VISIBLE_DEVICES=0 python -m litgpt.probe_gradient_similarity \
#       --model_name pythia-1b \
#       --train_data_dir /data/datasets/hf_cache/data/fineweb/sample-350BT/train/0 \
#       --resume /data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/lit_model.pth \
#       --out_dir /data/datasets/hf_cache/test \
#       --rank 0 \
#       --devices 1
# --out_dir /data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/less \