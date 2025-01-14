#!/bin/bash
#SBATCH --partition=general          
#SBATCH --job-name=probe_gradient_similarity
#SBATCH --gres=gpu:A6000:8                
#SBATCH --output=probe_gradient_similarity_%J.out
#SBATCH --error=probe_gradient_similarity_%J.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --mem=500G
#SBATCH --mail-type=BEGIN,END,FAIL  # Send email at job start, end, and failure
#SBATCH --mail-user=emilyx@andrew.cmu.edu  # Replace with your email address


# Usage
# cd Lightning-Pretrain
# conda activate myenv
# sbatch scripts/probe_gradient_similarity.sh

gpu_index=0
for s in {0..7}; do
    echo "Running Step"
    echo $s
    CUDA_VISIBLE_DEVICES=$gpu_index PYTHONUNBUFFERED=1 python -m litgpt.probe_gradient_similarity \
      --model_name pythia-1b \
      --train_data_dir /data/datasets/hf_cache/data/fineweb/sample-350BT/train/0 \
      --resume /data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/lit_model.pth \
      --out_dir /data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/less \
      --rank $s \
      --devices 1 > log_job_s${s}_gpu${gpu_index}.out 2>&1 &
    ((gpu_index=(gpu_index+1)%8))
done

wait
echo "All Done"

#testing
# CUDA_VISIBLE_DEVICES=0 python -m litgpt.probe_gradient_similarity \
#       --model_name pythia-1b \
#       --train_data_dir /data/datasets/hf_cache/data/fineweb/sample-350BT/train/0 \
#       --resume /data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/lit_model.pth \
#       --out_dir /data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/less \
#       --rank 0 \
#       --devices 1
# --out_dir /data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/less \