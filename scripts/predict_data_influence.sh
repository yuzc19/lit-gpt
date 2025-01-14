#!/bin/bash
#SBATCH --partition=general          
#SBATCH --job-name=predict_data_influence
#SBATCH --gres=gpu:8                
#SBATCH --output=logs/predict_data_influence_%J.out
#SBATCH --error=logs/predict_data_influence_%J.err
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --mail-type=BEGIN,END,FAIL  # Send email at job start, end, and failure
#SBATCH --mail-user=emilyx@andrew.cmu.edu  # Replace with your email address


# Usage
# cd Lightning-Pretrain
# conda activate myenv
# sbatch scripts/predict_data_influence.sh

base_dir="/data/datasets/hf_cache"
ckpt=10000
num_shards=16
split=0

# 5G RAM, 8GPU, 1h map, 3h predict 
gpu_index=0
for s in {0..7}; do
    echo $s
    CUDA_VISIBLE_DEVICES=$gpu_index python mates/predict_data_influence.py --ckpt $ckpt --shard $s $num_shards --base_dir $base_dir --split $split > log_job_s${s}_gpu${gpu_index}.out 2>&1 &
    ((gpu_index=(gpu_index+1)%8))
done
wait
echo "first half done"

# this part failed due to bug. need to rerun
gpu_index=0
for s in {8..15}; do
    echo $s
    CUDA_VISIBLE_DEVICES=$gpu_index python mates/predict_data_influence.py --ckpt $ckpt --shard $s $num_shards --base_dir $base_dir --split $split > log_job_s${s}_gpu${gpu_index}.out 2>&1 &
    ((gpu_index=(gpu_index+1)%8))
done
wait
echo "All done"


#test
# CUDA_VISIBLE_DEVICES=0 python mates/predict_data_influence.py --ckpt $ckpt --shard 1 16 --base_dir $base_dir