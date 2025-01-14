#!/bin/bash
#SBATCH --partition=general          
#SBATCH --job-name=embed_dataset
#SBATCH --gres=gpu:1                  
#SBATCH --output=embed_dataset_%A_%a.out
#SBATCH --error=embed_dataset_%A_%a.err
#SBATCH --array=0-7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --mem=200G


# Run the Python script with the appropriate shard index and GPU index
python embed.py --shard $SLURM_ARRAY_TASK_ID 8



# gpu_index=0
# for s in {0..7}; do
#     echo $s
#     CUDA_VISIBLE_DEVICES=$gpu_index python embed.py --shard $s 8 > log_job_s${s}_gpu${gpu_index}.out 2>&1 &
#     ((gpu_index=(gpu_index+1)%8))
# done