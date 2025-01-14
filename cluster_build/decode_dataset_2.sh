#!/bin/bash
#SBATCH --partition=general          
#SBATCH --job-name=decode_dataset
#SBATCH --gres=gpu:0                  
#SBATCH --output=decode_dataset_%A_%a.out
#SBATCH --error=decode_dataset_%A_%a.err
#SBATCH --array=8-15
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --mem=200G

python decode_litdata_dataset.py --shard $SLURM_ARRAY_TASK_ID --num_shards 16