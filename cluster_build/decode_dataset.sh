#!/bin/bash
#SBATCH --partition=general          
#SBATCH --job-name=decode_dataset
#SBATCH --gres=gpu:0                  
#SBATCH --output=decode_dataset_%A_%a.out
#SBATCH --error=decode_dataset_%A_%a.err
#SBATCH --array=0-7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --mem=200G

python decode_litdata_dataset.py --shard $SLURM_ARRAY_TASK_ID --num_shards 16

# Submit the second job array after the first one completes
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    sbatch --dependency=afterok:$SLURM_JOB_ID decode_dataset_2.sh
fi