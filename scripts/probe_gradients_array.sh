#!/bin/bash
#SBATCH --partition=preempt          
#SBATCH --job-name=probe_gradients
#SBATCH --gres=gpu:L40S:1              
#SBATCH --output=logs/probe_gradients_%A_%a.out
#SBATCH --error=logs/probe_gradients_%A_%a.err
#SBATCH --array=0-23
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emilyx@andrew.cmu.edu
#SBATCH --requeue

# description: calculates projected gradient for each datapoint. 17M datapoints per split for 350BT. 51200 per rank, 300 shards
# Usage
# cd Lightning-Pretrain
# conda activate myenv
# sbatch scripts/probe_gradients_array.sh

# Exit immediately on any error
set -e

# Define the total number of ranks and batch size
TOTAL_RANKS=300
BATCH_SIZE=24
STARTING_RANK=48 # default 0
CHECKPOINT_DIR="checkpoints"
mkdir -p $CHECKPOINT_DIR

# Load checkpoint if it exists
CHECKPOINT_FILE="$CHECKPOINT_DIR/checkpoint_${SLURM_ARRAY_TASK_ID}.pkl"
if [ -f $CHECKPOINT_FILE ]; then
    echo "Loading checkpoint from $CHECKPOINT_FILE"
    RANK=$(cat $CHECKPOINT_FILE)
else
    RANK=$((STARTING_RANK + SLURM_ARRAY_TASK_ID))
fi

# Loop to run jobs in batches of 24
while [ $RANK -lt $TOTAL_RANKS ]; do
    echo "Processing RANK $RANK"

    PYTHONUNBUFFERED=1 python -m litgpt.probe_gradients \
        --model_name pythia-1b \
        --train_data_dir /data/datasets/hf_cache/data/fineweb/sample-350BT/train/0 \
        --resume /data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/lit_model.pth \
        --out_dir /data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/grad \
        --rank $RANK \
        --devices 1 || { echo "Error processing RANK $RANK"; exit 1; }

    # Save checkpoint
    echo $((RANK + BATCH_SIZE)) > $CHECKPOINT_FILE

    # Increment rank for the next iteration
    RANK=$((RANK + BATCH_SIZE))
done