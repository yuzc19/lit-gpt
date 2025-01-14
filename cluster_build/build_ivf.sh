#!/bin/bash
#SBATCH --partition=general
#SBATCH --job-name=build_index
#SBATCH --output=logs/build-ivf-%x-%j.out
#SBATCH --error=logs/build-ivf-%x-%j.err
#SBATCH --nodes=1

#SBATCH --mem=400G
#SBATCH --gres=gpu:0

#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=BEGIN,END,FAIL  # Send email at job start, end, and failure
#SBATCH --mail-user=emilyx@andrew.cmu.edu  # Replace with your email address


## USAGE
# cd Lightning-Pretrain
# conda activate myenv
# sbatch cluster/build_ivf.sh


# index specification 
metrics='dot'
nlist=1000 # same cluster size when grouping 600M data into 10k clusters. data = 17M

# input embedding 
embed_path="/data/datasets/hf_cache/data/fineweb/sample-350BT/train_bge_micro_embeddings"

# index destination 
dest_dir="/data/datasets/hf_cache/data/fineweb/sample-350BT/train_bge_micro_embeddings_index_1000"

mkdir -p $dest_dir
echo $dest_dir

python cluster/build_by_kmeans_ds.py \
        --dest_dir $dest_dir \
        --dataset_path $embed_path \
        --metrics $metrics \
        --nlist $nlist 

#test
# python build_by_kmeans_ds.py \
#             --dest_dir ./out \
#             --dataset_path /data/datasets/hf_cache/data/fineweb/sample-350BT/train_bge_micro_embeddings \
#             --nlist 100 \


# Job ID: 147548
# Cluster: babel
# State: COMPLETED (exit code 0)
# Nodes: 1
# Cores per node: 16
# CPU Utilized: 01:47:33
# CPU Efficiency: 5.75% of 1-07:09:36 core-walltime
# Job Wall-clock time: 01:56:51
# Memory Utilized: 68.89 GB