#!/bin/bash
#SBATCH --partition=general          
#SBATCH --job-name=pretrain_decay_compare_v2
#SBATCH --gres=gpu:A6000:8                
#SBATCH --output=pretrain_decay_%J.out
#SBATCH --error=pretrain_decay_%J.err
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --mem=200G
#SBATCH --mail-type=BEGIN,END,FAIL  # Send email at job start, end, and failure
#SBATCH --mail-user=emilyx@andrew.cmu.edu  # Replace with your email address


# Usage
# cd Lightning-Pretrain
# conda activate myenv
# sbatch scripts/pretrain_decay_compare_v2.sh

export NCCL_P2P_DISABLE=1

method=bge_cluster_full_gradsim
data_path=/data/datasets/hf_cache/data/fineweb/sample-350BT/train/0/select_full_grad_sim_mab_decay

python3 -m litgpt pretrain \
  --model_name pythia-1b \
  --tokenizer_dir checkpoints/EleutherAI/pythia-1b \
  --data FineWeb \
  --data_path $data_path \
  --train.save_interval 200 \
  --train.micro_batch_size 8 \
  --train.max_tokens 104_857_600_00 \
  --resume /data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/lit_model.pth \
  --train.decay true \
  --train.log_interval 10 \
  --eval.interval 50 \
  --exp_name pythia_1b_step10000_decay_cluster_gradsim \
  --out_dir /data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/$method \
  --seed 1337 \
  --devices 8

echo "Done with $method"


method2=top_full_gradsim
data_path2=/data/datasets/hf_cache/data/fineweb/sample-350BT/train/0/select_full_grad_sim_top_decay

python3 -m litgpt pretrain \
  --model_name pythia-1b \
  --tokenizer_dir checkpoints/EleutherAI/pythia-1b \
  --data FineWeb \
  --data_path $data_path2 \
  --train.save_interval 200 \
  --train.micro_batch_size 8 \
  --train.max_tokens 104_857_600_00 \
  --resume /data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/lit_model.pth \
  --train.decay true \
  --train.log_interval 10 \
  --eval.interval 50 \
  --exp_name pythia_1b_step10000_decay_top_gradsim \
  --out_dir /data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/$method2 \
  --seed 1337 \
  --devices 8

echo "Done with $method2"

echo "start evaluation"

python -m litgpt evaluate \
  --checkpoint_dir /data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/$method/final \
  --num_fewshot 0 \
  --batch_size 4 \
  --tasks "sciq,arc_easy,arc_challenge,logiqa,mmlu,boolq,hellaswag,piqa,winogrande,lambada_openai" \
  --out_dir out/pythia-1b/step-10000-$method

echo "Done with evaluation for $method"

python -m litgpt evaluate \
  --checkpoint_dir /data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/$method2/final \
  --num_fewshot 0 \
  --batch_size 4 \
  --tasks "sciq,arc_easy,arc_challenge,logiqa,mmlu,boolq,hellaswag,piqa,winogrande,lambada_openai" \
  --out_dir out/pythia-1b/step-10000-$method2

echo "Done with evaluation for $method2"



