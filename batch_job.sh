#!/bin/bash

# for s in {0..7}
# do
#     sbatch <<EOT
# #!/bin/bash
# #SBATCH --job-name=doremi_$s
# #SBATCH --output=logs/myjob_$s.out
# #SBATCH --error=logs/myjob_$s.err
# #SBATCH --partition=general
# #SBATCH --gres=gpu:A6000:1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=32G
# #SBATCH --time=2-00:00:00

# # Your job commands go here
# python pretrain/c4_rand_test_comb.py --resume /data/user_data/zichunyu/out/c4-dsdm-pythia-410m/iter-080000-ckpt.pth --rank $s
# EOT
# done

# gpu_index=0
# for s in {0..7}; do
#     echo $s
#     CUDA_VISIBLE_DEVICES=$gpu_index nohup python pretrain/c4_rand_test.py --resume /data/user_data/zichunyu/out/c4-dsdm-pythia-410m/iter-120000-ckpt.pth --rank $s > logs/log_job_s${s}_gpu${gpu_index}.out 2>&1 &
#     ((gpu_index=(gpu_index+1)%4))
# done
# wait

gpu_index=0
for s in {8..15}; do
    echo $s
    CUDA_VISIBLE_DEVICES=$gpu_index nohup python pretrain/c4_rand_test.py --resume /data/datasets/hf_cache/c4-pythia-1b/iter-080000-ckpt/0-100/tar-lambada_openai-6087236-sample/iter-159992-ckpt.pth --rank $s > logs/log_job_s${s}_gpu${gpu_index}.out 2>&1 &
    ((gpu_index=(gpu_index+1)%8))
done
wait