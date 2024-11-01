# python -m litgpt pretrain \
#     --model_name pythia-1b \
#     --tokenizer_dir checkpoints/EleutherAI/pythia-1b \
#     --data FineWeb \
#     --data_path /data/users/zichunyu/data/fineweb/sample-350BT/train/0/cluster/5267030 \
#     --train.micro_batch_size 16 \
#     --train.max_tokens 104_857_600_00 \
#     --train.resume_steps 10000 \
#     --eval.interval 200000 \
#     --in_dir /data/users/zichunyu/out/pythia-1b/fineweb/sample-100BT \
#     --out_dir /data/users/zichunyu/out/pythia-1b/fineweb/sample-350BT/cluster \
#     --resume true \
#     --exp_name pythia-1b_fineweb_cluster_s10000

python -m litgpt pretrain \
    --model_name pythia-1b \
    --tokenizer_dir checkpoints/EleutherAI/pythia-1b \
    --data FineWeb \
    --data_path /data/users/zichunyu/data/fineweb/sample-350BT/train/0/cluster/5267030 \
    --train.save_interval 200 \
    --train.micro_batch_size 16 \
    --train.max_tokens 104_857_600_00 \
    --train.resume_steps 20000 \
    --train.decay true \
    --eval.interval 200000 \
    --out_dir /data/users/zichunyu/out/pythia-1b/fineweb/sample-350BT/cluster \
    --resume true \
    --seed 1337 \
    --exp_name pythia-1b_fineweb_cluster_d20000

# python -m litgpt pretrain \
#     --model_name pythia-410m \
#     --tokenizer_dir checkpoints/EleutherAI/pythia-410m \
#     --data FineWeb \
#     --data_path /data/users/zichunyu/data/fineweb/sample-100BT/train \
#     --train.save_interval 5000 \
#     --train.micro_batch_size 16 \
#     --train.max_tokens 50_000_000_000 \
#     --eval.interval 200000 \
#     --out_dir /data/users/zichunyu/out/pythia-410m/fineweb/sample-100BT_wsd \
#     --logger_name wandb \
#     --exp_name pythia-410m_fineweb_sample-100BT_wsd

# for c in 10000 20000 30000; do
#     python -m litgpt pretrain \
#     --model_name pythia-410m \
#     --tokenizer_dir checkpoints/EleutherAI/pythia-410m \
#     --data FineWeb \
#     --data_path /data/users/zichunyu/data/fineweb/sample-100BT/train \
#     --train.save_interval 200 \
#     --train.micro_batch_size 16 \
#     --train.resume_steps $c \
#     --train.decay true \
#     --eval.interval 200000 \
#     --in_dir /data/users/zichunyu/out/pythia-410m/fineweb/sample-100BT_wsd \
#     --out_dir /data/users/zichunyu/out/pythia-410m/fineweb/sample-100BT_wsd \
#     --resume true \
#     --seed 1337 \
#     --exp_name pythia-410m_fineweb_sample-100BT_wsd_d$c
# done

# for c in "0.3-0.7" "0.7-0.3" "0.9-0.1"; do
#     python -m litgpt pretrain \
#         --model_name pythia-1b \
#         --tokenizer_dir checkpoints/EleutherAI/pythia-1b \
#         --data FineWeb \
#         --data_path /data/users/zichunyu/data/games-fineweb-102400-${c} \
#         --train.save_interval 200 \
#         --train.micro_batch_size 16 \
#         --train.resume_steps 45000 \
#         --train.decay true \
#         --eval.interval 200000 \
#         --in_dir /data/users/zichunyu/out/pythia-1b/fineweb/sample-100BT \
#         --out_dir /data/users/zichunyu/out/pythia-1b/games-fineweb-102400-${c} \
#         --resume true \
#         --exp_name pythia-1b_games-fineweb-102400-${c}_d45000
# done

# python -m litgpt pretrain \
#     --model_name pythia-1b \
#     --tokenizer_dir checkpoints/EleutherAI/pythia-1b \
#     --data FineWeb \
#     --data_path /data/users/zichunyu/data/mates-fineweb-500000 \
#     --train.save_interval 200 \
#     --train.micro_batch_size 16 \
#     --train.resume_steps 45000 \
#     --train.decay true \
#     --eval.interval 200000 \
#     --in_dir /data/users/zichunyu/out/pythia-1b/fineweb/sample-100BT \
#     --out_dir /data/users/zichunyu/out/pythia-1b/mates-fineweb-500000 \
#     --resume true \
#     --logger_name wandb \
#     --exp_name pythia-1b_fineweb_mates_d45000

# python -m litgpt pretrain \
#     --model_name pythia-1b \
#     --tokenizer_dir checkpoints/EleutherAI/pythia-1b \
#     --data FineWeb \
#     --data_path ../manifold/scaling_mates/data/fineweb/sample-350BT/train/0/pythia-1b/mates/10000/5120000/0.5 \
#     --train.save_interval 200 \
#     --train.micro_batch_size 16 \
#     --train.resume_steps 10000 \
#     --train.decay true \
#     --eval.interval 200000 \
#     --in_dir /data/users/zichunyu/out/pythia-1b/fineweb/sample-100BT \
#     --out_dir /data/users/zichunyu/out/pythia-1b/fineweb/sample-350BT/mates/10000/5120000/0.5 \
#     --resume true \
#     --seed 1337 \
#     --exp_name pythia-1b_fineweb_mates_d10000

# python3 -m litgpt pretrain \
#     --model_name pythia-1b \
#     --tokenizer_dir checkpoints/EleutherAI/pythia-1b \
#     --data FineWeb \
#     --data_path ../manifold/scaling_mates/data/fineweb/sample-350BT/train/1/pythia-1b/mates/20000/5120000 \
#     --train.save_interval 200 \
#     --train.micro_batch_size 16 \
#     --train.resume_steps 20000 \
#     --train.decay true \
#     --eval.interval 200000 \
#     --in_dir ../out/pythia-1b/fineweb/sample-350BT/mates \
#     --out_dir ../manifold/scaling_mates/out/pythia-1b/fineweb/sample-350BT/mates \
#     --resume true \
#     --exp_name pythia-1b_fineweb_mates_d20000

# bash scripts/eval_batch_job.sh
