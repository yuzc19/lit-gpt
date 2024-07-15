# python3 -m litgpt pretrain \
#   --model_name pythia-410m \
#   --tokenizer_dir checkpoints/EleutherAI/pythia-410m \
#   --data FineWeb \
#   --data_path /data/users/zichunyu/data/fineweb/sample-100BT/pythia-1b/mates/10000-flan/2000000 \
#   --train.save_interval 5000 \
#   --train.micro_batch_size 16 \
#   --train.max_tokens 50_000_000 \
#   --eval.interval 200000 \
#   --out_dir /data/users/zichunyu/out/pythia-410m/fineweb/sample-350BT \
#   --logger_name tensorboard \
#   --exp_name pythia-410m_fineweb_sample-350BT

for c in 2000000 800000 266666; do
  python -m litgpt pretrain \
    --model_name pythia-1b \
    --tokenizer_dir checkpoints/EleutherAI/pythia-1b \
    --data FineWeb \
    --data_path /data/users/zichunyu/data/fineweb/sample-100BT/pythia-1b/mates/10000-flan/$c \
    --train.save_interval 200 \
    --train.micro_batch_size 16 \
    --train.resume_steps 10000 \
    --eval.interval 200000 \
    --out_dir /data/users/zichunyu/out/pythia-1b/fineweb/sample-100BT/mates/10000-flan/$c \
    --logger_name tensorboard \
    --resume true \
    --exp_name pythia-1b_fineweb_mates_d10000
done
