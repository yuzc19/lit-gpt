export WANDB_API_KEY=

python -m litgpt pretrain \
  --model_name pythia-410m \
  --tokenizer_dir checkpoints/EleutherAI/pythia-160m \
  --data FineWeb \
  --train.save_interval 5000 \
  --train.micro_batch_size 16 \
  --train.max_tokens 50_000_000_000 \
  --eval.interval 200000 \
  --out_dir /data/users/zichunyu/out/pythia-410m/fineweb/sample-100BT \
  --logger_name wandb \
  --exp_name pythia-410m_fineweb_sample-100BT
