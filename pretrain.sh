export WANDB_API_KEY=bda3630516768ce4285b4d8205dd6c0189520be1

python -m litgpt pretrain \
  --model_name pythia-160m \
  --tokenizer_dir checkpoints/EleutherAI/pythia-160m \
  --data FineWeb \
  --train.micro_batch_size 32 \
  --train.max_tokens 10_000_000_000 \
  --out_dir /data/users/zichunyu/out/pythia-160m-ddp/fineweb/sample-10BT \
  --logger_name wandb \
  --exp_name pythia-160m_fineweb_sample-10BT-ddp
