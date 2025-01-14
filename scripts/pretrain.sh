if [ "$decay" = "true" ]; then
  python3 -m litgpt pretrain \
    --model_name pythia-1b \
    --tokenizer_dir checkpoints/EleutherAI/pythia-1b \
    --data FineWeb \
    --data_path data/fineweb/sample-350BT/train/$split/pythia-1b/$method/$ckpt/5120000 \
    --train.save_interval 200 \
    --train.micro_batch_size 16 \
    --train.max_tokens 104_857_600_00 \
    --train.resume_steps $resume \
    --train.decay true \
    --eval.interval 200000 \
    --out_dir out/pythia-1b/fineweb/sample-350BT/$method \
    --seed 1337
else
  python3 -m litgpt pretrain \
    --model_name pythia-1b \
    --tokenizer_dir checkpoints/EleutherAI/pythia-1b \
    --data FineWeb \
    --data_path data/fineweb/sample-350BT/train/$split/pythia-1b/$method/$ckpt/5120000 \
    --train.micro_batch_size 16 \
    --train.max_tokens 104_857_600_00 \
    --train.resume_steps $resume \
    --eval.interval 200000 \
    --out_dir out/pythia-1b/fineweb/sample-350BT/$method
fi
