python -m litgpt evaluate \
  --checkpoint_dir /data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/random/final \
  --num_fewshot 0 \
  --batch_size 4 \
  --tasks "sciq,arc_easy,arc_challenge,logiqa,mmlu,boolq,hellaswag,piqa,winogrande,lambada_openai" \
  --out_dir out/pythia-1b/step-10000-decay-random
