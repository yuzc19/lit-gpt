python -m litgpt evaluate \
  --checkpoint_dir /data/users/zichunyu/out/custom-model/final \
  --num_fewshot 0 \
  --batch_size 4 \
  --tasks "arc_easy,arc_challenge,sciq,logiqa,hellaswag,piqa,winogrande,boolq,lambada_openai,lambada_standard,openbookqa,copa" \
  --out_dir out/pythia-160m/
