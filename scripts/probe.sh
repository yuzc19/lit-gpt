gpu_index=0
for s in {0..7}; do
    echo $s
    CUDA_VISIBLE_DEVICES=$gpu_index nohup python -m litgpt.probe_oracle_data_influence \
      --model_name pythia-1b \
      --tokenizer_dir checkpoints/EleutherAI/pythia-1b \
      --data Tulu \
      --train.resume_steps 10000 \
      --out_dir /data/users/zichunyu/out/pythia-1b/fineweb/sample-100BT \
      --rank $s \
      --devices 1 > log_job_s${s}_gpu${gpu_index}.out 2>&1 &
    ((gpu_index=(gpu_index+1)%8))
done
