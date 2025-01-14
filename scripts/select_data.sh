base_dir="/data/datasets/hf_cache"
ckpt=143000
num_shards=16
split=0

python mates/select_data.py \
    --base_dir $base_dir \
    --method mates \
    --split $split \
    --ckpt $ckpt \
    --shard_num $num_shards \

#10G RAM, 8 CPU, 30min