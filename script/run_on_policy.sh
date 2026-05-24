cd /root/neko/script
python on_policy_pipeline.py \
    --resume \
    --neko_init_path /data/ckpt_neko/v0-20260420-085555/checkpoint-345 \
    --maodie_init_path /data/ckpt_maodie/v0-20260419-152155/checkpoint-345 \
    --data_path /root/neko/data/train_neko.jsonl \
    --total_steps 100
