CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
WANDB_MODE=offline \
swift sft \
    --model /data/xxxx \
    --tuner_type full \
    --dataset /root/neko/data/xxxxx.jsonl \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 16 \
    --max_length 4096 \
    --group_by_length false \
    --packing false \
    --deepspeed zero3 \
    --gradient_checkpointing true \
    --attn_impl eager \
    --output_dir /data/ckpt_neko_9b2 \
    --eval_steps 50 \
    --save_steps 4000 \
    --logging_steps 10 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 2 \
    --dataset_num_proc 4 \
    --model_author neko9b \
    --model_name neko9b \
    --report_to wandb \