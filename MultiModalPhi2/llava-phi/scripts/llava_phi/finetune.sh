#!/bin/bash

deepspeed --master_port 29600 llava_phi/train/train.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./checkpoints/llavaPhi-v0-3b-pretrain \
    --version v0 \
    --data_path ./finetune_data/llava_instruct_150k.json \
    --image_folder ./finetune_data/images \
    --tune_mm_mlp_adapter True \
    --freeze_vision_tower False \
    --freeze_backbone False \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints/llavaPhi-v0-3b-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard
