#!/bin/bash
echo "start......"
deepspeed --include localhost:0 llava/train/train_mem.py \
   --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 3e-5 \
   --deepspeed ./scripts/zero3.json \
   --model_name_or_path liuhaotian/llava-v1.5-13b \
   --version v1 \
   --data_path ./traindata/train_sample_40kqa_combine_200kqa_gene.json \
   --image_folder /mnt/data2/zhiqiang/wenhao/data_sample_complexity \
   --vision_tower openai/clip-vit-large-patch14-336 \
   --mm_projector_type mlp2x_gelu \
   --mm_vision_select_layer -2 \
   --mm_use_im_start_end False \
   --mm_use_im_patch_token False \
   --image_aspect_ratio pad \
   --group_by_modality_length True \
   --bf16 True \
   --output_dir ./checkpoints/llava-lora-2epoch-40k_200k-gene \
   --num_train_epochs 2 \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 1 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 50000 \
   --save_total_limit 3 \
   --learning_rate 3e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --tf32 True \
   --model_max_length 2048 \
   --gradient_checkpointing True \
   --dataloader_num_workers 4 \
   --lazy_preprocess True \
   --report_to wandb




