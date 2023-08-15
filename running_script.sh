#!/bin/bash 

python model_moe.py \
    --model_save_path experiments/cross_encoder_model_with_moe_balance_loss \
    --model_name_or_path bert-base-uncased \
    --batch_size 128 \
    --max_seq_length 64 \
    --num_epochs 10 \
    --num_experts 2 \
    --top_routing 1 \
    --alpha_1 0.05 \
    --alpha_2 0.005 \
    --learning_rate 1e-4

python evaluation_moe.py \
    --model_name_or_path experiments/cross_encoder_model_with_moe_balance_loss \
    --task_set sts_sickr

python evaluation_moe.py \
    --model_name_or_path experiments/cross_encoder_model_with_moe_balance_loss \
    --task_set domain_transfer
