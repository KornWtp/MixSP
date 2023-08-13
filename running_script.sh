#!/bin/bash 

python main.py \
    --model_save_path experiments/cross_encoder_model_with_moe_aux_loss \
    --model_name_or_path bert-base-uncased \
    --batch_size 128 \
    --num_epochs 10 \
    --num_experts 2 \
    --learning_rate 1e-4

python evaluation.py \
    --model_name_or_path experiments/cross_encoder_model_with_moe_aux_loss \
    --task_set sts \
    --mode test