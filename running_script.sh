#!/bin/bash 

python main.py \
    --model_save_path experiments/cross_encoder_model_with_moe_balance \
    --model_name_or_path bert-base-uncased \
    --batch_size 128 \
    --max_seq_length 64 \
    --num_epochs 10 \
    --num_experts 2 \
    --top_routing 1\
    --temp 0.3 \
    --learning_rate 1e-4

python evaluation.py \
    --model_name_or_path experiments/cross_encoder_model_with_moe_balance \
    --task_set sts \
    --mode test