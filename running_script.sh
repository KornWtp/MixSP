#!/bin/bash 

python main.py \
    --model_save_path experiments/mixsp-simcse-base-model \
    --model_name_or_path princeton-nlp/sup-simcse-bert-base-uncased \
    --batch_size 16 \
    --max_seq_length 64 \
    --num_epochs 10 \
    --num_experts 2 \
    --top_routing 1 \
    --alpha_1 0.05 \
    --alpha_2 0.0005 \
    --learning_rate 5e-5