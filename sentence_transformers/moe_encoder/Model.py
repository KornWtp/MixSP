import os
import torch
import torch.nn as nn
from typing import Dict, Type, Callable, List
from transformers import AutoModel


class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])

    def forward(self, x):
        # Mixture of Experts (MoE)
        gate_outputs = self.gate(x)
        gate_probs = torch.softmax(gate_outputs, dim=1)

        expert_outputs = []
        for i in range(self.num_experts):
            expert_output = self.experts[i](x)
            expert_outputs.append(expert_output)

        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Weighted combination of expert outputs
        weighted_outputs = torch.bmm(gate_probs.unsqueeze(1), expert_outputs).squeeze(1)
        return weighted_outputs


class Model(nn.Module):
    def __init__(self, model_name:str, num_experts:int, config:Dict = {}):
        super(Model, self).__init__()
        expert_dim = config.hidden_size
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        self.moe = MoE(expert_dim, expert_dim, num_experts)

    def forward(self, features):
        # BERT encoding
        bert_output = self.bert(**features, return_dict=True)
        last_layer_hidden_states = bert_output.last_hidden_state
        pooled_output = bert_output.pooler_output
        avg_pooling = torch.mean(last_layer_hidden_states, dim=1)
        
        # Get the indexes of the [SEP] tokens
        input_ids = features['input_ids']
        sep_indices = (input_ids == 102).nonzero()[:, 1]

        # Split the token vectors for sentence A and sentence B
        sentence_a_embeddings = []
        sentence_b_embeddings = []
        for i in range(len(sep_indices)):
            if (i % 2) == 0:
                sentence_a_embedding = last_layer_hidden_states[int(i/2), 1:sep_indices[i], :]
                sentence_a_embeddings.append(torch.mean(sentence_a_embedding, dim=0))
            else:
                sentence_b_embedding = last_layer_hidden_states[int(i/2), sep_indices[i-1]+1:sep_indices[i], :]
                sentence_b_embeddings.append(torch.mean(sentence_b_embedding, dim=0))
        
        cls_embeddings = last_layer_hidden_states[:, 0, :]
        sentence_a_embeddings = torch.stack(sentence_a_embeddings, dim=0)
        sentence_b_embeddings = torch.stack(sentence_b_embeddings, dim=0)
        
        moe_input_sentence_a = sentence_a_embeddings + cls_embeddings
        moe_input_sentence_b = sentence_b_embeddings + cls_embeddings
        moe_output_sentence_a = self.moe(moe_input_sentence_a)
        moe_output_sentence_b = self.moe(moe_input_sentence_b)
        all_embeddings = [moe_output_sentence_a, moe_output_sentence_b]
        return all_embeddings


    def save_pretrained(self, save_path):
        self.bert.save_pretrained(save_path)
        torch.save(self.moe.state_dict(), f"{save_path}/moe_model.bin")
