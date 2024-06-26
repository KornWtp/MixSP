import os
import torch
import torch.nn as nn
from typing import Dict, Type, Callable, List
from transformers import AutoModel

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

        
class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, top_routing):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList([MLP(input_dim, input_dim*10, output_dim) for _ in range(num_experts)])
        
        # Define class frequencies 2 classes (04, 45)
        num_class_0 = 4343
        num_class_1 = 1406
        # Calculate class weights
        total_samples = num_class_0 + num_class_1
        class_weights = torch.tensor([total_samples / num_class_0, total_samples / num_class_1])
        self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=class_weights)

        if top_routing == 0:
            self.top_routing = None
        else:
            self.top_routing = top_routing

    def forward(self, x1, x2, labels = None):
        # Mixture of Experts (MoE)
        gate_outputs = self.gate(x1)
        gate_probs = torch.softmax(gate_outputs, dim=1)
        
        # Top routing
        if self.top_routing is not None:
            top_values, top_expert_indices = gate_probs.topk(self.top_routing, dim=1)
            gate_probs = torch.zeros_like(gate_probs)
            gate_probs.scatter_(1, top_expert_indices, top_values)

        expert_outputs = []
        for i in range(self.num_experts):
            expert_output = self.experts[i](x2)
            expert_outputs.append(expert_output)

        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Weighted combination of expert outputs
        weighted_outputs = torch.bmm(gate_probs.unsqueeze(1), expert_outputs).squeeze(1)
    
        if labels is not None:
            labels = [[0, 1] if label >= 0.8 else [1, 0] for label in labels]
            labels = torch.tensor(labels, dtype=torch.float32).to(gate_outputs.device)
            gate_loss = self.loss_fct(gate_probs, labels)
            return weighted_outputs, gate_loss
        else:
            return weighted_outputs


class Model(nn.Module):
    def __init__(self, model_name:str, num_experts:int, top_routing:int, config:Dict = {}):
        super(Model, self).__init__()
        expert_dim = config.hidden_size
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        self.moe = MoE(expert_dim, expert_dim, num_experts, top_routing)
        self.fc = nn.Linear(expert_dim*2, 1)
        self.experts = MLP(expert_dim, expert_dim*10, expert_dim)
        self.model_name = model_name


    def split_text_embed_roberta(self, features, embeddings):
        # Get the indexes of the [SEP] tokens
        input_ids = features['input_ids']
        sep_indices = (input_ids == 2).nonzero()[:, 1]

        # Split the token vectors for sentence A and sentence B
        sentence_a_embeddings = []
        sentence_b_embeddings = []
        for i in range(0, len(sep_indices), 3):
            sentence_a_embedding = embeddings[int(i/3), 1:sep_indices[i], :]
            sentence_b_embedding = embeddings[int(i/3), sep_indices[i+1]:sep_indices[i+2], :]
            sentence_a_embeddings.append(torch.mean(sentence_a_embedding, dim=0))
            sentence_b_embeddings.append(torch.mean(sentence_b_embedding, dim=0))
        sentence_a_embeddings = torch.stack(sentence_a_embeddings, dim=0)
        sentence_b_embeddings = torch.stack(sentence_b_embeddings, dim=0)

        return sentence_a_embeddings, sentence_b_embeddings
    

    def split_text_embed_bert(self, features, embeddings):
        # Get the indexes of the [SEP] tokens
        input_ids = features['input_ids']
        sep_indices = (input_ids == 102).nonzero()[:, 1]

        # Split the token vectors for sentence A and sentence B
        sentence_a_embeddings = []
        sentence_b_embeddings = []
        for i in range(len(sep_indices)):
            if (i % 2) == 0:
                sentence_a_embedding = embeddings[int(i/2), 1:sep_indices[i], :]
                sentence_a_embeddings.append(torch.mean(sentence_a_embedding, dim=0))
            else:
                sentence_b_embedding = embeddings[int(i/2), sep_indices[i-1]+1:sep_indices[i], :]
                sentence_b_embeddings.append(torch.mean(sentence_b_embedding, dim=0))
        sentence_a_embeddings = torch.stack(sentence_a_embeddings, dim=0)
        sentence_b_embeddings = torch.stack(sentence_b_embeddings, dim=0)

        return sentence_a_embeddings, sentence_b_embeddings


    def forward(self, features, labels = None):
        outputs = self.encoder(**features, return_dict=True)
        last_layer_hidden_states = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        avg_pooling = torch.mean(last_layer_hidden_states, dim=1)

        if 'roberta' in self.model_name:
            sentence_a_embeddings, sentence_b_embeddings = self.split_text_embed_roberta(features, last_layer_hidden_states)
        elif 'bert' in self.model_name:
            sentence_a_embeddings, sentence_b_embeddings = self.split_text_embed_bert(features, last_layer_hidden_states)
        else:
            raise NotImplementedError

        moe_input_sentence_a = sentence_a_embeddings + pooled_output
        moe_input_sentence_b = sentence_b_embeddings + pooled_output
        
        if labels is not None: 
            moe_output_sentence_a, loss_a = self.moe(moe_input_sentence_a, sentence_a_embeddings, labels)
            moe_output_sentence_b, loss_b = self.moe(moe_input_sentence_b, sentence_b_embeddings, labels)
            all_loss = (loss_a + loss_b)/2
            final_input = torch.cat((moe_output_sentence_a, moe_output_sentence_b), 1)
            final_output = self.fc(final_input)
            return final_output, all_loss
        else:
            moe_output_sentence_a = self.moe(moe_input_sentence_a, sentence_a_embeddings)
            moe_output_sentence_b = self.moe(moe_input_sentence_b, sentence_b_embeddings)
            final_input = torch.cat((moe_output_sentence_a, moe_output_sentence_b), 1)
            final_output = self.fc(final_input)
            return final_output


    def save_pretrained(self, save_path):
        self.encoder.save_pretrained(save_path)
        torch.save(self.moe.state_dict(), f"{save_path}/moe_model.bin")
        torch.save(self.fc.state_dict(), f"{save_path}/fc_model.bin")