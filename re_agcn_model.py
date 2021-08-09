import copy
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from model.bert import BertPreTrainedModel, BertModel
from model.agcn import TypeGraphConvolution

class ReAgcn(BertPreTrainedModel):
    def __init__(self, config):
        super(ReAgcn, self).__init__(config)
        self.bert = BertModel(config)
        self.dep_type_embedding = nn.Embedding(config.type_num, config.hidden_size, padding_idx=0)
        gcn_layer = TypeGraphConvolution(config.hidden_size, config.hidden_size)
        self.gcn_layer = nn.ModuleList([copy.deepcopy(gcn_layer) for _ in range(config.num_gcn_layers)])
        self.ensemble_linear = nn.Linear(1, config.num_gcn_layers)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*3, config.num_labels)
        self.apply(self.init_bert_weights)

    def valid_filter(self, sequence_output, valid_ids):
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=sequence_output.dtype,
                                   device=sequence_output.device)
        for i in range(batch_size):
            temp = sequence_output[i][valid_ids[i] == 1]
            valid_output[i][:temp.size(0)] = temp
        return valid_output

    def max_pooling(self, sequence, e_mask):
        entity_output = sequence * torch.stack([e_mask] * sequence.shape[-1], 2) + torch.stack(
            [(1.0 - e_mask) * -1000.0] * sequence.shape[-1], 2)
        entity_output = torch.max(entity_output, -2)[0]
        return entity_output.type_as(sequence)

    def extract_entity(self, sequence, e_mask):
        return self.max_pooling(sequence, e_mask)

    def get_attention(self, val_out, dep_embed, adj):
        batch_size, max_len, feat_dim = val_out.shape
        val_us = val_out.unsqueeze(dim=2)
        val_us = val_us.repeat(1,1,max_len,1)
        val_cat = torch.cat((val_us, dep_embed), -1)
        atten_expand = (val_cat.float() * val_cat.float().transpose(1,2))
        attention_score = torch.sum(atten_expand, dim=-1)
        attention_score = attention_score / feat_dim ** 0.5
        # softmax
        exp_attention_score = torch.exp(attention_score)
        exp_attention_score = torch.mul(exp_attention_score.float(), adj.float())
        sum_attention_score = torch.sum(exp_attention_score, dim=-1).unsqueeze(dim=-1).repeat(1,1,max_len)
        attention_score = torch.div(exp_attention_score, sum_attention_score + 1e-10)
        return attention_score

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, e1_mask=None, e2_mask=None,
                dep_adj_matrix=None, dep_type_matrix=None, valid_ids=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        if valid_ids is not None:
            valid_sequence_output = self.valid_filter(sequence_output, valid_ids)
        else:
            valid_sequence_output = sequence_output
        sequence_output = self.dropout(valid_sequence_output)

        dep_type_embedding_outputs = self.dep_type_embedding(dep_type_matrix)
        dep_adj_matrix = torch.clamp(dep_adj_matrix, 0, 1)
        for i, gcn_layer_module in enumerate(self.gcn_layer):
            attention_score = self.get_attention(sequence_output, dep_type_embedding_outputs, dep_adj_matrix)
            sequence_output = gcn_layer_module(sequence_output, attention_score, dep_type_embedding_outputs)
        e1_h = self.extract_entity(sequence_output, e1_mask)
        e2_h = self.extract_entity(sequence_output, e2_mask)

        pooled_output = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            return loss
        else:
            return logits