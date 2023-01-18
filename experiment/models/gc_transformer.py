#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   gc_transformer.py
@Time    :   2023/01/06 13:29:48
@Author  :   LiFeng
@Contact :   2807229316@qq.com
@Desc    :   GCN + TransformerEncoder
'''
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import GCNConv

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class GCNTransformer(nn.Module):
    def __init__(self, args, activation):
        super(GCNTransformer, self).__init__()
        self.activation = activation
        
        self.conv1 = GCNConv(args.feats_per_node, args.d_model)
    
        fea_size = args.num_hist_steps * args.d_model
        self.pos_layer = PositionalEncoding(d_model=fea_size)
        
        encoder_layers = TransformerEncoderLayer(fea_size, nhead = 8, batch_first=True)
        self.transformer_layer = TransformerEncoder(encoder_layers, num_layers=6)
               
    def forward(self, edge_index_list, node_feats_list, edge_feats_list, nodes_mask_list):
       
        # 对一批图进行先卷积,在列维度进行连接
        series_embedding = []
        
        for t, edge_index in enumerate(edge_index_list):
            node_feats = node_feats_list[t].to_dense()
            edge_feats = edge_feats_list[t]

            x = self.conv1(node_feats, edge_index, edge_feats)
            series_embedding.append(F.relu(x))
            
        series_embedding = torch.cat(series_embedding, dim=1)
        # 进行位置编码
        series_embedding = torch.unsqueeze(series_embedding, dim=1)
        input_embedding = self.pos_layer(series_embedding)
        # transformer encoder
        x = self.transformer_layer(torch.squeeze(input_embedding, dim=1))
        
        
        return F.relu(x)