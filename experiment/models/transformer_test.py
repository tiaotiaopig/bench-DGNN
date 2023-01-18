#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   transformer_test.py
@Time    :   2023/01/11 20:44:29
@Author  :   LiFeng
@Contact :   2807229316@qq.com
@Desc    :   None
'''

import math
import torch
from torch import nn
from torch import Tensor

embedding_layer = nn.Embedding(10, 128)

transformer_layer = nn.Transformer(d_model=128, batch_first=True)

src = torch.LongTensor([[0, 3, 4, 5, 6, 7, 8, 1, 2, 2]])

tgt = torch.LongTensor([[0, 3, 4, 5, 6, 7, 8, 1, 2]])

src_embedding, tgt_embedding = embedding_layer(src), embedding_layer(tgt)

res = transformer_layer(src_embedding, tgt_embedding)

print(res.size())


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
    
pos_layer = PositionalEncoding(d_model = 128)

res_2 = pos_layer(src_embedding)
print(res_2.size())