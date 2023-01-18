#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   gclstm2.py
@Time    :   2023/01/06 15:38:31
@Author  :   LiFeng
@Contact :   2807229316@qq.com
@Desc    :   GCLSTM另一种实现
'''
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GCLSTM

class GCLSTM2(nn.Module):
    
    def __init__(self, args, activation) -> None:
        super(GCLSTM2, self).__init__()
        self.activation = activation
        self.K = args.K
        self.recurrent_1 = GCLSTM(args.feats_per_node, args.layer_2_feats, args.K)
        
    def forward(self, edge_index_list, node_feats_list, edge_feats_list, nodes_mask_list) -> torch.Tensor:
        H, C = None, None
        
        for t, edge_index in enumerate(edge_index_list):
            node_feats = node_feats_list[t].to_dense()
            edge_feats = edge_feats_list[t]

            H, C = self.recurrent_1(node_feats, edge_index, edge_feats, H, C)
            
        return F.relu(H)