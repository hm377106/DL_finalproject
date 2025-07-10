import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from einops.layers.torch import Rearrange
from einops import rearrange

from ..layer.attention import FeatureAttention as Attention
from ..layer.ffn import FFN

'''
input [time_series: features]→[batch_size: seq_length: feature_dim]→
特徴量方向のself-attention→[time_series]
時系列方向のtransformer encoder→[time_series]
出力[1]
'''

class TransformerTranspose(nn.Module):
    def __init__(self, dim, heads, dim_head, hidden_dim, seq_length, dropout):
        """
        TransformerのEncoder Blockの実装．

        Arguments
        ---------
        dim : int
            埋め込みされた次元数．PatchEmbedのembed_dimと同じ値．
        heads : int
            Multi-Head Attentionのヘッドの数．
        dim_head : int
            Multi-Head Attentionの各ヘッドの次元数．
        mlp_dim : int
            Feed-Forward Networkの隠れ層の次元数．
        dropout : float
            Droptou層の確率p．
        """
        super().__init__()
        self.embedding = nn.Linear(seq_length, dim)

        self.attn_ln = nn.LayerNorm(dim)  # AttentionあとのLayerNorm
        self.attn_feature = Attention(dim, heads, dim_head, dropout, is_feature_attention=False)
        self.attn_time_series = Attention(dim, heads, dim_head, dropout, is_feature_attention=False)
        self.ffn_ln = nn.LayerNorm(dim)  # FFN前のLayerNorm
        self.ffn = FFN(dim, hidden_dim, dropout)

    def forward(self, x, return_attn=False):
        """
        x: (B, N, dim)
        B: バッチサイズ
        N: 系列長
        dim: 埋め込み次元
        """
        y, attn_feature = self.attn_feature(self.embedding(x))  
        x = y+x
        z, attn_time = self.attn_time_series(self.attn_ln(x))
        if return_attn:  # attention mapを返す（attention mapの可視化に利用）
            return attn_feature, attn_time
        
        x = z+x
        out = self.ffn(self.ffn_ln(x)) + x

        return out
