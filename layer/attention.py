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
'''
X[batch_num:batch_size:seq_length: feature_dim]
x:batchごとx[batch_size:seq_length:feature_dim]
'''

class FeatureAttention(nn.Module):
    def __init__(self, num_features, seq_length, num_heads, is_feature_attention=False):
        super().__init__()
        if is_feature_attention:
            self.attn = nn.MultiheadAttention(embed_dim=seq_length, num_heads=num_heads, batch_first=True)
        else:
            self.attn = nn.MultiheadAttention(embed_dim=num_features, num_heads=num_heads, batch_first=True)
    def forward(self, x, is_feature_attention=False):
        # x : (batch_size, seq_len, feature_dim)
        if is_feature_attention:
            x = x.transpose(1, 2)
        x, attn_weight = self.attn(x, x, x, need_weights=True)
        if is_feature_attention:    
            x = x.transpose(1, 2)
        return x,  attn_weight


