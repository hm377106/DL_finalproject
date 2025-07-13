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
    def __init__(self, input_dim, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):

        x, attn_weight = self.attn(x, x, x, need_weights=True)
        x = self.dropout(x)
        return x,  attn_weight


