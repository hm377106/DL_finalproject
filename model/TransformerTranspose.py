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

from layer.attention import FeatureAttention
from layer.ffn import FFN
from model.cnn import CNN

'''
input [time_series: features]→[batch_size: seq_length: feature_dim]→
特徴量方向のself-attention→[time_series]
時系列方向のtransformer encoder→[time_series]
出力[1]
'''

class TransformerTranspose(nn.Module):
    def __init__(
                self, d_model_feature, d_model_time, num_heads, hidden_dim, seq_length, features,
                dropout_attention, dropout_ffn,
                in_channels,         # 入力チャネル数
                out_channels_list,   # 各畳み込み層の出力チャネル数
                kernel_sizes,        # 各畳み込み層のカーネルサイズ
                strides,             # 各畳み込み層のストライド
                paddings,            # 各畳み込み層のパディング
                pooling_methods,     # 各層のプーリング方法
                d_cmodel,            # 入力テンソルの高さ
                d_tmodel,            # 入力テンソルの幅
                dropout_cnn,
                global_pooling="adaptive_avg",  # グローバルプーリング
                output_dim=1
                ):
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
        hidden_dim : int
            Feed-Forward Networkの隠れ層の次元数．
        dropout : float
            Droptou層の確率p．
        """
        super().__init__()
        self.legth2embedding = nn.Linear(seq_length, d_model_time)
        self.features2embedding = nn.Linear(features, d_model_feature)

        self.attn_ln_time = nn.LayerNorm(d_model_time)  # AttentionまえのLayerNorm
        self.attn_ln_feature = nn.LayerNorm(d_model_feature)
        self.attn_feature = FeatureAttention(input_dim=d_model_time, num_heads=num_heads, dropout=dropout_attention)
        self.attn_time_series = FeatureAttention(input_dim=d_model_feature, num_heads=num_heads, dropout=dropout_attention)
        self.ffn_feature = FFN(d_model_feature, hidden_dim, dropout_ffn)
        self.ffn_time = FFN(d_model_time, hidden_dim, dropout_ffn)
        self.ffn_ln_feature = nn.LayerNorm(d_model_feature)  # FFNまえのLayerNorm
        self.ffn_ln_time = nn.LayerNorm(d_model_time)

        self.cnn = CNN(
                in_channels,         # 入力チャネル数
                 out_channels_list,   # 各畳み込み層の出力チャネル数
                 kernel_sizes,        # 各畳み込み層のカーネルサイズ
                 strides,             # 各畳み込み層のストライド
                 paddings,            # 各畳み込み層のパディング
                 pooling_methods,     # 各層のプーリング方法
                 d_cmodel,            # 入力テンソルの高さ
                 d_tmodel,            # 入力テンソルの幅
                 global_pooling,  # グローバルプーリング
                 dropout_cnn,         # ドロップアウト率
                 output_dim=output_dim
                 )

    def forward(self, x, return_attn=False):
        """
        x: (B, N, D)
        B: バッチサイズ
        D: 特徴量数
        dim: 埋め込み次元
        """
        x = self.legth2embedding(x.transpose(1, 2)) #[B:N:D] -> [B:D:N] -> [B:D:d_model_time]
        # for _ in range(2):
        y, attn_feature = self.attn_feature(self.attn_ln_time(x))   #[B:D:d_model_time] -> [B:D:d_model_time]
        x = y+x
        x = self.ffn_time(self.ffn_ln_time(x)) + x

        x = self.features2embedding(x.transpose(1, 2)) #[B:D:dim] -> [B:d_model_time:D]->[B:d_model_time:d_model_feature]
        # for _ in range(2):
        z, attn_time = self.attn_time_series(self.attn_ln_feature(x))
        x = z+x
        out = self.ffn_feature(self.ffn_ln_feature(x)) + x
        out = out.unsqueeze(1)

        output = self.cnn(out)
        if return_attn:
            return output , attn_feature, attn_time, out
        else:
            return output





