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

from Utilities.Utilities import read_csv, dataloader
from config import config
from DataProcessing.PosEmbedding import SeasonalPositionalEncoding, TokenEmbedding, PositionalEmbedding
from model.TransformerTranspose import TransformerTranspose

path =config.path

x_train, y_train, x_valid, y_valid, x_test, y_test = read_csv(path, train_size=config.train_size, test_size=config.test_size, seq_length=config.seq_length, target_col=config.target_col)

#LSTMはpos_encodingいらない
pos_encoding = PositionalEmbedding(d_model=config.d_model, max_len=config.seq_length)
pe_train = pos_encoding(x_train)
pe_valid = pos_encoding(x_valid)
pe_test = pos_encoding(x_test)

train_loader = dataloader(x_train, y_train, batch_size=config.batch_size) #x[batch_size:64, seq_length:432, features:25]
valid_loader = dataloader(x_valid, y_valid, batch_size=config.batch_size) #y[64, 1]
test_loader = dataloader(x_test, y_test, batch_size=config.batch_size)

model = TransformerTranspose(
    dim=config.d_model, heads=config.num_heads, dim_head=config.dim_head, 
    mlp_dim=config.mlp_dim, seq_length=config.seq_length, dropout=config.dropout
    )

for loader in [train_loader, valid_loader, test_loader]:
    first_batch_x, first_batch_y = next(iter(loader))

    print("--- データローダーの動作確認 ---")
    print(len(loader.dataset))
    print(len(loader))
    print(f"元のx_trainの形状: {x_train.shape}")
    print(f"元のx_validの形状: {x_valid.shape}")
    print(f"元のx_testの形状: {x_test.shape}")
    print(f"バッチサイズ: {config.batch_size}")
    print(f"データローダーから取り出した最初のバッチの形状:")
    print(f"  - x_batchの形状: {first_batch_x.shape}")
    print(f"  - y_batchの形状: {first_batch_y.shape}")

