import os
import math
import random

import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from einops.layers.torch import Rearrange
from einops import rearrange
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from torch import Generator
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from einops.layers.torch import Rearrange
from einops import rearrange
import time

from Utilities.Utilities import read_csv, dataloader, create_sequences, split
from config import config
from DataProcessing.PosEmbedding import SeasonalPositionalEncoding, TokenEmbedding, PositionalEmbedding
from model.TransformerTranspose import TransformerTranspose
from sklearn.metrics import mean_absolute_error, mean_squared_error

path =config.path

x_train, y_train, x_test, y_test = read_csv(path, train_size=config.train_size, test_size=config.test_size, seq_length=config.seq_length, target_col=config.target_col)
x_train_data, y_train_data = create_sequences(x_train, y_train, config.seq_length, config.target_col)
# x_valid_data, y_valid_data =create_sequences(x_valid, y_valid, config.seq_length, config.target_col)
x_test_data, y_test_data = create_sequences(x_test, y_test, config.seq_length, config.target_col)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LSTMはpos_encodingいらない
# pos_encoding = PositionalEmbedding(d_model=config.d_model, max_len=config.seq_length)
# pe_train = pos_encoding(x_train)
# pe_valid = pos_encoding(x_valid)
# pe_test = pos_encoding(x_test)

train_loader, valid_loader = split(x_train_data, y_train_data, train_ratio=config.train_size, seed=config.seed, batch_size=config.batch_size) #x[batch_size:64, seq_length:432, features:25]
# valid_loader = dataloader(x_valid_data, y_valid_data, batch_size=config.batch_size) #y[64, 1]
test_loader = dataloader(x_test_data, y_test_data, batch_size=config.batch_size)

for loader in [train_loader, valid_loader, test_loader]:
    first_batch_x, first_batch_y = next(iter(loader))

    print("--- データローダーの動作確認 ---")
    print(len(loader.dataset))
    print(len(loader))
    print(f"元のx_trainの形状: {x_train.shape}")
    # print(f"元のx_validの形状: {x_valid.shape}")
    print(f"元のx_testの形状: {x_test.shape}")
    print(f"バッチサイズ: {config.batch_size}")
    print(f"データローダーから取り出した最初のバッチの形状:")
    print(f"  - x_batchの形状: {first_batch_x.shape}")
    print(f"  - y_batchの形状: {first_batch_y.shape}")


model = TransformerTranspose(
    d_model_feature=config.d_model_feature, d_model_time= config.d_model_time, num_heads=config.num_heads, hidden_dim=config.hidden_dim,
    seq_length=config.seq_length, features=config.features,
    dropout_attention=config.dropout_attention, dropout_ffn=config.dropout_ffn, dropout_cnn=config.dropout_cnn,
    in_channels=config.cnn_in_channels, out_channels_list=config.cnn_out_channels,
    kernel_sizes=config.cnn_kernel_sizes, strides=config.cnn_strides, paddings=config.cnn_paddings,
    pooling_methods=config.cnn_pooling, d_cmodel=config.d_model_time, d_tmodel=config.d_model_feature, global_pooling=config.cnn_global_pool,
    )
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)

loss_list =[]
valid_loss_list=[]
start_time = time.time()
model.train()
early_stop=0
early_stop_limit=config.early_stop
best_valid_loss=100000000000000
for epoch in range(config.num_epochs):
    total_loss = 0.0
    for x_batch, y_batch in train_loader:
        y_batch=y_batch.unsqueeze(1)
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        outputs=outputs.unsqueeze(1)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    valid_loss=0.0

    with torch.no_grad():
      for x_batch, y_batch in valid_loader:
        y_batch=y_batch.unsqueeze(1)
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        valid_loss+=loss.item()

    if epoch % 2 == 0:
        print(f'Current LR : {scheduler.get_last_lr()[0]:.6f}')
        print(f'Epoch Train [{epoch+1}/{config.num_epochs}], Loss: {total_loss/len(train_loader):.4f}')
        print(f'Epoch Valid [{epoch+1}/{config.num_epochs}], Loss: {valid_loss/len(valid_loader):.4f}')

    if valid_loss < best_valid_loss:
      best_valid_loss = valid_loss
      early_stop=0
    else:
      early_stop+=1
    loss_list.append(total_loss/len(train_loader))
    valid_loss_list.append(valid_loss/len(valid_loader))

    if early_stop >= early_stop_limit:
      print(f'early stoppint at Epoch {epoch+1}')
      break
    scheduler.step(valid_loss)

end_time = time.time()
print(f"train finish {end_time-start_time}s")

torch.save(model.state_dict(), config.output_model_path)

