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
import time

from Utilities.Utilities import read_csv, dataloader
from config import config
from DataProcessing.PosEmbedding import SeasonalPositionalEncoding, TokenEmbedding, PositionalEmbedding
from model.TransformerTranspose import TransformerTranspose
from sklearn.metrics import mean_absolute_error, mean_squared_error

path =config.path

x_train, y_train, x_valid, y_valid, x_test, y_test = read_csv(path, train_size=config.train_size, test_size=config.test_size, seq_length=config.seq_length, target_col=config.target_col)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LSTMはpos_encodingいらない
pos_encoding = PositionalEmbedding(d_model=config.d_model, max_len=config.seq_length)
pe_train = pos_encoding(x_train)
pe_valid = pos_encoding(x_valid)
pe_test = pos_encoding(x_test)

train_loader = dataloader(x_train, y_train, batch_size=config.batch_size) #x[batch_size:64, seq_length:432, features:25]
valid_loader = dataloader(x_valid, y_valid, batch_size=config.batch_size) #y[64, 1]
test_loader = dataloader(x_test, y_test, batch_size=config.batch_size)

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


model = TransformerTranspose(
    dim=config.d_model, num_heads=config.num_heads, hidden_dim=config.hidden_dim, 
    seq_length=config.seq_length, features=config.features, 
    dropout_attention=config.dropout_attention, dropout_ffn=config.dropout_ffn, dropout_cnn=config.dropout_cnn,
    in_channels=config.cnn_in_channels, out_channels_list=config.cnn_out_channels, 
    kernel_sizes=config.cnn_kernel_sizes, strides=config.cnn_strides, paddings=config.cnn_paddings,
    pooling_methods=config.cnn_pooling, d_cmodel=config.d_model, d_tmodel=config.d_model, global_pooling=config.cnn_global_pool,

    )
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

loss_list =[]
start_time = time.time()
model.train()
for epoch in range(config.num_epochs):
    total_loss = 0
    for x_batch, y_batch in train_loader:
        y_batch.unsqueeze(1)
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)[0]
        outputs.unsqueeze(1)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 2 == 0:
        print(f'Epoch [{epoch+1}/{config.num_epochs}], Loss: {total_loss/len(train_loader):.4f}')
    loss_list.append(total_loss)
end_time = time.time()
print(f"train finish {end_time-start_time}s")

torch.save(model.state_dict(), config.output_model_path)

model.eval()
predictions = []
actuals = []
all_attns1 = []
all_attns2 = []
all_hid = []

with torch.no_grad():
    for x_batch, y_batch in valid_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs, attns1, attns2, hid = model(x_batch, return_attention=True)
        predictions.extend(outputs.cpu().numpy().squeeze())
        actuals.extend(y_batch.cpu().numpy().squeeze())
        for attn in attns1:
            all_attns1.append(attn.cpu().numpy())
        for attn in attns2:
            all_attns2.append(attn.cpu().numpy())
        all_hid.append(hid.cpu().numpy())

all_attns1 = np.concatenate(all_attns1, axis=0)
all_attns2 = np.concatenate(all_attns2, axis=0)
all_hid = np.concatenate(all_hid, axis=0)

# MAEの計算
mae = mean_absolute_error([a for a in actuals], [p for p in predictions])
print(f"Mean Absolute Error (MAE) on test data: {mae:.4f}")

mse = mean_squared_error([a for a in actuals], [p for p in predictions])
print(f"Mean Absolute Error (MSE) on test data: {mse:.4f}")

x_valid = x_valid.reset_index(drop=True)
y_valid = y_valid.reset_index(drop=True)

nan_padding = [np.nan]*(config.sequence_length)
predictions_withnan = nan_padding + [p for p in predictions[:len(y_test)]]

x_valid["pred"] = predictions_withnan

x_valid.to_csv(config.train_output_file_path, index=False)
np.save(config.output_attnfeature_path, all_attns1)
np.save(config.output_attntime_path, all_attns2)
np.save(config.output_hid_path, all_hid)

print(f"Predictions saved to {config.train_output_file_path}")


