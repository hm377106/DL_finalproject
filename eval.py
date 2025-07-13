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
from model.TransformerTranspose import TransformerTranspose

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


model = TransformerTranspose(
    d_model_feature=config.d_model_feature, d_model_time= config.d_model_time, num_heads=config.num_heads, hidden_dim=config.hidden_dim,
    seq_length=config.seq_length, features=config.features,
    dropout_attention=config.dropout_attention, dropout_ffn=config.dropout_ffn, dropout_cnn=config.dropout_cnn,
    in_channels=config.cnn_in_channels, out_channels_list=config.cnn_out_channels,
    kernel_sizes=config.cnn_kernel_sizes, strides=config.cnn_strides, paddings=config.cnn_paddings,
    pooling_methods=config.cnn_pooling, d_cmodel=config.d_model_time, d_tmodel=config.d_model_feature, global_pooling=config.cnn_global_pool,
    )

MODEL_PATH = os.path.join(path, 'output_model.pth')

model.load_state_dict(torch.load(MODEL_PATH))

model.eval()
print('モデルの読み込み完了')


model.eval()
predictions = []
actuals = []
all_attns1 = []
all_attns2 = []
all_hid = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs, attns1, attns2, hid = model(x_batch, return_attn=True)
        predictions.extend(np.atleast_1d(outputs.cpu().numpy().squeeze()))
        actuals.extend(np.atleast_1d(y_batch.cpu().numpy().squeeze()))
        # predictions.extend(outputs.cpu().numpy().squeeze())
        # actuals.extend(y_batch.cpu().numpy().squeeze())
        for attn in attns1:
            all_attns1.append(attn.cpu().numpy())
        for attn in attns2:
            all_attns2.append(attn.cpu().numpy())
        all_hid.append(hid.cpu().numpy())

all_attns1 = np.concatenate(all_attns1, axis=0)
all_attns2 = np.concatenate(all_attns2, axis=0)
all_hid = np.concatenate(all_hid, axis=0)
# print(predictions)
# print(actuals.shape)
# MAEの計算
mae = mean_absolute_error([a for a in actuals], [p for p in predictions])
print(f"Mean Absolute Error (MAE) on test data: {mae:.4f}")
#MSE
mse = mean_squared_error([a for a in actuals], [p for p in predictions])
print(f"Mean Absolute Error (MSE) on test data: {mse:.4f}")

x_test = x_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# nan_padding = [np.nan]*(config.seq_length)
# predictions_withnan = nan_padding + [p for p in predictions]
print(len(predictions))
# y_valid["pred"] = predictions_withnan[:len(y_valid)]
# diff = len(y_test) - len(predictions)
# if diff > 0:
#   predictions = [np.nan]*diff + predictions
# else:
#   predictions = predictions[:len(y_test)]

print(len(y_test))
print(len(predictions))
pred_csv = pd.DataFrame({
    'Truth': [x for x in actuals],
    'Pred' : predictions
})

pred_csv.to_csv(config.train_output_file_path, index=False)
os.makedirs(config.output_attnfeature_path, exist_ok=True)
os.makedirs(config.output_attntime_path, exist_ok=True)
os.makedirs(config.output_hid_path, exist_ok=True)
np.save(config.output_attnfeature_path, all_attns1)
np.save(config.output_attntime_path, all_attns2)
np.save(config.output_hid_path, all_hid)

print(f"Predictions saved to {config.train_output_file_path}")



pred_csv[['Truth', 'Pred']].plot(figsize=(12, 6))

# グラフの装飾
plt.title('truth vs pred')
plt.xlabel('index')
plt.ylabel('log')
plt.grid(True)
plt.legend()
plt.show()



def calculate_inverse_exp(value):
  return np.exp(value) -1
pred_csv['Truth'] = pred_csv['Truth'].apply(calculate_inverse_exp)
pred_csv['Pred'] = pred_csv['Pred'].apply(calculate_inverse_exp)
pred_csv[['Truth', 'Pred']].plot(figsize=(12, 6))

# グラフの装飾
plt.title('truth vs pred')
plt.xlabel('index')
plt.ylabel('Wh')
plt.grid(True)
plt.legend()
plt.show()


pred_csv['loss'] = pred_csv['Truth'] - pred_csv['Pred']
mse = pred_csv['loss'].sum() / pred_csv['loss'].count()
print(mse)