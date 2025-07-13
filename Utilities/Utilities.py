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
# from ..config import config

def read_csv(path, train_size, test_size, seq_length, target_col):
    df = pd.read_csv(path)

    df['date']=pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    drop_cols = ['date', 'rv1', 'rv2']
    df = df.drop(columns=drop_cols)
    train_length = int(len(df)*train_size)
    test_length = int(len(df)*test_size)

    train_df = df[:test_length]
    test_df = df[test_length:]

    scaler = StandardScaler()
    # train_df = scaler.fit_transform(train_df)
    # valid_df = scaler.fit_transform(valid_df)
    # test_df = scaler.fit_transform(test_df)

    train_df = pd.DataFrame(train_df, columns=df.columns)
    # valid_df = pd.DataFrame(valid_df, columns=df.columns)
    test_df = pd.DataFrame(test_df, columns=df.columns)

    x_train, y_train = separate_target_feature(train_df, target_col)
    # x_valid, y_valid = separate_target_feature(valid_df, target_col)
    x_test, y_test = separate_target_feature(test_df, target_col)

    x_train_scaler = scaler.fit_transform(x_train)
    # x_valid_scaler = scaler.fit_transform(x_valid)
    x_test_scaler = scaler.fit_transform(x_test)

    x_train_df = pd.DataFrame(x_train_scaler, columns=x_train.columns)
    # x_valid_df = pd.DataFrame(x_valid_scaler, columns=x_valid.columns)
    x_test_df = pd.DataFrame(x_test_scaler, columns=x_test.columns)
    return x_train_df, y_train, x_test_df, y_test

def separate_target_feature(df, target_col):
  target_df=df[target_col]
  feature_df=df.drop(columns=target_col)
  return feature_df, target_df

def create_sequences(df_feature, df_target, seq_length, target_col):
    Features = []
    Target = []
    feature_only = df_feature
    for i in range(seq_length, len(feature_only)):
          Features.append(feature_only.iloc[i-seq_length:i].values)
          Target.append(df_target.iloc[i])

    Features = np.array(Features)
    Target = np.array(Target).reshape(-1, 1)
    return Features, Target

def split(x_train_df, y_train_df, train_ratio, seed, batch_size):
  x_data = torch.from_numpy(x_train_df).float()
  y_data = np.log1p(torch.from_numpy(y_train_df).float())
  dataset = TensorDataset(x_data, y_data)
  train_len = int(len(dataset) * train_ratio)
  valid_len = len(dataset) - train_len
  train_dataset, valid_dataset = random_split(dataset, [train_len, valid_len], generator=Generator().manual_seed(seed))
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
  return train_loader, valid_loader

def dataloader(x_data, y_data, batch_size, shuffle=False):

    x_data = torch.from_numpy(x_data).float()
    y_data= np.log1p(torch.from_numpy(y_data).float())
    dataset = TensorDataset(x_data, y_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
