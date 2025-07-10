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

# from ..config import config

def read_csv(path, train_size, test_size, seq_length, target_col):
    df = pd.read_csv(path)

    df['date']=pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)        

    drop_cols = ['date', 'rv1', 'rv2']
    df = df.drop(columns=drop_cols)
    train_length = int(len(df)*train_size)
    test_length = int(len(df)*test_size)

    train_df = df[:train_length]
    valid_df = df[train_length:test_length]
    test_df = df[test_length:]

    scaler = StandardScaler()
    train_df = scaler.fit_transform(train_df)
    valid_df = scaler.fit_transform(valid_df)
    test_df = scaler.fit_transform(test_df)

    train_df = pd.DataFrame(train_df, columns=df.columns)
    valid_df = pd.DataFrame(valid_df, columns=df.columns)
    test_df = pd.DataFrame(test_df, columns=df.columns)

    x_train, y_train = create_sequences(train_df, seq_length, target_col)
    x_valid, y_valid = create_sequences(valid_df, seq_length, target_col)
    x_test, y_test = create_sequences(test_df, seq_length, target_col)

    x_train = torch.from_numpy(x_train).float()
    x_valid = torch.from_numpy(x_valid).float()
    x_test = torch.from_numpy(x_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_valid = torch.from_numpy(y_valid).float()
    y_test= torch.from_numpy(y_test).float()

    return x_train, y_train, x_valid, y_valid, x_test, y_test

def create_sequences(df, seq_length, target_col):
    Features = []
    Target = []
    feature_only = df.drop(columns=target_col)
    for i in range(seq_length, len(feature_only)):
        Features.append(feature_only.iloc[i-seq_length:i].values)
        Target.append(df.iloc[i][target_col])

    Features = np.array(Features)
    Target = np.array(Target).reshape(-1, 1)
    return Features, Target


def dataloader(x_data, y_data, batch_size, shuffle=True):
    dataset = TensorDataset(x_data, y_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
