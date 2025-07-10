# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.dates as mdates

def plot_time_series(df, time_col, value_col):
    plt.figure(figsize=(12, 5))
    for col in value_col:
        plt.plot(df[time_col], df[col], marker='o', linestyle='-', label=col)
    plt.title(f"{value_col} over {time_col}")
    plt.xlabel(time_col)
    plt.ylabel(value_col)
    plt.grid(True)
    plt.legend()
    plt.show()

def calculate_correlation(df, col1, col2):

    correlation = df[[col1, col2]].corr().iloc[0, 1]
    print(f"correlation (Pearson): {col1} と {col2} = {correlation:.4f}")
    return correlation
    
def plot_diff(df, time_col, value_col):
    plt.figure(figsize=(12, 5))
    for col in value_col:
        diff = df[col].diff() 
        plt.plot(df[time_col], diff, marker='o', linestyle='-', label=col)
    plt.title(f"{value_col}.diff over {time_col}")
    plt.xlabel(time_col)
    plt.ylabel(f"{value_col}.diff")
    plt.legend()
    plt.grid(True)
    plt.show() 

def plot_correlation_heatmap(df, cols=None):
    if cols is not None:
        df_drop = df.drop(cols, axis=1)
    else:
        df_drop=df
    corr = df_drop.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, fmt=".2f", cmap='coolwarm', square=True, cbar=True)
    plt.title("Correlation Heatmap")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

def plot_time_series_bar(df, column, time_col='date', title='Time Series Bar Chart'):
    """
    時系列に沿った棒グラフを表示
    
    Parameters:
    - df: DataFrame
    - column: 縦軸にする1列の名前（数値列）
    - time_col: 横軸にする日付・時刻列（デフォルト 'date'）
    - title: グラフのタイトル
    """
    plt.figure(figsize=(12, 5))
    plt.bar(df[time_col], df[column], width=0.8, align='center')
    plt.xlabel('Time')
    plt.ylabel(column)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True, axis='y')
    plt.show()


def plot_by_year(df, time_val, value):
    plt.figure(figsize=(12, 5))
    plt.plot(df[time_val], df[value])

    # x軸のフォーマットを「年」に設定
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())       # 年ごとの目盛り
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # 年だけ表示
    plt.xlabel('Year')
    plt.ylabel('Appliances')
    plt.title('Appliance usage over years')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

def plot_time_range(df, columns, start, end, time_col='date', title='Time Series (Subset)'):
    df_filtered = df[(df[time_col] >= start) & (df[time_col] <= end)]

    plt.figure(figsize=(12, 5))
    for col in columns:
        plt.plot(df_filtered[time_col], df_filtered[col], label=col)

    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

