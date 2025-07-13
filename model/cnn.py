import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, 
                 in_channels,         # 入力チャネル数
                 out_channels_list,   # 各畳み込み層の出力チャネル数
                 kernel_sizes,        # 各畳み込み層のカーネルサイズ
                 strides,             # 各畳み込み層のストライド
                 paddings,            # 各畳み込み層のパディング
                 pooling_methods,     # 各層のプーリング方法
                 d_cmodel,            # 入力テンソルの高さ
                 d_tmodel,            # 入力テンソルの幅
                 global_pooling="adaptive_avg",  # グローバルプーリング
                 dropout=0.2,         # ドロップアウト率
                 output_dim=1):       # 出力次元数
                                                                
        super(CNN, self).__init__()
        
        layers = []
        current_in_channels = in_channels  # 最初の入力チャネル数
        current_h, current_w = d_cmodel, d_tmodel  # 初期の高さと幅

        # 畳み込み層とプーリング層を動的に構築
        for out_channels, kernel_size, stride, padding, pooling in zip(
            out_channels_list, kernel_sizes, strides, paddings, pooling_methods
        ):
            # 畳み込み層
            layers.append(
                nn.Conv2d(
                    current_in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            # 出力サイズを計算
            current_h = (current_h + 2 * padding - kernel_size) // stride + 1
            current_w = (current_w + 2 * padding - kernel_size) // stride + 1

            # プーリング層
            if pooling == "max":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                current_h //= 2
                current_w //= 2
            elif pooling == "avg":
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
                current_h //= 2
                current_w //= 2

            # 入力チャネル数を更新
            current_in_channels = out_channels

        # グローバルプーリング
        if global_pooling == "adaptive_avg":
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        elif global_pooling == "adaptive_max":
            layers.append(nn.AdaptiveMaxPool2d((1, 1)))

        # 畳み込みブロックをシーケンシャル化
        self.cnn = nn.Sequential(*layers)

        # 全結合層
        self.fc = nn.Linear(current_in_channels, output_dim)

    def forward(self, x):
        # 入力テンソルの形状: (batch_size, in_channels, d_cmodel, d_tmodel)
        # 例: (8, 1, 32, 32) （バッチサイズ=8、チャネル数=1、高さ=32、幅=32）
        
        x = self.cnn(x)  
        # 畳み込み + プーリング後のテンソル形状: (batch_size, out_channels_last, h_last, w_last)
        # 例: (8, 128, 4, 4) （最後の畳み込み層の出力チャネル数=128、高さと幅は縮小）
        
        x = x.view(x.size(0), -1)  
        # Flatten後のテンソル形状: (batch_size, out_channels_last * h_last * w_last)
        # 例: (8, 128 * 4 * 4) → (8, 2048) （全ての空間次元を1次元にまとめる）
        
        x = self.fc(x)  
        # 全結合層後のテンソル形状: (batch_size, output_dim)
        # 例: (8, 1) （出力次元=1なら1値のスカラーを出力）
        
        return x

