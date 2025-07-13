class Config():
    def __init__(
            self, seq_length, features, d_model_time, d_model_feature, periods, train_size, test_size, target_col, path,
            batch_size, num_heads, hidden_dim, dropout_attention, dropout_ffn, dropout_cnn, output_model_path, num_epochs,
            cnn_in_channels, cnn_out_channels, cnn_kernel_sizes, learning_rate,
            cnn_strides, cnn_paddings, cnn_pooling, cnn_global_pool, optimizer_factor, optimizer_patience,
            output_attnfeature_path, output_attntime_path, output_hid_path, train_output_file_path, early_stop, seed
            ):

        self.seq_length: int = seq_length
        self.features :int = features
        self.d_model_time: int = d_model_time
        self.d_model_feature : int = d_model_feature
        self.periods: list = periods
        self.train_size : float = train_size
        self.test_size : float = test_size
        self.target_col : str = target_col
        self.path :str = path
        self.batch_size : int = batch_size
        self.num_heads :int = num_heads
        assert d_model_time % num_heads == 0
        assert d_model_feature % num_heads == 0
        # self.dim_head : int = d_model/num_heads
        self.learning_rate : float = learning_rate

        self.dropout_attention : float = dropout_attention
        self.dropout_ffn : float = dropout_ffn
        self.dropout_cnn : float = dropout_cnn

        self.hidden_dim : int = hidden_dim
        self.output_model_path : str = output_model_path
        self.num_epochs : int = num_epochs
        self.cnn_in_channels : int = cnn_in_channels
        self.cnn_out_channels : list = cnn_out_channels
        self.cnn_kernel_sizes : list = cnn_kernel_sizes
        self.cnn_strides : list = cnn_strides
        self.cnn_paddings : list = cnn_paddings
        self.cnn_pooling :list = cnn_pooling
        self.cnn_global_pool :str = cnn_global_pool
        self.output_attnfeature_path: str = output_attnfeature_path
        self.output_attntime_path: str = output_attntime_path
        self.output_hid_path:str=output_hid_path
        self.train_output_file_path:str = train_output_file_path
        self.early_stop : int = early_stop
        self.optimizer_factor : float = optimizer_factor
        self.optimizer_patience : int = optimizer_patience
        self.seed : int = seed

config = Config(
    seq_length=266,
    features=25,
    d_model_feature=128,
    d_model_time=128,
    batch_size = 256,
    periods=[144, 1008],
    train_size=0.8,
    test_size=0.85,
    num_heads=1,
    hidden_dim=64,
    num_epochs=300,
    dropout_attention=0.5,
    dropout_ffn=0.5,
    early_stop=30,
    learning_rate = 1e-5,

    optimizer_factor=0.8,
    optimizer_patience=100,

    seed =42,
    #CNN
    cnn_in_channels=1,
    cnn_out_channels=[64, 32, 32],  # 畳み込み層の出力チャネル
    cnn_kernel_sizes=[3, 3, 3],  # カーネルサイズ
    cnn_strides=[1, 1, 1],  # ストライド
    cnn_paddings=[1, 1, 1],  # パディング
    cnn_pooling=["max", "max", "avg"],  # プーリング方法
    cnn_global_pool="adaptive_avg",  # グローバルプーリング
    dropout_cnn = 0.3,
    
    target_col = 'Appliances',
    path = 'dataset/energydata_complete.csv',
    output_model_path='output/output_model.pth',
    output_attnfeature_path='output/output_attn_feature.pth',
    output_attntime_path='output/output_attn_time.pth',
    output_hid_path='output/output_hid.pth',
)


''''
確認しなきゃいけないこと
transformer内各ステップでの次元
最後の分類方法：予測の仕方・正規化の戻し方
'''