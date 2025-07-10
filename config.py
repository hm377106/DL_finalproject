class Config():
    def __init__(
            self, seq_length, d_model, periods, train_size, test_size, target_col, path,
            batch_size, num_heads, hidden_dim, dropout
            ):
        self.seq_length: int = seq_length
        self.d_model: int = d_model
        self.periods: list = periods
        self.train_size : float = train_size
        self.test_size : float = test_size
        self.target_col : str = target_col
        self.path :str = path
        self.batch_size : int = batch_size
        self.num_heads :int = num_heads
        assert d_model & num_heads == 0
        self.dim_head : int = d_model/num_heads
        self.dropout : int = dropout
        self.hidden_dim : int = hidden_dim

config = Config(
    seq_length=432,
    d_model=128,
    batch_size = 64,
    periods=[144, 1008],
    train_size=0.7,
    test_size=0.85,
    num_heads=2,    
    hidden_dim=256,
    
    

    target_col = 'Appliances',
    path = 'dataset/energydata_complete.csv'
)


''''
確認しなきゃいけないこと
transformer内各ステップでの次元
最後の分類方法：予測の仕方・正規化の戻し方
'''