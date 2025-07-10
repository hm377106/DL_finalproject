class Config():
    def __init__(
            self, seq_length, d_model, periods, train_size, test_size, target_col, path,
            batch_size
            ):
        self.seq_length: int = seq_length
        self.d_model: int = d_model
        self.periods: list = periods
        self.train_size : float = train_size
        self.test_size : float = test_size
        self.target_col : str = target_col
        self.path :str = path
        self.batch_size : int = batch_size


config = Config(
    seq_length=432,
    d_model=128,
    batch_size = 64,
    periods=[144, 1008],
    train_size=0.7,
    test_size=0.85,

    target_col = 'Appliances',
    path = 'dataset/energydata_complete.csv'
)