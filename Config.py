#使用类实现一个配置文件
class Config:
    def __init__(self):
        self.multpath = './Model/'
        self.dimname = 'pollution'
        self.n_predictions = 10
        #skip层参数
        self.skip = 5
        #AR截取时间步
        self.highway_window = 5
        self.dropout = 0.4
        self.optimizer = 'adam'
        self.loss_metric = 'mse'
        self.lstm_batch_size = 16
        self.verbose = 1
        self.epochs = 100

