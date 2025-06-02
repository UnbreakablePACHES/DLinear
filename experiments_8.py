import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta

# 设置随机种子
rdseed = 123
def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
set_all_seeds(rdseed)

# DLinear模型定义
class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, kernel_size):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decomposition = SeriesDecomp(kernel_size)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

class CustomDataset(Dataset):
    def __init__(self, df, seq_len, pred_len):
        self.df = df
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __getitem__(self, index):
        input_begin = index
        input_end = input_begin + self.seq_len
        label_begin = input_end
        label_end = label_begin + self.pred_len
        seq_x = self.df[input_begin:input_end]
        seq_y = self.df[label_begin:label_end]
        return seq_x, seq_y

    def __len__(self):
        return len(self.df) - self.seq_len - self.pred_len + 1

def shift_date(start_date, days_to_shift):
    date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    shifted_date = date_obj + timedelta(days=days_to_shift)
    return shifted_date.strftime('%Y-%m-%d')

def reset_model_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def train_model(df_norm, config, num_epoch=200):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_set = CustomDataset(df=df_norm, seq_len=config['seq_len'], pred_len=config['pred_len'])
    data_loader = DataLoader(data_set, batch_size=config['batch_size'], shuffle=True, drop_last=False)

    model = DLinear(seq_len=config['seq_len'], pred_len=config['pred_len'], kernel_size=config['kernel_size']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.L1Loss().to(device)

    train_loss_ep = []
    for epoch in tqdm(range(num_epoch)):
        train_loss = []
        model.train()
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(loss.item())
        train_loss_ep.append(np.average(train_loss))
    return model

def predict_and_save(model, df_norm, config, history, start_date, end_date, ticker, rdseed):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    RES = pd.DataFrame()
    for i in range(11):
        r = df_norm[-config['seq_len']:, :]
        x = torch.tensor(r[-config['seq_len']:, :]).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            output = model(x.float().to(device))
        r = np.append(r, output.squeeze(0).to("cpu").detach().numpy(), axis=0)
        res_df = pd.DataFrame(r[-config['pred_len']:])
        res_df.columns = ["ret_pred"]
        RES = pd.concat([RES, res_df], axis=0, ignore_index=True) if not RES.empty else res_df
        start_date = shift_date(start_date, 24)
        end_date = shift_date(end_date, 24)
        reset_model_weights(model)
    REAL = history[(history['date'] >= '2021-01-04') & (history['date'] <= '2022-01-19')].copy()
    REAL['pred'] = RES.values
    plot_and_save(REAL, ticker, rdseed)
    REAL.to_csv(f'res/{rdseed}_{ticker}_pred.csv', index=False)
    return REAL

def plot_and_save(REAL, ticker, rdseed):
    plt.figure(figsize=(10, 5))
    plt.plot(REAL['pred'], label='Predicted', color='blue')
    plt.plot(REAL['Return'], label='Real', color='red')
    plt.title(f'{ticker} Predicted vs Real')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'image/{rdseed}_{ticker}.png', dpi=300)

def evaluate_predictions(real, pred):
    mae = mean_absolute_error(real, pred)
    mse = mean_squared_error(real, pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((real - pred) / real)) * 100
    return mae, mse, rmse, mape

def main():
    tickers = ["EEM", "EFA", "JPXN", "SPY", "VTI", "XLK", "AGG", "DBC"]
    start_date = '2019-10-01'
    end_date = '2021-01-01'
    num_epoch = 500
    config = dict(
        batch_size=64,
        kernel_size=21,
        learning_rate=0.0001,
        seq_len=48,
        pred_len=24,
    )

    all_metrics = pd.DataFrame(columns=['Ticker', 'MAE', 'MSE', 'RMSE', 'MAPE'])

    for ticker in tickers:
        print(f'Processing {ticker}')
        history = pd.read_csv(f'data/{ticker}.csv')
        history['date'] = pd.to_datetime(history['date'])

        data = history[(history['date'] >= start_date) & (history['date'] <= end_date)]
        df = data[["Return"]].reset_index(drop=True)
        df_norm = df.to_numpy()

        model = train_model(df_norm, config, num_epoch)
        REAL = predict_and_save(model, df_norm, config, history, start_date, end_date, ticker, rdseed)

        mae, mse, rmse, mape = evaluate_predictions(REAL['Return'], REAL['pred'])
        all_metrics = pd.concat([all_metrics, pd.DataFrame({'Ticker': [ticker], 'MAE': [mae], 'MSE': [mse], 'RMSE': [rmse], 'MAPE': [mape]})], ignore_index=True)

    all_metrics.to_csv(f'res/{len(tickers)}_{rdseed}_metrics.csv', index=False)
    print(all_metrics)

if __name__ == '__main__':
    main()

