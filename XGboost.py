import os
import random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

set_all_seeds(rdseed)

def shift_date(start_date, days_to_shift):
    date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    shifted_date = date_obj + timedelta(days=days_to_shift)
    return shifted_date.strftime('%Y-%m-%d')

def train_model_xgb(df_norm, config, num_epoch=200):
    seq_len = config['seq_len']
    pred_len = config['pred_len']
    
    # Prepare dataset
    X, y = [], []
    for i in range(len(df_norm) - seq_len - pred_len + 1):
        X.append(df_norm[i:i+seq_len].flatten())
        y.append(df_norm[i+seq_len:i+seq_len+pred_len].flatten())
    X = np.array(X)
    y = np.array(y)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=rdseed)
    
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=num_epoch, learning_rate=config['learning_rate'])
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True, early_stopping_rounds=10)
    
    return model

def predict_and_save_xgb(model, df_norm, config, history, start_date, end_date, ticker, rdseed):
    seq_len = config['seq_len']
    pred_len = config['pred_len']
    
    RES = pd.DataFrame()
    for i in range(11):
        r = df_norm[-seq_len:, :]
        X_test = r.flatten().reshape(1, -1)
        y_pred = model.predict(X_test)
        r = np.append(r, y_pred.reshape(pred_len, -1), axis=0)
        res_df = pd.DataFrame(r[-pred_len:])
        res_df.columns = ["ret_pred"]
        RES = pd.concat([RES, res_df], axis=0, ignore_index=True) if not RES.empty else res_df
        start_date = shift_date(start_date, 24)
        end_date = shift_date(end_date, 24)
    
    REAL = history[(history['date'] >= '2021-01-04') & (history['date'] <= '2022-01-19')].copy()
    REAL['pred'] = RES.values
    plot_and_save(REAL, ticker, rdseed)
    REAL.to_csv(f'res/XGboost/{rdseed}_{ticker}_pred.csv', index=False)
    return REAL

def plot_and_save(REAL, ticker, rdseed):
    plt.figure(figsize=(10, 5))
    plt.plot(REAL['pred'], label='Predicted', color='blue')
    plt.plot(REAL['Return'], label='Real', color='red')
    plt.title(f'{ticker} Predicted vs Real')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'image/XGboost/{rdseed}_{ticker}.png', dpi=300)

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
    num_epoch = 200
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

        model = train_model_xgb(df_norm, config, num_epoch)
        REAL = predict_and_save_xgb(model, df_norm, config, history, start_date, end_date, ticker, rdseed)

        mae, mse, rmse, mape = evaluate_predictions(REAL['Return'], REAL['pred'])
        all_metrics = pd.concat([all_metrics, pd.DataFrame({'Ticker': [ticker], 'MAE': [mae], 'MSE': [mse], 'RMSE': [rmse], 'MAPE': [mape]})], ignore_index=True)

    all_metrics.to_csv(f'res/XGboost/{len(tickers)}_{rdseed}_metrics.csv', index=False)
    print(all_metrics)

if __name__ == '__main__':
    main()

