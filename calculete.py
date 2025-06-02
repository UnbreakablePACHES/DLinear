import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_data(filepath: str):
    """
    从文件中加载数据的函数。

    参数：
    filepath (str): 文件路径。

    返回：
    data: 从文件中加载的数据。
    """
    return pd.read_csv(filepath)

def calculate_monthly_avg(pred_data: pd.DataFrame):
    """
    计算每个月的 pred 和 return 平均值。

    参数：
    pred_data (pd.DataFrame): 包含 pred 和 return 列的数据。

    返回：
    monthly_avg (pd.DataFrame): 每个月的 pred 和 return 平均值。
    """
    pred_data['date'] = pd.to_datetime(pred_data['date'])
    pred_data['year_month'] = pred_data['date'].dt.to_period('M')
    monthly_avg = pred_data.groupby('year_month').agg({'pred': 'mean', 'Return': 'mean'}).reset_index()
    return monthly_avg

def calculate_mean_return(real_data: pd.DataFrame):
    """
    计算每个月初的前50天的 return 平均值。

    参数：
    real_data (pd.DataFrame): 包含日期和 return 列的数据。

    返回：
    results_df (pd.DataFrame): 每个月初前50天的 return 平均值。
    """
    real_data['date'] = pd.to_datetime(real_data['date'])
    real_data.set_index('date', inplace=True)
    monthly_starts = real_data.resample('MS').first().index

    results = []
    for start_date in monthly_starts:
        date_range_start = start_date - pd.Timedelta(days=50)
        date_range_end = start_date - pd.Timedelta(days=1)
        filtered_data = real_data.loc[date_range_start:date_range_end]
        mean_return = filtered_data['Return'].mean()
        results.append({'month': start_date, 'mean_return': mean_return})

    results_df = pd.DataFrame(results)
    results_df['month'] = pd.to_datetime(results_df['month'])
    results_df['year_month'] = results_df['month'].dt.strftime('%Y-%m')
    return results_df

def merge_data(monthly_avg: pd.DataFrame, results_df: pd.DataFrame):
    """
    合并 monthly_avg 和 results_df 数据。

    参数：
    monthly_avg (pd.DataFrame): 每个月的 pred 和 return 平均值。
    results_df (pd.DataFrame): 每个月初前50天的 return 平均值。

    返回：
    merged_df (pd.DataFrame): 合并后的数据。
    """
    monthly_avg['year_month'] = monthly_avg['year_month'].astype(str)
    results_df['year_month'] = results_df['year_month'].astype(str)
    merged_df = pd.merge(monthly_avg, results_df[['year_month', 'mean_return']], how='left', on='year_month')
    return merged_df

def calculate_metrics(y_true, y_pred):
    """
    计算 MAE, MSE, RMSE 和 MAPE。

    参数：
    y_true: 实际值。
    y_pred: 预测值。

    返回：
    metrics (dict): 包含 MAE, MSE, RMSE 和 MAPE 的字典。
    """
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
    return metrics

def main():
    # 初始化一个空列表，用于存储每个指标的结果
    results_list = []

    # 加载数据
    tickers = ['IAI', 'IVV', 'ESGU', 'PICK', 'QUAL', 
               'SLV', 'IWB', 'HEWJ', 'RING', 'IAU', 
               'IYY', 'EWT', 'ITOT', 'IWV', 'IAK', 
               'ILCB', 'DIVB', 'ICVT', 'DGRO', 'IFRA']
    for ticker in tickers:
        # 加载预测数据和实际数据
        pred_data = load_data(f'res/123_{ticker}pred.csv')
        real_data = load_data(f'data/{ticker}.csv')

        # 计算每个月的预测值和实际值的平均
        monthly_avg = calculate_monthly_avg(pred_data)

        # 计算每个月初的前50天的实际值的平均
        results_df_ticker = calculate_mean_return(real_data)

        # 合并数据
        merged_df = merge_data(monthly_avg, results_df_ticker)

        # 计算指标
        pred_metrics = calculate_metrics(merged_df['Return'], merged_df['pred'])
        mean_return_metrics = calculate_metrics(merged_df['Return'], merged_df['mean_return'])

        # 将结果添加到结果列表中
        results_list.append({'Ticker': ticker, 'Metric': 'pred_mae', 'Value': pred_metrics['MAE']})
        results_list.append({'Ticker': ticker, 'Metric': 'pred_mse', 'Value': pred_metrics['MSE']})
        results_list.append({'Ticker': ticker, 'Metric': 'pred_rmse', 'Value': pred_metrics['RMSE']})
        results_list.append({'Ticker': ticker, 'Metric': 'pred_mape', 'Value': pred_metrics['MAPE']})

        results_list.append({'Ticker': ticker, 'Metric': 'mean_mae', 'Value': mean_return_metrics['MAE']})
        results_list.append({'Ticker': ticker, 'Metric': 'mean_mse', 'Value': mean_return_metrics['MSE']})
        results_list.append({'Ticker': ticker, 'Metric': 'mean_rmse', 'Value': mean_return_metrics['RMSE']})
        results_list.append({'Ticker': ticker, 'Metric': 'mean_mape', 'Value': mean_return_metrics['MAPE']})

    # 将结果列表转换为 DataFrame
    results_df = pd.DataFrame(results_list)

    return results_df

if __name__ == "__main__":
    results_df = main()
    transposed_df = results_df.pivot(index='Ticker', columns='Metric', values='Value')
    transposed_df.to_csv('Mean vs Pred.csv')



