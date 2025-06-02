# Data-Driven Portfolio Optimization with DLinear

This repository implements a two-stage deep learning-based framework for portfolio optimization using the **Predict-then-Optimize (PO)** paradigm. The proposed method integrates **DLinear** for return forecasting and **LSTM** for portfolio allocation, directly optimizing the Sharpe Ratio to construct robust investment strategies.

## 🧠 Key Features

- **Forecasting Model**:  
  Employs DLinear, a decomposition-based deep learning model, to predict multi-step asset returns.

- **Optimization Module**:  
  Uses an LSTM network to allocate weights that maximize the Sharpe Ratio based on predicted returns.

- **PO Framework**:  
  Follows a data-driven Predict-then-Optimize pipeline, bridging learning and decision-making.

- **Backtesting Engine**:  
  Simulates investment performance over various asset universes using realistic constraints and metrics.

## 📊 Datasets

- Daily return data from MSCI iShares ETFs
- Collected via [`yfinance`](https://pypi.org/project/yfinance/) between January 2021 and January 2022
- Three ETF universes used:
  - **Universe 1**: 8 manually selected diversified ETFs (e.g., SPY, EFA, AGG)
  - **Universe 2**: 20 randomly sampled ETFs
  - **Universe 3**: All MSCI ETFs launched before 2015

## 🧪 Evaluation

### Forecasting Metrics

- MAE / MAPE / MSE / RMSE  
- Compared DLinear with baseline models such as XGBoost

### Portfolio Performance

- Sharpe Ratio, Sortino Ratio, Volatility, Max Drawdown
- Benchmarked against:
  - Equally Weighted Strategy
  - SPY Index
  - Classical Mean-Variance Portfolio

## 📈 Results

- DLinear outperforms XGBoost in prediction accuracy
- The proposed two-stage strategy achieves:
  - **40% annualized return** in Universe 1
  - Consistent outperformance over the mean-variance strategy
  - Reasonable robustness even in high-dimensional ETF settings

## 🔧 Environment

- **Language**: Python 3.9
- **Libraries**: PyTorch, TensorFlow, scikit-learn, Backtrader

