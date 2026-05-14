"""
Feature engineering for VeilMetric. Provides two modes:

1. generate_contextual_data()  - Batch: processes thousands of synthetic
   portfolios over random 4-year windows of historical data, extracting
   market context (Nifty 50 momentum + volatility) as features and
   portfolio return/drawdown/volatility as targets.

2. get_live_market_context()   - Live: fetches the latest 1-year Nifty 50
   data via yfinance to compute real-time context features for inference.
"""

import logging
import time

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger("veilmetric.data")

NIFTY_COL_INDEX = 4
TRADING_DAYS_PER_YEAR = 252
WINDOW_YEARS = 4
WINDOW_SIZE = TRADING_DAYS_PER_YEAR * WINDOW_YEARS
PRE_HISTORY = TRADING_DAYS_PER_YEAR


def generate_contextual_data(profiles_csv="user_training_data.csv", market_csv="market_log_returns.csv"):
    """Build the enhanced training dataset by pairing each portfolio with a random market window."""
    log.info("Loading data from %s and %s", profiles_csv, market_csv)
    profiles_df = pd.read_csv(profiles_csv)
    weight_cols = [c for c in profiles_df.columns if c.startswith('w_')]
    weights_matrix = profiles_df[weight_cols].values
    num_portfolios = len(weights_matrix)

    returns_df = pd.read_csv(market_csv, index_col=0, parse_dates=True)
    returns_array = returns_df.values

    max_start_idx = len(returns_array) - WINDOW_SIZE
    start_indices = np.random.randint(PRE_HISTORY, max_start_idx, size=num_portfolios)

    context_momentum = np.zeros(num_portfolios)
    context_vol = np.zeros(num_portfolios)
    total_returns = np.zeros(num_portfolios)
    max_drawdowns = np.zeros(num_portfolios)
    std_devs = np.zeros(num_portfolios)

    log.info("Processing %d portfolio windows...", num_portfolios)
    start_time = time.time()

    for i in range(num_portfolios):
        s_idx = start_indices[i]
        w = weights_matrix[i]

        past_year_nifty = returns_array[s_idx - PRE_HISTORY : s_idx, NIFTY_COL_INDEX]
        context_momentum[i] = np.sum(past_year_nifty)
        context_vol[i] = np.std(past_year_nifty) * np.sqrt(TRADING_DAYS_PER_YEAR)

        window_ret = returns_array[s_idx : s_idx + WINDOW_SIZE]
        port_daily_ret = window_ret @ w

        total_returns[i] = np.exp(np.sum(port_daily_ret)) - 1.0
        std_devs[i] = np.std(port_daily_ret) * np.sqrt(TRADING_DAYS_PER_YEAR)

        cum_ret = np.exp(np.cumsum(port_daily_ret))
        max_drawdowns[i] = abs(np.min(
            (cum_ret - np.maximum.accumulate(cum_ret)) / np.maximum.accumulate(cum_ret)
        ))

    log.info("Finished processing in %.2fs", time.time() - start_time)

    final_df = profiles_df.copy()
    final_df['Context_Momentum'] = context_momentum
    final_df['Context_Vol'] = context_vol
    final_df['Target_Return'] = total_returns
    final_df['Target_Drawdown'] = max_drawdowns
    final_df['Target_Volatility'] = std_devs

    final_df.to_csv("enhanced_training_data.csv", index=False)
    log.info("Saved enhanced training data to enhanced_training_data.csv")


def get_live_market_context():
    """Fetch the trailing 1-year Nifty 50 log-return momentum and annualized volatility."""
    nifty_data = yf.download("^NSEI", period="1y", progress=False)['Close']
    log_returns = np.log(nifty_data / nifty_data.shift(1)).dropna()

    context_momentum = log_returns.sum().item()
    context_vol = (log_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)).item()

    return context_momentum, context_vol


def prepare_live_features(user_weights_dict):
    """Combine user portfolio weights with live market context into a model-ready DataFrame."""
    momentum, vol = get_live_market_context()

    features = pd.DataFrame([{
        "w_GC=F": user_weights_dict.get("Gold", 0),
        "w_SI=F": user_weights_dict.get("Silver", 0),
        "w_BTC-USD": user_weights_dict.get("Bitcoin", 0),
        "w_ETH-USD": user_weights_dict.get("Ethereum", 0),
        "w_^NSEI_USD": user_weights_dict.get("Nifty", 0),
        "w_NAM-INDIA.NS_USD": user_weights_dict.get("Nippon", 0),
        "Context_Momentum": momentum,
        "Context_Vol": vol
    }])

    return features


if __name__ == "__main__":
    generate_contextual_data()