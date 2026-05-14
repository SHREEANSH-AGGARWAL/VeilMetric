"""
Fetches 10 years of historical price data for 6 assets (Gold, Silver, BTC, ETH,
Nifty 50, Nippon India) via yfinance, converts Indian tickers to USD, computes
log returns, and saves the result as market_log_returns.csv.
"""

import logging

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger("veilmetric.ingest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

INDIAN_TICKERS = ["^NSEI", "NAM-INDIA.NS"]
GLOBAL_TICKERS = ["GC=F", "SI=F", "BTC-USD", "ETH-USD"]
FX_TICKER = ["INR=X"]

all_tickers = INDIAN_TICKERS + GLOBAL_TICKERS + FX_TICKER

log.info("Downloading 10-year close data for %d tickers", len(all_tickers))
all_data = yf.download(all_tickers, period="10y")["Close"]
data = all_data.ffill().dropna()

for ticker in INDIAN_TICKERS:
    data[f"{ticker}_USD"] = data[ticker] / data["INR=X"]

columns_to_keep = GLOBAL_TICKERS + [f"{t}_USD" for t in INDIAN_TICKERS]
clean_df = data[columns_to_keep].copy()

log_returns = np.log(clean_df / clean_df.shift(1)).dropna()
log_returns.to_csv("market_log_returns.csv")
log.info("Saved %d rows of log returns to market_log_returns.csv", len(log_returns))
