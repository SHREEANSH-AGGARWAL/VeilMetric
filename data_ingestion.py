
import pandas as pd 
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np 


IndianList=["^NSEI","NAM-INDIA.NS"]
GlobalList=["GC=F","SI=F","BTC-USD","ETH-USD"]     
USDRATEList= ["INR=X"]


combined_data_list = IndianList + GlobalList + USDRATEList 

all_data = yf.download(combined_data_list, period="10y")['Close']
data = all_data.ffill().dropna()


for ticker in IndianList:
    data[f'{ticker}_USD'] = data[ticker] / data['INR=X']


columns_to_keep = GlobalList + [f'{ticker}_USD' for ticker in IndianList]
clean_df = data[columns_to_keep].copy()

log_returns = np.log(clean_df / clean_df.shift(1)).dropna()

log_returns.to_csv("market_log_returns.csv")

# plt.figure(figsize=(12, 6))
# plt.plot(log_returns)

# plt.title('Historical Data')
# plt.xlabel('Date')
# plt.ylabel('Price')

# plt.legend(log_returns.columns)
# plt.show()





