# can also use only dirchlet distribution to make profiles
# but going by archetype is preferred here to account for exterme cases as well 


import pandas as pd 
import numpy as np
input_csv="market_log_returns.csv"
total_portfolios=1,000,000
returns = pd.read_csv(input_csv, index_col=0, parse_dates=True)
assets = returns.columns
num_assets = len(assets)

weights_list = []
n_random = int(total_portfolios * 0.4)
weights_list.append(np.random.dirichlet(np.ones(num_assets), size=n_random))


crypto_indices = [i for i, col in enumerate(assets) if 'BTC' in col or 'ETH' in col]
for _ in range(int(total_portfolios * 0.2)):
    w = np.random.rand(num_assets) * 0.1 # Small random weights for everything
    for idx in crypto_indices:
        w[idx] = np.random.uniform(0.4, 0.9) # Massive weights for crypto
    weights_list.append(w / w.sum()) # Normalize to 1.0
        
    # Bucket C: "Conservative TradFi" (20% of data)
    # High weights in G-Sec (NAM-INDIA) and Gold (GC=F)
safe_indices = [i for i, col in enumerate(assets) if 'NAM' in col or 'GC=F' in col]
for _ in range(int(total_portfolios * 0.2)):
    w = np.random.rand(num_assets) * 0.1
    for idx in safe_indices:
        w[idx] = np.random.uniform(0.4, 0.9)
    weights_list.append(w / w.sum())

    # Bucket D: "Single Asset Maxis" (20% of data)
    # Simulates users who put 95%+ of their net worth in just one asset
for _ in range(int(total_portfolios * 0.2)):
    w = np.random.rand(num_assets) * 0.05
    random_maxi_idx = np.random.randint(0, num_assets)
    w[random_maxi_idx] = 0.95
    weights_list.append(w / w.sum())

all_weights = np.vstack(weights_list)
x_df = pd.DataFrame(all_weights, columns=[f'w_{c}' for c in assets])


def calculate_cvar(w, ret_data,alpha=0.05):
    port_ret = ret_data @ w
    threshold = np.percentile(port_ret, alpha * 100)
    return abs(port_ret[port_ret <= threshold].mean())

print(f"Calculating CVaR for {len(x_df)} portfolios")
y = x_df.apply(lambda row: calculate_cvar(row.values, returns), axis=1)

x_df['target_cvar'] = y
x_df.to_csv("user_training_data.csv", index=False)
