import numpy as np
import pandas as pd
import time

def generate_contextual_data(profiles_csv="user_training_data.csv", market_csv="market_log_returns.csv"):
    print("1. Loading Data...")
    profiles_df = pd.read_csv(profiles_csv)
    weight_cols = [c for c in profiles_df.columns if c.startswith('w_')]
    weights_matrix = profiles_df[weight_cols].values
    num_portfolios = len(weights_matrix)

    returns_df = pd.read_csv(market_csv, index_col=0, parse_dates=True)
    # We use Nifty 50 (index 4) as our 'Market Context' indicator
    nifty_idx = 4 
    returns_array = returns_df.values
    
    window_size = 252 * 4 
    pre_history = 252 # 1 year of context
    
    # Ensure we have enough room for pre-history and the 4-year window
    max_start_idx = len(returns_array) - window_size
    start_indices = np.random.randint(pre_history, max_start_idx, size=num_portfolios)

    # Storage for features and targets
    context_momentum = np.zeros(num_portfolios)
    context_vol = np.zeros(num_portfolios)
    total_returns = np.zeros(num_portfolios)
    max_drawdowns = np.zeros(num_portfolios)
    std_devs = np.zeros(num_portfolios)

    print(f"2. Simulating 1M portfolios with Market Context... 🚀")
    start_time = time.time()
    
    for i in range(num_portfolios):
        s_idx = start_indices[i]
        w = weights_matrix[i]
        
        # --- FEATURE EXTRACTION: The 'Context' ---
        # Look at the 1 year BEFORE the portfolio starts
        past_year_nifty = returns_array[s_idx - pre_history : s_idx, nifty_idx]
        context_momentum[i] = np.sum(past_year_nifty) # Total return of Nifty in past year
        context_vol[i] = np.std(past_year_nifty) * np.sqrt(252) # Volatility of Nifty in past year
        
        # --- TARGET EXTRACTION: The 'Future' ---
        window_ret = returns_array[s_idx : s_idx + window_size]
        port_daily_ret = window_ret @ w
        
        total_returns[i] = np.exp(np.sum(port_daily_ret)) - 1.0
        std_devs[i] = np.std(port_daily_ret) * np.sqrt(252)
        
        cum_ret = np.exp(np.cumsum(port_daily_ret))
        max_drawdowns[i] = abs(np.min((cum_ret - np.maximum.accumulate(cum_ret)) / np.maximum.accumulate(cum_ret)))

    print(f"✅ Finished in {time.time() - start_time:.2f}s")

    # 3. Save Enhanced Dataset
    final_df = profiles_df[weight_cols].copy()
    final_df['Context_Momentum'] = context_momentum
    final_df['Context_Vol'] = context_vol
    final_df['Target_Return'] = total_returns
    final_df['Target_Drawdown'] = max_drawdowns
    final_df['Target_Volatility'] = std_devs
    
    final_df.to_csv("enhanced_training_data.csv", index=False)
    print("🎉 Dataset 'enhanced_training_data.csv' is ready!")

if __name__ == "__main__":
    generate_contextual_data()