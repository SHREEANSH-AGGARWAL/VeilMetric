import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import time

def train_multi_agent_system(data_path="enhanced_training_data.csv"):
    print(f"🚀 Loading 1 Million Simulated Profiles from {data_path}... ⏳")
    start_load = time.time()
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows in {time.time() - start_load:.2f} seconds.")
    
    # 1. Identify Features (X) - All columns starting with 'w_'
    feature_cols = [c for c in df.columns if c.startswith('w_')]
    X = df[feature_cols]

    # 2. Define our 3 Targets (y)
    targets = {
        "Return": "Target_TotalReturn",
        "Drawdown": "Target_MaxDrawdown",
        "Volatility": "Target_StdDev"
    }

    # 3. Consistent Train/Test Split (80% Train, 20% Test)
    # Using random_state=42 ensures all 3 agents are tested on the exact same 'unseen' users
    X_train, X_test, indices_train, indices_test = train_test_split(
        X, df.index, test_size=0.2, random_state=42
    )

    # 4. FHE-Optimized Hyperparameters
    # We keep depth shallow (5) to ensure the encrypted circuit isn't too slow later
    model_params = {
        'n_estimators': 100, 
        'max_depth': 5, 
        'learning_rate': 0.1, 
        'tree_method': 'hist', # Essential for speed on 1M rows
        'random_state': 42
    }

    print("\n🧠 Training Specialized Agents...")
    
    for name, col in targets.items():
        start_agent = time.time()
        print(f"   Training {name} Agent...", end=" ", flush=True)
        
        y_train = df.loc[indices_train, col]
        y_test = df.loc[indices_test, col]
        
        # Initialize and train
        model = xgb.XGBRegressor(**model_params)
        model.fit(X_train, y_train)
        
        # Evaluation
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        
        print(f"Done! ({time.time() - start_agent:.2f}s)")
        print(f"      $R^2$ Accuracy: {r2*100:.2f}% | Avg Error (MAE): {mae:.6f}")
        
        # 5. Save Model as JSON for FHE Compiler
        model_file = f"agent_{name.lower()}.json"
        model.save_model(model_file)

    print("\n✅ SUCCESS: All 3 agents saved.")
    print("Files created: agent_return.json, agent_drawdown.json, agent_volatility.json")

if __name__ == "__main__":
    train_multi_agent_system()