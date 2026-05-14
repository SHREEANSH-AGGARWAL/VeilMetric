import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import time

def train_multi_agent_system(data_path="enhanced_training_data.csv"):
    print(f" Starting ")
    start_load = time.time()
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows in {time.time() - start_load:.2f} seconds.")
    
 
    weight_cols = [c for c in df.columns if c.startswith('w_')]
    context_cols = ['Context_Momentum', 'Context_Vol']
    feature_cols = weight_cols + context_cols
    
    X = df[feature_cols]
    print(f"Features being used: {feature_cols}")

    # 2. Define our 3 Targets (y)
    targets = {
        "Return": "Target_Return",
        "Drawdown": "Target_Drawdown",
        "Volatility": "Target_Volatility"
    }

    # 3. Consistent Train/Test Split
    X_train, X_test, indices_train, indices_test = train_test_split(
        X, df.index, test_size=0.2, random_state=42
    )

    # 4. FHE-Optimized Hyperparameters
    model_params = {
        'n_estimators': 150, # Increased slightly for better learning
        'max_depth': 6,      # Increased depth to capture regime patterns
        'learning_rate': 0.1, 
        'tree_method': 'hist', 
        'random_state': 42
    }

    print("Training")
    
    for name, col in targets.items():
        start_agent = time.time()
        print(f"   Training {name} Agent...", end=" ", flush=True)
        
        y_train = df.loc[indices_train, col]
        y_test = df.loc[indices_test, col]
        
        model = xgb.XGBRegressor(**model_params)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        
        print(f"Done! ({time.time() - start_agent:.2f}s)")
        print(f"      $R^2$ Accuracy: {r2*100:.2f}% | Avg Error (MAE): {mae:.6f}")
        
        model.save_model(f"agent_{name.lower()}.json")

    print("\n Saved ")

if __name__ == "__main__":
    train_multi_agent_system()

