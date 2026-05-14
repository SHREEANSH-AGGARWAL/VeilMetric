import logging
import time

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

log = logging.getLogger("veilmetric.train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

def train_multi_agent_system(data_path="enhanced_training_data.csv"):
    log.info("Loading training data from %s", data_path)
    start_load = time.time()
    df = pd.read_csv(data_path)
    log.info("Loaded %d rows in %.2fs", len(df), time.time() - start_load)
    
 
    weight_cols = [c for c in df.columns if c.startswith('w_')]
    context_cols = ['Context_Momentum', 'Context_Vol']
    feature_cols = weight_cols + context_cols
    
    X = df[feature_cols]
    log.info("Features: %s", feature_cols)

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

    log.info("Training %d agents with params: %s", len(targets), model_params)
    
    for name, col in targets.items():
        start_agent = time.time()
        log.info("Training %s agent...", name)
        
        y_train = df.loc[indices_train, col]
        y_test = df.loc[indices_test, col]
        
        model = xgb.XGBRegressor(**model_params)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        
        log.info("%s done in %.2fs  |  R2=%.4f  MAE=%.6f", name, time.time() - start_agent, r2, mae)
        
        model.save_model(f"agent_{name.lower()}.json")

    log.info("All agents saved.")

if __name__ == "__main__":
    train_multi_agent_system()

