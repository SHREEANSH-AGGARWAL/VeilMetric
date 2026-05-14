import logging
import os
import time

import pandas as pd
from sklearn.model_selection import train_test_split
from concrete.ml.sklearn import XGBRegressor
from concrete.ml.deployment import FHEModelDev

log = logging.getLogger("veilmetric.fhe")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

def compile_and_export_fhe(data_path="enhanced_training_data.csv"):
    log.info("Starting FHE compilation from %s", data_path)
    df = pd.read_csv(data_path)
    
    # 1. Define Features (Weights + Context)
    feature_cols = [c for c in df.columns if c.startswith('w_')] + ['Context_Momentum', 'Context_Vol']
    X = df[feature_cols]

    targets = {
        "Return": "Target_Return",
        "Drawdown": "Target_Drawdown",
        "Volatility": "Target_Volatility"
    }

    # Create a secure folder for our cryptographic assets
    export_dir = "fhe_vault"
    os.makedirs(export_dir, exist_ok=True)

    for name, col in targets.items():
        log.info("Compiling %s agent", name)
        y = df[col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 2. Initialize Quantized Model 
        # Note: We lower n_estimators to 40 and max_depth to 4 to ensure the FHE execution is fast
        fhe_model = XGBRegressor(n_estimators=30, max_depth=4, n_bits=5, random_state=42)
        
        # 3. Train the Quantized Version
        start_train = time.time()
        log.info("Training quantized model...")
        fhe_model.fit(X_train, y_train)
        log.info("Trained in %.2fs", time.time() - start_train)

        log.info("Compiling FHE circuit...")
        start_compile = time.time()
        fhe_model.compile(X_train.head(1000)) 
        log.info("Compiled in %.2fs", time.time() - start_compile)

        # 5. Export the FHE Deployment Files
        agent_dir = os.path.join(export_dir, name.lower())
        dev = FHEModelDev(agent_dir, fhe_model)
        dev.save()
        log.info("Exported FHE assets to %s", agent_dir)

    log.info("All 3 agents compiled into FHE circuits.")

if __name__ == "__main__":
    compile_and_export_fhe()