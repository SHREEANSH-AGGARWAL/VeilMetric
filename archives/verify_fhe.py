import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from concrete.ml.deployment import FHEModelClient, FHEModelServer
import time
import numpy as np

def verify_vault(agent_name="return", num_samples=5):
  
    
    # 1. Load Test Data
    df = pd.read_csv("enhanced_training_data.csv")
    feature_cols = [c for c in df.columns if c.startswith('w_')] + ['Context_Momentum', 'Context_Vol']
    target_col = f"Target_{agent_name.capitalize()}"
    
    # Grab a tiny random sample of 5 users to test
    test_data = df.sample(num_samples, random_state=42)
    X_test = test_data[feature_cols].values
    y_true = test_data[target_col].values

    # 2. Load the Cryptographic Vaults
    vault_path = f"fhe_vault/{agent_name.lower()}"
    client = FHEModelClient(vault_path)
    server = FHEModelServer(vault_path)

    print("\nGenerating cryptographic keys")
    client.generate_private_and_evaluation_keys()
    serialized_evaluation_keys = client.get_serialized_evaluation_keys()

   
    y_pred = []
 
    start_time = time.time()
    
    for i in range(num_samples):
        # Isolate exactly 1 user at a time
        single_user = X_test[i:i+1] 
        
        # A. CLIENT: Encrypt
        encrypted_features = client.quantize_encrypt_serialize(single_user)
        
        # B. SERVER: Run XGBoost inside the vault
        encrypted_result = server.run(encrypted_features, serialized_evaluation_keys)
        
        # C. CLIENT: Decrypt the answer
        decrypted_result = client.deserialize_decrypt_dequantize(encrypted_result)
        
        # Save the raw number
        y_pred.append(float(np.array(decrypted_result).flatten()[0]))
        print(f" Processed User {i+1}/{num_samples}")

    execution_time = time.time() - start_time
    print(f"\n Total FHE execution time: {execution_time:.2f} seconds.")

    # 4. The Reveal (Accuracy Check)
    y_pred = np.array(y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
   
    print(f"FHE Accuracy (R² Score): {r2 * 100:.2f}%")
    print(f"Average Error Margin:    {mae * 100:.2f}%")
    
    if r2 > 0.85:
        print("Status: PASSED")
    else:
        print("Status: FAILED")

if __name__ == "__main__":
    verify_vault(agent_name="return", num_samples=5)