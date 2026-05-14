import pandas as pd
import numpy as np
import xgboost as xgb


returns = pd.read_csv("market_log_returns.csv", index_col=0, parse_dates=True)


model = xgb.XGBRegressor()
model.load_model("veilmetric_v1.json")


new_weights = [[15/100,5/100,0,0,70/100,10/100]]
new_user_df = pd.DataFrame(new_weights, columns=[f'w_{c}' for c in returns.columns])

predicted_risk = model.predict(new_user_df)[0]

def true_cvar(w, ret_data, alpha=0.05):
    port_ret = ret_data @ w
    threshold = np.percentile(port_ret, alpha * 100)
    return abs(port_ret[port_ret <= threshold].mean())

actual_risk = true_cvar(new_weights[0], returns)

print(f"Model Prediction: {predicted_risk * 100:.3f}%")
print(f"True Math Risk:   {actual_risk * 100:.3f}%")
print(f"Difference:       {abs(predicted_risk - actual_risk) * 100:.4f}%")