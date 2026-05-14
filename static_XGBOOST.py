import pandas as pd 
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

data_path = "user_training_data.csv"
print(f"Loading synthetic user data from {data_path}...")
df = pd.read_csv(data_path)
    
# 1. Split Features (Weights) and Target (CVaR)
X = df.drop('target_cvar', axis=1)
y = df['target_cvar']

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training XGBoost Regressor (Tree Method: hist)...")
model = xgb.XGBRegressor(
    n_estimators=150, 
    max_depth=6, 
    learning_rate=0.1, 
    tree_method='hist', # Super fast for large datasets
    random_state=42
)
model.fit(X_train, y_train)

# 3. Evaluate the Model
predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print("\n--- Model Performance ---")
print(f"R-Squared (Accuracy): {r2 * 100:.3f}%")
print(f"Mean Absolute Error: {mae:.6f} (Risk % variance)")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot A: Actual vs. Predicted (The "Diagonal Line" Test)
axes[0].scatter(y_test, predictions, alpha=0.3, color='royalblue', s=10)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_title('Actual vs. Predicted Risk (CVaR)')
axes[0].set_xlabel('True Risk Score')
axes[0].set_ylabel('Model Predicted Risk')
axes[0].grid(True, alpha=0.3)

# Plot B: Residuals (Error Distribution)
errors = y_test - predictions
axes[1].hist(errors, bins=50, color='purple', edgecolor='black', alpha=0.7)
axes[1].set_title('Error Distribution (Residuals)')
axes[1].set_xlabel('Prediction Error')
axes[1].set_ylabel('Frequency')
axes[1].grid(True, alpha=0.3)

# Plot C: Feature Importance by GAIN (True impact on accuracy)
xgb.plot_importance(model, importance_type='gain', ax=axes[2], title='Feature Importance (By Gain)', color='teal')

plt.tight_layout()
plt.show()

model_filename = "veilmetric_v1.json"
model.save_model(model_filename)
print(f" Model successfully saved to: {model_filename}")

expected_features = list(X.columns)
for i, feature in enumerate(expected_features):
    print(f"   [{i}] {feature}")