"""
Weight Prediction Model Training Script
========================================
This script trains machine learning models to predict weight variation
based on lifestyle factors including calories, physical activity, sleep,
screen time, and intermittent fasting patterns.

Author: Data Science Project
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import pickle
import json
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD AND EXPLORE DATA
# ============================================================================
print("=" * 70)
print("WEIGHT PREDICTION MODEL TRAINING")
print("=" * 70)

data = pd.read_csv("data.csv")

print(f"\n✓ Dataset loaded: {len(data)} records")
print(f"✓ Features: {list(data.columns[:-1])}")
print(f"✓ Target variable: WeightChange")

# Data statistics
print(f"\nDataset Statistics:")
print(f"  Shape: {data.shape}")
print(f"  Missing values: {data.isnull().sum().sum()}")
print(f"  Weight change range: {data['WeightChange'].min():.2f} to {data['WeightChange'].max():.2f} kg")

# ============================================================================
# 2. FEATURE CORRELATION ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("FEATURE CORRELATION ANALYSIS")
print("=" * 70)

correlations = data.corr()['WeightChange'].drop('WeightChange').sort_values(ascending=False)
print("\nFeature Correlations with Weight Change:")
for feature, corr in correlations.items():
    direction = "📈" if corr > 0 else "📉"
    print(f"  {direction} {feature:15s}: {corr:+.4f}")

# ============================================================================
# 3. DATA PREPARATION
# ============================================================================
print("\n" + "=" * 70)
print("DATA PREPARATION")
print("=" * 70)

X = data[['Calories', 'Steps', 'Sleep', 'ScreenTime', 'FastingHours']]
y = data['WeightChange']

print(f"\n✓ Features (X) shape: {X.shape}")
print(f"✓ Target (y) shape: {y.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n✓ Training set: {len(X_train)} samples")
print(f"✓ Testing set: {len(X_test)} samples")

# ============================================================================
# 4. TRAIN LINEAR REGRESSION MODEL
# ============================================================================
print("\n" + "=" * 70)
print("LINEAR REGRESSION MODEL")
print("=" * 70)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_r2 = r2_score(y_test, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))

print(f"\n✓ Model trained successfully!")
print(f"\nPerformance Metrics:")
print(f"  MAE (Mean Absolute Error):    {lr_mae:.4f} kg")
print(f"  RMSE (Root Mean Squared):     {lr_rmse:.4f} kg")
print(f"  R² Score:                     {lr_r2:.4f}")
print(f"  Accuracy:                     {(1 - (lr_mae / abs(y_test).mean())) * 100:.1f}%")

print(f"\nFeature Coefficients:")
for feature, coef in zip(X.columns, lr_model.coef_):
    direction = "↑" if coef > 0 else "↓"
    print(f"  {direction} {feature:15s}: {coef:+.6f}")

# ============================================================================
# 5. TRAIN RANDOM FOREST MODEL
# ============================================================================
print("\n" + "=" * 70)
print("RANDOM FOREST MODEL")
print("=" * 70)

rf_model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=15)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print(f"\n✓ Model trained successfully!")
print(f"\nPerformance Metrics:")
print(f"  MAE (Mean Absolute Error):    {rf_mae:.4f} kg")
print(f"  RMSE (Root Mean Squared):     {rf_rmse:.4f} kg")
print(f"  R² Score:                     {rf_r2:.4f}")
print(f"  Accuracy:                     {(1 - (rf_mae / abs(y_test).mean())) * 100:.1f}%")

print(f"\nFeature Importance:")
importance_dict = dict(zip(X.columns, rf_model.feature_importances_))
for feature in sorted(importance_dict, key=importance_dict.get, reverse=True):
    importance = importance_dict[feature]
    bar = "█" * int(importance * 50)
    print(f"  {feature:15s}: {bar} {importance:.4f} ({importance*100:.1f}%)")

# ============================================================================
# 6. MODEL COMPARISON
# ============================================================================
print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

if rf_r2 > lr_r2:
    print(f"\n🏆 BEST MODEL: Random Forest")
    best_model = rf_model
    best_name = "Random Forest"
else:
    print(f"\n🏆 BEST MODEL: Linear Regression")
    best_model = lr_model
    best_name = "Linear Regression"

print(f"\nComparison Summary:")
print(f"  Linear Regression R²:  {lr_r2:.4f}")
print(f"  Random Forest R²:      {rf_r2:.4f}")
print(f"  R² Improvement:        {(rf_r2 - lr_r2):.4f}")
print(f"  MAE Improvement:       {(lr_mae - rf_mae):.4f} kg")

# ============================================================================
# 7. SAVE MODELS AND RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("SAVING MODELS AND RESULTS")
print("=" * 70)

# Save best model
pickle.dump(best_model, open("model.pkl", "wb"))
print(f"\n✓ Best model saved: model.pkl ({best_name})")

# Save both models for comparison
pickle.dump(lr_model, open("model_lr.pkl", "wb"))
pickle.dump(rf_model, open("model_rf.pkl", "wb"))
print(f"✓ Both models saved for reference")

# Save results as JSON for dashboard
results = {
    'dataset_size': len(data),
    'features': list(X.columns),
    'correlations': {str(k): float(v) for k, v in correlations.items()},
    'linear_regression': {
        'mae': round(float(lr_mae), 4),
        'rmse': round(float(lr_rmse), 4),
        'r2': round(float(lr_r2), 4),
        'accuracy': round(float((1 - (lr_mae / abs(y_test).mean())) * 100), 1)
    },
    'random_forest': {
        'mae': round(float(rf_mae), 4),
        'rmse': round(float(rf_rmse), 4),
        'r2': round(float(rf_r2), 4),
        'accuracy': round(float((1 - (rf_mae / abs(y_test).mean())) * 100), 1)
    },
    'feature_importance': {str(k): round(float(v), 4) for k, v in importance_dict.items()},
    'best_model': best_name
}

with open('model_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved: model_results.json")

# ============================================================================
# 8. SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING COMPLETE ✓")
print("=" * 70)
print(f"\nFiles generated:")
print(f"  • model.pkl               - Best trained model")
print(f"  • model_lr.pkl            - Linear Regression model")
print(f"  • model_rf.pkl            - Random Forest model")
print(f"  • model_results.json      - Metrics and statistics")
print(f"\nBest Model: {best_name}")
print(f"R² Score: {max(lr_r2, rf_r2):.4f}")
print(f"MAE: {min(lr_mae, rf_mae):.4f} kg")
print("\n" + "=" * 70)
