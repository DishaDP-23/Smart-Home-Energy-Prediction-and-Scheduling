
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ==========================================
# 1. PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "house1_training.csv"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_FILE = MODEL_DIR / "xgboost_energy_model.joblib"
METRICS_FILE = RESULTS_DIR / "prediction_metrics.json"
PREDICTIONS_FILE = RESULTS_DIR / "test_predictions.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("XGBOOST HOUSEHOLD ENERGY PREDICTION")
print("=" * 60)

# ==========================================
# 2. LOAD PREPARED DATA
# ==========================================

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE, parse_dates=["Time"])
df = df.sort_values("Time").reset_index(drop=True)

if "target_power" not in df.columns:
    raise ValueError("target_power column not found.")

# Time is metadata, not a direct model feature.
# The target is household power in the next 15-minute interval.
X = df.drop(columns=["Time", "target_power"])
y = df["target_power"]

# Keep only numeric features.
X = X.select_dtypes(include=[np.number])

if X.isna().any().any() or y.isna().any():
    raise ValueError("Missing values found in training data.")

print(f"Total examples: {len(df):,}")
print(f"Number of features: {X.shape[1]}")

# ==========================================
# 3. CHRONOLOGICAL SPLIT
# ==========================================

n = len(df)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X.iloc[train_end:validation_end]
y_val = y.iloc[train_end:validation_end]

X_test = X.iloc[validation_end:]
y_test = y.iloc[validation_end:]

test_times = df["Time"].iloc[validation_end:].reset_index(drop=True)

print("\nDATA SPLIT")
print("-" * 60)
print(f"Training:   {len(X_train):,}")
print(f"Validation: {len(X_val):,}")
print(f"Testing:    {len(X_test):,}")

# ==========================================
# 4. TRAIN XGBOOST
# ==========================================

model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    eval_metric="rmse",
    early_stopping_rounds=40,
)

print("\nTraining model...")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)

print("\nTraining completed.")

# ==========================================
# 5. EVALUATE ON UNSEEN TEST DATA
# ==========================================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = r2_score(y_test, y_pred)

# MAPE can become misleading when actual power is
# zero or close to zero, so exclude very small values.
mape_mask = y_test.abs() > 1.0

if mape_mask.any():
    mape = (
        np.abs(
            (y_test[mape_mask].to_numpy()
             - y_pred[mape_mask])
            / y_test[mape_mask].to_numpy()
        ).mean() * 100
    )
else:
    mape = None

metrics = {
    "target": "next_15min_aggregate_power",
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2),
    "mape_percent_actual_abs_gt_1": (
        float(mape) if mape is not None else None
    ),
    "test_samples": int(len(y_test)),
    "features": int(X.shape[1]),
    "best_iteration": int(model.best_iteration),
}

print("\nTEST RESULTS")
print("-" * 60)
print(f"MAE:  {mae:.3f} W")
print(f"RMSE: {rmse:.3f} W")
print(f"R²:   {r2:.4f}")

if mape is not None:
    print(f"MAPE (actual power > 1 W): {mape:.2f}%")

# ==========================================
# 6. SAVE MODEL AND METRICS
# ==========================================

joblib.dump(model, MODEL_FILE)

with open(METRICS_FILE, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

predictions = pd.DataFrame({
    "Time": test_times,
    "actual_power": y_test.to_numpy(),
    "predicted_power": y_pred,
})

predictions.to_csv(PREDICTIONS_FILE, index=False)

print("\nSaved model:")
print(MODEL_FILE)

print("\nSaved metrics:")
print(METRICS_FILE)

print("\nSaved test predictions:")
print(PREDICTIONS_FILE)

# ==========================================
# 7. PLOT ACTUAL VS PREDICTED POWER
# ==========================================

# Show a limited sample to keep the graph readable.
plot_count = min(500, len(predictions))
plot_df = predictions.iloc[:plot_count]

plt.figure(figsize=(13, 5))
plt.plot(
    plot_df["Time"],
    plot_df["actual_power"],
    label="Actual power",
)
plt.plot(
    plot_df["Time"],
    plot_df["predicted_power"],
    label="Predicted power",
)

plt.xlabel("Time")
plt.ylabel("Aggregate power (W)")
plt.title("XGBoost: Actual vs Predicted Household Power")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()

plot_file = RESULTS_DIR / "actual_vs_predicted.png"
plt.savefig(plot_file, dpi=150)
plt.close()

# ==========================================
# 8. FEATURE IMPORTANCE
# ==========================================

importance = pd.Series(
    model.feature_importances_,
    index=X.columns,
).sort_values(ascending=False)

importance_file = RESULTS_DIR / "feature_importance.csv"
importance.to_csv(importance_file, header=["importance"])

plt.figure(figsize=(10, 6))
importance.head(15).sort_values().plot(kind="barh")

plt.xlabel("Feature importance")
plt.title("Top 15 XGBoost Features")
plt.tight_layout()

importance_plot = RESULTS_DIR / "feature_importance.png"
plt.savefig(importance_plot, dpi=150)
plt.close()

print("\nGenerated graphs:")
print(plot_file)
print(importance_plot)

print("\nTop 10 features:")
print(importance.head(10).to_string())

print("\n" + "=" * 60)
print("XGBOOST TRAINING AND EVALUATION COMPLETED")
print("=" * 60)
