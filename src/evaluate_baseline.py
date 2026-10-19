
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "house1_training.csv"

df = pd.read_csv(DATA_PATH)

# Match the chronological 70% / 15% / 15% split
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

test = df.iloc[val_end:].copy()

actual = test["target_power"]
predicted = test["Aggregate_current"]

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)

print("=" * 45)
print("PERSISTENCE BASELINE: NEXT 15-MINUTE POWER")
print("=" * 45)
print(f"Test examples: {len(test):,}")
print(f"MAE:  {mae:.3f} W")
print(f"RMSE: {rmse:.3f} W")
print(f"R²:   {r2:.4f}")

print("\nComparison with XGBoost")
print("-" * 45)
print("XGBoost MAE:  169.411 W")
print("XGBoost RMSE: 365.662 W")
print("XGBoost R²:   0.2703")

print("\nBaseline vs XGBoost:")
print(
    "MAE improvement from baseline: "
    f"{(mae - 169.411) / mae * 100:.2f}%"
    if mae != 0 else "MAE improvement cannot be calculated."
)

output = pd.DataFrame({
    "Time": test["Time"],
    "actual_power": actual,
    "baseline_prediction": predicted,
})

out_path = ROOT / "results" / "baseline_predictions.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
output.to_csv(out_path, index=False)

print(f"\nSaved baseline predictions to:\n{out_path}")
