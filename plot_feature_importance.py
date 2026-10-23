
from pathlib import Path

import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.appliance_mapping import APPLIANCE_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_ROOT / "models" / "xgboost_energy_model.joblib"
OUTPUT_FILE = PROJECT_ROOT / "rolling" / "model" / "feature_importance.png"

if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_FILE}")

model = joblib.load(MODEL_FILE)

# Extract the existing model's feature importance.
importance = pd.Series(
    model.feature_importances_,
    index=model.feature_names_in_,
).sort_values(ascending=False)

# Rename appliance references for display only.
def display_feature_name(feature):
    for original_name, friendly_name in APPLIANCE_NAMES.items():
        feature = feature.replace(original_name, friendly_name)
    return feature

display_importance = importance.head(15).copy()
display_importance.index = [
    display_feature_name(name)
    for name in display_importance.index
]

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 6))
display_importance.sort_values().plot(kind="barh")
plt.xlabel("Feature importance")
plt.title("Top 15 XGBoost Features")
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=150)
plt.close()

print("Feature-importance graph updated:")
print(OUTPUT_FILE)
