
from pathlib import Path
import pandas as pd
import numpy as np

# ==========================================
# 1. PATHS AND SETTINGS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "house1_15min.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "house1_training.csv"

FREQ = "15min"

# Historical power readings used for prediction.
LAGS = [1, 2, 4, 96]

ROLLING_WINDOWS = [4, 96]

print("=" * 60)
print("PREPARING XGBOOST TRAINING DATA")
print("=" * 60)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"File not found: {INPUT_FILE}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD AND VALIDATE DATA
# ==========================================

df = pd.read_csv(INPUT_FILE, parse_dates=["Time"])
df = df.sort_values("Time").set_index("Time")

if df.index.has_duplicates:
    raise ValueError("Duplicate timestamps found.")

if "Aggregate" not in df.columns:
    raise ValueError("Aggregate column not found.")

# Reindex to a complete 15-minute timeline.
# Missing intervals remain NaN; they are NOT filled with zero.
full_index = pd.date_range(
    start=df.index.min(),
    end=df.index.max(),
    freq=FREQ,
)

df = df.reindex(full_index)
df.index.name = "Time"

print(f"Complete timeline rows: {len(df):,}")
print(f"Missing aggregate intervals: {df['Aggregate'].isna().sum():,}")


# ==========================================
# 3. CREATE HISTORICAL FEATURES
# ==========================================

features = pd.DataFrame(index=df.index)

# Current household and appliance readings.
for col in df.columns:
    features[f"{col}_current"] = df[col]

# Historical household consumption.
for lag in LAGS:
    features[f"aggregate_lag_{lag}"] = (
        df["Aggregate"].shift(lag)
    )

# Rolling statistics use only current and earlier readings.
for window in ROLLING_WINDOWS:
    features[f"aggregate_mean_{window}"] = (
        df["Aggregate"]
        .rolling(window=window, min_periods=window)
        .mean()
    )

    features[f"aggregate_std_{window}"] = (
        df["Aggregate"]
        .rolling(window=window, min_periods=window)
        .std()
    )

# Calendar features for the interval being predicted.
target_time = pd.Series(
    df.index, index=df.index
).shift(-1)

features["target_hour"] = target_time.dt.hour
features["target_day_of_week"] = target_time.dt.dayofweek
features["target_is_weekend"] = (
    target_time.dt.dayofweek >= 5
).astype(float)

# Predict household power in the NEXT 15-minute interval.
target = df["Aggregate"].shift(-1).rename("target_power")

dataset = features.join(target)

# Remove rows where any required feature or target is missing.
# This also prevents training windows from crossing missing intervals.
dataset = dataset.replace([np.inf, -np.inf], np.nan)
dataset = dataset.dropna()

print(f"\nUsable training examples: {len(dataset):,}")

if len(dataset) < 100:
    raise ValueError(
        "Too few training examples. Check the timeline and feature settings."
    )


# ==========================================
# 4. CHRONOLOGICAL TRAIN / VALIDATION / TEST
# ==========================================

# Preserve time order; do not randomly shuffle.
n = len(dataset)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

train = dataset.iloc[:train_end]
validation = dataset.iloc[train_end:validation_end]
test = dataset.iloc[validation_end:]

print("\nCHRONOLOGICAL SPLIT")
print("-" * 60)

print(f"Training rows:   {len(train):,}")
print(f"Validation rows: {len(validation):,}")
print(f"Test rows:       {len(test):,}")

print("\nTraining target range:")
print(train.index.min(), "to", train.index.max())

print("\nValidation target range:")
print(validation.index.min(), "to", validation.index.max())

print("\nTest target range:")
print(test.index.min(), "to", test.index.max())


# ==========================================
# 5. SAVE PREPARED DATA
# ==========================================

dataset = dataset.reset_index(names="Time")
dataset.to_csv(OUTPUT_FILE, index=False)

print("\nPrepared dataset saved to:")
print(OUTPUT_FILE)

print("\nFirst 5 rows:")
print(dataset.head().to_string(index=False))

print("\n" + "=" * 60)
print("TRAINING DATA PREPARATION COMPLETED")
print("=" * 60)
