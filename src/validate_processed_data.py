
from pathlib import Path
import pandas as pd

# Locate project and processed data
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "house1_15min.csv"

if not DATA_FILE.exists():
    raise FileNotFoundError(f"File not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE, parse_dates=["Time"])
df = df.sort_values("Time").reset_index(drop=True)

print("=" * 60)
print("PROCESSED DATA VALIDATION")
print("=" * 60)

# Basic checks
print(f"\nRows: {len(df):,}")
print(f"Duplicate timestamps: {df['Time'].duplicated().sum():,}")
print(f"Missing values: {df.isna().sum().sum():,}")

# Calculate gaps between consecutive timestamps
time_diffs = df["Time"].diff()

print("\nTIME INTERVAL DISTRIBUTION")
print("-" * 60)
print(time_diffs.value_counts().head(10).to_string())

# Identify intervals with gaps larger than 15 minutes
gap_mask = time_diffs > pd.Timedelta(minutes=15)
gap_count = int(gap_mask.sum())

print(f"\nGaps larger than 15 minutes: {gap_count:,}")

if gap_count:
    gap_examples = pd.DataFrame({
        "previous_time": df.loc[gap_mask, "Time"].shift(1),
        "next_time": df.loc[gap_mask, "Time"],
        "gap": time_diffs[gap_mask],
    })

    # Show the first few gaps for inspection
    print("\nFirst 10 gaps:")
    print(gap_examples.head(10).to_string(index=False))

# Compare actual and expected number of intervals
expected_index = pd.date_range(
    start=df["Time"].min(),
    end=df["Time"].max(),
    freq="15min",
)

actual_index = pd.DatetimeIndex(df["Time"])

missing_times = expected_index.difference(actual_index)

print("\nTIMELINE COVERAGE")
print("-" * 60)
print(f"Expected 15-minute timestamps: {len(expected_index):,}")
print(f"Observed timestamps:           {len(actual_index):,}")
print(f"Missing timestamps:            {len(missing_times):,}")

if len(expected_index):
    coverage = len(actual_index) / len(expected_index) * 100
    print(f"Timestamp coverage: {coverage:.2f}%")

if not missing_times.empty:
    print("\nFirst 10 missing timestamps:")
    print(missing_times[:10].to_list())

print("\nValidation complete.")
