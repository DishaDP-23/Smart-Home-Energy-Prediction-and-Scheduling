
from pathlib import Path
import pandas as pd

# ==========================================
# 1. PATHS AND CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "CLEAN_House1.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "house1_15min.csv"

INTERVAL = "15min"
CHUNK_SIZE = 250_000

MEASUREMENT_COLUMNS = [
    "Aggregate",
    "Appliance1",
    "Appliance2",
    "Appliance3",
    "Appliance4",
    "Appliance5",
    "Appliance6",
    "Appliance7",
    "Appliance8",
    "Appliance9",
]

REQUIRED_COLUMNS = ["Time"] + MEASUREMENT_COLUMNS


# ==========================================
# 2. VALIDATE INPUT
# ==========================================

print("=" * 60)
print("REFIT HOUSE 1: DATA PREPROCESSING")
print("=" * 60)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input CSV not found: {INPUT_FILE}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Check the header before processing the complete file.
header = pd.read_csv(INPUT_FILE, nrows=0)
missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in header.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing from CSV: {missing_columns}"
    )

print(f"Input:  {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print(f"Interval: {INTERVAL}")


# ==========================================
# 3. PROCESS CSV IN CHUNKS
# ==========================================

# Store sums and counts for each time interval.
# Combining sums and counts handles intervals that
# span two consecutive CSV chunks.

interval_sums = None
interval_counts = None

total_rows = 0
invalid_timestamps = 0
invalid_measurements = 0
negative_measurements = 0

reader = pd.read_csv(
    INPUT_FILE,
    usecols=REQUIRED_COLUMNS,
    chunksize=CHUNK_SIZE,
)

for chunk_number, chunk in enumerate(reader, start=1):
    total_rows += len(chunk)

    # Parse timestamps.
    chunk["Time"] = pd.to_datetime(
        chunk["Time"],
        errors="coerce",
    )

    invalid_timestamps += int(chunk["Time"].isna().sum())

    # Convert measurements to numeric.
    for col in MEASUREMENT_COLUMNS:
        original = chunk[col]

        numeric = pd.to_numeric(
            original,
            errors="coerce",
        )

        invalid_measurements += int(
            (original.notna() & numeric.isna()).sum()
        )

        negative_measurements += int(
            (numeric < 0).sum()
        )

        chunk[col] = numeric

    # Remove rows with invalid timestamps or measurements.
    chunk = chunk.dropna(
        subset=["Time"] + MEASUREMENT_COLUMNS
    )

    if chunk.empty:
        continue

    # Assign each reading to a 15-minute interval.
    chunk["Time"] = chunk["Time"].dt.floor(INTERVAL)

    grouped = chunk.groupby("Time")[MEASUREMENT_COLUMNS]

    chunk_sums = grouped.sum()
    chunk_counts = grouped.count()

    if interval_sums is None:
        interval_sums = chunk_sums
        interval_counts = chunk_counts
    else:
        interval_sums = interval_sums.add(
            chunk_sums, fill_value=0
        )

        interval_counts = interval_counts.add(
            chunk_counts, fill_value=0
        )

    print(
        f"Processed chunk {chunk_number:,} | "
        f"Rows processed: {total_rows:,}"
    )


# ==========================================
# 4. CALCULATE INTERVAL AVERAGES
# ==========================================

if interval_sums is None or interval_sums.empty:
    raise ValueError(
        "No valid measurements were found in the input CSV."
    )

# Use the sum/count across all chunks for each interval.
processed = interval_sums.div(interval_counts)

processed = processed.sort_index()
processed.index.name = "Time"

# Convert the index back into a regular CSV column.
processed = processed.reset_index()

# ==========================================
# 5. REPORT DATA QUALITY
# ==========================================

print("\nDATA QUALITY SUMMARY")
print("-" * 60)

print(f"Input rows: {total_rows:,}")
print(f"Invalid timestamps: {invalid_timestamps:,}")
print(f"Invalid measurement values: {invalid_measurements:,}")
print(f"Negative measurement values: {negative_measurements:,}")
print(f"Output intervals: {len(processed):,}")

print("\nProcessed date range:")
print("Start:", processed["Time"].min())
print("End:  ", processed["Time"].max())

print("\nMissing values in processed data:")
print(processed.isna().sum().to_string())

# ==========================================
# 6. SAVE PROCESSED DATA
# ==========================================

processed.to_csv(OUTPUT_FILE, index=False)

print("\nFirst 5 processed rows:")
print(processed.head().to_string(index=False))

print("\nProcessed dataset saved successfully:")
print(OUTPUT_FILE)

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)
