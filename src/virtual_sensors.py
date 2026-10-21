
"""
Replay recorded household energy measurements from house1_15min.csv.

- Does not generate random or synthetic household readings.
- Saves replayed records to data/replayed_energy.csv.
- Optionally publishes recorded telemetry to ThingsBoard using MQTT.
- Preserves the original dataset timestamps.
"""

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from appliance_mapping import APPLIANCE_NAMES

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_FILE = PROJECT_ROOT / "house1_15min.csv"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "replayed_energy.csv"

load_dotenv(PROJECT_ROOT / ".env")

HOST = os.getenv("THINGSBOARD_HOST", "mqtt.thingsboard.cloud")
TOKEN = os.getenv("THINGSBOARD_TOKEN")

# Delay between replayed rows; this is not the dataset's sampling interval.
INTERVAL_SECONDS = 0.2

APPLIANCE_COLUMNS = [
    f"Appliance{i}" for i in range(1, 10)
]

OUTPUT_FIELDS = [
    "timestamp",
    *[APPLIANCE_NAMES[column] for column in APPLIANCE_COLUMNS],
    "total_power_w",
]


# --------------------------------------------------
# Load and validate the original dataset
# --------------------------------------------------

def load_recorded_data():
    """Load real measurements without modifying the source file."""

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {SOURCE_FILE}"
        )

    data = pd.read_csv(SOURCE_FILE)

    required_columns = {
        "Time",
        "Aggregate",
        *APPLIANCE_COLUMNS,
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )

    data = data[
        ["Time", "Aggregate", *APPLIANCE_COLUMNS]
    ].copy()

    data["Time"] = pd.to_datetime(
        data["Time"], errors="raise"
    )

    if data["Time"].duplicated().any():
        raise ValueError(
            "The source dataset contains duplicate timestamps."
        )

    data = data.sort_values("Time").reset_index(drop=True)

    numeric_columns = ["Aggregate", *APPLIANCE_COLUMNS]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column], errors="raise"
        )

    if data[numeric_columns].isna().any().any():
        raise ValueError(
            "The source dataset contains missing power readings."
        )

    if (data[numeric_columns] < 0).any().any():
        raise ValueError(
            "Negative power readings were found in the dataset."
        )

    return data


# --------------------------------------------------
# CSV output
# --------------------------------------------------

def save_reading(reading, writer):
    """Write one recorded reading to the replay CSV."""

    writer.writerow(reading)


# --------------------------------------------------
# Optional ThingsBoard MQTT connection
# --------------------------------------------------

def create_mqtt_client():
    """Connect to ThingsBoard when MQTT credentials are available."""

    if not TOKEN:
        print(
            "THINGSBOARD_TOKEN is not set. "
            "Local CSV replay will continue without MQTT."
        )
        return None

    if mqtt is None:
        print(
            "paho-mqtt is not installed. "
            "Local CSV replay will continue without MQTT."
        )
        return None

    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id="home-energy-dataset-replay",
    )

    client.username_pw_set(TOKEN)

    try:
        print(f"Connecting to {HOST}:1883 ...")
        client.connect(HOST, 1883, 60)
        client.loop_start()

        for _ in range(50):
            if client.is_connected():
                print("Connected to ThingsBoard MQTT.")
                return client

            time.sleep(0.1)

        client.loop_stop()
        client.disconnect()

        print(
            "ThingsBoard connection timed out. "
            "Local CSV replay will continue."
        )

    except (OSError, RuntimeError) as error:
        print(
            f"ThingsBoard connection failed: {error}\n"
            "Local CSV replay will continue."
        )

    return None


def publish_reading(client, reading):
    """Publish a recorded row using its original dataset timestamp."""

    if client is None or not client.is_connected():
        return False

    source_time = pd.Timestamp(reading["timestamp"])

    # Preserve an explicit timezone if present. For timezone-naive
    # dataset timestamps, use the computer's local timezone.
    timestamp_ms = int(source_time.timestamp() * 1000)

    telemetry_values = {
        key: float(value)
        for key, value in reading.items()
        if key != "timestamp"
    }

    payload = {
        "ts": timestamp_ms,
        "values": telemetry_values,
    }

    try:
        info = client.publish(
            "v1/devices/me/telemetry",
            json.dumps(payload),
            qos=1,
        )

        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            print(
                f"Telemetry could not be queued: {info.rc}"
            )
            return False

        info.wait_for_publish(timeout=10)

        if info.is_published():
            print("ThingsBoard confirmed telemetry.")
            return True

        print("WARNING: Telemetry confirmation timed out.")

    except (OSError, RuntimeError) as error:
        print(f"Telemetry publishing failed: {error}")

    return False


# --------------------------------------------------
# Historical dataset replay
# --------------------------------------------------

def run_replay(interval_seconds=INTERVAL_SECONDS):
    """Replay the source dataset sequentially, from first row to last."""

    if interval_seconds <= 0:
        raise ValueError(
            "interval_seconds must be greater than zero."
        )

    data = load_recorded_data()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    client = create_mqtt_client()

    print(f"Source dataset: {SOURCE_FILE}")
    print(f"Replay output:  {OUTPUT_FILE}")
    print(f"Recorded rows:  {len(data)}")
    print(f"Replay delay:   {interval_seconds} seconds")
    print(
        "\nIMPORTANT: Values are recorded historical measurements, "
        "not current physical sensor readings."
    )
    print("Press Ctrl+C to stop replay.\n")

    try:
        # Start a clean replay file. Never append to old synthetic data.
        with OUTPUT_FILE.open(
            "w", newline="", encoding="utf-8"
        ) as output_file:

            writer = csv.DictWriter(
                output_file,
                fieldnames=OUTPUT_FIELDS,
            )
            writer.writeheader()

            for index, row in data.iterrows():
                source_timestamp = pd.Timestamp(row["Time"]).isoformat()

                reading = {
                    "timestamp": source_timestamp,
                    **{
                        APPLIANCE_NAMES[column]: float(row[column])
                        for column in APPLIANCE_COLUMNS
                    },
                    "total_power_w": float(row["Aggregate"]),
                }

                save_reading(reading, writer)
                output_file.flush()

                publish_reading(client, reading)

                print(
                    f"[{index + 1}/{len(data)}] "
                    f"{source_timestamp} | "
                    f"Recorded aggregate: "
                    f"{reading['total_power_w']:.2f} W"
                )

                if index < len(data) - 1:
                    time.sleep(interval_seconds)

        print("\nDataset replay completed.")

    except KeyboardInterrupt:
        print("\nDataset replay stopped by user.")

    finally:
        if client is not None:
            try:
                if client.is_connected():
                    client.disconnect()
                client.loop_stop()
            except (OSError, RuntimeError):
                pass


if __name__ == "__main__":
    run_replay()
