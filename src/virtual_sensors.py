
"""
Virtual Sensor Simulator
AIoT-Based Intelligent Home Energy Management

Features:
- Realistic synthetic appliance operating patterns
- Timestamped CSV logging
- MQTT telemetry publishing to ThingsBoard
- Automatic MQTT reconnection attempts
- Compatible with the existing Streamlit dashboard
"""

import csv
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import paho.mqtt.client as mqtt
from dotenv import load_dotenv


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_FILE = DATA_DIR / "sample_energy.csv"

load_dotenv(PROJECT_ROOT / ".env")

HOST = os.getenv("THINGSBOARD_HOST", "mqtt.thingsboard.cloud")
TOKEN = os.getenv("THINGSBOARD_TOKEN")

INTERVAL_SECONDS = 2

# Rated power in watts.
APPLIANCES = {
    "lights": 60,
    "fan": 75,
    "refrigerator": 150,
    "television": 100,
    "washing_machine": 500,
}

CSV_FIELDS = [
    "timestamp",
    *APPLIANCES.keys(),
    "total_power_w",
]


# --------------------------------------------------
# Appliance simulation
# --------------------------------------------------

class ApplianceSimulator:
    """
    Maintains appliance states across readings instead
    of randomly switching every appliance every time.
    """

    def __init__(self):
        self.states = {
            "lights": False,
            "fan": False,
            "refrigerator": True,
            "television": False,
            "washing_machine": False,
        }

        self.next_change = {
            appliance: time.monotonic()
            + random.uniform(5, 20)
            for appliance in APPLIANCES
        }

        self.washing_machine_end = 0.0
        self.refrigerator_cycle_end = (
            time.monotonic() + random.uniform(10, 25)
        )

    def update_state(self, appliance, now):
        """Update an appliance according to its operating pattern."""

        if appliance == "refrigerator":
            # A refrigerator cycles between running and idle.
            if now >= self.refrigerator_cycle_end:
                self.states["refrigerator"] = (
                    not self.states["refrigerator"]
                )

                if self.states["refrigerator"]:
                    self.refrigerator_cycle_end = (
                        now + random.uniform(15, 35)
                    )
                else:
                    self.refrigerator_cycle_end = (
                        now + random.uniform(8, 20)
                    )

        elif appliance == "washing_machine":
            # Once started, the washing machine runs for
            # a longer cycle instead of switching every sample.
            if now >= self.washing_machine_end:
                if self.states["washing_machine"]:
                    self.states["washing_machine"] = False
                    self.next_change[appliance] = (
                        now + random.uniform(30, 60)
                    )
                elif now >= self.next_change[appliance]:
                    # Occasional simulated wash cycle.
                    if random.random() < 0.35:
                        self.states["washing_machine"] = True
                        self.washing_machine_end = (
                            now + random.uniform(30, 60)
                        )
                    else:
                        self.next_change[appliance] = (
                            now + random.uniform(20, 45)
                        )

        else:
            # Lights, fan, and television remain in their
            # current state for a while before changing.
            if now >= self.next_change[appliance]:
                self.states[appliance] = (
                    not self.states[appliance]
                )
                self.next_change[appliance] = (
                    now + random.uniform(15, 45)
                )

    def generate_reading(self):
        """Generate one timestamped household power reading."""

        now = time.monotonic()
        readings = {}

        for appliance, rated_power in APPLIANCES.items():
            self.update_state(appliance, now)

            if self.states[appliance]:
                # Running power varies around rated power.
                power = random.uniform(
                    0.7 * rated_power,
                    rated_power,
                )
            else:
                power = 0.0

            readings[appliance] = round(power, 2)

        readings["timestamp"] = datetime.now().isoformat(
            timespec="seconds"
        )

        readings["total_power_w"] = round(
            sum(
                readings[name]
                for name in APPLIANCES
            ),
            2,
        )

        return readings


# --------------------------------------------------
# CSV logging
# --------------------------------------------------

def save_reading(reading):
    """Append a reading to the CSV file."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = (
        CSV_FILE.exists()
        and CSV_FILE.stat().st_size > 0
    )

    with CSV_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(reading)


# --------------------------------------------------
# ThingsBoard MQTT connection
# --------------------------------------------------

def create_mqtt_client():
    """Connect to ThingsBoard using the device access token."""

    if not TOKEN:
        raise RuntimeError(
            "THINGSBOARD_TOKEN is missing from the project .env file."
        )

    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id="home-energy-virtual-sensors",
    )

    client.username_pw_set(TOKEN)

    print(f"Connecting to {HOST}:1883 ...")
    client.connect(HOST, 1883, 60)
    client.loop_start()

    for _ in range(50):
        if client.is_connected():
            print("Connected to ThingsBoard MQTT.")
            return client

        time.sleep(0.1)

    client.loop_stop()

    raise ConnectionError(
        "Could not establish the MQTT connection."
    )


def publish_reading(client, reading):
    """Publish appliance telemetry to ThingsBoard."""

    telemetry_values = {
        key: value
        for key, value in reading.items()
        if key != "timestamp"
    }

    timestamp_ms = int(
        datetime.fromisoformat(
            reading["timestamp"]
        ).timestamp() * 1000
    )

    payload = {
        "ts": timestamp_ms,
        "values": telemetry_values,
    }

    info = client.publish(
        "v1/devices/me/telemetry",
        json.dumps(payload),
        qos=1,
    )

    if info.rc != mqtt.MQTT_ERR_SUCCESS:
        print(
            "Publish could not be queued. "
            f"Error code: {info.rc}"
        )
        return False

    try:
        info.wait_for_publish(timeout=10)
    except RuntimeError as error:
        print(f"MQTT publish error: {error}")
        return False

    if info.is_published():
        print("ThingsBoard confirmed telemetry.")
        return True

    print("WARNING: MQTT publish confirmation timed out.")
    return False


# --------------------------------------------------
# Main simulator
# --------------------------------------------------

def run_simulator(interval_seconds=INTERVAL_SECONDS):
    """Run the simulator until the user stops it."""

    if interval_seconds <= 0:
        raise ValueError(
            "The interval must be greater than zero."
        )

    simulator = ApplianceSimulator()
    client = create_mqtt_client()

    print(f"Saving readings to: {CSV_FILE}")
    print(
        f"Publishing every {interval_seconds} seconds."
    )
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            if not client.is_connected():
                print("MQTT disconnected. Reconnecting...")

                try:
                    client.reconnect()

                    for _ in range(50):
                        if client.is_connected():
                            break
                        time.sleep(0.1)

                except (OSError, RuntimeError) as error:
                    print(f"Reconnection failed: {error}")
                    time.sleep(5)
                    continue

                if not client.is_connected():
                    print("Still disconnected. Retrying shortly.")
                    time.sleep(5)
                    continue

                print("MQTT reconnected.")

            reading = simulator.generate_reading()

            # Local logging continues even if publishing fails.
            try:
                save_reading(reading)
            except OSError as error:
                print(f"CSV write failed: {error}")

            publish_reading(client, reading)

            print(
                f"{reading['timestamp']} | "
                f"Total: {reading['total_power_w']} W | "
                "CSV logging attempted"
            )

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\nVirtual sensor simulator stopped.")

    finally:
        if client.is_connected():
            client.disconnect()

        client.loop_stop()


if __name__ == "__main__":
    run_simulator()
