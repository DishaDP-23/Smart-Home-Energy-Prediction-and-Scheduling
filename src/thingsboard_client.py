
import os
import json
import time
from pathlib import Path

import paho.mqtt.client as mqtt
from dotenv import load_dotenv

# Load .env from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

HOST = os.getenv("THINGSBOARD_HOST", "mqtt.thingsboard.cloud")
TOKEN = os.getenv("THINGSBOARD_TOKEN")


def send_test_reading():
    if not TOKEN:
        print("ERROR: THINGSBOARD_TOKEN is missing from .env")
        return

    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id="home-energy-test"
    )
    client.username_pw_set(TOKEN)

    try:
        print(f"Connecting to {HOST}:1883 ...")

        client.connect(HOST, 1883, 60)
        client.loop_start()

        # Wait for the MQTT connection.
        for _ in range(50):
            if client.is_connected():
                break
            time.sleep(0.1)

        if not client.is_connected():
            print("ERROR: MQTT connection was not established.")
            return

        print("MQTT connected successfully.")

        payload = {
            "total_power_w": 250.0,
            "source": "python_test"
        }

        info = client.publish(
            "v1/devices/me/telemetry",
            json.dumps(payload),
            qos=1
        )

        info.wait_for_publish(timeout=10)

        if info.is_published():
            print("Telemetry published:", payload)
        else:
            print("ERROR: Publish was not confirmed.")

    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}")

    finally:
        if client.is_connected():
            client.disconnect()
        client.loop_stop()


if __name__ == "__main__":
    print("ThingsBoard client file loaded.")
    send_test_reading()
