
"""Assumed appliance labels for display and telemetry.

These names are illustrative assumptions, not verified identities
from the original dataset. Keep the original dataset and model
feature names unchanged.
"""

APPLIANCE_NAMES = {
    "Appliance1": "Fan",
    "Appliance2": "Lights",
    "Appliance3": "Television",
    "Appliance4": "Microwave",
    "Appliance5": "Air Conditioner",
    "Appliance6": "Water Heater",
    "Appliance7": "Laptop",
    "Appliance8": "Iron",
    "Appliance9": "Refrigerator",
}


def get_appliance_name(original_name: str) -> str:
    """Return the friendly label, or the original name if unmapped."""
    return APPLIANCE_NAMES.get(original_name, original_name)
