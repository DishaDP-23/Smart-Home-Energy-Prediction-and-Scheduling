"""Generate deterministic SYNTHETIC background demand, not REFIT observations."""
import csv
import math
from datetime import datetime, timedelta
from pathlib import Path


def generate(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'predicted_background_kw', 'actual_background_kw', 'tariff_per_kwh'])
        for i in range(96):
            hour = i / 4
            predicted = 0.5 + 1.7 * math.exp(-((hour - 18.5) / 1.8) ** 2) + 0.5 * math.exp(-((hour - 8) / 1.5) ** 2)
            actual = max(0, predicted + 0.12 * math.sin(i * 0.7))
            tariff = 10 if 17 <= hour < 21 else (4 if hour >= 21 or hour < 6 else 6)
            writer.writerow([(datetime(2026, 1, 1) + timedelta(minutes=15*i)).isoformat(),
                             round(predicted, 4), round(actual, 4), tariff])


if __name__ == '__main__':
    generate(Path(__file__).parent / 'data' / 'sample_predictions.csv')
