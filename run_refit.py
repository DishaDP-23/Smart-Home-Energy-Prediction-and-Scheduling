"""Validate local REFIT data, reuse Person 1's training, run causal scheduling."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def prepare_source(source):
    import numpy as np
    import pandas as pd
    df = pd.read_csv(source)
    required = ['Aggregate', *[f'Appliance{i}' for i in range(1, 10)]]
    if set(['Time', *required]) - set(df.columns):
        raise ValueError('Expected Time, Aggregate and Appliance1..9 columns')
    df['Time'] = pd.to_datetime(df['Time'], errors='raise')
    if df['Time'].duplicated().any() or df['Time'].isna().any():
        raise ValueError('Duplicate or invalid timestamps')
    if not (df['Time'] == df['Time'].dt.floor('15min')).all():
        raise ValueError('Input timestamps must align to 15 minutes')
    df = df.sort_values('Time')
    values = df[required].apply(pd.to_numeric, errors='coerce')
    invalid = ~np.isfinite(values) | (values < 0)
    # Missing measurements remain missing so upstream feature creation drops
    # affected windows; never silently replace missing/negative demand with zero.
    df[required] = values.mask(invalid)
    destination = ROOT / 'data/processed/house1_15min.csv'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        raise ValueError('Source must differ from prepared output; original data is preserved')
    df.to_csv(destination, index=False)
    expected = pd.date_range(df.Time.min(), df.Time.max(), freq='15min')
    report = dict(source=str(source.resolve()), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  rows=len(df), first_time=str(df.Time.min()), last_time=str(df.Time.max()),
                  missing_intervals=len(expected.difference(pd.DatetimeIndex(df.Time))),
                  invalid_measurements_masked=int(invalid.sum().sum()),
                  negative_values=int((values < 0).sum().sum()), units='W', interval_minutes=15)
    output = ROOT / 'results/rolling'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'data_quality.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'house1_15min.csv')
    parser.add_argument('--skip-training', action='store_true', help='Reuse existing test predictions')
    args = parser.parse_args()
    if not args.skip_training:
        prepare_source(args.input)
        subprocess.run([sys.executable, str(ROOT / 'src/prepare_training_data.py')], check=True)
        subprocess.run([sys.executable, str(ROOT / 'src/train_model.py'),
                        '--model-dir', str(ROOT / 'models'),
                        '--results-dir', str(ROOT / 'results/rolling/model')], check=True)
    subprocess.run([sys.executable, str(ROOT / 'evaluate_rolling.py')], check=True)


if __name__ == '__main__':
    main()
