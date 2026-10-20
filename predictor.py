"""Inference adapter for Person 1's locally trained XGBoost model."""
from pathlib import Path


class Predictor:
    def __init__(self, model_path=None):
        import joblib
        self.model = joblib.load(model_path or Path(__file__).parent / 'models/xgboost_energy_model.joblib')

    def predict_next(self, history):
        """history: DataFrame with completed interval-start Time and REFIT W columns.

        Requires the last 97 consecutive 15-minute aggregate readings and the
        latest nine appliance readings. Feed exogenous recorded measurements,
        excluding added simulated jobs. Only load trusted local model artifacts.
        """
        import numpy as np
        import pandas as pd
        df = history.copy()
        df['Time'] = pd.to_datetime(df['Time'], errors='raise')
        if df['Time'].duplicated().any():
            raise ValueError('Duplicate history timestamps')
        df = df.sort_values('Time').set_index('Time')
        if len(df) < 97:
            raise ValueError('97 completed intervals required')
        last = df.index[-1]
        if last != last.floor('15min'):
            raise ValueError('History must align to 15 minutes')
        df = df.reindex(pd.date_range(end=last, periods=97, freq='15min'))
        columns = ['Aggregate', *[f'Appliance{i}' for i in range(1, 10)]]
        current = df[columns].iloc[-1].astype(float)
        aggregate = df['Aggregate'].astype(float)
        if not np.isfinite(current).all() or not np.isfinite(aggregate).all():
            raise ValueError('Missing or nonfinite required history')
        if (current < 0).any() or (aggregate < 0).any():
            raise ValueError('Negative measurements are invalid')
        values = {f'{column}_current': current[column] for column in columns}
        values.update({f'aggregate_lag_{lag}': aggregate.iloc[-lag-1] for lag in [1, 2, 4, 96]})
        for window in [4, 96]:
            values[f'aggregate_mean_{window}'] = aggregate.iloc[-window:].mean()
            values[f'aggregate_std_{window}'] = aggregate.iloc[-window:].std()
        target = last + pd.Timedelta(minutes=15)
        values.update(target_hour=target.hour, target_day_of_week=target.dayofweek,
                      target_is_weekend=float(target.dayofweek >= 5))
        features = pd.DataFrame([values])[list(self.model.feature_names_in_)]
        prediction = float(self.model.predict(features)[0])
        if not np.isfinite(prediction):
            raise ValueError('Model returned a nonfinite prediction')
        return dict(timestamp=target.isoformat(), forecast_issued_at=target.isoformat(),
                    predicted_aggregate_kw=max(0, prediction)/1000,
                    negative_prediction_clipped=prediction < 0)
