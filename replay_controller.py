"""Causal sensor-to-predictor-to-scheduler integration, without network side effects."""
from collections import deque
from datetime import timedelta
import json
from pathlib import Path

import pandas as pd
from evaluate_rolling import jobs_for_day, tariff_at
from rolling_scheduler import RollingScheduler


class ReplayController:
    def __init__(self, predictor, config):
        if config['slot_minutes'] != 15:
            raise ValueError('Integrated REFIT replay requires 15-minute slots')
        self.predictor = predictor
        self.config = config
        self.history = deque(maxlen=97)
        self.last_time = None
        self.engine = None
        self.blocked_day = None
        self.events = []

    def observe(self, row):
        """A row is a completed interval average; next dispatch is at Time + 15m."""
        now = pd.Timestamp(row['Time'])
        if self.last_time is not None and now <= self.last_time:
            raise ValueError('Replay timestamps must increase')
        if self.last_time is not None and now - self.last_time != pd.Timedelta(minutes=15):
            if self.engine:
                self.events.append({'timestamp': now.isoformat(), 'status': 'replay_gap',
                                    'reason': 'Scheduling interrupted by missing observations; resume next day after warmup'})
            self.history.clear()
            self.engine = None
            self.blocked_day = now.date()
        self.last_time = now
        self.history.append(dict(row))
        observed_added_kw = 0.0
        active = []
        if self.engine:
            for decision in self.engine.decisions.values():
                if decision['status'] == 'scheduled' and pd.Timestamp(decision['start']) <= now < pd.Timestamp(decision['end']):
                    observed_added_kw += decision['power_kw']
                    active.append(decision['appliance_id'])
        state = dict(observed_timestamp=now.isoformat(), forecast=None, status='warming_up',
                     history_intervals=len(self.history), current_simulated_appliance_kw=observed_added_kw,
                     current_total_kw=float(row['Aggregate'])/1000+observed_added_kw,
                     active_simulated_jobs=active, schedule=[], updates=[], replay_events=self.events[-20:])
        if len(self.history) < 97:
            return state
        forecast = self.predictor.predict_next(pd.DataFrame(self.history))
        target = pd.Timestamp(forecast['timestamp'])
        if target != now + pd.Timedelta(minutes=15):
            raise ValueError('Predictor returned a misaligned target timestamp')
        state['forecast'] = forecast
        if target.date() == self.blocked_day:
            state['status'] = 'scheduling_paused_after_gap'
            return state
        if not self.engine or self.engine.times[0].date() != target.date():
            timestamps = [t.isoformat() for t in pd.date_range(target, target.normalize()+pd.Timedelta(days=1), freq='15min', inclusive='left')]
            self.engine = RollingScheduler(timestamps, [tariff_at(t, self.config) for t in timestamps],
                                           jobs_for_day(target.date().isoformat(), self.config), self.config)
        updates = self.engine.step(target.isoformat(), forecast['predicted_aggregate_kw'])
        if self.engine.next_index == len(self.engine.times):
            self.engine.finish()
        state.update(status='running', schedule=list(self.engine.decisions.values()), updates=updates,
                     schedule_timestamp=target.isoformat(),
                     predicted_total_kw=forecast['predicted_aggregate_kw'] + self.engine.committed_kw[self.engine.next_index-1])
        return state


def write_state(path, state):
    """Publish an atomic snapshot so the dashboard never reads partial JSON."""
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2, default=str, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def telemetry_values(reading, state):
    values = {k: float(v) for k, v in reading.items() if k != 'timestamp'}
    values.update(replay_timestamp=reading['timestamp'], integration_status=state['status'],
                  simulated_appliance_power_w=state['current_simulated_appliance_kw']*1000,
                  simulated_total_power_w=state['current_total_kw']*1000,
                  schedule_json=json.dumps(state['schedule']),
                  scheduling_updates_json=json.dumps(state['updates']))
    if state['forecast']:
        values.update(predicted_aggregate_power_w=state['forecast']['predicted_aggregate_kw']*1000,
                      forecast_timestamp=state['forecast']['timestamp'])
    return values
