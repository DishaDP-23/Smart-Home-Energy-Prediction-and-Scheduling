import csv
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timedelta
from scheduler import Job
from rolling_scheduler import RollingScheduler, replay
from evaluate import evaluate
from evaluate_rolling import load_predictions, jobs_for_day


class RollingTests(unittest.TestCase):
    def setUp(self):
        self.times = [(datetime(2026, 1, 1)+timedelta(minutes=i*15)).isoformat() for i in range(8)]
        self.rows = [dict(timestamp=t, predicted_background_kw=1., actual_background_kw=1.,
                          tariff_per_kwh=4.) for t in self.times]
        self.config = dict(slot_minutes=15, max_household_kw=5., preferred_total_kw=3.5, off_peak_threshold=4.)
        self.job = Job('job', 1., 30, self.times[0], self.times[0], self.times[-1], 60)

    def test_future_forecasts_cannot_change_past_starts(self):
        first = replay(self.rows, [self.job], self.config)
        for row in self.rows[1:]:
            row['predicted_background_kw'] = 100
            row['actual_background_kw'] = 200
        second = replay(self.rows, [self.job], self.config)
        self.assertEqual(first, second)
        self.assertEqual(first[0]['start'], self.times[0])

    def test_prefix_causality_when_job_is_pending(self):
        engines = [RollingScheduler(self.times, [4]*8, [self.job], self.config) for _ in range(2)]
        for engine in engines:
            self.assertEqual(engine.step(self.times[0], 4.5), [])
        # Future forecasts are not even accepted by the step API.
        self.assertEqual(engines[0].step(self.times[1], 1)[0]['start'], self.times[1])
        self.assertEqual(engines[1].step(self.times[1], 5), [])

    def test_max_delay_relaxes_preference_not_cap(self):
        for row in self.rows:
            row['predicted_background_kw'] = 3
        decisions = replay(self.rows, [self.job], self.config)
        self.assertEqual(decisions[0]['start'], self.times[4])
        for row in self.rows:
            row['predicted_background_kw'] = 5
        decisions = replay(self.rows, [self.job], self.config)
        self.assertEqual(decisions[0]['status'], 'infeasible')

    def test_started_cycle_not_interrupted_and_overloads_reported(self):
        self.rows[1]['actual_background_kw'] = 5
        decisions = replay(self.rows, [self.job], self.config)
        result, load = evaluate(self.rows, [self.job], self.config, decisions)
        self.assertEqual(load[1], 6)
        self.assertEqual(result['actual_power_cap_violation_slots'], 1)
        self.assertEqual(result['scheduled_appliance_energy_kwh'], .5)

    def test_priority_and_committed_capacity(self):
        self.config['max_household_kw'] = 2
        high = Job(**{**self.job.__dict__, 'id': 'high', 'priority': 9})
        decisions = {d['appliance_id']: d for d in replay(self.rows, [self.job, high], self.config)}
        self.assertEqual(decisions['high']['start'], self.times[0])
        self.assertEqual(decisions['job']['start'], self.times[2])

    def test_rule_baseline_waits_for_complete_cheap_cycle(self):
        for row in self.rows[:3]:
            row['tariff_per_kwh'] = 10
        decisions = replay(self.rows, [self.job], self.config, 'rule_based')
        self.assertEqual(decisions[0]['start'], self.times[3])

    def test_forecast_timestamp_and_units(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'pred.csv'
            path.write_text('Time,actual_power,predicted_power\n2026-01-01 00:00:00,1200,1500\n', encoding='utf-8')
            rows, clipped = load_predictions(path)
            self.assertEqual(rows[0]['timestamp'], self.times[1])
            self.assertEqual(rows[0]['actual_background_kw'], 1.2)
            self.assertEqual(rows[0]['predicted_background_kw'], 1.5)
            self.assertEqual(clipped, 0)

    def test_call_order_and_finish(self):
        engine = RollingScheduler(self.times, [4]*8, [self.job], self.config)
        with self.assertRaises(ValueError):
            engine.step(self.times[1], 1)
        with self.assertRaises(ValueError):
            engine.finish()

    def test_date_rebasing_and_midnight(self):
        config = dict(daily_jobs=[dict(id='a', power_kw=1, duration_minutes=30, request_time='23:00',
                                     earliest_time='23:00', finish_time='24:00', max_delay_minutes=30, priority=1)])
        jobs = jobs_for_day('2015-06-01', config)
        self.assertEqual(jobs[0].latest_finish, '2015-06-02T00:00:00')


if __name__ == '__main__':
    unittest.main()
