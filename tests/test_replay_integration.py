import ast
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd

from replay_controller import ReplayController, telemetry_values, write_state


class FakePredictor:
    def predict_next(self, history):
        assert len(history) == 97
        return {'timestamp': (pd.Timestamp(history.Time.iloc[-1])+pd.Timedelta(minutes=15)).isoformat(),
                'predicted_aggregate_kw': float(history.Aggregate.iloc[-1])/1000}


class ReplayIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).parents[1]/'config/rolling_appliances.json').read_text())
        self.controller = ReplayController(FakePredictor(), self.config)

    def row(self, time, power=500):
        return dict(Time=pd.Timestamp(time), Aggregate=power,
                    **{f'Appliance{i}': 1 for i in range(1,10)})

    def warmup(self):
        for t in pd.date_range('2015-03-19 20:45', periods=97, freq='15min'):
            state = self.controller.observe(self.row(t))
        return state

    def test_causal_forecast_and_scheduling(self):
        state = self.warmup()
        self.assertEqual(state['forecast']['timestamp'], '2015-03-20T21:00:00')
        self.assertEqual(state['status'], 'running')
        self.assertTrue(state['updates'])
        self.assertEqual(state['current_simulated_appliance_kw'], 0)
        next_state = self.controller.observe(self.row('2015-03-20 21:00', 600))
        self.assertGreater(next_state['current_simulated_appliance_kw'], 0)
        self.assertEqual(next_state['forecast']['predicted_aggregate_kw'], .6)
        self.assertAlmostEqual(next_state['current_total_kw'], .6+next_state['current_simulated_appliance_kw'])

    def test_gap_resets_warmup_and_exposes_interruption(self):
        self.warmup()
        state = self.controller.observe(self.row('2015-03-20 22:00'))
        self.assertIsNone(state['forecast'])
        self.assertEqual(state['history_intervals'], 1)
        self.assertEqual(state['current_simulated_appliance_kw'], 0)
        self.assertEqual(state['replay_events'][0]['status'], 'replay_gap')

    def test_duplicate_or_backwards_time_rejected(self):
        self.controller.observe(self.row('2015-03-20 12:00'))
        with self.assertRaises(ValueError):
            self.controller.observe(self.row('2015-03-20 12:00'))

    def test_telemetry_includes_forecast_and_schedule(self):
        state = self.warmup()
        values = telemetry_values({'timestamp': state['observed_timestamp'], 'total_power_w':500}, state)
        self.assertEqual(values['predicted_aggregate_power_w'], 500)
        self.assertEqual(json.loads(values['schedule_json']), state['schedule'])
        self.assertEqual(values['forecast_timestamp'], '2015-03-20T21:00:00')

    def test_atomic_snapshot_roundtrip(self):
        state = self.warmup()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'state.json'
            write_state(path,state)
            self.assertEqual(json.loads(path.read_text())['forecast'],state['forecast'])
            self.assertFalse(path.with_suffix('.tmp').exists())

    def test_interval_average_energy_and_filter(self):
        # Extract pure helpers without running the Streamlit UI at import time.
        source=(Path(__file__).parents[1]/'src/dashboard.py').read_text(encoding='utf-8')
        functions=[node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef)
                   and node.name in ('calculate_energy_kwh','filter_data_by_time_range')]
        ns={'pd':pd,'TIME_RANGES':{'Last 15 minutes':pd.Timedelta(minutes=15)}}
        exec(compile(ast.Module(body=functions,type_ignores=[]),'dashboard_helpers','exec'),ns)
        data=pd.DataFrame({'timestamp':pd.date_range('2015-03-20',periods=2,freq='15min'), 'total_power_w':[1000,1000]})
        self.assertEqual(ns['calculate_energy_kwh'](data,'total_power_w'),.5)
        filtered=ns['filter_data_by_time_range'](data,'Last 15 minutes')
        self.assertEqual(len(filtered),1)
        self.assertEqual(ns['calculate_energy_kwh'](filtered,'total_power_w'),.25)


if __name__ == '__main__':
    unittest.main()
