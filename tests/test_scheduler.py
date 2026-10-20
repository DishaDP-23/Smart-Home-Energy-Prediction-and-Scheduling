import copy
import unittest
from scheduler import Job, schedule
from evaluate import evaluate


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{'timestamp': f'2026-01-01T00:{i*15:02d}:00',
                      'predicted_background_kw': 1, 'actual_background_kw': 1,
                      'tariff_per_kwh': t} for i, t in enumerate([10, 10, 2, 2])]
        self.config = {'slot_minutes': 15, 'max_household_kw': 3,
                       'off_peak_threshold': 2, 'peak_weight': 1, 'delay_weight': 0}
        self.job = Job('test', 1, 30, '2026-01-01T00:00:00',
                       '2026-01-01T00:00:00', '2026-01-01T01:00:00', 30)

    def test_cheapest_slot_and_energy_conservation(self):
        decisions = schedule(self.rows, [self.job], self.config)
        self.assertEqual(decisions[0]['start'], '2026-01-01T00:30:00')
        result, _ = evaluate(self.rows, [self.job], self.config, decisions)
        self.assertAlmostEqual(result['scheduled_appliance_energy_kwh'], 0.5)
        self.assertEqual(result['job_constraint_violations'], 0)
        self.assertEqual(result['estimated_cost'], 7)

    def test_unreachable_cap_reports_unserved(self):
        self.config['max_household_kw'] = 1.5
        decisions = schedule(self.rows, [self.job], self.config)
        self.assertEqual(decisions[0]['status'], 'infeasible')
        result, _ = evaluate(self.rows, [self.job], self.config, decisions)
        self.assertEqual(result['unserved_jobs'], 1)
        self.assertEqual(result['unserved_energy_kwh'], 0.5)

    def test_no_actual_data_leakage_and_realized_cap_check(self):
        before = schedule(self.rows, [self.job], self.config)
        for row in self.rows:
            row['actual_background_kw'] = 4
        self.assertEqual(before, schedule(self.rows, [self.job], self.config))
        result, _ = evaluate(self.rows, [self.job], self.config, before)
        self.assertEqual(result['actual_power_cap_violation_slots'], 4)

    def test_delay_limit(self):
        job = Job(**{**self.job.__dict__, 'max_delay_minutes': 0})
        for method in ('unscheduled', 'rule_based', 'adaptive'):
            self.assertEqual(schedule(self.rows, [job], self.config, method)[0]['start'], job.requested_start)

    def test_jobs_share_capacity(self):
        self.config['max_household_kw'] = 2
        other = Job(**{**self.job.__dict__, 'id': 'second'})
        decisions = schedule(self.rows, [self.job, other], self.config)
        result, _ = evaluate(self.rows, [self.job, other], self.config, decisions)
        self.assertEqual(result['scheduled_jobs'], 2)
        self.assertEqual(result['actual_power_cap_violation_slots'], 0)

    def test_invalid_grid_and_nan(self):
        rows = copy.deepcopy(self.rows)
        rows[1]['timestamp'] = rows[0]['timestamp']
        with self.assertRaises(ValueError):
            schedule(rows, [self.job], self.config)
        self.rows[0]['predicted_background_kw'] = float('nan')
        with self.assertRaises(ValueError):
            schedule(self.rows, [self.job], self.config)

    def test_horizon_and_finish_are_enforced(self):
        job = Job(**{**self.job.__dict__, 'duration_minutes': 75})
        self.assertEqual(schedule(self.rows, [job], self.config)[0]['status'], 'infeasible')
        job = Job(**{**self.job.__dict__, 'latest_finish': '2026-01-01T00:15:00'})
        self.assertEqual(schedule(self.rows, [job], self.config)[0]['status'], 'infeasible')


if __name__ == '__main__':
    unittest.main()
