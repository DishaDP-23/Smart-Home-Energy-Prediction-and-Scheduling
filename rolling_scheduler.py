"""Causal 15-minute dispatch for Person 1's one-step aggregate forecast.

The recorded household is an exogenous base load. Jobs are NEW simulated
loads, never the same appliance cycles already present in the recording.
"""
from datetime import datetime, timedelta
import math
from scheduler import validate


class RollingScheduler:
    """Stateful API: step() accepts exactly one forecast, not a future series.

    Once started, jobs run to completion. For admission checks only, the current
    one-step forecast is held constant over a job's duration; this is a declared
    persistence assumption, not a multi-step XGBoost forecast or a guarantee.
    """
    def __init__(self, timestamps, tariffs, jobs, config, method='adaptive'):
        if method not in ('adaptive', 'rule_based', 'unscheduled'):
            raise ValueError('Unknown method')
        if len(timestamps) != len(tariffs):
            raise ValueError('One tariff is required per timestamp')
        rows = [dict(timestamp=t, tariff_per_kwh=p, predicted_background_kw=0)
                for t, p in zip(timestamps, tariffs)]
        self.times = validate(rows, jobs, config)
        self.tariffs = [float(t) for t in tariffs]
        self.jobs = sorted(jobs, key=lambda j: (-j.priority, j.requested_start, j.id))
        self.config = dict(config)
        self.method = method
        self.minutes = config['slot_minutes']
        self.committed_kw = [0.0] * len(timestamps)
        self.decisions = {}
        self.next_index = 0
        target = config.get('preferred_total_kw', 3.5)
        if not math.isfinite(target) or target <= 0:
            raise ValueError('preferred_total_kw must be finite and positive')

    def _bounds(self, job):
        request = datetime.fromisoformat(job.requested_start)
        earliest = max(request, datetime.fromisoformat(job.earliest_start))
        latest = min(request + timedelta(minutes=job.max_delay_minutes),
                     datetime.fromisoformat(job.latest_finish) - timedelta(minutes=job.duration_minutes),
                     self.times[-1] + timedelta(minutes=self.minutes - job.duration_minutes))
        return request, earliest, latest

    def _reject(self, job, reason):
        result = dict(appliance_id=job.id, status='infeasible', start=None, end=None,
                      power_kw=job.power_kw, reason=reason)
        self.decisions[job.id] = result
        return result

    def step(self, timestamp, predicted_aggregate_kw):
        """Dispatch at the start of the predicted interval, after prior readings arrive.

        Only known requests, tariffs, commitments and this forecast affect decisions.
        Returns new started/rejected decisions; pending jobs stay internal.
        """
        i = self.next_index
        if i >= len(self.times) or datetime.fromisoformat(timestamp) != self.times[i]:
            raise ValueError('Call step once per interval, in exact chronological order')
        prediction = float(predicted_aggregate_kw)
        if not math.isfinite(prediction) or prediction < 0:
            raise ValueError('Forecast must be finite and nonnegative')
        now = self.times[i]
        updates = []
        for job in self.jobs:
            if job.id in self.decisions:
                continue
            request, earliest, latest = self._bounds(job)
            if now < request:
                continue
            duration = job.duration_minutes // self.minutes
            if self.method == 'unscheduled':
                if now != request or i + duration > len(self.times):
                    updates.append(self._reject(job, 'Requested start outside replay horizon'))
                    continue
            else:
                if now > latest:
                    updates.append(self._reject(job, 'Latest feasible start passed'))
                    continue
                if now < earliest:
                    continue
                peak = max(prediction + self.committed_kw[k] + job.power_kw
                           for k in range(i, i + duration))
                cap = self.config.get('max_household_kw')
                if cap is not None and peak > cap + 1e-9:
                    if now == latest:
                        updates.append(self._reject(job, 'Forecast admission cap blocks final start'))
                    continue
                candidates = [k for k in range(i, len(self.times) - duration + 1)
                              if self.times[k] <= latest]
                if self.method == 'rule_based':
                    off_peak = [k for k in candidates if all(
                        t <= self.config['off_peak_threshold'] for t in self.tariffs[k:k + duration])]
                    # No off-peak window remaining: start now if admission is feasible.
                    if off_peak and i != off_peak[0]:
                        continue
                else:
                    current_price = sum(self.tariffs[i:i + duration])
                    cheapest_price = min(sum(self.tariffs[k:k + duration]) for k in candidates)
                    favorable = (current_price <= cheapest_price + 1e-9
                                 and peak <= self.config.get('preferred_total_kw', 3.5))
                    # At the final allowed start, relax preferences, never the cap.
                    if not favorable and now != latest:
                        continue
            for k in range(i, i + duration):
                self.committed_kw[k] += job.power_kw
            result = dict(appliance_id=job.id, status='scheduled', start=now.isoformat(),
                          end=(now + timedelta(minutes=job.duration_minutes)).isoformat(),
                          power_kw=job.power_kw, delay_minutes=(now-request).total_seconds()/60)
            self.decisions[job.id] = result
            updates.append(result)
        self.next_index += 1
        return updates

    def finish(self):
        if self.next_index != len(self.times):
            raise ValueError('Replay must finish before finalizing decisions')
        for job in self.jobs:
            if job.id not in self.decisions:
                self._reject(job, 'Request/window outside replay horizon')
        return list(self.decisions.values())


def replay(rows, jobs, config, method='adaptive'):
    engine = RollingScheduler([r['timestamp'] for r in rows],
                              [r['tariff_per_kwh'] for r in rows], jobs, config, method)
    for row in rows:
        engine.step(row['timestamp'], row['predicted_background_kw'])
    return engine.finish()
