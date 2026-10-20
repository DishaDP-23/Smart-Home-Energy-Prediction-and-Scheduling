"""Deterministic, non-preemptive appliance scheduling. All power is in kW."""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class Job:
    id: str
    power_kw: float
    duration_minutes: int
    requested_start: str
    earliest_start: str
    latest_finish: str
    max_delay_minutes: int
    priority: int = 1  # Larger number means higher priority.


def validate(rows, jobs, config):
    step = config['slot_minutes']
    if not isinstance(step, int) or step <= 0 or not rows:
        raise ValueError('Positive integer slot_minutes and nonempty rows required')
    times = [datetime.fromisoformat(r['timestamp']) for r in rows]
    if any(b - a != timedelta(minutes=step) for a, b in zip(times, times[1:])):
        raise ValueError('Timestamps must be ordered, unique and evenly spaced')
    for row in rows:
        for field in ('predicted_background_kw', 'tariff_per_kwh'):
            value = float(row[field])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{field} must be finite and nonnegative')
    if len({j.id for j in jobs}) != len(jobs):
        raise ValueError('Job IDs must be unique')
    for j in jobs:
        if not math.isfinite(j.power_kw) or j.power_kw <= 0:
            raise ValueError('Job power must be finite and positive')
        if j.duration_minutes <= 0 or j.duration_minutes % step:
            raise ValueError('Job duration must be a positive multiple of slot_minutes')
        if j.max_delay_minutes < 0:
            raise ValueError('Maximum delay cannot be negative')
        for value in (j.requested_start, j.earliest_start, j.latest_finish):
            if (datetime.fromisoformat(value) - times[0]).total_seconds() % (step * 60):
                raise ValueError('Job timestamps must align with the slot grid')
        if datetime.fromisoformat(j.earliest_start) >= datetime.fromisoformat(j.latest_finish):
            raise ValueError('Operating window must have positive length')
    cap = config.get('max_household_kw')
    if cap is not None and (not math.isfinite(cap) or cap <= 0):
        raise ValueError('Power cap must be finite and positive')
    for key in ('cost_weight', 'peak_weight', 'delay_weight', 'off_peak_threshold'):
        if key in config and (not math.isfinite(config[key]) or config[key] < 0):
            raise ValueError(f'{key} must be finite and nonnegative')
    return times


def schedule(rows, jobs, config, method='adaptive'):
    """Return JSON-compatible decisions without using actual demand.

    requested_start is a release time: no method may start a job earlier.
    Adaptive and rule-based schedules enforce a forecast household power cap.
    The unscheduled baseline starts at request time; evaluation checks violations.
    """
    if method not in ('unscheduled', 'rule_based', 'adaptive'):
        raise ValueError('Unknown scheduling method')
    times = validate(rows, jobs, config)
    step = config['slot_minutes']
    load = [float(r['predicted_background_kw']) for r in rows]
    tariff = [float(r['tariff_per_kwh']) for r in rows]
    cap = config.get('max_household_kw')
    decisions = []
    for job in sorted(jobs, key=lambda j: (-j.priority, j.requested_start, j.id)):
        duration = job.duration_minutes // step
        request = datetime.fromisoformat(job.requested_start)
        earliest = max(request, datetime.fromisoformat(job.earliest_start))
        deadline = datetime.fromisoformat(job.latest_finish)
        latest = request + timedelta(minutes=job.max_delay_minutes)
        candidates = []
        for i, start in enumerate(times):
            if i + duration > len(rows):
                continue
            if method == 'unscheduled':
                if start == request:
                    candidates.append(i)
                continue
            if start < earliest or start > latest or start + timedelta(minutes=job.duration_minutes) > deadline:
                continue
            if cap is not None and any(load[k] + job.power_kw > cap + 1e-9 for k in range(i, i + duration)):
                continue
            candidates.append(i)
        if not candidates:
            decisions.append({'appliance_id': job.id, 'status': 'infeasible', 'start': None,
                              'end': None, 'power_kw': job.power_kw,
                              'reason': 'No feasible slot in the horizon under this method'})
            continue
        def score(i):
            trial = load.copy()
            for k in range(i, i + duration):
                trial[k] += job.power_kw
            cost = job.power_kw * step / 60 * sum(tariff[i:i + duration])
            delay_hours = (times[i] - request).total_seconds() / 3600
            return (config.get('cost_weight', 1) * cost
                    + config.get('peak_weight', 1) * max(trial)
                    + config.get('delay_weight', 0.1) * delay_hours, i)
        if method == 'adaptive':
            chosen = min(candidates, key=score)
        elif method == 'rule_based':
            threshold = config['off_peak_threshold']
            off_peak = [i for i in candidates if all(t <= threshold for t in tariff[i:i + duration])]
            chosen = min(off_peak or candidates)
        else:
            chosen = candidates[0]
        for k in range(chosen, chosen + duration):
            load[k] += job.power_kw
        start = times[chosen]
        decisions.append({'appliance_id': job.id, 'status': 'scheduled',
                          'start': start.isoformat(),
                          'end': (start + timedelta(minutes=job.duration_minutes)).isoformat(),
                          'power_kw': job.power_kw,
                          'delay_minutes': (start - request).total_seconds() / 60})
    return decisions
