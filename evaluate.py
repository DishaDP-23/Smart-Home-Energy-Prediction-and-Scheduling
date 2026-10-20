"""Compare schedules on the same actual background, requests and tariff."""
import argparse
import csv
import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from scheduler import Job, schedule, validate

ROOT = Path(__file__).resolve().parent


def evaluate(rows, jobs, config, decisions):
    times = validate(rows, jobs, config)
    step = config['slot_minutes']
    load = [float(r['actual_background_kw']) for r in rows]
    if any(not math.isfinite(x) or x < 0 for x in load):
        raise ValueError('Actual demand must be finite and nonnegative')
    lookup = {j.id: j for j in jobs}
    if len(decisions) != len(jobs) or {d['appliance_id'] for d in decisions} != set(lookup):
        raise ValueError('Exactly one decision is required per job')
    delays, violations, scheduled, energy = [], 0, 0, 0.0
    for decision in decisions:
        if decision['status'] == 'infeasible':
            continue
        if decision['status'] != 'scheduled':
            raise ValueError('Unknown decision status')
        job = lookup[decision['appliance_id']]
        start, end = (datetime.fromisoformat(decision[k]) for k in ('start', 'end'))
        if start not in times or end > times[-1] + timedelta(minutes=step):
            raise ValueError('Schedule falls outside evaluation horizon')
        if end - start != timedelta(minutes=job.duration_minutes):
            raise ValueError('Schedule does not preserve job duration')
        if decision['power_kw'] != job.power_kw:
            raise ValueError('Schedule does not preserve appliance power')
        delay = (start - datetime.fromisoformat(job.requested_start)).total_seconds() / 60
        violations += int(start < datetime.fromisoformat(job.earliest_start))
        violations += int(end > datetime.fromisoformat(job.latest_finish))
        violations += int(delay < 0 or delay > job.max_delay_minutes)
        first = times.index(start)
        for i in range(first, first + job.duration_minutes // step):
            load[i] += job.power_kw
        delays.append(delay)
        scheduled += 1
        energy += job.power_kw * job.duration_minutes / 60
    cap = config.get('max_household_kw')
    cap_slots = sum(x > cap + 1e-9 for x in load) if cap is not None else 0
    metrics = {
        'peak_kw': max(load),
        'estimated_cost': sum(p * float(r['tariff_per_kwh']) * step / 60 for p, r in zip(load, rows)),
        'total_energy_kwh': sum(load) * step / 60,
        'scheduled_appliance_energy_kwh': energy,
        'mean_delay_minutes': sum(delays) / len(delays) if delays else None,
        'max_delay_minutes': max(delays) if delays else None,
        'scheduled_jobs': scheduled,
        'unserved_jobs': len(jobs) - scheduled,
        'unserved_energy_kwh': sum(j.power_kw * j.duration_minutes / 60 for j in jobs) - energy,
        'job_constraint_violations': violations,
        'actual_power_cap_violation_slots': cap_slots,
    }
    return metrics, load


def run(input_path, config_path, output_path):
    with Path(input_path).open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    jobs = [Job(**item) for item in config['jobs']]
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    metrics, profiles, schedules = [], {}, {}
    for method in ('unscheduled', 'rule_based', 'adaptive'):
        decisions = schedule(rows, jobs, config, method)
        result, profiles[method] = evaluate(rows, jobs, config, decisions)
        metrics.append({'method': method, **result})
        schedules[method] = decisions
    # Savings are comparable only when all requested work is served feasibly.
    baseline = metrics[0]
    for result in metrics:
        comparable = all(r['unserved_jobs'] == 0 and r['job_constraint_violations'] == 0
                         for r in (baseline, result))
        result['same_work_comparable'] = comparable
        for metric in ('peak_kw', 'estimated_cost'):
            result[metric + '_reduction_percent'] = (
                100 * (baseline[metric] - result[metric]) / baseline[metric]
                if comparable and baseline[metric] > 0 else None)
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    (output / 'schedules.json').write_text(json.dumps(schedules, indent=2), encoding='utf-8')
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    with (output / 'load_profiles.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'actual_background_kw', *profiles])
        for i, row in enumerate(rows):
            writer.writerow([row['timestamp'], row['actual_background_kw'], *[v[i] for v in profiles.values()]])
    os.environ.setdefault('MPLCONFIGDIR', str(output.resolve() / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    times = [datetime.fromisoformat(r['timestamp']) for r in rows]
    fig, ax = plt.subplots(figsize=(11, 5))
    for method, load in profiles.items():
        ax.step(times, load, where='post', label=method.replace('_', ' ').title())
    if config.get('max_household_kw') is not None:
        ax.axhline(config['max_household_kw'], color='red', linestyle='--', label='Power cap')
    ax.set(title='Schedule comparison — actual background + appliance loads', ylabel='Interval-average power (kW)', xlabel='Time')
    ax.legend()
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / 'demand_comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, key, label in zip(axes, ['peak_kw', 'estimated_cost', 'mean_delay_minutes'],
                              ['Peak (kW)', f"Cost ({config.get('currency', 'units')})", 'Mean delay (minutes)']):
        ax.bar(['Unscheduled', 'Rule-based', 'Adaptive'], [r[key] or 0 for r in metrics], color=['#64748b', '#f59e0b', '#0d9488'])
        ax.set_title(label)
        ax.tick_params(axis='x', rotation=20)
    fig.suptitle('Scheduling metrics — inspect unserved jobs before comparing savings')
    fig.tight_layout()
    fig.savefig(output / 'metrics_comparison.png', dpi=160)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/sample_predictions.csv')
    parser.add_argument('--config', type=Path, default=ROOT / 'config/appliances.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    run(args.input, args.config, args.output)
