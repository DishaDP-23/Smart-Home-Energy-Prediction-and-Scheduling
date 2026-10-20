"""Evaluate additional simulated jobs on held-out REFIT aggregate forecasts."""
import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from scheduler import Job
from rolling_scheduler import replay
from evaluate import evaluate

ROOT = Path(__file__).resolve().parent
METHODS = ('unscheduled', 'rule_based', 'adaptive')


def load_predictions(path):
    """Person 1 Time is feature-interval start, not the target interval start."""
    with Path(path).open(encoding='utf-8-sig') as f:
        source = list(csv.DictReader(f))
    rows, seen, clipped = [], set(), 0
    for r in source:
        feature_time = datetime.fromisoformat(r['Time'])
        target = feature_time + timedelta(minutes=15)
        if r.get('target_time') and datetime.fromisoformat(r['target_time']) != target:
            raise ValueError('Target time does not match the next-interval model')
        if r.get('forecast_issued_at') and datetime.fromisoformat(r['forecast_issued_at']) > target:
            raise ValueError('Forecast is unavailable at dispatch time')
        if target in seen:
            raise ValueError('Duplicate prediction timestamps')
        seen.add(target)
        actual, predicted = float(r['actual_power']), float(r['predicted_power'])
        if not math.isfinite(actual) or actual < 0 or not math.isfinite(predicted):
            raise ValueError('Invalid actual power or forecast')
        clipped += int(predicted < 0)
        # These compatibility keys mean exogenous recorded aggregate in this
        # simulation, NOT appliance-subtracted physical background demand.
        rows.append(dict(timestamp=target.isoformat(), actual_background_kw=actual / 1000,
                         predicted_background_kw=max(0, predicted) / 1000))
    return sorted(rows, key=lambda r: r['timestamp']), clipped


def jobs_for_day(date, config):
    midnight = datetime.fromisoformat(date)
    def stamp(clock):
        hour, minute = map(int, clock.split(':'))
        if not (0 <= hour <= 24 and 0 <= minute < 60) or (hour == 24 and minute):
            raise ValueError('Invalid daily job time')
        return (midnight + timedelta(hours=hour, minutes=minute)).isoformat()
    return [Job(id=j['id'], power_kw=j['power_kw'], duration_minutes=j['duration_minutes'],
                requested_start=stamp(j['request_time']), earliest_start=stamp(j['earliest_time']),
                latest_finish=stamp(j['finish_time']), max_delay_minutes=j['max_delay_minutes'],
                priority=j['priority']) for j in config['daily_jobs']]


def tariff_at(timestamp, config):
    t = datetime.fromisoformat(timestamp)
    hour = t.hour + t.minute / 60
    matching = [p['rate'] for p in config['tariff_periods'] if p['start_hour'] <= hour < p['end_hour']]
    if len(matching) != 1:
        raise ValueError('Tariff periods must cover each interval exactly once')
    return matching[0]


def write_csv(path, records):
    if not records:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run(predictions, config_path, output):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    if config['slot_minutes'] != 15:
        raise ValueError('Person 1 model requires 15-minute slots')
    rows, clipped = load_predictions(predictions)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    by_day = defaultdict(list)
    for row in rows:
        by_day[row['timestamp'][:10]].append(row)
    if rows:
        date = datetime.fromisoformat(rows[0]['timestamp'][:10])
        last_date = datetime.fromisoformat(rows[-1]['timestamp'][:10])
        while date <= last_date:
            by_day.setdefault(date.date().isoformat(), [])
            date += timedelta(days=1)
    daily, profiles, schedules, omitted = [], [], {}, []
    for date, day in sorted(by_day.items()):
        expected = [(datetime.fromisoformat(date) + timedelta(minutes=15*i)).isoformat() for i in range(96)]
        if [r['timestamp'] for r in day] != expected:
            omitted.append(dict(date=date, intervals=len(day), reason='Incomplete day or gaps'))
            continue
        for row in day:
            row['tariff_per_kwh'] = tariff_at(row['timestamp'], config)
        jobs = jobs_for_day(date, config)
        schedules[date], loads = {}, {}
        day_metrics = []
        base_cap_slots = sum(r['actual_background_kw'] > config['max_household_kw'] + 1e-9 for r in day)
        for method in METHODS:
            decisions = replay(day, jobs, config, method)
            metrics, loads[method] = evaluate(day, jobs, config, decisions)
            schedules[date][method] = decisions
            job_cost = sum((p - r['actual_background_kw']) * r['tariff_per_kwh'] * .25
                           for p, r in zip(loads[method], day))
            metrics.update(date=date, method=method, added_job_cost=job_cost,
                           base_already_over_cap_slots=base_cap_slots,
                           new_over_cap_slots=sum(p > config['max_household_kw'] + 1e-9 and
                                                  r['actual_background_kw'] <= config['max_household_kw'] + 1e-9
                                                  for p, r in zip(loads[method], day)))
            day_metrics.append(metrics)
        comparable = all(m['unserved_jobs'] == 0 and m['job_constraint_violations'] == 0 for m in day_metrics)
        for metrics in day_metrics:
            metrics['all_methods_same_work'] = comparable
        daily.extend(day_metrics)
        for i, r in enumerate(day):
            profiles.append(dict(timestamp=r['timestamp'], actual_aggregate_kw=r['actual_background_kw'],
                                 predicted_aggregate_kw=r['predicted_background_kw'], tariff_per_kwh=r['tariff_per_kwh'],
                                 **{method + '_kw': loads[method][i] for method in METHODS}))
    if not daily:
        raise ValueError('No complete held-out days available')
    comparable_dates = sorted({r['date'] for r in daily if r['all_methods_same_work']})
    summary = []
    for method in METHODS:
        items = [r for r in daily if r['method'] == method]
        fair = [r for r in items if r['all_methods_same_work']]
        served = sum(r['scheduled_jobs'] for r in items)
        summary.append(dict(method=method, evaluated_days=len(items), comparable_days=len(fair),
                            scheduled_jobs=served, unserved_jobs=sum(r['unserved_jobs'] for r in items),
                            unserved_energy_kwh=sum(r['unserved_energy_kwh'] for r in items),
                            job_constraint_violations=sum(r['job_constraint_violations'] for r in items),
                            actual_power_cap_violation_slots=sum(r['actual_power_cap_violation_slots'] for r in items),
                            base_already_over_cap_slots=sum(r['base_already_over_cap_slots'] for r in items),
                            new_over_cap_slots=sum(r['new_over_cap_slots'] for r in items),
                            mean_served_delay_minutes=sum((r['mean_delay_minutes'] or 0)*r['scheduled_jobs'] for r in items)/served if served else None,
                            comparable_mean_daily_peak_kw=sum(r['peak_kw'] for r in fair)/len(fair) if fair else None,
                            comparable_total_cost=sum(r['estimated_cost'] for r in fair) if fair else None,
                            comparable_added_job_cost=sum(r['added_job_cost'] for r in fair) if fair else None))
    baseline = summary[0]
    for s in summary:
        for key in ('comparable_mean_daily_peak_kw', 'comparable_total_cost'):
            s[key + '_reduction_percent'] = (100*(baseline[key]-s[key])/baseline[key]
                                             if baseline[key] else None)
    write_csv(output / 'daily_metrics.csv', daily)
    write_csv(output / 'summary.csv', summary)
    write_csv(output / 'load_profiles.csv', profiles)
    (output / 'schedules.json').write_text(json.dumps(schedules, indent=2), encoding='utf-8')
    report = dict(scenario='REFIT aggregate plus additional simulated appliances',
                  forecast_horizon_minutes=15, admission_assumption='Current forecast held constant for job duration',
                  complete_test_days=len(schedules), excluded_days=omitted,
                  comparable_dates=comparable_dates, negative_predictions_clipped=clipped,
                  tariffs='Illustrative demo_units/kWh; not an actual electricity tariff', summary=summary)
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    plot(profiles, daily, summary, config, output)
    print(json.dumps(dict(complete_test_days=len(schedules), excluded_days=len(omitted), summary=summary), indent=2))
    return report


def plot(profiles, daily, summary, config, output):
    os.environ.setdefault('MPLCONFIGDIR', str(output.resolve() / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Fixed first complete day: do not cherry-pick the largest savings.
    first = profiles[:96]
    fig, ax = plt.subplots(figsize=(12, 5))
    times = [datetime.fromisoformat(r['timestamp']) for r in first]
    ax.step(times, [r['actual_aggregate_kw'] for r in first], where='post', linestyle=':', color='black', label='Recorded REFIT aggregate')
    for method in METHODS:
        ax.step(times, [r[method+'_kw'] for r in first], where='post', label=method.replace('_', ' ').title())
    ax.axhline(config['max_household_kw'], color='red', linestyle='--', label='Power cap')
    ax.set(title='First complete held-out day: REFIT + additional simulated jobs', ylabel='Interval-average power (kW)', xlabel='Time')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / 'first_day_comparison.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    keys = ['comparable_mean_daily_peak_kw', 'comparable_total_cost', 'unserved_jobs']
    titles = ['Mean daily peak (kW)\nComparable days only', 'Total cost (demo units)\nComparable days only', 'Unserved jobs\nAll complete days']
    for ax, key, title in zip(axes, keys, titles):
        values = [s[key] for s in summary]
        if any(v is None for v in values):
            ax.text(.5, .5, 'No comparable days', ha='center', transform=ax.transAxes)
        else:
            bars = ax.bar(['Unscheduled', 'Rule-based', 'Adaptive'], values, color=['#64748b', '#f59e0b', '#0d9488'])
            ax.bar_label(bars, fmt='%.0f' if key == 'unserved_jobs' else '%.2f', padding=3)
            ax.set_ylim(0, max(1, max(values) * 1.15))
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=20)
    fig.suptitle('Rolling scheduling — real aggregate, simulated jobs and tariffs')
    fig.tight_layout()
    fig.savefig(output / 'summary_comparison.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, default=ROOT / 'results/rolling/model/test_predictions.csv')
    parser.add_argument('--config', type=Path, default=ROOT / 'config/rolling_appliances.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/rolling')
    args = parser.parse_args()
    run(args.predictions, args.config, args.output)
