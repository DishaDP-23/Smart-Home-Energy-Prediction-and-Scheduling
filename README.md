# Smart Home Energy Prediction and Scheduling

Person 1's REFIT/XGBoost pipeline is integrated with Person 2's **rolling 15-minute scheduler**. The earlier synthetic day-ahead demo remains a separate example.

## What this experiment means

The input is real REFIT House 1 aggregate and appliance measurements. Person 1's model forecasts **next-interval total household power in watts**. We retain that target and feature set.

The scheduler adds **new, simulated appliance jobs** on top of the recorded household. They are not the same appliances already included in the recording. This avoids double-counting, but does **not** establish savings from shifting measured REFIT cycles. No channels are subtracted and no guessed appliance mapping is used. Job powers, durations, windows, priorities and tariffs are explicit experimental assumptions.

## Run locally

This checkout has a `.venv` with dependencies installed. In PowerShell:

```powershell
.\run_refit.ps1
```

This reads `house1_15min.csv`, preserves the source, checks data quality, writes prepared data, runs Person 1's feature creation and XGBoost training, and evaluates all complete held-out days.

Replay existing predictions without retraining:

```powershell
.\run_refit.ps1 -SkipTraining
```

This does not incorporate dataset/model changes. For another source, run `.\.venv\Scripts\python.exe run_refit.py --input "path/to/house1_15min.csv"`.

On a new machine with Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run_refit.ps1
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Files

| File | Purpose |
|---|---|
| `src/data_preprocessing.py` | Original raw preprocessing; not rerun for the already-resampled input |
| `src/prepare_training_data.py` | Person 1's historical features and chronological 70/15/15 split |
| `src/train_model.py` | Person 1's model/parameters; added configurable output paths and explicit target timestamps |
| `run_refit.py` | Validation and integrated pipeline runner |
| `predictor.py` | Live inference adapter matching the trained feature schema |
| `rolling_scheduler.py` | Stateful dispatch using one forecast at a time |
| `config/rolling_appliances.json` | Simulated daily requests, cap, preferred load threshold and illustrative tariffs |
| `evaluate_rolling.py` | Multi-day replay, fair comparisons and graphs |
| `evaluate.py` | Shared actual-load evaluation and original demo |
| `scheduler.py` | Shared Job/validation and original day-ahead algorithm; not used to select REFIT dispatch times |

## Rolling decisions

Every 15 minutes, the previous interval's measurements become available. XGBoost predicts the next interval; the scheduler decides which waiting jobs to start now. It never reads future forecasts or actual future readings. Tariffs and declared job requests are known in advance.

1. Process eligible jobs by priority (larger number first).
2. Enforce earliest start, finish deadline, maximum start delay and sufficient remaining replay time.
3. Check admission capacity including previously started jobs. For the remainder of a cycle, **hold the current aggregate forecast constant**. This is a persistence assumption, not an XGBoost multi-step forecast.
4. Adaptive dispatch prefers a cheapest remaining tariff window and projected total below `preferred_total_kw`. At the last allowed start, relax preferences but retain the admission cap.
5. Started jobs run uninterrupted. Later forecast errors can cause actual overload; report it instead of canceling a cycle or hiding it.
6. Jobs that cannot start are explicitly unserved. This greedy heuristic does not guarantee a globally optimal schedule.

Unscheduled operation starts at request time. The rule-based baseline waits for the earliest complete off-peak cycle, subject to the same forecast admission cap; if none remains, it attempts to start immediately. No method reads future demand. Sample dryer jobs are independent requests: washer-to-dryer dependencies are not modeled.

## Timestamp and unit contract

- REFIT `Time` is the start of a measured 15-minute interval.
- A training row at `Time=t` uses readings through that interval and predicts `t+15 minutes`.
- Exported predictions retain legacy `Time` and add `target_time` and `forecast_issued_at`, both `t+15 minutes`, assuming zero processing latency.
- The adapter checks alignment and converts W to kW. Negative forecasts are clipped and counted. Invalid actual values cause an error.
- Power is interval-average kW; energy is kWh. These peaks are not instantaneous electrical peaks.
- Timestamps have no timezone attached. Preserve their recorded clock; no timezone or daylight-saving conversion is inferred.

The source has missing intervals. Invalid/negative readings are masked, never filled with zero. Person 1's feature creation drops incomplete historical windows. Evaluation uses complete 96-interval test days and lists every omitted date, including entirely absent days. Scheduler parameters are fixed before testing; no test-set tuning is performed.

## Outputs

Inside `results/rolling/`:

- `data_quality.json`: source hash, dates, gaps and invalid counts.
- `model/`: locally regenerated predictions, model metrics and graphs. Original committed results remain unchanged.
- `daily_metrics.csv`, `summary.csv`, `report.json`: per-day and summary comparisons.
- `load_profiles.csv`, `schedules.json`: replay data and dispatch decisions.
- `first_day_comparison.png`: first complete test day, not selected for best savings.
- `summary_comparison.png`: comparable-day peak/cost and all-day unserved jobs.

Peak reduction refers to **mean daily peak**, not the single maximum across the dataset. Costs use illustrative tariffs, not real bills. Savings use the common days on which all three methods serve every job without window/delay violations. Unserved counts and overloads remain visible across all days. Existing aggregate overload is separated from new overload intervals caused by added jobs. Comparable savings do not imply zero overload.

Moving identical jobs preserves energy. Rule-based and adaptive scheduling can have identical costs but different peak and delay. Report all metrics, including worse outcomes. Models are regenerated locally; do not assume bit-for-bit reproduction of another machine's results.

## Person 3 integration

```python
from predictor import Predictor
from rolling_scheduler import RollingScheduler

# times: dispatch timestamps; tariffs: known prices; jobs: list[Job]
engine = RollingScheduler(times, tariffs, jobs, config, method='adaptive')
predictor = Predictor()  # trusted, locally generated model
# After each completed interval, history contains Time and REFIT columns:
forecast = predictor.predict_next(history)
updates = engine.step(forecast['timestamp'], forecast['predicted_aggregate_kw'])
```

Updates contain newly started/rejected jobs: appliance_id, status, start, end, power_kw and delay_minutes (served) or reason (rejected). Keep the same scheduler instance across intervals; call `finish()` after the entire horizon. The predictor requires 97 consecutive completed aggregate intervals and the latest nine appliance readings. Feed exogenous recorded readings; never feed added simulated appliance loads back into this model.

The model is Git-ignored; share separately or rerun training. MQTT, ThingsBoard and sensor replay remain Person 3's responsibilities.

## Earlier synthetic demo

`.\run_demo.ps1` runs the original day-ahead example. Its `results/` outputs are separate from REFIT results and must not be presented as REFIT findings.

Datasets, models, environments and generated replay results are Git-ignored. Nothing is automatically committed or pushed.
