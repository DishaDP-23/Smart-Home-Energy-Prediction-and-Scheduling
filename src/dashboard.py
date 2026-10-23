"""
Real-Time Home Energy Monitoring Dashboard
AIoT-Based Intelligent Home Energy Management

Features:
- Automatic dashboard refresh
- Latest household power and active appliance count
- Historical time-range filter
- Estimated energy consumption and electricity cost
- Peak and average household power for the selected period
- Appliance-wise energy contribution for the selected period
- Household power graph for the selected period
- Appliance breakdown and current readings
- Sensor freshness indicator
- Recent sensor readings table
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
import json

from predictor import Predictor
from src.appliance_mapping import get_appliance_name
# --------------------------------------------------
# Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Existing dashboard data
CSV_FILE = PROJECT_ROOT / "data" / "replayed_energy.csv"

# Confirmed ThingsBoard publishes — original measurement timestamps
HISTORICAL_FILE = PROJECT_ROOT / "historical_measurements.csv"

# Original historical dataset used for XGBoost forecasting
FORECAST_DATASET = PROJECT_ROOT / "house1_15min.csv"

# Existing rolling-scheduler evaluation results
SCHEDULING_RESULTS_DIR = PROJECT_ROOT / "rolling"
SCHEDULING_SUMMARY_FILE = SCHEDULING_RESULTS_DIR / "summary.csv"
SCHEDULING_REPORT_FILE = SCHEDULING_RESULTS_DIR / "report.json"
SCHEDULES_FILE = PROJECT_ROOT / "rolling" / "schedules.json"

REFRESH_SECONDS = 2
MAX_GRAPH_READINGS = 100
MAX_ENERGY_GAP_SECONDS = 900
ELECTRICITY_TARIFF = 6.0  # Estimated ₹ per kWh

TIME_RANGES = {
    "Last 15 minutes": timedelta(minutes=15),
    "Last 1 hour": timedelta(hours=1),
    "Last 6 hours": timedelta(hours=6),
    "Last 24 hours": timedelta(hours=24),
    "All available data": None,
}

st.set_page_config(
    page_title="Smart Home Energy Monitor",
    page_icon="⚡",
    layout="wide",
)


# --------------------------------------------------
# Load and validate sensor data
# --------------------------------------------------


def load_sensor_data():
    """Read the replay CSV and preserve original model column names."""

    if not CSV_FILE.exists():
        return None

    try:
        data = pd.read_csv(CSV_FILE)
    except (
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):
        return None

    required_columns = {"timestamp", "total_power_w"}

    if data.empty or not required_columns.issubset(data.columns):
        return None

    data["timestamp"] = pd.to_datetime(
        data["timestamp"], errors="coerce"
    )

    data["total_power_w"] = pd.to_numeric(
        data["total_power_w"], errors="coerce"
    )

    data = data.dropna(
        subset=["timestamp", "total_power_w"]
    )

    if data.empty:
        return None

    data = (
        data.sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # Friendly CSV/ThingsBoard names -> original internal names.
    from src.appliance_mapping import APPLIANCE_NAMES

    reverse_mapping = {
        friendly_name: original_name
        for original_name, friendly_name in APPLIANCE_NAMES.items()
    }

    data = data.rename(columns=reverse_mapping)

    appliance_columns = [
        column
        for column in data.columns
        if column not in ("timestamp", "total_power_w")
    ]

    for column in appliance_columns:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        ).fillna(0.0)

    return data



@st.cache_resource
def load_predictor():
    """Load the trained XGBoost model once."""
    return Predictor()


@st.cache_data
def load_forecast_history():
    """Load and prepare the original recorded dataset."""
    if not FORECAST_DATASET.exists():
        raise FileNotFoundError(
            f"Forecast dataset not found: {FORECAST_DATASET}"
        )

    history = pd.read_csv(FORECAST_DATASET)

    required_columns = [
        "Time",
        "Aggregate",
        *[f"Appliance{i}" for i in range(1, 10)],
    ]

    missing = [
        column for column in required_columns
        if column not in history.columns
    ]

    if missing:
        raise ValueError(
            f"Missing forecast dataset columns: {missing}"
        )

    history = history[required_columns].copy()
    history["Time"] = pd.to_datetime(
        history["Time"], errors="coerce"
    )

    for column in required_columns[1:]:
        history[column] = pd.to_numeric(
            history[column], errors="coerce"
        )

    history = (
    history.dropna(subset=required_columns)
    .sort_values("Time")
    .drop_duplicates(subset=["Time"], keep="last")
    .reset_index(drop=True)
    )

    # Keep only the latest uninterrupted 15-minute sequence.
    if not history.empty:
        time_gaps = history["Time"].diff()
        gap_positions = time_gaps[time_gaps != pd.Timedelta(minutes=15)].index

        if len(gap_positions) > 0:
            history = history.iloc[gap_positions[-1]:].reset_index(drop=True)

    return history

# --------------------------------------------------
# Historical time-range filtering
# --------------------------------------------------

def filter_data_by_time_range(data, selected_range):
    """Return readings within the selected historical period."""

    duration = TIME_RANGES[selected_range]

    if duration is None:
        return data.copy()

    latest_timestamp = data["timestamp"].max()
    cutoff = latest_timestamp - duration

    return (
        data.loc[data["timestamp"] >= cutoff]
        .copy()
        .reset_index(drop=True)
    )


# --------------------------------------------------
# Calculate energy from consecutive readings
# --------------------------------------------------

def calculate_energy_kwh(data, power_column):
    """
    Estimate energy using the trapezoidal rule.

    Only intervals greater than zero and no longer than
    MAX_ENERGY_GAP_SECONDS are included. This avoids counting
    long periods when the simulator was stopped.
    """

    if data is None or len(data) < 2:
        return 0.0

    elapsed_seconds = (
        data["timestamp"].diff().dt.total_seconds()
    )

    power = pd.to_numeric(
        data[power_column], errors="coerce"
    )

    average_power = (
        power + power.shift(1)
    ) / 2

    valid_intervals = (
        elapsed_seconds.gt(0)
        & elapsed_seconds.le(MAX_ENERGY_GAP_SECONDS)
        & average_power.notna()
    )

    energy_wh = (
        average_power[valid_intervals]
        * elapsed_seconds[valid_intervals]
        / 3600
    ).sum()

    return float(energy_wh / 1000)


def estimate_energy_kwh(data):
    """Estimate total household energy consumption."""

    return calculate_energy_kwh(
        data, "total_power_w"
    )


def calculate_appliance_analytics(
    data, appliance_columns
):
    """Calculate estimated energy and contribution per appliance."""

    rows = []

    for appliance in appliance_columns:
        energy_kwh = calculate_energy_kwh(
            data, appliance
        )

        rows.append({
            "Appliance": get_appliance_name(appliance),
            "Energy (kWh)": energy_kwh,
            "Estimated Cost (₹)": (
                energy_kwh * ELECTRICITY_TARIFF
            ),
        })

    analytics = pd.DataFrame(
        rows,
        columns=[
            "Appliance",
            "Energy (kWh)",
            "Estimated Cost (₹)",
        ],
    )

    total_appliance_energy = (
        analytics["Energy (kWh)"].sum()
        if not analytics.empty
        else 0.0
    )

    if total_appliance_energy > 0:
        analytics["Contribution (%)"] = (
            analytics["Energy (kWh)"]
            / total_appliance_energy
            * 100
        )
    else:
        analytics["Contribution (%)"] = 0.0

    return analytics


# --------------------------------------------------
# Dashboard
# --------------------------------------------------

def display_scheduling_evaluation():
    """Display the existing rolling-scheduler evaluation results."""

    st.header("Appliance Scheduling Performance")
    st.caption(
        "Offline evaluation on recorded REFIT household consumption, "
        "with additional simulated appliance jobs."
    )

    if not SCHEDULING_SUMMARY_FILE.exists():
        st.warning(
            "Scheduling summary not found. Expected file: "
            f"{SCHEDULING_SUMMARY_FILE}"
        )
        return

    try:
        summary = pd.read_csv(SCHEDULING_SUMMARY_FILE)

        required = [
            "method",
            "comparable_days",
            "scheduled_jobs",
            "unserved_jobs",
            "job_constraint_violations",
            "new_over_cap_slots",
            "mean_served_delay_minutes",
            "comparable_mean_daily_peak_kw",
            "comparable_total_cost",
            "comparable_mean_daily_peak_kw_reduction_percent",
            "comparable_total_cost_reduction_percent",
        ]

        missing = [column for column in required if column not in summary.columns]

        if missing:
            st.error(f"Scheduling summary is missing columns: {missing}")
            return

        summary = summary[required].copy()
        summary["method"] = summary["method"].astype(str)

        if summary["method"].duplicated().any():
            st.error("The scheduling summary contains duplicate methods.")
            return

        if not {"unscheduled", "rule_based", "adaptive"}.issubset(
            set(summary["method"])
        ):
            st.error("The summary must contain all three scheduling methods.")
            return

        summary = summary.set_index("method").loc[
            ["unscheduled", "rule_based", "adaptive"]
        ].reset_index()

        report = {}
        if SCHEDULING_REPORT_FILE.exists():
            with open(SCHEDULING_REPORT_FILE, encoding="utf-8") as file:
                report = json.load(file)

    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        st.error(f"Could not load scheduling evaluation: {exc}")
        return

    adaptive = summary.loc[summary["method"] == "adaptive"].iloc[0]
    unscheduled = summary.loc[summary["method"] == "unscheduled"].iloc[0]

    days = int(adaptive["comparable_days"])

    st.info(
        f"Evaluation scenario: {report.get('scenario', 'Not specified')}. "
        f"Results cover {days} comparable days."
    )

    # Main adaptive-scheduler results
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Mean Daily Peak",
        f"{adaptive['comparable_mean_daily_peak_kw']:.3f} kW",
        delta=(
            f"{adaptive['comparable_mean_daily_peak_kw_reduction_percent']:.2f}% "
            "vs. unscheduled"
        ),
    )

    col2.metric(
        "Simulated Cost",
        f"{adaptive['comparable_total_cost']:,.2f} demo units",
        delta=(
            f"{adaptive['comparable_total_cost_reduction_percent']:.2f}% "
            "vs. unscheduled"
        ),
    )

    col3.metric(
        "Unserved Jobs",
        f"{int(adaptive['unserved_jobs'])}",
    )

    col4.metric(
        "New Over-Cap Slots",
        f"{int(adaptive['new_over_cap_slots'])}",
    )

    st.subheader("Compare Scheduling Strategies")

    comparison = summary.set_index("method")

    comparison = comparison.rename(
        index={
            "unscheduled": "Unscheduled",
            "rule_based": "Rule-based",
            "adaptive": "Adaptive",
        }
    )

    peak_chart = comparison[
        ["comparable_mean_daily_peak_kw"]
    ].rename(
        columns={
            "comparable_mean_daily_peak_kw": "Mean Daily Peak (kW)"
        }
    )

    cost_chart = comparison[
        ["comparable_total_cost"]
    ].rename(
        columns={
            "comparable_total_cost": "Total Simulated Cost (demo units)"
        }
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Mean daily peak demand**")
        st.bar_chart(peak_chart)

    with chart_col2:
        st.markdown("**Total simulated cost**")
        st.bar_chart(cost_chart)

    display = summary[
        [
            "method",
            "comparable_days",
            "scheduled_jobs",
            "unserved_jobs",
            "mean_served_delay_minutes",
            "comparable_mean_daily_peak_kw",
            "comparable_mean_daily_peak_kw_reduction_percent",
            "comparable_total_cost",
            "comparable_total_cost_reduction_percent",
            "new_over_cap_slots",
            "job_constraint_violations",
        ]
    ].copy()

    display = display.rename(
        columns={
            "method": "Strategy",
            "comparable_days": "Comparable Days",
            "scheduled_jobs": "Jobs Scheduled",
            "unserved_jobs": "Unserved Jobs",
            "mean_served_delay_minutes": "Mean Delay (min)",
            "comparable_mean_daily_peak_kw": "Mean Daily Peak (kW)",
            "comparable_mean_daily_peak_kw_reduction_percent":
                "Peak Reduction (%)",
            "comparable_total_cost": "Total Cost (demo units)",
            "comparable_total_cost_reduction_percent":
                "Cost Reduction (%)",
            "new_over_cap_slots": "New Over-Cap Slots",
            "job_constraint_violations": "Job Constraint Violations",
        }
    )

    display["Mean Delay (min)"] = display["Mean Delay (min)"].round(1)
    display["Mean Daily Peak (kW)"] = display["Mean Daily Peak (kW)"].round(3)
    display["Peak Reduction (%)"] = display["Peak Reduction (%)"].round(2)
    display["Total Cost (demo units)"] = (
        display["Total Cost (demo units)"].round(2)
    )
    display["Cost Reduction (%)"] = display["Cost Reduction (%)"].round(2)

    st.dataframe(display, hide_index=True, use_container_width=True)

    st.caption(
        "Methodology: the current 15-minute forecast is held constant "
        "over each job's duration for admission checks. The appliance jobs "
        "are additional simulated loads, not shifts of measured appliance "
        "cycles. Tariffs are illustrative demo units, not actual electricity "
        "rates or verified bill savings. Over-cap slots include situations "
        "where household consumption exceeds the configured power cap."
    )

    
def display_appliance_schedules():
    """Display daily schedules for the three simulated appliances."""

    st.header("Daily Appliance Schedules")

    if not SCHEDULES_FILE.exists():
        st.warning(f"Schedule file not found: {SCHEDULES_FILE}")
        return

    try:
        with open(SCHEDULES_FILE, encoding="utf-8") as file:
            schedules = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not load appliance schedules: {exc}")
        return

    if not isinstance(schedules, dict) or not schedules:
        st.info("No appliance schedules are available.")
        return

    st.caption(
        "Schedules for the simulated washing machine, dishwasher, "
        "and dryer. Existing scheduling evaluation results are shown above."
    )

    # Inspect the available dates in the schedule file.
    dates = sorted(schedules.keys())

    selected_date = st.selectbox(
        "Select schedule date",
        options=dates,
        key="appliance_schedule_date",
    )

    daily_schedules = schedules[selected_date]

    if not isinstance(daily_schedules, dict) or not daily_schedules:
        st.info("No schedules are available for this date.")
        return

    strategy_labels = {
        "unscheduled": "Unscheduled",
        "rule_based": "Rule-based",
        "adaptive": "Adaptive",
    }

    available_strategies = [
        strategy for strategy in strategy_labels
        if strategy in daily_schedules
    ]

    if not available_strategies:
        st.warning("No recognized scheduling strategies were found.")
        return

    selected_strategy = st.selectbox(
        "Select scheduling strategy",
        options=available_strategies,
        format_func=lambda strategy: strategy_labels[strategy],
        key="appliance_schedule_strategy",
    )

    jobs = daily_schedules[selected_strategy]

    if not isinstance(jobs, list) or not jobs:
        st.info("No appliance jobs are listed for this date and strategy.")
        return

    appliance_labels = {
        "sim_washing_machine": "Washing Machine",
        "sim_dishwasher": "Dishwasher",
        "sim_dryer": "Dryer",
    }

    rows = []

    for job in jobs:
        if not isinstance(job, dict):
            continue

        appliance_id = job.get("appliance_id")

        if appliance_id not in appliance_labels:
            continue

        rows.append({
            "Appliance": appliance_labels[appliance_id],
            "Status": job.get("status", "Not specified"),
            "Start": job.get("start", "—"),
            "End": job.get("end", "—"),
            "Power (kW)": job.get("power_kw"),
            "Delay (min)": job.get("delay_minutes"),
        })

    if not rows:
        st.info(
            "No matching washing machine, dishwasher, or dryer "
            "jobs were found for this strategy."
        )
        return

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
    )


def display_live_schedule(latest_power_w, forecast_kw=None):
    """Recommend appliance start slots using current power and forecast."""

    st.header("Live Appliance Scheduling")

    st.caption(
        "Recommendations use the latest replayed household reading. "
        "These are simulated recommendations, not physical appliance control."
    )

    appliances = {
        "Washing Machine": {"power_kw": 1.0, "duration_min": 60},
        "Dishwasher": {"power_kw": 1.2, "duration_min": 90},
        "Dryer": {"power_kw": 1.8, "duration_min": 60},
    }

    current_kw = float(latest_power_w) / 1000
    planning_kw = (
        max(current_kw, float(forecast_kw))
        if forecast_kw is not None
        else current_kw
    )

    # Demo planning limit; adjust to your configured household power cap.
    max_household_kw = 3.0
    available_kw = max(0.0, max_household_kw - planning_kw)

    st.metric("Latest Household Power", f"{current_kw:.2f} kW")
    st.metric("Estimated Spare Capacity", f"{available_kw:.2f} kW")

    rows = []
    for name, appliance in appliances.items():
        fits = appliance["power_kw"] <= available_kw
        rows.append({
            "Appliance": name,
            "Power (kW)": appliance["power_kw"],
            "Duration (min)": appliance["duration_min"],
            "Recommendation": (
                "Can start now (estimated)"
                if fits else "Defer — insufficient spare capacity"
            ),
        })

    st.dataframe(pd.DataFrame(rows), hide_index=True,
                 use_container_width=True)

    st.caption(
        "This is a capacity-based demo recommendation, not a full "
        "duration-aware scheduler. The 3.0 kW limit is illustrative."
    )



def display_dashboard():

    st.title("⚡ Smart Home Energy Management")

    st.caption(
       "Historical Energy Monitoring | "
"XGBoost Forecasting | Adaptive Appliance Scheduling"
    )

    data = load_sensor_data()

    if data is None:
        st.warning(
            "No valid sensor readings found. "
            "Start the simulator using "
            "`python src/virtual_sensors.py`."
        )
        return

    appliance_columns = [
        column
        for column in data.columns
        if column not in (
            "timestamp",
            "total_power_w",
        )
    ]

    latest = data.iloc[-1]

    latest_power = float(
        latest["total_power_w"]
    )

    active_appliances = sum(
        float(latest[column]) > 0
        for column in appliance_columns
    )

    # Handle timezone-aware and timezone-naive timestamps.
    # Display the timestamp of the latest recorded measurement.
    latest_timestamp = data.iloc[-1]["timestamp"]

    st.info(
        "Historical dataset replay — displaying recorded "
        f"measurements. Latest record: {latest_timestamp}"
    )

    # --------------------------------------------------
    # Current household consumption
    # Always displays the latest sensor reading.
    # --------------------------------------------------

    st.subheader(
        "Current Household Consumption"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Latest Recorded Power",
        f"{latest_power:,.2f} W",
    )

    col2.metric(
        "Appliances Active in Latest Record",
        f"{active_appliances} / "
        f"{len(appliance_columns)}",
    )

    col3.metric(
        "Latest Reading",
        str(latest_timestamp),
    )

    st.caption(
        "Current power and active appliance count "
        "always reflect the latest available reading, "
        "regardless of the historical filter."
    )
    
    # --------------------------------------------------
    # XGBoost next-interval forecast
    # --------------------------------------------------

    st.subheader("Next-Interval Energy Forecast")

    try:
        predictor = load_predictor()

        # Use the replayed measurements already loaded by the dashboard.
        forecast_columns = [
            "timestamp",
            "total_power_w",
            *[f"Appliance{i}" for i in range(1, 10)],
        ]

        missing_columns = [
            column for column in forecast_columns
            if column not in data.columns
        ]

        if missing_columns:
            st.warning(
                "Forecast unavailable. Required replay columns are "
                f"missing: {missing_columns}"
            )
        else:
            # Map replay data to the model's expected input names.
            forecast_history = data[forecast_columns].copy()
            forecast_history = forecast_history.rename(
                columns={
                    "timestamp": "Time",
                    "total_power_w": "Aggregate",
                }
            )

            forecast_history["Time"] = pd.to_datetime(
                forecast_history["Time"], errors="coerce"
            )

            numeric_columns = [
                "Aggregate",
                *[f"Appliance{i}" for i in range(1, 10)],
            ]

            for column in numeric_columns:
                forecast_history[column] = pd.to_numeric(
                    forecast_history[column], errors="coerce"
                )

            forecast_history = (
                forecast_history
                .dropna(subset=["Time", *numeric_columns])
                .sort_values("Time")
                .drop_duplicates(subset=["Time"], keep="last")
                .reset_index(drop=True)
            )

            # Find the latest uninterrupted 15-minute sequence.
            if not forecast_history.empty:
                gaps = forecast_history["Time"].diff()
                gap_positions = gaps[
                    gaps != pd.Timedelta(minutes=15)
                ].index

                if len(gap_positions) > 0:
                    forecast_history = (
                        forecast_history
                        .iloc[gap_positions[-1]:]
                        .reset_index(drop=True)
                    )

            available = len(forecast_history)

            if available < 97:
                st.info(
                    "Collecting consecutive readings for forecasting: "
                    f"{available}/97. A forecast will be available "
                    "after 97 valid consecutive readings."
                )
            else:
                history_window = forecast_history.tail(97).copy()
                forecast = predictor.predict_next(history_window)

                forecast_col1, forecast_col2 = st.columns(2)

                forecast_col1.metric(
                    "Predicted Aggregate Power",
                    f"{forecast['predicted_aggregate_kw'] * 1000:,.2f} W",
                )

                forecast_col2.metric(
                    "Forecast Timestamp",
                    pd.Timestamp(forecast["timestamp"]).strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                )

                st.caption(
                    "XGBoost prediction for the next 15-minute interval, "
                    "based on the latest consecutive replayed measurements. "
                    "This is a forecast, not a measured power reading."
                )

    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
        st.warning(f"Forecast unavailable: {exc}")

    except Exception as exc:
        st.error(
            f"An unexpected forecasting error occurred: {exc}"
        )

        
    display_live_schedule(latest_power)

    st.divider()

    display_scheduling_evaluation()
    st.divider()

    display_appliance_schedules()

    st.divider()

    # --------------------------------------------------
    # Historical time-range selector
    # --------------------------------------------------

    st.subheader("Historical Analysis")

    selected_range = st.selectbox(
        "Select a time range",
        options=list(TIME_RANGES.keys()),
        index=0,
        help=(
            "The historical chart and analytics below "
            "use this period. Current household readings "
            "always show the latest reading."
        ),
    )

    filtered_data = filter_data_by_time_range(
        data, selected_range
    )

    if filtered_data.empty:
        st.info(
            "No readings are available "
            "for this time range."
        )
        return

    first_timestamp = (
        filtered_data["timestamp"].iloc[0]
    )

    last_timestamp = (
        filtered_data["timestamp"].iloc[-1]
    )

    st.caption(
        f"Selected period: {first_timestamp} to "
        f"{last_timestamp} | "
        f"{len(filtered_data)} readings"
    )

    # --------------------------------------------------
    # Energy and power analytics for selected period
    # --------------------------------------------------

    energy_kwh = estimate_energy_kwh(
        filtered_data
    )

    estimated_cost = (
        energy_kwh * ELECTRICITY_TARIFF
    )

    peak_power = float(
        filtered_data["total_power_w"].max()
    )

    average_power = float(
        filtered_data["total_power_w"].mean()
    )

    appliance_analytics = (
        calculate_appliance_analytics(
            filtered_data,
            appliance_columns,
        )
    )

    st.subheader(
        "Consumption for Selected Period"
    )

    energy_col, cost_col, peak_col, average_col = (
        st.columns(4)
    )

    energy_col.metric(
        "Estimated Energy",
        f"{energy_kwh:.6f} kWh",
    )

    cost_col.metric(
        "Estimated Cost",
        f"₹{estimated_cost:.4f}",
    )

    peak_col.metric(
        "Peak Power",
        f"{peak_power:,.2f} W",
    )

    average_col.metric(
        "Average Power",
        f"{average_power:,.2f} W",
    )

    st.caption(
        f"Estimated tariff: ₹{ELECTRICITY_TARIFF:.2f}/kWh. "
        "Energy and cost are calculated from valid "
        "consecutive readings; intervals longer than "
        f"{MAX_ENERGY_GAP_SECONDS} seconds are excluded. "
        "Results represent simulated data."
    )

    st.divider()

    # --------------------------------------------------
    # Household power graph for selected period
    # --------------------------------------------------

    st.subheader(
        "Household Power Over Time"
    )

    graph_data = filtered_data.tail(
        MAX_GRAPH_READINGS
    ).copy()

    chart_data = graph_data.set_index(
        "timestamp"
    )[["total_power_w"]]

    st.line_chart(
        chart_data,
        y="total_power_w",
        x_label="Time",
        y_label="Power (W)",
        height=350,
    )

    st.caption(
        f"Showing {len(graph_data)} of "
        f"{len(filtered_data)} readings "
        "in the selected period."
    )

    st.divider()

    # --------------------------------------------------
    # Current appliance readings
    # Always displays the latest sensor reading.
    # --------------------------------------------------

    st.subheader(
        "Current Appliance Readings"
    )

    appliance_data = pd.DataFrame({
        "Appliance": [
            get_appliance_name(column)
            for column in appliance_columns
        ],
        "Power (W)": [
            float(latest[column])
            for column in appliance_columns
        ],
    })

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        st.bar_chart(
            appliance_data.set_index(
                "Appliance"
            ),
            y="Power (W)",
            y_label="Power (W)",
            height=350,
        )

    with col_table:
        st.dataframe(
            appliance_data,
            hide_index=True,
            use_container_width=True,
        )

    

    st.divider()

    # --------------------------------------------------
    # Appliance energy contribution
    # For the selected historical period.
    # --------------------------------------------------

    st.subheader(
        "Appliance Energy Contribution"
    )

    st.caption(
        "Contribution is calculated from each "
        "appliance's estimated energy over valid "
        "intervals in the selected period, not just "
        "its latest power reading."
    )

    if not appliance_analytics.empty:

        contribution_chart = (
            appliance_analytics.set_index(
                "Appliance"
            )[["Contribution (%)"]]
        )

        st.bar_chart(
            contribution_chart,
            y="Contribution (%)",
            y_label="Contribution (%)",
            height=300,
        )

        display_analytics = (
            appliance_analytics.copy()
        )

        display_analytics["Energy (kWh)"] = (
            display_analytics["Energy (kWh)"].map(
                lambda value: f"{value:.6f}"
            )
        )

        display_analytics["Estimated Cost (₹)"] = (
            display_analytics[
                "Estimated Cost (₹)"
            ].map(
                lambda value: f"₹{value:.4f}"
            )
        )

        display_analytics["Contribution (%)"] = (
            display_analytics[
                "Contribution (%)"
            ].map(
                lambda value: f"{value:.2f}%"
            )
        )

        st.dataframe(
            display_analytics,
            hide_index=True,
            use_container_width=True,
        )

    st.divider()

    # --------------------------------------------------
    # Recent sensor readings
    # --------------------------------------------------

    with st.expander(
        "View recent sensor readings"
    ):
        recent_table = data.tail(10).sort_values(
            "timestamp",
            ascending=False,
        )

        recent_display = recent_table.rename(
        columns={
            column: get_appliance_name(column)
            for column in appliance_columns
        }
    )
        
        st.dataframe(
            recent_display,
            hide_index=True,
            use_container_width=True,
        )

    display_historical_measurements()

    st.caption(
        "Virtual sensor simulation | "
        "CSV storage + MQTT telemetry to ThingsBoard"
    )


# --------------------------------------------------
# Confirmed historical measurements — ADD-ON
# --------------------------------------------------

def display_historical_measurements():
    """Display successfully published measurements and allow CSV download."""

    st.divider()
    st.subheader("Confirmed Historical Measurements")

    st.caption(
        "Measurements whose MQTT publish was acknowledged by the client. "
        "Timestamps represent the original recorded measurements."
    )

    if not HISTORICAL_FILE.exists():
        st.info(
            "No confirmed historical measurements have been recorded yet. "
            "Start the virtual sensor to populate this file."
        )
        return

    try:
        historical_data = pd.read_csv(HISTORICAL_FILE)

    except (
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        st.warning(f"Could not load historical measurements: {exc}")
        return

    required_columns = {"timestamp", "total_power_w"}

    if not required_columns.issubset(historical_data.columns):
        st.warning(
            "The historical CSV is missing required columns: "
            "timestamp and/or total_power_w."
        )
        return

    if historical_data.empty:
        st.info("The historical measurements file is currently empty.")
        return

    # Preserve the original timestamp strings in the displayed/downloaded data.
    st.caption(f"Total confirmed records: {len(historical_data):,}")

    st.dataframe(
        historical_data,
        hide_index=True,
        use_container_width=True,
        height=350,
    )

    csv_data = historical_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Historical Measurements (CSV)",
        data=csv_data,
        file_name="historical_measurements.csv",
        mime="text/csv",
        key="download_historical_measurements",
    )


# --------------------------------------------------
# Automatic refresh
# --------------------------------------------------

try:
    from streamlit import fragment

    live_dashboard = fragment(
        run_every=f"{REFRESH_SECONDS}s"
    )(display_dashboard)

except ImportError:
    # Fallback for older Streamlit versions.
    live_dashboard = display_dashboard

live_dashboard()