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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.appliance_mapping import get_appliance_name
# --------------------------------------------------
# Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Existing dashboard data
CSV_FILE = PROJECT_ROOT / "data" / "replayed_energy.csv"

# Original historical dataset used for XGBoost forecasting
FORECAST_DATASET = PROJECT_ROOT / "house1_15min.csv"

# Existing rolling-scheduler evaluation results
SCHEDULING_RESULTS_DIR = PROJECT_ROOT / "results" / "rolling"
STATE_FILE = PROJECT_ROOT / "data" / "replay_state.json"
SCHEDULING_SUMMARY_FILE = SCHEDULING_RESULTS_DIR / "summary.csv"
SCHEDULING_REPORT_FILE = SCHEDULING_RESULTS_DIR / "report.json"

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



def load_replay_state():
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

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
        data.loc[data["timestamp"] > cutoff]
        .copy()
        .reset_index(drop=True)
    )


# --------------------------------------------------
# Calculate energy from consecutive readings
# --------------------------------------------------

def calculate_energy_kwh(data, power_column):
    """Each recorded value is a completed 15-minute interval average in W."""
    if data is None or data.empty:
        return 0.0
    values = pd.to_numeric(data[power_column], errors="coerce")
    return float(values.where(values >= 0).sum() * 0.25 / 1000)


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

    state = load_replay_state()
    if not state or state.get("status") == "starting":
        st.info("Waiting for the integrated sensor replay to publish a snapshot.")
    else:
        st.caption(f"Integration snapshot follows completed interval: {state['observed_timestamp']}")
        if not state.get("replay_running", False):
            st.info("Replay is stopped. Displaying its final saved snapshot.")
        forecast = state.get("forecast")
        if forecast:
            col1, col2 = st.columns(2)
            col1.metric("Predicted Aggregate Power", f"{forecast['predicted_aggregate_kw'] * 1000:,.2f} W")
            col2.metric("Forecast Timestamp", forecast["timestamp"])
        else:
            st.info(f"Forecast warmup: {state.get('history_intervals', 0)}/97 consecutive completed intervals.")
        st.subheader("Rolling Scheduling During Replay")
        st.caption("Additional simulated appliance jobs; recorded appliance channels are not shifted.")
        st.write(f"Integration status: {state['status']}")
        c1, c2 = st.columns(2)
        c1.metric("Simulated Appliance Power", f"{state['current_simulated_appliance_kw']:.3f} kW")
        c2.metric("Recorded + Simulated Power", f"{state['current_total_kw']:.3f} kW")
        if state.get("schedule"):
            st.dataframe(pd.DataFrame(state["schedule"]), hide_index=True, use_container_width=True)
        else:
            st.info("No jobs started or rejected in the current scheduling horizon yet.")
        if state.get("replay_events"):
            with st.expander("Replay gaps / scheduling interruptions"):
                st.dataframe(pd.DataFrame(state["replay_events"]), hide_index=True)

    display_scheduling_evaluation()
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
        "Energy sums completed 15-minute interval-average readings. "
        "Missing intervals contribute no estimate. Tariff is illustrative; readings are historical."
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

    st.subheader("Current Appliance Readings")
    st.caption("Appliance labels are illustrative assumptions; channel identities have not been verified.")

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
        "Contribution uses recorded 15-minute interval energy. Friendly appliance names "
        "are illustrative, unverified labels, not confirmed REFIT channel identities."
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

    st.caption(
        "Virtual sensor simulation | "
        "CSV storage + MQTT telemetry to ThingsBoard"
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