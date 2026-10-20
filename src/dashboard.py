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

# --------------------------------------------------
# Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_FILE = PROJECT_ROOT / "data" / "replayed_energy.csv"

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
    """Read the CSV and validate its contents."""

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
            "Appliance": appliance.replace(
                "_", " "
            ).title(),
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

def display_dashboard():

    st.title("⚡ Smart Home Energy Management")

    st.caption(
        "Live virtual sensor monitoring | "
        "Streamlit + Python + ThingsBoard MQTT"
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
        "Current Power",
        f"{latest_power:,.2f} W",
    )

    col2.metric(
        "Active Appliances",
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
            column.replace("_", " ").title()
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

        st.dataframe(
            recent_table,
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