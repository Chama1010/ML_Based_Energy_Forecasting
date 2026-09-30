import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Historical Energy Forecast Explorer",
    page_icon="⚡",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "final_random_forest.pkl"
FEATURES_PATH = PROJECT_DIR / "models" / "forecast_features.json"
DATA_PATH = PROJECT_DIR / "data" / "processed" / "engineered_energy_data.csv"
FORECAST_HORIZON = 24


def file_signature(path):
    file_path = Path(path)
    stat = file_path.stat()
    return str(file_path), stat.st_mtime_ns, stat.st_size


def is_git_lfs_pointer(path):
    with Path(path).open("rb") as artifact:
        return artifact.read(128).startswith(
            b"version https://git-lfs.github.com/spec/v1"
        )


@st.cache_resource(show_spinner="Loading the saved forecasting model...")
def load_model(path, modified_ns, size):
    del modified_ns, size
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if is_git_lfs_pointer(model_path):
        raise ValueError(
            "The model file is a Git LFS pointer. Retrieve the model with `git lfs pull`."
        )
    return joblib.load(model_path)


@st.cache_data(show_spinner="Loading the historical dataset...")
def load_historical_data(path, modified_ns, size):
    del modified_ns, size
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Historical data file not found: {data_path}")

    history = pd.read_csv(data_path, index_col=0, parse_dates=True)
    if not isinstance(history.index, pd.DatetimeIndex):
        raise ValueError("The historical dataset must have a datetime index.")
    if history.index.hasnans:
        raise ValueError("The historical dataset contains invalid timestamps.")
    if "Appliances" not in history.columns:
        raise ValueError("The historical dataset is missing the Appliances column.")

    history = history.sort_index()
    if history.index.has_duplicates:
        raise ValueError("The historical dataset contains duplicate timestamps.")
    history["Appliances"] = pd.to_numeric(history["Appliances"], errors="coerce")
    history = history.loc[history["Appliances"].notna()]
    if len(history) < 168:
        raise ValueError("At least 168 valid hourly observations are required.")
    return history


@st.cache_data(show_spinner="Loading forecast feature configuration...")
def load_feature_config(path, modified_ns, size):
    del modified_ns, size
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Forecast feature configuration not found: {config_path}")
    with config_path.open(encoding="utf-8") as config_file:
        features = json.load(config_file)
    if not isinstance(features, list) or not features or not all(
        isinstance(feature, str) for feature in features
    ):
        raise ValueError("The forecast feature configuration must be a non-empty list of names.")
    if len(features) != len(set(features)):
        raise ValueError("The forecast feature configuration contains duplicate names.")
    return features


def create_forecast_features(timestamp, history, features):
    values = history["Appliances"]
    features_row = {
        "hour": timestamp.hour,
        "day_of_week": timestamp.dayofweek,
        "is_weekend": int(timestamp.dayofweek >= 5),
        "lag_1": values.iloc[-1],
        "lag_2": values.iloc[-2],
        "lag_3": values.iloc[-3],
        "lag_24": values.iloc[-24],
        "lag_48": values.iloc[-48],
        "lag_168": values.iloc[-168],
        "rolling_mean_3": values.iloc[-3:].mean(),
        "rolling_mean_6": values.iloc[-6:].mean(),
    }
    feature_row = pd.DataFrame([features_row], index=[timestamp])[features]
    return feature_row.apply(pd.to_numeric, errors="raise")


@st.cache_data(show_spinner="Generating the historical 24-hour forecast...")
def generate_forecast(
    model_path,
    model_modified_ns,
    model_size,
    features_path,
    features_modified_ns,
    features_size,
    data_path,
    data_modified_ns,
    data_size,
):
    model = load_model(model_path, model_modified_ns, model_size)
    features = load_feature_config(features_path, features_modified_ns, features_size)
    history = load_historical_data(data_path, data_modified_ns, data_size)

    missing_features = [feature for feature in features if feature not in history.columns]
    if missing_features:
        raise ValueError(f"Historical data is missing model features: {missing_features}")
    if getattr(model, "n_features_in_", len(features)) != len(features):
        raise ValueError("The saved model and feature configuration have different feature counts.")
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None and list(model_features) != features:
        raise ValueError("The saved model feature order does not match its configuration.")

    last_timestamp = history.index[-1]
    future_timestamps = pd.date_range(
        start=last_timestamp + pd.Timedelta(hours=1),
        periods=FORECAST_HORIZON,
        freq="h",
    )
    recursive_history = history[["Appliances"]].copy()
    predictions = []

    for timestamp in future_timestamps:
        forecast_features = create_forecast_features(
            timestamp,
            recursive_history,
            features,
        )
        prediction = float(model.predict(forecast_features)[0])
        predictions.append(prediction)
        recursive_history.loc[timestamp, "Appliances"] = prediction

    forecast = pd.DataFrame(
        {"Predicted_Appliances": predictions},
        index=future_timestamps,
    )
    forecast.index.name = "date"
    return forecast


def classify_demand(forecast, low_threshold, high_threshold):
    classified = forecast.copy()
    classified["Demand_Level"] = "Normal"
    classified.loc[
        classified["Predicted_Appliances"] <= low_threshold,
        "Demand_Level",
    ] = "Low"
    classified.loc[
        classified["Predicted_Appliances"] >= high_threshold,
        "Demand_Level",
    ] = "High"
    return classified


def display_scale(units):
    return 1000.0 if units == "kWh" else 1.0


def format_energy(value, units):
    return f"{value / display_scale(units):,.2f} {units}"


def serialize_forecast_table(table):
    return table.reset_index().to_csv(
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )


def reset_controls():
    st.session_state["forecast_horizon"] = 24
    st.session_state["history_window"] = 48
    st.session_state["display_units"] = "Wh"
    st.session_state["low_percentile"] = 25
    st.session_state["high_percentile"] = 75
    st.session_state["demand_filter"] = ["Low", "Normal", "High"]
    st.session_state["forecast_hour"] = forecast_24h.index[0]
    st.session_state["historical_date_range"] = (
        history_df.index.min().date(),
        history_df.index.max().date(),
    )


def keep_high_percentile_above_low():
    low_value = st.session_state["low_percentile"]
    high_value = st.session_state.get("high_percentile", 75)
    st.session_state["high_percentile"] = min(max(high_value, low_value + 1), 99)


st.title("Historical Energy Forecast Explorer")
st.caption("Appliance-energy demonstration using one building's 2016 UCI dataset.")

try:
    model_signature = file_signature(MODEL_PATH)
    features_signature = file_signature(FEATURES_PATH)
    data_signature = file_signature(DATA_PATH)
    history_df = load_historical_data(*data_signature)
    feature_names = load_feature_config(*features_signature)
    forecast_24h = generate_forecast(
        *model_signature,
        *features_signature,
        *data_signature,
    )
except FileNotFoundError as error:
    st.error(f"Required project file is unavailable: {error}")
    st.stop()
except (OSError, ValueError, json.JSONDecodeError, EOFError) as error:
    st.error(f"Could not prepare the saved historical forecast: {error}")
    st.stop()
except Exception as error:
    st.error(f"Could not load the saved forecasting artifacts: {error}")
    st.stop()

latest_timestamp = history_df.index[-1]
forecast_start = forecast_24h.index[0]

if "forecast_horizon" not in st.session_state:
    reset_controls()
if "historical_date_range" not in st.session_state:
    st.session_state["historical_date_range"] = (
        history_df.index.min().date(),
        history_df.index.max().date(),
    )

with st.sidebar:
    st.header("Explorer Controls")
    st.button("Reset controls", on_click=reset_controls, use_container_width=True)
    horizon = st.slider(
        "Forecast horizon (hours)",
        min_value=1,
        max_value=24,
        key="forecast_horizon",
    )
    history_window = st.select_slider(
        "Historical chart context (hours)",
        options=[24, 48, 72, 168],
        key="history_window",
    )
    units = st.radio(
        "Display units",
        options=["Wh", "kWh"],
        horizontal=True,
        key="display_units",
    )

    with st.expander("Demand thresholds", expanded=False):
        st.caption(
            "Percentiles use the full recorded history. These settings change "
            "demand categories only, not numerical predictions."
        )
        low_percentile = st.select_slider(
            "Low-demand percentile",
            options=list(range(1, 99)),
            key="low_percentile",
            on_change=keep_high_percentile_above_low,
        )
        high_options = list(range(low_percentile + 1, 100))
        high_percentile = st.select_slider(
            "High-demand percentile",
            options=high_options,
            key="high_percentile",
        )

display_factor = display_scale(units)
selected_forecast = forecast_24h.iloc[:horizon].copy()
historical_consumption = history_df["Appliances"].dropna()
low_threshold = historical_consumption.quantile(low_percentile / 100)
high_threshold = historical_consumption.quantile(high_percentile / 100)
selected_forecast = classify_demand(
    selected_forecast,
    low_threshold,
    high_threshold,
)
selected_forecast["Displayed_Energy"] = (
    selected_forecast["Predicted_Appliances"] / display_factor
)

forecast_end = selected_forecast.index[-1]
st.info(
    f"Latest stored observation: **{latest_timestamp:%Y-%m-%d %H:%M}**  "
    f"| Forecast window: **{forecast_start:%Y-%m-%d %H:%M}** to "
    f"**{forecast_end:%Y-%m-%d %H:%M}** ({horizon} of 24 hours). "
    "Predictions continue directly after the historical endpoint; they are not live measurements."
)

forecast_tab, history_tab, model_tab = st.tabs(
    ["Forecast", "Historical Data", "Model Information"]
)

with forecast_tab:
    st.subheader("Forecast summary")
    total_energy = selected_forecast["Predicted_Appliances"].sum()
    average_hourly = selected_forecast["Predicted_Appliances"].mean()
    peak_timestamp = selected_forecast["Predicted_Appliances"].idxmax()
    peak_energy = selected_forecast.loc[peak_timestamp, "Predicted_Appliances"]
    high_demand_count = int((selected_forecast["Demand_Level"] == "High").sum())

    metric_columns = st.columns(4)
    metric_columns[0].metric("Total predicted energy", format_energy(total_energy, units))
    metric_columns[1].metric("Average hourly consumption", format_energy(average_hourly, units))
    metric_columns[2].metric(
        "Highest predicted hour",
        format_energy(peak_energy, units),
        delta=peak_timestamp.strftime("%Y-%m-%d %H:%M"),
        delta_color="off",
    )
    metric_columns[3].metric("High-demand hours", f"{high_demand_count} / {horizon}")

    st.subheader("Observed context and forecast")
    chart_history = history_df[["Appliances"]].tail(history_window).copy()
    chart_history["Timestamp"] = chart_history.index
    chart_history["Consumption"] = chart_history["Appliances"] / display_factor
    chart_history["Series"] = "Observed history"
    chart_forecast = selected_forecast[["Predicted_Appliances"]].copy()
    chart_forecast["Timestamp"] = chart_forecast.index
    chart_forecast["Consumption"] = chart_forecast["Predicted_Appliances"] / display_factor
    chart_forecast["Series"] = "Forecast"
    chart_data = pd.concat(
        [
            chart_history[["Timestamp", "Consumption", "Series"]],
            chart_forecast[["Timestamp", "Consumption", "Series"]],
        ],
        ignore_index=True,
    )
    line_chart = (
        alt.Chart(chart_data)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "Timestamp:T",
                title="Date and time",
                axis=alt.Axis(format="%b %d %H:%M", labelAngle=-35),
            ),
            y=alt.Y("Consumption:Q", title=f"Appliance energy ({units})"),
            color=alt.Color(
                "Series:N",
                title="",
                scale=alt.Scale(
                    domain=["Observed history", "Forecast"],
                    range=["#277da1", "#e76f51"],
                ),
            ),
            tooltip=[
                alt.Tooltip("Timestamp:T", title="Timestamp", format="%Y-%m-%d %H:%M"),
                alt.Tooltip("Series:N", title="Series"),
                alt.Tooltip("Consumption:Q", title=f"Energy ({units})", format=",.2f"),
            ],
        )
    )
    forecast_boundary = (
        alt.Chart(pd.DataFrame({"Timestamp": [forecast_start]}))
        .mark_rule(color="#333333", strokeDash=[5, 4])
        .encode(
            x="Timestamp:T",
            tooltip=[
                alt.Tooltip(
                    "Timestamp:T",
                    title="Forecast begins",
                    format="%Y-%m-%d %H:%M",
                )
            ],
        )
    )
    st.altair_chart(line_chart + forecast_boundary, width="stretch", theme="streamlit")

    st.subheader("Inspect a forecast hour")
    forecast_timestamps = list(selected_forecast.index)
    if st.session_state.get("forecast_hour") not in forecast_timestamps:
        st.session_state["forecast_hour"] = forecast_timestamps[0]
    selected_hour = st.selectbox(
        "Forecast timestamp",
        options=forecast_timestamps,
        format_func=lambda timestamp: timestamp.strftime("%Y-%m-%d %H:%M"),
        key="forecast_hour",
    )
    hour_result = selected_forecast.loc[selected_hour]
    selected_columns = st.columns(2)
    selected_columns[0].metric(
        "Predicted consumption",
        format_energy(hour_result["Predicted_Appliances"], units),
    )
    selected_columns[1].metric("Demand category", hour_result["Demand_Level"])

    st.subheader("Detailed forecast")
    st.caption("Demand filter applies only to this table and its CSV download.")
    st.caption(
        f"Low: at or below {format_energy(low_threshold, units)} "
        f"({low_percentile}th percentile) | High: at or above "
        f"{format_energy(high_threshold, units)} ({high_percentile}th percentile)."
    )
    demand_filter = st.multiselect(
        "Show demand categories",
        options=["Low", "Normal", "High"],
        key="demand_filter",
    )
    table = selected_forecast.loc[
        selected_forecast["Demand_Level"].isin(demand_filter),
        ["Displayed_Energy", "Demand_Level"],
    ].rename(
        columns={
            "Displayed_Energy": f"Predicted energy ({units})",
            "Demand_Level": "Demand category",
        }
    )
    table.index.name = "Timestamp"
    if table.empty:
        st.info("No forecast hours match the selected demand categories.")
    else:
        st.dataframe(table.round(3), width="stretch")
        csv_data = serialize_forecast_table(table)
        st.download_button(
            "Download filtered forecast CSV",
            data=csv_data,
            file_name="historical_energy_forecast.csv",
            mime="text/csv",
        )

with history_tab:
    st.subheader("Recorded appliance consumption")
    st.caption(
        "Browse recorded observations only. This does not change the forecast origin or model inputs."
    )
    first_date = history_df.index.min().date()
    last_date = latest_timestamp.date()
    selected_dates = st.date_input(
        "Recorded date range",
        min_value=first_date,
        max_value=last_date,
        key="historical_date_range",
    )
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        range_start, range_end = selected_dates
        start_timestamp = pd.Timestamp(range_start)
        end_timestamp = (
            pd.Timestamp(range_end)
            + pd.Timedelta(days=1)
            - pd.Timedelta(nanoseconds=1)
        )
        displayed_history = history_df.loc[
            start_timestamp:end_timestamp,
            ["Appliances"],
        ].copy()
    else:
        displayed_history = history_df.iloc[0:0][["Appliances"]].copy()

    if displayed_history.empty:
        st.info("Select a date range containing recorded observations.")
    else:
        displayed_history["Timestamp"] = displayed_history.index
        displayed_history["Consumption"] = (
            displayed_history["Appliances"] / display_factor
        )
        historical_chart = (
            alt.Chart(displayed_history.reset_index(drop=True))
            .mark_line(color="#277da1")
            .encode(
                x=alt.X(
                    "Timestamp:T",
                    title="Recorded date and time",
                    axis=alt.Axis(format="%b %d %H:%M", labelAngle=-35),
                ),
                y=alt.Y("Consumption:Q", title=f"Recorded consumption ({units})"),
                tooltip=[
                    alt.Tooltip("Timestamp:T", title="Timestamp", format="%Y-%m-%d %H:%M"),
                    alt.Tooltip("Consumption:Q", title=f"Consumption ({units})", format=",.2f"),
                ],
            )
        )
        st.altair_chart(historical_chart, width="stretch", theme="streamlit")

        stats = displayed_history["Appliances"].describe()
        stats_frame = stats.to_frame(name=units)
        stats_frame[units] = stats_frame[units] / display_factor
        st.dataframe(stats_frame.round(3), width="stretch")

        historical_table = displayed_history[["Appliances"]].rename(
            columns={"Appliances": f"Recorded consumption ({units})"}
        )
        historical_table[f"Recorded consumption ({units})"] /= display_factor
        historical_table.index.name = "Timestamp"
        st.dataframe(historical_table.round(3), width="stretch")

with model_tab:
    st.subheader("Saved model and data")
    model_columns = st.columns(2)
    model_columns[0].metric("Model", "Random Forest Regressor")
    model_columns[1].metric("Training rows in saved dataset", f"{len(history_df):,}")
    st.write(
        f"**Dataset coverage:** {history_df.index.min():%Y-%m-%d %H:%M} to "
        f"{latest_timestamp:%Y-%m-%d %H:%M}"
    )
    st.write("**Forecast features, in model order:**")
    st.code("\n".join(feature_names), language="text")
    st.write(
        "**Limitations:** This is a retrospective forecast demonstration from one building's "
        "2016 measurements. It is not live, has not been validated for present-day households, "
        "and recursive errors can accumulate across forecast hours."
    )