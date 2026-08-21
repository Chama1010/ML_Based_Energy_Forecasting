import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path


# PAGE CONFIGURATION
st.set_page_config(
    page_title="Energy Forecasting Dashboard",
    page_icon="⚡",
    layout="wide"
)


# PROJECT PATHS

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "final_random_forest.pkl"
)

FEATURES_PATH = (
    PROJECT_DIR
    / "models"
    / "forecast_features.json"
)

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "engineered_energy_data.csv"
)


# LOAD MODEL

model = joblib.load(MODEL_PATH)


# LOAD FEATURE CONFIGURATION

with open(FEATURES_PATH, "r") as f:
    features = json.load(f)


# LOAD HISTORICAL DATA

forecast_df = pd.read_csv(
    DATA_PATH,
    index_col=0,
    parse_dates=True
)

forecast_df = forecast_df.sort_index()


# FORECAST FEATURE CREATION

def create_forecast_features(timestamp, history):

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
        "rolling_mean_6": values.iloc[-6:].mean()
    }

    return pd.DataFrame(
        [features_row],
        index=[timestamp]
    )[features]


# GENERATE 24-HOUR RECURSIVE FORECAST

last_timestamp = forecast_df.index.max()

future_timestamps = pd.date_range(
    start=last_timestamp + pd.Timedelta(hours=1),
    periods=24,
    freq="h"
)

history = forecast_df[["Appliances"]].copy()

recursive_predictions = []

for timestamp in future_timestamps:

    forecast_features = create_forecast_features(
        timestamp,
        history
    )

    prediction = model.predict(
        forecast_features
    )[0]

    recursive_predictions.append(prediction)

    history.loc[
        timestamp,
        "Appliances"
    ] = prediction


# CREATE FORECAST DATAFRAME

forecast_results = pd.DataFrame(
    {
        "date": future_timestamps,
        "Predicted_Appliances": recursive_predictions
    }
)

forecast_results = forecast_results.set_index("date")


# FORECAST SUMMARY CALCULATIONS

average_consumption = (
    forecast_results["Predicted_Appliances"].mean()
)

peak_timestamp = (
    forecast_results["Predicted_Appliances"].idxmax()
)

peak_consumption = (
    forecast_results["Predicted_Appliances"].max()
)

minimum_timestamp = (
    forecast_results["Predicted_Appliances"].idxmin()
)

minimum_consumption = (
    forecast_results["Predicted_Appliances"].min()
)


# DEMAND CLASSIFICATION

historical_consumption = (
    forecast_df["Appliances"]
    .dropna()
)

low_threshold = (
    historical_consumption.quantile(0.25)
)

high_threshold = (
    historical_consumption.quantile(0.75)
)

forecast_results["Demand_Level"] = "Normal"

forecast_results.loc[
    forecast_results["Predicted_Appliances"] <= low_threshold,
    "Demand_Level"
] = "Low"

forecast_results.loc[
    forecast_results["Predicted_Appliances"] >= high_threshold,
    "Demand_Level"
] = "High"


# HIGH-DEMAND PERIODS
high_demand_periods = forecast_results[
    forecast_results["Demand_Level"] == "High"
].copy()


# DASHBOARD

st.title("⚡ Energy Forecasting Dashboard")

st.write(
    "24-hour recursive appliance energy consumption "
    "forecasting system."
)


# MODEL INFORMATION

st.subheader("Model Information")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Model",
        "Random Forest"
    )

with col2:
    st.metric(
        "Forecast Horizon",
        "24 Hours"
    )


# FORECAST INFORMATION

st.subheader("24-Hour Forecast")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Forecast Start",
        forecast_results.index[0].strftime(
            "%Y-%m-%d %H:%M"
        )
    )

with col2:
    st.metric(
        "Forecast End",
        forecast_results.index[-1].strftime(
            "%Y-%m-%d %H:%M"
        )
    )

with col3:
    st.metric(
        "Forecast Hours",
        len(forecast_results)
    )


# FORECAST SUMMARY

st.subheader("Forecast Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Average Predicted Consumption",
        f"{average_consumption:.1f} Wh"
    )

with col2:
    st.metric(
        "Peak Predicted Consumption",
        f"{peak_consumption:.1f} Wh"
    )

with col3:
    st.metric(
        "Minimum Predicted Consumption",
        f"{minimum_consumption:.1f} Wh"
    )

st.write(
    f"Peak expected at: "
    f"{peak_timestamp.strftime('%Y-%m-%d %H:%M')}"
)

st.write(
    f"Minimum expected at: "
    f"{minimum_timestamp.strftime('%Y-%m-%d %H:%M')}"
)


# 24-HOUR FORECAST CHART

st.subheader("24-Hour Energy Forecast")

st.line_chart(
    forecast_results[
        ["Predicted_Appliances"]
    ],
    y="Predicted_Appliances",
    x_label="Time",
    y_label="Predicted Consumption (Wh)"
)


# HISTORICAL + FORECAST COMPARISON

st.subheader("Historical and 24-Hour Forecast")

historical_for_chart = forecast_df[
    ["Appliances"]
].tail(48).copy()

historical_for_chart = historical_for_chart.rename(
    columns={
        "Appliances": "Actual Consumption"
    }
)

forecast_for_chart = forecast_results[
    ["Predicted_Appliances"]
].rename(
    columns={
        "Predicted_Appliances":
        "Forecast Consumption"
    }
)

comparison_chart = pd.concat(
    [
        historical_for_chart,
        forecast_for_chart
    ],
    axis=1
)

st.line_chart(
    comparison_chart,
    x_label="Time",
    y_label="Energy Consumption (Wh)"
)


# DEMAND CLASSIFICATION

st.subheader("Demand Classification")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Low-Demand Threshold",
        f"{low_threshold:.1f} Wh"
    )

with col2:
    st.metric(
        "High-Demand Threshold",
        f"{high_threshold:.1f} Wh"
    )


# HIGH-DEMAND PERIODS

st.subheader("High-Demand Periods")

if len(high_demand_periods) > 0:

    high_demand_display = high_demand_periods[
        ["Predicted_Appliances"]
    ].copy()

    high_demand_display = high_demand_display.rename(
        columns={
            "Predicted_Appliances":
            "Predicted Consumption (Wh)"
        }
    )

    st.dataframe(
        high_demand_display.round(2),
        use_container_width=True
    )

else:

    st.info(
        "No high-demand periods were identified "
        "in the next 24 hours."
    )


# ENERGY MANAGEMENT RECOMMENDATION

st.subheader("Energy Management Recommendation")

if len(high_demand_periods) > 0:

    peak_hour = peak_timestamp.strftime(
        "%Y-%m-%d %H:%M"
    )

    st.warning(
        f"Higher-than-usual consumption is predicted "
        f"during some periods of the next 24 hours. "
        f"The highest predicted consumption occurs "
        f"around {peak_hour}, at approximately "
        f"{peak_consumption:.1f} Wh."
    )

else:

    st.success(
        "No unusually high-demand period was detected "
        "in the next 24 hours based on the historical "
        "consumption distribution."
    )


# DETAILED FORECAST TABLE

st.subheader("Detailed 24-Hour Forecast")

forecast_table = forecast_results.copy()

forecast_table = forecast_table.rename(
    columns={
        "Predicted_Appliances":
        "Predicted Consumption (Wh)",
        "Demand_Level":
        "Demand Level"
    }
)

st.dataframe(
    forecast_table.round(2),
    use_container_width=True
)