# Appliance Energy Consumption Forecasting

A machine learning based time-series forecasting system for predicting short-term household appliance energy consumption.

This project transforms historical appliance energy measurements into an hourly forecasting problem, engineers temporal features, evaluates baseline and machine learning models, and deploys the selected model through an interactive Streamlit dashboard.

The final system generates a **24-hour recursive forecast** of appliance energy consumption.

---

## 1. Project Objectives

The main objectives of this project are to:

* Prepare and transform appliance energy consumption data for time-series forecasting.
* Engineer temporal and lag-based forecasting features.
* Establish simple forecasting baselines.
* Train and evaluate machine learning regression models.
* Compare Random Forest and XGBoost with the baseline approaches.
* Evaluate both one-step and 24-hour recursive forecasting performance.
* Select the best-performing model based on the final test results.
* Build a 24-hour recursive forecasting pipeline.
* Deploy the forecasting system through a Streamlit dashboard.

---

## 2. Dataset

This project uses the **Appliances Energy Prediction** dataset from the UCI Machine Learning Repository.

The dataset was originally collected and studied by:

**Luis M. Candanedo, Véronique Feldheim, and Dominique Deramaix**

The original dataset contains measurements collected from a low-energy residential building over approximately 4.5 months. The measurements were originally recorded at **10-minute intervals**.

### Dataset Source

UCI Machine Learning Repository:

https://doi.org/10.24432/C5VC8G

### Original Research Paper

Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017).

*Data driven prediction models of energy use of appliances in a low-energy house.*

Energy and Buildings, 140, 81–97.

DOI:

https://doi.org/10.1016/j.enbuild.2017.01.083

### Dataset Attribution

The dataset is **not original data created by this project**. It was obtained from the publicly available dataset provided through the UCI Machine Learning Repository.

The UCI dataset is distributed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

Appropriate attribution to the original dataset creators and source should therefore be retained when the dataset is redistributed or reused.

---

## 3. Project Approach

The project treats appliance energy prediction as a **time-series forecasting problem** rather than a conventional randomly shuffled regression problem.

The overall workflow is:

```text
Original Dataset
       |
       v
Data Preparation
       |
       v
Hourly Forecasting Dataset
       |
       v
Temporal Feature Engineering
       |
       v
Chronological Train / Validation / Test Split
       |
       +----------------------+
       |                      |
       v                      v
Baseline Models        Machine Learning Models
       |                      |
       |               +------+------+
       |               |             |
       |               v             v
       |         Random Forest    XGBoost
       |               |             |
       +---------------+-------------+
                       |
                       v
              Forecast Evaluation
                       |
                       v
                Model Comparison
                       |
                       v
               Random Forest
                       |
                       v
             24-Hour Forecast
                       |
                       v
             Streamlit Dashboard
```

---

## 4. Time-Series Transformation

The original dataset contains observations at 10-minute intervals.

For this project, the data was transformed into an **hourly forecasting framework** so that the model could predict appliance energy consumption on an hourly basis.

The forecasting workflow therefore operates on an hourly time index.

The exact preprocessing and transformation steps are implemented in the project notebooks.

---

## 5. Target Variable

The target variable is:

```text
Appliances
```

The original dataset defines `Appliances` as appliance energy consumption measured in **Wh**.

In this project, the target is used to generate hourly appliance energy consumption forecasts.

---

## 6. Feature Engineering

The final forecasting feature set consists of:

```text
hour
day_of_week
is_weekend
lag_1
lag_2
lag_3
lag_24
lag_48
lag_168
rolling_mean_3
rolling_mean_6
```

### Calendar Features

#### `hour`

The hour of the day.

Used to capture daily consumption patterns.

#### `day_of_week`

The numerical day of the week.

Used to distinguish different weekly consumption patterns.

#### `is_weekend`

Binary indicator representing whether the observation occurs on Saturday or Sunday.

---

### Lag Features

#### `lag_1`

Appliance consumption from the previous hour.

#### `lag_2`

Appliance consumption from two hours earlier.

#### `lag_3`

Appliance consumption from three hours earlier.

#### `lag_24`

Appliance consumption from approximately the same hour on the previous day.

#### `lag_48`

Appliance consumption from approximately the same hour two days earlier.

#### `lag_168`

Appliance consumption from approximately the same hour one week earlier.

These lag variables allow the models to use recent, daily, and weekly historical consumption patterns.

---

### Rolling Features

#### `rolling_mean_3`

Mean appliance consumption over the most recent three observations.

#### `rolling_mean_6`

Mean appliance consumption over the most recent six observations.

These features provide information about recent consumption levels while reducing the effect of individual fluctuations.

---

## 7. Baseline Models

Simple forecasting approaches were implemented before evaluating machine learning models.

This provides a reference point for determining whether the machine learning models provide meaningful improvement.

### Previous Hour

Uses the previous hour's appliance consumption as the prediction.

```text
Prediction(t) = Consumption(t - 1)
```

### Previous Day

Uses the corresponding previous-day consumption as the prediction.

```text
Prediction(t) = Consumption(t - 24h)
```

### Hourly Profile

Uses the typical appliance consumption associated with the corresponding hour of the day.

This baseline captures recurring intraday consumption patterns.

---

## 8. Random Forest

A `RandomForestRegressor` was trained using the engineered temporal features.

Random Forest was selected as the final deployment model because it achieved the strongest performance among the evaluated models on the final 24-hour recursive test.

The final trained model is stored as:

```text
models/final_random_forest.pkl
```

The forecasting feature configuration is stored as:

```text
models/forecast_features.json
```

---

## 9. XGBoost

XGBoost was evaluated as an alternative tree-based machine learning model.

The model was evaluated using both:

* One-step validation
* 24-hour recursive validation
* Final 24-hour recursive testing

An important finding was that XGBoost achieved stronger one-step validation performance than Random Forest, but this advantage did not carry over to recursive 24-hour forecasting.

Therefore, XGBoost was retained as a comparative model, while Random Forest was selected for deployment.

---

## 10. Evaluation Methodology

Two forecasting evaluation approaches were used.

### One-Step Evaluation

The model predicts the next observation using the available historical information.

This evaluates direct prediction performance.

### 24-Hour Recursive Evaluation

The model predicts the next 24 hours sequentially.

The prediction for one hour becomes part of the historical input used to generate the prediction for the following hour.

The process is:

```text
Historical Data
      |
      v
Predict t+1
      |
      v
Add prediction to history
      |
      v
Generate features for t+2
      |
      v
Predict t+2
      |
      v
Continue recursively
      |
      v
Predict t+24
```

This evaluation is more representative of the final application because the deployed system also produces a 24-hour recursive forecast.

---

## 11. Evaluation Metrics

The main evaluation metrics are:

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted consumption.

Lower values indicate better performance.

### Root Mean Squared Error (RMSE)

Measures prediction error while giving greater weight to larger errors.

Lower values indicate better performance.

### R² Score

Measures the proportion of variance in the target explained by the model.

Higher values indicate better explanatory performance.

---

## 12. Final Test Results

The final 24-hour recursive test results were:

| Model             |        MAE |       RMSE |
| ----------------- | ---------: | ---------: |
| Previous Hour     |     251.34 |     478.52 |
| Previous Day      |     268.20 |     483.99 |
| Hourly Profile    |     245.58 |     386.73 |
| **Random Forest** | **235.37** | **373.10** |
| XGBoost           |     250.31 |     404.79 |

### Final Random Forest

```text
MAE  = 235.37
RMSE = 373.10
R²   = 0.21785
```

### Final XGBoost

```text
MAE  = 250.31
RMSE = 404.79
R²   = 0.07935
```

Based on the final test results, **Random Forest was the best-performing model among the approaches evaluated in this project**.

The results should not be interpreted as claiming that Random Forest is universally the best model for appliance energy forecasting.

---

## 13. Important Model Finding

The XGBoost experiments demonstrated an important difference between one-step and multi-step forecasting.

XGBoost achieved:

```text
One-step Validation MAE  = 191.92
One-step Validation RMSE = 326.11
```

However, its 24-hour recursive validation performance was:

```text
Recursive Validation MAE  = 253.57
Recursive Validation RMSE = 378.88
```

After tuning, the recursive validation performance was:

```text
Tuned Recursive MAE  = 272.91
Tuned Recursive RMSE = 387.70
```

This shows that strong one-step prediction performance does not necessarily translate into strong multi-step recursive forecasting performance.

For this reason, the final model selection was based primarily on the forecasting scenario required by the application.

---

## 14. Final Forecasting Pipeline

The final system uses the Random Forest model to generate a 24-hour recursive forecast.

The process begins from the latest available historical observation.

For each future hour:

1. Generate the calendar features.
2. Retrieve the required lag values.
3. Calculate the rolling features.
4. Arrange the features according to the saved feature configuration.
5. Generate the Random Forest prediction.
6. Add the prediction to the forecasting history.
7. Use the updated history to generate features for the next hour.
8. Repeat until 24 future predictions are produced.

The final forecast contains:

```text
date
Predicted_Appliances
```

---

## 15. Streamlit Dashboard

The project includes an interactive Streamlit dashboard for displaying the forecasting results.

The dashboard provides:

* Model information
* Forecast horizon
* Forecast start and end times
* Number of forecast hours
* Average predicted consumption
* Peak predicted consumption
* Minimum predicted consumption
* Peak forecast timestamp
* Minimum forecast timestamp
* 24-hour forecast line chart
* Historical and forecast comparison
* Detailed forecast table

The dashboard loads the saved Random Forest model and forecasting configuration rather than retraining the model during application execution.

---

## 16. Technologies

* Python
* pandas
* NumPy
* scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* joblib
* Streamlit
* Jupyter Notebook
* Visual Studio Code

---

## 17. Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

---

## 18. Running the Dashboard

From the project root:

```bash
streamlit run app/app.py
```

Streamlit will provide a local URL that can be opened in a web browser.

---

## 19. Reproducibility

The project follows a chronological time-series workflow.

To reproduce the forecasting process:

* Preserve the chronological ordering of observations.
* Do not randomly shuffle the time-series data.
* Use the defined forecasting features.
* Preserve the chronological train, validation, and test periods.
* Use the saved model configuration.
* Use the saved final model for deployment.
* Maintain the same preprocessing and feature-engineering procedures.

---

## 20. Limitations

### Limited Observation Period

The original dataset covers approximately 4.5 months. Therefore, the dataset does not provide a full year of observations for learning long-term annual seasonality.

### Building-Specific Data

The original measurements were collected from a particular low-energy residential building. The resulting model should therefore not automatically be assumed to generalize to other buildings or households.

### Recursive Forecasting Error

The 24-hour forecasting process is recursive. Errors in earlier predictions can influence later predictions.

### Moderate R²

The final Random Forest achieved an R² of approximately 0.218 on the final recursive test. This indicates that substantial variation in appliance consumption remains unexplained.

### Limited Forecasting Inputs

The final forecasting model primarily uses historical appliance consumption and engineered temporal features. Additional real-time environmental or contextual variables could potentially improve forecasting performance.

---

## 21. Future Improvements

Possible future extensions include:

* Additional lag features
* Additional rolling statistics
* Exponentially weighted moving averages
* Holiday and special-day features
* Weather-aware forecasting
* More extensive hyperparameter optimization
* Direct multi-horizon forecasting
* Ensemble forecasting
* Additional gradient boosting models
* Deep learning time-series models
* Prediction intervals and uncertainty estimation
* Automated model retraining
* Forecast error monitoring
* Cloud deployment

These are potential extensions and are not part of the current final implementation.

---

## 22. Dataset and Research Attribution

The underlying dataset used in this project was created by the original researchers and is not an original dataset produced by this project.

Please cite the dataset and original research when using or redistributing the dataset.

### Dataset

Candanedo, L. (2017). *Appliances Energy Prediction*. UCI Machine Learning Repository.

https://doi.org/10.24432/C5VC8G

### Original Research

Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. *Energy and Buildings, 140*, 81–97.

https://doi.org/10.1016/j.enbuild.2017.01.083

### Original Data Repository

Luis M. Candanedo's original research data repository:

https://github.com/LuisM78/Appliances-energy-prediction-data

---

## 23. Comparability Notice

The results presented in this project are **not direct reproductions of the results reported in the original research paper**.

The original research and this project differ in areas including:

* Problem formulation
* Data transformation
* Feature engineering
* Model configuration
* Train/validation/test methodology
* Forecasting horizon
* Evaluation methodology

The original research is cited as the source of the dataset and as the academic work from which the dataset originated. The modelling pipeline and experimental results presented in this repository are part of this project.

---

## 24. Acknowledgement

This project acknowledges the work of:

**Luis M. Candanedo**
**Véronique Feldheim**
**Dominique Deramaix**

for collecting, publishing, and making the underlying appliance energy dataset available for research and educational use.

The machine learning forecasting pipeline, feature engineering, model comparison, recursive forecasting implementation, and Streamlit dashboard presented in this repository were developed as part of this project.
