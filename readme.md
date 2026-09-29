# Appliance Energy Consumption Forecasting

A historical machine-learning prototype for 24-hour household appliance energy forecasting using measurements collected from one building in 2016.

This project transforms the historical appliance energy measurements into an hourly forecasting problem, engineers temporal features, evaluates baseline and machine learning models, and provides a Streamlit dashboard for exploring a retrospective forecast from the dataset's last observation.

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
* Select a deployment model based on held-out 24-hour recursive test results.
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

Random Forest was selected for deployment because it achieved the lowest MAE among the evaluated models on the held-out 24-hour recursive test. The reported test metrics come from models trained on the chronological training period. Separately, `notebooks/04_forecasting.ipynb` refits the selected Random Forest configuration on all available engineered historical rows for the deployment artifact; that refitted model has not been evaluated on a later, independent period.

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

The model was evaluated using:

* One-step validation
* 24-hour recursive validation
* Held-out 24-hour recursive testing

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

The held-out test uses 19 non-overlapping forecast windows with origins every 24 hours, from 2016-05-08 05:00 through 2016-05-26 05:00. Each window predicts the following 24 hours, for 456 scored hourly predictions in total, ending at 2016-05-27 05:00. The remaining rows at the end of the source series are not part of these scored windows. These historical test results estimate performance for this dataset and split; they are not evidence of current-day performance or generalization to other households.

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

## 12. Held-Out Test Results

The following metrics summarize the held-out 24-hour recursive test windows described above. Random Forest and XGBoost were trained on the chronological training partition; these scores do not come from the all-history deployment refit.

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

Based on these held-out windows, **Random Forest had the lowest MAE among the approaches evaluated in this project**.

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

For this reason, model selection was based primarily on the forecasting scenario required by the application. The reported validation figures are separate from the held-out test and from the all-history deployment refit.

---

## 14. Final Forecasting Pipeline

The dashboard uses a Random Forest artifact refitted on all 3,121 available engineered hourly observations, from 2016-01-18 17:00 through 2016-05-27 17:00. Its 24-hour forecast is a retrospective continuation from that last observation, covering 2016-05-27 18:00 through 2016-05-28 17:00; it is not a live forecast for the present day.

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

The project includes an interactive Streamlit dashboard for displaying the retrospective forecasting results. It identifies the last historical observation and the forecast window. The displayed values are model predictions, not live measurements or validated predictions for present-day households.

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

### Data and Model Files

Place the UCI source file at `data/raw/energydata_complete.csv`. The raw-data directory is intentionally excluded from Git; obtain the dataset from the UCI source cited above and retain its attribution. The processed CSV files are included in the repository.

The deployment model is tracked with Git LFS. Install Git LFS, then retrieve the model payload after cloning:

```bash
git lfs install
git lfs pull
```

Confirm that `models/final_random_forest.pkl` is the full model file, not a small LFS pointer, before launching the dashboard.

### Reproducing the Notebooks

Run the notebooks in order from the project root or the `notebooks` directory:

1. `notebooks/01_data_understanding.ipynb` reads the raw dataset and writes `data/processed/hourly_energy_data.csv`.
2. `notebooks/02_feature_engineering.ipynb` writes `data/processed/engineered_energy_data.csv`.
3. `notebooks/03_model_training.ipynb` evaluates the existing chronological train/validation/test workflow.
4. `notebooks/04_forecasting.ipynb` refits and saves the deployment model and feature configuration, and writes a forecast CSV.

The fourth notebook intentionally overwrites the tracked model, feature configuration, and forecast output when its save cells are run. Skip those save cells during review-only runs, or preserve/restore the tracked artifacts first. The dashboard itself does not retrain the model.

---

## 18. Running the Dashboard

From the project root:

```bash
streamlit run app/app.py
```

Streamlit will provide a local URL that can be opened in a web browser.

The dashboard displays a retrospective forecast from the dataset's latest observation in May 2016. It does not fetch current measurements and must not be interpreted as a present-day forecast.

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
* Interpret held-out test scores only for their documented 2016 forecast windows; distinguish them from the deployment model refitted on all available history.

---

## 20. Limitations

### Limited Observation Period

The original dataset covers approximately 4.5 months. Therefore, the dataset does not provide a full year of observations for learning long-term annual seasonality.

### Building-Specific Data

The original measurements were collected from a particular low-energy residential building. The resulting model should therefore not automatically be assumed to generalize to other buildings or households.

### Historical Forecast Only

The source series ends on 2016-05-27. The dashboard forecast is a historical illustration from that endpoint, not a prediction for current conditions or a validated operational forecast.

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
