# Electricity Demand Forecasting

Time series forecasting of electricity consumption using XGBoost. The goal was to build a model that can reliably predict hourly electricity demand based on historical usage patterns, weather conditions, and calendar features — the kind of problem grid operators and energy companies deal with every day.

---

## The Problem

Electricity demand is not random. It follows patterns: mornings spike when people wake up, summers heat up with AC usage, weekends drop off. The challenge is capturing all of that structure — hourly, daily, weekly, seasonal — and translating it into a forecast that actually holds up on unseen data.

This project works through that end-to-end: raw CSV data, messy real-world missing values, feature engineering, model training with proper time-based splits, and evaluation.

---

## Dataset

The dataset contains hourly electricity readings with the following columns:

- `Timestamp` — hourly datetime index
- `Demand` — electricity consumption (target variable)
- `Temperature` — ambient temperature
- `Humidity` — relative humidity
- `hour`, `dayofweek`, `month`, `year`, `dayofyear` — pre-extracted time features

---

## What Was Done

**Data Cleaning**

Real data has gaps. Rather than dropping rows blindly, different strategies were applied depending on the column:
- Calendar features (`hour`, `dayofweek`, etc.) were forward-filled since they follow a strict sequence
- `Temperature` and `Humidity` were backward-filled to use the nearest future reading
- `Demand` itself was interpolated using time-based interpolation, which respects the temporal nature of the signal better than a simple mean fill

**Feature Engineering**

Raw timestamps do not mean much to a model. Additional features were derived to give it more context:

- `Quarter` — which quarter of the year
- `weekofyear` — ISO week number
- `is_weekend` — binary flag for Saturday/Sunday
- `Demand_lag_24hr` — demand from exactly 24 hours ago (same hour yesterday)
- `Demand_lag_168hr` — demand from 168 hours ago (same hour last week)
- `Demand_rolling_mean_24hr` — 24-hour rolling average
- `Demand_rolling_std_24hr` — 24-hour rolling standard deviation

The lag features are particularly important here. Electricity demand has strong autocorrelation — what happened yesterday at 9am is a strong predictor of what happens today at 9am.

**Exploratory Analysis**

Before modeling, the data was visualized to understand its structure:
- Demand over the full time range to check for trends and anomalies
- Hourly demand distribution using box plots — confirms the morning and evening peaks
- Monthly demand distribution — captures seasonal variation
- Scatter plot of demand vs temperature — shows the expected nonlinear relationship (both cold and hot extremes drive demand up)
- Correlation heatmap across all features

**Model**

XGBoost Regressor, trained with the following setup:

```python
XGBRegressor(
    n_estimators=1000,
    early_stopping_rounds=50,
    learning_rate=0.01,
    objective="reg:squarederror",
    random_state=42
)
```

The train/test split is time-based, not random — data through end of 2023 for training, 2024 onwards for testing. This is the correct way to evaluate a forecasting model. Shuffling time series data to create a random split would leak future information into training and produce misleading metrics.

---

## Results

| Metric | Score |
|--------|-------|
| RMSE | see notebook output |
| MAE | see notebook output |

The actual vs predicted plot shows the model tracking the demand signal closely, including the daily cycles and weekly patterns.

---

## Project Structure

```
.
├── electricity_consumption.csv          # raw input data
├── Electricity_Demand.ipynb             # full analysis and modeling notebook
└── electricity_xgb_prediction_model.pkl # saved trained model
```

---

## Stack

- Python 3
- pandas, numpy
- matplotlib, seaborn
- XGBoost
- scikit-learn
- joblib

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn xgboost scikit-learn joblib
jupyter notebook Electricity_Demand.ipynb
```

The trained model is saved as a `.pkl` file and can be loaded directly for inference without retraining:

```python
import joblib
model = joblib.load("electricity_xgb_prediction_model.pkl")
predictions = model.predict(X_new)
```
