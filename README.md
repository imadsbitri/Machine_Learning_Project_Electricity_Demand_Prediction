# Machine_Learning_Project_Electricity_Demand_Prediction
---

## Production Model & Performance Metrics

The predictive core is powered by an optimized **XGBoost Regressor**, selected for its capacity to interpret highly non-linear climate interactions and complex auto-regressive time-series features without requiring computationally expensive spatial architectures.

### Feature Importance Profile
The model leans heavily on engineered indicators, preventing drift and ensuring precise fitting:
1. `Demand_lag_24hr` / `Demand_lag_168hr`: High predictive weights, proving strong persistence of weekly grid routines.
2. `Temperature`: Primary exogenous driver, mapping consumption surges during extreme seasonal climate conditions.
3. `Hour` / `is_weekend`: Key socio-economic parameters mapping standard intraday utility spikes and industrial downscaling.

### Evaluation Metrics (Test Set - 2024)
Evaluated strictly against unseen calendar data, the framework achieves stable, production-ready accuracy scores:

* **Mean Absolute Error (MAE):** 74.45 MW
* **Root Mean Squared Error (RMSE):** 112.18 MW
* **Mean Absolute Percentage Error (MAPE):** 1.84%
* **R-squared Coefficient ($R^2$):** 0.9926

---

## Repository Structure
