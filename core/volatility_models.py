import streamlit as st
import pandas as pd
import numpy as np
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import requests
import pickle
from datetime import datetime, timedelta
from config.settings import NIFTY_CSV_URL, XGB_MODEL_URL, XGB_CSV_URL
from utils.logger import setup_logger
from utils.helpers import compute_realized_vol, calculate_rolling_and_fixed_hv

logger = setup_logger()

def run_garch_forecast():
    """Run GARCH volatility forecast."""
    try:
        nifty_df = pd.read_csv(NIFTY_CSV_URL)
        nifty_df.columns = nifty_df.columns.str.strip()
        nifty_df["Date"] = pd.to_datetime(nifty_df["Date"], format="%d-%b-%Y", errors="coerce")
        nifty_df = nifty_df.dropna(subset=["Date"]).set_index("Date")
        nifty_df = nifty_df.rename(columns={"Close": "NIFTY_Close"})
        if 'NIFTY_Close' not in nifty_df.columns:
            raise ValueError("CSV missing 'NIFTY_Close' column")
        nifty_df = nifty_df[["NIFTY_Close"]].dropna().sort_index()

        log_returns = np.log(nifty_df["NIFTY_Close"].pct_change() + 1).dropna() * 100
        model = arch_model(log_returns, vol="Garch", p=1, q=1)
        model_fit = model.fit(disp="off")
        forecast_horizon = 7
        garch_forecast = model_fit.forecast(horizon=forecast_horizon)
        garch_vols = np.sqrt(garch_forecast.variance.values[-1]) * np.sqrt(252)
        garch_vols = np.clip(garch_vols, 5, 50)
        last_date = nifty_df.index[-1]
        forecast_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=forecast_horizon)
        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Day": forecast_dates.day_name(),
            "Forecasted Volatility (%)": np.round(garch_vols, 2)
        })

        realized_vol = compute_realized_vol(nifty_df)
        rv_7d_df, hv_30d, hv_1y = calculate_rolling_and_fixed_hv(nifty_df["NIFTY_Close"])
        return forecast_df, realized_vol, hv_30d, hv_1y
    except Exception as e:
        logger.error(f"GARCH error: {e}")
        return None, 0, 0, 0

def run_xgboost_evaluation():
    """Evaluate XGBoost model performance."""
    try:
        xgb_df = pd.read_csv(XGB_CSV_URL)
        xgb_df = xgb_df.dropna()
        features = ['ATM_IV', 'Realized_Vol', 'IVP', 'Event_Impact_Score', 'FII_DII_Net_Long', 'PCR', 'VIX']
        target = 'Next_5D_Realized_Vol'
        if not all(col in xgb_df.columns for col in features + [target]):
            raise ValueError("CSV missing required columns")
        X = xgb_df[features]
        y = xgb_df[target] * 100

        response = requests.get(XGB_MODEL_URL)
        if response.status_code != 200:
            raise ValueError("Failed to load XGBoost model")
        xgb_model = pickle.loads(response.content)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        y_pred_train = xgb_model.predict(X_train)
        y_pred_test = xgb_model.predict(X_test)

        metrics = {
            'rmse_train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'mae_train': mean_absolute_error(y_train, y_pred_train),
            'mae_test': mean_absolute_error(y_test, y_pred_test),
            'r2_train': r2_score(y_train, y_pred_train),
            'r2_test': r2_score(y_test, y_pred_test),
            'feature_importances': xgb_model.feature_importances_,
            'features': features
        }
        return metrics
    except Exception as e:
        logger.error(f"XGBoost evaluation error: {e}")
        return None

def run_xgboost_prediction(new_data):
    """Run XGBoost prediction for new data."""
    try:
        response = requests.get(XGB_MODEL_URL)
        if response.status_code != 200:
            raise ValueError("Failed to load XGBoost model")
        xgb_model = pickle.loads(response.content)
        prediction = xgb_model.predict(new_data)[0]
        return prediction
    except Exception as e:
        logger.error(f"XGBoost prediction error: {e}")
        return None
