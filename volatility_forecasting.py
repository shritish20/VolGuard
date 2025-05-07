import logging
import pandas as pd
import numpy as np
from datetime import timedelta
from garch_forecasting import forecast_volatility_garch # Import GARCH function
from xgboost_forecasting import forecast_volatility_xgboost # Import XGBoost function
from data_processing import FEATURE_COLS # Import FEATURE_COLS

# Setup logging
logger = logging.getLogger(__name__)

def forecast_volatility_future(df, forecast_horizon):
    """
    Blends GARCH and XGBoost forecasts and calculates confidence.
    """
    try:
        logger.info("Starting volatility forecasting")
        df = df.copy()
        df.index = pd.to_datetime(df.index).normalize()

        # Ensure df has enough data and required columns for both models
        required_cols = list(FEATURE_COLS) + ['NIFTY_Close', 'Realized_Vol', 'VIX']
        if len(df) < 200 or not all(col in df.columns for col in required_cols):
            logger.error(f"Insufficient data ({len(df)} days) or missing columns ({[col for col in required_cols if col not in df.columns]}) for volatility forecasting.")
            return None, None, None, None, None, None, None, None


        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon, freq='B')

        # Ensure realized_vol calculation handles potential NaNs
        realized_vol = df["Realized_Vol"].dropna().iloc[-5:].mean() if not df["Realized_Vol"].dropna().empty and len(df["Realized_Vol"].dropna().iloc[-5:]) >= 1 else df["VIX"].iloc[-1] if "VIX" in df.columns and len(df) > 0 and not pd.isna(df["VIX"].iloc[-1]) else 15.0 # Fallback realized vol


        # --- Run GARCH Forecasting ---
        garch_vols = forecast_volatility_garch(df.tail(max(200, forecast_horizon * 2)), forecast_horizon) # Use a reasonable window for GARCH

        # --- Run XGBoost Forecasting ---
        # Pass the full DataFrame for XGBoost as it handles its own data splitting
        xgb_vols, rmse, feature_importances = forecast_volatility_xgboost(df, forecast_horizon)

        # Ensure garch_vols and xgb_vols are lists/arrays of the same length
        if not isinstance(garch_vols, (list, np.ndarray)):
             garch_vols = np.full(forecast_horizon, realized_vol) # Fallback if GARCH failed
        if not isinstance(xgb_vols, (list, np.ndarray)):
             xgb_vols = np.full(forecast_horizon, realized_vol) # Fallback if XGBoost failed

        # Ensure lengths match before blending
        min_horizon = min(len(garch_vols), len(xgb_vols), forecast_horizon)
        garch_vols = garch_vols[:min_horizon]
        xgb_vols = xgb_vols[:min_horizon]
        future_dates = future_dates[:min_horizon]


        # Blending GARCH and XGBoost
        if min_horizon > 0:
             # Calculate initial difference for weighting based on the first forecast day
             initial_garch_vol = garch_vols[0] if len(garch_vols) > 0 else realized_vol
             initial_xgb_vol = xgb_vols[0] if len(xgb_vols) > 0 else realized_vol

             garch_diff = np.abs(initial_garch_vol - realized_vol)
             xgb_diff = np.abs(initial_xgb_vol - realized_vol)

             # Avoid division by zero or very small numbers
             total_diff = garch_diff + xgb_diff
             garch_weight = xgb_diff / total_diff if total_diff > 0 else 0.5
             xgb_weight = 1 - garch_weight

             # Apply weights across the forecast horizon
             blended_vols = [(garch_weight * g) + (xgb_weight * x) for g, x in zip(garch_vols, xgb_vols)]

             # Confidence based on the agreement between models and deviation from realized
             model_agreement = 1 - (np.abs(initial_garch_vol - initial_xgb_vol) / max(initial_garch_vol, initial_xgb_vol) if max(initial_garch_vol, initial_xgb_vol) > 0 else 0)
             deviation_from_realized = 1 - (min(garch_diff, xgb_diff) / realized_vol if realized_vol > 0 else 0)
             confidence_score = min(100, max(30, (model_agreement * 0.6 + deviation_from_realized * 0.4) * 100)) # Adjust weighting and min confidence
        else:
             blended_vols = []
             confidence_score = 50 # Low confidence if no forecast days
             logger.warning("No forecast days available for blending.")


        forecast_log = pd.DataFrame({
            "Date": future_dates,
            "GARCH_Vol": garch_vols,
            "XGBoost_Vol": xgb_vols,
            "Blended_Vol": blended_vols,
            "Confidence": [confidence_score] * min_horizon if min_horizon > 0 else [] # Assign confidence per day
        })
        logger.debug("Volatility forecasting completed")
        return forecast_log, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, feature_importances
    except Exception as e:
        logger.error(f"Error in volatility forecasting: {str(e)}", exc_info=True)
        return None, None, None, None, None, None, None, None

