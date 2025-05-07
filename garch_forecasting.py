import logging
import pandas as pd
import numpy as np
from arch import arch_model

# Setup logging
logger = logging.getLogger(__name__)

def forecast_volatility_garch(df_garch, forecast_horizon):
    """
    Forecasts volatility using a GARCH(1,1) model.
    Takes a DataFrame with 'NIFTY_Close' and returns GARCH volatility forecasts.
    """
    try:
        logger.info("Forecasting volatility using GARCH(1,1)")

        if len(df_garch) < 100: # Minimum data for GARCH stability
             logger.warning(f"Insufficient data ({len(df_garch)} rows) for GARCH model. Skipping GARCH forecast.")
             # Return default values or handle appropriately in the caller
             return np.full(forecast_horizon, df_garch["VIX"].iloc[-1] if "VIX" in df_garch.columns and len(df_garch) > 0 else 15.0)


        # Ensure Log_Returns calculation handles potential NaNs at the beginning
        df_garch['Log_Returns'] = np.log(df_garch['NIFTY_Close'].pct_change() + 1).dropna() * 100

        if df_garch['Log_Returns'].empty or len(df_garch['Log_Returns']) < 100:
             logger.warning(f"Insufficient historical returns data ({len(df_garch['Log_Returns'])}) for GARCH.")
             # Return default values or handle appropriately in the caller
             return np.full(forecast_horizon, df_garch["VIX"].iloc[-1] if "VIX" in df_garch.columns and len(df_garch) > 0 else 15.0)


        garch_model = arch_model(df_garch['Log_Returns'], vol='Garch', p=1, q=1, rescale=False)
        garch_fit = garch_model.fit(disp="off")
        garch_forecast = garch_fit.forecast(horizon=forecast_horizon, reindex=False)

        # Convert conditional standard deviation to annualized volatility (%)
        garch_vols = np.sqrt(garch_forecast.variance.iloc[-1].values) * np.sqrt(252)
        garch_vols = np.clip(garch_vols, 5, 50) # Apply reasonable bounds

        logger.debug("GARCH forecast completed.")
        return garch_vols

    except Exception as e:
        logger.error(f"Error in GARCH volatility forecasting: {str(e)}", exc_info=True)
        # Return default values or handle appropriately in the caller
        return np.full(forecast_horizon, df_garch["VIX"].iloc[-1] if "VIX" in df_garch.columns and len(df_garch) > 0 else 15.0)

