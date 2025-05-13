import logging
import pandas as pd
import numpy as np
from arch import arch_model # Import the arch_model from the ARCH library
import warnings # To manage warnings from model fitting

# Setup logging for this module
logger = logging.getLogger(__name__)

# === GARCH Forecasting Function ===

def run_garch_forecast(vix_series: pd.Series, forecast_horizon: int):
    """
    Fits a GARCH(1,1) model to the input VIX time series and forecasts
    volatility for a specified future horizon.

    Args:
        vix_series (pd.Series): A pandas Series of historical VIX values.
                                Must have a datetime index and should be free of NaNs.
                                Expected to be sorted chronologically.
        forecast_horizon (int): The number of future periods (days) to forecast.

    Returns:
        np.ndarray or None: A numpy array containing the forecasted volatility
                            values for the specified horizon, or None if the
                            model fitting or forecasting fails.
    """
    logger.info(f"Starting GARCH volatility forecast for {forecast_horizon} days.")

    # --- 1. Validate and Prepare Data ---
    if vix_series is None or vix_series.empty:
        logger.error("Input VIX series is empty or None. Cannot run GARCH forecast.")
        return None
    # Ensure the VIX series is a pandas Series with a valid index
    if not isinstance(vix_series, pd.Series):
         logger.error(f"Input VIX series is not a pandas Series ({type(vix_series)}). Cannot run GARCH forecast.")
         return None
    # Ensure the index is a DatetimeIndex (or can be converted)
    try:
        vix_series.index = pd.to_datetime(vix_series.index)
        vix_series = vix_series.sort_index() # Ensure sorted
    except Exception as e:
        logger.error(f"Error converting VIX series index to datetime: {str(e)}. Cannot run GARCH forecast.")
        return None


    # Handle NaNs in the VIX series. GARCH models cannot handle NaNs.
    initial_nan_count = vix_series.isna().sum()
    if initial_nan_count > 0:
        logger.warning(f"NaNs found in input VIX series ({initial_nan_count}). Attempting imputation (ffill/bfill).")
        # Impute NaNs using forward fill then backward fill
        vix_series = vix_series.fillna(method='ffill').fillna(method='bfill')
        # If NaNs still exist (e.g., all values were NaN), fill with a default or the first valid value.
        # Using the mean of the series as a fallback if all NaNs remain
        if vix_series.isna().sum() > 0:
             mean_vix = vix_series.mean() # Calculate mean of available data
             vix_series = vix_series.fillna(mean_vix if pd.notna(mean_vix) else 15.0) # Fill remaining NaNs with mean or 15.0 default
             logger.warning(f"NaNs still present after ffill/bfill, filled with mean/default. Remaining NaNs: {vix_series.isna().sum()}")


    # Ensure we have enough data points to train a GARCH model
    # GARCH models need a reasonable history (e.g., at least 30-50 points)
    min_data_points = 50
    if len(vix_series) < min_data_points:
        logger.error(f"Insufficient data points ({len(vix_series)}) to train GARCH model. Need at least {min_data_points}.")
        return None


    # The GARCH model typically models the conditional variance of a time series.
    # It is usually applied to returns (percentage changes) rather than levels.
    # However, for modeling VIX level volatility directly, we can model the VIX series itself.
    # If the intention is to model the volatility of VIX *returns*, calculate percentage changes first.
    # Based on the call in volatility_forecasting.py, it seems we are passing the VIX level series directly.
    # Let's proceed with modeling the VIX level directly as the process, assuming it's treated like returns by the model.
    # If the goal is to model VIX returns volatility, the input series should be vix_series.pct_change().dropna().
    # Assuming we are modeling the VIX level directly here as per potential original intent from backend code structure.
    data_to_model = vix_series # Use the VIX series directly

    # Ensure data_to_model is numeric and has no remaining NaNs
    data_to_model = pd.to_numeric(data_to_model, errors='coerce').dropna()
    if data_to_model.empty:
         logger.error("Data for GARCH model is empty after numeric conversion and NaN drop. Cannot proceed.")
         return None


    # --- 2. Fit the GARCH Model ---
    logger.info("Fitting GARCH(1,1) model.")
    # Define and fit the GARCH model.
    # Using a simple GARCH(1,1) specification with ZeroMean and Normal distribution.
    # This is a common starting point, but optimal orders (p, q) might vary.
    # ZeroMean is often suitable for series assumed to fluctuate around zero (like returns)
    # or if the mean has already been removed. For VIX levels, ConstantMean might be more appropriate.
    # Let's use ConstantMean for VIX levels for better fitting of the level.
    try:
        # Suppress warnings from the ARCH library during fitting to keep logs cleaner unless an error occurs
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore") # Ignore warnings

            # Initialize and fit the GARCH model
            # mean='Constant' to model the mean level of VIX
            # vol='Garch' for GARCH variance model
            # p=1, q=1 for GARCH(1,1) order
            # dist='normal' for assuming normal distribution of residuals
            model = arch_model(data_to_model, mean='Constant', vol='Garch', p=1, q=1, dist='normal')
            results = model.fit(disp='off') # disp='off' suppresses iteration output

        logger.info("GARCH model fitting completed.")
        # logger.debug(results.summary()) # Uncomment to see model summary in logs


    except Exception as e:
        # Catch any errors during model fitting (e.g., convergence issues)
        logger.error(f"Error during GARCH model fitting: {str(e)}", exc_info=True)
        # Return None if fitting fails
        return None


    # --- 3. Forecast Future Volatility ---
    garch_forecast_values = None # Initialize as None
    try:
        logger.info(f"Generating GARCH forecast for horizon {forecast_horizon}.")
        # Use the fitted results to forecast future conditional volatility.
        # Forecast starts *after* the last date in the data_to_model series.
        forecast_results = results.forecast(horizon=forecast_horizon, start=None, reindex=True)

        # The forecast_results object contains mean forecast, conditional variance forecast, etc.
        # We need the conditional volatility forecast, which is the square root of the conditional variance forecast.
        # The conditional variance is in forecast_results.variance.iloc[-1] (the forecast from the last date)
        # .iloc[-1] selects the forecast starting from the last date in the input data.
        # Need to ensure forecast_results.variance is not empty and has at least one row.
        if forecast_results.variance is not None and not forecast_results.variance.empty:
            # Extract the variance forecasts for the specified horizon.
            # The columns are named 'h.1', 'h.2', ..., 'h.horizon'.
            # Select the variance forecasts corresponding to the horizon.
            # Ensure we only select columns that exist and match the horizon.
            variance_cols = [f'h.{i}' for i in range(1, forecast_horizon + 1)]
            # Filter for columns that actually exist in the variance DataFrame
            existing_variance_cols = [col for col in variance_cols if col in forecast_results.variance.columns]

            if existing_variance_cols:
                 # Extract the forecasts for the horizon from the *last row* of the variance DataFrame
                 # This row contains the forecasts originating from the last data point used for fitting.
                 # Convert to numpy array.
                 garch_forecast_variance = forecast_results.variance[existing_variance_cols].iloc[-1].values

                 # Calculate volatility as the square root of variance
                 garch_forecast_values = np.sqrt(garch_forecast_variance)

                 # Ensure forecast values are non-negative (square root is already >= 0, but a check is safe)
                 garch_forecast_values = np.maximum(0, garch_forecast_values)

                 logger.info(f"GARCH forecast generated: {len(garch_forecast_values)} values.")
                 logger.debug(f"GARCH forecast values (first 5): {garch_forecast_values[:5]}")

            else:
                 logger.error(f"Forecast variance columns for horizon {forecast_horizon} not found in GARCH forecast results.")
                 garch_forecast_values = None # Ensure None if expected columns are missing

        else:
            logger.error("GARCH forecast results did not contain expected variance data.")
            garch_forecast_values = None # Ensure None if variance data is missing


    except Exception as e:
        # Catch any errors during the forecasting process
        logger.error(f"Error during GARCH forecasting: {str(e)}", exc_info=True)
        garch_forecast_values = None # Ensure None if forecasting fails


    # --- 4. Return Results ---
    logger.info("GARCH volatility forecasting process completed.")
    # Return the numpy array of forecast values.
    return garch_forecast_values

# The __main__ block is removed as this module's functions are intended to be imported
# and used by other parts of the application.
# To test this function, you would typically call it from a separate script
# with a dummy pandas Series of VIX values and a forecast horizon.
