import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
# Assuming these modules are available and contain the respective forecasting functions
from xgboost_forecasting import run_xgboost_forecast # Import the XGBoost forecast function
from garch_forecasting import run_garch_forecast # Import the GARCH forecast function
# Assuming FEATURE_COLS is defined in data_processing or a common config file
# from data_processing import FEATURE_COLS # Import FEATURE_COLS if not defined here

# Setup logging for this module
logger = logging.getLogger(__name__)

# --- Define FEATURE_COLS if not imported from data_processing ---
# Ensure FEATURE_COLS is defined if it's not imported globally
# If you have a common config.py, import it from there.
try:
    from data_processing import FEATURE_COLS
    logger.info("FEATURE_COLS imported from data_processing.")
except ImportError:
    logger.warning("FEATURE_COLS not found in data_processing. Using a default list. Ensure consistency!")
    # Define a default list if import fails - MUST match features generated in data_processing
    FEATURE_COLS = [
        'VIX', 'ATM_IV', 'IVP', 'PCR', 'VIX_Change_Pct', 'IV_Skew',
        'Straddle_Price', 'Spot_MaxPain_Diff_Pct', 'Days_to_Expiry',
        'Event_Flag', 'FII_Index_Fut_Pos', 'FII_Option_Pos', 'Realized_Vol',
        'Advance_Decline_Ratio', 'Capital_Pressure_Index', 'Gamma_Bias',
        'NIFTY_Close', 'Total_Capital' # Assuming these are the features generated
    ]


# === Volatility Forecasting Function ===

def forecast_volatility_future(analysis_df: pd.DataFrame, forecast_horizon: int):
    """
    Runs both XGBoost and GARCH volatility forecasting models, blends their results,
    and calculates key forecast metrics.

    Args:
        analysis_df (pd.DataFrame): DataFrame containing historical and latest
                                   market data with generated features. Must
                                   have a datetime index and contain FEATURE_COLS.
        forecast_horizon (int): The number of future days to forecast volatility for.

    Returns:
        tuple: (forecast_log_df, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, feature_importances)
               - forecast_log_df (pd.DataFrame or None): DataFrame with daily forecast values.
               - garch_vols (np.ndarray or None): GARCH forecast values.
               - xgb_vols (np.ndarray or None): XGBoost forecast values.
               - blended_vols (np.ndarray or None): Blended forecast values.
               - realized_vol (float or None): Latest calculated historical realized volatility.
               - confidence_score (float or None): Confidence score for the forecast.
               - rmse (float or None): RMSE of the XGBoost model (on historical data).
               - feature_importances (list or None): Feature importances from XGBoost.
               Returns (None, None, None, None, None, None, None, None) if forecasting fails critically.
    """
    logger.info(f"Starting volatility forecasting for {forecast_horizon} days.")

    # --- 1. Validate Input Data ---
    if analysis_df is None or analysis_df.empty:
        logger.error("Analysis DataFrame is empty or None. Cannot run volatility forecast.")
        # Return None for all outputs if critical input data is missing
        return None, None, None, None, None, None, None, None
    # Check if necessary columns are present
    if 'VIX' not in analysis_df.columns or 'NIFTY_Close' not in analysis_df.columns:
         logger.error("Analysis DataFrame missing 'VIX' or 'NIFTY_Close' columns. Cannot run forecast.")
         return None, None, None, None, None, None, None, None

    # Ensure the analysis DataFrame has a datetime index and is sorted by date
    analysis_df.index = pd.to_datetime(analysis_df.index).normalize()
    analysis_df = analysis_df.sort_index()

    # Extract the latest realized volatility from the DataFrame (if calculated in data_processing)
    # Safely get the last value of the 'Realized_Vol' column
    realized_vol = analysis_df['Realized_Vol'].iloc[-1] if 'Realized_Vol' in analysis_df.columns and not analysis_df['Realized_Vol'].empty else None
    if pd.isna(realized_vol): realized_vol = None # Ensure it's None if NaN


    # --- 2. Run Individual Forecasting Models ---

    garch_vols = None
    try:
        logger.info("Running GARCH forecast.")
        # Call the GARCH forecasting function. It likely takes VIX series and horizon.
        # Pass the VIX series from the analysis DataFrame.
        garch_vols = run_garch_forecast(analysis_df['VIX'], forecast_horizon)
        if garch_vols is not None and isinstance(garch_vols, np.ndarray) and len(garch_vols) > 0:
             logger.info(f"GARCH forecast completed. {len(garch_vols)} values.")
        else:
             logger.warning("GARCH forecast returned None or empty array.")
             garch_vols = None # Ensure it's None if empty or invalid return

    except Exception as e:
        logger.error(f"Error running GARCH forecast: {str(e)}", exc_info=True)
        garch_vols = None # Ensure garch_vols is None if an error occurs


    xgb_results = None # XGBoost returns multiple outputs (forecast, rmse, feature_importances)
    xgb_vols = None
    rmse = None
    feature_importances = None
    try:
        logger.info("Running XGBoost forecast.")
        # Call the XGBoost forecasting function. It likely takes the full analysis_df, horizon, and FEATURE_COLS.
        # Pass a copy of the DataFrame and the list of feature columns.
        # Ensure run_xgboost_forecast can handle the DataFrame and returns expected types.
        xgb_results = run_xgboost_forecast(analysis_df.copy(), forecast_horizon, FEATURE_COLS)

        # Check if XGBoost forecast returned the expected tuple of results
        if xgb_results is not None and isinstance(xgb_results, tuple) and len(xgb_results) == 3:
             xgb_vols, rmse, feature_importances = xgb_results # Unpack the results
             if xgb_vols is not None and isinstance(xgb_vols, np.ndarray) and len(xgb_vols) > 0:
                  logger.info(f"XGBoost forecast completed. {len(xgb_vols)} values. RMSE: {rmse}")
             else:
                  logger.warning("XGBoost forecast returned None or empty array for vols.")
                  xgb_vols = None # Ensure xgb_vols is None if empty or invalid

             # Validate rmse and feature_importances
             if rmse is not None and not isinstance(rmse, (int, float)): rmse = None
             if feature_importances is not None and not isinstance(feature_importances, list): feature_importances = None # Ensure list format

        else:
             logger.warning(f"XGBoost forecast returned unexpected result type or structure: {type(xgb_results)}")
             xgb_vols, rmse, feature_importances = None, None, None # Set all to None if return is unexpected


    except Exception as e:
        logger.error(f"Error running XGBoost forecast: {str(e)}", exc_info=True)
        xgb_vols, rmse, feature_importances = None, None, None # Ensure all are None if an error occurs


    # --- 3. Blend Forecasts ---
    # Blend the forecasts if both are available. If only one is available, use that one.
    # If neither is available, blended_vols will be None.
    blended_vols = None
    confidence_score = None # Confidence score based on agreement or RMSE

    if garch_vols is not None and xgb_vols is not None:
        logger.info("Both GARCH and XGBoost forecasts available. Blending forecasts.")
        # Ensure both arrays have the same length for blending
        min_len = min(len(garch_vols), len(xgb_vols))
        if min_len > 0:
            # Simple average blending example (you could use weighted average, etc.)
            blended_vols = (garch_vols[:min_len] + xgb_vols[:min_len]) / 2.0
            logger.info(f"Blended {len(blended_vols)} forecast values.")

            # Calculate confidence score based on how close the forecasts are
            # Lower difference implies higher confidence. Scale inversely.
            # Example: use the average absolute difference between the two forecasts
            avg_diff = np.mean(np.abs(garch_vols[:min_len] - xgb_vols[:min_len]))
            # Scale this difference inversely to get a confidence score (e.g., 100 - scaled_diff)
            # Need a scaling factor; this is heuristic. Assume a diff of 5 VIX points means 0 confidence loss.
            scaling_factor = 100 / 5.0 # 100% confidence reduction for 5 points difference
            confidence_score = max(0, 100 - avg_diff * scaling_factor) # Score between 0 and 100, min 0

            logger.info(f"Forecast blending completed. Confidence score: {confidence_score:.2f}")
        else:
             logger.warning("GARCH and XGBoost forecasts have zero length. Cannot blend.")


    elif garch_vols is not None:
        logger.info("Only GARCH forecast available. Using GARCH forecast as blended.")
        blended_vols = garch_vols # Use GARCH if XGBoost failed
        confidence_score = 50.0 # Assign a medium confidence score if only one model ran

    elif xgb_vols is not None:
        logger.info("Only XGBoost forecast available. Using XGBoost forecast as blended.")
        blended_vols = xgb_vols # Use XGBoost if GARCH failed
        confidence_score = 50.0 # Assign a medium confidence score if only one model ran

    else:
        logger.error("Both GARCH and XGBoost forecasts failed or returned no values. Cannot produce blended forecast.")
        blended_vols = None # Ensure it's None if both failed
        confidence_score = 0.0 # Assign low confidence if neither model ran


    # Ensure blended_vols are non-negative, as volatility cannot be negative
    if blended_vols is not None:
         blended_vols = np.maximum(0, blended_vols) # Clip at 0


    # --- 4. Prepare Forecast Log DataFrame ---
    forecast_log_df = None # Initialize as None
    if blended_vols is not None and len(blended_vols) > 0:
        try:
            # Create a DataFrame with the forecast date range and the blended forecast values
            latest_date = analysis_df.index[-1] # Get the last date from the analysis_df
            # Generate future dates based on the latest date and forecast horizon.
            # Assume business days if forecasting trading volatility, or just calendar days.
            # Let's assume calendar days for simplicity based on horizon input.
            forecast_dates = [latest_date + timedelta(days=i) for i in range(1, forecast_horizon + 1)] # Forecast starts from *next* day

            # Ensure the length of dates matches the length of forecast values
            if len(forecast_dates) == len(blended_vols):
                 forecast_log_df = pd.DataFrame({
                     "Date": forecast_dates,
                     "Blended_Vol": blended_vols.tolist(), # Convert numpy array to list for DataFrame
                     "GARCH_Vol": garch_vols.tolist() if garch_vols is not None and len(garch_vols) == len(blended_vols) else [None] * len(blended_vols), # Include GARCH if available and same length
                     "XGBoost_Vol": xgb_vols.tolist() if xgb_vols is not None and len(xgb_vols) == len(blended_vols) else [None] * len(blended_vols) # Include XGBoost if available and same length
                 })
                 forecast_log_df = forecast_log_df.set_index("Date") # Set Date as index
                 logger.info(f"Forecast log DataFrame created with {len(forecast_log_df)} entries.")
                 logger.debug("Forecast log preview:")
                 logger.debug(forecast_log_df.head())

            else:
                 logger.error(f"Mismatch between forecast dates ({len(forecast_dates)}) and blended forecast values ({len(blended_vols)}) length. Cannot create forecast log.")
                 forecast_log_df = None # Set to None if lengths don't match

        except Exception as e:
             logger.error(f"Error creating forecast log DataFrame: {str(e)}", exc_info=True)
             forecast_log_df = None # Set to None on error


    # --- 5. Final Checks and Return ---
    logger.info("Volatility forecasting process completed.")
    # Return all calculated and generated outputs. Some may be None if errors occurred.
    return forecast_log_df, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, feature_importances

# The __main__ block is removed as this module's functions are intended to be imported
# and used by other parts of the application.
# To test this function, you would typically call it from a separate script
# with a dummy analysis_df DataFrame and mocked or actual outputs from
# run_garch_forecast and run_xgboost_forecast.
