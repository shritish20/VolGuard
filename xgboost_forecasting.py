import logging
import pandas as pd
import numpy as np
import xgboost as xgb # Import the XGBoost library
from sklearn.model_selection import train_test_split # Still useful for splitting historical data for evaluation
from sklearn.metrics import mean_squared_error # To evaluate model performance (RMSE)
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


# === XGBoost Forecasting Function ===

def run_xgboost_forecast(analysis_df: pd.DataFrame, forecast_horizon: int, feature_cols: list):
    """
    Trains an XGBoost Regressor model on historical data and forecasts
    volatility for a specified future horizon based on the latest features.

    Args:
        analysis_df (pd.DataFrame): DataFrame containing historical and latest
                                   market data with generated features. Must
                                   have a datetime index and contain FEATURE_COLS.
                                   Expected to be sorted by date.
        forecast_horizon (int): The number of future days to forecast volatility for.
        feature_cols (list): A list of column names to use as features for the model.
                             Must be present in analysis_df.

    Returns:
        tuple: (xgb_forecast_values, rmse, feature_importances)
               - xgb_forecast_values (np.ndarray or None): Forecasted volatility values for the horizon.
               - rmse (float or None): Root Mean Squared Error of the model on a historical test set.
               - feature_importances (list or None): List of feature importance values from the trained model.
               Returns (None, None, None) if forecasting fails.
    """
    logger.info(f"Starting XGBoost volatility forecast for {forecast_horizon} days.")

    # --- 1. Validate and Prepare Data ---
    if analysis_df is None or analysis_df.empty:
        logger.error("Analysis DataFrame is empty or None. Cannot run XGBoost forecast.")
        return None, None, None
    # Ensure required feature columns exist
    missing_cols = [col for col in feature_cols if col not in analysis_df.columns]
    if missing_cols:
        logger.error(f"Analysis DataFrame missing required feature columns: {missing_cols}. Cannot run XGBoost forecast.")
        return None, None, None
    # Ensure the target variable ('VIX') exists
    if 'VIX' not in analysis_df.columns:
         logger.error("Analysis DataFrame missing 'VIX' column (target variable). Cannot run XGBoost forecast.")
         return None, None, None

    # Select only the feature columns and the target variable
    # Ensure only feature_cols that exist in the DataFrame are selected
    existing_feature_cols = [col for col in feature_cols if col in analysis_df.columns]
    data = analysis_df[existing_feature_cols + ['VIX']].copy() # Work on a copy

    # IMPORTANT: Handle NaNs. XGBoost cannot handle NaNs.
    # Although data_processing attempts to fill NaNs, re-check here.
    # Option 1: Drop rows with NaNs in feature/target columns (simplest, but loses data)
    # data.dropna(subset=existing_feature_cols + ['VIX'], inplace=True)
    # Option 2: Impute NaNs (e.g., with mean, median, or forward fill)
    # Using ffill then bfill as a robust imputation strategy
    initial_nan_count = data.isna().sum().sum()
    if initial_nan_count > 0:
        logger.warning(f"NaNs found in data before XGBoost processing ({initial_nan_count}). Attempting imputation.")
        # Impute NaNs in the selected feature and target columns
        data[existing_feature_cols + ['VIX']] = data[existing_feature_cols + ['VIX']].fillna(method='ffill').fillna(method='bfill')
        # After ffill/bfill, if NaNs still exist (e.g., at the very beginning), fill them with 0 or mean.
        data[existing_feature_cols + ['VIX']] = data[existing_feature_cols + ['VIX']].fillna(0) # Fill any remaining with 0

        final_nan_count = data.isna().sum().sum()
        if final_nan_count > 0:
             logger.error(f"FATAL ERROR: NaNs still present in data after imputation for XGBoost: {final_nan_count}. Cannot proceed.")
             return None, None, None
        else:
             logger.info("NaNs successfully imputed for XGBoost processing.")


    if data.empty:
        logger.error("Data is empty after handling NaNs. Cannot train XGBoost model.")
        return None, None, None

    # Define features (X) and target (y)
    X = data[existing_feature_cols]
    y = data['VIX'] # Target variable is VIX

    # --- 2. Prepare Data for Training and Evaluation ---
    # Use a time-series split approach for evaluation (train on earlier data, test on later)
    # Use the last portion of the data as a test set to evaluate performance on recent history.
    # Define the size of the test set (e.g., last 10% or a fixed number of days)
    test_size = max(1, int(len(data) * 0.1)) # Use at least 1 day for test, or 10% of data
    if len(data) < 2 * test_size: # Ensure there's enough data for both train and test sets
        logger.warning(f"Data length ({len(data)}) is too short for train/test split with test size {test_size}. Using all data for training and skipping evaluation.")
        X_train, y_train = X, y
        X_test, y_test = pd.DataFrame(), pd.Series() # Empty test set
    else:
        X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
        y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
        logger.info(f"Data split into training ({len(X_train)} samples) and test ({len(X_test)} samples) sets.")

    if X_train.empty or y_train.empty:
        logger.error("Training data is empty after split. Cannot train XGBoost model.")
        return None, None, None


    # --- 3. Train the XGBoost Model ---
    logger.info("Training XGBoost Regressor model.")
    # Define XGBoost model parameters (example parameters, tuning might be needed)
    params = {
        'objective': 'reg:squarederror', # Regression task
        'n_estimators': 100,             # Number of boosting rounds
        'learning_rate': 0.1,            # Step size shrinkage
        'max_depth': 3,                  # Maximum depth of trees
        'subsample': 0.8,                # Fraction of samples used for training each tree
        'colsample_bytree': 0.8,         # Fraction of features used for training each tree
        'random_state': 42               # For reproducibility
    }

    model = xgb.XGBRegressor(**params)

    try:
        model.fit(X_train, y_train)
        logger.info("XGBoost model training completed.")
    except Exception as e:
        logger.error(f"Error during XGBoost model training: {str(e)}", exc_info=True)
        return None, None, None


    # --- 4. Evaluate Model (on Historical Test Set) ---
    rmse = None
    if not X_test.empty and not y_test.empty:
        logger.info("Evaluating XGBoost model on historical test set.")
        try:
            y_pred_test = model.predict(X_test)
            # Calculate Root Mean Squared Error
            rmse = mean_squared_error(y_test, y_pred_test, squared=False) # squared=False for RMSE
            logger.info(f"XGBoost Test RMSE: {rmse:.2f}")
        except Exception as e:
             logger.warning(f"Error during XGBoost model evaluation (RMSE calculation): {str(e)}. RMSE will be None.", exc_info=True)
             rmse = None # Ensure RMSE is None if calculation fails

    else:
        logger.warning("Historical test set is empty. Skipping RMSE calculation.")


    # --- 5. Predict Future Volatility ---
    # To predict future volatility, we need feature values for those future dates.
    # Since future feature values are unknown, we use the latest available feature values
    # (from the last row of the input analysis_df) as the input for predicting
    # volatility for each step in the forecast horizon. This is a simplification.

    # Get the feature values from the very last row of the *original, full* analysis_df
    # This row contains the latest real-time or imputed data.
    # Ensure the last row exists
    if analysis_df.empty:
        logger.error("Original analysis_df is empty. Cannot get latest features for prediction.")
        return None, rmse, None

    latest_features = analysis_df[existing_feature_cols].iloc[-1]

    # Repeat the latest feature vector for the number of days in the forecast horizon.
    # This assumes the relationships learned by the model based on historical features
    # can predict future volatility levels given the *current* market state (represented by latest_features).
    # This does NOT account for how features themselves might change over the horizon.
    future_X = pd.DataFrame([latest_features] * forecast_horizon, columns=existing_feature_cols)

    xgb_forecast_values = None
    if not future_X.empty:
        logger.info(f"Predicting volatility for the next {forecast_horizon} days using latest features.")
        try:
            # Predict future volatility values using the trained model and future_X features
            xgb_forecast_values = model.predict(future_X)

            # Ensure forecast values are non-negative
            xgb_forecast_values = np.maximum(0, xgb_forecast_values)

            logger.info(f"XGBoost forecast generated: {len(xgb_forecast_values)} values.")
            logger.debug(f"XGBoost forecast values (first 5): {xgb_forecast_values[:5]}")

        except Exception as e:
            logger.error(f"Error during XGBoost prediction: {str(e)}", exc_info=True)
            xgb_forecast_values = None # Ensure None if prediction fails

    else:
        logger.error("Future features DataFrame is empty. Cannot run XGBoost prediction.")
        xgb_forecast_values = None


    # --- 6. Extract Feature Importances ---
    feature_importances = None
    try:
        # Get feature importances from the trained model
        # Need to match importances to feature names
        if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
             feature_importances_values = model.feature_importances_
             # Ensure the number of importance values matches the number of features
             if len(feature_importances_values) == len(existing_feature_cols):
                 # Create a list of (feature_name, importance_value) tuples
                 feature_importances = list(zip(existing_feature_cols, feature_importances_values))
                 # Sort by importance descending
                 feature_importances = sorted(feature_importances, key=lambda item: item[1], reverse=True)
                 logger.debug("Feature importances extracted and sorted.")
             else:
                 logger.warning(f"Mismatch between number of features ({len(existing_feature_cols)}) and feature importances ({len(feature_importances_values)}). Cannot provide feature importances list.")
                 feature_importances = None # Set to None if mismatch occurs

        else:
            logger.warning("XGBoost model does not have 'feature_importances_' attribute or it is None.")
            feature_importances = None # Set to None if attribute is missing

    except Exception as e:
        logger.warning(f"Error extracting feature importances from XGBoost model: {str(e)}. Feature importances will be None.", exc_info=True)
        feature_importances = None # Ensure None if extraction fails


    # --- 7. Return Results ---
    logger.info("XGBoost volatility forecasting process completed.")
    # Return the forecast values, calculated RMSE, and feature importances list
    return xgb_forecast_values, rmse, feature_importances

# The __main__ block is removed as this module's functions are intended to be imported
# and used by other parts of the application.
# To test this function, you would typically call it from a separate script
# with a dummy analysis_df DataFrame containing the necessary columns and values.
