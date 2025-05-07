import logging
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from data_processing import FEATURE_COLS # Import the feature columns definition

# Setup logging
logger = logging.getLogger(__name__)

def forecast_volatility_xgboost(df_xgb, forecast_horizon):
    """
    Forecasts volatility using an XGBoost Regressor.
    Takes a DataFrame with features and 'Realized_Vol' and returns XGBoost volatility forecasts.
    """
    try:
        logger.info("Forecasting volatility using XGBoost")

        # Ensure target variable exists and drop NaNs caused by shift
        if "Realized_Vol" not in df_xgb.columns:
             logger.error("Realized_Vol feature is missing for XGBoost target.")
             # Return default values or handle appropriately in the caller
             return np.full(forecast_horizon, df_xgb["VIX"].iloc[-1] if "VIX" in df_xgb.columns and len(df_xgb) > 0 else 15.0), 0.0, None

        df_xgb['Target_Vol'] = df_xgb['Realized_Vol'].shift(-1)
        df_xgb = df_xgb.dropna(subset=['Target_Vol'] + FEATURE_COLS) # Drop NaNs in target or features

        if len(df_xgb) < 50: # Minimum data for XGBoost training
            logger.warning(f"Insufficient data ({len(df_xgb)} rows) for XGBoost training after dropping NaNs.")
            # Return default values or handle appropriately in the caller
            return np.full(forecast_horizon, df_xgb["VIX"].iloc[-1] if "VIX" in df_xgb.columns and len(df_xgb) > 0 else 15.0), 0.0, None


        X = df_xgb[FEATURE_COLS]
        y = df_xgb['Target_Vol']

        # Ensure split_index is valid
        split_index = int(len(X) * 0.8)
        if split_index < 1 or split_index >= len(X):
             split_index = max(1, len(X) - 50) # Ensure some test data, at least 50 points

        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        # Ensure there's data in train and test sets
        if X_train.empty or X_test.empty:
             logger.warning("Training or testing data is empty after split.")
             return np.full(forecast_horizon, df_xgb["VIX"].iloc[-1] if "VIX" in df_xgb.columns and len(df_xgb) > 0 else 15.0), 0.0, None


        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) # Use transform on test set

        model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.03, random_state=42)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        logger.debug(f"XGBoost training completed. RMSE: {rmse:.2f}")


        # Forecast using XGBoost
        xgb_vols = []
        # Start forecasting from the last known feature state
        if df_xgb.empty:
            logger.warning("DataFrame is empty for XGBoost forecasting.")
            return np.full(forecast_horizon, df_xgb["VIX"].iloc[-1] if "VIX" in df_xgb.columns and len(df_xgb) > 0 else 15.0), rmse, model.feature_importances_

        current_row = df_xgb[FEATURE_COLS].iloc[-1].copy()

        for i in range(forecast_horizon):
            # Prepare the current feature row for prediction
            current_row_df = pd.DataFrame([current_row], columns=FEATURE_COLS)
            current_row_scaled = scaler.transform(current_row_df)

            # Predict the next volatility
            next_vol = model.predict(current_row_scaled)[0]
            xgb_vols.append(next_vol)

            # Simulate feature changes for the next day's prediction
            # These simulations should ideally be more sophisticated (e.g., based on predicted price move)
            # For now, keep the current simulation logic as it was, but apply bounds and handle potential NaNs
            current_row["Days_to_Expiry"] = max(1, current_row["Days_to_Expiry"] - 1)
            current_row["VIX"] = np.clip(current_row["VIX"] * np.random.uniform(0.98, 1.02), 5, 50) # Add bounds
            current_row["Straddle_Price"] = np.clip(current_row["Straddle_Price"] * np.random.uniform(0.98, 1.02), 50, 400) # Add bounds
            # VIX_Change_Pct is a daily change, recalculate based on simulated VIX change
            prev_vix = df_xgb["VIX"].iloc[-1] if len(df_xgb)>0 else current_row["VIX"]
            current_row["VIX_Change_Pct"] = ((current_row["VIX"] / (prev_vix if prev_vix > 0 else current_row["VIX"])) - 1) * 100 if prev_vix > 0 else 0

            current_row["ATM_IV"] = current_row["VIX"] * (1 + np.random.normal(0, 0.05)) # Reduced noise
            current_row["Realized_Vol"] = np.clip(next_vol * np.random.uniform(0.98, 1.02), 5, 50) # Use predicted vol with noise
            current_row["IVP"] = np.clip(current_row["IVP"] * np.random.uniform(0.99, 1.01), 0, 100) # Add bounds
            current_row["PCR"] = np.clip(current_row["PCR"] + np.random.normal(0, 0.02), 0.7, 2.0) # Reduced noise
            current_row["Spot_MaxPain_Diff_Pct"] = np.clip(current_row["Spot_MaxPain_Diff_Pct"] * np.random.uniform(0.98, 1.02), 0.1, 5.0) # Add bounds
            # Event Flag simulation could be improved - maybe base on remaining DTE to next known expiry
            current_row["Event_Flag"] = 1 if current_row["Days_to_Expiry"] <= 3 else 0 # Simple rule based on DTE
            current_row["FII_Index_Fut_Pos"] += np.random.normal(0, 500) # Reduced noise
            current_row["FII_Option_Pos"] += np.random.normal(0, 200) # Reduced noise
            current_row["IV_Skew"] = np.clip(current_row["IV_Skew"] + np.random.normal(0, 0.05), -3, 3) # Reduced noise

            # Ensure no NaNs creep in during simulation - use backward/forward fill as a fallback
            current_row = current_row.fillna(method='bfill').fillna(method='ffill')


        xgb_vols = np.clip(xgb_vols, 5, 50)
        # Apply event spike to XGBoost forecast if the last known day was an event day
        if df_xgb["Event_Flag"].iloc[-1] == 1:
            xgb_vols = [v * 1.05 for v in xgb_vols] # Reduced spike effect


        logger.debug("XGBoost forecast completed.")
        return xgb_vols, rmse, model.feature_importances_

    except Exception as e:
        logger.error(f"Error in XGBoost volatility forecasting: {str(e)}", exc_info=True)
        # Return default values or handle appropriately in the caller
        return np.full(forecast_horizon, df_xgb["VIX"].iloc[-1] if "VIX" in df_xgb.columns and len(df_xgb) > 0 else 15.0), 0.0, None


