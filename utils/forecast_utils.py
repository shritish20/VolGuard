import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import logging
from config.settings import FEATURE_COLS

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@st.cache_data
def forecast_volatility_future(df, forecast_horizon):
    """Forecasts future volatility using GARCH and XGBoost."""
    try:
        logger.info("Forecasting volatility")
        df = df.copy()
        df.index = pd.to_datetime(df.index).normalize()

        if len(df) < 200 or not all(col in df.columns for col in FEATURE_COLS + ['NIFTY_Close', 'Realized_Vol']):
            st.error("Insufficient data or missing columns for forecasting.")
            logger.error(f"Insufficient data ({len(df)} days) or missing columns.")
            return None, None, None, None, None, None, None, None

        df_garch = df.tail(len(df))
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon, freq='B')

        df_garch['Log_Returns'] = np.log(df_garch['NIFTY_Close'].pct_change() + 1).dropna() * 100
        if df_garch['Log_Returns'].empty:
            st.error("Insufficient returns data for GARCH.")
            return None, None, None, None, None, None, None, None

        if len(df_garch['Log_Returns']) >= 100:
            garch_model = arch_model(df_garch['Log_Returns'], vol='Garch', p=1, q=1, rescale=False)
            garch_fit = garch_model.fit(disp="off")
            garch_forecast = garch_fit.forecast(horizon=forecast_horizon, reindex=False)
            garch_vols = np.sqrt(garch_forecast.variance.iloc[-1].values) * np.sqrt(252)
            garch_vols = np.clip(garch_vols, 5, 50)
            logger.debug("GARCH forecast completed.")
        else:
            logger.warning(f"Insufficient data ({len(df_garch['Log_Returns'])} returns) for GARCH.")
            garch_vols = np.full(forecast_horizon, df["VIX"].iloc[-1] if "VIX" in df.columns else 15.0)

        realized_vol = df["Realized_Vol"].dropna().iloc[-5:].mean() if not df["Realized_Vol"].dropna().empty else df["VIX"].iloc[-1] if "VIX" in df.columns else 15.0
        df_xgb = df.tail(len(df))
        df_xgb['Target_Vol'] = df_xgb['Realized_Vol'].shift(-1)
        df_xgb = df_xgb.dropna(subset=['Target_Vol'] + FEATURE_COLS)

        if len(df_xgb) < 50:
            st.error(f"Insufficient data ({len(df_xgb)} rows) for XGBoost.")
            return None, None, None, None, None, None, None, None

        X = df_xgb[FEATURE_COLS]
        y = df_xgb['Target_Vol']
        split_index = int(len(X) * 0.8) if int(len(X) * 0.8) >= 1 else max(1, len(X) - 50)
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.03, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        logger.debug(f"XGBoost training completed. RMSE: {rmse:.2f}")

        xgb_vols = []
        current_row = df_xgb[FEATURE_COLS].iloc[-1].copy()
        for _ in range(forecast_horizon):
            current_row_df = pd.DataFrame([current_row], columns=FEATURE_COLS)
            current_row_scaled = scaler.transform(current_row_df)
            next_vol = model.predict(current_row_scaled)[0]
            xgb_vols.append(next_vol)

            current_row["Days_to_Expiry"] = max(1, current_row["Days_to_Expiry"] - 1)
            current_row["VIX"] = np.clip(current_row["VIX"] * np.random.uniform(0.98, 1.02), 5, 50)
            current_row["Straddle_Price"] = np.clip(current_row["Straddle_Price"] * np.random.uniform(0.98, 1.02), 50, 400)
            current_row["VIX_Change_Pct"] = ((current_row["VIX"] / (df_xgb["VIX"].iloc[-1] if len(df_xgb)>0 else current_row["VIX"])) - 1) * 100
            current_row["ATM_IV"] = current_row["VIX"] * (1 + np.random.normal(0, 0.05))
            current_row["Realized_Vol"] = np.clip(next_vol * np.random.uniform(0.98, 1.02), 5, 50)
            current_row["IVP"] = np.clip(current_row["IVP"] * np.random.uniform(0.99, 1.01), 0, 100)
            current_row["PCR"] = np.clip(current_row["PCR"] + np.random.normal(0, 0.02), 0.7, 2.0)
            current_row["Spot_MaxPain_Diff_Pct"] = np.clip(current_row["Spot_MaxPain_Diff_Pct"] * np.random.uniform(0. nisso, 1.02), 0.1, 5.0)
            current_row["Event_Flag"] = 1 if current_row["Days_to_Expiry"] <= 3 else 0
            current_row["FII_Index_Fut_Pos"] += np.random.normal(0, 500)
            current_row["FII_Option_Pos"] += np.random.normal(0, 200)
vii
            current_row["IV_Skew"] = np.clip(current_row["IV_Skew"] + np.random.normal(0, 0.05), -3, 3)
            current_row = current_row.fillna(method='bfill').fillna(method='ffill')

        xgb_vols = np.clip(xgb_vols, 5, 50)
        if df["Event_Flag"].iloc[-1] == 1:
            xgb_vols = [v * 1.05 for v in xgb_vols]

        if len(garch_vols) != len(xgb_vols):
            logger.error("GARCH and XGBoost forecast horizons mismatch.")
            blended_vols = xgb_vols
            confidence_score = 50
        else:
            initial_garch_vol = garch_vols[0] if len(garch_vols) > 0 else realized_vol
            initial_xgb_vol = xgb_vols[0] if len(xgb_vols) > 0 else realized_vol
            garch_diff = np.abs(initial_garch_vol - realized_vol)
            xgb_diff = np.abs(initial_xgb_vol - realized_vol)
            total_diff = garch_diff + xgb_diff
            garch_weight = xgb_diff / total_diff if total_diff > 0 else 0.5
            xgb_weight = 1 - garch_weight
            blended_vols = [(garch_weight * g) + (xgb_weight * x) for g, x in zip(garch_vols, xgb_vols)]
            model_agreement = 1 - (np.abs(initial_garch_vol - initial_xgb_vol) / max(initial_garch_vol, initial_xgb_vol) if max(initial_garch_vol, initial_xgb_vol) > 0 else 0)
            deviation_from_realized = 1 - (min(garch_diff, xgb_diff) / realized_vol if realized_vol > 0 else 0)
            confidence_score = min(100, max(30, (model_agreement * 0.6 + deviation_from_realized * 0.4) * 100))

        forecast_log = pd.DataFrame({
            "Date": future_dates,
            "GARCH_Vol": garch_vols,
            "XGBoost_Vol": xgb_vols,
            "Blended_Vol": blended_vols,
            "Confidence": [confidence_score] * forecast_horizon
        })

        logger.debug("Volatility forecast completed")
        return forecast_log, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, model.feature_importances_
    except Exception as e:
        st.error(f"Error in volatility forecasting: {str(e)}")
        logger.error(f"Error in volatility forecasting: {str(e)}")
        return None, None, None, None, None, None, None, None
