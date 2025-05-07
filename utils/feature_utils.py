import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@st.cache_data
def generate_features(df, real_data, capital):
    """Generates features for the dataset."""
    try:
        logger.info("Generating features")
        df = df.copy()
        df.index = df.index.normalize()
        n_days = len(df)

        # Use real-time data if available
        base_pcr = real_data["pcr"] if real_data and real_data.get("pcr") is not None else df["PCR"].iloc[-1] if "PCR" in df.columns and len(df) > 1 else 1.0
        base_straddle_price = real_data["straddle_price"] if real_data and real_data.get("straddle_price") is not None else df["Straddle_Price"].iloc[-1] if "Straddle_Price" in df.columns and len(df) > 1 else 200.0
        base_max_pain_diff_pct = real_data["max_pain_diff_pct"] if real_data and real_data.get("max_pain_diff_pct") is not None else df["Spot_MaxPain_Diff_Pct"].iloc[-1] if "Spot_MaxPain_Diff_Pct" in df.columns and len(df) > 1 else 0.5
        base_vix_change_pct = real_data["vix_change_pct"] if real_data and real_data.get("vix_change_pct") is not None else df["VIX_Change_Pct"].iloc[-1] if "VIX_Change_Pct" in df.columns and len(df) > 1 else 0.0

        def calculate_days_to_expiry(dates):
            days_to_expiry = []
            fetched_expiry = datetime.strptime(real_data["expiry"], "%Y-%m-%d").date() if real_data and real_data.get("expiry") and real_data["expiry"] != "N/A" else None
            for date in dates:
                date_only = date.date()
                if fetched_expiry and date_only >= fetched_expiry:
                    dte = 0
                elif fetched_expiry and date_only < fetched_expiry:
                    dte = (fetched_expiry - date_only).days
                else:
                    days_ahead = (3 - date_only.weekday()) % 7
                    if days_ahead == 0:
                        days_ahead = 7
                    next_expiry_approx = date_only + timedelta(days=days_ahead)
                    dte = (next_expiry_approx - date_only).days
                days_to_expiry.append(dte)
            return np.array(days_to_expiry)

        df["Days_to_Expiry"] = calculate_days_to_expiry(df.index)
        df["Days_to_Expiry"] = np.clip(df["Days_to_Expiry"], 0, None)

        df["ATM_IV"] = df["VIX"]
        if real_data and real_data.get("vix") is not None:
            df["ATM_IV"].iloc[-1] = real_data["vix"]

        if real_data and real_data.get("expiry") and real_data["expiry"] != "N/A":
            fetched_expiry_dt = datetime.strptime(real_data["expiry"], "%Y-%m-%d").date()
            df["Event_Flag"] = np.where(
                (df.index.date == fetched_expiry_dt) | (df["Days_to_Expiry"] <= 3), 1, 0
            )
        else:
            df["Event_Flag"] = np.where((df.index.weekday == 3) | (df["Days_to_Expiry"] <= 3), 1, 0)

        def dynamic_ivp(x):
            if len(x) >= 5 and x.iloc[-1] is not None and not pd.isna(x.iloc[-1]):
                historical_values = x.iloc[:-1].dropna()
                if not historical_values.empty:
                    return (np.sum(historical_values <= x.iloc[-1]) / len(historical_values)) * 100
            return 50.0

        df["IVP"] = df["ATM_IV"].rolling(252, min_periods=5).apply(dynamic_ivp, raw=False)
        df["IVP"] = df["IVP"].interpolate(method='linear').fillna(50.0)

        market_trend = df["NIFTY_Close"].pct_change().rolling(5).mean().fillna(0)
        df["PCR"] = np.clip(base_pcr + np.random.normal(0, 0.05, n_days) + market_trend * -5, 0.7, 2.0)
        if real_data and real_data.get("pcr") is not None:
            df["PCR"].iloc[-1] = base_pcr

        df["VIX_Change_Pct"] = df["VIX"].pct_change().fillna(0) * 100
        if real_data and real_data.get("vix_change_pct") is not None:
            df["VIX_Change_Pct"].iloc[-1] = base_vix_change_pct

        df["Spot_MaxPain_Diff_Pct"] = np.clip(base_max_pain_diff_pct + np.random.normal(0, 0.1, n_days) + df["Days_to_Expiry"]*0.01, 0.1, 5.0)
        if real_data and real_data.get("max_pain_diff_pct") is not None:
            df["Spot_MaxPain_Diff_Pct"].iloc[-1] = base_max_pain_diff_pct

        fii_trend = np.random.normal(0, 5000, n_days)
        fii_trend[::10] *= -1.5
        df["FII_Index_Fut_Pos"] = np.cumsum(fii_trend).astype(int)
        df["FII_Option_Pos"] = np.cumsum(np.random.normal(0, 2000, n_days)).astype(int)
        df["IV_Skew"] = np.clip(np.random.normal(0, 0.5, n_days) + (df["VIX"] / 20 - 1) * 2 + (df["Days_to_Expiry"] / 15 - 1)*0.5, -3, 3)
        df["Realized_Vol"] = df["NIFTY_Close"].pct_change().rolling(5, min_periods=5).std() * np.sqrt(252) * 100
        df["Realized_Vol"] = df["Realized_Vol"].fillna(df["VIX"]).fillna(15.0)
        df["Advance_Decline_Ratio"] = np.clip(1.0 + np.random.normal(0, 0.1, n_days) + market_trend * 5, 0.7, 1.5)
        df["Capital_Pressure_Index"] = (df["FII_Index_Fut_Pos"] / 5e4 + df["FII_Option_Pos"] / 2e4 + df["PCR"]-1) / 3
        df["Capital_Pressure_Index"] = np.clip(df["Capital_Pressure_Index"], -1.5, 1.5)
        df["Gamma_Bias"] = np.clip(df["IV_Skew"] * (30 - np.clip(df["Days_to_Expiry"], 1, 30)) / 30, -2, 2)
        df["Total_Capital"] = capital
        df["PnL_Day"] = np.random.normal(0, capital * 0.005, n_days) * (1 - df["Event_Flag"] * 0.2)
        df["Straddle_Price"] = np.clip(base_straddle_price + np.random.normal(0, base_straddle_price*0.1, n_days), base_straddle_price*0.5, base_straddle_price*1.5)
        if real_data and real_data.get("straddle_price") is not None:
            df["Straddle_Price"].iloc[-1] = base_straddle_price

        if df.isna().sum().sum() > 0:
            logger.warning(f"NaNs found: {df.isna().sum().sum()}")
            df = df.apply(lambda x: x.interpolate(method='linear')).fillna(method='bfill').fillna(method='ffill')
            if df.isna().sum().sum() > 0:
                logger.error(f"NaNs still present: {df.isna().sum().sum()}")
                st.error("Error: Could not fill all missing data points.")

        try:
            df.to_csv("volguard_hybrid_data.csv", index=True)
            logger.debug("Saved volguard_hybrid_data.csv")
        except PermissionError:
            logger.error("Permission denied: volguard_hybrid_data.csv")
            st.error("Cannot save volguard_hybrid_data.csv: Permission denied")
        except Exception as e:
            logger.error(f"Error saving CSV: {e}")
            st.error(f"Error saving data file: {e}")

        logger.debug("Features generated successfully")
        return df
    except Exception as e:
        st.error(f"Error generating features: {str(e)}")
        logger.error(f"Error generating features: {str(e)}")
        return None
