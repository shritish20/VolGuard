import logging
import io
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from upstox_api import fetch_real_time_market_data  # Import from Upstox API module

# Setup logging
logger = logging.getLogger(__name__)

# Define feature columns used in modeling
FEATURE_COLS = [
    'VIX', 'ATM_IV', 'IVP', 'PCR', 'VIX_Change_Pct', 'IV_Skew', 'Straddle_Price',
    'Spot_MaxPain_Diff_Pct', 'Days_to_Expiry', 'Event_Flag', 'FII_Index_Fut_Pos',
    'FII_Option_Pos'
]

# Data Loading (API First, then CSV Fallback)
def load_data(upstox_client):
    """
    Attempts to load data from Upstox API first, falls back to CSV if API fails.
    Returns DataFrame for analysis, real-time data dict, and data source tag.
    """
    df = None
    real_data = None
    data_source = "CSV (FALLBACK)"  # Default source

    # Attempt to fetch real-time data from API
    real_data = fetch_real_time_market_data(upstox_client)

    if real_data and real_data.get("nifty_spot") is not None and real_data.get("vix") is not None:
        logger.info("Successfully fetched real-time data from Upstox API.")
        data_source = real_data["source"]

        latest_date = datetime.now().date()
        live_df_row = pd.DataFrame({
            "NIFTY_Close": [real_data["nifty_spot"]],
            "VIX": [real_data["vix"]]
        }, index=[pd.to_datetime(latest_date).normalize()])

        # Load historical CSV data
        try:
            nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
            vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"

            nifty = pd.read_csv(io.StringIO(requests.get(nifty_url).text), encoding="utf-8-sig")
            vix = pd.read_csv(io.StringIO(requests.get(vix_url).text))

            nifty.columns = nifty.columns.str.strip()
            vix.columns = vix.columns.str.strip()

            nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
            vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")

            nifty = nifty.dropna(subset=["Date"]).set_index("Date")[["Close"]].rename(columns={"Close": "NIFTY_Close"})
            vix = vix.dropna(subset=["Date"]).set_index("Date")[["Close"]].rename(columns={"Close": "VIX"})

            nifty.index = nifty.index.normalize()
            vix.index = vix.index.normalize()
            nifty = nifty.groupby(nifty.index).last()
            vix = vix.groupby(vix.index).last()

            common_dates = nifty.index.intersection(vix.index)
            historical_df = pd.DataFrame({
                "NIFTY_Close": nifty["NIFTY_Close"].loc[common_dates],
                "VIX": vix["VIX"].loc[common_dates]
            }, index=common_dates).dropna()

            historical_df = historical_df[historical_df.index < live_df_row.index[0]]

            df = pd.concat([historical_df, live_df_row])
            df = df.groupby(df.index).last()
            df = df.sort_index()
            df = df.ffill().bfill()

            logger.debug(f"Combined historical and live data. Shape: {df.shape}")

        except Exception as e:
            logger.error(f"Error loading historical CSV data while having live data: {str(e)}")
            df = live_df_row
            logger.warning("Proceeding with only live data point due to CSV error.")

    else:
        logger.warning("Failed to fetch real-time data from Upstox API. Falling back to CSV.")
        real_data = None

        try:
            nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
            vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"

            nifty = pd.read_csv(io.StringIO(requests.get(nifty_url).text), encoding="utf-8-sig")
            vix = pd.read_csv(io.StringIO(requests.get(vix_url).text))

            nifty.columns = nifty.columns.str.strip()
            vix.columns = vix.columns.str.strip()

            nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
            vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")

            nifty = nifty.dropna(subset=["Date"]).set_index("Date")[["Close"]].rename(columns={"Close": "NIFTY_Close"})
            vix = vix.dropna(subset=["Date"]).set_index("Date")[["Close"]].rename(columns={"Close": "VIX"})

            nifty.index = nifty.index.normalize()
            vix.index = vix.index.normalize()
            nifty = nifty.groupby(nifty.index).last()
            vix = vix.groupby(vix.index).last()

            common_dates = nifty.index.intersection(vix.index)
            df = pd.DataFrame({
                "NIFTY_Close": nifty["NIFTY_Close"].loc[common_dates],
                "VIX": vix["VIX"].loc[common_dates]
            }, index=common_dates).dropna()

            df = df.groupby(df.index).last()
            df = df.sort_index()
            df = df.ffill().bfill()

            logger.debug(f"Loaded data from CSV fallback. Shape: {df.shape}")

        except Exception as e:
            logger.error(f"Fatal error loading data from CSV fallback: {str(e)}")
            return None, None, "Data Load Failed"

    if df is None or len(df) < 2:
        logger.error("Insufficient data loaded for analysis.")
        return None, None, data_source

    logger.debug(f"Data loading successful. Final DataFrame shape: {df.shape}. Source: {data_source}")
    return df, real_data, data_source

# Feature Generation
def generate_features(df, real_data, capital):
    try:
        logger.info("Generating features")
        df = df.copy()
        df.index = df.index.normalize()
        n_days = len(df)

        base_pcr = real_data["pcr"] if real_data and real_data.get("pcr") is not None and not np.isnan(real_data["pcr"]) else df["PCR"].iloc[-1] if "PCR" in df.columns and len(df) > 1 and not pd.isna(df["PCR"].iloc[-1]) else 1.0
        base_straddle_price = real_data["straddle_price"] if real_data and real_data.get("straddle_price") is not None and not np.isnan(real_data["straddle_price"]) else df["Straddle_Price"].iloc[-1] if "Straddle_Price" in df.columns and len(df) > 1 and not pd.isna(df["Straddle_Price"].iloc[-1]) else 200.0
        base_max_pain_diff_pct = real_data["max_pain_diff_pct"] if real_data and real_data.get("max_pain_diff_pct") is not None and not np.isnan(real_data["max_pain_diff_pct"]) else df["Spot_MaxPain_Diff_Pct"].iloc[-1] if "Spot_MaxPain_Diff_Pct" in df.columns and len(df) > 1 and not pd.isna(df["Spot_MaxPain_Diff_Pct"].iloc[-1]) else 0.5
        base_vix_change_pct = real_data["vix_change_pct"] if real_data and real_data.get("vix_change_pct") is not None and not np.isnan(real_data["vix_change_pct"]) else df["VIX_Change_Pct"].iloc[-1] if "VIX_Change_Pct" in df.columns and len(df) > 1 and not pd.isna(df["VIX_Change_Pct"].iloc[-1]) else 0.0

        def calculate_days_to_expiry(dates):
            days_to_expiry = []
            fetched_expiry = None
            if real_data and real_data.get("expiry") and real_data["expiry"] != "N/A":
                try:
                    fetched_expiry = datetime.strptime(real_data["expiry"], "%Y-%m-%d").date()
                except ValueError:
                    logger.warning(f"Could not parse fetched expiry date string: {real_data['expiry']}")
                    fetched_expiry = None

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
        if real_data and real_data.get("vix") is not None and not np.isnan(real_data["vix"]):
            df.loc[df.index[-1], "ATM_IV"] = real_data["vix"]

        if real_data and real_data.get("expiry") and real_data["expiry"] != "N/A":
            try:
                fetched_expiry_dt = datetime.strptime(real_data["expiry"], "%Y-%m-%d").date()
                df["Event_Flag"] = np.where(
                    (df.index.date == fetched_expiry_dt) |
                    (df["Days_to_Expiry"] <= 3),
                    1, 0
                )
            except ValueError:
                logger.warning(f"Could not parse fetched expiry date string for Event Flag: {real_data['expiry']}")
                df["Event_Flag"] = np.where(
                    (df.index.weekday == 3) |
                    (df["Days_to_Expiry"] <= 3),
                    1, 0
                )
        else:
            df["Event_Flag"] = np.where(
                (df.index.weekday == 3) |
                (df["Days_to_Expiry"] <= 3),
                1, 0
            )

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
        df.loc[df.index[-1], "PCR"] = base_pcr

        df["VIX_Change_Pct"] = df["VIX"].pct_change().fillna(0) * 100
        df.loc[df.index[-1], "VIX_Change_Pct"] = base_vix_change_pct

        df["Spot_MaxPain_Diff_Pct"] = np.clip(base_max_pain_diff_pct + np.random.normal(0, 0.1, n_days) + df["Days_to_Expiry"]*0.01, 0.1, 5.0)
        df.loc[df.index[-1], "Spot_MaxPain_Diff_Pct"] = base_max_pain_diff_pct

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
        df.loc[df.index[-1], "Straddle_Price"] = base_straddle_price

        if df.isna().sum().sum() > 0:
            logger.warning(f"NaNs found after initial feature generation: {df.isna().sum().sum()}")
            df = df.apply(lambda x: x.interpolate(method='linear')).fillna(method='bfill').fillna(method='ffill')
            if df.isna().sum().sum() > 0:
                logger.error(f"NaNs still present after interpolation/fill: {df.isna().sum().sum()}")

        if not all(col in df.columns for col in FEATURE_COLS):
            missing = [col for col in FEATURE_COLS if col not in df.columns]
            logger.error(f"FATAL ERROR: Missing required FEATURE_COLS after generation: {missing}")
            return None

        logger.debug("Features generated successfully")
        return df
    except Exception as e:
        logger.error(f"Error generating features: {str(e)}")
        return None
