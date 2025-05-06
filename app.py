import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime, timedelta
import logging
from py5paisa import FivePaisaClient
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os
import plotly.express as px
import plotly.graph_objects as go

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(page_title="VolGuard: Options Trading", page_icon="📈", layout="wide")

# Custom CSS for modern, clean UI
st.markdown("""
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #f5f7fa; }
        .stApp { background: #ffffff; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .sidebar .stButton>button { background: #007bff; color: white; border-radius: 8px; padding: 10px; }
        .sidebar .stButton>button:hover { background: #0056b3; }
        .section { background: #f8f9fa; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
        .section-header { color: #333; font-size: 24px; font-weight: 600; margin-bottom: 10px; }
        .metric-card { background: #ffffff; border-radius: 8px; padding: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .trade-modal { background: #ffffff; border-radius: 8px; padding: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.2); }
        .stButton>button { background: #28a745; color: white; border-radius: 8px; padding: 10px 20px; }
        .stButton>button:hover { background: #218838; }
        .alert { background: #dc3545; color: white; padding: 10px; border-radius: 8px; }
        .success { background: #28a745; color: white; padding: 10px; border-radius: 8px; }
        .footer { text-align: center; color: #666; font-size: 12px; padding: 20px 0; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "client" not in st.session_state:
    st.session_state.client = None
if "trades" not in st.session_state:
    st.session_state.trades = []
if "journal_complete" not in st.session_state:
    st.session_state.journal_complete = False
if "violations" not in st.session_state:
    st.session_state.violations = 0
if "backtest_run" not in st.session_state:
    st.session_state.backtest_run = False
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None

# 5paisa Client Initialization
def initialize_5paisa_client(totp_code):
    try:
        cred = {
            "APP_NAME": st.secrets["fivepaisa"].get("APP_NAME", ""),
            "APP_SOURCE": st.secrets["fivepaisa"].get("APP_SOURCE", ""),
            "USER_ID": st.secrets["fivepaisa"].get("USER_ID", ""),
            "PASSWORD": st.secrets["fivepaisa"].get("PASSWORD", ""),
            "USER_KEY": st.secrets["fivepaisa"].get("USER_KEY", ""),
            "ENCRYPTION_KEY": st.secrets["fivepaisa"].get("ENCRYPTION_KEY", "")
        }
        client = FivePaisaClient(cred=cred)
        client.get_totp_session(
            st.secrets["fivepaisa"].get("CLIENT_CODE", ""),
            totp_code,
            st.secrets["fivepaisa"].get("PIN", "")
        )
        if client.get_access_token():
            logger.info("5paisa client initialized successfully")
            return client
        return None
    except Exception as e:
        logger.error(f"Error initializing 5paisa client: {str(e)}")
        return None

# Data Fetching
def max_pain(df, nifty_spot):
    try:
        calls = df[df["CPType"] == "CE"].set_index("StrikeRate")["OpenInterest"]
        puts = df[df["CPType"] == "PE"].set_index("StrikeRate")["OpenInterest"]
        strikes = df["StrikeRate"].unique()
        pain = [(K, sum(max(0, s - K) * calls.get(s, 0) for s in strikes) + sum(max(0, K - s) * puts.get(s, 0) for s in strikes)) for K in strikes]
        max_pain_strike = min(pain, key=lambda x: x[1])[0]
        max_pain_diff_pct = abs(nifty_spot - max_pain_strike) / nifty_spot * 100 if nifty_spot != 0 else 0
        return max_pain_strike, max_pain_diff_pct
    except Exception as e:
        logger.error(f"Error calculating max pain: {str(e)}")
        return None, None

def fetch_nifty_data(client):
    try:
        req_list = [
            {"Exch": "N", "ExchType": "C", "ScripCode": 999920000},  # NIFTY 50
            {"Exch": "N", "ExchType": "C", "ScripCode": 999920005}   # India VIX
        ]
        market_feed = client.fetch_market_feed(req_list)
        if not market_feed or "Data" not in market_feed or len(market_feed["Data"]) < 2:
            raise Exception("Failed to fetch market data")
        
        nifty_spot = market_feed["Data"][0].get("LastRate", market_feed["Data"][0].get("LastTradedPrice", 0))
        vix = market_feed["Data"][1].get("LastRate", market_feed["Data"][1].get("LastTradedPrice", 0))
        if not nifty_spot or not vix:
            raise Exception("Missing NIFTY or VIX price")

        expiries = client.get_expiry("N", "NIFTY")
        if not expiries or "Data" not in expiries or not expiries["Data"]:
            raise Exception("Failed to fetch expiries")
        expiry_timestamp = expiries["Data"][0]["Timestamp"]
        option_chain = client.get_option_chain("N", "NIFTY", expiry_timestamp)
        if not option_chain or "Options" not in option_chain:
            raise Exception("Failed to fetch option chain")

        df = pd.DataFrame(option_chain["Options"])
        if not all(col in df.columns for col in ["StrikeRate", "CPType", "LastRate", "OpenInterest"]):
            raise Exception("Required columns missing")

        df["StrikeRate"] = df["StrikeRate"].astype(float)
        atm_strike = df["StrikeRate"].iloc[(df["StrikeRate"] - nifty_spot).abs().argmin()]
        atm_data = df[df["StrikeRate"] == atm_strike]
        atm_call = atm_data[atm_data["CPType"] == "CE"]["LastRate"].iloc[0] if not atm_data[atm_data["CPType"] == "CE"].empty else 0
        atm_put = atm_data[atm_data["CPType"] == "PE"]["LastRate"].iloc[0] if not atm_data[atm_data["CPType"] == "PE"].empty else 0
        straddle_price = atm_call + atm_put

        calls = df[df["CPType"] == "CE"]
        puts = df[df["CPType"] == "PE"]
        call_oi = calls["OpenInterest"].sum()
        put_oi = puts["OpenInterest"].sum()
        pcr = put_oi / call_oi if call_oi != 0 else float("inf")

        max_pain_strike, max_pain_diff_pct = max_pain(df, nifty_spot)
        if max_pain_strike is None:
            raise Exception("Max pain calculation failed")

        vix_change_pct = 0
        iv_file = "vix_history.csv"
        if os.path.exists(iv_file):
            iv_history = pd.read_csv(iv_file)
            prev_vix = iv_history["VIX"].iloc[-1] if not iv_history.empty else vix
            vix_change_pct = ((vix - prev_vix) / prev_vix * 100) if prev_vix != 0 else 0
        pd.DataFrame({"Date": [datetime.now().date()], "VIX": [vix]}).to_csv(iv_file, mode='a', header=not os.path.exists(iv_file), index=False)

        return {
            "nifty_spot": nifty_spot,
            "vix": vix,
            "atm_strike": atm_strike,
            "straddle_price": straddle_price,
            "pcr": pcr,
            "max_pain_strike": max_pain_strike,
            "max_pain_diff_pct": max_pain_diff_pct,
            "vix_change_pct": vix_change_pct,
            "option_chain": df,
            "expiry": expiries["Data"][0]["ExpiryDate"]
        }
    except Exception as e:
        logger.error(f"Error fetching 5paisa API data: {str(e)}")
        return None

def load_data(client):
    try:
        real_data = fetch_nifty_data(client) if client else None
        data_source = "5paisa API (LIVE)" if real_data else "GitHub CSV (FALLBACK)"

        if real_data is None:
            nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
            vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"
            nifty = pd.read_csv(nifty_url, encoding="utf-8-sig")
            vix = pd.read_csv(vix_url)
            nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
            vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")
            nifty = nifty.dropna(subset=["Date"]).groupby("Date").last()[["Close"]].rename(columns={"Close": "NIFTY_Close"})
            vix = vix.dropna(subset=["Date"]).groupby("Date").last()[["Close"]].rename(columns={"Close": "VIX"})
            common_dates = nifty.index.intersection(vix.index)
            df = pd.DataFrame({
                "NIFTY_Close": nifty["NIFTY_Close"].loc[common_dates],
                "VIX": vix["VIX"].loc[common_dates]
            }, index=common_dates).sort_index().ffill().bfill()
        else:
            latest_date = datetime.now().date()
            df = pd.DataFrame({
                "NIFTY_Close": [real_data["nifty_spot"]],
                "VIX": [real_data["vix"]]
            }, index=[pd.to_datetime(latest_date)])
            nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
            vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"
            nifty = pd.read_csv(nifty_url, encoding="utf-8-sig")
            vix = pd.read_csv(vix_url)
            nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
            vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")
            nifty = nifty.dropna(subset=["Date"]).groupby("Date").last()[["Close"]].rename(columns={"Close": "NIFTY_Close"})
            vix = vix.dropna(subset=["Date"]).groupby("Date").last()[["Close"]].rename(columns={"Close": "VIX"})
            historical_df = pd.DataFrame({
                "NIFTY_Close": nifty["NIFTY_Close"],
                "VIX": vix["VIX"]
            }).dropna()
            historical_df = historical_df[historical_df.index < pd.to_datetime(latest_date)]
            df = pd.concat([historical_df, df]).groupby(level=0).last().sort_index()

        logger.info(f"Data loaded from {data_source}. Shape: {df.shape}")
        return df, real_data, data_source
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None, None, None

def fetch_portfolio_data(client, capital):
    try:
        positions = client.positions()
        if not positions:
            return {"weekly_pnl": 0, "margin_used": 0, "exposure": 0}
        total_pnl = sum(pos.get("ProfitLoss", 0) for pos in positions)
        total_margin = sum(pos.get("MarginUsed", 0) for pos in positions)
        total_exposure = sum(pos.get("Exposure", 0) for pos in positions) / capital * 100 if capital > 0 else 0
        return {"weekly_pnl": total_pnl, "margin_used": total_margin, "exposure": total_exposure}
    except Exception as e:
        logger.error(f"Error fetching portfolio data: {str(e)}")
        return {"weekly_pnl": 0, "margin_used": 0, "exposure": 0}

# Feature Generation
@st.cache_data
def generate_features(df, real_data, capital):
    try:
        df = df.copy()
        df.index = df.index.normalize()
        n_days = len(df)
        np.random.seed(42)

        base_pcr = real_data["pcr"] if real_data else 1.0
        base_straddle_price = real_data["straddle_price"] if real_data else 200.0
        base_max_pain_diff_pct = real_data["max_pain_diff_pct"] if real_data else 0.5
        base_vix_change_pct = real_data["vix_change_pct"] if real_data else 0.0

        df["Days_to_Expiry"] = [(3 - d.weekday()) % 7 or 7 for d in df.index]
        event_spike = np.where((df.index.month % 3 == 0) & (df.index.day < 5), 1.2, 1.0)
        df["ATM_IV"] = np.clip(df["VIX"] * (1 + np.random.normal(0, 0.1, n_days)) * event_spike, 5, 50)
        if real_data:
            df["ATM_IV"].iloc[-1] = real_data["vix"]

        df["IVP"] = df["ATM_IV"].rolling(252, min_periods=5).apply(lambda x: (np.sum(x[:-1] <= x[-1]) / (len(x) - 1)) * 100 if len(x) >= 5 else 50.0)
        df["IVP"] = df["IVP"].interpolate().fillna(50.0)

        market_trend = df["NIFTY_Close"].pct_change().rolling(5).mean().fillna(0)
        df["PCR"] = np.clip(base_pcr + np.random.normal(0, 0.1, n_days) + market_trend * -10, 0.7, 2.0)
        if real_data:
            df["PCR"].iloc[-1] = base_pcr

        df["VIX_Change_Pct"] = df["VIX"].pct_change().fillna(0) * 100
        if real_data:
            df["VIX_Change_Pct"].iloc[-1] = base_vix_change_pct

        df["Spot_MaxPain_Diff_Pct"] = np.clip(np.abs(np.random.lognormal(-2, 0.5, n_days)), 0.1, 1.0)
        if real_data:
            df["Spot_MaxPain_Diff_Pct"].iloc[-1] = base_max_pain_diff_pct

        df["Event_Flag"] = np.where((df.index.month % 3 == 0) & (df.index.day < 5) | (df["Days_to_Expiry"] <= 3), 1, 0)
        fii_trend = np.random.normal(0, 10000, n_days)
        fii_trend[::30] *= -1
        df["FII_Index_Fut_Pos"] = np.cumsum(fii_trend).astype(int)
        df["FII_Option_Pos"] = np.cumsum(np.random.normal(0, 5000, n_days)).astype(int)
        df["IV_Skew"] = np.clip(np.random.normal(0, 0.8, n_days) + (df["VIX"] / 15 - 1) * 3, -3, 3)
        df["Realized_Vol"] = np.clip(df["NIFTY_Close"].pct_change().rolling(5, min_periods=5).std() * np.sqrt(252) * 100, 0, 50).fillna(df["VIX"])
        df["Advance_Decline_Ratio"] = np.clip(1.0 + np.random.normal(0, 0.2, n_days) + market_trend * 10, 0.5, 2.0)
        df["Capital_Pressure_Index"] = np.clip((df["FII_Index_Fut_Pos"] / 3e4 + df["FII_Option_Pos"] / 1e4 + df["PCR"]) / 3, -2, 2)
        df["Gamma_Bias"] = np.clip(df["IV_Skew"] * (30 - df["Days_to_Expiry"]) / 30, -2, 2)
        df["Total_Capital"] = capital
        df["PnL_Day"] = np.random.normal(0, 5000, n_days) * (1 - df["Event_Flag"] * 0.5)
        df["Straddle_Price"] = np.clip(np.random.normal(base_straddle_price, 50, n_days), 50, 400)
        if real_data:
            df["Straddle_Price"].iloc[-1] = base_straddle_price

        df = df.interpolate().fillna(method='bfill')
        try:
            df.to_csv("volguard_hybrid_data.csv")
        except Exception as e:
            logger.error(f"Error saving volguard_hybrid_data.csv: {str(e)}")

        return df
    except Exception as e:
        logger.error(f"Error generating features: {str(e)}")
        return None

# Volatility Forecasting
feature_cols = ['VIX', 'ATM_IV', 'IVP', 'PCR', 'VIX_Change_Pct', 'IV_Skew', 'Straddle_Price',
                'Spot_MaxPain_Diff_Pct', 'Days_to_Expiry', 'Event_Flag', 'FII_Index_Fut_Pos', 'FII_Option_Pos']

@st.cache_data
def forecast_volatility_future(df, forecast_horizon):
    try:
        df = df.copy()
        df.index = pd.to_datetime(df.index).normalize()
        df_garch = df.tail(len(df))
        if len(df_garch) < 200:
            return None, None, None, None, None, None, None, None

        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon, freq='B')

        df_garch['Log_Returns'] = np.log(df_garch['NIFTY_Close'] / df_garch['NIFTY_Close'].shift(1)).dropna() * 100
        garch_model = arch_model(df_garch['Log_Returns'].dropna(), vol='Garch', p=1, q=1, rescale=False)
        garch_fit = garch_model.fit(disp="off")
        garch_forecast = garch_fit.forecast(horizon=forecast_horizon, reindex=False)
        garch_vols = np.clip(np.sqrt(garch_forecast.variance.iloc[-1].values) * np.sqrt(252), 5, 50)

        realized_vol = df["Realized_Vol"].dropna().iloc[-5:].mean()

        df_xgb = df.tail(len(df))
        df_xgb['Target_Vol'] = df_xgb['Realized_Vol'].shift(-1)
        df_xgb = df_xgb.dropna()

        X = df_xgb[feature_cols]
        y = df_xgb['Target_Vol']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

        split_index = int(len(X) * 0.8)
        X_train, X_test = X_scaled.iloc[:split_index], X_scaled.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.03, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        xgb_vols = []
        current_row = df_xgb[feature_cols].iloc[-1].copy()
        for i in range(forecast_horizon):
            current_row_scaled = scaler.transform([current_row])
            next_vol = model.predict(current_row_scaled)[0]
            xgb_vols.append(next_vol)
            current_row["Days_to_Expiry"] = max(1, current_row["Days_to_Expiry"] - 1)
            current_row["VIX"] *= np.random.uniform(0.98, 1.02)
            current_row["Straddle_Price"] *= np.random.uniform(0.98, 1.02)
            current_row["VIX_Change_Pct"] = (current_row["VIX"] / df_xgb["VIX"].iloc[-1] - 1) * 100
            current_row["ATM_IV"] = current_row["VIX"] * (1 + np.random.normal(0, 0.1))
            current_row["IVP"] = current_row["IVP"] * np.random.uniform(0.99, 1.01)
            current_row["PCR"] = np.clip(current_row["PCR"] + np.random.normal(0, 0.05), 0.7, 2.0)
            current_row["Spot_MaxPain_Diff_Pct"] = np.clip(current_row["Spot_MaxPain_Diff_Pct"] * np.random.uniform(0.95, 1.05), 0.1, 1.0)
            current_row["Event_Flag"] = df_xgb["Event_Flag"].iloc[-1]
            current_row["FII_Index_Fut_Pos"] += np.random.normal(0, 1000)
            current_row["FII_Option_Pos"] += np.random.normal(0, 500)
            current_row["IV_Skew"] = np.clip(current_row["IV_Skew"] + np.random.normal(0, 0.1), -3, 3)

        xgb_vols = np.clip(xgb_vols, 5, 50)
        if df["Event_Flag"].iloc[-1] == 1:
            xgb_vols = [v * 1.1 for v in xgb_vols]

        garch_diff = np.abs(garch_vols[0] - realized_vol)
        xgb_diff = np.abs(xgb_vols[0] - realized_vol)
        garch_weight = xgb_diff / (garch_diff + xgb_diff) if (garch_diff + xgb_diff) > 0 else 0.5
        xgb_weight = 1 - garch_weight
        blended_vols = [(garch_weight * g) + (xgb_weight * x) for g, x in zip(garch_vols, xgb_vols)]
        confidence_score = min(100, max(50, 80 - abs(garch_diff - xgb_diff)))

        forecast_log = pd.DataFrame({
            "Date": future_dates,
            "GARCH_Vol": garch_vols,
            "XGBoost_Vol": xgb_vols,
            "Blended_Vol": blended_vols,
            "Confidence": [confidence_score] * forecast_horizon
        })
        return forecast_log, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, model.feature_importances_
    except Exception as e:
        logger.error(f"Error in volatility forecasting: {str(e)}")
        return None, None, None, None, None, None, None, None

# Backtesting
def run_backtest(df, capital, strategy_choice, start_date, end_date):
    try:
        df = df.groupby(df.index).last().loc[start_date:end_date].copy()
        if len(df) < 50:
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        backtest_results = []
        lot_size = 25
        base_transaction_cost = 0.002
        stt = 0.0005
        portfolio_pnl = 0
        risk_free_rate = 0.06 / 126

        def run_strategy_engine(day_data, avg_vol, portfolio_pnl):
            iv = day_data["ATM_IV"]
            hv = day_data["Realized_Vol"]
            iv_hv_gap = iv - hv
            iv_skew = day_data["IV_Skew"]
            dte = day_data["Days_to_Expiry"]
            event_flag = day_data["Event_Flag"]

            if portfolio_pnl < -0.1 * day_data["Total_Capital"]:
                return None, None, "Portfolio drawdown limit reached", [], 0, 0, 0

            regime = "EVENT-DRIVEN" if event_flag == 1 else "LOW" if avg_vol < 15 else "MEDIUM" if avg_vol < 20 else "HIGH"
            strategy = "Undefined"
            reason = "N/A"
            tags = []
            risk_reward = 1.5 if iv_hv_gap > 5 else 1.0

            if regime == "LOW":
                if iv_hv_gap > 5 and dte < 10:
                    strategy, reason, tags, risk_reward = "Butterfly Spread", "Low vol & short expiry", ["Neutral", "Theta"], 2.0
                else:
                    strategy, reason, tags = "Iron Fly", "Low vol and time decay", ["Neutral", "Theta"]
            elif regime == "MEDIUM":
                if iv_hv_gap > 3 and iv_skew > 2:
                    strategy, reason, tags, risk_reward = "Iron Condor", "Medium vol and skew", ["Neutral", "Theta"], 1.8
                else:
                    strategy, reason, tags = "Short Strangle", "Balanced vol", ["Neutral", "Premium Selling"]
            elif regime == "HIGH":
                if iv_hv_gap > 10:
                    strategy, reason, tags, risk_reward = "Jade Lizard", "High IV + call skew", ["Skewed", "Defined Risk"], 1.2
                else:
                    strategy, reason, tags = "Iron Condor", "High vol", ["Neutral", "Theta"]
            elif regime == "EVENT-DRIVEN":
                if iv > 30 and dte < 5:
                    strategy, reason, tags, risk_reward = "Short Straddle", "Event + IV spike", ["Volatility", "Neutral"], 1.5
                else:
                    strategy, reason, tags = "Calendar Spread", "Event uncertainty", ["Volatility", "Calendar"]

            capital_alloc = {"LOW": 0.10, "MEDIUM": 0.08, "HIGH": 0.06, "EVENT-DRIVEN": 0.06}
            deploy = day_data["Total_Capital"] * capital_alloc.get(regime, 0.06)
            max_loss = deploy * 0.025
            return regime, strategy, reason, tags, deploy, max_loss, risk_reward

        for i in range(1, len(df)):
            day_data = df.iloc[i]
            avg_vol = df["Realized_Vol"].iloc[max(0, i-5):i].mean()
            regime, strategy, reason, tags, deploy, max_loss, risk_reward = run_strategy_engine(day_data, avg_vol, portfolio_pnl)

            if strategy is None or (strategy_choice != "All Strategies" and strategy != strategy_choice):
                continue

            total_cost = base_transaction_cost + (0.001 if "Iron" in strategy else 0) + stt
            slippage = 0.005 * min(day_data["ATM_IV"] / 20, 2.5) * (1.5 if day_data["Days_to_Expiry"] < 5 else 1.0)
            entry_price = day_data["Straddle_Price"]
            lots = max(1, min(int(deploy / (entry_price * lot_size)), 2))
            decay_factor = max(0.75, 1 - day_data["Days_to_Expiry"] / 10)
            premium = entry_price * lot_size * lots * (1 - slippage - total_cost) * decay_factor
            premium *= 0.8 if np.random.rand() < 0.05 else 1.0
            premium *= 0.9 if np.random.rand() < 0.10 else 1.0

            iv_factor = min(day_data["ATM_IV"] / avg_vol, 1.5) if avg_vol != 0 else 1.0
            breakeven = entry_price * (1 + iv_factor * (0.04 if (day_data["Event_Flag"] == 1 or day_data["ATM_IV"] > 25) else 0.06))
            nifty_move = abs(day_data["NIFTY_Close"] - df.iloc[i-1]["NIFTY_Close"])
            loss = min(max(0, nifty_move - breakeven) * lot_size * lots, premium * (0.6 if strategy in ["Iron Fly", "Iron Condor"] else 0.8))
            pnl = premium - loss

            if np.random.rand() < (0.35 if day_data["Event_Flag"] == 1 else 0.20):
                shock_factor = nifty_move / (day_data["ATM_IV"] * 100) if day_data["ATM_IV"] != 0 else 1.0
                pnl -= abs(pnl) * min(shock_factor * 1.5, 2.0)
            if (day_data["Event_Flag"] == 1 or day_data["ATM_IV"] > 25) and np.random.rand() < 0.08:
                pnl -= premium * np.random.uniform(0.5, 1.0)
            if np.random.rand() < 0.02:
                pnl -= premium * np.random.uniform(1.0, 1.5)

            pnl = max(-max_loss, min(pnl, max_loss * 1.5))
            portfolio_pnl += pnl

            backtest_results.append({
                "Date": day_data.name,
                "Regime": regime,
                "Strategy": strategy,
                "PnL": pnl,
                "Capital_Deployed": deploy,
                "Max_Loss": max_loss,
                "Risk_Reward": risk_reward
            })

        backtest_df = pd.DataFrame(backtest_results)
        if backtest_df.empty:
            return backtest_df, 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        total_pnl = backtest_df["PnL"].sum()
        win_rate = len(backtest_df[backtest_df["PnL"] > 0]) / len(backtest_df)
        max_drawdown = (backtest_df["PnL"].cumsum().cummax() - backtest_df["PnL"].cumsum()).max()

        backtest_df.set_index("Date", inplace=True)
        returns = backtest_df["PnL"] / df["Total_Capital"].reindex(backtest_df.index, method="ffill").fillna(capital)
        nifty_returns = df["NIFTY_Close"].pct_change().reindex(backtest_df.index, method="ffill").fillna(0)
        excess_returns = returns - nifty_returns - risk_free_rate
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(126) if excess_returns.std() != 0 else 0
        sortino_ratio = excess_returns.mean() / excess_returns[excess_returns < 0].std() * np.sqrt(126) if len(excess_returns[excess_returns < 0]) > 0 and excess_returns[excess_returns < 0].std() != 0 else 0
        calmar_ratio = (total_pnl / capital) / (max_drawdown / capital) if max_drawdown != 0 else float('inf')

        strategy_perf = backtest_df.groupby("Strategy")["PnL"].agg(['sum', 'count', 'mean']).reset_index()
        strategy_perf["Win_Rate"] = backtest_df.groupby("Strategy")["PnL"].apply(lambda x: len(x[x > 0]) / len(x) if len(x) > 0 else 0).reset_index(drop=True)
        regime_perf = backtest_df.groupby("Regime")["PnL"].agg(['sum', 'count', 'mean']).reset_index()
        regime_perf["Win_Rate"] = backtest_df.groupby("Regime")["PnL"].apply(lambda x: len(x[x > 0]) / len(x) if len(x) > 0 else 0).reset_index(drop=True)

        return backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

# Strategy Generation
@st.cache_data
def generate_trading_strategy(df, forecast_log, realized_vol, risk_tolerance, confidence_score, capital):
    try:
        df = df.copy()
        latest = df.iloc[-1]
        avg_vol = np.mean(forecast_log["Blended_Vol"])
        iv = latest["ATM_IV"]
        hv = latest["Realized_Vol"]
        iv_hv_gap = iv - hv
        iv_skew = latest["IV_Skew"]
        dte = latest["Days_to_Expiry"]
        event_flag = latest["Event_Flag"]

        risk_flags = []
        if latest["VIX"] > 25:
            risk_flags.append("High Volatility")
        if latest["Spot_MaxPain_Diff_Pct"] > 70:
            risk_flags.append("High Exposure")
        if latest["PnL_Day"] < -0.05 * capital:
            risk_flags.append("High Loss")
        if latest["VIX_Change_Pct"] > 10:
            risk_flags.append("VIX Spike")

        if risk_flags:
            st.session_state.violations += 1
            if st.session_state.violations >= 2 and not st.session_state.journal_complete:
                return None

        regime = "EVENT-DRIVEN" if event_flag == 1 else "LOW" if avg_vol < 15 else "MEDIUM" if avg_vol < 20 else "HIGH"
        strategy = "Undefined"
        reason = "N/A"
        tags = []
        risk_reward = 1.0
        confidence = 0.5 + 0.5 * (1 - np.abs(forecast_log["GARCH_Vol"].iloc[0] - forecast_log["XGBoost_Vol"].iloc[0]) / max(forecast_log["GARCH_Vol"].iloc[0], forecast_log["XGBoost_Vol"].iloc[0]))

        if regime == "LOW":
            if iv_hv_gap > 5 and dte < 10:
                strategy, reason, tags, risk_reward = "Butterfly Spread", "Low vol & short expiry", ["Neutral", "Theta"], 2.0
            else:
                strategy, reason, tags = "Iron Fly", "Low volatility", ["Neutral", "Theta"]
        elif regime == "MEDIUM":
            if iv_hv_gap > 3 and iv_skew > 2:
                strategy, reason, tags, risk_reward = "Iron Condor", "Medium vol and skew", ["Neutral", "Theta"], 1.8
            else:
                strategy, reason, tags = "Short Strangle", "Balanced vol", ["Neutral", "Premium Selling"]
        elif regime == "HIGH":
            if iv_hv_gap > 10:
                strategy, reason, tags, risk_reward = "Jade Lizard", "High IV", ["Skewed", "Defined Risk"], 1.2
            else:
                strategy, reason, tags = "Iron Condor", "High vol", ["Neutral", "Theta"]
        elif regime == "EVENT-DRIVEN":
            if iv > 30 and dte < 5:
                strategy, reason, tags, risk_reward = "Short Straddle", "Event + IV spike", ["Volatility", "Neutral"], 1.5
            else:
                strategy, reason, tags = "Calendar Spread", "Event uncertainty", ["Volatility", "Calendar"]

        capital_alloc = {"LOW": 0.35, "MEDIUM": 0.25, "HIGH": 0.15, "EVENT-DRIVEN": 0.2}
        position_size = {"Conservative": 0.5, "Moderate": 1.0, "Aggressive": 1.5}[risk_tolerance]
        deploy = capital * capital_alloc.get(regime, 0.2) * position_size
        max_loss = deploy * 0.2
        total_exposure = deploy / capital

        return {
            "Regime": regime,
            "Strategy": strategy,
            "Reason": reason,
            "Tags": tags,
            "Confidence": confidence,
            "Risk_Reward": risk_reward,
            "Deploy": deploy,
            "Max_Loss": max_loss,
            "Exposure": total_exposure,
            "Risk_Flags": risk_flags
        }
    except Exception as e:
        logger.error(f"Error generating strategy: {str(e)}")
        return None

# Trading Functions
def place_trade(client, strategy, real_data, capital):
    try:
        if not real_data or "option_chain" not in real_data or "atm_strike" not in real_data:
            return False, "Invalid real-time data"

        option_chain = real_data["option_chain"]
        atm_strike = real_data["atm_strike"]
        lot_size = 25
        deploy = strategy["Deploy"]
        max_loss = strategy["Max_Loss"]
        expiry = real_data["expiry"]

        premium_per_lot = real_data["straddle_price"] * lot_size
        lots = max(1, min(int(deploy / premium_per_lot), int(max_loss / (premium_per_lot * 0.2))))

        orders = []
        if strategy["Strategy"] == "Short Straddle":
            strikes = [(atm_strike, "CE", "S"), (atm_strike, "PE", "S")]
        elif strategy["Strategy"] == "Short Strangle":
            call_strike = atm_strike + 100
            put_strike = atm_strike - 100
            strikes = [(call_strike, "CE", "S"), (put_strike, "PE", "S")]
        elif strategy["Strategy"] == "Iron Condor":
            call_sell_strike = atm_strike + 100
            call_buy_strike = call_sell_strike + 100
            put_sell_strike = atm_strike - 100
            put_buy_strike = put_sell_strike - 100
            strikes = [
                (call_sell_strike, "CE", "S"),
                (call_buy_strike, "CE", "B"),
                (put_sell_strike, "PE", "S"),
                (put_buy_strike, "PE", "B")
            ]
        elif strategy["Strategy"] == "Iron Fly":
            call_sell_strike = atm_strike
            call_buy_strike = atm_strike + 100
            put_sell_strike = atm_strike
            put_buy_strike = atm_strike - 100
            strikes = [
                (call_sell_strike, "CE", "S"),
                (call_buy_strike, "CE", "B"),
                (put_sell_strike, "PE", "S"),
                (put_buy_strike, "PE", "B")
            ]
        elif strategy["Strategy"] == "Butterfly Spread":
            call_sell_strike = atm_strike
            call_buy_strike = atm_strike + 200
            put_buy_strike = atm_strike - 200
            strikes = [
                (call_sell_strike, "CE", "S"),
                (call_buy_strike, "CE", "B"),
                (put_buy_strike, "PE", "B")
            ]
        elif strategy["Strategy"] == "Jade Lizard":
            call_sell_strike = atm_strike + 100
            put_sell_strike = atm_strike - 100
            put_buy_strike = atm_strike - 200
            strikes = [
                (call_sell_strike, "CE", "S"),
                (put_sell_strike, "PE", "S"),
                (put_buy_strike, "PE", "B")
            ]
        elif strategy["Strategy"] == "Calendar Spread":
            call_sell_strike = atm_strike
            call_buy_strike = atm_strike
            strikes = [
                (call_sell_strike, "CE", "S"),
                (call_buy_strike, "CE", "B")
            ]
        else:
            return False, "Unsupported strategy"

        for strike, cp_type, buy_sell in strikes:
            opt_data = option_chain[(option_chain["StrikeRate"] == strike) & (option_chain["CPType"] == cp_type)]
            if opt_data.empty:
                return False, f"No option data for {cp_type} at strike {strike}"
            scrip_code = int(opt_data["ScripCode"].iloc[0])
            price = float(opt_data["LastRate"].iloc[0])
            order = {
                "Exchange": "N",
                "ExchangeType": "D",
                "ScripCode": scrip_code,
                "Quantity": lot_size * lots,
                "Price": 0,
                "OrderType": "SELL" if buy_sell == "S" else "BUY",
                "IsIntraday": False
            }
            orders.append(order)

        for order in orders:
            response = client.place_order(
                OrderType=order["OrderType"],
                Exchange=order["Exchange"],
                ExchangeType=order["ExchangeType"],
                ScripCode=order["ScripCode"],
                Qty=order["Quantity"],
                Price=order["Price"],
                IsIntraday=order["IsIntraday"]
            )
            if response.get("Status") != 0:
                return False, f"Order failed: {response.get('Message', 'Unknown error')}"

        return True, f"Trade placed: {strategy['Strategy']} with {lots} lots"
    except Exception as e:
        logger.error(f"Error placing trade: {str(e)}")
        return False, f"Trade failed: {str(e)}"

def square_off_positions(client):
    try:
        response = client.squareoff_all()
        return response.get("Status") == 0
    except Exception as e:
        logger.error(f"Error squaring off positions: {str(e)}")
        return False

# Sidebar
with st.sidebar:
    st.header("🔐 Login")
    totp_code = st.text_input("TOTP Code", type="password")
    if st.button("Login"):
        client = initialize_5paisa_client(totp_code)
        if client:
            st.session_state.client = client
            st.session_state.logged_in = True
            st.success("Logged in successfully")
        else:
            st.error("Login failed")

    if st.session_state.logged_in:
        st.header("⚙️ Controls")
        capital = st.number_input("Capital (₹)", min_value=100000, value=1000000, step=100000)
        risk_tolerance = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"])
        forecast_horizon = st.slider("Forecast Horizon (days)", 1, 30, 7)
        start_date = st.date_input("Backtest Start", pd.to_datetime("2024-01-01"))
        end_date = st.date_input("Backtest End", pd.to_datetime("2025-04-29"))
        strategy_choice = st.selectbox("Backtest Strategy", ["All Strategies", "Butterfly Spread", "Iron Condor", "Iron Fly", "Short Strangle", "Calendar Spread", "Jade Lizard", "Short Straddle"])
        if st.button("Square Off Positions"):
            if square_off_positions(st.session_state.client):
                st.success("Positions squared off")
            else:
                st.error("Failed to square off")

# Main App
if not st.session_state.logged_in:
    st.info("Please login to proceed")
else:
    st.markdown("<h1 style='text-align: center; color: #007bff;'>VolGuard: Options Trading Dashboard</h1>", unsafe_allow_html=True)
    
    if st.button("Run Analysis"):
        with st.spinner("Analyzing market data..."):
            st.session_state.backtest_run = False
            st.session_state.backtest_results = None
            st.session_state.violations = 0
            st.session_state.journal_complete = False

            df, real_data, data_source = load_data(st.session_state.client)
            if df is not None:
                df = generate_features(df, real_data, capital)
                if df is not None:
                    backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf = run_backtest(
                        df, capital, strategy_choice, start_date, end_date
                    )
                    st.session_state.backtest_run = True
                    st.session_state.backtest_results = {
                        "backtest_df": backtest_df,
                        "total_pnl": total_pnl,
                        "win_rate": win_rate,
                        "max_drawdown": max_drawdown,
                        "sharpe_ratio": sharpe_ratio,
                        "sortino_ratio": sortino_ratio,
                        "calmar_ratio": calmar_ratio,
                        "strategy_perf": strategy_perf,
                        "regime_perf": regime_perf
                    }

                    # Market Snapshot
                    with st.container():
                        st.markdown('<div class="section"><h2 class="section-header">📊 Market Snapshot</h2>', unsafe_allow_html=True)
                        last_date = df.index[-1].strftime("%d-%b-%Y")
                        last_nifty = df["NIFTY_Close"].iloc[-1]
                        prev_nifty = df["NIFTY_Close"].iloc[-2] if len(df) >= 2 else last_nifty
                        last_vix = df["VIX"].iloc[-1]
                        regime = "LOW" if last_vix < 15 else "MEDIUM" if last_vix < 20 else "HIGH"
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("NIFTY 50", f"₹{last_nifty:,.2f}", f"{(last_nifty - prev_nifty)/prev_nifty*100:+.2f}%")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("India VIX", f"{last_vix:.2f}%", f"{df['VIX_Change_Pct'].iloc[-1]:+.2f}%")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col3:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("PCR", f"{df['PCR'].iloc[-1]:.2f}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col4:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Straddle Price", f"₹{df['Straddle_Price'].iloc[-1]:,.2f}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown(f"<p><b>Last Updated:</b> {last_date} | <b>Source:</b> {data_source}</p>", unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Volatility Forecast
                    with st.container():
                        st.markdown('<div class="section"><h2 class="section-header">📈 Volatility Forecast</h2>', unsafe_allow_html=True)
                        forecast_log, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, feature_importances = forecast_volatility_future(df, forecast_horizon)
                        if forecast_log is not None:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                st.metric("Avg Blended Volatility", f"{np.mean(blended_vols):.2f}%")
                                st.markdown('</div>', unsafe_allow_html=True)
                            with col2:
                                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                st.metric("Realized Volatility", f"{realized_vol:.2f}%")
                                st.markdown('</div>', unsafe_allow_html=True)
                            with col3:
                                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                st.metric("Confidence", f"{int(confidence_score)}%")
                                st.markdown('</div>', unsafe_allow_html=True)
                            forecast_df = pd.DataFrame({
                                "Date": forecast_log["Date"],
                                "GARCH": garch_vols,
                                "XGBoost": xgb_vols,
                                "Blended": blended_vols
                            })
                            fig = px.line(forecast_df, x="Date", y=["GARCH", "XGBoost", "Blended"], title="Volatility Forecast")
                            st.plotly_chart(fig, use_container_width=True)
                            st.markdown("### Feature Importance")
                            feature_df = pd.DataFrame({
                                'Feature': feature_cols,
                                'Importance': feature_importances
                            }).sort_values(by='Importance', ascending=False)
                            fig = px.bar(feature_df, x='Importance', y='Feature', orientation='h')
                            st.plotly_chart(fig, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Trading Strategies
                    with st.container():
                        st.markdown('<div class="section"><h2 class="section-header">🎯 Trading Strategies</h2>', unsafe_allow_html=True)
                        strategy = generate_trading_strategy(df, forecast_log, realized_vol, risk_tolerance, confidence_score, capital)
                        if strategy is None:
                            st.markdown('<div class="alert">🚨 Complete Journal to Unlock Trading</div>', unsafe_allow_html=True)
                        else:
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <h3>{strategy['Strategy']}</h3>
                                        <p><b>Regime:</b> {strategy['Regime']}</p>
                                        <p><b>Reason:</b> {strategy['Reason']}</p>
                                        <p><b>Confidence:</b> {strategy['Confidence']:.2f}</p>
                                        <p><b>Risk-Reward:</b> {strategy['Risk_Reward']:.2f}:1</p>
                                        <p><b>Capital:</b> ₹{strategy['Deploy']:,.0f}</p>
                                        <p><b>Max Loss:</b> ₹{strategy['Max_Loss']:,.0f}</p>
                                        <p><b>Tags:</b> {', '.join(strategy['Tags'])}</p>
                                    </div>
                                """, unsafe_allow_html=True)
                            with col2:
                                if st.button("Trade Now", key="trade_button"):
                                    st.session_state.show_trade_modal = True
                                    st.session_state.strategy = strategy

                            if strategy["Risk_Flags"]:
                                st.markdown(f'<div class="alert">⚠️ Risk Flags: {", ".join(strategy["Risk_Flags"])}</div>', unsafe_allow_html=True)

                            if st.session_state.get("show_trade_modal", False):
                                with st.container():
                                    st.markdown('<div class="trade-modal">', unsafe_allow_html=True)
                                    st.subheader("Confirm Trade")
                                    st.write(f"**Strategy**: {strategy['Strategy']}")
                                    st.write(f"**Capital**: ₹{strategy['Deploy']:,.0f}")
                                    st.write(f"**Max Loss**: ₹{strategy['Max_Loss']:,.0f}")
                                    st.write(f"**Expected Risk-Reward**: {strategy['Risk_Reward']:.2f}:1")
                                    if real_data:
                                        lots = max(1, min(int(strategy['Deploy'] / (real_data['straddle_price'] * 25)), int(strategy['Max_Loss'] / (real_data['straddle_price'] * 25 * 0.2))))
                                        st.write(f"**Lots**: {lots}")
                                        st.write(f"**Expiry**: {real_data['expiry']}")
                                    confirm = st.checkbox("I confirm the trade details")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if st.button("Execute Trade") and confirm:
                                            success, message = place_trade(st.session_state.client, strategy, real_data, capital)
                                            if success:
                                                st.session_state.trades.append({
                                                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                    "Strategy": strategy["Strategy"],
                                                    "Regime": strategy["Regime"],
                                                    "Risk_Level": "High" if strategy["Risk_Flags"] else "Low",
                                                    "Outcome": "Pending"
                                                })
                                                try:
                                                    pd.DataFrame(st.session_state.trades).to_csv("trade_log.csv", index=False)
                                                except Exception as e:
                                                    logger.error(f"Error saving trade_log.csv: {str(e)}")
                                                st.markdown(f'<div class="success">{message}</div>', unsafe_allow_html=True)
                                            else:
                                                st.markdown(f'<div class="alert">{message}</div>', unsafe_allow_html=True)
                                            st.session_state.show_trade_modal = False
                                    with col2:
                                        if st.button("Cancel"):
                                            st.session_state.show_trade_modal = False
                                    st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Portfolio
                    with st.container():
                        st.markdown('<div class="section"><h2 class="section-header">💼 Portfolio</h2>', unsafe_allow_html=True)
                        portfolio_data = fetch_portfolio_data(st.session_state.client, capital)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Weekly P&L", f"₹{portfolio_data['weekly_pnl']:,.2f}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Margin Used", f"₹{portfolio_data['margin_used']:,.2f}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col3:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Exposure", f"{portfolio_data['exposure']:.2f}%")
                            st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown("### Open Positions")
                        try:
                            positions = st.session_state.client.positions()
                            pos_df = pd.DataFrame(positions) if isinstance(positions, list) else pd.DataFrame([positions])
                            st.dataframe(pos_df, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error fetching positions: {e}")
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Journal
                    with st.container():
                        st.markdown('<div class="section"><h2 class="section-header">📝 Journal</h2>', unsafe_allow_html=True)
                        with st.form(key="journal_form"):
                            reason_strategy = st.selectbox("Strategy Reason", ["High IV", "Low Risk", "Event Opportunity", "Other"])
                            override_risk = st.radio("Override Risk Flags?", ("Yes", "No"))
                            expected_outcome = st.text_area("Expected Outcome")
                            if st.form_submit_button("Submit Journal"):
                                score = (3 if override_risk == "No" else 0) + (3 if reason_strategy != "Other" else 0) + (3 if expected_outcome else 0) + (1 if portfolio_data["weekly_pnl"] > 0 else 0)
                                score = min(score, 10)
                                journal_entry = {
                                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "Strategy_Reason": reason_strategy,
                                    "Override_Risk": override_risk,
                                    "Expected_Outcome": expected_outcome,
                                    "Discipline_Score": score
                                }
                                try:
                                    pd.DataFrame([journal_entry]).to_csv("journal_log.csv", mode='a', header=not os.path.exists("journal_log.csv"), index=False)
                                except Exception as e:
                                    logger.error(f"Error saving journal_log.csv: {str(e)}")
                                st.markdown(f'<div class="success">Journal Saved! Score: {score}/10</div>', unsafe_allow_html=True)
                                st.session_state.journal_complete = True
                                if st.session_state.violations > 0:
                                    st.session_state.violations = 0
                                    st.success("Discipline Lock Removed")
                        if os.path.exists("journal_log.csv"):
                            journal_df = pd.read_csv("journal_log.csv")
                            st.dataframe(journal_df, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Backtest
                    with st.container():
                        st.markdown('<div class="section"><h2 class="section-header">📉 Backtest Results</h2>', unsafe_allow_html=True)
                        if st.session_state.backtest_run and st.session_state.backtest_results is not None:
                            results = st.session_state.backtest_results
                            if results["backtest_df"].empty:
                                st.warning("No trades generated. Adjust parameters.")
                            else:
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                    st.metric("Total P&L", f"₹{results['total_pnl']:,.2f}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                with col2:
                                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                    st.metric("Win Rate", f"{results['win_rate']*100:.2f}%")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                with col3:
                                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                    st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                with col4:
                                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                    st.metric("Max Drawdown", f"₹{results['max_drawdown']:,.2f}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                cum_pnl = results["backtest_df"]["PnL"].cumsum()
                                fig = px.line(x=cum_pnl.index, y=cum_pnl, title="Cumulative P&L")
                                st.plotly_chart(fig, use_container_width=True)
                                st.markdown("### Strategy Performance")
                                st.dataframe(results["strategy_perf"].style.format({"sum": "₹{:,.2f}", "mean": "₹{:,.2f}", "Win_Rate": "{:.2%}"}))
                                st.markdown("### Regime Performance")
                                st.dataframe(results["regime_perf"].style.format({"sum": "₹{:,.2f}", "mean": "₹{:,.2f}", "Win_Rate": "{:.2%}"}))
                        else:
                            st.info("Run analysis to view backtest results")
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Trade History
                    with st.container():
                        st.markdown('<div class="section"><h2 class="section-header">📜 Trade History</h2>', unsafe_allow_html=True)
                        if st.session_state.trades:
                            trade_df = pd.DataFrame(st.session_state.trades)
                            st.dataframe(trade_df, use_container_width=True)
                            if st.button("Export Trade History"):
                                trade_df.to_csv("trade_history_export.csv", index=False)
                                st.success("Trade history exported")
                        else:
                            st.info("No trades executed yet")
                        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="footer">VolGuard © 2025 | Built by Shritish Shukla & Salman Azim</div>', unsafe_allow_html=True)
