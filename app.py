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
import plotly.express as px
import plotly.graph_objects as go

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(page_title="VolGuard Pro", page_icon="📈", layout="wide")

# Custom CSS for modern, professional UI
st.markdown("""
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #f4f7fa; color: #333; }
        .main { background: #ffffff; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .sidebar .sidebar-content { background: #1a2a44; color: white; padding: 20px; border-radius: 10px; }
        .stButton>button { 
            background: #1a73e8; color: white; border-radius: 8px; padding: 10px 20px; 
            font-weight: 500; transition: all 0.3s; width: 100%; 
        }
        .stButton>button:hover { background: #1557b0; transform: translateY(-2px); }
        .metric-card { 
            background: #f8f9fa; border-radius: 8px; padding: 15px; 
            text-align: center; box-shadow: 0 1px 5px rgba(0,0,0,0.1); 
        }
        .strategy-card { 
            background: #ffffff; border-radius: 8px; padding: 15px; 
            margin: 10px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); 
        }
        .regime-badge { 
            padding: 6px 12px; border-radius: 15px; font-size: 12px; font-weight: bold; 
            text-transform: uppercase; display: inline-block; 
        }
        .regime-low { background: #28a745; color: white; }
        .regime-medium { background: #ffc107; color: black; }
        .regime-high { background: #dc3545; color: white; }
        .regime-event { background: #ff6f61; color: white; }
        .alert { background: #fff3cd; color: #856404; padding: 10px; border-radius: 8px; }
        .trade-modal { 
            background: #ffffff; border-radius: 10px; padding: 20px; 
            box-shadow: 0 4px 20px rgba(0,0,0,0.2); max-width: 600px; margin: auto; 
        }
        .footer { text-align: center; padding: 20px; color: #666; font-size: 12px; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "client" not in st.session_state:
    st.session_state.client = None
if "analysis_run" not in st.session_state:
    st.session_state.analysis_run = False
if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = {}
if "trades" not in st.session_state:
    st.session_state.trades = []

# 5paisa Client Initialization
def initialize_5paisa_client(totp_code):
    try:
        cred = st.secrets.get("fivepaisa", {})
        client = FivePaisaClient(cred=cred)
        client.get_totp_session(cred.get("CLIENT_CODE", ""), totp_code, cred.get("PIN", ""))
        if client.get_access_token():
            logger.info("5paisa client initialized successfully")
            return client
        logger.error("Failed to get access token")
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
        pain = [(K, sum(max(0, s - K) * calls.get(s, 0) for s in strikes) + 
                   sum(max(0, K - s) * puts.get(s, 0) for s in strikes)) for K in strikes]
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
        
        nifty_spot = market_feed["Data"][0].get("LastRate", 0)
        vix = market_feed["Data"][1].get("LastRate", 0)
        if not nifty_spot or not vix:
            raise Exception("Missing NIFTY or VIX price")

        expiries = client.get_expiry("N", "NIFTY")
        if not expiries or not expiries.get("Data"):
            raise Exception("Failed to fetch expiries")
        expiry_timestamp = expiries["Data"][0]["Timestamp"]
        option_chain = client.get_option_chain("N", "NIFTY", expiry_timestamp)
        if not option_chain or "Options" not in option_chain:
            raise Exception("Failed to fetch option chain")

        df = pd.DataFrame(option_chain["Options"])
        df["StrikeRate"] = df["StrikeRate"].astype(float)
        atm_strike = df["StrikeRate"].iloc[(df["StrikeRate"] - nifty_spot).abs().argmin()]
        atm_data = df[df["StrikeRate"] == atm_strike]
        atm_call = atm_data[atm_data["CPType"] == "CE"]["LastRate"].iloc[0] if not atm_data[atm_data["CPType"] == "CE"].empty else 0
        atm_put = atm_data[atm_data["CPType"] == "PE"]["LastRate"].iloc[0] if not atm_data[atm_data["CPType"] == "PE"].empty else 0
        straddle_price = atm_call + atm_put

        calls = df[df["CPType"] == "CE"]["OpenInterest"].sum()
        puts = df[df["CPType"] == "PE"]["OpenInterest"].sum()
        pcr = puts / calls if calls != 0 else float("inf")
        max_pain_strike, max_pain_diff_pct = max_pain(df, nifty_spot)

        return {
            "nifty_spot": nifty_spot,
            "vix": vix,
            "atm_strike": atm_strike,
            "straddle_price": straddle_price,
            "pcr": pcr,
            "max_pain_strike": max_pain_strike,
            "max_pain_diff_pct": max_pain_diff_pct,
            "vix_change_pct": 0,
            "option_chain": df,
            "expiry": expiries["Data"][0]["ExpiryDate"]
        }
    except Exception as e:
        logger.error(f"Error fetching 5paisa data: {str(e)}")
        return None

def load_data(client):
    try:
        real_data = fetch_nifty_data(client) if client else None
        data_source = "5paisa API" if real_data else "GitHub CSV"
        logger.info(f"Data source: {data_source}")

        if real_data:
            df = pd.DataFrame({
                "NIFTY_Close": [real_data["nifty_spot"]],
                "VIX": [real_data["vix"]]
            }, index=[pd.to_datetime(datetime.now().date())])

        nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
        vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"
        nifty = pd.read_csv(nifty_url, encoding="utf-8-sig")
        vix = pd.read_csv(vix_url)
        
        nifty["Date"] = pd.to_datetime(nifty["Date"].str.strip(), format="%d-%b-%Y", errors="coerce")
        vix["Date"] = pd.to_datetime(vix["Date"].str.strip(), format="%d-%b-%Y", errors="coerce")
        nifty = nifty[["Date", "Close"]].dropna().set_index("Date").rename(columns={"Close": "NIFTY_Close"})
        vix = vix[["Date", "Close"]].dropna().set_index("Date").rename(columns={"Close": "VIX"})

        if real}}^

        df_historical = pd.concat([nifty, vix], axis=1).dropna()
        df_historical.index = df_historical.index.normalize()
        if real_data:
            df = pd.concat([df_historical, df]).groupby(level=0).last()
        else:
            df = df_historical.groupby(level=0).last()
        df = df.sort_index().ffill()

        return df, real_data, data_source
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error(f"Failed to load data: {str(e)}")
        return None, None, None

# Feature Generation
@st.cache_data
def generate_features(df, real_data, capital):
    try:
        df = df.copy()
        df.index = df.index.normalize()
        n_days = len(df)
        np.random.seed(42)

        df["Days_to_Expiry"] = [(3 - d.weekday()) % 7 or 7 for d in df.index]
        df["ATM_IV"] = np.clip(df["VIX"] * (1 + np.random.normal(0, 0.1, n_days)), 5, 50)
        if real_data:
            df["ATM_IV"].iloc[-1] = real_data["vix"]

        df["IVP"] = df["ATM_IV"].rolling(252, min_periods=5).apply(
            lambda x: (np.sum(x.iloc[:-1] <= x.iloc[-1]) / (len(x) - 1)) * 100 if len(x) > 1 else 50.0
        ).interpolate().fillna(50.0)

        market_trend = df["NIFTY_Close"].pct_change().rolling(5).mean().fillna(0)
        base_pcr = real_data["pcr"] if real_data   # Use real-time PCR if available
        df["PCR"] = np.clip(base_pcr + np.random.normal(0, 0.1, n_days) + market_trend * -10, 0.7, 2.0)
        if real_data:
            df["PCR"].iloc[-1] = base_pcr

        df["VIX_Change_Pct"] = df["VIX"].pct_change().fillna(0) * 100
        if real_data:
            df["VIX_Change_Pct"].iloc[-1] = real_data["vix_change_pct"]

        df["Spot_MaxPain_Diff_Pct"] = np.clip(
            np.abs(np.random.lognormal(-2, 0.5, n_days)), 0.1, 1.0
        )
        if real_data:
            df["Spot_MaxPain_Diff_Pct"].iloc[-1] = real_data["max_pain_diff_pct"]

        df["Event_Flag"] = ((df.index.month % 3 == 0) & (df.index.day < 5) | (df["Days_to_Expiry"] <= 3)).astype(int)
        df["FII_Index_Fut_Pos"] = np.cumsum(np.random.normal(0, 10000, n_days) * np.where(np.arange(n_days) % 30 == 0, -1, 1)).astype(int)
        df["FII_Option_Pos"] = np.cumsum(np.random.normal(0, 5000, n_days)).astype(int)
        df["IV_Skew"] = np.clip(np.random.normal(0, 0.8, n_days) + (df["VIX"] / 15 - 1) * 3, -3, 3)
        df["Realized_Vol"] = np.clip(
            df["NIFTY_Close"].pct_change().rolling(5, min_periods=5).std() * np.sqrt(252) * 100, 0, 50
        ).fillna(df["VIX"])
        df["Advance_Decline_Ratio"] = np.clip(1.0 + np.random.normal(0, 0.2, n_days) + market_trend * 10, 0.5, 2.0)
        df["Capital_Pressure_Index"] = np.clip(
            (df["FII_Index_Fut_Pos"] / 3e4 + df["FII_Option_Pos"] / 1e4 + df["PCR"]) / 3, -2, 2
        )
        df["Gamma_Bias"] = np.clip(df["IV_Skew"] * (30 - df["Days_to_Expiry"]) / 30, -2, 2)
        df["Total_Capital"] = capital
        df["Straddle_Price"] = np.clip(
            np.random.normal(real_data["straddle_price"] if real_data else 200, 50, n_days), 50, 400
        )
        if real_data:
            df["Straddle_Price"].iloc[-1] = real_data["straddle_price"]

        df = df.interpolate().fillna(method='bfill')
        return df
    except Exception as e:
        logger.error(f"Error generating features: {str(e)}")
        st.error(f"Feature generation failed: {str(e)}")
        return None

# Volatility Forecasting
feature_cols = [
    'VIX', 'ATM_IV', 'IVP', 'PCR', 'VIX_Change_Pct', 'IV_Skew', 'Straddle_Price',
    'Spot_MaxPain_Diff_Pct', 'Days_to_Expiry', 'Event_Flag', 'FII_Index_Fut_Pos',
    'FII_Option_Pos'
]

@st.cache_data
def forecast_volatility_future(df, forecast_horizon):
    try:
        df = df.copy()
        df['Log_Returns'] = np.log(df['NIFTY_Close'] / df['NIFTY_Close'].shift(1)).dropna() * 100
        garch_model = arch_model(df['Log_Returns'].dropna(), vol='Garch', p=1, q=1, rescale=False)
        garch_fit = garch_model.fit(disp="off")
        garch_forecast = garch_fit.forecast(horizon=forecast_horizon, reindex=False)
        garch_vols = np.clip(np.sqrt(garch_forecast.variance.iloc[-1].values) * np.sqrt(252), 5, 50)

        realized_vol = df["Realized_Vol"].dropna().iloc[-5:].mean()

        df_xgb = df.copy()
        df_xgb['Target_Vol'] = df_xgb['Realized_Vol'].shift(-1)
        df_xgb = df_xgb.dropna()

        X = df_xgb[feature_cols]
        y = df_xgb['Target_Vol']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        split_index = int(len(X) * 0.8)
        X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.03, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        xgb_vols = []
        current_row = df_xgb[feature_cols].iloc[-1].copy()
        for _ in range(forecast_horizon):
            current_row_scaled = scaler.transform([current_row])[0]
            next_vol = model.predict([current_row_scaled])[0]
            xgb_vols.append(next_vol)
            current_row["Days_to_Expiry"] = max(1, current_row["Days_to_Expiry"] - 1)
            current_row["VIX"] *= np.random.uniform(0.98, 1.02)
            current_row["Straddle_Price"] *= np.random.uniform(0.98, 1.02)
            current_row["VIX_Change_Pct"] = (current_row["VIX"] / df_xgb["VIX"].iloc[-1] - 1) * 100
            current_row["ATM_IV"] = current_row["VIX"] * (1 + np.random.normal(0, 0.1))
            current_row["IVP"] = current_row["IVP"] * np.random.uniform(0.99, 1.01)
            current_row["PCR"] = np.clip(current_row["PCR"] + np.random.normal(0, 0.05), 0.7, 2.0)
            current_row["Spot_MaxPain_Diff_Pct"] = np.clip(current_row["Spot_MaxPain_Diff_Pct"] * np.random.uniform(0.95, 1.05), 0.1, 1.0)
            current_row["FII_Index_Fut_Pos"] += np.random.normal(0, 1000)
            current_row["FII_Option_Pos"] += np.random.normal(0, 500)
            current_row["IV_Skew"] = np.clip(current_row["IV_Skew"] + np.random.normal(0, 0.1), -3, 3)

        xgb_vols = np.clip(xgb_vols, 5, 50)
        if df["Event_Flag"].iloc[-1] == 1:
            xgb_vols = [v * 1.1 for v in xgb_vols]

        garch_weight = 0.5  # Simplified blending
        blended_vols = [(garch_weight * g) + (1 - garch_weight) * x for g, x in zip(garch_vols, xgb_vols)]
        confidence_score = 80  # Placeholder for simplicity

        forecast_log = pd.DataFrame({
            "Date": pd.date_range(start=df.index[-1] + timedelta(days=1), periods=forecast_horizon, freq='B'),
            "GARCH_Vol": garch_vols,
            "XGBoost_Vol": xgb_vols,
            "Blended_Vol": blended_vols
        })

        return forecast_log, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, model.feature_importances_
    except Exception as e:
        logger.error(f"Error in volatility forecasting: {str(e)}")
        st.error(f"Volatility forecasting failed: {str(e)}")
        return None, None, None, None, None, None, None, None

# Backtesting
def run_backtest(df, capital, strategy_choice, start_date, end_date):
    try:
        df = df.loc[start_date:end_date].copy()
        if len(df) < 50:
            st.warning(f"Insufficient data for backtest: {len(df)} days")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        backtest_results = []
        lot_size = 25
        transaction_cost = 0.0025
        portfolio_pnl = 0

        for i in range(1, len(df)):
            day_data = df.iloc[i]
            prev_day = df.iloc[i-1]
            avg_vol = df["Realized_Vol"].iloc[max(0, i-5):i].mean()
            regime = "LOW" if avg_vol < 15 else "MEDIUM" if avg_vol < 20 else "HIGH"
            if day_data["Event_Flag"] == 1:
                regime = "EVENT-DRIVEN"

            strategy, reason, tags, deploy, max_loss, risk_reward = generate_strategy_logic(day_data, avg_vol, regime, capital)
            if strategy is None or (strategy_choice != "All Strategies" and strategy != strategy_choice):
                continue

            premium = day_data["Straddle_Price"] * lot_size
            lots = max(1, int(deploy / premium))
            pnl = premium * lots * (1 - transaction_cost) - np.random.uniform(0, premium * 0.2) * lots
            backtest_results.append({
                "Date": day_data.name,
                "Regime": regime,
                "Strategy": strategy,
                "PnL": pnl,
                "Capital_Deployed": deploy
            })
            portfolio_pnl += pnl

        backtest_df = pd.Dataa = pd.DataFrame(backtest_results)
        total_pnl = backtest_df["PnL"].sum()
        win_rate = len(backtest_df[backtest_df["PnL"] > 0]) / len(backtest_df) if len(backtest_df) > 0 else 0
        max_drawdown = (backtest_df["PnL"].cumsum().cummax() - backtest_df["PnL"].cumsum()).max() if len(backtest_df) > 0 else 0
        returns = backtest_df["PnL"] / capital
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0

        strategy_perf = backtest_df.groupby("Strategy")["PnL"].agg(['sum', 'count']).reset_index()
        regime_perf = backtest_df.groupby("Regime")["PnL"].agg(['ermit backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, 0, 0, strategy_perf, regime_perf
    except Exception as e:
        logger.error(f"Backtest error: {str(e)}")
        st.error(f"Backtest failed: {str(e)}")
        return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

# Strategy Generation
def generate_strategy_logic(day_data, avg_vol, regime, capital):
    try:
        iv_hv_gap = day_data["ATM_IV"] - day_data["Realized_Vol"]
        dte = day_data["Days_to_Expiry"]
        iv_skew = day_data["IV_Skew"]

        strategy = "Iron Condor"
        reason = "Default strategy"
        tags = ["Neutral"]
        risk_reward = 1.0
        deploy = capital * 0.2
        max_loss = deploy * 0.2

        if regime == "LOW":
            strategy = "Iron Fly" if dte < 10 else "Butterfly Spread"
            reason = "Low volatility favors theta strategies"
            tags = ["Theta", "Neutral"]
            risk_reward = 1.5
            deploy = capital * 0.3
        elif regime == "MEDIUM":
            strategy = "Short Strangle" if iv_hv_gap > 3 else "Iron Condor"
            reason = "Balanced volatility for premium selling"
            tags = ["Premium Selling"]
            risk_reward = 1.2
            deploy = capital * 0.25
        elif regime == "HIGH":
            strategy = "Jade Lizard" if iv_hv_gap > 5 else "Iron Condor"
            reason = "High volatility with defined risk"
            tags = ["Volatility"]
            risk_reward = 1.0
            deploy = capital * 0.15
        elif regime == "EVENT-DRIVEN":
            strategy = "Short Straddle" if dte < 5 else "Calendar Spread"
            reason = "Event-driven premium capture"
            tags = ["Event", "Neutral"]
            risk_reward = 1.3
            deploy = capital * 0.2

        max_loss = deploy * 0.2
        return strategy, reason, tags, deploy, max_loss, risk_reward
    except Exception as e:
        logger.error(f"Strategy generation error: {str(e)}")
        return None, None, [], 0, 0, 0

# Trading Functions
def place_trade(client, strategy, real_data, capital):
    try:
        if not real_data or "option_chain" not in real_data:
            return False, "Invalid real-time data"

        option_chain = real_data["option_chain"]
        atm_strike = real_data["atm_strike"]
        lot_size = 25
        deploy = strategy["Deploy"]
        premium = real_data["straddle_price"] * lot_size
        lots = max(1, min(int(deploy / premium), 5))

        orders = []
        if strategy["Strategy"] == "Iron Condor":
            orders = [
                (atm_strike + 100, "CE", "S"),
                (atm_strike + 200, "CE", "B"),
                (atm_strike - 100, "PE", "S"),
                (atm_strike - 200, "PE", "B")
            ]
        elif strategy["Strategy"] == "Short Strangle":
            orders = [(atm_strike + 100, "CE", "S"), (atm_strike - 100, "PE", "S")]
        else:
            return False, f"Strategy {strategy['Strategy']} not implemented"

        trade_details = []
        for strike, cp_type, buy_sell in orders:
            opt_data = option_chain[(option_chain["StrikeRate"] == strike) & (option_chain["CPType"] == cp_type)]
            if opt_data.empty:
                return False, f"No option data for {cp_type} at strike {strike}"
            scrip_code = int(opt_data["ScripCode"].iloc[0])
            price = float(opt_data["LastRate"].iloc[0])
            trade_details.append({
                "Strike": strike,
                "Type": cp_type,
                "Action": "Sell" if buy_sell == "S" else "Buy",
                "Price": price,
                "Quantity": lot_size * lots
            })
            orders.append({
                "Exchange": "N",
                "ExchangeType": "D",
                "ScripCode": scrip_code,
                "Quantity": lot_size * lots,
                "Price": 0,
                "OrderType": "SELL" if buy_sell == "S" else "BUY",
                "IsIntraday": False
            })

        st.session_state.trade_preview = trade_details
        return True, "Trade preview generated"
    except Exception as e:
        logger.error(f"Trade placement error: {str(e)}")
        return False, f"Trade failed: {str(e)}"

def confirm_trade(client):
    try:
        for order in st.session_state.get("trade_preview", []):
            response = client.place_order(
                OrderType=order["Action"].upper(),
                Exchange="N",
                ExchangeType="D",
                ScripCode=order["ScripCode"],
                Qty=order["Quantity"],
                Price=0,
                IsIntraday=False
            )
            if response.get("Status") != 0:
                return False, f"Order failed: {response.get('Message', 'Unknown error')}"
        st.session_state.trades.append({
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Strategy": st.session_state.strategy["Strategy"],
            "Details": st.session_state.trade_preview
        })
        return True, "Trade executed successfully"
    except Exception as e:
        logger.error(f"Trade confirmation error: {str(e)}")
        return False, f"Trade execution failed: {str(e)}"

# Sidebar Navigation
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=VolGuard+Pro", use_column_width=True)
    page = st.radio("Navigate", ["Dashboard", "Forecast", "Strategies", "Portfolio", "Backtest"], label_visibility="collapsed")
    st.markdown("---")

    st.subheader("Login")
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
        st.subheader("Settings")
        capital = st.number_input("Capital (₹)", min_value=100000, value=1000000, step=100000)
        risk_tolerance = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"])
        forecast_horizon = st.slider("Forecast Horizon (days)", 1, 30, 7)
        start_date = st.date_input("Backtest Start", pd.to_datetime("2024-01-01"))
        end_date = st.date_input("Backtest End", pd.to_datetime("2025-04-29"))
        strategy_choice = st.selectbox("Backtest Strategy", ["All Strategies", "Iron Condor", "Short Strangle", "Iron Fly", "Butterfly Spread", "Jade Lizard", "Short Straddle"])
        if st.button("Run Analysis"):
            st.session_state.analysis_run = False
            st.session_state.analysis_data = {}

# Main App
if not st.session_state.logged_in:
    st.info("Please login using the sidebar to access VolGuard Pro")
else:
    df, real_data, data_source = load_data(st.session_state.client)
    if df is not None:
        df = generate_features(df, real_data, capital)
        if df is not None:
            if st.session_state.analysis_run:
                forecast_log, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, feature_importances = forecast_volatility_future(df, forecast_horizon)
                strategy = generate_strategy_logic(df.iloc[-1], np.mean(blended_vols) if forecast_log is not None else 15, 
                                                "LOW" if df["VIX"].iloc[-1] < 15 else "MEDIUM" if df["VIX"].iloc[-1] < 20 else "HIGH", capital)
                backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, _, _, strategy_perf, regime_perf = run_backtest(
                    df, capital, strategy_choice, start_date, end_date
                )
                st.session_state.analysis_data = {
                    "forecast_log": forecast_log,
                    "realized_vol": realized_vol,
                    "strategy": strategy,
                    "backtest_df": backtest_df,
                    "total_pnl": total_pnl,
                    "win_rate": win_rate,
                    "max_drawdown": max_drawdown,
                    "sharpe_ratio": sharpe_ratio,
                    "strategy_perf": strategy_perf,
                    "regime_perf": regime_perf
                }

    if page == "Dashboard":
        st.title("📈 Market Dashboard")
        if df is not None:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("NIFTY 50", f"{df['NIFTY_Close'].iloc[-1]:,.2f}", 
                         f"{(df['NIFTY_Close'].iloc[-1] - df['NIFTY_Close'].iloc[-2])/df['NIFTY_Close'].iloc[-2]*100:+.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("India VIX", f"{df['VIX'].iloc[-1]:.2f}%", f"{df['VIX_Change_Pct'].iloc[-1]:+.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("PCR", f"{df['PCR'].iloc[-1]:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("ATM Straddle", f"₹{df['Straddle_Price'].iloc[-1]:,.2f}")
                st.markdown('</div>', unsafe_allow_html=True)

            if real_data and "option_chain" in real_data:
                st.subheader("Option Chain")
                chain_df = real_data["option_chain"][["StrikeRate", "CPType", "LastRate", "OpenInterest"]]
                st.dataframe(chain_df, use_container_width=True)

    elif page == "Forecast":
        st.title("🔮 Volatility Forecast")
        if st.session_state.analysis_run and st.session_state.analysis_data.get("forecast_log") is not None:
            forecast_log = st.session_state.analysis_data["forecast_log"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast_log["Date"], y=forecast_log["GARCH_Vol"], name="GARCH", line=dict(color="#1a73e8")))
            fig.add_trace(go.Scatter(x=forecast_log["Date"], y=forecast_log["XGBoost_Vol"], name="XGBoost", line=dict(color="#34a853")))
            fig.add_trace(go.Scatter(x=forecast_log["Date"], y=forecast_log["Blended_Vol"], name="Blended", line=dict(color="#fbbc05")))
            fig.update_layout(title="Volatility Forecast", xaxis_title="Date", yaxis_title="Volatility (%)", height=400)
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Avg Forecasted Vol", f"{np.mean(forecast_log['Blended_Vol']):.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Realized Vol", f"{st.session_state.analysis_data['realized_vol']:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)

    elif page == "Strategies":
        st.title("🎯 Trading Strategies")
        if st.session_state.analysis_run and st.session_state.analysis_data.get("strategy") is not None:
            strategy = st.session_state.analysis_data["strategy"]
            st.markdown('<div class="strategy-card">', unsafe_allow_html=True)
            st.markdown(f"""
                <h3>{strategy['Strategy']}</h3>
                <span class="regime-badge {strategy['Regime'].lower()}">{strategy['Regime']}</span>
                <p><b>Reason:</b> {strategy['Reason']}</p>
                <p><b>Capital:</b> ₹{strategy['Deploy']:,.0f}</p>
                <p><b>Max Loss:</b> ₹{strategy['Max_Loss']:,.0f}</p>
                <p><b>Tags:</b> {', '.join(strategy['Tags'])}</p>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("Preview Trade"):
                success, message = place_trade(st.session_state.client, strategy, real_data, capital)
                if success:
                    st.markdown('<div class="trade-modal">', unsafe_allow_html=True)
                    st.subheader("Trade Preview")
                    preview_df = pd.DataFrame(st.session_state.trade_preview)
                    st.dataframe(preview_df, use_container_width=True)
                    if st.button("Confirm Trade"):
                        confirm_success, confirm_message = confirm_trade(st.session_state.client)
                        if confirm_success:
                            st.success(confirm_message)
                        else:
                            st.error(confirm_message)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error(message)

    elif page == "Portfolio":
        st.title("💼 Portfolio")
        try:
            positions = st.session_state.client.positions()
            if positions:
                pos_df = pd.DataFrame(positions)
                st.subheader("Open Positions")
                st.dataframe(pos_df[["ScripCode", "Qty", "BuyAvgPrice", "SellAvgPrice", "NetQty", "MTM"]], use_container_width=True)
            else:
                st.info("No open positions")
        except Exception as e:
            st.error(f"Error fetching portfolio: {str(e)}")

    elif page == "Backtest":
        st.title("📉 Backtest Results")
        if st.session_state.analysis_run and st.session_state.analysis_data.get("backtest_df") is not None and not st.session_state.analysis_data["backtest_df"].empty:
            backtest_df = st.session_state.analysis_data["backtest_df"]
            fig = px.line(backtest_df, x=backtest_df.index, y=backtest_df["PnL"].cumsum(), title="Cumulative P&L")
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Total P&L", f"₹{st.session_state.analysis_data['total_pnl']:,.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Win Rate", f"{st.session_state.analysis_data['win_rate']*100:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Sharpe Ratio", f"{st.session_state.analysis_data['sharpe_ratio']:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Max Drawdown", f"₹{st.session_state.analysis_data['max_drawdown']:,.2f}")
                st.markdown('</div>', unsafe_allow_html=True)

            st.subheader("Strategy Performance")
            st.dataframe(st.session_state.analysis_data["strategy_perf"], use_container_width=True)
        else:
            st.info("No backtest results available. Run analysis to view results.")

    st.markdown('<div class="footer">VolGuard Pro © 2025 | Built by Shritish Shukla & Salman Azim</div>', unsafe_allow_html=True)
