import streamlit as st
import pandas as pd
import numpy as np
import time
import logging
import os
from datetime import datetime, timedelta
from py5paisa import FivePaisaClient
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page config
st.set_page_config(page_title="VolGuard Pro", layout="wide", page_icon="📈")

# CSS for styling
st.markdown("""
<style>
    .main { background-color: #f5f6f5; }
    .card { background-color: #ffffff; border-radius: 10px; padding: 20px; margin: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    .strategy-card { background-color: #f9f9f9; border-radius: 8px; padding: 15px; margin: 10px; width: 300px; display: inline-block; vertical-align: top; }
    .strategy-carousel { display: flex; overflow-x: auto; padding: 10px; }
    .regime-badge { padding: 5px 10px; border-radius: 12px; font-size: 12px; }
    .LOW { background-color: #d4edda; color: #155724; }
    .MEDIUM { background-color: #fff3cd; color: #856404; }
    .HIGH { background-color: #f8d7da; color: #721c24; }
    .EVENT-DRIVEN { background-color: #d1ecf1; color: #0c5460; }
    .alert-banner { background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; margin: 10px 0; }
    .safety-green { background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; }
    .safety-yellow { background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; }
    .safety-red { background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "client" not in st.session_state:
    st.session_state.client = None
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None
if "journal_complete" not in st.session_state:
    st.session_state.journal_complete = False
if "discipline_violations" not in st.session_state:
    st.session_state.discipline_violations = 0
if "last_regime" not in st.session_state:
    st.session_state.last_regime = None
if "cached_portfolio" not in st.session_state:
    st.session_state.cached_portfolio = {"weekly_pnl": 0, "margin_used": 0, "exposure": 0}
if "client_id" not in st.session_state:
    st.session_state.client_id = "YOUR_CLIENT_ID"  # Replace with your 5paisa Client ID
if "mpin" not in st.session_state:
    st.session_state.mpin = "YOUR_MPIN"  # Replace with your 5paisa MPIN

# Data fetching functions
@st.cache_data
def load_data():
    try:
        nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
        vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"
        nifty_df = pd.read_csv(nifty_url)
        vix_df = pd.read_csv(vix_url)
        nifty_df["Date"] = pd.to_datetime(nifty_df["Date"])
        vix_df["Date"] = pd.to_datetime(vix_df["Date"])
        df = nifty_df.merge(vix_df[["Date", "VIX"]], on="Date", how="left")
        df["VIX"] = df["VIX"].fillna(method="ffill")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error("Failed to load historical data.")
        return pd.DataFrame()

def fetch_nifty_data(client):
    try:
        spot_data = client.get_market_data("NIFTY 50")
        option_chain = client.get_option_chain("NIFTY", "INDEX")
        atm_strike = round(spot_data["Close"] / 100) * 100
        atm_options = option_chain[option_chain["StrikeRate"] == atm_strike]
        call_iv = atm_options[atm_options["CPType"] == "CE"]["IV"].iloc[0] if not atm_options[atm_options["CPType"] == "CE"].empty else 15
        put_iv = atm_options[atm_options["CPType"] == "PE"]["IV"].iloc[0] if not atm_options[atm_options["CPType"] == "PE"].empty else 15
        atm_iv = (call_iv + put_iv) / 2
        straddle_price = atm_options["LastRate"].sum()
        put_oi = atm_options[atm_options["CPType"] == "PE"]["OpenInterest"].iloc[0] if not atm_options[atm_options["CPType"] == "PE"].empty else 1
        call_oi = atm_options[atm_options["CPType"] == "CE"]["OpenInterest"].iloc[0] if not atm_options[atm_options["CPType"] == "CE"].empty else 1
        pcr = put_oi / call_oi if call_oi != 0 else 1
        prev_atm_iv = atm_iv  # Placeholder; ideally fetch previous day's IV
        vix_change_pct = ((atm_iv - prev_atm_iv) / prev_atm_iv * 100) if prev_atm_iv != 0 else 0
        return {
            "spot": spot_data["Close"],
            "vix": spot_data["VIX"],
            "atm_strike": atm_strike,
            "atm_iv": atm_iv,
            "straddle_price": straddle_price,
            "pcr": pcr,
            "vix_change_pct": vix_change_pct,
            "option_chain": option_chain
        }
    except Exception as e:
        logger.error(f"Error fetching nifty data: {str(e)}")
        return None

def generate_synthetic_features(df, real_data=None):
    df = df.copy()
    n_days = len(df)
    base_iv = real_data["atm_iv"] if real_data else df["VIX"].iloc[-1]
    base_pcr = real_data["pcr"] if real_data else 1.0
    base_vix_change_pct = real_data["vix_change_pct"] if real_data else 0
    market_trend = np.linspace(0, 0.1, n_days)
    event_spike = np.where(df["Date"].dt.day % 90 == 0, 1.2, 1)
    df["ATM_IV"] = df["VIX"] * (1 + np.random.normal(0, 0.1, n_days)) * event_spike
    df["ATM_IV"] = np.clip(df["ATM_IV"], 5, 50)
    df["IVP"] = df["ATM_IV"].rolling(252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False).fillna(0.5)
    df["PCR"] = np.clip(base_pcr + np.random.normal(0, 0.1, n_days) + market_trend * -10, 0.7, 2.0)
    df["VIX_Change_Pct"] = df["VIX"].pct_change().fillna(0) * 100
    df["IV_Skew"] = np.random.normal(0, 0.5, n_days) + df["VIX"] * 0.01
    df["Straddle_Price"] = real_data["straddle_price"] if real_data else df["VIX"] * 10
    df["Spot_MaxPain_Diff_Pct"] = np.random.lognormal(0, 0.1, n_days) * 0.5
    df["Days_to_Expiry"] = (df["Date"] + pd.offsets.MonthEnd(0) - df["Date"]).dt.days
    df["Event_Flag"] = np.where((df["Days_to_Expiry"] < 3) | (df["Date"].dt.day % 90 == 0), 1, 0)
    df["FII_Index_Fut_Pos"] = np.cumsum(np.random.normal(0, 1000, n_days))
    df["FII_Option_Pos"] = np.cumsum(np.random.normal(0, 1000, n_days))
    if real_data:
        df["ATM_IV"].iloc[-1] = base_iv
        df["PCR"].iloc[-1] = base_pcr
        df["VIX_Change_Pct"].iloc[-1] = base_vix_change_pct
        df["Straddle_Price"].iloc[-1] = real_data["straddle_price"]
    return df

def fetch_portfolio_data(client, capital):
    try:
        positions = client.positions()
        if not positions:
            raise Exception("Failed to fetch positions")
        total_pnl = sum(position.get("ProfitLoss", 0) for position in positions)
        total_margin = sum(position.get("MarginUsed", 0) for position in positions)
        total_exposure = sum(position.get("Exposure", 0) for position in positions) / capital * 100
        portfolio_data = {
            "weekly_pnl": total_pnl,
            "margin_used": total_margin,
            "exposure": total_exposure,
            "positions": positions
        }
        st.session_state.cached_portfolio = portfolio_data  # Cache portfolio
        # Auto Stop-Loss Check
        for position in positions:
            if position.get("ProfitLoss", 0) < -position.get("MaxLoss", 0):
                client.close_position(position["ScripCode"], position["Qty"])
                st.warning(f"Auto Stop-Loss triggered for {position['ScripCode']}")
        return portfolio_data
    except Exception as e:
        logger.error(f"Error fetching portfolio data: {str(e)}")
        return st.session_state.cached_portfolio  # Return cached data

# Forecasting and strategy functions
def forecast_volatility_future(df, forecast_horizon=7):
    df = df.copy()
    df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Realized_Vol"] = df["Log_Returns"].rolling(5).std() * np.sqrt(252) * 100
    df_garch = df[["Log_Returns"]].dropna()
    if len(df_garch) < 200:
        return pd.DataFrame(), 0, 0
    model = arch_model(df_garch["Log_Returns"] * 100, vol="Garch", p=1, q=1, dist="normal")
    garch_fit = model.fit(disp="off")
    garch_forecast = garch_fit.forecast(horizon=forecast_horizon, method="simulation")
    garch_vol = np.sqrt(garch_forecast.variance.values[-1, :]) * np.sqrt(252) / 100
    garch_vol = np.clip(garch_vol, 5, 50)
    feature_cols = ['VIX', 'ATM_IV', 'IVP', 'PCR', 'VIX_Change_Pct', 'IV_Skew', 'Straddle_Price',
                    'Spot_MaxPain_Diff_Pct', 'Days_to_Expiry', 'Event_Flag', 'FII_Index_Fut_Pos', 'FII_Option_Pos']
    X = df[feature_cols].dropna()
    y = df["Realized_Vol"].shift(-1).dropna()
    X = X.iloc[:len(y)]
    if len(X) < 10:
        return pd.DataFrame(), 0, 0
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.03, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    forecast_log = []
    current_row = X.iloc[-1:].copy()
    for i in range(forecast_horizon):
        current_scaled = scaler.transform(current_row[feature_cols])
        pred_vol = model.predict(current_scaled)[0]
        pred_vol = np.clip(pred_vol, 5, 50)
        if current_row["Event_Flag"].iloc[0] == 1:
            pred_vol *= 1.1
        forecast_log.append({"Date": df["Date"].iloc[-1] + timedelta(days=i+1), "XGBoost_Vol": pred_vol, "GARCH_Vol": garch_vol[i]})
        current_row["VIX"] *= np.random.uniform(0.98, 1.02)
        current_row["ATM_IV"] = current_row["VIX"] * (1 + np.random.normal(0, 0.1))
        current_row["ATM_IV"] = np.clip(current_row["ATM_IV"], 5, 50)
        current_row["IVP"] = current_row["ATM_IV"].rolling(252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False).fillna(0.5)
        current_row["PCR"] = np.clip(current_row["PCR"] + np.random.normal(0, 0.1), 0.7, 2.0)
        current_row["VIX_Change_Pct"] = (current_row["VIX"] - current_row["VIX"].shift(1)) / current_row["VIX"].shift(1) * 100
        current_row["VIX_Change_Pct"] = current_row["VIX_Change_Pct"].fillna(0)
        current_row["IV_Skew"] = np.random.normal(0, 0.5) + current_row["VIX"] * 0.01
        current_row["Straddle_Price"] = current_row["VIX"] * 10
        current_row["Spot_MaxPain_Diff_Pct"] = np.random.lognormal(0, 0.1) * 0.5
        current_row["Days_to_Expiry"] = max(current_row["Days_to_Expiry"] - 1, 0)
        current_row["Event_Flag"] = 1 if current_row["Days_to_Expiry"] < 3 else 0
        current_row["FII_Index_Fut_Pos"] += np.random.normal(0, 1000)
        current_row["FII_Option_Pos"] += np.random.normal(0, 1000)
    forecast_log = pd.DataFrame(forecast_log)
    realized_vol = df["Realized_Vol"].iloc[-1]
    garch_diff = np.abs(garch_vol.mean() - realized_vol)
    xgb_diff = np.abs(forecast_log["XGBoost_Vol"].mean() - realized_vol)
    garch_weight = max(0.3, 1 - garch_diff / (garch_diff + xgb_diff + 1e-10))
    xgb_weight = 1 - garch_weight
    forecast_log["Volatility"] = garch_weight * forecast_log["GARCH_Vol"] + xgb_weight * forecast_log["XGBoost_Vol"]
    confidence = max(50, 80 - abs(garch_diff - xgb_diff))
    return forecast_log, rmse, confidence

def generate_trading_strategy(df, forecast_log, capital, risk_profile, dte_preference, intraday_mode=False):
    strategies = []
    vix = df["VIX"].iloc[-1]
    ivp = df["IVP"].iloc[-1]
    pcr = df["PCR"].iloc[-1]
    regime = "LOW" if vix < 15 else "MEDIUM" if vix < 18 else "HIGH" if vix < 25 else "EVENT-DRIVEN"
    forecast_vol = forecast_log["Volatility"].mean() if not forecast_log.empty else vix
    risk_flag = False
    safety_class = "safety-green"
    if vix > 25:
        risk_flag = True
        safety_class = "safety-red"
        st.session_state.discipline_violations += 1
    if df["Exposure"].iloc[-1] > 70:
        risk_flag = True
        safety_class = "safety-yellow"
        st.session_state.discipline_violations += 1
    if df["Weekly_Loss"].iloc[-1] > 5:
        risk_flag = True
        safety_class = "safety-yellow"
        st.session_state.discipline_violations += 1
    if intraday_mode and (vix <= 18 or df["Event_Flag"].iloc[-1] == 0 or not st.session_state.journal_complete):
        risk_flag = True
        safety_class = "safety-red"
        st.session_state.discipline_violations += 1
    base_strategies = [
        {"Strategy": "Iron Condor", "Regime": ["LOW", "MEDIUM"], "Confidence": 0.85, "Risk_Reward": 0.3, "Max_Loss": capital * 0.05, "Deploy": capital * 0.2, "Reason": "Stable volatility, wide breakevens"},
        {"Strategy": "Short Strangle", "Regime": ["LOW"], "Confidence": 0.80, "Risk_Reward": 0.25, "Max_Loss": capital * 0.07, "Deploy": capital * 0.15, "Reason": "Low volatility, high premium capture"},
        {"Strategy": "Jade Lizard", "Regime": ["MEDIUM", "HIGH"], "Confidence": 0.75, "Risk_Reward": 0.4, "Max_Loss": capital * 0.04, "Deploy": capital * 0.25, "Reason": "Skewed IV, directional bias"},
        {"Strategy": "Calendar Spread", "Regime": ["MEDIUM"], "Confidence": 0.70, "Risk_Reward": 0.35, "Max_Loss": capital * 0.03, "Deploy": capital * 0.2, "Reason": "Time decay advantage"},
        {"Strategy": "Ratio Backspread", "Regime": ["HIGH", "EVENT-DRIVEN"], "Confidence": 0.65, "Risk_Reward": 0.5, "Max_Loss": capital * 0.06, "Deploy": capital * 0.3, "Reason": "High volatility, asymmetric payoff"}
    ]
    for strat in base_strategies:
        if regime in strat["Regime"] and (not intraday_mode or strat["Strategy"] in ["Jade Lizard", "Ratio Backspread"]):
            if intraday_mode:
                strat["Deploy"] = min(strat["Deploy"], capital * 0.2)
                strat["Max_Loss"] = strat["Deploy"] * 0.5
            if risk_profile == "Conservative" and strat["Max_Loss"] > capital * 0.05:
                continue
            if risk_profile == "Aggressive" and strat["Risk_Reward"] < 0.4:
                continue
            if dte_preference < 15 and strat["Strategy"] in ["Calendar Spread"]:
                continue
            strategies.append(strat)
    strategies = sorted(strategies, key=lambda x: x["Confidence"], reverse=True)[:5]
    return strategies, risk_flag, regime, safety_class

def run_backtest(df, capital, strategy_choice, start_date, end_date):
    backtest_df = pd.DataFrame(columns=["Date", "Regime", "Strategy", "PnL", "Capital_Deployed"])
    total_pnl = 0
    wins = 0
    max_drawdown = 0
    strategy_perf = {}
    regime_perf = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "EVENT-DRIVEN": 0}
    df_backtest = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()
    for i, row in df_backtest.iterrows():
        vix = row["VIX"]
        regime = "LOW" if vix < 15 else "MEDIUM" if vix < 18 else "HIGH" if vix < 25 else "EVENT-DRIVEN"
        strategies, _, _, _ = generate_trading_strategy(df.iloc[:i+1], pd.DataFrame(), capital, "Moderate", 15)
        selected_strat = next((s for s in strategies if strategy_choice == "All Strategies" or s["Strategy"] == strategy_choice), None)
        if selected_strat:
            deploy = selected_strat["Deploy"]
            max_loss = selected_strat["Max_Loss"]
            premium = row["Straddle_Price"] * 0.1
            slippage = 0.02 * premium if vix > 20 else 0.01 * premium
            txn_cost = deploy * 0.002
            move = row["Close"] - df["Close"].iloc[i-1] if i > 0 else 0
            breakeven = premium * 2
            pnl = premium - max(0, abs(move) - breakeven) - slippage - txn_cost
            if row["Event_Flag"] == 1:
                pnl *= 0.8
            total_pnl += pnl
            if pnl > 0:
                wins += 1
            capital_deployed = deploy
            backtest_df = pd.concat([backtest_df, pd.DataFrame([{
                "Date": row["Date"], "Regime": regime, "Strategy": selected_strat["Strategy"],
                "PnL": pnl, "Capital_Deployed": capital_deployed
            }])], ignore_index=True)
            strategy_perf[selected_strat["Strategy"]] = strategy_perf.get(selected_strat["Strategy"], 0) + pnl
            regime_perf[regime] += pnl
    win_rate = wins / len(backtest_df) if len(backtest_df) > 0 else 0
    returns = backtest_df["PnL"].cumsum() / capital
    max_drawdown = (returns.cummax() - returns).max() if len(returns) > 0 else 0
    sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
    sortino_ratio = (returns.mean() * 252) / (returns[returns < 0].std() * np.sqrt(252)) if returns[returns < 0].std() != 0 else 0
    calmar_ratio = (returns.mean() * 252) / max_drawdown if max_drawdown != 0 else 0
    return (backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio,
            pd.DataFrame(list(strategy_perf.items()), columns=["Strategy", "PnL"]),
            pd.DataFrame(list(regime_perf.items()), columns=["Regime", "PnL"]))

# Main app
def main():
    st.title("VolGuard Pro 🚀")
    st.markdown("Your AI-powered copilot for NIFTY 50 option selling.")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        # TOTP-only login
        totp = st.text_input("TOTP", type="password")
        if st.button("Login"):
            try:
                client_id = st.session_state.client_id
                mpin = st.session_state.mpin
                if client_id == "YOUR_CLIENT_ID" or mpin == "YOUR_MPIN":
                    st.error("Please set valid Client ID and MPIN in st.session_state.")
                else:
                    st.session_state.client = Client(client_id, totp, mpin)
                    st.session_state.logged_in = True
                    st.success("✅ Logged in successfully")
            except Exception as e:
                st.error(f"Login failed: {str(e)}")
                st.session_state.logged_in = False
        # One-time credential setup
        if not st.session_state.logged_in and (st.session_state.client_id == "YOUR_CLIENT_ID" or st.session_state.mpin == "YOUR_MPIN"):
            st.markdown("### Set Credentials (One-Time)")
            new_client_id = st.text_input("Client ID")
            new_mpin = st.text_input("MPIN", type="password")
            if st.button("Save Credentials"):
                if new_client_id and new_mpin:
                    st.session_state.client_id = new_client_id
                    st.session_state.mpin = new_mpin
                    st.success("Credentials saved. Enter TOTP to login.")
                else:
                    st.error("Client ID and MPIN cannot be empty.")
        capital = st.slider("Capital (₹)", 100000, 1000000, 500000, 50000)
        risk_profile = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"])
        dte_preference = st.slider("DTE Preference (days)", 7, 30, 15)
        forecast_horizon = st.slider("Forecast Horizon (days)", 7, 30, 7)
        start_date = st.date_input("Backtest Start Date", datetime.now() - timedelta(days=365))
        end_date = st.date_input("Backtest End Date", datetime.now())
        strategy_choice = st.selectbox("Strategy", ["All Strategies", "Iron Condor", "Short Strangle", "Jade Lizard", "Calendar Spread", "Ratio Backspread"])
        intraday_mode = st.checkbox("Intraday Mode", disabled=(st.session_state.client is None or not st.session_state.journal_complete))
        run_button = st.button("Run Analysis")

    # Load data
    df = load_data()
    real_data = fetch_nifty_data(st.session_state.client) if st.session_state.logged_in else None
    if real_data:
        df = df[df["Date"] < datetime.now().date()]
        latest_data = pd.DataFrame([{
            "Date": datetime.now(), "Close": real_data["spot"], "VIX": real_data["vix"]
        }])
        df = pd.concat([df, latest_data], ignore_index=True)
    df = generate_synthetic_features(df, real_data)
    df["Weekly_Loss"] = df["Close"].pct_change(5).fillna(0) * -100
    df["Exposure"] = np.random.uniform(0, 100, len(df))

    # Tabs
    tabs = st.tabs(["Snapshot", "Forecast", "Strategy", "Portfolio", "Journal", "Backtest"])

    # Snapshot Tab
    with tabs[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Market Snapshot")
        if real_data:
            st.markdown(f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (LIVE)")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("NIFTY 50", f"{real_data['spot']:,.2f}")
            col2.metric("VIX", f"{real_data['vix']:.2f}", f"{real_data['vix_change_pct']:.2f}%")
            pcr_color = "green" if real_data['pcr'] < 1 else "yellow" if real_data['pcr'] <= 1.5 else "red"
            col3.metric("PCR", f"{real_data['pcr']:.2f}", delta_color=pcr_color)
            implied_move = real_data['straddle_price'] / real_data['spot'] * 100
            col4.metric("Straddle Price", f"₹{real_data['straddle_price']:,.2f}", f"{implied_move:.2f}% move")
            st.markdown("### Option Chain (ATM ±2 Strikes)")
            atm_strike = real_data["atm_strike"]
            strikes = real_data["option_chain"][real_data["option_chain"]["StrikeRate"].isin([atm_strike-200, atm_strike-100, atm_strike, atm_strike+100, atm_strike+200])]
            strikes["OI_Change"] = strikes["OpenInterest"].pct_change().fillna(0) * 100
            st.dataframe(strikes[["StrikeRate", "CPType", "LastRate", "OpenInterest", "IV", "OI_Change"]], use_container_width=True)
        else:
            st.markdown("**Last Updated**: Using cached data (DEMO)")
            st.warning("Live data unavailable. Showing cached metrics.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Forecast Tab
    with tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔮 Volatility Forecast")
        forecast_log, rmse, confidence = forecast_volatility_future(df, forecast_horizon)
        if not forecast_log.empty:
            st.line_chart(forecast_log.set_index("Date")[["Volatility", "GARCH_Vol", "XGBoost_Vol"]], color=["#e94560", "#0a9396", "#005f73"])
            col1, col2, col3 = st.columns(3)
            col1.metric("Model RMSE", f"{rmse:.2f}")
            col2.metric("Confidence Score", f"{confidence:.0f}%", delta_color="green" if confidence > 75 else "yellow" if confidence > 50 else "red")
            vix_range = f"{forecast_log['Volatility'].min():.2f}–{forecast_log['Volatility'].max():.2f}"
            col3.metric("VIX Range (7d)", vix_range)
        else:
            st.warning("Insufficient data for volatility forecast.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Strategy Tab
    with tabs[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎯 Trading Strategy")
        if run_button:
            strategies, risk_flag, regime, safety_class = generate_trading_strategy(df, forecast_log, capital, risk_profile, dte_preference, intraday_mode)
            if st.session_state.discipline_violations >= 2 and not st.session_state.journal_complete:
                st.error("🚫 Strategy panel locked. Complete journaling to unlock.")
            elif strategies:
                if st.session_state.last_regime and st.session_state.last_regime != regime:
                    st.markdown(f'<div class="alert-banner">⚠️ Regime Shift Detected: {st.session_state.last_regime} → {regime}</div>', unsafe_allow_html=True)
                st.session_state.last_regime = regime
                st.markdown(f'<div class="{safety_class}">Trade Safety: {"Safe" if safety_class == "safety-green" else "Warning" if safety_class == "safety-yellow" else "Blocked"}</div>', unsafe_allow_html=True)
                st.markdown('<div class="strategy-carousel">', unsafe_allow_html=True)
                for strat in strategies:
                    regime_class = strat["Regime"][0]
                    st.markdown(f"""
                        <div class="strategy-card">
                            <h4>{strat['Strategy']}</h4>
                            <span class="regime-badge {regime_class}">{regime}</span>
                            <p><b>Reason:</b> {strat['Reason']}</p>
                            <p><b>Confidence:</b> {strat['Confidence']:.2f}</p>
                            <p><b>Risk-Reward:</b> {strat['Risk_Reward']:.2f}:1</p>
                            <p><b>Capital:</b> ₹{strat['Deploy']:,.0f}</p>
                            <p><b>Max Loss:</b> ₹{strat['Max_Loss']:,.0f}</p>
                    """, unsafe_allow_html=True)
                    if not risk_flag and st.session_state.logged_in:
                        if st.button("Trade Now", key=strat["Strategy"]):
                            try:
                                legs = [
                                    {"ScripCode": f"NIFTY-{strat['Strategy']}-CE", "Qty": 25, "Price": 100, "BuySell": "SELL"},
                                    {"ScripCode": f"NIFTY-{strat['Strategy']}-PE", "Qty": 25, "Price": 100, "BuySell": "SELL"},
                                    {"ScripCode": f"NIFTY-{strat['Strategy']}-CE-HEDGE", "Qty": 25, "Price": 50, "BuySell": "BUY"},
                                    {"ScripCode": f"NIFTY-{strat['Strategy']}-PE-HEDGE", "Qty": 25, "Price": 50, "BuySell": "BUY"}
                                ]
                                for leg in legs:
                                    st.session_state.client.place_order(**leg)
                                trade_log = pd.DataFrame([{
                                    "Date": datetime.now(), "Strategy": strat["Strategy"], "Regime": regime,
                                    "Risk_Level": risk_profile, "Outcome": "Pending"
                                }])
                                trade_log.to_csv("trade_log.csv", mode="a", index=False, header=not os.path.exists("trade_log.csv"))
                                st.success(f"Trade placed: {strat['Strategy']}")
                            except Exception as e:
                                st.error(f"Trade failed: {str(e)}")
                    st.markdown("</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("No suitable strategies found.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Portfolio Tab
    with tabs[3]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("💼 Portfolio")
        if st.session_state.logged_in:
            portfolio_data = fetch_portfolio_data(st.session_state.client, capital)
            col1, col2, col3 = st.columns(3)
            col1.metric("Weekly P&L", f"₹{portfolio_data['weekly_pnl']:,.2f}", delta_color="green" if portfolio_data['weekly_pnl'] >= 0 else "red")
            col2.metric("Margin Used", f"₹{portfolio_data['margin_used']:,.2f}", delta_color="yellow" if portfolio_data['margin_used'] > capital * 0.5 else "green")
            col3.metric("Exposure", f"{portfolio_data['exposure']:.2f}%", delta_color="red" if portfolio_data['exposure'] > 70 else "green")
            if portfolio_data.get("positions"):
                st.markdown("### Open Positions")
                positions_df = pd.DataFrame(portfolio_data["positions"])
                st.dataframe(positions_df[["ScripCode", "Qty", "AvgPrice", "LastPrice", "ProfitLoss", "MaxLoss"]], use_container_width=True)
        else:
            st.warning("Login to view portfolio.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Journal Tab
    with tabs[4]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📝 Discipline Hub")
        journal_entry = st.text_area("Why this strategy? Did you override warnings?")
        if st.button("Submit Journal"):
            if journal_entry:
                discipline_score = min(10, 8 + len(journal_entry.split()) // 10 - st.session_state.discipline_violations)
                journal_log = pd.DataFrame([{
                    "Date": datetime.now(), "Entry": journal_entry, "Discipline_Score": discipline_score
                }])
                journal_log.to_csv("journal_log.csv", mode="a", index=False, header=not os.path.exists("journal_log.csv"))
                st.session_state.journal_complete = True
                st.session_state.discipline_violations = 0
                st.success("Journal submitted. Discipline panel unlocked.")
            else:
                st.error("Journal entry cannot be empty.")
        if os.path.exists("journal_log.csv"):
            journal_df = pd.read_csv("journal_log.csv")
            st.markdown("### Discipline Trend")
            st.line_chart(journal_df.set_index("Date")["Discipline_Score"], color="#e94560")
            st.markdown("### Journal Entries")
            st.dataframe(journal_df[["Date", "Entry", "Discipline_Score"]], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Backtest Tab
    with tabs[5]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📉 Backtest Results")
        if run_button:
            backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf = run_backtest(df, capital, strategy_choice, start_date, end_date)
            if not backtest_df.empty:
                col1, col2, col3 = st.columns(3)
                col1.metric("Total PnL", f"₹{total_pnl:,.2f}", delta_color="green" if total_pnl >= 0 else "red")
                col2.metric("Win Rate", f"{win_rate*100:.2f}%", delta_color="green" if win_rate > 0.6 else "yellow")
                col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}", delta_color="green" if sharpe_ratio > 1 else "yellow")
                st.line_chart(backtest_df["PnL"].cumsum(), color="#e94560")
                st.markdown("### Strategy Performance")
                st.dataframe(strategy_perf, use_container_width=True)
                st.markdown("### Regime Performance")
                st.dataframe(regime_perf, use_container_width=True)
            else:
                st.warning("No trades generated for the selected strategy and period.")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
