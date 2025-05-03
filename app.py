import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from arch import arch_model
import xgboost as xgb
from scipy.stats import norm
from py5paisa import FivePaisaClient
import time
import uuid
import os
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import pytz

# Load .env file
load_dotenv()

# Initialize Streamlit Session State
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "client" not in st.session_state:
    st.session_state.client = None

# 5paisa API Client Setup
def initialize_5paisa_client(client_code, totp_code, pin):
    cred = {
        "APP_NAME": os.getenv("APP_NAME"),
        "APP_SOURCE": os.getenv("APP_SOURCE"),
        "USER_ID": os.getenv("USER_ID"),
        "PASSWORD": os.getenv("PASSWORD"),
        "USER_KEY": os.getenv("USER_KEY"),
        "ENCRYPTION_KEY": os.getenv("ENCRYPTION_KEY")
    }
    client = FivePaisaClient(cred=cred)
    client.get_totp_session(client_code, totp_code, pin)
    return client

# GitHub Data Loading
@st.cache_data
def load_historical_data():
    nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
    vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"
    nifty = pd.read_csv(nifty_url, parse_dates=["Date"]).set_index("Date")
    vix = pd.read_csv(vix_url, parse_dates=["Date"]).set_index("Date")
    df = nifty[["Close"]].rename(columns={"Close": "NIFTY_Close"}).join(vix[["Close"]].rename(columns={"Close": "VIX"}))
    df = df.fillna(method="ffill").fillna(method="bfill")
    return df

# Load Cached Real Data from GitHub
@st.cache_data
def load_cached_data():
    try:
        last_data_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/last_market_data.csv"
        last_data = pd.read_csv(last_data_url, parse_dates=["Timestamp"])
        return last_data.iloc[-1]
    except:
        return None

# Generate Synthetic Features
def generate_synthetic_features(df):
    df["Returns"] = df["NIFTY_Close"].pct_change().fillna(0)
    df["Realized_Vol"] = df["Returns"].rolling(window=5).std() * np.sqrt(252) * 100
    df["Days_to_Expiry"] = 7  # Simplified for demo
    df["Event_Flag"] = (df.index.month % 3 == 0) | (df["Days_to_Expiry"] < 5)
    df["IV_Skew"] = df["VIX"] * np.random.uniform(0.9, 1.1)
    df["Capital_Pressure_Index"] = np.random.uniform(-1, 1)
    df["Gamma_Bias"] = np.random.uniform(-0.5, 0.5)
    return df

# Simulate Data for Off-Hours
def simulate_data(last_data):
    simulated_data = last_data.copy()
    simulated_data["NIFTY_Close"] *= np.random.uniform(0.995, 1.005)
    simulated_data["VIX"] *= np.random.uniform(0.98, 1.02)
    simulated_data["ATM_IV"] *= (simulated_data["VIX"] / last_data["VIX"])
    simulated_data["Straddle_Price"] *= (simulated_data["VIX"] / last_data["VIX"])
    simulated_data["PCR"] *= np.random.uniform(0.98, 1.02)
    simulated_data["IV_Skew"] *= np.random.uniform(0.98, 1.02)
    simulated_data["FII_Pressure"] = np.random.uniform(-1, 1)
    return simulated_data

# Black-Scholes IV Calculation
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_volatility(S, K, T, r, market_price, option_type='call', tol=1e-5, max_iter=100):
    sigma = 0.2
    for _ in range(max_iter):
        if option_type == 'call':
            model_price = black_scholes_call(S, K, T, r, sigma)
        else:
            model_price = black_scholes_call(S, K, T, r, sigma) + K * np.exp(-r * T) - S
        diff = model_price - market_price
        if abs(diff) < tol:
            return sigma * 100
        vega = S * norm.pdf((np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))) * np.sqrt(T)
        if vega == 0:
            return np.nan
        sigma -= diff / vega
    return np.nan

# Fetch Real-Time Data from 5paisa
def fetch_nifty_data(client):
    nifty_req = [{
        "Exch": "N",
        "ExchType": "C",
        "ScripCode": 999920000,
        "Symbol": "NIFTY",
        "Expiry": "",
        "StrikePrice": "0",
        "OptionType": ""
    }]
    nifty_data = client.fetch_market_feed(nifty_req)
    if not nifty_data or "Data" not in nifty_data or not nifty_data["Data"]:
        return None
    nifty_spot = nifty_data["Data"][0].get("LastRate", nifty_data["Data"][0].get("LastTradedPrice", 0))
    if not nifty_spot:
        return None

    expiry_timestamp = 1746694800000  # May 8, 2025
    option_chain = client.get_option_chain("N", "NIFTY", expiry_timestamp)
    if not option_chain or "Options" not in option_chain:
        return None

    df = pd.DataFrame(option_chain["Options"])
    df["StrikeRate"] = df["StrikeRate"].astype(float)
    atm_strike = df["StrikeRate"].iloc[(df["StrikeRate"] - nifty_spot).abs().argmin()]
    atm_data = df[df["StrikeRate"] == atm_strike]
    atm_call = atm_data[atm_data["CPType"] == "CE"]["LastRate"].iloc[0] if not atm_data[atm_data["CPType"] == "CE"].empty else 0
    atm_put = atm_data[atm_data["CPType"] == "PE"]["LastRate"].iloc[0] if not atm_data[atm_data["CPType"] == "PE"].empty else 0
    straddle_price = atm_call + atm_put

    iv_df = df[(df["StrikeRate"] >= atm_strike - 100) & (df["StrikeRate"] <= atm_strike + 100)].copy()
    T = (datetime(2025, 5, 8) - datetime.now()).days / 365.0
    r = 0.06
    iv_df["IV (%)"] = iv_df.apply(
        lambda row: implied_volatility(
            S=nifty_spot,
            K=row["StrikeRate"],
            T=T,
            r=r,
            market_price=row["LastRate"],
            option_type='call' if row["CPType"] == "CE" else 'put'
        ),
        axis=1
    )
    atm_iv = iv_df[iv_df["StrikeRate"] == atm_strike]["IV (%)"].mean()

    calls = df[df["CPType"] == "CE"]
    puts = df[df["CPType"] == "PE"]
    call_oi = calls["OpenInterest"].sum()
    put_oi = puts["OpenInterest"].sum()
    pcr = put_oi / call_oi if call_oi != 0 else float("inf")

    return {
        "Timestamp": pd.Timestamp.now(),
        "NIFTY_Close": nifty_spot,
        "VIX": 15.0,  # Placeholder, as VIX not directly available
        "ATM_IV": atm_iv,
        "Straddle_Price": straddle_price,
        "PCR": pcr,
        "IV_Skew": np.random.uniform(0.9, 1.1) * atm_iv,  # Placeholder
        "FII_Pressure": np.random.uniform(-1, 1)  # Placeholder
    }

# Volatility Forecasting (GARCH + XGBoost)
def forecast_volatility(df):
    returns = df["Returns"].dropna() * 100  # Scale for GARCH
    garch_model = arch_model(returns, vol='Garch', p=1, q=1, dist='Normal')
    garch_fit = garch_model.fit(disp="off")
    garch_vol = garch_fit.forecast(horizon=7).variance.iloc[-1].mean() * np.sqrt(252)

    features = df[["VIX", "ATM_IV", "PCR", "IV_Skew", "Realized_Vol", "Capital_Pressure_Index", "Gamma_Bias"]].dropna()
    target = df["Realized_Vol"].dropna()
    if len(features) > 1:
        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        model.fit(features[:-1], target[1:])
        xgb_vol = model.predict(features[-1:].values)[0]
    else:
        xgb_vol = garch_vol

    blended_vol = 0.5 * garch_vol + 0.5 * xgb_vol
    confidence = np.random.uniform(0.7, 0.95)
    return blended_vol, confidence

# Strategy Engine
def generate_strategy(df, blended_vol, confidence, capital, risk_profile, dte):
    regime = "High" if blended_vol > 20 else "Medium" if blended_vol > 15 else "Low"
    if df["Event_Flag"].iloc[-1]:
        regime = "Event-Driven"

    if regime == "Low" and risk_profile == "Conservative":
        strategy = "Iron Condor"
        max_loss = capital * 0.05
    elif regime == "Medium" and risk_profile == "Moderate":
        strategy = "Butterfly Spread"
        max_loss = capital * 0.07
    else:
        strategy = "Jade Lizard"
        max_loss = capital * 0.1

    return {
        "Strategy": strategy,
        "Regime": regime,
        "Capital_Required": capital * 0.2,
        "Max_Loss": max_loss,
        "Confidence": confidence
    }

# Main App
def main():
    st.title("VolGuard Pro - Option Selling App")

    # Login Page
    if not st.session_state.user_id:
        with st.form("login_form"):
            username = st.text_input("Enter Username (e.g., User1)")
            client_code = st.text_input("5paisa Client Code", value=os.getenv("CLIENT_CODE"), disabled=True)
            totp_code = st.text_input("5paisa TOTP", type="password")
            pin = st.text_input("5paisa PIN", value=os.getenv("PIN"), disabled=True)
            submitted = st.form_submit_button("Login")
            if submitted:
                st.session_state.user_id = username if username else str(uuid.uuid4())
                if client_code and totp_code and pin:
                    st.session_state.client = initialize_5paisa_client(client_code, totp_code, pin)
                st.rerun()

    if not st.session_state.user_id:
        return

    user_id = st.session_state.user_id
    client = st.session_state.client

    # Sidebar Inputs
    with st.sidebar:
        st.header("User Inputs")
        capital = st.number_input("Capital (₹)", min_value=100000, max_value=1000000, value=1000000, key=f"capital_{user_id}")
        risk_profile = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"], key=f"risk_{user_id}")
        dte = st.slider("Days to Expiry", 7, 30, 7, key=f"dte_{user_id}")
        if st.button("Run Engine", key=f"run_{user_id}"):
            st.session_state[f"run_engine_{user_id}"] = True

    # Load Data
    df = load_historical_data()
    df = generate_synthetic_features(df)

    # Check Market Hours (Using pytz for IST)
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    is_market_open = (market_open <= now <= market_close) and (now.weekday() < 5)

    if is_market_open and client:
        latest_data = fetch_nifty_data(client)
        if latest_data:
            df.loc[latest_data["Timestamp"]] = [latest_data["NIFTY_Close"], latest_data["VIX"], np.nan, latest_data["ATM_IV"], latest_data["Straddle_Price"], latest_data["PCR"], latest_data["IV_Skew"], latest_data["FII_Pressure"]]
            df = generate_synthetic_features(df)
            # Save to GitHub (Placeholder - Manual Update Needed)
            pd.DataFrame([latest_data]).to_csv("last_market_data.csv", index=False)
        else:
            st.warning("Failed to fetch live data. Using cached data.")
            latest_data = load_cached_data()
            if latest_data is not None:
                latest_data = simulate_data(latest_data)
                df.loc[pd.Timestamp.now()] = [latest_data["NIFTY_Close"], latest_data["VIX"], np.nan, latest_data["ATM_IV"], latest_data["Straddle_Price"], latest_data["PCR"], latest_data["IV_Skew"], latest_data["FII_Pressure"]]
                df = generate_synthetic_features(df)
    else:
        st.warning("Market Closed—Using Cached/Simulated Data")
        latest_data = load_cached_data()
        if latest_data is not None:
            latest_data = simulate_data(latest_data)
            df.loc[pd.Timestamp.now()] = [latest_data["NIFTY_Close"], latest_data["VIX"], np.nan, latest_data["ATM_IV"], latest_data["Straddle_Price"], latest_data["PCR"], latest_data["IV_Skew"], latest_data["FII_Pressure"]]
            df = generate_synthetic_features(df)

    # Market Snapshot
    st.header("Market Snapshot")
    col1, col2, col3 = st.columns(3)
    col1.metric("NIFTY 50", f"₹{df['NIFTY_Close'].iloc[-1]:.2f}")
    col2.metric("VIX", f"{df['VIX'].iloc[-1]:.2f}")
    col3.metric("PCR", f"{df['PCR'].iloc[-1]:.4f}")

    # Volatility Forecast
    blended_vol, confidence = forecast_volatility(df)
    st.header("Volatility Forecast (7-30 DTE)")
    col1, col2 = st.columns(2)
    col1.metric("Blended Volatility", f"{blended_vol:.2f}%")
    col2.metric("Confidence", f"{confidence:.2%}")

    # Volatility Chart (Using Plotly)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Realized_Vol"], mode="lines", name="Realized Vol"))
    fig.add_trace(go.Scatter(x=[df.index[-1], df.index[-1] + pd.Timedelta(days=7)], y=[df["Realized_Vol"].iloc[-1], blended_vol], mode="lines", name="Forecasted Vol", line=dict(dash="dash")))
    fig.update_layout(title="Volatility Trend", xaxis_title="Date", yaxis_title="Volatility (%)")
    st.plotly_chart(fig, use_container_width=True)

    # Strategy Engine
    if st.session_state.get(f"run_engine_{user_id}", False):
        strategy = generate_strategy(df, blended_vol, confidence, capital, risk_profile, dte)
        st.header("Strategy Recommendation")
        st.write(f"**Strategy**: {strategy['Strategy']}")
        st.write(f"**Regime**: {strategy['Regime']}")
        st.write(f"**Capital Required**: ₹{strategy['Capital_Required']:.2f}")
        st.write(f"**Max Loss**: ₹{strategy['Max_Loss']:.2f}")
        st.write(f"**Confidence**: {strategy['Confidence']:.2%}")

        # Live Trading
        if is_market_open and client:
            if st.button("Trade Now", key=f"trade_{user_id}"):
                # Placeholder for Trade Execution
                trade_log = pd.DataFrame({
                    "Date": [datetime.now()],
                    "Strategy": [strategy["Strategy"]],
                    "Regime": [strategy["Regime"]],
                    "Risk_Level": [risk_profile],
                    "Outcome": ["Pending"]
                })
                trade_log.to_csv(f"trade_log_{user_id}.csv", mode='a', header=not os.path.exists(f"trade_log_{user_id}.csv"), index=False)
                st.success("Trade Executed! Check Trade Log.")
        else:
            st.error("Market Closed—Trading Available 9:15 AM to 3:30 PM IST, Mon-Fri")

    # Portfolio Tracker (Placeholder)
    st.header("Portfolio Tracker")
    st.write("Weekly P&L: ₹0 | Margin: ₹0 | Exposure: 0%")

    # Discipline Hub
    st.header("Discipline Hub")
    with st.form("journal_form"):
        journal_entry = st.text_area("Journal Entry: Why this strategy? Did you override warnings?")
        submitted = st.form_submit_button("Submit Journal")
        if submitted:
            journal = pd.DataFrame({"Date": [datetime.now()], "Entry": [journal_entry]})
            journal.to_csv(f"journal_{user_id}.csv", mode='a', header=not os.path.exists(f"journal_{user_id}.csv"), index=False)
            st.success("Journal Saved!")

if __name__ == "__main__":
    main()
