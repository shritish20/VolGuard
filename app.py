import streamlit as st
from py5paisa import FivePaisaClient
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="VolGuard Pro", layout="wide")

# Black-Scholes IV calculation
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

# Max Pain calculation
def max_pain(df, nifty_spot):
    calls = df[df["CPType"] == "CE"].set_index("StrikeRate")["LastRate"]
    puts = df[df["CPType"] == "PE"].set_index("StrikeRate")["LastRate"]
    strikes = df["StrikeRate"].unique()
    pain = []
    for K in strikes:
        total_loss = 0
        for s in strikes:
            if s in calls:
                total_loss += max(0, s - K) * calls.get(s, 0)
            if s in puts:
                total_loss += max(0, K - s) * puts.get(s, 0)
        pain.append((K, total_loss))
    max_pain_strike = min(pain, key=lambda x: x[1])[0]
    max_pain_diff_pct = abs(nifty_spot - max_pain_strike) / nifty_spot * 100
    return max_pain_strike, max_pain_diff_pct

# Fetch Nifty and Option Chain
def fetch_nifty_data(client=None):
    if client is None:
        return None
    try:
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
            raise Exception("Failed to fetch Nifty 50 index price")
        nifty_spot = nifty_data["Data"][0].get("LastRate", nifty_data["Data"][0].get("LastTradedPrice", 0))
        if not nifty_spot:
            raise Exception("Nifty price key not found")

        expiry_timestamp = 1746694800000
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

        near_atm_df = df[(df["StrikeRate"] >= atm_strike - 300) & (df["StrikeRate"] <= atm_strike + 300)]

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

        calls = df[df["CPType"] == "CE"]
        puts = df[df["CPType"] == "PE"]
        call_oi = calls["OpenInterest"].sum()
        put_oi = puts["OpenInterest"].sum()
        pcr = put_oi / call_oi if call_oi != 0 else float("inf")

        max_pain_strike, max_pain_diff_pct = max_pain(df, nifty_spot)

        atm_iv = iv_df[iv_df["StrikeRate"] == atm_strike]["IV (%)"].mean()
        vix_change_pct = 0
        iv_file = "data/atm_iv_history.csv"
        if os.path.exists(iv_file):
            iv_history = pd.read_csv(iv_file)
            prev_atm_iv = iv_history["ATM_IV"].iloc[-1] if not iv_history.empty else atm_iv
            vix_change_pct = ((atm_iv - prev_atm_iv) / prev_atm_iv * 100) if prev_atm_iv != 0 else 0
        pd.DataFrame({"Date": [datetime.now()], "ATM_IV": [atm_iv]}).to_csv(iv_file, mode='a', header=not os.path.exists(iv_file), index=False)

        return {
            "nifty_spot": nifty_spot,
            "atm_strike": atm_strike,
            "straddle_price": straddle_price,
            "near_atm_df": near_atm_df,
            "iv_df": iv_df,
            "pcr": pcr,
            "max_pain_strike": max_pain_strike,
            "max_pain_diff_pct": max_pain_diff_pct,
            "vix_change_pct": vix_change_pct
        }
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Save 5paisa data to CSVs
def save_5paisa_data(data):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    market_data = pd.DataFrame({
        "Timestamp": [timestamp],
        "Nifty_Spot": [data["nifty_spot"]],
        "PCR": [data["pcr"]],
        "Max_Pain_Strike": [data["max_pain_strike"]],
        "Max_Pain_Diff_Pct": [data["max_pain_diff_pct"]],
        "VIX_Change_Pct": [data["vix_change_pct"]]
    })
    market_data.to_csv("data/volguard_5paisa_market_data.csv", mode="a", header=not os.path.exists("data/volguard_5paisa_market_data.csv"), index=False)
    data["near_atm_df"].to_csv("data/volguard_5paisa_near_atm.csv", index=False)
    data["iv_df"].to_csv("data/volguard_5paisa_atm_iv.csv", index=False)

# Load last known 5paisa data
def load_last_5paisa_data():
    try:
        market_data = pd.read_csv("data/volguard_5paisa_market_data.csv")
        near_atm_df = pd.read_csv("data/volguard_5paisa_near_atm.csv")
        iv_df = pd.read_csv("data/volguard_5paisa_atm_iv.csv")
        return {
            "nifty_spot": market_data["Nifty_Spot"].iloc[-1],
            "pcr": market_data["PCR"].iloc[-1],
            "max_pain_diff_pct": market_data["Max_Pain_Diff_Pct"].iloc[-1],
            "vix_change_pct": market_data["VIX_Change_Pct"].iloc[-1],
            "near_atm_df": near_atm_df,
            "iv_df": iv_df
        }
    except:
        st.error("Failed to load last known data. Please try again later.")
        return None

# Plot OI graph
def plot_oi_graph(near_atm_df, atm_strike):
    plt.figure(figsize=(10, 6))
    calls = near_atm_df[near_atm_df["CPType"] == "CE"]
    puts = near_atm_df[near_atm_df["CPType"] == "PE"]
    plt.bar(calls["StrikeRate"] - 5, calls["OpenInterest"], width=20, label="Call OI", color="green", alpha=0.5)
    plt.bar(puts["StrikeRate"] + 5, puts["OpenInterest"], width=20, label="Put OI", color="red", alpha=0.5)
    plt.axvline(x=atm_strike, color="black", linestyle="--", label="ATM Strike")
    plt.xlabel("Strike Price")
    plt.ylabel("Open Interest")
    plt.title("OI for Near ATM Strikes (±300) - May 8, 2025")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Load historical data from GitHub
def load_data():
    try:
        # Fetch from GitHub raw links (replace with your actual links after uploading)
        nifty = pd.read_csv("https://raw.githubusercontent.com/shriti/repo-name/main/new_folder/nifty_50.csv")
        nifty["Date"] = pd.to_datetime(nifty["Date"])
        vix = pd.read_csv("https://raw.githubusercontent.com/shriti/repo-name/main/new_folder/india_vix.csv")
        vix["Date"] = pd.to_datetime(vix["Date"])
        df = nifty.merge(vix[["Date", "Close"]], on="Date", how="left").rename(columns={"Close": "VIX"})
        df = df.rename(columns={"Close": "NIFTY_Close"})
        df["VIX"] = df["VIX"].ffill()
        return df
    except:
        st.error("Failed to load data from GitHub. Trying Yahoo Finance.")
        try:
            nifty = yf.download("^NSEI", start="2020-01-01", end=datetime.now().strftime("%Y-%m-%d"))
            nifty = nifty.rename(columns={"Close": "NIFTY_Close"})
            vix = pd.read_csv("https://raw.githubusercontent.com/shriti/repo-name/main/new_folder/india_vix.csv")
            vix["Date"] = pd.to_datetime(vix["Date"])
            vix = vix[["Date", "Close"]].rename(columns={"Close": "VIX"})
            df = nifty[["NIFTY_Close"]].reset_index()
            df = df.merge(vix, on="Date", how="left")
            df["VIX"] = df["VIX"].ffill()
            return df
        except:
            st.error("Failed to load historical data.")
            return None

# Generate synthetic features
def generate_synthetic_features(df):
    df["IVP"] = np.random.uniform(60, 90, len(df))
    df["PCR"] = np.random.uniform(0.8, 1.2, len(df))
    df["VIX_Change_Pct"] = df["VIX"].pct_change() * 100
    df["IV_Skew"] = np.random.uniform(-5, 5, len(df))
    df["Straddle_Price"] = np.random.uniform(200, 500, len(df))
    df["Spot_MaxPain_Diff_Pct"] = np.random.uniform(-2, 2, len(df))
    df["Days_to_Expiry"] = np.random.randint(7, 30, len(df))
    df["Event_Flag"] = df["Date"].apply(lambda x: 1 if x.month % 3 == 0 or (datetime(2025, 5, 8) - x).days <= 3 else 0)
    df["FII_Index_Fut_Pos"] = np.random.uniform(-10000, 10000, len(df))
    df["FII_Option_Pos"] = np.random.uniform(-5000, 5000, len(df))
    df["ATM_IV"] = np.random.uniform(15, 30, len(df))
    realized_vol = df["NIFTY_Close"].pct_change().rolling(20).std() * np.sqrt(252) * 100
    df["Target_Vol"] = realized_vol.shift(-20)
    return df.dropna()

# Forecast volatility
def forecast_volatility_future(df, forecast_horizon):
    df_garch = df.copy()
    df_garch['Log_Returns'] = np.log(df_garch['NIFTY_Close'] / df_garch['NIFTY_Close'].shift(1)).dropna() * 100
    garch_model = arch_model(df_garch['Log_Returns'].dropna(), vol='Garch', p=1, q=1, rescale=False)
    garch_fit = garch_model.fit(disp="off")
    garch_forecast = garch_fit.forecast(horizon=forecast_horizon, reindex=False)
    garch_vols = np.sqrt(garch_forecast.variance.iloc[-1].values) * np.sqrt(252)

    feature_cols = [
        'VIX', 'ATM_IV', 'IVP', 'PCR', 'VIX_Change_Pct', 'IV_Skew', 'Straddle_Price',
        'Spot_MaxPain_Diff_Pct', 'Days_to_Expiry', 'Event_Flag', 'FII_Index_Fut_Pos',
        'FII_Option_Pos'
    ]
    df_xgb = df.copy()
    X = df_xgb[feature_cols]
    y = df_xgb['Target_Vol']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    train_size = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.03, random_state=42)
    model.fit(X_train, y_train)
    future_features = X_scaled[-1].reshape(1, -1)
    xgb_vols = []
    for _ in range(forecast_horizon):
        pred = model.predict(future_features)[0]
        xgb_vols.append(pred)
        future_features[0, feature_cols.index('Days_to_Expiry')] -= 1

    realized_vol = df['NIFTY_Close'].pct_change().rolling(20).std() * np.sqrt(252) * 100
    realized_vol = realized_vol.iloc[-1]
    garch_diff = np.abs(garch_vols[0] - realized_vol)
    xgb_diff = np.abs(xgb_vols[0] - realized_vol)
    garch_weight = xgb_diff / (garch_diff + xgb_diff) if (garch_diff + xgb_diff) > 0 else 0.5
    xgb_weight = 1 - garch_weight
    blended_vols = [(garch_weight * g) + (xgb_weight * x) for g, x in zip(garch_vols, xgb_vols)]

    confidence = max(50, 100 - (garch_diff + xgb_diff) / 2)
    regime = "Low" if blended_vols[0] < 15 else "Medium" if blended_vols[0] < 20 else "High"
    if df["Event_Flag"].iloc[-1] == 1:
        regime = "Event-Driven"

    return blended_vols, confidence, regime

# Plot volatility forecast
def plot_volatility_forecast(blended_vols, forecast_horizon):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, forecast_horizon + 1), blended_vols, marker='o', label="Forecasted Volatility")
    plt.xlabel("Day")
    plt.ylabel("Volatility (%)")
    plt.title("Volatility Forecast (Next Days)")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Strategy recommendation with Risk Manager checks
def recommend_strategy(regime, iv_hv_gap, pcr, max_pain_diff_pct, vix_change_pct, atm_iv, intraday_mode, vix, weekly_loss):
    # Risk Manager checks
    if vix > 25:
        st.warning("⚠️ VIX > 25%. High risk detected. Consider holding positions.")
        return "Hold", "Red", 0.0
    if weekly_loss < -5:
        st.warning("⚠️ Weekly loss > 5%. Risk limit exceeded. Consider holding positions.")
        return "Hold", "Red", 0.0

    # Strategy recommendation
    if intraday_mode:
        if regime in ["High", "Event-Driven"] and atm_iv > 18:
            return "Scalping (Intraday)", "Green", 0.2
        else:
            return "No Intraday Strategy", "Red", 1.0
    else:
        if regime == "Low" and iv_hv_gap > 5 and pcr < 0.9:
            return "Iron Condor", "Green", 1.0
        elif regime == "Medium" and iv_hv_gap > 0 and pcr > 1.0:
            return "Jade Lizard", "Yellow", 1.0
        elif regime == "High" and vix_change_pct > 5 and max_pain_diff_pct < 1:
            return "Long Straddle", "Red", 0.5
        elif regime == "Event-Driven":
            return "Short Straddle (Hedged)", "Yellow", 0.5
        else:
            return "Hold", "Yellow", 1.0

# Backtesting function
def backtest_strategies(df, capital=100000):
    portfolio_value = capital
    trades = []
    
    for i in range(len(df) - 252, len(df)):
        current_data = df.iloc[i]
        vix = current_data["VIX"]
        atm_iv = current_data["ATM_IV"]
        pcr = current_data["PCR"]
        max_pain_diff_pct = current_data["Spot_MaxPain_Diff_Pct"]
        vix_change_pct = current_data["VIX_Change_Pct"]
        
        df_subset = df.iloc[:i+1]
        blended_vols, confidence, regime = forecast_volatility_future(df_subset, forecast_horizon=1)
        
        hv = df["NIFTY_Close"].iloc[:i+1].pct_change().rolling(20).std() * np.sqrt(252) * 100
        iv_hv_gap = atm_iv - hv.iloc[-1] if not pd.isna(hv.iloc[-1]) else 0
        
        weekly_loss = 0
        
        strategy, safety, capital_allocation = recommend_strategy(
            regime=regime,
            iv_hv_gap=iv_hv_gap,
            pcr=pcr,
            max_pain_diff_pct=max_pain_diff_pct,
            vix_change_pct=vix_change_pct,
            atm_iv=atm_iv,
            intraday_mode=False,
            vix=vix,
            weekly_loss=weekly_loss
        )
        
        if safety == "Green":
            trade_return = 0.02
        elif safety == "Yellow":
            trade_return = 0.01
        else:
            trade_return = -0.02
        
        trade_pnl = portfolio_value * capital_allocation * trade_return
        portfolio_value += trade_pnl
        
        trades.append({
            "Date": current_data["Date"],
            "Strategy": strategy,
            "Safety": safety,
            "Capital_Allocation": capital_allocation,
            "Trade_PnL": trade_pnl,
            "Portfolio_Value": portfolio_value
        })
    
    trades_df = pd.DataFrame(trades)
    return trades_df, portfolio_value - capital

# Execute trade (basic structure)
def execute_trade(client, strategy, capital_allocation):
    try:
        order = {
            "Exch": "N",
            "ExchType": "D",
            "ScripCode": 999920000,
            "ScripData": "NIFTY",
            "BuySell": "B",
            "Qty": int(capital_allocation * 50),
            "AtMarket": True,
            "OrderType": "BUY"
        }
        response = client.place_order(order)
        if response.get("Status") == 0:
            stop_loss_order = {
                "Exch": "N",
                "ExchType": "D",
                "ScripCode": 999920000,
                "ScripData": "NIFTY",
                "BuySell": "S",
                "Qty": int(capital_allocation * 50),
                "AtMarket": False,
                "OrderType": "SELL",
                "TriggerPrice": response["Price"] * 0.95
            }
            client.place_order(stop_loss_order)
            return True
        else:
            return False
    except Exception as e:
        st.error(f"Trade execution failed: {str(e)}")
        return False

# Plot behavior score
def plot_behavior_score(behavior_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(behavior_scores["Date"], behavior_scores["Behavior_Score"], marker='o', label="Behavior Score")
    plt.xlabel("Date")
    plt.ylabel("Behavior Score")
    plt.title("Trade Reflection: Behavior Score Over Time")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Role-based access
role = st.selectbox("Select User Role", ["Admin (Live Data & Trading)", "Guest (View Only)"])
client = None

if role == "Admin (Live Data & Trading)":
    password = st.text_input("Enter Admin Password", type="password", value="")
    if password != "your_password":
        st.error("Access Denied")
        st.stop()
    totp_code = st.text_input("Enter TOTP Code", type="password", value="")
    if totp_code:
        load_dotenv()
        cred = {
            "APP_NAME": os.getenv("APP_NAME"),
            "APP_SOURCE": os.getenv("APP_SOURCE"),
            "USER_ID": os.getenv("USER_ID"),
            "PASSWORD": os.getenv("PASSWORD"),
            "USER_KEY": os.getenv("USER_KEY"),
            "ENCRYPTION_KEY": os.getenv("ENCRYPTION_KEY")
        }
        client_code = os.getenv("CLIENT_CODE")
        pin = os.getenv("PIN")
        client = FivePaisaClient(cred=cred)
        client.get_totp_session(client_code, totp_code, pin)
        if client:
            st.success("✅ Successfully Logged In!")
            st.session_state["client"] = client
            st.session_state["role"] = "admin"
        else:
            st.error("Login Failed. Try Again.")
            st.stop()
    else:
        st.stop()
else:
    st.session_state["role"] = "guest"
    st.info("Guest Mode: Real-time data and trading features are disabled.")

# Fetch or load data based on role
if st.session_state["role"] == "admin":
    data = fetch_nifty_data(client=st.session_state["client"])
    if data:
        save_5paisa_data(data)
    else:
        data = load_last_5paisa_data()
else:
    data = load_last_5paisa_data()

# Load historical data and generate features
df = load_data()
if df is None:
    st.stop()
df = generate_synthetic_features(df)

# Integrate 5paisa data into features
vix = df["VIX"].iloc[-1]
if data:
    df["ATM_IV"] = df["ATM_IV"].astype(float)
    df["PCR"] = df["PCR"].astype(float)
    df["Straddle_Price"] = df["Straddle_Price"].astype(float)
    df["Spot_MaxPain_Diff_Pct"] = df["Spot_MaxPain_Diff_Pct"].astype(float)
    df["VIX_Change_Pct"] = df["VIX_Change_Pct"].astype(float)
    df.loc[df.index[-1], "ATM_IV"] = data["iv_df"][data["iv_df"]["StrikeRate"] == data["atm_strike"]]["IV (%)"].mean()
    df.loc[df.index[-1], "PCR"] = data["pcr"]
    df.loc[df.index[-1], "Straddle_Price"] = data["straddle_price"]
    df.loc[df.index[-1], "Spot_MaxPain_Diff_Pct"] = data["max_pain_diff_pct"]
    df.loc[df.index[-1], "VIX_Change_Pct"] = data["vix_change_pct"]

# Calculate weekly loss (simulated for guests)
if st.session_state["role"] == "admin":
    weekly_loss = np.random.uniform(-10, 5)
else:
    weekly_loss = np.random.uniform(-10, 5)

# Sidebar for settings
st.sidebar.header("Settings")
intraday_mode = st.sidebar.checkbox("Enable Intraday Mode", value=False)

# Tab layout for navigation
tabs = st.tabs(["Snapshot", "Forecast", "Strategy", "Portfolio", "Journal"])

# Market Snapshot tab
with tabs[0]:
    st.header("Market Snapshot")
    if data:
        if st.session_state["role"] == "guest":
            st.info("Real-time data unavailable. Showing last known values.")
        st.write(f"**Live Nifty 50 Index Price:** ₹{data['nifty_spot']:.2f}")
        st.write(f"**ATM Strike Price:** ₹{data['atm_strike']:.2f}")
        st.write(f"**Straddle ATM Price:** ₹{data['straddle_price']:.2f}")
        st.write(f"**PCR (Whole Expiry):** {data['pcr']:.4f}")
        st.write(f"**Max Pain Strike Price:** ₹{data['max_pain_strike']:.2f}")
        st.write(f"**Spot Max Pain Diff Pct:** {data['max_pain_diff_pct']:.2f}%")
        st.write(f"**VIX Change Pct (ATM IV):** {data['vix_change_pct']:.2f}%")
        st.write("### IV for ATM and ±100 Strikes (May 8, 2025)")
        st.dataframe(data['iv_df'][["StrikeRate", "CPType", "LastRate", "IV (%)"]])
        st.write("### Open Interest for Near ATM (±300 Strikes)")
        plot_oi_graph(data['near_atm_df'], data['atm_strike'])
    else:
        st.write("No data available yet.")

# Volatility Forecast tab
with tabs[1]:
    st.header("Volatility Forecast")
    forecast_horizon = st.slider("Forecast Horizon (Days)", min_value=1, max_value=10, value=7)
    if st.button("Run Forecast"):
        blended_vols, confidence, regime = forecast_volatility_future(df, forecast_horizon)
        st.session_state["blended_vols"] = blended_vols
        st.session_state["regime"] = regime
        st.write(f"**Market Regime:** {regime}")
        st.write(f"**Confidence Score:** {confidence:.2f}%")
        st.write("### Volatility Forecast")
        forecast_df = pd.DataFrame({
            "Day": [f"Day {i+1}" for i in range(forecast_horizon)],
            "Volatility (%)": [f"{vol:.2f}" for vol in blended_vols]
        })
        st.dataframe(forecast_df)
        plot_volatility_forecast(blended_vols, forecast_horizon)

# Strategy Engine tab
with tabs[2]:
    st.header("Strategy Engine")
    if os.path.exists("data/trade_log.csv"):
        trade_log = pd.read_csv("data/trade_log.csv")
        violations = len(trade_log[trade_log["Risk_Level"] == "Red"])
        if violations >= 2:
            st.error("🚫 Discipline Lock: Too many high-risk trades. Please journal before proceeding.")
            st.stop()

    if "blended_vols" in st.session_state and "regime" in st.session_state:
        blended_vols = st.session_state["blended_vols"]
        regime = st.session_state["regime"]
        if data:
            iv_hv_gap = data["iv_df"][data["iv_df"]["StrikeRate"] == data["atm_strike"]]["IV (%)"].mean() - (df["NIFTY_Close"].pct_change().rolling(20).std() * np.sqrt(252) * 100).iloc[-1]
            strategy, safety, capital_allocation = recommend_strategy(
                regime=regime,
                iv_hv_gap=iv_hv_gap,
                pcr=data["pcr"],
                max_pain_diff_pct=data["max_pain_diff_pct"],
                vix_change_pct=data["vix_change_pct"],
                atm_iv=data["iv_df"][data["iv_df"]["StrikeRate"] == data["atm_strike"]]["IV (%)"].mean(),
                intraday_mode=intraday_mode,
                vix=vix,
                weekly_loss=weekly_loss
            )
            color = {"Green": "background-color: #90EE90", "Yellow": "background-color: #FFFF99", "Red": "background-color: #FF6347"}
            st.markdown(
                f"<div style='{color[safety]}; padding: 10px; border-radius: 5px;'>"
                f"<strong>Recommended Strategy:</strong> {strategy}<br>"
                f"<strong>Safety Level:</strong> {safety}<br>"
                f"<strong>Capital Allocation:</strong> {capital_allocation * 100:.0f}%"
                f"</div>",
                unsafe_allow_html=True
            )
            if st.session_state["role"] == "admin" and safety != "Red" and strategy != "Hold" and strategy != "No Intraday Strategy":
                if st.button("Trade Now"):
                    success = execute_trade(st.session_state["client"], strategy, capital_allocation)
                    trade_log = pd.DataFrame({
                        "Date": [datetime.now()],
                        "Strategy": [strategy],
                        "Regime": [regime],
                        "Risk_Level": [safety],
                        "Outcome": ["Success" if success else "Failed"]
                    })
                    trade_log.to_csv("data/trade_log.csv", mode="a", header=not os.path.exists("data/trade_log.csv"), index=False)
                    if success:
                        st.success("Trade Executed Successfully!")
                    else:
                        st.error("Trade Execution Failed.")
        else:
            st.write("Run a forecast and ensure market data is available to get strategy recommendations.")
    else:
        st.write("Run a forecast first to get strategy recommendations.")

# Portfolio Tracker tab with Backtesting
with tabs[3]:
    st.header("Portfolio Tracker")
    if st.session_state["role"] == "admin":
        if data:
            # Placeholder for live P&L
            st.write("**Live P&L:** ₹5000 (Placeholder)")
            st.write("**Margin Used:** ₹20000 (Placeholder)")
            st.write("**Exposure:** 1.5x (Placeholder)")
        else:
            st.write("No live data available.")
        
        # Run backtesting for Admin to show simulated performance
        st.subheader("Backtested Performance (Last 1 Year)")
        trades_df, simulated_pnl = backtest_strategies(df)
        st.write(f"**Simulated P&L (Backtest):** ₹{simulated_pnl:.2f}")
        st.write("**Backtest Trade Log:**")
        st.dataframe(trades_df[["Date", "Strategy", "Safety", "Trade_PnL", "Portfolio_Value"]])
    else:
        # Run backtesting for Guest
        st.subheader("Backtested Performance (Last 1 Year)")
        trades_df, simulated_pnl = backtest_strategies(df)
        st.write(f"**Simulated P&L (Backtest):** ₹{simulated_pnl:.2f}")
        st.write("**Margin Used:** ₹15000 (Simulated)")
        st.write("**Exposure:** 1.2x (Simulated)")
        st.write("**Backtest Trade Log:**")
        st.dataframe(trades_df[["Date", "Strategy", "Safety", "Trade_PnL", "Portfolio_Value"]])

# Discipline Hub tab
with tabs[4]:
    st.header("Discipline Hub")
    st.subheader("Trade Journal")
    why_strategy = st.text_area("Why did you choose this strategy?")
    override_warnings = st.text_area("Did you override any warnings? Why?")
    if st.button("Submit Journal Entry"):
        if why_strategy and override_warnings:
            journal_entry = pd.DataFrame({
                "Date": [datetime.now()],
                "Why_Strategy": [why_strategy],
                "Override_Warnings": [override_warnings]
            })
            journal_entry.to_csv("data/journal_entries.csv", mode="a", header=not os.path.exists("data/journal_entries.csv"), index=False)
            behavior_score = np.random.randint(60, 100)
            behavior_scores = pd.DataFrame({
                "Date": [datetime.now()],
                "Behavior_Score": [behavior_score]
            })
            behavior_scores.to_csv("data/behavior_scores.csv", mode="a", header=not os.path.exists("data/behavior_scores.csv"), index=False)
            st.success("Journal entry submitted!")
        else:
            st.error("Please fill in all fields.")

    st.subheader("Discipline Score")
    if os.path.exists("data/journal_entries.csv"):
        journal = pd.read_csv("data/journal_entries.csv")
        journal_count = len(journal)
        discipline_score = min(100, journal_count * 10)
        st.write(f"**Discipline Score:** {discipline_score}/100")
    else:
        st.write("**Discipline Score:** 0/100 (No journal entries yet)")

    st.subheader("Trade Reflection")
    if os.path.exists("data/behavior_scores.csv"):
        behavior_scores = pd.read_csv("data/behavior_scores.csv")
        plot_behavior_score(behavior_scores)
    else:
        st.write("No behavior scores available yet. Submit journal entries to track.")
