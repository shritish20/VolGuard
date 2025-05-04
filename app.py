import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, timedelta
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import requests
import io
import warnings
import logging
from py5paisa import FivePaisaClient
import os
import plotly.graph_objects as go

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# Page config
st.set_page_config(page_title="VolGuard Pro", page_icon="🛡️", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        .main { background: linear-gradient(135deg, #1a1a2e, #0f1c2e); color: #e5e5e5; font-family: 'Inter', sans-serif; }
        .stTabs [data-baseweb="tab-list"] { background: #16213e; border-radius: 10px; padding: 10px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3); }
        .stTabs [data-baseweb="tab"] { color: #a0a0a0; font-weight: 500; padding: 10px 20px; border-radius: 8px; transition: background 0.3s; }
        .stTabs [data-baseweb="tab"][aria-selected="true"] { background: #e94560; color: white; font-weight: 700; }
        .stTabs [data-baseweb="tab"]:hover { background: #2a2a4a; color: white; }
        .sidebar .stButton>button { width: 100%; background: #0f3460; color: white; border-radius: 10px; padding: 12px; margin: 5px 0; transition: transform 0.3s; }
        .sidebar .stButton>button:hover { transform: scale(1.05); background: #e94560; }
        .card { background: linear-gradient(145deg, rgba(22, 33, 62, 0.85), rgba(10, 25, 47, 0.9)); border-radius: 15px; padding: 20px; margin: 15px 0; box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4); transition: transform 0.3s; }
        .card:hover { transform: translateY(-5px); }
        .strategy-carousel { display: flex; overflow-x: auto; gap: 20px; padding: 10px; scrollbar-width: thin; }
        .strategy-card { flex: 0 0 auto; width: 300px; background: #16213e; border-radius: 15px; padding: 20px; transition: transform 0.3s; }
        .strategy-card:hover { transform: scale(1.05); }
        .sidebar-card { background: #16213e; border-radius: 15px; padding: 15px; margin-top: 10px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3); font-size: 14px; }
        .stMetric { background: rgba(15, 52, 96, 0.7); border-radius: 15px; padding: 15px; text-align: center; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3); }
        .gauge { width: 100px; height: 100px; border-radius: 50%; background: conic-gradient(#e94560 0% 50%, #00d4ff 50% 100%); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 18px; box-shadow: 0 0 15px rgba(233, 69, 96, 0.5); }
        .progress-bar { background: #16213e; border-radius: 10px; height: 20px; width: 100%; overflow: hidden; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #e94560, #ffcc00); transition: width 1s ease-in-out; }
        .regime-badge { padding: 8px 15px; border-radius: 20px; font-weight: bold; font-size: 14px; text-transform: uppercase; }
        .regime-low { background: #28a745; color: white; }
        .regime-medium { background: #ffc107; color: black; }
        .regime-high { background: #dc3545; color: white; }
        .regime-event { background: #ff6f61; color: white; }
        .alert-banner { background: #dc3545; color: white; padding: 15px; border-radius: 10px; position: sticky; top: 0; z-index: 100; }
        .stButton>button { background: #e94560; color: white; border-radius: 10px; padding: 12px 25px; font-size: 16px; transition: transform 0.3s; }
        .stButton>button:hover { transform: scale(1.05); background: #ffcc00; }
        .footer { text-align: center; padding: 20px; color: #a0a0a0; font-size: 14px; border-top: 1px solid rgba(255, 255, 255, 0.1); margin-top: 30px; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
for key in ["client", "backtest_run", "backtest_results", "violations", "journal_complete", "trades", "logged_in", "selected_strategy", "dte_preference"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "trades" else [] if key == "trades" else False if key in ["backtest_run", "journal_complete", "logged_in"] else 0 if key == "violations" else 15 if key == "dte_preference" else None

# Sidebar: 5paisa Login
with st.sidebar:
    st.header("🔐 5paisa Login")
    client_code = st.text_input("Client Code", placeholder="Enter Client Code")
    totp_code = st.text_input("TOTP", type="password", placeholder="Enter TOTP")
    mpin = st.text_input("MPIN", type="password", placeholder="Enter MPIN")
    if st.button("Login"):
        try:
            cred = {
                "APP_NAME": st.secrets["fivepaisa"]["APP_NAME"],
                "APP_SOURCE": st.secrets["fivepaisa"]["APP_SOURCE"],
                "USER_ID": st.secrets["fivepaisa"]["USER_ID"],
                "PASSWORD": st.secrets["fivepaisa"]["PASSWORD"],
                "USER_KEY": st.secrets["fivepaisa"]["USER_KEY"],
                "ENCRYPTION_KEY": st.secrets["fivepaisa"]["ENCRYPTION_KEY"]
            }
            client = FivePaisaClient(cred=cred)
            client.get_totp_session(client_code, totp_code, mpin)
            st.session_state.client = client
            st.session_state.logged_in = True
            st.success("✅ Logged in")
        except Exception as e:
            st.error(f"❌ Login failed: {str(e)}")
            logger.error(f"Login error: {str(e)}")

    if st.session_state.logged_in:
        st.header("⚙️ Trading Controls")
        capital = st.number_input("Capital (₹)", min_value=100000, value=1000000, step=100000)
        risk_tolerance = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"], index=1)
        forecast_horizon = st.slider("Forecast Horizon (days)", 1, 30, 7)
        st.session_state.dte_preference = st.slider("DTE Preference (days)", 7, 30, 15)
        st.markdown("**Backtest Parameters**")
        start_date = st.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
        end_date = st.date_input("End Date", value=pd.to_datetime("2025-05-04"))
        strategy_choice = st.selectbox("Strategy", ["All Strategies", "Butterfly Spread", "Iron Condor", "Iron Fly", "Short Strangle", "Calendar Spread", "Jade Lizard"])
        run_button = st.button("Run Analysis")
        st.markdown("**Motto:** Deploy with edge, survive, outlast.")
        if st.session_state.selected_strategy:
            strategy, regime, iv_hv_gap, iv_skew, vix = st.session_state.selected_strategy
            st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
            st.markdown(f"**{strategy['Strategy']}: Why?**\n- **Regime**: {regime}\n- **IV-HV Gap**: {iv_hv_gap:.2f}%\n- **IV Skew**: {iv_skew:.2f}\n- **VIX**: {vix:.2f}\n- **Confidence**: {strategy['Confidence']:.0%}")
            st.markdown('</div>', unsafe_allow_html=True)

# Helper Functions
def get_expiry_dates():
    try:
        client = st.session_state.client
        today = datetime.now()
        expiries = [int((today + timedelta(days=i)).timestamp() * 1000) for i in range(30) if (today + timedelta(days=i)).weekday() == 3]
        return expiries
    except:
        return [int((datetime.now() + timedelta(days=7)).timestamp() * 1000)]

def calculate_days_to_expiry(expiry_timestamp):
    expiry_date = datetime.fromtimestamp(expiry_timestamp / 1000)
    return max(1, (expiry_date - datetime.now()).days)

def black_scholes_call(S, K, T, r, sigma):
    try:
        T = max(T, 1e-6)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return max(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), 0)
    except:
        return 0

def black_scholes_put(S, K, T, r, sigma):
    try:
        T = max(T, 1e-6)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return max(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1), 0)
    except:
        return 0

def implied_volatility(S, K, T, r, market_price, option_type='call'):
    sigma = 0.2
    for _ in range(100):
        if option_type == 'call':
            model_price = black_scholes_call(S, K, T, r, sigma)
        else:
            model_price = black_scholes_put(S, K, T, r, sigma)
        diff = model_price - market_price
        if abs(diff) < 1e-5:
            return sigma * 100
        vega = S * norm.pdf((np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))) * np.sqrt(T)
        if vega == 0:
            return np.nan
        sigma -= diff / vega
    return np.nan

def max_pain(df, nifty_spot):
    calls = df[df["CPType"] == "CE"].set_index("StrikeRate")["OpenInterest"]
    puts = df[df["CPType"] == "PE"].set_index("StrikeRate")["OpenInterest"]
    strikes = df["StrikeRate"].unique()
    pain = [(K, sum(max(0, s - K) * calls.get(s, 0) for s in strikes if s in calls) + 
                sum(max(0, K - s) * puts.get(s, 0) for s in strikes if s in puts)) for K in strikes]
    max_pain_strike = min(pain, key=lambda x: x[1])[0]
    return max_pain_strike, abs(nifty_spot - max_pain_strike) / nifty_spot * 100

def fetch_nifty_data():
    try:
        client = st.session_state.client
        nifty_req = [{"Exch": "N", "ExchType": "C", "ScripCode": 999920000, "Symbol": "NIFTY"}]
        nifty_data = client.fetch_market_feed(nifty_req)
        nifty_spot = nifty_data["Data"][0].get("LastRate", 0)
        
        expiries = get_expiry_dates()
        dte_preference = st.session_state.dte_preference
        valid_expiries = [e for e in expiries if 7 <= calculate_days_to_expiry(e) <= 30]
        expiry_timestamp = min(valid_expiries, key=lambda x: abs(calculate_days_to_expiry(x) - dte_preference))
        
        option_chain = client.get_option_chain("N", "NIFTY", expiry_timestamp)
        df = pd.DataFrame(option_chain["Options"])
        df["StrikeRate"] = df["StrikeRate"].astype(float)
        atm_strike = df["StrikeRate"].iloc[(df["StrikeRate"] - nifty_spot).abs().argmin()]
        atm_data = df[df["StrikeRate"] == atm_strike]
        atm_call = atm_data[atm_data["CPType"] == "CE"]["LastRate"].iloc[0] if not atm_data[atm_data["CPType"] == "CE"].empty else 0
        atm_put = atm_data[atm_data["CPType"] == "PE"]["LastRate"].iloc[0] if not atm_data[atm_data["CPType"] == "PE"].empty else 0
        straddle_price = atm_call + atm_put
        
        T = calculate_days_to_expiry(expiry_timestamp) / 365.0
        r = 0.06
        iv_df = df[(df["StrikeRate"] >= atm_strike - 100) & (df["StrikeRate"] <= atm_strike + 100)].copy()
        iv_df["IV (%)"] = iv_df.apply(
            lambda row: implied_volatility(S=nifty_spot, K=row["StrikeRate"], T=T, r=r, market_price=row["LastRate"],
                                          option_type='call' if row["CPType"] == "CE" else 'put'), axis=1)
        
        calls = df[df["CPType"] == "CE"]
        puts = df[df["CPType"] == "PE"]
        pcr = puts["OpenInterest"].sum() / calls["OpenInterest"].sum() if calls["OpenInterest"].sum() != 0 else 1.0
        
        max_pain_strike, max_pain_diff_pct = max_pain(df, nifty_spot)
        
        vix_data = client.fetch_market_feed([{"Exch": "N", "ExchType": "C", "Symbol": "INDIA VIX"}])
        vix = vix_data["Data"][0]["LastRate"] if vix_data and "Data" in vix_data else 15.0
        iv_file = "atm_iv_history.csv"
        atm_iv = iv_df[iv_df["StrikeRate"] == atm_strike]["IV (%)"].mean()
        vix_change_pct = 0
        if os.path.exists(iv_file):
            iv_history = pd.read_csv(iv_file)
            prev_atm_iv = iv_history["ATM_IV"].iloc[-1] if not iv_history.empty else atm_iv
            vix_change_pct = ((atm_iv - prev_atm_iv) / prev_atm_iv * 100) if prev_atm_iv != 0 else 0
        pd.DataFrame({"Date": [datetime.now()], "ATM_IV": [atm_iv]}).to_csv(iv_file, mode='a', header=not os.path.exists(iv_file), index=False)
        
        return {
            "nifty_spot": nifty_spot, "atm_strike": atm_strike, "straddle_price": straddle_price, "pcr": pcr,
            "max_pain_strike": max_pain_strike, "max_pain_diff_pct": max_pain_diff_pct, "vix": vix,
            "vix_change_pct": vix_change_pct, "atm_iv": atm_iv, "option_chain": df, "expiry_timestamp": expiry_timestamp
        }
    except Exception as e:
        logger.error(f"Error fetching 5paisa data: {str(e)}")
        return None

@st.cache_data(ttl=300)
def load_data():
    try:
        real_data = fetch_nifty_data()
        if real_data is None:
            nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
            vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"
            nifty = pd.read_csv(io.StringIO(requests.get(nifty_url).text), encoding="utf-8-sig")
            nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
            nifty = nifty[["Date", "Close"]].set_index("Date").rename(columns={"Close": "NIFTY_Close"})
            vix = pd.read_csv(io.StringIO(requests.get(vix_url).text))
            vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")
            vix = vix[["Date", "Close"]].set_index("Date").rename(columns={"Close": "VIX"})
            common_dates = nifty.index.intersection(vix.index)
            df = pd.DataFrame({"NIFTY_Close": nifty["NIFTY_Close"].loc[common_dates], "VIX": vix["VIX"].loc[common_dates]}, index=common_dates)
        else:
            latest_date = datetime.now().date()
            df = pd.DataFrame({"NIFTY_Close": [real_data["nifty_spot"]], "VIX": [real_data["atm_iv"]]}, index=[pd.to_datetime(latest_date)])
            nifty = pd.read_csv(io.StringIO(requests.get("https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv").text), encoding="utf-8-sig")
            nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
            nifty = nifty[["Date", "Close"]].set_index("Date").rename(columns={"Close": "NIFTY_Close"})
            vix = pd.read_csv(io.StringIO(requests.get("https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv").text))
            vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")
            vix = vix[["Date", "Close"]].set_index("Date").rename(columns={"Close": "VIX"})
            historical_df = pd.DataFrame({"NIFTY_Close": nifty["NIFTY_Close"], "VIX": vix["VIX"]}).dropna()
            historical_df = historical_df[historical_df.index < pd.to_datetime(latest_date)]
            df = pd.concat([historical_df, df]).sort_index()
        return df, real_data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None, None

@st.cache_data
def generate_synthetic_features(df, real_data, capital):
    try:
        n_days = len(df)
        np.random.seed(42)
        risk_free_rate = 0.06
        strike_step = 100
        base_pcr = real_data["pcr"] if real_data else 1.0
        base_iv = real_data["atm_iv"] if real_data else 20.0
        base_straddle_price = real_data["straddle_price"] if real_data else 200.0
        base_max_pain_diff_pct = real_data["max_pain_diff_pct"] if real_data else 0.5
        base_vix_change_pct = real_data["vix_change_pct"] if real_data else 0.0
        expiry_timestamp = real_data["expiry_timestamp"] if real_data else int((datetime.now() + timedelta(days=7)).timestamp() * 1000)
        
        df["Days_to_Expiry"] = [(datetime.fromtimestamp(expiry_timestamp / 1000) - d).days for d in df.index]
        event_spike = np.where((df.index.month % 3 == 0) & (df.index.day < 5), 1.2, 1.0)
        df["ATM_IV"] = np.clip(df["VIX"] * (1 + np.random.normal(0, 0.1, n_days)) * event_spike, 5, 50)
        if real_data:
            df["ATM_IV"].iloc[-1] = base_iv
        df["IVP"] = df["ATM_IV"].rolling(252, min_periods=5).apply(lambda x: (np.sum(x.iloc[:-1] <= x.iloc[-1]) / (len(x) - 1)) * 100 if len(x) >= 5 else 50.0).interpolate().fillna(50.0)
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
        df["FII_Index_Fut_Pos"] = np.cumsum(np.random.normal(0, 10000, n_days) * np.where(np.arange(n_days) % 30 == 0, -1, 1)).astype(int)
        df["FII_Option_Pos"] = np.cumsum(np.random.normal(0, 5000, n_days)).astype(int)
        df["IV_Skew"] = np.clip(np.random.normal(0, 0.8, n_days) + (df["VIX"] / 15 - 1) * 3, -3, 3)
        df["Realized_Vol"] = np.clip(df["NIFTY_Close"].pct_change().rolling(5, min_periods=5).std() * np.sqrt(252) * 100, 0, 50).fillna(df["VIX"])
        df["Advance_Decline_Ratio"] = np.clip(1.0 + np.random.normal(0, 0.2, n_days) + market_trend * 10, 0.5, 2.0)
        df["Capital_Pressure_Index"] = np.clip((df["FII_Index_Fut_Pos"] / 3e4 + df["FII_Option_Pos"] / 1e4 + df["PCR"]) / 3, -2, 2)
        df["Gamma_Bias"] = np.clip(df["IV_Skew"] * (30 - df["Days_to_Expiry"]) / 30, -2, 2)
        df["Total_Capital"] = capital
        df["PnL_Day"] = np.random.normal(0, 5000, n_days) * (1 - df["Event_Flag"] * 0.5)
        straddle_prices = [np.clip((black_scholes_call(S, round(S / strike_step) * strike_step, D / 365, risk_free_rate, A / 100) + 
                                   black_scholes_put(S, round(S / strike_step) * strike_step, D / 365, risk_free_rate, A / 100)) * (S / 1000), 50, 400)
                          for S, D, A in zip(df["NIFTY_Close"], df["Days_to_Expiry"], df["ATM_IV"])]
        df["Straddle_Price"] = straddle_prices
        if real_data:
            df["Straddle_Price"].iloc[-1] = base_straddle_price
        df = df.interpolate().fillna(method='bfill')
        df.to_csv("volguard_hybrid_data.csv")
        return df
    except Exception as e:
        logger.error(f"Error generating features: {str(e)}")
        return None

def fetch_portfolio_data(capital):
    try:
        client = st.session_state.client
        positions = client.positions()
        total_pnl = sum(p.get("ProfitLoss", 0) for p in positions)
        total_margin = sum(p.get("MarginUsed", 0) for p in positions)
        total_exposure = sum(p.get("Exposure", 0) for p in positions) / capital * 100 if capital > 0 else 0
        return {"weekly_pnl": total_pnl, "margin_used": total_margin, "exposure": total_exposure, "positions": positions}
    except:
        return {"weekly_pnl": 0, "margin_used": 0, "exposure": 0, "positions": []}

feature_cols = ['VIX', 'ATM_IV', 'IVP', 'PCR', 'VIX_Change_Pct', 'IV_Skew', 'Straddle_Price', 'Spot_MaxPain_Diff_Pct', 'Days_to_Expiry', 'Event_Flag', 'FII_Index_Fut_Pos', 'FII_Option_Pos']

@st.cache_data
def forecast_volatility_future(df, forecast_horizon):
    try:
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
        for _ in range(forecast_horizon):
            current_row_df = pd.DataFrame([current_row], columns=feature_cols)
            current_row_scaled = scaler.transform(current_row_df)
            next_vol = model.predict(current_row_scaled)[0]
            xgb_vols.append(next_vol)
            current_row["Days_to_Expiry"] = max(1, current_row["Days_to_Expiry"] - 1)
            current_row["VIX"] *= np.random.uniform(0.98, 1.02)
            current_row["Straddle_Price"] *= np.random.uniform(0.98, 1.02)
            current_row["VIX_Change_Pct"] = (current_row["VIX"] / df_xgb["VIX"].iloc[-1] - 1) * 100
            current_row["ATM_IV"] = current_row["VIX"] * (1 + np.random.normal(0, 0.1))
            current_row["Realized_Vol"] = np.clip(next_vol * np.random.uniform(0.95, 1.05), 5, 50)
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
        blended_vols = [(garch_weight * g) + (1 - garch_weight) * x for g, x in zip(garch_vols, xgb_vols)]
        confidence_score = min(100, max(50, 80 - abs(garch_diff - xgb_diff)))
        iv_change = df["Straddle_Price"].pct_change().iloc[-1] * 100
        bias_tag = "FOMO Risk" if iv_change > 10 else "Neutral"
        forecast_log = pd.DataFrame({"Date": future_dates, "GARCH_Vol": garch_vols, "XGBoost_Vol": xgb_vols, "Blended_Vol": blended_vols, "Confidence": [confidence_score] * forecast_horizon})
        return forecast_log, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, bias_tag
    except Exception as e:
        logger.error(f"Error in volatility forecasting: {str(e)}")
        return None, None, None, None, None, None, None, None

@st.cache_data
def generate_trading_strategy(df, forecast_log, realized_vol, risk_tolerance, confidence_score, bias_tag, capital):
    try:
        latest = df.iloc[-1]
        avg_vol = np.mean(forecast_log["Blended_Vol"])
        iv_hv_gap = latest["ATM_IV"] - latest["Realized_Vol"]
        iv_skew = latest["IV_Skew"]
        vix = latest["VIX"]
        dte = latest["Days_to_Expiry"]
        event_flag = latest["Event_Flag"]
        portfolio_data = fetch_portfolio_data(capital)
        weekly_pnl = portfolio_data["weekly_pnl"]
        risk_flags = []
        if vix > 25:
            risk_flags.append("High Volatility Risk")
        if portfolio_data["exposure"] > 70:
            risk_flags.append("High Exposure Risk")
        if weekly_pnl < -0.05 * capital:
            risk_flags.append("High Loss Risk")
        if latest["VIX_Change_Pct"] > 10:
            risk_flags.append("High VIX Spike")
        if risk_flags:
            st.session_state.violations += 1
            if st.session_state.violations >= 2 and not st.session_state.journal_complete:
                st.error("🚨 Discipline Lock: Complete Journaling")
                return [], risk_flags, None, iv_hv_gap, iv_skew, vix, dte
        regime = "EVENT-DRIVEN" if event_flag == 1 else "LOW" if avg_vol < 15 else "MEDIUM" if avg_vol < 20 else "HIGH"
        strategies = [
            {"Strategy": "Butterfly Spread", "Reason": "Low vol & short expiry", "Confidence": 0.8 if iv_hv_gap > 5 and dte < 10 else 0.6, "Tags": ["Neutral", "Theta"], "Risk_Reward": 2.0, "Capital_Required": capital * 0.3, "Max_Loss": capital * 0.1, "Strikes": {"buy_call_lower": latest["NIFTY_Close"] - 100, "sell_call_middle": latest["NIFTY_Close"], "buy_call_upper": latest["NIFTY_Close"] + 100}},
            {"Strategy": "Iron Condor", "Reason": "High vol favors wide range", "Confidence": 0.75 if regime == "HIGH" else 0.65, "Tags": ["Neutral", "Theta"], "Risk_Reward": 1.8, "Capital_Required": capital * 0.35, "Max_Loss": capital * 0.12, "Strikes": {"sell_call": latest["NIFTY_Close"] + 200, "buy_call": latest["NIFTY_Close"] + 300, "sell_put": latest["NIFTY_Close"] - 200, "buy_put": latest["NIFTY_Close"] - 300}},
            {"Strategy": "Iron Fly", "Reason": "Low vol and time decay", "Confidence": 0.7 if regime == "LOW" else 0.55, "Tags": ["Neutral", "Theta"], "Risk_Reward": 1.5, "Capital_Required": capital * 0.25, "Max_Loss": capital * 0.08, "Strikes": {"sell_call": latest["NIFTY_Close"], "buy_call": latest["NIFTY_Close"] + 100, "sell_put": latest["NIFTY_Close"], "buy_put": latest["NIFTY_Close"] - 100}},
            {"Strategy": "Short Strangle", "Reason": "Balanced vol", "Confidence": 0.65 if regime == "MEDIUM" else 0.5, "Tags": ["Neutral", "Premium Selling"], "Risk_Reward": 1.6, "Capital_Required": capital * 0.4, "Max_Loss": capital * 0.15, "Strikes": {"sell_call": latest["NIFTY_Close"] + 400, "sell_put": latest["NIFTY_Close"] - 400}},
            {"Strategy": "Calendar Spread", "Reason": "IV skew and medium vol", "Confidence": 0.7 if iv_hv_gap > 3 and iv_skew > 2 else 0.6, "Tags": ["Volatility Play"], "Risk_Reward": 1.8, "Capital_Required": capital * 0.2, "Max_Loss": capital * 0.07, "Strikes": {"sell_call_near": latest["NIFTY_Close"], "buy_call_far": latest["NIFTY_Close"]}},
            {"Strategy": "Jade Lizard", "Reason": "High IV + call skew", "Confidence": 0.65 if iv_hv_gap > 10 else 0.55, "Tags": ["Skewed", "Volatility"], "Risk_Reward": 1.2, "Capital_Required": capital * 0.3, "Max_Loss": capital * 0.1, "Strikes": {"sell_put": latest["NIFTY_Close"] - 100, "sell_call": latest["NIFTY_Close"], "buy_call": latest["NIFTY_Close"] + 100}}
        ]
        if regime in ["HIGH", "EVENT-DRIVEN"] and vix > 18 and st.session_state.journal_complete:
            strategies.append({"Strategy": "Intraday Straddle", "Reason": "High IV", "Confidence": 0.7, "Tags": ["Intraday"], "Risk_Reward": 1.5, "Capital_Required": capital * 0.2, "Max_Loss": capital * 0.05, "Strikes": {"sell_call": latest["NIFTY_Close"], "sell_put": latest["NIFTY_Close"]}})
        filtered_strategies = [s for s in strategies if s["Capital_Required"] <= capital]
        if risk_tolerance == "Conservative":
            filtered_strategies = [s for s in filtered_strategies if s["Max_Loss"] <= capital * 0.1]
        elif risk_tolerance == "Aggressive":
            filtered_strategies = [s for s in filtered_strategies if s["Confidence"] >= 0.6]
        filtered_strategies = sorted(filtered_strategies, key=lambda x: x["Confidence"], reverse=True)[:5]
        return filtered_strategies, risk_flags, regime, iv_hv_gap, iv_skew, vix, dte
    except Exception as e:
        logger.error(f"Error generating strategy: {str(e)}")
        return [], [], None, 0, 0, 0, 0

def monitor_trades(strategy):
    try:
        client = st.session_state.client
        positions = client.positions()
        for pos in positions:
            if pos["ProfitLoss"] < -strategy["Max_Loss"] * 0.5:
                client.place_order(OrderType="S" if pos["BuySell"] == "B" else "B", Exchange="N", ExchangeType="C", ScripCode=pos["ScripCode"], Qty=pos["Qty"], Price=pos["LastRate"], StopLossPrice=pos["LastRate"] * 0.95)
                st.warning(f"Stop-Loss Triggered for {pos['ScripCode']}")
                trade_log = {"Date": datetime.now(), "Strategy": strategy["Strategy"], "ScripCode": pos["ScripCode"], "Action": "Stop-Loss Exit", "PnL": pos["ProfitLoss"]}
                st.session_state.trades.append(trade_log)
                pd.DataFrame(st.session_state.trades).to_csv("trade_log.csv", index=False)
    except Exception as e:
        logger.error(f"Stop-Loss Error: {str(e)}")

def check_regime_shift(df):
    try:
        client = st.session_state.client
        vix = client.fetch_market_feed([{"Exch": "N", "ExchType": "C", "Symbol": "INDIA VIX"}])["Data"][0]["LastRate"]
        option_chain = client.get_option_chain("N", "NIFTY", df["expiry_timestamp"].iloc[-1])
        last_rates = [x["LastRate"] for x in option_chain["Options"]]
        atm_strike = min([x["StrikeRate"] for x in option_chain["Options"]], key=lambda x: abs(x - df["NIFTY_Close"].iloc[-1]))
        atm_call = next((r for s, r, cp in zip([x["StrikeRate"] for x in option_chain["Options"]], last_rates, [x["CPType"] for x in option_chain["Options"]]) if s == atm_strike and cp == "CE"), 0)
        iv_change = (atm_call / df["Straddle_Price"].iloc[-1] - 1) * 100 if df["Straddle_Price"].iloc[-1] > 0 else 0
        return "EVENT-DRIVEN" if vix > 18 and iv_change > 10 else "HIGH" if vix > 18 else "MEDIUM" if vix > 15 else "LOW"
    except:
        return df["Regime"].iloc[-1] if "Regime" in df else "MEDIUM"

def run_backtest(df, capital, strategy_choice, start_date, end_date):
    try:
        df_backtest = df.loc[start_date:end_date].copy()
        if len(df_backtest) < 50:
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()
        backtest_results = []
        lot_size = 25
        base_transaction_cost = 0.002
        stt = 0.0005
        portfolio_pnl = 0
        risk_free_rate = 0.06 / 126
        for i in range(1, len(df_backtest)):
            day_data = df_backtest.iloc[i]
            prev_day = df_backtest.iloc[i-1]
            date = day_data.name
            avg_vol = df_backtest["Realized_Vol"].iloc[max(0, i-5):i].mean()
            iv_hv_gap = day_data["ATM_IV"] - day_data["Realized_Vol"]
            regime = "EVENT-DRIVEN" if day_data["Event_Flag"] == 1 else "LOW" if avg_vol < 15 else "MEDIUM" if avg_vol < 20 else "HIGH"
            strategy = "Butterfly Spread" if regime == "LOW" and iv_hv_gap > 5 and day_data["Days_to_Expiry"] < 10 else \
                       "Iron Fly" if regime == "LOW" else \
                       "Iron Condor" if regime == "MEDIUM" and iv_hv_gap > 3 else \
                       "Short Strangle" if regime == "MEDIUM" else \
                       "Jade Lizard" if regime == "HIGH" and iv_hv_gap > 10 else \
                       "Iron Condor" if regime == "HIGH" else \
                       "Calendar Spread" if regime == "EVENT-DRIVEN" and day_data["ATM_IV"] > 30 and day_data["Days_to_Expiry"] < 5 else \
                       "Iron Fly"
            if strategy_choice != "All Strategies" and strategy != strategy_choice:
                continue
            if portfolio_pnl < -0.1 * day_data["Total_Capital"]:
                continue
            extra_cost = 0.001 if "Iron" in strategy else 0
            total_cost = base_transaction_cost + extra_cost + stt
            slippage = 0.005 * min(day_data["ATM_IV"] / 20, 2.5) * (1.5 if day_data["Days_to_Expiry"] < 5 else 1.0) * (1.8 if strategy == "Iron Condor" else 2.2 if strategy == "Butterfly Spread" else 1.5 if strategy == "Iron Fly" else 1.6 if strategy == "Short Strangle" else 1.3 if strategy == "Calendar Spread" else 1.4)
            entry_price = day_data["Straddle_Price"]
            lots = max(1, min(int((day_data["Total_Capital"] * (0.10 if regime == "LOW" else 0.08 if regime == "MEDIUM" else 0.06)) / (entry_price * lot_size)), 2))
            decay_factor = max(0.75, 1 - day_data["Days_to_Expiry"] / 10)
            premium = entry_price * lot_size * lots * (1 - slippage - total_cost) * decay_factor
            premium = premium * 0.8 if np.random.rand() < 0.05 else premium
            premium = premium * 0.9 if np.random.rand() < 0.10 else premium
            iv_factor = min(day_data["ATM_IV"] / avg_vol, 1.5) if avg_vol != 0 else 1.0
            breakeven = entry_price * (1 + iv_factor * (0.04 if (day_data["Event_Flag"] == 1 or day_data["ATM_IV"] > 25) else 0.06))
            nifty_move = abs(day_data["NIFTY_Close"] - prev_day["NIFTY_Close"])
            loss = min(max(0, nifty_move - breakeven) * lot_size * lots, premium * (0.6 if strategy in ["Iron Fly", "Iron Condor"] else 0.8))
            pnl = premium - loss
            shock_prob = 0.35 if day_data["Event_Flag"] == 1 else 0.20
            if np.random.rand() < shock_prob:
                shock_factor = nifty_move / (day_data["ATM_IV"] * 100) if day_data["ATM_IV"] != 0 else 1.0
                pnl -= abs(pnl) * min(shock_factor * 1.5, 2.0)
            if (day_data["Event_Flag"] == 1 or day_data["ATM_IV"] > 25) and np.random.rand() < 0.08:
                pnl -= premium * np.random.uniform(0.5, 1.0)
            if np.random.rand() < 0.02:
                pnl -= premium * np.random.uniform(1.0, 1.5)
            max_loss = (day_data["Total_Capital"] * (0.10 if regime == "LOW" else 0.08 if regime == "MEDIUM" else 0.06)) * 0.025
            pnl = max(-max_loss, min(pnl, max_loss * 1.5))
            portfolio_pnl += pnl
            backtest_results.append({"Date": date, "Regime": regime, "Strategy": strategy, "PnL": pnl, "Capital_Deployed": day_data["Total_Capital"] * (0.10 if regime == "LOW" else 0.08 if regime == "MEDIUM" else 0.06), "Max_Loss": max_loss, "Risk_Reward": 2.0 if regime == "LOW" else 1.8 if regime == "MEDIUM" else 1.2})
        backtest_df = pd.DataFrame(backtest_results)
        if len(backtest_df) == 0:
            return backtest_df, 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()
        total_pnl = backtest_df["PnL"].sum()
        win_rate = len(backtest_df[backtest_df["PnL"] > 0]) / len(backtest_df) if len(backtest_df) > 0 else 0
        max_drawdown = (backtest_df["PnL"].cumsum().cummax() - backtest_df["PnL"].cumsum()).max() if len(backtest_df) > 0 else 0
        backtest_df.set_index("Date", inplace=True)
        returns = backtest_df["PnL"] / df_backtest["Total_Capital"].reindex(backtest_df.index, method="ffill")
        nifty_returns = df_backtest["NIFTY_Close"].pct_change().reindex(backtest_df.index, method="ffill").fillna(0)
        excess_returns = returns - nifty_returns - risk_free_rate
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(126) if excess_returns.std() != 0 else 0
        sortino_ratio = excess_returns.mean() / excess_returns[excess_returns < 0].std() * np.sqrt(126) if len(excess_returns[excess_returns < 0]) > 0 and excess_returns[excess_returns < 0].std() != 0 else 0
        calmar_ratio = (total_pnl / capital) / (max_drawdown / capital) if max_drawdown != 0 else 0
        strategy_perf = backtest_df.groupby("Strategy")["PnL"].agg(['sum', 'count', 'mean']).reset_index()
        strategy_perf["Win_Rate"] = backtest_df.groupby("Strategy")["PnL"].apply(lambda x: len(x[x > 0]) / len(x) if len(x) > 0 else 0).reset_index(drop=True)
        regime_perf = backtest_df.groupby("Regime")["PnL"].agg(['sum', 'count', 'mean']).reset_index()
        regime_perf["Win_Rate"] = backtest_df.groupby("Regime")["PnL"].apply(lambda x: len(x[x > 0]) / len(x) if len(x) > 0 else 0).reset_index(drop=True)
        return backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

def reflection_chart():
    if os.path.exists("journal_log.csv"):
        journal_df = pd.read_csv("journal_log.csv")
        if not journal_df.empty:
            journal_df["Date"] = pd.to_datetime(journal_df["Date"])
            journal_df = journal_df.sort_values("Date")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=journal_df["Date"], y=journal_df["Discipline_Score"], mode="lines+markers", name="Discipline Score", line=dict(color="#e94560")))
            fig.update_layout(title="Discipline Score Over Time", xaxis_title="Date", yaxis_title="Discipline Score", template="plotly_dark")
            return fig
    return None

# Main Execution
if not st.session_state.logged_in:
    st.info("Please login to 5paisa from the sidebar.")
else:
    st.markdown("<h1 style='color: #e94560; text-align: center;'>🛡️ VolGuard Pro: Your AI Trading Copilot</h1>", unsafe_allow_html=True)
    tabs = st.tabs(["Snapshot", "Forecast", "Strategy", "Portfolio", "Journal", "Backtest"])
    
    if run_button:
        with st.spinner("Running VolGuard Analysis..."):
            st.session_state.backtest_run = False
            st.session_state.backtest_results = None
            st.session_state.violations = 0
            st.session_state.journal_complete = False
            df, real_data = load_data()
            if df is not None:
                df = generate_synthetic_features(df, real_data, capital)
                if df is not None:
                    df["Regime"] = check_regime_shift(df)
                    df["expiry_timestamp"] = real_data["expiry_timestamp"] if real_data else int((datetime.now() + timedelta(days=7)).timestamp() * 1000)
                    
                    # Snapshot Tab
                    with tabs[0]:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader("📊 Market Snapshot")
                        last_date = df.index[-1].strftime("%d-%b-%Y")
                        last_nifty = df["NIFTY_Close"].iloc[-1]
                        prev_nifty = df["NIFTY_Close"].iloc[-2] if len(df) >= 2 else last_nifty
                        last_vix = df["VIX"].iloc[-1]
                        regime = df["Regime"].iloc[-1]
                        regime_class = {"LOW": "regime-low", "MEDIUM": "regime-medium", "HIGH": "regime-high", "EVENT-DRIVEN": "regime-event"}.get(regime, "regime-low")
                        st.markdown(f'<div class="gauge">{regime}</div><div style="text-align: center;">Market Regime</div>', unsafe_allow_html=True)
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("NIFTY 50", f"{last_nifty:,.2f}", f"{(last_nifty - prev_nifty)/prev_nifty*100:+.2f}%")
                        with col2:
                            st.metric("India VIX", f"{last_vix:.2f}%", f"{df['VIX_Change_Pct'].iloc[-1]:+.2f}%")
                        with col3:
                            st.metric("PCR", f"{df['PCR'].iloc[-1]:.2f}")
                        with col4:
                            st.metric("Straddle Price", f"₹{df['Straddle_Price'].iloc[-1]:,.2f}")
                        st.markdown(f"**Last Updated**: {last_date} {'(LIVE)' if real_data else '(DEMO)'}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Forecast Tab
                    with tabs[1]:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader("📈 Volatility Forecast")
                        with st.spinner("Predicting volatility..."):
                            forecast_log, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, bias_tag = forecast_volatility_future(df, forecast_horizon)
                        if forecast_log is not None:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Avg Blended Volatility", f"{np.mean(blended_vols):.2f}%")
                            with col2:
                                st.metric("Realized Volatility", f"{realized_vol:.2f}%")
                            with col3:
                                st.metric("Model RMSE", f"{rmse:.2f}%")
                                st.markdown(f'<div class="gauge">{int(confidence_score)}%</div><div style="text-align: center;">Confidence</div>', unsafe_allow_html=True)
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=forecast_log["Date"], y=garch_vols, mode="lines", name="GARCH", line=dict(color="#e94560")))
                            fig.add_trace(go.Scatter(x=forecast_log["Date"], y=xgb_vols, mode="lines", name="XGBoost", line=dict(color="#00d4ff")))
                            fig.add_trace(go.Scatter(x=forecast_log["Date"], y=blended_vols, mode="lines", name="Blended", line=dict(color="#ffcc00")))
                            fig.update_layout(title="Volatility Forecast", xaxis_title="Date", yaxis_title="Volatility (%)", template="plotly_dark")
                            st.plotly_chart(fig)
                            st.markdown(f"**Behavioral Bias**: {bias_tag}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Strategy Tab
                    with tabs[2]:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader("🎯 Trading Strategies")
                        strategies, risk_flags, regime, iv_hv_gap, iv_skew, vix, dte = generate_trading_strategy(df, forecast_log, realized_vol, risk_tolerance, confidence_score, bias_tag, capital)
                        if strategies:
                            st.markdown('<div class="strategy-carousel">', unsafe_allow_html=True)
                            for strategy in strategies:
                                st.markdown(f"""
                                    <div class="strategy-card">
                                        <h4>{strategy['Strategy']}</h4>
                                        <p><b>Reason:</b> {strategy['Reason']}</p>
                                        <p><b>Confidence:</b> {strategy['Confidence']:.0%}</p>
                                        <p><b>Risk-Reward:</b> {strategy['Risk_Reward']:.2f}</p>
                                        <p><b>Max Loss:</b> ₹{strategy['Max_Loss']:,.0f}</p>
                                        <p><b>Tags:</b> {', '.join(strategy['Tags'])}</p>
                                        <p><b>Strikes:</b> {', '.join([f"{k}: {v:,.0f}" for k, v in strategy['Strikes'].items()])}</p>
                                    </div>
                                """, unsafe_allow_html=True)
                                if st.button(f"Select {strategy['Strategy']}", key=strategy['Strategy']):
                                    st.session_state.selected_strategy = (strategy, regime, iv_hv_gap, iv_skew, vix)
                                    monitor_trades(strategy)
                            st.markdown('</div>', unsafe_allow_html=True)
                        if risk_flags:
                            st.markdown('<div class="alert-banner">', unsafe_allow_html=True)
                            st.write("**Risk Alerts**: " + ", ".join(risk_flags))
                            st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Portfolio Tab
                    with tabs[3]:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader("💼 Portfolio Overview")
                        portfolio_data = fetch_portfolio_data(capital)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Weekly PnL", f"₹{portfolio_data['weekly_pnl']:,.2f}")
                        with col2:
                            st.metric("Margin Used", f"₹{portfolio_data['margin_used']:,.2f}")
                        with col3:
                            st.metric("Exposure", f"{portfolio_data['exposure']:.2f}%")
                            exposure_pct = min(portfolio_data['exposure'] / 100, 1.0)
                            st.markdown(f'<div class="progress-bar"><div class="progress-fill" style="width: {exposure_pct*100}%"></div></div>', unsafe_allow_html=True)
                        if portfolio_data["positions"]:
                            st.write("**Open Positions**")
                            pos_df = pd.DataFrame(portfolio_data["positions"])
                            st.dataframe(pos_df[["ScripCode", "Qty", "ProfitLoss", "LastRate"]])
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Journal Tab
                    with tabs[4]:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader("📖 Trading Journal")
                        discipline_score = st.slider("Discipline Score (0-100)", 0, 100, 50)
                        reflection = st.text_area("Reflection Notes", placeholder="What went well? What can improve?")
                        if st.button("Submit Journal"):
                            journal_entry = {"Date": datetime.now(), "Discipline_Score": discipline_score, "Reflection": reflection}
                            journal_df = pd.DataFrame([journal_entry])
                            journal_df.to_csv("journal_log.csv", mode='a', header=not os.path.exists("journal_log.csv"), index=False)
                            st.session_state.journal_complete = True
                            st.success("Journal entry saved!")
                        fig = reflection_chart()
                        if fig:
                            st.plotly_chart(fig)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Backtest Tab
                    with tabs[5]:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader("🔍 Backtest Results")
                        backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf = run_backtest(df, capital, strategy_choice, start_date, end_date)
                        if not backtest_df.empty:
                            st.session_state.backtest_run = True
                            st.session_state.backtest_results = (backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf)
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total PnL", f"₹{total_pnl:,.2f}")
                            with col2:
                                st.metric("Win Rate", f"{win_rate:.2%}")
                            with col3:
                                st.metric("Max Drawdown", f"₹{max_drawdown:,.2f}")
                            col4, col5, col6 = st.columns(3)
                            with col4:
                                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                            with col5:
                                st.metric("Sortino Ratio", f"{sortino_ratio:.2f}")
                            with col6:
                                st.metric("Calmar Ratio", f"{calmar_ratio:.2f}")
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df["PnL"].cumsum(), mode="lines", name="Cumulative PnL", line=dict(color="#e94560")))
                            fig.update_layout(title="Backtest Cumulative PnL", xaxis_title="Date", yaxis_title="PnL (₹)", template="plotly_dark")
                            st.plotly_chart(fig)
                            st.write("**Strategy Performance**")
                            st.dataframe(strategy_perf)
                            st.write("**Regime Performance**")
                            st.dataframe(regime_perf)
                        else:
                            st.warning("No backtest results available. Adjust parameters and try again.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Footer
                    st.markdown('<div class="footer">', unsafe_allow_html=True)
                    st.markdown("**SEBI Disclaimer**: Trading involves risks. Past performance is not indicative of future results. Consult a financial advisor.")
                    st.markdown('</div>', unsafe_allow_html=True)
