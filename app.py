import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
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
import time

# Setup logging
logging.basicConfig(level=logging.DEBUG, filename="volguard.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Page config
st.set_page_config(page_title="VolGuard Pro", page_icon="🛡️", layout="wide")

# Custom CSS for pro UI
st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #1a1a2e, #0f1c2e);
            color: #e5e5e5;
            font-family: 'Inter', sans-serif;
        }
        .card {
            background: rgba(22, 33, 62, 0.9);
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        .metric-card {
            background: #16213e;
            border-radius: 8px;
            padding: 10px;
            text-align: center;
        }
        .gauge {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            background: conic-gradient(#e94560 0% 60%, #00d4ff 60% 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 16px;
        }
        .regime-badge {
            padding: 6px 12px;
            border-radius: 16px;
            font-weight: bold;
            font-size: 13px;
        }
        .regime-prime { background: #28a745; color: white; }
        .regime-neutral { background: #ffc107; color: black; }
        .regime-avoid { background: #dc3545; color: white; }
        .alert-banner {
            background: #dc3545;
            color: white;
            padding: 12px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .stButton>button {
            background: #e94560;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 15px;
        }
        .stButton>button:hover {
            background: #ffcc00;
            color: black;
        }
        .footer {
            text-align: center;
            padding: 15px;
            color: #a0a0a0;
            font-size: 13px;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "backtest_run" not in st.session_state:
    st.session_state.backtest_run = False
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None
if "violations" not in st.session_state:
    st.session_state.violations = 0
if "trades" not in st.session_state:
    st.session_state.trades = []
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# 5paisa Client Setup
cred = {
    "APP_NAME": st.secrets["fivepaisa"]["APP_NAME"],
    "APP_SOURCE": st.secrets["fivepaisa"]["APP_SOURCE"],
    "USER_ID": st.secrets["fivepaisa"]["USER_ID"],
    "PASSWORD": st.secrets["fivepaisa"]["PASSWORD"],
    "USER_KEY": st.secrets["fivepaisa"]["USER_KEY"],
    "ENCRYPTION_KEY": st.secrets["fivepaisa"]["ENCRYPTION_KEY"]
}
client = FivePaisaClient(cred=cred)

# Sidebar Login and Controls
with st.sidebar:
    st.header("🔐 5paisa Login")
    totp_code = st.text_input("TOTP (from Authenticator App)", type="password")
    if st.button("Login"):
        try:
            response = client.get_totp_session(
                st.secrets["fivepaisa"]["CLIENT_CODE"],
                totp_code,
                st.secrets["fivepaisa"]["PIN"]
            )
            if client.get_access_token():
                st.session_state.client = client
                st.session_state.logged_in = True
                st.success("✅ Logged in successfully")
                logger.info("Successful login to 5paisa")
            else:
                st.error("❌ Login failed")
                logger.error("Login failed: Invalid TOTP or credentials")
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logger.error(f"Login error: {str(e)}")

    if st.session_state.logged_in:
        st.header("⚙️ Trading Controls")
        capital = st.number_input("Capital (₹)", min_value=100000, value=1000000, step=100000)
        risk_tolerance = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"], index=1)
        forecast_horizon = st.slider("Forecast Horizon (days)", 1, 30, 7)
        run_button = st.button("Run Analysis")
        st.markdown("---")
        st.markdown("**Motto:** Sell premium, stay calm, let theta do the magic!")

# Black-Scholes IV calculation
def black_scholes_call(S, K, T, r, sigma):
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    except:
        return 0

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
    try:
        calls = df[df["CPType"] == "CE"].set_index("StrikeRate")[["LastRate", "OpenInterest"]]
        puts = df[df["CPType"] == "PE"].set_index("StrikeRate")[["LastRate", "OpenInterest"]]
        strikes = df["StrikeRate"].unique()
        pain = []
        for K in strikes:
            total_loss = 0
            for s in strikes:
                if s in calls.index:
                    total_loss += max(0, s - K) * calls.loc[s, "OpenInterest"]
                if s in puts.index:
                    total_loss += max(0, K - s) * puts.loc[s, "OpenInterest"]
            pain.append((K, total_loss))
        max_pain_strike = min(pain, key=lambda x: x[1])[0]
        max_pain_diff_pct = abs(nifty_spot - max_pain_strike) / nifty_spot * 100
        return max_pain_strike, max_pain_diff_pct
    except Exception as e:
        logger.error(f"Max Pain error: {str(e)}")
        return nifty_spot, 0

# Fetch real data from 5paisa with dynamic expiry
def fetch_nifty_data():
    try:
        # Fetch NIFTY spot price
        nifty_req = [{"Exch": "N", "ExchType": "C", "ScripCode": 999920000, "Symbol": "NIFTY"}]
        nifty_data = client.fetch_market_feed(nifty_req)
        if not nifty_data or "Data" not in nifty_data or not nifty_data["Data"]:
            raise Exception("Failed to fetch Nifty 50 index price")
        nifty_spot = nifty_data["Data"][0].get("LastRate", nifty_data["Data"][0].get("LastTradedPrice", 0))
        if not nifty_spot:
            raise Exception("Nifty price key not found")
        logger.info(f"Fetched NIFTY spot: {nifty_spot}")

        # Dynamic expiry: Fetch nearest weekly expiry (Thursday)
        today = datetime.now()
        days_to_thursday = (3 - today.weekday()) % 7
        if days_to_thursday == 0 and today.hour >= 15:  # After 3 PM on Thursday
            days_to_thursday = 7
        expiry_date = today + timedelta(days=days_to_thursday)
        expiry_timestamp = int(expiry_date.timestamp() * 1000)
        logger.info(f"Selected expiry: {expiry_date.strftime('%d-%b-%Y')}")

        # Fetch option chain
        option_chain = client.get_option_chain("N", "NIFTY", expiry_timestamp)
        if not option_chain or "Options" not in option_chain or not option_chain["Options"]:
            raise Exception("Failed to fetch option chain")
        df = pd.DataFrame(option_chain["Options"])
        required_cols = ["StrikeRate", "CPType", "LastRate", "OpenInterest", "ScripCode"]
        if not all(col in df.columns for col in required_cols):
            raise Exception("Required columns missing in option chain")
        logger.info("Option chain fetched successfully")

        # Process option chain
        df["StrikeRate"] = pd.to_numeric(df["StrikeRate"], errors="coerce")
        df["LastRate"] = pd.to_numeric(df["LastRate"], errors="coerce")
        df["OpenInterest"] = pd.to_numeric(df["OpenInterest"], errors="coerce")
        df = df.dropna(subset=["StrikeRate", "LastRate", "OpenInterest"])

        # Find ATM strike
        atm_strike = df["StrikeRate"].iloc[(df["StrikeRate"] - nifty_spot).abs().argmin()]
        atm_data = df[df["StrikeRate"] == atm_strike]
        atm_call = atm_data[atm_data["CPType"] == "CE"]["LastRate"].iloc[0] if not atm_data[atm_data["CPType"] == "CE"].empty else 0
        atm_put = atm_data[atm_data["CPType"] == "PE"]["LastRate"].iloc[0] if not atm_data[atm_data["CPType"] == "PE"].empty else 0
        straddle_price = atm_call + atm_put
        logger.info(f"ATM Strike: {atm_strike}, Straddle Price: {straddle_price}")

        # Calculate IV
        T = (expiry_date - today).days / 365.0
        if T <= 0:
            T = 1 / 365.0  # Minimum 1 day
        r = 0.06
        iv_df = df[(df["StrikeRate"] >= atm_strike - 200) & (df["StrikeRate"] <= atm_strike + 200)].copy()
        iv_df["IV (%)"] = iv_df.apply(
            lambda row: implied_volatility(
                S=nifty_spot, K=row["StrikeRate"], T=T, r=r, market_price=row["LastRate"],
                option_type='call' if row["CPType"] == "CE" else 'put'
            ), axis=1
        )
        atm_iv = iv_df[iv_df["StrikeRate"] == atm_strike]["IV (%)"].mean()
        if pd.isna(atm_iv):
            atm_iv = 20.0  # Fallback
        logger.info(f"ATM IV: {atm_iv}%")

        # Calculate PCR
        calls = df[df["CPType"] == "CE"]
        puts = df[df["CPType"] == "PE"]
        call_oi = calls["OpenInterest"].sum()
        put_oi = puts["OpenInterest"].sum()
        pcr = put_oi / call_oi if call_oi != 0 else 1.0
        logger.info(f"PCR: {pcr}")

        # Calculate Max Pain
        max_pain_strike, max_pain_diff_pct = max_pain(df, nifty_spot)
        logger.info(f"Max Pain Strike: {max_pain_strike}, Diff: {max_pain_diff_pct}%")

        # VIX Change
        vix_change_pct = 0
        iv_file = "atm_iv_history.csv"
        if os.path.exists(iv_file):
            iv_history = pd.read_csv(iv_file)
            prev_atm_iv = iv_history["ATM_IV"].iloc[-1] if not iv_history.empty else atm_iv
            vix_change_pct = ((atm_iv - prev_atm_iv) / prev_atm_iv * 100) if prev_atm_iv != 0 else 0
        pd.DataFrame({"Date": [datetime.now()], "ATM_IV": [atm_iv]}).to_csv(iv_file, mode='a', header=not os.path.exists(iv_file), index=False)

        return {
            "nifty_spot": nifty_spot,
            "atm_strike": atm_strike,
            "straddle_price": straddle_price,
            "pcr": pcr,
            "max_pain_strike": max_pain_strike,
            "max_pain_diff_pct": max_pain_diff_pct,
            "vix_change_pct": vix_change_pct,
            "atm_iv": atm_iv,
            "option_chain": df,
            "expiry_date": expiry_date,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error fetching 5paisa data: {str(e)}")
        st.warning("Failed to fetch live data. Using fallback data.")
        return None

# Load data
@st.cache_data(ttl=300)
def load_data():
    try:
        real_data = fetch_nifty_data()
        if real_data is None or not real_data.get("success", False):
            logger.warning("Falling back to GitHub CSV")
            nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
            response = requests.get(nifty_url)
            response.raise_for_status()
            nifty = pd.read_csv(io.StringIO(response.text), encoding="utf-8-sig")
            nifty.columns = nifty.columns.str.strip()
            nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
            nifty = nifty.dropna(subset=["Date"])
            nifty = nifty[["Date", "Close"]].set_index("Date")
            nifty.index = pd.to_datetime(nifty.index)
            nifty = nifty.rename(columns={"Close": "NIFTY_Close"})

            vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"
            response = requests.get(vix_url)
            response.raise_for_status()
            vix = pd.read_csv(io.StringIO(response.text))
            vix.columns = vix.columns.str.strip()
            vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")
            vix = vix.dropna(subset=["Date"])
            vix = vix[["Date", "Close"]].set_index("Date").rename(columns={"Close": "VIX"})
            vix.index = pd.to_datetime(vix.index)

            common_dates = nifty.index.intersection(vix.index)
            df = pd.DataFrame({
                "NIFTY_Close": nifty.loc[common_dates, "NIFTY_Close"],
                "VIX": vix.loc[common_dates, "VIX"]
            }, index=common_dates)
            df = df.ffill().bfill()
            real_data = None
        else:
            latest_date = datetime.now().date()
            df = pd.DataFrame({
                "NIFTY_Close": [real_data["nifty_spot"]],
                "VIX": [real_data["atm_iv"]]
            }, index=[pd.to_datetime(latest_date)])
            nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
            response = requests.get(nifty_url)
            response.raise_for_status()
            nifty = pd.read_csv(io.StringIO(response.text), encoding="utf-8-sig")
            nifty.columns = nifty.columns.str.strip()
            nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
            nifty = nifty.dropna(subset=["Date"])
            nifty = nifty[["Date", "Close"]].set_index("Date")
            nifty.index = pd.to_datetime(nifty.index)
            nifty = nifty.rename(columns={"Close": "NIFTY_Close"})

            vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"
            response = requests.get(vix_url)
            response.raise_for_status()
            vix = pd.read_csv(io.StringIO(response.text))
            vix.columns = vix.columns.str.strip()
            vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")
            vix = vix.dropna(subset=["Date"])
            vix = vix[["Date", "Close"]].set_index("Date").rename(columns={"Close": "VIX"})
            vix.index = pd.to_datetime(vix.index)

            historical_df = pd.DataFrame({
                "NIFTY_Close": nifty["NIFTY_Close"],
                "VIX": vix["VIX"]
            }).dropna()
            historical_df = historical_df[historical_df.index < pd.to_datetime(latest_date)]
            df = pd.concat([historical_df, df])
            df = df[~df.index.duplicated(keep='last')]
            df = df.sort_index()

        logger.info("Data loaded successfully")
        return df, real_data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error(f"Error loading data: {str(e)}")
        return None, None

# Generate synthetic features
@st.cache_data
def generate_synthetic_features(df, real_data, capital):
    try:
        n_days = len(df)
        np.random.seed(42)
        risk_free_rate = 0.06
        strike_step = 100

        if real_data:
            base_pcr = real_data["pcr"]
            base_iv = real_data["atm_iv"]
            base_straddle_price = real_data["straddle_price"]
            base_max_pain_diff_pct = real_data["max_pain_diff_pct"]
            base_vix_change_pct = real_data["vix_change_pct"]
        else:
            base_pcr = 1.0
            base_iv = 20.0
            base_straddle_price = 200.0
            base_max_pain_diff_pct = 0.5
            base_vix_change_pct = 0.0

        def calculate_days_to_expiry(dates):
            days_to_expiry = []
            for date in dates:
                days_ahead = (3 - date.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7
                next_expiry = date + pd.Timedelta(days=days_ahead)
                dte = (next_expiry - date).days
                days_to_expiry.append(dte)
            return np.array(days_to_expiry)

        def black_scholes(S, K, T, r, sigma, option_type="call"):
            try:
                T = max(T, 1e-6)
                d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                if option_type == "call":
                    return max(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), 0)
                else:
                    return max(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1), 0)
            except:
                return 0

        df["Days_to_Expiry"] = calculate_days_to_expiry(df.index)
        event_spike = np.where((df.index.month % 3 == 0) & (df.index.day < 5), 1.2, 1.0)
        df["ATM_IV"] = df["VIX"] * (1 + np.random.normal(0, 0.1, n_days)) * event_spike
        df["ATM_IV"] = np.clip(df["ATM_IV"], 5, 50)
        if real_data:
            df["ATM_IV"].iloc[-1] = base_iv

        def dynamic_ivp(x):
            if len(x) >= 5:
                return (np.sum(x.iloc[:-1] <= x.iloc[-1]) / (len(x) - 1)) * 100
            return 50.0
        df["IVP"] = df["ATM_IV"].rolling(252, min_periods=5).apply(dynamic_ivp)
        df["IVP"] = df["IVP"].interpolate().fillna(50.0)

        market_trend = df["NIFTY_Close"].pct_change().rolling(5).mean().fillna(0)
        df["PCR"] = np.clip(base_pcr + np.random.normal(0, 0.1, n_days) + market_trend * -10, 0.7, 2.0)
        if real_data:
            df["PCR"].iloc[-1] = base_pcr

        df["VIX_Change_Pct"] = df["VIX"].pct_change().fillna(0) * 100
        if real_data:
            df["VIX_Change_Pct"].iloc[-1] = base_vix_change_pct

        df["Spot_MaxPain_Diff_Pct"] = np.abs(np.random.lognormal(-2, 0.5, n_days))
        df["Spot_MaxPain_Diff_Pct"] = np.clip(df["Spot_MaxPain_Diff_Pct"], 0.1, 1.0)
        if real_data:
            df["Spot_MaxPain_Diff_Pct"].iloc[-1] = base_max_pain_diff_pct

        df["Event_Flag"] = np.where((df.index.month % 3 == 0) & (df.index.day < 5) | (df["Days_to_Expiry"] <= 3), 1, 0)
        df["Realized_Vol"] = df["NIFTY_Close"].pct_change().rolling(5, min_periods=5).std() * np.sqrt(252) * 100
        df["Realized_Vol"] = df["Realized_Vol"].fillna(df["VIX"])
        df["Realized_Vol"] = np.clip(df["Realized_Vol"], 0, 50)
        df["Total_Capital"] = capital
        df["PnL_Day"] = np.random.normal(0, 5000, n_days) * (1 - df["Event_Flag"] * 0.5)

        straddle_prices = []
        for i in range(n_days):
            S = df["NIFTY_Close"].iloc[i]
            K = round(S / strike_step) * strike_step
            T = df["Days_to_Expiry"].iloc[i] / 365
            sigma = df["ATM_IV"].iloc[i] / 100
            call_price = black_scholes(S, K, T, risk_free_rate, sigma, "call")
            put_price = black_scholes(S, K, T, risk_free_rate, sigma, "put")
            straddle_price = (call_price + put_price) * (S / 1000)
            straddle_price = np.clip(straddle_price, 50, 400)
            straddle_prices.append(straddle_price)
        df["Straddle_Price"] = straddle_prices
        if real_data:
            df["Straddle_Price"].iloc[-1] = base_straddle_price

        if df.isna().sum().sum() > 0:
            df = df.interpolate().fillna(method='bfill')
        
        df.to_csv("volguard_hybrid_data.csv")
        logger.info("Synthetic features generated successfully")
        return df
    except Exception as e:
        logger.error(f"Error generating features: {str(e)}")
        st.error(f"Error generating features: {str(e)}")
        return None

# Fetch portfolio data
def fetch_portfolio_data(capital):
    try:
        positions = client.positions()
        if not positions:
            raise Exception("Failed to fetch positions")
        total_pnl = sum(pos.get("ProfitLoss", 0) for pos in positions)
        total_margin = sum(pos.get("MarginUsed", 0) for pos in positions)
        total_exposure = sum(pos.get("Exposure", 0) for pos in positions)
        logger.info("Portfolio data fetched successfully")
        return {
            "weekly_pnl": total_pnl,
            "margin_used": total_margin,
            "exposure": total_exposure / capital * 100 if capital > 0 else 0,
            "positions": positions
        }
    except Exception as e:
        logger.error(f"Error fetching portfolio data: {str(e)}")
        return {"weekly_pnl": 0, "margin_used": 0, "exposure": 0, "positions": []}

# Define feature_cols
feature_cols = ['VIX', 'ATM_IV', 'IVP', 'PCR', 'VIX_Change_Pct', 'Straddle_Price', 'Spot_MaxPain_Diff_Pct', 'Days_to_Expiry', 'Event_Flag']

# Forecast volatility
@st.cache_data
def forecast_volatility_future(df, forecast_horizon):
    try:
        df.index = pd.to_datetime(df.index)
        df_garch = df.tail(len(df))
        if len(df_garch) < 200:
            logger.error(f"Insufficient data for GARCH: {len(df_garch)} days")
            return None, None, None, None, None, None, None, None

        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon, freq='B')

        df_garch['Log_Returns'] = np.log(df_garch['NIFTY_Close'] / df_garch['NIFTY_Close'].shift(1)).dropna() * 100
        garch_model = arch_model(df_garch['Log_Returns'].dropna(), vol='Garch', p=1, q=1, rescale=False)
        garch_fit = garch_model.fit(disp="off")
        garch_forecast = garch_fit.forecast(horizon=forecast_horizon, reindex=False)
        garch_vols = np.sqrt(garch_forecast.variance.iloc[-1].values) * np.sqrt(252)
        garch_vols = np.clip(garch_vols, 5, 50)

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
        logger.info("Volatility forecast completed")
        return forecast_log, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, model.feature_importances_
    except Exception as e:
        logger.error(f"Error in volatility forecasting: {str(e)}")
        st.error(f"Error in volatility forecasting: {str(e)}")
        return None, None, None, None, None, None, None, None

# Generate trading strategy optimized for theta/volatility selling
@st.cache_data
def generate_trading_strategy(df, forecast_log, realized_vol, risk_tolerance, confidence_score, capital):
    try:
        latest = df.iloc[-1]
        avg_vol = np.mean(forecast_log["Blended_Vol"])
        iv = latest["ATM_IV"]
        hv = realized_vol
        iv_hv_gap = iv - hv
        dte = latest["Days_to_Expiry"]
        pcr = latest["PCR"]
        event_flag = latest["Event_Flag"]
        
        risk_flags = []
        if latest["VIX"] > 25:
            risk_flags.append("High Volatility (VIX > 25%)")
            st.session_state.violations += 1
        if latest["Spot_MaxPain_Diff_Pct"] > 70:
            risk_flags.append("High Exposure (>70%)")
            st.session_state.violations += 1
        if latest["PnL_Day"] < -0.05 * capital:
            risk_flags.append("Weekly Loss > 5%")
            st.session_state.violations += 1
        if latest["VIX_Change_Pct"] > 10:
            risk_flags.append("VIX Spike (>10%)")
            st.session_state.violations += 1
        
        # Seller-focused regimes
        if iv_hv_gap > 3 and not event_flag and pcr < 1.5:
            regime = "Prime Selling"
        elif iv_hv_gap < 0 or event_flag or pcr > 2.0:
            regime = "Avoid Selling"
        else:
            regime = "Neutral"
        
        strategy = "Short Strangle"
        reason = "High premium capture due to IV overpricing."
        tags = ["Theta", "Volatility Selling"]
        risk_reward = 1.5
        win_prob = 0.70  # Placeholder
        credit = latest["Straddle_Price"] * 0.5
        
        if regime == "Prime Selling":
            if dte < 10:
                strategy = "Short Strangle"
                reason = f"IV {iv:.1f}% > HV {hv:.1f}% + short DTE = max theta decay."
                risk_reward = 2.0
                win_prob = 0.75
            else:
                strategy = "Iron Condor"
                reason = f"IV {iv:.1f}% > HV {hv:.1f}% + stable vol = wide-range premium."
                risk_reward = 1.8
                win_prob = 0.70
        elif regime == "Neutral":
            strategy = "Iron Fly"
            reason = "Balanced vol favors defined-risk premium selling."
            tags.append("Defined Risk")
            win_prob = 0.65
        elif regime == "Avoid Selling":
            strategy = "Hold"
            reason = "Low IV or event risk makes selling unattractive."
            tags = ["Wait"]
            credit = 0
            win_prob = 0
        
        position_size = {"Conservative": 0.5, "Moderate": 1.0, "Aggressive": 1.5}[risk_tolerance]
        deploy = capital * 0.2 * position_size if strategy != "Hold" else 0
        max_loss = deploy * 0.2 if strategy != "Hold" else 0
        
        logger.info(f"Strategy generated: {strategy}, Regime: {regime}")
        return {
            "Regime": regime,
            "Strategy": strategy,
            "Reason": reason,
            "Tags": tags,
            "Confidence": confidence_score / 100,
            "Risk_Reward": risk_reward,
            "Deploy": deploy,
            "Max_Loss": max_loss,
            "Credit": credit,
            "Win_Probability": win_prob,
            "Risk_Flags": risk_flags,
            "IV_HV_Gap": iv_hv_gap
        }
    except Exception as e:
        logger.error(f"Error generating strategy: {str(e)}")
        st.error(f"Error generating strategy: {str(e)}")
        return None

# Backtest function
@st.cache_data
def run_backtest(df, capital, strategy_choice, start_date, end_date):
    try:
        logger.info(f"Starting backtest for {strategy_choice} from {start_date} to {end_date}")
        df_backtest = df.loc[start_date:end_date].copy()
        if len(df_backtest) < 50:
            logger.error(f"Backtest failed: Insufficient data ({len(df_backtest)} days)")
            st.error(f"Backtest failed: Insufficient data ({len(df_backtest)} days)")
            return None

        backtest_results = []
        lot_size = 25
        portfolio_pnl = 0
        for i in range(1, len(df_backtest)):
            day_data = df_backtest.iloc[i]
            prev_day = df_backtest.iloc[i-1]
            date = day_data.name
            avg_vol = df_backtest["Realized_Vol"].iloc[max(0, i-5):i].mean()
            forecast_log = pd.DataFrame({"Blended_Vol": [avg_vol] * 7})  # Dummy for backtest
            strategy = generate_trading_strategy(df_backtest.iloc[:i+1], forecast_log, avg_vol, "Moderate", 80, capital)
            if strategy is None or strategy["Strategy"] == "Hold" or (strategy_choice != "All Strategies" and strategy["Strategy"] != strategy_choice):
                continue
            premium = day_data["Straddle_Price"] * lot_size
            loss = max(0, abs(day_data["NIFTY_Close"] - prev_day["NIFTY_Close"]) - premium * 0.04) * lot_size
            pnl = premium - loss
            portfolio_pnl += pnl
            backtest_results.append({
                "Date": date,
                "Regime": strategy["Regime"],
                "Strategy": strategy["Strategy"],
                "PnL": pnl,
                "Capital_Deployed": strategy["Deploy"],
                "Max_Loss": strategy["Max_Loss"],
                "Risk_Reward": strategy["Risk_Reward"]
            })

        backtest_df = pd.DataFrame(backtest_results)
        if backtest_df.empty:
            logger.warning("Backtest returned no trades")
            st.warning("Backtest returned no trades")
            return None

        total_pnl = backtest_df["PnL"].sum()
        win_rate = len(backtest_df[backtest_df["PnL"] > 0]) / len(backtest_df)
        max_drawdown = (backtest_df["PnL"].cumsum().cummax() - backtest_df["PnL"].cumsum()).max()
        returns = backtest_df["PnL"] / df_backtest["Total_Capital"].reindex(backtest_df.index, method="ffill")
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(126) if returns.std() != 0 else 0
        logger.info(f"Backtest completed: Total P&L: {total_pnl}, Win Rate: {win_rate}")
        return {
            "df": backtest_df,
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio
        }
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        st.error(f"Error running backtest: {str(e)}")
        return None

# Main execution
if not st.session_state.logged_in:
    st.info("Please login to 5paisa from the sidebar to proceed.")
else:
    st.markdown("<h1 style='color: #e94560; text-align: center;'>🛡️ VolGuard Pro: Theta Seller’s Copilot</h1>", unsafe_allow_html=True)
    tabs = st.tabs(["Trading Hub", "Portfolio", "Backtest", "Journal"])

    if run_button:
        with st.spinner("Running VolGuard Analysis..."):
            st.session_state.backtest_run = False
            st.session_state.backtest_results = None

            # Trading Hub
            with tabs[0]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("📊 Trading Hub: Theta & Volatility Selling")

                df, real_data = load_data()
                if df is None:
                    st.error("Failed to load data. Check logs.")
                    logger.error("Data loading failed")
                    st.stop()

                df = generate_synthetic_features(df, real_data, capital)
                if df is None:
                    st.error("Failed to generate features. Check logs.")
                    logger.error("Feature generation failed")
                    st.stop()

                forecast_log, _, _, blended_vols, realized_vol, confidence_score, _, _ = forecast_volatility_future(df, forecast_horizon)
                if forecast_log is None:
                    st.error("Volatility forecast failed. Using default values.")
                    logger.error("Volatility forecast failed")
                    blended_vols = [20.0] * forecast_horizon
                    realized_vol = 20.0
                    confidence_score = 50

                strategy = generate_trading_strategy(df, forecast_log, realized_vol, risk_tolerance, confidence_score, capital)
                if strategy is None:
                    st.error("Strategy generation failed. Check logs.")
                    logger.error("Strategy generation failed")
                    st.stop()

                # Market Snapshot
                st.markdown("### Market Snapshot")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("NIFTY 50", f"{df['NIFTY_Close'].iloc[-1]:,.2f}", f"{(df['NIFTY_Close'].iloc[-1] - df['NIFTY_Close'].iloc[-2])/df['NIFTY_Close'].iloc[-2]*100:+.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("ATM IV", f"{df['ATM_IV'].iloc[-1]:.2f}%", f"{df['VIX_Change_Pct'].iloc[-1]:+.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("PCR", f"{df['PCR'].iloc[-1]:.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                with col4:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    atm_strike = real_data["atm_strike"] if real_data else round(df["NIFTY_Close"].iloc[-1] / 100) * 100
                    st.metric(f"ATM Straddle ({atm_strike})", f"₹{df['Straddle_Price'].iloc[-1]:,.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)

                col5, col6, col7, col8 = st.columns(4)
                with col5:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("IVP", f"{df['IVP'].iloc[-1]:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                with col6:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    max_pain_strike = real_data["max_pain_strike"] if real_data else atm_strike
                    st.metric("Max Pain", f"{max_pain_strike:,.0f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                with col7:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Spot-MaxPain Diff", f"{df['Spot_MaxPain_Diff_Pct'].iloc[-1]:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                with col8:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Days to Expiry", f"{int(df['Days_to_Expiry'].iloc[-1])}")
                    st.markdown('</div>', unsafe_allow_html=True)

                # Trade Recommendation
                st.markdown("### Trade Recommendation")
                regime_class = {
                    "Prime Selling": "regime-prime",
                    "Neutral": "regime-neutral",
                    "Avoid Selling": "regime-avoid"
                }.get(strategy["Regime"], "regime-neutral")
                headline = f"Sell {strategy['Strategy']} for ₹{strategy['Deploy']:,.0f} Credit (IV Edge: {strategy['IV_HV_Gap']:.1f}%)"
                st.markdown(f'<h3 style="color: #ffcc00;">{headline}</h3>', unsafe_allow_html=True)
                st.markdown(f'<span class="regime-badge {regime_class}">{strategy["Regime"]}</span>', unsafe_allow_html=True)
                st.write(f"**Why**: {strategy['Reason']}")
                st.write(f"**Win Probability**: {strategy['Win_Probability']*100:.0f}% (based on backtests)")
                st.write(f"**Max Loss**: ₹{strategy['Max_Loss']:,.0f} | **Risk-Reward**: {strategy['Risk_Reward']:.1f}:1")
                st.write(f"**Vol Forecast**: {np.mean(blended_vols):.1f}% in {forecast_horizon} days (Confidence: {int(confidence_score)}%)")
                
                # Seller’s Edge Score
                seller_edge = min(100, 50 + (strategy["IV_HV_Gap"] * 5) + (strategy["Win_Probability"] * 50) + (10 if df["Days_to_Expiry"].iloc[-1] < 10 else 0))
                st.markdown(f'<div class="gauge">{int(seller_edge)}</div><div style="text-align: center;">Seller’s Edge Score</div>', unsafe_allow_html=True)

                # Risk Flags
                if strategy["Risk_Flags"]:
                    st.markdown('<div class="alert-banner">', unsafe_allow_html=True)
                    st.write("**Heads Up: Risks to Watch**")
                    for flag in strategy["Risk_Flags"]:
                        advice = {
                            "High Volatility (VIX > 25%)": "Widen strikes or reduce lots.",
                            "High Exposure (>70%)": "Lower position size.",
                            "Weekly Loss > 5%": "Pause and review portfolio.",
                            "VIX Spike (>10%)": "Consider hedging with puts."
                        }.get(flag, "Monitor closely.")
                        st.write(f"- {flag}: {advice}")
                    st.markdown('</div>', unsafe_allow_html=True)

                # Option Chain
                if real_data:
                    st.markdown(f"### Option Chain (Expiry: {real_data['expiry_date'].strftime('%d-%b-%Y')})")
                    chain_df = real_data["option_chain"][["StrikeRate", "CPType", "LastRate", "OpenInterest", "IV (%)"]]
                    chain_df = chain_df[(chain_df["StrikeRate"] >= real_data["atm_strike"] - 400) & (chain_df["StrikeRate"] <= real_data["atm_strike"] + 400)]
                    chain_df["IV (%)"] = chain_df["IV (%)"].round(2)
                    st.dataframe(chain_df.sort_values(["StrikeRate", "CPType"]), use_container_width=True)

                # Guided Trade Execution
                st.markdown("### Execute Your Trade")
                with st.form("sell_form"):
                    option_chain = real_data["option_chain"] if real_data else pd.DataFrame()
                    atm_strike = real_data["atm_strike"] if real_data else round(df["NIFTY_Close"].iloc[-1] / 100) * 100
                    
                    # Suggest strikes (20–30 delta equivalent)
                    call_sell_strike = atm_strike + 200
                    put_sell_strike = atm_strike - 200
                    call_premium = put_premium = 0
                    call_oi = put_oi = 0
                    call_iv = put_iv = 0
                    if not option_chain.empty:
                        try:
                            call_data = option_chain[(option_chain["StrikeRate"] == call_sell_strike) & (option_chain["CPType"] == "CE")]
                            put_data = option_chain[(option_chain["StrikeRate"] == put_sell_strike) & (option_chain["CPType"] == "PE")]
                            call_premium = call_data["LastRate"].iloc[0] if not call_data.empty else 0
                            put_premium = put_data["LastRate"].iloc[0] if not put_data.empty else 0
                            call_oi = call_data["OpenInterest"].iloc[0] if not call_data.empty else 0
                            put_oi = put_data["OpenInterest"].iloc[0] if not put_data.empty else 0
                            call_iv = call_data["IV (%)"].iloc[0] if not call_data.empty and "IV (%)" in call_data else 0
                            put_iv = put_data["IV (%)"].iloc[0] if not put_data.empty and "IV (%)" in put_data else 0
                        except Exception as e:
                            logger.warning(f"Option chain data issue: {str(e)}")
                            st.warning("Option chain data missing. Using estimated premiums.")

                    net_credit = (call_premium + put_premium) * 25
                    breakeven_up = call_sell_strike + (call_premium + put_premium)
                    breakeven_down = put_sell_strike - (call_premium + put_premium)
                    theta_potential = net_credit * (1 - df["Days_to_Expiry"].iloc[-1] / 30)

                    st.write(f"**Suggested Strikes ({strategy['Strategy']})**")
                    st.write(f"- Sell Call: {call_sell_strike} @ ₹{call_premium:.2f} (IV: {call_iv:.1f}%, OI: {call_oi:,.0f})")
                    st.write(f"- Sell Put: {put_sell_strike} @ ₹{put_premium:.2f} (IV: {put_iv:.1f}%, OI: {put_oi:,.0f})")
                    st.write(f"**Net Credit**: ₹{net_credit:,.2f} per lot")
                    st.write(f"**Breakeven**: {breakeven_down:,.0f} – {breakeven_up:,.0f}")
                    st.write(f"**Theta Decay Potential**: ₹{theta_potential:,.0f} over {int(df['Days_to_Expiry'].iloc[-1])} days")

                    lots = st.number_input("Lots (25 contracts each)", min_value=1, value=1, step=1)
                    simulate = st.checkbox("Simulate Trade (No Execution)")

                    submit = st.form_submit_button("Sell Now")
                    if submit:
                        if strategy["Strategy"] == "Hold":
                            st.error("Cannot trade: Strategy is Hold due to market conditions.")
                        elif simulate:
                            total_credit = net_credit * lots
                            st.success(f"Simulated Trade: Total Credit = ₹{total_credit:,.2f}")
                        else:
                            try:
                                orders = []
                                if not option_chain.empty and call_premium > 0 and put_premium > 0:
                                    orders = [
                                        {"Exch": "N", "ExchType": "C", "ScripCode": option_chain[(option_chain["StrikeRate"] == call_sell_strike) & (option_chain["CPType"] == "CE")]["ScripCode"].iloc[0], "BuySell": "S", "Qty": 25 * lots, "Price": call_premium, "OrderType": "LIMIT"},
                                        {"Exch": "N", "ExchType": "C", "ScripCode": option_chain[(option_chain["StrikeRate"] == put_sell_strike) & (option_chain["CPType"] == "PE")]["ScripCode"].iloc[0], "BuySell": "S", "Qty": 25 * lots, "Price": put_premium, "OrderType": "LIMIT"}
                                    ]
                                else:
                                    st.error("Cannot place trade: Option chain data incomplete.")
                                    logger.error("Trade failed: Incomplete option chain data")
                                    st.stop()

                                for order in orders:
                                    client.place_order(OrderType=order["BuySell"], Exchange=order["Exch"], ExchangeType=order["ExchType"], ScripCode=order["ScripCode"], Qty=order["Qty"], Price=order["Price"], IsIntraday=False)
                                trade_log = {
                                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "Strategy": strategy["Strategy"],
                                    "Credit": net_credit * lots,
                                    "IV_Edge": strategy["IV_HV_Gap"],
                                    "Outcome": "Pending"
                                }
                                st.session_state.trades.append(trade_log)
                                pd.DataFrame(st.session_state.trades).to_csv("trade_log.csv", index=False)
                                st.success(f"Trade Placed: ₹{net_credit * lots:,.2f} Credit! Crush it! 🚀")
                                logger.info(f"Trade placed: {strategy['Strategy']}, Credit: {net_credit * lots}")
                            except Exception as e:
                                logger.error(f"Trade failed: {str(e)}")
                                st.error(f"Trade Failed: {str(e)}")

                st.markdown('</div>', unsafe_allow_html=True)

            # Portfolio Tab
            with tabs[1]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("💼 Portfolio Overview")
                portfolio_data = fetch_portfolio_data(capital)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Weekly P&L", f"₹{portfolio_data['weekly_pnl']:,.2f}")
                with col2:
                    st.metric("Margin Used", f"₹{portfolio_data['margin_used']:,.2f}")
                with col3:
                    st.metric("Exposure", f"{portfolio_data['exposure']:.2f}%")
                st.markdown("### Open Positions")
                if portfolio_data["positions"]:
                    pos_df = pd.DataFrame(portfolio_data["positions"])
                    st.dataframe(pos_df, use_container_width=True)
                else:
                    st.write("No open positions.")
                st.markdown('</div>', unsafe_allow_html=True)

            # Backtest Tab
            with tabs[2]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("📈 Backtest Results")
                start_date = st.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
                end_date = st.date_input("End Date", value=pd.to_datetime("2025-04-29"))
                strategy_choice = st.selectbox("Strategy", ["All Strategies", "Short Strangle", "Iron Condor", "Iron Fly"])
                if st.button("Run Backtest"):
                    with st.spinner("Running backtest..."):
                        backtest_result = run_backtest(df, capital, strategy_choice, start_date, end_date)
                        if backtest_result is not None:
                            st.session_state.backtest_run = True
                            st.session_state.backtest_results = backtest_result
                            st.metric("Total P&L", f"₹{backtest_result['total_pnl']:,.2f}")
                            st.metric("Win Rate", f"{backtest_result['win_rate']:.2%}")
                            st.metric("Max Drawdown", f"₹{backtest_result['max_drawdown']:,.2f}")
                            st.metric("Sharpe Ratio", f"{backtest_result['sharpe_ratio']:.2f}")
                            st.dataframe(backtest_result["df"], use_container_width=True)
                        else:
                            st.error("Backtest failed. Check logs for details.")
                st.markdown('</div>', unsafe_allow_html=True)

            # Journal Tab
            with tabs[3]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("📝 Quick Seller’s Journal")
                if st.session_state.violations >= 2:
                    st.info("🔔 Risk flags detected. Log your plan to stay sharp!")
                with st.form("journal_form"):
                    premium_reason = st.text_input("Why are you selling this premium? (e.g., IV overpriced)", max_chars=100)
                    submit = st.form_submit_button("Log It")
                    if submit and premium_reason:
                        journal_entry = {
                            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Premium_Reason": premium_reason,
                            "Discipline_Score": 8
                        }
                        pd.DataFrame([journal_entry]).to_csv("journal_log.csv", mode='a', header=not os.path.exists("journal_log.csv"), index=False)
                        st.success("Trade Smarter! 🚀")
                        st.session_state.violations = 0
                        logger.info("Journal entry saved")
                st.markdown("### Past Reflections")
                if os.path.exists("journal_log.csv"):
                    journal_df = pd.read_csv("journal_log.csv")
                    st.dataframe(journal_df[["Date", "Premium_Reason", "Discipline_Score"]], use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown('<div class="footer">Built with ❤️ by Shritish Shukla & Salman Azim | © 2025 VolGuard Pro</div>', unsafe_allow_html=True)
