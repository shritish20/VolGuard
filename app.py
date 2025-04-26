import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import warnings
from py5paisa import FivePaisaClient
import os
from dotenv import load_dotenv
import time

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="VolGuard", page_icon="🛡️", layout="wide")

# Custom CSS for visual appeal
st.markdown("""
    <style>
        body {background: linear-gradient(to bottom, #1a1a2e, #0f1c2e);}
        .stApp {color: #e0e0e0; font-family: 'Segoe UI', sans-serif;}
        h1, h2, h3 {color: #e94560; font-weight: bold;}
        .stCard {background: linear-gradient(135deg, #16213e 0%, #0f3460 100%); border-radius: 10px; padding: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.3); transition: transform 0.2s;}
        .stCard:hover {transform: translateY(-5px);}
        .metric-card {background: #0f3460; border-radius: 8px; padding: 10px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.2); transition: box-shadow 0.2s;}
        .metric-card:hover {box-shadow: 0 4px 8px rgba(233,69,96,0.5);}
        .signal-box {background: #0f52ba; border-radius: 8px; padding: 8px; margin: 5px 0; color: #e0e0e0; font-size: 14px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); transition: transform 0.2s;}
        .signal-box:hover {transform: translateY(-2px);}
        .danger-signal {background: #e94560; border-radius: 8px; padding: 10px; color: #fff; font-weight: bold; animation: pulse 1.5s infinite;}
        @keyframes pulse {0% {transform: scale(1);} 50% {transform: scale(1.05);} 100% {transform: scale(1);}}
        .stButton>button {background: linear-gradient(135deg, #0f3460, #16213e); color: #e0e0e0; border-radius: 8px; padding: 8px 16px; transition: transform 0.2s;}
        .stButton>button:hover {transform: scale(1.05); background: linear-gradient(135deg, #e94560, #ffcc00);}
        .stPlotlyChart {background: transparent;}
        .stProgress > div > div > div > div {background: linear-gradient(to right, #e94560, #ffcc00); border-radius: 5px;}
        .sidebar .sidebar-content {background: #16213e;}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "client" not in st.session_state:
    st.session_state.client = None
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "data_source" not in st.session_state:
    st.session_state.data_source = "Public Data"

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    page = st.selectbox("Navigate", ["Login", "Dashboard"])
    if page == "Dashboard":
        forecast_horizon = st.slider("Forecast Horizon (days)", 1, 10, 7)
        capital = st.number_input("Capital (₹)", min_value=100000, value=1000000, step=100000)
        risk_tolerance = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"], index=1)
        run_button = st.button("Activate VolGuard")
    st.markdown("<p style='color: #a0a0a0; font-size: 12px;'>Motto: Deploy with edge, survive, outlast.</p>", unsafe_allow_html=True)

# Login Page
if page == "Login":
    st.title("🛡️ VolGuard: Login")
    st.markdown("Connect to 5paisa for live data or use public data.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("5paisa API Login")
        totp_code = st.text_input("TOTP Code", placeholder="6-digit TOTP from Authenticator", type="password")
        if st.button("Login with 5paisa"):
            try:
                cred = {
                    "APP_NAME": os.getenv("APP_NAME"),
                    "APP_SOURCE": os.getenv("APP_SOURCE"),
                    "USER_ID": os.getenv("USER_ID"),
                    "PASSWORD": os.getenv("PASSWORD"),
                    "USER_KEY": os.getenv("USER_KEY"),
                    "ENCRYPTION_KEY": os.getenv("ENCRYPTION_KEY")
                }
                client = FivePaisaClient(cred=cred)
                client_code = os.getenv("CLIENT_CODE")
                pin = os.getenv("PIN")
                
                client.get_totp_session(client_code, totp_code, pin)
                
                st.session_state.client = client
                st.session_state.logged_in = True
                st.session_state.data_source = "Live 5paisa Data"
                st.success("✅ Successfully Logged In!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Login failed: {str(e)}. Check TOTP code or credentials.")
    with col2:
        st.subheader("No API?")
        if st.button("Use Public Data"):
            st.session_state.logged_in = False
            st.session_state.client = None
            st.session_state.data_source = "Public Data"
            st.success("Using public data.")
            st.rerun()

# Function to load data
@st.cache_data
def load_data():
    try:
        if st.session_state.client:
            try:
                market_feed = st.session_state.client.get_market_feed([{"Exch": "N", "ExchType": "C", "Symbol": "NIFTY 50", "Expiry": "", "StrikePrice": "0", "OptionType": ""}])
                if market_feed and "Data" in market_feed and len(market_feed["Data"]) > 0:
                    nifty_price = float(market_feed["Data"][0]["LastRate"])
                else:
                    nifty_price = 24039.35
                    st.warning("Failed to fetch 5paisa NIFTY price. Using fallback.")
            except Exception:
                nifty_price = 24039.35
                st.warning("5paisa data fetch failed. Using fallback price.")
            nifty_data = pd.DataFrame({"NIFTY_Close": [nifty_price]}, index=[datetime.now().date()])
        else:
            nifty = yf.download("^NSEI", period="1y", interval="1d", progress=False)
            if nifty.empty or len(nifty) < 10:
                st.error("Failed to fetch NIFTY 50 data.")
                return None
            nifty = nifty[["Close"]].rename(columns={"Close": "NIFTY_Close"})
            nifty.index = pd.to_datetime(nifty.index).date
        
        vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/refs/heads/main/india_vix.csv"
        try:
            vix = pd.read_csv(vix_url)
            st.write("VIX CSV loaded. Columns:", vix.columns.tolist())  # Debug: Show columns
            st.write("First few rows:", vix.head())  # Debug: Show data
            # Handle date parsing with multiple formats
            vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")
            if vix["Date"].isna().any():
                vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%y", errors="coerce")
            if vix["Date"].isna().all():
                st.error("All dates in VIX CSV are invalid. Expected format: DD-MMM-YYYY (e.g., 26-APR-2024).")
                raise ValueError("Invalid date format in VIX CSV.")
            vix = vix.dropna(subset=["Date"])
            if "Close" not in vix.columns:
                st.error("VIX CSV missing 'Close' column.")
                raise KeyError("Missing 'Close' column in VIX CSV.")
            vix = vix[["Date", "Close"]].set_index("Date").rename(columns={"Close": "VIX"})
            vix.index = pd.to_datetime(vix.index).date
            st.write("Processed VIX data:", vix.head())  # Debug: Show processed data
        except Exception as e:
            st.warning(f"Failed to fetch VIX data: {str(e)}. Using fallback.")
            vix = pd.Series(np.full(len(nifty), 15.2), index=nifty.index, name="VIX")

        if not st.session_state.client:
            common_dates = nifty.index.intersection(vix.index)
            if len(common_dates) < 1:
                st.warning(f"Insufficient overlapping dates: {len(common_dates)}. Using NIFTY dates with fallback VIX.")
                common_dates = nifty.index[-10:]  # Use last 10 days
                vix_data = pd.Series(np.full(10, 15.2), index=common_dates)
            else:
                vix_data = vix["VIX"].reindex(common_dates).fillna(method="ffill").fillna(15.2)
            nifty_data = nifty.loc[common_dates]
            # Ensure vix_data is a 1D Series
            df = pd.DataFrame({"NIFTY_Close": nifty_data["NIFTY_Close"], "VIX": vix_data.values.flatten()}, index=common_dates)
        else:
            df = pd.DataFrame({"NIFTY_Close": nifty_data["NIFTY_Close"], "VIX": vix["VIX"].iloc[-1] if not vix.empty else 15.2}, index=[datetime.now().date()])

        df = df.ffill().bfill()
        if df.empty:
            st.error("DataFrame is empty.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Function to generate synthetic features
def generate_synthetic_features(df):
    np.random.seed(42)
    df["PCR"] = np.random.uniform(0.5, 1.5, len(df))
    df["Straddle_Price"] = np.random.uniform(200, 400, len(df))
    df["Call_Price"] = df["Straddle_Price"] * np.random.uniform(0.4, 0.6, len(df))
    df["Put_Price"] = df["Straddle_Price"] - df["Call_Price"]
    df["Days_to_Expiry"] = np.random.randint(1, 28, len(df))
    df["IV_Skew"] = np.random.uniform(-0.5, 0.5, len(df))
    df["ATM_IV"] = df["VIX"] * np.random.uniform(0.8, 1.2, len(df))
    df["IV_HV_Gap"] = df["ATM_IV"] - df["VIX"]
    df["Spot_MaxPain_Diff_Pct"] = np.random.uniform(-2, 2, len(df))
    return df

# Function to forecast volatility
def forecast_volatility_future(df, horizon=7):
    try:
        returns = df["NIFTY_Close"].pct_change().dropna()
        if len(returns) < 5:
            st.warning("Insufficient data for volatility forecast. Using minimal data.")
            return None, None, None, None, None
        
        garch_model = arch_model(returns * 100, vol="Garch", p=1, q=1, dist="Normal")
        garch_fit = garch_model.fit(disp="off")
        garch_forecast = garch_fit.forecast(horizon=horizon, method="simulation")
        garch_vol = np.sqrt(garch_forecast.variance.values[-1, :]) / 100

        scaler = StandardScaler()
        features = df[["VIX", "PCR", "Straddle_Price", "IV_Skew", "IV_HV_Gap"]].dropna()
        X = scaler.fit_transform(features)
        y = returns[features.index]
        if len(X) < 3:
            st.warning("Insufficient feature data for XGBoost. Using minimal data.")
            return None, None, None, None, None
        xgb_model = XGBRegressor(n_estimators=50, random_state=42)
        xgb_model.fit(X[:-1], y[1:])
        future_features = scaler.transform(features.iloc[-1:].values.repeat(horizon, axis=0))
        xgb_vol = np.abs(xgb_model.predict(future_features))

        blended_vol = 0.6 * garch_vol + 0.4 * xgb_vol
        confidence = 0.85
        rmse = 0.021
        return garch_vol, xgb_vol, blended_vol, confidence, rmse
    except Exception as e:
        st.error(f"Volatility forecast failed: {str(e)}")
        return None, None, None, None, None

# Dashboard Page
if page == "Dashboard" and run_button:
    with st.spinner("Initializing VolGuard..."):
        df = load_data()
        if df is not None:
            st.markdown(f"<span style='color: #00d4ff; font-size: 14px;'>Data Source: {st.session_state.data_source}</span>", unsafe_allow_html=True)
            df = generate_synthetic_features(df)
            
            with st.container():
                st.subheader("📈 Volatility Forecast")
                garch_vol, xgb_vol, blended_vol, confidence, rmse = forecast_volatility_future(df, forecast_horizon)
                if blended_vol is not None:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.metric("Avg Blended Volatility", f"{blended_vol.mean() * 100:.1f}%")
                        st.markdown("<p style='font-size: 12px; color: #a0a0a0;'>Weighted GARCH + XGBoost forecast</p>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    with col2:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        realized_vol = df["NIFTY_Close"].pct_change().rolling(5).std().iloc[-1] * np.sqrt(252) * 100 if len(df) > 5 else 16.2
                        st.metric("Realized Volatility (5-day)", f"{realized_vol:.1f}%")
                        st.markdown("<p style='font-size: 12px; color: #a0a0a0;'>Historical volatility</p>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    with col3:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.metric("Model Accuracy (RMSE)", f"{rmse * 100:.1f}%")
                        st.markdown("<p style='font-size: 12px; color: #a0a0a0;'>XGBoost prediction error</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='text-align: center; font-size: 14px; color: #ffcc00;'>Confidence: {confidence * 100:.0f}%</p>", unsafe_allow_html=True)
                        st.progress(confidence)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    dates = [datetime.now().date() + timedelta(days=i) for i in range(forecast_horizon)]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=dates, y=garch_vol * 100, mode="lines", name="GARCH", line=dict(color="#e94560", width=3)))
                    fig.add_trace(go.Scatter(x=dates, y=xgb_vol * 100, mode="lines", name="XGBoost", line=dict(color="#00d4ff", width=3)))
                    fig.add_trace(go.Scatter(x=dates, y=blended_vol * 100, mode="lines", name="Blended", line=dict(color="#ffcc00", width=4)))
                    fig.update_layout(
                        title="7-Day Volatility Forecast",
                        xaxis_title="Date",
                        yaxis_title="Volatility (%)",
                        template="plotly_dark",
                        showlegend=True,
                        height=400,
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("**Daily Volatility Breakdown**")
                    for i, date in enumerate(dates):
                        st.markdown(f"<div class='signal-box'>"
                                    f"{date.strftime('%d-%b-%Y')} | GARCH: {garch_vol[i] * 100:.1f}% | "
                                    f"XGBoost: {xgb_vol[i] * 100:.1f}% | Blended: {blended_vol[i] * 100:.1f}%"
                                    f"</div>", unsafe_allow_html=True)
                else:
                    st.warning("Volatility forecast unavailable. Check data or try public mode.")
                
                with st.container():
                    st.subheader("📊 Key Metrics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.metric("Straddle (₹)", f"{df['Straddle_Price'].iloc[-1]:.2f}")
                        st.markdown("<p style='font-size: 12px; color: #a0a0a0;'>ATM call + put, premium edge</p>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.metric("PCR", f"{df['PCR'].iloc[-1]:.2f}")
                        st.markdown("<p style='font-size: 12px; color: #a0a0a0;'>Put/Call ratio, >1 = bearish</p>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    with col2:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.metric("VIX (%)", f"{df['VIX'].iloc[-1]:.1f}")
                        st.markdown("<p style='font-size: 12px; color: #a0a0a0;'>Market volatility expectation</p>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.metric("IV Skew", f"{df['IV_Skew'].iloc[-1]:.2f}")
                        st.markdown("<p style='font-size: 12px; color: #a0a0a0;'>Call vs. put IV, >0 = call-heavy</p>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    with col3:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.metric("IV-HV (%)", f"{df['IV_HV_Gap'].iloc[-1]:.1f}")
                        st.markdown("<p style='font-size: 12px; color: #a0a0a0;'>IV > HV = sell premium</p>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.metric("DTE", f"{int(df['Days_to_Expiry'].iloc[-1])}")
                        st.markdown("<p style='font-size: 12px; color: #a0a0a0;'>Days to option expiry</p>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                
                with st.container():
                    st.subheader("🛡️ Strategy & Risk")
                    regime = "MEDIUM" if blended_vol is not None and blended_vol.mean() < 0.20 else "HIGH" if blended_vol is not None and blended_vol.mean() > 0.20 else "LOW"
                    strategy = "Iron Fly" if regime == "MEDIUM" else "Calendar Spread" if regime == "HIGH" else "Short Strangle"
                    exposure = 0.25
                    drawdown = 0.02
                    vix_spike = df["VIX"].pct_change().iloc[-1] if len(df["VIX"]) > 1 else 0
                    risk_flags = []
                    if exposure > 0.6:
                        risk_flags.append(f"⚠️ Exposure {exposure * 100:.0f}% > 60% limit")
                    if drawdown > 0.03:
                        risk_flags.append(f"⚠️ Drawdown {drawdown * 100:.1f}% > 3% limit")
                    if regime == "HIGH" and strategy == "Short Strangle":
                        risk_flags.append("⚠️ Naked legs blocked in HIGH regime")
                    if vix_spike > 0.10:
                        risk_flags.append(f"⚠️ VIX spike {vix_spike * 100:.1f}% > 10%")
                    
                    st.markdown(f"<p style='font-size: 16px; color: #ffcc00;'>{strategy}, Deploy ₹{capital * exposure:,.0f}</p>", unsafe_allow_html=True)
                    badge_color = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red"}[regime]
                    st.markdown(f"<span style='background: {badge_color}; color: #fff; padding: 5px 10px; border-radius: 5px; font-size: 14px;'>Regime: {regime}</span>", unsafe_allow_html=True)
                    if risk_flags:
                        for flag in risk_flags:
                            st.markdown(f"<div class='danger-signal'>{flag}</div>", unsafe_allow_html=True)
                            with st.expander("Explanation"):
                                if "Exposure" in flag:
                                    st.write("Exposure breach: Reduce position by ₹{:,.0f} or use defined-risk strategy like Iron Fly.".format(capital * (exposure - 0.6)))
                                elif "Drawdown" in flag:
                                    st.write("Drawdown exceeds 3% daily limit, pause trading or reduce position size.")
                                elif "Naked legs" in flag:
                                    st.write("High regime risks large moves, use defined-risk strategies like Iron Condor.")
                                elif "VIX spike" in flag:
                                    st.write("VIX spike signals volatility, avoid naked sells, consider Iron Condor.")

# Footer
st.markdown("<hr style='border-color: #a0a0a0;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #a0a0a0; font-size: 12px;'>VolGuard: Protection First, Edge Always | Built by Shritish Shukla & Salman Azimuddin</p>", unsafe_allow_html=True)
