import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import warnings
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Page config
st.set_page_config(page_title="VolGuard", page_icon="🛡️", layout="wide")

# Custom CSS for improved visibility
st.markdown("""
    <style>
        .main {background-color: #f0f2f5; color: #2b3e50;}
        .stButton>button {background-color: #2b3e50; color: white; border-radius: 5px; padding: 8px 16px;}
        .stMetric {background-color: #ffffff; border-radius: 8px; padding: 12px;}
        h1 {color: #2b3e50; font-family: 'Arial', sans-serif;}
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🛡️ VolGuard: AI-Powered Trading Copilot")
st.markdown("Your disciplined partner for volatility-driven options trading.")
st.write(f"Date: {datetime.now().strftime('%d-%b-%Y')}")

# Sidebar
with st.sidebar:
    st.header("⚙️ Parameters")
    forecast_horizon = st.slider("Forecast Horizon (days)", 1, 10, 7)
    capital = st.number_input("Capital (₹)", min_value=100000, value=1000000, step=100000)
    risk_tolerance = st.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"], index=1)
    
    st.header("Actions")
    run_button = st.button("Run VolGuard")
    st.markdown("---")
    st.info("Built by Shritish Shukla & AI Co-Founder | Protection First, Edge Always")

# Function to load data from GitHub
@st.cache_data
def load_data():
    try:
        # URLs to your GitHub CSV files (raw content)
        nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/Nifty50.csv"
        vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"

        # Load NIFTY data
        nifty = pd.read_csv(nifty_url)
        nifty.columns = nifty.columns.str.strip()
        if "Date" not in nifty.columns or "Close" not in nifty.columns:
            st.error("Nifty50.csv is missing required columns: 'Date' or 'Close'. Check the file.")
            return None

        nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
        nifty = nifty.dropna(subset=["Date"])
        if nifty.empty:
            st.error("NIFTY data is empty or invalid after parsing dates.")
            return None
        if not pd.api.types.is_numeric_dtype(nifty["Close"]):
            st.error("NIFTY 'Close' column contains non-numeric values.")
            return None
        nifty = nifty[["Date", "Close"]]
        nifty = nifty.set_index("Date")
        nifty = nifty[~nifty.index.duplicated(keep='first')]

        # Load VIX data
        vix = pd.read_csv(vix_url)
        vix.columns = vix.columns.str.strip()
        if "Date" not in vix.columns or "Close" not in vix.columns:
            st.error("india_vix.csv is missing required columns: 'Date' or 'Close'. Check the file.")
            return None

        vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")
        vix = vix.dropna(subset=["Date"])
        if vix.empty:
            st.error("VIX data is empty or invalid after parsing dates.")
            return None
        if not pd.api.types.is_numeric_dtype(vix["Close"]):
            st.error("VIX 'Close' column contains non-numeric values.")
            return None
        vix = vix[["Date", "Close"]]
        vix = vix.set_index("Date").rename(columns={"Close": "VIX"})
        vix = vix[~vix.index.duplicated(keep='first')]

        # Align dates between NIFTY and VIX
        common_dates = nifty.index.intersection(vix.index)
        if len(common_dates) < 10:
            st.error(f"Insufficient overlapping dates between NIFTY and VIX data. Found only {len(common_dates)} common dates.")
            return None
        nifty = nifty.loc[common_dates]
        vix = vix.loc[common_dates]

        # Create DataFrame
        df = pd.DataFrame({
            "NIFTY_Close": nifty["Close"],
            "VIX": vix["VIX"]
        }, index=common_dates)

        # Remove timestamp from index (keep only date)
        df.index = df.index.date

        # Validate data
        if df["NIFTY_Close"].isna().sum() > 0 or df["VIX"].isna().sum() > 0:
            st.warning("Found NaN values in data. Filling with forward-fill and back-fill.")
            df = df.ffill().bfill()
        if df.empty:
            st.error("DataFrame is empty after processing.")
            return None

        return df

    except Exception as e:
        st.error(f"Error loading data from GitHub: {str(e)}")
        return None

# Function to generate synthetic options features
def generate_synthetic_features(df):
    n_days = len(df)
    np.random.seed(42)  # For reproducibility

    # Constants
    risk_free_rate = 0.06
    strike_step = 100

    # Black-Scholes Pricing
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

    # Synthetic Features
    # ATM_IV: VIX-based with dynamic noise and event spikes
    event_spike = np.where((pd.to_datetime(df.index).month % 3 == 0) & (pd.to_datetime(df.index).day < 5), 1.2, 1.0)
    df["ATM_IV"] = df["VIX"] * (1 + np.random.normal(0, 0.1, n_days)) * event_spike
    df["ATM_IV"] = np.clip(df["ATM_IV"], 5, 50)

    # IVP: Percentile rank with dynamic window
    def dynamic_ivp(x):
        if len(x) >= 5:
            return (np.sum(x.iloc[:-1] <= x.iloc[-1]) / (len(x) - 1)) * 100
        return 50.0  # Default for very early data

    df["IVP"] = df["ATM_IV"].rolling(252, min_periods=5).apply(dynamic_ivp)
    df["IVP"] = df["IVP"].interpolate().fillna(50.0)

    # PCR: Enhanced with stronger momentum signal
    market_trend = df["NIFTY_Close"].pct_change().rolling(5).mean().fillna(0)
    df["PCR"] = np.clip(1.0 + np.random.normal(0, 0.1, n_days) + market_trend * -10, 0.7, 2.0)

    # VIX_Change_Pct
    df["VIX_Change_Pct"] = df["VIX"].pct_change().fillna(0) * 100

    # Spot_MaxPain_Diff_Pct: Realistic max pain differences
    df["Spot_MaxPain_Diff_Pct"] = np.abs(np.random.lognormal(-2, 0.5, n_days))
    df["Spot_MaxPain_Diff_Pct"] = np.clip(df["Spot_MaxPain_Diff_Pct"], 0.1, 1.0)

    # Days_to_Expiry: Realistic expiry cycles
    df["Days_to_Expiry"] = np.random.choice([1, 3, 7, 14, 21, 28], n_days)

    # Event_Flag: Earnings or expiry events
    df["Event_Flag"] = np.where((pd.to_datetime(df.index).month % 3 == 0) & (pd.to_datetime(df.index).day < 5) | (df["Days_to_Expiry"] <= 3), 1, 0)

    # FII Positions: Trend-based with reversals
    fii_trend = np.random.normal(0, 10000, n_days)
    fii_trend[::30] *= -1  # Periodic reversals
    df["FII_Index_Fut_Pos"] = np.cumsum(fii_trend).astype(int)
    df["FII_Option_Pos"] = np.cumsum(np.random.normal(0, 5000, n_days)).astype(int)

    # IV_Skew: Enhanced market correlation
    df["IV_Skew"] = np.clip(np.random.normal(0, 0.8, n_days) + (df["VIX"] / 15 - 1) * 3, -3, 3)

    # Realized Volatility: Handle early periods and cap outliers
    df["Realized_Vol"] = df["NIFTY_Close"].pct_change().rolling(5, min_periods=5).std() * np.sqrt(252) * 100
    df["Realized_Vol"] = df["Realized_Vol"].fillna(df["VIX"])
    df["Realized_Vol"] = np.clip(df["Realized_Vol"], 0, 50)

    # Advance/Decline Ratio: Simulate market breadth
    df["Advance_Decline_Ratio"] = np.clip(1.0 + np.random.normal(0, 0.2, n_days) + market_trend * 10, 0.5, 2.0)

    # Capital Pressure Index: Wider range
    df["Capital_Pressure_Index"] = (df["FII_Index_Fut_Pos"] / 3e4 + df["FII_Option_Pos"] / 1e4 + df["PCR"]) / 3
    df["Capital_Pressure_Index"] = np.clip(df["Capital_Pressure_Index"], -2, 2)

    # Gamma Bias: Proxy with DTE and IV_Skew
    df["Gamma_Bias"] = np.clip(df["IV_Skew"] * (30 - df["Days_to_Expiry"]) / 30, -2, 2)

    # Capital and PnL
    df["Total_Capital"] = capital
    df["PnL_Day"] = np.random.normal(0, 5000, n_days) * (1 - df["Event_Flag"] * 0.5)

    # Straddle Pricing
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

    # Final Sanity Checks
    missing = df.isna().sum().sum()
    if missing > 0:
        st.warning(f"Filling {missing} missing values...")
        df = df.interpolate().fillna(method='bfill')

    return df

# Main execution
if run_button:
    with st.spinner("Loading data from GitHub..."):
        df = load_data()
        if df is not None:
            st.success("✅ Data loaded successfully!")
            st.write("Sample Data:")
            st.dataframe(df.head())
            st.write("Data Info:")
            st.write(df.info())
            st.write("VIX Statistics:")
            st.write(f"Mean: {df['VIX'].mean():.2f}, Min: {df['VIX'].min():.2f}, Max: {df['VIX'].max():.2f}")

            # Generate synthetic features
            with st.spinner("Generating synthetic options features..."):
                df = generate_synthetic_features(df)
                st.write("Synthetic Features Generated:")
                feature_cols = [
                    "NIFTY_Close", "VIX", "ATM_IV", "IVP", "PCR", "VIX_Change_Pct",
                    "Spot_MaxPain_Diff_Pct", "Straddle_Price", "IV_Skew", "Realized_Vol",
                    "Days_to_Expiry", "Event_Flag", "FII_Index_Fut_Pos", "FII_Option_Pos",
                    "Advance_Decline_Ratio", "Capital_Pressure_Index", "Gamma_Bias"
                ]
                st.dataframe(df[feature_cols].tail())
                st.write("Feature Statistics:")
                st.dataframe(df[feature_cols].describe())
