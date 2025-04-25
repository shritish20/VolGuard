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
        st.write(f"Nifty50.csv columns: {list(nifty.columns)}")  # Debug: Show column names
        if "Date" not in nifty.columns or "Close" not in nifty.columns:
            st.error("Nifty50.csv is missing required columns: 'Date' or 'Close'. Check the file.")
            return None
        nifty = nifty[["Date", "Close"]]
        nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%y", errors="coerce")
        nifty = nifty.dropna(subset=["Date"])
        if nifty.empty:
            st.error("NIFTY data is empty or invalid after parsing dates.")
            return None
        if not pd.api.types.is_numeric_dtype(nifty["Close"]):
            st.error("NIFTY 'Close' column contains non-numeric values.")
            return None
        nifty = nifty.set_index("Date")
        nifty = nifty[~nifty.index.duplicated(keep='first')]  # Remove duplicate dates

        # Load VIX data
        vix = pd.read_csv(vix_url)
        vix.columns = vix.columns.str.strip()
        st.write(f"india_vix.csv columns: {list(vix.columns)}")  # Debug: Show column names
        if "Date" not in vix.columns or "Close" not in vix.columns:
            st.error("india_vix.csv is missing required columns: 'Date' or 'Close'. Check the file.")
            return None
        vix = vix[["Date", "Close"]]
        vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%y", errors="coerce")
        vix = vix.dropna(subset=["Date"])
        if vix.empty:
            st.error("VIX data is empty or invalid after parsing dates.")
            return None
        if not pd.api.types.is_numeric_dtype(vix["Close"]):
            st.error("VIX 'Close' column contains non-numeric values.")
            return None
        vix = vix.set_index("Date").rename(columns={"Close": "VIX"})
        vix = vix[~vix.index.duplicated(keep='first')]  # Remove duplicate dates

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

# Main execution
if run_button:
    with st.spinner("Loading data from GitHub..."):
        df = load_data()
        if df is not None:
            st.success("✅ Data loaded successfully!")
            st.write("Sample Data:")
            st.dataframe(df.head())
            st.write("Data Info:")
            st.write(df.info())  # Additional debug: Show data types and non-null counts
            st.write("VIX Statistics:")
            st.write(f"Mean: {df['VIX'].mean():.2f}, Min: {df['VIX'].min():.2f}, Max: {df['VIX'].max():.2f}")
        else:
            st.error("Failed to load data. Please check the GitHub URLs or CSV file formats.")
