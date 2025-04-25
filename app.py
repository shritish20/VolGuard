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
        st.write(f"Nifty50.csv columns: {list(nifty.columns)}")
        if "Date" not in nifty.columns or "Close" not in nifty.columns:
            st.error("Nifty50.csv is missing required columns: 'Date' or 'Close'. Check the file.")
            return None

        # Debug: Show raw date data
        st.write("Raw NIFTY Date Sample (first 5 rows):")
        st.write(nifty["Date"].head())

        # Try different date formats
        date_formats = ["%d-%b-%y", "%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"]
        nifty_dates = None
        for fmt in date_formats:
            try:
                nifty_dates = pd.to_datetime(nifty["Date"], format=fmt, errors="coerce")
                if nifty_dates.notna().sum() > 0:  # If at least one date parsed successfully
                    st.write(f"Successfully parsed NIFTY dates with format: {fmt}")
                    break
            except Exception as e:
                continue

        if nifty_dates is None or nifty_dates.notna().sum() == 0:
            st.error("Failed to parse NIFTY dates with any known format. Please check the date format in Nifty50.csv.")
            return None

        nifty["Date"] = nifty_dates
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
        st.write(f"india_vix.csv columns: {list(vix.columns)}")
        if "Date" not in vix.columns or "Close" not in vix.columns:
            st.error("india_vix.csv is missing required columns: 'Date' or 'Close'. Check the file.")
            return None

        # Debug: Show raw date data
        st.write("Raw VIX Date Sample (first 5 rows):")
        st.write(vix["Date"].head())

        # Try different date formats
        vix_dates = None
        for fmt in date_formats:
            try:
                vix_dates = pd.to_datetime(vix["Date"], format=fmt, errors="coerce")
                if vix_dates.notna().sum() > 0:
                    st.write(f"Successfully parsed VIX dates with format: {fmt}")
                    break
            except Exception as e:
                continue

        if vix_dates is None or vix_dates.notna().sum() == 0:
            st.error("Failed to parse VIX dates with any known format. Please check the date format in india_vix.csv.")
            return None

        vix["Date"] = vix_dates
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
            st.write(df.info())
            st.write("VIX Statistics:")
            st.write(f"Mean: {df['VIX'].mean():.2f}, Min: {df['VIX'].min():.2f}, Max: {df['VIX'].max():.2f}")
        else:
            st.error("Failed to load data. Please check the GitHub URLs or CSV file formats.")
