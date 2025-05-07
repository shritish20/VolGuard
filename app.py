import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import re # Import regex for date parsing
from datetime import datetime, timedelta
import logging
from py5paisa import FivePaisaClient
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(page_title="VolGuard Pro", page_icon="🛡️", layout="wide") # Corrected emoji

# Custom CSS
st.markdown("""
    <style>
        .main { background: linear-gradient(135deg, #1a1a2e, #0f1c2e); color: #e5e5e5; font-family: 'Inter', sans-serif; }
        .stTabs [data-baseweb="tab-list"] { background: #16213e; border-radius: 10px; padding: 10px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3); }
        .stTabs [data-baseweb="tab"] { color: #a0a0a0; font-weight: 500; padding: 10px 20px; border-radius: 8px; }
        .stTabs [data-baseweb="tab"][aria-selected="true"] { background: #e94560; color: white; font-weight: 700; }
        .stTabs [data-baseweb="tab"]:hover { background: #2a2a4a; color: white; }
        .sidebar .stButton>button { width: 100%; background: #0f3460; color: white; border-radius: 10px; padding: 12px; margin: 5px 0; }
        .sidebar .stButton>button:hover { transform: scale(1.05); background: #e94560; }
        .card { background: linear-gradient(145deg, rgba(22, 33, 62, 0.85), rgba(10, 25, 47, 0.9)); border-radius: 15px; padding: 20px; margin: 15px 0; box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4); }
        .card:hover { transform: translateY(-5px); }
        .strategy-carousel { display: flex; overflow-x: auto; gap: 20px; padding: 10px; }
        .strategy-card { flex: 0 0 auto; width: 300px; background: #16213e; border-radius: 15px; padding: 20px; }
        .strategy-card:hover { transform: scale(1.05); }
        .stMetric { background: rgba(15, 52, 96, 0.7); border-radius: 15px; padding: 15px; text-align: center; }
        .gauge { width: 100px; height: 100px; border-radius: 50%; background: conic-gradient(#e94560 0% 50%, #00d4ff 50% 100%); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 18px; }
        .regime-badge { padding: 8px 15px; border-radius: 20px; font-weight: bold; font-size: 14px; text-transform: uppercase; }
        .regime-low { background: #28a745; color: white; }
        .regime-medium { background: #ffc107; color: black; }
        .regime-high { background: #dc3545; color: white; }
        .regime-event { background: #ff6f61; color: white; }
        .alert-banner { background: #dc3545; color: white; padding: 15px; border-radius: 10px; position: sticky; top: 0; z-index: 100; }
        .stButton>button { background: #e94560; color: white; border-radius: 10px; padding: 12px 25px; font-size: 16px; }
        .stButton>button:hover { transform: scale(1.05); background: #ffcc00; }
        .footer { text-align: center; padding: 20px; color: #a0a0a0; font-size: 14px; border-top: 1px solid rgba(255, 255, 255, 0.1); margin-top: 30px; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "backtest_run" not in st.session_state:
    st.session_state.backtest_run = False
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None
if "violations" not in st.session_state:
    st.session_state.violations = 0
if "journal_complete" not in st.session_state:
    st.session_state.journal_complete = False
if "trades" not in st.session_state:
    st.session_state.trades = []
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "client" not in st.session_state:
    st.session_state.client = None
if "real_time_market_data" not in st.session_state:
     st.session_state.real_time_market_data = None
if "api_portfolio_data" not in st.session_state:
     st.session_state.api_portfolio_data = {}
if "prepared_orders" not in st.session_state:
     st.session_state.prepared_orders = None


# --- Helper function to parse the 5paisa date string format (from Notebook) ---
def parse_5paisa_date_string(date_string):
    """
    Parses the /Date(1234567890000+0000)/ format string to a numerical timestamp (in milliseconds).
    Returns the numerical timestamp or None if parsing fails.
    """
    if not isinstance(date_string, str):
        return None
    match = re.search(r'/Date\((\d+)[+-]\d+\)/', date_string)
    if match:
        return int(match.group(1))
    return None

# --- Helper function to format timestamp to readable date string (from Notebook) ---
def format_timestamp_to_date_str(timestamp_ms):
     """
     Converts a timestamp in milliseconds to a readableYYYY-MM-DD string.
     """
     if timestamp_ms is None:
          return "N/A"
     try:
          # Convert milliseconds to seconds
          timestamp_s = timestamp_ms / 1000
          # Convert timestamp to datetime object
          dt_object = datetime.fromtimestamp(timestamp_s)
          return dt_object.strftime("%Y-%m-%d")
     except Exception:
          return "N/A"

# --- Helper function for Max Pain Calculation (from Notebook) ---
def _calculate_max_pain(df: pd.DataFrame, nifty_spot: float):
    """
    Helper function to calculate the max pain strike.
    """
    try:
        if df.empty or "StrikeRate" not in df.columns or "CPType" not in df.columns or "OpenInterest" not in df.columns:
            logger.warning("Option chain data is incomplete or empty for max pain calculation.")
            return None, None

        try:
            df["StrikeRate"] = pd.to_numeric(df["StrikeRate"], errors='coerce')
            df["OpenInterest"] = pd.to_numeric(df["OpenInterest"], errors='coerce').fillna(0)
            df = df.dropna(subset=["StrikeRate", "OpenInterest"]).copy()
        except Exception as e:
            logger.error(f"Error converting columns for max pain: {e}")
            return None, None

        calls = df[df["CPType"] == "CE"].set_index("StrikeRate")["OpenInterest"]
        puts = df[df["CPType"] == "PE"].set_index("StrikeRate")["OpenInterest"]
        strikes = df["StrikeRate"].unique()
        strikes.sort()

        pain = []
        for K in strikes:
            total_loss = 0
            for s in strikes:
                 if s in calls:
                      total_loss += max(0, K - s) * calls.get(s, 0)
                 if s in puts:
                      total_loss += max(0, s - K) * puts.get(s, 0)
            pain.append((K, total_loss))

        if not pain:
             logger.warning("No valid strikes to calculate max pain.")
             return None, None

        max_pain_strike = min(pain, key=lambda x: x[1])[0]
        max_pain_diff_pct = abs(nifty_spot - max_pain_strike) / nifty_spot * 100 if nifty_spot != 0 else 0

        logger.debug(f"Max Pain calculated: Strike={max_pain_strike}, Diff%={max_pain_diff_pct:.2f}")
        return max_pain_strike, max_pain_diff_pct
    except Exception as e:
        logger.error(f"Error calculating max pain: {str(e)}")
        return None, None


# 5paisa Client Initialization
def initialize_5paisa_client(totp_code):
    try:
        logger.info("Initializing 5paisa client")
        # Using st.secrets assumes you have a secrets.toml file with these keys
        # Example:
        # [fivepaisa]
        # APP_NAME = "YOUR_APP_NAME"
        # APP_SOURCE = "YOUR_APP_SOURCE"
        # USER_ID = "YOUR_USER_ID"
        # PASSWORD = "YOUR_PASSWORD"
        # USER_KEY = "YOUR_USER_KEY"
        # ENCRYPTION_KEY = "YOUR_ENCRYPTION_KEY"
        # CLIENT_CODE = "YOUR_CLIENT_CODE"
        # PIN = "YOUR_PIN"
        cred = {
            "APP_NAME": st.secrets["fivepaisa"]["APP_NAME"],
            "APP_SOURCE": st.secrets["fivepaisa"]["APP_SOURCE"],
            "USER_ID": st.secrets["fivepaisa"]["USER_ID"],
            "PASSWORD": st.secrets["fivepaisa"]["PASSWORD"],
            "USER_KEY": st.secrets["fivepaisa"]["USER_KEY"],
            "ENCRYPTION_KEY": st.secrets["fivepaisa"]["ENCRYPTION_KEY"]
        }
        client = FivePaisaClient(cred=cred)
        logger.info("Attempting TOTP session...")
        client.get_totp_session(
            st.secrets["fivepaisa"]["CLIENT_CODE"],
            totp_code,
            st.secrets["fivepaisa"]["PIN"]
        )
        if client.get_access_token():
            logger.info("5paisa client initialized and session obtained successfully")
            return client
        else:
            logger.error("Failed to get access token after TOTP session attempt.")
            return None
    except Exception as e:
        logger.error(f"Error initializing 5paisa client or getting session: {str(e)}")
        st.error(f"Login failed: {str(e)}. Check credentials and TOTP.") # Provide user feedback
        return None

# Data Fetching Functions aligned with Notebook
def fetch_real_time_market_data(client):
    """
    Fetches real-time NIFTY 50, India VIX, and Option Chain data from 5paisa API,
    aligned with the working notebook logic.
    """
    if client is None or not client.get_access_token():
        logger.warning("5paisa client not available or not logged in.")
        return None

    logger.info("Fetching real-time market data from 5paisa API (aligned with notebook)")
    nifty_spot = None
    vix = None
    atm_strike = None
    straddle_price = 0
    pcr = 0
    max_pain_strike = None
    max_pain_diff_pct = None
    expiry_date_str = None
    df_option_chain = pd.DataFrame()
    expiry_timestamp = None

    try:
        # 1. Fetch NIFTY 50 (Notebook logic)
        nifty_req = [{
            "Exch": "N",
            "ExchType": "C",
            "ScripCode": 999920000,
            "Symbol": "NIFTY",
            "Expiry": "",
            "StrikePrice": "0",
            "OptionType": ""
        }]
        logger.debug(f"Fetching NIFTY market feed for: {nifty_req}")
        nifty_market_feed = client.fetch_market_feed(nifty_req)

        if not nifty_market_feed or not isinstance(nifty_market_feed, dict) or "Data" not in nifty_market_feed or not isinstance(nifty_market_feed["Data"], list) or not nifty_market_feed["Data"]:
            logger.error(f"Failed to fetch NIFTY 50 market feed or unexpected format. Response: {nifty_market_feed}")
        else:
             nifty_data = nifty_market_feed["Data"][0]
             nifty_spot = nifty_data.get("LastRate", nifty_data.get("LastTradedPrice", 0))
             if not nifty_spot:
                 logger.warning("NIFTY price not found in market feed data after parsing.")
             else:
                 logger.info(f"Fetched NIFTY Spot: {nifty_spot}")

        # 2. Fetch India VIX (Notebook logic)
        vix_req = [{
            "Exch": "N",
            "ExchType": "C",
            "ScripCode": 999920005,
            "Symbol": "INDIAVIX",
            "Expiry": "",
            "StrikePrice": "0",
            "OptionType": ""
        }]
        logger.debug(f"Fetching VIX market feed for: {vix_req}")
        vix_market_feed = client.fetch_market_feed(vix_req)

        if not vix_market_feed or not isinstance(vix_market_feed, dict) or "Data" not in vix_market_feed or not isinstance(vix_market_feed["Data"], list) or not vix_market_feed["Data"]:
            logger.warning(f"Failed to fetch India VIX market feed or unexpected format. Response: {vix_market_feed}")
        else:
            vix_data = vix_market_feed["Data"][0]
            vix = vix_data.get("LTP", vix_data.get("LastRate", vix_data.get("LastTradedPrice", 0)))
            if not vix:
                 logger.warning("VIX price not found in market feed data after trying LTP/LastRate.")
                 vix = None
            else:
                 logger.info(f"Fetched VIX: {vix}")

        # 3. Fetch NIFTY expiries (Notebook logic: checks "Expiry" key and parses date string)
        logger.debug("Fetching NIFTY expiries")
        expiries = client.get_expiry("N", "NIFTY")
        logger.debug(f"Expiries response type: {type(expiries)}, value: {expiries}")

        if not expiries or not isinstance(expiries, dict) or "Expiry" not in expiries or not isinstance(expiries["Expiry"], list) or not expiries["Expiry"]:
            logger.error(f"Failed to fetch NIFTY expiries or unexpected format. Response: {expiries}")
        else:
            first_expiry = expiries["Expiry"][0] # Access using the correct key "Expiry" as per notebook
            expiry_date_string_from_api = first_expiry.get("ExpiryDate")

            if not expiry_date_string_from_api:
                 logger.error("Expiry data missing ExpiryDate in the first expiry item.")
            else:
                 expiry_timestamp = parse_5paisa_date_string(expiry_date_string_from_api) # Use notebook helper

                 if expiry_timestamp is not None:
                      expiry_date_str = format_timestamp_to_date_str(expiry_timestamp) # Use notebook helper
                      logger.info(f"Fetched first expiry: {expiry_date_str} (Timestamp: {expiry_timestamp})")
                 else:
                      logger.error(f"Could not parse timestamp from ExpiryDate string: {expiry_date_string_from_api}")

        # 4. Fetch Option Chain (using the parsed timestamp)
        if expiry_timestamp is not None:
            logger.debug(f"Fetching Option Chain for expiry timestamp: {expiry_timestamp}")
            option_chain = client.get_option_chain("N", "NIFTY", expiry_timestamp)
            logger.debug(f"Option chain response type: {type(option_chain)}, value: {option_chain}")

            if not option_chain or not isinstance(option_chain, dict) or "Options" not in option_chain or not isinstance(option_chain["Options"], list) or not option_chain["Options"]:
                logger.error(f"Failed to fetch NIFTY option chain or unexpected format. Response: {option_chain}")
            else:
                 df_option_chain = pd.DataFrame(option_chain["Options"])
                 logger.debug(f"Option chain DataFrame created with shape: {df_option_chain.shape}")

                 required_cols = ["StrikeRate", "CPType", "LastRate", "OpenInterest", "ScripCode"]
                 if not all(col in df_option_chain.columns for col in required_cols):
                     missing = [col for col in required_cols if col not in df_option_chain.columns]
                     logger.warning(f"Required columns missing in option chain data DataFrame: {missing}")

                 df_option_chain["StrikeRate"] = pd.to_numeric(df_option_chain["StrikeRate"], errors='coerce')
                 df_option_chain["OpenInterest"] = pd.to_numeric(df_option_chain["OpenInterest"], errors='coerce').fillna(0)
                 df_option_chain = df_option_chain.dropna(subset=["StrikeRate", "OpenInterest"]).copy()

                 if df_option_chain.empty:
                     logger.warning("Option chain DataFrame is empty after cleaning missing strikes/OI.")

        else:
            logger.error("Cannot fetch Option Chain: Valid expiry timestamp was not obtained.")


        # 5. Calculate ATM, Straddle, PCR, Max Pain (only if Nifty and Option Chain are available)
        if nifty_spot is not None and nifty_spot != 0 and not df_option_chain.empty:
             logger.debug(f"Calculating ATM, Straddle, PCR, Max Pain for NIFTY spot: {nifty_spot}")
             if pd.api.types.is_numeric_dtype(df_option_chain["StrikeRate"]):
                 atm_strike = df_option_chain["StrikeRate"].iloc[(df_option_chain["StrikeRate"] - nifty_spot).abs().argmin()]
                 atm_data = df_option_chain[df_option_chain["StrikeRate"] == atm_strike]

                 if 'LastRate' in atm_data.columns and pd.api.types.is_numeric_dtype(atm_data["LastRate"]):
                     atm_call_data = atm_data[atm_data["CPType"] == "CE"]
                     atm_call = atm_call_data["LastRate"].iloc[0] if not atm_call_data.empty else 0
                     atm_put_data = atm_data[atm_data["CPType"] == "PE"]
                     atm_put = atm_put_data["LastRate"].iloc[0] if not atm_put_data.empty else 0
                     straddle_price = (atm_call + atm_put) if atm_call is not None and atm_put is not None else 0
                 else:
                      logger.warning("LastRate column missing or not numeric for straddle calculation.")

                 if 'OpenInterest' in df_option_chain.columns and pd.api.types.is_numeric_dtype(df_option_chain["OpenInterest"]):
                     calls_oi_sum = df_option_chain[df_option_chain["CPType"] == "CE"]["OpenInterest"].sum()
                     puts_oi_sum = df_option_chain[df_option_chain["CPType"] == "PE"]["OpenInterest"].sum()
                     pcr = puts_oi_sum / calls_oi_sum if calls_oi_sum != 0 else float("inf")
                 else:
                     logger.warning("OpenInterest column missing or not numeric for PCR calculation.")

                 max_pain_strike, max_pain_diff_pct = _calculate_max_pain(df_option_chain, nifty_spot)
                 logger.debug(f"Calculated ATM Strike: {atm_strike}, Straddle: {straddle_price}, PCR: {pcr}, Max Pain: {max_pain_strike}")
             else:
                  logger.warning("StrikeRate column is not numeric, cannot calculate ATM strike.")

        elif nifty_spot is None or nifty_spot == 0:
             logger.warning("NIFTY spot price not available for calculating derivatives metrics.")
        elif df_option_chain.empty:
             logger.warning("Option chain is empty, cannot calculate derivatives metrics.")

        logger.info("Real-time market data fetching and processing function completed.")

        return {
            "nifty_spot": nifty_spot,
            "vix": vix,
            "atm_strike": atm_strike,
            "straddle_price": straddle_price,
            "pcr": pcr,
            "max_pain_strike": max_pain_strike,
            "max_pain_diff_pct": max_pain_diff_pct,
            "expiry": expiry_date_str,
            "option_chain": df_option_chain,
            "source": "5paisa API (LIVE)" # Tag the source
        }

    except Exception as e:
        logger.error(f"An unexpected error occurred during real-time data fetch: {str(e)}", exc_info=True)
        st.error(f"Error fetching real-time market data: {str(e)}") # User feedback
        return None


def fetch_all_api_portfolio_data(client):
    """
    Fetches comprehensive portfolio and account data from 5paisa API, aligned with notebook.
    """
    if client is None or not client.get_access_token():
        logger.warning("5paisa client not available or not logged in for portfolio data.")
        return {}

    logger.info("Fetching all API portfolio/account data (aligned with notebook)")
    portfolio_data = {}

    try:
        portfolio_data["holdings"] = client.holdings()
        portfolio_data["margin"] = client.margin()
        portfolio_data["positions"] = client.positions()
        portfolio_data["order_book"] = client.order_book()
        portfolio_data["trade_book"] = client.get_tradebook()
        portfolio_data["market_status"] = client.get_market_status()
        logger.info("All API portfolio/account data fetched.")
    except Exception as e:
        logger.error(f"Error fetching some portfolio data: {str(e)}")
        st.warning(f"Could not fetch all portfolio data: {str(e)}") # User feedback

    return portfolio_data


# Data Loading (API First, then CSV Fallback)
def load_data(client):
    """
    Attempts to load data from 5paisa API first, falls back to CSV if API fails.
    Returns DataFrame for analysis, real-time data dict, and data source tag.
    """
    df = None
    real_data = None
    data_source = "CSV (FALLBACK)" # Default source

    # Attempt to fetch real-time data from API
    real_data = fetch_real_time_market_data(client)

    if real_data and real_data.get("nifty_spot") is not None and real_data.get("vix") is not None:
        logger.info("Successfully fetched real-time data from 5paisa API.")
        data_source = real_data["source"] # Use the source tag from fetch function

        latest_date = datetime.now().date()
        live_df_row = pd.DataFrame({
            "NIFTY_Close": [real_data["nifty_spot"]],
            "VIX": [real_data["vix"]]
        }, index=[pd.to_datetime(latest_date).normalize()])

        # Load historical CSV data
        try:
            nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
            vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"

            nifty = pd.read_csv(io.StringIO(requests.get(nifty_url).text), encoding="utf-8-sig")
            vix = pd.read_csv(io.StringIO(requests.get(vix_url).text))

            nifty.columns = nifty.columns.str.strip()
            vix.columns = vix.columns.str.strip()

            nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
            vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")

            nifty = nifty.dropna(subset=["Date"]).set_index("Date")[["Close"]].rename(columns={"Close": "NIFTY_Close"})
            vix = vix.dropna(subset=["Date"]).set_index("Date")[["Close"]].rename(columns={"Close": "VIX"})

            # Normalize dates and remove duplicates, keeping last
            nifty.index = nifty.index.normalize()
            vix.index = vix.index.normalize()
            nifty = nifty.groupby(nifty.index).last()
            vix = vix.groupby(vix.index).last()

            common_dates = nifty.index.intersection(vix.index)
            historical_df = pd.DataFrame({
                "NIFTY_Close": nifty["NIFTY_Close"].loc[common_dates],
                "VIX": vix["VIX"].loc[common_dates]
            }, index=common_dates).dropna()

            # Ensure historical data is before the live data date to avoid duplicates on the same day
            historical_df = historical_df[historical_df.index < live_df_row.index[0]]

            # Combine historical and real-time data
            df = pd.concat([historical_df, live_df_row])
            df = df.groupby(df.index).last() # Final deduplication if any
            df = df.sort_index()
            df = df.ffill().bfill() # Fill any remaining gaps

            logger.debug(f"Combined historical and live data. Shape: {df.shape}")

        except Exception as e:
             logger.error(f"Error loading historical CSV data while having live data: {str(e)}")
             # If CSV loading fails but live data is there, just use the live data point
             df = live_df_row
             logger.warning("Proceeding with only live data point due to CSV error.")


    else:
        logger.warning("Failed to fetch real-time data from 5paisa API. Falling back to CSV.")
        real_data = None # Ensure real_data is None on fallback

        try:
            # Load historical CSV data (full range)
            nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
            vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"

            nifty = pd.read_csv(io.StringIO(requests.get(nifty_url).text), encoding="utf-8-sig")
            vix = pd.read_csv(io.StringIO(requests.get(vix_url).text))

            nifty.columns = nifty.columns.str.strip()
            vix.columns = vix.columns.str.strip()

            nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
            vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")

            nifty = nifty.dropna(subset=["Date"]).set_index("Date")[["Close"]].rename(columns={"Close": "NIFTY_Close"})
            vix = vix.dropna(subset=["Date"]).set_index("Date")[["Close"]].rename(columns={"Close": "VIX"})

            # Normalize dates and remove duplicates, keeping last
            nifty.index = nifty.index.normalize()
            vix.index = vix.index.normalize()
            nifty = nifty.groupby(nifty.index).last()
            vix = vix.groupby(vix.index).last()

            common_dates = nifty.index.intersection(vix.index)
            df = pd.DataFrame({
                "NIFTY_Close": nifty["NIFTY_Close"].loc[common_dates],
                "VIX": vix["VIX"].loc[common_dates]
            }, index=common_dates).dropna()

            df = df.groupby(df.index).last() # Final deduplication
            df = df.sort_index()
            df = df.ffill().bfill() # Fill any remaining gaps

            logger.debug(f"Loaded data from CSV fallback. Shape: {df.shape}")

        except Exception as e:
            logger.error(f"Fatal error loading data from CSV fallback: {str(e)}")
            st.error(f"Fatal Error: Could not load data from API or CSV fallback: {str(e)}") # User feedback
            return None, None, "Data Load Failed"


    # Ensure data is sufficient for analysis
    if df is None or len(df) < 2:
         st.error("Insufficient data loaded for analysis.")
         return None, None, data_source


    logger.debug(f"Data loading successful. Final DataFrame shape: {df.shape}. Source: {data_source}")
    return df, real_data, data_source


# Max Pain calculation (already have from notebook helpers)
# _calculate_max_pain and max_pain are defined earlier


# Feature Generation (Keep as is, uses df)
@st.cache_data
def generate_features(df, real_data, capital):
    # ... (This function remains the same, it takes df and real_data as inputs)
    # Ensure it's using the real_data values correctly if available
    try:
        logger.info("Generating features")
        df = df.copy()
        df.index = df.index.normalize()
        n_days = len(df)
        # Use a fixed seed only for reproducible parts that don't rely on live data randomness
        # np.random.seed(42) # Remove or modify for live data freshness

        # Use actual real_data values if available, fallback to historical patterns or defaults
        base_pcr = real_data["pcr"] if real_data and real_data.get("pcr") is not None else df["PCR"].iloc[-1] if "PCR" in df.columns and len(df) > 1 else 1.0 # Use last historical PCR if no live data
        base_straddle_price = real_data["straddle_price"] if real_data and real_data.get("straddle_price") is not None else df["Straddle_Price"].iloc[-1] if "Straddle_Price" in df.columns and len(df) > 1 else 200.0 # Use last historical straddle if no live data
        base_max_pain_diff_pct = real_data["max_pain_diff_pct"] if real_data and real_data.get("max_pain_diff_pct") is not None else df["Spot_MaxPain_Diff_Pct"].iloc[-1] if "Spot_MaxPain_Diff_Pct" in df.columns and len(df) > 1 else 0.5 # Use last historical diff if no live data
        base_vix_change_pct = real_data["vix_change_pct"] if real_data and real_data.get("vix_change_pct") is not None else df["VIX_Change_Pct"].iloc[-1] if "VIX_Change_Pct" in df.columns and len(df) > 1 else 0.0 # Use last historical change if no live data


        def calculate_days_to_expiry(dates):
            # This needs to be smarter for live data - it should use the *fetched* expiry date
            # For historical data, the Friday calculation is an approximation
            days_to_expiry = []
            fetched_expiry = datetime.strptime(real_data["expiry"], "%Y-%m-%d").date() if real_data and real_data.get("expiry") and real_data["expiry"] != "N/A" else None

            for date in dates:
                date_only = date.date()
                if fetched_expiry and date_only >= fetched_expiry:
                     dte = 0 # On or after expiry day
                elif fetched_expiry and date_only < fetched_expiry:
                     dte = (fetched_expiry - date_only).days
                else:
                    # Fallback for historical data or failed API expiry fetch
                    days_ahead = (3 - date_only.weekday()) % 7 # Days to next Thursday
                    if days_ahead == 0: # If today is Thursday, next expiry is in 7 days
                        days_ahead = 7
                    next_expiry_approx = date_only + timedelta(days=days_ahead)
                    dte = (next_expiry_approx - date_only).days
                days_to_expiry.append(dte)
            return np.array(days_to_expiry)


        df["Days_to_Expiry"] = calculate_days_to_expiry(df.index)
        # Ensure DTE is non-negative
        df["Days_to_Expiry"] = np.clip(df["Days_to_Expiry"], 0, None)


        # These synthetic features need care when mixed with live data
        # For the latest row (live data), use actual fetched values where possible
        df["ATM_IV"] = df["VIX"] # Use VIX as ATM IV approximation if not directly fetched
        if real_data and real_data.get("vix") is not None:
             df["ATM_IV"].iloc[-1] = real_data["vix"] # Use live VIX for latest ATM_IV

        # Event Flag based on fetched expiry date if available
        if real_data and real_data.get("expiry") and real_data["expiry"] != "N/A":
             fetched_expiry_dt = datetime.strptime(real_data["expiry"], "%Y-%m-%d").date()
             df["Event_Flag"] = np.where(
                  (df.index.date == fetched_expiry_dt) | # Expiry day
                  (df["Days_to_Expiry"] <= 3), # Near expiry (within 3 days)
                  1, 0
             )
        else:
             # Fallback event flag based on approximate Thursday expiry
             df["Event_Flag"] = np.where(
                  (df.index.weekday == 3) | # Thursdays
                  (df["Days_to_Expiry"] <= 3),
                  1, 0
             )


        # Re-calculate features that depend on previous rows or live data
        def dynamic_ivp(x):
            if len(x) >= 5 and x.iloc[-1] is not None and not pd.isna(x.iloc[-1]):
                # Only consider non-nan historical values for percentile rank
                historical_values = x.iloc[:-1].dropna()
                if not historical_values.empty:
                    return (np.sum(historical_values <= x.iloc[-1]) / len(historical_values)) * 100
            return 50.0 # Default if not enough data or latest is NaN/None
        # Apply IVP calculation, ensuring it handles potential NaNs at the end
        df["IVP"] = df["ATM_IV"].rolling(252, min_periods=5).apply(dynamic_ivp, raw=False) # Use raw=False to pass series
        # Fill initial NaNs and potential NaNs from apply
        df["IVP"] = df["IVP"].interpolate(method='linear').fillna(50.0) # Use linear interpolation

        market_trend = df["NIFTY_Close"].pct_change().rolling(5).mean().fillna(0)
        df["PCR"] = np.clip(base_pcr + np.random.normal(0, 0.05, n_days) + market_trend * -5, 0.7, 2.0) # Reduced random noise
        if real_data and real_data.get("pcr") is not None:
            df["PCR"].iloc[-1] = base_pcr # Use live PCR for latest

        df["VIX_Change_Pct"] = df["VIX"].pct_change().fillna(0) * 100
        if real_data and real_data.get("vix_change_pct") is not None:
             df["VIX_Change_Pct"].iloc[-1] = base_vix_change_pct # Use live VIX change for latest

        # These synthetic features are less tied to live data directly, keep approximation
        df["Spot_MaxPain_Diff_Pct"] = np.clip(base_max_pain_diff_pct + np.random.normal(0, 0.1, n_days) + df["Days_to_Expiry"]*0.01, 0.1, 5.0) # Slightly adjust logic
        if real_data and real_data.get("max_pain_diff_pct") is not None:
             df["Spot_MaxPain_Diff_Pct"].iloc[-1] = base_max_pain_diff_pct # Use live max pain diff for latest

        # FII data, Skew, Realized Vol, etc. are synthetic or historical approximations
        # If live FII/Skew were available from API, they'd be used here.
        # Keeping the random generation for now as they aren't in notebook API calls
        fii_trend = np.random.normal(0, 5000, n_days) # Reduced random noise
        fii_trend[::10] *= -1.5 # More frequent directional changes
        df["FII_Index_Fut_Pos"] = np.cumsum(fii_trend).astype(int)
        df["FII_Option_Pos"] = np.cumsum(np.random.normal(0, 2000, n_days)).astype(int) # Reduced random noise

        df["IV_Skew"] = np.clip(np.random.normal(0, 0.5, n_days) + (df["VIX"] / 20 - 1) * 2 + (df["Days_to_Expiry"] / 15 - 1)*0.5, -3, 3) # Slightly adjust logic

        # Realized Vol needs historical data window
        df["Realized_Vol"] = df["NIFTY_Close"].pct_change().rolling(5, min_periods=5).std() * np.sqrt(252) * 100
        # Fill initial NaNs and potential NaNs by using VIX or a reasonable default
        df["Realized_Vol"] = df["Realized_Vol"].fillna(df["VIX"]).fillna(15.0) # Fill with VIX, then a default

        df["Advance_Decline_Ratio"] = np.clip(1.0 + np.random.normal(0, 0.1, n_days) + market_trend * 5, 0.7, 1.5) # Reduced random noise and range
        df["Capital_Pressure_Index"] = (df["FII_Index_Fut_Pos"] / 5e4 + df["FII_Option_Pos"] / 2e4 + df["PCR"]-1) / 3 # Scale FII impact
        df["Capital_Pressure_Index"] = np.clip(df["Capital_Pressure_Index"], -1.5, 1.5) # Reduced range
        df["Gamma_Bias"] = np.clip(df["IV_Skew"] * (30 - np.clip(df["Days_to_Expiry"], 1, 30)) / 30, -2, 2) # Ensure DTE is > 0 for division

        df["Total_Capital"] = capital # Assign capital to every row for backtest scaling

        # Synthetic PnL Day - used in strategy evaluation but not for backtest PnL calculation
        df["PnL_Day"] = np.random.normal(0, capital * 0.005, n_days) * (1 - df["Event_Flag"] * 0.2) # Scale PnL to capital

        df["Straddle_Price"] = np.clip(base_straddle_price + np.random.normal(0, base_straddle_price*0.1, n_days), base_straddle_price*0.5, base_straddle_price*1.5) # Base around base_straddle_price
        if real_data and real_data.get("straddle_price") is not None:
             df["Straddle_Price"].iloc[-1] = base_straddle_price # Use live straddle for latest


        # Final check and interpolate any remaining NaNs
        if df.isna().sum().sum() > 0:
            logger.warning(f"NaNs found after initial feature generation: {df.isna().sum().sum()}")
            # Use a combination of interpolation and backward/forward fill
            df = df.apply(lambda x: x.interpolate(method='linear')).fillna(method='bfill').fillna(method='ffill')
            if df.isna().sum().sum() > 0:
                 logger.error(f"NaNs still present after interpolation/fill: {df.isna().sum().sum()}")
                 st.error("Error: Could not fill all missing data points.")


        # Save to CSV - ensure index is not saved as a separate column unless intended
        try:
            # Save with index=True if Date is the index, it will be the first column
            df.to_csv("volguard_hybrid_data.csv", index=True)
            logger.debug("volguard_hybrid_data.csv saved successfully")
        except PermissionError:
            logger.error("Permission denied when writing to volguard_hybrid_data.csv")
            st.error("Cannot save volguard_hybrid_data.csv: Permission denied")
        except Exception as e:
             logger.error(f"Error saving volguard_hybrid_data.csv: {e}")
             st.error(f"Error saving data file: {e}")


        logger.debug("Features generated successfully")
        return df
    except Exception as e:
        st.error(f"Error generating features: {str(e)}")
        logger.error(f"Error generating features: {str(e)}")
        return None


# Volatility Forecasting (Keep as is, uses df features)
feature_cols = [
    'VIX', 'ATM_IV', 'IVP', 'PCR', 'VIX_Change_Pct', 'IV_Skew', 'Straddle_Price',
    'Spot_MaxPain_Diff_Pct', 'Days_to_Expiry', 'Event_Flag', 'FII_Index_Fut_Pos',
    'FII_Option_Pos'
]

@st.cache_data
def forecast_volatility_future(df, forecast_horizon):
    # ... (This function remains largely the same, uses df)
    try:
        logger.info("Forecasting volatility")
        df = df.copy()
        df.index = pd.to_datetime(df.index).normalize()
        # Ensure df has enough data and required columns
        if len(df) < 200 or not all(col in df.columns for col in list(feature_cols) + ['NIFTY_Close', 'Realized_Vol']):
            st.error("Insufficient data or missing columns for volatility forecasting.")
            logger.error(f"Insufficient data ({len(df)} days) or missing columns for forecasting.")
            return None, None, None, None, None, None, None, None

        df_garch = df.tail(len(df)) # Use entire df for GARCH if needed, or a window
        # GARCH needs at least 2 observations for returns, but more for stability
        if len(df_garch) < 2:
             st.error("Insufficient data for GARCH model.")
             return None, None, None, None, None, None, None, None


        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon, freq='B')

        # Ensure Log_Returns calculation handles potential NaNs at the beginning
        df_garch['Log_Returns'] = np.log(df_garch['NIFTY_Close'].pct_change() + 1).dropna() * 100

        if df_garch['Log_Returns'].empty:
             st.error("Insufficient historical returns data for GARCH.")
             return None, None, None, None, None, None, None, None

        # Fit GARCH only if sufficient data
        if len(df_garch['Log_Returns']) >= 100: # A reasonable minimum for GARCH
             garch_model = arch_model(df_garch['Log_Returns'], vol='Garch', p=1, q=1, rescale=False)
             garch_fit = garch_model.fit(disp="off")
             garch_forecast = garch_fit.forecast(horizon=forecast_horizon, reindex=False)
             # Convert conditional standard deviation to annualized volatility (%)
             garch_vols = np.sqrt(garch_forecast.variance.iloc[-1].values) * np.sqrt(252)
             garch_vols = np.clip(garch_vols, 5, 50)
             logger.debug("GARCH forecast completed.")
        else:
             logger.warning(f"Insufficient data ({len(df_garch['Log_Returns'])} returns) for GARCH model. Skipping GARCH.")
             garch_vols = np.full(forecast_horizon, df["VIX"].iloc[-1] if "VIX" in df.columns and len(df) > 0 else 15.0) # Fallback vols


        # Ensure Realized Volatility calculation handles potential NaNs
        realized_vol = df["Realized_Vol"].dropna().iloc[-5:].mean() if not df["Realized_Vol"].dropna().empty and len(df["Realized_Vol"].dropna().iloc[-5:]) >= 1 else df["VIX"].iloc[-1] if "VIX" in df.columns and len(df) > 0 else 15.0 # Fallback realized vol


        df_xgb = df.tail(len(df)) # Use entire df for XGBoost
        # Ensure target variable exists and drop NaNs caused by shift
        if "Realized_Vol" not in df_xgb.columns:
             st.error("Realized_Vol feature is missing for XGBoost target.")
             return None, None, None, None, None, None, None, None

        df_xgb['Target_Vol'] = df_xgb['Realized_Vol'].shift(-1)
        df_xgb = df_xgb.dropna(subset=['Target_Vol'] + feature_cols) # Drop NaNs in target or features

        if len(df_xgb) < 50: # A reasonable minimum for XGBoost training
            st.error(f"Insufficient data ({len(df_xgb)} rows) for XGBoost training after dropping NaNs.")
            return None, None, None, None, None, None, None, None


        X = df_xgb[feature_cols]
        y = df_xgb['Target_Vol']

        # Ensure split_index is valid
        split_index = int(len(X) * 0.8)
        if split_index < 1 or split_index >= len(X):
             split_index = max(1, len(X) - 50) # Ensure some test data, at least 50 points

        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) # Use transform on test set

        model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.03, random_state=42)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        logger.debug(f"XGBoost training completed. RMSE: {rmse:.2f}")


        # Forecast using XGBoost
        xgb_vols = []
        # Start forecasting from the last known feature state
        current_row = df_xgb[feature_cols].iloc[-1].copy()

        for i in range(forecast_horizon):
            # Prepare the current feature row for prediction
            current_row_df = pd.DataFrame([current_row], columns=feature_cols)
            current_row_scaled = scaler.transform(current_row_df)

            # Predict the next volatility
            next_vol = model.predict(current_row_scaled)[0]
            xgb_vols.append(next_vol)

            # Simulate feature changes for the next day's prediction
            # These simulations should ideally be more sophisticated (e.g., based on predicted price move)
            # For now, keep the current simulation logic as it was
            current_row["Days_to_Expiry"] = max(1, current_row["Days_to_Expiry"] - 1)
            current_row["VIX"] = np.clip(current_row["VIX"] * np.random.uniform(0.98, 1.02), 5, 50) # Add bounds
            current_row["Straddle_Price"] = np.clip(current_row["Straddle_Price"] * np.random.uniform(0.98, 1.02), 50, 400) # Add bounds
            # VIX_Change_Pct is a daily change, recalculate based on simulated VIX change
            current_row["VIX_Change_Pct"] = ((current_row["VIX"] / (df_xgb["VIX"].iloc[-1] if len(df_xgb)>0 else current_row["VIX"])) - 1) * 100 # Use last actual VIX or current if no history
            current_row["ATM_IV"] = current_row["VIX"] * (1 + np.random.normal(0, 0.05)) # Reduced noise
            current_row["Realized_Vol"] = np.clip(next_vol * np.random.uniform(0.98, 1.02), 5, 50) # Use predicted vol with noise
            current_row["IVP"] = np.clip(current_row["IVP"] * np.random.uniform(0.99, 1.01), 0, 100) # Add bounds
            current_row["PCR"] = np.clip(current_row["PCR"] + np.random.normal(0, 0.02), 0.7, 2.0) # Reduced noise
            current_row["Spot_MaxPain_Diff_Pct"] = np.clip(current_row["Spot_MaxPain_Diff_Pct"] * np.random.uniform(0.98, 1.02), 0.1, 5.0) # Add bounds
            # Event Flag simulation could be improved - maybe base on remaining DTE to next known expiry
            current_row["Event_Flag"] = 1 if current_row["Days_to_Expiry"] <= 3 else 0 # Simple rule based on DTE
            current_row["FII_Index_Fut_Pos"] += np.random.normal(0, 500) # Reduced noise
            current_row["FII_Option_Pos"] += np.random.normal(0, 200) # Reduced noise
            current_row["IV_Skew"] = np.clip(current_row["IV_Skew"] + np.random.normal(0, 0.05), -3, 3) # Reduced noise

            # Ensure no NaNs creep in during simulation
            current_row = current_row.fillna(method='bfill').fillna(method='ffill')


        xgb_vols = np.clip(xgb_vols, 5, 50)
        # Apply event spike to XGBoost forecast if the last known day was an event day
        if df["Event_Flag"].iloc[-1] == 1:
            xgb_vols = [v * 1.05 for v in xgb_vols] # Reduced spike effect


        # Blending GARCH and XGBoost
        # Ensure garch_vols and xgb_vols are of the same length if horizon differs (shouldn't if forecast_horizon is used)
        if len(garch_vols) != len(xgb_vols):
             logger.error("GARCH and XGBoost forecast horizons mismatch.")
             # Fallback to just XGBoost if mismatch
             blended_vols = xgb_vols
             confidence_score = 50 # Low confidence due to issue
        else:
            # Calculate initial difference for weighting
            initial_garch_vol = garch_vols[0] if len(garch_vols) > 0 else realized_vol
            initial_xgb_vol = xgb_vols[0] if len(xgb_vols) > 0 else realized_vol

            garch_diff = np.abs(initial_garch_vol - realized_vol)
            xgb_diff = np.abs(initial_xgb_vol - realized_vol)

            # Avoid division by zero or very small numbers
            total_diff = garch_diff + xgb_diff
            garch_weight = xgb_diff / total_diff if total_diff > 0 else 0.5
            xgb_weight = 1 - garch_weight

            # Apply weights across the forecast horizon
            blended_vols = [(garch_weight * g) + (xgb_weight * x) for g, x in zip(garch_vols, xgb_vols)]

            # Confidence based on the agreement between models and deviation from realized
            model_agreement = 1 - (np.abs(initial_garch_vol - initial_xgb_vol) / max(initial_garch_vol, initial_xgb_vol) if max(initial_garch_vol, initial_xgb_vol) > 0 else 0)
            deviation_from_realized = 1 - (min(garch_diff, xgb_diff) / realized_vol if realized_vol > 0 else 0)
            confidence_score = min(100, max(30, (model_agreement * 0.6 + deviation_from_realized * 0.4) * 100)) # Adjust weighting and min confidence


        forecast_log = pd.DataFrame({
            "Date": future_dates,
            "GARCH_Vol": garch_vols,
            "XGBoost_Vol": xgb_vols,
            "Blended_Vol": blended_vols,
            "Confidence": [confidence_score] * forecast_horizon
        })
        logger.debug("Volatility forecast completed")
        return forecast_log, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, model.feature_importances_
    except Exception as e:
        st.error(f"Error in volatility forecasting: {str(e)}")
        logger.error(f"Error in volatility forecasting: {str(e)}")
        return None, None, None, None, None, None, None, None


# Backtesting (Keep as is, uses df features)
def run_backtest(df, capital, strategy_choice, start_date, end_date):
    # ... (This function remains the same, uses df)
    try:
        logger.info(f"Starting backtest for {strategy_choice} from {start_date} to {end_date}")
        if df.empty:
            st.error("Backtest failed: No data available")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        df = df.groupby(df.index).last().copy() # Ensure unique index and copy
        df = df.loc[start_date:end_date].copy() # Slice and copy
        if len(df) < 50:
            st.error(f"Backtest failed: Insufficient data ({len(df)} days) in selected range.")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        required_cols = ["NIFTY_Close", "ATM_IV", "Realized_Vol", "IV_Skew", "Days_to_Expiry", "Event_Flag", "Total_Capital", "Straddle_Price", "PCR", "VIX_Change_Pct", "Spot_MaxPain_Diff_Pct"] # Added relevant features
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Backtest failed: Missing required columns after date slicing: {missing_cols}")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()


        backtest_results = []
        lot_size = 25
        # Transaction costs as a percentage of premium or trade value might be more realistic
        # For simplicity here, keeping base transaction cost as a factor
        base_transaction_cost_factor = 0.002
        stt_factor = 0.0005 # Securities Transaction Tax
        portfolio_pnl = 0
        risk_free_rate_daily = 0.06 / 252 # Approx daily risk-free rate assuming 252 trading days
        # nifty_returns = df["NIFTY_Close"].pct_change() # Calculate outside loop if needed

        def run_strategy_engine(day_data, avg_vol_forecast, portfolio_pnl, capital):
            try:
                # Use day_data (real historical/live features) for strategy decision
                iv = day_data["ATM_IV"]
                hv = day_data["Realized_Vol"]
                iv_hv_gap = iv - hv
                iv_skew = day_data["IV_Skew"]
                dte = day_data["Days_to_Expiry"]
                event_flag = day_data["Event_Flag"]
                pcr = day_data["PCR"] # Use PCR in strategy logic
                vix_change_pct = day_data["VIX_Change_Pct"] # Use VIX change

                # Drawdown limit check based on total capital
                if portfolio_pnl < -0.10 * capital: # 10% drawdown limit
                    return None, None, "Portfolio drawdown limit reached", [], 0, 0, 0

                # Determine regime based on blended forecast volatility (avg_vol_forecast)
                if avg_vol_forecast is None:
                     regime = "MEDIUM" # Default if forecast failed
                elif avg_vol_forecast < 15:
                    regime = "LOW"
                elif avg_vol_forecast < 20:
                    regime = "MEDIUM"
                else:
                    regime = "HIGH"

                strategy = "Undefined"
                reason = "N/A"
                tags = []
                risk_reward = 1.0 # Base risk-reward

                # Strategy selection logic based on regime and real-time indicators
                if regime == "LOW":
                    if iv_hv_gap > 3 and dte < 15: # Adjust thresholds
                        strategy = "Butterfly Spread"
                        reason = "Low vol & moderate expiry favors pinning strategies"
                        tags = ["Neutral", "Theta", "Expiry Play"]
                        risk_reward = 2.5 # Higher reward potential
                    else:
                        strategy = "Iron Fly"
                        reason = "Low volatility environment favors delta-neutral Iron Fly"
                        tags = ["Neutral", "Theta", "Range Bound"]
                        risk_reward = 1.8

                elif regime == "MEDIUM":
                    if iv_hv_gap > 2 and iv_skew > 1: # Adjust thresholds
                        strategy = "Iron Condor"
                        reason = "Medium vol and skew favor wide-range Iron Condor"
                        tags = ["Neutral", "Theta", "Range Bound"]
                        risk_reward = 1.5
                    elif pcr > 1.2 and dte < 10: # Example: Bullish bias + short expiry
                     strategy = "Short Put Vertical Spread" # Example directional strategy
                     reason = "Medium vol, bullish PCR, and short expiry"
                     tags = ["Directional", "Bullish", "Defined Risk"]
                     risk_reward = 1.2
                    else:
                        strategy = "Short Strangle"
                        reason = "Balanced vol, premium-rich environment for Short Strangle"
                        tags = ["Neutral", "Premium Selling", "Volatility Harvest"]
                        risk_reward = 1.6

                elif regime == "HIGH":
                    if iv_hv_gap > 8 or vix_change_pct > 5: # High gap or spike
                        strategy = "Jade Lizard"
                        reason = "High IV spike/gap favors Jade Lizard for defined upside risk"
                        tags = ["Skewed", "Volatility", "Defined Risk"]
                        risk_reward = 1.0
                    elif dte < 10: # High vol + near expiry
                         strategy = "Iron Condor" # Still viable for premium capture
                         reason = "High vol and near expiry favors wide premium collection"
                         tags = ["Neutral", "Theta", "Range Bound"]
                         risk_reward = 1.3
                    else:
                         strategy = "Long Put" # Example protective/bearish strategy in high vol
                         reason = "High vol suggests potential downside risk, protective put"
                         tags = ["Directional", "Bearish", "Protection"]
                         risk_reward = 2.0 # High potential if market drops

                elif regime == "EVENT-DRIVEN":
                    if iv > 35 and dte < 3: # Higher IV threshold, very near expiry
                        strategy = "Short Straddle"
                        reason = "Extreme IV + very near expiry event â†’ max premium capture"
                        tags = ["Volatility", "Event", "Neutral"]
                        risk_reward = 1.8 # Higher potential reward due to premium
                    else:
                        strategy = "Calendar Spread"
                        reason = "Event-based uncertainty favors term structure opportunity"
                        tags = ["Volatility", "Event", "Calendar"]
                        risk_reward = 1.5


                # Capital allocation based on regime
                capital_alloc_pct = {"LOW": 0.10, "MEDIUM": 0.08, "HIGH": 0.06, "EVENT-DRIVEN": 0.07}.get(regime, 0.07)
                deploy = capital * capital_alloc_pct # Simple allocation

                # Dynamic max loss based on strategy risk and capital allocation
                max_loss_pct = {"Iron Condor": 0.02, "Butterfly Spread": 0.015, "Iron Fly": 0.02, "Short Strangle": 0.03, "Calendar Spread": 0.015, "Jade Lizard": 0.025, "Short Straddle": 0.04, "Short Put Vertical Spread": 0.015, "Long Put": 0.03}.get(strategy, 0.025)
                max_loss = deploy * max_loss_pct # Max loss absolute value

                return regime, strategy, reason, tags, deploy, max_loss, risk_reward
            except Exception as e:
                logger.error(f"Error in backtest strategy engine: {str(e)}")
                return None, None, f"Strategy engine failed: {str(e)}", [], 0, 0, 0

        def calculate_trade_pnl(strategy, day_data, prev_day_data, deploy, max_loss):
             try:
                # Simplified PnL calculation for backtest
                # A real backtest would simulate option prices based on Black-Scholes or similar,
                # tracking Greeks, volatility changes, and time decay. This is a simplification.
                premium = day_data["Straddle_Price"] # Use Straddle Price as a proxy for strategy premium
                lot_size = 25
                # Approximate lots based on deployable capital and proxy premium
                lots = max(1, int(deploy / (premium * lot_size)) if premium > 0 else 1)
                if lots == 0: lots = 1 # Ensure at least 1 lot if deploy > 0

                # Simulate market move impact
                nifty_move_abs_pct = abs(day_data["NIFTY_Close"] / prev_day_data["NIFTY_Close"] - 1) * 100 if prev_day_data["NIFTY_Close"] > 0 else 0
                implied_move_1sd = day_data["ATM_IV"] / np.sqrt(252) # Daily 1-sigma move in %

                # PnL depends on strategy type and market move relative to volatility
                base_pnl = 0 # Default PnL if no clear outcome simulated
                strategy_sensitivity = {
                    "Iron Condor": -0.5, # Loses when market moves beyond range
                    "Butterfly Spread": -0.8, # Very sensitive to market pinning
                    "Iron Fly": -0.6, # Sensitive to move away from strike
                    "Short Strangle": -0.7, # Loses from large moves
                    "Calendar Spread": 0.3, # Benefits from time decay and changing term structure
                    "Jade Lizard": -0.4, # Limited upside risk, but loses on large moves
                    "Short Straddle": -1.0, # Loses heavily from large moves
                    "Short Put Vertical Spread": 0.8 if day_data["NIFTY_Close"] > prev_day_data["NIFTY_Close"] else -1.2, # Directional
                    "Long Put": -0.5 if day_data["NIFTY_Close"] > prev_day_data["NIFTY_Close"] else 1.5 # Directional
                }.get(strategy, -0.5) # Default sensitivity

                # PnL simulation based on premium and market move vs expected move
                # This is highly simplified. A real backtest needs option Greeks simulation.
                # premium_collected = premium * lot_size * lots # Simplified premium
                # expected_daily_move_pct = day_data["ATM_IV"] / np.sqrt(252)

                # Simulate PnL based on the strategy's sensitivity to unexpected moves
                # If actual move > implied move, strategies selling premium tend to lose
                # If actual move < implied move, strategies selling premium tend to win (from decay)

                move_ratio = nifty_move_abs_pct / implied_move_1sd if implied_move_1sd > 0 else 2.0 # Ratio of actual move to expected move

                # Base daily gain/loss as a percentage of deployed capital
                # Strategies selling premium benefit from decay unless move is large
                base_daily_gain_pct = 0.001 # Small gain from theta decay daily
                if "Short" in strategy or "Iron" in strategy or "Jade Lizard" in strategy: # Premium selling strategies
                    # Lose more if move is large relative to expected
                    loss_factor = max(0, move_ratio - 1.0) * abs(strategy_sensitivity)
                    base_daily_gain_pct -= loss_factor * 0.02 # Higher loss factor

                elif "Long" in strategy or "Calendar Spread" in strategy: # Buying premium or more complex
                    # May benefit from large moves or volatility spikes
                    gain_factor = max(0, move_ratio - 0.5) * abs(strategy_sensitivity) * np.sign(day_data["NIFTY_Close"] - prev_day_data["NIFTY_Close"]) # Simplified directionality
                    base_daily_gain_pct += gain_factor * 0.015 # Higher gain factor

                # Apply decay factor - strategies selling premium benefit from time passing
                decay_benefit_factor = 0.0005 * max(0, 15 - day_data["Days_to_Expiry"]) if "Short" in strategy or "Iron" in strategy or "Jade Lizard" in strategy else 0
                base_daily_gain_pct += decay_benefit_factor

                # Apply event shock if applicable
                if day_data["Event_Flag"] == 1:
                    event_impact = np.random.uniform(-0.03, 0.03) * abs(strategy_sensitivity) # Random positive or negative shock
                    base_daily_gain_pct += event_impact

                # Convert percentage PnL to absolute PnL based on deployed capital
                pnl = deploy * base_daily_gain_pct * np.random.uniform(0.8, 1.2) # Add some daily randomness

                # Ensure PnL is within max_loss boundaries
                # Winning PnL should also be capped relative to deploy or max_loss/risk_reward
                max_win = max_loss * risk_reward if risk_reward is not None else max_loss * 1.0
                pnl = max(-max_loss, min(pnl, max_win))


                # Simulate transaction costs (simplified)
                num_legs = {"Short Straddle": 2, "Short Strangle": 2, "Iron Condor": 4, "Iron Fly": 4, "Butterfly Spread": 3, "Jade Lizard": 3, "Calendar Spread": 2, "Short Put Vertical Spread": 2, "Long Put": 1}.get(strategy, 2)
                transaction_cost = deploy * base_transaction_cost_factor * num_legs + deploy * stt_factor # Simplified cost calculation
                pnl -= transaction_cost

                # Add small random noise to final PnL
                pnl += np.random.normal(0, deploy * 0.001)


                return pnl

             except Exception as e:
                logger.error(f"Error calculating trade PnL for {strategy} on {day_data.name}: {str(e)}")
                return 0 # Return 0 PnL on error


        # Get forecast for the backtest period to use avg_vol_forecast
        # Note: In a true backtest, forecast would be re-run *each day* using past data only.
        # This uses a single forecast for the whole period, which is a simplification.
        # A more accurate backtest requires simulating data availability day-by-day.
        # For this version, we'll use the blended forecast from the main run if available,
        # or fall back to simple average of historical realized vol in the backtest window.

        # Try to get the forecast_log from session state if available from the main run
        blended_vols_forecast = None
        if 'forecast_log' in st.session_state and st.session_state.forecast_log is not None:
             # Map forecast dates to backtest dates - this is tricky and likely inaccurate
             # if backtest period doesn't align with forecast horizon.
             # A better approach is needed for real backtesting with forecasting.
             # For now, we'll use the first blended vol from the main forecast as a proxy for "current" market volatility
             # and use historical realized vol for the rest of the backtest.
             blended_vols_forecast = st.session_state.forecast_log["Blended_Vol"].iloc[0] if not st.session_state.forecast_log.empty else None
             logger.debug(f"Using blended forecast from main run: {blended_vols_forecast}")

        # Backtest loop
        for i in range(1, len(df)):
            try:
                day_data = df.iloc[i]
                prev_day_data = df.iloc[i-1]
                date = day_data.name

                # Use the blended forecast from the main run if available, otherwise use a simple historical average
                # This is a backtest simplification. A rigorous backtest would only use data available *before* the trade date.
                avg_vol_for_strategy = blended_vols_forecast if blended_vols_forecast is not None else df["Realized_Vol"].iloc[max(0, i-5):i].mean()

                regime, strategy, reason, tags, deploy, max_loss, risk_reward = run_strategy_engine(
                    day_data, avg_vol_for_strategy, portfolio_pnl, capital # Pass capital to engine
                )

                # Filter strategies if a specific one is chosen
                if strategy is None or (strategy_choice != "All Strategies" and strategy != strategy_choice):
                    continue

                # Calculate PnL for the day's trade
                # Pass day_data and prev_day_data for PnL calculation
                pnl = calculate_trade_pnl(strategy, day_data, prev_day_data, deploy, max_loss)

                portfolio_pnl += pnl # Accumulate portfolio PnL

                backtest_results.append({
                    "Date": date,
                    "Regime": regime,
                    "Strategy": strategy,
                    "PnL": pnl,
                    "Cumulative_PnL": portfolio_pnl, # Track cumulative PnL
                    "Capital_Deployed": deploy,
                    "Max_Loss": max_loss,
                    "Risk_Reward": risk_reward
                    # Could add entry price, exit price simulation here for more detail
                })
            except Exception as e:
                logger.error(f"Error in backtest loop at index {i}: {str(e)}")
                continue # Continue backtest even if one day fails

        backtest_df = pd.DataFrame(backtest_results)

        if backtest_df.empty:
            logger.warning(f"No trades generated for {strategy_choice} from {start_date} to {end_date}")
            return backtest_df, 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        # Final performance metrics calculation
        total_pnl = backtest_df["PnL"].sum()
        win_rate = len(backtest_df[backtest_df["PnL"] > 0]) / len(backtest_df) if len(backtest_df) > 0 else 0

        # Calculate Max Drawdown correctly from Cumulative PnL
        backtest_df['Cumulative_PnL'] = backtest_df['PnL'].cumsum()
        backtest_df['Peak'] = backtest_df['Cumulative_PnL'].cummax()
        backtest_df['Drawdown'] = backtest_df['Peak'] - backtest_df['Cumulative_PnL']
        max_drawdown = backtest_df['Drawdown'].max() if not backtest_df.empty else 0


        backtest_df.set_index("Date", inplace=True)

        # Calculate daily returns based on capital at start of each day (simplified)
        # Assuming capital grows by cumulative PnL
        capital_series = pd.Series(capital, index=df.index).add(backtest_df['Cumulative_PnL'].reindex(df.index).fillna(method='ffill').fillna(0))
        daily_backtest_pnl = backtest_df['PnL'].reindex(capital_series.index).fillna(0) # Align PnL with capital dates
        daily_returns = daily_backtest_pnl / capital_series.shift(1).fillna(capital) # Daily return based on previous day's capital
        daily_returns = daily_returns.dropna() # Drop first NaN

        # Ensure NIFTY returns are aligned and calculated correctly
        df_aligned = df.reindex(daily_returns.index)
        nifty_daily_returns = df_aligned["NIFTY_Close"].pct_change().dropna()
        # Reindex daily_returns to match nifty_daily_returns for excess return calculation
        daily_returns_aligned = daily_returns.reindex(nifty_daily_returns.index).fillna(0)


        excess_returns = daily_returns_aligned - nifty_daily_returns - risk_free_rate_daily
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() != 0 else 0 # Annualized Sharpe
        sortino_std_negative = excess_returns[excess_returns < 0].std()
        sortino_ratio = excess_returns.mean() / sortino_std_negative * np.sqrt(252) if sortino_std_negative != 0 else 0 # Annualized Sortino
        calmar_ratio = (total_pnl / capital) / (max_drawdown / capital) if max_drawdown != 0 and capital != 0 else float('inf')


        # Performance by Strategy and Regime
        strategy_perf = backtest_df.groupby("Strategy")["PnL"].agg(['sum', 'count', 'mean']).reset_index()
        strategy_win_rates = backtest_df.groupby("Strategy")["PnL"].apply(lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0).reset_index(name="Win_Rate")
        strategy_perf = pd.merge(strategy_perf, strategy_win_rates, on="Strategy")

        regime_perf = backtest_df.groupby("Regime")["PnL"].agg(['sum', 'count', 'mean']).reset_index()
        regime_win_rates = backtest_df.groupby("Regime")["PnL"].apply(lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0).reset_index(name="Win_Rate")
        regime_perf = pd.merge(regime_perf, regime_win_rates, on="Regime")


        logger.debug("Backtest completed successfully")
        return backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf
    except Exception as e:
        st.error(f"Error running backtest: {str(e)}")
        logger.error(f"Error running backtest: {str(e)}")
        return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()


# Strategy Generation (Keep as is, uses df and forecast_log)
@st.cache_data
def generate_trading_strategy(df, forecast_log, realized_vol, risk_tolerance, confidence_score, capital):
    # ... (This function remains the same, uses df, forecast_log)
    # Ensure it uses the latest data point from df
    try:
        logger.info("Generating trading strategy")
        df = df.copy()
        df.index = df.index.normalize()
        if df.empty:
             logger.error("Cannot generate strategy: Input DataFrame is empty.")
             return None

        latest = df.iloc[-1]
        # Ensure required columns exist in the latest row
        required_latest_cols = ["ATM_IV", "Realized_Vol", "IV_Skew", "PCR", "Days_to_Expiry", "Event_Flag", "VIX", "Spot_MaxPain_Diff_Pct", "PnL_Day"]
        if not all(col in latest.index for col in required_latest_cols):
             missing = [col for col in required_latest_cols if col not in latest.index]
             logger.error(f"Missing required columns in latest data for strategy generation: {missing}")
             st.error(f"Cannot generate strategy: Missing data ({', '.join(missing)})")
             return None


        avg_vol = np.mean(forecast_log["Blended_Vol"]) if forecast_log is not None and not forecast_log.empty else latest["Realized_Vol"] # Use realized as fallback
        iv = latest["ATM_IV"]
        hv = latest["Realized_Vol"]
        iv_hv_gap = iv - hv
        iv_skew = latest["IV_Skew"]
        pcr = latest["PCR"]
        dte = latest["Days_to_Expiry"]
        event_flag = latest["Event_Flag"]
        latest_vix = latest["VIX"]
        spot_max_pain_diff_pct = latest["Spot_MaxPain_Diff_Pct"]
        pnl_day = latest["PnL_Day"]


        risk_flags = []
        if latest_vix > 25:
            risk_flags.append("VIX > 25% - High Volatility Risk")
        if spot_max_pain_diff_pct > 3: # Adjust threshold
            risk_flags.append(f"Spot-Max Pain Diff > {spot_max_pain_diff_pct:.1f}% - Potential Pinning Risk")
        if pnl_day < -0.01 * capital: # Daily loss > 1% of capital
            risk_flags.append(f"Recent Daily Loss ({pnl_day:,.0f} â‚¹) - Consider reducing size")
        if latest["VIX_Change_Pct"] > 8: # Adjust threshold
            risk_flags.append(f"High VIX Spike Detected ({latest['VIX_Change_Pct']:+.1f}%)")


        if risk_flags:
            st.session_state.violations += 1
            if st.session_state.violations >= 2 and not st.session_state.journal_complete:
                # If violations exceed limit and journaling isn't complete, return None to enforce lock
                return None
            # Otherwise, allow strategy generation but show warnings


        # Regime determination based on average forecast volatility
        if avg_vol < 15:
            regime = "LOW"
        elif avg_vol < 20:
            regime = "MEDIUM"
        else:
            regime = "HIGH"

        # Add Event-Driven regime check
        if event_flag == 1 or dte <= 3: # Within 3 days of expiry or explicit event flag
             regime = "EVENT-DRIVEN"


        strategy = "Undefined"
        reason = "N/A"
        tags = []
        risk_reward = 1.0

        # Strategy selection logic - refine based on combined factors
        if regime == "LOW":
            if iv_hv_gap > 2 and dte < 10:
                strategy = "Butterfly Spread"
                reason = "Low forecast vol, IV > HV, near expiry favors pinning"
                tags = ["Neutral", "Theta", "Expiry Play"]
                risk_reward = 2.5
            elif iv_skew < -1: # Negative skew
                 strategy = "Short Put" # Simple directional play
                 reason = "Low forecast vol, negative IV skew suggests put selling opportunity"
                 tags = ["Directional", "Bullish", "Premium Selling"]
                 risk_reward = 1.5
            else:
                strategy = "Iron Fly"
                reason = "Low forecast volatility favors delta-neutral Iron Fly"
                tags = ["Neutral", "Theta", "Range Bound"]
                risk_reward = 1.8

        elif regime == "MEDIUM":
            if iv_hv_gap > 1.5 and iv_skew > 0.5:
                strategy = "Iron Condor"
                reason = "Medium forecast vol, IV > HV, positive skew favors Iron Condor"
                tags = ["Neutral", "Theta", "Range Bound"]
                risk_reward = 1.5
            elif pcr > 1.1:
                 strategy = "Short Put Vertical Spread" # Bullish bias
                 reason = "Medium forecast vol, bullish PCR suggests defined risk put spread"
                 tags = ["Directional", "Bullish", "Defined Risk"]
                 risk_reward = 1.2
            elif pcr < 0.9:
                 strategy = "Short Call Vertical Spread" # Bearish bias
                 reason = "Medium forecast vol, bearish PCR suggests defined risk call spread"
                 tags = ["Directional", "Bearish", "Defined Risk"]
                 risk_reward = 1.2
            else:
                strategy = "Short Strangle"
                reason = "Medium forecast vol, balanced indicators, premium capture"
                tags = ["Neutral", "Premium Selling", "Volatility Harvest"]
                risk_reward = 1.6

        elif regime == "HIGH":
            if iv_hv_gap > 5 or latest_vix > 28: # High IV spike
                strategy = "Jade Lizard"
                reason = "High IV spike, IV > HV favors Jade Lizard for defined upside"
                tags = ["Skewed", "Volatility", "Defined Risk"]
                risk_reward = 1.0
            elif iv_skew < -2: # Extreme negative skew
                 strategy = "Long Put" # Protective or bearish play
                 reason = "High forecast vol, extreme negative skew suggests downside protection"
                 tags = ["Directional", "Bearish", "Protection"]
                 risk_reward = 2.0
            else:
                 strategy = "Iron Condor" # Still an option for premium if range is wide enough
                 reason = "High forecast vol, wide expected range favors Iron Condor premium"
                 tags = ["Neutral", "Theta", "Range Bound"]
                 risk_reward = 1.3

        elif regime == "EVENT-DRIVEN":
            if iv > 30 and dte < 3:
                strategy = "Short Straddle"
                reason = "High IV, very near expiry event â†’ max premium capture"
                tags = ["Volatility", "Event", "Neutral"]
                risk_reward = 1.8
            else:
                strategy = "Calendar Spread"
                reason = "Event-based uncertainty favors term structure opportunity"
                tags = ["Volatility", "Event", "Calendar"]
                risk_reward = 1.5


        # Confidence score based on forecast agreement and deviation from realized (from forecast function)
        confidence_score_from_forecast = confidence_score if confidence_score is not None else 50 # Use calculated confidence, default to 50

        # Capital allocation based on regime and risk tolerance
        capital_alloc_pct = {"LOW": 0.10, "MEDIUM": 0.08, "HIGH": 0.06, "EVENT-DRIVEN": 0.07}.get(regime, 0.08) # Match backtest allocation
        position_size_multiplier = {"Conservative": 0.5, "Moderate": 1.0, "Aggressive": 1.5}[risk_tolerance]
        deploy = capital * capital_alloc_pct * position_size_multiplier # Scale by risk tolerance

        # Max loss calculation matching backtest logic
        max_loss_pct = {"Iron Condor": 0.02, "Butterfly Spread": 0.015, "Iron Fly": 0.02, "Short Strangle": 0.03, "Calendar Spread": 0.015, "Jade Lizard": 0.025, "Short Straddle": 0.04, "Short Put Vertical Spread": 0.015, "Long Put": 0.03, "Short Put": 0.02}.get(strategy, 0.025) # Added Short Put
        max_loss = deploy * max_loss_pct


        total_exposure = deploy / capital if capital > 0 else 0

        # Behavior Score (Keep as is)
        behavior_score = 8 if total_exposure < 0.10 else 6 # Adjust threshold to 10% exposure
        behavior_warnings = ["Consider reducing position size (Exposure > 10%)"] if behavior_score < 8 else []


        logger.debug(f"Trading strategy generated: {strategy}")
        return {
            "Regime": regime,
            "Strategy": strategy,
            "Reason": reason,
            "Tags": tags,
            "Confidence": confidence_score_from_forecast, # Use calculated confidence
            "Risk_Reward": risk_reward,
            "Deploy": deploy,
            "Max_Loss": max_loss,
            "Exposure": total_exposure * 100, # Display as percentage
            "Risk_Flags": risk_flags,
            "Behavior_Score": behavior_score,
            "Behavior_Warnings": behavior_warnings
        }
    except Exception as e:
        st.error(f"Error generating strategy: {str(e)}")
        logger.error(f"Error generating strategy: {str(e)}")
        return None


# Trading Functions - Modified for Confirmation UI
def prepare_trade_orders(strategy, real_data, capital):
    """
    Prepares the list of individual orders for a given strategy based on real-time data.
    Returns a list of dictionaries representing the orders.
    """
    logger.info(f"Preparing trade orders for: {strategy['Strategy']}")
    if not real_data or "option_chain" not in real_data or "atm_strike" not in real_data or "expiry" not in real_data or "straddle_price" not in real_data:
        logger.error("Invalid or incomplete real-time data for order preparation.")
        st.error("Cannot prepare orders: Real-time data missing.")
        return None

    option_chain = real_data["option_chain"]
    atm_strike = real_data["atm_strike"]
    expiry_date_str = real_data["expiry"]
    straddle_price_live = real_data["straddle_price"]

    if option_chain.empty or atm_strike is None or expiry_date_str == "N/A" or straddle_price_live is None or straddle_price_live <= 0:
         logger.error("Essential market data (Option Chain, ATM, Expiry, Straddle) is not valid for order preparation.")
         st.error("Cannot prepare orders: Essential market data is incomplete.")
         return None


    lot_size = 25 # Standard NIFTY lot size
    deploy = strategy["Deploy"]
    max_loss = strategy["Max_Loss"] # Max loss for the strategy

    # Determine lots based on deployable capital and live straddle price
    # Ensure we don't divide by zero or a very small number
    premium_per_lot = straddle_price_live * lot_size if straddle_price_live > 0 else 200 * lot_size # Fallback premium if live is zero
    lots = max(1, int(deploy / premium_per_lot) if premium_per_lot > 0 else 1) # Ensure at least 1 lot if deploy > 0

    # Optional: Cap lots based on a percentage of max loss per lot (e.g., max loss per lot is 20% of premium)
    # max_loss_per_lot_approx = premium_per_lot * 0.2
    # max_lots_by_loss = int(max_loss / max_loss_per_lot_approx) if max_loss_per_lot_approx > 0 else lots
    # lots = min(lots, max_lots_by_loss)
    # Ensure lots is a reasonable number, e.g., max 10 lots
    lots = min(lots, 10)


    orders_to_place = [] # List to hold prepared order dictionaries

    # Define strikes and types based on the strategy
    strategy_legs = []
    expiry_timestamp = parse_5paisa_date_string(option_chain["ExpiryDate"].iloc[0]) # Get timestamp from OC data

    if strategy["Strategy"] == "Short Straddle":
        strategy_legs = [(atm_strike, "CE", "S"), (atm_strike, "PE", "S")]
    elif strategy["Strategy"] == "Short Strangle":
        # Need to find strikes approximately 100 points away - use option chain strikes
        strikes_sorted = option_chain["StrikeRate"].sort_values().tolist()
        call_strike = None
        put_strike = None
        # Find OTM call strike >= atm_strike + 100
        for strike in strikes_sorted:
             if strike >= atm_strike + 100:
                  call_strike = strike
                  break
        # Find OTM put strike <= atm_strike - 100
        for strike in reversed(strikes_sorted):
             if strike <= atm_strike - 100:
                  put_strike = strike
                  break

        if call_strike is None or put_strike is None:
             logger.error(f"Could not find suitable strikes for Short Strangle around {atm_strike}.")
             st.error("Cannot prepare Short Strangle: Suitable strikes not found.")
             return None

        strategy_legs = [(call_strike, "CE", "S"), (put_strike, "PE", "S")]

    elif strategy["Strategy"] == "Iron Condor":
        # Need 4 strikes: Buy OTM Put, Sell OTM Put, Sell OTM Call, Buy OTM Call
        strikes_sorted = option_chain["StrikeRate"].sort_values().tolist()
        put_sell_strike = None
        put_buy_strike = None
        call_sell_strike = None
        call_buy_strike = None

        # Find strikes: Put Buy < Put Sell <= ATM < Call Sell < Call Buy
        # Start from ATM and move outwards
        put_sell_strike = next((s for s in reversed(strikes_sorted) if s <= atm_strike - 50), None) # Approx 50-100 points below ATM
        put_buy_strike = next((s for s in reversed(strikes_sorted) if s < put_sell_strike - 100 if put_sell_strike is not None), None) # Approx 100 points below sell put

        call_sell_strike = next((s for s in strikes_sorted if s >= atm_strike + 50), None) # Approx 50-100 points above ATM
        call_buy_strike = next((s for s in strikes_sorted if s > call_sell_strike + 100 if call_sell_strike is not None), None) # Approx 100 points above sell call

        if None in [put_sell_strike, put_buy_strike, call_sell_strike, call_buy_strike]:
            logger.error("Could not find suitable strikes for Iron Condor.")
            st.error("Cannot prepare Iron Condor: Suitable strikes not found.")
            return None

        strategy_legs = [
            (put_buy_strike, "PE", "B"),
            (put_sell_strike, "PE", "S"),
            (call_sell_strike, "CE", "S"),
            (call_buy_strike, "CE", "B")
        ]

    elif strategy["Strategy"] == "Iron Fly":
        # Sell ATM Straddle, Buy OTM wings (e.g., 100 points out)
        strikes_sorted = option_chain["StrikeRate"].sort_values().tolist()
        put_buy_strike = next((s for s in reversed(strikes_sorted) if s <= atm_strike - 100), None)
        call_buy_strike = next((s for s in strikes_sorted if s >= atm_strike + 100), None)

        if put_buy_strike is None or call_buy_strike is None:
             logger.error("Could not find suitable strikes for Iron Fly wings.")
             st.error("Cannot prepare Iron Fly: Suitable wing strikes not found.")
             return None

        strategy_legs = [
            (atm_strike, "PE", "S"),
            (atm_strike, "CE", "S"),
            (put_buy_strike, "PE", "B"),
            (call_buy_strike, "CE", "B")
        ]

    elif strategy["Strategy"] == "Butterfly Spread":
        # Sell ATM Call, Buy OTM Call, Buy ITM Put (or vice versa for puts)
        # Using a Call Butterfly example: Buy ITM Call, Sell 2x ATM Call, Buy OTM Call
        # Simplified: Sell ATM Call, Buy OTM Call, Buy OTM Put (similar strikes)
         strikes_sorted = option_chain["StrikeRate"].sort_values().tolist()
         otm_strike_lower = next((s for s in reversed(strikes_sorted) if s <= atm_strike - 100), None)
         otm_strike_upper = next((s for s in strikes_sorted if s >= atm_strike + 100), None)

         if otm_strike_lower is None or otm_strike_upper is None:
              logger.error("Could not find suitable strikes for Butterfly wings.")
              st.error("Cannot prepare Butterfly: Suitable wing strikes not found.")
              return None

         # Using a Call Butterfly structure: Buy ITM, Sell 2x ATM, Buy OTM
         # Need to select strikes - simplify by using ATM and +/- 100 points if available
         strike_itm = next((s for s in reversed(strikes_sorted) if s < atm_strike), None)
         strike_otm = next((s for s in strikes_sorted if s > atm_strike), None)

         # A standard butterfly is equidistant. Let's try ATM +/- 100 points
         strike_lower_wing = next((s for s in reversed(strikes_sorted) if s <= atm_strike - 100), None)
         strike_upper_wing = next((s for s in strikes_sorted if s >= atm_strike + 100), None)

         if strike_lower_wing is None or strike_upper_wing is None:
              logger.error("Could not find suitable strikes for Butterfly +/- 100 points.")
              st.error("Cannot prepare Butterfly: Suitable +/- 100 strikes not found.")
              return None

         # Assuming a short call butterfly (sell ATM, buy wings):
         strategy_legs = [
             (strike_lower_wing, "CE", "B"), # Buy ITM Call
             (atm_strike, "CE", "S"), # Sell ATM Call
             (atm_strike, "CE", "S"), # Sell ATM Call (2x quantity)
             (strike_upper_wing, "CE", "B") # Buy OTM Call
         ]
         # Need to adjust quantities below for the doubled leg


    elif strategy["Strategy"] == "Jade Lizard":
         # Short OTM Call, Short OTM Put, Long further OTM Put
         strikes_sorted = option_chain["StrikeRate"].sort_values().tolist()
         call_sell_strike = next((s for s in strikes_sorted if s >= atm_strike + 100), None) # Approx 100 points OTM Call
         put_sell_strike = next((s for s in reversed(strikes_sorted) if s <= atm_strike - 100), None) # Approx 100 points OTM Put
         put_buy_strike = next((s for s in reversed(strikes_sorted) if s < put_sell_strike - 100 if put_sell_strike is not None), None) # Further OTM Put

         if None in [call_sell_strike, put_sell_strike, put_buy_strike]:
              logger.error("Could not find suitable strikes for Jade Lizard.")
              st.error("Cannot prepare Jade Lizard: Suitable strikes not found.")
              return None

         strategy_legs = [
             (call_sell_strike, "CE", "S"),
             (put_sell_strike, "PE", "S"),
             (put_buy_strike, "PE", "B")
         ]

    elif strategy["Strategy"] == "Calendar Spread":
         # Sell Near Month, Buy Far Month (same strike)
         # This requires fetching two different expiry option chains - NOT CURRENTLY IMPLEMENTED IN fetch_real_time_market_data
         # For simplicity in preparing orders, we will simulate this using the same expiry but note the limitation.
         # A real calendar spread needs the next expiry data.
         logger.warning("Calendar Spread requires fetching next expiry data, which is not fully supported in current fetch_real_time_market_data.")
         st.warning("Calendar Spread order preparation is a simplified example as next expiry data is not fetched.")

         # Find the fetched expiry timestamp
         fetched_expiry_ts = parse_5paisa_date_string(option_chain["ExpiryDate"].iloc[0])
         if fetched_expiry_ts is None:
              logger.error("Could not get expiry timestamp from option chain data.")
              st.error("Cannot prepare Calendar Spread: Expiry timestamp missing.")
              return None

         # To truly implement, you would need to call client.get_expiry again, find the *next* expiry,
         # then call client.get_option_chain for that next expiry.
         # For this example, we will just use the same expiry data as if it were the 'near' leg,
         # and mark the 'far' leg conceptually (won't have real ScripCode from the *actual* far expiry).
         # THIS IS A SIMPLIFICATION FOR DEMO PURPOSES.

         strategy_legs = [
             (atm_strike, "CE", "S", "Near"), # Sell Near Call @ ATM
             # (atm_strike, "CE", "B", "Far")  # Buy Far Call @ ATM - Requires next expiry data
         ]
         # Since we can't get the Far Leg ScripCode accurately with the current data,
         # we will just prepare the Short leg as a placeholder.
         # A full implementation needs modification of fetch_real_time_market_data or a new fetch function.


    elif strategy["Strategy"] == "Short Put Vertical Spread":
         # Sell OTM Put, Buy further OTM Put
         strikes_sorted = option_chain["StrikeRate"].sort_values().tolist()
         put_sell_strike = next((s for s in reversed(strikes_sorted) if s <= atm_strike - 50), None) # Sell slightly OTM
         put_buy_strike = next((s for s in reversed(strikes_sorted) if s < put_sell_strike - 100 if put_sell_strike is not None), None) # Buy further OTM

         if put_sell_strike is None or put_buy_strike is None:
              logger.error("Could not find suitable strikes for Short Put Vertical Spread.")
              st.error("Cannot prepare Short Put Vertical Spread: Suitable strikes not found.")
              return None

         strategy_legs = [
             (put_sell_strike, "PE", "S"),
             (put_buy_strike, "PE", "B")
         ]

    elif strategy["Strategy"] == "Short Call Vertical Spread":
         # Sell OTM Call, Buy further OTM Call
         strikes_sorted = option_chain["StrikeRate"].sort_values().tolist()
         call_sell_strike = next((s for s in strikes_sorted if s >= atm_strike + 50), None) # Sell slightly OTM
         call_buy_strike = next((s for s in strikes_sorted if s > call_sell_strike + 100 if call_sell_strike is not None), None) # Buy further OTM

         if call_sell_strike is None or call_buy_strike is None:
              logger.error("Could not find suitable strikes for Short Call Vertical Spread.")
              st.error("Cannot prepare Short Call Vertical Spread: Suitable strikes not found.")
              return None

         strategy_legs = [
             (call_sell_strike, "CE", "S"),
             (call_buy_strike, "CE", "B")
         ]

    elif strategy["Strategy"] == "Short Put":
         # Simple Short Put
         strikes_sorted = option_chain["StrikeRate"].sort_values().tolist()
         put_sell_strike = next((s for s in reversed(strikes_sorted) if s <= atm_strike - 50), None) # Sell OTM Put

         if put_sell_strike is None:
              logger.error("Could not find suitable strike for Short Put.")
              st.error("Cannot prepare Short Put: Suitable strike not found.")
              return None

         strategy_legs = [
             (put_sell_strike, "PE", "S")
         ]

    elif strategy["Strategy"] == "Long Put":
         # Simple Long Put
         strikes_sorted = option_chain["StrikeRate"].sort_values().tolist()
         put_buy_strike = next((s for s in reversed(strikes_sorted) if s <= atm_strike - 50), None) # Buy OTM Put

         if put_buy_strike is None:
              logger.error("Could not find suitable strike for Long Put.")
              st.error("Cannot prepare Long Put: Suitable strike not found.")
              return None

         strategy_legs = [
             (put_buy_strike, "PE", "B")
         ]


    else:
        logger.error(f"Unsupported strategy for order preparation: {strategy['Strategy']}")
        st.warning(f"Order preparation not supported for strategy: {strategy['Strategy']}")
        return None

    # Prepare the actual order dictionaries
    for leg in strategy_legs:
        strike, cp_type, buy_sell = leg[:3] # Handle potential extra elements like "Near"/"Far"
        quantity_multiplier = 2 if strategy["Strategy"] == "Butterfly Spread" and (strike, cp_type, buy_sell) == (atm_strike, "CE", "S") else 1 # Handle doubled leg in Butterfly

        opt_data = option_chain[
            (option_chain["StrikeRate"] == strike) &
            (option_chain["CPType"] == cp_type)
            # Ensure this is for the fetched expiry timestamp if multiple expiries were fetched
            # (Assuming option_chain only contains data for the first expiry now)
            # (option_chain["ExpiryDate"] == fetched_expiry_ts) # Need to filter by timestamp if fetching multiple expiries
        ]

        if opt_data.empty:
            logger.error(f"No option chain data found for {cp_type} at strike {strike} for expiry {expiry_date_str}. Skipping leg.")
            st.warning(f"Skipping order leg: No data for {cp_type} @ {strike}.")
            continue # Skip this leg if data isn't found

        # Assuming the first match is correct if duplicates exist (shouldn't for unique strike/type/expiry)
        scrip_code = int(opt_data["ScripCode"].iloc[0])
        # Use LastRate as a proxy for current market price
        latest_price = float(opt_data["LastRate"].iloc[0]) if not pd.isna(opt_data["LastRate"].iloc[0]) else 0.0
        # Could also fetch live market depth here for Bids/Offers for better price estimation if needed

        # Determine default price for Market or Limit order display
        # For simplicity, propose a Market Order (Price=0) for now, user confirms
        # Could allow user to change to Limit Price in the UI confirmation step if implemented
        proposed_price = 0 # 0 for Market Order

        orders_to_place.append({
            "Strategy": strategy["Strategy"],
            "Leg_Type": f"{buy_sell} {cp_type}",
            "Strike": strike,
            "Expiry": expiry_date_str, # Use the formatted date string
            "Exchange": "N",
            "ExchangeType": "D", # Derivatives
            "ScripCode": scrip_code,
            "Quantity_Lots": lots * quantity_multiplier, # Show lots for clarity
            "Quantity_Units": lots * quantity_multiplier * lot_size, # Show units for API
            "Proposed_Price": proposed_price, # Proposed price (0 for Market)
            "Last_Price_API": latest_price # Show the last traded price from API
        })

    if not orders_to_place:
         logger.error("No valid order legs were prepared.")
         st.error("Order preparation failed: No valid order legs found.")
         return None

    logger.info(f"Successfully prepared {len(orders_to_place)} order legs.")
    return orders_to_place


def execute_trade_orders(client, prepared_orders):
    """
    Executes a list of prepared orders using the 5paisa API.
    """
    logger.info(f"Attempting to execute {len(prepared_orders)} trade orders.")
    if client is None or not client.get_access_token():
        logger.error("5paisa client not available or not logged in for order execution.")
        st.error("Cannot place orders: 5paisa client not logged in.")
        return False

    if not prepared_orders:
        logger.warning("No prepared orders to execute.")
        st.warning("No orders to place.")
        return False

    all_successful = True
    responses = []

    for order in prepared_orders:
        try:
            logger.info(f"Placing order leg: {order['Leg_Type']} {order['Quantity_Units']} units of ScripCode {order['ScripCode']}")
            # Note: IsIntraday should potentially come from strategy logic or user choice
            response = client.place_order(
                OrderType=order["Leg_Type"].split(" ")[0].upper(), # 'BUY' or 'SELL'
                Exchange=order["Exchange"],
                ExchangeType=order["ExchangeType"],
                ScripCode=order["ScripCode"],
                Qty=order["Quantity_Units"], # Use quantity in units for API
                Price=order["Proposed_Price"], # Use the proposed price (likely 0 for Market)
                IsIntraday=False # Assuming positional for strategies unless specified otherwise
                # Can add validation/logic for other parameters like StopLossPrice here if needed
            )
            responses.append({"Order": order, "Response": response})
            logger.debug(f"Place order response for {order['ScripCode']}: {response}")

            if response.get("Status") != 0:
                all_successful = False
                error_message = response.get('Message', 'Unknown error')
                st.error(f"❌ Order failed for {order['Leg_Type']} {order['Strike']} {order['Expiry']} (ScripCode {order['ScripCode']}): {error_message}")
                logger.error(f"Order failed for {order['ScripCode']}: {error_message}")
            else:
                 st.success(f"✅ Order placed successfully for {order['Leg_Type']} {order['Strike']} {order['Expiry']} (ScripCode {order['ScripCode']}). Order ID: {response.get('ClientOrderID', 'N/A')}")
                 logger.info(f"Order placed successfully for {order['ScripCode']}. Response: {response}")

        except Exception as e:
            all_successful = False
            st.error(f"An unexpected error occurred placing order for {order['ScripCode']}: {str(e)}")
            logger.error(f"Unexpected error during order placement for {order['ScripCode']}: {str(e)}", exc_info=True)
            responses.append({"Order": order, "Response": {"Status": -1, "Message": f"Exception: {e}"}})

    if all_successful:
        st.balloons() # Celebrate if all orders were sent successfully (not necessarily filled)
        st.success("🚀 All requested orders have been sent to the broker.")
    else:
         st.warning("Some orders failed or encountered errors.")

    return all_successful # Indicate if the execution attempt was fully successful


def square_off_positions(client):
    """
    Calls the squareoff_all API endpoint.
    """
    try:
        logger.info("Attempting to square off all positions via API...")
        if client is None or not client.get_access_token():
            logger.error("5paisa client not available or not logged in for square off.")
            st.error("Cannot square off: 5paisa client not logged in.")
            return False

        # Add confirmation step before calling the API
        confirm_square_off = st.sidebar.button("Confirm Square Off ALL Positions", key="confirm_square_off")
        if confirm_square_off:
             st.sidebar.warning("Attempting to square off all positions...")
             response = client.squareoff_all()
             logger.debug(f"Square off all response: {response}")

             if response.get("Status") == 0:
                 st.sidebar.success("✅ Request to square off all positions sent successfully.")
                 logger.info("Square off all positions request sent successfully.")
                 return True
             else:
                 message = response.get("Message", "Unknown error")
                 st.sidebar.error(f"❌ Failed to square off positions: {message}")
                 logger.error(f"Square off all failed: {message}")
                 return False
        else:
             st.sidebar.warning("Click 'Confirm Square Off ALL Positions' to proceed.")
             return False # Indicate not yet confirmed/executed


    except Exception as e:
        st.sidebar.error(f"Error squaring off positions: {str(e)}")
        logger.error(f"Error squaring off positions: {str(e)}", exc_info=True)
        return False


# Sidebar Login and Controls
with st.sidebar:
    st.header("🔑 5paisa Login") # Corrected emoji
    totp_code = st.text_input("TOTP (from Authenticator App)", type="password")
    # Add input fields for credentials if not using st.secrets (less secure)
    # Or guide user to use st.secrets
    # app_name = st.text_input("APP_NAME")
    # ...

    if st.button("Login to 5paisa"): # Added more descriptive button text
        st.session_state.client = initialize_5paisa_client(totp_code)
        # Login status message handled inside initialize_5paisa_client

    if st.session_state.logged_in: # Check session state flag
        st.header("⚙️ Trading Controls") # Corrected emoji
        capital = st.number_input("Capital (₹)", min_value=100000, value=1000000, step=100000, format="%d") # Corrected currency, format
        risk_tolerance = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"], index=1)
        forecast_horizon = st.slider("Forecast Horizon (days)", 1, 30, 7)
        # dte_preference = st.slider("DTE Preference (days)", 7, 30, 15) # Not directly used in logic currently
        st.markdown("---") # Separator
        st.markdown("**Backtest Parameters**")
        # Set sensible defaults for date inputs
        default_start_date = datetime.now().date() - timedelta(days=365)
        default_end_date = datetime.now().date() + timedelta(days=0) # Today
        start_date = st.date_input("Start Date", value=default_start_date)
        end_date = st.date_input("End Date", value=default_end_date)
        strategy_choice = st.selectbox("Backtest Strategy Filter", ["All Strategies", "Butterfly Spread", "Iron Condor", "Iron Fly", "Short Strangle", "Calendar Spread", "Jade Lizard", "Short Straddle", "Short Put Vertical Spread", "Short Call Vertical Spread", "Short Put", "Long Put"]) # Added new strategies
        st.markdown("---") # Separator
        run_button = st.button("📊 Run Analysis") # Corrected emoji and text

        st.markdown("---") # Separator
        st.header("💥 Emergency Actions") # Corrected emoji
        st.warning("Use with EXTREME CAUTION!")
        if st.button("🚨 Square Off All Positions"): # Corrected emoji
             # Trigger square off logic
             square_off_positions(st.session_state.client)
             # Note: The confirmation step is now handled inside square_off_positions


# Main Execution Area
if not st.session_state.logged_in:
    st.info("Please login to 5paisa from the sidebar to proceed. You need a secrets.toml file with your API credentials.") # Added info about secrets.toml
else:
    st.markdown("<h1 style='color: #e94560; text-align: center;'>🛡️ VolGuard Pro: Your AI Trading Copilot</h1>", unsafe_allow_html=True) # Corrected emoji

    # Check if analysis has been run
    if run_button:
        with st.spinner("Running VolGuard Analysis... Fetching data and generating insights."):
            # Clear previous run results
            st.session_state.backtest_run = False
            st.session_state.backtest_results = None
            st.session_state.violations = 0
            st.session_state.journal_complete = False
            st.session_state.prepared_orders = None # Clear prepared orders from previous run

            # Load Data (API first)
            df, real_data, data_source = load_data(st.session_state.client)
            st.session_state.real_time_market_data = real_data # Store real_data in session state

            # Fetch all API portfolio data if data loading was successful (meaning client was available)
            if data_source != "Data Load Failed" and st.session_state.client:
                 st.session_state.api_portfolio_data = fetch_all_api_portfolio_data(st.session_state.client)
            else:
                 st.session_state.api_portfolio_data = {} # Ensure it's empty if no client/data

            if df is not None:
                # Generate Features
                df = generate_features(df, st.session_state.real_time_market_data, capital) # Pass real_data to feature generation

                if df is not None:
                     # Store df in session state if needed for other tabs without re-running
                     st.session_state.analysis_df = df

                     # Run Backtest (using the loaded and featured df)
                     backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf = run_backtest(
                        df, capital, strategy_choice, start_date, end_date
                     )

                     # Store backtest results
                     st.session_state.backtest_run = True
                     st.session_state.backtest_results = {
                         "backtest_df": backtest_df,
                         "total_pnl": total_pnl,
                         "win_rate": win_rate,
                         "max_drawdown": max_drawdown,
                         "sharpe_ratio": sharpe_ratio,
                         "sortino_ratio": sortino_ratio,
                         "calmar_ratio": calmar_ratio,
                         "strategy_perf": strategy_perf,
                         "regime_perf": regime_perf
                     }

                     # Volatility Forecasting (uses df features)
                     with st.spinner("Predicting market volatility..."):
                         forecast_log, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, feature_importances = forecast_volatility_future(df, forecast_horizon)
                         st.session_state.forecast_log = forecast_log # Store forecast log
                         st.session_state.forecast_metrics = { # Store other forecast results
                             "garch_vols": garch_vols, "xgb_vols": xgb_vols, "blended_vols": blended_vols,
                             "realized_vol": realized_vol, "confidence_score": confidence_score,
                             "rmse": rmse, "feature_importances": feature_importances
                         }

                     # Generate Trading Strategy (uses df, forecast_log, etc.)
                     st.session_state.generated_strategy = generate_trading_strategy(
                         df, st.session_state.forecast_log, st.session_state.forecast_metrics["realized_vol"], risk_tolerance, st.session_state.forecast_metrics["confidence_score"], capital
                     )

            else:
                st.error("Analysis could not be completed due to data loading failure.")


    # Define tabs
    tabs = st.tabs(["📈 Snapshot", "🔮 Forecast", "🧠 Strategy", "💼 Portfolio", "📝 Journal", "🔬 Backtest"]) # Corrected emojis


    # --- Snapshot Tab ---
    with tabs[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Market Snapshot") # Corrected emoji
        if st.session_state.analysis_df is not None and not st.session_state.analysis_df.empty:
            df = st.session_state.analysis_df
            latest_date = df.index[-1].strftime("%d-%b-%Y")
            last_nifty = df["NIFTY_Close"].iloc[-1]
            prev_nifty = df["NIFTY_Close"].iloc[-2] if len(df) >= 2 else last_nifty
            last_vix = df["VIX"].iloc[-1]

            # Determine regime based on latest VIX value from the df
            regime = "LOW" if last_vix < 15 else "MEDIUM" if last_vix < 20 else "HIGH"
            # Use generated strategy regime if available and not locked
            if 'generated_strategy' in st.session_state and st.session_state.generated_strategy is not None:
                 regime = st.session_state.generated_strategy["Regime"]

            regime_class = {
                 "LOW": "regime-low",
                 "MEDIUM": "regime-medium",
                 "HIGH": "regime-high",
                 "EVENT-DRIVEN": "regime-event"
            }.get(regime, "regime-low") # Default to low if regime is unexpected


            st.markdown(f'<div style="text-align: center;"><span class="regime-badge {regime_class}">{regime} Market Regime</span></div>', unsafe_allow_html=True) # Display regime badge
            st.markdown('<div class="gauge" style="margin: 20px auto;">Gauge Here</div>', unsafe_allow_html=True) # Placeholder for gauge if desired

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("NIFTY 50", f"{last_nifty:,.2f}", f"{(last_nifty - prev_nifty)/prev_nifty*100:+.2f}%")
            with col2:
                st.metric("India VIX", f"{last_vix:.2f}%", f"{df['VIX_Change_Pct'].iloc[-1]:+.2f}%" if 'VIX_Change_Pct' in df.columns else "N/A")
            with col3:
                st.metric("PCR", f"{df['PCR'].iloc[-1]:.2f}" if 'PCR' in df.columns else "N/A")
            with col4:
                st.metric("Straddle Price", f"₹{df['Straddle_Price'].iloc[-1]:,.2f}" if 'Straddle_Price' in df.columns else "N/A")

            # Display data source
            source_tag = "Data Load Failed"
            if 'data_source' in locals() and data_source: # Check if data_source was set by load_data
                 source_tag = data_source
            st.markdown(f"**Last Updated**: {latest_date} | **Source**: {source_tag}")

            # Display raw real-time data fetched if available
            if st.session_state.real_time_market_data and st.session_state.real_time_market_data["source"] == "5paisa API (LIVE)":
                 with st.expander("Raw 5paisa API Data"):
                      st.json(st.session_state.real_time_market_data) # Display the raw fetched data
            elif source_tag == "CSV (FALLBACK)":
                 st.info("Data loaded from CSV fallback. Real-time API data could not be fetched.")


        else:
            st.info("Run the analysis to see the market snapshot.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Forecast Tab ---
    with tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔮 Volatility Forecast") # Corrected emoji
        if 'forecast_log' in st.session_state and st.session_state.forecast_log is not None:
             forecast_log = st.session_state.forecast_log
             forecast_metrics = st.session_state.forecast_metrics

             col1, col2, col3 = st.columns(3)
             with col1:
                 st.metric("Avg Blended Volatility", f"{np.mean(forecast_metrics['blended_vols']):.2f}%")
             with col2:
                 st.metric("Realized Volatility (5-Day Avg)", f"{forecast_metrics['realized_vol']:.2f}%")
             with col3:
                 st.metric("Model RMSE", f"{forecast_metrics['rmse']:.2f}%")
                 st.markdown(f'<div class="gauge" style="margin: 0 auto;">{int(forecast_metrics["confidence_score"])}%</div><div style="text-align: center;">Confidence</div>', unsafe_allow_html=True)
             st.line_chart(pd.DataFrame({
                 "GARCH": forecast_metrics["garch_vols"],
                 "XGBoost": forecast_metrics["xgb_vols"],
                 "Blended": forecast_metrics["blended_vols"]
             }, index=forecast_log["Date"]), color=["#e94560", "#00d4ff", "#ffcc00"])
             st.markdown("### Feature Importance")
             feature_importance = pd.DataFrame({
                 'Feature': feature_cols, # Use the defined feature_cols
                 'Importance': forecast_metrics["feature_importances"]
             }).sort_values(by='Importance', ascending=False)
             st.dataframe(feature_importance, use_container_width=True)
        else:
             st.info("Run the analysis to see the volatility forecast.")
        st.markdown('</div>', unsafe_allow_html=True)


    # --- Strategy Tab ---
    with tabs[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧠 Trading Strategy") # Corrected emoji

        if st.session_state.violations >= 2 and not st.session_state.journal_complete:
            st.markdown('<div class="alert-banner">🚨 Discipline Lock: Complete Journaling to Unlock Trading</div>', unsafe_allow_html=True) # Corrected emoji
        elif 'generated_strategy' in st.session_state and st.session_state.generated_strategy is not None:
            strategy = st.session_state.generated_strategy
            real_data = st.session_state.real_time_market_data # Get real_data from session state

            regime_class = {
                "LOW": "regime-low", "MEDIUM": "regime-medium", "HIGH": "regime-high", "EVENT-DRIVEN": "regime-event"
            }.get(strategy["Regime"], "regime-low")

            st.markdown('<div class="strategy-carousel">', unsafe_allow_html=True)
            st.markdown(f"""
                <div class="strategy-card">
                    <h4>{strategy["Strategy"]}</h4>
                    <span class="regime-badge {regime_class}">{strategy["Regime"]} Regime</span>
                    <p><b>Reason:</b> {strategy["Reason"]}</p>
                    <p><b>Confidence:</b> {strategy["Confidence"]:.2f}</p>
                    <p><b>Risk-Reward:</b> {strategy["Risk_Reward"]:.2f}:1</p>
                    <p><b>Capital Deploy:</b> ₹{strategy["Deploy"]:,.0f}</p>
                    <p><b>Max Loss:</b> ₹{strategy["Max_Loss"]:,.0f}</p>
                    <p><b>Exposure:</b> {strategy["Exposure"]:.2f}%</p>
                    <p><b>Tags:</b> {', '.join(strategy["Tags"])}</p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if strategy["Risk_Flags"]:
                st.markdown(f'<div class="alert-banner">⚠️ Risk Flags: {", ".join(strategy["Risk_Flags"])}</div>', unsafe_allow_html=True) # Corrected emoji

            if strategy["Behavior_Warnings"]:
                 for warning in strategy["Behavior_Warnings"]:
                      st.warning(f"⚠️ Behavioral Warning: {warning}") # Corrected emoji

            st.markdown("---") # Separator
            st.subheader("Ready to Trade?")

            # --- Prepare Orders and Show Confirmation UI ---
            if st.button("📝 Prepare Orders for this Strategy"): # Corrected emoji
                 st.session_state.prepared_orders = prepare_trade_orders(strategy, real_data, capital) # Store prepared orders

            if st.session_state.prepared_orders:
                 st.markdown("### Proposed Order Details:")
                 st.warning("REVIEW THESE ORDERS CAREFULLY BEFORE PLACING!")

                 # Display prepared orders in a table
                 orders_df = pd.DataFrame(st.session_state.prepared_orders)
                 # Drop columns not needed for user review if desired
                 orders_display_cols = ['Leg_Type', 'Strike', 'Expiry', 'Quantity_Lots', 'Quantity_Units', 'Proposed_Price', 'Last_Price_API', 'ScripCode']
                 st.dataframe(orders_df[orders_display_cols], use_container_width=True)

                 st.markdown("---") # Separator

                 # Add confirmation button
                 if st.button("✅ Confirm and Place Orders"): # Corrected emoji
                     # Execute the prepared orders
                     execute_trade_orders(st.session_state.client, st.session_state.prepared_orders)
                     # Optionally clear prepared orders after placing
                     # st.session_state.prepared_orders = None
                 else:
                      st.info("Click 'Confirm and Place Orders' to send the orders to the broker.")

            elif 'generated_strategy' in st.session_state and st.session_state.generated_strategy is not None:
                 st.info("Click 'Prepare Orders for this Strategy' to see the order details before trading.")


        else:
            st.info("Run the analysis to generate a trading strategy.")
        st.markdown('</div>', unsafe_allow_html=True)


    # --- Portfolio Tab ---
    with tabs[3]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("💼 Portfolio Overview") # Corrected emoji

        # Fetch and display portfolio summary metrics (using the function that gets positions)
        portfolio_summary = fetch_portfolio_data(st.session_state.client, capital) # Call this function every time tab is active

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current P&L (Today/Holding)", f"₹{portfolio_summary['weekly_pnl']:,.2f}") # Label reflects data source
        with col2:
            st.metric("Margin Used", f"₹{portfolio_summary['margin_used']:,.2f}")
        with col3:
            st.metric("Exposure", f"{portfolio_summary['exposure']:.2f}%")

        st.markdown("---") # Separator

        # Display detailed portfolio data fetched from API if available
        if st.session_state.api_portfolio_data:
            st.subheader("Comprehensive Account Data (from 5paisa API)")

            # Holdings
            with st.expander("📈 Holdings"): # Corrected emoji
                 if st.session_state.api_portfolio_data.get("holdings"):
                      holdings_df = pd.DataFrame(st.session_state.api_portfolio_data["holdings"])
                      st.dataframe(holdings_df, use_container_width=True)
                 else:
                      st.info("No holdings found or could not fetch.")

            # Margin
            with st.expander("💰 Margin Details"): # Corrected emoji
                 if st.session_state.api_portfolio_data.get("margin"):
                      # Margin data is a dictionary, display keys and values
                      margin_data = st.session_state.api_portfolio_data["margin"]
                      for key, value in margin_data.items():
                           st.write(f"**{key}**: {value}")
                 else:
                      st.info("No margin data found or could not fetch.")

            # Positions
            with st.expander("📊 Open Positions"): # Corrected emoji
                 if st.session_state.api_portfolio_data.get("positions"):
                      positions_df = pd.DataFrame(st.session_state.api_portfolio_data["positions"])
                      st.dataframe(positions_df, use_container_width=True)
                 else:
                      st.info("No open positions found or could not fetch.")

            # Order Book (Open Orders)
            with st.expander("📋 Order Book"): # Corrected emoji
                 if st.session_state.api_portfolio_data.get("order_book"):
                      order_book_df = pd.DataFrame(st.session_state.api_portfolio_data["order_book"])
                      st.dataframe(order_book_df, use_container_width=True)
                 else:
                      st.info("No open orders found or could not fetch.")

            # Trade Book (Executed Trades)
            with st.expander("📜 Trade Book"): # Corrected emoji
                 if st.session_state.api_portfolio_data.get("trade_book"):
                      trade_book_df = pd.DataFrame(st.session_state.api_portfolio_data["trade_book"])
                      st.dataframe(trade_book_df, use_container_width=True)
                 else:
                      st.info("No executed trades found or could not fetch.")

            # Market Status
            with st.expander("📰 Market Status"): # Corrected emoji
                 if st.session_state.api_portfolio_data.get("market_status"):
                      st.json(st.session_state.api_portfolio_data["market_status"])
                 else:
                      st.info("Market status not available or could not fetch.")


        else:
             st.info("Connect to 5paisa and run analysis to fetch detailed portfolio data.")

        st.markdown('</div>', unsafe_allow_html=True)


    # --- Journal Tab ---
    with tabs[4]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📝 Discipline Hub") # Corrected emoji
        with st.form(key="journal_form"):
            st.markdown("Reflect on your trading decisions and build discipline.")
            reason_strategy = st.selectbox("Why did you choose this strategy?", ["High IV", "Low Risk", "Event Opportunity", "Bullish Bias", "Bearish Bias", "Range Bound Expectation", "Expiry Play", "Other"]) # Added more options
            override_risk = st.radio("Did you override any system risk flags?", ("No", "Yes"), index=0) # Default to No
            expected_outcome = st.text_area("Describe your trade plan, entry/exit criteria, and expected outcome.")
            lessons_learned = st.text_area("After the trade, what were the lessons learned (optional, for review)?") # Added lessons learned

            submit_journal = st.form_submit_button("💾 Save Journal Entry") # Corrected emoji

            if submit_journal:
                score = 0
                # Basic scoring logic - can be enhanced
                if override_risk == "No":
                    score += 3
                if reason_strategy != "Other":
                    score += 2 # Slightly less score for specific reason
                if expected_outcome:
                    score += 3
                if lessons_learned:
                    score += 1 # Reward post-trade reflection
                # Add score based on recent trade outcome if applicable (need to track trade outcomes)
                # For simplicity, using a placeholder based on portfolio PnL direction
                if portfolio_summary['weekly_pnl'] > 0: # Use portfolio PnL as proxy
                    score += 1

                score = min(score, 10) # Cap score at 10


                journal_entry = {
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Strategy_Reason": reason_strategy,
                    "Override_Risk": override_risk,
                    "Expected_Outcome": expected_outcome,
                    "Lessons_Learned": lessons_learned,
                    "Discipline_Score": score
                }
                journal_df_entry = pd.DataFrame([journal_entry]) # Create DataFrame for the single entry

                journal_file = "journal_log.csv"
                try:
                    # Append to CSV
                    journal_df_entry.to_csv(journal_file, mode='a', header=not os.path.exists(journal_file), index=False)
                    st.success(f"✅ Journal Entry Saved! Discipline Score: {score}/10") # Corrected emoji
                    logger.info(f"Journal entry saved. Score: {score}")

                    # If score is high enough and there were violations, clear violations and unlock
                    if score >= 7 and st.session_state.violations > 0: # Adjusted threshold
                         st.session_state.violations = 0
                         st.session_state.journal_complete = True # Mark journaling complete
                         st.success("🔓 Discipline Lock Removed! Keep up the good work.") # Corrected emoji
                         logger.info("Discipline lock removed.")

                except PermissionError:
                    logger.error("Permission denied when writing to journal_log.csv")
                    st.error("❌ Cannot save journal_log.csv: Permission denied") # Corrected emoji
                except Exception as e:
                    logger.error(f"Error saving journal entry: {e}")
                    st.error(f"❌ Error saving journal entry: {e}") # Corrected emoji


        st.markdown("### Past Entries")
        journal_file = "journal_log.csv"
        if os.path.exists(journal_file):
            try:
                journal_df = pd.read_csv(journal_file)
                # Format date column for better display
                if 'Date' in journal_df.columns:
                    journal_df['Date'] = pd.to_datetime(journal_df['Date']).dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(journal_df, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error reading journal_log.csv: {e}") # Corrected emoji
        else:
            st.info("No journal entries found yet.")
        st.markdown('</div>', unsafe_allow_html=True)


    # --- Backtest Tab ---
    with tabs[5]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔬 Backtest Results") # Corrected emoji
        if st.session_state.backtest_run and st.session_state.backtest_results is not None:
            results = st.session_state.backtest_results
            if results["backtest_df"].empty:
                st.warning("No trades generated for the selected parameters. Try adjusting the date range or strategy.")
            else:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total P&L", f"₹{results['total_pnl']:,.2f}") # Corrected currency
                with col2:
                    st.metric("Win Rate", f"{results['win_rate']*100:.2f}%")
                with col3:
                    st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                with col4:
                    st.metric("Max Drawdown", f"₹{results['max_drawdown']:,.2f}") # Corrected currency

                st.markdown("### Cumulative P&L")
                # Ensure cumulative PnL column exists and is numeric
                if 'Cumulative_PnL' in results["backtest_df"].columns:
                     st.line_chart(results["backtest_df"]["Cumulative_PnL"], color="#e94560")
                else:
                     st.warning("Cumulative P&L data not available.")


                st.markdown("### Performance by Strategy")
                # Format the performance DataFrames
                strategy_perf_formatted = results["strategy_perf"].style.format({
                    "sum": "₹{:,.2f}", # Corrected currency
                    "mean": "₹{:,.2f}", # Corrected currency
                    "Win_Rate": "{:.2%}"
                })
                st.dataframe(strategy_perf_formatted, use_container_width=True)


                st.markdown("### Performance by Regime")
                regime_perf_formatted = results["regime_perf"].style.format({
                    "sum": "₹{:,.2f}", # Corrected currency
                    "mean": "₹{:,.2f}", # Corrected currency
                    "Win_Rate": "{:.2%}"
                })
                st.dataframe(regime_perf_formatted, use_container_width=True)

                st.markdown("### Detailed Backtest Trades")
                # Format the detailed trades DataFrame
                detailed_trades_formatted = results["backtest_df"].style.format({
                    "PnL": "₹{:,.2f}", # Corrected currency
                    "Cumulative_PnL": "₹{:,.2f}", # Corrected currency
                    "Capital_Deployed": "₹{:,.2f}", # Corrected currency
                    "Max_Loss": "₹{:,.2f}", # Corrected currency
                    "Risk_Reward": "{:.2f}"
                })
                st.dataframe(detailed_trades_formatted, use_container_width=True)


        else:
            st.info("Run the analysis to view backtest results.")
        st.markdown('</div>', unsafe_allow_html=True)


    st.markdown('<div class="footer">Built with ❤️ by Shritish Shukla & Salman Azim | © 2025 VolGuard</div>', unsafe_allow_html=True) # Corrected emoji ❤️

