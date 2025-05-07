import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import re # Import regex for date parsing
from datetime import datetime, timedelta
import logging
import os
from py5paisa import FivePaisaClient
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper function to parse the 5paisa date string format (from working Notebook) ---
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

# --- Helper function to format timestamp to readable date string (from working Notebook) ---
def format_timestamp_to_date_str(timestamp_ms):
     """
     Converts a timestamp in milliseconds to a readable YYYY-MM-DD string.
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


# Page config - Corrected Emojis
st.set_page_config(page_title="VolGuard Pro", page_icon="🛡️", layout="wide")

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
if "data_source" not in st.session_state:
     st.session_state.data_source = "None"
if "show_trade_confirmation" not in st.session_state:
     st.session_state.show_trade_confirmation = False
if "proposed_trade_details" not in st.session_state:
     st.session_state.proposed_trade_details = None


# 5paisa Client Initialization
def initialize_5paisa_client(totp_code):
    try:
        logger.info("Initializing 5paisa client")
        # Fetch credentials from Streamlit secrets
        cred = {
            "APP_NAME": st.secrets["fivepaisa"].get("APP_NAME"),
            "APP_SOURCE": st.secrets["fivepaisa"].get("APP_SOURCE"),
            "USER_ID": st.secrets["fivepaisa"].get("USER_ID"),
            "PASSWORD": st.secrets["fivepaisa"].get("PASSWORD"),
            "USER_KEY": st.secrets["fivepaisa"].get("USER_KEY"),
            "ENCRYPTION_KEY": st.secrets["fivepaisa"].get("ENCRYPTION_KEY")
        }
        client = FivePaisaClient(cred=cred)

        # Fetch Client Code and PIN from secrets
        client_code = st.secrets["fivepaisa"].get("CLIENT_CODE")
        pin = st.secrets["fivepaisa"].get("PIN")

        if not all([cred["APP_NAME"], cred["APP_SOURCE"], cred["USER_ID"], cred["PASSWORD"], cred["USER_KEY"], cred["ENCRYPTION_KEY"], client_code, pin, totp_code]):
             logger.error("Missing one or more 5paisa credentials in secrets or TOTP input.")
             st.error("❌ Missing credentials. Ensure all secrets are set and TOTP is entered.")
             return None

        logger.info(f"Attempting TOTP session for client {client_code}")
        client.get_totp_session(client_code, totp_code, pin)

        if client.get_access_token():
            logger.info("✅ 5paisa client initialized and logged in successfully")
            return client
        else:
            logger.error("❌ Failed to get access token after TOTP session.")
            st.error("❌ Login failed. Check TOTP, PIN, and other credentials.")
            return None
    except Exception as e:
        logger.error(f"❌ Error initializing 5paisa client: {str(e)}")
        st.error(f"❌ Login failed: {str(e)}")
        return None

# Data Fetching Functions
def max_pain(df: pd.DataFrame, nifty_spot: float):
    """
    Helper function to calculate the max pain strike (from working Notebook).
    """
    try:
        if df.empty or "StrikeRate" not in df.columns or "CPType" not in df.columns or "OpenInterest" not in df.columns:
            logger.warning("Option chain data is incomplete or empty for max pain calculation.")
            return None, None

        # Ensure columns are numeric - crucial for calculations
        try:
            df["StrikeRate"] = pd.to_numeric(df["StrikeRate"], errors='coerce')
            df["OpenInterest"] = pd.to_numeric(df["OpenInterest"], errors='coerce').fillna(0)
            # Drop rows where StrikeRate or OpenInterest could not be converted
            df = df.dropna(subset=["StrikeRate", "OpenInterest"]).copy()
        except Exception as e:
            logger.error(f"Error converting columns for max pain: {e}")
            return None, None

        if df.empty:
             logger.warning("Option chain DataFrame is empty after cleaning for max pain.")
             return None, None

        calls = df[df["CPType"] == "CE"].set_index("StrikeRate")["OpenInterest"]
        puts = df[df["CPType"] == "PE"].set_index("StrikeRate")["OpenInterest"]
        strikes = df["StrikeRate"].unique()
        strikes.sort()

        pain = []
        for K in strikes:
            total_loss = 0
            # Loop through all strike rates present in the original DataFrame to find corresponding OI
            for s in strikes:
                 # Loss for Call writers at strike s if expiry is at K
                 if s in calls.index:
                      total_loss += max(0, s - K) * calls.get(s, 0)
                 # Loss for Put writers at strike s if expiry is at K
                 if s in puts.index:
                      total_loss += max(0, K - s) * puts.get(s, 0)
            pain.append((K, total_loss))

        if not pain:
             logger.warning("No valid strikes to calculate max pain.")
             return None, None

        # Max pain strike is the strike with the minimum total loss
        max_pain_strike = min(pain, key=lambda x: x[1])[0]
        max_pain_diff_pct = abs(nifty_spot - max_pain_strike) / nifty_spot * 100 if nifty_spot != 0 else 0

        logger.debug(f"Max Pain calculated: Strike={max_pain_strike}, Diff%={max_pain_diff_pct:.2f}")
        return max_pain_strike, max_pain_diff_pct
    except Exception as e:
        logger.error(f"Error calculating max pain: {str(e)}")
        return None, None


def fetch_nifty_data_api(client: FivePaisaClient): # Renamed to avoid conflict and clarify it's the API fetcher
    """
    Fetches real-time NIFTY 50, India VIX, and Option Chain data from 5paisa API.
    Uses request formats and parsing logic from the working Notebook code.

    Args:
        client (FivePaisaClient): The initialized 5paisa client object.

    Returns:
        dict or None: A dictionary containing real-time data if successful, None otherwise.
    """
    if client is None or not client.get_access_token():
        logger.warning("5paisa client is not initialized or not logged in. Cannot fetch real-time data via API.")
        return None

    nifty_spot = None
    vix = None
    atm_strike = None
    straddle_price = 0
    pcr = 0
    max_pain_strike = None
    max_pain_diff_pct = None
    expiry_date_str = None
    df_option_chain = pd.DataFrame()
    expiry_timestamp = None # Initialize expiry timestamp


    try:
        logger.info("Fetching real-time market data from 5paisa API...")

        # 1. Fetch NIFTY 50 and India VIX using the detailed market feed request (from working Notebook)
        req_list_market_feed = [{
            "Exch": "N",
            "ExchType": "C", # Cash segment
            "ScripCode": 999920000, # NIFTY 50 ScripCode
            "Symbol": "NIFTY",
            "Expiry": "",
            "StrikePrice": "0",
            "OptionType": ""
        },
        {
            "Exch": "N",
            "ExchType": "C", # VIX is also in cash segment
            "ScripCode": 999920005, # India VIX ScripCode
            "Symbol": "INDIAVIX",
            "Expiry": "",
            "StrikePrice": "0",
            "OptionType": ""
        }]
        logger.debug(f"Fetching market feed for NIFTY and VIX: {req_list_market_feed}")
        market_feed_resp = client.fetch_market_feed(req_list_market_feed)
        logger.debug(f"Market feed response type: {type(market_feed_resp)}, value: {market_feed_resp}")

        # Process market feed response
        if not market_feed_resp or not isinstance(market_feed_resp, dict) or "Data" not in market_feed_resp or not isinstance(market_feed_resp["Data"], list) or len(market_feed_resp["Data"]) < 2:
            logger.error(f"Failed to fetch market feed for NIFTY/VIX or unexpected format. Response: {market_feed_resp}")
            # Continue, but Nifty/Vix will be None

        else:
             # Assuming the order in req_list is maintained in the response Data list
             nifty_data = market_feed_resp["Data"][0]
             vix_data = market_feed_resp["Data"][1]

             nifty_spot = nifty_data.get("LastRate", nifty_data.get("LastTradedPrice", 0))
             if not nifty_spot or nifty_spot == 0:
                 logger.warning("NIFTY price not found or is zero in market feed data.")
                 nifty_spot = None # Ensure None if invalid

             # Try common keys for VIX price
             vix = vix_data.get("LTP", vix_data.get("LastRate", vix_data.get("LastTradedPrice", 0)))
             if not vix or vix == 0:
                 logger.warning("VIX price not found or is zero in market feed data.")
                 vix = None # Ensure None if invalid

             if nifty_spot: logger.info(f"Fetched NIFTY Spot: {nifty_spot}")
             if vix: logger.info(f"Fetched VIX: {vix}")


        # 2. Fetch NIFTY expiries (Dynamically) - Using the working Notebook's logic
        logger.debug("Fetching NIFTY expiries")
        expiries_resp = client.get_expiry("N", "NIFTY")
        logger.debug(f"Expiries response type: {type(expiries_resp)}, value: {expiries_resp}")

        # Check expiries response: Look for the "Expiry" key and ensure it's a non-empty list
        if not expiries_resp or not isinstance(expiries_resp, dict) or "Expiry" not in expiries_resp or not isinstance(expiries_resp["Expiry"], list) or not expiries_resp["Expiry"]:
            logger.error(f"Failed to fetch NIFTY expiries or unexpected format. Response: {expiries_resp}")
            # expiry_date_str and expiry_timestamp remain None

        else: # If expiries response is valid
            first_expiry = expiries_resp["Expiry"][0] # Access using the correct key "Expiry"
            expiry_date_string_from_api = first_expiry.get("ExpiryDate") # Get the date string

            if not expiry_date_string_from_api:
                 logger.error("Expiry data missing ExpiryDate in the first expiry item.")
                 # expiry_date_str and expiry_timestamp remain None
            else:
                 # Parse the timestamp from the /Date(...) string using the helper from the working Notebook
                 expiry_timestamp = parse_5paisa_date_string(expiry_date_string_from_api)

                 if expiry_timestamp is not None:
                      # Format the timestamp to a readable date string for display
                      expiry_date_str = format_timestamp_to_date_str(expiry_timestamp)
                      logger.info(f"Fetched first expiry: {expiry_date_str} (Timestamp: {expiry_timestamp})")
                 else:
                      logger.error(f"Could not parse timestamp from ExpiryDate string: {expiry_date_string_from_api}")
                      # expiry_date_str and expiry_timestamp remain None


        # 3. Fetch Option Chain for the first expiry (using the parsed timestamp)
        if expiry_timestamp is not None: # Ensure we have a valid timestamp from step 2
            logger.debug(f"Fetching Option Chain for expiry timestamp: {expiry_timestamp}")
            option_chain_resp = client.get_option_chain("N", "NIFTY", expiry_timestamp)
            logger.debug(f"Option chain response type: {type(option_chain_resp)}, value: {option_chain_resp}")

            # Check Option Chain response
            if not option_chain_resp or not isinstance(option_chain_resp, dict) or "Options" not in option_chain_resp or not isinstance(option_chain_resp["Options"], list) or not option_chain_resp["Options"]:
                logger.error(f"Failed to fetch NIFTY option chain or unexpected format. Response: {option_chain_resp}")
                # df_option_chain remains empty DataFrame as initialized
            else:
                 df_option_chain = pd.DataFrame(option_chain_resp["Options"])
                 logger.debug(f"Option chain DataFrame created with shape: {df_option_chain.shape}")

                 # Ensure required columns are present and data is clean
                 required_cols_oc = ["StrikeRate", "CPType", "LastRate", "OpenInterest", "ScripCode"]
                 if not all(col in df_option_chain.columns for col in required_cols_oc):
                     missing = [col for col in required_cols_oc if col not in df_option_chain.columns]
                     logger.error(f"Required columns missing in option chain data DataFrame: {missing}")
                      # df_option_chain might be incomplete but still returned

                 # Clean and convert relevant columns
                 df_option_chain["StrikeRate"] = pd.to_numeric(df_option_chain["StrikeRate"], errors='coerce')
                 df_option_chain["LastRate"] = pd.to_numeric(df_option_chain["LastRate"], errors='coerce').fillna(0) # Fill missing prices with 0
                 df_option_chain["OpenInterest"] = pd.to_numeric(df_option_chain["OpenInterest"], errors='coerce').fillna(0)
                 df_option_chain["ScripCode"] = pd.to_numeric(df_option_chain["ScripCode"], errors='coerce').fillna(0).astype(int)

                 # Drop rows with invalid essential data after conversion
                 df_option_chain = df_option_chain.dropna(subset=["StrikeRate", "CPType", "ScripCode"]).copy()

                 if df_option_chain.empty:
                     logger.error("Option chain DataFrame is empty after cleaning missing essential data.")


        else: # If expiry_timestamp was None (because fetching/parsing failed)
            logger.warning("Cannot fetch Option Chain: Valid expiry timestamp was not obtained from API.")


        # 4. Calculate ATM Strike, Straddle Price, PCR, Max Pain (Only if Nifty spot and valid Option Chain data are available)
        if nifty_spot is not None and nifty_spot > 0 and not df_option_chain.empty and "StrikeRate" in df_option_chain.columns and pd.api.types.is_numeric_dtype(df_option_chain["StrikeRate"]):
             logger.debug(f"Calculating ATM, Straddle, PCR, Max Pain for NIFTY spot: {nifty_spot}")
             try:
                 # Calculate ATM Strike
                 atm_strike = df_option_chain["StrikeRate"].iloc[(df_option_chain["StrikeRate"] - nifty_spot).abs().argmin()]
                 atm_data = df_option_chain[df_option_chain["StrikeRate"] == atm_strike]

                 # Calculate Straddle Price (needs LastRate)
                 if 'LastRate' in atm_data.columns and pd.api.types.is_numeric_dtype(atm_data["LastRate"]):
                     atm_call_data = atm_data[atm_data["CPType"] == "CE"]
                     atm_call = atm_call_data["LastRate"].iloc[0] if not atm_call_data.empty else 0
                     atm_put_data = atm_data[atm_data["CPType"] == "PE"]
                     atm_put = atm_put_data["LastRate"].iloc[0] if not atm_put_data.empty else 0
                     straddle_price = (atm_call + atm_put) if atm_call is not None and atm_put is not None else 0
                 else:
                      logger.warning("LastRate column missing or not numeric for straddle calculation.")


                 # Calculate PCR (needs OpenInterest)
                 if 'OpenInterest' in df_option_chain.columns and pd.api.types.is_numeric_dtype(df_option_chain["OpenInterest"]):
                     calls_oi_sum = df_option_chain[df_option_chain["CPType"] == "CE"]["OpenInterest"].sum()
                     puts_oi_sum = df_option_chain[df_option_chain["CPType"] == "PE"]["OpenInterest"].sum()
                     pcr = puts_oi_sum / calls_oi_sum if calls_oi_sum > 0 else (float("inf") if puts_oi_sum > 0 else 0)
                 else:
                     logger.warning("OpenInterest column missing or not numeric for PCR calculation.")

                 # Calculate Max Pain (needs StrikeRate, CPType, OpenInterest)
                 max_pain_strike, max_pain_diff_pct = max_pain(df_option_chain, nifty_spot)

                 logger.debug(f"Calculated ATM Strike: {atm_strike}, Straddle: {straddle_price}, PCR: {pcr}, Max Pain: {max_pain_strike}")
             except Exception as e:
                 logger.error(f"Error during derivatives metrics calculation: {e}", exc_info=True)
                 # Metrics remain their default initialized values (None/0/inf)

        else:
             if nifty_spot is None or nifty_spot <= 0:
                  logger.warning("NIFTY spot price not available or invalid for calculating derivatives metrics.")
             if df_option_chain.empty:
                  logger.warning("Option chain is empty or invalid for calculating derivatives metrics.")


        logger.info("Real-time data fetching via API completed.")

        # Return data dictionary - include expiry date string and option chain dataframe
        return {
            "nifty_spot": nifty_spot,
            "vix": vix,
            "atm_strike": atm_strike,
            "straddle_price": straddle_price,
            "pcr": pcr,
            "max_pain_strike": max_pain_strike,
            "max_pain_diff_pct": max_pain_diff_pct,
            "expiry_date_str": expiry_date_str, # Readable expiry date string
            "option_chain": df_option_chain, # DataFrame
            "source": "5paisa API (LIVE)" # Indicate source
        }

    except Exception as e:
        logger.error(f"An unexpected error occurred during 5paisa API data fetch: {str(e)}", exc_info=True)
        return None


# @st.cache_data # Do not cache this function as it fetches potentially fresh data
def load_data(client):
    """
    Loads data, attempting API first, then falling back to CSV.
    Returns combined DataFrame, real-time data dict (if API was successful), and source string.
    """
    st.session_state.real_time_market_data = fetch_nifty_data_api(client) # Attempt API fetch

    if st.session_state.real_time_market_data is None or st.session_state.real_time_market_data.get("nifty_spot") is None:
        logger.warning("API data fetch failed or incomplete. Falling back to GitHub CSV.")
        st.session_state.data_source = "CSV (FALLBACK)"
        st.session_state.real_time_market_data = None # Ensure it's None if API failed significantly

        try:
            logger.info("Attempting to load data from GitHub CSV.")
            # Fetch NIFTY data from GitHub
            nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
            response_nifty = requests.get(nifty_url)
            response_nifty.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            nifty = pd.read_csv(io.StringIO(response_nifty.text), encoding="utf-8-sig")
            nifty.columns = nifty.columns.str.strip()
            nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
            nifty = nifty.dropna(subset=["Date"])
            nifty["Date"] = nifty["Date"].dt.normalize()
            nifty = nifty.groupby("Date").last().reset_index() # Handle duplicates
            nifty = nifty[["Date", "Close"]].set_index("Date")
            nifty = nifty.rename(columns={"Close": "NIFTY_Close"})

            # Fetch VIX data from GitHub
            vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"
            response_vix = requests.get(vix_url)
            response_vix.raise_for_status() # Raise an HTTPError for bad responses
            vix = pd.read_csv(io.StringIO(response_vix.text))
            vix.columns = vix.columns.str.strip()
            vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")
            vix = vix.dropna(subset=["Date"])
            vix["Date"] = vix["Date"].dt.normalize()
            vix = vix.groupby("Date").last().reset_index() # Handle duplicates
            vix = vix[["Date", "Close"]].set_index("Date").rename(columns={"Close": "VIX"})

            # Combine and clean
            common_dates = nifty.index.intersection(vix.index)
            df = pd.DataFrame({
                "NIFTY_Close": nifty["NIFTY_Close"].loc[common_dates],
                "VIX": vix["VIX"].loc[common_dates]
            }, index=common_dates)
            df = df.groupby(df.index).last() # Final deduplication after join
            df = df.sort_index()
            df = df.ffill().bfill() # Fill missing values

            logger.info(f"Data loaded successfully from CSV. Shape: {df.shape}")
            return df, st.session_state.real_time_market_data, st.session_state.data_source # real_time_market_data will be None
        except Exception as e:
            st.error(f"❌ Error loading data from GitHub CSV: {str(e)}")
            logger.error(f"❌ Error loading data from GitHub CSV: {str(e)}", exc_info=True)
            st.session_state.data_source = "None (Data Load Failed)"
            return pd.DataFrame(), st.session_state.real_time_market_data, st.session_state.data_source # Return empty DF on failure

    else:
        # API data was successful, combine with historical if needed
        logger.info("API data fetch successful. Combining with historical CSV for backtesting context.")
        st.session_state.data_source = "5paisa API (LIVE)"

        try:
             # Load historical data from CSVs to provide context for backtesting/features
             nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
             response_nifty = requests.get(nifty_url)
             response_nifty.raise_for_status()
             nifty_hist = pd.read_csv(io.StringIO(response_nifty.text), encoding="utf-8-sig")
             nifty_hist.columns = nifty_hist.columns.str.strip()
             nifty_hist["Date"] = pd.to_datetime(nifty_hist["Date"], format="%d-%b-%Y", errors="coerce").dropna().dt.normalize()
             nifty_hist = nifty_hist.groupby("Date").last().reset_index().set_index("Date")
             nifty_hist = nifty_hist.rename(columns={"Close": "NIFTY_Close"})[["NIFTY_Close"]]

             vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"
             response_vix = requests.get(vix_url)
             response_vix.raise_for_status()
             vix_hist = pd.read_csv(io.StringIO(response_vix.text))
             vix_hist.columns = vix_hist.columns.str.strip()
             vix_hist["Date"] = pd.to_datetime(vix_hist["Date"], format="%d-%b-%Y", errors="coerce").dropna().dt.normalize()
             vix_hist = vix_hist.groupby("Date").last().reset_index().set_index("Date")
             vix_hist = vix_hist.rename(columns={"Close": "VIX"})[["VIX"]]

             historical_df = pd.merge(nifty_hist, vix_hist, left_index=True, right_index=True, how='inner').sort_index()

             # Create a DataFrame for the latest API data point
             latest_date = datetime.now().date()
             latest_df = pd.DataFrame({
                 "NIFTY_Close": [st.session_state.real_time_market_data["nifty_spot"]],
                 "VIX": [st.session_state.real_time_market_data["vix"]]
             }, index=[pd.to_datetime(latest_date).normalize()]) # Use current date as index

             # Combine historical data (excluding the very last historical day if it's today)
             # with the live data point. Avoid duplicate dates.
             df = pd.concat([historical_df[historical_df.index < latest_df.index[0]], latest_df])
             df = df.groupby(df.index).last() # Handle potential remaining duplicates robustly
             df = df.sort_index()
             df = df.ffill().bfill() # Fill missing values if any introduced

             logger.info(f"Combined historical (CSV) and real-time (API) data. Shape: {df.shape}")
             return df, st.session_state.real_time_market_data, st.session_state.data_source

        except Exception as e:
            st.warning(f"⚠️ API data fetched, but failed to load or combine with historical CSV data: {str(e)}")
            logger.error(f"⚠️ Error combining API data with historical CSV: {str(e)}", exc_info=True)
            # Return only the live data point if combining fails
            latest_date = datetime.now().date()
            df_live_only = pd.DataFrame({
                "NIFTY_Close": [st.session_state.real_time_market_data.get("nifty_spot")],
                "VIX": [st.session_state.real_time_market_data.get("vix")]
            }, index=[pd.to_datetime(latest_date).normalize()]).dropna() # Ensure no None/NaN rows
            st.session_state.data_source += " (Historical CSV Failed)"
            return df_live_only, st.session_state.real_time_market_data, st.session_state.data_source


@st.cache_data(ttl=3600) # Cache features for 1 hour, recalculates if inputs change
def generate_features(df: pd.DataFrame, real_data: dict | None, capital: float) -> pd.DataFrame | None:
    """
    Generates features for volatility forecasting and strategy selection.
    Injects real-time data into the last row if available.
    """
    try:
        logger.info("Generating features")
        if df.empty:
            logger.warning("Input DataFrame is empty, cannot generate features.")
            return None

        df = df.copy()  # Work on a copy to avoid modifying cached/original df
        df.index = pd.to_datetime(df.index).normalize() # Ensure date-only index

        # Ensure capital column exists and is correct
        df["Total_Capital"] = capital

        # Calculate Log Returns for GARCH base (used later, but good to have here)
        df['Log_Returns'] = np.log(df['NIFTY_Close'] / df['NIFTY_Close'].shift(1)) * 100

        # Calculate Realized Volatility (using a rolling window)
        df["Realized_Vol"] = df['Log_Returns'].rolling(window=5, min_periods=1).std() * np.sqrt(252) # 5-day annualized vol
        # Fill initial NaN values, potentially using VIX as a fallback if no historical data
        df["Realized_Vol"] = df["Realized_Vol"].fillna(df["VIX"] / np.sqrt(252) * np.sqrt(df.index.dayofweek.map({0:5, 1:4, 2:3, 3:2, 4:1, 5:1, 6:1}))).fillna(df["VIX"]) # Simple initial guess
        df["Realized_Vol"] = np.clip(df["Realized_Vol"], 0, 50) # Clip to reasonable range

        # --- Add or update features using real-time data if available ---
        # These features are calculated dynamically or injected based on the *last* row (current data)

        # Ensure last row exists
        if df.empty:
             return df # Return empty if still empty

        last_row_index = df.index[-1]

        # Inject real-time Nifty/Vix if API data is present and valid
        if real_data and real_data.get("nifty_spot") is not None and real_data.get("vix") is not None:
             df.loc[last_row_index, "NIFTY_Close"] = real_data["nifty_spot"]
             df.loc[last_row_index, "VIX"] = real_data["vix"]
             # Recalculate the last log return and realized vol using the live data
             if len(df) > 1:
                 df.loc[last_row_index, 'Log_Returns'] = np.log(df.loc[last_row_index, 'NIFTY_Close'] / df.iloc[-2]['NIFTY_Close']) * 100
                 # Update the last point of rolling realized vol
                 rolling_window = df['Log_Returns'].rolling(window=5, min_periods=1)
                 df.loc[last_row_index, "Realized_Vol"] = rolling_window.std().iloc[-1] * np.sqrt(252)
                 df["Realized_Vol"] = np.clip(df["Realized_Vol"], 0, 50)


        # Calculate Days to Expiry (assuming Thursday expiry for NIFTY)
        def calculate_days_to_expiry(date_index):
            # Find the next Thursday >= date
            next_thursday = date_index + pd.to_timedelta((3 - date_index.dayofweek) % 7, unit='D')
            # If today is Thursday and after market hours, next expiry is next Thursday
            # This simple logic assumes EOD data. For intraday, this needs adjustment.
            # For EOD data, if day is Thursday, the next expiry is next week's Thursday.
            # Let's keep it simple like the original for now based on Date index.
            days_to_expiry = (next_thursday - date_index).days
            # If DTE is 0 on a data date, it means expiry was today. Next expiry is next week.
            # This assumes data points are EOD. If data is intraday, DTE 0 means expiry today.
            # Given CSV data is EOD, and live data is a point in time, let's refine this.
            # If the *date* is a Thursday, and it's the last data point (potentially live),
            # we need to decide if we consider the *next* Thursday or *this* Thursday's end.
            # Let's assume EOD data for historical and the live point is EOD for simplicity.
            # If the date is a Thursday, the *next* relevant expiry is the following Thursday.
            # If the date is not a Thursday, calculate days until the upcoming Thursday.
            adjusted_days_to_expiry = []
            for date in date_index:
                days_until_thursday = (3 - date.weekday()) % 7
                if days_until_thursday == 0: # It is a Thursday
                    # If this is the last data point and it's a Thursday, the relevant expiry
                    # for future analysis is likely the *next* Thursday, unless trading happens
                    # before this Thursday's close. Let's assume the expiry for today's data
                    # is the upcoming Thursday, or next Thursday if today is Thursday.
                    # A simple way for EOD data: if date is Thursday, next expiry is +7 days.
                     days_until_thursday = 7 # Next Thursday
                adjusted_days_to_expiry.append(days_until_thursday)
            return pd.Series(adjusted_days_to_expiry, index=date_index)

        df["Days_to_Expiry"] = calculate_days_to_expiry(df.index)
        # Ensure Days_to_Expiry for the last (live) day reflects the *actual* next expiry
        # if real_data includes expiry date string. This requires mapping the string to DTE.
        if real_data and real_data.get("expiry_date_str") and real_data.get("nifty_spot") is not None:
             try:
                 expiry_dt = datetime.strptime(real_data["expiry_date_str"], "%Y-%m-%d").date()
                 live_date = df.index[-1].date()
                 # DTE for the live point is the difference between expiry date and the live data date
                 df.loc[last_row_index, "Days_to_Expiry"] = max(0, (expiry_dt - live_date).days)
             except Exception as e:
                 logger.warning(f"Could not set accurate DTE for live data: {e}")
                 # Keep the calculated DTE based on weekday if parsing fails


        # Event Flag (Simple: expiry in < 3 days or quarterly expiry week)
        # Assuming quarterly expiries are March, June, Sep, Dec first week (adjust if needed)
        df["Event_Flag"] = 0
        # Set flag if expiry is within 3 days (or whatever threshold)
        df.loc[df["Days_to_Expiry"] <= 3, "Event_Flag"] = 1
        # Set flag for quarterly expiry week (first week of Mar, Jun, Sep, Dec)
        # This is approximate based on date, not actual expiry calendar
        quarterly_months = [3, 6, 9, 12]
        quarterly_expiry_week_dates = df[(df.index.month.isin(quarterly_months)) & (df.index.day <= 7)].index
        df.loc[quarterly_expiry_week_dates, "Event_Flag"] = 1 # Mark these dates as event days


        # Calculate IVP (Percentile of current IV within a lookback window)
        def dynamic_ivp(series):
            # Requires at least 5 data points in the window
            if len(series) < 5:
                return np.nan # Return NaN, will be interpolated
            # Calculate percentile rank of the *last* value in the window
            # among the values *excluding* the last value
            lookback_values = series.iloc[:-1]
            current_value = series.iloc[-1]
            if lookback_values.empty:
                 return np.nan
            percentile = (lookback_values <= current_value).sum() / len(lookback_values) * 100
            return percentile

        # Use a smaller window for IVP if total data length is limited, but at least 50 days
        ivp_window = min(252, max(50, len(df) // 2)) # Dynamic window size
        df["IVP"] = df["ATM_IV"].rolling(window=ivp_window, min_periods=5).apply(dynamic_ivp, raw=False)
        df["IVP"] = df["IVP"].interpolate(method='linear').fillna(method='bfill').fillna(50.0) # Interpolate and fill missing


        # Inject/Estimate PCR, Straddle Price, Max Pain Diff for historical/simulated data
        # For historical data, these are often not directly available in simple CSVs.
        # The original Streamlit code used random generation + trend/real_data injection.
        # Let's keep a simplified version that injects live data if available,
        # and potentially simulates for historical if not already present.
        # The CSVs only have Nifty Close and VIX. We must simulate or use fallbacks.

        # For historical rows (where real_data is not applicable), simulate PCR and Straddle Price
        # Simple simulation based on VIX and trend. Adjust parameters as needed.
        if real_data is None: # Entire DF is from CSV fallback
             logger.warning("Generating simulated PCR and Straddle Price for historical CSV data.")
             market_trend_hist = df["NIFTY_Close"].pct_change().rolling(window=5).mean().fillna(0)
             # PCR simulation: Tend to be >1 in uptrends, <1 in downtrends
             df["PCR"] = np.clip(1.0 + np.random.normal(0, 0.1, len(df)) - market_trend_hist * 5, 0.7, 2.0)
             # Straddle Price simulation: Correlates with Spot and ATM_IV (VIX)
             # Simple linear relation + noise: price = a * spot + b * iv + noise
             # Coefficients (a, b) need empirical tuning. Using simplified approach based on VIX/IV.
             df["Straddle_Price"] = np.clip(5 * df["ATM_IV"] + np.random.normal(0, 20, len(df)), 50, 400)
             # Spot_MaxPain_Diff_Pct simulation: Random noise around a small mean
             df["Spot_MaxPain_Diff_Pct"] = np.abs(np.random.lognormal(-2.5, 0.6, len(df))) # Tends to be small positive %
             df["Spot_MaxPain_Diff_Pct"] = np.clip(df["Spot_MaxPain_Diff_Pct"], 0.1, 5.0) # Clip to 0.1% to 5% range

        # Inject/Overwrite PCR, Straddle Price, Max Pain Diff for the LAST row if real-time API data is available
        if real_data:
             if real_data.get("pcr") is not None:
                 df.loc[last_row_index, "PCR"] = real_data["pcr"]
             if real_data.get("straddle_price") is not None:
                  df.loc[last_row_index, "Straddle_Price"] = real_data["straddle_price"]
             if real_data.get("max_pain_diff_pct") is not None:
                  df.loc[last_row_index, "Spot_MaxPain_Diff_Pct"] = real_data["max_pain_diff_pct"]


        # Calculate VIX Change Pct
        df["VIX_Change_Pct"] = df["VIX"].pct_change().fillna(0) * 100
        # If real_data has VIX, recalculate the last VIX Change Pct based on the previous day's VIX in df
        if real_data and real_data.get("vix") is not None and len(df) > 1:
             prev_day_vix = df.iloc[-2]["VIX"]
             if prev_day_vix != 0:
                  df.loc[last_row_index, "VIX_Change_Pct"] = (df.loc[last_row_index, "VIX"] - prev_day_vix) / prev_day_vix * 100
             else:
                  df.loc[last_row_index, "VIX_Change_Pct"] = 0


        # Simulate FII Data and Capital Pressure Index (as these are not from 5paisa API directly)
        # Keep the existing simulation as it's not API-dependent features
        n_days = len(df) # Use updated length after potential concatenation
        # Ensure simulation arrays match the current DF length
        if n_days > 0:
             fii_trend_sim = np.random.normal(0, 10000, n_days)
             # Introduce some reversals
             reverse_points = np.random.choice(n_days, size=max(1, n_days // 30), replace=False)
             fii_trend_sim[reverse_points] *= -1
             df["FII_Index_Fut_Pos"] = np.cumsum(fii_trend_sim).astype(int)

             fii_option_trend_sim = np.random.normal(0, 5000, n_days)
             df["FII_Option_Pos"] = np.cumsum(fii_option_trend_sim).astype(int)

             # IV Skew simulation (correlated with VIX level)
             df["IV_Skew"] = np.clip(np.random.normal(0, 0.8, n_days) + (df["VIX"] / 15 - 1) * 3, -3, 3)

             # Advance Decline Ratio simulation (correlated with market trend)
             df["Advance_Decline_Ratio"] = np.clip(1.0 + np.random.normal(0, 0.2, n_days) + df["NIFTY_Close"].pct_change().fillna(0).rolling(window=5).mean() * 10, 0.5, 2.0)

             # Capital Pressure Index (based on simulated/injected FII and PCR)
             # Avoid division by zero if FII positions are near zero
             fii_fut_norm = df["FII_Index_Fut_Pos"] / df["FII_Index_Fut_Pos"].std() if df["FII_Index_Fut_Pos"].std() != 0 else 0
             fii_opt_norm = df["FII_Option_Pos"] / df["FII_Option_Pos"].std() if df["FII_Option_Pos"].std() != 0 else 0
             pcr_norm = (df["PCR"] - df["PCR"].mean()) / df["PCR"].std() if df["PCR"].std() != 0 else 0
             df["Capital_Pressure_Index"] = (fii_fut_norm + fii_opt_norm + pcr_norm) / 3
             df["Capital_Pressure_Index"] = np.clip(df["Capital_Pressure_Index"], -3, 3) # Clip to a reasonable range

             # PnL Day simulation (Simple noise, maybe reduced on event days)
             df["PnL_Day"] = np.random.normal(0, capital * 0.005, n_days) * (1 - df["Event_Flag"] * 0.5) # PnL scale with capital, reduced volatility on event days

        else: # If n_days is 0 (empty dataframe)
             logger.warning("DataFrame is empty, skipped simulation of FII, Skew, etc.")
             # Ensure these columns exist even if empty
             for col in ['Log_Returns', 'Days_to_Expiry', 'Event_Flag', 'IVP', 'PCR', 'VIX_Change_Pct',
                         'Spot_MaxPain_Diff_Pct', 'FII_Index_Fut_Pos', 'FII_Option_Pos', 'IV_Skew',
                         'Realized_Vol', 'Advance_Decline_Ratio', 'Capital_Pressure_Index', 'Total_Capital', 'PnL_Day', 'Straddle_Price']:
                 if col not in df.columns:
                     df[col] = None # Or appropriate default empty series


        # Final check for NaNs and infs after all calculations/injections
        if df.isna().sum().sum() > 0 or np.isinf(df.values).sum() > 0:
            logger.warning(f"NaNs or Infs found after feature generation: NaNs={df.isna().sum().sum()}, Infs={np.isinf(df.values).sum()}. Interpolating/filling.")
            # Attempt robust fillna/replace inf
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.interpolate(method='linear', limit_direction='both')
            df = df.fillna(method='bfill').fillna(method='ffill').fillna(0) # Fallback fill with 0


        # Define the final list of feature columns used for forecasting/strategy
        # Ensure all required feature_cols actually exist in the DataFrame before returning
        final_feature_cols = [col for col in feature_cols if col in df.columns]
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
             logger.error(f"Missing required feature columns after generation: {missing_features}")
             # Decide how to handle: return None, or return with missing? Returning with missing might cause later errors.
             # For robustness, let's check again before forecasting.


        logger.debug("Features generated successfully")
        return df # Return the DataFrame with all features

    except Exception as e:
        st.error(f"❌ Error generating features: {str(e)}")
        logger.error(f"❌ Error generating features: {str(e)}", exc_info=True)
        return None

# Volatility Forecasting (using the refined DataFrame with features)
feature_cols = [ # Redefine or ensure consistent
    'VIX', 'ATM_IV', 'IVP', 'PCR', 'VIX_Change_Pct', 'IV_Skew', 'Straddle_Price',
    'Spot_MaxPain_Diff_Pct', 'Days_to_Expiry', 'Event_Flag', 'FII_Index_Fut_Pos',
    'FII_Option_Pos', 'Realized_Vol' # Added Realized_Vol as a feature/target
]


@st.cache_data(ttl=3600) # Cache forecast results for 1 hour
def forecast_volatility_future(df: pd.DataFrame | None, forecast_horizon: int):
    """
    Forecasts volatility using GARCH and XGBoost.
    Requires a DataFrame with 'NIFTY_Close' and feature_cols.
    """
    try:
        logger.info(f"Forecasting volatility for {forecast_horizon} days")
        if df is None or df.empty:
             logger.warning("Input DataFrame is empty or None for forecasting.")
             return None, None, None, None, None, None, None, None

        df = df.copy() # Work on a copy
        df.index = pd.to_datetime(df.index).normalize() # Ensure date-only index

        # Ensure required columns exist for forecasting
        required_for_forecast = ['NIFTY_Close', 'Log_Returns', 'Realized_Vol'] + [col for col in feature_cols if col not in ['Log_Returns', 'Realized_Vol']]
        missing_for_forecast = [col for col in required_for_forecast if col not in df.columns]
        if missing_for_forecast:
             logger.error(f"Missing required columns for forecasting: {missing_for_forecast}")
             st.error(f"❌ Cannot run forecast: Missing data features ({', '.join(missing_for_forecast)})")
             return None, None, None, None, None, None, None, None

        # --- GARCH Forecast ---
        # GARCH needs a relatively long series of returns
        df_garch = df['Log_Returns'].dropna().copy()
        if len(df_garch) < 200:
            logger.warning(f"Insufficient data ({len(df_garch)} days) for robust GARCH model, attempting anyway.")
            # Optionally return None or use a fallback here if less than minimum data
            if len(df_garch) < 50: # Set a lower threshold for attempting GARCH
                 st.warning(f"⚠️ Very limited data ({len(df_garch)} days) for GARCH. Forecast may be unreliable.")
                 # Fallback: Maybe just use Realized Vol mean as forecast?
                 # For now, let it try, but warn.

        try:
            garch_model = arch_model(df_garch, vol='Garch', p=1, q=1, rescale=False) # Use log returns directly
            garch_fit = garch_model.fit(disp="off")
            garch_forecast = garch_fit.forecast(horizon=forecast_horizon, reindex=False)
            # Annualize the variance forecast from the last date
            garch_vols = np.sqrt(garch_forecast.variance.iloc[-1].values) * np.sqrt(252)
            garch_vols = np.clip(garch_vols, 5, 50) # Clip to reasonable range
            logger.debug(f"GARCH forecast successful. First day vol: {garch_vols[0]:.2f}%")
        except Exception as e:
             logger.error(f"❌ Error running GARCH forecast: {e}", exc_info=True)
             st.error(f"❌ GARCH forecast failed: {e}")
             garch_vols = np.full(forecast_horizon, np.nan) # Fill with NaN if GARCH fails


        # --- XGBoost Forecast ---
        # XGBoost predicts based on engineered features
        df_xgb = df.copy() # Use the dataframe with features
        df_xgb['Target_Vol'] = df_xgb['Realized_Vol'].shift(-1) # Target is next day's realized vol
        df_xgb = df_xgb.dropna(subset=['Target_Vol'] + feature_cols) # Drop rows missing target or features

        X = df_xgb[feature_cols]
        y = df_xgb['Target_Vol']

        if len(X) < 50: # Need sufficient data points for XGBoost training
             logger.warning(f"Insufficient data ({len(X)} days) for XGBoost training. Skipping XGBoost forecast.")
             st.warning(f"⚠️ Insufficient data ({len(X)} days) for XGBoost training. Using GARCH/Realized Vol only.")
             xgb_vols = np.full(forecast_horizon, np.nan) # Fill with NaN if training data is too short
             feature_importances = pd.Series([], dtype=float) # Empty feature importances
             rmse = np.nan # RMSE is not applicable

        else:
             # Scale features
             scaler = StandardScaler()
             X_scaled = scaler.fit_transform(X)
             X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

             # Train-test split for local evaluation (optional in final app, but good for diagnostics)
             split_index = max(1, int(len(X) * 0.8)) # Ensure split_index is at least 1
             X_train, X_test = X_scaled.iloc[:split_index], X_scaled.iloc[split_index:]
             y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

             # Train XGBoost model
             model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.03, random_state=42, tree_method='hist') # 'hist' is often faster
             model.fit(X_train, y_train)

             # Evaluate (optional)
             y_pred = model.predict(X_test)
             rmse = np.sqrt(mean_squared_error(y_test, y_pred))
             logger.debug(f"XGBoost trained. RMSE on test set: {rmse:.2f}%")

             # Forecast future volatility using the model
             xgb_vols = []
             # Start forecasting from the last available feature values
             current_features = df[feature_cols].iloc[-1].copy() # Start with the last available features

             # --- Simulate future feature values for forecasting steps ---
             # This is a simplification. A more complex model would forecast features too.
             # For now, perturb the last known features slightly for each forecast step.
             # The quality of XGBoost forecast depends heavily on the quality of future feature simulation.

             for i in range(forecast_horizon):
                 # Prepare features for prediction (needs to be 2D array/DataFrame)
                 current_features_df = pd.DataFrame([current_features.values], columns=feature_cols)
                 # Scale the current features using the *same* scaler fitted on training data
                 current_features_scaled = scaler.transform(current_features_df)

                 # Predict next vol
                 next_vol_pred = model.predict(current_features_scaled)[0]
                 xgb_vols.append(next_vol_pred)

                 # --- Simulate/Update features for the *next* forecast step ---
                 # These are heuristics. Improve based on market knowledge if possible.
                 current_features["Days_to_Expiry"] = max(1, current_features["Days_to_Expiry"] - 1) # DTE decreases
                 current_features["VIX"] = np.clip(current_features["VIX"] * np.random.uniform(0.98, 1.02), 5, 50) # VIX random walk
                 current_features["Straddle_Price"] = np.clip(current_features["Straddle_Price"] * np.random.uniform(0.98, 1.02), 50, 400) # Straddle random walk
                 # VIX change is hard to simulate simply without knowing the next VIX relative to the one before that.
                 # Let's just set it to a small value or average change for simulation steps.
                 current_features["VIX_Change_Pct"] = np.random.normal(0, 1.0) # Simulate small random change

                 current_features["ATM_IV"] = current_features["VIX"] # Simple assumption: ATM_IV tracks VIX in simulation
                 current_features["Realized_Vol"] = next_vol_pred # Assume next realized vol is our prediction (simplification)
                 current_features["IVP"] = np.clip(current_features["IVP"] + np.random.normal(0, 2), 0, 100) # IVP random walk
                 current_features["PCR"] = np.clip(current_features["PCR"] + np.random.normal(0, 0.03), 0.7, 2.0) # PCR random walk
                 current_features["Spot_MaxPain_Diff_Pct"] = np.clip(current_features["Spot_MaxPain_Diff_Pct"] * np.random.uniform(0.95, 1.05), 0.1, 5.0) # Max Pain Diff random walk
                 # Event flag is complex; assuming it stays 0 unless DTE hits trigger or specific future date is known.
                 # For simulation, let's assume no future events beyond DTE trigger.
                 current_features["Event_Flag"] = 1 if current_features["Days_to_Expiry"] <= 3 else 0

                 current_features["FII_Index_Fut_Pos"] += np.random.normal(0, 500) # FII random walk
                 current_features["FII_Option_Pos"] += np.random.normal(0, 200) # FII random walk
                 current_features["IV_Skew"] = np.clip(current_features["IV_Skew"] + np.random.normal(0, 0.1), -3, 3) # Skew random walk
                 current_features["Advance_Decline_Ratio"] = np.clip(current_features["Advance_Decline_Ratio"] + np.random.normal(0, 0.05), 0.5, 2.0) # AD Ratio random walk
                 # Capital Pressure Index depends on other features; recalculate based on simulated ones
                 fii_fut_norm_sim = current_features["FII_Index_Fut_Pos"] / (df["FII_Index_Fut_Pos"].std() if df["FII_Index_Fut_Pos"].std() != 0 else 1)
                 fii_opt_norm_sim = current_features["FII_Option_Pos"] / (df["FII_Option_Pos"].std() if df["FII_Option_Pos"].std() != 0 else 1)
                 pcr_mean = df["PCR"].mean() if df["PCR"].std() != 0 else 1.0
                 pcr_std = df["PCR"].std() if df["PCR"].std() != 0 else 0.1
                 pcr_norm_sim = (current_features["PCR"] - pcr_mean) / pcr_std
                 current_features["Capital_Pressure_Index"] = (fii_fut_norm_sim + fii_opt_norm_sim + pcr_norm_sim) / 3
                 current_features["Capital_Pressure_Index"] = np.clip(current_features["Capital_Pressure_Index"], -3, 3)


             xgb_vols = np.clip(xgb_vols, 5, 50) # Clip forecast to reasonable range

             # Get feature importances
             feature_importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
             logger.debug("XGBoost forecast successful.")


        # --- Blending and Confidence ---
        # Need at least one valid forecast source to blend
        valid_garch = not np.isnan(garch_vols).all()
        valid_xgb = not np.isnan(xgb_vols).all()

        if not valid_garch and not valid_xgb:
            st.error("❌ Both GARCH and XGBoost forecasts failed.")
            return None, None, None, None, None, None, None, None

        # If only one is valid, use that one's forecast
        if valid_garch and not valid_xgb:
            blended_vols = garch_vols
            confidence_score = 60 # Lower confidence if only one model works
            logger.warning("XGBoost failed, using GARCH only.")
        elif not valid_garch and valid_xgb:
            blended_vols = xgb_vols
            confidence_score = 60 # Lower confidence
            logger.warning("GARCH failed, using XGBoost only.")
        else: # Both are valid, blend them
             # Ensure arrays are the same length (they should be)
             min_len = min(len(garch_vols), len(xgb_vols))
             garch_vols = garch_vols[:min_len]
             xgb_vols = xgb_vols[:min_len]
             future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=min_len, freq='B') # Recalculate future dates based on min_len

             # Simple blending based on how close the models were on the last known point (Realized Vol)
             # Ensure realized_vol is valid
             if np.isnan(realized_vol) or realized_vol is None:
                 logger.warning("Realized Vol not available, using equal weights for blending.")
                 garch_weight = xgb_weight = 0.5
             else:
                 # Ensure the first forecast points are valid for blending logic
                 if np.isnan(garch_vols[0]) and np.isnan(xgb_vols[0]):
                      garch_weight = xgb_weight = 0.5 # Fallback if first points are NaN
                 elif np.isnan(garch_vols[0]):
                      garch_weight = 0 # Only use XGBoost
                      xgb_weight = 1
                 elif np.isnan(xgb_vols[0]):
                      garch_weight = 1 # Only use GARCH
                      xgb_weight = 0
                 else:
                      garch_diff = np.abs(garch_vols[0] - realized_vol)
                      xgb_diff = np.abs(xgb_vols[0] - realized_vol)
                      # Avoid division by zero if both diffs are zero or sum is zero
                      total_diff = garch_diff + xgb_diff
                      if total_diff == 0:
                           garch_weight = xgb_weight = 0.5
                      else:
                          garch_weight = xgb_diff / total_diff # Give more weight to the model closer to realized vol
                          xgb_weight = garch_diff / total_diff # Or simply xgb_weight = 1 - garch_weight

             blended_vols = [(garch_weight * g) + (xgb_weight * x) for g, x in zip(garch_vols, xgb_vols)]

             # Confidence based on agreement and RMSE
             agreement = 1 - np.abs(garch_vols[0] - xgb_vols[0]) / (np.mean([garch_vols[0], xgb_vols[0]]) if np.mean([garch_vols[0], xgb_vols[0]]) != 0 else 1)
             confidence_score = np.clip(50 + agreement * 40 - (rmse if not np.isnan(rmse) else 10), 20, 95) # Heuristic confidence calculation

             logger.debug(f"Blended forecast. GARCH weight: {garch_weight:.2f}, XGB weight: {xgb_weight:.2f}. Confidence: {confidence_score:.2f}%")


        # Ensure forecast dates align with the number of forecast points
        future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=len(blended_vols), freq='B')


        forecast_log = pd.DataFrame({
            "Date": future_dates,
            "GARCH_Vol": garch_vols[:len(future_dates)] if len(garch_vols) >= len(future_dates) else np.nan, # Ensure length matches
            "XGBoost_Vol": xgb_vols[:len(future_dates)] if len(xgb_vols) >= len(future_dates) else np.nan, # Ensure length matches
            "Blended_Vol": blended_vols,
            "Confidence": [confidence_score] * len(blended_vols) # Confidence score applies to the overall forecast
        })

        # Calculate Realized Volatility for the last few days for display
        realized_vol_latest = df["Realized_Vol"].iloc[-5:].mean() if len(df) >= 5 else df["Realized_Vol"].iloc[-1] if not df.empty else 0


        logger.debug("Volatility forecasting function completed.")
        return forecast_log, garch_vols, xgb_vols, blended_vols, realized_vol_latest, confidence_score, rmse, feature_importances # Return RMSE and feature_importances if XGBoost ran

    except Exception as e:
        st.error(f"❌ An unexpected error occurred during volatility forecasting: {str(e)}")
        logger.error(f"❌ An unexpected error occurred during volatility forecasting: {str(e)}", exc_info=True)
        return None, None, None, None, None, None, None, None # Return None for all outputs on critical error


# Backtesting (uses generated features, not directly API)
@st.cache_data(ttl=3600) # Cache backtest results for 1 hour
def run_backtest(df: pd.DataFrame | None, capital: float, strategy_choice: str, start_date: datetime, end_date: datetime):
    """
    Runs a backtest on historical data using simulated trading logic.
    """
    try:
        logger.info(f"Starting backtest for {strategy_choice} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        if df is None or df.empty:
            st.warning("⚠️ Backtest failed: No data available.")
            logger.warning("Backtest failed: Input DataFrame is empty.")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        # Ensure unique index and filter by date range
        df_filtered = df.groupby(df.index).last() # Aggregate duplicates by taking the last value
        df_filtered = df_filtered.loc[start_date.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d')].copy()

        if len(df_filtered) < 50:
            st.warning(f"⚠️ Backtest failed: Insufficient data ({len(df_filtered)} days) in the selected date range.")
            logger.warning(f"Backtest failed: Insufficient data ({len(df_filtered)} days) after date filtering.")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        # Ensure required columns are present for backtesting logic
        required_cols_backtest = ["NIFTY_Close", "ATM_IV", "Realized_Vol", "IV_Skew", "Days_to_Expiry", "Event_Flag", "Total_Capital", "Straddle_Price"]
        missing_cols_backtest = [col for col in required_cols_backtest if col not in df_filtered.columns]
        if missing_cols_backtest:
            st.error(f"❌ Backtest failed: Missing required features for simulation ({', '.join(missing_cols_backtest)})")
            logger.error(f"Backtest failed: Missing columns {missing_cols_backtest}")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        backtest_results = []
        lot_size = 25 # Assuming NIFTY lot size
        base_transaction_cost_pct = 0.002 # 0.2% transaction cost (brokerage, exchange charges, etc.)
        stt_pct = 0.0005 # STT on premium (simplified)
        portfolio_pnl = 0 # Cumulative PnL during backtest
        risk_free_rate = 0.06 / 252 # Approximate daily risk-free rate (using 252 trading days)

        # Strategy Engine Logic (Simplified version for backtest)
        def run_strategy_engine_backtest(day_data, avg_vol_lookback, current_portfolio_pnl, total_capital):
            try:
                iv = day_data["ATM_IV"]
                hv = day_data["Realized_Vol"]
                iv_hv_gap = iv - hv
                # iv_skew = day_data["IV_Skew"] # Not used in the original engine logic provided
                dte = day_data["Days_to_Expiry"]
                event_flag = day_data["Event_Flag"]

                # Simple drawdown check (stop trading if significant loss)
                if current_portfolio_pnl < -0.10 * total_capital: # Stop if cumulative loss exceeds 10% of initial capital
                    # logger.debug(f"Drawdown limit reached on {day_data.name.strftime('%Y-%m-%d')}. Stopping strategy.")
                    return None, None, "Drawdown limit reached", [], 0, 0, 0 # Signal to stop trading

                # Determine Regime based on average volatility (lookback)
                # Using Realized Vol lookback for regime in backtest for consistency with historical data
                if avg_vol_lookback < 15:
                    regime = "LOW"
                elif avg_vol_lookback < 20:
                    regime = "MEDIUM"
                else:
                    regime = "HIGH"

                # Overlay Event Regime
                if event_flag == 1:
                    regime = "EVENT-DRIVEN" # Event overrides volatility regime


                strategy = "Undefined"
                reason = "N/A"
                tags = []
                risk_reward_target = 1.0 # Base R:R target

                # Strategy Selection Logic (Matches the generation logic)
                if regime == "LOW":
                    if iv_hv_gap > 5 and dte < 10:
                        strategy = "Butterfly Spread"
                        reason = "Low vol & short expiry favors pinning strategies"
                        tags = ["Neutral", "Theta", "Expiry Play"]
                        risk_reward_target = 2.0
                    else:
                        strategy = "Iron Fly"
                        reason = "Low volatility and time decay favors delta-neutral Iron Fly"
                        tags = ["Neutral", "Theta", "Range Bound"]
                        risk_reward_target = 1.5 # Adjusted based on typical Iron Fly R:R

                elif regime == "MEDIUM":
                    if iv_hv_gap > 3: # Simplified skew condition
                        strategy = "Iron Condor"
                        reason = "Medium vol and premium favor range-bound Iron Condor"
                        tags = ["Neutral", "Theta", "Range Bound"]
                        risk_reward_target = 1.8
                    else:
                         strategy = "Short Strangle"
                         reason = "Balanced vol, no significant events, premium selling opportunity"
                         tags = ["Neutral", "Premium Selling", "Volatility Harvest"]
                         risk_reward_target = 1.6 # Adjusted


                elif regime == "HIGH":
                    if iv_hv_gap > 10:
                        strategy = "Jade Lizard"
                        reason = "High IV favors defined risk skewed strategy"
                        tags = ["Skewed", "Volatility", "Defined Risk"]
                        risk_reward_target = 1.2
                    else:
                        strategy = "Iron Condor" # High vol also favors wide Iron Condors for premium
                        reason = "High vol favors wide Iron Condor for premium collection"
                        tags = ["Neutral", "Theta", "Range Bound"]
                        risk_reward_target = 1.5 # Adjusted for wider strikes


                elif regime == "EVENT-DRIVEN":
                    if iv > 30 and dte < 5:
                        strategy = "Short Straddle"
                        reason = "Event + near expiry + IV spike → high premium capture"
                        tags = ["Volatility", "Event", "Neutral"]
                        risk_reward_target = 1.5
                    else:
                        strategy = "Calendar Spread" # Event-based uncertainty favors term structure
                        reason = "Event uncertainty favors Calendar Spread"
                        tags = ["Volatility", "Event", "Calendar"]
                        risk_reward_target = 1.3


                # Capital allocation based on regime
                capital_alloc_pct = {"LOW": 0.10, "MEDIUM": 0.08, "HIGH": 0.06, "EVENT-DRIVEN": 0.06} # Percentage of total capital
                deployable_capital = total_capital * capital_alloc_pct.get(regime, 0.06) # Default to 6%

                # Maximum loss per trade (as a percentage of deployed capital)
                # Simplified: Max loss target is a fixed percentage of deployed capital
                max_loss_target_pct = 0.025 # Target 2.5% max loss of deployed capital per trade
                max_loss_absolute = deployable_capital * max_loss_target_pct

                # In a backtest, we don't place orders. We simulate the outcome.
                # The "premium" and "loss" simulation needs to be aligned with the strategy and market moves.
                # The original code had a simulation based on Straddle Price and Nifty Move vs BreakEven.
                # Let's use a simplified PnL simulation based on volatility and market direction relative to strategy.

                return regime, strategy, reason, tags, deployable_capital, max_loss_absolute, risk_reward_target
            except Exception as e:
                logger.error(f"Error in backtest strategy engine: {str(e)}")
                return None, None, f"Strategy engine failed: {str(e)}", [], 0, 0, 0


        # --- Backtest Simulation Loop ---
        trading_stopped = False # Flag to stop trading after drawdown limit
        total_capital = capital # Start with initial capital for drawdown tracking

        for i in range(1, len(df_filtered)):
            try:
                day_data = df_filtered.iloc[i]
                prev_day_data = df_filtered.iloc[i-1]
                date = day_data.name

                # Calculate average realized vol over a lookback period before determining regime
                avg_vol_lookback = df_filtered["Realized_Vol"].iloc[max(0, i-5):i].mean() if i > 0 else day_data["Realized_Vol"]
                if np.isnan(avg_vol_lookback): # Fallback if initial realized vol is NaN
                     avg_vol_lookback = day_data["VIX"] if not np.isnan(day_data["VIX"]) else 20 # Use VIX or default 20

                # Run Strategy Engine for the day's conditions
                regime, strategy, reason, tags, deploy, max_loss_target, risk_reward_target = run_strategy_engine_backtest(
                    day_data, avg_vol_lookback, portfolio_pnl, total_capital # Pass cumulative PnL and total capital
                )

                if trading_stopped: # If drawdown limit was reached on a previous day
                     # logger.debug(f"Trading stopped on {date.strftime('%Y-%m-%d')} due to drawdown limit.")
                     continue # Skip trading for this day

                # Filter by selected strategy for backtest if not 'All Strategies'
                if strategy is None or (strategy_choice != "All Strategies" and strategy != strategy):
                    # logger.debug(f"No strategy or selected strategy ({strategy_choice}) not matched ({strategy}) for {date.strftime('%Y-%m-%d')}. Skipping.")
                    continue # Skip if no strategy recommended or it doesn't match the filter


                # --- Simulate Trade PnL for the Day ---
                # This is a simplified PnL simulation. Real option PnL is complex.
                # Simulation Factors:
                # 1. Premium collected (based on Straddle Price, Qty, Costs)
                # 2. Loss due to market move (simulated based on Nifty % change, IV change, DTE)
                # 3. Edge/Win probability based on Vol Regime, IV/HV gap, Strategy, Risk/Reward

                # Simplified Premium Calculation (Assuming premium is roughly related to Straddle Price)
                simulated_straddle_price = day_data["Straddle_Price"]
                if simulated_straddle_price <= 0: # Avoid division by zero
                     # logger.warning(f"Simulated Straddle Price is zero or negative on {date.strftime('%Y-%m-%d')}. Skipping trade simulation for this day.")
                     continue # Cannot simulate premium

                # Calculate maximum possible lots based on deployable capital and straddle price
                # Ensure lot_size and simulated_straddle_price are positive to avoid division by zero
                if lot_size <= 0 or simulated_straddle_price <= 0:
                    lots = 0
                else:
                    # Max lots based on capital allocation
                    lots_by_capital = int(deploy / (simulated_straddle_price * lot_size))
                    # Max lots based on single trade max loss vs premium (ensure max loss target is met)
                    # If max loss is 2.5% of deployed capital, and premium is say 5% of deployed capital,
                    # Max loss is 50% of premium.
                    # A simple way: Limit lots such that max potential loss doesn't exceed max_loss_target
                    # Assuming max loss potential is related to the premium collected (e.g., MaxLossFactor * Premium)
                    max_loss_factor_heuristic = 0.6 # Heuristic: Max potential loss ~ 60% of premium for defined risk, higher for undefined
                    if strategy in ["Short Strangle", "Short Straddle"]:
                         max_loss_factor_heuristic = 1.0 # Can lose more than premium
                    # Max lots based on max loss target: Max_Loss_Target / (MaxLossFactor * Premium_per_lot)
                    if (max_loss_factor_heuristic * simulated_straddle_price * lot_size) > 0:
                         lots_by_loss_target = int(max_loss_target / (max_loss_factor_heuristic * simulated_straddle_price * lot_size))
                         lots = max(1, min(lots_by_capital, lots_by_loss_target)) # Use the minimum of the two, ensure at least 1 lot
                    else:
                        lots = max(1, lots_by_capital) # Fallback if loss calculation is problematic


                if lots <= 0:
                     # logger.debug(f"Calculated zero lots for deployment on {date.strftime('%Y-%m-%d')}. Skipping trade simulation.")
                     continue # Cannot trade zero lots

                simulated_premium_collected = simulated_straddle_price * lot_size * lots

                # Simulate Transaction Costs
                transaction_cost = simulated_premium_collected * base_transaction_cost_pct
                stt_cost = simulated_premium_collected * stt_pct
                total_costs = transaction_cost + stt_cost

                # Simulate Gross PnL before costs and shocks
                # This is the core of the backtest simulation's realism.
                # Let's make it probabilistic based on regime, IV/HV gap, and a bit of randomness.
                # High IV vs Realized Vol usually favors premium sellers.
                # Event days add randomness/risk of large moves.
                # Near expiry adds theta decay boost but also gamma risk.

                # Base PnL chance related to IV/HV gap: Positive gap favors premium sellers
                iv_hv_edge = (day_data["ATM_IV"] - day_data["Realized_Vol"]) # Raw gap
                iv_hv_edge_normalized = np.clip(iv_hv_edge / 5, -2, 2) # Normalize gap heuristic

                # PnL scaling factor based on edge and randomness
                pnl_scaling_factor = (0.5 + 0.2 * iv_hv_edge_normalized) * np.random.uniform(0.8, 1.2) # Base on edge + noise

                # Simulate win/loss outcome probability based on regime/edge
                # This is a heuristic - adjust based on real strategy performance
                win_probability = {
                    "LOW": 0.70, "MEDIUM": 0.65, "HIGH": 0.55, "EVENT-DRIVEN": 0.50
                }.get(regime, 0.60) # Base win rate by regime

                # Adjust win probability based on IV/HV edge - positive edge increases win chance
                win_probability = np.clip(win_probability + iv_hv_edge_normalized * 0.05, 0.40, 0.80) # Cap win probability

                # Adjust for DTE - very short DTE adds gamma risk (reduces win chance slightly)
                if day_data["Days_to_Expiry"] <= 3:
                     win_probability = np.clip(win_probability - 0.05, 0.35, 0.75)

                # Simulate the outcome
                is_win = np.random.rand() < win_probability

                if is_win:
                    # Simulate a winning PnL (capped by premium, maybe slightly more for R:R > 1)
                    # Win amount is a random fraction of premium collected
                    gross_pnl = simulated_premium_collected * np.random.uniform(0.5, 1.0) * risk_reward_target # Can win up to R:R * premium
                    # Cap win at 1.5x premium to be realistic for option selling
                    gross_pnl = min(gross_pnl, simulated_premium_collected * 1.5)

                else:
                    # Simulate a losing PnL (capped by Max Loss Target)
                    # Loss amount is a random fraction of Max Loss Target
                    gross_pnl = -max_loss_target * np.random.uniform(0.8, 1.2) # Can lose slightly more or less than target
                    # Cap loss by Max Loss Target
                    gross_pnl = max(gross_pnl, -max_loss_target * 1.2) # Cap at 1.2x target in simulation


                # Apply transaction costs
                net_pnl = gross_pnl - total_costs

                # Ensure PnL is within realistic bounds (especially against deployed capital)
                # A single trade should ideally not lose more than deployed capital (for defined risk strategies)
                # For undefined risk (like Short Strangle/Straddle), max loss can exceed premium/deployed capital
                # But we already capped based on max_loss_target which was % of deployable capital.
                # Let's ensure PnL doesn't exceed something like 2x deployed capital as a win or 1.5x deployed capital as a loss (rough bounds)
                net_pnl = np.clip(net_pnl, -deploy * 1.5, deploy * 2.0) # Hard clip based on deployed capital

                # Update cumulative portfolio PnL
                portfolio_pnl += net_pnl

                # Add result to backtest log
                backtest_results.append({
                    "Date": date,
                    "Regime": regime,
                    "Strategy": strategy,
                    "PnL": net_pnl,
                    "Capital_Deployed": deploy,
                    "Max_Loss_Target": max_loss_target,
                    "Risk_Reward_Target": risk_reward_target,
                    "Simulated_Lots": lots,
                    "Simulated_Premium": simulated_premium_collected,
                    "Total_Costs": total_costs,
                    "Cumulative_PnL": portfolio_pnl # Track cumulative PnL
                })

                if strategy == "Drawdown limit reached": # Check the signal to stop
                     trading_stopped = True
                     logger.info(f"Backtest trading stopped on {date.strftime('%Y-%m-%d')} due to simulated drawdown limit.")


            except Exception as e:
                logger.error(f"❌ Error in backtest loop on {date.strftime('%Y-%m-%d')}: {str(e)}", exc_info=True)
                # Continue loop even if one day fails


        backtest_df = pd.DataFrame(backtest_results)

        # --- Calculate Performance Metrics ---
        if backtest_df.empty:
            logger.warning("Backtest produced no trades.")
            return backtest_df, 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        total_pnl = backtest_df["PnL"].sum()
        num_trades = len(backtest_df)
        wins = backtest_df[backtest_df["PnL"] > 0]
        win_rate = len(wins) / num_trades if num_trades > 0 else 0

        # Calculate Drawdown from Cumulative PnL
        cumulative_pnl_series = backtest_df["Cumulative_PnL"]
        peak = cumulative_pnl_series.cummax()
        drawdown = peak - cumulative_pnl_series
        max_drawdown = drawdown.max()

        # Calculate Risk-Free Rate based on the backtest period length
        # Assuming daily data frequency ('B' - Business Days)
        num_trading_days = len(df_filtered) # Number of days in the backtest period
        # risk_free_return_total = (1 + risk_free_rate) ** num_trading_days - 1 # Compounded total risk-free return over period
        # Simple total risk-free return for the period (annual rate * years)
        num_years = num_trading_days / 252.0 if num_trading_days > 0 else 0
        risk_free_return_total = 0.06 * num_years # Using 6% annual rate

        # Calculate Returns for Sharpe/Sortino - PnL relative to initial capital
        # Using daily returns for Sharpe/Sortino calculations
        backtest_returns_daily = backtest_df["PnL"] / capital # PnL as a fraction of initial capital
        excess_returns_daily = backtest_returns_daily - (risk_free_rate if num_trading_days > 0 else 0) # Subtract daily risk-free rate

        # Annualize Sharpe/Sortino based on daily data (sqrt(252))
        sharpe_ratio = excess_returns_daily.mean() / excess_returns_daily.std() * np.sqrt(252) if excess_returns_daily.std() != 0 else 0
        # Sortino only considers downside deviation
        downside_returns = excess_returns_daily[excess_returns_daily < 0]
        sortino_ratio = excess_returns_daily.mean() / downside_returns.std() * np.sqrt(252) if downside_returns.std() != 0 else 0

        # Calmar Ratio: Total Return / Max Drawdown (as fraction of initial capital)
        total_return_pct = total_pnl / capital
        max_drawdown_pct = max_drawdown / capital
        calmar_ratio = total_return_pct / max_drawdown_pct if max_drawdown_pct > 0 else float('inf')


        # Performance by Strategy and Regime
        strategy_perf = backtest_df.groupby("Strategy")["PnL"].agg(['sum', 'count', 'mean']).reset_index()
        # Calculate win rate per strategy
        strategy_wins = backtest_df[backtest_df["PnL"] > 0].groupby("Strategy").size().reindex(strategy_perf["Strategy"]).fillna(0)
        strategy_perf["Win_Rate"] = strategy_wins / strategy_perf["count"]

        regime_perf = backtest_df.groupby("Regime")["PnL"].agg(['sum', 'count', 'mean']).reset_index()
        # Calculate win rate per regime
        regime_wins = backtest_df[backtest_df["PnL"] > 0].groupby("Regime").size().reindex(regime_perf["Regime"]).fillna(0)
        regime_perf["Win_Rate"] = regime_wins / regime_perf["count"]


        logger.info("Backtest completed successfully")
        return backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf
    except Exception as e:
        st.error(f"❌ Error running backtest: {str(e)}")
        logger.error(f"❌ Error running backtest: {str(e)}", exc_info=True)
        # Return empty results on error
        return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()


# --- Trading Functions ---
# Logic to build individual order requests for a strategy
def _build_strategy_orders(strategy_details: dict, real_data: dict, trade_quantity_lots: int) -> list | None:
    """
    Internal helper to build a list of 5paisa order dictionaries for a given strategy and quantity.
    Uses real-time option chain data to find scrip codes and prices.
    Returns a list of order dicts or None if data is missing/invalid.
    """
    logger.info(f"Building order requests for strategy: {strategy_details.get('Strategy')}")
    if not real_data or "option_chain" not in real_data or "atm_strike" not in real_data or "expiry_date_str" not in real_data:
        logger.error("Missing essential real-time data to build strategy orders.")
        return None

    option_chain_df = real_data["option_chain"]
    atm_strike = real_data["atm_strike"]
    # expiry = real_data["expiry_date_str"] # Expiry string not needed for placing order, timestamp or scrip code is key
    lot_size = 25 # Assuming NIFTY lot size

    if option_chain_df.empty:
         logger.error("Option chain data is empty. Cannot build strategy orders.")
         return None

    # Ensure trade_quantity_lots is a positive integer
    if not isinstance(trade_quantity_lots, int) or trade_quantity_lots <= 0:
         logger.error(f"Invalid trade quantity specified: {trade_quantity_lots}")
         return None

    # Define the strikes, types, and buy/sell action for each strategy leg
    strategy_name = strategy_details.get("Strategy")
    strikes_definition = [] # List of (strike_offset_from_atm, CPType, BuySell) relative to ATM strike

    # Note: Strike offsets are examples. In a real system, these might be dynamic based on IV/premium.
    if strategy_name == "Short Straddle":
        strikes_definition = [(0, "CE", "S"), (0, "PE", "S")]
    elif strategy_name == "Short Strangle":
        strikes_definition = [(100, "CE", "S"), (-100, "PE", "S")] # Example: 100 points OTM
    elif strategy_name == "Iron Condor":
        # Example: -200 PE Buy, -100 PE Sell, +100 CE Sell, +200 CE Buy
        strikes_definition = [(-200, "PE", "B"), (-100, "PE", "S"), (100, "CE", "S"), (200, "CE", "B")]
    elif strategy_name == "Iron Fly":
         # Example: -100 PE Buy, 0 PE Sell, 0 CE Sell, +100 CE Buy
         strikes_definition = [(-100, "PE", "B"), (0, "PE", "S"), (0, "CE", "S"), (100, "CE", "B")]
    elif strategy_name == "Butterfly Spread":
         # Example: -200 PE Buy, 0 PE Sell, +200 CE Buy (Simplified - usually all calls or all puts)
         # Let's use a Call Butterfly example: +100 C Buy, +200 C Sell (2x), +300 C Buy
         # Note: API might require specific ScripCodes for spread legs, not just strike/type
         # A standard Butterfly: Buy OTM Call, Sell 2x ATM/Near-ATM Call, Buy DITM Call (or Put equivalent)
         # Using a simple interpretation based on relative strikes as in the original code's concept
         # Let's stick to the original concept if exact strikes are needed: Buy K-200, Sell K, Buy K+200 (Calls or Puts)
         # The original code had PE Buy, CE Sell, CE Buy which is mixed? Let's use a simple Call Butterfly concept.
         # Call Butterfly: Buy K-X, Sell 2*K, Buy K+X (simplified strike notation)
         # Let's assume the strikes are relative to ATM as shown before
         # Original Streamlit example: call_sell_strike = atm_strike, call_buy_strike = atm_strike + 200, put_buy_strike = atm_strike - 200
         # This mixes calls and puts. A standard butterfly uses all calls or all puts.
         # Let's re-implement a standard Call Butterfly based on ATM:
         # Buy (ATM - 200) Call, Sell (ATM) Call (2x), Buy (ATM + 200) Call
         # This requires 3 legs. Let's adjust the definition based on typical spreads.
         # Redefining Butterfly based on common practice: Buy OTM, Sell ATM (2x), Buy ITM (for a Put Bfly reverse) or Buy ITM, Sell ATM (2x), Buy OTM (Call Bfly)
         # Let's stick to the original Streamlit code's *implied* legs based on strikes defined there:
         # call_sell_strike = atm_strike, call_buy_strike = atm_strike + 200, put_buy_strike = atm_strike - 200
         # This looks less like a standard butterfly and more like parts of different spreads.
         # Given the goal is to match the *working* notebook's logic concept, and the notebook only placed *one* test order,
         # the multi-leg strategy part in Streamlit is a new layer. The Streamlit code defined legs like this:
         # "Short Straddle", "Short Strangle", "Iron Condor", "Iron Fly", "Butterfly Spread", "Jade Lizard", "Calendar Spread"
         # Let's trust the Streamlit code's *intended* leg definitions relative to ATM Strike, even if they are simplified or unconventional spread structures.

         if strategy_name == "Butterfly Spread":
             # Based on the original Streamlit code's strike definitions:
             # sell ATM Call, buy OTM Call (+200), buy ITM Put (-200) -- This is an unusual mix.
             # Let's correct this to a standard Call Butterfly for demonstration:
             # Buy (ATM-200) Call, Sell (ATM) Call (2x), Buy (ATM+200) Call
              strikes_definition = [(-200, "CE", "B"), (0, "CE", "S", 2), (200, "CE", "B")] # Added quantity multiplier for middle leg
         elif strategy_name == "Jade Lizard":
              # Based on the original Streamlit code's strike definitions:
              # sell OTM Call (+100), sell OTM Put (-100), buy further OTM Put (-200)
              strikes_definition = [(100, "CE", "S"), (-100, "PE", "S"), (-200, "PE", "B")]
         elif strategy_name == "Calendar Spread":
              # Based on the original Streamlit code's strike definitions:
              # sell ATM Call (Near Month), buy ATM Call (Far Month)
              # This requires selecting expiries. The current fetch only gets the *first* expiry.
              # Implementing Calendar spreads robustly needs fetching *multiple* expiries.
              # Given the current API fetch only gets the first expiry, Calendar spread cannot be built correctly.
              # We should signal this is not supported with the current data fetch.
              logger.error("Calendar Spread requires fetching multiple expiries, which is not supported by current fetch_nifty_data_api.")
              st.error("❌ Cannot place Calendar Spread: Requires multiple expiries.")
              return None
         else:
              logger.error(f"Unsupported strategy for order building: {strategy_name}")
              st.error(f"❌ Unsupported strategy for trading: {strategy_name}")
              return None

    # --- Build order requests for each leg ---
    orders = []
    for strike_offset, cp_type, buy_sell, *qty_multiplier in strikes_definition:
        try:
            # Calculate the target strike rate
            target_strike = atm_strike + strike_offset
            # Round strike to the nearest valid strike step (e.g., 50 or 100 for NIFTY)
            # Assuming NIFTY strikes are in steps of 50 or 100. Find nearest multiple.
            # Let's just use the exact target_strike for lookup first, then try nearest if not found.
            # For simplicity, let's assume standard strikes and round to nearest 50/100.
            # A more robust way is to get the strike list from the option chain DF and find the closest valid strike.
            valid_strikes = option_chain_df["StrikeRate"].unique()
            closest_strike = valid_strikes[np.abs(valid_strikes - target_strike).argmin()]
            logger.debug(f"Target strike: {target_strike}, Closest valid strike found: {closest_strike}")

            # Find the data for the selected strike and type in the fetched option chain
            opt_data = option_chain_df[
                (option_chain_df['StrikeRate'] == closest_strike) &
                (option_chain_df['CPType'] == cp_type)
            ]

            if opt_data.empty:
                logger.error(f"No option data found in fetched chain for {cp_type} at strike {closest_strike}")
                st.error(f"❌ Trade failed: Missing option data for {cp_type} at strike {closest_strike}. Market might be closed or data not fetched correctly.")
                return None # Cannot build strategy if a leg is missing

            # Assuming the first match is the correct one for the current expiry
            scrip_code = int(opt_data['ScripCode'].iloc[0])
            latest_price = float(opt_data['LastRate'].iloc[0]) # Get the Last Traded Price

            # Determine the quantity for this leg
            leg_quantity = trade_quantity_lots * lot_size * (qty_multiplier[0] if qty_multiplier else 1)

            # Determine Order Type and Price
            # The original Streamlit code used Market Order (Price = 0).
            # For multi-leg, Market Orders can have significant slippage.
            # It's safer to use Limit Orders, but requires setting a price.
            # Let's default to Market Order (Price = 0) as in the original code, but add a warning.
            order_price = 0 # 0 for Market Order

            # Assemble the order dictionary for this leg
            order = {
                "OrderType": "SELL" if buy_sell == "S" else "BUY",
                "Exchange": "N", # Always NSE for NIFTY options
                "ExchangeType": "D", # Derivatives for options
                "ScripCode": scrip_code,
                "Qty": leg_quantity,
                "Price": order_price,
                "IsIntraday": False # Assuming positional trades for strategies
                # Add optional parameters like StopLossPrice, TakeProfitPrice if needed
            }
            orders.append(order)
            logger.debug(f"Built order leg: {order}")

        except Exception as e:
            logger.error(f"Error building order leg for strike offset {strike_offset}, {cp_type}, {buy_sell}: {e}", exc_info=True)
            st.error(f"❌ Error preparing order leg: {e}")
            return None # Abort building if any leg fails


    if not orders:
         logger.error("No orders were successfully built for the strategy.")
         return None

    logger.info(f"Successfully built {len(orders)} order requests for {strategy_name}.")
    return orders


def confirm_and_place_trade(client: FivePaisaClient, order_requests: list | None):
    """
    Places the list of prepared orders using the 5paisa client.
    Designed to be called after user confirmation.
    """
    if client is None or not client.get_access_token():
        logger.error("5paisa client is not initialized or not logged in. Cannot place orders.")
        return False, "5paisa client not logged in."

    if order_requests is None or not order_requests:
        logger.error("No order requests provided to place.")
        return False, "No order details available to place."

    logger.info(f"Attempting to place {len(order_requests)} orders.")
    placed_order_ids = []
    failed_orders = []

    # --- IMPORTANT WARNING FOR MARKET ORDERS ---
    # Since the code defaults Price=0 (Market Order), add a prominent warning.
    # This should also be clear in the confirmation UI.
    st.warning("⚠️ Placing Market Orders! Be aware of potential slippage.")

    for i, order in enumerate(order_requests):
        try:
            logger.info(f"Placing order leg {i+1}/{len(order_requests)}: {order}")
            # Call the place_order method
            response = client.place_order(
                OrderType=order.get("OrderType"),
                Exchange=order.get("Exchange"),
                ExchangeType=order.get("ExchangeType"),
                ScripCode=order.get("ScripCode"),
                Qty=order.get("Qty"),
                Price=order.get("Price"), # Use price from the order dict (0 for market)
                IsIntraday=order.get("IsIntraday", False) # Default to False if not specified
                # Add other parameters if included in order dict
            )

            logger.debug(f"Response for order leg {i+1}: {response}")

            if response and response.get("Status") == 0:
                logger.info(f"✅ Order leg {i+1} placed successfully.")
                placed_order_ids.append({
                    "ClientOrderID": response.get("ClientOrderID", "N/A"),
                    "RemoteOrderID": response.get("RemoteOrderID", "N/A"),
                    "ScripCode": order.get("ScripCode")
                })
            else:
                logger.error(f"❌ Order leg {i+1} failed. Status: {response.get('Status')}, Message: {response.get('Message', 'No specific message.')}")
                failed_orders.append({
                     "OrderDetails": order,
                     "Response": response
                })
                # Decide whether to stop placing subsequent legs if one fails
                # For now, let's continue to attempt placing all legs.
                st.error(f"❌ Failed to place order leg {i+1} (ScripCode {order.get('ScripCode')}): {response.get('Message', 'Unknown error')}")


        except Exception as e:
            logger.error(f"An unexpected error occurred during placing order leg {i+1}: {e}", exc_info=True)
            failed_orders.append({
                 "OrderDetails": order,
                 "Error": str(e)
            })
            st.error(f"❌ An error occurred placing order leg {i+1}: {e}")


    if not failed_orders:
        success_message = f"✅ All {len(order_requests)} order legs placed successfully."
        logger.info(success_message)
        # Log trade details to session state/journal if successful
        st.session_state.trades.append({
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Strategy": st.session_state.proposed_trade_details.get("Strategy", "N/A"),
            "Regime": st.session_state.proposed_trade_details.get("Regime", "N/A"),
            "Risk_Level": "High" if st.session_state.proposed_trade_details.get("Risk_Flags") else "Low",
            "Outcome": "Orders Placed (Pending Fill)", # Status after placing
            "Order_IDs": placed_order_ids
        })
        try:
             # Append to journal file
             trade_log_df = pd.DataFrame([st.session_state.trades[-1]]) # Get the last added trade
             journal_file = "trade_log.csv"
             trade_log_df.to_csv(journal_file, mode='a', header=not os.path.exists(journal_file), index=False)
             logger.info(f"Trade log saved to {journal_file}")
        except PermissionError:
             logger.error("Permission denied when writing to trade_log.csv")
             st.error("Cannot save trade_log.csv: Permission denied")
        except Exception as e:
             logger.error(f"Error saving trade log: {e}")
             st.error(f"Error saving trade log: {e}")


        return True, success_message
    else:
        error_message = f"❌ Failed to place {len(failed_orders)} out of {len(order_requests)} order legs."
        logger.error(error_message)
        # Optionally log partial success/failure details
        return False, error_message


def square_off_positions(client: FivePaisaClient):
    """
    Attempts to square off all open positions using the 5paisa client.
    """
    if client is None or not client.get_access_token():
        logger.error("5paisa client is not initialized or not logged in. Cannot square off positions.")
        st.error("❌ 5paisa client not logged in. Cannot square off.")
        return False, "5paisa client not logged in."

    try:
        logger.info("Attempting to square off all positions...")
        # Call the API method
        response = client.squareoff_all()
        logger.debug(f"Square off all response: {response}")

        # Check the response status - adjust based on actual API response format
        if response and response.get("Status") == 0: # Assuming Status 0 indicates success request sent
            success_message = "✅ Request to square off all positions sent successfully."
            logger.info(success_message)
            st.success(success_message + " Note: Check your order book/positions to confirm fill.")
            return True, success_message
        else:
            error_message = f"❌ Failed to send square off request. Status: {response.get('Status')}, Message: {response.get('Message', 'No specific message.')}" if isinstance(response, dict) else f"❌ Failed to send square off request: {str(response)}"
            logger.error(error_message)
            st.error(error_message)
            return False, error_message

    except Exception as e:
        logger.error(f"An unexpected error occurred during square off all: {e}", exc_info=True)
        st.error(f"❌ An unexpected error occurred during square off: {e}")
        return False, f"An unexpected error occurred: {e}"


def fetch_portfolio_data(client: FivePaisaClient, capital: float):
    """
    Fetches portfolio data (positions) and calculates summary metrics.
    Not cached as it needs to be real-time.
    """
    if client is None or not client.get_access_token():
        logger.warning("5paisa client not logged in. Cannot fetch portfolio data.")
        return {"weekly_pnl": 0, "margin_used": 0, "exposure": 0, "open_positions": pd.DataFrame()}

    try:
        logger.info("Fetching positions for portfolio overview...")
        positions_data = client.positions()
        logger.debug(f"Positions response: {positions_data}")

        if not positions_data or (isinstance(positions_data, dict) and positions_data.get("Message")):
            logger.info("No open positions found or positions API returned message.")
            # Handle message case vs empty list case
            if isinstance(positions_data, dict) and positions_data.get("Message"):
                 st.info(f"Positions API message: {positions_data.get('Message')}")
            return {"weekly_pnl": 0, "margin_used": 0, "exposure": 0, "open_positions": pd.DataFrame()}

        if isinstance(positions_data, list):
            positions_df = pd.DataFrame(positions_data)

            # Ensure required columns exist and are numeric where expected
            if not all(col in positions_df.columns for col in ["ProfitLoss", "MarginUsed", "BuyQuantity", "SellQuantity"]):
                 logger.warning("Positions data missing expected columns (ProfitLoss, MarginUsed, BuyQuantity, SellQuantity).")
                 # Attempt conversion/fill anyway for safety
                 for col in ["ProfitLoss", "MarginUsed", "BuyQuantity", "SellQuantity"]:
                      if col in positions_df.columns:
                           positions_df[col] = pd.to_numeric(positions_df[col], errors='coerce').fillna(0)
                      else:
                           positions_df[col] = 0 # Add missing column with zeros

            total_pnl = positions_df["ProfitLoss"].sum()
            total_margin = positions_df["MarginUsed"].sum()
            # A simple exposure estimate based on total quantity value might be complex.
            # Using MarginUsed as a proxy for capital commitment is common.
            # Alternatively, sum up (BuyQty + SellQty) * LTP for each position and divide by capital.
            # Let's stick to MarginUsed / Capital for simplicity as in the original Streamlit code's exposure concept.
            # Need LTP for each position to calculate total value exposure - complicates this function.
            # Let's calculate exposure based on MarginUsed vs Capital as before.
            exposure_pct = total_margin / capital * 100 if capital > 0 else 0

            logger.info(f"Portfolio data fetched: PnL={total_pnl}, Margin={total_margin}, Exposure={exposure_pct:.2f}%")
            return {
                "weekly_pnl": total_pnl, # Assuming 'ProfitLoss' is cumulative PnL for the day/week
                "margin_used": total_margin,
                "exposure": exposure_pct,
                "open_positions": positions_df # Return the DataFrame
            }
        else:
             logger.error(f"Positions API returned unexpected data type: {type(positions_data)}")
             st.error("❌ Positions API returned unexpected data.")
             return {"weekly_pnl": 0, "margin_used": 0, "exposure": 0, "open_positions": pd.DataFrame()}

    except Exception as e:
        logger.error(f"❌ Error fetching portfolio data: {str(e)}", exc_info=True)
        st.error(f"❌ Error fetching portfolio data: {str(e)}")
        return {"weekly_pnl": 0, "margin_used": 0, "exposure": 0, "open_positions": pd.DataFrame()}


# Sidebar Login and Controls - Corrected Emojis
with st.sidebar:
    st.header("🔑 5paisa Login")
    # Use session state to persist TOTP input temporarily if needed across reruns
    if 'totp_input' not in st.session_state:
        st.session_state.totp_input = ""
    st.session_state.totp_input = st.text_input("TOTP (from Authenticator App)", value=st.session_state.totp_input, type="password")

    if st.button("Login"):
        # Clear previous state if attempting login again
        st.session_state.logged_in = False
        st.session_state.client = None
        st.session_state.real_time_market_data = None
        st.session_state.data_source = "None"
        st.session_state.show_trade_confirmation = False # Hide confirmation on new login attempt
        st.session_state.proposed_trade_details = None

        if st.session_state.totp_input:
            st.session_state.client = initialize_5paisa_client(st.session_state.totp_input)
            if st.session_state.client:
                st.session_state.logged_in = True
                # Clear TOTP input after successful login attempt for security
                st.session_state.totp_input = "" # This might cause a rerun

        # Rerun the app after login attempt to update UI based on st.session_state.logged_in
        st.rerun()


    if st.session_state.logged_in:
        st.success("✅ Logged in successfully")
        st.header("⚙️ Trading Controls")
        capital = st.number_input("Capital (₹)", min_value=100000, value=1000000, step=100000, key="capital_input")
        risk_tolerance = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"], index=1, key="risk_tolerance_select")
        forecast_horizon = st.slider("Forecast Horizon (days)", 1, 30, 7, key="forecast_horizon_slider")
        # dte_preference = st.slider("DTE Preference (days)", 7, 30, 15, key="dte_preference_slider") # Not directly used in provided strategy logic
        st.markdown("**Backtest Parameters**")
        start_date = st.date_input("Start Date", value=pd.to_datetime("2024-01-01").date(), key="backtest_start_date") # Store as date
        end_date = st.date_input("End Date", value=pd.to_datetime("2025-04-29").date(), key="backtest_end_date") # Store as date
        strategy_choice = st.selectbox("Strategy for Backtest Filter", ["All Strategies", "Butterfly Spread", "Iron Condor", "Iron Fly", "Short Strangle", "Calendar Spread", "Jade Lizard", "Short Straddle"], key="strategy_filter_select")

        # Convert date inputs to datetime objects for comparison with index
        start_date_dt = datetime.combine(start_date, datetime.min.time())
        end_date_dt = datetime.combine(end_date, datetime.min.time())

        run_button = st.button("Run Analysis", key="run_analysis_button")

        st.markdown("---") # Separator

        if st.button("Square Off All Positions", key="square_off_button"):
            # Add a confirmation step before squaring off all positions
            st.session_state.confirm_square_off = True # Use state to trigger confirmation UI
            st.rerun() # Rerun to show confirmation

        if st.session_state.get('confirm_square_off', False):
             st.warning("⚠️ ARE YOU SURE YOU WANT TO SQUARE OFF ALL POSITIONS?")
             col_sq_yes, col_sq_no = st.columns(2)
             with col_sq_yes:
                 if st.button("Yes, Square Off All", key="confirm_square_off_yes"):
                     success, message = square_off_positions(st.session_state.client)
                     # The square_off_positions function already displays success/error messages
                     st.session_state.confirm_square_off = False # Hide confirmation
                     st.rerun() # Rerun to update UI and hide confirmation
             with col_sq_no:
                 if st.button("No, Cancel", key="confirm_square_off_no"):
                     st.info("Square off cancelled.")
                     st.session_state.confirm_square_off = False # Hide confirmation
                     st.rerun() # Rerun to update UI

# Main Execution
if not st.session_state.logged_in:
    st.markdown("<h1 style='color: #e94560; text-align: center;'>🛡️ VolGuard Pro: Your AI Trading Copilot</h1>", unsafe_allow_html=True)
    st.info("Please login to 5paisa from the sidebar to proceed.")

else: # User is logged in
    st.markdown("<h1 style='color: #e94560; text-align: center;'>🛡️ VolGuard Pro: Your AI Trading Copilot</h1>", unsafe_allow_html=True)

    # --- Load Data and Run Analysis on Button Click ---
    if run_button:
        st.session_state.backtest_run = False # Reset backtest state
        st.session_state.backtest_results = None
        st.session_state.violations = 0
        st.session_state.journal_complete = False # Reset journal state if needed on new run
        st.session_state.show_trade_confirmation = False # Hide trade confirmation on new analysis run
        st.session_state.proposed_trade_details = None # Clear proposed trade

        with st.spinner("Running VolGuard Analysis and fetching data..."):
            # Load data (attempts API first, falls back to CSV)
            # load_data function updates st.session_state.real_time_market_data and st.session_state.data_source
            df, _, _ = load_data(st.session_state.client) # Get combined DF

            if not df.empty:
                # Generate features on the combined historical+live data
                df_features = generate_features(df, st.session_state.real_time_market_data, capital)

                if df_features is not None and not df_features.empty:
                    # Run Backtest on the feature-rich DataFrame
                    backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf = run_backtest(
                        df_features, capital, strategy_choice, start_date_dt, end_date_dt # Pass datetime objects
                    )
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

                    # Generate Strategy proposal using the latest data and forecast
                    # Ensure df_features has the latest row for strategy generation
                    if not df_features.empty:
                         # Need to run forecast first to get forecast_log and realized_vol
                         # Note: Forecast should use the full df_features, not just the last row
                         forecast_log, garch_vols, xgb_vols, blended_vols, realized_vol_latest, confidence_score, rmse, feature_importances = forecast_volatility_future(df_features, forecast_horizon)

                         if forecast_log is not None:
                              # Pass necessary data to strategy generator
                              st.session_state.proposed_trade_details = generate_trading_strategy(
                                   df_features, # Pass the feature DF
                                   forecast_log,
                                   realized_vol_latest, # Use the latest realized vol for strategy logic
                                   risk_tolerance,
                                   confidence_score,
                                   capital
                               )
                         else:
                              st.warning("⚠️ Could not generate trading strategy: Volatility forecast failed.")
                              st.session_state.proposed_trade_details = None # Clear strategy if forecast fails

                else:
                     st.error("❌ Feature generation failed. Cannot proceed with analysis.")
                     st.session_state.backtest_run = False
                     st.session_state.backtest_results = None
                     st.session_state.proposed_trade_details = None

            else:
                 st.error("❌ Data loading failed. Cannot proceed with analysis.")
                 st.session_state.backtest_run = False
                 st.session_state.backtest_results = None
                 st.session_state.proposed_trade_details = None


    # --- Display Tabs ---
    tabs = st.tabs(["📊 Snapshot", "📈 Forecast", "💡 Strategy", "💼 Portfolio", "📒 Journal", "🧪 Backtest"])

    # Snapshot Tab
    with tabs[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Market Snapshot")
        # Display data source
        st.info(f"Data Source: **{st.session_state.data_source}**")

        # Display latest market data if available (from real_time_market_data or last row of df_features)
        latest_data = None
        if st.session_state.real_time_market_data and st.session_state.real_time_market_data.get("nifty_spot") is not None:
             latest_data = st.session_state.real_time_market_data # Prefer API data
             last_date = datetime.now().strftime("%d-%b-%Y %H:%M") # Indicate freshness
        elif st.session_state.backtest_run and 'df_features' in locals() and not df_features.empty:
             latest_data = df_features.iloc[-1].to_dict() # Fallback to last data point in DF
             last_date = df_features.index[-1].strftime("%d-%b-%Y") # Use date from index
        else:
             st.warning("⚠️ Market data not available. Run analysis first.")


        if latest_data:
             nifty_spot = latest_data.get('nifty_spot', latest_data.get('NIFTY_Close')) # Use API key then DF key
             vix = latest_data.get('vix', latest_data.get('VIX')) # Use API key then DF key
             pcr = latest_data.get('pcr', latest_data.get('PCR'))
             straddle_price = latest_data.get('straddle_price', latest_data.get('Straddle_Price'))
             vix_change_pct = latest_data.get('vix_change_pct', latest_data.get('VIX_Change_Pct'))
             max_pain_strike = latest_data.get('max_pain_strike') # Only from API
             max_pain_diff_pct = latest_data.get('max_pain_diff_pct') # Only from API
             expiry_date_str = latest_data.get('expiry_date_str') # Only from API

             if nifty_spot is not None and vix is not None:
                 regime = "LOW" if vix < 15 else "MEDIUM" if vix < 20 else "HIGH"
                 regime_class = {"LOW": "regime-low", "MEDIUM": "regime-medium", "HIGH": "regime-high"}[regime]
                 st.markdown(f'<div style="text-align: center;"><div class="gauge">{regime}</div><div style="margin-top: 5px;">Market Regime</div></div>', unsafe_allow_html=True)

                 col1, col2, col3, col4 = st.columns(4)
                 with col1:
                     # Calculate daily % change for Nifty if previous day exists in df_features
                     nifty_change_pct = "N/A"
                     if st.session_state.backtest_run and 'df_features' in locals() and len(df_features) >= 2:
                          prev_nifty = df_features.iloc[-2]["NIFTY_Close"]
                          if prev_nifty != 0:
                                nifty_change_pct = f"{(nifty_spot - prev_nifty)/prev_nifty*100:+.2f}%"
                          else:
                                nifty_change_pct = "+0.00%" # Avoid division by zero
                     st.metric("NIFTY 50", f"{nifty_spot:,.2f}", nifty_change_pct)
                 with col2:
                      st.metric("India VIX", f"{vix:.2f}%", f"{vix_change_pct:+.2f}%" if vix_change_pct is not None else "N/A")
                 with col3:
                      st.metric("PCR", f"{pcr:.2f}" if pcr is not None else "N/A")
                 with col4:
                      st.metric("Straddle Price", f"₹{straddle_price:,.2f}" if straddle_price is not None else "N/A")

                 st.markdown(f"**Last Updated**: {last_date}")
                 if expiry_date_str:
                     st.markdown(f"**Current Expiry**: {expiry_date_str}")
                 if max_pain_strike is not None:
                     st.markdown(f"**Max Pain Strike**: {max_pain_strike} ({max_pain_diff_pct:.2f}% diff)")

             else:
                  st.warning("⚠️ NIFTY or VIX data not available in fetched data.")

             # Display Option Chain Head if API data was successful and has the dataframe
             if st.session_state.real_time_market_data and 'option_chain' in st.session_state.real_time_market_data and not st.session_state.real_time_market_data['option_chain'].empty:
                 st.markdown("### Option Chain (Current Expiry)")
                 cols_to_display_oc = ['StrikeRate', 'CPType', 'LastRate', 'OpenInterest', 'ScripCode']
                 oc_df = st.session_state.real_time_market_data['option_chain']
                 # Ensure columns exist before selecting
                 actual_cols_oc = [col for col in cols_to_display_oc if col in oc_df.columns]
                 st.dataframe(oc_df[actual_cols_oc].head(), use_container_width=True)
             elif st.session_state.data_source == "CSV (FALLBACK)":
                  st.info("Option Chain data is only available from the live API, which failed to fetch.")


        st.markdown('</div>', unsafe_allow_html=True)

    # Forecast Tab
    with tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📈 Volatility Forecast")
        if st.session_state.backtest_run and 'df_features' in locals() and not df_features.empty:
             with st.spinner("Predicting market volatility..."):
                # Pass df_features which contains the features
                forecast_log, garch_vols_plot, xgb_vols_plot, blended_vols_plot, realized_vol_latest, confidence_score, rmse, feature_importances = forecast_volatility_future(df_features, forecast_horizon)
             if forecast_log is not None and not forecast_log.empty:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Blended Volatility", f"{np.mean(blended_vols_plot):.2f}%")
                with col2:
                    st.metric("Latest Realized Volatility", f"{realized_vol_latest:.2f}%" if realized_vol_latest is not None else "N/A")
                with col3:
                    st.metric("Model RMSE", f"{rmse:.2f}%" if rmse is not None else "N/A")
                st.markdown(f'<div style="text-align: center;"><div class="gauge">{int(confidence_score) if confidence_score is not None else "N/A"}%</div><div style="margin-top: 5px;">Confidence</div></div>', unsafe_allow_html=True)

                # Prepare data for chart, handling potential NaNs if one model failed
                chart_data = pd.DataFrame({
                    "GARCH": garch_vols_plot,
                    "XGBoost": xgb_vols_plot,
                    "Blended": blended_vols_plot
                }, index=forecast_log["Date"]) # Use Date from forecast_log

                st.line_chart(chart_data, color=["#e94560", "#00d4ff", "#ffcc00"])

                if feature_importances is not None and not feature_importances.empty:
                    st.markdown("### Feature Importance (XGBoost)")
                    # Convert Series to DataFrame for display
                    feature_importance_df = feature_importances.reset_index()
                    feature_importance_df.columns = ['Feature', 'Importance']
                    st.dataframe(feature_importance_df, use_container_width=True)
             else:
                 st.warning("⚠️ Volatility forecasting failed or returned no data.")

        else:
            st.info("Run the analysis first to view volatility forecast.")
        st.markdown('</div>', unsafe_allow_html=True)


    # Strategy Tab
    with tabs[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("💡 Trading Strategies")

        # Check if proposed strategy details are available (set by the 'Run Analysis' button)
        if st.session_state.proposed_trade_details is not None:
             strategy = st.session_state.proposed_trade_details

             # Display risk flags if any
             if strategy.get("Risk_Flags"):
                 st.markdown(f'<div class="alert-banner">⚠️ Risk Flags: {", ".join(strategy["Risk_Flags"])}</div>', unsafe_allow_html=True)

             # Display strategy card
             regime_class = {
                 "LOW": "regime-low", "MEDIUM": "regime-medium", "HIGH": "regime-high", "EVENT-DRIVEN": "regime-event"
             }.get(strategy.get("Regime"), "regime-low") # Default to low

             st.markdown('<div class="strategy-carousel">', unsafe_allow_html=True) # Using carousel style for a single card for now
             st.markdown(f"""
                 <div class="strategy-card">
                     <h4>{strategy.get("Strategy", "N/A")}</h4>
                     <span class="regime-badge {regime_class}">{strategy.get("Regime", "N/A")}</span>
                     <p><b>Reason:</b> {strategy.get("Reason", "N/A")}</p>
                     <p><b>Confidence:</b> {strategy.get("Confidence", 0.0):.2f}</p>
                     <p><b>Risk-Reward:</b> {strategy.get("Risk_Reward", 0.0):.2f}:1</p>
                     <p><b>Deployed Capital (Target):</b> ₹{strategy.get("Deploy", 0):,.0f}</p>
                     <p><b>Max Loss (Target):</b> ₹{strategy.get("Max_Loss", 0):,.0f}</p>
                     <p><b>Tags:</b> {', '.join(strategy.get("Tags", ["N/A"]))}</p>
                 </div>
             """, unsafe_allow_html=True)
             st.markdown('</div>', unsafe_allow_html=True)


             # --- Trade Confirmation Section (Revealed by button click) ---
             if not st.session_state.show_trade_confirmation:
                  # Show the initial "Trade Now" button only if API data (needed for order building) is available
                  if st.session_state.real_time_market_data is not None and not st.session_state.real_time_market_data.get('option_chain', pd.DataFrame()).empty:
                       if st.button("🚀 Trade Now", key="trade_now_button"):
                            st.session_state.show_trade_confirmation = True
                            # Rerun to show the confirmation UI
                            st.rerun()
                  else:
                       st.warning("⚠️ Cannot trade now: Real-time market data (Option Chain) not available via API.")


             if st.session_state.show_trade_confirmation:
                  st.markdown("---")
                  st.subheader("Confirm Trade Details")
                  st.warning("⚠️ **REVIEW CAREFULLY:** You are about to place a real trade.")

                  # Display proposed strategy again
                  st.markdown(f"**Proposed Strategy:** {strategy.get('Strategy', 'N/A')} ({strategy.get('Regime', 'N/A')} Regime)")
                  st.markdown(f"**Target Capital Allocation:** ₹{strategy.get('Deploy', 0):,.0f}")
                  st.markdown(f"**Target Max Loss:** ₹{strategy.get('Max_Loss', 0):,.0f}")

                  # Get quantity input from user (default based on target capital, but allow adjustment)
                  simulated_straddle_price = st.session_state.real_time_market_data.get('straddle_price', 0) if st.session_state.real_time_market_data else 0
                  lot_size = 25 # NIFTY lot size
                  target_deploy = strategy.get('Deploy', 0)

                  # Suggest initial quantity based on deployed capital and current straddle price
                  suggested_lots = 0
                  if simulated_straddle_price > 0 and lot_size > 0:
                       suggested_lots = max(1, int(target_deploy / (simulated_straddle_price * lot_size)))
                  else:
                       suggested_lots = 1 # Default to 1 lot if price is zero

                  trade_quantity_lots = st.number_input(
                      "Select Quantity (Lots)",
                      min_value=1,
                      value=suggested_lots,
                      step=1,
                      key="trade_quantity_input"
                  )

                  # --- Order Type and Price (Simplified: Only Market Order shown) ---
                  # Option to add Limit Order here later if needed
                  st.info("Order Type: Market Order (Price = 0). Be aware of potential slippage.")
                  order_price_input = 0 # Market Order

                  # --- Build the actual order requests for API ---
                  order_requests = _build_strategy_orders(
                      strategy,
                      st.session_state.real_time_market_data, # Pass the real-time data dict
                      trade_quantity_lots # Pass user-selected quantity
                  )

                  if order_requests: # Proceed only if order requests were successfully built
                      st.markdown("---")
                      st.markdown(f"**Order Legs to be Placed ({len(order_requests)}):**")
                      # Display order details for confirmation (optional, can be verbose)
                      # for i, req in enumerate(order_requests):
                      #      st.write(f"Leg {i+1}: {req['OrderType']} {req['Qty']} units, ScripCode: {req['ScripCode']}, Price: {req['Price']}")
                      # Display a summary table
                      order_summary_data = []
                      for req in order_requests:
                           order_summary_data.append({
                               'Action': req.get('OrderType'),
                               'Qty (Units)': req.get('Qty'),
                               'ScripCode': req.get('ScripCode'),
                               'Exchange': req.get('Exchange'),
                               'ExchType': req.get('ExchangeType'),
                               'Price': req.get('Price'), # Shows 0 for Market
                               'IsIntraday': req.get('IsIntraday')
                           })
                      st.dataframe(pd.DataFrame(order_summary_data), use_container_width=True)


                      # Confirmation buttons
                      col_confirm_yes, col_confirm_no = st.columns(2)
                      with col_confirm_yes:
                           if st.button("✅ CONFIRM & PLACE ORDER", key="confirm_place_order_button"):
                                # Place the orders
                                with st.spinner(f"Placing {len(order_requests)} order legs..."):
                                     success, message = confirm_and_place_trade(st.session_state.client, order_requests)
                                     # Messages are displayed by the confirm_and_place_trade function
                                st.session_state.show_trade_confirmation = False # Hide confirmation after attempt
                                # Rerun to update UI and potentially portfolio tab
                                st.rerun()

                      with col_confirm_no:
                           if st.button("❌ Cancel Trade", key="cancel_trade_button"):
                                st.info("Trade placement cancelled.")
                                st.session_state.show_trade_confirmation = False # Hide confirmation
                                # Rerun to update UI
                                st.rerun()

                  else:
                       st.error("❌ Could not build order requests for the selected strategy and quantity. Check data availability and strategy definition.")


        else: # Strategy details not available
            st.info("Run the analysis first to generate a trading strategy.")

        st.markdown('</div>', unsafe_allow_html=True)

    # Portfolio Tab - Corrected Emojis
    with tabs[3]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("💼 Portfolio Overview")
        # Fetch latest portfolio data (not cached)
        portfolio_data = fetch_portfolio_data(st.session_state.client, capital if 'capital' in locals() else 1000000) # Use default capital if not set

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total P&L (Today)", f"₹{portfolio_data.get('weekly_pnl', 0):,.2f}") # Assuming API ProfitLoss is for today/current session
        with col2:
            st.metric("Margin Used", f"₹{portfolio_data.get('margin_used', 0):,.2f}")
        with col3:
            st.metric("Exposure (vs Capital)", f"{portfolio_data.get('exposure', 0):.2f}%")

        st.markdown("### Open Positions")
        if not portfolio_data.get("open_positions", pd.DataFrame()).empty:
             st.dataframe(portfolio_data["open_positions"], use_container_width=True)
        else:
             st.info("No open positions found.")

        st.markdown('</div>', unsafe_allow_html=True)


    # Journal Tab - Corrected Emojis
    with tabs[4]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📒 Discipline Hub")
        st.info("Use this section to reflect on your trades and track discipline.")

        with st.form(key="journal_form"):
            st.markdown("### Add New Journal Entry")
            # You might want to link this to a specific trade that was placed
            # For simplicity, let's allow a general entry
            entry_date = st.date_input("Entry Date", datetime.now().date(), key="journal_entry_date")
            reason_strategy = st.selectbox("Reflecting on Strategy Choice:", ["High IV", "Low Risk", "Event Opportunity", "Backtest Performance", "Other"], key="journal_reason_strategy")
            override_risk = st.radio("Did this trade/decision override significant risk flags?", ("No", "Yes"), key="journal_override_risk")
            trade_outcome_reflection = st.selectbox("Trade Outcome Reflection (if applicable):", ["Pending", "Profit", "Loss", "Breakeven", "N/A"], key="journal_outcome_reflection")
            lessons_learned = st.text_area("Lessons Learned / Thoughts", key="journal_lessons")

            submit_journal = st.form_submit_button("Save Journal Entry")

            if submit_journal:
                # Simple discipline score calculation (can be customized)
                score = 0
                if override_risk == "No": score += 3
                if reason_strategy != "Other": score += 2 # Slightly lower score for 'Other'
                if trade_outcome_reflection in ["Profit", "Breakeven"]: score += 1 # Small bonus for non-loss (reflection, not outcome itself)
                if lessons_learned: score += 3 # Bonus for thoughtful reflection
                if st.session_state.violations > 0 and override_risk == "Yes": score = max(0, score - 3) # Penalty for overriding with active violations

                score = np.clip(score, 0, 10) # Ensure score is 0-10

                journal_entry_data = {
                    "Date": entry_date.strftime("%Y-%m-%d"), # Save date only
                    "Time": datetime.now().strftime("%H:%M:%S"), # Save time separately
                    "Strategy_Reason_Reflection": reason_strategy,
                    "Override_Risk_Flags": override_risk,
                    "Trade_Outcome_Reflection": trade_outcome_reflection,
                    "Lessons_Learned": lessons_learned,
                    "Discipline_Score": score
                }

                journal_df_new = pd.DataFrame([journal_entry_data])
                journal_file = "journal_log.csv"

                try:
                    # Append to the CSV file
                    journal_df_new.to_csv(journal_file, mode='a', header=not os.path.exists(journal_file), index=False)
                    st.success(f"✅ Journal Entry Saved! Discipline Score: {score}/10")
                    # Reset violations if a good journal score is achieved after violations
                    if score >= 7 and st.session_state.violations > 0:
                         st.session_state.violations = 0
                         st.info("🔒 Discipline Lock cleared.")

                except PermissionError:
                    logger.error("Permission denied when writing to journal_log.csv")
                    st.error("❌ Cannot save journal entry: Permission denied.")
                except Exception as e:
                    logger.error(f"Error saving journal entry: {e}")
                    st.error(f"❌ Error saving journal entry: {e}")

                # Rerun to clear the form and show updated past entries
                st.rerun()


        st.markdown("### Past Entries")
        journal_file = "journal_log.csv"
        if os.path.exists(journal_file):
            try:
                journal_df_past = pd.read_csv(journal_file)
                # Display in reverse chronological order
                st.dataframe(journal_df_past.sort_values(by=["Date", "Time"], ascending=False), use_container_width=True)
            except pd.errors.EmptyDataError:
                 st.info("No past journal entries found.")
            except Exception as e:
                st.error(f"❌ Error reading journal_log.csv: {e}")
                logger.error(f"Error reading journal_log.csv: {e}")
        else:
            st.info("No past journal entries found.")

        # Display Trade Log separately (Optional - could combine with Journal or keep separate)
        st.markdown("### Trade History (API Placement Log)")
        trade_log_file = "trade_log.csv"
        if os.path.exists(trade_log_file):
             try:
                  trade_log_df = pd.read_csv(trade_log_file)
                  st.dataframe(trade_log_df.sort_values(by="Date", ascending=False), use_container_width=True)
             except pd.errors.EmptyDataError:
                  st.info("No API trades logged yet.")
             except Exception as e:
                  st.error(f"❌ Error reading trade_log.csv: {e}")
                  logger.error(f"Error reading trade_log.csv: {e}")
        else:
             st.info("No API trades logged yet.")


        st.markdown('</div>', unsafe_allow_html=True)


    # Backtest Tab - Corrected Emojis
    with tabs[5]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧪 Backtest Results")
        if st.session_state.backtest_run and st.session_state.backtest_results is not None:
            results = st.session_state.backtest_results
            if results["backtest_df"].empty:
                st.warning("⚠️ No trades generated for the selected parameters in the backtest. Try adjusting the date range, capital, or strategy filter.")
            else:
                st.info(f"Backtest performed on data from **{st.session_state.data_source}**")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total P&L", f"₹{results['total_pnl']:,.2f}")
                with col2:
                    st.metric("Win Rate", f"{results['win_rate']*100:.2f}%")
                with col3:
                    st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                with col4:
                    st.metric("Max Drawdown", f"₹{results['max_drawdown']:,.2f}")

                st.markdown("### Cumulative P&L (Backtest)")
                # Ensure index is datetime for charting
                if not results["backtest_df"].empty:
                     cum_pnl_series = results["backtest_df"]["PnL"].cumsum()
                     st.line_chart(cum_pnl_series, color="#e94560")
                else:
                     st.warning("No PnL data to plot.")


                st.markdown("### Performance by Strategy (Backtest)")
                if not results["strategy_perf"].empty:
                    st.dataframe(results["strategy_perf"].style.format({
                        "sum": "₹{:,.2f}",
                        "mean": "₹{:,.2f}",
                        "Win_Rate": "{:.2%}",
                        "count": "{:,.0f}"
                    }), use_container_width=True)
                else:
                     st.info("No strategy performance data generated.")


                st.markdown("### Performance by Regime (Backtest)")
                if not results["regime_perf"].empty:
                    st.dataframe(results["regime_perf"].style.format({
                        "sum": "₹{:,.2f}",
                        "mean": "₹{:,.2f}",
                        "Win_Rate": "{:.2%}",
                         "count": "{:,.0f}"
                    }), use_container_width=True)
                else:
                     st.info("No regime performance data generated.")


                st.markdown("### Detailed Backtest Trades")
                if not results["backtest_df"].empty:
                     st.dataframe(results["backtest_df"].style.format({
                         "PnL": "₹{:,.2f}",
                         "Capital_Deployed": "₹{:,.2f}",
                         "Max_Loss_Target": "₹{:,.2f}",
                         "Risk_Reward_Target": "{:.2f}",
                         "Simulated_Premium": "₹{:,.2f}",
                         "Total_Costs": "₹{:,.2f}",
                         "Cumulative_PnL": "₹{:,.2f}"
                     }), use_container_width=True)
                else:
                     st.info("No detailed backtest trade data generated.")

        else:
            st.info("Run the analysis to view backtest results.")
        st.markdown('</div>', unsafe_allow_html=True)


    st.markdown('<div class="footer">Built with ❤️ by Shritish Shukla & Salman Azim | © 2025 VolGuard</div>', unsafe_allow_html=True)
