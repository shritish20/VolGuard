import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime, timedelta
import logging
from py5paisa import FivePaisaClient
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os
import pytz # Import for timezone handling

# --- Configuration ---
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Timezone for logging/timestamps (India Standard Time)
IST = pytz.timezone('Asia/Kolkata')

# GitHub Raw Data URLs
NIFTY_HISTORICAL_URL = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
VIX_HISTORICAL_URL = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"

# Strategy Parameters (can be adjusted)
LOT_SIZE = 50 # NIFTY Lot Size (adjust if needed)
BASE_TRANSACTION_COST_PCT = 0.002 # 0.2%
STT_PCT = 0.0005 # 0.05%
RISK_FREE_RATE_DAILY = 0.06 / 252 # Approx risk-free rate per trading day

# --- Session State Initialization ---
# Use st.session_state for state persistence within a single session
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "client" not in st.session_state:
    st.session_state.client = None
if "analysis_run" not in st.session_state:
    st.session_state.analysis_run = False
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None
if "violations" not in st.session_state:
    st.session_state.violations = 0
if "journal_complete" not in st.session_state:
    st.session_state.journal_complete = False
# Use session state for logs instead of writing to filesystem
if "trade_log" not in st.session_state:
    st.session_state.trade_log = []
if "journal_log" not in st.session_state:
    st.session_state.journal_log = []
if "vix_history" not in st.session_state:
    st.session_state.vix_history = pd.DataFrame(columns=["Date", "VIX"]) # Store VIX history if needed, though can be derived from main data
if "real_time_data" not in st.session_state:
    st.session_state.real_time_data = None
if "historical_data" not in st.session_state:
     st.session_state.historical_data = None
if "feature_data" not in st.session_state:
     st.session_state.feature_data = None
if "forecast_data" not in st.session_state: # Initialize forecast_data state
     st.session_state.forecast_data = None
if "strategy_recommendation" not in st.session_state: # Initialize strategy_recommendation state
     st.session_state.strategy_recommendation = None
if "strategy_errors" not in st.session_state: # Initialize strategy_errors state
     st.session_state.strategy_errors = []
if "portfolio_data" not in st.session_state: # Initialize portfolio_data state
     st.session_state.portfolio_data = None
if "generated_orders" not in st.session_state: # Initialize generated_orders state
    st.session_state.generated_orders = None
if "generated_order_error" not in st.session_state: # Initialize generated_order_error state
    st.session_state.generated_order_error = None


# --- Initialize button state ---
# This ensures the variable exists even if the button is not rendered
run_analysis_button = False


# --- FivePaisa Client Initialization ---
def initialize_5paisa_client(totp_code):
    """Initializes the 5paisa client and gets session token."""
    try:
        logger.info("Attempting 5paisa client initialization")
        # Ensure all required keys are in secrets
        required_keys = ["APP_NAME", "APP_SOURCE", "USER_ID", "PASSWORD", "USER_KEY", "ENCRYPTION_KEY", "CLIENT_CODE", "PIN"]
        for key in required_keys:
            if not st.secrets.get("fivepaisa", {}).get(key):
                 logger.error(f"Missing 5paisa secret key: {key}")
                 st.error(f"Configuration Error: Missing 5paisa secret '{key}'. Please add it to your Streamlit secrets.")
                 return None

        cred = {
            "APP_NAME": st.secrets["fivepaisa"]["APP_NAME"],
            "APP_SOURCE": st.secrets["fivepaisa"]["APP_SOURCE"],
            "USER_ID": st.secrets["fivepaisa"]["USER_ID"],
            "PASSWORD": st.secrets["fivepaisa"]["PASSWORD"],
            "USER_KEY": st.secrets["fivepaisa"]["USER_KEY"],
            "ENCRYPTION_KEY": st.secrets["fivepaisa"]["ENCRYPTION_KEY"]
        }
        client = FivePaisaClient(cred=cred)
        logger.info("Calling get_totp_session...")
        client.get_totp_session(
            st.secrets["fivepaisa"]["CLIENT_CODE"],
            totp_code,
            st.secrets["fivepaisa"]["PIN"]
        )

        # Check if access token is obtained - py5paisa client stores this internally
        # A simple check is if the client object was created without exceptions
        # More robust would be calling a method that requires auth, but this is often complex
        logger.info("5paisa client potentially initialized. Checking access token...")
        # py5paisa v3 doesn't expose get_access_token() directly on the client object in the same way
        # We'll rely on subsequent API calls to confirm authentication status.
        # For now, consider it 'initialized' if the above steps complete without error.
        logger.info("5paisa client initialization process completed.")
        return client

    except Exception as e:
        logger.error(f"Error initializing 5paisa client: {str(e)}")
        st.error(f"Failed to connect to 5paisa: {e}. Check your credentials and TOTP.")
        return None

# --- Data Fetching ---
def fetch_nifty_data_5paisa(client):
    """Fetches real-time NIFTY 50 and India VIX data from 5paisa."""
    if not client:
        logger.error("5paisa client not initialized.")
        return None

    try:
        logger.info("Fetching real-time data from 5paisa API")
        # Using ScripCode from 5paisa documentation
        req_list = [
            {"Exch": "N", "ExchType": "C", "ScripCode": 999920000},  # NIFTY 50
            {"Exch": "N", "ExchType": "C", "ScripCode": 999920005}   # India VIX
        ]
        market_feed = client.fetch_market_feed(req_list)

        if not market_feed or "Data" not in market_feed or len(market_feed["Data"]) < 2:
            logger.warning("Failed to fetch NIFTY 50 or India VIX from market feed.")
            # Attempting fetch_live_feed as an alternative if market_feed fails
            try:
                 logger.info("Trying fetch_live_feed as alternative...")
                 live_feed = client.fetch_live_feed(req_list)
                 if not live_feed or "Data" not in live_feed or len(live_feed["Data"]) < 2:
                     raise Exception("fetch_live_feed also failed")
                 market_feed = live_feed # Use live_feed data if successful
                 logger.info("Successfully fetched data using fetch_live_feed.")
            except Exception as live_e:
                 logger.error(f"Both market_feed and fetch_live_feed failed: {live_e}")
                 raise Exception("Failed to fetch NIFTY 50 or India VIX from 5paisa API.")


        nifty_data = market_feed["Data"][0]
        vix_data = market_feed["Data"][1]
        nifty_spot = nifty_data.get("LastRate", nifty_data.get("LastTradedPrice", 0))
        vix = vix_data.get("LastRate", vix_data.get("LastTradedPrice", 0))

        if not nifty_spot or not vix or nifty_spot <= 0 or vix <= 0:
             raise Exception(f"Missing or invalid NIFTY ({nifty_spot}) or VIX ({vix}) price.")

        # Fetch Expiry Dates
        expiries_resp = client.get_expiry("N", "NIFTY")
        if not expiries_resp or "Data" not in expiries_resp or not expiries_resp["Data"]:
            logger.warning("Failed to fetch expiries. Proceeding without expiry info if needed later.")
            expiry_date_str = (datetime.now(IST) + timedelta(days=7 - datetime.now(IST).weekday() + 3) % 7).strftime("%Y-%m-%d") # Estimate next Thursday expiry
            next_expiry_timestamp = None # Cannot fetch option chain without timestamp
        else:
            # Assuming the first expiry in the list is the nearest weekly expiry
            next_expiry_info = expiries_resp["Data"][0]
            expiry_date_str = next_expiry_info["ExpiryDate"] # This is often a date string
            next_expiry_timestamp = next_expiry_info["Timestamp"] # This is a timestamp used for option chain

        # Fetch Option Chain for the nearest expiry if timestamp is available
        option_chain_df = pd.DataFrame()
        atm_strike = 0
        straddle_price = 0
        pcr = 0
        max_pain_strike = 0
        max_pain_diff_pct = 0

        if next_expiry_timestamp:
            option_chain_resp = client.get_option_chain("N", "NIFTY", next_expiry_timestamp)
            if not option_chain_resp or "Options" not in option_chain_resp:
                logger.warning("Failed to fetch option chain.")
            else:
                option_chain_df = pd.DataFrame(option_chain_resp["Options"])
                required_cols = ["StrikeRate", "CPType", "LastRate", "OpenInterest", "ScripCode"] # Include ScripCode
                if not all(col in option_chain_df.columns for col in required_cols):
                    logger.warning(f"Option chain missing required columns. Found: {option_chain_df.columns.tolist()}. Required: {required_cols}")
                    option_chain_df = pd.DataFrame() # Clear if columns are missing
                else:
                    # Ensure data types are correct
                    option_chain_df["StrikeRate"] = option_chain_df["StrikeRate"].astype(float)
                    option_chain_df["OpenInterest"] = option_chain_df["OpenInterest"].astype(float)
                    option_chain_df["LastRate"] = option_chain_df["LastRate"].astype(float)
                    option_chain_df["ScripCode"] = option_chain_df["ScripCode"].astype(int)


        if not option_chain_df.empty:
            try:
                # Calculate ATM Strike
                atm_strike = option_chain_df["StrikeRate"].iloc[(option_chain_df["StrikeRate"] - nifty_spot).abs().argmin()]

                # Calculate Straddle Price at ATM Strike
                atm_data = option_chain_df[option_chain_df["StrikeRate"] == atm_strike]
                atm_call_price = atm_data[atm_data["CPType"] == "CE"]["LastRate"].mean() # Use mean in case of duplicates
                atm_put_price = atm_data[atm_data["CPType"] == "PE"]["LastRate"].mean() # Use mean in case of duplicates
                # Handle potential NaN if CE/PE not found at ATM strike
                atm_call_price = atm_call_price if not pd.isna(atm_call_price) else 0
                atm_put_price = atm_put_price if not pd.isna(atm_put_price) else 0
                straddle_price = atm_call_price + atm_put_price

                # Calculate PCR (using total OI)
                total_call_oi = option_chain_df[option_chain_df["CPType"] == "CE"]["OpenInterest"].sum()
                total_put_oi = option_chain_df[option_chain_df["CPType"] == "PE"]["OpenInterest"].sum()
                pcr = total_put_oi / total_call_oi if total_call_oi != 0 else float("inf")

                # Calculate Max Pain
                max_pain_strike, max_pain_diff_pct = max_pain(option_chain_df, nifty_spot)
                if max_pain_strike is None:
                     logger.warning("Max pain calculation failed using option chain data.")

            except Exception as oc_calc_e:
                logger.error(f"Error calculating option chain metrics: {oc_calc_e}")
                # Reset option chain related data if calculation fails
                option_chain_df = pd.DataFrame()
                atm_strike = 0
                straddle_price = 0
                pcr = 0
                max_pain_strike = 0
                max_pain_diff_pct = 0

        # Calculate VIX change percentage (requires historical VIX, handled in load_data)
        # For real-time fetch, we just get the current VIX. VIX change is calculated later.
        vix_change_pct = 0 # Placeholder, calculated in load_data

        logger.info("Real-time data fetched successfully from 5paisa API")
        return {
            "nifty_spot": nifty_spot,
            "vix": vix,
            "atm_strike": atm_strike,
            "straddle_price": straddle_price,
            "pcr": pcr,
            "max_pain_strike": max_pain_strike,
            "max_pain_diff_pct": max_pain_diff_pct,
            "vix_change_pct": vix_change_pct, # Will be updated by load_data
            "option_chain": option_chain_df,
            "expiry": expiry_date_str
        }
    except Exception as e:
        logger.error(f"Error fetching 5paisa API data: {str(e)}")
        st.warning(f"Could not fetch real-time data from 5paisa: {str(e)}. Using historical CSV data.")
        return None

def max_pain(df, nifty_spot):
    """Calculates the max pain strike level from option chain data."""
    try:
        if df.empty or not all(col in df.columns for col in ["StrikeRate", "CPType", "OpenInterest"]):
            logger.warning("Max pain calculation requires non-empty df with StrikeRate, CPType, OpenInterest.")
            return None, None

        calls = df[df["CPType"] == "CE"].set_index("StrikeRate")["OpenInterest"]
        puts = df[df["CPType"] == "PE"].set_index("StrikeRate")["OpenInterest"]
        strikes = df["StrikeRate"].unique()

        if len(strikes) < 2:
             logger.warning("Max pain requires at least two unique strikes.")
             return None, None

        pain = []
        # Ensure strikes are sorted for potential future use, though not strictly needed for min()
        sorted_strikes = sorted(strikes)

        for K in sorted_strikes:
            total_loss = 0
            # Loss for Call writers at strike s if Nifty closes at K
            # Iterate efficiently over relevant strikes
            for s in calls.index:
                 if s >= K: # Calls at strikes >= K expire worthless or ITM below K (loss for writer)
                      total_loss += max(0, s - K) * calls.get(s, 0)

            # Loss for Put writers at strike s if Nifty closes at K
            for s in puts.index:
                 if s <= K: # Puts at strikes <= K expire worthless or ITM above K (loss for writer)
                       total_loss += max(0, K - s) * puts.get(s, 0)

            pain.append((K, total_loss))

        if not pain:
             logger.warning("Max pain calculation resulted in empty pain list.")
             return None, None

        # Find the strike with the minimum total loss
        max_pain_strike = min(pain, key=lambda x: x[1])[0]
        max_pain_diff_pct = abs(nifty_spot - max_pain_strike) / nifty_spot * 100 if nifty_spot != 0 else 0
        return max_pain_strike, max_pain_diff_pct
    except Exception as e:
        logger.error(f"Error calculating max pain: {str(e)}")
        return None, None


@st.cache_data(ttl=3600) # Cache for 1 hour
def load_historical_data():
    """Loads historical NIFTY and VIX data from GitHub CSVs."""
    try:
        logger.info("Loading historical data from GitHub CSVs")

        # Load Nifty Data
        nifty_response = requests.get(NIFTY_HISTORICAL_URL)
        nifty_response.raise_for_status() # Raise an exception for bad status codes
        nifty = pd.read_csv(io.StringIO(nifty_response.text), encoding="utf-8-sig")
        nifty.columns = nifty.columns.str.strip()
        nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
        nifty = nifty.dropna(subset=["Date"]).set_index("Date")[["Close"]].rename(columns={"Close": "NIFTY_Close"})

        # Load VIX Data
        vix_response = requests.get(VIX_HISTORICAL_URL)
        vix_response.raise_for_status() # Raise an exception for bad status codes
        vix = pd.read_csv(io.StringIO(vix_response.text))
        vix.columns = vix.columns.str.strip()
        vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")
        vix = vix.dropna(subset=["Date"]).set_index("Date")[["Close"]].rename(columns={"Close": "VIX"})

        # Merge dataframes on date index
        df_historical = pd.merge(nifty, vix, left_index=True, right_index=True, how="inner")

        # Handle potential duplicates by taking the last value and sort
        df_historical = df_historical.groupby(df_historical.index).last()
        df_historical = df_historical.sort_index()
        df_historical = df_historical.ffill().bfill() # Fill missing values

        logger.info(f"Historical data loaded. Shape: {df_historical.shape}")
        return df_historical

    except requests.exceptions.RequestException as re:
        logger.error(f"Error fetching historical data from GitHub: {re}")
        st.error(f"Failed to load historical data from GitHub: {re}")
        return pd.DataFrame() # Return empty dataframe on failure
    except Exception as e:
        logger.error(f"Error loading or processing historical data: {e}")
        st.error(f"An error occurred while processing historical data: {e}")
        return pd.DataFrame() # Return empty dataframe on failure


def prepare_data(client, capital):
    """Loads historical and real-time data, combines them, and prepares the final dataframe."""
    logger.info("Preparing data...")
    historical_df = load_historical_data()
    if historical_df.empty:
        return pd.DataFrame(), None, "Historical data not available"

    # Fetch real-time data first
    real_data = fetch_nifty_data_5paisa(client)
    st.session_state.real_time_data = real_data # Store real-time data in session state

    data_source = "CSV (Historical)"
    combined_df = historical_df.copy() # Start with historical data

    if real_data and real_data["nifty_spot"] is not None and real_data["vix"] is not None:
        logger.info("Combining real-time data with historical data.")
        data_source = "5paisa API (LIVE) + CSV"
        # Get the date of the latest historical data point
        last_hist_date = historical_df.index.max() if not historical_df.empty else pd.NaT
        latest_real_date = pd.to_datetime(datetime.now(IST).date()).normalize()

        # Only append real-time data if it's for a new date or if historical is empty
        if pd.isnat(last_hist_date) or latest_real_date > last_hist_date:
             real_time_df = pd.DataFrame({
                 "NIFTY_Close": [real_data["nifty_spot"]],
                 "VIX": [real_data["vix"]]
             }, index=[latest_real_date])
             combined_df = pd.concat([historical_df, real_time_df])

             # Recalculate VIX Change Pct incorporating the new real-time data
             combined_df["VIX_Change_Pct"] = combined_df["VIX"].pct_change() * 100
             # Update real_data dict with the calculated change percent
             if latest_real_date in combined_df.index and len(combined_df) > 1:
                  real_data["vix_change_pct"] = combined_df.loc[latest_real_date, "VIX_Change_Pct"]

        elif latest_real_date == last_hist_date:
             # If real-time data is for the same day as last historical, update that row
             logger.info("Updating latest historical row with real-time data.")
             combined_df.loc[last_hist_date, "NIFTY_Close"] = real_data["nifty_spot"]
             combined_df.loc[last_hist_date, "VIX"] = real_data["vix"]
             # Recalculate VIX Change Pct for the last row
             if len(combined_df) > 1:
                  combined_df["VIX_Change_Pct"] = combined_df["VIX"].pct_change() * 100
                  real_data["vix_change_pct"] = combined_df.loc[last_hist_date, "VIX_Change_Pct"]


        # If real-time date is before last historical date (shouldn't happen with current date)
        # or if concatenation/update failed, just use historical.
        # Ensure combined_df is sorted and cleaned after potential concat/update
        combined_df = combined_df.groupby(combined_df.index).last() # Handle potential duplicates
        combined_df = combined_df.sort_index()
        combined_df = combined_df.ffill().bfill() # Final fill missing values

    else:
        logger.warning("Real-time data fetch failed or data is invalid. Using only historical data.")
        # If only historical is used, calculate VIX_Change_Pct here if not already present
        if "VIX_Change_Pct" not in combined_df.columns:
             combined_df["VIX_Change_Pct"] = combined_df["VIX"].pct_change().fillna(0) * 100


    if combined_df.empty:
        return pd.DataFrame(), None, "No data available after processing."

    logger.debug(f"Combined data prepared. Shape: {combined_df.shape}. Source: {data_source}")
    st.session_state.historical_data = combined_df # Store combined data in session state
    return combined_df, real_data, data_source


def fetch_portfolio_data(client):
    """Fetches current portfolio positions from 5paisa."""
    if not client:
        logger.warning("5paisa client not initialized for portfolio fetch.")
        return {"total_pnl": 0, "margin_used": 0, "exposure_value": 0, "positions": []}

    try:
        logger.info("Fetching portfolio data from 5paisa API")
        # py5paisa v3 client positions() might return a list of dicts or None/empty
        positions_data = client.positions()

        # Check for common indicators of failed API call vs no positions
        if positions_data is None: # API call failed
            logger.error("5paisa positions() returned None.")
            return {"total_pnl": 0, "margin_used": 0, "exposure_value": 0, "positions": [], "error": "API call failed"}
        if not isinstance(positions_data, list) or (isinstance(positions_data, list) and not positions_data): # Not a list or empty list
             logger.info("No positions found in 5paisa account.")
             return {"total_pnl": 0, "margin_used": 0, "exposure_value": 0, "positions": []}

        total_pnl = 0
        total_margin = 0
        total_exposure_value = 0
        cleaned_positions = []

        for pos in positions_data:
            # Basic validation for a position dictionary
            if not isinstance(pos, dict) or 'ScripName' not in pos:
                 logger.warning(f"Skipping invalid position data format: {pos}")
                 continue

            try:
                pnl = pos.get("ProfitLoss", 0.0)
                margin = pos.get("MarginUsed", 0.0)
                exposure = pos.get("Exposure", 0.0)
                scrip_name = pos.get("ScripName", "N/A")
                buy_avg = pos.get("BuyAvg", 0.0)
                sell_avg = pos.get("SellAvg", 0.0)
                net_qty = pos.get("BuyQty", 0) - pos.get("SellQty", 0) # Net Quantity
                ltp = pos.get("LastRate", 0.0)
                # Construct a cleaner representation
                cleaned_positions.append({
                    "Scrip Name": scrip_name,
                    "Quantity": net_qty,
                    "Buy Avg": buy_avg,
                    "Sell Avg": sell_avg,
                    "LTP": ltp,
                    "P&L": pnl,
                    "Margin Used": margin,
                    "Exposure Value": exposure # Note: This might not be % Exposure
                })
                total_pnl += pnl
                total_margin += margin
                total_exposure_value += exposure
            except Exception as pos_e:
                logger.error(f"Error processing individual position data: {pos_e} - Data: {pos}")
                continue # Skip problematic position

        logger.info(f"Portfolio data fetched: Total PnL={total_pnl}, Margin Used={total_margin}, Exposure Value={total_exposure_value}")
        return {
            "total_pnl": total_pnl,
            "margin_used": total_margin,
            "exposure_value": total_exposure_value,
            "positions": cleaned_positions
        }
    except Exception as e:
        logger.error(f"Error fetching portfolio data: {str(e)}")
        st.error(f"Failed to fetch portfolio data: {str(e)}")
        return {"total_pnl": 0, "margin_used": 0, "exposure_value": 0, "positions": [], "error": str(e)}


# --- Feature Generation ---
@st.cache_data
def generate_features(df, real_data, capital):
    """Generates features for volatility forecasting and strategy selection."""
    try:
        logger.info("Generating features")
        df = df.copy()  # Avoid modifying cached dataframe
        df.index = df.index.normalize()  # Ensure date-only index

        if df.empty:
             logger.error("Cannot generate features from empty dataframe.")
             return pd.DataFrame()

        n_days = len(df)

        # Use real data points if available, otherwise use historical averages/fallbacks
        # Prioritize real_data, then last row of df if available, else default
        base_pcr = real_data["pcr"] if real_data and real_data["pcr"] is not None and real_data["pcr"] != 0 else df["PCR"].iloc[-1] if "PCR" in df.columns and len(df) > 0 else 1.0
        base_straddle_price = real_data["straddle_price"] if real_data and real_data["straddle_price"] is not None and real_data["straddle_price"] != 0 else df["Straddle_Price"].iloc[-1] if "Straddle_Price" in df.columns and len(df) > 0 else 200.0
        base_max_pain_diff_pct = real_data["max_pain_diff_pct"] if real_data and real_data["max_pain_diff_pct"] is not None else df["Spot_MaxPain_Diff_Pct"].iloc[-1] if "Spot_MaxPain_Diff_Pct" in df.columns and len(df) > 0 else 0.5
        base_vix_change_pct = real_data["vix_change_pct"] if real_data and real_data["vix_change_pct"] is not None else df["VIX_Change_Pct"].iloc[-1] if "VIX_Change_Pct" in df.columns and len(df) > 0 else 0.0
        latest_vix = real_data["vix"] if real_data and real_data["vix"] is not None else df["VIX"].iloc[-1] if len(df) > 0 and "VIX" in df.columns else 15.0
        latest_nifty = real_data["nifty_spot"] if real_data and real_data["nifty_spot"] is not None else df["NIFTY_Close"].iloc[-1] if len(df) > 0 and "NIFTY_Close" in df.columns else 18000.0


        # Calculate Days to Expiry - This needs to be dynamic per row for backtesting
        # For the *last* row (real-time), it's based on the *next* expiry
        # For historical rows, it's based on the next expiry *after* that date
        def calculate_days_to_expiry_series(dates):
            days_to_expiry = []
            for date in dates:
                # Find the next Thursday from this date
                # weekday() Monday is 0, Sunday is 6. Thursday is 3.
                days_ahead = (3 - date.weekday()) % 7
                if days_ahead == 0: # If it's already Thursday, next expiry is in 7 days (for next week)
                     days_ahead = 7
                next_expiry = date + pd.Timedelta(days=days_ahead)
                dte = (next_expiry - date).days
                days_to_expiry.append(dte)
            return pd.Series(days_to_expiry, index=dates)

        df["Days_to_Expiry"] = calculate_days_to_expiry_series(df.index)
        # If the latest data point is a Thursday that is an expiry day, DTE would be 0.
        # We should treat this as 7 days to the *next* expiry for forecasting purposes.
        if len(df) > 0 and df.index[-1].weekday() == 3 and df["Days_to_Expiry"].iloc[-1] == 0:
            df.loc[df.index[-1], "Days_to_Expiry"] = 7 # Treat expiry day as 7 days to next cycle


        # ATM IV - Use VIX as base, add noise/event spikes
        # The synthetic generation from the original code is kept for historical data,
        # but the last value is replaced by the real-time VIX.
        np.random.seed(42) # For reproducibility of synthetic parts
        # Ensure length matches df
        # Re-evaluate event_spike based on the calculated DTE
        event_spike_condition = (df.index.month % 3 == 0) & (df["Days_to_Expiry"] <= 3) # Event near quarterly expiry
        event_spike = np.where(event_spike_condition, 1.2, 1.0)
        random_iv_noise = np.random.normal(0, 0.1, n_days)
        # Apply synthetic generation only to historical part
        df["ATM_IV"] = df["VIX"] * (1 + random_iv_noise) * event_spike
        df["ATM_IV"] = np.clip(df["ATM_IV"], 5, 50)
        # Replace the last ATM_IV with the latest real VIX value if available
        if real_data and real_data["vix"] is not None and len(df) > 0:
            df.loc[df.index[-1], "ATM_IV"] = latest_vix


        # IVP - Calculate based on rolling ATM_IV (need at least 5 periods)
        def dynamic_ivp(series):
            # Ensure the series contains numbers before calculating quantile
            numeric_series = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_series) >= 5:
                # Calculate the rank (percentile) of the last value within the window
                return numeric_series.rank(pct=True).iloc[-1] * 100 if len(numeric_series) > 1 else 50.0
            return 50.0 # Default for insufficient data or only one value
        # Apply rolling calculation. window=252 (approx trading days in a year), min_periods=5
        df["IVP"] = df["ATM_IV"].rolling(window=252, min_periods=5).apply(dynamic_ivp, raw=False)
        df["IVP"] = df["IVP"].interpolate(method='linear', limit_direction='both').fillna(50.0) # Interpolate and fill any remaining NaNs


        # PCR - Synthesize based on market trend for historical, use real data for latest
        market_trend = df["NIFTY_Close"].pct_change().rolling(5).mean().fillna(0)
        # Synthesize for full history
        # Base PCR around 1.0, add noise and inverse relation to market trend (rising market -> PCR drops as calls favored)
        df["PCR"] = np.clip(1.0 + np.random.normal(0, 0.1, n_days) - market_trend * 5, 0.7, 2.0) # Reduced market trend impact
        # Replace the last PCR with the latest real PCR value if available
        if real_data and real_data["pcr"] is not None and len(df) > 0:
            df.loc[df.index[-1], "PCR"] = base_pcr

        # VIX Change Pct - Calculated in prepare_data if real data available. If not, calculate here.
        if "VIX_Change_Pct" not in df.columns:
             df["VIX_Change_Pct"] = df["VIX"].pct_change().fillna(0) * 100
        # Ensure the last value is updated if real data was fetched (prepare_data handles this)
        # If real_data exists but VIX_change_pct was not calculated in prepare_data (e.g. only 1 row), calculate it now
        if real_data and real_data["vix_change_pct"] is None and len(df) > 1:
             df["VIX_Change_Pct"] = df["VIX"].pct_change().fillna(0) * 100
             real_data["vix_change_pct"] = df["VIX_Change_Pct"].iloc[-1] # Update real_data


        # Spot_MaxPain_Diff_Pct - Synthesize for historical, use real data for latest
        # Synthesize based on a distribution that typically favors smaller differences
        df["Spot_MaxPain_Diff_Pct"] = np.random.lognormal(mean=-2, sigma=0.5, size=n_days) * 100 # mean=-2, sigma=0.5 gives values mostly below 10%
        df["Spot_MaxPain_Diff_Pct"] = np.clip(df["Spot_MaxPain_Diff_Pct"], 0.1, 10.0) # Clip to a more realistic percentage range (0.1% to 10%)
        # Replace the last value with real data if available
        if real_data and real_data["max_pain_diff_pct"] is not None and len(df) > 0:
             df.loc[df.index[-1], "Spot_MaxPain_Diff_Pct"] = base_max_pain_diff_pct

        # Event Flag - Keep quarterly expiry week logic, already used for event_spike
        df["Event_Flag"] = np.where(event_spike_condition, 1, 0)

        # FII Data - Keep synthetic generation as real FII data is not directly available via 5paisa API
        # Use more realistic step sizes maybe?
        fii_trend_fut = np.random.normal(0, 5000, n_days) # Smaller daily change
        fii_trend_fut[::30] *= -1 # Introduce some reversals every ~month
        df["FII_Index_Fut_Pos"] = np.cumsum(fii_trend_fut).astype(int)
        fii_trend_opt = np.random.normal(0, 2000, n_days) # Smaller daily change
        df["FII_Option_Pos"] = np.cumsum(fii_trend_opt).astype(int)

        # IV Skew - Synthesize based on VIX level and add noise
        # Higher VIX often means higher skew. Negative correlation with Nifty trend?
        df["IV_Skew"] = np.clip(np.random.normal(0, 0.5, n_days) + (df["VIX"] / 20 - 1) * 2 - market_trend * 5, -3, 3) # Reduced VIX impact, added inverse market trend
        # Use a more realistic range and dynamic calculation if possible with Option Chain?
        # For now, keep synthetic to align with original code logic base.
        # If real_data has option chain, could calculate actual IV skew across strikes.
        # Let's keep it synthetic for consistency unless we add full option chain analysis.

        # Realized Volatility - Calculate based on rolling NIFTY returns
        df["Realized_Vol"] = df["NIFTY_Close"].pct_change().rolling(5, min_periods=1).std() * np.sqrt(252) * 100
        df["Realized_Vol"] = df["Realized_Vol"].fillna(df["VIX"]) # Fill initial NaNs with VIX
        df["Realized_Vol"] = np.clip(df["Realized_Vol"], 0, 50)

        # Advance Decline Ratio - Synthesize based on market trend
        df["Advance_Decline_Ratio"] = np.clip(1.0 + np.random.normal(0, 0.1, n_days) + market_trend * 5, 0.7, 1.3) # Reduced scale and impact

        # Capital Pressure Index / Gamma Bias - Keep original synthetic logic
        # Adjust Capital Pressure Index formula - use PCR-1 as deviation from 1.0
        df["Capital_Pressure_Index"] = (df["FII_Index_Fut_Pos"] / 5e4 + df["FII_Option_Pos"] / 2e4 + (df["PCR"]-1)*5) / 3 # Adjusted divisors and PCR factor
        df["Capital_Pressure_Index"] = np.clip(df["Capital_Pressure_Index"], -2, 2)
        # Gamma Bias is often related to IV Skew and DTE - keep simplified formula
        # Skew impact increases as DTE decreases
        df["Gamma_Bias"] = np.clip(df["IV_Skew"] * (30 - df["Days_to_Expiry"]) / 30 * 0.5, -2, 2) # Reduced impact multiplier


        # Straddle Price - Synthesize for historical, use real data for latest
        # Use Nifty price and VIX/IV to make it more realistic
        # Simplified Black-Scholes like estimate for ATM Straddle (needs T, r)
        # Approximation: Straddle Price ~ 2 * Nifty_Spot * IV * sqrt(DTE/365)
        r_annual = 0.06 # Assume 6% annual risk-free rate
        trading_days_in_year = 252 # Approx trading days
        # Ensure DTE is positive for sqrt
        dte_years = np.maximum(1, df["Days_to_Expiry"]) / trading_days_in_year
        df["Straddle_Price_Synth"] = 2 * df["NIFTY_Close"] * (df["ATM_IV"]/100) * np.sqrt(dte_years) # Basic BS estimate components
        # Add some noise around the theoretical value
        df["Straddle_Price_Synth"] = df["Straddle_Price_Synth"] * np.random.uniform(0.9, 1.1, n_days)
        df["Straddle_Price_Synth"] = np.clip(df["Straddle_Price_Synth"], 50, 600) # Wider reasonable range

        # Replace the 'Straddle_Price' column with the synthesized values initially
        # This column name is used by the backtest and strategy functions
        df["Straddle_Price"] = df["Straddle_Price_Synth"]

        # Replace the last Straddle_Price with the latest real value if available
        if real_data and real_data["straddle_price"] is not None and real_data["straddle_price"] != 0 and len(df) > 0:
             df.loc[df.index[-1], "Straddle_Price"] = base_straddle_price

        # Remove the temporary synthetic column
        df = df.drop(columns=["Straddle_Price_Synth"])


        # Add Capital as a constant column for backtesting context (used by backtest func)
        # Note: Capital in backtest should ideally be dynamic or based on initial capital.
        # Keeping this for now as it was in original code, but will ensure backtest uses input capital.
        # This column might not be strictly needed for features, but keeping it if subsequent steps rely on it.
        df["Total_Capital"] = capital

        # Ensure all columns used for forecasting exist and handle NaNs
        feature_cols_required = [
             'VIX', 'ATM_IV', 'IVP', 'PCR', 'VIX_Change_Pct', 'IV_Skew', 'Straddle_Price',
             'Spot_MaxPain_Diff_Pct', 'Days_to_Expiry', 'Event_Flag', 'FII_Index_Fut_Pos',
             'FII_Option_Pos', 'Realized_Vol', 'Gamma_Bias', 'Capital_Pressure_Index',
             'NIFTY_Close' # Include Nifty Close for returns calculation etc.
         ]
        for col in feature_cols_required:
            if col not in df.columns:
                df[col] = 0.0 # Add missing columns with default 0
                logger.warning(f"Added missing feature column: {col}")

        # Final check and fill NaNs - important before training/forecasting
        if df.isna().sum().sum() > 0:
            logger.warning(f"NaNs found in feature dataframe: {df.isna().sum()[df.isna().sum() > 0].to_dict()}")
            # Use interpolation for most, but ffill/bfill for start/end or stubborn NaNs
            df = df.interpolate(method='linear', limit_direction='both')
            df = df.fillna(method='bfill').fillna(method='ffill')

        # Final check for infinite values which can occur with division by zero etc.
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0) # Replace any remaining inf/NaN with 0


        logger.debug(f"Features generated successfully. Shape: {df.shape}")
        st.session_state.feature_data = df # Store feature data in session state
        return df
    except Exception as e:
        st.error(f"Error generating features: {str(e)}")
        logger.error(f"Error generating features: {str(e)}")
        return pd.DataFrame() # Return empty dataframe on error

# --- Volatility Forecasting ---
# Define feature columns used for XGBoost model
XGB_FEATURE_COLS = [
    'VIX', 'ATM_IV', 'IVP', 'PCR', 'VIX_Change_Pct', 'IV_Skew', 'Straddle_Price',
    'Spot_MaxPain_Diff_Pct', 'Days_to_Expiry', 'Event_Flag', 'FII_Index_Fut_Pos',
    'FII_Option_Pos', 'Realized_Vol', 'Gamma_Bias', 'Capital_Pressure_Index'
]


@st.cache_data
def forecast_volatility_future(df, forecast_horizon):
    """Forecasts volatility using GARCH and XGBoost."""
    try:
        logger.info(f"Forecasting volatility for {forecast_horizon} days.")
        df = df.copy()  # Avoid modifying cached dataframe
        df.index = pd.to_datetime(df.index).normalize()  # Ensure date-only index

        if df.empty or len(df) < max(252, forecast_horizon + 1): # Need enough data for rolling calcs, training, and forecasting
             st.warning(f"Insufficient data for robust forecasting: need at least {max(252, forecast_horizon + 1)} days, have {len(df)}")
             # Provide a basic forecast if data is too short
             if len(df) > 0 and "VIX" in df.columns:
                  last_vix = df["VIX"].iloc[-1]
             else:
                  last_vix = 15.0 # Default if no data
             future_dates = pd.date_range(start=df.index.max() + timedelta(days=1), periods=forecast_horizon, freq='B')
             basic_forecast = pd.DataFrame({
                  "Date": future_dates,
                  "GARCH_Vol": last_vix,
                  "XGBoost_Vol": last_vix,
                  "Blended_Vol": last_vix,
                  "Confidence": 50 # Low confidence
             })
             return basic_forecast, [last_vix] * forecast_horizon, [last_vix] * forecast_horizon, [last_vix] * forecast_horizon, last_vix, 50.0, 999.99, pd.Series([0.0]*len(XGB_FEATURE_COLS), index=XGB_FEATURE_COLS).values # Return basic forecast and dummy metrics


        # --- GARCH Model ---
        # Use a recent window for GARCH if historical data is very long
        garch_df = df.tail(500).copy() # Use last 500 days for GARCH
        if len(garch_df) < 5 or "NIFTY_Close" not in garch_df.columns:
             logger.warning("Not enough data or NIFTY_Close missing for GARCH model.")
             garch_vols = np.full(forecast_horizon, garch_df["VIX"].iloc[-1] if len(garch_df) > 0 and "VIX" in garch_df.columns else 15.0)
        else:
            garch_df['Log_Returns'] = np.log(garch_df['NIFTY_Close'] / garch_df['NIFTY_Close'].shift(1)).dropna() * 100
            if len(garch_df['Log_Returns']) < 5:
                 logger.warning("Not enough log returns for GARCH model.")
                 garch_vols = np.full(forecast_horizon, garch_df["VIX"].iloc[-1] if len(garch_df) > 0 and "VIX" in garch_df.columns else 15.0)
            else:
                try:
                    # Ensure log returns are finite
                    log_returns_finite = garch_df['Log_Returns'][np.isfinite(garch_df['Log_Returns'])]
                    if len(log_returns_finite) < 5:
                         logger.warning("Not enough finite log returns for GARCH model.")
                         garch_vols = np.full(forecast_horizon, garch_df["VIX"].iloc[-1] if len(garch_df) > 0 and "VIX" in garch_df.columns else 15.0)
                    else:
                        garch_model = arch_model(log_returns_finite, vol='Garch', p=1, q=1, rescale=False)
                        garch_fit = garch_model.fit(disp="off", show_warning=False) # Suppress fit warnings
                        garch_forecast = garch_fit.forecast(horizon=forecast_horizon, reindex=False)
                        # Ensure variance is non-negative before sqrt
                        # Accessing the correct part of the forecast result
                        forecast_variance = garch_forecast.variance.iloc[-1].values if not garch_forecast.variance.empty else np.zeros(forecast_horizon)
                        garch_vols = np.sqrt(np.maximum(0, forecast_variance)) * np.sqrt(252) # Annualize
                        garch_vols = np.clip(garch_vols, 5, 50) # Clip to reasonable range
                except Exception as garch_e:
                    logger.error(f"GARCH model fitting or forecasting failed: {garch_e}")
                    garch_vols = np.full(forecast_horizon, garch_df["VIX"].iloc[-1] if len(garch_df) > 0 and "VIX" in garch_df.columns else 15.0) # Fallback to last VIX


        realized_vol = df["Realized_Vol"].dropna().iloc[-5:].mean() if len(df["Realized_Vol"].dropna()) >= 5 else (df["VIX"].iloc[-1] if len(df)>0 and "VIX" in df.columns else 15.0)


        # --- XGBoost Model ---
        xgb_df = df.copy()
        # Target is next day's Realized Volatility
        xgb_df['Target_Vol'] = xgb_df['Realized_Vol'].shift(-1)
        xgb_df = xgb_df.dropna(subset=['Target_Vol']) # Drop rows where target is NaN

        # Ensure XGBoost features exist and handle NaNs before scaling/training
        current_xgb_features = [col for col in XGB_FEATURE_COLS if col in xgb_df.columns]
        if not current_xgb_features:
             logger.error("No valid feature columns found for XGBoost training.")
             # Fallback for XGBoost forecast
             xgb_vols = np.full(forecast_horizon, df["VIX"].iloc[-1] if len(df)>0 and "VIX" in df.columns else 15.0)
             rmse = 999.99
             feature_importances = np.full(len(XGB_FEATURE_COLS), 0.0)
        else:
            X = xgb_df[current_xgb_features]
            y = xgb_df['Target_Vol']

            if len(X) < 100: # Need sufficient data for XGBoost training
                 logger.warning(f"Insufficient data for XGBoost training: need at least 100 days, have {len(X)}")
                 # Fallback for XGBoost forecast
                 xgb_vols = np.full(forecast_horizon, df["VIX"].iloc[-1] if len(df)>0 and "VIX" in df.columns else 15.0)
                 rmse = 999.99
                 feature_importances = np.full(len(XGB_FEATURE_COLS), 0.0)
            else:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_scaled_df = pd.DataFrame(X_scaled, columns=current_xgb_features, index=X.index)

                split_index = max(1, int(len(X_scaled_df) * 0.8)) # Ensure train set is at least 1
                X_train, X_test = X_scaled_df.iloc[:split_index], X_scaled_df.iloc[split_index:]
                y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

                model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.03, random_state=42, objective='reg:squarederror')
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    # Ensure y_test is not empty
                    if len(y_test) > 0:
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    else:
                        rmse = 0.0 # No test data, cannot calculate RMSE, set to 0 or NaN?

                    feature_importances_raw = model.feature_importances_
                    # Create a series to map importance to original feature names
                    feature_importances_series = pd.Series(feature_importances_raw, index=current_xgb_features)
                    # Ensure the output feature_importances array matches XGB_FEATURE_COLS order, filling 0 for missing
                    feature_importances = pd.Series(0.0, index=XGB_FEATURE_COLS)
                    feature_importances.update(feature_importances_series)
                    feature_importances = feature_importances.values

                    # Forecast future using XGBoost - simulate feature evolution
                    xgb_vols = []
                    # Start forecasting from the last available feature row
                    if not X_scaled_df.empty:
                        current_features_scaled = X_scaled_df.iloc[-1].copy() # Start from last scaled features
                        # Need the unscaled last features to simulate evolution
                        last_unscaled_features = xgb_df[current_xgb_features].iloc[-1].copy()

                        for i in range(forecast_horizon):
                            # Predict next vol using current scaled features
                            current_scaled_df = pd.DataFrame([current_features_scaled.values], columns=current_features_scaled.index)
                            next_vol_pred = model.predict(current_scaled_df)[0]
                            xgb_vols.append(next_vol_pred)

                            # Simulate evolution of unscaled features for the *next* day
                            # This simulation is a simplification; a more complex model would predict future features
                            # Use the predicted vol (next_vol_pred) to influence the simulation
                            # Simulate a date for DTE calculation
                            simulated_date = df.index.max() + timedelta(days=i + 1)
                            simulated_dte = calculate_days_to_expiry_series([simulated_date]).iloc[0]
                            if 'Days_to_Expiry' in last_unscaled_features.index:
                                last_unscaled_features['Days_to_Expiry'] = simulated_dte # Sync DTE simulation

                            # Simulate Event_Flag based on the simulated date and DTE
                            simulated_event_flag = 1 if simulated_dte <= 3 and simulated_date.month % 3 == 0 else 0
                            if 'Event_Flag' in last_unscaled_features.index:
                                last_unscaled_features['Event_Flag'] = simulated_event_flag


                            # Simple random walk/noise for other features
                            for feature in last_unscaled_features.index:
                                if feature not in ['Days_to_Expiry', 'Event_Flag', 'Realized_Vol']: # Exclude calculated/synced features
                                    # Apply small random noise or trend based on feature type
                                    noise_scale = 0.02 # Default noise percentage
                                    additive_noise = 0 # Default additive noise

                                    if feature in ['VIX', 'ATM_IV', 'Straddle_Price']:
                                        # Base next VIX/IV/Straddle on the predicted next_vol_pred, with noise
                                        if feature == 'VIX':
                                             last_unscaled_features['VIX'] = next_vol_pred * np.random.uniform(0.95, 1.05)
                                        elif feature == 'ATM_IV':
                                             last_unscaled_features['ATM_IV'] = next_vol_pred * np.random.uniform(0.98, 1.08)
                                        elif feature == 'Straddle_Price':
                                             # Re-calculate Straddle Price based on simulated Nifty, IV, DTE
                                             # Need to simulate Nifty movement first
                                             # For simplicity, assume Nifty moves based on predicted vol
                                             simulated_nifty_move_pct = np.random.normal(0, next_vol_pred / 100 / np.sqrt(252))
                                             simulated_nifty = last_unscaled_features.get('NIFTY_Close', df['NIFTY_Close'].iloc[-1]) * (1 + simulated_nifty_move_pct)
                                             # Recalculate Straddle Price estimate
                                             simulated_dte_years = max(1, simulated_dte) / 252
                                             last_unscaled_features['Straddle_Price'] = 2 * simulated_nifty * (last_unscaled_features['ATM_IV']/100) * np.sqrt(simulated_dte_years) * np.random.uniform(0.95, 1.05)
                                             last_unscaled_features['Straddle_Price'] = np.clip(last_unscaled_features['Straddle_Price'], 50, 600) # Clip


                                    elif feature in ['FII_Index_Fut_Pos', 'FII_Option_Pos']:
                                         additive_noise = abs(last_unscaled_features[feature])*0.005 + 500 # Additive noise related to position size + base
                                         last_unscaled_features[feature] += np.random.normal(0, additive_noise)
                                    elif feature in ['PCR', 'IV_Skew', 'Spot_MaxPain_Diff_Pct', 'Advance_Decline_Ratio', 'Capital_Pressure_Index', 'Gamma_Bias']:
                                         additive_noise = abs(last_unscaled_features[feature])*0.01 + 0.005 # Additive noise related to value + base
                                         last_unscaled_features[feature] += np.random.normal(0, additive_noise)


                            # Update Realized_Vol to be the predicted next_vol_pred (or close to it)
                            if 'Realized_Vol' in last_unscaled_features.index:
                                last_unscaled_features['Realized_Vol'] = next_vol_pred * np.random.uniform(0.95, 1.05) # Add slight noise around prediction


                            # Ensure values are within reasonable bounds after simulation
                            last_unscaled_features['VIX'] = np.clip(last_unscaled_features.get('VIX', 15.0), 5, 50)
                            last_unscaled_features['ATM_IV'] = np.clip(last_unscaled_features.get('ATM_IV', 15.0), 5, 50)
                            last_unscaled_features['PCR'] = np.clip(last_unscaled_features.get('PCR', 1.0), 0.7, 2.0)
                            last_unscaled_features['Spot_MaxPain_Diff_Pct'] = np.clip(last_unscaled_features.get('Spot_MaxPain_Diff_Pct', 0.5), 0.1, 10.0)
                            last_unscaled_features['IV_Skew'] = np.clip(last_unscaled_features.get('IV_Skew', 0), -3, 3)
                            last_unscaled_features['Realized_Vol'] = np.clip(last_unscaled_features.get('Realized_Vol', 15.0), 5, 50)
                            last_unscaled_features['Advance_Decline_Ratio'] = np.clip(last_unscaled_features.get('Advance_Decline_Ratio', 1.0), 0.7, 1.3)
                            last_unscaled_features['Capital_Pressure_Index'] = np.clip(last_unscaled_features.get('Capital_Pressure_Index', 0), -2, 2)
                            last_unscaled_features['Gamma_Bias'] = np.clip(last_unscaled_features.get('Gamma_Bias', 0), -2, 2)
                            last_unscaled_features['Straddle_Price'] = np.clip(last_unscaled_features.get('Straddle_Price', 200.0), 50, 600)


                            # Scale the simulated next features for the next prediction step
                            next_unscaled_df = pd.DataFrame([last_unscaled_features.values], columns=last_unscaled_features.index)
                            # Use the original scaler fitted on the training data
                            current_features_scaled = pd.Series(scaler.transform(next_unscaled_df)[0], index=last_unscaled_features.index)

                    else: # Fallback if X_scaled_df is empty (shouldn't happen if len(X) >= 100 check passes)
                         xgb_vols = np.full(forecast_horizon, df["VIX"].iloc[-1] if len(df)>0 and "VIX" in df.columns else 15.0)


                    xgb_vols = np.clip(xgb_vols, 5, 50) # Clip XGBoost forecast
                    # Apply event spike post-clipping if the *last* day was an event day
                    # This was for the *last historical* day. For forecast, check simulated event flag.
                    # Let's apply a potential uplift based on the *predicted* average volatility and event flags in the forecast horizon
                    # A simpler approach: Check if *any* day in the forecast horizon is an event day, apply a small uplift.
                    # Re-generate simulated DTE/Event Flags for the forecast horizon dates
                    forecast_dates = pd.date_range(start=df.index.max() + timedelta(days=1), periods=forecast_horizon, freq='B')
                    forecast_dtes = calculate_days_to_expiry_series(forecast_dates)
                    forecast_event_flags = np.where((forecast_dtes <= 3) & (forecast_dates.month % 3 == 0), 1, 0)

                    if np.any(forecast_event_flags == 1):
                         xgb_vols = [v * 1.05 for v in xgb_vols] # Apply a 5% uplift if any event day in horizon


                except Exception as xgb_e:
                    logger.error(f"XGBoost model fitting or forecasting failed: {xgb_e}")
                    xgb_vols = np.full(forecast_horizon, df["VIX"].iloc[-1] if len(df)>0 and "VIX" in df.columns else 15.0) # Fallback to last VIX
                    rmse = 999.99
                    feature_importances = np.full(len(XGB_FEATURE_COLS), 0.0)


        # --- Blended Forecast ---
        # Ensure lengths match, pad with last value if GARCH/XGBoost forecast was shorter than horizon (shouldn't happen with arch.forecast(horizon=...))
        garch_vols_padded = np.pad(garch_vols, (0, max(0, forecast_horizon - len(garch_vols))), mode='edge')
        xgb_vols_padded = np.pad(xgb_vols, (0, max(0, forecast_horizon - len(xgb_vols))), mode='edge')

        # Ensure both arrays have values before blending
        if len(garch_vols_padded) > 0 and len(xgb_vols_padded) > 0:
            # Blend based on which model's *first* forecast point is closer to the last Realized Vol
            garch_first_pred = garch_vols_padded[0] if len(garch_vols_padded) > 0 else realized_vol
            xgb_first_pred = xgb_vols_padded[0] if len(xgb_vols_padded) > 0 else realized_vol

            garch_diff = np.abs(garch_first_pred - realized_vol)
            xgb_diff = np.abs(xgb_first_pred - realized_vol)

            # Avoid division by zero if both diffs are zero or very small
            total_diff = garch_diff + xgb_diff
            if total_diff > 1e-6: # Use a small threshold to avoid division by zero
                 garch_weight = xgb_diff / total_diff
                 xgb_weight = garch_diff / total_diff # Weights should sum to 1
            else:
                 garch_weight = 0.5
                 xgb_weight = 0.5

            blended_vols = [(garch_weight * g) + (xgb_weight * x) for g, x in zip(garch_vols_padded, xgb_vols_padded)]
        else:
             # Fallback if either forecast failed entirely
            blended_vols = np.full(forecast_horizon, realized_vol)
            garch_weight = 0.5
            xgb_weight = 0.5


        # Confidence Score - Based on agreement between models and RMSE
        # Scale RMSE: Higher RMSE means lower confidence. Lower RMSE is better.
        # Max reasonable RMSE could be e.g., 10-15%. Map 0 RMSE to high confidence, high RMSE to low.
        max_expected_rmse = 15.0 # Define a threshold for 'high' RMSE
        rmse_confidence = max(0, 100 - (rmse / max_expected_rmse) * 50) # Example scaling

        # Model Agreement Confidence: High difference between GARCH/XGBoost forecasts = lower confidence
        agreement_diff = np.abs(np.mean(garch_vols_padded) - np.mean(xgb_vols_padded)) # Use mean difference over horizon
        max_expected_diff = 5.0 # Define a threshold for 'high' difference
        agreement_confidence = max(0, 100 - (agreement_diff / max_expected_diff) * 50) # Example scaling

        # Combine RMSE and Agreement confidence (example blend)
        confidence_score = 0.6 * rmse_confidence + 0.4 * agreement_confidence
        confidence_score = np.clip(confidence_score, 10, 100) # Clip score to a range


        future_dates = pd.date_range(start=df.index.max() + timedelta(days=1), periods=forecast_horizon, freq='B') # Start from day *after* last data point

        forecast_log = pd.DataFrame({
            "Date": future_dates,
            "GARCH_Vol": garch_vols_padded,
            "XGBoost_Vol": xgb_vols_padded,
            "Blended_Vol": blended_vols,
            "Confidence": [confidence_score] * forecast_horizon # Apply same confidence to all forecast days for simplicity
        })

        logger.debug("Volatility forecast completed")
        return forecast_log, garch_vols_padded, xgb_vols_padded, blended_vols, realized_vol, confidence_score, rmse, feature_importances
    except Exception as e:
        st.error(f"Error in volatility forecasting: {str(e)}")
        logger.error(f"Error in volatility forecasting: {str(e)}")
        # Return empty/fallback data on error
        future_dates = pd.date_range(start=df.index.max() + timedelta(days=1), periods=forecast_horizon, freq='B')
        fallback_vix = df["VIX"].iloc[-1] if len(df) > 0 and "VIX" in df.columns else 15.0
        basic_forecast = pd.DataFrame({
             "Date": future_dates,
             "GARCH_Vol": fallback_vix,
             "XGBoost_Vol": fallback_vix,
             "Blended_Vol": fallback_vix,
             "Confidence": 20 # Very low confidence on error
        })
        # Ensure feature_importances matches length even on error
        feature_importances_fallback = np.full(len(XGB_FEATURE_COLS), 0.0)
        return basic_forecast, [fallback_vix] * forecast_horizon, [fallback_vix] * forecast_horizon, [fallback_vix] * forecast_horizon, fallback_vix, 20.0, 999.99, feature_importances_fallback


# --- Strategy Generation ---
@st.cache_data
def generate_trading_strategy(df, forecast_log, realized_vol, risk_tolerance, confidence_score, capital):
    """Generates a recommended trading strategy based on market conditions and forecast."""
    try:
        logger.info("Generating trading strategy")
        df = df.copy()
        df.index = df.index.normalize()

        if df.empty or forecast_log is None or forecast_log.empty:
             logger.warning("Cannot generate strategy due to empty data or forecast.")
             return None, ["Insufficient data or forecast not available for strategy generation."]


        latest = df.iloc[-1]
        # Use the blended forecast for the next day (index 0 of forecast_log) as the primary indicator
        next_day_forecast_vol = forecast_log["Blended_Vol"].iloc[0] if not forecast_log.empty else latest["VIX"] if "VIX" in latest else 15.0

        # Use current ATM IV and Realized Vol from latest data
        iv = latest["ATM_IV"] if "ATM_IV" in latest else latest["VIX"] if "VIX" in latest else 15.0
        hv = latest["Realized_Vol"] if "Realized_Vol" in latest else latest["VIX"] if "VIX" in latest else 15.0
        iv_hv_gap = iv - hv
        iv_skew = latest["IV_Skew"] if "IV_Skew" in latest else 0.0
        pcr = latest["PCR"] if "PCR" in latest else 1.0
        dte = latest["Days_to_Expiry"] if "Days_to_Expiry" in latest else 7 # Assume 7 days if missing
        event_flag = latest["Event_Flag"] if "Event_Flag" in latest else 0
        latest_vix = latest["VIX"] if "VIX" in latest else 15.0
        latest_vix_change_pct = latest["VIX_Change_Pct"] if "VIX_Change_Pct" in latest else 0.0
        latest_max_pain_diff_pct = latest["Spot_MaxPain_Diff_Pct"] if "Spot_MaxPain_Diff_Pct" in latest else 0.5


        # Determine Volatility Regime based on NEXT DAY's FORECASTED volatility
        if event_flag == 1: # Event takes precedence
            regime = "EVENT-DRIVEN"
        elif next_day_forecast_vol < 15:
            regime = "LOW"
        elif next_day_forecast_vol < 20:
            regime = "MEDIUM"
        else:
            regime = "HIGH"

        strategy = "Undefined"
        reason = "N/A"
        tags = []
        risk_reward_target = 1.5 # Base target risk-reward

        # Strategy Selection Logic (Refined)
        if regime == "LOW":
            if iv_hv_gap > 3 and dte < 15: # Higher gap in low vol indicates potential compression
                strategy = "Butterfly Spread"
                reason = f"Low forecasted vol ({next_day_forecast_vol:.1f}%) & short expiry ({dte} days) favors pinning strategies with high Theta decay."
                tags = ["Neutral", "Theta Decay", "Expiry Play"]
                risk_reward_target = 2.5 # Higher potential R:R for Butterflies
            else:
                strategy = "Iron Fly"
                reason = f"Low forecasted vol ({next_day_forecast_vol:.1f}%) and time decay favors delta-neutral Iron Fly in expected range."
                tags = ["Neutral", "Theta Decay", "Range Bound"]
                risk_reward_target = 2.0 # Iron Fly generally has better R:R than Iron Condor

        elif regime == "MEDIUM":
            if iv_hv_gap > 2 and iv_skew > 1.5: # Gap + Skew suggests specific opportunity
                strategy = "Iron Condor"
                reason = f"Medium forecasted vol ({next_day_forecast_vol:.1f}%) and moderate skew favor premium collection with defined risk."
                tags = ["Neutral", "Theta Decay", "Range Bound", "Defined Risk"]
                risk_reward_target = 1.8
            else:
                # Consider Short Strangle if conditions are stable, otherwise wider Iron Condor
                if event_flag == 0 and dte > 7 and confidence_score > 70: # Stable, not too close to expiry, high confidence
                    strategy = "Short Strangle"
                    reason = f"Balanced forecasted vol ({next_day_forecast_vol:.1f}%) and reasonable confidence ({int(confidence_score)}%) offer good premium capture in stable conditions."
                    tags = ["Neutral", "Premium Selling", "Volatility Harvest"]
                    risk_reward_target = 1.6 # Higher premium but potentially higher undefined risk
                else:
                    strategy = "Iron Condor"
                    reason = f"Medium forecasted vol ({next_day_forecast_vol:.1f}%) favors defined risk premium collection."
                    tags = ["Neutral", "Theta Decay", "Range Bound", "Defined Risk"]
                    risk_reward_target = 1.7


        elif regime == "HIGH":
            if iv_hv_gap > 5 and iv_skew > 2: # High IV + Call Skew
                strategy = "Jade Lizard"
                reason = f"High forecasted vol ({next_day_forecast_vol:.1f}%) and notable call skew suggest potential for Jade Lizard (defined upside risk)."
                tags = ["Skewed", "Volatility", "Defined Risk"]
                risk_reward_target = 1.3
            else:
                # High vol favors defined risk strategies, Iron Condor is a general choice
                 strategy = "Iron Condor"
                 reason = f"High forecasted vol ({next_day_forecast_vol:.1f}%) favors collecting premium with defined risk."
                 tags = ["Neutral", "Theta Decay", "Range Bound", "Defined Risk"]
                 risk_reward_target = 1.5

        elif regime == "EVENT-DRIVEN":
            if iv > 25 and dte < 5: # High IV spike near event/expiry
                strategy = "Short Straddle"
                reason = f"Event risk near expiry ({dte} days) with high IV spike ({iv:.1f}%) offers significant premium capture potential."
                tags = ["Volatility", "Event", "Neutral", "High Premium"]
                risk_reward_target = 1.4 # High premium, but also high risk

            else:
                # If IV is high but not extreme, or DTE is not very low, consider Calendar
                # Calendar spread logic for recommendation only (order generation is difficult)
                # Let's stick to simpler strategies for order generation as discussed.
                # If Event/High IV but Short Straddle conditions not met, maybe Iron Condor for defined risk during event?
                strategy = "Iron Condor" # Fallback to defined risk
                reason = f"Event risk present, but conditions not ideal for Short Straddle. Recommend defined risk Iron Condor."
                tags = ["Volatility", "Event", "Defined Risk"]
                risk_reward_target = 1.3


        # --- Risk Management based on risk tolerance and regime ---
        # Base capital allocation % per regime
        capital_alloc_pct = {
            "LOW": 0.15, # Can deploy more in low vol (less risk of large swings)
            "MEDIUM": 0.10,
            "HIGH": 0.07,
            "EVENT-DRIVEN": 0.08 # Managed risk even during events
        }

        # Multiplier based on risk tolerance
        position_size_multiplier = {"Conservative": 0.7, "Moderate": 1.0, "Aggressive": 1.3}[risk_tolerance]

        # Calculate recommended deployable capital
        recommended_deploy_capital = capital * capital_alloc_pct.get(regime, 0.1) * position_size_multiplier

        # Define Max Recommended Loss (as a percentage of deployed capital)
        # Riskier strategies/regimes might have a slightly wider allowed loss initially
        max_loss_pct_of_deploy = {
            "Short Straddle": 0.3, # Higher initial loss buffer for high premium capture
            "Short Strangle": 0.25,
            "Iron Condor": 0.15,
            "Iron Fly": 0.2,
            "Butterfly Spread": 0.1, # Lower loss tolerance for debit/low credit spreads
            "Jade Lizard": 0.2,
            # "Calendar Spread": 0.15, # Removed as strategy, keeping logic comment
            "Undefined": 0.1
        }.get(strategy, 0.15) # Default max loss percentage

        recommended_max_loss = recommended_deploy_capital * max_loss_pct_of_deploy

        # Calculate Exposure % relative to total capital
        total_exposure_pct = (recommended_deploy_capital / capital) * 100 if capital > 0 else 0

        # --- Risk Flags (Based on latest data and thresholds) ---
        risk_flags = []
        if latest_vix > 22: # Higher threshold for warning
            risk_flags.append(f"VIX ({latest_vix:.1f}%) is elevated.")
        if latest_vix_change_pct > 8:
             risk_flags.append(f"Significant VIX spike today (+{latest_vix_change_pct:.1f}%).")
        if latest_max_pain_diff_pct > 3.5: # Moderate diff is normal, larger suggests directional skew/pinning risk
             risk_flags.append(f"Spot ({latest['NIFTY_Close']:,.0f}) is {latest_max_pain_diff_pct:.1f}% away from Max Pain ({real_data['max_pain_strike'] if real_data and real_data['max_pain_strike'] else 'N/A'}).")
        if iv_skew > 2.5:
             risk_flags.append(f"High IV Skew ({iv_skew:.1f}). Call/Put premiums are significantly different.")
        if total_exposure_pct > 20: # Warning if allocation is high relative to total capital
            risk_flags.append(f"Recommended Exposure ({total_exposure_pct:.1f}%) is relatively high based on your capital.")
        # Add a check for very low confidence score
        if confidence_score < 60:
            risk_flags.append(f"Volatility Forecast Confidence ({int(confidence_score)}%) is low.")
        if regime == "EVENT-DRIVEN":
             risk_flags.append(f"Market is in an EVENT-DRIVEN regime ({dte} days to expiry). Increased uncertainty.")


        # --- Discipline/Behavior Score ---
        # Calculate Behavior Score (Placeholder - can integrate journaling here later)
        # For now, just check if recommended allocation is excessively high regardless of tolerance
        behavior_warnings = []
        behavior_score = 10 # Start high

        # Deduct points for high allocation relative to profile
        if recommended_deploy_capital > 0.25 * capital and risk_tolerance == "Conservative":
             behavior_warnings.append("Recommended allocation seems high for a Conservative profile.")
             behavior_score -= 2
        if recommended_deploy_capital > 0.35 * capital and risk_tolerance == "Moderate":
             behavior_warnings.append("Recommended allocation seems high for a Moderate profile.")
             behavior_score -= 1
        if recommended_deploy_capital > 0.5 * capital and risk_tolerance == "Aggressive":
             behavior_warnings.append("Recommended allocation seems very high.")
             behavior_score -= 1

        # Deduct points for risk flags
        if risk_flags:
             behavior_warnings.append(f"Note: {len(risk_flags)} risk flag(s) detected.")
             # Deduct points based on number/severity of flags (simple: 1 point per flag up to 5)
             behavior_score -= min(len(risk_flags), 5)

        # Deduct points if forecast confidence is low
        if confidence_score < 50:
             behavior_warnings.append(f"Note: Low forecast confidence ({int(confidence_score)}%) makes analysis less reliable.")
             behavior_score -= 2

        behavior_score = max(1, min(behavior_score, 10)) # Clip score between 1 and 10


        # --- Violation Tracking ---
        # If there are risk flags and the user hasn't journaled since the last flag appeared
        if risk_flags and not st.session_state.journal_complete:
            st.session_state.violations += 1 # Increment violation count if flags are present
            logger.warning(f"Risk flags detected. Incrementing violation count to {st.session_state.violations}.")
        elif not risk_flags and st.session_state.violations > 0 and not st.session_state.journal_complete:
             # If violations accumulated but no flags this time, and not journaled,
             # don't increment, but keep the lock if violations >= 2.
             pass # Violations persist until journaled
        elif st.session_state.journal_complete:
            # If journal was completed, violations were reset. Start fresh.
             pass # Violations should be 0 here


        logger.debug(f"Trading strategy generated: {strategy} ({regime}). Violations: {st.session_state.violations}, Journal Complete: {st.session_state.journal_complete}")
        return {
            "Regime": regime,
            "Strategy": strategy,
            "Reason": reason,
            "Tags": tags,
            "Forecast_Confidence": confidence_score, # Confidence in the *forecast*
            "Risk_Reward_Target": risk_reward_target,
            "Recommended_Deploy_Capital": recommended_deploy_capital,
            "Recommended_Max_Loss": recommended_max_loss,
            "Recommended_Exposure_Pct": total_exposure_pct,
            "Risk_Flags": risk_flags,
            "Behavior_Score": behavior_score,
            "Behavior_Warnings": behavior_warnings,
            "Latest_Data": latest # Include latest data for order generation
        }, [] # Return empty list for errors if successful


    except Exception as e:
        st.error(f"Error generating strategy: {str(e)}")
        logger.error(f"Error generating strategy: {str(e)}")
        return None, [f"An error occurred during strategy generation: {str(e)}"]


def generate_trade_orders_display(strategy, real_time_data, capital, lot_size=LOT_SIZE):
    """Generates the details of the orders for the recommended strategy to be displayed."""
    logger.info(f"Generating order details for strategy: {strategy['Strategy']}")

    if not real_time_data or "option_chain" not in real_time_data or real_time_data["option_chain"].empty:
        return None, "Cannot generate orders: Real-time option chain data is missing or empty."
    if real_time_data.get("atm_strike", 0) == 0:
         return None, "Cannot generate orders: ATM strike could not be determined from real-time data."
    if real_time_data.get("expiry") is None:
        return None, "Cannot generate orders: Expiry date could not be determined from real-time data."


    option_chain_df = real_time_data["option_chain"]
    atm_strike = real_time_data["atm_strike"]
    expiry_date_str = real_time_data["expiry"]
    recommended_deploy_capital = strategy["Recommended_Deploy_Capital"]

    # Calculate number of lots based on recommended capital and current straddle price (as a proxy for premium)
    # This is a simplification. A real system would calculate premium per lot more accurately for each leg.
    # Use ATM Straddle price from real_time_data as a base premium reference
    base_premium_per_lot = real_time_data.get("straddle_price", 0) * lot_size

    # Fallback premium estimate if real straddle price is zero or missing
    if base_premium_per_lot == 0 or pd.isna(base_premium_per_lot):
         atm_iv = real_time_data.get("vix", 15.0) # Use VIX as ATM IV fallback
         nifty_spot = real_time_data.get("nifty_spot", 18000.0)
         # Need DTE for the latest date to estimate straddle price
         if st.session_state.feature_data is not None and not st.session_state.feature_data.empty:
             latest_dte = st.session_state.feature_data["Days_to_Expiry"].iloc[-1]
         else:
              latest_dte = 7 # Fallback DTE
         dte_years = max(1, latest_dte) / 252.0 # Ensure DTE is positive
         # Rough approximation: Straddle Price ~ 2 * Spot * IV * sqrt(DTE/365) * Lot Size
         base_premium_per_lot = 2 * nifty_spot * (atm_iv/100) * np.sqrt(dte_years) * lot_size
         logger.warning(f"Using estimated premium per lot ({base_premium_per_lot:.2f}) as real straddle price was missing or zero.")


    if base_premium_per_lot == 0 or pd.isna(base_premium_per_lot):
         return None, "Cannot estimate premium per lot to calculate recommended quantity."

    # Calculate recommended lots. Ensure at least 1 lot if deployable capital is positive.
    # Use a safety factor - maybe only allocate 80% of recommended capital to determine lots
    # as the actual premium might be higher/lower.
    allocation_for_lots = recommended_deploy_capital * 0.8
    recommended_lots = int(allocation_for_lots / base_premium_per_lot) if base_premium_per_lot > 0 else 0
    recommended_lots = max(1, recommended_lots) if recommended_deploy_capital > 0 else 0

    if recommended_lots == 0:
         return None, "Recommended capital is insufficient to trade even 1 lot."


    # --- Determine strikes for the chosen strategy ---
    strikes_config = [] # List of (strike_adjustment_from_ATM, CPType, BuySell, LegDescription)

    if strategy["Strategy"] == "Short Straddle":
        # Sell ATM Call, Sell ATM Put
        strikes_config = [(0, "CE", "S", "Short Call"), (0, "PE", "S", "Short Put")]
    elif strategy["Strategy"] == "Short Strangle":
        # Sell OTM Call, Sell OTM Put (e.g., 100 points out or based on IV/DTE)
        # Let's use a fixed points out for simplicity, but dynamic could be better
        otm_adjustment_points = 100 # Points OTM
        strikes_config = [(otm_adjustment_points, "CE", "S", "Short Call"), (-otm_adjustment_points, "PE", "S", "Short Put")]
    elif strategy["Strategy"] == "Iron Condor":
        # Sell OTM Call, Buy Far OTM Call | Sell OTM Put, Buy Far OTM Put
        # Use fixed points relative to ATM
        short_otm_adj_points = 150 # Slightly wider short legs
        long_otm_adj_points = 300 # Wider protection legs
        strikes_config = [
            (short_otm_adj_points, "CE", "S", "Short Call"),
            (long_otm_adj_points, "CE", "B", "Long Call"),
            (-short_otm_adj_points, "PE", "S", "Short Put"),
            (-long_otm_adj_points, "PE", "B", "Long Put")
        ]
    elif strategy["Strategy"] == "Iron Fly":
        # Sell ATM Straddle, Buy OTM Wings
        wing_adj_points = 100
        strikes_config = [
            (0, "CE", "S", "Short Call"),
            (wing_adj_points, "CE", "B", "Long Call (OTM)"),
            (0, "PE", "S", "Short Put"), # Sell ATM Put
            (-wing_adj_points, "PE", "B", "Long Put (OTM)")
        ]
    elif strategy["Strategy"] == "Butterfly Spread":
        # Assuming Call Butterfly: Buy ITM, Sell 2x ATM, Buy OTM
        # Define in relation to ATM: Buy ATM-Wing, Sell 2x ATM, Buy ATM+Wing
        wing_adj_points = 100
        strikes_config = [
            (-wing_adj_points, "CE", "B", "Long Call (ITM)"),
            (0, "CE", "S", "Short 2x Call (ATM)"), # Note: Quantity will be 2x here
            (wing_adj_points, "CE", "B", "Long Call (OTM)")
        ]
        # Adjust lots for the 2x leg - Butterfly quantity is often quoted per unit of 2x legs
        # The calculated recommended_lots is based on total capital/premium for the *entire* strategy.
        # For Butterfly, the number of *units* is half the total lots for the middle leg.
        # Let's keep recommended_lots as the standard contract count (like Short Straddle) and adjust leg quantity below.

    elif strategy["Strategy"] == "Jade Lizard":
        # Short OTM Call, Short OTM Put, Long Far OTM Put (Defined upside, undefined downside but buffered)
        short_call_adj_points = 100
        short_put_adj_points = -100
        long_put_adj_points = -200 # Further OTM put to define risk
        strikes_config = [
            (short_call_adj_points, "CE", "S", "Short Call"),
            (short_put_adj_points, "PE", "S", "Short Put"),
            (long_put_adj_points, "PE", "B", "Long Put (Protection)")
        ]

    else:
        # Fallback for Undefined or unhandled strategies (shouldn't happen if strategy logic is sound)
        return None, f"Order generation logic not defined for strategy: {strategy['Strategy']}"


    # --- Generate Order Details ---
    order_details_list = []

    for adj_points, cp_type, buy_sell, leg_desc in strikes_config:
        # Calculate the target strike based on ATM strike and adjustment
        target_strike = atm_strike + adj_points

        # Find the closest available strike in the option chain for this CP type and expiry
        available_strikes_for_type = option_chain_df[option_chain_df["CPType"] == cp_type]["StrikeRate"].unique()
        if not available_strikes_for_type.any():
             return None, f"No available strikes found for {cp_type} options in the option chain for this expiry."

        # Find the strike in the available list that is closest to our target strike
        closest_strike = available_strikes_for_type[(np.abs(available_strikes_for_type - target_strike)).argmin()]

        # Find the scrip code for the closest strike, CP type, and expiry
        # Assuming the option chain DF only contains the relevant expiry fetched earlier
        option_scrip_data = option_chain_df[
            (option_chain_df["StrikeRate"] == closest_strike) &
            (option_chain_df["CPType"] == cp_type)
        ]

        if option_scrip_data.empty:
             return None, f"Could not find ScripCode for {cp_type} at strike {closest_strike} for the selected expiry."

        # Use .iloc[0] because there should be only one row matching strike/CPType for a given expiry
        scrip_code = int(option_scrip_data["ScripCode"].iloc[0])
        # Use LTP as a reference price, but recommend Market order for fills
        last_traded_price = float(option_scrip_data["LastRate"].iloc[0])

        # Determine quantity for this specific leg
        leg_quantity_units = recommended_lots * lot_size
        # For Butterfly's 2x middle leg, double the quantity
        if "2x" in leg_desc: # Check for "2x" in description
             leg_quantity_units *= 2

        order_details = {
            "Action": "SELL" if buy_sell == "S" else "BUY",
            "Instrument": f"NIFTY 50 {cp_type}",
            "Strike": closest_strike,
            "Expiry": expiry_date_str,
            "Quantity (Lots)": leg_quantity_units // lot_size, # Display in Lots
            "Quantity (Units)": leg_quantity_units,
            "ScripCode": scrip_code,
            "Reference LTP": last_traded_price,
            "Recommended Order Type": "Market Order", # Usually best for spreads
            "Leg": leg_desc
            # You could add theoretical price calculation here based on BS or IV if needed
        }
        order_details_list.append(order_details)


    logger.info(f"Generated {len(order_details_list)} order details.")
    if not order_details_list:
         return None, "Order details list is empty after processing."

    return order_details_list, "" # Return list of orders and empty error message


# --- Backtesting ---
def run_backtest(df, capital, strategy_choice, start_date, end_date):
    """Runs a backtest simulation based on historical data and strategy logic."""
    try:
        logger.info(f"Starting backtest for {strategy_choice} from {start_date} to {end_date}")

        # Filter data based on date range
        df_backtest = df.loc[start_date:end_date].copy()

        if df_backtest.empty:
            st.warning("Backtest failed: No data available in the selected date range.")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame(), [capital], 0.0, 0.0, 0.0 # Return empty/fallback
        if len(df_backtest) < 50: # Need minimum data for meaningful backtest
            st.warning(f"Backtest failed: Insufficient data ({len(df_backtest)} days) in the selected date range. Need at least 50 days.")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame(), [capital], 0.0, 0.0, 0.0 # Return empty/fallback


        # Ensure required columns for backtesting simulation are present
        required_cols_backtest = ["NIFTY_Close", "ATM_IV", "Realized_Vol", "IV_Skew", "Days_to_Expiry", "Event_Flag", "Straddle_Price", "VIX"]
        missing_cols = [col for col in required_cols_backtest if col not in df_backtest.columns]
        if missing_cols:
            st.error(f"Backtest failed: Missing required data columns for simulation: {missing_cols}")
            logger.error(f"Missing backtest columns: {missing_cols}")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame(), [capital], 0.0, 0.0, 0.0 # Return empty/fallback

        # Ensure all required columns are numeric, coercing errors
        for col in required_cols_backtest:
             df_backtest[col] = pd.to_numeric(df_backtest[col], errors='coerce')

        # Re-fill any NaNs introduced by coercion
        if df_backtest[required_cols_backtest].isna().sum().sum() > 0:
             logger.warning(f"NaNs found in critical backtest columns after coercion: {df_backtest[required_cols_backtest].isna().sum()[df_backtest[required_cols_backtest].isna().sum() > 0].to_dict()}")
             df_backtest = df_backtest.fillna(method='bfill').fillna(method='ffill')
             # If still NaNs (e.g., first row), fill with a reasonable default
             for col in required_cols_backtest:
                  if df_backtest[col].isna().any():
                      df_backtest[col] = df_backtest[col].fillna(df_backtest[col].mean() if df_backtest[col].mean() is not np.nan else (df_backtest[col].iloc[0] if not df_backtest[col].empty else 1.0)) # Use mean if possible, else first value, else 1.0/sensible default


        backtest_results = []
        portfolio_value = float(capital) # Start with initial capital, ensure float
        daily_portfolio_values = [portfolio_value] # Track portfolio value daily
        peak_portfolio_value = portfolio_value
        drawdowns = [0.0] # Track value drawdown
        drawdown_pcts = [0.0] # Track percentage drawdown

        lot_size = LOT_SIZE # Use configured lot size
        base_transaction_cost = BASE_TRANSACTION_COST_PCT
        stt_cost = STT_PCT

        # Simulate trading day by day
        for i in range(1, len(df_backtest)):
            try:
                day_data = df_backtest.iloc[i]
                prev_day_data = df_backtest.iloc[i-1]
                date = day_data.name

                # --- Strategy Engine for Backtest ---
                # Recalculate average volatility based on trailing data for realism
                trailing_vol_window = 20 # Days
                # Ensure window does not go before the start of the backtest data
                start_window = max(0, i-trailing_vol_window)
                avg_vol_trailing = df_backtest["Realized_Vol"].iloc[start_window:i].mean() if i > start_window else day_data["Realized_Vol"] # Use day's vol if window too small

                # Simplified strategy engine within backtest loop
                iv_i = day_data["ATM_IV"]
                hv_i = day_data["Realized_Vol"]
                iv_hv_gap_i = iv_i - hv_i
                iv_skew_i = day_data["IV_Skew"]
                dte_i = day_data["Days_to_Expiry"]
                event_flag_i = day_data["Event_Flag"]
                straddle_price_i = day_data["Straddle_Price"] # Use the synthetic/adjusted straddle price

                # Determine Regime for the day based on trailing average volatility
                if event_flag_i == 1:
                    regime = "EVENT-DRIVEN"
                elif avg_vol_trailing < 15:
                    regime = "LOW"
                elif avg_vol_trailing < 20:
                    regime = "MEDIUM"
                else:
                    regime = "HIGH"

                # Determine Strategy for the day based on regime and simple rules
                current_strategy = "Undefined"
                # Note: The strategy rules here must match the rules in generate_trading_strategy
                # for the backtest to simulate the recommended strategy logic.
                if regime == "LOW":
                    if iv_hv_gap_i > 3 and dte_i < 15:
                        current_strategy = "Butterfly Spread"
                    else:
                        current_strategy = "Iron Fly"
                elif regime == "MEDIUM":
                    if iv_hv_gap_i > 2 and iv_skew_i > 1.5:
                        current_strategy = "Iron Condor"
                    # Add checks for Short Strangle conditions if applicable (match strategy func)
                    # Simplification for backtest: Always pick one if regime/gap/skew fits
                    elif event_flag_i == 0 and dte_i > 7: # Add conditions like in strategy func
                         current_strategy = "Short Strangle"
                    else:
                         current_strategy = "Iron Condor" # Default for medium if strangle conditions not met
                elif regime == "HIGH":
                    if iv_hv_gap_i > 5 and iv_skew_i > 2:
                        current_strategy = "Jade Lizard"
                    else:
                         current_strategy = "Iron Condor"
                elif regime == "EVENT-DRIVEN":
                    if iv_i > 25 and dte_i < 5:
                        current_strategy = "Short Straddle"
                    else:
                         current_strategy = "Iron Condor" # Fallback defined risk during event


                # Filter strategy if user selected a specific one for backtest
                if strategy_choice != "All Strategies" and current_strategy != strategy_choice:
                    current_strategy = "No Trade" # Explicitly mark as no trade day

                # Determine Risk-Reward target for logging (based on strategy logic)
                risk_reward = {
                     "Butterfly Spread": 2.5,
                     "Iron Fly": 2.0,
                     "Iron Condor": 1.8,
                     "Short Strangle": 1.6,
                     "Jade Lizard": 1.3,
                     "Short Straddle": 1.4,
                     "No Trade": 0.0, # No risk reward for no trade
                     "Undefined": 0.0 # Should not happen if logic is complete
                }.get(current_strategy, 1.0)


                # --- Simulate Trade Execution and P&L ---
                simulated_pnl = 0.0
                deploy_capital_for_trade = 0.0
                max_loss_for_trade = 0.0

                if current_strategy != "No Trade":
                    # Estimate deployable capital for this trade (simplified)
                    # Using a fixed percentage of *current portfolio value* based on regime
                    capital_alloc_pct_backtest = {"LOW": 0.15, "MEDIUM": 0.10, "HIGH": 0.07, "EVENT-DRIVEN": 0.08}.get(regime, 0.1)
                    deploy_capital_for_trade = portfolio_value * capital_alloc_pct_backtest
                    # Define Max Recommended Loss (as a percentage of deployed capital)
                    max_loss_pct_of_deploy_backtest = {
                        "Short Straddle": 0.3,
                        "Short Strangle": 0.25,
                        "Iron Condor": 0.15,
                        "Iron Fly": 0.2,
                        "Butterfly Spread": 0.1,
                        "Jade Lizard": 0.2,
                        "Undefined": 0.1 # Should not happen
                    }.get(current_strategy, 0.15)

                    max_loss_for_trade = deploy_capital_for_trade * max_loss_pct_of_deploy_backtest


                    # Estimate premium received (simplified, based on straddle price as proxy)
                    # Adjust premium based on strategy type (e.g., straddle/strangle yield more premium than iron condor)
                    premium_factor = {
                        "Short Straddle": 1.0,
                        "Short Strangle": 0.8,
                        "Iron Condor": 0.4, # Lower premium factor for spreads
                        "Iron Fly": 0.6,
                        "Butterfly Spread": 0.3, # Butterflies are often debit or low credit
                        "Jade Lizard": 0.7,
                    }.get(current_strategy, 0.5)

                    # Ensure straddle_price_i is not zero or NaN before division
                    estimated_premium_base = straddle_price_i * lot_size if straddle_price_i > 0 and not pd.isna(straddle_price_i) else 0
                    # Calculate number of units based on deployed capital and base premium per unit (straddle)
                    # Then estimate total premium based on the strategy's premium factor
                    num_units = int(deploy_capital_for_trade / estimated_premium_base) if estimated_premium_base > 0 else 0
                    num_units = max(1, num_units) if deploy_capital_for_trade > 0 else 0 # Ensure at least 1 unit if deploying

                    estimated_premium = estimated_premium_base * num_units * premium_factor # Total estimated premium for the position


                    # Add Transaction Costs and Slippage (simplified simulation)
                    extra_cost_sim = 0.001 if "Iron" in current_strategy or "Butterfly" in current_strategy or "Jade" in current_strategy else 0 # Example extra cost for spreads
                    total_cost_pct = base_transaction_cost + extra_cost_sim + stt_cost
                    # Dynamic slippage simulation based on IV and DTE
                    slippage_pct = get_dynamic_slippage_sim(current_strategy, iv_i, dte_i)

                    estimated_premium_after_costs = estimated_premium * (1 - total_cost_pct - slippage_pct)

                    # Simulate P&L based on market move, IV, and DTE
                    # Ensure prev_day_data['NIFTY_Close'] is not zero before division
                    nifty_move_pct = (day_data["NIFTY_Close"] - prev_day_data["NIFTY_Close"]) / prev_day_data["NIFTY_Close"] if prev_day_data.get("NIFTY_Close", 0) != 0 else 0.0

                    # P&L Simulation Logic (Highly Simplified)
                    # Premium Decay (Theta): Increases P&L over time - more decay as expiry nears
                    theta_decay = estimated_premium_after_costs * (1 - np.clip(dte_i / 30, 0.1, 1.0)) * 0.5 # Decay factor, capped at 10% of premium

                    # Simulate loss from directional move (Gamma risk for neutral strategies)
                    # Loss is proportional to the absolute Nifty move relative to expected move (based on IV)
                    expected_daily_move_pct = (iv_i / 100) / np.sqrt(252) if iv_i > 0 else 0.0
                    if expected_daily_move_pct > 1e-6: # Avoid division by zero
                         move_factor = abs(nifty_move_pct) / expected_daily_move_pct
                    else:
                         move_factor = 0.0
                    # If move is larger than expected, incur loss. Loss proportional to move_factor and deployed capital.
                    # Impact varies by strategy - Butterfly/Iron Fly are very sensitive near expiry
                    move_impact_factor = {
                         "Short Straddle": 1.5, "Short Strangle": 1.2,
                         "Iron Fly": 1.8, "Butterfly Spread": 2.0, # Higher impact near expiry
                         "Iron Condor": 0.8, "Jade Lizard": 1.0, # Lower impact for wider/defined risk
                         "Undefined": 1.0
                    }.get(current_strategy, 1.0)
                    loss_from_move = max(0, move_factor - 0.5) * deploy_capital_for_trade * 0.3 * move_impact_factor # Simplified loss calculation, only for significant moves

                    # Simulate P&L from IV change (vega effect - short vol strategies lose when IV goes up)
                    vix_change_pct_i = day_data["VIX_Change_Pct"] if "VIX_Change_Pct" in day_data else 0.0
                    # Vega impact proportional to VIX change and deployed capital (simplified)
                    # Short vol strategies (straddle, strangle, condor, fly) have negative vega
                    # Long vol strategies (calendar) have positive vega (not simulated here)
                    vega_impact_factor = {
                         "Short Straddle": -1.0, "Short Strangle": -0.8,
                         "Iron Condor": -0.5, "Iron Fly": -0.7,
                         "Butterfly Spread": -0.3, "Jade Lizard": -0.6,
                         "Undefined": 0.0 # Should not happen
                    }.get(current_strategy, 0.0)

                    iv_change_impact = vega_impact_factor * (vix_change_pct_i / 10.0) * deploy_capital_for_trade * 0.1 # Simplified impact calculation


                    simulated_pnl = estimated_premium_after_costs + theta_decay - loss_from_move + iv_change_impact

                    # Apply random shocks (as in original code)
                    simulated_pnl = apply_volatility_shock_sim(simulated_pnl, nifty_move_pct, iv_i, event_flag_i, deploy_capital_for_trade)
                    simulated_pnl = apply_gap_and_crash_sim(simulated_pnl, event_flag_i, iv_i, estimated_premium_after_costs)

                    # Apply Max Loss limit (stop loss simulation)
                    # If PnL is below -MaxLoss, simulate exiting the trade with max loss
                    if simulated_pnl < -max_loss_for_trade:
                         simulated_pnl = -max_loss_for_trade # Cap loss at MaxLoss

                    # Apply Profit Target (optional but can simulate managing winners)
                    # If PnL exceeds Profit Target (e.g., 50% of deployed), simulate exiting with that profit
                    profit_target = deploy_capital_for_trade * 0.5
                    if simulated_pnl > profit_target:
                         simulated_pnl = profit_target # Cap profit at target


                    # Ensure PnL is within a reasonable range relative to deployed capital
                    # This prevents extreme outliers from the simplified simulation
                    # Cap daily PnL between -100% and +100% of deployed capital
                    simulated_pnl = np.clip(simulated_pnl, -deploy_capital_for_trade, deploy_capital_for_trade)


                # Update portfolio value
                portfolio_value += simulated_pnl

                # Log the trade day result
                backtest_results.append({
                    "Date": date,
                    "Regime": regime,
                    "Strategy": current_strategy,
                    "PnL": simulated_pnl,
                    "Capital_Deployed": deploy_capital_for_trade,
                    "Max_Loss": max_loss_for_trade, # Log the simulated max loss for the trade
                    "Risk_Reward_Target": risk_reward, # Log the target risk reward
                    "Realized_Vol": hv_i, # Log key metrics for analysis
                    "ATM_IV": iv_i,
                    "Days_to_Expiry": dte_i,
                    "Event_Flag": event_flag_i,
                    "Portfolio_Value": portfolio_value # Log portfolio value at end of day
                })

                # Track portfolio value and drawdown daily
                # The portfolio value was already updated
                daily_portfolio_values.append(portfolio_value)
                peak_portfolio_value = max(peak_portfolio_value, portfolio_value)
                drawdown = peak_portfolio_value - portfolio_value
                drawdowns.append(drawdown)
                # Percentage Drawdown calculation
                drawdown_pct = (drawdown / peak_portfolio_value) * 100 if peak_portfolio_value > 0 else 0.0
                drawdown_pcts.append(drawdown_pct)


            except Exception as e:
                logger.error(f"Error in backtest simulation loop at date {date}: {str(e)}")
                # Log an error entry for this day and continue
                backtest_results.append({
                    "Date": date, "Regime": "Error", "Strategy": "Error", "PnL": 0.0,
                    "Capital_Deployed": 0.0, "Max_Loss": 0.0, "Risk_Reward_Target": 0.0,
                    "Realized_Vol": np.nan, "ATM_IV": np.nan, "Days_to_Expiry": np.nan, "Event_Flag": np.nan,
                    "Portfolio_Value": portfolio_value, # Keep portfolio value same as previous day
                    "Error": str(e)
                })
                daily_portfolio_values.append(portfolio_value) # Keep portfolio value same
                drawdowns.append(drawdowns[-1] if drawdowns else 0.0) # Keep drawdown same
                drawdown_pcts.append(drawdown_pcts[-1] if drawdown_pcts else 0.0) # Keep percentage drawdown same
                continue # Move to next day


        backtest_df = pd.DataFrame(backtest_results)

        if backtest_df.empty:
            st.warning("No simulation results generated for the selected parameters.")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame(), [capital], 0.0, 0.0, 0.0 # Return empty/fallback

        # Filter out 'No Trade' days and error days for performance metrics calculations
        trade_days_df = backtest_df[(backtest_df["Strategy"] != "No Trade") & (backtest_df["Strategy"] != "Error")].copy()

        if trade_days_df.empty:
            st.warning("No actual trades executed in the backtest simulation after filtering.")
            # Still return the full backtest_df for logging/plotting portfolio value
            return backtest_df, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, pd.DataFrame(), pd.DataFrame(), daily_portfolio_values, max(drawdown_pcts), 0.0, 0.0


        total_pnl = trade_days_df["PnL"].sum()
        win_rate = len(trade_days_df[trade_days_df["PnL"] > 0]) / len(trade_days_df) if len(trade_days_df) > 0 else 0.0
        losing_trades_count = len(trade_days_df[trade_days_df["PnL"] < 0])
        avg_win = trade_days_df[trade_days_df["PnL"] > 0]["PnL"].mean() if (trade_days_df["PnL"] > 0).any() else 0.0
        avg_loss = trade_days_df[trade_days_df["PnL"] < 0]["PnL"].mean() if (trade_days_df["PnL"] < 0).any() else 0.0
        # Calculate Profit Factor - sum of winning PnL / absolute sum of losing PnL
        total_winning_pnl = trade_days_df[trade_days_df["PnL"] > 0]["PnL"].sum()
        total_losing_pnl = trade_days_df[trade_days_df["PnL"] < 0]["PnL"].sum()
        profit_factor = abs(total_winning_pnl / total_losing_pnl) if total_losing_pnl != 0.0 else (float('inf') if total_winning_pnl > 0 else 0.0) # Handle division by zero and cases with only wins or losses
        expectancy = (avg_win * win_rate) + (avg_loss * (1 - win_rate)) # Average PnL per trade


        # Calculate Sharpe Ratio, Sortino Ratio, Calmar Ratio on the *daily portfolio values*
        # This is more standard practice than using individual trade returns.
        # Use the tracked daily_portfolio_values for returns series
        portfolio_returns = pd.Series(daily_portfolio_values).pct_change().dropna()

        if len(portfolio_returns) > 1 and portfolio_returns.std() != 0:
            # Calculate Sharpe Ratio (annualized)
            # Assumes returns are daily, scale by sqrt(trading days per year, e.g., 252)
            # Need excess returns: portfolio_return - risk_free_rate (per period)
            excess_returns = portfolio_returns - RISK_FREE_RATE_DAILY
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        if len(portfolio_returns) > 1 and len(excess_returns[excess_returns < 0]) > 0 and excess_returns[excess_returns < 0].std() != 0:
            # Calculate Sortino Ratio (annualized) - uses downside standard deviation
            downside_returns = excess_returns[excess_returns < 0]
            sortino_ratio = excess_returns.mean() / downside_returns.std() * np.sqrt(252)
        else:
            sortino_ratio = 0.0

        # Max Drawdown is already calculated and tracked
        max_drawdown_value_result = max(drawdowns)
        max_drawdown_pct_result = max(drawdown_pcts)


        # Calculate Calmar Ratio using Total PnL / Max Drawdown (Value)
        calmar_ratio = (total_pnl / capital) / (max_drawdown_value_result / capital) if max_drawdown_value_result > 0 else (float('inf') if total_pnl > 0 else 0.0) # Handle zero drawdown and negative PnL cases


        # Performance metrics by strategy and regime (only for actual trade days)
        strategy_perf = trade_days_df.groupby("Strategy")["PnL"].agg(['sum', 'count', 'mean']).reset_index()
        # Calculate Win Rate per strategy
        strategy_win_rate_counts = trade_days_df.groupby("Strategy").size()
        strategy_wins_counts = trade_days_df[trade_days_df["PnL"] > 0].groupby("Strategy").size()
        strategy_win_rate = strategy_wins_counts / strategy_win_rate_counts
        strategy_win_rate = strategy_win_rate.reset_index(name="Win_Rate")
        strategy_perf = pd.merge(strategy_perf, strategy_win_rate, on="Strategy", how="left").fillna(0.0) # Merge and fill missing win rates with 0

        regime_perf = trade_days_df.groupby("Regime")["PnL"].agg(['sum', 'count', 'mean']).reset_index()
        # Calculate Win Rate per regime
        regime_win_rate_counts = trade_days_df.groupby("Regime").size()
        regime_wins_counts = trade_days_df[trade_days_df["PnL"] > 0].groupby("Regime").size()
        regime_win_rate = regime_wins_counts / regime_win_rate_counts
        regime_win_rate = regime_win_rate.reset_index(name="Win_Rate")
        regime_perf = pd.merge(regime_perf, regime_win_rate, on="Regime", how="left").fillna(0.0) # Merge and fill missing win rates with 0


        logger.debug("Backtest completed successfully")
        return (backtest_df, total_pnl, win_rate, max_drawdown_value_result, sharpe_ratio, sortino_ratio,
                calmar_ratio, strategy_perf, regime_perf, daily_portfolio_values, max_drawdown_pct_result,
                profit_factor, expectancy)

    except Exception as e:
        st.error(f"Error running backtest: {str(e)}")
        logger.error(f"Error running backtest: {str(e)}")
        # Return empty/fallback data on error
        return pd.DataFrame(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, pd.DataFrame(), pd.DataFrame(), [capital], 0.0, 0.0, 0.0


# Backtest Simulation Helper Functions (from original code, kept for consistency)
def get_dynamic_slippage_sim(strategy, iv, dte):
    """Simulates dynamic slippage during backtest."""
    base_slippage = 0.0005 # Lower base slippage for simulation realism?
    iv_multiplier = min(iv / 20.0, 2.0) # Less extreme multiplier
    dte_factor = 1.5 if dte < 5 else 1.0
    strategy_multipliers = {
        "Iron Condor": 1.5, # Lower multiplier
        "Butterfly Spread": 1.8,
        "Iron Fly": 1.2,
        "Short Strangle": 1.3,
        "Calendar Spread": 1.0, # Calendar might have lower slippage on average - though not used in backtest sim
        "Jade Lizard": 1.2,
        "Short Straddle": 1.3,
        "No Trade": 0.0,
        "Error": 0.0
    }
    return base_slippage * strategy_multipliers.get(strategy, 1.0) * iv_multiplier * dte_factor

def apply_volatility_shock_sim(pnl, nifty_move_pct, iv, event_flag, deployed_capital):
    """Simulates volatility shock impact during backtest."""
    shock_prob = 0.30 if event_flag == 1 else 0.10 # Lower shock probability
    if np.random.rand() < shock_prob:
        # Shock magnitude proportional to move vs expected move and deployed capital
        expected_daily_move_pct = (iv / 100.0) / np.sqrt(252.0) if iv > 0 else 0.0
        if expected_daily_move_pct > 1e-6: # Avoid division by zero
             move_factor = abs(nifty_move_pct) / expected_daily_move_pct
        else:
             move_factor = 0.0
        # Shock is negative, proportional to move_factor and deployed capital
        # Use a base shock size relative to deployed capital
        base_shock_size = deployed_capital * 0.05 # Example: 5% of deployed capital base
        shock_magnitude = base_shock_size * min(move_factor * 0.8, 3.0) # Magnitude increases with move factor, capped
        shock = -shock_magnitude * np.sign(nifty_move_pct if nifty_move_pct != 0 else np.random.choice([-1, 1])) # Shock direction might relate to move, default random if no move

        # Add shock to PnL, but don't let total loss exceed deployed capital significantly from this shock
        return pnl + np.clip(shock, -deployed_capital * 1.5, deployed_capital * 0.5) # Allow larger loss than deployed, but capped
    return pnl

def apply_gap_and_crash_sim(pnl, event_flag, iv, estimated_premium):
    """Simulates gap/crash events during backtest."""
    # Gap Down Risk (more likely with high IV/Event)
    # Base probability is low, increased by IV and Event flag
    gap_prob = 0.01 + (iv/50.0)*0.02 + event_flag*0.03 # Base 1%, add up to 2%, add 3% for event
    if np.random.rand() < gap_prob:
        gap_loss_factor = np.random.uniform(1.0, 2.5) # Loss is 1x to 2.5x estimated premium
        gap_loss = estimated_premium * gap_loss_factor
        pnl -= gap_loss
        # logger.debug(f"Simulated Gap Down Loss: {gap_loss:.2f}")

    # Market Crash (Rare, severe)
    crash_prob = 0.002 # Very low probability (e.g., 1 in 500 days)
    if np.random.rand() < crash_prob:
        crash_loss_factor = np.random.uniform(3.0, 8.0) # Severe loss
        # Loss is related to initial deployed capital, not just premium
        # Use a proxy for deployed capital if needed, or relate to abs(pnl) or initial_capital
        # Let's use estimated premium as a base for lack of better info in this func
        crash_loss = estimated_premium * crash_loss_factor # This might be too small for real crash
        # A better crash sim relates to the *market value* of the position delta/gamma exposure during a large gap
        # For this simplified sim, just apply a large negative shock relative to premium.
        # Or, a fixed percentage of portfolio value? Let's try premium * factor for simplicity as in original code base
        pnl -= crash_loss
        # logger.debug(f"Simulated Market Crash Loss: {crash_loss:.2f}")

    return pnl


# --- Streamlit UI Layout ---
st.set_page_config(page_title="VolGuard Pro", page_icon="🛡️", layout="wide")

# Custom CSS (refined)
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
        .main { background: linear-gradient(135deg, #0e172a, #1a2035); color: #e0e0e0; }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px; /* Space between tabs */
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #1a2035; /* Darker background for inactive tabs */
            color: #a0a0a0; /* Grey text for inactive tabs */
            font-weight: 500;
            padding: 10px 20px;
            border-radius: 8px 8px 0 0; /* Rounded top corners */
            transition: background-color 0.2s ease-in-out, color 0.2s ease-in-out;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: #e94560; /* Active tab color */
            color: white;
            font-weight: 700;
            border-bottom-color: #e94560 !important; /* Match border to active color */
        }
         .stTabs [data-baseweb="tab"]:hover {
            background: #2a3147; /* Hover color */
            color: white;
         }
        .sidebar .stButton>button {
            width: 100%;
            background: #0f3460;
            color: white;
            border-radius: 8px;
            padding: 10px;
            margin: 4px 0;
            font-size: 1rem;
            transition: all 0.2s ease-in-out;
        }
        .sidebar .stButton>button:hover {
            transform: translateY(-2px);
            background: #1a4a8a;
        }
        .sidebar .stButton>button:active {
            transform: translateY(0);
        }
         .stDownloadButton > button {
             background-color: #0f3460 !important;
             color: white !important;
             border-radius: 8px !important;
             padding: 10px !important;
             margin: 4px 0 !important;
             font-size: 1rem !important;
             transition: all 0.2s ease-in-out !important;
             width: 100%;
         }
        .stDownloadButton > button:hover {
             transform: translateY(-2px);
             background-color: #1a4a8a !important;
        }

        div[data-baseweb="select"] > div {
            border-radius: 8px; /* Rounded select boxes */
        }
         div[data-baseweb="input"] input {
             border-radius: 8px; /* Rounded text inputs */
         }
         div[data-baseweb="input"] textarea {
             border-radius: 8px; /* Rounded text areas */
         }

        .card {
            background: linear-gradient(145deg, rgba(26, 32, 53, 0.9), rgba(14, 23, 42, 0.9));
            border-radius: 12px;
            padding: 25px;
            margin: 20px 0;
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(40, 51, 74, 0.5); /* Subtle border */
        }
        .card h3, .card h4 { color: #e94560; margin-bottom: 15px; }

        .stMetric > div { /* Target the div containing metric value and label */
            background: rgba(15, 52, 96, 0.5); /* More subtle background */
            border-radius: 10px;
            padding: 12px;
             border: 1px solid rgba(40, 51, 74, 0.6);
        }
        .stMetric label { color: #a0a0a0; font-size: 0.9rem; } /* Label color */
        .stMetric div[data-testid="stMetricDelta"] svg { fill: white !important; } /* Make delta arrow visible */

        .gauge-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 20px auto;
            width: fit-content; /* Center the gauge */
        }
         .gauge {
             width: 120px; height: 120px;
             border-radius: 50%;
             position: relative;
             display: flex; align-items: center; justify-content: center;
             font-weight: bold; font-size: 1.1rem;
             color: white;
             box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
         }
         .gauge::before {
             content: '';
             position: absolute;
             top: 50%; left: 50%;
             width: 80%; height: 80%;
             background: #0e172a; /* Inner circle color */
             border-radius: 50%;
             transform: translate(-50%, -50%);
         }
         .gauge span { position: relative; z-index: 1; } /* Ensure text is above inner circle */

         /* Conic gradient for gauge fill - update with actual percentage via data attribute or class */
         /* Default background for the gauge before filling */
        .gauge { background: #0e172a; } /* Dark background */

        /* Specific class for confidence gauge */
         .gauge.confidence {
             /* Use the --percentage CSS variable to control the gradient fill */
             background: conic-gradient(#00d4ff var(--percentage), #0e172a var(--percentage) 100%);
         }
        /* Classes for regime gauge */
         .gauge.low { background: conic-gradient(#28a745 50%, #0e172a 50%); }
         .gauge.medium { background: conic-gradient(#ffc107 75%, #0e172a 75%); color: #0e172a; } /* Medium might need dark text */
         .gauge.high { background: conic-gradient(#dc3545 100%, #0e172a 100%); }
         .gauge.event { background: conic-gradient(#ff6f61 100%, #0e172a 100%); }


        .regime-badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 15px;
            font-weight: bold;
            font-size: 0.8rem;
            text-transform: uppercase;
            margin-bottom: 10px;
            text-align: center;
        }
        .regime-low { background: #28a745; color: white; }
        .regime-medium { background: #ffc107; color: black; }
        .regime-high { background: #dc3545; color: white; }
        .regime-event { background: #ff6f61; color: white; }

        .alert-banner {
            background: rgba(220, 53, 69, 0.9); /* Red */
            color: white;
            padding: 12px;
            border-radius: 8px;
            margin: 15px 0;
            font-weight: 500;
            display: flex;
            align-items: center;
        }
         .alert-banner svg { margin-right: 10px; } /* Icon spacing */

        .info-banner {
            background: rgba(23, 162, 184, 0.9); /* Info blue */
            color: white;
            padding: 12px;
            border-radius: 8px;
            margin: 15px 0;
            font-weight: 500;
             display: flex;
            align-items: center;
        }
         .info-banner svg { margin-right: 10px; } /* Icon spacing */

        .warning-banner {
            background: rgba(255, 193, 7, 0.9); /* Warning yellow */
            color: black;
            padding: 12px;
            border-radius: 8px;
            margin: 15px 0;
            font-weight: 500;
             display: flex;
            align-items: center;
        }
         .warning-banner svg { margin-right: 10px; } /* Icon spacing */


        .stButton>button {
            background: #e94560; /* Primary action button */
            color: white;
            border-radius: 8px;
            padding: 12px 25px;
            font-size: 1rem;
            margin-top: 15px;
            transition: all 0.2s ease-in-out;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            background: #ff6f61; /* Lighter hover */
        }
         .stButton>button:active {
            transform: translateY(0);
        }

        .stException { /* Style for error messages */
            background: #341f28; /* Dark red background */
            color: #ffb3b3; /* Light red text */
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e94560;
            margin-bottom: 15px;
        }

        .footer { text-align: center; padding: 20px; color: #a0a0a0; font-size: 0.9rem; border-top: 1px solid rgba(255, 255, 255, 0.1); margin-top: 30px; }

        /* Adjust columns for strategy card - simple float layout */
        .strategy-card .row { display: flex; flex-wrap: wrap; margin: 0 -10px; }
        .strategy-card .col { flex: 1; padding: 0 10px; min-width: 150px; }
        .strategy-card .col p { margin-bottom: 5px; }

    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://raw.githubusercontent.com/shritish20/VolGuard/main/logo_vanguard.png", use_column_width=True) # Use your logo URL
    st.header("ðŸ” 5paisa Login")

    if not st.session_state.logged_in:
        totp_code = st.text_input("TOTP (from Authenticator App)", type="password", key="sidebar_totp")
        if st.button("Login to 5paisa", key="sidebar_login_btn"):
            if totp_code:
                client = initialize_5paisa_client(totp_code)
                if client:
                    st.session_state.client = client
                    st.session_state.logged_in = True
                    st.success("âœ… Logged in successfully!")
                    # Rerun to update UI
                    st.rerun()
                # Error message is shown within initialize_5paisa_client
            else:
                st.warning("Please enter TOTP.")
    else:
        st.success("âœ… Connected to 5paisa API")
        if st.button("Logout", key="sidebar_logout_btn"):
             st.session_state.logged_in = False
             st.session_state.client = None
             st.session_state.analysis_run = False # Reset state on logout
             st.session_state.backtest_results = None
             st.session_state.real_time_data = None
             st.session_state.historical_data = None
             st.session_state.feature_data = None
             st.session_state.forecast_data = None # Reset forecast state
             st.session_state.strategy_recommendation = None # Reset strategy state
             st.session_state.strategy_errors = [] # Reset strategy errors state
             st.session_state.portfolio_data = None # Reset portfolio state
             st.session_state.generated_orders = None # Reset generated orders state
             st.session_state.generated_order_error = None # Reset generated order error state
             st.session_state.trade_log = [] # Clear session logs
             st.session_state.journal_log = [] # Clear session logs
             st.session_state.violations = 0 # Reset violations
             st.session_state.journal_complete = False # Reset journal complete status
             st.success("Logged out.")
             st.rerun()


    st.header("⚙️ Settings & Controls")
    capital = st.number_input("Trading Capital (₹)", min_value=100000.0, value=1000000.0, step=100000.0, format="%.2f", help="Your total capital for position sizing and performance calculation.")
    risk_tolerance = st.select_slider("Risk Profile", options=["Conservative", "Moderate", "Aggressive"], value="Moderate", help="Adjusts recommended position sizing.")
    forecast_horizon = st.slider("Volatility Forecast Horizon (Days)", 1, 30, 7, help="Number of future trading days to forecast volatility.")
    # DTE Preference is now primarily part of strategy logic, less a user setting for the core analysis run
    # dte_preference = st.slider("DTE Preference (Days)", 7, 30, 15)

    st.markdown("---") # Separator
    st.markdown("**Backtest Parameters**")
    backtest_start_date = st.date_input("Start Date", value=pd.to_datetime("2024-01-01").date(), key="backtest_start_date")
    backtest_end_date = st.date_input("End Date", value=pd.to_datetime("2025-04-29").date(), key="backtest_end_date")
    backtest_strategy_choice = st.selectbox("Strategy to Backtest",
                                              ["All Strategies", "Butterfly Spread", "Iron Condor",
                                               "Iron Fly", "Short Strangle", "Jade Lizard", "Short Straddle"],
                                               key="backtest_strategy_select")
    st.markdown("---") # Separator


    # --- Action Buttons ---
    if st.session_state.logged_in:
        # The button is defined here, its result is assigned to the variable initialized earlier
        run_analysis_button = st.button("📊 Run VolGuard Analysis", key="run_analysis_btn")
        st.markdown("---")
        if st.button("🚨 Square Off All Positions", key="square_off_btn"):
            with st.spinner("Attempting to square off positions..."):
                success = square_off_positions(st.session_state.client)
                if success:
                    st.success("✅ All positions squared off successfully!")
                    st.session_state.real_time_data = None # Clear cached data that might show old positions
                    st.session_state.portfolio_data = fetch_portfolio_data(st.session_state.client) # Refresh portfolio data
                else:
                    st.error("❌ Failed to square off positions.")
    else:
         st.info("Login to enable analysis and trading controls.")


    # --- Data Downloads (for Session Logs) ---
    st.markdown("---")
    st.header("💾 Session Data")
    st.warning("Session data (Journal, Trades) is stored only for your current browser session and is NOT persistent across deployments or restarts on Streamlit Cloud. Use download buttons to save.")

    if st.session_state.journal_log:
        journal_csv = pd.DataFrame(st.session_state.journal_log).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Journal Log (CSV)",
            data=journal_csv,
            file_name="volguard_journal_log.csv",
            mime="text/csv",
            key="download_journal_btn"
        )
    if st.session_state.trade_log:
        trade_csv = pd.DataFrame(st.session_state.trade_log).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Trade Log (CSV)",
            data=trade_csv,
            file_name="volguard_trade_log.csv",
            mime="text/csv",
            key="download_trade_btn"
        )
    # Can add download for backtest results if needed


# --- Main App Area ---
st.markdown("<h1 style='color: #e94560; text-align: center;'>🛡️ VolGuard Pro: Your AI Trading Copilot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #a0a0a0;'>Data-Driven Insights & Strategy Recommendations for Option Sellers</p>", unsafe_allow_html=True)

# Create tabs
tabs = st.tabs(["📈 Dashboard", "🔭 Forecast", "💡 Strategy", "💼 Portfolio", "📝 Discipline Journal", "📊 Backtest"]) # Renamed strategy tab

# --- Execution Flow ---
# This check is now safe because run_analysis_button is always defined
if run_analysis_button and st.session_state.logged_in:
    with st.spinner("Running VolGuard Analysis..."):
        # Reset analysis state flags
        st.session_state.analysis_run = False
        st.session_state.backtest_results = None
        st.session_state.forecast_data = None
        st.session_state.strategy_recommendation = None
        st.session_state.strategy_errors = []
        st.session_state.generated_orders = None # Clear previously generated orders on new run
        st.session_state.generated_order_error = None # Clear previous error


        # 1. Load and Prepare Data
        df_combined, real_time_data, data_source_info = prepare_data(st.session_state.client, capital)

        if df_combined is not None and not df_combined.empty:
             # 2. Generate Features
             df_features = generate_features(df_combined, real_time_data, capital)

             if not df_features.empty:
                 # 3. Run Backtest (using the historical portion of features)
                 # Ensure backtest range does not include the latest real-time data point if df_features includes it
                 # Backtest should run up to the day *before* the last day if the last day is real-time.
                 last_data_date = df_features.index.max().date()
                 backtest_end_date_effective = min(backtest_end_date, last_data_date)

                 if backtest_start_date >= backtest_end_date_effective or len(df_features.loc[pd.to_datetime(backtest_start_date).normalize():pd.to_datetime(backtest_end_date_effective).normalize()]) < 50:
                      st.warning("Backtest date range is invalid or too short after filtering. Please select a valid range with at least 50 days of data.")
                      # Populate backtest results with empty/fallback data
                      backtest_results = (pd.DataFrame(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, pd.DataFrame(), pd.DataFrame(), [capital], 0.0, 0.0, 0.0)
                 else:
                    backtest_results = run_backtest(
                        df_features, capital, backtest_strategy_choice,
                        pd.to_datetime(backtest_start_date).normalize(),
                        pd.to_datetime(backtest_end_date_effective).normalize()
                    )

                 st.session_state.backtest_results = backtest_results # Store backtest results

                 # 4. Volatility Forecasting (using all features)
                 if len(df_features) >= max(100, forecast_horizon + 1): # Need sufficient data for XGBoost and horizon
                     forecast_log, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, feature_importances = forecast_volatility_future(df_features, forecast_horizon)
                     st.session_state.forecast_data = { # Store forecast results
                         "forecast_log": forecast_log,
                         "garch_vols": garch_vols,
                         "xgb_vols": xgb_vols,
                         "blended_vols": blended_vols,
                         "realized_vol": realized_vol,
                         "confidence_score": confidence_score,
                         "rmse": rmse,
                         "feature_importances": feature_importances
                     }
                 else:
                     st.warning(f"Skipping volatility forecasting due to insufficient data ({len(df_features)} days). Need at least {max(100, forecast_horizon+1)} days.")
                     st.session_state.forecast_data = None # Clear forecast state if skipped

                 # 5. Generate Trading Strategy Recommendation (using latest features and forecast)
                 if st.session_state.forecast_data: # Only generate strategy if forecast was successful
                    strategy_recommendation, strategy_errors = generate_trading_strategy(
                        df_features,
                        st.session_state.forecast_data["forecast_log"],
                        st.session_state.forecast_data["realized_vol"],
                        risk_tolerance,
                        st.session_state.forecast_data["confidence_score"],
                        capital
                    )
                    st.session_state.strategy_recommendation = strategy_recommendation
                    st.session_state.strategy_errors = strategy_errors
                 else:
                     st.session_state.strategy_recommendation = None
                     st.session_state.strategy_errors = ["Skipped strategy generation because volatility forecasting failed or was skipped."]


                 # 6. Fetch Portfolio Data (Current)
                 portfolio_data = fetch_portfolio_data(st.session_state.client)
                 st.session_state.portfolio_data = portfolio_data


                 st.session_state.analysis_run = True # Mark analysis as complete
                 st.success("Analysis complete!")

             else:
                  st.error("Feature generation failed. Cannot proceed.")
        else:
            st.error("Data loading failed. Cannot proceed with analysis.")


# --- Populate Tabs ---

# Dashboard Tab
with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("ðŸ“Š Market Snapshot")
    if st.session_state.analysis_run and st.session_state.feature_data is not None:
        df_features = st.session_state.feature_data
        if not df_features.empty:
            latest_data = df_features.iloc[-1]
            prev_data = df_features.iloc[-2] if len(df_features) >= 2 else latest_data
            real_data_info = st.session_state.real_time_data # Use stored real-time data for display if available

            # Prioritize real-time data for current values if available and valid
            current_vix = real_data_info["vix"] if real_data_info and real_data_info["vix"] is not None else latest_data.get("VIX", 0.0)
            current_nifty = real_data_info["nifty_spot"] if real_data_info and real_data_info["nifty_spot"] is not None else latest_data.get("NIFTY_Close", 0.0)
            current_pcr = real_data_info["pcr"] if real_data_info and real_data_info["pcr"] is not None else latest_data.get("PCR", 0.0)
            current_straddle = real_data_info["straddle_price"] if real_data_info and real_data_info["straddle_price"] is not None else latest_data.get("Straddle_Price", 0.0)
            current_max_pain = real_data_info["max_pain_strike"] if real_data_info and real_data_info["max_pain_strike"] is not None else 0.0
            current_max_pain_diff = real_data_info["max_pain_diff_pct"] if real_data_info and real_data_info["max_pain_diff_pct"] is not None else latest_data.get("Spot_MaxPain_Diff_Pct", 0.0)


            # Determine Regime for display based on current VIX
            display_regime = "LOW" if current_vix < 15 else "MEDIUM" if current_vix < 20 else "HIGH"
            # Add event flag check for display regime if analysis ran and strategy generated
            if st.session_state.strategy_recommendation and st.session_state.strategy_recommendation["Regime"] == "EVENT-DRIVEN":
                display_regime = "EVENT-DRIVEN" # Override if event regime is active

            regime_class = {
                "LOW": "regime-low",
                "MEDIUM": "regime-medium",
                "HIGH": "regime-high",
                "EVENT-DRIVEN": "regime-event"
            }[display_regime]
            # Use percentage based on VIX for gradient fill (example mapping)
            vix_percent_for_gauge = min(100, max(0, int((current_vix / 30.0) * 100))) # Map VIX 0-30 to 0-100%
            # If in EVENT-DRIVEN regime, force gauge to 100% fill with event color
            if display_regime == "EVENT-DRIVEN":
                 vix_percent_for_gauge = 100


            st.markdown('<div class="gauge-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="gauge {regime_class}" style="--percentage: {vix_percent_for_gauge}%;"><span>{display_regime}</span></div>', unsafe_allow_html=True)
            st.markdown('<div style="text-align: center; margin-top: 5px;">Market Regime</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


            col1, col2, col3, col4 = st.columns(4)
            with col1:
                # Ensure prev_data Close is not zero for percentage change calculation
                prev_nifty_close = prev_data.get("NIFTY_Close", current_nifty)
                nifty_change = (current_nifty - prev_nifty_close)/prev_nifty_close * 100 if prev_nifty_close != 0 else 0.0
                st.metric("NIFTY 50 Spot", f"₹{current_nifty:,.2f}", f"{nifty_change:+.2f}%")
            with col2:
                vix_change = latest_data.get("VIX_Change_Pct", 0.0) if "VIX_Change_Pct" in latest_data else 0.0
                st.metric("India VIX", f"{current_vix:.2f}%", f"{vix_change:+.2f}%")
            with col3:
                st.metric("PCR", f"{current_pcr:.2f}")
            with col4:
                 st.metric("ATM Straddle Price", f"₹{current_straddle:,.2f}")


            st.markdown(f"**Latest Data Date**: {latest_data.name.strftime('%d-%b-%Y')} | **Data Source**: {data_source_info}")

            if current_max_pain != 0.0:
                st.info(f"💡 Max Pain Strike: {current_max_pain:.0f} ({current_max_pain_diff:.1f}% from Spot)")

            # Optional: Display a small chart of recent Nifty/VIX
            if st.session_state.historical_data is not None and not st.session_state.historical_data.empty:
                 # Use the combined data (historical + latest real-time) for the chart
                 recent_history = df_features.tail(60) # Last 60 days from features df
                 if not recent_history.empty:
                     st.line_chart(recent_history[["NIFTY_Close", "VIX"]], y=["NIFTY_Close", "VIX"], color=["#00d4ff", "#e94560"])
                     st.markdown("<sub>Recent 60-day NIFTY Close and India VIX</sub>", unsafe_allow_html=True)

        else:
            st.info("Insufficient data to display market snapshot.")

    else:
        st.info("Run analysis to see the market snapshot.")
    st.markdown('</div>', unsafe_allow_html=True)


# Forecast Tab
with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("ðŸ“ˆ Volatility Forecast")
    if st.session_state.analysis_run and st.session_state.forecast_data is not None:
        forecast_data = st.session_state.forecast_data
        forecast_log = forecast_data["forecast_log"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Blended Forecast Vol", f"{np.mean(forecast_data['blended_vols']):.2f}%")
        with col2:
            st.metric("Current Realized Vol", f"{forecast_data['realized_vol']:.2f}%")
        with col3:
            st.metric("XGBoost Model RMSE", f"{forecast_data['rmse']:.2f}%")

        st.markdown('<div class="gauge-container">', unsafe_allow_html=True)
        # Use custom CSS variable for dynamic gradient fill
        st.markdown(f'<div class="gauge confidence" style="--percentage: {int(forecast_data["confidence_score"])}%;"><span>{int(forecast_data["confidence_score"])}%</span></div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: center; margin-top: 5px;">Forecast Confidence Score</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


        if not forecast_log.empty:
            st.line_chart(forecast_log.set_index("Date")[["GARCH_Vol", "XGBoost_Vol", "Blended_Vol"]], color=["#e94560", "#00d4ff", "#ffcc00"])
        else:
             st.warning("Forecast log is empty.")


        with st.expander("Model Details & Feature Importance"):
             st.markdown("### Feature Importance (XGBoost)")
             feature_importance = pd.DataFrame({
                 'Feature': XGB_FEATURE_COLS, # Use the standard list
                 'Importance': forecast_data["feature_importances"]
             }).sort_values(by='Importance', ascending=False)
             st.dataframe(feature_importance, use_container_width=True)
             st.markdown("""
                 <p><i>Feature importance indicates how much each factor contributed to the XGBoost model's volatility predictions.</i></p>
             """, unsafe_allow_html=True)
             st.markdown("### Model Comparison")
             if not forecast_log.empty:
                 comp_df = forecast_log.set_index("Date")
                 st.dataframe(comp_df.style.format("{:.2f}"), use_container_width=True)
             else:
                  st.warning("Forecast log is empty.")


    elif st.session_state.analysis_run and st.session_state.forecast_data is None:
         st.warning("Volatility forecast skipped due to insufficient historical data.")
    else:
        st.info("Run analysis to see the volatility forecast.")
    st.markdown('</div>', unsafe_allow_html=True)


# Strategy Tab
with tabs[2]: # This is the 3rd tab, index 2
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💡 Trading Strategy Recommendation")
    if st.session_state.analysis_run:
        if st.session_state.strategy_recommendation:
            strategy = st.session_state.strategy_recommendation
            regime_class = {
                "LOW": "regime-low",
                "MEDIUM": "regime-medium",
                "HIGH": "regime-high",
                "EVENT-DRIVEN": "regime-event"
            }.get(strategy["Regime"], "regime-low")

            st.markdown(f"""
                <div class="strategy-card card"> <h4>Recommended Strategy: {strategy["Strategy"]}</h4>
                    <span class="regime-badge {regime_class}">{strategy["Regime"]} Regime</span>
                    <p><b>Reasoning:</b> {strategy["Reason"]}</p>
                    <p><b>Key Factors:</b> {', '.join(strategy["Tags"])}</p>
                    <hr style="border-top: 1px solid rgba(255,255,255,0.1); margin: 15px 0;"/>
                    <div class="row"> <div class="col">
                            <p><b>Forecast Confidence:</b> {int(strategy["Forecast_Confidence"])}%</p>
                        </div>
                        <div class="col">
                             <p><b>Target Risk-Reward:</b> {strategy["Risk_Reward_Target"]:.1f}:1</p>
                        </div>
                    </div>
                    <p><b>Recommended Capital Allocation:</b> ₹{strategy["Recommended_Deploy_Capital"]:,.0f} ({strategy["Recommended_Exposure_Pct"]:.1f}% of Total Capital)</p>
                    <p><b>Recommended Max Loss (per trade):</b> ₹{strategy["Recommended_Max_Loss"]:,.0f}</p>

                </div>
            """, unsafe_allow_html=True)

            # Risk Flags and Behavior Warnings
            if strategy["Risk_Flags"]:
                for flag in strategy["Risk_Flags"]:
                    st.markdown(f'<div class="warning-banner">⚠️ Risk Flag: {flag}</div>', unsafe_allow_html=True)

            if strategy["Behavior_Warnings"]:
                 for warning in strategy["Behavior_Warnings"]:
                      st.markdown(f'<div class="info-banner">ℹ️ Behavioral Note: {warning}</div>', unsafe_allow_html=True)


            # Discipline Lock Check - Check session state variables
            is_locked = st.session_state.violations >= 2 and not st.session_state.journal_complete
            if is_locked:
                 st.markdown('<div class="alert-banner">ðŸš¨ **Discipline Lock Active:** You have accumulated multiple warnings without journaling. Complete the journal entry in the Discipline Journal tab to remove the lock.</div>', unsafe_allow_html=True)

            # Generate Order Details Button (if not locked and real-time data is available)
            if not is_locked and st.session_state.real_time_data is not None:
                 # Reset generated orders state when analysis is re-run, but display them only after clicking the button
                 if st.button("📋 Generate Order Details", key="generate_orders_btn"):
                      order_details, order_error = generate_trade_orders_display(
                          strategy,
                          st.session_state.real_time_data,
                          capital,
                          LOT_SIZE
                      )
                      if order_details:
                           st.session_state.generated_orders = order_details # Store generated orders
                           st.session_state.generated_order_error = None
                           st.success("Order details generated below.")
                           # Log this "intended" trade action to session state
                           trade_log_entry = {
                               "Timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                               "Strategy Recommended": strategy["Strategy"],
                               "Regime": strategy["Regime"],
                               "Risk Flags Present": len(strategy["Risk_Flags"]) > 0,
                               "Behavior Score": strategy["Behavior_Score"],
                               "Recommended Capital": strategy["Recommended_Deploy_Capital"],
                               "Action": "Order Details Generated"
                               # Do NOT log PnL here - this is not a live trade execution
                           }
                           st.session_state.trade_log.append(trade_log_entry)
                      else:
                           st.session_state.generated_orders = None
                           st.session_state.generated_order_error = order_error
                           st.error(f"Could not generate order details: {order_error}")

            elif st.session_state.real_time_data is None:
                 st.warning("Real-time data is required to generate specific order details for live trading.")
            elif st.session_state.logged_in and not st.session_state.analysis_run:
                 st.info("Run analysis first to get a strategy recommendation and generate order details.")


            # Display Generated Order Details
            if st.session_state.get("generated_orders"):
                 st.markdown("### Recommended Orders to Place Manually")
                 st.markdown("""
                     <div class="info-banner">
                     ℹ️ **Action Required:** These are the specific legs and parameters for the recommended strategy.
                     Please place these orders **manually** on your 5paisa or other trading platform.
                     VolGuard Pro does **not** place live trades for you.
                     </div>
                 """, unsafe_allow_html=True)
                 for order in st.session_state.generated_orders:
                      st.json(order) # Display each order detail as JSON or formatted text

            if st.session_state.get("generated_order_error"):
                 st.error(f"Order Generation Error: {st.session_state.generated_order_error}")


        elif st.session_state.strategy_errors:
             for err in st.session_state.strategy_errors:
                  st.warning(f"Strategy Generation Warning: {err}")
             st.info("Could not generate a strategy recommendation based on current conditions or data.")

        else:
             st.info("Strategy recommendation will appear here after running analysis.")
    else:
        st.info("Run analysis to get a strategy recommendation.")
    st.markdown('</div>', unsafe_allow_html=True)


# Portfolio Tab
with tabs[3]: # This is the 4th tab, index 3
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💼 Portfolio Overview")
    if st.session_state.logged_in: # Only show portfolio if logged in
        if st.session_state.analysis_run and st.session_state.portfolio_data:
            portfolio_data = st.session_state.portfolio_data

            if "error" in portfolio_data:
                 st.error(f"Failed to fetch portfolio data: {portfolio_data['error']}")
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total MTM P&L", f"₹{portfolio_data['total_pnl']:,.2f}")
                with col2:
                    st.metric("Margin Used", f"₹{portfolio_data['margin_used']:,.2f}")
                with col3:
                     # Calculate percentage exposure relative to total capital
                     exposure_pct = (portfolio_data['margin_used'] / capital) * 100 if capital > 0 else 0.0
                     st.metric("Exposure (of Capital)", f"{exposure_pct:.2f}%")


                st.markdown("### Open Positions")
                if portfolio_data["positions"]:
                    # Convert list of dicts to DataFrame for better display
                    pos_df = pd.DataFrame(portfolio_data["positions"])
                    # Format columns for currency/numbers
                    st.dataframe(pos_df.style.format({
                        "Buy Avg": "₹{:,.2f}",
                        "Sell Avg": "₹{:,.2f}",
                        "LTP": "₹{:,.2f}",
                        "P&L": "₹{:,.2f}",
                        "Margin Used": "₹{:,.2f}",
                         "Exposure Value": "₹{:,.2f}" # Display value as currency
                    }), use_container_width=True)
                else:
                    st.info("No open positions found in your 5paisa account.")

        elif st.session_state.logged_in and not st.session_state.analysis_run:
             st.info("Run analysis to fetch and see your current portfolio overview.")
        else:
             st.warning("Portfolio data not available.")
    else:
        st.info("Login to 5paisa from the sidebar to see your portfolio overview.")
    st.markdown('</div>', unsafe_allow_html=True)

# Journal Tab
with tabs[4]: # This is the 5th tab, index 4
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📝 Discipline Journal")
    st.markdown("""
        <div class="info-banner">
        ℹ️ The Discipline Journal helps you track your decision-making and adherence to your plan.
        Completing entries helps maintain focus and can unlock features if you accumulate warnings.
        </div>
    """, unsafe_allow_html=True)

    # Check violation status from session state
    is_locked = st.session_state.violations >= 2 and not st.session_state.journal_complete
    if is_locked:
         st.markdown('<div class="alert-banner">ðŸš¨ **Discipline Lock Active:** You have accumulated multiple warnings without journaling. Complete the journal entry below to remove the lock.</div>', unsafe_allow_html=True)
    elif st.session_state.violations > 0:
         st.markdown(f'<div class="warning-banner">⚠️ **Pending Violations:** You have {st.session_state.violations} risk flag(s) detected since your last journal entry. Journaling is recommended.</div>', unsafe_allow_html=True)
         # Note: The `journal_complete` flag should be reset when new violations occur.
         # This is handled in the Strategy Generation section now.


    with st.form(key="journal_form"):
        st.markdown("#### New Journal Entry")
        entry_date = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
        st.write(f"Entry Timestamp: {entry_date}")

        # You can pre-fill strategy/regime if analysis was just run
        latest_strategy_rec = st.session_state.strategy_recommendation
        if latest_strategy_rec:
             st.info(f"Last analysis recommendation: **{latest_strategy_rec['Strategy']}** in **{latest_strategy_rec['Regime']}** regime.")
             # Store these for logging in the entry if user submits
             last_rec_strategy = latest_strategy_rec['Strategy']
             last_rec_regime = latest_strategy_rec['Regime']
             last_rec_risk_flags_present = len(latest_strategy_rec['Risk_Flags']) > 0
        else:
             last_rec_strategy = "N/A"
             last_rec_regime = "N/A"
             last_rec_risk_flags_present = False
             st.info("Run analysis first to get a recommendation context for your journal entry.")


        reason_strategy = st.selectbox("Reason for considering/choosing today's strategy (if applicable):",
                                         ["Following VolGuard Recommendation", "High IV Environment", "Low Risk Setup",
                                          "Event Opportunity", "Other (Explain Below)"], key="journal_reason_strategy")

        override_risk = st.radio("Did you decide to proceed despite risk flags or behavioral warnings from analysis?",
                                 ("Yes", "No", "N/A - No flags/warnings"), index=2, key="journal_override_risk")

        decision_notes = st.text_area("Notes on your trading decision today (e.g., why you followed/deviated from recommendation, specific setup, thoughts):", key="journal_decision_notes")
        expected_outcome = st.text_area("What outcome do you expect from today's trading activity or setup?", key="journal_expected_outcome")
        learnings = st.text_area("Learnings or observations from today's market action:", key="journal_learnings")


        submit_journal = st.form_submit_button("Submit Journal Entry")

        if submit_journal:
            # Calculate discipline score (example logic)
            score = 0
            if override_risk == "No": score += 3
            if override_risk == "N/A - No flags/warnings": score += 4 # Higher score if no flags were present
            if reason_strategy == "Following VolGuard Recommendation": score += 2
            # Add score based on detail in text areas
            if decision_notes: score += min(len(decision_notes.split()) // 10, 3) # Score based on word count (min 10 words per point)
            if expected_outcome: score += min(len(expected_outcome.split()) // 10, 2)
            if learnings: score += min(len(learnings.split()) // 10, 3)
            # Add score based on recent PnL if available in portfolio_data
            if st.session_state.portfolio_data:
                 recent_pnl = st.session_state.portfolio_data.get("total_pnl", 0)
                 if recent_pnl > 0: score += 1 # Small bonus for positive PnL


            score = max(1, min(score, 10)) # Clip score between 1 and 10

            journal_entry = {
                "Timestamp": entry_date,
                "Strategy_Reason": reason_strategy,
                "Override_Risk_Flags": override_risk,
                "Decision_Notes": decision_notes,
                "Expected_Outcome": expected_outcome,
                "Learnings": learnings,
                "Discipline_Score": score,
                # Log the state of violations/journal complete at time of entry
                "Violations_Before_Entry": st.session_state.violations,
                "Journal_Complete_Before_Entry": st.session_state.journal_complete,
                 "Risk_Flags_Present_During_Analysis": last_rec_risk_flags_present, # Log if flags were present in the analysis context for this journal
                 "Recommended_Strategy": last_rec_strategy,
                 "Recommended_Regime": last_rec_regime
            }
            st.session_state.journal_log.append(journal_entry) # Add to session state log

            st.success(f"Journal Entry Saved! Your Discipline Score for this entry: {score}/10")

            # Reset violations and remove lock if applicable
            if is_locked or st.session_state.violations > 0:
                 st.session_state.violations = 0
                 st.session_state.journal_complete = True # Mark journal as completed after submission
                 logger.info("Violations reset and journal_complete set to True after journal submission.")
                 st.success("✅ Discipline Lock Removed!") # Re-display success if it was locked

            if score >= 8: # Confetti for good scores
                st.balloons() # Use Streamlit's built-in confetti

            # Optional: Rerun to clear form and update UI
            st.rerun()


    st.markdown("---")
    st.markdown("#### Past Journal Entries (Current Session)")
    if st.session_state.journal_log:
        journal_df = pd.DataFrame(st.session_state.journal_log)
        # Display in reverse chronological order
        st.dataframe(journal_df.sort_values(by="Timestamp", ascending=False).style.format({
            "Discipline_Score": "{}/10"
        }), use_container_width=True)
        st.info("Entries are saved only for this session. Download the CSV from the sidebar to keep them.")
    else:
        st.info("No journal entries yet for this session.")

    st.markdown('</div>', unsafe_allow_html=True)


# Backtest Tab
with tabs[5]: # This is the 6th tab, index 5
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("ðŸ“‰ Backtest Results")
    st.markdown("""
        <div class="warning-banner">
        ⚠️ **Backtest Disclaimer:** The backtesting simulation uses simplified models for P&L calculation, slippage, and market shocks.
        Results are indicative and **not** a guarantee of future performance. Transaction costs, taxes, and execution quality are simplified.
        Always use backtesting results as a guide, not a definitive forecast.
        </div>
    """, unsafe_allow_html=True)


    if st.session_state.analysis_run and st.session_state.backtest_results is not None:
        results = st.session_state.backtest_results
        (backtest_df, total_pnl, win_rate, max_drawdown_value, sharpe_ratio,
         sortino_ratio, calmar_ratio, strategy_perf, regime_perf,
         daily_portfolio_values, max_drawdown_pct, profit_factor, expectancy) = results

        # Filter out 'No Trade' days for displaying trade metrics like win rate etc.
        trade_days_df = backtest_df[(backtest_df["Strategy"] != "No Trade") & (backtest_df["Strategy"] != "Error")].copy()


        if backtest_df.empty or trade_days_df.empty: # Check if there were any actual trades
            st.warning("No trades were simulated for the selected backtest period and strategy. Try adjusting the dates or selecting 'All Strategies'.")
        else:
            st.markdown(f"**Backtest Period:** {backtest_start_date.strftime('%Y-%m-%d')} to {backtest_end_date.strftime('%Y-%m-%d')}")
            st.markdown(f"**Strategy Tested:** {backtest_strategy_choice}")
            st.markdown(f"**Starting Capital:** ₹{capital:,.0f}")
            st.markdown(f"**Total Trade Days:** {len(trade_days_df):,.0f}")


            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Net P&L", f"₹{total_pnl:,.2f}")
            with col2:
                st.metric("Win Rate", f"{win_rate*100:.2f}%")
            with col3:
                st.metric("Max Drawdown (Value)", f"₹{max_drawdown_value:,.2f}")
            with col4:
                 st.metric("Max Drawdown (%)", f"{max_drawdown_pct:.2f}%")

            col5, col6, col7, col8 = st.columns(4)
            with col5:
                 st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            with col6:
                 st.metric("Sortino Ratio", f"{sortino_ratio:.2f}")
            with col7:
                 # Handle infinity case for Calmar
                 calmar_display = f"{calmar_ratio:.2f}" if calmar_ratio != float('inf') and calmar_ratio != -float('inf') else ("∞" if total_pnl > 0 else "N/A") # N/A for negative PnL with zero drawdown
                 st.metric("Calmar Ratio", calmar_display)
            with col8:
                 profit_factor_display = f"{profit_factor:.2f}" if profit_factor != float('inf') and profit_factor != -float('inf') else ("∞" if total_winning_pnl > 0 else "N/A")
                 st.metric("Profit Factor", profit_factor_display)

            col9, col10 = st.columns(2)
            with col9:
                 st.metric("Avg Winning Trade", f"₹{avg_win:,.2f}")
            with col10:
                 st.metric("Avg Losing Trade", f"₹{avg_loss:,.2f}")


            st.markdown("### Cumulative Portfolio Value")
            # Plot cumulative portfolio value over the backtest period
            if daily_portfolio_values and not df_backtest.empty:
                # Ensure the dates for the portfolio values match the backtest_df index
                port_value_series = pd.Series(daily_portfolio_values, index=df_backtest.index[:len(daily_portfolio_values)]).dropna() # Align with dates
                st.line_chart(port_value_series, color="#00d4ff")
            else:
                 st.warning("Cumulative portfolio value data is not available.")


            st.markdown("### Performance by Strategy")
            if not strategy_perf.empty:
                 st.dataframe(strategy_perf.style.format({
                     "sum": "₹{:,.2f}",
                     "mean": "₹{:,.2f}",
                     "Win_Rate": "{:.2%}",
                     "count": "{:,.0f}" # Format trade count
                 }), use_container_width=True)
            else:
                 st.info("No trades executed for any strategy in this backtest.")


            st.markdown("### Performance by Regime")
            if not regime_perf.empty:
                 st.dataframe(regime_perf.style.format({
                     "sum": "₹{:,.2f}",
                     "mean": "₹{:,.2f}",
                     "Win_Rate": "{:.2%}",
                     "count": "{:,.0f}" # Format trade count
                 }), use_container_width=True)
            else:
                st.info("No trades executed in any regime in this backtest.")


            with st.expander("Detailed Backtest Trades"):
                 if not backtest_df.empty:
                      st.dataframe(backtest_df.style.format({
                          "PnL": "₹{:,.2f}",
                          "Capital_Deployed": "₹{:,.2f}",
                          "Max_Loss": "₹{:,.2f}",
                          "Risk_Reward_Target": "{:.2f}",
                          "Realized_Vol": "{:.2f}%",
                          "ATM_IV": "{:.2f}%",
                          "Days_to_Expiry": "{:.0f}",
                          "Portfolio_Value": "₹{:,.2f}"
                      }), use_container_width=True)
                 else:
                     st.info("No detailed trade data available.")

    else:
        st.info("Run analysis to view backtest results.")
    st.markdown('</div>', unsafe_allow_html=True)


# --- Footer ---
st.markdown('<div class="footer">Built with ❤️ by Shritish Shukla & Salman Azim | © 2025 VolGuard</div>', unsafe_allow_html=True)

# --- Optional: Display Session State (for debugging) ---
# with st.expander("Session State (Debug)"):
#     st.json(st.session_state)
 l
