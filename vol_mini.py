# === VOLGUARD PRO - STREAMLIT APP ===
# Based on the original Colab script logic
# Features: Sidebar Token Input, Dashboard Layout, Option Chain, Metrics, Depth, IV Skew, Portfolio
# FIX: Added robust error handling for converting OI values to integer

import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import datetime, date
import logging
import time
import upstox_client
from upstox_client.rest import ApiException

# === CONFIG ===
INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"
BASE_URL = "https://api.upstox.com/v2"

# === LOGGING (for console, Streamlit handles its own logs) ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === STREAMLIT SETUP ===
st.set_page_config(layout="wide", page_title="Volguard PRO - Nifty Option Chain Dashboard")
st.title("📈 Volguard PRO: Nifty Option Chain Dashboard")

# Initialize session state
# Session state helps retain information (like previous OI or initialized client) across user interactions
if 'prev_oi' not in st.session_state:
    st.session_state['prev_oi'] = {} # Stores OI from the last successful fetch for change calculation
if 'upstox_client' not in st.session_state:
     st.session_state['upstox_client'] = None # Stores the initialized Upstox client bundle
if 'initialized_token' not in st.session_state:
     st.session_state['initialized_token'] = None # Stores the token used for the current client
if 'latest_data' not in st.session_state:
     st.session_state['latest_data'] = None # Stores the results of the last data fetch


# === INIT CLIENT ===
@st.cache_resource # Use caching to avoid re-initializing the client on every Streamlit rerun unless token changes
def initialize_upstox_client(_access_token: str): # Use underscore to differentiate from potentially local var
    """Initialize Upstox API client with access token and validate."""
    if not _access_token:
        # This case is handled by the UI prompting the user for token
        return None

    try:
        configuration = upstox_client.Configuration()
        configuration.access_token = _access_token
        client = upstox_client.ApiClient(configuration)

        # Validate token by fetching profile - confirms token works
        user_api = upstox_client.UserApi(client)
        user_api.get_profile(api_version="2") 
        
        logger.info("Token validated and Upstox client initialized successfully.")
        
        # Return a bundle of initialized API clients and the token
        return {
            "client": client,
            "access_token": _access_token,
            "user_api": user_api,
            "options_api": upstox_client.OptionsApi(client),
            "portfolio_api": upstox_client.PortfolioApi(client),
            "order_api": upstox_client.OrderApi(client),
        }
    except ApiException as e:
        logger.error(f"API Error during client initialization. Token used: {_access_token[:5]}...{_access_token[-5:]}. Error: {e}")
        # Clear cached client if validation fails for the token
        st.error(f"API Error during client initialization. Please check your Access Token and permissions: {e}")
        st.cache_resource.clear() # Clear cache on API error during init
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during client initialization: {e}")
        st.error(f"An unexpected error occurred during client initialization: {e}")
        st.cache_resource.clear() # Clear cache on unexpected error during init
        return None


# === API CALL FUNCTIONS (Adapted from your Colab script) ===

def get_nearest_expiry(options_api_client):
    """Get the nearest expiry date for the instrument using Options API client."""
    try:
        # Uses the options_api object from the client bundle
        response = options_api_client.get_option_contracts(instrument_key=INSTRUMENT_KEY)
        contracts = response.to_dict().get("data", [])
        expiry_dates = set()

        for contract in contracts:
            exp = contract.get("expiry")
            if isinstance(exp, str):
                # Handle potential time part if any, just keep date
                exp = datetime.strptime(exp.split('T')[0], "%Y-%m-%d")
            expiry_dates.add(exp)

        expiry_list = sorted(expiry_dates)
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) # Compare only dates

        valid_expiries = [e.strftime("%Y-%m-%d") for e in expiry_list if e >= today]
        
        nearest = valid_expiries[0] if valid_expiries else None
        logger.info(f"Nearest expiry found: {nearest}")
        # time.sleep(0.1) # Simple Rate limiting - can be adjusted or removed in Streamlit depending on fetch frequency
        return nearest
    except ApiException as e:
        logger.error(f"Expiry fetch failed via SDK: {e}")
        st.error(f"Could not fetch expiry dates: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred fetching expiry: {e}")
        st.error(f"An error occurred fetching expiry dates: {e}")
        return None

def fetch_vix(access_token_str):
    """Fetch India VIX value using REST API."""
    try:
        url = f"{BASE_URL}/market-quote/quotes"
        headers = {"Authorization": f"Bearer {access_token_str}", "Accept": "application/json"}
        res = requests.get(url, headers=headers, params={"instrument_key": "NSE_INDEX|India VIX"})
        res.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        data = res.json().get('data', {}).get('NSE_INDEX|India VIX', {})
        vix = data.get('last_price')
        logger.info(f"Fetched India VIX: {vix}")
        # time.sleep(0.1) # Simple Rate limiting
        return float(vix) if vix is not None else None
    except requests.exceptions.RequestException as e:
        logger.error(f"VIX fetch error: {e}")
        st.warning("Could not fetch India VIX. Service may be unavailable or token invalid.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred fetching VIX: {e}")
        return None


def fetch_option_chain(options_api_client, expiry_date_str):
    """Fetch option chain data for given expiry using Upstox SDK."""
    try:
        # Uses the SDK's get_put_call_option_chain as in your Colab script
        res = options_api_client.get_put_call_option_chain(instrument_key=INSTRUMENT_KEY, expiry_date=expiry_date_str)
        chain_data = res.to_dict().get('data', [])
        logger.info(f"Fetched option chain with {len(chain_data)} strikes.")
        # time.sleep(0.1) # Simple Rate limiting
        return chain_data
    except ApiException as e:
        logger.error(f"Option chain fetch error via SDK: {e}")
        st.error(f"Could not fetch option chain data: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred fetching option chain: {e}")
        st.error(f"An error occurred fetching option chain data: {e}")
        return []


def get_market_depth(access_token_str, token):
    """Fetch market depth for a given instrument using REST API."""
    if not token:
        return {"bid_volume": 0, "ask_volume": 0}
    try:
        url = f"{BASE_URL}/market-quote/depth"
        headers = {"Authorization": f"Bearer {access_token_str}", "Accept": "application/json"}
        res = requests.get(url, headers=headers, params={"instrument_key": token})
        res.raise_for_status() # Raise an exception for bad status codes
        depth = res.json().get('data', {}).get(token, {}).get('depth', {})
        bid_volume = sum(item.get('quantity', 0) for item in depth.get('buy', []))
        ask_volume = sum(item.get('quantity', 0) for item in depth.get('sell', []))
        logger.info(f"Fetched depth for {token}: Bid={bid_volume}, Ask={ask_volume}")
        # time.sleep(0.1) # Simple Rate limiting
        return {"bid_volume": bid_volume, "ask_volume": ask_volume}
    except requests.exceptions.RequestException as e:
        logger.error(f"Depth fetch error for {token}: {e}")
        # st.warning(f"Could not fetch depth for {token}: {e}") # Avoid too many warnings in UI
        return {}
    except Exception as e:
        logger.error(f"An unexpected error occurred fetching depth for {token}: {e}")
        return {}

def get_user_data(user_api_client, portfolio_api_client, order_api_client):
    """Fetch user portfolio and order data using Upstox SDK."""
    if not (user_api_client and portfolio_api_client and order_api_client):
         logger.warning("API clients not available for portfolio data.")
         return {}

    data = {}
    try:
        data['margin'] = user_api_client.get_user_fund_margin(api_version="v2").to_dict()
        logger.info("Fetched funds")
    except ApiException as e:
        logger.error(f"Funds fetch error: {e}")
        logger.warning("Funds service may be unavailable or token invalid.")
        data['margin'] = {}
    except Exception as e:
        logger.error(f"An unexpected error occurred fetching funds: {e}")
        data['margin'] = {}

    try:
        data['holdings'] = portfolio_api_client.get_holdings(api_version="v2").to_dict()
        logger.info("Fetched holdings")
    except Exception as e:
        logger.error(f"Holdings fetch error: {e}")
        st.warning("Could not fetch holdings.")
        data['holdings'] = {}

    try:
        data['positions'] = portfolio_api_client.get_positions(api_version="v2").to_dict()
        logger.info("Fetched positions")
    except Exception as e:
        logger.error(f"Positions fetch error: {e}")
        st.warning("Could not fetch positions.")
        data['positions'] = {}

    try:
        data['orders'] = order_api_client.get_order_book(api_version="v2").to_dict()
        logger.info("Fetched orders")
    except Exception as e:
        logger.error(f"Orders fetch error: {e}")
        st.warning("Could not fetch orders.")
        data['orders'] = {}

    try:
        # Note: Upstox SDK v2 has get_trade_book, not get_trade_history as in some older docs
        data['trades'] = order_api_client.get_trade_book(api_version="v2").to_dict()
        logger.info("Fetched trades")
    except Exception as e:
        logger.error(f"Trades fetch error: {e}")
        st.warning("Could not fetch trades.")
        data['trades'] = {}

    # time.sleep(0.1) # Simple Rate limiting
    return data


# === DATA PROCESSING AND METRICS (Adapted from your Colab script) ===

def process_chain(chain_data):
    """Process raw option chain data into a DataFrame and calculate total OI."""
    # Uses st.session_state['prev_oi'] for persistence in Streamlit
    if 'prev_oi' not in st.session_state:
         st.session_state['prev_oi'] = {} # Ensure it exists in session state

    rows, total_ce_oi, total_pe_oi = [], 0, 0
    current_oi = {} # Dictionary to store current OI for change calculation

    for item in chain_data:
        ce = item.get('call_options', {})
        pe = item.get('put_options', {})
        ce_md, pe_md = ce.get('market_data', {}), pe.get('market_data', {})
        ce_gk, pe_gk = ce.get('option_greeks', {}), pe.get('option_greeks', {})
        strike = item.get('strike_price')

        if strike is None:
            continue # Skip if strike price is missing

        # --- FIX: Robust Extraction and Conversion of OI ---
        # Get the raw value first, could be None or something else
        raw_ce_oi = ce_md.get("oi") 
        raw_pe_oi = pe_md.get("oi")

        # Convert to integer, default to 0 if None or conversion fails
        try:
            ce_oi_val = int(raw_ce_oi) if pd.notna(raw_ce_oi) and raw_ce_oi is not None else 0
        except (ValueError, TypeError):
            logger.warning(f"Could not convert CE OI value '{raw_ce_oi}' for strike {strike} to int. Defaulting to 0.")
            ce_oi_val = 0

        try:
            pe_oi_val = int(raw_pe_oi) if pd.notna(raw_pe_oi) and raw_pe_oi is not None else 0
        except (ValueError, TypeError):
            logger.warning(f"Could not convert PE OI value '{raw_pe_oi}' for strike {strike} to int. Defaulting to 0.")
            pe_oi_val = 0
        # --- End FIX ---

        # Calculate OI change and percentage using session state's prev_oi
        # Use .get() with default 0 in case strike was not present in the previous fetch
        prev_ce_oi = st.session_state['prev_oi'].get(f"{strike}_CE", 0)
        prev_pe_oi = st.session_state['prev_oi'].get(f"{strike}_PE", 0)

        ce_oi_change = ce_oi_val - prev_ce_oi
        pe_oi_change = pe_oi_val - prev_pe_oi

        # Handle division by zero for percentage change calculation
        ce_oi_change_pct = (ce_oi_change / prev_ce_oi * 100) if prev_ce_oi > 0 else 0
        pe_oi_change_pct = (pe_oi_change / prev_pe_oi * 100) if prev_pe_oi > 0 else 0

        # Store current OI for next run's calculation
        current_oi[f"{strike}_CE"] = ce_oi_val
        current_oi[f"{strike}_PE"] = pe_oi_val


        row = {
            "Strike": strike,
            "CE_LTP": ce_md.get("ltp"),
            "CE_IV": ce_gk.get("iv"), # Key from your Colab script
            "CE_Delta": ce_gk.get("delta"),
            "CE_Theta": ce_gk.get("theta"),
            "CE_Vega": ce_gk.get("vega"),
            "CE_OI": ce_oi_val,
            "CE_OI_Change": ce_oi_change,
            "CE_OI_Change_Pct": ce_oi_change_pct,
            "CE_Volume": ce_md.get("volume", 0),
            "PE_LTP": pe_md.get("ltp"),
            "PE_IV": pe_gk.get("iv"), # Key from your Colab script
            "PE_Delta": pe_gk.get("delta"),
            "PE_Theta": pe_gk.get("theta"),
            "PE_Vega": pe_gk.get("vega"),
            "PE_OI": pe_oi_val,
            "PE_OI_Change": pe_oi_change,
            "PE_OI_Change_Pct": pe_oi_change_pct,
            "PE_Volume": pe_md.get("volume", 0),
            "Strike_PCR": pe_oi_val / ce_oi_val if ce_oi_val > 0 else (1 if pe_oi_val > 0 else 0), # Handle division by zero
            "CE_Token": ce.get("instrument_key"),
            "PE_Token": pe.get("instrument_key")
        }
        total_ce_oi += ce_oi_val
        total_pe_oi += pe_oi_val
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("Strike").reset_index(drop=True)

    # Update prev_oi in session state for the next run
    st.session_state['prev_oi'] = current_oi

    return df, total_ce_oi, total_pe_oi

def calculate_metrics(df, ce_oi_total, pe_oi_total, spot):
    """Calculate key option metrics using logic from your Colab script."""
    if df.empty or spot is None:
         return None, None, None, None # Return None if data is insufficient

    # ATM Strike (Closest strike to spot)
    atm_strike_row = df.iloc[(df['Strike'] - spot).abs().argsort()[:1]]
    atm_strike = atm_strike_row['Strike'].values[0] if not atm_strike_row.empty else None

    # Total PCR
    pcr = round(pe_oi_total / ce_oi_total, 2) if ce_oi_total > 0 else (1 if pe_oi_total > 0 else 0) # Handle division by zero


    # Max Pain (Standard Calculation from your Colab script)
    max_pain_strike = None
    if not df.empty:
        max_pain_series = []
        strikes = df['Strike'].unique() # Use unique strikes as potential expiry points
        # Calculate loss for each strike price assuming it's the expiry price
        for assumed_expiry_price in strikes:
            # Total seller gain (which is total buyer loss)
            total_seller_gain_at_strike = sum(
                (row['CE_OI'] * max(0, assumed_expiry_price - row['Strike'])) +
                (row['PE_OI'] * max(0, row['Strike'] - assumed_expiry_price))
                for index, row in df.iterrows()
            )
            max_pain_series.append((assumed_expiry_price, total_seller_gain_at_strike))

        if max_pain_series:
             # Max Pain is the strike with the MINIMUM total seller gain (maximum buyer loss)
            max_pain_strike = min(max_pain_series, key=lambda item: item[1])[0]
    else:
        max_pain_strike = None # No strikes to calculate max pain


    # Straddle Price at ATM (Sum of ATM CE LTP + PE LTP from your Colab script)
    straddle_price = None
    if atm_strike is not None:
        straddle_row = df[df["Strike"] == atm_strike]
        if not straddle_row.empty:
            ce_ltp = straddle_row['CE_LTP'].values[0]
            pe_ltp = straddle_row['PE_LTP'].values[0]
            # Ensure LTPs are not None before summing
            if pd.notna(ce_ltp) and pd.notna(pe_ltp):
                 straddle_price = float(ce_ltp + pe_ltp)


    return pcr, max_pain_strike, straddle_price, atm_strike

def plot_iv_skew(df, spot, atm_strike):
    """Plot IV Skew using Matplotlib."""
    if df.empty or spot is None or atm_strike is None:
        logger.warning("Insufficient data for IV Skew plot.")
        return None

    # Filter for valid IV data points (> 0 and not NaN) as in your Colab script
    valid = df[(df['CE_IV'].notna()) & (df['PE_IV'].notna()) & (df['CE_IV'] > 0) & (df['PE_IV'] > 0)].copy() # Use copy to avoid warnings if manipulating later

    if valid.empty:
        logger.warning("No valid IV data points available for plotting.")
        return None

    # Convert IV to percentage if they are small decimals (common in some APIs)
    # Check a sample to decide if conversion is needed. Assume they are already % for now like your screenshot
    # If API returns like 0.15, uncomment below:
    # if valid['CE_IV'].iloc[0] < 5 and valid['CE_IV'].iloc[0] > 0: # Heuristic check
    #    valid['CE_IV'] = valid['CE_IV'] * 100
    #    valid['PE_IV'] = valid['PE_IV'] * 100


    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(valid['Strike'], valid['CE_IV'], label='Call IV', color='blue', marker='o', linestyle='-')
    ax.plot(valid['Strike'], valid['PE_IV'], label='Put IV', color='red', marker='o', linestyle='-')
    ax.axvline(spot, color='gray', linestyle='--', label=f'Spot ({spot:.2f})')
    ax.axvline(atm_strike, color='green', linestyle=':', label=f'ATM ({atm_strike})')

    # Highlight the ATM strike data point if it's in the valid data
    if atm_strike in valid['Strike'].values:
         atm_valid_ivs = valid[valid['Strike'] == atm_strike][['CE_IV', 'PE_IV']].values.flatten()
         ax.scatter([atm_strike] * len(atm_valid_ivs), atm_valid_ivs, color='green', zorder=5, s=100) # Larger markers

    ax.set_title("IV Skew")
    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Implied Volatility (%)")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    return fig


# === MAIN DATA FETCHING LOGIC ===

# This function orchestrates the calls using the initialized client bundle
def fetch_all_data(upstox_client_bundle):
    """Fetches all market and portfolio data using the initialized client."""
    if not upstox_client_bundle:
        # This case should ideally be prevented by disabling the button
        st.error("Upstox client not initialized. Cannot fetch data.")
        return None

    # Extract clients/token from the bundle
    options_api_client = upstox_client_bundle.get("options_api")
    access_token_str = upstox_client_bundle.get("access_token")
    user_api_client = upstox_client_bundle.get("user_api")
    portfolio_api_client = upstox_client_bundle.get("portfolio_api")
    order_api_client = upstox_client_bundle.get("order_api")

    if not (options_api_client and access_token_str and user_api_client and portfolio_api_client and order_api_client):
         st.error("Error: Required API clients are not available.")
         return None


    data = {}
    
    # --- Fetch Market Data ---
    st.subheader("Fetching Market Data...")
    
    with st.spinner("Fetching India VIX..."):
        data['vix'] = fetch_vix(access_token_str)

    with st.spinner("Fetching nearest expiry..."):
        data['expiry'] = get_nearest_expiry(options_api_client)
    
    if not data['expiry']:
        st.error("Failed to get nearest expiry. Cannot fetch option chain.")
        # Return partial data; dependent steps will be skipped
        return data

    with st.spinner(f"Fetching option chain for {data['expiry']}..."):
        chain_raw = fetch_option_chain(options_api_client, data['expiry'])

    if not chain_raw:
        st.warning("Option chain data is empty.")
        # Ensure dependent data structures are empty or None
        data['option_chain_df'] = pd.DataFrame()
        data['nifty_spot'] = None
        data['total_ce_oi'] = 0
        data['total_pe_oi'] = 0
    else:
        # Extract spot price from the first element (assuming it's present)
        data['nifty_spot'] = chain_raw[0].get("underlying_spot_price")
        
        # Fallback to fetch spot separately if missing (as implemented in your Colab script)
        if data['nifty_spot'] is None:
             logger.warning("Nifty spot price missing in option chain data, attempting to fetch separately.")
             try:
                 url = f"{BASE_URL}/market-quote/quotes"
                 headers = {"Authorization": f"Bearer {access_token_str}", "Accept": "application/json"}
                 res = requests.get(url, headers=headers, params={"instrument_key": INSTRUMENT_KEY})
                 res.raise_for_status()
                 spot_data = res.json().get('data', {}).get(INSTRUMENT_KEY, {})
                 data['nifty_spot'] = spot_data.get('last_price')
                 logger.info(f"Fetched Nifty Spot separately: {data['nifty_spot']}")
             except Exception as e:
                  logger.error(f"Failed to fetch Nifty Spot separately: {e}")
                  st.error("Failed to fetch Nifty spot price.")
                  data['nifty_spot'] = None # Ensure spot is None if separate fetch fails
        
        if data['nifty_spot'] is not None:
             with st.spinner("Processing option chain..."):
                # Process the raw chain data into a DataFrame and get total OI
                data['option_chain_df'], data['total_ce_oi'], data['total_pe_oi'] = process_chain(chain_raw)
        else:
             # If spot is still None, we can't process or calculate metrics reliably
             data['option_chain_df'] = pd.DataFrame()
             data['total_ce_oi'] = 0
             data['total_pe_oi'] = 0
             st.error("Could not determine Nifty spot price. Cannot process option chain.")


    # Calculations and dependent fetches only if we have spot and processed chain data
    if data.get('option_chain_df') is not None and not data['option_chain_df'].empty and data.get('nifty_spot') is not None:
         with st.spinner("Calculating metrics..."):
             # Calculate key metrics based on the processed DataFrame
             data['pcr'], data['max_pain_strike'], data['straddle_price'], data['atm_strike'] = calculate_metrics(
                 data['option_chain_df'], data['total_ce_oi'], data['total_pe_oi'], data['nifty_spot']
             )

         with st.spinner("Fetching market depth for ATM strikes..."):
            # Fetch depth only if ATM strike and corresponding tokens were found
            if data.get('atm_strike') is not None:
                 atm_row = data['option_chain_df'][data['option_chain_df']['Strike'] == data['atm_strike']]
                 if not atm_row.empty:
                     ce_token = atm_row['CE_Token'].values[0] if pd.notna(atm_row['CE_Token'].values[0]) else None
                     pe_token = atm_row['PE_Token'].values[0] if pd.notna(atm_row['PE_Token'].values[0]) else None
                     data['ce_depth'] = get_market_depth(access_token_str, ce_token)
                     data['pe_depth'] = get_market_depth(access_token_str, pe_token)
                 else:
                      logger.warning(f"ATM strike {data['atm_strike']} not found in processed dataframe for depth fetch.")
                      data['ce_depth'] = {}
                      data['pe_depth'] = {}
            else:
                logger.warning("ATM strike not available for depth fetch.")
                data['ce_depth'] = {}
                data['pe_depth'] = {}
    else:
        # Ensure these keys exist even if data fetching/processing failed partially
        data['pcr'] = None
        data['max_pain_strike'] = None
        data['straddle_price'] = None
        data['atm_strike'] = None
        data['ce_depth'] = {}
        data['pe_depth'] = {}


    # --- Fetch User/Portfolio Data ---
    st.subheader("Fetching User Data...")
    with st.spinner("Fetching portfolio and user data..."):
        data['user_data'] = get_user_data(user_api_client, portfolio_api_client, order_api_client)


    data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.success("Data fetch attempt complete.") # Indicate process finished, even if some parts failed
    return data


# === STREAMLIT UI LAYOUT AND DISPLAY ===

# --- Sidebar for Token Input and Fetch Button ---
st.sidebar.header("Settings")
# Token input field
token_input = st.sidebar.text_input("Enter Upstox Access Token:", type="password")

# Initialize client if token is entered and client is not already initialized with this token
if token_input and (st.session_state['upstox_client'] is None or st.session_state['initialized_token'] != token_input):
    # Clear previous state and re-initialize
    st.session_state['upstox_client'] = None
    st.session_state['initialized_token'] = None
    st.session_state['latest_data'] = None
    st.session_state['prev_oi'] = {} # Reset OI history on new token

    with st.spinner("Initializing Upstox client..."):
        st.session_state['upstox_client'] = initialize_upstox_client(token_input)
        if st.session_state['upstox_client']:
             st.session_state['initialized_token'] = token_input # Store the token that successfully initialized the client
             st.sidebar.success("Client initialized. You can now fetch data.")
        else:
             st.sidebar.error("Client initialization failed. Please check your token.")

elif not token_input and st.session_state['upstox_client'] is not None:
    # If token is cleared in the input field, invalidate the current client and data
    st.session_state['upstox_client'] = None
    st.session_state['initialized_token'] = None
    st.session_state['latest_data'] = None
    st.session_state['prev_oi'] = {} # Reset OI history
    st.sidebar.warning("Access token cleared. Client de-initialized.")


# Fetch button - only enable if client is initialized
fetch_button = st.sidebar.button("Fetch Latest Data", disabled=st.session_state['upstox_client'] is None)

# --- Trigger Data Fetch ---
if fetch_button and st.session_state['upstox_client'] is not None:
    # Fetch data and store in session state
    st.session_state['latest_data'] = fetch_all_data(st.session_state['upstox_client'])

# --- Display Data ---
if st.session_state['latest_data']:
    data = st.session_state['latest_data']

    st.header("📊 Market Snapshot")
    
    # Use columns for a clean layout of key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nifty Spot", f"{data.get('nifty_spot', 'N/A'):,.2f}" if data.get('nifty_spot') is not None else "N/A") # Added comma formatting
    col2.metric("India VIX", f"{data.get('vix', 'N/A'):,.2f}" if data.get('vix') is not None else "N/A") # Added comma formatting
    col3.metric("Total PCR", f"{data.get('pcr', 'N/A'):.2f}" if data.get('pcr') is not None else "N/A") # Added formatting
    col4.metric("Nearest Expiry", data.get('expiry', 'N/A'))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("ATM Strike", data.get('atm_strike', 'N/A'))
    col6.metric("ATM Straddle Price", f"{data.get('straddle_price', 'N/A'):,.2f}" if data.get('straddle_price') is not None else "N/A") # Added comma formatting
    col7.metric("Max Pain Strike", data.get('max_pain_strike', 'N/A'))
    
    ce_depth = data.get('ce_depth', {})
    pe_depth = data.get('pe_depth', {})
    col8.metric("ATM Depth (B/A)", f"CE: {ce_depth.get('bid_volume', 0):,}/{ce_depth.get('ask_volume', 0):,} | PE: {pe_depth.get('bid_volume', 0):,}/{pe_depth.get('ask_volume', 0):,}") # Added comma formatting

    st.markdown(f"*(Last Updated: {data.get('timestamp', 'N/A')})*")


    st.header("📉 IV Skew Plot")
    # Pass the dataframe and relevant metrics to the plotting function
    option_chain_df = data.get('option_chain_df')
    if option_chain_df is not None and not option_chain_df.empty and data.get('nifty_spot') is not None and data.get('atm_strike') is not None:
        iv_fig = plot_iv_skew(option_chain_df, data['nifty_spot'], data['atm_strike'])
        if iv_fig:
            st.pyplot(iv_fig)
            plt.close(iv_fig) # Close the figure to free up memory
        else:
            st.info("IV Skew plot not available due to insufficient valid IV data (check strikes with IV > 0).")
    else:
        st.info("IV Skew plot requires Nifty spot price, ATM strike, and valid option chain data.")


    st.header("🔑 Key Strikes (ATM ± 6)")
    if option_chain_df is not None and not option_chain_df.empty and data.get('atm_strike') is not None:
        atm_strike = data['atm_strike']
        # Ensure atm_strike exists in the DataFrame before trying to find index
        if atm_strike in option_chain_df['Strike'].values:
            atm_idx = option_chain_df.index[option_chain_df['Strike'] == atm_strike].tolist()[0] # Get index
            start_idx = max(0, atm_idx - 6)
            end_idx = min(len(option_chain_df) - 1, atm_idx + 6)
            key_strikes_df_raw = option_chain_df.iloc[start_idx : end_idx + 1].copy() # Get the slice and copy

            # --- Apply Formatting for Display ---
            # This replicates the formatting logic from your Colab script's print output
            def format_oi_change_highlight(change):
                 if pd.isna(change): return "-"
                 # Highlight changes > 500,000 with an asterisk (adjust threshold as needed)
                 return f"{change:,.0f}*" if abs(change) > 500000 else f"{change:,.0f}" # Format with commas

            def format_large_number(num):
                 if pd.isna(num) or num == 0: return "-"
                 return f"{num:,.0f}" # Format with commas

            def format_float(num, decimals=2):
                 if pd.isna(num): return "-"
                 return f"{num:,.{decimals}f}" # Format with commas and decimals

            def format_pct(num):
                 if pd.isna(num): return "-"
                 return f"{num:.1f}%" if abs(num) < 10000 else f"{num:,.0f}%" # Format percentage


            key_strikes_df_formatted = key_strikes_df_raw[[
                 'Strike' # Keep Strike as is
            ]].copy()

            # Format other columns
            for col in ['CE_LTP', 'CE_IV', 'CE_Delta', 'CE_Theta', 'CE_Vega']:
                if col in key_strikes_df_raw.columns:
                     key_strikes_df_formatted[col] = key_strikes_df_raw[col].apply(format_float)
            for col in ['PE_LTP', 'PE_IV', 'PE_Delta', 'PE_Theta', 'PE_Vega']:
                if col in key_strikes_df_raw.columns:
                     key_strikes_df_formatted[col] = key_strikes_df_raw[col].apply(format_float)

            for col in ['CE_OI', 'CE_Volume']:
                if col in key_strikes_df_raw.columns:
                     key_strikes_df_formatted[col] = key_strikes_df_raw[col].apply(format_large_number)
            for col in ['PE_OI', 'PE_Volume']:
                if col in key_strikes_df_raw.columns:
                     key_strikes_df_formatted[col] = key_strikes_df_raw[col].apply(format_large_number)

            # Apply special formatting for OI Change
            if 'CE_OI_Change' in key_strikes_df_raw.columns:
                key_strikes_df_formatted['CE_OI_Change'] = key_strikes_df_raw['CE_OI_Change'].apply(format_oi_change_highlight)
            if 'PE_OI_Change' in key_strikes_df_raw.columns:
                key_strikes_df_formatted['PE_OI_Change'] = key_strikes_df_raw['PE_OI_Change'].apply(format_oi_change_highlight)


            # Apply special formatting for OI Change Pct
            if 'CE_OI_Change_Pct' in key_strikes_df_raw.columns:
                 key_strikes_df_formatted['CE_OI_Change_%'] = key_strikes_df_raw['CE_OI_Change_Pct'].apply(format_pct)
            if 'PE_OI_Change_Pct' in key_strikes_df_raw.columns:
                 key_strikes_df_formatted['PE_OI_Change_%'] = key_strikes_df_raw['PE_OI_Change_Pct'].apply(format_pct)


            # Apply special formatting for Strike PCR
            if 'Strike_PCR' in key_strikes_df_raw.columns:
                 key_strikes_df_formatted['Strike_PCR'] = key_strikes_df_raw['Strike_PCR'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")


            # Define desired column order and names for display
            display_cols_order = [
                'Strike', 'CE_LTP', 'CE_IV', 'CE_Delta', 'CE_Theta', 'CE_Vega',
                'CE_OI', 'CE_OI_Change', 'CE_OI_Change_%', 'CE_Volume',
                'PE_LTP', 'PE_IV', 'PE_Delta', 'PE_Theta', 'PE_Vega',
                'PE_OI', 'PE_OI_Change', 'PE_OI_Change_%', 'PE_Volume',
                'Strike_PCR'
            ]

            # Rename formatted columns to original names for ordered selection
            rename_map = {
                col + '_Formatted': col for col in [
                    'CE_LTP', 'CE_IV', 'CE_Delta', 'CE_Theta', 'CE_Vega',
                    'CE_OI', 'CE_Volume', 'PE_LTP', 'PE_IV', 'PE_Delta',
                    'PE_Theta', 'PE_Vega', 'PE_OI', 'PE_Volume'
                    ] if col + '_Formatted' in key_strikes_df_formatted.columns
                }
            if 'CE_OI_Change' in key_strikes_df_formatted.columns: rename_map['CE_OI_Change'] = 'CE_OI_Change' # Keep name if formatted in place
            if 'PE_OI_Change' in key_strikes_df_formatted.columns: rename_map['PE_OI_Change'] = 'PE_OI_Change'
            if 'CE_OI_Change_%' in key_strikes_df_formatted.columns: rename_map['CE_OI_Change_%'] = 'CE_OI_Change_%'
            if 'PE_OI_Change_%' in key_strikes_df_formatted.columns: rename_map['PE_OI_Change_%'] = 'PE_OI_Change_%'
            if 'Strike_PCR' in key_strikes_df_formatted.columns: rename_map['Strike_PCR'] = 'Strike_PCR'


            # Apply renaming where needed (only to formatted cols that got new names)
            key_strikes_df_formatted = key_strikes_df_formatted.rename(columns=rename_map)

            # Select and order columns for final display, keeping only those that exist
            final_display_cols = [col for col in display_cols_order if col in key_strikes_df_formatted.columns]


            st.dataframe(key_strikes_df_formatted[final_display_cols], hide_index=True, use_container_width=True)
            st.markdown("* Significant OI Change (e.g., > 500,000) is marked with an asterisk.")

        else:
             st.warning(f"ATM strike {atm_strike} found but not present in the current option chain data range.")
    else:
        st.info("Option chain data or ATM strike not available to display key strikes.")


    st.header("📖 Full Option Chain Data")
    if option_chain_df is not None and not option_chain_df.empty:
        # Re-apply the same formatting for the full table
        def format_oi_change_highlight_full(change): # Needed a slightly different func name due to scope in a larger script
             if pd.isna(change): return "-"
             return f"{change:,.0f}*" if abs(change) > 500000 else f"{change:,.0f}"

        def format_large_number_full(num):
             if pd.isna(num) or num == 0: return "-"
             return f"{num:,.0f}"

        def format_float_full(num, decimals=2):
             if pd.isna(num): return "-"
             return f"{num:,.{decimals}f}"

        def format_pct_full(num):
             if pd.isna(num): return "-"
             return f"{num:.1f}%" if abs(num) < 10000 else f"{num:,.0f}%"


        full_display_df_formatted = option_chain_df[['Strike']].copy()

        # Format other columns
        for col in ['CE_LTP', 'CE_IV', 'CE_Delta', 'CE_Theta', 'CE_Vega']:
            if col in option_chain_df.columns:
                 full_display_df_formatted[col] = option_chain_df[col].apply(format_float_full)
        for col in ['PE_LTP', 'PE_IV', 'PE_Delta', 'PE_Theta', 'PE_Vega']:
            if col in option_chain_df.columns:
                 full_display_df_formatted[col] = option_chain_df[col].apply(format_float_full)

        for col in ['CE_OI', 'CE_Volume']:
            if col in option_chain_df.columns:
                 full_display_df_formatted[col] = option_chain_df[col].apply(format_large_number_full)
        for col in ['PE_OI', 'PE_Volume']:
            if col in option_chain_df.columns:
                 full_display_df_formatted[col] = option_chain_df[col].apply(format_large_number_full)

        # Apply special formatting for OI Change
        if 'CE_OI_Change' in option_chain_df.columns:
             full_display_df_formatted['CE_OI_Change'] = option_chain_df['CE_OI_Change'].apply(format_oi_change_highlight_full)
        if 'PE_OI_Change' in option_chain_df.columns:
             full_display_df_formatted['PE_OI_Change'] = option_chain_df['PE_OI_Change'].apply(format_oi_change_highlight_full)

        # Apply special formatting for OI Change Pct
        if 'CE_OI_Change_Pct' in option_chain_df.columns:
             full_display_df_formatted['CE_OI_Change_%'] = option_chain_df['CE_OI_Change_Pct'].apply(format_pct_full)
        if 'PE_OI_Change_Pct' in option_chain_df.columns:
             full_display_df_formatted['PE_OI_Change_%'] = option_chain_df['PE_OI_Change_Pct'].apply(format_pct_full)


        # Apply special formatting for Strike PCR
        if 'Strike_PCR' in option_chain_df.columns:
             full_display_df_formatted['Strike_PCR'] = option_chain_df['Strike_PCR'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")

        # Define desired column order and names for display (same as key strikes)
        display_cols_order_full = [
            'Strike', 'CE_LTP', 'CE_IV', 'CE_Delta', 'CE_Theta', 'CE_Vega',
            'CE_OI', 'CE_OI_Change', 'CE_OI_Change_%', 'CE_Volume',
            'PE_LTP', 'PE_IV', 'PE_Delta', 'PE_Theta', 'PE_Vega',
            'PE_OI', 'PE_OI_Change', 'PE_OI_Change_%', 'PE_Volume',
            'Strike_PCR'
        ]

        # Rename formatted columns to original names for ordered selection
        rename_map_full = {
            col + '_Formatted': col for col in [
                'CE_LTP', 'CE_IV', 'CE_Delta', 'CE_Theta', 'CE_Vega',
                'CE_OI', 'CE_Volume', 'PE_LTP', 'PE_IV', 'PE_Delta',
                'PE_Theta', 'PE_Vega', 'PE_OI', 'PE_Volume'
                ] if col + '_Formatted' in full_display_df_formatted.columns
            }
        # Keep names for columns that got special handling in place
        if 'CE_OI_Change' in full_display_df_formatted.columns: rename_map_full['CE_OI_Change'] = 'CE_OI_Change'
        if 'PE_OI_Change' in full_display_df_formatted.columns: rename_map_full['PE_OI_Change'] = 'PE_OI_Change'
        if 'CE_OI_Change_%' in full_display_df_formatted.columns: rename_map_full['CE_OI_Change_%'] = 'CE_OI_Change_%'
        if 'PE_OI_Change_%' in full_display_df_formatted.columns: rename_map_full['PE_OI_Change_%'] = 'PE_OI_Change_%'
        if 'Strike_PCR' in full_display_df_formatted.columns: rename_map_full['Strike_PCR'] = 'Strike_PCR'


        # Apply renaming where needed
        full_display_df_formatted = full_display_df_formatted.rename(columns=rename_map_full)

        # Select and order columns for final display, keeping only those that exist
        final_display_cols_full = [col for col in display_cols_order_full if col in full_display_df_formatted.columns]


        st.dataframe(full_display_df_formatted[final_display_cols_full], hide_index=True, use_container_width=True)
    else:
        st.info("Full option chain data not available.")


    st.header("👤 Portfolio and User Data")
    user_data = data.get('user_data', {})

    if user_data:
        # Use tabs for organizing portfolio data
        tabs = st.tabs(["Funds/Margin", "Holdings", "Positions", "Orders", "Trades"])

        with tabs[0]:
            st.subheader("Funds/Margin")
            if user_data.get('margin', {}).get('data'):
                 # Use columns to display Equity and Commodity side-by-side if both exist
                 margin_col1, margin_col2 = st.columns(2)
                 equity_data = user_data['margin']['data'].get('equity', {})
                 commodity_data = user_data['margin']['data'].get('commodity', {})

                 with margin_col1:
                      st.write("**Equity:**")
                      if equity_data:
                           st.json(equity_data)
                      else:
                           st.info("No Equity margin data.")

                 with margin_col2:
                      st.write("**Commodity:**")
                      if commodity_data:
                           st.json(commodity_data)
                      else:
                           st.info("No Commodity margin data.")
                 
                 # Display any other top-level keys in 'data' if they exist
                 other_margin_keys = {k: v for k, v in user_data['margin']['data'].items() if k not in ['equity', 'commodity']}
                 if other_margin_keys:
                      st.write("**Other Margin Data:**")
                      st.json(other_margin_keys)

            else:
                st.info("No funds/margin data available.")

        with tabs[1]:
            st.subheader("Holdings")
            if user_data.get('holdings', {}).get('data'):
                 holdings_df = pd.DataFrame(user_data['holdings']['data'])
                 # Select relevant columns if they exist and format numbers
                 holding_cols_raw = ['tradingsymbol', 'quantity', 'average_price', 'last_price', 'close_price', 'pnl', 'product_type', 'exchange', 'isin']
                 holding_cols_formatted = ['Tradingsymbol', 'Quantity', 'Avg. Price', 'Last Price', 'Close Price', 'P&L', 'Product Type', 'Exchange', 'ISIN']
                 display_holding_cols = [col for col in holding_cols_raw if col in holdings_df.columns]

                 if display_holding_cols:
                      holdings_display_df = holdings_df[display_holding_cols].copy()
                      # Apply formatting
                      for col in ['quantity']:
                          if col in holdings_display_df.columns:
                               holdings_display_df[col] = holdings_display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "-")
                      for col in ['average_price', 'last_price', 'close_price', 'pnl']:
                          if col in holdings_display_df.columns:
                               holdings_display_df[col] = holdings_display_df[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "-")

                      holdings_display_df.columns = holding_cols_formatted[:len(display_holding_cols)] # Rename columns for display
                      st.dataframe(holdings_display_df, hide_index=True, use_container_width=True)
                 else:
                      st.info("No holdings data available or no displayable columns found.")
            else:
                st.info("No holdings data available.")

        with tabs[2]:
            st.subheader("Positions")
            if user_data.get('positions', {}).get('data'):
                 positions_df = pd.DataFrame(user_data['positions']['data'])
                 # Select relevant columns if they exist and format numbers
                 position_cols_raw = ['tradingsymbol', 'quantity', 'product_type', 'exchange', 'instrument_type', 'option_type', 'strike_price', 'expiry_date', 'average_price', 'last_price', 'realized_pnl', 'unrealized_pnl', 'buy_quantity', 'sell_quantity']
                 position_cols_formatted = ['Tradingsymbol', 'Qty', 'Product', 'Exchange', 'Type', 'Option', 'Strike', 'Expiry', 'Avg. Price', 'Last Price', 'Realized P&L', 'Unrealized P&L', 'Buy Qty', 'Sell Qty']
                 display_position_cols = [col for col in position_cols_raw if col in positions_df.columns]

                 if display_position_cols:
                      positions_display_df = positions_df[display_position_cols].copy()
                       # Apply formatting
                      for col in ['quantity', 'buy_quantity', 'sell_quantity']:
                          if col in positions_display_df.columns:
                               positions_display_df[col] = positions_display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "-")
                      for col in ['average_price', 'last_price', 'realized_pnl', 'unrealized_pnl']:
                          if col in positions_display_df.columns:
                               positions_display_df[col] = positions_display_df[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "-")

                      positions_display_df.columns = position_cols_formatted[:len(display_position_cols)] # Rename columns for display
                      st.dataframe(positions_display_df, hide_index=True, use_container_width=True)
                 else:
                      st.info("No positions data available or no displayable columns found.")
            else:
                st.info("No positions data available.")

        with tabs[3]:
            st.subheader("Orders")
            if user_data.get('orders', {}).get('data'):
                 orders_df = pd.DataFrame(user_data['orders']['data'])
                  # Select relevant columns if they exist and format numbers
                 order_cols_raw = ['tradingsymbol', 'exchange', 'status', 'quantity', 'order_type', 'transaction_type', 'price', 'average_price', 'order_timestamp', 'status_message']
                 order_cols_formatted = ['Tradingsymbol', 'Exchange', 'Status', 'Qty', 'Type', 'Txn Type', 'Price', 'Avg. Price', 'Timestamp', 'Message']
                 display_order_cols = [col for col in order_cols_raw if col in orders_df.columns]

                 if display_order_cols:
                      orders_display_df = orders_df[display_order_cols].copy()
                       # Apply formatting
                      for col in ['quantity']:
                          if col in orders_display_df.columns:
                               orders_display_df[col] = orders_display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "-")
                      for col in ['price', 'average_price']:
                          if col in orders_display_df.columns:
                               orders_display_df[col] = orders_display_df[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "-")

                      orders_display_df.columns = order_cols_formatted[:len(display_order_cols)] # Rename columns for display
                      st.dataframe(orders_display_df, hide_index=True, use_container_width=True)
                 else:
                      st.info("No order book data available or no displayable columns found.")

            else:
                st.info("No order book data available.")

        with tabs[4]:
            st.subheader("Trades")
            if user_data.get('trades', {}).get('data'):
                 trades_df = pd.DataFrame(user_data['trades']['data'])
                  # Select relevant columns if they exist and format numbers
                 trade_cols_raw = ['tradingsymbol', 'exchange', 'quantity', 'price', 'trade_timestamp', 'order_id', 'trade_id']
                 trade_cols_formatted = ['Tradingsymbol', 'Exchange', 'Qty', 'Price', 'Timestamp', 'Order ID', 'Trade ID']
                 display_trade_cols = [col for col in trade_cols_raw if col in trades_df.columns]

                 if display_trade_cols:
                      trades_display_df = trades_df[display_trade_cols].copy()
                       # Apply formatting
                      for col in ['quantity']:
                          if col in trades_display_df.columns:
                               trades_display_df[col] = trades_display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "-")
                      for col in ['price']:
                          if col in trades_display_df.columns:
                               trades_display_df[col] = trades_display_df[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "-")

                      trades_display_df.columns = trade_cols_formatted[:len(display_trade_cols)] # Rename columns for display
                      st.dataframe(trades_display_df, hide_index=True, use_container_width=True)
                 else:
                      st.info("No trade book data available or no displayable columns found.")
            else:
                st.info("No trade book data available.")


    else:
        st.info("Portfolio and User data not available (Check token, permissions, and market hours).")

# --- Initial State Message ---
if st.session_state['latest_data'] is None and st.session_state['upstox_client'] is None:
     st.info("☝️ Please enter your Upstox Access Token in the sidebar to get started.")
elif st.session_state['latest_data'] is None and st.session_state['upstox_client'] is not None:
     st.info("✅ Client initialized successfully. Click 'Fetch Latest Data' in the sidebar to load the Nifty option chain and market data.")
