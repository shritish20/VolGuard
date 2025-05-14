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
st.set_page_config(layout="wide", page_title="Volguard PRO - Nifty Option Chain")
st.title("📈 Volguard PRO: Nifty Option Chain & Analysis")

# Initialize session state
if 'prev_oi' not in st.session_state:
    st.session_state['prev_oi'] = {}
if 'upstox_client' not in st.session_state:
     st.session_state['upstox_client'] = None
if 'initialized_token' not in st.session_state:
     st.session_state['initialized_token'] = None
if 'latest_data' not in st.session_state:
     st.session_state['latest_data'] = None


# === INIT CLIENT ===
@st.cache_resource # Cache the client object itself
def initialize_upstox_client(access_token: str):
    """Initialize Upstox API client with access token."""
    if not access_token:
        st.warning("Please enter your Upstox Access Token.")
        return None

    try:
        configuration = upstox_client.Configuration()
        configuration.access_token = access_token
        client = upstox_client.ApiClient(configuration)

        # Validate token by fetching profile
        # Note: Fetching profile here adds latency but confirms token validity upfront
        user_api = upstox_client.UserApi(client)
        user_api.get_profile(api_version="2") # Call to validate token
        
        logger.info("Token validated and Upstox client initialized successfully.")
        # Return a bundle of useful objects
        return {
            "client": client,
            "access_token": access_token,
            "user_api": user_api,
            "options_api": upstox_client.OptionsApi(client),
            "portfolio_api": upstox_client.PortfolioApi(client),
            "order_api": upstox_client.OrderApi(client),
        }
    except ApiException as e:
        logger.error(f"Error initializing Upstox client or validating token: {e}")
        st.error(f"Error initializing Upstox client. Please check your Access Token: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during client initialization: {e}")
        st.error(f"An unexpected error occurred during client initialization: {e}")
        return None


# === API CALL FUNCTIONS (These remain largely the same but use the bundle or token) ===

def fetch_vix(access_token):
    """Fetch India VIX value using REST API."""
    try:
        url = f"{BASE_URL}/market-quote/quotes"
        headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
        res = requests.get(url, headers=headers, params={"instrument_key": "NSE_INDEX|India VIX"})
        res.raise_for_status() # Raise an exception for bad status codes
        data = res.json().get('data', {}).get('NSE_INDEX|India VIX', {})
        vix = data.get('last_price')
        logger.info(f"Fetched India VIX: {vix}")
        #time.sleep(0.1) # Simple rate limiting - adjust or remove based on needs and Streamlit's rerun
        return float(vix) if vix is not None else None
    except requests.exceptions.RequestException as e:
        logger.error(f"VIX fetch error: {e}")
        st.warning("Could not fetch India VIX. Service may be unavailable or token invalid.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred fetching VIX: {e}")
        return None


def get_nearest_expiry(options_api):
    """Get the nearest expiry date for the instrument."""
    try:
        response = options_api.get_option_contracts(instrument_key=INSTRUMENT_KEY)
        contracts = response.to_dict().get("data", [])
        expiry_dates = set()

        for contract in contracts:
            exp = contract.get("expiry")
            if isinstance(exp, str):
                exp = datetime.strptime(exp.split('T')[0], "%Y-%m-%d")
            expiry_dates.add(exp)

        expiry_list = sorted(expiry_dates)
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) # Compare only dates

        valid_expiries = [e.strftime("%Y-%m-%d") for e in expiry_list if e >= today]
        
        nearest = valid_expiries[0] if valid_expiries else None
        logger.info(f"Nearest expiry found: {nearest}")
        #time.sleep(0.1) # Simple rate limiting
        return nearest
    except ApiException as e:
        logger.error(f"Expiry fetch failed via SDK: {e}")
        st.error(f"Could not fetch expiry dates: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred fetching expiry: {e}")
        st.error(f"An error occurred fetching expiry dates: {e}")
        return None


def fetch_option_chain(options_api, expiry):
    """Fetch option chain data for given expiry using Upstox SDK."""
    try:
        # Using the SDK's get_put_call_option_chain which is more direct
        res = options_api.get_put_call_option_chain(instrument_key=INSTRUMENT_KEY, expiry_date=expiry)
        chain_data = res.to_dict().get('data', [])
        logger.info(f"Fetched option chain with {len(chain_data)} strikes.")
        #time.sleep(0.1) # Simple rate limiting
        return chain_data
    except ApiException as e:
        logger.error(f"Option chain fetch error via SDK: {e}")
        st.error(f"Could not fetch option chain data: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred fetching option chain: {e}")
        st.error(f"An error occurred fetching option chain data: {e}")
        return []


def fetch_market_depth_by_scrip(access_token, token):
    """Fetch market depth for a given instrument using REST API."""
    if not token:
        return {"bid_volume": 0, "ask_volume": 0}
    try:
        url = f"{BASE_URL}/market-quote/depth"
        headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
        res = requests.get(url, headers=headers, params={"instrument_key": token})
        res.raise_for_status() # Raise an exception for bad status codes
        depth = res.json().get('data', {}).get(token, {}).get('depth', {})
        bid_volume = sum(item.get('quantity', 0) for item in depth.get('buy', []))
        ask_volume = sum(item.get('quantity', 0) for item in depth.get('sell', []))
        logger.info(f"Fetched depth for {token}: Bid={bid_volume}, Ask={ask_volume}")
        #time.sleep(0.1) # Simple rate limiting
        return {"bid_volume": bid_volume, "ask_volume": ask_volume}
    except requests.exceptions.RequestException as e:
        logger.error(f"Depth fetch error for {token}: {e}")
        # st.warning(f"Could not fetch depth for {token}: {e}") # Avoid too many warnings
        return {}
    except Exception as e:
        logger.error(f"An unexpected error occurred fetching depth for {token}: {e}")
        return {}

def fetch_all_api_portfolio_data(upstox_client_bundle):
    """Fetch all portfolio-related data using Upstox SDK."""
    portfolio_api = upstox_client_bundle.get("portfolio_api")
    order_api = upstox_client_bundle.get("order_api")
    user_api = upstox_client_bundle.get("user_api")

    if not (portfolio_api and order_api and user_api):
         st.warning("API objects not available for portfolio data.")
         return {}

    data = {}
    try:
        data['margin'] = user_api.get_user_fund_margin(api_version="2").to_dict()
        logger.info("Fetched funds/margin.")
    except ApiException as e:
        logger.error(f"Funds fetch error via SDK: {e}")
        st.warning("Could not fetch funds/margin. Service may be unavailable or token invalid.")
        data['margin'] = {}
    except Exception as e:
        logger.error(f"An unexpected error occurred fetching funds: {e}")
        data['margin'] = {}

    try:
        data['holdings'] = portfolio_api.get_holdings(api_version="2").to_dict()
        logger.info("Fetched holdings.")
    except Exception as e:
        logger.error(f"Holdings fetch error via SDK: {e}")
        st.warning("Could not fetch holdings.")
        data['holdings'] = {}

    try:
        data['positions'] = portfolio_api.get_positions(api_version="2").to_dict()
        logger.info("Fetched positions.")
    except Exception as e:
        logger.error(f"Positions fetch error via SDK: {e}")
        st.warning("Could not fetch positions.")
        data['positions'] = {}

    try:
        # Note: Upstox SDK v2 has get_trade_book, not get_trade_history
        data['trades'] = order_api.get_trade_book(api_version="2").to_dict()
        logger.info("Fetched trades.")
    except Exception as e:
        logger.error(f"Trades fetch error via SDK: {e}")
        st.warning("Could not fetch trades.")
        data['trades'] = {}

    try:
        data['orders'] = order_api.get_order_book(api_version="2").to_dict()
        logger.info("Fetched orders.")
    except Exception as e:
        logger.error(f"Orders fetch error via SDK: {e}")
        st.warning("Could not fetch orders.")
        data['orders'] = {}

    #time.sleep(0.1) # Simple rate limiting
    return data


# === DATA PROCESSING AND METRICS ===

def process_chain(chain_data):
    """Process raw option chain data into a DataFrame and calculate total OI."""
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

        ce_oi_val = ce_md.get("oi", 0) or 0 # Ensure OI is treated as number, default to 0
        pe_oi_val = pe_md.get("oi", 0) or 0

        # Calculate OI change and percentage using session state
        # Use .get() with default 0 in case strike was not present in the previous fetch
        prev_ce_oi = st.session_state['prev_oi'].get(f"{strike}_CE", 0)
        prev_pe_oi = st.session_state['prev_oi'].get(f"{strike}_PE", 0)
        
        ce_oi_change = ce_oi_val - prev_ce_oi
        pe_oi_change = pe_oi_val - prev_pe_oi
        
        ce_oi_change_pct = (ce_oi_change / prev_ce_oi * 100) if prev_ce_oi > 0 else 0
        pe_oi_change_pct = (pe_oi_change / prev_pe_oi * 100) if prev_pe_oi > 0 else 0

        # Store current OI for next run's calculation
        current_oi[f"{strike}_CE"] = ce_oi_val
        current_oi[f"{strike}_PE"] = pe_oi_val

        row = {
            "Strike": strike,
            "CE_LTP": ce_md.get("last_price"),
            "CE_IV": ce_gk.get("implied_volatility"),
            "CE_Delta": ce_gk.get("delta"),
            "CE_Theta": ce_gk.get("theta"),
            "CE_Vega": ce_gk.get("vega"),
            "CE_OI": ce_oi_val,
            "CE_OI_Change": ce_oi_change,
            "CE_OI_Change_Pct": ce_oi_change_pct,
            "CE_Volume": ce_md.get("volume", 0),
            "PE_LTP": pe_md.get("last_price"),
            "PE_IV": pe_gk.get("implied_volatility"),
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
    
    # Update previous OI for the next run
    st.session_state['prev_oi'] = current_oi

    return df, total_ce_oi, total_pe_oi

def calculate_metrics(df, total_ce_oi, total_pe_oi, spot_price):
    """Calculate key option metrics."""
    if df.empty or spot_price is None:
        return None, None, None, None # Return None if data is insufficient

    # ATM Strike
    # Find the row where strike is closest to spot price
    atm_strike_row = df.iloc[(df['Strike'] - spot_price).abs().argsort()[:1]]
    atm_strike = atm_strike_row['Strike'].values[0] if not atm_strike_row.empty else None

    # Total PCR
    pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else (1 if total_pe_oi > 0 else 0) # Handle division by zero

    # Max Pain (Standard Calculation)
    # Max pain is the strike price at which option buyers would lose the maximum amount of money
    # Calculate loss for each strike price assuming it's the expiry price
    max_pain_strike = None
    if not df.empty:
        max_pain_series = []
        strikes = df['Strike'].unique() # Use unique strikes as potential expiry points
        for assumed_expiry_price in strikes:
            total_buyer_loss = 0
            # Sum up the potential loss for buyers at this assumed expiry price
            # Loss for CE buyers: max(0, assumed_expiry_price - Strike) * CE_OI
            # Loss for PE buyers: max(0, Strike - assumed_expiry_price) * PE_OI
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

    # Straddle Price at ATM
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
        #st.warning("Insufficient data to plot IV Skew.") # Avoid warning on every rerun without data
        return None

    valid = df[(df['CE_IV'].notna()) & (df['PE_IV'].notna()) & (df['CE_IV'] > 0) & (df['PE_IV'] > 0)]
    
    if valid.empty:
        #st.warning("No valid IV data points available for plotting.") # Avoid warning on every rerun without data
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(valid['Strike'], valid['CE_IV'], label='Call IV', color='blue', marker='o', linestyle='-')
    ax.plot(valid['Strike'], valid['PE_IV'], label='Put IV', color='red', marker='o', linestyle='-')
    ax.axvline(spot, color='gray', linestyle='--', label=f'Spot ({spot:.2f})')
    ax.axvline(atm_strike, color='green', linestyle=':', label=f'ATM ({atm_strike})')

    ax.set_title("IV Skew")
    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Implied Volatility (%)")
    ax.legend()
    ax.grid(True)
    
    # Highlight the ATM strike on the x-axis if it's within the data range
    if atm_strike in valid['Strike'].values:
         atm_ivs = valid[valid['Strike'] == atm_strike][['CE_IV', 'PE_IV']].values.flatten()
         ax.scatter([atm_strike] * len(atm_ivs), atm_ivs, color='green', zorder=5)

    plt.tight_layout()
    return fig


# === MAIN DATA FETCHING LOGIC ===

def fetch_all_data(upstox_client_bundle):
    """Fetches all market and portfolio data."""
    if not upstox_client_bundle:
        # This case should be handled by the UI logic before calling
        # st.error("Upstox client not initialized.")
        return None

    options_api = upstox_client_bundle["options_api"]
    access_token = upstox_client_bundle["access_token"] # Get token from bundle
    
    data = {}
    
    with st.spinner("Fetching India VIX..."):
        data['vix'] = fetch_vix(access_token)

    with st.spinner("Fetching nearest expiry..."):
        data['expiry'] = get_nearest_expiry(options_api)
    
    if not data['expiry']:
        st.error("Failed to get nearest expiry. Cannot fetch option chain.")
        # Return partial data, but don't reset session state data completely
        return data # Indicate failure by missing key data points

    with st.spinner(f"Fetching option chain for {data['expiry']}..."):
        chain_raw = fetch_option_chain(options_api, data['expiry'])

    if not chain_raw:
        st.warning("Option chain data is empty.")
        data['option_chain_df'] = pd.DataFrame()
        data['nifty_spot'] = None
        data['total_ce_oi'] = 0
        data['total_pe_oi'] = 0
    else:
        # Extract spot price from the first element (assuming it's present)
        data['nifty_spot'] = chain_raw[0].get("underlying_spot_price")
        
        # Fallback to fetch spot separately if missing
        if data['nifty_spot'] is None:
             logger.warning("Nifty spot price missing in option chain data, attempting to fetch separately.")
             try:
                 url = f"{BASE_URL}/market-quote/quotes"
                 headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
                 res = requests.get(url, headers=headers, params={"instrument_key": INSTRUMENT_KEY})
                 res.raise_for_status()
                 spot_data = res.json().get('data', {}).get(INSTRUMENT_KEY, {})
                 data['nifty_spot'] = spot_data.get('last_price')
                 logger.info(f"Fetched Nifty Spot separately: {data['nifty_spot']}")
             except Exception as e:
                  logger.error(f"Failed to fetch Nifty Spot separately: {e}")
                  st.error("Failed to fetch Nifty spot price.")
        
        if data['nifty_spot'] is not None:
             with st.spinner("Processing option chain..."):
                data['option_chain_df'], data['total_ce_oi'], data['total_pe_oi'] = process_chain(chain_raw)
        else:
             data['option_chain_df'] = pd.DataFrame()
             data['total_ce_oi'] = 0
             data['total_pe_oi'] = 0


    # Calculations and dependent fetches only if we have spot and processed chain data
    if data.get('option_chain_df') is not None and not data['option_chain_df'].empty and data.get('nifty_spot') is not None:
         with st.spinner("Calculating metrics..."):
             data['pcr'], data['max_pain_strike'], data['straddle_price'], data['atm_strike'] = calculate_metrics(
                 data['option_chain_df'], data['total_ce_oi'], data['total_pe_oi'], data['nifty_spot']
             )

         with st.spinner("Fetching market depth for ATM strikes..."):
            if data.get('atm_strike') is not None:
                 # Find the row corresponding to ATM strike to get tokens
                 atm_row = data['option_chain_df'][data['option_chain_df']['Strike'] == data['atm_strike']]
                 if not atm_row.empty:
                     ce_token = atm_row['CE_Token'].values[0] if pd.notna(atm_row['CE_Token'].values[0]) else None
                     pe_token = atm_row['PE_Token'].values[0] if pd.notna(atm_row['PE_Token'].values[0]) else None
                     data['ce_depth'] = fetch_market_depth_by_scrip(access_token, ce_token)
                     data['pe_depth'] = fetch_market_depth_by_scrip(access_token, pe_token)
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


    with st.spinner("Fetching portfolio data..."):
        data['portfolio_data'] = fetch_all_api_portfolio_data(upstox_client_bundle)

    data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.success("Data fetch attempt complete.") # Success message even if some parts failed
    return data


# === STREAMLIT UI LAYOUT AND DISPLAY ===

st.sidebar.header("Settings")
# Token input field
token = st.sidebar.text_input("Enter Upstox Access Token:", type="password")

# Initialize client if token is entered and client is not already initialized with this token
if token and (st.session_state['upstox_client'] is None or st.session_state['initialized_token'] != token):
    with st.spinner("Initializing Upstox client..."):
        st.session_state['upstox_client'] = initialize_upstox_client(token)
        if st.session_state['upstox_client']:
             st.session_state['initialized_token'] = token # Store the token that successfully initialized the client
             st.sidebar.success("Client initialized.")
        else:
             st.session_state['initialized_token'] = None # Clear token if initialization failed
             st.sidebar.error("Client initialization failed.")
elif not token and st.session_state['upstox_client'] is not None:
    # If token is cleared, invalidate the client
    st.session_state['upstox_client'] = None
    st.session_state['initialized_token'] = None
    st.sidebar.warning("Access token cleared. Client de-initialized.")
    st.session_state['latest_data'] = None # Clear displayed data too


# Fetch button - only enable if client is initialized
fetch_button = st.sidebar.button("Fetch Latest Data", disabled=st.session_state['upstox_client'] is None)

# --- Main Data Display ---
if fetch_button and st.session_state['upstox_client'] is not None:
    # Fetch data and store in session state
    st.session_state['latest_data'] = fetch_all_data(st.session_state['upstox_client'])

# Display the latest fetched data if available in session state
if st.session_state['latest_data']:
    data = st.session_state['latest_data']

    st.header("Market Snapshot")
    
    # Use columns for better layout of key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nifty Spot", f"{data.get('nifty_spot', 'N/A'):.2f}" if data.get('nifty_spot') is not None else "N/A")
    col2.metric("India VIX", f"{data.get('vix', 'N/A'):.2f}" if data.get('vix') is not None else "N/A")
    col3.metric("Total PCR", data.get('pcr', 'N/A'))
    col4.metric("Nearest Expiry", data.get('expiry', 'N/A'))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("ATM Strike", data.get('atm_strike', 'N/A'))
    col6.metric("ATM Straddle Price", f"{data.get('straddle_price', 'N/A'):.2f}" if data.get('straddle_price') is not None else "N/A")
    col7.metric("Max Pain Strike", data.get('max_pain_strike', 'N/A'))
    
    ce_depth = data.get('ce_depth', {})
    pe_depth = data.get('pe_depth', {})
    col8.metric("ATM Depth", f"CE: {ce_depth.get('bid_volume', 0)}/{ce_depth.get('ask_volume', 0)} | PE: {pe_depth.get('bid_volume', 0)}/{pe_depth.get('ask_volume', 0)}")
    
    st.write(f"*(Last Updated: {data.get('timestamp', 'N/A')})*")

    st.header("Key Strikes (ATM ± 6)")
    option_chain_df = data.get('option_chain_df')

    if option_chain_df is not None and not option_chain_df.empty and data.get('atm_strike') is not None:
        atm_strike = data['atm_strike']
        # Ensure atm_strike exists in the DataFrame before trying to find index
        if atm_strike in option_chain_df['Strike'].values:
            atm_idx = option_chain_df[option_chain_df['Strike'] == atm_strike].index[0]
            start_idx = max(0, atm_idx - 6)
            end_idx = min(len(option_chain_df) - 1, atm_idx + 6)
            key_strikes_df = option_chain_df.iloc[start_idx : end_idx + 1].copy() # Use copy to avoid SettingWithCopyWarning

            # Apply formatting and highlight significant OI changes
            def format_oi_change(change):
                if pd.isna(change):
                    return "-"
                # Highlight changes > 500,000 with an asterisk (adjust threshold as needed)
                return f"{change:,.0f}*" if abs(change) > 500000 else f"{change:,.0f}" # Format with commas

            key_strikes_df['CE_OI_Change'] = key_strikes_df['CE_OI_Change'].apply(format_oi_change)
            key_strikes_df['PE_OI_Change'] = key_strikes_df['PE_OI_Change'].apply(format_oi_change)
            
            # Format percentage columns
            # Handle large percentages cleanly if needed, currently caps at 10000 for detailed decimal
            key_strikes_df['CE_OI_Change_Pct'] = key_strikes_df['CE_OI_Change_Pct'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) and abs(x) < 10000 else (f"{x:,.0f}%" if pd.notna(x) else "-"))
            key_strikes_df['PE_OI_Change_Pct'] = key_strikes_df['PE_OI_Change_Pct'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) and abs(x) < 10000 else (f"{x:,.0f}%" if pd.notna(x) else "-"))
            
             # Format large numbers like OI and Volume
            def format_large_number(num):
                if pd.isna(num) or num == 0:
                    return "-"
                return f"{num:,.0f}" # Format with commas

            key_strikes_df['CE_OI'] = key_strikes_df['CE_OI'].apply(format_large_number)
            key_strikes_df['PE_OI'] = key_strikes_df['PE_OI'].apply(format_large_number)
            key_strikes_df['CE_Volume'] = key_strikes_df['CE_Volume'].apply(format_large_number)
            key_strikes_df['PE_Volume'] = key_strikes_df['PE_Volume'].apply(format_large_number)
            
            # Format other numerical columns
            key_strikes_df['CE_LTP'] = key_strikes_df['CE_LTP'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
            key_strikes_df['PE_LTP'] = key_strikes_df['PE_LTP'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
            key_strikes_df['CE_IV'] = key_strikes_df['CE_IV'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
            key_strikes_df['PE_IV'] = key_strikes_df['PE_IV'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
            key_strikes_df['CE_Delta'] = key_strikes_df['CE_Delta'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
            key_strikes_df['PE_Delta'] = key_strikes_df['PE_Delta'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
            key_strikes_df['CE_Theta'] = key_strikes_df['CE_Theta'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
            key_strikes_df['PE_Theta'] = key_strikes_df['PE_Theta'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
            key_strikes_df['CE_Vega'] = key_strikes_df['CE_Vega'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
            key_strikes_df['PE_Vega'] = key_strikes_df['PE_Vega'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
            key_strikes_df['Strike_PCR'] = key_strikes_df['Strike_PCR'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")


            # Drop token columns for display unless needed
            display_df = key_strikes_df.drop(columns=['CE_Token', 'PE_Token'], errors='ignore')

            st.dataframe(display_df, hide_index=True, use_container_width=True)
            st.write("* Significant OI Change (e.g., > 500,000) is marked with an asterisk.")

        else:
             st.warning("ATM strike found but not present in the current chain data range.")
    else:
        st.info("Option chain data or ATM strike not available to display key strikes.")


    st.header("IV Skew Plot")
    iv_fig = plot_iv_skew(option_chain_df, data.get('nifty_spot'), data.get('atm_strike'))
    if iv_fig:
        st.pyplot(iv_fig)
        plt.close(iv_fig) # Close the figure to free up memory
    else:
        st.info("IV Skew plot not available due to insufficient data.")


    st.header("Full Option Chain Data")
    if option_chain_df is not None and not option_chain_df.empty:
        # Drop token columns from full chain display too
        full_display_df = option_chain_df.drop(columns=['CE_Token', 'PE_Token'], errors='ignore')
        # Re-apply formatting for full table display consistency
        full_display_df['CE_OI_Change'] = full_display_df['CE_OI_Change'].apply(lambda x: f"{x:,.0f}*" if pd.notna(x) and abs(x) > 500000 else (f"{x:,.0f}" if pd.notna(x) else "-"))
        full_display_df['PE_OI_Change'] = full_display_df['PE_OI_Change'].apply(lambda x: f"{x:,.0f}*" if pd.notna(x) and abs(x) > 500000 else (f"{x:,.0f}" if pd.notna(x) else "-"))
        full_display_df['CE_OI_Change_Pct'] = full_display_df['CE_OI_Change_Pct'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) and abs(x) < 10000 else (f"{x:,.0f}%" if pd.notna(x) else "-"))
        full_display_df['PE_OI_Change_Pct'] = full_display_df['PE_OI_Change_Pct'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) and abs(x) < 10000 else (f"{x:,.0f}%" if pd.notna(x) else "-"))
        full_display_df['CE_OI'] = full_display_df['CE_OI'].apply(format_large_number)
        full_display_df['PE_OI'] = full_display_df['PE_OI'].apply(format_large_number)
        full_display_df['CE_Volume'] = full_display_df['CE_Volume'].apply(format_large_number)
        full_display_df['PE_Volume'] = full_display_df['PE_Volume'].apply(format_large_number)
        full_display_df['CE_LTP'] = full_display_df['CE_LTP'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        full_display_df['PE_LTP'] = full_display_df['PE_LTP'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        full_display_df['CE_IV'] = full_display_df['CE_IV'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        full_display_df['PE_IV'] = full_display_df['PE_IV'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        full_display_df['CE_Delta'] = full_display_df['CE_Delta'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        full_display_df['PE_Delta'] = full_display_df['PE_Delta'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        full_display_df['CE_Theta'] = full_display_df['CE_Theta'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        full_display_df['PE_Theta'] = full_display_df['PE_Theta'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        full_display_df['CE_Vega'] = full_display_df['CE_Vega'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        full_display_df['PE_Vega'] = full_display_df['PE_Vega'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        full_display_df['Strike_PCR'] = full_display_df['Strike_PCR'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")


        st.dataframe(full_display_df, hide_index=True, use_container_width=True)
    else:
        st.info("Full option chain data not available.")


    st.header("Portfolio and User Data")
    portfolio_data = data.get('portfolio_data', {})

    if portfolio_data:
        tabs = st.tabs(["Funds/Margin", "Holdings", "Positions", "Orders", "Trades"])

        with tabs[0]:
            st.subheader("Funds/Margin")
            if portfolio_data.get('margin', {}).get('data'):
                 equity_data = portfolio_data['margin']['data'].get('equity', {})
                 if equity_data:
                      st.write("**Equity:**")
                      st.json(equity_data)
                 commodity_data = portfolio_data['margin']['data'].get('commodity', {})
                 if commodity_data:
                      st.write("**Commodity:**")
                      st.json(commodity_data)
                 if not equity_data and not commodity_data:
                      st.info("No funds/margin data available.")
                 
                 # Display remaining keys at the top level of 'data' if any
                 other_margin_keys = {k: v for k, v in portfolio_data['margin']['data'].items() if k not in ['equity', 'commodity']}
                 if other_margin_keys:
                      st.write("**Other Margin Data:**")
                      st.json(other_margin_keys)

            else:
                st.info("No funds/margin data available.")

        with tabs[1]:
            st.subheader("Holdings")
            if portfolio_data.get('holdings', {}).get('data'):
                 # Ensure columns exist before selecting
                holdings_df = pd.DataFrame(portfolio_data['holdings']['data'])
                # Select relevant columns if they exist
                holding_cols = ['instrument_token', 'quantity', 'average_price', 'last_price', 'close_price', 'pnl', 'product_type', 'exchange', 'tradingsymbol', 'isin']
                display_holding_cols = [col for col in holding_cols if col in holdings_df.columns]
                st.dataframe(holdings_df[display_holding_cols], hide_index=True, use_container_width=True)
            else:
                st.info("No holdings data available.")

        with tabs[2]:
            st.subheader("Positions")
            if portfolio_data.get('positions', {}).get('data'):
                 # Ensure columns exist before selecting
                positions_df = pd.DataFrame(portfolio_data['positions']['data'])
                # Select relevant columns if they exist
                position_cols = ['instrument_token', 'quantity', 'multiplier', 'average_price', 'last_price', 'close_price', 'product_type', 'exchange', 'tradingsymbol', 'realized_pnl', 'unrealized_pnl', 'buy_price', 'sell_price', 'buy_quantity', 'sell_quantity', 'instrument_type', 'option_type', 'strike_price', 'expiry_date']
                display_position_cols = [col for col in position_cols if col in positions_df.columns]
                st.dataframe(positions_df[display_position_cols], hide_index=True, use_container_width=True)
            else:
                st.info("No positions data available.")

        with tabs[3]:
            st.subheader("Orders")
            if portfolio_data.get('orders', {}).get('data'):
                 orders_df = pd.DataFrame(portfolio_data['orders']['data'])
                 # Select relevant columns if they exist
                 order_cols = ['order_id', 'tradingsymbol', 'exchange', 'product', 'order_type', 'transaction_type', 'status', 'quantity', 'average_price', 'price', 'trigger_price', 'placed_by', 'order_timestamp', 'exchange_timestamp', 'instrument_token']
                 display_order_cols = [col for col in order_cols if col in orders_df.columns]
                 st.dataframe(orders_df[display_order_cols], hide_index=True, use_container_width=True)
            else:
                st.info("No order book data available.")

        with tabs[4]:
            st.subheader("Trades")
            if portfolio_data.get('trades', {}).get('data'):
                 trades_df = pd.DataFrame(portfolio_data['trades']['data'])
                 # Select relevant columns if they exist
                 trade_cols = ['trade_id', 'order_id', 'tradingsymbol', 'exchange', 'product', 'instrument_token', 'order_type', 'transaction_type', 'quantity', 'price', 'trade_timestamp']
                 display_trade_cols = [col for col in trade_cols if col in trades_df.columns]
                 st.dataframe(trades_df[display_trade_cols], hide_index=True, use_container_width=True)
            else:
                st.info("No trade book data available.")

    else:
        st.info("Portfolio and User data not available (Check token and market hours).")

# Initial state before fetching data
if st.session_state['latest_data'] is None and st.session_state['upstox_client'] is None:
     st.info("Please enter your Upstox Access Token in the sidebar to get started.")
elif st.session_state['latest_data'] is None and st.session_state['upstox_client'] is not None:
     st.info("Client initialized. Click 'Fetch Latest Data' in the sidebar to load the Nifty option chain and market data.")
