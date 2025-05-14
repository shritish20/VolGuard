# === VOLGUARD PRO - STREAMLIT APP with Volatility Analysis ===
# Based on the original Colab script logic + Integrated Volatility
# Features: Sidebar Token Input, Dashboard with Tabs, Option Chain, Metrics, Depth, IV Skew, Portfolio, Volatility Analysis (GARCH, RV, ATM IV)
# Historical data fetched from user's GitHub repository.

import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
import logging
import time
import numpy as np # Added for volatility calculations
from arch import arch_model # Added for GARCH model
import io # Added to read CSV from URL response

# === CONFIG ===
INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"
BASE_URL = "https://api.upstox.com/v2"
NIFTY_HISTORICAL_CSV_URL = "https://raw.githubusercontent.com/shritish20/VolGuard/refs/heads/main/nifty_50.csv" # Your GitHub CSV URL

# === LOGGING (for console, Streamlit handles its own logs) ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === STREAMLIT SETUP ===
st.set_page_config(layout="wide", page_title="Volguard PRO - Nifty Dashboard")
st.title("📈 Volguard PRO: Nifty Options & Volatility Dashboard")

# Initialize session state
# Session state helps retain information across user interactions (reruns)
if 'prev_oi' not in st.session_state:
    st.session_state['prev_oi'] = {} # Stores OI from the last successful fetch for change calculation
if 'upstox_client' not in st.session_state:
     st.session_state['upstox_client'] = None # Stores the initialized Upstox client bundle
if 'initialized_token' not in st.session_state:
     st.session_state['initialized_token'] = None # Stores the token used for the current client
if 'latest_data' not in st.session_state:
     st.session_state['latest_data'] = None # Stores the results of the last data fetch


# === INIT API CLIENT ===
@st.cache_resource(ttl="1h") # Cache the client object itself, re-initialize if app restarts or cache expires (e.g., hourly)
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
        # Note: Fetching profile here adds latency but confirms token validity upfront
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
        logger.error(f"API Error during client initialization. Error: {e}")
        # Clear cached client if validation fails for the token
        st.error(f"API Error during client initialization. Please check your Access Token and permissions: {e}")
        st.cache_resource.clear() # Clear cache on API error during init
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during client initialization: {e}")
        st.error(f"An unexpected error occurred during client initialization: {e}")
        st.cache_resource.clear() # Clear cache on unexpected error during init
        return None

# === HISTORICAL DATA LOADING ===
@st.cache_data(ttl="1d") # Cache historical data, refresh daily
def load_nifty_historical_data(csv_url: str):
    """Fetches Nifty historical data from URL and preprocesses it."""
    try:
        logger.info(f"Attempting to fetch historical data from {csv_url}")
        response = requests.get(csv_url)
        response.raise_for_status() # Raise an exception for bad status codes

        # Read the content into a pandas DataFrame
        df = pd.read_csv(io.StringIO(response.text))

        # Preprocess data as in your Colab script
        df.columns = df.columns.str.strip()
        # Use errors='coerce' and dropna to handle potential parsing issues
        df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y", errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")
        df = df.rename(columns={"Close": "NIFTY_Close"})
        df = df[["NIFTY_Close"]].dropna().sort_index() # Keep only close price and handle missing

        logger.info(f"Successfully loaded and preprocessed historical data. Shape: {df.shape}")
        return df
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching historical data from URL {csv_url}: {e}")
        st.error(f"Error fetching historical data from {csv_url}. Please check the URL and your internet connection.")
        return pd.DataFrame() # Return empty DataFrame on error
    except Exception as e:
        logger.error(f"An error occurred during historical data processing: {e}")
        st.error(f"An error occurred processing historical data: {e}")
        return pd.DataFrame()


# === VOLATILITY CALCULATIONS ===
@st.cache_data(ttl="1h") # Cache volatility calculations, re-calculate hourly
def calculate_volatilities(nifty_historical_df: pd.DataFrame):
    """Calculates GARCH forecast and Realized Volatility."""
    if nifty_historical_df.empty or nifty_historical_df.shape[0] < 252 + 5: # Need enough data for 1-year HV and 5-day RV
        st.warning("Insufficient historical data to calculate volatilities.")
        return None # Return None if not enough data

    # Calculate log returns (%) as in your script
    log_returns = np.log(nifty_historical_df["NIFTY_Close"].pct_change() + 1).dropna() * 100

    if log_returns.empty:
         st.warning("Could not calculate log returns from historical data.")
         return None

    volatility_results = {}

    # 1. GARCH(1,1) Forecast (5 days)
    try:
        # Ensure enough data points for GARCH (typically needs a good history)
        if len(log_returns) > 100: # Arbitrary threshold, adjust if needed
            logger.info("Fitting GARCH(1,1) model...")
            # Use try-except as fitting can sometimes fail
            try:
                model = arch_model(log_returns, vol="Garch", p=1, q=1)
                model_fit = model.fit(disp="off") # disp="off" hides fitting output

                # Forecast next 5 business days (horizon=5)
                forecast_horizon = 5
                garch_forecast = model_fit.forecast(horizon=forecast_horizon)

                # Annualize the forecast volatility (std dev) and clip
                # Take the forecast standard deviations from the last step of the forecast
                # Ensure we handle potential non-finite values
                forecast_std_devs = garch_forecast.variance.values[-1]
                if np.any(~np.isfinite(forecast_std_devs)):
                    logger.warning("GARCH forecast contained non-finite values.")
                    garch_vols_annualized = [np.nan] * forecast_horizon
                else:
                    garch_vols_annualized = np.sqrt(forecast_std_devs) * np.sqrt(252) # Annualize
                    garch_vols_annualized = np.clip(garch_vols_annualized, 5, 50) # Clip as in your script

                volatility_results['garch_forecast_5d'] = np.round(garch_vols_annualized, 2)
                logger.info(f"GARCH 5-day forecast (annualized): {volatility_results['garch_forecast_5d']}")

            except Exception as e:
                logger.error(f"Error fitting or forecasting with GARCH model: {e}")
                st.warning(f"Could not fit or forecast with GARCH model: {e}")
                volatility_results['garch_forecast_5d'] = [np.nan] * 5 # Indicate failure

        else:
            logger.warning("Not enough log returns to fit GARCH model.")
            st.warning("Insufficient historical data points to fit the GARCH model effectively.")
            volatility_results['garch_forecast_5d'] = [np.nan] * 5 # Indicate insufficient data

    except Exception as e:
         logger.error(f"An unexpected error occurred during GARCH processing: {e}")
         volatility_results['garch_forecast_5d'] = [np.nan] * 5 # Indicate failure


    # 2. Realized Volatility (5-day, 30-day, 1-year)
    try:
        # Need at least 5 days of returns for 5-day RV, 30 for 30-day, 252 for 1-year
        if len(log_returns) >= 5:
            # 5-Day Realized Volatility (annualized std dev over last 5 trading days)
            rv_5d = log_returns[-5:].std() * np.sqrt(252) # Annualize
            volatility_results['rv_5d'] = round(rv_5d, 2)
        else:
             volatility_results['rv_5d'] = np.nan
             logger.warning("Not enough log returns for 5-day RV calculation.")


        if len(log_returns) >= 30:
            # 30-Day Historical Volatility (annualized std dev over last 30 trading days)
            rv_30d = log_returns[-30:].std() * np.sqrt(252) # Annualize
            volatility_results['rv_30d'] = round(rv_30d, 2)
        else:
             volatility_results['rv_30d'] = np.nan
             logger.warning("Not enough log returns for 30-day RV calculation.")


        if len(log_returns) >= 252:
            # 1-Year Historical Volatility (annualized std dev over last 252 trading days)
            rv_1y = log_returns[-252:].std() * np.sqrt(252) # Annualize
            volatility_results['rv_1y'] = round(rv_1y, 2)
        else:
            volatility_results['rv_1y'] = np.nan
            logger.warning("Not enough log returns for 1-year RV calculation.")

        logger.info(f"Realized Volatilities: 5d={volatility_results.get('rv_5d')}, 30d={volatility_results.get('rv_30d')}, 1y={volatility_results.get('rv_1y')}")

    except Exception as e:
        logger.error(f"Error calculating Realized Volatility: {e}")
        st.warning(f"Could not calculate Realized Volatility: {e}")
        volatility_results['rv_5d'] = np.nan
        volatility_results['rv_30d'] = np.nan
        volatility_results['rv_1y'] = np.nan


    # Get dates for the GARCH forecast
    last_hist_date = nifty_historical_df.index[-1] if not nifty_historical_df.empty else datetime.now().date()
    forecast_dates = pd.bdate_range(start=last_hist_date + timedelta(days=1), periods=5) # 5 business days

    volatility_results['garch_forecast_dates'] = forecast_dates.strftime("%Y-%m-%d").tolist() # Store as list of strings

    return volatility_results


# === API CALL FUNCTIONS (Adapted from your Colab script) ===
# (These remain similar to the previous Streamlit version, using the client bundle)

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
         return None, None, None, None, None # Added ATM IV return

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

    # ATM IV (Average of ATM CE IV and PE IV)
    atm_iv = None
    if atm_strike is not None:
         atm_row = df[df["Strike"] == atm_strike]
         if not atm_row.empty:
              ce_iv = atm_row['CE_IV'].values[0]
              pe_iv = atm_row['PE_IV'].values[0]
              if pd.notna(ce_iv) and pd.notna(pe_iv) and ce_iv > 0 and pe_iv > 0: # Only average if both are valid and > 0
                   atm_iv = (float(ce_iv) + float(pe_iv)) / 2.0
              elif pd.notna(ce_iv) and ce_iv > 0: # Use CE IV if only CE is valid
                   atm_iv = float(ce_iv)
              elif pd.notna(pe_iv) and pe_iv > 0: # Use PE IV if only PE is valid
                   atm_iv = float(pe_iv)


    return pcr, max_pain_strike, straddle_price, atm_strike, atm_iv

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

def plot_volatility_comparison(historical_df, garch_forecast_5d, garch_forecast_dates, rv_5d, atm_iv):
    """Plots historical RV, GARCH forecast, and ATM IV."""
    if historical_df is None or historical_df.empty:
         st.warning("Historical data not available for volatility comparison plot.")
         return None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot rolling 30-day RV for historical context
    if historical_df.shape[0] >= 30:
        log_returns = np.log(historical_df["NIFTY_Close"].pct_change() + 1).dropna() * 100
        # Calculate rolling 30-day std dev, annualize, and shift by -30 days to align with period end
        rolling_rv_30d = log_returns.rolling(window=30).std().dropna() * np.sqrt(252)
        ax.plot(rolling_rv_30d.index, rolling_rv_30d, label='Rolling 30-Day RV', color='orange', alpha=0.7)
    else:
        st.warning("Not enough historical data for Rolling 30-Day RV plot.")


    # Plot GARCH 5-day forecast
    if garch_forecast_5d is not None and garch_forecast_dates and len(garch_forecast_5d) == len(garch_forecast_dates):
        # Ensure forecast dates are datetime objects for plotting
        forecast_dates_dt = pd.to_datetime(garch_forecast_dates)
        ax.plot(forecast_dates_dt, garch_forecast_5d, marker='o', linestyle='--', color='purple', label='GARCH (1,1) 5-Day Forecast')

    # Add current 5-day Realized Volatility as a point or line from the last data point
    if rv_5d is not None and not historical_df.empty:
         last_date = historical_df.index[-1]
         # Add a point at the last historical date, representing the RV up to that point
         ax.scatter(last_date, rv_5d, color='blue', zorder=5, s=100, label=f'Latest 5-Day RV ({rv_5d:.2f}%)')


    # Add current ATM IV as a horizontal line
    if atm_iv is not None:
         ax.axhline(atm_iv, color='red', linestyle='-', label=f'Current ATM IV ({atm_iv:.2f}%)')


    ax.set_title("Volatility Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Volatility (%)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig


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


# === MAIN DATA FETCHING ORCHESTRATION ===

# This function orchestrates ALL data fetching (market, historical, user)
def fetch_all_data(upstox_client_bundle):
    """Fetches all market, historical volatility, and portfolio data."""
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

    # --- Fetch Historical Data for Volatility ---
    st.subheader("Loading Historical Data...")
    with st.spinner(f"Fetching Nifty historical data from {NIFTY_HISTORICAL_CSV_URL}..."):
        data['nifty_historical_df'] = load_nifty_historical_data(NIFTY_HISTORICAL_CSV_URL)


    # --- Calculate Volatilities ---
    st.subheader("Calculating Volatilities...")
    if data.get('nifty_historical_df') is not None and not data['nifty_historical_df'].empty:
        with st.spinner("Calculating GARCH forecast and Realized Volatility..."):
            data['volatility_metrics'] = calculate_volatilities(data['nifty_historical_df'])
    else:
        data['volatility_metrics'] = None
        st.warning("Cannot calculate volatilities without historical data.")


    # --- Fetch Live Market Data ---
    st.subheader("Fetching Live Market Data...") # Separate spinner for API calls

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
        data['atm_iv'] = None # Ensure ATM IV is None if chain is empty
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
                # calculate_metrics also returns ATM IV now
                data['option_chain_df'], data['total_ce_oi'], data['total_pe_oi'], data['pcr'], data['max_pain_strike'], data['straddle_price'], data['atm_strike'], data['atm_iv'] = calculate_metrics(
                    chain_raw, data['total_ce_oi'], data['total_pe_oi'], data['nifty_spot'] # Pass chain_raw here for process_chain
                    # Note: process_chain returns df, total_ce_oi, total_pe_oi
                    # calculate_metrics takes df, total_ce_oi, total_pe_oi, spot and returns pcr, max_pain_strike, straddle_price, atm_strike, atm_iv
                    # Need to chain these calls correctly. Let's adjust the flow.
                    )

             # Corrected flow: Process chain first, then calculate metrics from the resulting DF
             processed_chain_result = process_chain(chain_raw)
             data['option_chain_df'], data['total_ce_oi'], data['total_pe_oi'] = processed_chain_result

             if data.get('option_chain_df') is not None and not data['option_chain_df'].empty and data.get('nifty_spot') is not None:
                  with st.spinner("Calculating metrics..."):
                       data['pcr'], data['max_pain_strike'], data['straddle_price'], data['atm_strike'], data['atm_iv'] = calculate_metrics(
                           data['option_chain_df'], data['total_ce_oi'], data['total_pe_oi'], data['nifty_spot']
                       )
             else:
                  st.warning("Processed option chain empty. Cannot calculate metrics.")
                  data['pcr'] = None
                  data['max_pain_strike'] = None
                  data['straddle_price'] = None
                  data['atm_strike'] = None
                  data['atm_iv'] = None # ATM IV also None


        else:
             # If spot is still None, we can't process or calculate metrics reliably
             data['option_chain_df'] = pd.DataFrame()
             data['total_ce_oi'] = 0
             data['total_pe_oi'] = 0
             data['pcr'] = None
             data['max_pain_strike'] = None
             data['straddle_price'] = None
             data['atm_strike'] = None
             data['atm_iv'] = None
             st.error("Could not determine Nifty spot price. Cannot process option chain.")


    # Calculations and dependent fetches only if we have spot and processed chain data for market depth
    if data.get('option_chain_df') is not None and not data['option_chain_df'].empty and data.get('atm_strike') is not None and data.get('nifty_spot') is not None:
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
    st.cache_resource.clear() # Clear client cache on new token attempt

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
    st.cache_resource.clear() # Clear client cache on clear token
    st.sidebar.warning("Access token cleared. Client de-initialized.")


# Fetch button - only enable if client is initialized
fetch_button = st.sidebar.button("Fetch Latest Data", disabled=st.session_state['upstox_client'] is None)

# --- Trigger Data Fetch ---
if fetch_button and st.session_state['upstox_client'] is not None:
    # Fetch data and store in session state
    st.session_state['latest_data'] = fetch_all_data(st.session_state['upstox_client'])

# --- Display Data using Tabs ---
if st.session_state['latest_data']:
    data = st.session_state['latest_data']

    # Define the tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Market & Option Chain", "📉 Volatility Analysis", "👤 Portfolio Data", "⚙️ Raw Data (Debug)"])

    # --- Tab 1: Market & Option Chain ---
    with tab1:
        st.header("📊 Market & Option Chain Snapshot")
        
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

        st.subheader("🔑 Key Strikes (ATM ± 6)")
        option_chain_df = data.get('option_chain_df')

        if option_chain_df is not None and not option_chain_df.empty and data.get('atm_strike') is not None:
            atm_strike = data['atm_strike']
            # Ensure atm_strike exists in the DataFrame before trying to find index
            if atm_strike in option_chain_df['Strike'].values:
                atm_idx = option_chain_df.index[option_chain_df['Strike'] == atm_strike].tolist()[0] # Get index
                start_idx = max(0, atm_idx - 6)
                end_idx = min(len(option_chain_df) - 1, atm_idx + 6)
                key_strikes_df_raw = option_chain_df.iloc[start_idx : end_idx + 1].copy() # Get the slice and copy

                # --- Apply Formatting for Display ---
                def format_oi_change_highlight(change):
                     if pd.isna(change): return "-"
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

                # Format other columns (check if column exists before applying)
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

                # Apply special formatting for OI Change (check if column exists before applying)
                if 'CE_OI_Change' in key_strikes_df_raw.columns:
                    key_strikes_df_formatted['CE_OI_Change'] = key_strikes_df_raw['CE_OI_Change'].apply(format_oi_change_highlight)
                if 'PE_OI_Change' in key_strikes_df_raw.columns:
                    key_strikes_df_formatted['PE_OI_Change'] = key_strikes_df_raw['PE_OI_Change'].apply(format_oi_change_highlight)


                # Apply special formatting for OI Change Pct (check if column exists before applying)
                if 'CE_OI_Change_Pct' in key_strikes_df_raw.columns:
                     key_strikes_df_formatted['CE_OI_Change_%'] = key_strikes_df_raw['CE_OI_Change_Pct'].apply(format_pct)
                if 'PE_OI_Change_Pct' in key_strikes_df_raw.columns:
                     key_strikes_df_formatted['PE_OI_Change_%'] = key_strikes_df_raw['PE_OI_Change_Pct'].apply(format_pct)


                # Apply special formatting for Strike PCR (check if column exists before applying)
                if 'Strike_PCR' in key_strikes_df_raw.columns:
                     key_strikes_df_formatted['Strike_PCR'] = key_strikes_df_raw['Strike_PCR'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")


                # Define desired column order for display, keeping only those that exist after formatting
                display_cols_order = [
                    'Strike', 'CE_LTP', 'CE_IV', 'CE_Delta', 'CE_Theta', 'CE_Vega',
                    'CE_OI', 'CE_OI_Change', 'CE_OI_Change_%', 'CE_Volume',
                    'PE_LTP', 'PE_IV', 'PE_Delta', 'PE_Theta', 'PE_Vega',
                    'PE_OI', 'PE_OI_Change', 'PE_OI_Change_%', 'PE_Volume',
                    'Strike_PCR'
                ]

                # Rename columns for display headers if needed (e.g., add '%' to % columns)
                display_names = {
                     'CE_OI_Change_%': 'CE_OI_Change (%)',
                     'PE_OI_Change_%': 'PE_OI_Change (%)'
                }

                # Select and order columns for final display, keeping only those that exist and applying display names
                final_display_cols = [col for col in display_cols_order if col in key_strikes_df_formatted.columns]


                st.dataframe(key_strikes_df_formatted[final_display_cols].rename(columns=display_names), hide_index=True, use_container_width=True)
                st.markdown("* Significant OI Change (e.g., > 500,000) is marked with an asterisk.")

            else:
                 st.warning(f"ATM strike {atm_strike} found but not present in the current option chain data range.")
        else:
            st.info("Option chain data or ATM strike not available to display key strikes.")


        st.subheader("📖 Full Option Chain Data")
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

            # Format other columns (check if column exists)
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

            # Apply special formatting for OI Change (check if column exists)
            if 'CE_OI_Change' in option_chain_df.columns:
                 full_display_df_formatted['CE_OI_Change'] = option_chain_df['CE_OI_Change'].apply(format_oi_change_highlight_full)
            if 'PE_OI_Change' in option_chain_df.columns:
                 full_display_df_formatted['PE_OI_Change'] = option_chain_df['PE_OI_Change'].apply(format_oi_change_highlight_full)

            # Apply special formatting for OI Change Pct (check if column exists)
            if 'CE_OI_Change_Pct' in option_chain_df.columns:
                 full_display_df_formatted['CE_OI_Change_%'] = option_chain_df['CE_OI_Change_Pct'].apply(format_pct_full)
            if 'PE_OI_Change_Pct' in option_chain_df.columns:
                 full_display_df_formatted['PE_OI_Change_%'] = option_chain_df['PE_OI_Change_Pct'].apply(format_pct_full)


            # Apply special formatting for Strike PCR (check if column exists)
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

            # Rename columns for display headers if needed (e.g., add '%' to % columns)
            display_names_full = {
                 'CE_OI_Change_%': 'CE_OI_Change (%)',
                 'PE_OI_Change_%': 'PE_OI_Change (%)'
            }

            # Select and order columns for final display, keeping only those that exist
            final_display_cols_full = [col for col in display_cols_order_full if col in full_display_df_formatted.columns]

            st.dataframe(full_display_df_formatted[final_display_cols_full].rename(columns=display_names_full), hide_index=True, use_container_width=True)
        else:
            st.info("Full option chain data not available.")


    # --- Tab 2: Volatility Analysis ---
    with tab2:
        st.header("📉 Volatility Analysis")

        vol_metrics = data.get('volatility_metrics')
        atm_iv = data.get('atm_iv') # Get ATM IV from option chain metrics

        if vol_metrics is not None:
            # Display key metrics
            vol_col1, vol_col2, vol_col3, vol_col4 = st.columns(4)
            vol_col1.metric("Latest 5-Day Realized Volatility", f"{vol_metrics.get('rv_5d', 'N/A'):.2f}%" if pd.notna(vol_metrics.get('rv_5d')) else "N/A")
            # Display the first day of GARCH forecast as a representative value
            garch_1d_forecast = vol_metrics.get('garch_forecast_5d')
            vol_col2.metric("GARCH(1,1) 1-Day Forecast", f"{garch_1d_forecast[0]:.2f}%" if garch_1d_forecast is not None and len(garch_1d_forecast) > 0 and pd.notna(garch_1d_forecast[0]) else "N/A")
            vol_col3.metric("Current ATM IV", f"{atm_iv:.2f}%" if atm_iv is not None else "N/A")
            vol_col4.metric("Implied Volatility (India VIX)", f"{data.get('vix', 'N/A'):.2f}" if data.get('vix') is not None else "N/A")


            st.subheader("GARCH(1,1) 5-Day Forecast Details")
            garch_forecast_5d_values = vol_metrics.get('garch_forecast_5d')
            garch_forecast_dates = vol_metrics.get('garch_forecast_dates')

            if garch_forecast_5d_values is not None and garch_forecast_dates:
                forecast_df = pd.DataFrame({
                    "Date": garch_forecast_dates,
                    "Forecasted Volatility (%)": garch_forecast_5d_values
                })
                # Format percentages in the table
                forecast_df['Forecasted Volatility (%)'] = forecast_df['Forecasted Volatility (%)'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                st.dataframe(forecast_df, hide_index=True, use_container_width=True)
            else:
                st.info("GARCH forecast details not available.")


            st.subheader("Historical Realized Volatility")
            # Display fixed historical volatilities
            hist_vol_col1, hist_vol_col2 = st.columns(2)
            hist_vol_col1.metric("Latest 30-Day Realized Volatility", f"{vol_metrics.get('rv_30d', 'N/A'):.2f}%" if pd.notna(vol_metrics.get('rv_30d')) else "N/A")
            hist_vol_col2.metric("Latest 1-Year Realized Volatility", f"{vol_metrics.get('rv_1y', 'N/A'):.2f}%" if pd.notna(vol_metrics.get('rv_1y')) else "N/A")


            st.subheader("Volatility Comparison Plot")
            historical_df = data.get('nifty_historical_df')
            if historical_df is not None and not historical_df.empty:
                vol_plot_fig = plot_volatility_comparison(
                    historical_df,
                    garch_forecast_5d=vol_metrics.get('garch_forecast_5d'),
                    garch_forecast_dates=vol_metrics.get('garch_forecast_dates'),
                    rv_5d=vol_metrics.get('rv_5d'),
                    atm_iv=atm_iv
                )
                if vol_plot_fig:
                    st.pyplot(vol_plot_fig)
                    plt.close(vol_plot_fig) # Close the figure
                else:
                     st.info("Volatility comparison plot could not be generated.")

            else:
                 st.info("Historical data is required to plot volatility comparison.")


        else:
            st.info("Volatility calculations not available. Please ensure historical data is loaded correctly.")


    # --- Tab 3: Portfolio Data ---
    with tab3:
        st.header("👤 Portfolio and User Data")
        user_data = data.get('user_data', {})

        if user_data:
            # Use tabs for organizing portfolio data within this main tab
            portfolio_tab1, portfolio_tab2, portfolio_tab3, portfolio_tab4, portfolio_tab5 = st.tabs(["Funds/Margin", "Holdings", "Positions", "Orders", "Trades"])

            with portfolio_tab1:
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

            with portfolio_tab2:
                st.subheader("Holdings")
                if user_data.get('holdings', {}).get('data'):
                     holdings_df = pd.DataFrame(user_data['holdings']['data'])
                     holding_cols_raw = ['tradingsymbol', 'quantity', 'average_price', 'last_price', 'close_price', 'pnl', 'product_type', 'exchange', 'isin']
                     holding_cols_formatted = ['Tradingsymbol', 'Quantity', 'Avg. Price', 'Last Price', 'Close Price', 'P&L', 'Product Type', 'Exchange', 'ISIN']
                     display_holding_cols = [col for col in holding_cols_raw if col in holdings_df.columns]

                     if display_holding_cols:
                          holdings_display_df = holdings_df[display_holding_cols].copy()
                          for col in ['quantity']:
                              if col in holdings_display_df.columns:
                                   holdings_display_df[col] = holdings_display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "-")
                          for col in ['average_price', 'last_price', 'close_price', 'pnl']:
                              if col in holdings_display_df.columns:
                                   holdings_display_df[col] = holdings_display_df[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "-")

                          holdings_display_df.columns = holding_cols_formatted[:len(display_holding_cols)]
                          st.dataframe(holdings_display_df, hide_index=True, use_container_width=True)
                     else:
                          st.info("No holdings data available or no displayable columns found.")
                else:
                    st.info("No holdings data available.")

            with portfolio_tab3:
                st.subheader("Positions")
                if user_data.get('positions', {}).get('data'):
                     positions_df = pd.DataFrame(user_data['positions']['data'])
                     position_cols_raw = ['tradingsymbol', 'quantity', 'product_type', 'exchange', 'instrument_type', 'option_type', 'strike_price', 'expiry_date', 'average_price', 'last_price', 'realized_pnl', 'unrealized_pnl', 'buy_quantity', 'sell_quantity']
                     position_cols_formatted = ['Tradingsymbol', 'Qty', 'Product', 'Exchange', 'Type', 'Option', 'Strike', 'Expiry', 'Avg. Price', 'Last Price', 'Realized P&L', 'Unrealized P&L', 'Buy Qty', 'Sell Qty']
                     display_position_cols = [col for col in position_cols_raw if col in positions_df.columns]

                     if display_position_cols:
                          positions_display_df = positions_df[display_position_cols].copy()
                          for col in ['quantity', 'buy_quantity', 'sell_quantity']:
                              if col in positions_display_df.columns:
                                   positions_display_df[col] = positions_display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "-")
                          for col in ['average_price', 'last_price', 'realized_pnl', 'unrealized_pnl']:
                              if col in positions_display_df.columns:
                                   positions_display_df[col] = positions_display_df[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "-")

                          positions_display_df.columns = position_cols_formatted[:len(display_position_cols)]
                          st.dataframe(positions_display_df, hide_index=True, use_container_width=True)
                     else:
                          st.info("No positions data available or no displayable columns found.")
                else:
                    st.info("No positions data available.")

            with portfolio_tab4:
                st.subheader("Orders")
                if user_data.get('orders', {}).get('data'):
                     orders_df = pd.DataFrame(user_data['orders']['data'])
                     order_cols_raw = ['tradingsymbol', 'exchange', 'status', 'quantity', 'order_type', 'transaction_type', 'price', 'average_price', 'order_timestamp', 'status_message']
                     order_cols_formatted = ['Tradingsymbol', 'Exchange', 'Status', 'Qty', 'Type', 'Txn Type', 'Price', 'Avg. Price', 'Timestamp', 'Message']
                     display_order_cols = [col for col in order_cols_raw if col in orders_df.columns]

                     if display_order_cols:
                          orders_display_df = orders_df[display_order_cols].copy()
                          for col in ['quantity']:
                              if col in orders_display_df.columns:
                                   orders_display_df[col] = orders_display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "-")
                          for col in ['price', 'average_price']:
                              if col in orders_display_df.columns:
                                   orders_display_df[col] = orders_display_df[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "-")

                          orders_display_df.columns = order_cols_formatted[:len(display_order_cols)]
                          st.dataframe(orders_display_df, hide_index=True, use_container_width=True)
                     else:
                          st.info("No order book data available or no displayable columns found.")

                else:
                    st.info("No order book data available.")

            with portfolio_tab5:
                st.subheader("Trades")
                if user_data.get('trades', {}).get('data'):
                     trades_df = pd.DataFrame(user_data['trades']['data'])
                     trade_cols_raw = ['tradingsymbol', 'exchange', 'quantity', 'price', 'trade_timestamp', 'order_id', 'trade_id']
                     trade_cols_formatted = ['Tradingsymbol', 'Exchange', 'Qty', 'Price', 'Timestamp', 'Order ID', 'Trade ID']
                     display_trade_cols = [col for col in trade_cols_raw if col in trades_df.columns]

                     if display_trade_cols:
                          trades_display_df = trades_df[display_trade_cols].copy()
                          for col in ['quantity']:
                              if col in trades_display_df.columns:
                                   trades_display_df[col] = trades_display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "-")
                          for col in ['price']:
                              if col in trades_display_df.columns:
                                   trades_display_df[col] = trades_display_df[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "-")

                          trades_display_df.columns = trade_cols_formatted[:len(display_trade_cols)]
                          st.dataframe(trades_display_df, hide_index=True, use_container_width=True)
                     else:
                          st.info("No trade book data available or no displayable columns found.")
                else:
                    st.info("No trade book data available.")

        else:
            st.info("Portfolio and User data not available (Check token, permissions, and market hours).")


    # --- Tab 4: Raw Data (Debug) ---
    with tab4:
         st.header("⚙️ Raw Fetched Data")
         st.write("This tab shows the raw data fetched from the APIs for debugging purposes.")
         st.json(data)


# --- Initial State Message ---
if st.session_state['latest_data'] is None and st.session_state['upstox_client'] is None:
     st.info("☝️ Please enter your Upstox Access Token in the sidebar to get started.")
elif st.session_state['latest_data'] is None and st.session_state['upstox_client'] is not None:
     st.info("✅ Client initialized successfully. Click 'Fetch Latest Data' in the sidebar to load the Nifty option chain and market data.")
