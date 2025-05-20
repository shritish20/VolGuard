import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from upstox_client import Configuration, ApiClient, OptionsApi
from upstox_client.rest import ApiException  # Corrected import
from retrying import retry
from utils.logger import setup_logger
from config.settings import INSTRUMENT_KEY, UPSTOX_BASE_URL

logger = setup_logger()

@st.cache_data
def get_nearest_expiry(_options_api, instrument_key):
    """Fetch the nearest valid expiry date for the given instrument."""
    logger.info(f"Fetching expiry dates for instrument: {instrument_key}")
    try:
        res = _options_api.get_option_contracts(instrument_key=instrument_key)
        expiries = res.to_dict().get('data', [])
        if not expiries:
            logger.error("No expiry dates returned from API")
            return None
        expiry_dates = [e['expiry_date'] for e in expiries]
        today = datetime.now().date()
        valid_expiries = [datetime.strptime(date, '%Y-%m-%d').date() for date in expiry_dates if datetime.strptime(date, '%Y-%m-%d').date() >= today]
        if not valid_expiries:
            logger.error("No valid future expiry dates found")
            return None
        nearest_expiry = min(valid_expiries).strftime('%Y-%m-%d')
        logger.info(f"Nearest expiry: {nearest_expiry}")
        return nearest_expiry
    except Exception as e:
        logger.error(f"Error fetching expiry: {str(e)}")
        return None

@st.cache_data
def fetch_option_chain(_options_api, instrument_key, expiry):
    """Fetch option chain data for the given instrument and expiry."""
    logger.info(f"Fetching option chain for {instrument_key}, expiry {expiry}")
    try:
        res = _options_api.get_put_call_option_chain(instrument_key=instrument_key, expiry_date=expiry)
        data = res.to_dict().get('data', [])
        logger.info(f"Option chain response: {data[:2]}")  # Log first two entries
        return data
    except ApiException as e:
        logger.error(f"Option chain fetch failed: {str(e)}")
        return []

def process_chain(chain):
    """Process option chain data into a structured DataFrame."""
    logger.info("Processing option chain data")
    try:
        df = pd.DataFrame(chain)
        if df.empty:
            logger.error("Option chain data is empty")
            return df, 0, 0
        
        # Extract relevant fields
        df['Strike'] = df['strike_price']
        df['CE_LTP'] = df['call_option'].apply(lambda x: x.get('last_price', 0))
        df['CE_IV'] = df['call_option'].apply(lambda x: x.get('implied_volatility', 0))
        df['CE_Delta'] = df['call_option'].apply(lambda x: x.get('delta', 0))
        df['CE_Theta'] = df['call_option'].apply(lambda x: x.get('theta', 0))
        df['CE_Vega'] = df['call_option'].apply(lambda x: x.get('vega', 0))
        df['CE_OI'] = df['call_option'].apply(lambda x: x.get('open_interest', 0))
        df['CE_OI_Change'] = df['call_option'].apply(lambda x: x.get('oi_change', 0))
        df['CE_OI_Change_Pct'] = df['call_option'].apply(lambda x: x.get('oi_change_pct', 0))
        df['CE_Volume'] = df['call_option'].apply(lambda x: x.get('volume', 0))
        df['CE_Token'] = df['call_option'].apply(lambda x: x.get('instrument_key', ''))

        df['PE_LTP'] = df['put_option'].apply(lambda x: x.get('last_price', 0))
        df['PE_IV'] = df['put_option'].apply(lambda x: x.get('implied_volatility', 0))
        df['PE_Delta'] = df['put_option'].apply(lambda x: x.get('delta', 0))
        df['PE_Theta'] = df['put_option'].apply(lambda x: x.get('theta', 0))
        df['PE_Vega'] = df['put_option'].apply(lambda x: x.get('vega', 0))
        df['PE_OI'] = df['put_option'].apply(lambda x: x.get('open_interest', 0))
        df['PE_OI_Change'] = df['put_option'].apply(lambda x: x.get('oi_change', 0))
        df['PE_OI_Change_Pct'] = df['put_option'].apply(lambda x: x.get('oi_change_pct', 0))
        df['PE_Volume'] = df['put_option'].apply(lambda x: x.get('volume', 0))
        df['PE_Token'] = df['put_option'].apply(lambda x: x.get('instrument_key', ''))

        # Calculate additional metrics
        df['Strike_PCR'] = df['PE_OI'] / (df['CE_OI'] + 1e-10)
        df['OI_Skew'] = df['CE_OI'] - df['PE_OI']
        df['IV_Skew_Slope'] = df['CE_IV'] - df['PE_IV']

        ce_oi = df['CE_OI'].sum()
        pe_oi = df['PE_OI'].sum()
        logger.info(f"Processed option chain with {len(df)} rows, CE OI: {ce_oi}, PE OI: {pe_oi}")
        return df, ce_oi, pe_oi
    except Exception as e:
        logger.error(f"Error processing chain: {str(e)}")
        return pd.DataFrame(), 0, 0

def calculate_metrics(df, ce_oi, pe_oi, spot):
    """Calculate key market metrics from option chain data."""
    logger.info("Calculating market metrics")
    try:
        pcr = pe_oi / (ce_oi + 1e-10)
        max_pain = df.groupby('Strike').apply(
            lambda x: sum(max(0, x['Strike'].iloc[0] - s) * df[df['Strike'] == s]['CE_OI'].sum() +
                          max(0, s - x['Strike'].iloc[0]) * df[df['Strike'] == s]['PE_OI'].sum()
                          for s in df['Strike'].unique())
        ).idxmin()
        
        atm_strike = df['Strike'].iloc[(df['Strike'] - spot).abs().argmin()]
        straddle_price = (df[df['Strike'] == atm_strike]['CE_LTP'].iloc[0] +
                         df[df['Strike'] == atm_strike]['PE_LTP'].iloc[0])
        atm_iv = df[df['Strike'] == atm_strike]['CE_IV'].iloc[0]
        
        logger.info(f"Metrics: PCR={pcr}, Max Pain={max_pain}, ATM Strike={atm_strike}, ATM IV={atm_iv}")
        return pcr, max_pain, straddle_price, atm_strike, atm_iv
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return 0, 0, 0, 0, 0

@retry(stop_max_attempt_number=3, wait_fixed=2000)
def get_market_depth(access_token, base_url, instrument_key):
    """Fetch market depth for a given instrument."""
    logger.info(f"Fetching market depth for {instrument_key}")
    try:
        headers = {'Authorization': f'Bearer {access_token}'}
        response = requests.get(f"{base_url}/market-depth/{instrument_key}", headers=headers)
        response.raise_for_status()
        data = response.json().get('data', {})
        logger.info(f"Market depth: {data}")
        return data
    except Exception as e:
        logger.error(f"Error fetching market depth: {str(e)}")
        return {}

def run_volguard(access_token):
    """Run VolGuard data fetching pipeline."""
    logger.info("Starting VolGuard data fetch")

    if not access_token:
        logger.error("No access token provided")
        st.error("Please provide a valid Upstox access token.")
        return None, None, None, None, None

    try:
        logger.info("Initializing Upstox API client")
        configuration = Configuration()
        configuration.access_token = access_token
        client = ApiClient(configuration)
        options_api = OptionsApi(client)
        instrument_key = INSTRUMENT_KEY
        base_url = UPSTOX_BASE_URL

        logger.info(f"Fetching nearest expiry for {instrument_key}")
        expiry = get_nearest_expiry(options_api, instrument_key)
        if not expiry:
            logger.error("Failed to fetch expiry date")
            st.error("Could not fetch expiry date. Check your access token or API connectivity.")
            return None, None, None, None, None
        logger.info(f"Nearest expiry: {expiry}")

        logger.info(f"Fetching option chain for expiry {expiry}")
        chain = fetch_option_chain(options_api, instrument_key, expiry)
        if not chain:
            logger.error("Option chain fetch returned empty data")
            st.error("Failed to fetch option chain data. Possible API issue or invalid token.")
            return None, None, None, None, None
        logger.info(f"Option chain fetched with {len(chain)} entries")

        spot = chain[0].get("underlying_spot_price") or 0
        if not spot:
            logger.error("Spot price not found in option chain")
            st.error("Spot price not available in option chain data.")
            return None, None, None, None, None
        logger.info(f"Spot price: {spot}")

        logger.info("Processing option chain data")
        df, ce_oi, pe_oi = process_chain(chain)
        if df.empty:
            logger.error("Processed option chain DataFrame is empty")
            st.error("Failed to process option chain data.")
            return None, None, None, None, None
        logger.info(f"Option chain processed with {len(df)} rows")

        logger.info("Calculating market metrics")
        pcr, max_pain, straddle_price, atm_strike, atm_iv = calculate_metrics(df, ce_oi, pe_oi, spot)
        logger.info(f"Metrics: PCR={pcr}, Max Pain={max_pain}, ATM Strike={atm_strike}, ATM IV={atm_iv}")

        logger.info("Fetching market depth for ATM options")
        ce_depth = get_market_depth(access_token, base_url, df[df['Strike'] == atm_strike]['CE_Token'].values[0])
        pe_depth = get_market_depth(access_token, base_url, df[df['Strike'] == atm_strike]['PE_Token'].values[0])
        logger.info(f"CE Depth: {ce_depth}, PE Depth: {pe_depth}")

        logger.info("Generating IV skew plot")
        from utils.helpers import plot_iv_skew
        iv_skew_fig = plot_iv_skew(df, spot, atm_strike)
        if not iv_skew_fig:
            logger.warning("IV skew plot could not be generated")

        result = {
            "nifty_spot": spot,
            "atm_strike": atm_strike,
            "straddle_price": straddle_price,
            "pcr": pcr,
            "max_pain": max_pain,
            "expiry": expiry,
            "iv_skew_data": df.to_dict(),
            "ce_depth": ce_depth,
            "pe_depth": pe_depth,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "atm_iv": atm_iv
        }
        st.session_state.option_chain = chain
        logger.info("VolGuard data fetch completed successfully")
        return result, df, iv_skew_fig, atm_strike, atm_iv

    except Exception as e:
        logger.error(f"Volguard run error: {str(e)}")
        st.error(f"Error fetching data: {str(e)}. Check logs for details.")
        return None, None, None, None, None
