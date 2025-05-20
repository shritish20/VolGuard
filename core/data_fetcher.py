import streamlit as st
import pandas as pd
from datetime import datetime
from upstox_client import Configuration, ApiClient, OptionsApi, UserApi, PortfolioApi, OrderApi
from upstox_client.rest import ApiException
from retrying import retry
import requests
from config.settings import UPSTOX_BASE_URL, INSTRUMENT_KEY
from utils.logger import setup_logger

logger = setup_logger()

@st.cache_data(ttl=300)
def get_nearest_expiry(options_api, instrument_key):
    """Fetch nearest expiry date for the given instrument."""
    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def fetch_expiry():
        try:
            response = options_api.get_option_contracts(instrument_key=instrument_key)
            return response.to_dict().get("data", [])
        except ApiException as e:
            logger.error(f"Expiry fetch failed: {e}")
            raise

    try:
        contracts = fetch_expiry()
        expiry_dates = set()
        for contract in contracts:
            exp = contract.get("expiry")
            if isinstance(exp, str):
                exp = datetime.strptime(exp, "%Y-%m-%d")
            expiry_dates.add(exp)
        expiry_list = sorted(expiry_dates)
        today = datetime.now()
        valid_expiries = [e.strftime("%Y-%m-%d") for e in expiry_list if e >= today]
        return valid_expiries[0] if valid_expiries else None
    except Exception as e:
        logger.error(f"Expiry fetch error: {e}")
        return None

def fetch_option_chain(options_api, instrument_key, expiry):
    """Fetch option chain data."""
    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def fetch_chain():
        try:
            res = options_api.get_put_call_option_chain(instrument_key=instrument_key, expiry_date=expiry)
            return res.to_dict().get('data', [])
        except ApiException as e:
            logger.error(f"Option chain fetch failed: {e}")
            raise

    try:
        return fetch_chain()
    except Exception as e:
        logger.error(f"Option chain fetch error: {e}")
        return []

@st.cache_data(ttl=300)
def get_user_details(access_token):
    """Fetch user details from Upstox API."""
    try:
        configuration = Configuration()
        configuration.access_token = access_token
        client = ApiClient(configuration)
        user_api = UserApi(client)
        portfolio_api = PortfolioApi(client)
        order_api = OrderApi(client)
        details = {}
        details['profile'] = user_api.get_profile(api_version="v2").to_dict()
        details['funds'] = user_api.get_user_fund_margin(api_version="v2").to_dict()
        details['holdings'] = portfolio_api.get_holdings(api_version="v2").to_dict()
        details['positions'] = portfolio_api.get_positions(api_version="v2").to_dict()
        details['orders'] = order_api.get_order_book(api_version="v2").to_dict()
        details['trades'] = order_api.get_trade_history(api_version="v2").to_dict()
        return details
    except Exception as e:
        logger.error(f"User details fetch error: {e}")
        return {'error': str(e)}

@st.cache_data(ttl=300)
def get_market_depth(access_token, base_url, token):
    """Fetch market depth for a given instrument."""
    try:
        @retry(stop_max_attempt_number=3, wait_fixed=2000)
        def fetch_depth():
            headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
            url = f"{base_url}/market-quote/depth"
            res = requests.get(url, headers=headers, params={"instrument_key": token})
            res.raise_for_status()
            return res.json().get('data', {}).get(token, {}).get('depth', {})

        depth = fetch_depth()
        bid_volume = sum(item.get('quantity', 0) for item in depth.get('buy', []))
        ask_volume = sum(item.get('quantity', 0) for item in depth.get('sell', []))
        return {"bid_volume": bid_volume, "ask_volume": ask_volume}
    except Exception as e:
        logger.error(f"Depth fetch error for {token}: {e}")
        return {"bid_volume": 0, "ask_volume": 0}

def process_chain(data):
    """Process option chain data."""
    try:
        rows, ce_oi, pe_oi = [], 0, 0
        prev_oi = st.session_state.get('prev_oi', {})
        for r in data:
            ce = r.get('call_options', {})
            pe = r.get('put_options', {})
            ce_md, pe_md = ce.get('market_data', {}), pe.get('market_data', {})
            ce_gk, pe_gk = ce.get('option_greeks', {}), pe.get('option_greeks', {})
            strike = r.get('strike_price', 0)
            ce_oi_val = ce_md.get("oi", 0) or 0
            pe_oi_val = pe_md.get("oi", 0) or 0
            ce_oi_change = ce_oi_val - prev_oi.get(f"{strike}_CE", 0)
            pe_oi_change = pe_oi_val - prev_oi.get(f"{strike}_PE", 0)
            ce_oi_change_pct = (ce_oi_change / prev_oi.get(f"{strike}_CE", 1) * 100) if prev_oi.get(f"{strike}_CE", 0) else 0
            pe_oi_change_pct = (pe_oi_change / prev_oi.get(f"{strike}_PE", 1) * 100) if prev_oi.get(f"{strike}_PE", 0) else 0
            strike_pcr = pe_oi_val / (ce_oi_val or 1)
            row = {
                "Strike": strike,
                "CE_LTP": ce_md.get("ltp", 0) or 0,
                "CE_IV": ce_gk.get("iv", 0) or 0,
                "CE_Delta": ce_gk.get("delta", 0) or 0,
                "CE_Theta": ce_gk.get("theta", 0) or 0,
                "CE_Vega": ce_gk.get("vega", 0) or 0,
                "CE_OI": ce_oi_val,
                "CE_OI_Change": ce_oi_change,
                "CE_OI_Change_Pct": ce_oi_change_pct,
                "CE_Volume": ce_md.get("volume", 0) or 0,
                "PE_LTP": pe_md.get("ltp", 0) or 0,
                "PE_IV": pe_gk.get("iv", 0) or 0,
                "PE_Delta": pe_gk.get("delta", 0) or 0,
                "PE_Theta": pe_gk.get("theta", 0) or 0,
                "PE_Vega": pe_gk.get("vega", 0) or 0,
                "PE_OI": pe_oi_val,
                "PE_OI_Change": pe_oi_change,
                "PE_OI_Change_Pct": pe_oi_change_pct,
                "PE_Volume": pe_md.get("volume", 0) or 0,
                "Strike_PCR": strike_pcr,
                "CE_Token": ce.get("instrument_key", ""),
                "PE_Token": pe.get("instrument_key", "")
            }
            ce_oi += ce_oi_val
            pe_oi += pe_oi_val
            rows.append(row)
            prev_oi[f"{strike}_CE"] = ce_oi_val
            prev_oi[f"{strike}_PE"] = pe_oi_val
        st.session_state.prev_oi = prev_oi
        df = pd.DataFrame(rows).sort_values("Strike")
        if not df.empty:
            df['OI_Skew'] = (df['PE_OI'] - df['CE_OI']) / (df['PE_OI'] + df['CE_OI'] + 1)
            valid_iv = df[(df['CE_IV'] > 0) & (df['PE_IV'] > 0)]
            if len(valid_iv) > 2:
                iv_diff = (valid_iv['PE_IV'] - valid_iv['CE_IV']).abs()
                df['IV_Skew_Slope'] = iv_diff.rolling(window=3).mean().reindex(df.index, fill_value=0)
            else:
                df['IV_Skew_Slope'] = 0
        return df, ce_oi, pe_oi
    except Exception as e:
        logger.error(f"Option chain processing error: {e}")
        return pd.DataFrame(), 0, 0

def calculate_metrics(df, ce_oi_total, pe_oi_total, spot):
    """Calculate market metrics."""
    try:
        if df.empty:
            return 0, 0, 0, 0, 0
        atm = df.iloc[(df['Strike'] - spot).abs().argsort()[:1]]
        atm_strike = atm['Strike'].values[0] if not atm.empty else spot
        pcr = pe_oi_total / (ce_oi_total or 1)
        min_pain = float('inf')
        max_pain = spot
        for strike in df['Strike']:
            pain = 0
            for s in df['Strike']:
                if s <= strike:
                    pain += df[df['Strike'] == s]['CE_OI'].iloc[0] * max(0, strike - s)
                if s >= strike:
                    pain += df[df['Strike'] == s]['PE_OI'].iloc[0] * max(0, s - strike)
            if pain < min_pain:
                min_pain = pain
                max_pain = strike
        straddle_price = float(atm['CE_LTP'].values[0] + atm['PE_LTP'].values[0]) if not atm.empty else 0
        atm_iv = (atm['CE_IV'].values[0] + atm['PE_IV'].values[0]) / 2 if not atm.empty else 0
        return pcr, max_pain, straddle_price, atm_strike, atm_iv
    except Exception as e:
        logger.error(f"Metrics calculation error: {e}")
        return 0, 0, 0, 0, 0

def run_volguard(access_token):
    """Run VolGuard data fetching pipeline."""
    logger = setup_logger()
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
