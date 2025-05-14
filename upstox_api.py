# upstox_api.py
import logging
import pandas as pd
import requests
import time
from datetime import datetime
import upstox_client
from upstox_client.rest import ApiException

# === CONFIG ===
base_url = "https://api.upstox.com/v2"
logger = logging.getLogger(__name__)

# === INIT CLIENT ===
def initialize_upstox_client(access_token: str):
    if not access_token:
        logger.error("Access token is missing.")
        return None
    try:
        configuration = upstox_client.Configuration()
        configuration.access_token = access_token
        client = upstox_client.ApiClient(configuration)

        user_api = upstox_client.UserApi(client)
        user_profile = user_api.get_profile(api_version="v2")
        logger.info(f"Token validated for {user_profile.data.get('user_name')}")
        return {
            "client": client,
            "access_token": access_token,
            "user_api": user_api,
            "options_api": upstox_client.OptionsApi(client),
            "portfolio_api": upstox_client.PortfolioApi(client),
            "order_api": upstox_client.OrderApi(client),
        }
    except ApiException as e:
        logger.error(f"Token validation failed: {e.body}")
        return None

# === FETCH VIX ===
def fetch_vix(access_token):
    try:
        url = f"{base_url}/market-quote/quotes"
        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
        res = requests.get(url, headers=headers, params={"instrument_key": "NSE_INDEX|India VIX"})
        vix = res.json().get("data", {}).get("NSE_INDEX|India VIX", {}).get("last_price")
        return float(vix) if vix else None
    except Exception as e:
        logger.error(f"Error fetching VIX: {e}")
        return None

# === FETCH OPTION CHAIN ===
def get_nearest_expiry(options_api):
    try:
        contracts = options_api.get_option_contracts(instrument_key="NSE_INDEX|Nifty 50").to_dict().get("data", [])
        expiry_dates = sorted({datetime.strptime(c["expiry"], "%Y-%m-%d") for c in contracts})
        today = datetime.now()
        return next((e.strftime("%Y-%m-%d") for e in expiry_dates if e >= today), None)
    except Exception as e:
        logger.error(f"Failed to get nearest expiry: {e}")
        return None

def fetch_option_chain(options_api, expiry):
    try:
        chain = options_api.get_put_call_option_chain(instrument_key="NSE_INDEX|Nifty 50", expiry_date=expiry)
        return chain.to_dict().get("data", [])
    except Exception as e:
        logger.error(f"Error fetching option chain: {e}")
        return []

# === PROCESS CHAIN + METRICS ===
def process_chain(chain):
    rows, ce_oi, pe_oi = [], 0, 0
    for item in chain:
        ce, pe = item.get("call_options", {}), item.get("put_options", {})
        ce_md, pe_md = ce.get("market_data", {}), pe.get("market_data", {})
        ce_gk, pe_gk = ce.get("option_greeks", {}), pe.get("option_greeks", {})
        strike = item["strike_price"]
        ce_oi_val, pe_oi_val = ce_md.get("oi", 0), pe_md.get("oi", 0)
        row = {
            "Strike": strike,
            "CE_LTP": ce_md.get("ltp"),
            "CE_IV": ce_gk.get("iv"),
            "CE_OI": ce_oi_val,
            "CE_Volume": ce_md.get("volume", 0),
            "PE_LTP": pe_md.get("ltp"),
            "PE_IV": pe_gk.get("iv"),
            "PE_OI": pe_oi_val,
            "PE_Volume": pe_md.get("volume", 0),
            "Strike_PCR": pe_oi_val / ce_oi_val if ce_oi_val else 0,
            "CE_Token": ce.get("instrument_key"),
            "PE_Token": pe.get("instrument_key"),
        }
        ce_oi += ce_oi_val
        pe_oi += pe_oi_val
        rows.append(row)
    return pd.DataFrame(rows), ce_oi, pe_oi

def calculate_metrics(df, ce_oi, pe_oi, spot):
    atm_strike = df.iloc[(df["Strike"] - spot).abs().argsort()[:1]]["Strike"].values[0]
    pcr = round(pe_oi / ce_oi, 2) if ce_oi else 0
    straddle = df[df["Strike"] == atm_strike]
    straddle_price = float(straddle["CE_LTP"].values[0] + straddle["PE_LTP"].values[0])
    max_pain = df["Strike"][((df["Strike"] - df["Strike"].mean()) ** 2).argsort().iloc[0]]
    return pcr, max_pain, straddle_price, atm_strike

# === FETCH MARKET DEPTH ===
def fetch_market_depth_by_scrip(access_token, token):
    try:
        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
        url = f"{base_url}/market-quote/depth"
        res = requests.get(url, headers=headers, params={"instrument_key": token})
        depth = res.json().get("data", {}).get(token, {}).get("depth", {})
        return {
            "bid_volume": sum(d.get("quantity", 0) for d in depth.get("buy", [])),
            "ask_volume": sum(d.get("quantity", 0) for d in depth.get("sell", []))
        }
    except Exception as e:
        logger.error(f"Depth fetch error: {e}")
        return {"bid_volume": 0, "ask_volume": 0}

# === MAIN FUNCTION ===
def fetch_real_time_market_data(upstox_client):
    options_api = upstox_client["options_api"]
    access_token = upstox_client["access_token"]

    vix = fetch_vix(access_token)
    expiry = get_nearest_expiry(options_api)
    if not expiry:
        return {}

    chain = fetch_option_chain(options_api, expiry)
    if not chain:
        return {}

    spot = chain[0].get("underlying_spot_price")
    df, ce_oi, pe_oi = process_chain(chain)
    pcr, max_pain, straddle_price, atm_strike = calculate_metrics(df, ce_oi, pe_oi, spot)

    ce_depth = fetch_market_depth_by_scrip(access_token, df[df["Strike"] == atm_strike]["CE_Token"].values[0])
    pe_depth = fetch_market_depth_by_scrip(access_token, df[df["Strike"] == atm_strike]["PE_Token"].values[0])

    return {
        "nifty_spot": spot,
        "vix": vix,
        "pcr": pcr,
        "max_pain_strike": max_pain,
        "straddle_price": straddle_price,
        "atm_strike": atm_strike,
        "expiry": expiry,
        "option_chain": df,
        "ce_depth": ce_depth,
        "pe_depth": pe_depth,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
