import logging
import pandas as pd
import requests
from datetime import datetime
import upstox_client

# === CONFIG ===
base_url = "https://api.upstox.com/v2"
logger = logging.getLogger(__name__)

# === INIT CLIENT ===
def initialize_upstox_client(access_token: str):
    """Initialize Upstox API client with access token."""
    configuration = upstox_client.Configuration()
    configuration.access_token = access_token
    client = upstox_client.ApiClient(configuration)

    user_api = upstox_client.UserApi(client)
    user_api.get_profile(api_version="2")  # Call to validate token
    
    logger.info("Token validated for user.")
    return {
        "client": client,
        "access_token": access_token,
        "user_api": user_api,
        "options_api": upstox_client.OptionApi(client),
        "portfolio_api": upstox_client.PortfolioApi(client),
        "order_api": upstox_client.OrderApi(client),
    }

# === FETCH VIX ===
def fetch_vix(access_token):
    """Fetch India VIX value."""
    url = f"{base_url}/market-quote/quotes"
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
    res = requests.get(url, headers=headers, params={"instrument_key": "NSE_INDEX|India VIX"})
    data = res.json().get("data", {}).get("NSE_INDEX|India VIX", {})
    vix = data.get("last_price")
    return float(vix) if vix else None

# === FETCH OPTION CHAIN ===
def get_nearest_expiry(options_api):
    """Get the nearest expiry date for NIFTY options."""
    contracts = options_api.get_option_contracts(instrument_key="NSE_INDEX|Nifty 50", api_version="2").data
    expiry_dates = sorted({datetime.strptime(c.expiry_date, "%Y-%m-%d") for c in contracts})
    today = datetime.now()
    return next((e.strftime("%Y-%m-%d") for e in expiry_dates if e >= today), None)

def fetch_option_chain(options_api, expiry):
    """Fetch option chain for given expiry."""
    chain = options_api.get_full_market_quote(
        instrument_key=f"NSE_INDEX|Nifty 50&expiry={expiry}",
        api_version="2"
    )
    return chain.data if chain and chain.data else []

# === PROCESS CHAIN + METRICS ===
def process_chain(chain):
    """Process option chain data into a DataFrame."""
    rows, ce_oi, pe_oi = [], 0, 0
    for item in chain:
        ce, pe = item.get("call_options", {}), item.get("put_options", {})
        ce_md, pe_md = ce.get("market_data", {}), pe.get("market_data", {})
        ce_gk, pe_gk = ce.get("option_greeks", {}), pe.get("option_greeks", {})
        strike = item.get("strike_price", 0)
        ce_oi_val, pe_oi_val = ce_md.get("oi", 0), pe_md.get("oi", 0)
        row = {
            "Strike": strike,
            "CE_LTP": ce_md.get("last_price"),
            "CE_IV": ce_gk.get("implied_volatility"),
            "CE_OI": ce_oi_val,
            "CE_Volume": ce_md.get("volume", 0),
            "PE_LTP": pe_md.get("last_price"),
            "PE_IV": pe_gk.get("implied_volatility"),
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
    """Calculate key option metrics like PCR, max pain, and straddle price."""
    atm_strike = df.iloc[(df["Strike"] - spot).abs().argsort()[:1]]["Strike"].values[0]
    pcr = round(pe_oi / ce_oi, 2) if ce_oi else 0
    straddle = df[df["Strike"] == atm_strike]
    straddle_price = float(straddle["CE_LTP"].values[0] + straddle["PE_LTP"].values[0]) if not straddle.empty else 0
    max_pain = df["Strike"][((df["Strike"] - df["Strike"].mean()) ** 2).argsort().iloc[0]] if not df.empty else None
    return pcr, max_pain, straddle_price, atm_strike

# === FETCH MARKET DEPTH ===
def fetch_market_depth_by_scrip(access_token, token):
    """Fetch market depth for a given instrument."""
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
    url = f"{base_url}/market-quote/depth"
    res = requests.get(url, headers=headers, params={"instrument_key": token})
    depth = res.json().get("data", {}).get(token, {}).get("depth", {})
    return {
        "bid_volume": sum(d.get("quantity", 0) for d in depth.get("buy", [])),
        "ask_volume": sum(d.get("quantity", 0) for d in depth.get("sell", []))
    }

# === REAL-TIME MARKET SNAPSHOT ===
def fetch_real_time_market_data(upstox_client):
    """Fetch real-time market data snapshot."""
    options_api = upstox_client["options_api"]
    access_token = upstox_client["access_token"]

    vix = fetch_vix(access_token)
    expiry = get_nearest_expiry(options_api)

    chain = fetch_option_chain(options_api, expiry)

    spot = chain[0].get("underlying_spot_price") if chain else None
    df, ce_oi, pe_oi = process_chain(chain)
    pcr, max_pain, straddle_price, atm_strike = calculate_metrics(df, ce_oi, pe_oi, spot)

    ce_depth = fetch_market_depth_by_scrip(access_token, df[df["Strike"] == atm_strike]["CE_Token"].values[0]) if not df.empty else {"bid_volume": 0, "ask_volume": 0}
    pe_depth = fetch_market_depth_by_scrip(access_token, df[df["Strike"] == atm_strike]["PE_Token"].values[0]) if not df.empty else {"bid_volume": 0, "ask_volume": 0}

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

# === PORTFOLIO DATA ===
def fetch_all_api_portfolio_data(upstox_client):
    """Fetch all portfolio-related data."""
    portfolio_api = upstox_client["portfolio_api"]
    order_api = upstox_client["order_api"]
    user_api = upstox_client["user_api"]

    data = {}
    data['margin'] = user_api.get_user_fund_margin(api_version="2").to_dict()
    data['holdings'] = portfolio_api.get_holdings(api_version="2").to_dict()
    data['positions'] = portfolio_api.get_positions(api_version="2").to_dict()
    data['orders'] = order_api.get_order_book(api_version="2").to_dict()
    data['trades'] = order_api.get_trade_book(api_version="2").to_dict()

    return data

# === ORDER MANAGEMENT ===
def prepare_trade_orders(strategy):
    """Placeholder: Prepare trade orders based on strategy."""
    return strategy.get("Orders", [])

def execute_trade_orders(upstox_client, orders):
    """Execute trade orders via Upstox API."""
    order_api = upstox_client["order_api"]
    results = []
    for order in orders:
        res = order_api.place_order(body=order, api_version="2")
        results.append({"status": "success", "order_id": res.data.get("order_id")})
    return len([r for r in results if r["status"] == "success"]) > 0, results

def square_off_positions(upstox_client):
    """Square off all open positions."""
    portfolio_api = upstox_client["portfolio_api"]
    positions = portfolio_api.get_positions(api_version="2").to_dict().get("data", [])
    closed = 0
    for pos in positions:
        if pos.get("quantity", 0) != 0:
            closed += 1  # Placeholder for actual square-off logic
    return closed > 0
