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
        "options_api": upstox_client.OptionsApi(client),
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
    response = options_api.get_option_contracts(instrument_key="NSE_INDEX|Nifty 50")
    contracts = response.to_dict().get("data", [])
    expiry_dates = set()

    for contract in contracts:
        exp = contract.get("expiry")
        if isinstance(exp, str):
            exp = datetime.strptime(exp, "%Y-%m-%d")
        expiry_dates.add(exp)

    expiry_list = sorted(expiry_dates)
    today = datetime.now()
    valid_expiries = [e.strftime("%Y-%m-%d") for e in expiry_list if e >= today]
    return valid_expiries[0]

def fetch_option_chain(options_api, expiry, access_token):
    """Fetch option chain for given expiry using REST API."""
    # Step 1: Get all option contracts for Nifty 50 with the given expiry
    response = options_api.get_option_contracts(instrument_key="NSE_INDEX|Nifty 50")
    contracts = response.to_dict().get("data", [])
    
    # Filter contracts by expiry and collect instrument keys
    instrument_keys = []
    for contract in contracts:
        contract_expiry = contract.get("expiry")
        if isinstance(contract_expiry, str):
            contract_expiry = datetime.strptime(contract_expiry, "%Y-%m-%d").strftime("%Y-%m-%d")
        if contract_expiry == expiry:
            instrument_keys.append(contract["instrument_key"])
    
    # Add Nifty 50 spot price instrument key
    instrument_keys.append("NSE_INDEX|Nifty 50")
    
    # Step 2: Fetch quotes using REST API
    url = f"{base_url}/market-quote/quotes"
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
    res = requests.get(url, headers=headers, params={"instrument_key": ",".join(instrument_keys)})
    raw_data = res.json().get("data", {})
    
    # Step 3: Group by strike price and format into the expected structure
    chain_data = {}
    spot_price = None
    for token, quote in raw_data.items():
        if token == "NSE_INDEX|Nifty 50":
            spot_price = quote.get("last_price")
            continue
        strike = quote.get("strike_price")
        if not strike:
            continue
        if strike not in chain_data:
            chain_data[strike] = {"strike_price": strike}
        option_type = quote.get("option_type")
        quote["instrument_key"] = token
        if option_type == "CE":
            chain_data[strike]["call_options"] = {
                "market_data": {
                    "last_price": quote.get("last_price"),
                    "oi": quote.get("oi", 0),
                    "volume": quote.get("volume", 0)
                },
                "option_greeks": {
                    "implied_volatility": quote.get("iv", 0)
                },
                "instrument_key": token
            }
        elif option_type == "PE":
            chain_data[strike]["put_options"] = {
                "market_data": {
                    "last_price": quote.get("last_price"),
                    "oi": quote.get("oi", 0),
                    "volume": quote.get("volume", 0)
                },
                "option_greeks": {
                    "implied_volatility": quote.get("iv", 0)
                },
                "instrument_key": token
            }
    
    # Convert to list and add spot price
    chain = list(chain_data.values())
    if spot_price and chain:
        chain[0]["underlying_spot_price"] = spot_price
    return chain

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

    chain = fetch_option_chain(options_api, expiry, access_token)

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
