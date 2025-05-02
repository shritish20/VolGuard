import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import norm

# --- Black-Scholes IV Calculation ---
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_volatility(S, K, T, r, market_price, option_type='call', tol=1e-5, max_iter=100):
    sigma = 0.2
    for _ in range(max_iter):
        if option_type == 'call':
            model_price = black_scholes_call(S, K, T, r, sigma)
        else:
            model_price = black_scholes_call(S, K, T, r, sigma) + K * np.exp(-r * T) - S
        diff = model_price - market_price
        if abs(diff) < tol:
            return sigma * 100
        vega = S * norm.pdf((np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))) * np.sqrt(T)
        if vega == 0:
            return np.nan
        sigma -= diff / vega
    return np.nan

# --- Max Pain Calculation ---
def max_pain(df, nifty_spot):
    calls = df[df["CPType"] == "CE"].set_index("StrikeRate")["LastRate"]
    puts = df[df["CPType"] == "PE"].set_index("StrikeRate")["LastRate"]
    strikes = df["StrikeRate"].unique()
    pain = []
    for K in strikes:
        total_loss = 0
        for s in strikes:
            total_loss += max(0, s - K) * calls.get(s, 0)
            total_loss += max(0, K - s) * puts.get(s, 0)
        pain.append((K, total_loss))
    return min(pain, key=lambda x: x[1])[0]

# --- Main Live Fetch Function ---
def get_volguard_live_data(client):
    # Get Nifty Index Spot
    nifty_req = [{
        "Exch": "N",
        "ExchType": "C",
        "ScripCode": 999920000,
        "Symbol": "NIFTY"
    }]
    feed = client.fetch_market_feed(nifty_req)
    nifty_spot = feed["Data"][0].get("LastRate", 0)

    # Fetch Option Chain
    expiry_timestamp = 1746694800000  # update manually
    option_chain = client.get_option_chain("N", "NIFTY", expiry_timestamp)
    df = pd.DataFrame(option_chain["Options"])
    df["StrikeRate"] = df["StrikeRate"].astype(float)

    # Calculate Straddle, PCR
    atm_strike = df["StrikeRate"].iloc[(df["StrikeRate"] - nifty_spot).abs().argmin()]
    atm_data = df[df["StrikeRate"] == atm_strike]
    atm_call = atm_data[atm_data["CPType"] == "CE"]["LastRate"].values[0]
    atm_put = atm_data[atm_data["CPType"] == "PE"]["LastRate"].values[0]
    straddle_price = atm_call + atm_put

    calls = df[df["CPType"] == "CE"]
    puts = df[df["CPType"] == "PE"]
    pcr = puts["OpenInterest"].sum() / calls["OpenInterest"].sum()

    # IV for ±100 Strike Range
    T = (datetime(2025, 5, 8) - datetime.now()).days / 365
    r = 0.06
    iv_df = df[(df["StrikeRate"] >= atm_strike - 100) & (df["StrikeRate"] <= atm_strike + 100)].copy()
    iv_df["IV (%)"] = iv_df.apply(lambda row:
        implied_volatility(
            nifty_spot, row["StrikeRate"], T, r,
            row["LastRate"], 'call' if row["CPType"] == "CE" else 'put'),
        axis=1)

    atm_iv = iv_df[iv_df["StrikeRate"] == atm_strike]["IV (%)"].mean()

    max_pain_strike = max_pain(df, nifty_spot)
    max_pain_diff = abs(nifty_spot - max_pain_strike) / nifty_spot * 100

    return {
        "nifty_spot": nifty_spot,
        "atm_strike": atm_strike,
        "straddle_price": straddle_price,
        "atm_iv": atm_iv,
        "pcr": pcr,
        "max_pain_strike": max_pain_strike,
        "max_pain_diff_pct": max_pain_diff,
        "iv_df": iv_df
    }
