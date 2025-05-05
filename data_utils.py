import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import requests
import io
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Constants
STRIKE_STEP = 50  # Nifty strike multiples
RISK_FREE_RATE = 0.06
FALLBACK_DATA_URLS = {
    "nifty": "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv",
    "vix": "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"
}

def get_next_expiry() -> int:
    """Calculate next Thursday expiry timestamp in 5paisa format"""
    now = datetime.now()
    days_to_thursday = (3 - now.weekday()) % 7  # 0=Monday, 3=Thursday
    if days_to_thursday == 0 and now.hour >= 15:  # After market close on Thursday
        days_to_thursday = 7
    expiry_date = (now + timedelta(days=days_to_thursday)).replace(hour=15, minute=30, second=0, microsecond=0)
    return int(expiry_date.timestamp() * 1000)

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes for call options with input validation"""
    try:
        T = max(T, 1/252)  # Minimum 1 day
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    except Exception as e:
        logger.warning(f"Black-Scholes error: {e} (S={S}, K={K}, T={T})")
        return 0.0

def implied_volatility(S: float, K: float, T: float, r: float, market_price: float, option_type: str = 'call', tol: float = 1e-5, max_iter: int = 100) -> float:
    """Calculate implied volatility using Newton-Raphson"""
    sigma = 0.2
    for _ in range(max_iter):
        if option_type == 'call':
            model_price = black_scholes_call(S, K, T, r, sigma)
        else:
            model_price = black_scholes_call(S, K, T, r, sigma) + K * np.exp(-r * T) - S
        diff = model_price - market_price
        if abs(diff) < tol:
            return sigma * 100
        vega = S * norm.pdf((np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))) * np.sqrt(T)
        if vega == 0:
            return np.nan
        sigma -= diff / vega
    return np.nan

def max_pain(df: pd.DataFrame, nifty_spot: float) -> tuple[float, float]:
    """Calculate max pain strike and difference percentage"""
    calls = df[df["CPType"] == "CE"].set_index("StrikeRate")["LastRate"]
    puts = df[df["CPType"] == "PE"].set_index("StrikeRate")["LastRate"]
    strikes = df["StrikeRate"].unique()
    pain = []
    for K in strikes:
        total_loss = 0
        for s in strikes:
            if s in calls:
                total_loss += max(0, s - K) * calls.get(s, 0)
            if s in puts:
                total_loss += max(0, K - s) * puts.get(s, 0)
        pain.append((K, total_loss))
    max_pain_strike = min(pain, key=lambda x: x[1])[0]
    max_pain_diff_pct = abs(nifty_spot - max_pain_strike) / nifty_spot * 100
    return max_pain_strike, max_pain_diff_pct

def fetch_nifty_data(client) -> dict | None:
    """Fetch live Nifty data from 5paisa with robust error handling"""
    try:
        nifty_req = [{"Exch": "N", "ExchType": "C", "ScripCode": 999920000, "Symbol": "NIFTY"}]
        nifty_data = client.fetch_market_feed(nifty_req)
        if not nifty_data or "Data" not in nifty_data or not nifty_data["Data"]:
            raise Exception("Failed to fetch Nifty 50 index price")
        nifty_spot = nifty_data["Data"][0].get("LastRate", nifty_data["Data"][0].get("LastTradedPrice", 0))
        if not nifty_spot or nifty_spot <= 0:
            raise Exception("Invalid Nifty price")

        expiry_timestamp = get_next_expiry()
        option_chain = client.get_option_chain("N", "NIFTY", expiry_timestamp)
        if not option_chain or "Options" not in option_chain:
            raise Exception("Failed to fetch option chain")

        df = pd.DataFrame(option_chain["Options"])
        required_cols = ["StrikeRate", "CPType", "LastRate", "OpenInterest"]
        if not all(col in df.columns for col in required_cols):
            raise Exception(f"Required columns missing: {required_cols}")
        if df["LastRate"].min() < 0:
            raise Exception("Negative prices detected in option chain")

        df["StrikeRate"] = df["StrikeRate"].astype(float)
        atm_strike = df["StrikeRate"].iloc[(df["StrikeRate"] - nifty_spot).abs().argmin()]
        atm_data = df[df["StrikeRate"] == atm_strike]
        if atm_data.empty:
            raise ValueError(f"No ATM strikes found near {atm_strike}")
        atm_call = atm_data[atm_data["CPType"] == "CE"]["LastRate"].iloc[0] if not atm_data[atm_data["CPType"] == "CE"].empty else 0
        atm_put = atm_data[atm_data["CPType"] == "PE"]["LastRate"].iloc[0] if not atm_data[atm_data["CPType"] == "PE"].empty else 0
        straddle_price = atm_call + atm_put

        iv_df = df[(df["StrikeRate"] >= atm_strike - 100) & (df["StrikeRate"] <= atm_strike + 100)].copy()
        T = (datetime.fromtimestamp(expiry_timestamp/1000) - datetime.now()).days / 365.0
        r = RISK_FREE_RATE
        iv_df["IV (%)"] = iv_df.apply(
            lambda row: implied_volatility(
                S=nifty_spot, K=row["StrikeRate"], T=max(T, 0.002), r=r, market_price=row["LastRate"],
                option_type='call' if row["CPType"] == "CE" else 'put'
            ), axis=1
        )

        calls = df[df["CPType"] == "CE"]
        puts = df[df["CPType"] == "PE"]
        call_oi = calls["OpenInterest"].sum()
        put_oi = puts["OpenInterest"].sum()
        pcr = put_oi / call_oi if call_oi != 0 else float("inf")

        max_pain_strike, max_pain_diff_pct = max_pain(df, nifty_spot)

        atm_iv = iv_df[iv_df["StrikeRate"] == atm_strike]["IV (%)"].mean()
        vix_change_pct = 0
        if "iv_history" not in st.session_state:
            st.session_state.iv_history = pd.DataFrame(columns=["Date", "ATM_IV"])
        prev_atm_iv = st.session_state.iv_history["ATM_IV"].iloc[-1] if not st.session_state.iv_history.empty else atm_iv
        vix_change_pct = ((atm_iv - prev_atm_iv) / prev_atm_iv * 100) if prev_atm_iv != 0 else 0
        st.session_state.iv_history = pd.concat([
            st.session_state.iv_history,
            pd.DataFrame({"Date": [datetime.now()], "ATM_IV": [atm_iv]})
        ], ignore_index=True)

        return {
            "nifty_spot": nifty_spot,
            "atm_strike": atm_strike,
            "straddle_price": straddle_price,
            "pcr": pcr,
            "max_pain_strike": max_pain_strike,
            "max_pain_diff_pct": max_pain_diff_pct,
            "vix_change_pct": vix_change_pct,
            "atm_iv": atm_iv,
            "option_chain": df
        }
    except Exception as e:
        logger.error(f"Error fetching 5paisa data: {str(e)}")
        return None

@st.cache_data(ttl=300)
def load_data() -> tuple[pd.DataFrame | None, dict | None]:
    """Load live and historical data with fallback"""
    try:
        real_data = fetch_nifty_data(st.session_state.client)
        if real_data is None:
            logger.warning("Falling back to GitHub CSV")
            nifty_url = FALLBACK_DATA_URLS["nifty"]
            response = requests.get(nifty_url)
            response.raise_for_status()
            nifty = pd.read_csv(io.StringIO(response.text), encoding="utf-8-sig")
            nifty.columns = nifty.columns.str.strip()
            nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
            nifty = nifty.dropna(subset=["Date"])
            nifty = nifty[["Date", "Close"]].set_index("Date")
            nifty.index = pd.to_datetime(nifty.index)
            nifty = nifty.rename(columns={"Close": "NIFTY_Close"})
            nifty_series = nifty["NIFTY_Close"].squeeze()

            vix_url = FALLBACK_DATA_URLS["vix"]
            response = requests.get(vix_url)
            response.raise_for_status()
            vix = pd.read_csv(io.StringIO(response.text), encoding="utf-8-sig")
            vix.columns = vix.columns.str.strip()
            vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")
            vix = vix.dropna(subset=["Date"])
            vix = vix[["Date", "Close"]].set_index("Date").rename(columns={"Close": "VIX"})
            vix.index = pd.to_datetime(vix.index)
            vix_series = vix["VIX"].squeeze()

            common_dates = nifty_series.index.intersection(vix_series.index)
            df = pd.DataFrame({
                "NIFTY_Close": nifty_series.loc[common_dates],
                "VIX": vix_series.loc[common_dates]
            }, index=common_dates)
            df = df.ffill().bfill()
        else:
            latest_date = datetime.now().date()
            df = pd.DataFrame({
                "NIFTY_Close": [real_data["nifty_spot"]],
                "VIX": [real_data["atm_iv"]]
            }, index=[pd.to_datetime(latest_date)])
            nifty_url = FALLBACK_DATA_URLS["nifty"]
            response = requests.get(nifty_url)
            response.raise_for_status()
            nifty = pd.read_csv(io.StringIO(response.text), encoding="utf-8-sig")
            nifty.columns = nifty.columns.str.strip()
            nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
            nifty = nifty.dropna(subset=["Date"])
            nifty = nifty[["Date", "Close"]].set_index("Date")
            nifty.index = pd.to_datetime(nifty.index)
            nifty = nifty.rename(columns={"Close": "NIFTY_Close"})

            vix_url = FALLBACK_DATA_URLS["vix"]
            response = requests.get(vix_url)
            response.raise_for_status()
            vix = pd.read_csv(io.StringIO(response.text), encoding="utf-8-sig")
            vix.columns = vix.columns.str.strip()
            vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")
            vix = vix.dropna(subset=["Date"])
            vix = vix[["Date", "Close"]].set_index("Date").rename(columns={"Close": "VIX"})
            vix.index = pd.to_datetime(vix.index)

            historical_df = pd.DataFrame({
                "NIFTY_Close": nifty["NIFTY_Close"],
                "VIX": vix["VIX"]
            }).dropna()
            historical_df = historical_df[historical_df.index < pd.to_datetime(latest_date)]
            df = pd.concat([historical_df, df])
            df = df[~df.index.duplicated(keep='last')]
            df = df.sort_index()

        logger.debug("Data loaded successfully.")
        return df, real_data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logger.error(f"Error loading data: {str(e)}")
        return None, None

@st.cache_data
def generate_synthetic_features(df: pd.DataFrame, real_data: dict | None, capital: float) -> pd.DataFrame | None:
    """Generate synthetic features for trading analysis"""
    try:
        n_days = len(df)
        np.random.seed(42)
        risk_free_rate = RISK_FREE_RATE
        strike_step = STRIKE_STEP

        if real_data:
            base_pcr = real_data["pcr"]
            base_iv = real_data["atm_iv"]
            base_straddle_price = real_data["straddle_price"]
            base_max_pain_diff_pct = real_data["max_pain_diff_pct"]
            base_vix_change_pct = real_data["vix_change_pct"]
        else:
            base_pcr = 1.0
            base_iv = 20.0
            base_straddle_price = 200.0
            base_max_pain_diff_pct = 0.5
            base_vix_change_pct = 0.0

        def calculate_days_to_expiry(dates):
            days_to_expiry = []
            for date in dates:
                days_ahead = (3 - date.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7
                next_expiry = date + pd.Timedelta(days=days_ahead)
                dte = (next_expiry - date).days
                days_to_expiry.append(dte)
            return np.array(days_to_expiry)

        df["Days_to_Expiry"] = calculate_days_to_expiry(df.index)
        event_spike = np.where((df.index.month % 3 == 0) & (df.index.day < 5), 1.2, 1.0)
        df["ATM_IV"] = df["VIX"] * (1 + np.random.normal(0, 0.1, n_days)) * event_spike
        df["ATM_IV"] = np.clip(df["ATM_IV"], 5, 50)
        if real_data:
            df["ATM_IV"].iloc[-1] = base_iv

        def dynamic_ivp(x):
            if len(x) >= 5:
                return (np.sum(x.iloc[:-1] <= x.iloc[-1]) / (len(x) - 1)) * 100
            return 50.0
        df["IVP"] = df["ATM_IV"].rolling(252, min_periods=5).apply(dynamic_ivp)
        df["IVP"] = df["IVP"].interpolate().fillna(50.0)

        market_trend = df["NIFTY_Close"].pct_change().rolling(5).mean().fillna(0)
        df["PCR"] = np.clip(base_pcr + np.random.normal(0, 0.1, n_days) + market_trend * -10, 0.7, 2.0)
        if real_data:
            df["PCR"].iloc[-1] = base_pcr

        df["VIX_Change_Pct"] = df["VIX"].pct_change().fillna(0) * 100
        if real_data:
            df["VIX_Change_Pct"].iloc[-1] = base_vix_change_pct

        df["Spot_MaxPain_Diff_Pct"] = np.abs(np.random.lognormal(-2, 0.5, n_days))
        df["Spot_MaxPain_Diff_Pct"] = np.clip(df["Spot_MaxPain_Diff_Pct"], 0.1, 1.0)
        if real_data:
            df["Spot_MaxPain_Diff_Pct"].iloc[-1] = base_max_pain_diff_pct

        df["Event_Flag"] = np.where((df.index.month % 3 == 0) & (df.index.day < 5) | (df["Days_to_Expiry"] <= 3), 1, 0)
        fii_trend = np.random.normal(0, 10000, n_days)
        fii_trend[::30] *= -1
        df["FII_Index_Fut_Pos"] = np.cumsum(fii_trend).astype(int)
        df["FII_Option_Pos"] = np.cumsum(np.random.normal(0, 5000, n_days)).astype(int)
        df["IV_Skew"] = np.clip(np.random.normal(0, 0.8, n_days) + (df["VIX"] / 15 - 1) * 3, -3, 3)
        df["Realized_Vol"] = df["NIFTY_Close"].pct_change().rolling(5, min_periods=5).std() * np.sqrt(252) * 100
        df["Realized_Vol"] = df["Realized_Vol"].fillna(df["VIX"])
        df["Realized_Vol"] = np.clip(df["Realized_Vol"], 0, 50)
        df["Advance_Decline_Ratio"] = np.clip(1.0 + np.random.normal(0, 0.2, n_days) + market_trend * 10, 0.5, 2.0)
        df["Capital_Pressure_Index"] = (df["FII_Index_Fut_Pos"] / 3e4 + df["FII_Option_Pos"] / 1e4 + df["PCR"]) / 3
        df["Capital_Pressure_Index"] = np.clip(df["Capital_Pressure_Index"], -2, 2)
        df["Gamma_Bias"] = np.clip(df["IV_Skew"] * (30 - df["Days_to_Expiry"]) / 30, -2, 2)
        df["Total_Capital"] = capital
        df["PnL_Day"] = np.random.normal(0, 5000, n_days) * (1 - df["Event_Flag"] * 0.5)

        straddle_prices = []
        for i in range(n_days):
            S = df["NIFTY_Close"].iloc[i]
            K = round(S / strike_step) * strike_step
            T = df["Days_to_Expiry"].iloc[i] / 365
            sigma = df["ATM_IV"].iloc[i] / 100
            call_price = black_scholes_call(S, K, T, risk_free_rate, sigma)
            put_price = black_scholes_call(S, K, T, risk_free_rate, sigma) + K * np.exp(-risk_free_rate * T) - S
            straddle_price = (call_price + put_price) * (S / 1000)
            straddle_price = np.clip(straddle_price, 50, 400)
            straddle_prices.append(straddle_price)
        df["Straddle_Price"] = straddle_prices
        if real_data:
            df["Straddle_Price"].iloc[-1] = base_straddle_price

        if df.isna().sum().sum() > 0:
            df = df.interpolate().fillna(method='bfill')

        logger.debug("Synthetic features generated successfully.")
        return df
    except Exception as e:
        st.error(f"Error generating features: {str(e)}")
        logger.error(f"Error generating features: {str(e)}")
        return None
