# main.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import pickle
from datetime import datetime, timedelta
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import upstox_client
from upstox_client.rest import ApiException
import logging
import time
import io

# === Streamlit Configuration ===
st.set_page_config(page_title="VolGuard Pro", layout="wide")
st.title("VolGuard Pro: Nifty 50 Options & Volatility Analysis")
st.markdown("Analyze real-time Nifty 50 options, forecast volatility, and predict market movements.")

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VolGuard")

# === Global OI Storage ===
prev_oi = {}  # Store previous OI for change tracking

# === Session State to Store VolGuard Data ===
if 'volguard_data' not in st.session_state:
    st.session_state.volguard_data = None

# === Helper Functions ===
def get_nearest_expiry(options_api, instrument_key):
    try:
        response = options_api.get_option_contracts(instrument_key=instrument_key)
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
        nearest = valid_expiries[0] if valid_expiries else None
        time.sleep(0.5)  # Rate limiting
        return nearest
    except Exception as e:
        logger.error(f"Expiry fetch failed: {e}")
        return None

def fetch_vix(access_token, base_url):
    try:
        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
        url = f"{base_url}/market-quote/quotes"
        res = requests.get(url, headers=headers, params={"instrument_key": "NSE_INDEX|India VIX"})
        vix = res.json().get('data', {}).get('NSE_INDEX|India VIX', {}).get('last_price')
        time.sleep(0.5)
        return vix
    except Exception as e:
        logger.error(f"VIX fetch error: {e}")
        return None

def fetch_option_chain(options_api, instrument_key, expiry):
    try:
        res = options_api.get_put_call_option_chain(instrument_key=instrument_key, expiry_date=expiry)
        time.sleep(0.5)
        return res.to_dict().get('data', [])
    except Exception as e:
        logger.error(f"Option chain fetch error: {e}")
        return []

def process_chain(data):
    global prev_oi
    rows, ce_oi, pe_oi = [], 0, 0
    for r in data:
        ce = r.get('call_options', {})
        pe = r.get('put_options', {})
        ce_md, pe_md = ce.get('market_data', {}), pe.get('market_data', {})
        ce_gk, pe_gk = ce.get('option_greeks', {}), pe.get('option_greeks', {})
        strike = r['strike_price']
        ce_oi_val = ce_md.get("oi", 0)
        pe_oi_val = pe_md.get("oi", 0)
        ce_oi_change = ce_oi_val - prev_oi.get(f"{strike}_CE", 0)
        pe_oi_change = pe_oi_val - prev_oi.get(f"{strike}_PE", 0)
        ce_oi_change_pct = (ce_oi_change / prev_oi[f"{strike}_CE"] * 100) if prev_oi.get(f"{strike}_CE", 0) else 0
        pe_oi_change_pct = (pe_oi_change / prev_oi[f"{strike}_PE"] * 100) if prev_oi.get(f"{strike}_PE", 0) else 0
        strike_pcr = pe_oi_val / ce_oi_val if ce_oi_val else 0
        row = {
            "Strike": strike,
            "CE_LTP": ce_md.get("ltp"),
            "CE_IV": ce_gk.get("iv"),
            "CE_Delta": ce_gk.get("delta"),
            "CE_Theta": ce_gk.get("theta"),
            "CE_Vega": ce_gk.get("vega"),
            "CE_OI": ce_oi_val,
            "CE_OI_Change": ce_oi_change,
            "CE_OI_Change_Pct": ce_oi_change_pct,
            "CE_Volume": ce_md.get("volume", 0),
            "PE_LTP": pe_md.get("ltp"),
            "PE_IV": pe_gk.get("iv"),
            "PE_Delta": pe_gk.get("delta"),
            "PE_Theta": pe_gk.get("theta"),
            "PE_Vega": ce_gk.get("vega"),
            "PE_OI": pe_oi_val,
            "PE_OI_Change": pe_oi_change,
            "PE_OI_Change_Pct": pe_oi_change_pct,
            "PE_Volume": pe_md.get("volume", 0),
            "Strike_PCR": strike_pcr,
            "CE_Token": ce.get("instrument_key"),
            "PE_Token": pe.get("instrument_key")
        }
        ce_oi += ce_oi_val
        pe_oi += pe_oi_val
        rows.append(row)
        prev_oi[f"{strike}_CE"] = ce_oi_val
        prev_oi[f"{strike}_PE"] = pe_oi_val
    df = pd.DataFrame(rows).sort_values("Strike")
    return df, ce_oi, pe_oi

def calculate_metrics(df, ce_oi_total, pe_oi_total, spot):
    atm = df.iloc[(df['Strike'] - spot).abs().argsort()[:1]]
    atm_strike = atm['Strike'].values[0]
    pcr = round(pe_oi_total / ce_oi_total, 2) if ce_oi_total else 0
    max_pain_series = df.apply(lambda x: sum(df['CE_OI'] * abs(df['Strike'] - x['Strike'])) +
                                         sum(df['PE_OI'] * abs(df['Strike'] - x['Strike'])), axis=1)
    strike_with_pain = df.loc[max_pain_series.idxmin(), 'Strike']
    straddle_price = float(atm['CE_LTP'].values[0] + atm['PE_LTP'].values[0])
    return pcr, strike_with_pain, straddle_price, atm_strike

def plot_iv_skew(df, spot, atm_strike):
    valid = df[(df['CE_IV'] > 0) & (df['PE_IV'] > 0)]
    if valid.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(valid['Strike'], valid['CE_IV'], label='Call IV', color='blue')
    ax.plot(valid['Strike'], valid['PE_IV'], label='Put IV', color='red')
    ax.axvline(spot, color='gray', linestyle='--', label='Spot')
    ax.axvline(atm_strike, color='green', linestyle=':', label='ATM')
    ax.set_title("IV Skew")
    ax.set_xlabel("Strike")
    ax.set_ylabel("IV")
    ax.legend()
    ax.grid(True)
    return fig

def get_market_depth(access_token, base_url, token):
    try:
        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
        url = f"{base_url}/market-quote/depth"
        res = requests.get(url, headers=headers, params={"instrument_key": token})
        depth = res.json().get('data', {}).get(token, {}).get('depth', {})
        bid_volume = sum(item.get('quantity', 0) for item in depth.get('buy', []))
        ask_volume = sum(item.get('quantity', 0) for item in depth.get('sell', []))
        time.sleep(0.5)
        return {"bid_volume": bid_volume, "ask_volume": ask_volume}
    except Exception as e:
        logger.error(f"Depth fetch error for {token}: {e}")
        return {}

def calculate_rolling_and_fixed_hv(nifty_close):
    log_returns = np.log(nifty_close.pct_change() + 1).dropna()
    last_7d_std = log_returns[-7:].std()
    rolling_rv_annualized = last_7d_std * np.sqrt(252) * 100
    last_date = nifty_close.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=7)
    rv_7d_df = pd.DataFrame({
        "Date": future_dates,
        "Day": future_dates.day_name(),
        "7-Day Realized Volatility (%)": np.round([rolling_rv_annualized]*7, 2)
    })
    hv_30d = log_returns[-30:].std() * np.sqrt(252) * 100
    hv_1y = log_returns[-252:].std() * np.sqrt(252) * 100
    return rv_7d_df, round(hv_30d, 2), round(hv_1y, 2)

def compute_realized_vol(nifty_df):
    log_returns = np.log(nifty_df["NIFTY_Close"].pct_change() + 1).dropna()
    last_7d_std = log_returns[-7:].std() * np.sqrt(252) * 100
    return last_7d_std if not np.isnan(last_7d_std) else 0

def run_volguard(access_token):
    try:
        configuration = upstox_client.Configuration()
        configuration.access_token = access_token
        client = upstox_client.ApiClient(configuration)
        options_api = upstox_client.OptionsApi(client)
        instrument_key = "NSE_INDEX|Nifty 50"
        base_url = "https://api.upstox.com/v2"

        vix = fetch_vix(access_token, base_url)
        expiry = get_nearest_expiry(options_api, instrument_key)
        if not expiry:
            return None, None, None, None
        chain = fetch_option_chain(options_api, instrument_key, expiry)
        if not chain:
            return None, None, None, None
        spot = chain[0].get("underlying_spot_price")
        if not spot:
            return None, None, None, None

        df, ce_oi, pe_oi = process_chain(chain)
        if df.empty:
            return None, None, None, None
        pcr, max_pain, straddle_price, atm_strike = calculate_metrics(df, ce_oi, pe_oi, spot)
        ce_depth = get_market_depth(access_token, base_url, df[df['Strike'] == atm_strike]['CE_Token'].values[0])
        pe_depth = get_market_depth(access_token, base_url, df[df['Strike'] == atm_strike]['PE_Token'].values[0])
        iv_skew_fig = plot_iv_skew(df, spot, atm_strike)

        result = {
            "nifty_spot": spot,
            "vix": vix,
            "atm_strike": atm_strike,
            "straddle_price": straddle_price,
            "pcr": pcr,
            "max_pain": max_pain,
            "expiry": expiry,
            "iv_skew_data": df.to_dict(),
            "ce_depth": ce_depth,
            "pe_depth": pe_depth,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "atm_iv": df[df['Strike'] == atm_strike]['CE_IV'].values[0] if not df[df['Strike'] == atm_strike].empty else 0
        }
        return result, df, iv_skew_fig, atm_strike
    except Exception as e:
        logger.error(f"Volguard run error: {e}")
        return None, None, None, None

# === Streamlit Tabs ===
tab1, tab2, tab3, tab4 = st.tabs(["VolGuard: Options Analysis", "GARCH: Volatility Forecast", "XGBoost: Volatility Prediction", "Dashboard: Volatility Comparison"])

# === Tab 1: VolGuard ===
with tab1:
    st.header("VolGuard: Real-Time Options Analysis")
    st.warning("Upstox API is available from 5:30 AM to 12:00 AM IST. Data may not load outside these hours.")
    access_token = st.text_input("Enter Upstox Access Token", type="password")
    
    if st.button("Run VolGuard"):
        if not access_token:
            st.error("Please enter a valid Upstox access token.")
        else:
            with st.spinner("Fetching options data..."):
                result, df, iv_skew_fig, atm_strike = run_volguard(access_token)
                if result:
                    st.session_state.volguard_data = result  # Store for XGBoost and Dashboard
                    st.success("Data fetched successfully!")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Market Snapshot")
                        st.write(f"**Timestamp**: {result['timestamp']}")
                        st.write(f"**Nifty Spot**: {result['nifty_spot']}")
                        st.write(f"**India VIX**: {result['vix']}")
                        st.write(f"**Expiry**: {result['expiry']}")
                        st.write(f"**ATM Strike**: {result['atm_strike']}")
                        st.write(f"**Straddle Price**: {result['straddle_price']}")
                        st.write(f"**PCR**: {result['pcr']}")
                        st.write(f"**Max Pain**: {result['max_pain']}")
                        st.write(f"**CE Depth**: Bid Volume={result['ce_depth'].get('bid_volume', 0)}, Ask Volume={result['ce_depth'].get('ask_volume', 0)}")
                        st.write(f"**PE Depth**: Bid Volume={result['pe_depth'].get('bid_volume', 0)}, Ask Volume={result['pe_depth'].get('ask_volume', 0)}")
                    with col2:
                        if iv_skew_fig:
                            st.subheader("IV Skew Plot")
                            st.pyplot(iv_skew_fig)
                    
                    st.subheader("Key Strikes (ATM ± 6)")
                    atm_idx = df[df['Strike'] == atm_strike].index[0]
                    key_strikes = df.iloc[max(0, atm_idx-6):atm_idx+7][[
                        'Strike', 'CE_LTP', 'CE_IV', 'CE_Delta', 'CE_Theta', 'CE_Vega', 'CE_OI', 
                        'CE_OI_Change', 'CE_OI_Change_Pct', 'CE_Volume', 'PE_LTP', 'PE_IV', 'PE_Delta', 
                        'PE_Theta', 'PE_Vega', 'PE_OI', 'PE_OI_Change', 'PE_OI_Change_Pct', 'PE_Volume', 
                        'Strike_PCR'
                    ]]
                    key_strikes['CE_OI_Change'] = key_strikes['CE_OI_Change'].apply(
                        lambda x: f"{x:.1f}*" if x > 500000 else f"{x:.1f}"
                    )
                    key_strikes['PE_OI_Change'] = key_strikes['PE_OI_Change'].apply(
                        lambda x: f"{x:.1f}*" if x > 500000 else f"{x:.1f}"
                    )
                    st.dataframe(key_strikes, use_container_width=True)
                else:
                    st.error("Failed to fetch options data. Check your access token or API availability.")

# === Tab 2: GARCH ===
with tab2:
    st.header("GARCH: 7-Day Volatility Forecast")
    try:
        nifty_df = pd.read_csv("https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv")
        nifty_df.columns = nifty_df.columns.str.strip()
        nifty_df["Date"] = pd.to_datetime(nifty_df["Date"], format="%d-%b-%Y", errors="coerce")
        nifty_df = nifty_df.dropna(subset=["Date"]).set_index("Date")
        nifty_df = nifty_df.rename(columns={"Close": "NIFTY_Close"})
        nifty_df = nifty_df[["NIFTY_Close"]].dropna().sort_index()

        log_returns = np.log(nifty_df["NIFTY_Close"].pct_change() + 1).dropna() * 100
        model = arch_model(log_returns, vol="Garch", p=1, q=1)
        model_fit = model.fit(disp="off")
        forecast_horizon = 7
        garch_forecast = model_fit.forecast(horizon=forecast_horizon)
        garch_vols = np.sqrt(garch_forecast.variance.values[-1]) * np.sqrt(252)
        garch_vols = np.clip(garch_vols, 5, 50)
        forecast_dates = pd.bdate_range(start=datetime(2025, 5, 15), periods=7)  # Start from May 15, 2025
        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Day": forecast_dates.day_name(),
            "Forecasted Volatility (%)": np.round(garch_vols, 2)
        })
        st.subheader("GARCH Volatility Forecast")
        st.dataframe(forecast_df, use_container_width=True)

        rv_7d_df, hv_30d, hv_1y = calculate_rolling_and_fixed_hv(nifty_df["NIFTY_Close"])
        st.subheader("Historical Volatility")
        st.write(f"**30-Day HV (Annualized)**: {hv_30d}%")
        st.write(f"**1-Year HV (Annualized)**: {hv_1y}%")
        st.subheader("7-Day Realized Volatility Forecast")
        st.dataframe(rv_7d_df, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading GARCH data: {e}")

# === Tab 3: XGBoost ===
with tab3:
    st.header("XGBoost: 5-Day Volatility Prediction")
    xgb_model_url = "https://drive.google.com/uc?export=download&id=1Gs86p1p8wsGe1lp498KC-OVn0e87Gv-R"
    xgb_csv_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/synthetic_volguard_dataset.csv"

    # Model Evaluation Section
    st.subheader("Evaluate Trained Model")
    if st.button("Run Model Evaluation"):
        try:
            with st.spinner("Loading XGBoost data and model..."):
                xgb_df = pd.read_csv(xgb_csv_url)
                xgb_df = xgb_df.dropna()
                features = ['ATM_IV', 'Realized_Vol', 'IVP', 'Event_Impact_Score', 'FII_DII_Net_Long', 'PCR', 'VIX']
                target = 'Next_5D_Realized_Vol'
                if not all(col in xgb_df.columns for col in features + [target]):
                    st.error("CSV missing required columns!")
                    st.stop()
                X = xgb_df[features]
                y = xgb_df[target] * 100  # Convert to percentage for consistency

                response = requests.get(xgb_model_url)
                if response.status_code != 200:
                    st.error("Failed to load xgb_model.pkl from Google Drive.")
                    st.stop()
                xgb_model = pickle.loads(response.content)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                y_pred_train = xgb_model.predict(X_train)
                y_pred_test = xgb_model.predict(X_test)

                rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
                rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
                mae_train = mean_absolute_error(y_train, y_pred_train)
                mae_test = mean_absolute_error(y_test, y_pred_test)
                r2_train = r2_score(y_train, y_pred_train)
                r2_test = r2_score(y_test, y_pred_test)

                st.subheader("Model Performance")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Training Metrics**")
                    st.write(f"RMSE: {rmse_train:.4f}%")
                    st.write(f"MAE: {mae_train:.4f}%")
                    st.write(f"R²: {r2_train:.4f}")
                with col2:
                    st.write("**Test Metrics**")
                    st.write(f"RMSE: {rmse_test:.4f}%")
                    st.write(f"MAE: {mae_test:.4f}%")
                    st.write(f"R²: {r2_test:.4f}")

                fig, ax = plt.subplots(figsize=(10, 6))
                xgb_importances = xgb_model.feature_importances_
                sorted_idx = np.argsort(xgb_importances)
                ax.barh(range(len(features)), xgb_importances[sorted_idx], align='center')
                ax.set_yticks(range(len(features)))
                ax.set_yticklabels(np.array(features)[sorted_idx])
                ax.set_xlabel("Feature Importance")
                ax.set_title("XGBoost Feature Importances")
                st.subheader("Feature Importances")
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Error running model evaluation: {e}")

    # Real-Time Prediction Section
    st.subheader("Predict with New Data")
    st.info("Use VolGuard data (if available) or enter values manually. IVP, Event_Impact_Score, and FII_DII_Net_Long require external data.")
    
    try:
        nifty_df = pd.read_csv("https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv")
        nifty_df.columns = nifty_df.columns.str.strip()
        nifty_df["Date"] = pd.to_datetime(nifty_df["Date"], format="%d-%b-%Y", errors="coerce")
        nifty_df = nifty_df.dropna(subset=["Date"]).set_index("Date")
        nifty_df = nifty_df.rename(columns={"Close": "NIFTY_Close"})
        nifty_df = nifty_df[["NIFTY_Close"]].dropna().sort_index()
        realized_vol = compute_realized_vol(nifty_df)
    except Exception as e:
        realized_vol = 0
        st.warning(f"Could not compute Realized Volatility: {e}")

    # Auto-fill from VolGuard if available
    atm_iv = st.session_state.volguard_data['atm_iv'] * 100 if st.session_state.volguard_data and st.session_state.volguard_data['atm_iv'] else 0  # Convert to %
    pcr = st.session_state.volguard_data['pcr'] if st.session_state.volguard_data else 0
    vix = st.session_state.volguard_data['vix'] if st.session_state.volguard_data else 0

    col1, col2 = st.columns(2)
    with col1:
        atm_iv_input = st.number_input("ATM IV (%)", value=float(atm_iv), min_value=0.0, step=0.1)
        realized_vol_input = st.number_input("Realized Volatility (%)", value=float(realized_vol), min_value=0.0, step=0.1)
        ivp_input = st.number_input("IV Percentile (0–100)", value=50.0, min_value=0.0, max_value=100.0, step=1.0)
    with col2:
        event_score_input = st.number_input("Event Impact Score (0–2)", value=1.0, min_value=0.0, max_value=2.0, step=1.0)
        fii_dii_input = st.number_input("FII/DII Net Long (₹ Cr)", value=0.0, step=100.0)
        pcr_input = st.number_input("Put-Call Ratio", value=float(pcr), min_value=0.0, step=0.01)
        vix_input = st.number_input("India VIX (%)", value=float(vix), min_value=0.0, step=0.1)

    if st.button("Predict Volatility"):
        try:
            with st.spinner("Loading model and predicting..."):
                response = requests.get(xgb_model_url)
                if response.status_code != 200:
                    st.error("Failed to load xgb_model.pkl from Google Drive.")
                    st.stop()
                xgb_model = pickle.loads(response.content)

                # Create feature DataFrame
                new_data = pd.DataFrame({
                    'ATM_IV': [atm_iv_input],
                    'Realized_Vol': [realized_vol_input],
                    'IVP': [ivp_input],
                    'Event_Impact_Score': [event_score_input],
                    'FII_DII_Net_Long': [fii_dii_input],
                    'PCR': [pcr_input],
                    'VIX': [vix_input]
                })

                # Predict and convert to percentage
                prediction = xgb_model.predict(new_data)[0]
                st.session_state.xgb_prediction = prediction  # Store for Dashboard
                st.success(f"Predicted Next 5-Day Realized Volatility: {prediction:.2f}%")
        except Exception as e:
            st.error(f"Error predicting volatility: {e}")

# === Tab 4: Dashboard ===
with tab4:
    st.header("Dashboard: Volatility Comparison")
    st.info("Compares XGBoost, GARCH, Realized Volatility, and ATM IV. Metrics include IV-RV, PCR, VIX, and more.")

    try:
        # Load Nifty data for Realized Volatility
        nifty_df = pd.read_csv("https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv")
        nifty_df.columns = nifty_df.columns.str.strip()
        nifty_df["Date"] = pd.to_datetime(nifty_df["Date"], format="%d-%b-%Y", errors="coerce")
        nifty_df = nifty_df.dropna(subset=["Date"]).set_index("Date")
        nifty_df = nifty_df.rename(columns={"Close": "NIFTY_Close"})
        nifty_df = nifty_df[["NIFTY_Close"]].dropna().sort_index()
        realized_vol = compute_realized_vol(nifty_df)

        # GARCH Forecast
        log_returns = np.log(nifty_df["NIFTY_Close"].pct_change() + 1).dropna() * 100
        model = arch_model(log_returns, vol="Garch", p=1, q=1)
        model_fit = model.fit(disp="off")
        garch_forecast = model_fit.forecast(horizon=7)
        garch_vols = np.sqrt(garch_forecast.variance.values[-1]) * np.sqrt(252)
        garch_vols = np.clip(garch_vols, 5, 50)

        # XGBoost Prediction (from session state or default)
        xgb_vol = st.session_state.get('xgb_prediction', 15.0)  # Default 15% if not predicted
        xgb_vols = [xgb_vol] * 7  # Extend to 7 days for plotting

        # ATM IV and Metrics from VolGuard
        atm_iv = st.session_state.volguard_data['atm_iv'] * 100 if st.session_state.volguard_data and st.session_state.volguard_data['atm_iv'] else 20.0  # Default 20%
        atm_iv_vols = [atm_iv] * 7  # Extend to 7 days
        pcr = st.session_state.volguard_data['pcr'] if st.session_state.volguard_data else 1.0
        vix = st.session_state.volguard_data['vix'] if st.session_state.volguard_data else 15.0
        straddle_price = st.session_state.volguard_data['straddle_price'] if st.session_state.volguard_data else 0
        max_pain = st.session_state.volguard_data['max_pain'] if st.session_state.volguard_data else 0
        iv_rv = atm_iv - realized_vol

        # Realized Volatility
        rv_vols = [realized_vol] * 7

        # Plot
        dates = pd.bdate_range(start=datetime(2025, 5, 15), periods=7)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates, garch_vols, label='GARCH Forecast', color='blue', marker='o')
        ax.plot(dates, xgb_vols, label='XGBoost Prediction', color='green', marker='s')
        ax.plot(dates, rv_vols, label='Realized Volatility', color='red', marker='^')
        ax.plot(dates, atm_iv_vols, label='ATM IV', color='purple', marker='d')
        ax.set_title("Volatility Comparison (May 15–21, 2025)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Volatility (%)")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        st.subheader("Volatility Plot")
        st.pyplot(fig)

        # Metrics Table
        metrics = {
            "Metric": ["IV-RV", "PCR", "VIX", "Straddle Price", "Max Pain", "Realized Volatility", "ATM IV", "XGBoost Volatility", "GARCH Volatility (Day 1)"],
            "Value": [
                f"{iv_rv:.2f}%",
                f"{pcr:.2f}",
                f"{vix:.2f}%",
                f"{straddle_price:.2f}",
                f"{max_pain:.2f}",
                f"{realized_vol:.2f}%",
                f"{atm_iv:.2f}%",
                f"{xgb_vol:.2f}%",
                f"{garch_vols[0]:.2f}%"
            ]
        }
        metrics_df = pd.DataFrame(metrics)
        st.subheader("Key Metrics")
        st.dataframe(metrics_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading dashboard: {e}")
        st.write("Run VolGuard and XGBoost tabs first to populate data.")
