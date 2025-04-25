import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import warnings
from datetime import datetime, timedelta
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import time

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Page config
st.set_page_config(page_title="VolGuard", page_icon="🛡️", layout="wide")

# Custom CSS for a sleek, modern look
st.markdown("""
    <style>
        .main {background-color: #1a1a2e; color: #e5e5e5; font-family: 'Arial', sans-serif;}
        .stButton>button {background-color: #0f3460; color: white; border-radius: 10px; padding: 10px 20px; font-size: 16px;}
        .stMetric {background-color: #16213e; border-radius: 15px; padding: 15px; text-align: center;}
        h1 {color: #e94560; font-size: 32px; text-align: center;}
        h2 {color: #00d4ff; font-size: 20px; margin-bottom: 10px;}
        .stDataFrame {background-color: #16213e; border-radius: 10px; padding: 10px;}
        .card {background-color: #16213e; border-radius: 15px; padding: 15px; margin: 10px 0; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);}
        .gauge {width: 100px; height: 100px; border-radius: 50%; background: conic-gradient(#e94560 0% 50%, #00d4ff 50% 100%); display: inline-block; text-align: center; line-height: 100px; color: white; font-weight: bold;}
        .risk-flag {color: #e94560; font-size: 18px;}
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🛡️ VolGuard: AI-Powered Trading Copilot")
st.markdown("**Protection First, Edge Always** | Built by Shritish & Salman")

# Sidebar
with st.sidebar:
    st.header("⚙️ Trading Parameters")
    forecast_horizon = st.slider("Forecast Horizon (days)", 1, 10, 7)
    capital = st.number_input("Capital (₹)", min_value=100000, value=1000000, step=100000)
    risk_tolerance = st.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"], index=1)
    run_button = st.button("Run VolGuard")
    st.markdown("---")
    st.markdown("**Philosophy:** Deploy edge, survive, compound, outlast.")

# Function to load data from GitHub
@st.cache_data
def load_data():
    try:
        nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/Nifty50.csv"
        vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"

        nifty = pd.read_csv(nifty_url)
        nifty.columns = nifty.columns.str.strip()
        if "Date" not in nifty.columns or "Close" not in nifty.columns:
            st.error("Nifty50.csv is missing required columns.")
            return None

        nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
        nifty = nifty.dropna(subset=["Date"])
        if nifty.empty:
            st.error("NIFTY data is empty.")
            return None
        nifty = nifty[["Date", "Close"]].set_index("Date")
        nifty = nifty[~nifty.index.duplicated(keep='first')]

        vix = pd.read_csv(vix_url)
        vix.columns = vix.columns.str.strip()
        if "Date" not in vix.columns or "Close" not in vix.columns:
            st.error("india_vix.csv is missing required columns.")
            return None

        vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")
        vix = vix.dropna(subset=["Date"])
        if vix.empty:
            st.error("VIX data is empty.")
            return None
        vix = vix[["Date", "Close"]].set_index("Date").rename(columns={"Close": "VIX"})
        vix = vix[~vix.index.duplicated(keep='first')]

        common_dates = nifty.index.intersection(vix.index)
        if len(common_dates) < 10:
            st.error(f"Insufficient overlapping dates: {len(common_dates)} found.")
            return None
        df = pd.DataFrame({"NIFTY_Close": nifty["Close"], "VIX": vix["VIX"]}, index=common_dates)
        df.index = df.index.date

        if df["NIFTY_Close"].isna().sum() > 0 or df["VIX"].isna().sum() > 0:
            st.warning("Filling NaN values.")
            df = df.ffill().bfill()
        if df.empty:
            st.error("DataFrame is empty.")
            return None

        return df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Function to generate synthetic options features
def generate_synthetic_features(df):
    n_days = len(df)
    np.random.seed(42)

    risk_free_rate = 0.06
    strike_step = 100

    def black_scholes(S, K, T, r, sigma, option_type="call"):
        try:
            T = max(T, 1e-6)
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if option_type == "call":
                return max(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), 0)
            else:
                return max(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1), 0)
        except:
            return 0

    event_spike = np.where((pd.to_datetime(df.index).month % 3 == 0) & (pd.to_datetime(df.index).day < 5), 1.2, 1.0)
    df["ATM_IV"] = df["VIX"] * (1 + np.random.normal(0, 0.1, n_days)) * event_spike
    df["ATM_IV"] = np.clip(df["ATM_IV"], 5, 50)

    def dynamic_ivp(x):
        if len(x) >= 5:
            return (np.sum(x.iloc[:-1] <= x.iloc[-1]) / (len(x) - 1)) * 100
        return 50.0

    df["IVP"] = df["ATM_IV"].rolling(len(df), min_periods=5).apply(dynamic_ivp)
    df["IVP"] = df["IVP"].interpolate().fillna(50.0)

    market_trend = df["NIFTY_Close"].pct_change().rolling(5).mean().fillna(0)
    df["PCR"] = np.clip(1.0 + np.random.normal(0, 0.1, n_days) + market_trend * -10, 0.7, 2.0)
    df["VIX_Change_Pct"] = df["VIX"].pct_change().fillna(0) * 100
    df["Spot_MaxPain_Diff_Pct"] = np.abs(np.random.lognormal(-2, 0.5, n_days))
    df["Spot_MaxPain_Diff_Pct"] = np.clip(df["Spot_MaxPain_Diff_Pct"], 0.1, 1.0)
    df["Days_to_Expiry"] = np.random.choice([1, 3, 7, 14, 21, 28], n_days)
    df["Event_Flag"] = np.where((pd.to_datetime(df.index).month % 3 == 0) & (pd.to_datetime(df.index).day < 5) | (df["Days_to_Expiry"] <= 3), 1, 0)
    df["Realized_Vol"] = df["NIFTY_Close"].pct_change().rolling(5, min_periods=5).std() * np.sqrt(252) * 100
    df["Realized_Vol"] = df["Realized_Vol"].fillna(df["VIX"])
    df["Realized_Vol"] = np.clip(df["Realized_Vol"], 0, 50)

    straddle_prices = []
    call_prices = []
    put_prices = []
    for i in range(n_days):
        S = df["NIFTY_Close"].iloc[i]
        K = round(S / strike_step) * strike_step
        T = df["Days_to_Expiry"].iloc[i] / 365
        sigma = df["ATM_IV"].iloc[i] / 100
        call_price = black_scholes(S, K, T, risk_free_rate, sigma, "call")
        put_price = black_scholes(S, K, T, risk_free_rate, sigma, "put")
        straddle_price = (call_price + put_price) * (S / 1000)
        straddle_price = np.clip(straddle_price, 50, 400)
        straddle_prices.append(straddle_price)
        call_prices.append(call_price * (S / 1000))
        put_prices.append(put_price * (S / 1000))
    df["Straddle_Price"] = straddle_prices
    df["Call_Price"] = call_prices
    df["Put_Price"] = put_prices

    if df.isna().sum().sum() > 0:
        df = df.interpolate().fillna(method='bfill')
    return df

# Function to forecast volatility (future)
def forecast_volatility_future(df, forecast_horizon):
    df.index = pd.to_datetime(df.index)
    df_garch = df.tail(len(df))
    if len(df_garch) < 200:
        st.error(f"Insufficient data for GARCH: {len(df_garch)} days.")
        return None, None, None

    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon, freq='B')

    start_time = time.time()
    df_garch['Log_Returns'] = np.log(df_garch['NIFTY_Close'] / df_garch['NIFTY_Close'].shift(1)).dropna() * 100
    if len(df_garch['Log_Returns'].dropna()) < 200:
        st.error(f"Insufficient log returns: {len(df_garch['Log_Returns'].dropna())}.")
        return None, None, None
    garch_model = arch_model(df_garch['Log_Returns'].dropna(), vol='Garch', p=1, q=1, rescale=False)
    garch_fit = garch_model.fit(disp="off")
    garch_forecast = garch_fit.forecast(horizon=forecast_horizon, reindex=False)
    garch_vols = np.sqrt(garch_forecast.variance.iloc[-1].values) * np.sqrt(252)
    garch_vols = np.clip(garch_vols, 5, 50)
    if df["Event_Flag"].iloc[-1] == 1:
        garch_vols *= 1.1
    st.write(f"GARCH fit time: {time.time() - start_time:.2f}s")

    realized_vol = df["Realized_Vol"].dropna().iloc[-5:].mean()

    start_time = time.time()
    df_xgb = df.tail(len(df))
    df_xgb['Target_Vol'] = df_xgb['Realized_Vol'].shift(-1)
    df_xgb = df_xgb.dropna()

    feature_cols = ['VIX', 'ATM_IV', 'PCR', 'Realized_Vol', 'Days_to_Expiry', 'VIX_Change_Pct']
    X = df_xgb[feature_cols]
    y = df_xgb['Target_Vol']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

    split_index = int(len(X) * 0.8)
    X_train, X_test = X_scaled.iloc[:split_index], X_scaled.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    st.write(f"XGBoost fit time: {time.time() - start_time:.2f}s")

    xgb_vols = []
    current_row = df_xgb[feature_cols].iloc[-1].copy()
    for _ in range(forecast_horizon):
        current_row_scaled = scaler.transform([current_row])
        next_vol = model.predict(current_row_scaled)[0]
        xgb_vols.append(next_vol)
        current_row["Days_to_Expiry"] = max(1, current_row["Days_to_Expiry"] - 1)
        current_row["VIX"] *= np.random.uniform(0.98, 1.02)
        current_row["VIX_Change_Pct"] = (current_row["VIX"] / df_xgb["VIX"].iloc[-1] - 1) * 100
        current_row["ATM_IV"] = current_row["VIX"] * (1 + np.random.normal(0, 0.1))
        current_row["Realized_Vol"] = np.clip(current_row["Realized_Vol"] * np.random.uniform(0.95, 1.05), 5, 50)

    xgb_vols = np.clip(xgb_vols, 5, 50)
    if df["Event_Flag"].iloc[-1] == 1:
        xgb_vols = [v * 1.1 for v in xgb_vols]

    garch_diff = np.abs(garch_vols[0] - realized_vol)
    xgb_diff = np.abs(xgb_vols[0] - realized_vol)
    garch_weight = xgb_diff / (garch_diff + xgb_diff) if (garch_diff + xgb_diff) > 0 else 0.5
    xgb_weight = 1 - garch_weight
    blended_vols = [(garch_weight * g) + (xgb_weight * x) for g, x in zip(garch_vols, xgb_vols)]
    confidence_score = min(100, max(50, 80 - abs(garch_diff - xgb_diff)))  # Simplified confidence

    forecast_log = pd.DataFrame({
        "Date": future_dates,
        "Blended_Vol": blended_vols,
        "Confidence": [confidence_score] * forecast_horizon
    })
    return forecast_log, blended_vols, realized_vol, confidence_score

# Function to generate trading signals
def generate_trading_signals(df, forecast_log, realized_vol, risk_tolerance, confidence_score):
    signals = []
    position_size = {"Conservative": 0.1, "Moderate": 0.2, "Aggressive": 0.3}[risk_tolerance]
    risk_flag = "🚩 High Risk" if confidence_score < 70 or df["Event_Flag"].iloc[-1] == 1 else "✅ Safe"

    for i, row in forecast_log.iterrows():
        blended_vol = row["Blended_Vol"]
        ivp = df["IVP"].iloc[-1]
        pcr = df["PCR"].iloc[-1]
        strike = round(df["NIFTY_Close"].iloc[-1] / 100) * 100
        signal = "Hold"
        action = None

        if blended_vol > realized_vol + 2 and ivp > 75 and confidence_score >= 70:
            signal = "Buy Call"
            action = f"Buy Call at {strike}, Premium ~{df['Call_Price'].iloc[-1]:.2f}"
        elif blended_vol < realized_vol - 2 and pcr < 0.8 and confidence_score >= 70:
            signal = "Buy Put"
            action = f"Buy Put at {strike}, Premium ~{df['Put_Price'].iloc[-1]:.2f}"
        elif df["Event_Flag"].iloc[-1] == 1 and confidence_score >= 70:
            signal = "Buy Straddle"
            action = f"Buy Straddle at {strike}, Premium ~{df['Straddle_Price'].iloc[-1]:.2f}"

        signals.append({
            "Date": row["Date"].strftime("%d-%b-%Y"),
            "Signal": signal,
            "Action": action,
            "Position Size": f"{int(position_size * 100)}% of Capital",
            "Risk Flag": risk_flag
        })
    return pd.DataFrame(signals)

# Function to backtest strategy (simplified historical simulation)
def backtest_strategy(df, signals_df):
    df = df.copy()
    df['Signal'] = "Hold"
    df['Position'] = 0.0
    df['Trade_Cost'] = 0.0
    df['PnL'] = 0.0

    for _, signal in signals_df.iterrows():
        if signal["Date"] in df.index.astype(str):
            df.loc[df.index[df.index.astype(str) == signal["Date"]].tolist()[0], "Signal"] = signal["Signal"]
            position_size = float(signal["Position Size"].replace("% of Capital", "")) / 100
            df.loc[df.index[df.index.astype(str) == signal["Date"]].tolist()[0], "Position"] = position_size

    for i in range(1, len(df)):
        if df["Signal"].iloc[i] == "Buy Call":
            price_change = (df["NIFTY_Close"].iloc[i] - df["NIFTY_Close"].iloc[i-1]) / df["NIFTY_Close"].iloc[i-1]
            option_return = price_change * 2
            cost = df["Call_Price"].iloc[i] * df["Position"].iloc[i]
            df.loc[df.index[i], "Trade_Cost"] = -cost
            df.loc[df.index[i], "PnL"] = (option_return * cost)
        elif df["Signal"].iloc[i] == "Buy Put":
            price_change = (df["NIFTY_Close"].iloc[i-1] - df["NIFTY_Close"].iloc[i]) / df["NIFTY_Close"].iloc[i-1]
            option_return = price_change * 2
            cost = df["Put_Price"].iloc[i] * df["Position"].iloc[i]
            df.loc[df.index[i], "Trade_Cost"] = -cost
            df.loc[df.index[i], "PnL"] = (option_return * cost)
        elif df["Signal"].iloc[i] == "Buy Straddle":
            price_change = abs(df["NIFTY_Close"].iloc[i] - df["NIFTY_Close"].iloc[i-1]) / df["NIFTY_Close"].iloc[i-1]
            option_return = price_change * 1.5
            cost = df["Straddle_Price"].iloc[i] * df["Position"].iloc[i]
            df.loc[df.index[i], "Trade_Cost"] = -cost
            df.loc[df.index[i], "PnL"] = (option_return * cost)

    df['Cumulative_PnL'] = df['PnL'].cumsum()
    df['Cumulative_Returns'] = (df['Cumulative_PnL'] / capital) * 100
    returns = df['PnL'] / capital
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
    max_drawdown = (df['Cumulative_PnL'].cummax() - df['Cumulative_PnL']).max() / capital * 100
    behavioral_score = min(10, max(1, 10 - (abs(max_drawdown) / 3)))  # Simplified score

    return {
        "Total_Return": df['Cumulative_Returns'].iloc[-1],
        "Sharpe_Ratio": sharpe_ratio,
        "Max_Drawdown": max_drawdown,
        "PnL_Series": df['Cumulative_PnL'],
        "Behavioral_Score": behavioral_score
    }

# Main execution
if run_button:
    with st.spinner("Analyzing market conditions..."):
        df = load_data()
        if df is not None:
            df = generate_synthetic_features(df)

            with st.spinner("Forecasting volatility..."):
                forecast_log, blended_vols, realized_vol, confidence_score = forecast_volatility_future(df, forecast_horizon)

            if forecast_log is not None:
                # Volatility Forecast Card
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("📈 Volatility Forecast")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Blended Volatility (Avg)", f"{np.mean(blended_vols):.2f}%")
                with col2:
                    st.metric("Realized Volatility (5-day)", f"{realized_vol:.2f}%")
                st.markdown(f'<div class="gauge">{int(confidence_score)}%</div>', unsafe_allow_html=True)
                st.markdown("**Confidence Score**", unsafe_allow_html=True)
                st.line_chart(pd.DataFrame({
                    "Date": forecast_log["Date"],
                    "Blended_Vol": forecast_log["Blended_Vol"]
                }).set_index("Date"))
                st.markdown('</div>', unsafe_allow_html=True)

                # Trading Signals Card
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("🎯 Trading Signals (Future Forecast)")
                signals_df = generate_trading_signals(df, forecast_log, realized_vol, risk_tolerance, confidence_score)
                st.dataframe(signals_df[["Date", "Signal", "Action", "Position Size", "Risk Flag"]], use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Backtest Results Card
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("📊 Backtest Performance")
                signals_df_historical = generate_trading_signals(df, forecast_log, realized_vol, risk_tolerance, historical=True)
                backtest_results = backtest_strategy(df, signals_df_historical)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Return", f"{backtest_results['Total_Return']:.2f}%")
                with col2:
                    st.metric("Sharpe Ratio", f"{backtest_results['Sharpe_Ratio']:.2f}")
                with col3:
                    st.metric("Max Drawdown", f"{backtest_results['Max_Drawdown']:.2f}%")
                st.markdown(f'<div class="gauge">{int(backtest_results["Behavioral_Score"])}/10</div>', unsafe_allow_html=True)
                st.markdown("**Behavioral Score**", unsafe_allow_html=True)
                st.line_chart(pd.DataFrame({
                    "Date": df.index,
                    "Cumulative PnL": backtest_results["PnL_Series"]
                }).set_index("Date"))
                st.markdown('</div>', unsafe_allow_html=True)

                # Journaling Prompt
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("📝 Journaling Prompt")
                st.text_area("Reflect on your trading discipline today:", height=100)
                st.button("Save Reflection")
                st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Set parameters and click 'Run VolGuard' to begin.")

# Export Button
st.markdown('<div style="text-align: center; margin-top: 20px;">', unsafe_allow_html=True)
st.button("Export to PDF/CSV")
st.markdown('</div>', unsafe_allow_html=True)
