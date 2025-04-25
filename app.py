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

# Custom CSS for improved visibility
st.markdown("""
    <style>
        .main {background-color: #f0f2f5; color: #2b3e50;}
        .stButton>button {background-color: #2b3e50; color: white; border-radius: 5px; padding: 8px 16px;}
        .stMetric {background-color: #ffffff; border-radius: 8px; padding: 12px;}
        h1, h2, h3 {color: #2b3e50; font-family: 'Arial', sans-serif;}
        .stDataFrame {background-color: #ffffff; border-radius: 8px; padding: 10px;}
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🛡️ VolGuard: AI-Powered Trading Copilot")
st.markdown("Your disciplined partner for volatility-driven options trading.")
st.write(f"Date: {datetime.now().strftime('%d-%b-%Y')}")

# Sidebar
with st.sidebar:
    st.header("⚙️ Parameters")
    forecast_horizon = st.slider("Forecast Horizon (days)", 1, 10, 7)
    capital = st.number_input("Capital (₹)", min_value=100000, value=1000000, step=100000)
    risk_tolerance = st.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"], index=1)
    st.header("Actions")
    run_button = st.button("Run VolGuard")
    st.markdown("---")
    st.info("Built by Shritish Shukla & AI Co-Founder | Protection First, Edge Always")

# Function to load data from GitHub
@st.cache_data
def load_data():
    try:
        nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/Nifty50.csv"
        vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"

        nifty = pd.read_csv(nifty_url)
        nifty.columns = nifty.columns.str.strip()
        if "Date" not in nifty.columns or "Close" not in nifty.columns:
            st.error("Nifty50.csv is missing required columns: 'Date' or 'Close'.")
            return None

        nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
        nifty = nifty.dropna(subset=["Date"])
        if nifty.empty:
            st.error("NIFTY data is empty or invalid.")
            return None
        if not pd.api.types.is_numeric_dtype(nifty["Close"]):
            st.error("NIFTY 'Close' column contains non-numeric values.")
            return None
        nifty = nifty[["Date", "Close"]].set_index("Date")
        nifty = nifty[~nifty.index.duplicated(keep='first')]

        vix = pd.read_csv(vix_url)
        vix.columns = vix.columns.str.strip()
        if "Date" not in vix.columns or "Close" not in vix.columns:
            st.error("india_vix.csv is missing required columns: 'Date' or 'Close'.")
            return None

        vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")
        vix = vix.dropna(subset=["Date"])
        if vix.empty:
            st.error("VIX data is empty or invalid.")
            return None
        if not pd.api.types.is_numeric_dtype(vix["Close"]):
            st.error("VIX 'Close' column contains non-numeric values.")
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
            st.warning("Filling NaN values with forward-fill and back-fill.")
            df = df.ffill().bfill()
        if df.empty:
            st.error("DataFrame is empty after processing.")
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

    # Synthetic Features
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
        call_price = black_scholes(S, K, T, risk_free_rate, sigma, "call")
        put_price = black_scholes(S, K, T, risk_free_rate, sigma, "put")
        straddle_price = (call_price + put_price) * (S / 1000)
        straddle_price = np.clip(straddle_price, 50, 400)
        straddle_prices.append(straddle_price)
    df["Straddle_Price"] = straddle_prices

    if df.isna().sum().sum() > 0:
        st.warning("Filling missing values...")
        df = df.interpolate().fillna(method='bfill')
    return df

# Function to forecast volatility
def forecast_volatility(df, forecast_horizon):
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

    feature_cols = ['VIX', 'ATM_IV', 'PCR', 'IV_Skew', 'Realized_Vol', 'Days_to_Expiry', 'VIX_Change_Pct']
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

    forecast_log = pd.DataFrame({
        "Date": future_dates,
        "GARCH_Vol": garch_vols,
        "XGBoost_Vol": xgb_vols,
        "Blended_Vol": blended_vols
    })
    return forecast_log, blended_vols, realized_vol

# Function to generate trading signals
def generate_trading_signals(df, forecast_log, realized_vol, risk_tolerance):
    signals = []
    position_size = {"Conservative": 0.1, "Moderate": 0.2, "Aggressive": 0.3}[risk_tolerance]
    for i, row in forecast_log.iterrows():
        blended_vol = row["Blended_Vol"]
        ivp = df["IVP"].iloc[-1]
        pcr = df["PCR"].iloc[-1]
        event_flag = df["Event_Flag"].iloc[-1]
        strike = round(df["NIFTY_Close"].iloc[-1] / 100) * 100
        signal = "Hold"
        action = None

        if blended_vol > realized_vol + 2 and ivp > 75:
            signal = "Buy Call"
            action = f"Buy Call at Strike {strike}, Premium ~{df['Straddle_Price'].iloc[-1]/2:.2f}"
        elif blended_vol < realized_vol - 2 and pcr < 0.8:
            signal = "Buy Put"
            action = f"Buy Put at Strike {strike}, Premium ~{df['Straddle_Price'].iloc[-1]/2:.2f}"
        elif event_flag == 1:
            signal = "Buy Straddle"
            action = f"Buy Straddle at Strike {strike}, Premium ~{df['Straddle_Price'].iloc[-1]:.2f}"

        signals.append({"Date": row["Date"].strftime("%d-%b-%Y"), "Signal": signal, "Action": action,
                       "Position Size": f"{position_size*100}% of Capital"})
    return pd.DataFrame(signals)

# Function to backtest strategy
def backtest_strategy(df, signals_df):
    df = df.copy()
    df['Signal'] = "Hold"
    for _, signal in signals_df.iterrows():
        if signal["Date"] in df.index:
            df.loc[signal["Date"], "Signal"] = signal["Signal"]

    df['Returns'] = df["NIFTY_Close"].pct_change()
    df['Strategy_Returns'] = df['Returns'] * (df['Signal'].map({"Buy Call": 1, "Buy Put": -1, "Buy Straddle": 0, "Hold": 0}).fillna(0))
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod() - 1
    sharpe_ratio = df['Strategy_Returns'].mean() / df['Strategy_Returns'].std() * np.sqrt(252) if df['Strategy_Returns'].std() != 0 else 0
    max_drawdown = (df['Cumulative_Returns'].cummax() - df['Cumulative_Returns']).max()

    return {
        "Total_Return": df['Cumulative_Returns'].iloc[-1] * 100,
        "Sharpe_Ratio": sharpe_ratio,
        "Max_Drawdown": max_drawdown * 100
    }

# Main execution
if run_button:
    with st.spinner("Loading data from GitHub..."):
        df = load_data()
        if df is not None:
            st.success(f"✅ Data loaded successfully! ({len(df)} days)")
            st.subheader("Sample Data")
            st.dataframe(df.head())
            st.subheader("Data Info")
            st.write(df.info())
            st.subheader("VIX Statistics")
            st.write(f"Mean: {df['VIX'].mean():.2f}, Min: {df['VIX'].min():.2f}, Max: {df['VIX'].max():.2f}")

            with st.spinner("Generating synthetic features..."):
                df = generate_synthetic_features(df)
                st.subheader("Synthetic Features (Last 5 Days)")
                st.dataframe(df[["NIFTY_Close", "VIX", "ATM_IV", "IVP", "PCR", "Realized_Vol"]].tail())
                st.subheader("Feature Statistics")
                st.dataframe(df[["NIFTY_Close", "VIX", "ATM_IV", "IVP", "PCR", "Realized_Vol"]].describe())

            with st.spinner("Forecasting volatility..."):
                forecast_log, blended_vols, realized_vol = forecast_volatility(df, forecast_horizon)
                if forecast_log is not None:
                    st.subheader("📈 Volatility Forecast")
                    forecast_df = pd.DataFrame({
                        "Date": [d.strftime("%d-%b-%Y") for d in forecast_log["Date"]],
                        "GARCH (%)": [f"{v:.2f}" for v in forecast_log["GARCH_Vol"]],
                        "XGBoost (%)": [f"{v:.2f}" for v in forecast_log["XGBoost_Vol"]],
                        "Blended (%)": [f"{v:.2f}" for v in forecast_log["Blended_Vol"]]
                    })
                    st.dataframe(forecast_df, use_container_width=True)

                    st.subheader("Volatility Chart")
                    chart_data = pd.DataFrame({
                        "Date": df.index[-30:].append(forecast_log["Date"]),
                        "Realized_Vol": df["Realized_Vol"].iloc[-30:].tolist() + [np.nan] * forecast_horizon,
                        "Blended_Vol": [np.nan] * 30 + forecast_log["Blended_Vol"].tolist()
                    })
                    st.line_chart(chart_data.set_index("Date"))

                    st.subheader("Forecast Metrics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Avg Forecasted Volatility", f"{np.mean(blended_vols):.2f}%")
                    with col2:
                        st.metric("Recent Realized Volatility (5-day)", f"{realized_vol:.2f}%")

                    # Strategy Engine
                    st.subheader("🎯 Trading Signals")
                    signals_df = generate_trading_signals(df, forecast_log, realized_vol, risk_tolerance)
                    st.dataframe(signals_df, use_container_width=True)

                    # Backtesting
                    st.subheader("📊 Backtest Results")
                    backtest_results = backtest_strategy(df, signals_df)
                    st.write(f"Total Return: {backtest_results['Total_Return']:.2f}%")
                    st.write(f"Sharpe Ratio: {backtest_results['Sharpe_Ratio']:.2f}")
                    st.write(f"Max Drawdown: {backtest_results['Max_Drawdown']:.2f}%")
else:
    st.info("Set parameters in the sidebar and click 'Run VolGuard' to start.")
