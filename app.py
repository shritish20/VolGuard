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
        h1 {color: #2b3e50; font-family: 'Arial', sans-serif;}
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
        # URLs to your GitHub CSV files (raw content)
        nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/Nifty50.csv"
        vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"

        # Load NIFTY data
        nifty = pd.read_csv(nifty_url)
        nifty.columns = nifty.columns.str.strip()
        if "Date" not in nifty.columns or "Close" not in nifty.columns:
            st.error("Nifty50.csv is missing required columns: 'Date' or 'Close'. Check the file.")
            return None

        nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
        nifty = nifty.dropna(subset=["Date"])
        if nifty.empty:
            st.error("NIFTY data is empty or invalid after parsing dates.")
            return None
        if not pd.api.types.is_numeric_dtype(nifty["Close"]):
            st.error("NIFTY 'Close' column contains non-numeric values.")
            return None
        nifty = nifty[["Date", "Close"]]
        nifty = nifty.set_index("Date")
        nifty = nifty[~nifty.index.duplicated(keep='first')]

        # Load VIX data
        vix = pd.read_csv(vix_url)
        vix.columns = vix.columns.str.strip()
        if "Date" not in vix.columns or "Close" not in vix.columns:
            st.error("india_vix.csv is missing required columns: 'Date' or 'Close'. Check the file.")
            return None

        vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")
        vix = vix.dropna(subset=["Date"])
        if vix.empty:
            st.error("VIX data is empty or invalid after parsing dates.")
            return None
        if not pd.api.types.is_numeric_dtype(vix["Close"]):
            st.error("VIX 'Close' column contains non-numeric values.")
            return None
        vix = vix[["Date", "Close"]]
        vix = vix.set_index("Date").rename(columns={"Close": "VIX"})
        vix = vix[~vix.index.duplicated(keep='first')]

        # Align dates between NIFTY and VIX
        common_dates = nifty.index.intersection(vix.index)
        if len(common_dates) < 10:
            st.error(f"Insufficient overlapping dates between NIFTY and VIX data. Found only {len(common_dates)} common dates.")
            return None
        nifty = nifty.loc[common_dates]
        vix = vix.loc[common_dates]

        # Create DataFrame
        df = pd.DataFrame({
            "NIFTY_Close": nifty["Close"],
            "VIX": vix["VIX"]
        }, index=common_dates)

        # Remove timestamp from index (keep only date)
        df.index = df.index.date

        # Validate data
        if df["NIFTY_Close"].isna().sum() > 0 or df["VIX"].isna().sum() > 0:
            st.warning("Found NaN values in data. Filling with forward-fill and back-fill.")
            df = df.ffill().bfill()
        if df.empty:
            st.error("DataFrame is empty after processing.")
            return None

        return df

    except Exception as e:
        st.error(f"Error loading data from GitHub: {str(e)}")
        return None

# Function to generate synthetic options features
def generate_synthetic_features(df):
    n_days = len(df)
    np.random.seed(42)  # For reproducibility

    # Constants
    risk_free_rate = 0.06
    strike_step = 100

    # Black-Scholes Pricing
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
    # ATM_IV: VIX-based with dynamic noise and event spikes
    event_spike = np.where((pd.to_datetime(df.index).month % 3 == 0) & (pd.to_datetime(df.index).day < 5), 1.2, 1.0)
    df["ATM_IV"] = df["VIX"] * (1 + np.random.normal(0, 0.1, n_days)) * event_spike
    df["ATM_IV"] = np.clip(df["ATM_IV"], 5, 50)

    # IVP: Percentile rank with dynamic window
    def dynamic_ivp(x):
        if len(x) >= 5:
            return (np.sum(x.iloc[:-1] <= x.iloc[-1]) / (len(x) - 1)) * 100
        return 50.0  # Default for very early data

    df["IVP"] = df["ATM_IV"].rolling(252, min_periods=5).apply(dynamic_ivp)
    df["IVP"] = df["IVP"].interpolate().fillna(50.0)

    # PCR: Enhanced with stronger momentum signal
    market_trend = df["NIFTY_Close"].pct_change().rolling(5).mean().fillna(0)
    df["PCR"] = np.clip(1.0 + np.random.normal(0, 0.1, n_days) + market_trend * -10, 0.7, 2.0)

    # VIX_Change_Pct
    df["VIX_Change_Pct"] = df["VIX"].pct_change().fillna(0) * 100

    # Spot_MaxPain_Diff_Pct: Realistic max pain differences
    df["Spot_MaxPain_Diff_Pct"] = np.abs(np.random.lognormal(-2, 0.5, n_days))
    df["Spot_MaxPain_Diff_Pct"] = np.clip(df["Spot_MaxPain_Diff_Pct"], 0.1, 1.0)

    # Days_to_Expiry: Realistic expiry cycles
    df["Days_to_Expiry"] = np.random.choice([1, 3, 7, 14, 21, 28], n_days)

    # Event_Flag: Earnings or expiry events
    df["Event_Flag"] = np.where((pd.to_datetime(df.index).month % 3 == 0) & (pd.to_datetime(df.index).day < 5) | (df["Days_to_Expiry"] <= 3), 1, 0)

    # FII Positions: Trend-based with reversals
    fii_trend = np.random.normal(0, 10000, n_days)
    fii_trend[::30] *= -1  # Periodic reversals
    df["FII_Index_Fut_Pos"] = np.cumsum(fii_trend).astype(int)
    df["FII_Option_Pos"] = np.cumsum(np.random.normal(0, 5000, n_days)).astype(int)

    # IV_Skew: Enhanced market correlation
    df["IV_Skew"] = np.clip(np.random.normal(0, 0.8, n_days) + (df["VIX"] / 15 - 1) * 3, -3, 3)

    # Realized Volatility: Handle early periods and cap outliers
    df["Realized_Vol"] = df["NIFTY_Close"].pct_change().rolling(5, min_periods=5).std() * np.sqrt(252) * 100
    df["Realized_Vol"] = df["Realized_Vol"].fillna(df["VIX"])
    df["Realized_Vol"] = np.clip(df["Realized_Vol"], 0, 50)

    # Advance/Decline Ratio: Simulate market breadth
    df["Advance_Decline_Ratio"] = np.clip(1.0 + np.random.normal(0, 0.2, n_days) + market_trend * 10, 0.5, 2.0)

    # Capital Pressure Index: Wider range
    df["Capital_Pressure_Index"] = (df["FII_Index_Fut_Pos"] / 3e4 + df["FII_Option_Pos"] / 1e4 + df["PCR"]) / 3
    df["Capital_Pressure_Index"] = np.clip(df["Capital_Pressure_Index"], -2, 2)

    # Gamma Bias: Proxy with DTE and IV_Skew
    df["Gamma_Bias"] = np.clip(df["IV_Skew"] * (30 - df["Days_to_Expiry"]) / 30, -2, 2)

    # Capital and PnL
    df["Total_Capital"] = capital
    df["PnL_Day"] = np.random.normal(0, 5000, n_days) * (1 - df["Event_Flag"] * 0.5)

    # Straddle Pricing
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

    # Final Sanity Checks
    missing = df.isna().sum().sum()
    if missing > 0:
        st.warning(f"Filling {missing} missing values...")
        df = df.interpolate().fillna(method='bfill')

    return df

# Function to forecast volatility
def forecast_volatility(df, forecast_horizon):
    # Convert index to datetime if not already
    df.index = pd.to_datetime(df.index)

    # GARCH: Use last 252 days (1 year of trading days)
    df_garch = df.tail(252)
    if len(df_garch) != 252:
        st.error(f"Expected 252 days of data for GARCH, but got {len(df_garch)} days.")
        return None, None, None

    # Generate future dates for forecasts (business days)
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon, freq='B')
    future_dates_str = [d.strftime('%d-%b-%Y') for d in future_dates]

    # GARCH(1,1) Forecast
    start_time = time.time()
    df_garch['Log_Returns'] = np.log(df_garch['NIFTY_Close'] / df_garch['NIFTY_Close'].shift(1)).dropna()
    returns = df_garch['Log_Returns'].dropna() * 100  # Scale for GARCH stability
    if len(returns) < 200:  # Ensure enough data for GARCH
        st.error(f"Insufficient log returns for GARCH: {len(returns)} returns available.")
        return None, None, None
    garch_model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
    garch_fit = garch_model.fit(disp="off")
    garch_forecast = garch_fit.forecast(horizon=forecast_horizon, reindex=False)
    garch_vols = np.sqrt(garch_forecast.variance.iloc[-1].values) * np.sqrt(252)  # Annualize
    garch_vols = np.clip(garch_vols, 5, 50)  # Cap to realistic range
    if df["Event_Flag"].iloc[-1] == 1:
        garch_vols *= 1.1
    st.write(f"GARCH fit time: {time.time() - start_time:.2f} seconds")

    # Realized Volatility Reference
    realized_vol = df["Realized_Vol"].dropna().iloc[-5:].mean()

    # XGBoost: Use last 252 days (same as GARCH, since we only have 1 year of data)
    start_time = time.time()
    df_xgb = df.tail(252)
    df_xgb['Target_Vol'] = df_xgb['Realized_Vol'].shift(-1)
    df_xgb = df_xgb.dropna()

    feature_cols = [
        'VIX', 'ATM_IV', 'PCR', 'IV_Skew', 'Realized_Vol', 'Days_to_Expiry'
    ]  # Reduced features for efficiency

    X = df_xgb[feature_cols]
    y = df_xgb['Target_Vol']

    # Scale features for XGBoost
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

    split_index = int(len(X) * 0.8)
    X_train, X_test = X_scaled.iloc[:split_index], X_scaled.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42)  # Optimized parameters
    model.fit(X_train, y_train)
    st.write(f"XGBoost fit time: {time.time() - start_time:.2f} seconds")

    xgb_vols = []
    current_row = X.iloc[-1].copy()
    current_row_df = pd.DataFrame([current_row], columns=feature_cols)
    current_row_scaled = scaler.transform(current_row_df)
    for _ in range(forecast_horizon):
        next_vol = model.predict(current_row_scaled)[0]
        xgb_vols.append(next_vol)
        current_row["Days_to_Expiry"] = max(1, current_row["Days_to_Expiry"] - 1)
        current_row["VIX"] *= np.random.uniform(0.98, 1.02)
        current_row_df = pd.DataFrame([current_row], columns=feature_cols)
        current_row_scaled = scaler.transform(current_row_df)

    xgb_vols = np.clip(xgb_vols, 5, 50)
    if df["Event_Flag"].iloc[-1] == 1:
        xgb_vols = [v * 1.1 for v in xgb_vols]

    # Blend Forecasts
    garch_diff = np.abs(garch_vols[0] - realized_vol)
    xgb_diff = np.abs(xgb_vols[0] - realized_vol)
    garch_weight = xgb_diff / (garch_diff + xgb_diff) if (garch_diff + xgb_diff) > 0 else 0.5
    xgb_weight = 1 - garch_weight
    blended_vols = [(garch_weight * g) + (xgb_weight * x) for g, x in zip(garch_vols, xgb_vols)]

    # Create forecast log
    forecast_log = pd.DataFrame({
        "Date": future_dates,
        "GARCH_Vol": garch_vols,
        "XGBoost_Vol": xgb_vols,
        "Blended_Vol": blended_vols
    })

    return forecast_log, blended_vols, realized_vol

# Main execution
if run_button:
    with st.spinner("Loading data from GitHub..."):
        df = load_data()
        if df is not None:
            st.success("✅ Data loaded successfully!")
            st.write("Sample Data:")
            st.dataframe(df.head())
            st.write("Data Info:")
            st.write(df.info())
            st.write("VIX Statistics:")
            st.write(f"Mean: {df['VIX'].mean():.2f}, Min: {df['VIX'].min():.2f}, Max: {df['VIX'].max():.2f}")

            # Generate synthetic features
            with st.spinner("Generating synthetic options features..."):
                df = generate_synthetic_features(df)
                st.write("Synthetic Features Generated:")
                feature_cols = [
                    "NIFTY_Close", "VIX", "ATM_IV", "IVP", "PCR", "VIX_Change_Pct",
                    "Spot_MaxPain_Diff_Pct", "Straddle_Price", "IV_Skew", "Realized_Vol",
                    "Days_to_Expiry", "Event_Flag", "FII_Index_Fut_Pos", "FII_Option_Pos",
                    "Advance_Decline_Ratio", "Capital_Pressure_Index", "Gamma_Bias"
                ]
                st.dataframe(df[feature_cols].tail())
                st.write("Feature Statistics:")
                st.dataframe(df[feature_cols].describe())

            # Forecast volatility
            with st.spinner("Forecasting volatility..."):
                max_wait = 300  # 5-minute timeout
                start_time = time.time()
                try:
                    forecast_log, blended_vols, realized_vol = forecast_volatility(df, forecast_horizon)
                    if forecast_log is None:
                        st.error("Forecasting failed. Check error messages above.")
                    elif time.time() - start_time > max_wait:
                        st.error("Forecasting took too long. Please reduce dataset or optimize parameters.")
                    else:
                        # Display forecast
                        st.subheader("📈 Volatility Forecast")
                        forecast_df = pd.DataFrame({
                            "Date": [d.strftime("%d-%b-%Y") for d in forecast_log["Date"]],
                            "GARCH (%)": [f"{v:.2f}" for v in forecast_log["GARCH_Vol"]],
                            "XGBoost (%)": [f"{v:.2f}" for v in forecast_log["XGBoost_Vol"]],
                            "Blended (%)": [f"{v:.2f}" for v in forecast_log["Blended_Vol"]]
                        })
                        st.dataframe(forecast_df, use_container_width=True)

                        # Display metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Avg Forecasted Volatility", f"{np.mean(blended_vols):.2f}%")
                        with col2:
                            st.metric("Recent Realized Volatility (5-day)", f"{realized_vol:.2f}%")
                except Exception as e:
                    st.error(f"Error during forecasting: {str(e)}")
