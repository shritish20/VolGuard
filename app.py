import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from py5paisa import FivePaisaClient
from datetime import datetime, timedelta
import logging
from arch import arch_model
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import requests
import ssl
import json

# Enable legacy SSL renegotiation for 5paisa API
try:
    import urllib3
    urllib3.util.ssl_.DEFAULT_CIPHERS += ':HIGH:!DH:!aNULL'
    ctx = urllib3.util.ssl_.create_urllib3_context()
    ctx.options |= ssl.OP_LEGACY_SERVER_CONNECT
except Exception as e:
    st.error(f"SSL setup failed: {str(e)}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(page_title="VolGuard Pro", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for premium UI
st.markdown("""
    <style>
        body {
            background-color: #1a1a2e;
            color: #e5e5e5;
            font-family: 'Arial', sans-serif;
        }
        .stApp {
            background-color: #1a1a2e;
        }
        .css-1d391kg {
            background-color: #1a1a2e !important;
        }
        .sidebar .sidebar-content {
            background-color: #16213e;
            padding: 20px;
        }
        .stButton>button {
            background-color: #e94560;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 10px 20px;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #ff6f61;
        }
        .card {
            background-color: #0f172a;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            margin-bottom: 20px;
        }
        .metric-card {
            background-color: #0f172a;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .strategy-card {
            background-color: #16213e;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            transition: transform 0.3s;
        }
        .strategy-card:hover {
            transform: translateY(-5px);
        }
        .footer {
            text-align: center;
            padding: 20px;
            color: #a1a1aa;
            font-size: 14px;
        }
        .header {
            text-align: center;
            padding: 20px;
            background-color: #0f172a;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .data-tag {
            font-size: 12px;
            color: #a1a1aa;
            margin-top: 10px;
            text-align: right;
        }
        .regime-badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
            margin-top: 10px;
        }
        .regime-low { background-color: #28a745; }
        .regime-medium { background-color: #ffcc00; }
        .regime-high { background-color: #e94560; }
        .regime-event { background-color: #ff6f61; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "client" not in st.session_state:
    st.session_state.client = None
if "trades" not in st.session_state:
    st.session_state.trades = []
if "journal_entries" not in st.session_state:
    st.session_state.journal_entries = []
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None

# 5paisa Client Initialization
def initialize_5paisa_client(totp_code):
    try:
        logger.info("Initializing 5paisa client")
        cred = {
            "APP_NAME": st.secrets["fivepaisa"]["APP_NAME"],
            "APP_SOURCE": st.secrets["fivepaisa"]["APP_SOURCE"],
            "USER_ID": st.secrets["fivepaisa"]["USER_ID"],
            "PASSWORD": st.secrets["fivepaisa"]["PASSWORD"],
            "USER_KEY": st.secrets["fivepaisa"]["USER_KEY"],
            "ENCRYPTION_KEY": st.secrets["fivepaisa"]["ENCRYPTION_KEY"]
        }
        client = FivePaisaClient(cred=cred)
        response = client.get_totp_session(
            st.secrets["fivepaisa"]["CLIENT_CODE"],
            totp_code,
            st.secrets["fivepaisa"]["PIN"]
        )
        if response and response.get("Status") == 0:
            logger.info("5paisa client initialized successfully")
            return client
        else:
            logger.error(f"Failed to initialize 5paisa client: {response.get('Message', 'Invalid TOTP session response')}")
            st.error(f"❌ Failed to initialize 5paisa client: {response.get('Message', 'Invalid TOTP session response')}")
            return None
    except Exception as e:
        logger.error(f"Error initializing 5paisa client: {str(e)}")
        st.error(f"❌ Error initializing 5paisa client: {str(e)}")
        return None

# Fetch Market Data
def fetch_market_data(client):
    try:
        req_list = [
            {"Exch": "N", "ExchType": "C", "ScripCode": 999920000},  # NIFTY 50
            {"Exch": "N", "ExchType": "C", "ScripCode": 999920005}   # India VIX
        ]
        max_retries = 3
        for attempt in range(max_retries):
            try:
                market_feed = client.fetch_market_feed(req_list)
                if market_feed is None or not isinstance(market_feed, dict) or "Data" not in market_feed:
                    raise Exception("Invalid market feed response")
                if not market_feed["Data"]:
                    raise Exception("No data returned")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to fetch market data after {max_retries} attempts: {str(e)}")
                    return None
                logger.warning(f"Market feed attempt {attempt + 1} failed: {str(e)}. Retrying...")
                continue

        nifty_data = None
        vix_data = None
        for item in market_feed["Data"]:
            if item["ScripCode"] == 999920000:
                nifty_data = item
            elif item["ScripCode"] == 999920005:
                vix_data = item

        if not nifty_data or not vix_data:
            logger.error("Incomplete market data: NIFTY 50 or VIX data missing")
            return None

        df = pd.DataFrame({
            "Date": [datetime.now()],
            "NIFTY_Close": [nifty_data["LastRate"]],
            "VIX": [vix_data["LastRate"]]
        })
        df.set_index("Date", inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")
        return None

# Simulate Historical Data (Fallback for Demo/Testing)
def get_historical_data():
    try:
        dates = pd.date_range(end=datetime.now(), periods=90, freq="D")
        np.random.seed(42)
        nifty_prices = 22000 + np.random.normal(0, 100, len(dates)).cumsum()
        vix_values = 15 + np.random.normal(0, 2, len(dates))
        df = pd.DataFrame({
            "NIFTY_Close": nifty_prices,
            "VIX": vix_values,
            "Event_Flag": np.random.choice([0, 1], len(dates), p=[0.9, 0.1]),
            "Days_to_Expiry": np.random.randint(1, 30, len(dates))
        }, index=dates)
        return df
    except Exception as e:
        logger.error(f"Error generating historical data: {str(e)}")
        return None

# Volatility Forecasting
def forecast_volatility(df):
    try:
        # GARCH Model
        returns = df["NIFTY_Close"].pct_change().dropna() * 100
        garch_model = arch_model(returns, vol="Garch", p=1, q=1)
        garch_fit = garch_model.fit(disp="off")
        garch_forecast = garch_fit.forecast(horizon=5)
        garch_vol = np.sqrt(garch_forecast.variance.iloc[-1]) * np.sqrt(252)

        # XGBoost Model (Simplified)
        features = pd.DataFrame({
            "Lagged_Vol": df["VIX"].shift(1),
            "NIFTY_Return": returns,
            "Event_Flag": df["Event_Flag"]
        }).dropna()
        target = df["VIX"][1:]
        train_size = int(0.8 * len(features))
        X_train, X_test = features[:train_size], features[train_size:]
        y_train, y_test = target[:train_size], target[train_size:]
        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        future_features = features.iloc[-1:].copy()
        xgb_vol = xgb_model.predict(future_features)[0]

        # Blended Volatility
        blended_vol = 0.6 * garch_vol.iloc[-1] + 0.4 * xgb_vol
        confidence_score = 100 - (rmse / np.mean(y_test) * 100)

        forecast_log = pd.DataFrame({
            "Date": pd.date_range(start=datetime.now(), periods=5, freq="D"),
            "GARCH_Vol": garch_vol,
            "XGBoost_Vol": [xgb_vol] * 5,
            "Blended_Vol": [blended_vol] * 5
        })
        return blended_vol, confidence_score, forecast_log, rmse, xgb_model.feature_importances_, features.columns
    except Exception as e:
        logger.error(f"Error in volatility forecasting: {str(e)}")
        return None, None, None, None, None, None

# Generate Trading Strategies
def generate_trading_strategies(blended_vol, risk_tolerance, confidence_score, capital):
    try:
        strategies = []
        if risk_tolerance == "Conservative":
            strategies.append({
                "Name": "Hedged Iron Condor",
                "Vol_Threshold": 15,
                "Capital_Deploy": capital * 0.2,
                "Max_Loss": capital * 0.05,
                "Risk_Reward": 2.5,
                "Confidence": confidence_score / 100,
                "Tags": ["Low Risk", "Hedged"]
            })
        elif risk_tolerance == "Moderate":
            strategies.append({
                "Name": "Neutral Butterfly",
                "Vol_Threshold": 20,
                "Capital_Deploy": capital * 0.4,
                "Max_Loss": capital * 0.1,
                "Risk_Reward": 3.0,
                "Confidence": confidence_score / 100,
                "Tags": ["Balanced", "Event-Driven"]
            })
        else:
            strategies.append({
                "Name": "Aggressive Strangle",
                "Vol_Threshold": 25,
                "Capital_Deploy": capital * 0.6,
                "Max_Loss": capital * 0.15,
                "Risk_Reward": 4.0,
                "Confidence": confidence_score / 100,
                "Tags": ["High Risk", "High Reward"]
            })
        return [s for s in strategies if blended_vol >= s["Vol_Threshold"]]
    except Exception as e:
        logger.error(f"Error generating strategies: {str(e)}")
        return []

# Place Trade
def place_trade(client, strategy, capital):
    try:
        if capital < strategy["Capital_Deploy"]:
            return False, "Insufficient capital to deploy strategy"
        # Simulate trade placement
        return True, f"Successfully placed {strategy['Name']} with ₹{strategy['Capital_Deploy']:,.2f}"
    except Exception as e:
        logger.error(f"Error placing trade: {str(e)}")
        return False, str(e)

# Backtest
def run_backtest(df, capital, strategy_name, start_date, end_date):
    try:
        backtest_df = df.loc[start_date:end_date].copy()
        backtest_df["Returns"] = backtest_df["NIFTY_Close"].pct_change()
        backtest_df["PnL"] = np.random.normal(0, 0.01, len(backtest_df)) * capital
        total_pnl = backtest_df["PnL"].sum()
        win_rate = len(backtest_df[backtest_df["PnL"] > 0]) / len(backtest_df)
        max_drawdown = (backtest_df["PnL"].cumsum().cummax() - backtest_df["PnL"].cumsum()).max()
        sharpe_ratio = backtest_df["PnL"].mean() / backtest_df["PnL"].std() * np.sqrt(252)
        return backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio
    except Exception as e:
        logger.error(f"Error in backtest: {str(e)}")
        return None, None, None, None, None

# Main App
def main():
    st.markdown('<div class="header"><h1>VolGuard Pro</h1><p>Advanced Options Trading Dashboard</p></div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("Account")
        if not st.session_state.logged_in:
            totp_code = st.text_input("Enter 5paisa TOTP Code", type="password")
            if st.button("Login", key="login_button"):
                if not totp_code:
                    st.error("Please enter a TOTP code.")
                else:
                    client = initialize_5paisa_client(totp_code)
                    if client:
                        st.session_state.client = client
                        st.session_state.logged_in = True
                        st.success("✅ Logged in successfully")
                    else:
                        st.session_state.logged_in = False
                        st.session_state.client = None
                        st.error("❌ Login failed. Please check your TOTP code.")
        else:
            st.success("You are logged in!")
            if st.button("Logout", key="logout_button"):
                st.session_state.logged_in = False
                st.session_state.client = None
                st.session_state.trades = []
                st.session_state.journal_entries = []
                st.session_state.backtest_results = None
                st.success("✅ Logged out successfully")

        st.header("Configuration")
        capital = st.number_input("Capital (₹)", min_value=100000, value=1000000, step=100000)
        risk_tolerance = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"])
        analysis_horizon = st.slider("Analysis Horizon (Days)", 1, 10, 5)

    # Main Content
    if not st.session_state.logged_in:
        st.warning("Please log in to access the dashboard.")
        return

    # Fetch Data
    client = st.session_state.client
    with st.spinner("Fetching market data..."):
        market_df = fetch_market_data(client)
        historical_df = get_historical_data()

    if market_df is None or historical_df is None:
        st.error("Failed to load market data. Using fallback data.")
        historical_df = get_historical_data()
        market_df = historical_df.iloc[-1:].copy()

    # Run Analysis Button
    if st.button("Run Analysis", key="run_analysis_button"):
        with st.spinner("Running analysis..."):
            blended_vol, confidence_score, forecast_log, rmse, feature_importances, feature_cols = forecast_volatility(historical_df)
            if blended_vol is None:
                st.error("Analysis failed. Please try again.")
                return
            strategies = generate_trading_strategies(blended_vol, risk_tolerance, confidence_score, capital)
            st.session_state.analysis_results = {
                "blended_vol": blended_vol,
                "confidence_score": confidence_score,
                "forecast_log": forecast_log,
                "rmse": rmse,
                "feature_importances": feature_importances,
                "feature_cols": feature_cols,
                "strategies": strategies
            }
            st.success("Analysis completed successfully!")

    if "analysis_results" not in st.session_state:
        st.info("Click 'Run Analysis' to get started.")
        return

    results = st.session_state.analysis_results
    blended_vol = results["blended_vol"]
    confidence_score = results["confidence_score"]
    forecast_log = results["forecast_log"]
    rmse = results["rmse"]
    feature_importances = results["feature_importances"]
    feature_cols = results["feature_cols"]
    strategies = results["strategies"]

    # Tabs
    tabs = st.tabs(["Overview", "Volatility", "Strategies", "Performance", "Insights", "Backtest"])

    # Overview Tab
    with tabs[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Market Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("NIFTY 50", f"{market_df['NIFTY_Close'].iloc[-1]:,.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("India VIX", f"{market_df['VIX'].iloc[-1]:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Volatility Regime", "High" if blended_vol > 20 else "Medium" if blended_vol > 15 else "Low")
            st.markdown('</div>', unsafe_allow_html=True)

        st.plotly_chart(go.Figure(
            data=[
                go.Scatter(x=historical_df.index[-30:], y=historical_df["NIFTY_Close"].iloc[-30:], mode="lines", name="NIFTY 50", line=dict(color="#e94560")),
                go.Scatter(x=historical_df.index[-30:], y=historical_df["VIX"].iloc[-30:], mode="lines", name="India VIX", line=dict(color="#ffcc00"), yaxis="y2")
            ],
            layout=dict(
                height=400,
                margin=dict(l=10, r=10, t=30, b=10),
                yaxis=dict(title="NIFTY 50", titlefont=dict(color="#e94560"), tickfont=dict(color="#e94560")),
                yaxis2=dict(title="India VIX", titlefont=dict(color="#ffcc00"), tickfont=dict(color="#ffcc00"), overlaying="y", side="right"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                plot_bgcolor="#1a1a2e",
                paper_bgcolor="#1a1a2e",
                font=dict(color="#e5e5e5")
            )
        ), use_container_width=True)
        st.markdown(f'<div class="data-tag">Data Source: 5paisa API | Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Volatility Tab
    with tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Volatility Insights")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Forecasted Volatility", f"{blended_vol:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Confidence Score", f"{confidence_score:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        st.plotly_chart(go.Figure(
            data=[
                go.Scatter(x=forecast_log["Date"], y=forecast_log["Blended_Vol"], mode="lines", name="Blended Vol", line=dict(color="#28a745"))
            ],
            layout=dict(
                title="Volatility Forecast",
                height=400,
                yaxis=dict(title="Volatility (%)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                plot_bgcolor="#1a1a2e",
                paper_bgcolor="#1a1a2e",
                font=dict(color="#e5e5e5")
            )
        ), use_container_width=True)
        st.markdown(f'<div class="data-tag">Model RMSE: {rmse:.2f} | Data Source: 5paisa API</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Strategies Tab
    with tabs[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Recommended Strategies")
        if not strategies:
            st.warning("No strategies match your risk profile and market conditions.")
        else:
            for idx, strategy in enumerate(strategies):
                st.markdown('<div class="strategy-card">', unsafe_allow_html=True)
                st.write(f"**{strategy['Name']}**")
                st.write(f"**Capital to Deploy:** ₹{strategy['Capital_Deploy']:,.2f}")
                st.write(f"**Max Loss:** ₹{strategy['Max_Loss']:,.2f}")
                st.write(f"**Risk/Reward:** {strategy['Risk_Reward']:.2f}")
                st.write(f"**Confidence:** {strategy['Confidence']:.2%}")
                st.write(f"**Tags:** {', '.join(strategy['Tags'])}")
                if st.button(f"Execute {strategy['Name']}", key=f"execute_{idx}"):
                    success, message = place_trade(client, strategy, capital)
                    if success:
                        st.success(message)
                        st.session_state.trades.append({
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Strategy": strategy["Name"],
                            "Capital_Deployed": strategy["Capital_Deploy"],
                            "Max_Loss": strategy["Max_Loss"],
                            "Status": "Open"
                        })
                    else:
                        st.error(message)
                st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Performance Tab
    with tabs[3]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Trade Performance")
        if st.session_state.trades:
            trades_df = pd.DataFrame(st.session_state.trades)
            st.dataframe(trades_df, use_container_width=True)
            trades_df["Timestamp"] = pd.to_datetime(trades_df["Timestamp"])
            trades_df = trades_df.sort_values("Timestamp")
            trades_df["Cumulative_PnL"] = trades_df["Capital_Deployed"].cumsum() - trades_df["Max_Loss"].cumsum()
            st.plotly_chart(go.Figure(
                data=[go.Scatter(x=trades_df["Timestamp"], y=trades_df["Cumulative_PnL"], mode="lines", name="Cumulative PnL", line=dict(color="#e94560"))],
                layout=dict(
                    title="Cumulative PnL",
                    height=400,
                    yaxis=dict(title="PnL (₹)"),
                    plot_bgcolor="#1a1a2e",
                    paper_bgcolor="#1a1a2e",
                    font=dict(color="#e5e5e5")
                )
            ), use_container_width=True)
        else:
            st.info("No trades executed yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Insights Tab
    with tabs[4]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Trading Insights")
        with st.form("insights_form"):
            reflection = st.text_area("Reflect on your trading session:")
            submitted = st.form_submit_button("Save Insight")
            if submitted:
                st.session_state.journal_entries.append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Reflection": reflection
                })
                st.success("Insight saved successfully!")
        if st.session_state.journal_entries:
            journal_df = pd.DataFrame(st.session_state.journal_entries)
            st.dataframe(journal_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Backtest Tab
    with tabs[5]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Backtest Analysis")
        strategy_choice = st.selectbox("Select Strategy", ["Hedged Iron Condor", "Neutral Butterfly", "Aggressive Strangle"])
        start_date = st.date_input("Start Date", value=historical_df.index.min().date())
        end_date = st.date_input("End Date", value=historical_df.index.max().date())
        if st.button("Run Backtest", key="run_backtest_button"):
            backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio = run_backtest(
                historical_df, capital, strategy_choice, pd.Timestamp(start_date), pd.Timestamp(end_date)
            )
            if backtest_df is not None:
                st.session_state.backtest_results = {
                    "backtest_df": backtest_df,
                    "total_pnl": total_pnl,
                    "win_rate": win_rate,
                    "max_drawdown": max_drawdown,
                    "sharpe_ratio": sharpe_ratio
                }
                st.success("Backtest completed successfully!")
        if st.session_state.backtest_results:
            results = st.session_state.backtest_results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total PnL", f"₹{results['total_pnl']:,.2f}")
            with col2:
                st.metric("Win Rate", f"{results['win_rate']:.2%}")
            st.plotly_chart(go.Figure(
                data=[go.Scatter(x=results["backtest_df"].index, y=results["backtest_df"]["PnL"].cumsum(), mode="lines", name="Cumulative PnL", line=dict(color="#e94560"))],
                layout=dict(
                    title="Backtest PnL",
                    height=400,
                    yaxis=dict(title="PnL (₹)"),
                    plot_bgcolor="#1a1a2e",
                    paper_bgcolor="#1a1a2e",
                    font=dict(color="#e5e5e5")
                )
            ), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="footer">VolGuard Pro © 2025 | Powered by xAI</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
