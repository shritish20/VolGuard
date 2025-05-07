import streamlit as st
from datetime import datetime, timedelta
from config.settings import LOT_SIZE
from utils.data_utils import load_data
from utils.feature_utils import generate_features
from utils.forecast_utils import forecast_volatility_future
from utils.strategy_utils import generate_trading_strategy
from utils.backtest_utils import run_backtest
from utils.trade_utils import initialize_5paisa_client, square_off_positions
from ui.components import apply_custom_css, render_sidebar
from ui.snapshot_tab import render_snapshot_tab
from ui.forecast_tab import render_forecast_tab
from ui.strategy_tab import render_strategy_tab
from ui.portfolio_tab import render_portfolio_tab
from ui.journal_tab import render_journal_tab
from ui.backtest_tab import render_backtest_tab

# Initialize session state
def initialize_session_state():
    defaults = {
        "backtest_run": False,
        "backtest_results": None,
        "violations": 0,
        "journal_complete": False,
        "trades": [],
        "logged_in": False,
        "client": None,
        "real_time_market_data": None,
        "api_portfolio_data": {},
        "prepared_orders": None,
        "analysis_df": None,
        "forecast_log": None,
        "forecast_metrics": None,
        "generated_strategy": None
    }
骑兵
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Main app logic
def main():
    # Page config
    st.set_page_config(page_title="VolGuard Pro", page_icon="🛡️", layout="wide")

    # Apply custom CSS
    apply_custom_css()

    # Initialize session state
    initialize_session_state()

    # Render sidebar
    totp_code, capital, risk_tolerance, forecast_horizon, start_date, end_date, strategy_choice, run_button = render_sidebar()

    # Main execution area
    if not st.session_state.logged_in:
        st.info("Please login to 5paisa from the sidebar to proceed. You need a secrets.toml file with your API credentials.")
    else:
        st.markdown("<h1 style='color: #e94560; text-align: center;'>🛡️ VolGuard Pro: Your AI Trading Copilot</h1>", unsafe_allow_html=True)

        # Handle run button
        if run_button:
            with st.spinner("Running VolGuard Analysis... Fetching data and generating insights."):
                # Reset previous results
                st.session_state.backtest_run = False
                st.session_state.backtest_results = None
                st.session_state.violations = 0
                st.session_state.journal_complete = False
                st.session_state.prepared_orders = None

                # Load data
                df, real_data, data_source = load_data(st.session_state.client)
                st.session_state.real_time_market_data = real_data

                # Fetch portfolio data if client is available
                if data_source != "Data Load Failed" and st.session_state.client:
                    st.session_state.api_portfolio_data = fetch_all_api_portfolio_data(st.session_state.client)
                else:
                    st.session_state.api_portfolio_data = {}

                if df is not None:
                    # Generate features
                    df = generate_features(df, st.session_state.real_time_market_data, capital)
                    if df is not None:
                        st.session_state.analysis_df = df

                        # Run backtest
                        backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf = run_backtest(
                            df, capital, strategy_choice, start_date, end_date
                        )
                        st.session_state.backtest_run = True
                        st.session_state.backtest_results = {
                            "backtest_df": backtest_df,
                            "total_pnl": total_pnl,
                            "win_rate": win_rate,
                            "max_drawdown": max_drawdown,
                            "sharpe_ratio": sharpe_ratio,
                            "sortino_ratio": sortino_ratio,
                            "calmar_ratio": calmar_ratio,
                            "strategy_perf": strategy_perf,
                            "regime_perf": regime_perf
                        }

                        # Volatility forecasting
                        with st.spinner("Predicting market volatility..."):
                            forecast_log, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, feature_importances = forecast_volatility_future(
                                df, forecast_horizon
                            )
                            st.session_state.forecast_log = forecast_log
                            st.session_state.forecast_metrics = {
                                "garch_vols": garch_vols,
                                "xgb_vols": xgb_vols,
                                "blended_vols": blended_vols,
                                "realized_vol": realized_vol,
                                "confidence_score": confidence_score,
                                "rmse": rmse,
                                "feature_importances": feature_importances
                            }

                        # Generate trading strategy
                        st.session_state.generated_strategy = generate_trading_strategy(
                            df,
                            st.session_state.forecast_log,
                            st.session_state.forecast_metrics["realized_vol"],
                            risk_tolerance,
                            st.session_state.forecast_metrics["confidence_score"],
                            capital
                        )
                    else:
                        st.error("Analysis could not be completed due to feature generation failure.")
                else:
                    st.error("Analysis could not be completed due to data loading failure.")

        # Define tabs
        tabs = st.tabs(["📈 Snapshot", "🔮 Forecast", "🧠 Strategy", "💼 Portfolio", "📓 Journal", "🔬 Backtest"])

        # Render tabs
        with tabs[0]:
            render_snapshot_tab()
        with tabs[1]:
            render_forecast_tab()
        with tabs[2]:
            render_strategy_tab(capital)
        with tabs[3]:
            render_portfolio_tab(capital)
        with tabs[4]:
            render_journal_tab(capital)
        with tabs[5]:
            render_backtest_tab()

        # Footer
        st.markdown('<div class="footer">Built with ❤️ by Shritish Shukla & Salman Azim | © 2025 VolGuard</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
