import streamlit as st
from config.settings import set_page_config, apply_custom_css
from session.state_manager import initialize_session_state
from ui.components import render_top_bar
from ui.tabs.snapshot import render_snapshot_tab
from ui.tabs.forecast import render_forecast_tab
from ui.tabs.prediction import render_prediction_tab
from ui.tabs.strategies import render_strategies_tab
from ui.tabs.dashboard import render_dashboard_tab
from ui.tabs.journal import render_journal_tab
from ui.tabs.backtest import render_backtest_tab

def main():
    # Set page configuration and custom CSS
    set_page_config()
    apply_custom_css()

    # Initialize session state
    initialize_session_state()

    # Sidebar controls
    st.sidebar.header("VolGuard Pro 2.0 - Trading Copilot")
    total_capital = st.sidebar.slider("Total Capital (₹)", 100000, 5000000, 1000000, 10000, help="Your total trading capital.")
    risk_profile = st.sidebar.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"], help="Your risk tolerance for strategy recommendations.")
    st.sidebar.subheader("Risk Management Settings")
    max_exposure_pct = st.sidebar.slider("Max Exposure (%)", 10.0, 100.0, st.session_state.risk_settings['max_exposure_pct'], 1.0, help="Maximum capital to deploy at any time.")
    max_loss_per_trade_pct = st.sidebar.slider("Max Loss per Trade (%)", 1.0, 10.0, st.session_state.risk_settings['max_loss_per_trade_pct'], 0.1, help="Maximum loss allowed per trade.")
    daily_loss_limit_pct = st.sidebar.slider("Daily Loss Limit (%)", 1.0, 10.0, st.session_state.risk_settings['daily_loss_limit_pct'], 0.1, help="Maximum daily loss allowed.")
    run_engine = st.sidebar.button("Run Engine", help="Generate strategy recommendations.")

    # Update risk settings
    st.session_state.risk_settings.update({
        'max_exposure_pct': max_exposure_pct,
        'max_loss_per_trade_pct': max_loss_per_trade_pct,
        'daily_loss_limit_pct': daily_loss_limit_pct,
        'total_capital': total_capital,
        'risk_profile': risk_profile
    })

    # Render top bar
    render_top_bar()

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Snapshot", "Forecast", "Prediction", "Strategies", "Dashboard", "Journal", "Backtest"
    ])

    # Render each tab
    with tab1:
        render_snapshot_tab()
    with tab2:
        render_forecast_tab()
    with tab3:
        render_prediction_tab()
    with tab4:
        render_strategies_tab(run_engine=run_engine)
    with tab5:
        render_dashboard_tab()
    with tab6:
        render_journal_tab()
    with tab7:
        render_backtest_tab()

    # Footer
    st.markdown("""
        <div style='text-align: center; margin-top: 20px;'>
            <p style='color: #FAFAFA;'>VolGuard Pro 2.0 - Built with ❤️ by Shritish Shukla & Salman Azim</p>
            <p style='color: #FAFAFA;'>For support, contact: shritish@example.com</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
