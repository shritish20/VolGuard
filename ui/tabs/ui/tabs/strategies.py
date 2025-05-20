import streamlit as st
import pandas as pd
from core.strategy_engine import execute_strategy, calculate_regime_score, monte_carlo_expiry_simulation
from utils.helpers import generate_payout_chart
from ui.components import render_metric_card, render_risk_status
from utils.logger import setup_logger

logger = setup_logger()

def render_strategies_tab(run_engine=False):
    """Render the Strategies tab."""
    st.header("Strategies")
    
    if not st.session_state.volguard_data or not st.session_state.option_chain:
        st.warning("Please run VolGuard in the Snapshot tab to fetch market data.")
        return

    # Strategy Selection
    strategies = [
        "Iron_Fly", "Iron_Condor", "Short_Straddle", "Short_Strangle",
        "Bull_Put_Credit", "Bear_Call_Credit", "Jade_Lizard"
    ]
    strategy = st.selectbox("Select Strategy", strategies, help="Choose a trading strategy to execute.")
    quantity = st.number_input("Quantity (Lots)", min_value=1, max_value=100, value=1, step=1, help="Number of lots (1 lot = 75 units).")
    otm_distance = st.slider("OTM Distance", 50, 500, 50, 50, help="Distance from ATM for OTM strikes.")
    
    # Market Regime
    df = pd.DataFrame(st.session_state.volguard_data['iv_skew_data'])
    regime_score, regime, explanation = calculate_regime_score(
        st.session_state.atm_iv, st.session_state.realized_vol, df['Strike_PCR'].mean(),
        vix=15.0, iv_skew_slope=df['IV_Skew_Slope'].mean()
    )
    st.subheader("Market Regime")
    render_metric_card("Regime", regime, "assessment")
    render_metric_card("Score", f"{regime_score:.2f}", "score")
    st.markdown(f"<p style='color: #FAFAFA;'>{explanation}</p>", unsafe_allow_html=True)

    if run_engine:
        with st.spinner(f"Executing {strategy.replace('_', ' ')}..."):
            spot_price = st.session_state.volguard_data['nifty_spot']
            total_capital = st.session_state.risk_settings['total_capital']
            risk_settings = st.session_state.risk_settings
            access_token = st.session_state.get('access_token', '')
            
            order_results, total_pnl, entry_price, max_loss, legs = execute_strategy(
                access_token, st.session_state.option_chain, spot_price, strategy,
                quantity, df, total_capital, risk_settings
            )
            
            if order_results:
                st.success(f"Strategy {strategy.replace('_', ' ')} executed successfully!")
                render_metric_card("Total P&L", f"₹{total_pnl:,.2f}", "monetization_on")
                render_metric_card("Entry Price", f"₹{entry_price:,.2f}", "attach_money")
                render_metric_card("Max Loss", f"₹{max_loss:,.2f}", "warning")
                
                # Payout Chart
                payout_fig = generate_payout_chart(df, legs, spot_price)
                if payout_fig:
                    st.subheader("Payout Chart")
                    st.plotly_chart(payout_fig, use_container_width=True)
                
                # Monte Carlo Simulation
                sim_results = monte_carlo_expiry_simulation(legs, spot_price)
                if sim_results:
                    sim_df = pd.DataFrame(sim_results, columns=['P&L'])
                    st.subheader("Monte Carlo Simulation")
                    render_metric_card("Avg P&L", f"₹{sim_df['P&L'].mean():,.2f}", "calculate")
                    render_metric_card("P&L Std Dev", f"₹{sim_df['P&L'].std():,.2f}", "calculate")
                    render_metric_card("Worst 5% P&L", f"₹{sim_df['P&L'].quantile(0.05):,.2f}", "warning")
            else:
                st.error("Strategy execution failed. Check logs for details.")
