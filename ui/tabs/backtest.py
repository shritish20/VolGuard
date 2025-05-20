import streamlit as st
import pandas as pd
from core.backtest_engine import run_backtest
from ui.components import render_metric_card
from utils.logger import setup_logger

logger = setup_logger()

def render_backtest_tab():
    """Render the Backtest tab."""
    st.header("Backtest Strategies")
    
    strategies = [
        "Iron_Fly", "Iron_Condor", "Short_Straddle", "Short_Strangle",
        "Bull_Put_Credit", "Bear_Call_Credit", "Jade_Lizard"
    ]
    strategy = st.selectbox("Select Strategy", strategies, help="Choose a strategy to backtest.")
    quantity = st.number_input("Quantity (Lots)", min_value=1, max_value=100, value=1, step=1, help="Number of lots (1 lot = 75 units).")
    period = st.slider("Backtest Period (Days)", 30, 365, 90, help="Number of days to backtest.")
    
    if st.button("Run Backtest"):
        with st.spinner(f"Running backtest for {strategy.replace('_', ' ')}..."):
            backtest_df, total_pnl, win_rate, avg_pnl, max_drawdown, fig = run_backtest(strategy, quantity, period)
            if backtest_df is not None:
                st.subheader("Backtest Results")
                render_metric_card("Total P&L", f"₹{total_pnl:,.2f}", "monetization_on")
                render_metric_card("Win Rate", f"{win_rate:.2f}%", "percent")
                render_metric_card("Avg P&L per Trade", f"₹{avg_pnl:,.2f}", "calculate")
                render_metric_card("Max Drawdown", f"₹{max_drawdown:,.2f}", "warning")
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("Backtest Data"):
                    st.dataframe(backtest_df)
            else:
                st.error("Backtest failed. Check logs for details.")
