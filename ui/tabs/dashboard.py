import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.helpers import calculate_discipline_score
from ui.components import render_metric_card, render_highlight_card
from utils.logger import setup_logger

logger = setup_logger()

def render_dashboard_tab():
    """Render the Dashboard tab."""
    st.header("Trading Dashboard")
    
    # Trade Metrics
    metrics = st.session_state.trade_metrics
    st.subheader("Performance Metrics")
    col1, col2 = st.columns(2)
    with col1:
        render_metric_card("Total Trades", metrics['total_trades'], "shopping_cart")
        render_metric_card("Winning Trades", metrics['winning_trades'], "thumb_up")
        render_metric_card("Losing Trades", metrics['losing_trades'], "thumb_down")
    with col2:
        render_metric_card("Total P&L", f"₹{metrics['total_pnl']:,.2f}", "monetization_on")
        win_rate = (metrics['winning_trades'] / metrics['total_trades'] * 100) if metrics['total_trades'] > 0 else 0
        render_metric_card("Win Rate", f"{win_rate:.2f}%", "percent")
        avg_pnl = metrics['total_pnl'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0
        render_metric_card("Avg P&L per Trade", f"₹{avg_pnl:,.2f}", "calculate")

    # P&L History Plot
    if metrics['pnl_history']:
        pnl_df = pd.DataFrame(metrics['pnl_history'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pnl_df['timestamp'], y=pnl_df['pnl'].cumsum(),
            mode='lines', name='Cumulative P&L',
            line=dict(color='#4CAF50')
        ))
        fig.update_layout(
            title="Cumulative P&L Over Time",
            xaxis_title="Timestamp",
            yaxis_title="Cumulative P&L (₹)",
            template="plotly_dark",
            plot_bgcolor="#121212",
            paper_bgcolor="#121212",
            font=dict(color="#FAFAFA")
        )
        st.plotly_chart(fig, use_container_width=True)

    # Trading Discipline
    discipline_score, violations = calculate_discipline_score(st.session_state.trade_log)
    st.subheader("Trading Discipline")
    render_highlight_card("Discipline Score", f"{discipline_score:.2f}", "verified")
    if violations:
        st.markdown("<h4>Violations</h4>", unsafe_allow_html=True)
        for violation in violations:
            st.markdown(f"<div class='alert-red'>{violation}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='alert-green'>No trading discipline violations detected.</div>", unsafe_allow_html=True)
