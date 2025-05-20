import streamlit as st
import plotly.graph_objects as go

def render_top_bar():
    """Render the top bar with quick stats."""
    total_capital = st.session_state.risk_settings['total_capital']
    exposure_pct = (st.session_state.deployed_capital / total_capital) * 100 if total_capital > 0 else 0
    st.markdown(f"""
        <div class='top-bar'>
            <div><i class="material-icons">percent</i><p>Exposure: {exposure_pct:.1f}%</p></div>
            <div><i class="material-icons">monetization_on</i><p>Daily P&L: ₹{st.session_state.daily_pnl:,.2f}</p></div>
            <div><i class="material-icons">warning</i><p>Risk Status: {st.session_state.risk_status.capitalize()}</p></div>
        </div>
    """, unsafe_allow_html=True)

def render_risk_status(risk_status, risk_message):
    """Render risk status alert."""
    if risk_status == "green":
        st.markdown(f"<div class='alert-green'>{risk_message}</div>", unsafe_allow_html=True)
    elif risk_status == "yellow":
        st.markdown(f"<div class='alert-yellow'>{risk_message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='alert-red'>{risk_message}</div>", unsafe_allow_html=True)

def render_metric_card(title, value, icon="info"):
    """Render a metric card."""
    st.markdown(f"""
        <div class='metric-card'>
            <h4><i class='material-icons'>{icon}</i> {title}</h4>
            <p>{value}</p>
        </div>
    """, unsafe_allow_html=True)

def render_highlight_card(title, value, icon="trending_up"):
    """Render a highlight card."""
    st.markdown(f"""
        <div class='highlight-card'>
            <h4><i class='material-icons'>{icon}</i> {title}</h4>
            <p>{value}</p>
        </div>
    """, unsafe_allow_html=True)
