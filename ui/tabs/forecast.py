import streamlit as st
from core.volatility_models import run_garch_forecast
from ui.components import render_metric_card, render_risk_status
from utils.logger import setup_logger

logger = setup_logger()

def render_forecast_tab():
    """Render the Forecast tab."""
    st.header("GARCH: 7-Day Volatility Forecast")
    forecast_df, realized_vol, hv_30d, hv_1y = run_garch_forecast()
    if forecast_df is not None:
        st.session_state.realized_vol = realized_vol
        st.subheader("GARCH Volatility Forecast")
        for idx, row in forecast_df.iterrows():
            render_metric_card(
                f"{row['Date'].strftime('%Y-%m-%d')} ({row['Day']})",
                f"Forecasted Volatility: {row['Forecasted Volatility (%)']}%",
                "event"
            )

        avg_vol = forecast_df["Forecasted Volatility (%)"].mean()
        st.subheader("Trading Insight")
        if avg_vol > 20:
            st.markdown("<div class='alert-yellow'>Volatility is high (>20%). Consider defensive strategies like Iron Condor.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='alert-green'>Volatility is moderate. Explore strategies like Jade Lizard.</div>", unsafe_allow_html=True)

        st.subheader("Historical Volatility")
        render_metric_card("30-Day HV", f"{hv_30d:.2f}%", "history")
        render_metric_card("1-Year HV", f"{hv_1y:.2f}%", "history")
    else:
        st.error("Error loading GARCH data. Please check the CSV source.")
