import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from core.volatility_models import run_xgboost_evaluation, run_xgboost_prediction
from utils.helpers import compute_realized_vol
from ui.components import render_metric_card, render_highlight_card
from utils.logger import setup_logger
from config.settings import NIFTY_CSV_URL

logger = setup_logger()

def render_prediction_tab():
    """Render the Prediction tab."""
    st.header("XGBoost: 7-Day Volatility Prediction")
    
    # Model Evaluation
    st.subheader("Evaluate Trained Model")
    if st.button("Run Model Evaluation"):
        with st.spinner("Loading XGBoost data and model..."):
            metrics = run_xgboost_evaluation()
            if metrics:
                st.subheader("Model Performance")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<h3>Training Metrics</h3>", unsafe_allow_html=True)
                    render_metric_card("RMSE", f"{metrics['rmse_train']:.4f}%", "bar_chart")
                    render_metric_card("MAE", f"{metrics['mae_train']:.4f}%", "bar_chart")
                    render_metric_card("R²", f"{metrics['r2_train']:.4f}", "bar_chart")
                with col2:
                    st.markdown("<h3>Test Metrics</h3>", unsafe_allow_html=True)
                    render_metric_card("RMSE", f"{metrics['rmse_test']:.4f}%", "bar_chart")
                    render_metric_card("MAE", f"{metrics['mae_test']:.4f}%", "bar_chart")
                    render_metric_card("R²", f"{metrics['r2_test']:.4f}", "bar_chart")

                # Feature Importance Plot
                fig = go.Figure()
                sorted_idx = np.argsort(metrics['feature_importances'])  # Corrected line
                fig.add_trace(go.Bar(
                    x=metrics['feature_importances'][sorted_idx],
                    y=np.array(metrics['features'])[sorted_idx],
                    orientation='h',
                    marker=dict(color='#4CAF50')
                ))
                fig.update_layout(
                    title="Feature Importance",
                    xaxis_title="Importance",
                    yaxis_title="Feature",
                    template="plotly_dark",
                    plot_bgcolor="#121212",
                    paper_bgcolor="#121212",
                    font=dict(color="#FAFAFA")
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Error loading XGBoost model or data.")

    # Volatility Prediction
    st.subheader("Predict Next 7-Day Volatility")
    with st.form("xgb_input_form"):
        col1, col2 = st.columns(2)
        with col1:
            atm_iv = st.number_input("ATM IV (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
            realized_vol = st.number_input("Realized Volatility (%)", min_value=0.0, max_value=100.0, value=12.0, step=0.1)
            ivp = st.number_input("IV Percentile", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
            event_impact = st.number_input("Event Impact Score", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
        with col2:
            fii_dii = st.number_input("FII/DII Net Long", min_value=-1000000.0, max_value=1000000.0, value=0.0, step=1000.0)
            pcr = st.number_input("PCR", min_value=0.0, max_value=10.0, value=1.0, step=0.01)
            vix = st.number_input("VIX", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
        submitted = st.form_submit_button("Predict Volatility")
        
        if submitted:
            new_data = pd.DataFrame({
                'ATM_IV': [atm_iv],
                'Realized_Vol': [realized_vol],
                'IVP': [ivp],
                'Event_Impact_Score': [event_impact],
                'FII_DII_Net_Long': [fii_dii],
                'PCR': [pcr],
                'VIX': [vix]
            })
            with st.spinner("Running XGBoost prediction..."):
                prediction = run_xgboost_prediction(new_data)
                if prediction is not None:
                    st.session_state.xgb_prediction = prediction
                    render_highlight_card("Predicted 7-Day Volatility", f"{prediction:.2f}%", "trending_up")
                    if prediction > 20:
                        st.markdown("<div class='alert-yellow'>High volatility predicted. Consider hedged strategies.</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='alert-green'>Moderate volatility predicted. Explore directional strategies.</div>", unsafe_allow_html=True)
                else:
                    st.error("Error running XGBoost prediction.")
