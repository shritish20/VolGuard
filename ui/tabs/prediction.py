import streamlit as st
import pandas as pd
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

                fig = go.Figure()
                sorted_idx = np.argsort(metrics['feature
