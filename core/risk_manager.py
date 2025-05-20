import streamlit as st
from utils.logger import setup_logger

logger = setup_logger()

def check_risk(capital_to_deploy, max_loss, daily_pnl, atm_iv, realized_vol, total_capital, risk_settings):
    """Perform risk checks for a trade."""
    try:
        max_deployed_capital = total_capital * (risk_settings['max_exposure_pct'] / 100)
        max_loss_per_trade = total_capital * (risk_settings['max_loss_per_trade_pct'] / 100)
        daily_loss_limit = total_capital * (risk_settings['daily_loss_limit_pct'] / 100)

        new_deployed_capital = st.session_state.deployed_capital + capital_to_deploy
        new_exposure_pct = (new_deployed_capital / total_capital) * 100 if total_capital > 0 else 0
        new_daily_pnl = daily_pnl + st.session_state.daily_pnl

        vol_factor = 1.0
        if atm_iv > 0 and realized_vol > 0:
            iv_rv_ratio = atm_iv / realized_vol
            if iv_rv_ratio > 1.5:
                vol_factor = 0.7
            elif iv_rv_ratio < 0.8:
                vol_factor = 1.2

        adjusted_max_exposure = max_deployed_capital * vol_factor
        adjusted_exposure_pct = (new_deployed_capital / adjusted_max_exposure) * 100 if adjusted_max_exposure > 0 else 0

        if new_exposure_pct > risk_settings['max_exposure_pct'] or new_deployed_capital > adjusted_max_exposure:
            return "red", f"Exposure exceeds {risk_settings['max_exposure_pct']:.1f}% (adjusted: {adjusted_exposure_pct:.1f}%)! Cannot deploy ₹{capital_to_deploy:,.2f}."
        if max_loss > max_loss_per_trade:
            return "red", f"Max loss per trade exceeds ₹{max_loss_per_trade:,.2f} ({risk_settings['max_loss_per_trade_pct']}% of capital)!"
        if new_daily_pnl < -daily_loss_limit:
            return "red", f"Daily loss limit exceeded! Max loss allowed today: ₹{daily_loss_limit:,.2f}."
        if new_exposure_pct > risk_settings['max_exposure_pct'] * 0.8:
            return "yellow", f"Exposure nearing {risk_settings['max_exposure_pct']}% (current: {new_exposure_pct:.1f}%). Proceed with caution."
        return "green", "Safe to trade."
    except Exception as e:
        logger.error(f"Risk check error: {e}")
        return "red", "Risk calculation failed. Please check inputs and try again."
