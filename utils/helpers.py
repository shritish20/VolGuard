import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

def plot_iv_skew(df, spot, atm_strike):
    """Generate IV skew plot."""
    try:
        valid = df[(df['CE_IV'] > 0) & (df['PE_IV'] > 0)]
        if valid.empty:
            return None
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=valid['Strike'], y=valid['CE_IV'], mode='lines+markers', name='Call IV', line=dict(color='#4CAF50')))
        fig.add_trace(go.Scatter(x=valid['Strike'], y=valid['PE_IV'], mode='lines+markers', name='Put IV', line=dict(color='#FFA726')))
        fig.add_vline(x=spot, line=dict(color='#FAFAFA', dash='dash'), name='Spot')
        fig.add_vline(x=atm_strike, line=dict(color='#388E3C', dash='dot'), name='ATM')
        fig.update_layout(
            title="IV Skew",
            xaxis_title="Strike",
            yaxis_title="IV (%)",
            template="plotly_dark",
            showlegend=True,
            margin=dict(l=40, r=40, t=40, b=40),
            plot_bgcolor='#121212',
            paper_bgcolor='#121212',
            font=dict(color='#FAFAFA')
        )
        return fig
    except Exception as e:
        from utils.logger import setup_logger
        logger = setup_logger()
        logger.error(f"IV skew plot error: {e}")
        return None

def generate_payout_chart(df, legs, spot):
    """Generate strategy payout chart."""
    try:
        strikes = df["Strike"].tolist()
        spot_range = np.linspace(spot * 0.95, spot * 1.05, 100)
        pnl = []

        for s in spot_range:
            total = 0
            for leg in legs:
                strike = leg.get('strike')
                qty = int(leg.get('quantity', 0)) * 75
                opt_type = 'CE' if 'CE' in leg['instrument_key'] else 'PE'
                action = leg['action']
                intrinsic = max(0, s - strike) if opt_type == "CE" else max(0, strike - s)
                payoff = -intrinsic if action == "SELL" else intrinsic
                total += payoff * qty
            pnl.append(total)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spot_range, y=pnl, mode="lines", name="P/L", line=dict(color="#4CAF50")))
        fig.add_vline(x=spot, line=dict(color="white", dash="dash"))
        fig.update_layout(
            title="Strategy Payout at Expiry",
            xaxis_title="Spot Price at Expiry",
            yaxis_title="Net P&L (₹)",
            template="plotly_dark",
            plot_bgcolor="#121212",
            paper_bgcolor="#121212",
            font=dict(color="#FAFAFA"),
            height=400
        )
        return fig
    except Exception as e:
        from utils.logger import setup_logger
        logger = setup_logger()
        logger.error(f"Payout chart error: {e}")
        return None

def compute_realized_vol(nifty_df):
    """Calculate realized volatility from Nifty data."""
    try:
        required_cols = ['NIFTY_Close']
        if not all(col in nifty_df.columns for col in required_cols):
            raise ValueError("CSV missing required column: 'NIFTY_Close'")
        log_returns = np.log(nifty_df["NIFTY_Close"].pct_change() + 1).dropna()
        last_7d_std = log_returns[-7:].std() * np.sqrt(252) * 100
        return last_7d_std if not np.isnan(last_7d_std) else 0
    except Exception as e:
        from utils.logger import setup_logger
        logger = setup_logger()
        logger.error(f"Realized vol error: {e}")
        return 0

def calculate_rolling_and_fixed_hv(nifty_close):
    """Calculate rolling and fixed historical volatility."""
    try:
        log_returns = np.log(nifty_close.pct_change() + 1).dropna()
        last_7d_std = log_returns[-7:].std()
        rolling_rv_annualized = last_7d_std * np.sqrt(252) * 100
        last_date = nifty_close.index[-1]
        future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=7)
        rv_7d_df = pd.DataFrame({
            "Date": future_dates,
            "Day": future_dates.day_name(),
            "7-Day Realized Volatility (%)": np.round([rolling_rv_annualized]*7, 2)
        })
        hv_30d = log_returns[-30:].std() * np.sqrt(252) * 100
        hv_1y = log_returns[-252:].std() * np.sqrt(252) * 100
        return rv_7d_df, hv_30d, hv_1y
    except Exception as e:
        from utils.logger import setup_logger
        logger = setup_logger()
        logger.error(f"HV calculation error: {e}")
        return pd.DataFrame(), 0, 0

def calculate_discipline_score(trade_log, regime_score_threshold=60, max_trades_per_day=3):
    """Calculate trading discipline score."""
    violations = []
    score = 100

    if not trade_log:
        return 100, violations

    trades_df = pd.DataFrame(trade_log)
    trades_df['date_only'] = pd.to_datetime(trades_df['date']).dt.date

    # Rule 1: Avoid trading in Risk-Red Regimes
    risk_trades = trades_df[trades_df['regime_score'] < regime_score_threshold]
    if not risk_trades.empty:
        violations.append(f"{len(risk_trades)} trade(s) executed during Risk-Red regime.")
        score -= len(risk_trades) * 10

    # Rule 2: Avoid Overtrading
    trade_counts = trades_df.groupby('date_only').size()
    over_trades = trade_counts[trade_counts > max_trades_per_day]
    if not over_trades.empty:
        violations.append(f"{len(over_trades)} day(s) with overtrading.")
        score -= len(over_trades) * 10

    # Rule 3: Risk-Reward Violation
    high_risk = trades_df[trades_df['max_loss'] > 0.05 * 1000000]
    if not high_risk.empty:
        violations.append(f"{len(high_risk)} high-risk trade(s) exceeding 5% capital loss.")
        score -= len(high_risk) * 5

    score = max(score, 0)
    return score, violations
