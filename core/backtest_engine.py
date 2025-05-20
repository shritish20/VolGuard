import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from config.settings import NIFTY_CSV_URL
from core.strategy_engine import build_strategy_legs, find_atm_strike
from utils.logger import setup_logger

logger = setup_logger()

def run_backtest(strategy_name, quantity, period):
    """Run backtest for a strategy."""
    try:
        nifty_df = pd.read_csv(NIFTY_CSV_URL)
        nifty_df.columns = nifty_df.columns.str.strip()
        nifty_df["Date"] = pd.to_datetime(nifty_df["Date"], format="%d-%b-%Y", errors="coerce")
        nifty_df = nifty_df.dropna(subset=["Date"]).set_index("Date")
        nifty_df = nifty_df.rename(columns={"Close": "NIFTY_Close"})
        if 'NIFTY_Close' not in nifty_df.columns:
            raise ValueError("CSV missing 'NIFTY_Close' column")
        nifty_df = nifty_df[["NIFTY_Close"]].dropna().sort_index()

        backtest_data = nifty_df.tail(period)
        simulated_pnl = []
        for idx, row in backtest_data.iterrows():
            spot = row['NIFTY_Close']
            strikes = np.arange(spot - 500, spot + 500, 50)
            atm_strike = find_atm_strike(spot, strikes)
            legs = build_strategy_legs(
                [{'strike_price': s, 'call_options': {'instrument_key': f'NSE_FO|CALL_{s}'}, 'put_options': {'instrument_key': f'NSE_FO|PUT_{s}'}} for s in strikes],
                spot, strategy_name, quantity
            )
            expiry_spot = spot * (1 + np.random.normal(0, 0.02))
            total_pnl = 0
            for leg in legs:
                strike = leg['strike']
                qty = int(leg.get('quantity', 0)) * 75
                opt_type = 'CE' if 'CE' in leg['instrument_key'] else 'PE'
                action = leg['action']
                intrinsic = max(0, expiry_spot - strike) if opt_type == "CE" else max(0, strike - expiry_spot)
                payoff = -intrinsic if action == "SELL" else intrinsic
                total_pnl += payoff * qty
            simulated_pnl.append(total_pnl)

        backtest_df = pd.DataFrame({
            'Date': backtest_data.index,
            'P&L': simulated_pnl
        })
        total_pnl = backtest_df['P&L'].sum()
        win_rate = (backtest_df['P&L'] > 0).mean() * 100
        avg_pnl = backtest_df['P&L'].mean()
        max_drawdown = (backtest_df['P&L'].cumsum().cummax() - backtest_df['P&L'].cumsum()).max()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=backtest_df['Date'], y=backtest_df['P&L'].cumsum(),
            mode='lines', name='Cumulative P&L',
            line=dict(color='#4CAF50')
        ))
        fig.update_layout(
            title=f"Backtest: {strategy_name.replace('_', ' ')}",
            xaxis_title="Date",
            yaxis_title="Cumulative P&L (₹)",
            template="plotly_dark",
            plot_bgcolor="#121212",
            paper_bgcolor="#121212",
            font=dict(color="#FAFAFA")
        )
        return backtest_df, total_pnl, win_rate, avg_pnl, max_drawdown, fig
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return None, 0, 0, 0, 0, None
