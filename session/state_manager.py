import streamlit as st

def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        'volguard_data': None,
        'xgb_prediction': None,
        'atm_iv': 0.0,
        'realized_vol': 0.0,
        'strategies': [],
        'journal_entries': [],
        'trade_log': [],
        'deployed_capital': 0.0,
        'daily_pnl': 0.0,
        'user_details': None,
        'option_chain': None,
        'trade_metrics': {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'pnl_history': []
        },
        'risk_settings': {
            'max_exposure_pct': 50.0,
            'max_loss_per_trade_pct': 5.0,
            'daily_loss_limit_pct': 5.0,
            'total_capital': 1000000,
            'risk_profile': 'Moderate'
        },
        'risk_status': 'green',
        'prev_oi': {}
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
