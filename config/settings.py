# Constants and configuration settings

LOT_SIZE = 25  # Standard NIFTY lot size

FEATURE_COLS = [
    'VIX', 'ATM_IV', 'IVP', 'PCR', 'VIX_Change_Pct', 'IV_Skew', 'Straddle_Price',
    'Spot_MaxPain_Diff_Pct', 'Days_to_Expiry', 'Event_Flag', 'FII_Index_Fut_Pos', 'FII_Option_Pos'
]

# Transaction cost factors
BASE_TRANSACTION_COST_FACTOR = 0.002
STT_FACTOR = 0.0005

# Risk-free rate
RISK_FREE_RATE_DAILY = 0.06 / 252  # Approx daily risk-free rate assuming 252 trading days

# Capital allocation percentages by regime
CAPITAL_ALLOC_PCT = {
    "LOW": 0.10,
    "MEDIUM": 0.08,
    "HIGH": 0.06,
    "EVENT-DRIVEN": 0.07
}

# Max loss percentages by strategy
MAX_LOSS_PCT = {
    "Iron Condor": 0.02,
    "Butterfly Spread": 0.015,
    "Iron Fly": 0.02,
    "Short Strangle": 0.03,
    "Calendar Spread": 0.015,
    "Jade Lizard": 0.025,
    "Short Straddle": 0.04,
    "Short Put Vertical Spread": 0.015,
    "Long Put": 0.03,
    "Short Put": 0.02,
    "Short Call Vertical Spread": 0.015
}
