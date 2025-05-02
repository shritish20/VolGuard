import pandas as pd
from datetime import datetime

def get_today_data_patched(use_live=False, client=None):
    df = pd.read_csv("nifty_50.csv")
    last_row = df.iloc[-1].copy()

    # Add VIX if available
    vix_df = pd.read_csv("india_vix.csv")
    today = datetime.now().strftime("%Y-%m-%d")
    vix_today = vix_df[vix_df["Date"] == today]
    if not vix_today.empty:
        last_row["VIX"] = vix_today["VIX"].values[0]
        last_row["VIX_Change"] = vix_today["VIX_Change"].values[0]

    # Add live values if client exists
    if use_live and client:
        from fivepaisa_live import get_volguard_live_data
        live = get_volguard_live_data(client)
        last_row["NIFTY_Close"] = live["nifty_spot"]
        last_row["ATM_IV"] = live["atm_iv"]
        last_row["StraddlePrice"] = live["straddle_price"]
        last_row["PCR"] = live["pcr"]
        last_row["MaxPainDiffPct"] = live["max_pain_diff_pct"]

    return last_row
