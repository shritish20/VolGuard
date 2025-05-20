import streamlit as st

def set_page_config():
    st.set_page_config(page_title="VolGuard Pro 2.0 - AI Trading Copilot", layout="wide")

def apply_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
        body {
            font-family: 'Roboto', sans-serif;
        }
        .stApp {
            background: #121212;
            color: #FAFAFA;
        }
        .css-1d391kg {
            background: #1E1E1E;
            padding: 20px;
            border-right: 1px solid #4CAF50;
        }
        .css-1d391kg h1 {
            color: #4CAF50;
            font-size: 1.6em;
            margin-bottom: 20px;
        }
        .css-1d391kg .stButton>button {
            background: #4CAF50;
            color: #FAFAFA;
            border-radius: 8px;
            padding: 10px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        .css-1d391kg .stButton>button:hover {
            background: #388E3C;
            box-shadow: 0 0 15px rgba(76, 175, 80, 0.5);
        }
        .top-bar {
            background: #1E1E1E;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
            display: flex;
            justify-content: space-between;
            align-items: center;
            border: 1px solid #4CAF50;
        }
        .top-bar div {
            margin: 0 15px;
            display: flex;
            align-items: center;
        }
        .top-bar div p {
            margin: 0 0 0 8px;
            font-size: 1.1em;
            color: #FAFAFA;
        }
        .top-bar i {
            color: #4CAF50;
        }
        .stTabs [role="tab"] {
            background: transparent;
            color: #FAFAFA;
            border-bottom: 2px solid transparent;
            padding: 10px 20px;
            margin-right: 5px;
            transition: all 0.3s ease;
        }
        .stTabs [role="tab"][aria-selected="true"] {
            border-bottom: 2px solid #4CAF50;
            color: #4CAF50;
        }
        .stTabs [role="tab"]:hover {
            color: #FFA726;
        }
        .stButton>button {
            background: #4CAF50;
            color: #FAFAFA;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background: #388E3C;
            box-shadow: 0 0 15px rgba(76, 175, 80, 0.5);
        }
        .metric-card {
            background: #1E1E1E;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
            border: 1px solid #4CAF50;
            transition: transform 0.3s ease;
            width: 100%;
            max-width: 600px;
        }
        .metric-card:hover {
            transform: scale(1.02);
        }
        .metric-card h4 {
            color: #4CAF50;
            margin: 0;
            display: flex;
            align-items: center;
        }
        .metric-card h4 i {
            margin-right: 8px;
            color: #FFA726;
        }
        .metric-card p {
            color: #FAFAFA;
            font-size: 1.1em;
            margin: 5px 0 0 0;
        }
        .highlight-card {
            background: #4CAF50;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 4px 10px rgba(76, 175, 80, 0.5);
            border: 1px solid #388E3C;
            width: 100%;
            max-width: 600px;
        }
        .highlight-card h4 {
            color: #FAFAFA;
            margin: 0;
            display: flex;
            align-items: center;
        }
        .highlight-card h4 i {
            margin-right: 8px;
            color: #FAFAFA;
        }
        .highlight-card p {
            color: #FAFAFA;
            font-size: 1.3em;
            margin: 5px 0 0 0;
        }
        .alert-green {
            background-color: #388E3C;
            color: #FAFAFA;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .alert-yellow {
            background-color: #FFA726;
            color: #121212;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .alert-red {
            background-color: #EF5350;
            color: #FAFAFA;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        h1, h2, h3, h4 {
            color: #4CAF50;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes scaleUp {
            from { opacity: 0; transform: scale(0.95); }
            to { opacity: 1; transform: scale(1); }
        }
        .tooltip {
            position: relative;
            display: inline-block;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #1E1E1E;
            color: #FAFAFA;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        @media (max-width: 600px) {
            .metric-card, .highlight-card {
                max-width: 100%;
            }
            .top-bar {
                flex-direction: column;
                align-items: flex-start;
            }
            .top-bar div {
                margin: 5px 0;
            }
        }
        </style>
        <link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons">
    """, unsafe_allow_html=True)

# API settings
UPSTOX_BASE_URL = "https://api.upstox.com/v2"
NIFTY_CSV_URL = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
XGB_MODEL_URL = "https://drive.google.com/uc?export=download&id=1Gs86p1p8wsGe1lp498KC-OVn0e87Gv-R"
XGB_CSV_URL = "https://raw.githubusercontent.com/shritish20/VolGuard/main/synthetic_volguard_dataset.csv"
INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"
