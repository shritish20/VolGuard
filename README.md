# VolGuard: AI-Powered Trading Copilot 🛡️

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-ff4b4b)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

**Protection First, Edge Always**  
VolGuard is an AI-driven trading copilot designed to help traders navigate market volatility with precision and discipline. Leveraging a GARCH(1,1) model for volatility forecasting, VolGuard provides actionable insights, regime-based strategies, and risk management tools to deploy capital with an edge, survive market turbulence, and outlast uncertainty.

[🌐 Try VolGuard Live](https://volguard-knftjtqn9kztizkcn7mxpa.streamlit.app/) | [📂 View Source Code](#installation)

---

## 📸 Screenshot

![VolGuard Screenshot](https://via.placeholder.com/800x400.png?text=VolGuard+Screenshot)  
*VolGuard's sleek interface displaying volatility forecasts and trading strategies.*

---

## 📜 Project Vision

VolGuard embodies the philosophy: *"We don’t predict direction—we predict conditions. We deploy edge, survive, and outlast."* It’s built for traders who prioritize capital protection while seeking consistent market edges. Whether you're conservative, moderate, or aggressive, VolGuard empowers you with data-driven insights to trade smarter.

---

## ✨ Features

- **Volatility Forecasting**  
  - Predicts market volatility using a GARCH(1,1) model, validated against industry benchmarks (e.g., 14.43% to 15.10% over 6 days).  
  - Displays daily volatility breakdowns with a sleek line chart.

- **Regime Classification**  
  - Classifies market conditions into LOW, MEDIUM, HIGH, or EVENT-DRIVEN regimes based on volatility levels.  
  - Visual indicators for quick regime identification.

- **Trading Strategy Recommendations**  
  - Suggests strategies like Butterfly Spread, Short Strangle, or Jade Lizard based on market conditions.  
  - Includes confidence scores, risk-reward ratios, and capital deployment suggestions.

- **Risk Management**  
  - Flags potential risks (e.g., high VIX spikes, excessive drawdowns) to protect your capital.  
  - Behavioral monitoring with a discipline score and actionable warnings.

- **Interactive UI**  
  - Built with Streamlit for a modern, user-friendly experience.  
  - Export forecasts and strategies as CSV files for further analysis.

- **Journaling**  
  - Reflect on your trading discipline with a built-in journaling prompt.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python  
  - `yfinance`: Fetches NIFTY 50 data.  
  - `arch`: Implements GARCH volatility modeling.  
  - `pandas`, `numpy`: Data manipulation.  
  - `scipy`, `xgboost`, `scikit-learn`: Analytics and synthetic features.  
- **Data Sources**:  
  - NIFTY 50 historical data (`^NSEI`) via Yahoo Finance.  
  - India VIX data from a GitHub-hosted CSV.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package manager)

Install the required dependencies:
```bash
pip install streamlit pandas numpy scipy yfinance arch xgboost scikit-learn
