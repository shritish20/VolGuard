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
InstallationClone the repository:git clone https://github.com/shritish20/VolGuard.git
cd VolGuardEnsure the app.py file (provided in the repository) is in the project directory.Run the Streamlit app:streamlit run app.pyOpen your browser and navigate to:http://localhost:8501
📖 UsageConfigure SettingsAdjust the Forecast Horizon (default: 6 days).Enter your Capital (e.g., ₹1,000,000).Select your Risk Profile (Conservative, Moderate, Aggressive).Activate VolGuardClick the "Activate VolGuard" button to generate forecasts and strategies.Explore InsightsVolatility Forecast: View GARCH predictions (e.g., 14.43% to 15.10% over 6 days).Regime & Strategy: Analyze the market regime and recommended trading strategy.Risk Flags: Check for warnings (e.g., high VIX spikes).Export: Download forecasts and strategies as CSV files.Journal: Reflect on your trading discipline.Example OutputVolatility Forecast (6-day horizon):28-Apr-2025: 14.43%29-Apr-2025: 14.78%30-Apr-2025: 14.95%01-May-2025: 15.04%02-May-2025: 15.08%05-May-2025: 15.10%Regime: LOW (Avg Vol: 14.90%)Strategy: Butterfly SpreadReason: Low volatility and short expiry favor pinning strategies.Capital to Deploy: ₹350,000 (for a ₹1M portfolio, Moderate risk)
.🔍 MethodologyVolatility Forecasting:Uses a GARCH(1,1) model on NIFTY 50 log returns, with default rescaling to match industry benchmarks (validated against a Colab implementation).Annualized using √252 for trading days.Regime Classification:LOW: <15% volatilityMEDIUM: 15–20% volatilityHIGH: >20% volatilityEVENT-DRIVEN: Triggered by event flags (e.g., policy days, expiries).Strategy Engine:Maps regimes to strategies using market signals (e.g., IV-HV gaps, skew, expiry dynamics).Balances risk and reward with confidence scores.Risk Filters:Monitors VIX spikes, drawdowns, and exposure to ensure capital protection.
📈 Future EnhancementsTelegram Alerts: Real-time notifications for high-risk conditions.PDF Export: Generate professional PDF reports of insights.Visual Risk Dashboard: Interactive visualizations for risk and portfolio metrics.XGBoost Integration: Reintroduce XGBoost for blended volatility forecasts.
🤝 ContributingWe welcome contributions! To get started:Fork the repository.Create a feature branch:git checkout -b feature/YourFeatureCommit your changes:git commit -m "Add YourFeature"Push to the branch:git push origin feature/YourFeatureOpen a pull request.
📜 LicenseThis project is licensed under the MIT License. See the LICENSE file for details.
📧 ContactFor questions or feedback, reach out to Shritish Shukla at shritish@amityonline.com
🙌 AcknowledgmentsDevelopers: Shritish Shukla & Salman
Inspiration: Built to address the need for disciplined, edge-driven trading in volatile markets.VolGuard: Deploy with edge, survive, outlast.---

### How to Use
1. **Copy the Text**: Copy the entire block above.
2. **Create `README.md`**:
   - In your GitHub repository, go to the root directory.
   - Click "Add file" > "Create new file".
   - Name the file `README.md`.
3. **Paste the Content**: Paste the copied text into the editor.
4. **Commit the File**:
   - Add a commit message like "Add README.md".
   - Click "Commit new file".
5. **Verify**: The README will render automatically on your repository’s main page.
