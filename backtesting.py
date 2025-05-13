import logging
import pandas as pd
import numpy as np
from datetime import datetime, date # Import date specifically

# Setup logging for this module
logger = logging.getLogger(__name__)

# === Backtesting Function ===

def run_backtest(analysis_df: pd.DataFrame, start_date: date, end_date: date, initial_capital: float, strategy_filter: str = "All Strategies"):
    """
    Runs a backtest simulation of trading strategies based on signals in the
    analysis DataFrame over a specified date range.

    Args:
        analysis_df (pd.DataFrame): DataFrame containing historical data with
                                   generated features and strategy recommendations.
                                   Must have a datetime index and columns used by
                                   the strategy logic (e.g., 'Recommended_Strategy',
                                   'PnL_Day'). Expected to be sorted by date.
        start_date (date): The start date for the backtest period (inclusive).
        end_date (date): The end date for the backtest period (inclusive).
        initial_capital (float): The starting capital for the backtest.
        strategy_filter (str): Optional. Filter backtest to days where this specific
                               strategy was recommended ('All Strategies' to include all).

    Returns:
        dict or None: A dictionary containing backtest performance metrics and
                      trade log, or None if backtesting fails.
        Example return dict:
        {
            'total_return': float,
            'sharpe_ratio': float,
            'max_drawdown': float,
            'trade_log': pd.DataFrame, # Detailed log of simulated trades
            'cumulative_pnl': pd.Series # Time series of cumulative PnL
        }
    """
    logger.info(f"Starting backtest from {start_date} to {end_date} with initial capital ₹{initial_capital}.")
    logger.info(f"Strategy filter: {strategy_filter}")

    # --- 1. Validate and Prepare Data ---
    if analysis_df is None or analysis_df.empty:
        logger.error("Analysis DataFrame is empty or None. Cannot run backtest.")
        return None

    # Ensure analysis_df has a datetime index and is sorted
    try:
        analysis_df.index = pd.to_datetime(analysis_df.index).normalize()
        analysis_df = analysis_df.sort_index()
    except Exception as e:
        logger.error(f"Error processing analysis_df index: {str(e)}. Cannot run backtest.", exc_info=True)
        return None

    # Ensure essential columns for backtesting exist
    required_cols = ['Recommended_Strategy', 'PnL_Day'] # Assuming these columns are in analysis_df
    if not all(col in analysis_df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in analysis_df.columns]
        logger.error(f"Analysis DataFrame missing required columns for backtesting: {missing_cols}. Ensure strategy generation and data processing are correct.")
        return None

    # Filter the analysis DataFrame for the specified date range
    # Use .loc with string formatted dates for robust range slicing
    try:
        backtest_df = analysis_df.loc[str(start_date):str(end_date)].copy() # Work on a copy
        logger.info(f"Filtered data for backtest period. Shape: {backtest_df.shape}")
    except KeyError:
        logger.error(f"Date range {start_date} to {end_date} not found in analysis_df index.")
        logger.info(f"Analysis DataFrame index range: {analysis_df.index.min().date()} to {analysis_df.index.max().date()}")
        return None
    except Exception as e:
        logger.error(f"Error filtering data for backtest date range: {str(e)}", exc_info=True)
        return None


    if backtest_df.empty:
        logger.warning(f"Filtered DataFrame for backtest period ({start_date} to {end_date}) is empty. No data to backtest.")
        return { # Return zero/empty results gracefully
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'trade_log': pd.DataFrame(columns=['Date', 'Strategy', 'Action', 'PnL', 'Cumulative_PnL']),
            'cumulative_pnl': pd.Series([0.0], index=[pd.to_datetime(start_date).normalize()])
        }

    # --- 2. Simulate Trading ---
    capital = initial_capital # Current capital during backtest
    cumulative_pnl = 0.0 # Cumulative PnL during backtest
    daily_pnls = [] # List to store daily PnL for cumulative calculation and metrics
    trade_log_list = [] # List to store details of simulated trades

    logger.info("Starting backtest simulation loop.")

    # Iterate through the filtered DataFrame day by day
    # Ensure index is DateTimeIndex for iteration
    if not isinstance(backtest_df.index, pd.DatetimeIndex):
        logger.error("Backtest DataFrame index is not DatetimeIndex. Cannot iterate.")
        return None

    # Assuming 'Recommended_Strategy' column exists and contains strategy names or 'None'/'No Trade'
    # Assuming 'PnL_Day' column exists and represents the synthetic daily PnL if a trade was active

    for date_index, row in backtest_df.iterrows():
        date_only = date_index.date() # Get just the date part
        # Ensure row data is accessible and expected columns exist
        recommended_strategy = row.get('Recommended_Strategy')
        daily_pnl = pd.to_numeric(row.get('PnL_Day'), errors='coerce').fillna(0.0) # Get synthetic daily PnL, default to 0 if NaN

        # Check if a strategy was recommended for this day AND if it matches the filter (if filter is active)
        if recommended_strategy and recommended_strategy != 'None' and recommended_strategy != 'No Trade':
             # Apply the strategy filter if it's not "All Strategies"
             if strategy_filter == "All Strategies" or recommended_strategy == strategy_filter:

                # --- Simulate Trade Action (Entry/Holding/Exit) ---
                # In this simplified backtest, we assume:
                # - A strategy recommendation implies being in a trade for the day.
                # - 'PnL_Day' represents the outcome of that trade for the day.
                # - We are always in the recommended trade for the full day if filtered.

                # Update cumulative PnL
                cumulative_pnl += daily_pnl
                # Log the daily PnL for later metrics
                daily_pnls.append(daily_pnl)

                # Log the simulated trade/day action
                trade_log_list.append({
                    'Date': date_only,
                    'Strategy': recommended_strategy,
                    'Action': 'Trade Active', # Simplified action
                    'PnL': daily_pnl,
                    'Cumulative_PnL': cumulative_pnl
                })
                logger.debug(f"{date_only}: Strategy '{recommended_strategy}' active. Daily PnL: {daily_pnl:.2f}, Cumulative PnL: {cumulative_pnl:.2f}")

             else:
                 # Log days where a strategy was recommended but filtered out
                 logger.debug(f"{date_only}: Strategy '{recommended_strategy}' recommended, but filtered out.")
                 daily_pnls.append(0.0) # Add 0 PnL for filtered out days
                 trade_log_list.append({ # Log a 'No Trade' action due to filter
                    'Date': date_only,
                    'Strategy': recommended_strategy, # Still log the recommended strategy
                    'Action': f'Filtered ({strategy_filter})', # Indicate it was filtered
                    'PnL': 0.0,
                    'Cumulative_PnL': cumulative_pnl # Cumulative PnL doesn't change
                 })


        else:
            # Log days where no strategy was recommended
            logger.debug(f"{date_only}: No strategy recommended or 'None'/'No Trade'.")
            daily_pnls.append(0.0) # Add 0 PnL for days with no trade
            trade_log_list.append({ # Log a 'No Trade' action
                 'Date': date_only,
                 'Strategy': recommended_strategy if recommended_strategy else 'N/A',
                 'Action': 'No Trade',
                 'PnL': 0.0,
                 'Cumulative_PnL': cumulative_pnl # Cumulative PnL doesn't change
            })


    logger.info("Backtest simulation loop finished.")

    # --- 3. Calculate Performance Metrics ---
    logger.info("Calculating backtest performance metrics.")

    # Create DataFrame for trade log
    trade_log_df = pd.DataFrame(trade_log_list)
    # Ensure 'Date' column is datetime and set as index
    if 'Date' in trade_log_df.columns:
        trade_log_df['Date'] = pd.to_datetime(trade_log_df['Date']).normalize()
        trade_log_df = trade_log_df.set_index('Date') # Set Date as index

    # Calculate Cumulative PnL Series
    # Ensure 'PnL' column exists and is numeric before calculating cumulative sum
    if 'PnL' in trade_log_df.columns:
         trade_log_df['PnL'] = pd.to_numeric(trade_log_df['PnL'], errors='coerce').fillna(0.0)
         cumulative_pnl_series = trade_log_df['PnL'].cumsum() # Calculate cumulative sum of daily PnLs
         # If trade_log_df is empty, cumulative_pnl_series will be empty. Add initial capital point.
         if cumulative_pnl_series.empty:
             cumulative_pnl_series = pd.Series([0.0], index=[pd.to_datetime(start_date).normalize()])
             logger.warning("Cumulative PnL series is empty, defaulting to initial date with 0.0 PnL.")
         # Add the initial capital as a starting point for the PnL series visualization
         cumulative_pnl_series = pd.concat([pd.Series([0.0], index=[pd.to_datetime(start_date).normalize()]), cumulative_pnl_series])
         # Remove any duplicate index entries that might occur from concat
         cumulative_pnl_series = cumulative_pnl_series.groupby(cumulative_pnl_series.index).last()
         cumulative_pnl_series = cumulative_pnl_series.sort_index() # Ensure sorted by date

    else:
         logger.error("PnL column missing from trade log. Cannot calculate cumulative PnL.")
         cumulative_pnl_series = pd.Series([0.0], index=[pd.to_datetime(start_date).normalize()])


    # Calculate Total Return
    # Total Return = (Ending Cumulative PnL) / Initial Capital
    # Handle case where trade_log_df is empty (cumulative_pnl remains 0.0)
    if not trade_log_df.empty and 'Cumulative_PnL' in trade_log_df.columns:
         # Use the last value of the Cumulative_PnL column from the trade log DataFrame
         final_cumulative_pnl = trade_log_df['Cumulative_PnL'].iloc[-1]
    else:
         final_cumulative_pnl = 0.0 # Default to 0.0 if trade log is empty

    # Calculate Total Return relative to initial capital
    total_return = (final_cumulative_pnl / initial_capital) if initial_capital != 0 else 0.0
    logger.info(f"Total Return: {total_return:.2%}")


    # Calculate Sharpe Ratio
    # Sharpe Ratio = (Average Daily Return - Risk-Free Rate) / Standard Deviation of Daily Returns
    # Assuming Risk-Free Rate = 0 for simplicity in this backtest.
    # Need a series of daily *returns*, not just PnL.
    # Daily Return = Daily PnL / Starting Capital for the day. This requires tracking daily capital.
    # OR, more simply, calculate daily returns from the *equity curve* (Cumulative PnL + Initial Capital).
    equity_curve = cumulative_pnl_series + initial_capital # Ensure initial capital is numeric
    equity_curve = pd.to_numeric(equity_curve, errors='coerce').fillna(initial_capital) # Fill NaNs in equity curve with initial capital

    daily_returns = equity_curve.pct_change().dropna() # Calculate daily percentage change (returns), drop first NaN
    # Annualize Sharpe Ratio assuming 252 trading days in a year
    annualization_factor = np.sqrt(252) # For annualizing standard deviation of daily returns

    sharpe_ratio = 0.0 # Default Sharpe Ratio
    if not daily_returns.empty:
        mean_daily_return = daily_returns.mean()
        std_daily_return = daily_returns.std()
        # Avoid division by zero if standard deviation is zero (e.g., no trading activity, all PnL is 0)
        if std_daily_return != 0:
            sharpe_ratio = (mean_daily_return / std_daily_return) * annualization_factor
        else:
             logger.warning("Standard deviation of daily returns is zero. Sharpe Ratio cannot be calculated (set to 0.0).")
             sharpe_ratio = 0.0 # Set to 0 if std dev is 0
    else:
        logger.warning("Daily returns series is empty. Sharpe Ratio cannot be calculated (set to 0.0).")
        sharpe_ratio = 0.0 # Set to 0 if no daily returns


    logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")


    # Calculate Maximum Drawdown
    # Drawdown = (Peak - Trough) / Peak
    # Max Drawdown is the largest percentage drop from a peak in the equity curve.
    # Need to handle cases where equity curve is flat or only increasing.
    max_drawdown = 0.0 # Default Max Drawdown

    if not equity_curve.empty:
        # Calculate the cumulative maximum of the equity curve
        cumulative_max = equity_curve.cummax()
        # Calculate the drawdown from the cumulative maximum
        drawdowns = (cumulative_max - equity_curve) / cumulative_max
        # Find the maximum value in the drawdowns series
        max_drawdown = drawdowns.max()
        if pd.isna(max_drawdown): max_drawdown = 0.0 # Ensure it's 0.0 if max is NaN

    logger.info(f"Maximum Drawdown: {max_drawdown:.2%}")


    # --- 4. Prepare Results Dictionary ---
    results = {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'trade_log': trade_log_df, # Return the full trade log DataFrame
        'cumulative_pnl': cumulative_pnl_series # Return the cumulative PnL Series
    }

    logger.info("Backtesting process completed successfully.")
    return results

# The __main__ block is removed as this module's functions are intended to be imported
# and used by other parts of the application.
# To test this function, you would typically call it from a separate script
# with a dummy analysis_df DataFrame containing dates, 'Recommended_Strategy',
# and 'PnL_Day' columns, along with backtest parameters.
