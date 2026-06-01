import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# --- Settings ---
ticker = "AAPL"
start = "2020-01-01"
end = "2024-01-01"

# --- Get Data ---
data = yf.download(ticker, start=start, end=end, auto_adjust=True)
prices = data["Close"].squeeze()

# --- Calculate Returns ---
daily_returns = prices.pct_change().dropna()
cumulative_returns = (1 + daily_returns).cumprod()

# --- Key Stats ---
annualized_return = float(daily_returns.mean()) * 252
annualized_vol = float(daily_returns.std()) * (252 ** 0.5)
sharpe_ratio = annualized_return / annualized_vol
max_drawdown = float(((cumulative_returns - cumulative_returns.cummax()) / cumulative_returns.cummax()).min())

print(f"Ticker: {ticker}")
print(f"Annualized Return: {annualized_return:.2%}")
print(f"Annualized Volatility: {annualized_vol:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")

# --- Plot ---
plt.figure(figsize=(10, 5))
plt.plot(cumulative_returns)
plt.title(f"{ticker} Cumulative Returns")
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.grid(True)
plt.tight_layout()
plt.savefig("returns_chart.png")
plt.show()