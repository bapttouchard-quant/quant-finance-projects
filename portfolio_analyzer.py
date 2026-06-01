import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# --- User Input ---
print("=" * 50)
print("   Multi-Asset Portfolio Analyzer")
print("=" * 50)
print("\nEnter the stock tickers you want to analyze.")
print("Type them separated by spaces (e.g. AAPL MSFT JPM)")
print("Press Enter when done.\n")

user_input = input("Your tickers: ").strip().upper()
tickers = [t for t in user_input.split() if t != "SPY"]

if len(tickers) < 2:
    print("\nPlease enter at least 2 tickers. Restart and try again.")
    exit()

start = input("\nStart date (YYYY-MM-DD, e.g. 2020-01-01): ").strip()
end = "2026-05-01"
print(f"End date set to: {end}")
print(f"\nFetching data for: {', '.join(tickers)}...")

# --- Get Data ---
all_tickers = list(set(tickers + ["SPY"]))
data = yf.download(all_tickers, start=start, end=end, auto_adjust=True)["Close"]

if isinstance(data, pd.Series):
    data = data.to_frame()

missing = [t for t in tickers if t not in data.columns or data[t].isnull().all()]
if missing:
    print(f"\nWarning: No data found for {missing}, removing from analysis.")
    tickers = [t for t in tickers if t not in missing]

data = data[[t for t in tickers + ["SPY"] if t in data.columns]]
daily_returns = data.pct_change().dropna()
spy_returns = daily_returns["SPY"].squeeze()
asset_returns = daily_returns[tickers]

# ============================================================
# 1. STATS
# ============================================================
print("\n" + "=" * 65)
print(f"{'Asset':<8} {'Ann.Return':>10} {'Volatility':>10} {'Sharpe':>8} {'Beta':>8} {'VaR 95%':>10}")
print("=" * 65)

stats = {}
for ticker in tickers:
    ret = float(asset_returns[ticker].mean()) * 252
    vol = float(asset_returns[ticker].std()) * (252 ** 0.5)
    sharpe = ret / vol
    cov = float(asset_returns[ticker].cov(spy_returns))
    beta = cov / float(spy_returns.var())
    var_95 = float(np.percentile(asset_returns[ticker], 5))
    stats[ticker] = {"ret": ret, "vol": vol, "sharpe": sharpe, "beta": beta, "var_95": var_95}
    print(f"{ticker:<8} {ret:>10.2%} {vol:>10.2%} {sharpe:>8.2f} {beta:>8.2f} {var_95:>10.2%}")

print("=" * 65)
print("VaR 95%: worst expected daily loss 95% of the time")

# ============================================================
# 2. EFFICIENT FRONTIER
# ============================================================
print("\nSimulating efficient frontier...")

n_portfolios = 5000
n_assets = len(tickers)
results = np.zeros((3, n_portfolios))
weights_record = []

mean_returns = asset_returns.mean() * 252
cov_matrix = asset_returns.cov() * 252

for i in range(n_portfolios):
    weights = np.random.random(n_assets)
    weights /= weights.sum()
    weights_record.append(weights)
    port_return = float(np.dot(weights, mean_returns))
    port_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
    sharpe = port_return / port_vol
    results[0, i] = port_vol
    results[1, i] = port_return
    results[2, i] = sharpe

max_sharpe_idx = np.argmax(results[2])
best_weights = weights_record[max_sharpe_idx]

print("\nOptimal Portfolio (Max Sharpe Ratio):")
print("-" * 30)
for ticker, weight in zip(tickers, best_weights):
    print(f"  {ticker:<8} {weight:.2%}")
print(f"\n  Expected Return:     {results[1, max_sharpe_idx]:.2%}")
print(f"  Expected Volatility: {results[0, max_sharpe_idx]:.2%}")
print(f"  Sharpe Ratio:        {results[2, max_sharpe_idx]:.2f}")

# ============================================================
# 3. DASHBOARD
# ============================================================
fig = plt.figure(figsize=(20, 26))
fig.patch.set_facecolor("#0f0f0f")
gs = GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.35)

title = ", ".join(tickers)
fig.suptitle(f"Portfolio Analysis: {title}\n{start} to {end}",
             fontsize=16, color="white", fontweight="bold", y=0.98)

colors = plt.cm.tab10(np.linspace(0, 1, len(tickers)))

def style_ax(ax, title):
    ax.set_facecolor("#1a1a1a")
    ax.set_title(title, color="white", fontsize=11, pad=10)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.grid(True, color="#2a2a2a", linewidth=0.5)

# --- Plot 1: Cumulative Returns ---
ax1 = fig.add_subplot(gs[0, :2])
cumulative_returns = (1 + asset_returns).cumprod()
for i, ticker in enumerate(tickers):
    ax1.plot(cumulative_returns[ticker], label=ticker, color=colors[i])
style_ax(ax1, "Cumulative Returns")
ax1.set_ylabel("Growth of $1")
ax1.legend(fontsize=8, labelcolor="white", facecolor="#1a1a1a", edgecolor="#444444")

# --- Plot 2: Rolling 30-Day Volatility ---
ax2 = fig.add_subplot(gs[0, 2:])
rolling_vol = asset_returns.rolling(window=30).std() * (252 ** 0.5)
for i, ticker in enumerate(tickers):
    ax2.plot(rolling_vol[ticker], label=ticker, color=colors[i])
style_ax(ax2, "Rolling 30-Day Annualized Volatility")
ax2.set_ylabel("Volatility")
ax2.legend(fontsize=8, labelcolor="white", facecolor="#1a1a1a", edgecolor="#444444")

# --- Plot 3: Drawdown ---
ax3 = fig.add_subplot(gs[1, :2])
for i, ticker in enumerate(tickers):
    cum = (1 + asset_returns[ticker]).cumprod()
    drawdown = (cum - cum.cummax()) / cum.cummax()
    ax3.plot(drawdown, label=ticker, color=colors[i])
style_ax(ax3, "Drawdown Chart")
ax3.set_ylabel("Drawdown")
ax3.legend(fontsize=8, labelcolor="white", facecolor="#1a1a1a", edgecolor="#444444")

# --- Plot 4: Correlation Matrix (double height) ---
ax4 = fig.add_subplot(gs[1:3, 2:])
corr_matrix = asset_returns.corr()
im = ax4.imshow(corr_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
cbar = plt.colorbar(im, ax=ax4)
cbar.set_label("Correlation", color="white")
cbar.ax.yaxis.set_tick_params(color="white")
cbar.ax.tick_params(colors="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
ax4.set_xticks(range(len(tickers)))
ax4.set_yticks(range(len(tickers)))
ax4.set_xticklabels(tickers, color="white", fontsize=10)
ax4.set_yticklabels(tickers, color="white", fontsize=10)
for i in range(len(tickers)):
    for j in range(len(tickers)):
        ax4.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                 ha="center", va="center", fontsize=10, color="black", fontweight="bold")
ax4.set_facecolor("#1a1a1a")
ax4.set_title("Correlation Matrix", color="white", fontsize=11, pad=10)
ax4.tick_params(colors="white")
for spine in ax4.spines.values():
    spine.set_edgecolor("#444444")

# --- Plot 5: Efficient Frontier ---
ax5 = fig.add_subplot(gs[2, :2])
scatter = ax5.scatter(results[0], results[1], c=results[2],
                      cmap="viridis", alpha=0.5, s=8)
cbar2 = plt.colorbar(scatter, ax=ax5)
cbar2.set_label("Sharpe Ratio", color="white")
cbar2.ax.yaxis.set_tick_params(color="white")
cbar2.ax.tick_params(colors="white")
plt.setp(cbar2.ax.yaxis.get_ticklabels(), color="white")
ax5.scatter(results[0, max_sharpe_idx], results[1, max_sharpe_idx],
            color="red", marker="*", s=300, label="Max Sharpe", zorder=5)
style_ax(ax5, "Efficient Frontier")
ax5.set_xlabel("Volatility")
ax5.set_ylabel("Expected Return")
ax5.legend(fontsize=8, labelcolor="white", facecolor="#1a1a1a", edgecolor="#444444")

# --- Plot 6: Optimal Weights Pie ---
ax6 = fig.add_subplot(gs[3, :2])
ax6.set_facecolor("#1a1a1a")
wedges, texts, autotexts = ax6.pie(
    best_weights,
    labels=tickers,
    autopct="%1.1f%%",
    colors=colors,
    textprops={"color": "white", "fontsize": 9}
)
for at in autotexts:
    at.set_color("white")
ax6.set_title("Optimal Portfolio Weights (Max Sharpe)", color="white", fontsize=11, pad=10)

# --- Plot 7: Stats Table ---
ax7 = fig.add_subplot(gs[3, 2:])
ax7.set_facecolor("#1a1a1a")
ax7.axis("off")
ax7.set_title("Risk & Return Summary", color="white", fontsize=11, pad=10)

col_labels = ["Ticker", "Ann. Return", "Volatility", "Sharpe", "Beta", "VaR 95%"]
table_data = [
    [
        ticker,
        f"{stats[ticker]['ret']:.2%}",
        f"{stats[ticker]['vol']:.2%}",
        f"{stats[ticker]['sharpe']:.2f}",
        f"{stats[ticker]['beta']:.2f}",
        f"{stats[ticker]['var_95']:.2%}"
    ]
    for ticker in tickers
]

table = ax7.table(
    cellText=table_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

for (row, col), cell in table.get_celld().items():
    cell.set_facecolor("#2a2a2a" if row % 2 == 0 else "#1a1a1a")
    cell.set_text_props(color="white")
    cell.set_edgecolor("#444444")

plt.savefig("portfolio_analysis.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()

print("\nDone! Dashboard saved as portfolio_analysis.png")