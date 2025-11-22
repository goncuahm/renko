# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import floor

st.set_page_config(page_title="Renko Strategy Backtester (Long+Short + TV Breakout)", layout="wide")
st.title("🧱 Renko Strategy Backtester — Long & Short (TradingView-style breakouts)")

# ------------------------------
# Sidebar Parameters
# ------------------------------
st.sidebar.header("🔧 User Parameters")

ticker = st.sidebar.text_input("Ticker Symbol (e.g., XU030.IS)", "XU030.IS")
period = st.sidebar.selectbox("Data Period", ["6mo", "1y", "2y", "3y", "5y"], index=1)
tcost = st.sidebar.number_input("Transaction Cost per Trade (e.g., 0.002 = 0.2%)", value=0.0020, step=0.0005)
capital = st.sidebar.number_input("Initial Capital (TRY)", value=1_000_000, step=50_000)

st.sidebar.markdown("### Renko Parameters")
method = st.sidebar.selectbox("Method", ["ATR", "Traditional"])
atr_period = st.sidebar.slider("ATR Period", 5, 50, 14)
brick_size = st.sidebar.number_input("Traditional Brick Size (price units)", value=10.0, step=0.5, format="%.6f")
source_choice = st.sidebar.selectbox("Source for Renko price", ["close", "hl"])
reversal = st.sidebar.number_input("Reversal (bricks)", min_value=1, max_value=5, value=2)
length = st.sidebar.slider("Length for Breakout (number of bricks)", 1, 4, 1)

st.sidebar.markdown("---")
st.sidebar.caption("Strategy: go LONG on confirmed Renko up-breakout, go SHORT on confirmed Renko down-breakout. Exits occur on opposite breakouts.")

# ------------------------------
# Data download
# ------------------------------
st.subheader(f"📊 Downloading data for: {ticker}")
df = yf.download(ticker, period=period, auto_adjust=True)

if df.empty:
    st.error("⚠️ No data found for the selected ticker.")
    st.stop()

df = df.dropna()[["Open", "High", "Low", "Close", "Volume"]].copy()
df.index = pd.to_datetime(df.index)

# ------------------------------
# Helper: ATR
# ------------------------------
def compute_atr(df, n=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, min_periods=n).mean()
    return atr

# ------------------------------
# Build Renko bricks (iterative)
# ------------------------------
def build_renko(df, box, source="close", reversal=2):
    """
    Build renko bricks similar to TradingView script:
    returns renko DataFrame indexed by the date (bar when brick(s) were formed)
    columns: brick_open, brick_close, trend (1 up, -1 down), brick_run (run length)
    """
    if source == "close":
        prices = df["Close"].values
    elif source == "hl":
        prices = ((df["High"] + df["Low"]) / 2).values
    else:
        prices = df["Close"].values

    dates = df.index.to_list()
    bricks = []

    # initial beginprice
    beginprice = floor(df["Open"].iloc[0] / box) * box
    trend = 0  # unknown
    for i, p in enumerate(prices):
        currentprice = p

        if trend == 0:
            if abs(beginprice - currentprice) >= box * reversal:
                numcell = int(abs(beginprice - currentprice) // box)
                if beginprice > currentprice:
                    # down bricks
                    for b in range(numcell):
                        o = beginprice - b * box
                        c = o - box
                        bricks.append({"date": dates[i], "brick_open": o, "brick_close": c, "trend": -1})
                    beginprice = bricks[-1]["brick_close"]
                    trend = -1
                else:
                    for b in range(numcell):
                        o = beginprice + b * box
                        c = o + box
                        bricks.append({"date": dates[i], "brick_open": o, "brick_close": c, "trend": 1})
                    beginprice = bricks[-1]["brick_close"]
                    trend = 1
            else:
                continue
        else:
            # trending up
            if trend == 1:
                # add consecutive up bricks while possible
                while currentprice - beginprice >= box:
                    o = beginprice
                    c = beginprice + box
                    bricks.append({"date": dates[i], "brick_open": o, "brick_close": c, "trend": 1})
                    beginprice = c
                # check reversal to down
                if beginprice - currentprice >= box * reversal:
                    numcell = int((beginprice - currentprice) // box)
                    for b in range(1, numcell + 1):
                        o = beginprice - (b - 1) * box
                        c = o - box
                        bricks.append({"date": dates[i], "brick_open": o, "brick_close": c, "trend": -1})
                    beginprice = bricks[-1]["brick_close"]
                    trend = -1
            # trending down
            else:
                while beginprice - currentprice >= box:
                    o = beginprice
                    c = beginprice - box
                    bricks.append({"date": dates[i], "brick_open": o, "brick_close": c, "trend": -1})
                    beginprice = c
                # check reversal to up
                if currentprice - beginprice >= box * reversal:
                    numcell = int((currentprice - beginprice) // box)
                    for b in range(1, numcell + 1):
                        o = beginprice + (b - 1) * box
                        c = o + box
                        bricks.append({"date": dates[i], "brick_open": o, "brick_close": c, "trend": 1})
                    beginprice = bricks[-1]["brick_close"]
                    trend = 1

    renko = pd.DataFrame(bricks)
    if renko.empty:
        return renko

    renko = renko.reset_index(drop=True)
    renko["brick_open"] = renko["brick_open"].astype(float)
    renko["brick_close"] = renko["brick_close"].astype(float)
    renko["trend"] = renko["trend"].astype(int)

    # compute contiguous run lengths
    run_lengths = []
    run = 0
    last = None
    for t in renko["trend"].values:
        if t == last:
            run += 1
        else:
            run = 1
            last = t
        run_lengths.append(run)
    renko["brick_run"] = run_lengths

    # add index as datetime
    renko["date"] = pd.to_datetime(renko["date"])
    renko = renko.set_index("date")
    return renko

# ------------------------------
# TradingView-style Brick breakout functions (approximate port)
# ------------------------------
def f_brickhigh(renko, idx, box, Length):
    """
    Approximate port of the TradingView f_Brickhigh logic.
    Starting from brick index idx where trend==1, compute whether
    the accumulated count across prior alternating bricks reaches Length.
    """
    # require that current brick trend is up
    if renko["trend"].iloc[idx] != 1:
        return False
    # initial count: number of bricks in current positive run minus 0
    # In Pine they use floor((iclose - iopen)/box) - 1; for our single-brick representation that equates to run-1 sometimes.
    _l = max(0, int(renko["brick_run"].iloc[idx]) - 1)
    # if already enough
    if _l >= Length:
        return True
    # walk backward through bricks (like Pine)
    # we will iterate backward and add counts when encountering prior bricks, similar to the original loop
    for x in range(idx - 1, -1, -1):
        # if trend changed between x and x+1:
        if renko["trend"].iloc[x] != renko["trend"].iloc[x + 1]:
            # if the next segment (x+1) was up and we are searching for more up bricks, accumulate length
            if renko["trend"].iloc[x + 1] == 1:
                # count bricks in that segment
                cnt = int(renko["brick_run"].iloc[x + 1])
                _l += cnt
            # if next segment (x+1) was down, we need to estimate how many up bricks can be inferred through its reversal
            else:
                # for down segment, we can compute how many "would-be" up bricks lie beyond the current iclose
                # approximate by checking if previous down bricks crossed beyond current brick_close
                # We'll conservatively add 0 for down segments (safe approximation)
                pass
        # early exit
        if _l >= Length:
            return True
    return False

def f_bricklow(renko, idx, box, Length):
    if renko["trend"].iloc[idx] != -1:
        return False
    _l = max(0, int(renko["brick_run"].iloc[idx]) - 1)
    if _l >= Length:
        return True
    for x in range(idx - 1, -1, -1):
        if renko["trend"].iloc[x] != renko["trend"].iloc[x + 1]:
            if renko["trend"].iloc[x + 1] == -1:
                cnt = int(renko["brick_run"].iloc[x + 1])
                _l += cnt
        if _l >= Length:
            return True
    return False

# ------------------------------
# Decide box size
# ------------------------------
if method == "ATR":
    atr = compute_atr(df, atr_period)
    box = float(max(atr.dropna().iloc[-1], 1e-12))
    st.info(f"Using ATR method. ATR({atr_period}) latest = {box:.6f} → brick size (box) = {box:.6f}")
else:
    box = float(brick_size)
    st.info(f"Using Traditional brick size = {box:.6f}")

# ------------------------------
# Build renko bricks
# ------------------------------
with st.spinner("Building Renko bricks..."):
    renko = build_renko(df, box=box, source=source_choice, reversal=int(reversal))

if renko.empty:
    st.warning("No Renko bricks were generated. Try smaller brick size or longer data period.")
    st.stop()

st.write(f"Generated {len(renko)} Renko bricks (last date {renko.index[-1].date()})")

# ------------------------------
# Use TV-like breakout logic to create 'breakout signal' on renko bricks
#  - setA -> breakout uptrend started  (1)
#  - setB -> breakout downtrend started (-1)
# ------------------------------
signals = pd.Series(0, index=renko.index)
for i in range(len(renko)):
    if renko["trend"].iloc[i] == 1:
        if f_brickhigh(renko, i, box, length):
            signals.iloc[i] = 1
    elif renko["trend"].iloc[i] == -1:
        if f_bricklow(renko, i, box, length):
            signals.iloc[i] = -1
renko["signal"] = signals

# ------------------------------
# Backtest: Long & Short logic
# ------------------------------
trades = []
position = 0  # 0 flat, 1 long, -1 short
entry_price = None
entry_date = None
cash = float(capital)
equity_ts = []  # list of (date, equity)
shares = 0

# iterate over renko bricks in chronological order
for dt, row in renko.iterrows():
    sig = int(row["signal"])
    price = float(row["brick_close"])

    # LONG entry
    if sig == 1 and position == 0:
        # buy with all capital
        shares = int(cash // price)
        if shares > 0:
            entry_price = price
            entry_date = dt
            position = 1
            cash -= shares * price * (1 + tcost)
            trades.append({"buy_date": entry_date, "buy_price": entry_price,
                           "sell_date": None, "sell_price": None,
                           "shares": shares, "side": "long", "pnl": None})
    # LONG exit (close long) if signal -1
    elif sig == -1 and position == 1:
        sell_price = price
        sell_date = dt
        # close long
        cash += shares * sell_price * (1 - tcost)
        entry = trades[-1]
        pnl = (sell_price - entry["buy_price"]) * shares - (entry["buy_price"] * shares * tcost + sell_price * shares * tcost)
        entry.update({"sell_date": sell_date, "sell_price": sell_price, "pnl": pnl})
        position = 0
        shares = 0

    # SHORT entry
    if sig == -1 and position == 0:
        # short with all capital; compute shares borrowed = floor(capital / price)
        shares = int(cash // price)
        if shares > 0:
            entry_price = price
            entry_date = dt
            position = -1
            # when shorting, we assume proceeds are credited to cash but we keep margin = not modeled — simple sim:
            cash += shares * price * (1 - tcost)  # receive proceeds net of tcost for initiating short
            trades.append({"buy_date": entry_date, "buy_price": entry_price,
                           "sell_date": None, "sell_price": None,
                           "shares": shares, "side": "short", "pnl": None})
    # SHORT exit (buy to cover) when sig == 1
    elif sig == 1 and position == -1:
        cover_price = price
        cover_date = dt
        # close short: we pay shares * price and pay tcost
        cash -= shares * cover_price * (1 + tcost)
        entry = trades[-1]
        # For short pnl: (entry_price - cover_price) * shares minus trade costs
        pnl = (entry["buy_price"] - cover_price) * shares - (entry["buy_price"] * shares * tcost + cover_price * shares * tcost)
        entry.update({"sell_date": cover_date, "sell_price": cover_price, "pnl": pnl})
        position = 0
        shares = 0

    # mark-to-market equity at this renko brick date
    if position == 0:
        current_equity = cash
    elif position == 1:
        current_equity = cash + shares * price
    else:  # short
        # when short: proceeds were added to cash earlier; current equity = cash - (shares * market_price)
        current_equity = cash - shares * price

    equity_ts.append((dt, current_equity))

# convert equity to series
equity_df = pd.DataFrame(equity_ts, columns=["date", "equity"]).set_index("date")
equity_df = equity_df.sort_index()

total_pnl = equity_df["equity"].iloc[-1] - capital
total_return_pct = (equity_df["equity"].iloc[-1] / capital - 1) * 100

# ------------------------------
# Create a daily equity curve mapped to original daily df.index
# by forward-filling the last renko equity onto the daily index
# ------------------------------
daily_eq = pd.Series(index=df.index, dtype=float)
last_eq = capital
renko_it = iter(equity_df.itertuples())
next_renko = None
try:
    next_renko = next(renko_it)
except StopIteration:
    next_renko = None

for day in df.index:
    # advance renko pointer while renko date <= day
    while next_renko is not None and next_renko.Index <= day:
        last_eq = next_renko.equity
        try:
            next_renko = next(renko_it)
        except StopIteration:
            next_renko = None
    daily_eq.loc[day] = last_eq

daily_eq = daily_eq.ffill().fillna(capital)

# ------------------------------
# Compute trade-level summary and drawdown
# ------------------------------
closed_trades = [t for t in trades if t["pnl"] is not None]
if closed_trades:
    trades_df = pd.DataFrame(closed_trades)
    trades_df["buy_date"] = pd.to_datetime(trades_df["buy_date"])
    trades_df["sell_date"] = pd.to_datetime(trades_df["sell_date"])
    trades_df["return_pct"] = trades_df["pnl"] / (trades_df["shares"] * trades_df["buy_price"])
else:
    trades_df = pd.DataFrame(columns=["buy_date", "buy_price", "sell_date", "sell_price", "shares", "side", "pnl", "return_pct"])

# drawdown
cum_ret = daily_eq / capital
running_max = cum_ret.cummax()
drawdown = (cum_ret - running_max) / running_max
max_dd = drawdown.min()

# ------------------------------
# Show summary metrics
# ------------------------------
st.subheader("📋 Strategy Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Initial Capital (TRY)", f"{capital:,.0f}")
col2.metric("Final Equity (TRY)", f"{equity_df['equity'].iloc[-1]:,.0f}")
col3.metric("Total Return %", f"{total_return_pct:.2f}%")
st.write(f"**Total P&L:** {total_pnl:,.0f} TRY | **Number of Renko bricks:** {len(renko)}")
st.write(f"**Closed trades:** {len(closed_trades)} | **Max Drawdown:** {max_dd:.2%}")

# ------------------------------
# Plots
#  - price with renko verticals & buy/sell markers (daily mapping)
#  - equity curve & drawdown
# ------------------------------
fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})

ax_price = axes[0]
ax_eq = axes[1]
ax_dd = axes[2]

# Price series
ax_price.plot(df.index, df["Close"], label=f"{ticker} Close", lw=1.2)
ax_price.set_title(f"{ticker} Close with Renko brick events (box={box:.6f})")
ax_price.grid(True)

# Vertical lines where renko bricks were created
for dt, r in renko.iterrows():
    ax_price.axvline(dt, color='gray', alpha=0.25, lw=0.6)

# Add renko step overlay (for visual clarity)
renko_steps_x = []
renko_steps_y = []
for i, r in renko.reset_index().iterrows():
    renko_steps_x.append(r["date"])
    renko_steps_y.append(r["brick_open"])
    renko_steps_x.append(r["date"])
    renko_steps_y.append(r["brick_close"])
ax_price.step(renko_steps_x, renko_steps_y, where='post', label='Renko step', lw=1.6)

# Mark buy/sell on price (map trade dates to actual price series)
for t in closed_trades:
    if t["side"] == "long":
        ax_price.scatter(t["buy_date"], t["buy_price"], marker="^", s=100, color="green", label="Long Entry" if "Long Entry" not in ax_price.get_legend_handles_labels()[1] else "")
        ax_price.scatter(t["sell_date"], t["sell_price"], marker="v", s=100, color="red", label="Long Exit" if "Long Exit" not in ax_price.get_legend_handles_labels()[1] else "")
    else:
        # short: entry is a short (we labeled with '^' at buy_date since buy_date holds entry_date)
        ax_price.scatter(t["buy_date"], t["buy_price"], marker="v", s=100, color="orange", label="Short Entry" if "Short Entry" not in ax_price.get_legend_handles_labels()[1] else "")
        ax_price.scatter(t["sell_date"], t["sell_price"], marker="^", s=100, color="blue", label="Short Exit" if "Short Exit" not in ax_price.get_legend_handles_labels()[1] else "")

ax_price.legend(loc='upper left', fontsize=9)

# Equity curve (daily)
ax_eq.plot(daily_eq.index, daily_eq.values, lw=1.6)
ax_eq.set_ylabel("Equity (TRY)")
ax_eq.set_title("Daily Equity Curve (forward-filled from Renko events)")
ax_eq.grid(True)

# Drawdown
ax_dd.plot(drawdown.index, drawdown.values, lw=1.2)
ax_dd.set_ylabel("Drawdown")
ax_dd.set_title("Drawdown (from peak equity)")
ax_dd.grid(True)

st.pyplot(fig)

# ------------------------------
# Renko & Signals table + Trades table
# ------------------------------
st.subheader("Renko bricks (recent)")
st.dataframe(renko.tail(100)[["brick_open", "brick_close", "trend", "brick_run", "signal"]])

st.subheader("Closed Trades")
if not trades_df.empty:
    display_df = trades_df.sort_values("buy_date").reset_index(drop=True)
    display_df["buy_date"] = display_df["buy_date"].dt.date
    display_df["sell_date"] = display_df["sell_date"].dt.date
    display_df = display_df[["buy_date", "sell_date", "side", "buy_price", "sell_price", "shares", "pnl", "return_pct"]]
    st.dataframe(display_df)
    st.write(f"Average return per closed trade: {display_df['return_pct'].mean():.2%}")
else:
    st.info("No closed trades available yet.")

st.caption("This tool is educational and approximate. The breakout logic is an approximation of the TradingView script's f_Brickhigh/f_Bricklow logic. Validate results before live trading.")
