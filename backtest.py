"""Backtest a signal series and compute accuracy metrics."""
import numpy as np
import pandas as pd


def compute_metrics(close: pd.Series, signal: pd.Series, forward_bars: int = 5,
                    long_only: bool = True) -> dict:
    """
    signal: 1 (long), -1 (short), 0 (flat). Position is taken at the close of bar t,
            P&L is realized over the following bar(s) (no lookahead).

    Returns dict of metrics + a per-bar returns Series.
    """
    close = close.dropna()
    signal = signal.reindex(close.index).fillna(0)
    if long_only:
        signal = signal.clip(lower=0)

    bar_ret = close.pct_change().fillna(0)
    # position held during bar t is signal at t-1
    position = signal.shift(1).fillna(0)
    strat_ret = position * bar_ret

    # Trade-level: identify each entry-exit pair
    trades = _extract_trades(close, signal)

    # Signal "accuracy": forward N-bar return after each event matches direction
    events = signal.diff().fillna(0)
    entries = events[events != 0].index
    correct = 0
    total = 0
    for ts in entries:
        idx = close.index.get_loc(ts)
        if idx + forward_bars >= len(close):
            continue
        fwd = close.iloc[idx + forward_bars] / close.iloc[idx] - 1
        direction = signal.loc[ts]
        if direction == 0:
            continue
        if (direction > 0 and fwd > 0) or (direction < 0 and fwd < 0):
            correct += 1
        total += 1
    forward_accuracy = (correct / total) if total else float("nan")

    # Strategy returns
    cum_strat = (1 + strat_ret).cumprod()
    cum_bh = (1 + bar_ret).cumprod()
    total_strat = cum_strat.iloc[-1] - 1 if len(cum_strat) else 0
    total_bh = cum_bh.iloc[-1] - 1 if len(cum_bh) else 0

    # Sharpe (assume daily; annualize 252; if hourly etc still informative)
    if strat_ret.std() > 0:
        sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252)
    else:
        sharpe = float("nan")

    # Drawdown
    running_max = cum_strat.cummax()
    drawdown = (cum_strat / running_max) - 1
    max_dd = drawdown.min() if len(drawdown) else 0

    win_trades = [t for t in trades if t["pnl_pct"] > 0]
    win_rate = (len(win_trades) / len(trades)) if trades else float("nan")
    avg_trade = np.mean([t["pnl_pct"] for t in trades]) if trades else float("nan")

    return {
        "forward_accuracy": forward_accuracy,
        "forward_bars": forward_bars,
        "n_signals": total,
        "n_trades": len(trades),
        "win_rate": win_rate,
        "avg_trade_pct": avg_trade,
        "total_return_strategy": total_strat,
        "total_return_buyhold": total_bh,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "equity_curve": cum_strat,
        "buyhold_curve": cum_bh,
        "trades": trades,
    }


def _extract_trades(close: pd.Series, signal: pd.Series):
    trades = []
    pos = 0
    entry_price = None
    entry_time = None
    entry_dir = 0
    for ts, s in signal.items():
        if s != pos:
            # close existing
            if pos != 0 and entry_price is not None:
                exit_price = close.loc[ts]
                pnl = (exit_price / entry_price - 1) * entry_dir
                trades.append({
                    "entry_time": entry_time, "exit_time": ts,
                    "entry_price": entry_price, "exit_price": exit_price,
                    "direction": entry_dir, "pnl_pct": pnl,
                })
                entry_price = None
            # open new
            if s != 0:
                entry_price = close.loc[ts]
                entry_time = ts
                entry_dir = s
            pos = s
    return trades
