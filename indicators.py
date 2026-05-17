"""Technical indicators and signal generation.

Includes:
- Trend/momentum: SMA, EMA, RSI, MACD
- Anchored VWAP (with auto-anchor helpers: swing high/low, recent date)
- Smart Money Concepts: Order Blocks, Fair Value Gaps, Liquidity Sweeps,
  Break of Structure (BOS), Change of Character (CHoCH)
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def ma_crossover_signals(close: pd.Series, fast_period: int, slow_period: int, ma_type: str = "SMA") -> pd.DataFrame:
    """Returns DataFrame with columns: fast, slow, signal (1 long, -1 short, 0 flat), event (golden/death cross marker)."""
    fn = sma if ma_type.upper() == "SMA" else ema
    fast = fn(close, fast_period)
    slow = fn(close, slow_period)
    signal = np.where(fast > slow, 1, np.where(fast < slow, -1, 0))
    signal = pd.Series(signal, index=close.index)
    # event: where signal changes
    prev = signal.shift(1).fillna(0)
    event = np.where((prev <= 0) & (signal > 0), "golden",
                     np.where((prev >= 0) & (signal < 0), "death", ""))
    return pd.DataFrame({"fast": fast, "slow": slow, "signal": signal, "event": event}, index=close.index)


def rsi_signals(close: pd.Series, period: int = 14, lower: int = 30, upper: int = 70) -> pd.DataFrame:
    r = rsi(close, period)
    # Long when crosses up from below lower; short when crosses down from above upper
    signal = pd.Series(0, index=close.index)
    state = 0
    for i in range(1, len(r)):
        if pd.isna(r.iloc[i]):
            signal.iloc[i] = state
            continue
        if r.iloc[i - 1] < lower and r.iloc[i] >= lower:
            state = 1
        elif r.iloc[i - 1] > upper and r.iloc[i] <= upper:
            state = -1
        signal.iloc[i] = state
    prev = signal.shift(1).fillna(0)
    event = np.where((prev <= 0) & (signal > 0), "buy",
                     np.where((prev >= 0) & (signal < 0), "sell", ""))
    return pd.DataFrame({"rsi": r, "signal": signal, "event": event}, index=close.index)


def macd_signals(close: pd.Series, fast: int = 12, slow: int = 26, signal_period: int = 9) -> pd.DataFrame:
    macd_line, signal_line, hist = macd(close, fast, slow, signal_period)
    signal = np.where(macd_line > signal_line, 1, np.where(macd_line < signal_line, -1, 0))
    signal = pd.Series(signal, index=close.index)
    prev = signal.shift(1).fillna(0)
    event = np.where((prev <= 0) & (signal > 0), "bull_cross",
                     np.where((prev >= 0) & (signal < 0), "bear_cross",
                              ""))
    return pd.DataFrame({"macd": macd_line, "signal_line": signal_line, "hist": hist,
                         "signal": signal, "event": event}, index=close.index)


# ---------------------------------------------------------------------------
# Swing point detection (used by AVWAP, BOS, and liquidity sweeps)
# ---------------------------------------------------------------------------

def swing_highs(high: pd.Series, lookback: int = 5) -> pd.Series:
    """Boolean series, True at bars where high is a local maximum within ±lookback bars."""
    n = len(high)
    out = pd.Series(False, index=high.index)
    arr = high.values
    for i in range(lookback, n - lookback):
        window = arr[i - lookback:i + lookback + 1]
        if arr[i] == window.max() and (window == arr[i]).sum() == 1:
            out.iloc[i] = True
    return out


def swing_lows(low: pd.Series, lookback: int = 5) -> pd.Series:
    """Boolean series, True at bars where low is a local minimum within ±lookback bars."""
    n = len(low)
    out = pd.Series(False, index=low.index)
    arr = low.values
    for i in range(lookback, n - lookback):
        window = arr[i - lookback:i + lookback + 1]
        if arr[i] == window.min() and (window == arr[i]).sum() == 1:
            out.iloc[i] = True
    return out


# ---------------------------------------------------------------------------
# Anchored VWAP
# ---------------------------------------------------------------------------

def anchored_vwap(high: pd.Series, low: pd.Series, close: pd.Series,
                  volume: pd.Series, anchor_idx: int) -> pd.Series:
    """Volume-weighted average price computed cumulatively from anchor_idx forward.
    Bars before the anchor are NaN.
    """
    typical = (high + low + close) / 3.0
    pv = typical * volume
    out = pd.Series(np.nan, index=close.index, dtype=float)
    if anchor_idx < 0 or anchor_idx >= len(close):
        return out
    cum_pv = pv.iloc[anchor_idx:].cumsum()
    cum_v = volume.iloc[anchor_idx:].cumsum().replace(0, np.nan)
    out.iloc[anchor_idx:] = (cum_pv / cum_v).values
    return out


def avwap_signals(ohlcv: pd.DataFrame, anchor: str = "swing_low",
                  swing_lookback: int = 10,
                  anchor_date: Optional[str] = None) -> pd.DataFrame:
    """Anchored-VWAP signals.

    anchor:
      - "swing_low"  : anchor at the most recent confirmed swing low
      - "swing_high" : anchor at the most recent confirmed swing high
      - "date"       : anchor at `anchor_date` (YYYY-MM-DD); fallback to first bar
      - "first"      : anchor at the first bar (regular VWAP)

    Long signal: close crosses above AVWAP.  Short signal: close crosses below.
    """
    close = ohlcv["Close"]
    high = ohlcv["High"]
    low = ohlcv["Low"]
    volume = ohlcv["Volume"]

    if anchor == "swing_low":
        sl = swing_lows(low, swing_lookback)
        # most recent confirmed swing low (leave room for confirmation lag)
        idxs = np.where(sl.values)[0]
        anchor_idx = int(idxs[-1]) if len(idxs) else 0
    elif anchor == "swing_high":
        sh = swing_highs(high, swing_lookback)
        idxs = np.where(sh.values)[0]
        anchor_idx = int(idxs[-1]) if len(idxs) else 0
    elif anchor == "date" and anchor_date:
        try:
            ts = pd.to_datetime(anchor_date)
            # normalize tz if the data index is tz-aware
            if close.index.tz is not None and ts.tz is None:
                ts = ts.tz_localize(close.index.tz)
            elif close.index.tz is None and ts.tz is not None:
                ts = ts.tz_localize(None)
            anchor_idx = int(close.index.get_indexer([ts], method="nearest")[0])
        except Exception:
            anchor_idx = 0
    else:
        anchor_idx = 0

    av = anchored_vwap(high, low, close, volume, anchor_idx)

    signal = pd.Series(np.where(close > av, 1, np.where(close < av, -1, 0)),
                       index=close.index)
    prev = signal.shift(1).fillna(0)
    event = np.where((prev <= 0) & (signal > 0), "cross_up",
                     np.where((prev >= 0) & (signal < 0), "cross_down", ""))
    return pd.DataFrame({"avwap": av, "signal": signal, "event": event,
                         "anchor_idx": anchor_idx}, index=close.index)


# ---------------------------------------------------------------------------
# Fair Value Gaps (FVG / imbalances)
#
# Bullish FVG (3-candle): low[i] > high[i-2]   (gap is between high[i-2] and low[i])
# Bearish FVG (3-candle): high[i] < low[i-2]   (gap is between high[i] and low[i-2])
# ---------------------------------------------------------------------------

def fair_value_gaps(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Return one row per detected FVG with: time, kind ('bull'|'bear'), top, bottom, mid."""
    high = ohlc["High"].values
    low = ohlc["Low"].values
    idx = ohlc.index
    rows = []
    for i in range(2, len(ohlc)):
        # bullish FVG: low of bar i > high of bar i-2
        if low[i] > high[i - 2]:
            top, bot = low[i], high[i - 2]
            rows.append({"time": idx[i], "kind": "bull",
                         "top": top, "bottom": bot, "mid": (top + bot) / 2})
        # bearish FVG: high of bar i < low of bar i-2
        elif high[i] < low[i - 2]:
            top, bot = low[i - 2], high[i]
            rows.append({"time": idx[i], "kind": "bear",
                         "top": top, "bottom": bot, "mid": (top + bot) / 2})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Order Blocks
# An "order block" is the last opposite-color candle before a strong displacement
# move. A bullish OB is the last DOWN candle before a strong UP move that breaks
# the prior swing high. A bearish OB is the last UP candle before a strong DOWN
# move that breaks the prior swing low.
# ---------------------------------------------------------------------------

def order_blocks(ohlc: pd.DataFrame, displacement_atr_mult: float = 1.5,
                 atr_period: int = 14, swing_lookback: int = 5) -> pd.DataFrame:
    """Detect order blocks; one row per OB."""
    high = ohlc["High"]
    low = ohlc["Low"]
    close = ohlc["Close"]
    open_ = ohlc["Open"]

    # ATR for displacement threshold
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period, min_periods=atr_period).mean()

    sh_idx = np.where(swing_highs(high, swing_lookback).values)[0]
    sl_idx = np.where(swing_lows(low, swing_lookback).values)[0]

    rows = []
    for i in range(swing_lookback + 1, len(ohlc)):
        if pd.isna(atr.iloc[i]):
            continue
        body = close.iloc[i] - open_.iloc[i]
        # bullish displacement
        if body > displacement_atr_mult * atr.iloc[i]:
            # find prior swing high broken by this candle's close
            prior = sh_idx[sh_idx < i]
            if len(prior) and close.iloc[i] > high.iloc[prior[-1]]:
                # find last bearish (down) candle before i
                for j in range(i - 1, max(0, i - 20), -1):
                    if close.iloc[j] < open_.iloc[j]:
                        rows.append({
                            "time": ohlc.index[j],
                            "kind": "bull",
                            "top": float(high.iloc[j]),
                            "bottom": float(low.iloc[j]),
                            "displacement_time": ohlc.index[i],
                        })
                        break
        # bearish displacement
        elif body < -displacement_atr_mult * atr.iloc[i]:
            prior = sl_idx[sl_idx < i]
            if len(prior) and close.iloc[i] < low.iloc[prior[-1]]:
                for j in range(i - 1, max(0, i - 20), -1):
                    if close.iloc[j] > open_.iloc[j]:
                        rows.append({
                            "time": ohlc.index[j],
                            "kind": "bear",
                            "top": float(high.iloc[j]),
                            "bottom": float(low.iloc[j]),
                            "displacement_time": ohlc.index[i],
                        })
                        break
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["time", "kind"]).reset_index(drop=True)
    return df


def ob_fvg_signals(ohlcv: pd.DataFrame, use_obs: bool = True, use_fvgs: bool = True,
                   displacement_atr_mult: float = 1.5,
                   swing_lookback: int = 5) -> pd.DataFrame:
    """Long signal when price taps into a bullish OB or bullish FVG and closes back above it.
    Short signal when price taps into a bearish OB/FVG and closes back below it.

    'Tap' = bar's low ≤ zone top (for bullish) and prior bar's low > zone top.
    """
    close = ohlcv["Close"]
    high = ohlcv["High"]
    low = ohlcv["Low"]

    bull_zones = []  # list of (start_idx, top, bottom)
    bear_zones = []

    if use_obs:
        obs = order_blocks(ohlcv, displacement_atr_mult=displacement_atr_mult,
                           swing_lookback=swing_lookback)
        for _, r in obs.iterrows():
            i0 = ohlcv.index.get_loc(r["time"])
            if r["kind"] == "bull":
                bull_zones.append((i0, r["top"], r["bottom"]))
            else:
                bear_zones.append((i0, r["top"], r["bottom"]))

    if use_fvgs:
        fvgs = fair_value_gaps(ohlcv)
        for _, r in fvgs.iterrows():
            i0 = ohlcv.index.get_loc(r["time"])
            if r["kind"] == "bull":
                bull_zones.append((i0, r["top"], r["bottom"]))
            else:
                bear_zones.append((i0, r["top"], r["bottom"]))

    signal = pd.Series(0, index=close.index, dtype=int)
    state = 0
    events = []

    for i in range(1, len(close)):
        # check bullish zones — long entry on tap-and-reclaim
        for z_start, z_top, z_bot in bull_zones:
            if z_start >= i:
                continue
            tapped_now = low.iloc[i] <= z_top and low.iloc[i] >= z_bot
            tapped_prev = low.iloc[i - 1] <= z_top and low.iloc[i - 1] >= z_bot
            if tapped_now and not tapped_prev and close.iloc[i] > z_top:
                state = 1
                events.append((i, "long_tap"))
                break
        for z_start, z_top, z_bot in bear_zones:
            if z_start >= i:
                continue
            tapped_now = high.iloc[i] >= z_bot and high.iloc[i] <= z_top
            tapped_prev = high.iloc[i - 1] >= z_bot and high.iloc[i - 1] <= z_top
            if tapped_now and not tapped_prev and close.iloc[i] < z_bot:
                state = -1
                events.append((i, "short_tap"))
                break
        signal.iloc[i] = state

    event_arr = np.full(len(close), "", dtype=object)
    for i, kind in events:
        event_arr[i] = kind
    return pd.DataFrame({"signal": signal, "event": event_arr}, index=close.index)


# ---------------------------------------------------------------------------
# Liquidity Sweeps + Break of Structure (BOS) / Change of Character (CHoCH)
# ---------------------------------------------------------------------------

def bos_signals(ohlc: pd.DataFrame, swing_lookback: int = 5,
                require_sweep: bool = True) -> pd.DataFrame:
    """Long when close > most recent confirmed swing high (bullish BOS). Optionally
    require that the prior bar's wick first swept the swing low (liquidity grab + BOS = CHoCH).
    Short when close < most recent confirmed swing low.
    """
    high = ohlc["High"]
    low = ohlc["Low"]
    close = ohlc["Close"]

    sh_mask = swing_highs(high, swing_lookback)
    sl_mask = swing_lows(low, swing_lookback)
    sh_idx = np.where(sh_mask.values)[0]
    sl_idx = np.where(sl_mask.values)[0]

    signal = pd.Series(0, index=close.index, dtype=int)
    state = 0
    event_arr = np.full(len(close), "", dtype=object)

    for i in range(swing_lookback + 1, len(close)):
        prior_sh = sh_idx[sh_idx < i]
        prior_sl = sl_idx[sl_idx < i]
        last_sh = high.iloc[prior_sh[-1]] if len(prior_sh) else np.nan
        last_sl = low.iloc[prior_sl[-1]] if len(prior_sl) else np.nan

        # bullish BOS: close > last swing high
        if not np.isnan(last_sh) and close.iloc[i] > last_sh:
            sweep_ok = True
            if require_sweep and not np.isnan(last_sl):
                # check if any of the last 5 bars wicked below last_sl but closed above
                window_low = low.iloc[max(0, i - 5):i + 1]
                window_close = close.iloc[max(0, i - 5):i + 1]
                sweep_ok = (window_low.min() < last_sl) and (window_close.iloc[-1] > last_sl)
            if sweep_ok and state != 1:
                state = 1
                event_arr[i] = "bull_bos"
        elif not np.isnan(last_sl) and close.iloc[i] < last_sl:
            sweep_ok = True
            if require_sweep and not np.isnan(last_sh):
                window_high = high.iloc[max(0, i - 5):i + 1]
                window_close = close.iloc[max(0, i - 5):i + 1]
                sweep_ok = (window_high.max() > last_sh) and (window_close.iloc[-1] < last_sh)
            if sweep_ok and state != -1:
                state = -1
                event_arr[i] = "bear_bos"
        signal.iloc[i] = state

    return pd.DataFrame({"signal": signal, "event": event_arr}, index=close.index)


def liquidity_sweep_signals(ohlc: pd.DataFrame, swing_lookback: int = 5) -> pd.DataFrame:
    """Pure liquidity-sweep reversal signals (no BOS confirmation):
    Long after a bar wicks below recent swing low but closes above it.
    Short after a bar wicks above recent swing high but closes below it.
    """
    high = ohlc["High"]
    low = ohlc["Low"]
    close = ohlc["Close"]

    sh_idx = np.where(swing_highs(high, swing_lookback).values)[0]
    sl_idx = np.where(swing_lows(low, swing_lookback).values)[0]

    signal = pd.Series(0, index=close.index, dtype=int)
    state = 0
    event_arr = np.full(len(close), "", dtype=object)

    for i in range(swing_lookback + 1, len(close)):
        prior_sh = sh_idx[sh_idx < i]
        prior_sl = sl_idx[sl_idx < i]
        last_sh = high.iloc[prior_sh[-1]] if len(prior_sh) else np.nan
        last_sl = low.iloc[prior_sl[-1]] if len(prior_sl) else np.nan

        # bullish sweep: wick below SL, close above SL
        if not np.isnan(last_sl) and low.iloc[i] < last_sl < close.iloc[i]:
            if state != 1:
                state = 1
                event_arr[i] = "sweep_low"
        elif not np.isnan(last_sh) and high.iloc[i] > last_sh > close.iloc[i]:
            if state != -1:
                state = -1
                event_arr[i] = "sweep_high"
        signal.iloc[i] = state

    return pd.DataFrame({"signal": signal, "event": event_arr}, index=close.index)
