# algo.py
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

DATA_DIR = Path("./data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252

# ---------- Helpers / TA ----------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.clip(0, 100).fillna(50.0)

def hv_annualized(returns: pd.Series, n: int = 20) -> pd.Series:
    rv = returns.rolling(n, min_periods=n).std()
    return (rv * math.sqrt(TRADING_DAYS)).rename("hv20")

def bollinger(s: pd.Series, n: int = 20, k: float = 2.0):
    m = s.rolling(n, min_periods=n).mean()
    sd = s.rolling(n, min_periods=n).std()
    upper = m + k * sd
    lower = m - k * sd
    bw = (upper - lower) / m
    return m, upper, lower, bw

def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(s, fast) - ema(s, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def dmi_adx(df: pd.DataFrame, n: int = 14):
    up_move = df["high"].diff()
    dn_move = -df["low"].diff()
    plus_dm  = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    tr = true_range(df)
    tr_sm = pd.Series(tr).ewm(alpha=1/n, adjust=False).mean()
    plus_sm = pd.Series(plus_dm).ewm(alpha=1/n, adjust=False).mean()
    minus_sm = pd.Series(minus_dm).ewm(alpha=1/n, adjust=False).mean()
    plus_di = 100 * (plus_sm / (tr_sm + 1e-12))
    minus_di = 100 * (minus_sm / (tr_sm + 1e-12))
    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12))
    adx = pd.Series(dx).ewm(alpha=1/n, adjust=False).mean()
    return plus_di.rename("DI+"), minus_di.rename("DI-"), adx.rename("ADX")

def linreg_slope_norm(close: pd.Series, win: int, scale: pd.Series) -> pd.Series:
    def slope_window(x):
        y = x.values
        x_idx = np.arange(len(y))
        if len(y) < 2:
            return np.nan
        coef = np.polyfit(x_idx, y, 1)[0]
        return coef
    raw = close.rolling(win).apply(slope_window, raw=False)
    scale_eps = scale.replace(0, np.nan)
    return (raw / scale_eps).clip(-5, 5)

# ---------- Candle tags (simple) ----------
def detect_candles(df: pd.DataFrame) -> pd.DataFrame:
    o = df["open"]; c = df["close"]; h = df["high"]; l = df["low"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upsh = (h - np.maximum(c, o))
    lowsh = (np.minimum(c, o) - l)
    doji = (body <= 0.1 * rng)
    hammer = (lowsh >= 2 * body) & (upsh <= body) & (c > o)
    shooting = (upsh >= 2 * body) & (lowsh <= body) & (c < o)

    prev_o = o.shift(1); prev_c = c.shift(1)
    prev_dir = np.sign(prev_c - prev_o)
    bull_eng = (prev_dir < 0) & (c > o) & (c >= prev_o) & (o <= prev_c)
    bear_eng = (prev_dir > 0) & (c < o) & (o >= prev_c) & (c <= prev_o)
    return pd.DataFrame({
        "doji": doji.fillna(False),
        "hammer": hammer.fillna(False),
        "shooting_star": shooting.fillna(False),
        "bullish_engulf": bull_eng.fillna(False),
        "bearish_engulf": bear_eng.fillna(False),
    })

# ---------- Feature engineering ----------
def compute_feature_frame(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return daily
    df = daily.copy()

    # Accept either 'timestamp' (tz-aware) or 'date'
    if "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("UTC").dt.date
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("UTC").dt.date
    else:
        return pd.DataFrame()

    df = df.groupby("date", as_index=False).agg(
        open=("open","first"),
        high=("high","max"),
        low=("low","min"),
        close=("close","last"),
        volume=("volume","sum")
    )
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    df["ATR14"] = atr(df, 14)
    df["RSI14"] = rsi(df["close"], 14)
    df["SMA20"] = sma(df["close"], 20)
    df["SMA50"] = sma(df["close"], 50)
    df["EMA21"] = ema(df["close"], 21)

    macd_line, macd_sig, macd_hist = macd(df["close"], 12, 26, 9)
    df["MACD"] = macd_line; df["MACD_SIG"] = macd_sig; df["MACD_HIST"] = macd_hist

    m, ub, lb, bw = bollinger(df["close"], 20, 2.0)
    df["BB_UP"] = ub; df["BB_LO"] = lb; df["BB_BW"] = bw

    df["HV20"] = hv_annualized(df["ret"], 20)
    di_plus, di_minus, adx = dmi_adx(df, 14)
    df["DI_PLUS"] = di_plus; df["DI_MINUS"] = di_minus; df["ADX14"] = adx

    df["VOL_MA20"] = sma(df["volume"], 20)
    df["VOL_RATIO"] = (df["volume"] / (df["VOL_MA20"] + 1e-9)).clip(lower=0)

    df["HH20"] = df["high"].rolling(20, min_periods=5).max()
    df["LL20"] = df["low"].rolling(20, min_periods=5).min()
    df["break_20_up"] = (df["close"] > df["HH20"]).astype(int)
    df["break_20_dn"] = (df["close"] < df["LL20"]).astype(int)
    df["prev_high"] = df["high"].shift(1)
    df["prev_low"]  = df["low"].shift(1)
    df["break_prev_up"] = (df["close"] > df["prev_high"]).astype(int)
    df["break_prev_dn"] = (df["close"] < df["prev_low"]).astype(int)
    df["gap_pct"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

    look = 60
    df["HV20_pct"] = df["HV20"].rolling(look, min_periods=5).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    df["BB_BW_pct_inv"] = df["BB_BW"].rolling(look, min_periods=5).apply(lambda x: 1.0 - pd.Series(x).rank(pct=True).iloc[-1], raw=False)

    df["slope10_norm"] = linreg_slope_norm(df["close"], 10, df["ATR14"])
    df["sma_state"] = np.where(df["SMA20"] > df["SMA50"], 1.0, -1.0)
    df["sma_dist"]  = (df["SMA20"] - df["SMA50"]).abs() / df["close"]

    df = pd.concat([df, detect_candles(df)], axis=1)

    macd_norm = np.tanh((df["MACD"] - df["MACD_SIG"]) / (df["close"].rolling(20).std() + 1e-9))
    adx_trend = ((df["DI_PLUS"] - df["DI_MINUS"]) / 100.0) * ((df["ADX14"] - 20) / 20).clip(lower=0)

    s_dir = (
        0.25 * df["slope10_norm"].clip(-1, 1) +
        0.20 * (df["sma_state"] * (df["sma_dist"] * 10).clip(0, 0.6)) +
        0.25 * macd_norm.clip(-1, 1) +
        0.15 * adx_trend.clip(-1, 1) +
        0.15 * ((df["RSI14"] - 50.0) / 50.0).clip(-1, 1)
    )
    s_dir = s_dir + np.where(df["RSI14"] > 70, -0.10, 0.0) + np.where(df["RSI14"] < 30, +0.10, 0.0)
    df["S_DIR"] = s_dir.clip(-1, 1)

    atr_pct = (df["ATR14"] / df["close"]).replace([np.inf, -np.inf], np.nan)
    atr_mean10 = atr_pct.rolling(10, min_periods=10).mean()
    atr_std10  = atr_pct.rolling(10, min_periods=10).std()
    atr_z10 = ((atr_pct - atr_mean10) / (atr_std10 + 1e-9)).clip(-2, 2)
    atr_z10 = atr_z10.apply(lambda z: float(np.tanh(z / 2)))

    s_vol = (
        0.40 * df["HV20_pct"].clip(0, 1).fillna(0) +
        0.30 * df["BB_BW_pct_inv"].clip(0, 1).fillna(0) +
        0.20 * df["VOL_RATIO"].pipe(lambda s: (s - 1.0)).clip(lower=0, upper=3) / 3.0 +
        0.10 * atr_z10.fillna(0)
    )
    df["S_VOL"] = s_vol.clip(0, 1)

    brk_up = (df["break_20_up"] | df["break_prev_up"]) & (df["VOL_RATIO"] >= 1.5)
    brk_dn = (df["break_20_dn"] | df["break_prev_dn"]) & (df["VOL_RATIO"] >= 1.5)
    s_brk = np.where(brk_up, 1.0, np.where(brk_dn, -1.0, 0.0))
    df["S_BRK"] = pd.Series(s_brk, index=df.index).clip(-1, 1)

    tags = []
    for i in range(len(df)):
        t = []
        if brk_up.iloc[i]: t.append("Breakout↑")
        if brk_dn.iloc[i]: t.append("Breakdown↓")
        if df["MACD"].iloc[i] > df["MACD_SIG"].iloc[i]: t.append("MACD↑")
        if df["MACD"].iloc[i] < df["MACD_SIG"].iloc[i]: t.append("MACD↓")
        if df["ADX14"].iloc[i] >= 25: t.append("Strong trend")
        if df["BB_BW_pct_inv"].iloc[i] >= 0.6: t.append("Squeeze")
        if df["RSI14"].iloc[i] <= 30: t.append("Oversold")
        if df["RSI14"].iloc[i] >= 70: t.append("Overbought")
        if df["bullish_engulf"].iloc[i]: t.append("Bull Engulf")
        if df["bearish_engulf"].iloc[i]: t.append("Bear Engulf")
        if df["hammer"].iloc[i]: t.append("Hammer")
        if df["shooting_star"].iloc[i]: t.append("Shooting Star")
        if df["doji"].iloc[i]: t.append("Doji")
        tags.append(", ".join(t))
    df["TAGS"] = tags
    return df

# ---------- Black–Scholes ----------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def black_scholes_price(typ: str, S: float, K: float, vol: float, T: float, r: float = 0.0) -> float:
    vol = max(vol, 1e-6); T = max(T, 1e-6)
    d1 = (math.log(S / K) + (r + 0.5 * vol * vol) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    if typ == "call":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

# ---------- P/L and ranking ----------
def simulate_pl(cands: pd.DataFrame, hv20: float, hold_days: int = 2) -> pd.DataFrame:
    """
    Expects (at minimum):
      - entry_price: per-CONTRACT dollars
      - contract_size: usually 100
      - play_type in {"call","put","straddle"}
      - K, S0, T0, IV_used (and call_iv/put_iv for straddles)
    Returns P/L columns in PER-CONTRACT dollars.
    """
    if cands.empty:
        return cands
    out = cands.copy()
    out["contract_size"] = out.get("contract_size", 100)
    H = max(hold_days, 1) / TRADING_DAYS
    sigma_H = float(hv20 or 0.0) * math.sqrt(H)

    def after_value_per_share(row, S1, T_left):
        if row["play_type"] == "straddle":
            civ = float(row.get("call_iv") or row["IV_used"])
            piv = float(row.get("put_iv") or row["IV_used"])
            call_val = black_scholes_price("call", S1, row["K"], civ, T_left, 0.0)
            put_val  = black_scholes_price("put",  S1, row["K"], piv, T_left, 0.0)
            return call_val + put_val
        else:
            return black_scholes_price(row["play_type"], S1, row["K"], row["IV_used"], T_left, 0.0)

    S0 = float(out["S0"].iloc[0])
    S_up, S_flat, S_dn = S0 * math.exp(+sigma_H), S0, S0 * math.exp(-sigma_H)
    T_left = np.maximum(out["T0"] - H, 1.0 / TRADING_DAYS / 10.0)

    per_share_up   = out.apply(lambda r: after_value_per_share(r, S_up,   T_left.loc[r.name]), axis=1)
    per_share_flat = out.apply(lambda r: after_value_per_share(r, S_flat, T_left.loc[r.name]), axis=1)
    per_share_dn   = out.apply(lambda r: after_value_per_share(r, S_dn,   T_left.loc[r.name]), axis=1)

    out["P_up"]   = per_share_up   * out["contract_size"] - out["entry_price"]
    out["P_flat"] = per_share_flat * out["contract_size"] - out["entry_price"]
    out["P_dn"]   = per_share_dn   * out["contract_size"] - out["entry_price"]

    out["units"] = out["units"].astype(int)
    out["Afford?"] = np.where(out["units"] > 0, "Yes", "No")
    out["Exp P/L"] = (out["P_up"] + out["P_flat"] + out["P_dn"]) / 3.0

    out["score"] = np.where(out["entry_price"] > 0, out["Exp P/L"] / out["entry_price"], -1e9)
    return out

def rank_top_k(cands_pl: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    if cands_pl.empty:
        return cands_pl

    df = cands_pl.copy()

    # Coerce key fields; use safe defaults so ranking never crashes
    if "spread_pct" not in df.columns:
        df["spread_pct"] = np.nan
    df["spread_pct"] = pd.to_numeric(df["spread_pct"], errors="coerce").fillna(1.0)  # 100% spread as neutral/worse

    if "oi" not in df.columns:
        df["oi"] = 0
    df["oi"] = pd.to_numeric(df["oi"], errors="coerce").fillna(0).astype(int)

    if "score" not in df.columns:
        df["score"] = -1e9
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(-1e9)

    # Rank: higher score, then tighter spreads, then higher OI
    ranked = df.sort_values(["score", "spread_pct", "oi"], ascending=[False, True, False]).head(k).copy()

    # Compact output
    cols = [
        "underlying","play_type","dte","strike_rule","target_delta","entry_price",
        "P_up","P_flat","P_dn","Exp P/L","score","units","Afford?",
        "K","S0","IV_used","oi","vol","bid","ask","spread_pct","pricing_source"
    ]
    cols = [c for c in cols if c in ranked.columns]
    show = ranked[cols].rename(columns={
        "underlying":"Underlying",
        "play_type":"Play",
        "dte":"DTE",
        "strike_rule":"Strike rule",
        "target_delta":"Target Δ",
        "entry_price":"Cost/contract",
        "P_up":"P/L up",
        "P_flat":"P/L flat",
        "P_dn":"P/L down",
        "K":"Strike",
        "IV_used":"IV used",
        "oi":"OI",
        "vol":"Vol",
        "spread_pct":"Spread %",
        "pricing_source":"Pricing",
    })

    # Final formatting
    for col in ["Cost/contract","P/L up","P/L flat","P/L down","Exp P/L","score","Spread %","IV used"]:
        if col in show.columns:
            show[col] = pd.to_numeric(show[col], errors="coerce")
    if "score" in show.columns:
        show["Score"] = show["score"].round(3)
        show = show.drop(columns=["score"])
    if "Spread %" in show.columns:
        show["Spread %"] = (show["Spread %"] * 100).round(1)
    if "IV used" in show.columns:
        show["IV used"] = (show["IV used"] * 100).round(1)
    if "Target Δ" in show.columns:
        show["Target Δ"] = pd.to_numeric(show["Target Δ"], errors="coerce").round(2)
    return show


def save_artifacts(features_df: pd.DataFrame, picks_df: pd.DataFrame, ticker: str, params: Dict):
    if not features_df.empty:
        features_df.to_parquet(DATA_DIR / f"features_{ticker.lower()}.parquet", index=False)
    if not picks_df.empty:
        with open(DATA_DIR / f"picks_{ticker.lower()}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "params": params,
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "rows": picks_df.to_dict(orient="records"),
            }) + "\n")
