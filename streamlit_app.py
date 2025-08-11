# streamlit_app.py
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
import os
import io
import time

import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load .env early so tokens are available
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)

# Local modules (logic/algorithms: DO NOT MODIFY)
from algo import compute_feature_frame, simulate_pl, rank_top_k, save_artifacts
from tradier_provider import TradierProvider

# -------------------- App Defaults (unchanged) --------------------
BASELINE_UNIVERSE = [
    # Index/ETF
    "SPY","QQQ","IWM","DIA","TLT","HYG","XLF","XLE","XLY","XLP","XLV","XLI","XLB","XOP","SMH","SOXL","SOXS",
    # Mega/AI/Semis
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AMD","INTC","AVGO","MU","TSM","ASML","ARM","SMCI",
    # Big tech & platform
    "CRM","NFLX","ADBE","ORCL","SHOP","SNOW","NOW","UBER","ABNB",
    # Financials
    "JPM","GS","MS","BAC","C","SCHW","COIN",
    # Energy/Materials/Industrials
    "CVX","XOM","OXY","CAT","BA","NOC","LMT","GE",
    # Autos/EV
    "GM","F","NIO","RIVN","LCID",
    # Healthcare/Biotech
    "UNH","LLY","PFE","MRNA","BMY","GILD",
    # Retail/Consumer
    "COST","WMT","HD","LOW","NKE","TGT","SBUX","MCD",
    # Social/Media
    "PINS","ROKU","SNAP","DIS","PARA",
    # Software/Payments
    "SQ","PYPL","INTU","PANW","CRWD","ZS","DDOG",
    # Misc high-turnover
    "PLTR","NET","TDOC","DKNG","RBLX","U","AFRM","AI","SOUN",
    # China ADRs (active)
    "BABA","JD","PDD","BIDU","NIO","XPEV",
]
DTE_MIN, DTE_MAX = 7, 21
HOLD_DAYS = 2
DEFAULT_RISK = 500
TOP_K_DEFAULT = 4
MAX_TICKERS_DEFAULT = 30
MIN_BARS_FOR_FEATURES = 140
TRADIER_REQ_TIMEOUT = 20

# -------------------- Page config --------------------
st.set_page_config(page_title="One-Click Option Picks", page_icon="⚡", layout="wide")
st.title("⚡ One-Click Option Picks")

# -------------------- Session state (UI only) --------------------
def _init_state():
    ss = st.session_state
    ss.setdefault("universe", BASELINE_UNIVERSE.copy())
    ss.setdefault("max_tickers", MAX_TICKERS_DEFAULT)
    ss.setdefault("risk_dollars", DEFAULT_RISK)
    ss.setdefault("top_k", TOP_K_DEFAULT)
    ss.setdefault("last_status", "")
_init_state()

# -------------------- Small helpers (logic unchanged) --------------------
def _safe_list(x):
    if x is None: return []
    return x if isinstance(x, list) else [x]

def _tradier_base():
    env = (os.getenv("TRADIER_ENV") or "production").strip().lower()
    return "https://api.tradier.com" if env == "production" else "https://sandbox.tradier.com"

def _tradier_headers():
    tok = os.getenv("TRADIER_ACCESS_TOKEN") or (st.secrets.get("TRADIER_ACCESS_TOKEN", "") if hasattr(st, "secrets") else "")
    if not tok:
        return None
    return {"Authorization": f"Bearer {tok}", "Accept": "application/json"}

def _date_window(days_back: int = 420) -> tuple[str, str]:
    today = date.today()
    start = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")
    return start, end

@st.cache_data(show_spinner=False, ttl=180)
def fetch_daily_tradier_cached(symbol: str, start_s: str, end_s: str) -> pd.DataFrame:
    """
    Pull daily bars from Tradier Market Data.
    /v1/markets/history?symbol=SYM&interval=daily&start=YYYY-MM-DD&end=YYYY-MM-DD
    """
    headers = _tradier_headers()
    if not headers:
        return pd.DataFrame()
    url = f"{_tradier_base()}/v1/markets/history"
    params = {"symbol": symbol, "interval": "daily", "start": start_s, "end": end_s}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=TRADIER_REQ_TIMEOUT)
    except Exception:
        return pd.DataFrame()
    if r.status_code != 200:
        return pd.DataFrame()
    data = r.json()
    rows = _safe_list(data.get("history", {}).get("day"))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Normalize
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    return df[["date", "open", "high", "low", "close", "volume"]]

def normalize_daily_for_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the frame has: timestamp (tz-aware UTC), open/high/low/close, volume
    Works for Tradier ('date') or anything with 'timestamp'.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    ts = None
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    elif "date" in df.columns:
        ts = pd.to_datetime(df["date"], errors="coerce", utc=True)
    else:
        return pd.DataFrame()
    df["timestamp"] = ts
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    keep = ["timestamp", "open", "high", "low", "close", "volume"]
    return df[[c for c in keep if c in df.columns]]

def is_market_open_now_et() -> bool:
    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:
        return False
    open_t = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_t <= now_et <= close_t

def header_cards(tkr: str, sig_row: pd.Series):
    colA, colB, colC, colD, colE, colF = st.columns(6)
    colA.metric(f"{tkr} last", f"{sig_row['close']:.2f}")
    colB.metric("ATR14", f"{sig_row['ATR14']:.2f}")
    colC.metric("HV20", f"{sig_row['HV20']:.2%}")
    colD.metric("RSI14", f"{sig_row['RSI14']:.0f}")
    trend_state = "Uptrend" if sig_row["SMA20"] > sig_row["SMA50"] else "Downtrend"
    colE.metric("Trend", trend_state)
    colF.metric("ADX14", f"{sig_row['ADX14']:.1f}")

def build_explanation(idea: pd.Series, sig: pd.Series) -> str:
    sdir, svol, sbrk = float(sig["S_DIR"]), float(sig["S_VOL"]), float(sig["S_BRK"])
    adx, rsi = float(sig["ADX14"]), float(sig["RSI14"])
    tags = str(sig.get("TAGS", ""))
    play = idea["Play"]; dte = int(idea["DTE"]); strike = float(idea["Strike"])
    iv = float(idea.get("IV used", 0.0)) if "IV used" in idea else None
    spread = float(idea.get("Spread %", 0.0)) if "Spread %" in idea else 0.0
    oi = int(idea.get("OI", 0))
    cost = float(idea["Cost/contract"]); units = int(idea.get("units", 0))
    pricing = str(idea.get("Pricing", "")).strip()

    if play == "straddle":
        bias_txt = f"Direction unclear (S_dir {sdir:.2f}), vol elevated (S_vol {svol:.2f})."
        why_txt = "ATM straddle targets movement either way."
    elif play == "call":
        bias_txt = f"Bullish (S_dir {sdir:.2f})" + (", breakout confirmed" if sbrk > 0 else "")
        why_txt = "Δ~0.30 call balances probability and premium."
    else:
        bias_txt = f"Bearish (S_dir {sdir:.2f})" + (", breakdown confirmed" if sbrk < 0 else "")
        why_txt = "Δ~0.30 put captures downside with limited cost."

    trend_txt = "Strong trend" if adx >= 25 else "Weak/neutral trend"
    rsi_txt = "Overbought" if rsi >= 70 else ("Oversold" if rsi <= 30 else "Neutral RSI")
    tag_txt = f" • {tags}" if tags else ""
    iv_txt = f", IV ~{iv:.1f}%" if iv else ""
    liq_txt = f" OI {oi}, spread ~{spread:.1f}%." if "Spread %" in idea else ""
    price_src = f" Priced via {pricing}." if pricing else ""
    return f"{bias_txt} {trend_txt}; {rsi_txt}. {why_txt}{tag_txt} Strike ~{strike:.2f}, {dte} DTE{iv_txt}. Cost ≈ ${cost:,.2f} ({units}x).{liq_txt}{price_src}"

def plain_english_column(top_df: pd.DataFrame, signals_by_ticker: dict) -> pd.Series:
    out = []
    for _, row in top_df.iterrows():
        tkr = row.get("Underlying", "")
        sig = signals_by_ticker.get(tkr)
        out.append("Signals unavailable." if sig is None else build_explanation(row, sig))
    return pd.Series(out, index=top_df.index, name="Why this trade")

# Cache the provider object (UI perf only; logic unchanged)
@st.cache_resource(show_spinner=False)
def _get_provider():
    return TradierProvider()

# -------------------- Sidebar: global controls --------------------
with st.sidebar:
    st.success("Click **Scan now**. Defaults handle the rest.")
    with st.form("scan_form", clear_on_submit=False):
        with st.expander("Advanced", expanded=False):
            st.multiselect(
                "Universe",
                BASELINE_UNIVERSE,
                default=st.session_state.universe,
                key="universe",
                help="Symbols to include."
            )
            st.slider("Max tickers per run", 10, 60, key="max_tickers", help="Keeps API load steady.")
            st.number_input("Budget per idea ($)", 50, 5000, step=50, key="risk_dollars")
            st.slider("Max ideas to show", 2, 10, key="top_k")
        st.caption(
            f"TRADIER_ENV={os.getenv('TRADIER_ENV','(unset)')} | "
            f"TRADIER_TOKEN={'yes' if bool(os.getenv('TRADIER_ACCESS_TOKEN')) else 'no'}"
        )
        run_btn = st.form_submit_button("Scan now", type="primary")

# -------------------- Main content --------------------
tabs = st.tabs(["Ideas", "Signals", "Logs"])
ideas_tab, signals_tab, logs_tab = tabs

# -------------------- Run scan (logic preserved; UI wrapped) --------------------
if run_btn:
    try:
        token = os.getenv("TRADIER_ACCESS_TOKEN") or st.secrets.get("TRADIER_ACCESS_TOKEN", "")
        if not token:
            st.error("Tradier token missing. Add TRADIER_ACCESS_TOKEN to .env or .streamlit/secrets.toml and rerun.")
            st.stop()

        market_open = is_market_open_now_et()
        with ideas_tab:
            st.caption(
                f"Routing: {'Market OPEN → live-mid pricing' if market_open else 'Market CLOSED → theoretical fallbacks'}."
            )

        # Hidden per-mode defaults (unchanged)
        if market_open:
            min_oi_default = 50
            max_spread_default = 0.50
            use_theo = False
            theo_spread_thresh = 1.00
            relax_gates = False
        else:
            min_oi_default = 10
            max_spread_default = 2.00
            use_theo = True
            theo_spread_thresh = 0.80
            relax_gates = True

        d_start, d_end = _date_window(days_back=420)
        provider = _get_provider()

        # Universe slice (unchanged logic)
        universe = list(dict.fromkeys([u.strip().upper() for u in st.session_state.universe if u.strip()]))
        if not universe:
            st.error("Universe is empty.")
            st.stop()
        offset = pd.Timestamp.utcnow().dayofyear % len(universe)
        to_scan = (universe[offset:] + universe[:offset])[:int(st.session_state.max_tickers)]

        # Status panel for long steps
        with logs_tab:
            status_box = st.empty()

        with st.status("Scanning…", expanded=False) as status:
            status.update(label=f"Fetching daily history and computing signals for {len(to_scan)} tickers…")
            # 1) Signals
            signals_map, feat_store = {}, {}
            for t in to_scan:
                daily = fetch_daily_tradier_cached(t, d_start, d_end)
                daily = normalize_daily_for_features(daily)
                if daily.empty or len(daily) < MIN_BARS_FOR_FEATURES:
                    continue
                feat = compute_feature_frame(daily).dropna().reset_index(drop=True)
                if feat.empty:
                    continue
                last = feat.iloc[-1].copy()
                last.name = t
                signals_map[t] = last
                feat_store[t] = feat

            if not signals_map:
                status.update(state="complete")
                st.info("No signals computed. Try again near market hours or expand the universe.")
                st.stop()

            status.update(label="Building option candidates…")
            rows = []
            for t, sig in signals_map.items():
                cands = provider.build_live_candidates(
                    symbol=t,
                    spot=float(sig["close"]),
                    s_dir=float(sig["S_DIR"]),
                    s_vol=float(sig["S_VOL"]),
                    s_brk=float(sig["S_BRK"]),
                    hv20=float(sig["HV20"]),
                    risk_dollars=float(st.session_state.risk_dollars),
                    dte_min=DTE_MIN, dte_max=DTE_MAX,
                    min_oi=min_oi_default, min_vol=0, max_spread=max_spread_default,
                    max_expiries=2,
                    use_theoretical_fallback=use_theo,
                    theo_spread_threshold=theo_spread_thresh,
                    relax_gates_if_empty=relax_gates,
                )
                if isinstance(cands, pd.DataFrame) and not cands.empty:
                    rows.append(cands)
            all_cands = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

            if all_cands.empty:
                st.warning("No contracts passed initial gates. Loosening filters and forcing theoretical pricing once…")
                rows = []
                for t, sig in signals_map.items():
                    cands = provider.build_live_candidates(
                        symbol=t,
                        spot=float(sig["close"]),
                        s_dir=float(sig["S_DIR"]),
                        s_vol=float(sig["S_VOL"]),
                        s_brk=float(sig["S_BRK"]),
                        hv20=float(sig["HV20"]),
                        risk_dollars=float(st.session_state.risk_dollars),
                        dte_min=DTE_MIN, dte_max=DTE_MAX,
                        min_oi=0, min_vol=0, max_spread=9.99,
                        max_expiries=2,
                        use_theoretical_fallback=True,
                        theo_spread_threshold=0.10,
                        relax_gates_if_empty=True,
                    )
                    if isinstance(cands, pd.DataFrame) and not cands.empty:
                        rows.append(cands)
                all_cands = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

            if all_cands.empty:
                status.update(state="complete")
                st.info("Still no candidates after aggressive fallback. Try closer to the open or widen the universe.")
                st.stop()

            status.update(label="Running scenario P/L and ranking…")
            parts = []
            for tkr, g in all_cands.groupby("underlying"):
                hv20 = float(signals_map[tkr]["HV20"])
                parts.append(simulate_pl(g, hv20=hv20, hold_days=HOLD_DAYS))
            all_pl = pd.concat(parts, ignore_index=True)

            if "pricing_source" in all_pl.columns and "Pricing" not in all_pl.columns:
                all_pl["Pricing"] = all_pl["pricing_source"].map({"mid": "mid", "theoretical": "theoretical"}).fillna("mid")

            top = rank_top_k(all_pl, k=int(st.session_state.top_k))
            if top.empty:
                status.update(state="complete")
                st.info("No ranked ideas after scoring.")
                st.stop()

            if "Pricing" not in top.columns and "Pricing" in all_pl.columns:
                try:
                    top = top.merge(
                        all_pl[["underlying", "play_type", "dte", "K", "Pricing"]],
                        left_on=["Underlying", "Play", "DTE", "Strike"],
                        right_on=["underlying", "play_type", "dte", "K"],
                        how="left"
                    ).drop(columns=["underlying", "play_type", "dte", "K"])
                except Exception:
                    pass

            # Header quick stats for top underlying
            first_tkr = top.get("Underlying", pd.Series(dtype=str)).iloc[0] if "Underlying" in top.columns and len(top) else None
            with ideas_tab:
                if first_tkr and first_tkr in feat_store:
                    header_cards(first_tkr, signals_map[first_tkr])

            # Plain-English rationale column
            explanations = plain_english_column(top, signals_map)
            top = top.copy()
            top.insert(len(top.columns), "Why this trade", explanations)

            status.update(state="complete")

        # -------------------- Ideas tab: main table --------------------
        with ideas_tab:
            # Column configs if present
            colcfg = {}
            if "DTE" in top.columns: colcfg["DTE"] = st.column_config.NumberColumn("DTE", help="Days to expiry", format="%d")
            if "Strike" in top.columns: colcfg["Strike"] = st.column_config.NumberColumn("Strike", format="%.2f")
            if "Cost/contract" in top.columns: colcfg["Cost/contract"] = st.column_config.NumberColumn("Cost/contract", format="$%.2f")
            if "IV used" in top.columns: colcfg["IV used"] = st.column_config.NumberColumn("IV used", format="%.1f%%")
            if "Spread %" in top.columns: colcfg["Spread %"] = st.column_config.NumberColumn("Spread %", format="%.1f%%")
            if "HV20" in top.columns: colcfg["HV20"] = st.column_config.NumberColumn("HV20", format="%.2f")
            if "Why this trade" in top.columns: colcfg["Why this trade"] = st.column_config.TextColumn("Why this trade", width="large")

            st.dataframe(
                top,
                use_container_width=True,
                column_config=colcfg,
                hide_index=True
            )

            # Download
            try:
                buf = io.StringIO()
                top.to_csv(buf, index=False)
                st.download_button("Download ideas (CSV)", buf.getvalue(), file_name="option_ideas.csv", type="secondary")
            except Exception:
                pass

        # -------------------- Signals tab: show latest signal rows --------------------
        with signals_tab:
            st.caption("Latest signals used for ranking.")
            if signals_map:
                sig_df = pd.DataFrame(signals_map).T.reset_index(names="Ticker")
                st.dataframe(sig_df, use_container_width=True, hide_index=True)
            else:
                st.info("No signals available.")

        # -------------------- Save artifacts (unchanged) --------------------
        try:
            if first_tkr:
                save_artifacts(
                    features_df=feat_store[first_tkr],
                    picks_df=top[top["Underlying"] == first_tkr],
                    ticker=first_tkr,
                    params={"dte": [DTE_MIN, DTE_MAX], "risk": float(st.session_state.risk_dollars), "hold_days": HOLD_DAYS}
                )
        except Exception:
            pass

        st.toast("Scan complete.", icon="✅")

    except Exception as e:
        st.error("Something went wrong while scanning.")
        with logs_tab:
            with st.expander("Details"):
                st.exception(e)
