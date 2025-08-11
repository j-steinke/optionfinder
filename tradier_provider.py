# tradier_provider.py
import os
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

# Load .env early so TRADIER_ACCESS_TOKEN is available
load_dotenv(override=False)

TRADING_DAYS = 252

def _base_url() -> str:
    env = (os.getenv("TRADIER_ENV") or "production").strip().lower()
    return "https://api.tradier.com" if env == "production" else "https://sandbox.tradier.com"

def _headers() -> Dict[str, str]:
    token = os.getenv("TRADIER_ACCESS_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TRADIER_ACCESS_TOKEN is missing. Put it in your .env")
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}

def _safe_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def _dte(exp: date, today: Optional[date] = None) -> int:
    today = today or date.today()
    return max((exp - today).days, 0)

def _mid(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is None or ask is None:
        return None
    if ask <= 0:
        return None
    if bid is None or bid < 0:
        bid = 0.0
    return (bid + ask) / 2.0

# Import Black–Scholes from algo (fallback local if import order changes)
try:
    from algo import black_scholes_price
except Exception:
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

@dataclass
class OptionQuote:
    symbol: str
    option_type: str  # "call" | "put"
    strike: float
    expiration: date
    dte: int
    bid: float
    ask: float
    last: Optional[float]
    volume: int
    open_interest: int
    delta: Optional[float]
    gamma: Optional[float]
    theta: Optional[float]
    vega: Optional[float]
    bid_iv: Optional[float]
    mid_iv: Optional[float]
    ask_iv: Optional[float]
    smv_vol: Optional[float]

    @property
    def mid(self) -> Optional[float]:
        return _mid(self.bid, self.ask)

    @property
    def spread_pct(self) -> Optional[float]:
        m = self.mid
        if m is None or m <= 0:
            return None
        return max(self.ask - self.bid, 0.0) / m

class TradierProvider:
    """
    Expirations + chains (Greeks/IV) with robust fallbacks:
    - pre-open theoretical pricing
    - relaxed gates
    - synthetic ATM ideas if Tradier returns no expirations/chains
    """
    def __init__(self, throttle_sec: float = 0.4):
        self.base = _base_url()
        self.headers = _headers()
        self.throttle_sec = throttle_sec

    # ------------------ Raw API ------------------
    def get_expirations(self, symbol: str) -> List[date]:
        url = f"{self.base}/v1/markets/options/expirations"
        params = {"symbol": symbol, "includeAllRoots": "true"}
        r = requests.get(url, headers=self.headers, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        dates = []
        exp_obj = data.get("expirations", {})
        if "date" in exp_obj and isinstance(exp_obj["date"], list):
            dates = exp_obj["date"]
        else:
            dates = [x["date"] if isinstance(x, dict) else x for x in _safe_list(exp_obj.get("expiration"))]
        out: List[date] = []
        for ds in dates:
            try:
                out.append(_parse_date(ds))
            except Exception:
                continue
        time.sleep(self.throttle_sec)
        return sorted(out)

    def get_chain(self, symbol: str, expiration: date, want_greeks: bool = True) -> List[OptionQuote]:
        url = f"{self.base}/v1/markets/options/chains"
        params = {"symbol": symbol, "expiration": expiration.strftime("%Y-%m-%d")}
        if want_greeks:
            params["greeks"] = "true"
        r = requests.get(url, headers=self.headers, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        raw_opts = _safe_list(data.get("options", {}).get("option"))
        out: List[OptionQuote] = []
        for o in raw_opts:
            g = o.get("greeks") or {}
            out.append(OptionQuote(
                symbol=o["symbol"],
                option_type=o.get("option_type"),
                strike=float(o["strike"]),
                expiration=_parse_date(o["expiration_date"]),
                dte=_dte(_parse_date(o["expiration_date"])),
                bid=float(o.get("bid") or 0.0),
                ask=float(o.get("ask") or 0.0),
                last=(float(o["last"]) if o.get("last") is not None else None),
                volume=int(o.get("volume") or 0),
                open_interest=int(o.get("open_interest") or 0),
                delta=(float(g["delta"]) if g.get("delta") is not None else None),
                gamma=(float(g["gamma"]) if g.get("gamma") is not None else None),
                theta=(float(g["theta"]) if g.get("theta") is not None else None),
                vega=(float(g["vega"]) if g.get("vega") is not None else None),
                bid_iv=(float(g["bid_iv"]) if g.get("bid_iv") is not None else None),
                mid_iv=(float(g["mid_iv"]) if g.get("mid_iv") is not None else None),
                ask_iv=(float(g["ask_iv"]) if g.get("ask_iv") is not None else None),
                smv_vol=(float(g["smv_vol"]) if g.get("smv_vol") is not None else None),
            ))
        time.sleep(self.throttle_sec)
        return out

    # ------------------ Helpers / pickers ------------------
    def _filter_liquidity(self, df: pd.DataFrame, min_oi: int, min_vol: int, max_spread: float) -> pd.DataFrame:
        df = df.copy()
        df["mid"] = (df["bid"] + df["ask"]) / 2.0
        df["spread_pct"] = (df["ask"] - df["bid"]).clip(lower=0) / df["mid"].replace(0, pd.NA)
        return df[
            (df["open_interest"] >= min_oi) &
            (df["volume"] >= min_vol) &
            (df["mid"] > 0) &
            (df["spread_pct"].fillna(1.0) <= max_spread)
        ]

    def _filter_for_theoretical(self, df: pd.DataFrame, min_oi: int) -> pd.DataFrame:
        # Pre-open: allow mid<=0, ignore spread; keep a light OI floor
        return df[df["open_interest"] >= min_oi].copy()

    def _nearest_delta(self, df: pd.DataFrame, target: float, side: str, spot: float) -> Optional[pd.Series]:
        if df.empty:
            return None
        df = df.copy()
        if side == "call":
            df = df[df["strike"] >= spot * 0.99]
        else:
            df = df[df["strike"] <= spot * 1.01]
        if df.empty:
            return None
        df["delta_abs"] = pd.to_numeric(df.get("delta"), errors="coerce").abs()
        df["delta_diff"] = (df["delta_abs"] - target).abs()
        df["mid"] = (df["bid"] + df["ask"]) / 2.0
        df["spread_pct"] = (df["ask"] - df["bid"]).clip(lower=0) / df["mid"].replace(0, pd.NA)
        df = df.sort_values(by=["delta_diff", "spread_pct", "open_interest"], ascending=[True, True, False])
        return df.iloc[0] if not df.empty else None

    def select_directional(self, chain_df: pd.DataFrame, side: str, spot: float,
                           target_delta: float, min_oi: int, min_vol: int, max_spread: float) -> Optional[pd.Series]:
        df = chain_df[chain_df["option_type"] == side].copy()
        df = self._filter_liquidity(df, min_oi, min_vol, max_spread)
        if df.empty:
            return None
        return self._nearest_delta(df, target_delta, side, spot)

    def select_directional_theo(self, chain_df: pd.DataFrame, side: str, spot: float,
                                target_delta: float, min_oi: int) -> Optional[pd.Series]:
        df = chain_df[chain_df["option_type"] == side].copy()
        df = self._filter_for_theoretical(df, min_oi)
        if df.empty:
            return None
        return self._nearest_delta(df, target_delta, side, spot)

    def select_straddle(self, chain_df: pd.DataFrame, spot: float,
                        min_oi: int, min_vol: int, max_spread: float) -> Optional[Tuple[pd.Series, pd.Series]]:
        df = self._filter_liquidity(chain_df.copy(), min_oi, min_vol, max_spread)
        if df.empty:
            return None
        df["moneyness"] = (df["strike"] - spot).abs()
        strikes_ranked = df.groupby("strike")["moneyness"].first().reset_index().sort_values("moneyness")
        for _, row in strikes_ranked.iterrows():
            K = row["strike"]
            call = df[(df["strike"] == K) & (df["option_type"] == "call")]
            put  = df[(df["strike"] == K) & (df["option_type"] == "put")]
            if not call.empty and not put.empty:
                return call.iloc[0], put.iloc[0]
        return None

    def select_straddle_theo(self, chain_df: pd.DataFrame, spot: float, min_oi: int) -> Optional[Tuple[pd.Series, pd.Series]]:
        df = self._filter_for_theoretical(chain_df.copy(), min_oi)
        if df.empty:
            return None
        df["moneyness"] = (df["strike"] - spot).abs()
        strikes_ranked = df.groupby("strike")["moneyness"].first().reset_index().sort_values("moneyness")
        for _, row in strikes_ranked.iterrows():
            K = row["strike"]
            call = df[(df["strike"] == K) & (df["option_type"] == "call")]
            put  = df[(df["strike"] == K) & (df["option_type"] == "put")]
            if not call.empty and not put.empty:
                return call.iloc[0], put.iloc[0]
        return None

    # ------------------ Synthetic fallback (when expirations/chains are empty) ------------------
    def _pick_preferred_expiry(self, dte_min: int, dte_max: int) -> Tuple[date, int]:
        """Pick a Friday inside [dte_min, dte_max]; else use midpoint."""
        today = date.today()
        for d in range(dte_min, dte_max + 1):
            dt = today + timedelta(days=d)
            if dt.weekday() == 4:
                return dt, d
        d = max(dte_min, min(dte_max, (dte_min + dte_max) // 2))
        return today + timedelta(days=d), d

    def _make_synth_row(self, symbol: str, side: str, spot: float, K: float, dte: int,
                        hv20: float, risk_dollars: float, iv_mult: float, play_type_label: str) -> Dict:
        T = max(dte, 1) / TRADING_DAYS
        iv_used = max(hv20 * iv_mult, 1e-6)
        mid = black_scholes_price("call" if side == "call" else "put", spot, K, iv_used, T, 0.0)  # per share
        entry_contract = mid * 100.0
        units = int(risk_dollars // entry_contract) if entry_contract > 0 else 0
        return {
            "underlying": symbol,
            "play_type": side if play_type_label == "dir" else "straddle",
            "symbol_opt": f"{symbol}-synth-{side}-{K:.2f}-{dte}",
            "dte": dte,
            "strike_rule": ("ATM (call+put)" if play_type_label == "straddle" else f"SYNTH {side.upper()} ≈ Δ0.30"),
            "target_delta": 0.30,
            "entry_price": (entry_contract if play_type_label != "straddle" else None),  # straddle handled separately
            "contract_size": 100,
            "units": units if units > 0 else 0,
            "affordable": bool(units > 0),
            "pricing_source": "theoretical",
            "K": float(K),
            "S0": float(spot),
            "T0": T,
            "IV_used": iv_used,
            "bid": 0.0, "ask": 0.0, "spread_pct": None,
            "oi": 0, "vol": 0,
        }

    def _synthetic_candidates(self, symbol: str, spot: float, s_dir: float, s_vol: float, s_brk: float,
                              hv20: float, risk_dollars: float, dte_min: int, dte_max: int,
                              iv_fallback_mult: float) -> pd.DataFrame:
        _, dte = self._pick_preferred_expiry(dte_min, dte_max)
        K_call = round(spot * 1.05, 2)
        K_put  = round(spot * 0.95, 2)
        K_atm  = round(spot, 2)

        rows = []
        if s_dir >= 0.25 and s_brk >= 0:
            rows.append(self._make_synth_row(symbol, "call", spot, K_call, dte, hv20, risk_dollars, iv_fallback_mult, "dir"))
        if s_dir <= -0.25 and s_brk <= 0:
            rows.append(self._make_synth_row(symbol, "put",  spot, K_put,  dte, hv20, risk_dollars, iv_fallback_mult, "dir"))

        want_straddle = (abs(s_dir) < 0.20 and s_vol >= 0.45) or (s_vol >= 0.65) or not rows
        if want_straddle:
            T = max(dte, 1) / TRADING_DAYS
            iv_used = max(hv20 * iv_fallback_mult, 1e-6)
            c_mid = black_scholes_price("call", spot, K_atm, iv_used, T, 0.0)
            p_mid = black_scholes_price("put",  spot, K_atm, iv_used, T, 0.0)
            entry_contract = (c_mid + p_mid) * 100.0
            units = int(risk_dollars // entry_contract) if entry_contract > 0 else 0
            rows.append({
                "underlying": symbol,
                "play_type": "straddle",
                "symbol_opt": f"{symbol}-synth-straddle-{K_atm:.2f}-{dte}",
                "dte": dte,
                "strike_rule": "ATM (call+put)",
                "target_delta": 0.50,
                "entry_price": entry_contract,
                "contract_size": 100,
                "units": units if units > 0 else 0,
                "affordable": bool(units > 0),
                "pricing_source": "theoretical",
                "K": float(K_atm),
                "S0": float(spot),
                "T0": T,
                "IV_used": iv_used,
                "call_iv": iv_used, "put_iv": iv_used,
                "bid": 0.0, "ask": 0.0, "spread_pct": None,
                "oi": 0, "vol": 0,
            })
        return pd.DataFrame(rows)

    # ------------------ Main (market-open + pre-open capable + synthetic safety net) ------------------
    def build_live_candidates(
        self,
        symbol: str,
        spot: float,
        s_dir: float,
        s_vol: float,
        s_brk: float,
        hv20: float,
        risk_dollars: float,
        dte_min: int,
        dte_max: int,
        iv_fallback_mult: float = 1.2,
        min_oi: int = 50,
        min_vol: int = 0,
        max_spread: float = 0.5,
        max_expiries: int = 2,
        use_theoretical_fallback: bool = False,
        theo_spread_threshold: float = 1.0,
        relax_gates_if_empty: bool = True,
    ) -> pd.DataFrame:
        """
        Build candidates. If use_theoretical_fallback=True and quotes are unusable,
        price with Black–Scholes using mid_iv (or hv20 * iv_fallback_mult).
        If expirations/chains are empty, synthesize ATM ideas so you never get an empty table.
        """
        expirations_all = self.get_expirations(symbol)
        expirations = [e for e in expirations_all if dte_min <= _dte(e) <= dte_max][:max_expiries]

        if not expirations_all:
            return self._synthetic_candidates(symbol, spot, s_dir, s_vol, s_brk, hv20, risk_dollars, dte_min, dte_max, iv_fallback_mult)

        if not expirations:
            alt = [e for e in expirations_all if _dte(e) <= 30][:max_expiries]
            expirations = alt

        rows: List[Dict] = []

        # Gates
        bullish_gate  = (s_dir >= 0.40) and (s_brk >= 0)
        bearish_gate  = (s_dir <= -0.40) and (s_brk <= 0)
        straddle_gate = (abs(s_dir) < 0.20 and s_vol >= 0.50) or (s_vol >= 0.65)

        # Strict/normal pass
        for exp in expirations:
            chain = self.get_chain(symbol, exp, want_greeks=True)
            if not chain:
                continue
            cdf = pd.DataFrame([c.__dict__ for c in chain])
            if cdf.empty:
                continue

            if bullish_gate:
                pick = self.select_directional(cdf, "call", spot, 0.30, min_oi, min_vol, max_spread)
                if pick is None and use_theoretical_fallback:
                    pick = self.select_directional_theo(cdf, "call", spot, 0.30, max(5, min_oi))
                if pick is not None:
                    rows.append(self._row_from_pick("call", symbol, spot, hv20, risk_dollars, pick, iv_fallback_mult, use_theoretical_fallback, theo_spread_threshold))

            if bearish_gate:
                pick = self.select_directional(cdf, "put", spot, 0.30, min_oi, min_vol, max_spread)
                if pick is None and use_theoretical_fallback:
                    pick = self.select_directional_theo(cdf, "put", spot, 0.30, max(5, min_oi))
                if pick is not None:
                    rows.append(self._row_from_pick("put", symbol, spot, hv20, risk_dollars, pick, iv_fallback_mult, use_theoretical_fallback, theo_spread_threshold))

            if straddle_gate:
                pair = self.select_straddle(cdf, spot, min_oi, min_vol, max_spread)
                if pair is None and use_theoretical_fallback:
                    pair = self.select_straddle_theo(cdf, spot, max(5, min_oi))
                if pair is not None:
                    call, put = pair
                    rows.append(self._row_from_straddle(symbol, spot, hv20, risk_dollars, call, put, iv_fallback_mult, use_theoretical_fallback, theo_spread_threshold))

        # Relaxed pass (pre-open friendliness) if empty
        if not rows and relax_gates_if_empty:
            for exp in expirations:
                chain = self.get_chain(symbol, exp, want_greeks=True)
                if not chain:
                    continue
                cdf = pd.DataFrame([c.__dict__ for c in chain])
                if cdf.empty:
                    continue

                if s_dir >= 0.25:
                    pick = self.select_directional_theo(cdf, "call", spot, 0.30, max(5, min_oi//2))
                    if pick is not None:
                        rows.append(self._row_from_pick("call", symbol, spot, hv20, risk_dollars, pick, iv_fallback_mult, True, theo_spread_threshold))
                if s_dir <= -0.25:
                    pick = self.select_directional_theo(cdf, "put", spot, 0.30, max(5, min_oi//2))
                    if pick is not None:
                        rows.append(self._row_from_pick("put", symbol, spot, hv20, risk_dollars, pick, iv_fallback_mult, True, theo_spread_threshold))
                if s_vol >= 0.45:
                    pair = self.select_straddle_theo(cdf, spot, max(5, min_oi//2))
                    if pair is not None:
                        call, put = pair
                        rows.append(self._row_from_straddle(symbol, spot, hv20, risk_dollars, call, put, iv_fallback_mult, True, theo_spread_threshold))

        if not rows:
            return self._synthetic_candidates(symbol, spot, s_dir, s_vol, s_brk, hv20, risk_dollars, dte_min, dte_max, iv_fallback_mult)

        return pd.DataFrame(rows)

    # ------------------ Row builders (with theoretical fallback) ------------------
    def _row_from_pick(
        self,
        side: str,
        symbol: str,
        spot: float,
        hv20: float,
        risk_dollars: float,
        pick: pd.Series,
        iv_fallback_mult: float,
        use_theoretical: bool,
        theo_spread_threshold: float,
    ) -> Dict:
        mid = float(pick.get("mid") or 0.0)
        spread_pct = float(pick.get("spread_pct") or 9e9)
        iv_used = float(pick.get("mid_iv") or pick.get("smv_vol") or hv20 * iv_fallback_mult or 0.0)
        dte = int(pick["dte"])
        T = dte / TRADING_DAYS
        K = float(pick["strike"])

        pricing_source = "mid"
        if (mid <= 0) or (use_theoretical and (spread_pct is None or spread_pct > theo_spread_threshold)):
            theo = black_scholes_price(side, spot, K, iv_used, T, 0.0)
            mid = max(theo, 0.01)  # per-share
            pricing_source = "theoretical"

        entry_contract = mid * 100.0
        units = int(risk_dollars // entry_contract) if entry_contract > 0 else 0

        return {
            "underlying": symbol,
            "play_type": side,
            "symbol_opt": str(pick.get("symbol") or f"{symbol}-theo-{side}-{K:.2f}-{dte}"),
            "dte": dte,
            "strike_rule": f"Nearest Δ≈0.30 ({side})",
            "target_delta": 0.30,
            "entry_price": entry_contract,
            "contract_size": 100,
            "units": units if units > 0 else 0,
            "affordable": bool(units > 0),
            "pricing_source": pricing_source,
            "K": K,
            "S0": float(spot),
            "T0": T,
            "IV_used": iv_used,
            "bid": float(pick.get("bid", 0.0)),
            "ask": float(pick.get("ask", 0.0)),
            "spread_pct": None if spread_pct >= 9e9 else spread_pct,
            "oi": int(pick.get("open_interest", 0)),
            "vol": int(pick.get("volume", 0)),
        }

    def _row_from_straddle(
        self,
        symbol: str,
        spot: float,
        hv20: float,
        risk_dollars: float,
        call: pd.Series,
        put: pd.Series,
        iv_fallback_mult: float,
        use_theoretical: bool,
        theo_spread_threshold: float,
    ) -> Dict:
        dte = int(call["dte"])
        T = dte / TRADING_DAYS
        K = float(call["strike"])
        mid_call = float(call.get("mid") or 0.0)
        mid_put  = float(put.get("mid")  or 0.0)
        sp_call = float(call.get("spread_pct") or 9e9)
        sp_put  = float(put.get("spread_pct")  or 9e9)

        iv_call = float(call.get("mid_iv") or call.get("smv_vol") or hv20 * iv_fallback_mult or 0.0)
        iv_put  = float(put.get("mid_iv")  or put.get("smv_vol")  or hv20 * iv_fallback_mult or 0.0)

        pricing_source = "mid"
        if (mid_call <= 0 or mid_put <= 0) or (use_theoretical and ((sp_call is None or sp_call > theo_spread_threshold) or (sp_put is None or sp_put > theo_spread_threshold))):
            mid_call = black_scholes_price("call", spot, K, iv_call, T, 0.0)
            mid_put  = black_scholes_price("put",  spot, K, iv_put,  T, 0.0)
            pricing_source = "theoretical"

        entry_contract = (mid_call + mid_put) * 100.0
        units = int(risk_dollars // entry_contract) if entry_contract > 0 else 0

        return {
            "underlying": symbol,
            "play_type": "straddle",
            "symbol_opt": f"{symbol}-straddle-{K:.2f}-{dte}",
            "dte": dte,
            "strike_rule": "ATM (call+put)",
            "target_delta": 0.50,
            "entry_price": entry_contract,
            "contract_size": 100,
            "units": units if units > 0 else 0,
            "affordable": bool(units > 0),
            "pricing_source": pricing_source,
            "K": K,
            "S0": float(spot),
            "T0": T,
            "IV_used": max(iv_call, iv_put),
            "call_iv": iv_call, "put_iv": iv_put,
            "bid": 0.0, "ask": 0.0,
            "spread_pct": None,   # not meaningful for straddle aggregate
            "oi": int(min(call.get("open_interest", 0), put.get("open_interest", 0))),
            "vol": int(min(call.get("volume", 0), put.get("volume", 0))),
        }
