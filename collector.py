# collector.py
import os
from datetime import date, timedelta
from typing import List, Dict, Any

import pandas as pd
from dotenv import load_dotenv
from polygon import RESTClient

def _to_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return obj  # last resort

def _normalize_agg_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df.empty:
        return df
    rename = {"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close",
              "v": "volume", "vw": "vwap", "n": "transactions"}
    for k, v in rename.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["ticker"] = ticker
    return df.sort_values("timestamp").reset_index(drop=True)

class PolygonStocksCollector:
    """Free-plan friendly: daily and minute aggregates only."""
    def __init__(self, paginate: bool = True, trace: bool = False):
        load_dotenv(override=False)
        key = os.getenv("POLYGON_API_KEY")
        if not key:
            raise RuntimeError("Set POLYGON_API_KEY in your environment or .env file.")
        self.client = RESTClient(api_key=key, pagination=paginate, trace=trace)

    def get_daily_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for agg in self.client.list_aggs(
            ticker=ticker, multiplier=1, timespan="day", from_=start, to=end, adjusted=True, limit=50_000
        ):
            rows.append(_to_dict(agg))
        return _normalize_agg_df(pd.DataFrame(rows), ticker)

    def get_minute_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for agg in self.client.list_aggs(
            ticker=ticker, multiplier=1, timespan="minute", from_=start, to=end, adjusted=True, limit=50_000
        ):
            rows.append(_to_dict(agg))
        return _normalize_agg_df(pd.DataFrame(rows), ticker)

def default_date_windows():
    today = date.today()
    daily_start = (today - timedelta(days=120)).strftime("%Y-%m-%d")  # 60 trading days cushion
    daily_end = today.strftime("%Y-%m-%d")
    minute_start = (today - timedelta(days=5)).strftime("%Y-%m-%d")
    minute_end = daily_end
    return daily_start, daily_end, minute_start, minute_end
