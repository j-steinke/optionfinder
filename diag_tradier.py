# diag_tradier.py
import os, sys, json
from datetime import datetime, date
from pathlib import Path
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)

def base_url():
    env = (os.getenv("TRADIER_ENV") or "production").strip().lower()
    return "https://api.tradier.com" if env == "production" else "https://sandbox.tradier.com"

def headers():
    tok = os.getenv("TRADIER_ACCESS_TOKEN", "").strip()
    if not tok:
        raise SystemExit("TRADIER_ACCESS_TOKEN missing in env")
    return {"Authorization": f"Bearer {tok}", "Accept": "application/json"}

def get(url, **params):
    r = requests.get(url, headers=headers(), params=params, timeout=20)
    return r.status_code, r.text, (r.headers.get("content-type") or "")

def main(sym: str):
    bu = base_url()
    print(f"ENV={os.getenv('TRADIER_ENV','(unset)')}  BASE={bu}")

    # 0) Quotes (equity) — should work if token’s good
    sc, body, ct = get(f"{bu}/v1/markets/quotes", symbols=sym)
    print(f"quotes status={sc} ct={ct}\n{body[:400]}\n")

    # 1) Lookup option symbols — verifies roots (should show something like AMD)
    sc, body, ct = get(f"{bu}/v1/markets/lookup/options", underlying=sym)
    print(f"lookup options status={sc} ct={ct}\n{body[:400]}\n")

    # 2) Expirations (the failing one)
    sc, body, ct = get(f"{bu}/v1/markets/options/expirations", symbol=sym, includeAllRoots="true", strikes="false")
    print(f"expirations status={sc} ct={ct}\n{body[:800]}\n")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "AMD")
