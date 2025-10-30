#!/usr/bin/env python3
# yahoo_to_mongo.py with Yahoo -> Stooq fallback
import os, sys
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
import yfinance as yf
from pandas_datareader import data as web
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "marketdata")

def get_coll():
    client = MongoClient(MONGODB_URI)
    coll = client[DB_NAME]["daily_bars"]
    coll.create_index([("symbol", 1), ("date", 1)], unique=True, name="uq_symbol_date")
    return coll

def _normalize(df: pd.DataFrame, symbol: str, source: str) -> pd.DataFrame:
    df = df.copy()
    if "Date" in df.columns:
        df.rename(columns={"Date":"date"}, inplace=True)
    if "Open" in df.columns: df.rename(columns={"Open":"open"}, inplace=True)
    if "High" in df.columns: df.rename(columns={"High":"high"}, inplace=True)
    if "Low"  in df.columns: df.rename(columns={"Low":"low"},   inplace=True)
    if "Close" in df.columns: df.rename(columns={"Close":"close"}, inplace=True)
    if "Adj Close" in df.columns: df.rename(columns={"Adj Close":"adj_close"}, inplace=True)
    if "Volume" in df.columns: df.rename(columns={"Volume":"volume"}, inplace=True)

    # Index->column if needed
    if "date" not in df.columns:
        if df.index.name and df.index.name.lower() in ("date","datetime"):
            df["date"] = df.index
        else:
            df["date"] = df.index

    # Ensure UTC midnight timestamps
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.tz_localize(timezone.utc)

    # If provider lacks adj_close (e.g., stooq), create a placeholder = close
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    # Standard fields
    df["symbol"] = symbol.upper()
    df["source"] = source
    df["tz"] = "America/New_York"

    # Coerce numeric
    for c in ["open","high","low","close","adj_close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep only what we need
    cols = ["date","symbol","open","high","low","close","adj_close","volume","source","tz"]
    df = df[[c for c in cols if c in df.columns]]
    df = df.dropna(subset=["open","high","low","close"])
    df = df.reset_index(drop=True)
    return df

def fetch_yahoo(symbol: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=False,
                     progress=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    return _normalize(df, symbol, source="yahoo")

def fetch_stooq(symbol: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    # Stooq returns descending index; we’ll sort ascending.
    df = web.DataReader(symbol, "stooq", start=start, end=end)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.sort_index().reset_index()
    return _normalize(df, symbol, source="stooq")

def fetch_any(symbol: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    df = fetch_yahoo(symbol, start, end)
    if not df.empty:
        return df
    # fallback
    return fetch_stooq(symbol, start, end)

def last_date(coll, symbol: str):
    doc = coll.find({"symbol": symbol.upper()}).sort("date", -1).limit(1)
    L = list(doc)
    return None if not L else L[0]["date"]

def upsert(coll, df: pd.DataFrame) -> int:
    if df.empty: return 0
    ops = []
    now = datetime.now(timezone.utc)
    for _, r in df.iterrows():
        doc = {
            "symbol": r["symbol"],
            "date": r["date"].to_pydatetime(),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "adj_close": float(r["adj_close"]) if pd.notna(r["adj_close"]) else None,
            "volume": int(r["volume"]) if ("volume" in r and pd.notna(r["volume"])) else None,
            "source": r["source"],
            "tz": r["tz"],
            "ingested_at": now,
        }
        ops.append(UpdateOne(
            {"symbol": doc["symbol"], "date": doc["date"]},
            {"$set": doc},
            upsert=True
        ))
    if not ops: return 0
    coll.bulk_write(ops, ordered=False)
    return len(ops)

def backfill(symbols: List[str], start: Optional[str], end: Optional[str]):
    coll = get_coll()
    total = 0
    for s in symbols:
        s = s.upper()
        s_start = start
        ld = last_date(coll, s)
        if ld:
            s_start = ld.date().isoformat()
        print(f"[{s}] fetch start={s_start}, end={end}")
        df = fetch_any(s, s_start, end)
        n = upsert(coll, df) if not df.empty else 0
        total += n
        print(f"[{s}] upserted {n} (source={'-' if df.empty else df['source'].iloc[0] if 'source' in df.columns else 'unknown'})")
    print(f"Done. Total rows upserted: {total}")

if __name__ == "__main__":
    syms = sys.argv[1:] or ["SPY","QQQ","AAPL","MSFT","NVDA","AMZN"]
    backfill(syms, start="1990-01-01", end=None)
