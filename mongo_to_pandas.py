#!/usr/bin/env python3
import os
import pandas as pd
from datetime import timezone
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "marketdata")

def load_bars(symbol: str, start: str = None, end: str = None, use_adj_close=False) -> pd.DataFrame:
    client = MongoClient(MONGODB_URI)
    coll = client[DB_NAME]["daily_bars"]

    q = {"symbol": symbol.upper()}
    if start or end:
        q["date"] = {}
        if start:
            q["date"]["$gte"] = pd.to_datetime(start).tz_localize(timezone.utc).to_pydatetime()
        if end:
            q["date"]["$lte"] = pd.to_datetime(end).tz_localize(timezone.utc).to_pydatetime()
        if not q["date"]:
            del q["date"]

    cur = coll.find(q, {"_id": 0}).sort("date", 1)
    df = pd.DataFrame(list(cur))
    if df.empty:
        return df

    df.index = pd.to_datetime(df["date"])
    df.drop(columns=["date"], inplace=True)

    # Choose price column for signals
    if use_adj_close and "adj_close" in df.columns and df["adj_close"].notna().any():
        df["px"] = df["adj_close"]
    else:
        df["px"] = df["close"]

    # Keep expected columns if present
    keep = [c for c in ["open","high","low","close","adj_close","volume","px","source"] if c in df.columns]
    return df[keep]
