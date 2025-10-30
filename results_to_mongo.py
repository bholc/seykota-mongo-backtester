#!/usr/bin/env python3
import os
from datetime import datetime, timezone
from typing import Dict, List

from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "marketdata")

def get_db():
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    # Ensure indexes
    db.runs.create_index([("run_id", 1)], unique=True, name="uq_run_id")
    db.runs.create_index([("created_at", -1)], name="ix_created_at")
    db.trades.create_index([("run_id", 1), ("symbol", 1), ("ts", 1)], name="ix_trades_run_sym_ts")
    db.equity.create_index([("run_id", 1), ("ts", 1)], name="ix_equity_run_ts")
    return db

def new_run_id(prefix="run"):
    # e.g., run_2025-10-30T13-05-22Z_NNNN
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{prefix}_{now}"

def write_run(db, run_doc: Dict):
    db.runs.insert_one(run_doc)

def write_trades(db, run_id: str, symbol: str, trade_rows: List[Dict]):
    if not trade_rows:
        return
    docs = []
    for r in trade_rows:
        docs.append({
            "run_id": run_id,
            "symbol": symbol,
            "ts": r["ts"].to_pydatetime() if hasattr(r["ts"], "to_pydatetime") else r["ts"],
            "event": r["event"],
            "px": float(r["px"]),
            "pnl": float(r.get("pnl", 0.0)),
            "equity_after": float(r.get("equity", 0.0)),
        })
    db.trades.insert_many(docs)

def write_equity(db, run_id: str, equity_series):
    if equity_series is None or equity_series.empty:
        return
    docs = []
    for ts, eq in equity_series.items():
        docs.append({
            "run_id": run_id,
            "ts": ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
            "equity": float(eq),
        })
    db.equity.insert_many(docs)
