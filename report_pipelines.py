#!/usr/bin/env python3
# report_pipelines.py — Aggregation pipelines via PyMongo (MongoDB 5+), Python 3.8.10 compatible


import os
from pprint import pprint
from datetime import datetime
from typing import List, Tuple, Dict, Any

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "marketdata")

#  helpers 

def get_db():
    return MongoClient(MONGODB_URI)[DB_NAME]

def latest_portfolio_run(db, engine: str = "sey_portfolio_ema") -> Tuple[str, Dict[str, Any]]:
    doc = db.runs.find_one({"engine": engine}, sort=[("created_at", -1)])
    if not doc:
        raise RuntimeError(f"No runs found for engine={engine}")
    return doc["run_id"], doc

#aggregation pipelines 

def equity_cagr_maxdd(db, run_id: str) -> Dict[str, Any]:
    """
    Computes:
      - first/last timestamps & equity
      - Max Drawdown (fraction)
      - Years elapsed (via date subtraction, no $dateDiff)
      - CAGR
    """
    pipe = [
        {"$match": {"run_id": run_id}},
        {"$sort": {"ts": 1}},
        {"$setWindowFields": {
            "sortBy": {"ts": 1},
            "output": {
                "cumMaxEq": {"$max": "$equity", "window": {"documents": ["unbounded", "current"]}}
            }
        }},
        {"$addFields": {"dd": {"$subtract": [{"$divide": ["$equity", "$cumMaxEq"]}, 1]}}},
        {"$group": {
            "_id": None,
            "firstTs": {"$first": "$ts"},
            "lastTs": {"$last": "$ts"},
            "firstEq": {"$first": "$equity"},
            "lastEq": {"$last": "$equity"},
            "n": {"$sum": 1},
            "maxDD": {"$min": "$dd"}
        }},
        # Years = (lastTs - firstTs) / ms_per_year
        {"$addFields": {
            "years": {
                "$divide": [
                    {"$subtract": ["$lastTs", "$firstTs"]},
                    1000 * 60 * 60 * 24 * 365.25
                ]
            },
            "CAGR": {
                "$cond": [
                    {"$gt": ["$years", 0]},
                    {"$subtract": [
                        {"$pow": [{"$divide": ["$lastEq", "$firstEq"]}, {"$divide": [1, "$years"]}]},
                        1
                    ]},
                    None
                ]
            }
        }},
        {"$project": {"_id": 0}}
    ]
    out = list(db.equity.aggregate(pipe))
    return out[0] if out else {}

def equity_sharpe(db, run_id: str) -> Dict[str, Any]:
    """
    Annualized mean/volatility and Sharpe (rf=0) from daily equity returns.
    """
    pipe = [
        {"$match": {"run_id": run_id}},
        {"$sort": {"ts": 1}},
        {"$setWindowFields": {
            "sortBy": {"ts": 1},
            "output": {"prevEq": {"$shift": {"output": "$equity", "by": 1}}}  # previous row
        }},
        {"$addFields": {
            "ret": {
                "$cond": [
                    {"$and": [{"$ne": ["$prevEq", None]}, {"$gt": ["$prevEq", 0]}]},
                    {"$subtract": [{"$divide": ["$equity", "$prevEq"]}, 1]},
                    None
                ]
            }
        }},
        {"$match": {"ret": {"$ne": None}}},
        {"$group": {
            "_id": None,
            "n": {"$sum": 1},
            "meanRet": {"$avg": "$ret"},
            "stdRet": {"$stdDevPop": "$ret"}
        }},
        {"$addFields": {
            "annMean": {"$multiply": ["$meanRet", 252]},
            "annVol": {"$multiply": ["$stdRet", {"$sqrt": 252}]},
            "Sharpe": {
                "$cond": [
                    {"$gt": ["$stdRet", 0]},
                    {"$divide": [
                        {"$multiply": ["$meanRet", 252]},
                        {"$multiply": ["$stdRet", {"$sqrt": 252}]}
                    ]},
                    None
                ]
            }
        }},
        {"$project": {"_id": 0}}
    ]
    out = list(db.equity.aggregate(pipe))
    return out[0] if out else {}

def trades_summary_by_symbol(db, run_id: str) -> List[Dict[str, Any]]:
    """
    Summarize realized SELL P&L per symbol: trades, winRate, ProfitFactor, avgWin/avgLoss, sumPnL.
    """
    pipe = [
        {"$match": {"run_id": run_id, "event": {"$in": ["BUY", "SELL"]}}},  # safety
        {"$match": {"event": "SELL"}},
        {"$addFields": {
            "win": {"$gt": ["$pnl", 0]},
            "loss": {"$lt": ["$pnl", 0]},
            "grossWin": {"$cond": [{"$gt": ["$pnl", 0]}, "$pnl", 0]},
            "grossLoss": {"$cond": [{"$lt": ["$pnl", 0]}, {"$abs": "$pnl"}, 0]}
        }},
        {"$group": {
            "_id": "$symbol",
            "trades": {"$sum": 1},
            "wins": {"$sum": {"$cond": ["$win", 1, 0]}},
            "losses": {"$sum": {"$cond": ["$loss", 1, 0]}},
            "sumPnL": {"$sum": "$pnl"},
            "grossWin": {"$sum": "$grossWin"},
            "grossLoss": {"$sum": "$grossLoss"},
            "avgWin": {"$avg": {"$cond": [{"$gt": ["$pnl", 0]}, "$pnl", None]}},
            "avgLoss": {"$avg": {"$cond": [{"$lt": ["$pnl", 0]}, "$pnl", None]}}
        }},
        {"$addFields": {
            "winRate": {"$cond": [{"$gt": ["$trades", 0]}, {"$divide": ["$wins", "$trades"]}, None]},
            "profitFactor": {"$cond": [{"$gt": ["$grossLoss", 0]}, {"$divide": ["$grossWin", "$grossLoss"]}, None]}
        }},
        {"$project": {
            "_id": 0,
            "symbol": "$_id",
            "trades": 1,
            "wins": 1,
            "losses": 1,
            "winRate": 1,
            "sumPnL": 1,
            "profitFactor": 1,
            "avgWin": 1,
            "avgLoss": 1
        }},
        {"$sort": {"profitFactor": -1}}
    ]
    return list(db.trades.aggregate(pipe))

def equity_slice(db, run_id: str, start_iso: str, end_iso: str) -> List[Dict[str, Any]]:
    """
    Return time-sliced equity series for a given run_id.
    Uses Python to build datetime bounds (no $dateFromString).
    """
    start_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))

    pipe = [
        {"$match": {
            "run_id": run_id,
            "ts": {"$gte": start_dt, "$lte": end_dt}
        }},
        {"$project": {"_id": 0, "ts": 1, "equity": 1}},
        {"$sort": {"ts": 1}}
    ]
    return list(db.equity.aggregate(pipe))

def compare_runs_params(db, engine: str = "sey_portfolio_ema") -> List[Dict[str, Any]]:
    """
    Compare parameter sets across runs for a given engine.
    """
    pipe = [
        {"$match": {"engine": engine}},
        {"$project": {
            "_id": 0,
            "run_id": 1,
            "created_at": 1,
            "FAST": "$params.FAST",
            "SLOW": "$params.SLOW",
            "CAP": "$params.PORTFOLIO_HEAT_CAP",
            "HEAT": "$params.HEAT",
            "CAGR": "$stats.CAGR",
            "Sharpe": "$stats.Sharpe",
            "MaxDD": "$stats.MaxDD"
        }},
        {"$sort": {"created_at": -1}}
    ]
    return list(db.runs.aggregate(pipe))

def materialize_trades_summary(db, run_id: str, into_collection: str = "reports_trades_by_symbol") -> int:
    """
    Writes a per-symbol trades report into a materialized collection via $merge.
    Returns inserted/updated count (best-effort; count the docs for run afterward).
    """
    pipe = [
        {"$match": {"run_id": run_id, "event": "SELL"}},
        {"$addFields": {
            "win": {"$gt": ["$pnl", 0]},
            "loss": {"$lt": ["$pnl", 0]},
            "grossWin": {"$cond": [{"$gt": ["$pnl", 0]}, "$pnl", 0]},
            "grossLoss": {"$cond": [{"$lt": ["$pnl", 0]}, {"$abs": "$pnl"}, 0]}
        }},
        {"$group": {
            "_id": "$symbol",
            "trades": {"$sum": 1},
            "wins": {"$sum": {"$cond": ["$win", 1, 0]}},
            "losses": {"$sum": {"$cond": ["$loss", 1, 0]}},
            "sumPnL": {"$sum": "$pnl"},
            "grossWin": {"$sum": "$grossWin"},
            "grossLoss": {"$sum": "$grossLoss"},
            "avgWin": {"$avg": {"$cond": [{"$gt": ["$pnl", 0]}, "$pnl", None]}},
            "avgLoss": {"$avg": {"$cond": [{"$lt": ["$pnl", 0]}, "$pnl", None]}}
        }},
        {"$addFields": {
            "winRate": {"$cond": [{"$gt": ["$trades", 0]}, {"$divide": ["$wins", "$trades"]}, None]},
            "profitFactor": {"$cond": [{"$gt": ["$grossLoss", 0]}, {"$divide": ["$grossWin", "$grossLoss"]}, None]}
        }},
        {"$project": {
            "_id": 0,
            "symbol": "$_id",
            "run_id": {"$literal": run_id},
            "trades": 1,
            "wins": 1,
            "losses": 1,
            "winRate": 1,
            "sumPnL": 1,
            "profitFactor": 1,
            "avgWin": 1,
            "avgLoss": 1
        }},
        {"$merge": {"into": into_collection, "whenMatched": "replace", "whenNotMatched": "insert"}}
    ]
    # Exhaust cursor to execute
    list(db.trades.aggregate(pipe))
    return db[into_collection].count_documents({"run_id": run_id})

# CLI entry 

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="sey_portfolio_ema", help="Engine filter for latest run")
    ap.add_argument("--run-id", default=None, help="Explicit run_id to report on")
    ap.add_argument("--slice-start", default="2015-01-01", help="Equity slice start (ISO)")
    ap.add_argument("--slice-end", default="2025-01-01", help="Equity slice end (ISO)")
    ap.add_argument("--materialize", action="store_true", help="Materialize trades summary into reports collection")
    args = ap.parse_args()

    db = get_db()

    if args.run_id:
        run_id = args.run_id
        run_doc = db.runs.find_one({"run_id": run_id}) or {}
        if not run_doc:
            raise SystemExit(f"run_id not found: {run_id}")
    else:
        run_id, run_doc = latest_portfolio_run(db, engine=args.engine)

    print("Run:", run_id)
    print("Params:", run_doc.get("params", {}))
    print()

    print("== Equity CAGR & MaxDD ==")
    pprint(equity_cagr_maxdd(db, run_id)); print()

    print("== Equity Sharpe ==")
    pprint(equity_sharpe(db, run_id)); print()

    print("== Trades summary by symbol (top 10) ==")
    rows = trades_summary_by_symbol(db, run_id)
    pprint(rows[:10]); print()

    print(f"== Equity slice {args.slice_start} → {args.slice_end} ==")
    pprint(equity_slice(db, run_id, args.slice_start, args.slice_end)[:5]); print()

    print("== Compare runs/params (latest 5) ==")
    cmp_rows = compare_runs_params(db, engine=args.engine)
    pprint(cmp_rows[:5]); print()

    if args.materialize:
        count = materialize_trades_summary(db, run_id)
        print(f"Materialized reports_trades_by_symbol rows for run_id={run_id}: {count}")
