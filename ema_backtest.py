#!/usr/bin/env python3
# ema_backtest.py — Seykota-style EMA(15/150) + ATR(20), heat sizing, with MongoDB persistence

import numpy as np
import pandas as pd
from datetime import datetime, timezone
import argparse

from mongo_to_pandas import load_bars

from results_to_mongo import (
    get_db,
    new_run_id,
    write_run,
    write_trades,
    write_equity,
)

# Parameters (Seykota defaults)
FAST = 15
SLOW = 150
ATR_LAG = 20
ATR_MULT = 5.0
HEAT = 0.10
START_EQUITY = 1_000_000.0


# Indicators 
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["px"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df: pd.DataFrame, lag: int) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(span=lag, adjust=False).mean()

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = ema(df["px"], FAST)
    df["ema_slow"] = ema(df["px"], SLOW)
    sig = (df["ema_fast"] > df["ema_slow"]).astype(int)
    df["buy"]  = (sig.eq(1) & sig.shift(1).fillna(0).eq(0))
    df["sell"] = (sig.eq(0) & sig.shift(1).fillna(0).eq(1))
    return df


# Sizing 
def sey_position_size(equity: float, heat: float, atr_val: float, atr_mult: float) -> int:
    risk_per_unit = atr_mult * float(atr_val)
    if risk_per_unit <= 0 or np.isnan(risk_per_unit):
        return 0
    return int(round((equity * heat) / risk_per_unit))


# Backtest 
def backtest_single(symbol: str, use_adj_close: bool = False, start: str = "1990-01-01") -> pd.DataFrame:
    df = load_bars(symbol, start=start, use_adj_close=use_adj_close)
    if df.empty:
        raise SystemExit(f"No data for {symbol}. Did you import it?")

    for c in ["open", "high", "low", "px"]:
        if c not in df.columns:
            raise SystemExit(f"{symbol}: missing column '{c}' in DB data")

    df = df.dropna(subset=["open", "high", "low", "px"])
    df = generate_signals(df)
    df["atr"] = atr(df, ATR_LAG)

    equity = START_EQUITY
    position = 0
    entry_px = np.nan

    recs = []
    for ts, row in df.iterrows():
        # Sell on open if we were long and got a sell signal yesterday
        if position > 0 and row["sell"]:
            exit_px = row["open"]
            pnl = position * (exit_px - entry_px)
            equity += pnl
            position = 0
            entry_px = np.nan
            recs.append({"ts": ts, "event": "SELL", "px": float(exit_px), "pnl": float(pnl), "equity": float(equity)})

        # Buy on open if we were flat and got a buy signal yesterday 
        if position == 0 and row["buy"]:
            size = sey_position_size(equity, HEAT, row["atr"], ATR_MULT)
            if size > 0:
                entry_px = row["open"]
                position = size
                recs.append({"ts": ts, "event": "BUY", "px": float(entry_px), "pnl": 0.0, "equity": float(equity)})

        # MTM equity each bar
        ref_entry = entry_px if not np.isnan(entry_px) else row["px"]
        cur_px = row["px"] if not np.isnan(row["px"]) else row["open"]
        mtm_equity = equity + position * (cur_px - ref_entry)
        recs.append({"ts": ts, "event": "MTM", "px": float(cur_px), "pnl": 0.0, "equity": float(mtm_equity)})

    blotter = pd.DataFrame.from_records(recs).set_index("ts")
    return blotter


# Stats
def perf_stats(equity_curve: pd.Series) -> dict:
    eq = equity_curve.dropna()
    if eq.empty:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan}
    ret = eq.pct_change().fillna(0.0)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (252 / len(eq)) - 1 if len(eq) > 0 else np.nan
    vol = ret.std() * np.sqrt(252)
    sharpe = (ret.mean() * 252) / vol if vol and vol > 0 else np.nan
    dd = (eq / eq.cummax() - 1.0)
    maxdd = dd.min()
    return {"CAGR": float(cagr), "Vol": float(vol), "Sharpe": float(sharpe), "MaxDD": float(maxdd)}


# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol", nargs="?", default="AAPL")
    parser.add_argument("--start", default="1990-01-01")
    parser.add_argument("--adj", action="store_true", help="Use adjusted close for signals")
    parser.add_argument("--persist", action="store_true", help="Write run, trades, and equity to MongoDB")
    args = parser.parse_args()

    sym = args.symbol.upper()
    blotter = backtest_single(sym, use_adj_close=args.adj, start=args.start)

    # Equity series for stats
    eq = blotter.loc[blotter["event"] == "MTM", "equity"]
    stats = perf_stats(eq)
    print(sym, stats)

    if args.persist:
        params = {
            "FAST": FAST,
            "SLOW": SLOW,
            "ATR_LAG": ATR_LAG,
            "ATR_MULT": ATR_MULT,
            "HEAT": HEAT,
            "START_EQUITY": START_EQUITY,
            "use_adj_close": args.adj,
            "start": args.start,
        }
        run_id = new_run_id(prefix=f"sey_{sym}")
        run_doc = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc),
            "engine": "sey_ema",
            "universe": [sym],
            "params": params,
            "stats": stats,
        }

        db = get_db()
        write_run(db, run_doc)

        # Trades = BUY/SELL rows only
        trade_rows = blotter[blotter["event"].isin(["BUY", "SELL"])].reset_index().to_dict("records")
        write_trades(db, run_id, sym, trade_rows)

        # Equity = MTM rows
        write_equity(db, run_id, eq)

        print(f"Persisted to Mongo with run_id={run_id}")
