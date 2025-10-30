#!/usr/bin/env python3
# portfolio_backtest.py — portfolio backtest with heat cap + Mongo persistence

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import argparse

from mongo_to_pandas import load_bars
from ema_backtest import ema, atr, FAST, SLOW, ATR_LAG, ATR_MULT, HEAT, START_EQUITY
from results_to_mongo import get_db, new_run_id, write_run, write_trades, write_equity

# ---------- Signals layer ----------
def build_symbol_frame(symbol: str, start: str = "1990-01-01", use_adj_close: bool = False) -> pd.DataFrame:
    df = load_bars(symbol, start=start, use_adj_close=use_adj_close)
    if df.empty:
        raise RuntimeError(f"{symbol}: no data found in Mongo. Did you ingest it?")
    need = ["open", "high", "low", "px"]
    for c in need:
        if c not in df.columns:
            raise RuntimeError(f"{symbol}: missing column '{c}' in DB data")
    df = df.dropna(subset=["open", "high", "low", "px"]).copy()

    # Indicators
    df["ema_fast"] = ema(df["px"], FAST)
    df["ema_slow"] = ema(df["px"], SLOW)

    # No-lookahead cross: prior-day EMAs drive today's orders
    fast_prev = df["ema_fast"].shift(1)
    slow_prev = df["ema_slow"].shift(1)
    cross_up   = (fast_prev > slow_prev) & (fast_prev.shift(1) <= slow_prev.shift(1))
    cross_down = (fast_prev < slow_prev) & (fast_prev.shift(1) >= slow_prev.shift(1))
    df["buy"]  = cross_up.fillna(False)
    df["sell"] = cross_down.fillna(False)

    df["atr"] = atr(df, ATR_LAG)
    df["symbol"] = symbol.upper()
    return df

def build_universe_frames(symbols: List[str], start: str = "1990-01-01", use_adj_close: bool = False) -> Dict[str, pd.DataFrame]:
    return {s.upper(): build_symbol_frame(s, start=start, use_adj_close=use_adj_close) for s in symbols}

# ---------- Portfolio simulator with heat cap ----------
def sey_position_size(equity: float, heat: float, atr_val: float, atr_mult: float) -> int:
    risk_per_unit = atr_mult * float(atr_val)
    if risk_per_unit <= 0 or np.isnan(risk_per_unit):
        return 0
    return int(np.floor((equity * heat) / risk_per_unit))

def portfolio_backtest(
    symbols: List[str],
    start: str = "1990-01-01",
    use_adj_close: bool = False,
    per_trade_heat: float = HEAT,          # e.g., 0.10
    portfolio_heat_cap: float = 0.20,      # e.g., 0.20
    atr_mult: float = ATR_MULT,            # 5.0
    start_equity: float = START_EQUITY
) -> Tuple[pd.DataFrame, pd.Series]:
    frames = build_universe_frames(symbols, start=start, use_adj_close=use_adj_close)
    all_dates = sorted(set().union(*[df.index.tolist() for df in frames.values()]))

    equity = start_equity
    positions: Dict[str, int] = {s: 0 for s in frames}
    entries: Dict[str, float] = {s: np.nan for s in frames}

    def current_heat(ts) -> float:
        if equity <= 0:
            return 0.0
        h = 0.0
        for s, df in frames.items():
            if positions[s] != 0 and ts in df.index:
                atr_val = float(df.at[ts, "atr"]) if not np.isnan(df.at[ts, "atr"]) else 0.0
                h += (abs(positions[s]) * atr_mult * atr_val) / equity
        return h

    def mtm_equity(ts) -> float:
        mtm = equity
        for s, df in frames.items():
            if positions[s] != 0 and ts in df.index:
                px = float(df.at[ts, "px"]) if not np.isnan(df.at[ts, "px"]) else float(df.at[ts, "open"])
                mtm += positions[s] * (px - (entries[s] if not np.isnan(entries[s]) else px))
        return float(mtm)

    trade_rows = []
    equity_rows = []

    for ts in all_dates:
        # 1) Sells first (today's open)
        for s, df in frames.items():
            if ts not in df.index:
                continue
            if positions[s] > 0 and bool(df.at[ts, "sell"]):
                exit_px = float(df.at[ts, "open"])
                pnl = positions[s] * (exit_px - entries[s])
                equity += pnl
                trade_rows.append({"ts": ts, "symbol": s, "event": "SELL", "px": exit_px, "pnl": pnl, "equity_after": float(equity)})
                positions[s] = 0
                entries[s] = np.nan

        # 2) Gather buy candidates
        buy_candidates = []
        for s, df in frames.items():
            if ts not in df.index or positions[s] != 0:
                continue
            if bool(df.at[ts, "buy"]):
                atr_val = float(df.at[ts, "atr"]) if not np.isnan(df.at[ts, "atr"]) else np.nan
                open_px = float(df.at[ts, "open"]) if not np.isnan(df.at[ts, "open"]) else np.nan
                if np.isnan(atr_val) or atr_val <= 0 or np.isnan(open_px):
                    continue
                full_size = sey_position_size(equity, per_trade_heat, atr_val, atr_mult)
                if full_size > 0:
                    buy_candidates.append({"symbol": s, "atr": atr_val, "open": open_px, "full_size": int(full_size), "desired_heat": per_trade_heat})

        # 3) Enforce portfolio heat cap
        if buy_candidates:
            existing = current_heat(ts)
            remaining = max(0.0, portfolio_heat_cap - existing)
            total_desired = sum(c["desired_heat"] for c in buy_candidates)
            scale = 1.0 if total_desired <= 0 else min(1.0, remaining / total_desired)

            for c in buy_candidates:
                size = int(np.floor(c["full_size"] * scale))
                if size <= 0:
                    continue
                s = c["symbol"]
                positions[s] = size
                entries[s] = c["open"]
                trade_rows.append({"ts": ts, "symbol": s, "event": "BUY", "px": c["open"], "pnl": 0.0, "equity_after": float(equity)})

        # 4) Portfolio MTM
        equity_rows.append({"ts": ts, "equity": mtm_equity(ts)})

    blotter = pd.DataFrame(trade_rows).set_index("ts").sort_index()
    equity_ser = pd.DataFrame(equity_rows).set_index("ts").sort_index()["equity"]
    return blotter, equity_ser

def perf_stats(eq: pd.Series) -> dict:
    eq = eq.dropna()
    if eq.empty:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan}
    ret = eq.pct_change().fillna(0.0)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (252 / len(eq)) - 1 if len(eq) > 0 else np.nan
    vol = ret.std() * np.sqrt(252)
    sharpe = (ret.mean() * 252) / vol if vol and vol > 0 else np.nan
    dd = (eq / eq.cummax() - 1.0)
    maxdd = float(dd.min()) if not dd.empty else np.nan
    return {"CAGR": float(cagr), "Vol": float(vol), "Sharpe": float(sharpe), "MaxDD": maxdd}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["SPY","AAPL","MSFT"])
    parser.add_argument("--start", default="1990-01-01")
    parser.add_argument("--adj", action="store_true", help="Use adjusted close for signals")
    parser.add_argument("--cap", type=float, default=0.20, help="Portfolio heat cap (e.g., 0.20)")
    parser.add_argument("--heat", type=float, default=HEAT, help="Per-trade heat (e.g., 0.10)")
    parser.add_argument("--persist", action="store_true", help="Write run + portfolio trades/equity to MongoDB")
    args = parser.parse_args()

    blotter, equity = portfolio_backtest(
        symbols=args.symbols,
        start=args.start,
        use_adj_close=args.adj,
        per_trade_heat=args.heat,
        portfolio_heat_cap=args.cap,
        atr_mult=ATR_MULT,
        start_equity=START_EQUITY
    )

    stats = perf_stats(equity)
    print("Symbols:", args.symbols)
    print("Stats:", stats)

    if args.persist:
        db = get_db()
        run_id = new_run_id(prefix="sey_port")
        run_doc = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc),
            "engine": "sey_portfolio_ema",
            "universe": [s.upper() for s in args.symbols],
            "params": {
                "FAST": FAST, "SLOW": SLOW,
                "ATR_LAG": ATR_LAG, "ATR_MULT": ATR_MULT,
                "HEAT": args.heat, "PORTFOLIO_HEAT_CAP": args.cap,
                "START_EQUITY": START_EQUITY,
                "use_adj_close": args.adj,
                "start": args.start,
            },
            "stats": stats,
        }
        write_run(db, run_doc)

        # Persist BUY/SELL rows (include symbol)
        if not blotter.empty:
            rows = blotter.reset_index().to_dict("records")
            write_trades(db, run_id, "PORTFOLIO", rows)

        # Persist portfolio equity series
        write_equity(db, run_id, equity)

        print(f"Persisted portfolio run_id={run_id}")
