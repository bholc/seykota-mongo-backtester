Seykota EMA Portfolio Backtester (MongoDB)

A Python-based portfolio backtesting system inspired by Ed Seykota’s EMA trend-following model, fully integrated with MongoDB for data persistence and in-database analytics.
This project demonstrates real-world use of MongoDB as a time-series data store and analytics engine for quantitative trading systems.

This project has the following Features:
Market Data Pipeline
-Downloads historical OHLCV data (Yahoo! Finance with Stooq fallback)
-Stores daily bars in MongoDB under marketdata.daily_bars
Seykota EMA System
-Implements classic EMA crossover (e.g. 15/150) with ATR-based position sizing
-Risk managed using account heat and optional portfolio heat cap
Portfolio Simulation
-Runs multiple symbols simultaneously under a unified run_id
-Tracks capital allocation, equity, and trade-level P&L
MongoDB Persistence
-All runs, trades, and equity curves stored as documents:
  runs — parameters, metadata, stats
  trades — BUY/SELL fills, realized P&L
  equity — equity curve per day
Aggregation Pipelines (in-DB analytics)
-Computes CAGR, Sharpe, and Max Drawdown inside MongoDB
-Summarizes trades per symbol (win rate, profit factor, average win/loss)
Extensible Framework
-Easy to add stop logic, transaction costs, or experiment tracking (run_group_id)

Project Structure
seykota-mongo-backtester/
├── ema_backtest.py         # Single-asset Seykota EMA backtest
├── portfolio_backtest.py   # Multi-asset portfolio backtest w/ heat cap
├── yahoo_to_mongo.py       # Data ingestion (Yahoo/Stooq → MongoDB)
├── mongo_to_pandas.py      # Query helper: Mongo → Pandas DataFrame
├── results_to_mongo.py     # Write runs, trades, and equity to Mongo
├── report_pipelines.py     # PyMongo aggregation pipelines (CAGR, Sharpe, PF)
├── .env.example            # Example MongoDB connection vars
├── .gitignore              # Ignores venv, logs, and secrets
└── README.md               # You’re reading it!

Installation (bash)
# Clone this repository
git clone https://github.com/<your-username>/seykota-mongo-backtester.git
cd seykota-mongo-backtester
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
# (Optional) Create .env from example
Copy-Item .env.example .env
# Edit DB_NAME / MONGODB_URI if needed

Usage
1. Ingest data:
   python yahoo_to_mongo.py SPY AAPL MSFT
2. Run a single-asset backtest:
   python ema_backtest.py AAPL --persist
3. Run a multi-asset portfolio backtest:
   python portfolio_backtest.py --symbols SPY AAPL MSFT --cap 0.20 --heat 0.10 --persist
4. Generate reports via aggregation pipelines:
   python report_pipelines.py

Outputs include:
Equity CAGR & MaxDD
Sharpe ratio
Trade summary by symbol (win rate, profit factor, avg win/loss)
Equity slice between chosen dates
Run-to-run parameter comparison

