# Stock Market Analyse

Works out of the box on synthetic sample data. When your Shoonya API access is ready, add credentials to `.env` and it automatically switches to live NSE data.

## What's New: Foundational Book Analytics & Educational Hub
The application analyzes stocks and options chains using proven quantitative frameworks extracted from the 21 investment & trading books in your `Books/` library:

1. **Steve Nison (*Japanese Candlestick Charting Techniques, 2nd Edition*)**:
   - Automated recognition of reversal lines: Hammer, Hanging Man, Shooting Star, Inverted Hammer
   - Multi-bar patterns: Bullish & Bearish Engulfing, Morning Star, Evening Star, Piercing Line, Dark Cloud Cover
   - Indecision & Windows: Doji (Dragonfly, Gravestone), Rising & Falling Windows (Gaps)
   - Western Technical Confirmation: Volume expansion (> 20MA) and moving average alignment under Nison's Rule of Confluence

2. **CBSE & National Stock Exchange (*Introduction to Financial Markets - Course 405*)**:
   - Primary and secondary capital market infrastructure
   - NEAT Price-Time priority order matching rules
   - NSDL/CDSL demat clearing and SEBI investor protection tenets

3. **William O'Neil (*How to Make Money in Stocks*)**:
   - CAN SLIM technical scoring (0-100)
   - Volume surge detection vs 50-day average
   - Stage 2 uptrend confirmation (Price > 20 > 50 > 200 SMA)
   - 52-week high breakout proximity & Relative Strength (RS) rating
   - Unconditional 7% stop-loss calculation & 20-25% profit targets

2. **Tom Williams (*Master the Market* - Volume Spread Analysis)**:
   - Institutional footprints: Stopping Volume, Upthrusts, No Demand, No Supply, Absorption Volume, Effort vs Result divergence
   - Automatic classification of bar close position (Upper, Middle, Lower third) and relative spread

3. **Bill Williams (*Trading Chaos*)**:
   - The Alligator Indicator: Jaw (13), Teeth (8), Lips (5)
   - State classification: Sleeping (range-bound) vs Awakening/Hungry (bullish/bearish trend)
   - 5-bar High/Low Fractals for breakout entries and trailing stops
   - Awesome Oscillator (AO) momentum & Market Facilitation Index (MFI 4-color bars)

4. **Perry Kaufman (*Trading Systems and Methods*)**:
   - Kaufman Adaptive Moving Average (KAMA) that speeds up in trends and flattens in chop
   - Efficiency Ratio (ER) measuring directional signal vs market noise
   - Market regime filtering: Strong Trend (ER >= 0.50), Moderate, or Choppy/Noise (ER < 0.25)
   - ATR volatility-adjusted position sizing and stop guidelines

5. **Edward Thorp & Sheen Kassouf (*Beat the Market*)**:
   - Realized Volatility (HV 20-day) vs Implied Volatility (ATM IV) spread
   - Volatility Regime classification: Rich (Seller Edge) vs Cheap (Buyer Edge) vs Fair
   - 1-Sigma Expected Move calculation: `Spot * (IV / 100) * sqrt(T)`
   - 68.2% probability confidence cone & Max Pain gravitational pinning
   - Algorithmic option strategy recommender (Bull Put Spread, Iron Condor, Long Straddle, Covered Call)

6. **Benjamin Graham & Warren Buffett (*The Intelligent Investor* & *The Warren Buffett Way*)**:
   - Graham Number valuation: `sqrt(22.5 * EPS * BVPS)`
   - Margin of Safety calculation
   - Graham Defensive Investor rule: `P/E * P/B <= 22.5`
   - Economic Moat & Compounding Quality Score

7. **Open-Source 90+ Technical Indicators & SuperTrend Engine**:
   - SuperTrend (10, 3) trailing ATR volatility band with bullish green support / bearish red resistance detection
   - 90+ industry standard technical indicators across Trend (Ichimoku Cloud, ADX +DI/-DI, MACD, Moving Averages 20/50/200, Aroon, Parabolic SAR), Momentum (RSI 14, Stochastic %K/%D, Williams %R, CCI, Awesome Oscillator), Volatility (Bollinger Bands %B, Donchian Channels 20, Keltner Channels), and Volume/Flow (VWAP, Chaikin Money Flow, Money Flow Index)
   - Automated Ensemble Consensus Voting with category breakdowns and Master Consensus Score (%)

8. **Educational Knowledge Base & Interactive PDF Reader**:
   - Dedicated tab `📚 Books & Wisdom` featuring structured dossiers for all 25 books (including Sheldon Natenberg, James Dalton, Dr. Van K. Tharp, and Mark Douglas)
   - Interactive PDF page reader directly extracts excerpts from books in `Books/`
   - Interactive multiple-choice knowledge quizzes for every book
   - Educational guidelines on how to apply each book to stocks and options

9. **Real-Time Institutional Market Flow & Macro Volatility Engine**:
   - Live / synthetic **India VIX** tracker and volatility regime classifier (Low, Moderate, High, Extreme Panic)
   - Real-time **FII & DII Institutional Net Flow** tracker (Cash market net buying/selling in ₹ Crores + Index Futures Long/Short ratio)
   - Macro institutional alignment wired directly into the AI Trade Conviction Engine

---

## Quick Start
```bash
pip install -r requirements.txt
python run_app.py              # or double-click run.bat on Windows
```
Runs on SAMPLE data. Test offline suite with:
```bash
python test_core.py
```

### Dashboard Tabs
- **🔔 90%+ Conviction Alerts**: Real-time notification desk scanning for high-conviction trade setups (Conviction ≥ 90%) with separate actionable lists for Stocks and Option Chains, one-click Paper Trading execution, Live Shoonya broker orders, and direct chart inspection.
- **📊 Stock Analysis**: Candlestick charts with interactive overlays (SuperTrend, Donchian Channels, Kaufman KAMA, Alligator, Fractals, SMA), volume bars, and book scorecards (including James Dalton's Volume Profile/POC and Van Tharp Expectancy).
- **⛓️ Option Chain & Strategy Payoff Lab**: PCR, Max Pain, ATM IV, Thorp Volatility Edge (IV vs HV), Sheldon Natenberg Volatility Skew, Expected Move (±1σ), OI Wall charts, and Interactive Strategy Payoff visualizer.
- **🤖 AI Trade & Automation Lab**: Dual-mode, dual-asset engine:
  - **Macro Institutional Barometer**: Real-time India VIX meter and FII/DII net flow.
  - **90+ Open-Source Indicator Consensus Radar**: Visual master consensus bar and category breakdown (Trend, Momentum, Volatility, Flow).
  - **Stock Trade Recommendation**: Concrete Equity BUY/SELL action, conviction score, entry, strict stop-loss, profit targets, and position sizing.
  - **Option Chain Trade Recommendation**: Concrete Option CALL/PUT or Spread action (Bull Call Spread, Bear Put Spread, Long Call, Long Put, Bull Put Spread, Bear Call Spread, Iron Condor, Long Straddle) with exact strikes, expiry, net premium, stop-loss, and profit targets.
  - **Mark Douglas Discipline Guard**: Automated mindset and risk-freeze protection.
  - **Paper Trading Mode**: Beginners can evaluate AI trade signals, execute virtual stock and option orders (₹5,00,000 virtual balance), and let automated triggers enforce stop-losses and targets with ₹0 risk.
  - **Live Broker Execution Mode**: Advanced traders can send live orders to Shoonya API with attached Stop-Loss Limit (SL-LMT) orders and an Emergency Global Kill Switch.
- **🎯 Watchlist & Scanner**: Multi-stock scanner with filter presets (CAN SLIM Breakouts, VSA Reversals, Kaufman Trends, Graham Value).
- **⏳ Backtest**: Realistic, lookahead-free backtesting of book strategies (Kaufman KAMA, Alligator Trend, O'Neil Breakout, SMA Crossover) with transaction costs.
- **📚 Books & Wisdom**: Comprehensive educational encyclopedia with core philosophies, formulas, stock/option instructions, risk tenets, interactive quizzes, and interactive PDF excerpt reader across 25 masterworks.
- **🔌 Connection**: Shoonya API credential checker and live connection test.

---

## Files
- `institutional_data.py`: Real-time India VIX volatility regime and FII/DII institutional net flow barometer.
- `open_source_indicators.py`: 90+ open-source indicators ensemble engine and vectorised SuperTrend (10, 3).
- `ai_trade_engine.py`: AI recommendation and conviction scoring engine synthesizing open-source indicators, macro flow, and book analytics for stocks and option chains.
- `paper_trading.py`: SQLite-backed virtual paper trading engine with automated triggers for stocks and options.
- `live_execution.py`: Shoonya live broker order placement (Equity & NFO Options) and emergency kill switch.
- `options_payoff.py`: Interactive P&L payoff graph visualizer for multi-leg option strategies.
- `book_analytics.py`: Quantitative formulas and algorithms from the books (including Dalton Volume Profile, Natenberg Skew, Van Tharp Expectancy, Douglas Discipline).
- `book_knowledge.py`: Educational dossiers, formulas, guidance, interactive quizzes, and PDF excerpt extractor for 25 books.
- `indicators.py`: Technical indicators (SMA, EMA, RSI, MACD, Bollinger, KAMA, Alligator).
- `options_analytics.py`: Greeks, IV bisection, Max Pain, PCR, Thorp Volatility Edge, Expected Move.
- `screener.py`: Multi-symbol scanner with book strategy presets.
- `backtest.py`: Lookahead-free backtester with book-based strategy models.
- `dashboard.py`: Streamlit frontend.
- `providers.py`: Data provider interface (Shoonya live + synthetic sample).
- `storage.py`: SQLite caching engine.
- `test_core.py`: Automated test suite (29 unit tests).
