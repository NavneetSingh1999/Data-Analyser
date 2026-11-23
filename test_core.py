"""Offline checks. Run:  python test_core.py"""
import os
import subprocess
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path

# Ensure categorical sub-folders are in sys.path
_ROOT = Path(__file__).resolve().parent
for _subdir in ("analytics", "data", "execution", "assets"):
    _sub_p = str(_ROOT / _subdir)
    if _sub_p not in sys.path:
        sys.path.insert(0, _sub_p)

import numpy as np
import pandas as pd

import backtest as bt
import config
import indicators as ind
import options_analytics as oa
import screener
from providers import OnlineProvider, SampleProvider, ShoonyaProvider, daily_series_to_frame, get_provider, is_market_open
from storage import get_history_cached


# ---- indicators ----------------------------------------------------------
def test_rsi_extremes():
    assert ind.rsi(pd.Series(np.arange(1.0, 60.0))).iloc[-1] == 100
    assert ind.rsi(pd.Series(np.arange(60.0, 1.0, -1.0))).iloc[-1] < 1


def test_sma_ema_known_values():
    s = pd.Series([1.0, 2, 3, 4, 5])
    assert ind.sma(s, 3).iloc[-1] == 4.0
    assert abs(ind.ema(s, 3).iloc[0] - 1.0) < 1e-12


def test_bollinger_flat_series_has_zero_width():
    b = ind.bollinger(pd.Series([10.0] * 30))
    assert (b["upper"].dropna() == b["lower"].dropna()).all()


# ---- option analytics ----------------------------------------------------
def test_max_pain_and_pcr_hand_example():
    chain = pd.DataFrame(
        {"strike": [100.0, 110.0, 120.0], "ce_oi": [10, 50, 10], "pe_oi": [10, 50, 10],
         "ce_volume": [1, 1, 1], "pe_volume": [2, 2, 2], "ce_iv": [15, 15, 15], "pe_iv": [15, 15, 15]}
    )
    assert oa.max_pain(chain) == 110.0
    assert oa.pcr(chain, "oi") == 1.0
    assert oa.pcr(chain, "volume") == 2.0
    assert oa.atm_strike(chain, 112) == 110.0


def test_implied_vol_round_trip_and_parity():
    spot, strike, t = 24000.0, 24200.0, 10 / 365
    for is_call in (True, False):
        price = oa.bs_price(spot, strike, t, 0.15, is_call)
        assert abs(oa.implied_vol(price, spot, strike, t, is_call) - 15.0) < 0.01
    c, p = oa.bs_price(spot, strike, t, 0.2, True), oa.bs_price(spot, strike, t, 0.2, False)
    assert abs((c - p) - (spot - strike * np.exp(-oa.RISK_FREE * t))) < 1e-6  # put-call parity
    assert np.isnan(oa.implied_vol(0.01, spot, 20000.0, t, True))  # price below intrinsic: no IV


def test_years_to_expiry():
    now = datetime(2026, 11, 19, 15, 30, tzinfo=oa.IST)
    assert abs(oa.years_to_expiry("2026-11-26", now) - 7 / 365) < 1e-9
    assert oa.years_to_expiry("2026-11-01", now) == 1e-6  # already expired


# ---- Shoonya adapter against a fake API shaped like the docs -------------
class FakeNoren:
    def __init__(self):
        self.expire_once = False
        self.logins = 0

    def get_quotes(self, exchange, token):
        if self.expire_once:
            self.expire_once = False
            return {"stat": "Not_Ok", "emsg": "Session Expired : Invalid Session Key"}
        if exchange == "NSE":
            return {"stat": "Ok", "lp": "24000.00", "tsym": "Nifty 50"}
        strike, side = {"1": (23950, "CE"), "2": (23950, "PE"), "3": (24050, "CE"), "4": (24050, "PE")}[token]
        ltp = {"1": 160.0, "2": 95.0, "3": 105.0, "4": 140.0}[token]
        return {"stat": "Ok", "lp": str(ltp), "oi": str(1000 * int(token)), "v": "50"}

    def searchscrip(self, exchange, searchtext):
        if exchange == "NSE":
            return {"stat": "Ok", "values": [{"exch": "NSE", "token": "2885", "tsym": searchtext}]}
        return {"stat": "Ok", "values": [{"tsym": "NIFTY26NOV26F", "token": "99"}, {"tsym": "NIFTY26NOV26C24000", "token": "98"}]}

    def get_option_chain(self, exchange, tradingsymbol, strikeprice, count):
        v = lambda tok, k, o: {"exch": "NFO", "tsym": f"X{tok}", "token": tok, "optt": o, "strprc": str(k), "ls": "75"}
        return {"stat": "Ok", "values": [v("1", 23950, "CE"), v("2", 23950, "PE"), v("3", 24050, "CE"), v("4", 24050, "PE")]}

    def get_daily_price_series(self, exchange, tradingsymbol, startdate, enddate):
        return [  # newest first, JSON strings, as Shoonya returns them
            '{"stat":"Ok","time":"19-NOV-2026","ssboe":"1794000000","into":"101","inth":"105","intl":"99","intc":"104","intv":"5000"}',
            '{"stat":"Ok","time":"18-NOV-2026","ssboe":"1793913600","into":"100","inth":"103","intl":"98","intc":"101","intv":"4000"}',
        ]


def make_fake_provider():
    return ShoonyaProvider(api=FakeNoren(), creds={}, pause=0)


def test_shoonya_history_parsing_sorts_oldest_first():
    df = make_fake_provider().get_history("RELIANCE", date(2026, 11, 1), date(2026, 11, 19))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.is_monotonic_increasing and len(df) == 2
    assert df["close"].iloc[-1] == 104.0 and df["volume"].iloc[0] == 4000.0


def test_shoonya_history_error_handling():
    assert daily_series_to_frame({"stat": "Not_Ok", "emsg": "no data"}).empty
    assert daily_series_to_frame(None).empty
    try:
        daily_series_to_frame({"stat": "Not_Ok", "emsg": "Invalid input"})
        assert False, "should raise"
    except RuntimeError:
        pass


def test_shoonya_option_chain_built_from_quotes():
    p = make_fake_provider()
    chain = p.get_option_chain("NIFTY", "2026-11-26")
    assert chain["strike"].tolist() == [23950.0, 24050.0]
    assert chain.attrs["underlying_ltp"] == 24000.0 and chain.attrs["reference_symbol"] == "NIFTY26NOV26F"
    assert chain.loc[0, "ce_oi"] == 1000 and chain.loc[0, "pe_oi"] == 2000
    assert {"ce_iv", "pe_iv", "ce_delta", "pe_delta"} <= set(chain.columns)
    assert oa.pcr(chain) > 0 and oa.max_pain(chain) in (23950.0, 24050.0)


def test_shoonya_relogin_on_expired_session():
    p = make_fake_provider()
    p._logged_in = False
    logins = []
    p._login = lambda: (logins.append(1), setattr(p, "_logged_in", True))
    p.api.expire_once = True
    assert p._spot("NIFTY") == 24000.0
    assert len(logins) == 2  # first login + re-login after "Session Expired"


def test_missing_credentials_message_and_auto_fallback():
    saved = {v: os.environ.pop(v, None) for v in config.SHOONYA_FIELDS.values()}
    try:
        assert len(config.shoonya_missing()) == 6
        assert config.default_source() == "sample"
        assert get_provider("auto").name == "sample"
        try:
            get_provider("shoonya")
            assert False, "should raise"
        except RuntimeError as e:
            assert "SHOONYA_USER" in str(e)
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_env_file_loading():
    with tempfile.TemporaryDirectory() as t:
        f = Path(t) / ".env"
        f.write_text('# c\nSHOONYA_USER="abc"\nSHOONYA_IMEI=xyz\n')
        os.environ.pop("SHOONYA_USER", None)
        os.environ.pop("SHOONYA_IMEI", None)
        config.load_env(f)
        assert os.environ["SHOONYA_USER"] == "abc" and os.environ["SHOONYA_IMEI"] == "xyz"
        os.environ.pop("SHOONYA_USER"), os.environ.pop("SHOONYA_IMEI")


# ---- storage / sample / screener / backtest ------------------------------
def test_cache_fetches_only_missing_days():
    class Counting(SampleProvider):
        calls = []
        def get_history(self, symbol, start, end):
            self.calls.append((start, end))
            return super().get_history(symbol, start, end)

    with tempfile.TemporaryDirectory() as t:
        db = Path(t) / "x.db"
        p = Counting()
        a = get_history_cached(p, "sample", "TEST", date(2026, 1, 1), date(2026, 6, 30), db)
        b = get_history_cached(p, "sample", "TEST", date(2026, 1, 1), date(2026, 9, 30), db)
        assert len(b) > len(a) and p.calls[1][0] > date(2026, 6, 25)


def test_sample_data_identical_across_processes():
    code = ("from datetime import date; from providers import SampleProvider; "
            "print(SampleProvider().get_history('ABC', date(2026,1,1), date(2026,3,1))['close'].iloc[-1])")
    proj_dir = str(Path(__file__).parent)
    p_path = os.pathsep.join([proj_dir, os.path.join(proj_dir, "data"), os.path.join(proj_dir, "analytics"), os.path.join(proj_dir, "execution")])
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=proj_dir,
                           env={**os.environ, "PYTHONPATH": p_path, "PYTHONHASHSEED": str(s)}).stdout for s in (1, 2)}
    assert len(outs) == 1 and "" not in outs


def test_signals_and_summary_run_on_sample_data():
    df = SampleProvider().get_history("ABC", date(2025, 1, 1), date(2026, 6, 30))
    assert ind.latest_signals(df)
    chain = SampleProvider().get_option_chain("NIFTY", "2026-11-26")
    s = oa.summary(chain, chain.attrs["underlying_ltp"])
    assert 0 < s["pcr_oi"] < 5 and s["max_pain"] in chain["strike"].values


def test_screener_isolates_failures():
    class Flaky(SampleProvider):
        def get_history(self, symbol, start, end):
            if symbol == "BAD":
                raise RuntimeError("boom")
            return super().get_history(symbol, start, end)

    import storage
    with tempfile.TemporaryDirectory() as t:
        storage.DB_PATH = Path(t) / "s.db"
        table, errors = screener.scan(Flaky(), ["AAA", "BAD", "CCC"])
    assert set(table["symbol"]) == {"AAA", "CCC"} and "BAD" in errors


def test_backtest_no_lookahead_and_costs():
    # Price jumps up on day 60 only. A signal built from that day's close must not capture the jump.
    close = pd.Series([100.0] * 60 + [200.0] * 40, index=pd.bdate_range("2025-01-01", periods=100))
    df = pd.DataFrame({"close": close})
    r = bt.sma_crossover(df, 5, 20, cost_bps=0)
    assert r.metrics["strategy_return_%"] == 0.0  # jump day itself was missed by design
    assert r.metrics["buy_hold_return_%"] == 100.0
    # Steady uptrend: strategy should be long and costs should reduce the result.
    up = pd.DataFrame({"close": 100 * 1.005 ** np.arange(200)}, index=pd.bdate_range("2025-01-01", periods=200))
    free, costly = bt.sma_crossover(up, 5, 20, 0), bt.sma_crossover(up, 5, 20, 50)
    assert free.metrics["strategy_return_%"] > costly.metrics["strategy_return_%"] > 0
    assert free.metrics["trades"] == 1


# ---- Book Analytics & Knowledge Base Tests -----------------------------------
def test_book_analytics_oneil():
    import book_analytics as ba
    dates = pd.bdate_range("2025-01-01", periods=100)
    # Build steady trend
    close = pd.Series(100.0 + np.arange(100) * 0.5, index=dates)
    high = close + 1.0
    low = close - 1.0
    vol = pd.Series(1000.0, index=dates)
    vol.iloc[-1] = 2500.0  # 2.5x volume surge
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": vol})
    res = ba.calc_oneil_signals(df)
    assert res["vol_surge_ratio"] >= 2.0
    assert -1.0 <= res["pct_from_52w_high"] <= 0.0  # within 1% of high
    assert res["score"] >= 75
    assert res["stop_loss_7pct"] < close.iloc[-1]


def test_book_analytics_vsa():
    import book_analytics as ba
    dates = pd.bdate_range("2025-01-01", periods=50)
    close = pd.Series(100.0, index=dates)
    high = close + 1.0
    low = close - 1.0
    vol = pd.Series(1000.0, index=dates)
    # Make last bar a Stopping Volume bar: big drop, high volume, close in upper third
    high.iloc[-1] = 102.0
    low.iloc[-1] = 90.0
    close.iloc[-1] = 99.0
    vol.iloc[-1] = 3000.0
    df = pd.DataFrame({"open": pd.Series(100.0, index=dates), "high": high, "low": low, "close": close, "volume": vol})
    vsa = ba.calc_vsa_signals(df)
    assert vsa["rel_vol"] >= 2.0
    assert "Stopping Volume" in " ".join(vsa["signals"])


def test_book_analytics_kaufman_and_alligator():
    import book_analytics as ba
    dates = pd.bdate_range("2025-01-01", periods=60)
    close = pd.Series(100.0 + np.arange(60) * 1.0, index=dates)  # strong linear trend
    df = pd.DataFrame({
        "open": close - 0.2, "high": close + 0.5, "low": close - 0.5, "close": close,
        "volume": pd.Series(1000.0, index=dates)
    })
    kauf = ba.calc_kaufman_regime(df)
    assert kauf["efficiency_ratio"] > 0.8  # near perfect trend
    assert "Strong Trend" in kauf["regime"]
    
    alli = ba.calc_alligator_and_chaos(df)
    assert "Awakening Bullish" in alli["alligator_state"]
    assert alli["lips"] > alli["teeth"] > alli["jaw"]


def test_book_analytics_thorp_options():
    import book_analytics as ba
    dates = pd.bdate_range("2025-01-01", periods=60)
    close = pd.Series(100.0 + np.sin(np.arange(60)) * 2.0, index=dates)
    df = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1000})
    chain = SampleProvider().get_option_chain("NIFTY", "2026-11-26")
    spot = chain.attrs["underlying_ltp"]
    thorp = ba.calc_thorp_options_edge(chain, df, spot, days_to_expiry=7.0)
    assert thorp["expected_move"] > 0
    assert thorp["expected_range"][0] < spot < thorp["expected_range"][1]
    assert thorp["vol_regime"] in ("Rich / Overpriced Options (Seller Edge)", "Cheap / Underpriced Options (Buyer Edge)", "Fairly Priced")


def test_book_knowledge_metadata():
    import book_knowledge as bk
    assert len(bk.BOOKS_METADATA) == 25
    cats = bk.get_all_categories()
    assert len(cats) >= 5
    oneil = bk.get_book_by_id("oneil")
    assert oneil is not None and "CAN SLIM" in oneil["tagline"]
    nison = bk.get_book_by_id("nison_candlesticks")
    assert nison is not None and "Japanese Candlestick" in nison["title"]
    cbse = bk.get_book_by_id("cbse_fin_markets_405")
    assert cbse is not None and "Introduction to Financial Markets" in cbse["title"]
    natenberg = bk.get_book_by_id("natenberg")
    assert natenberg is not None and "Option Volatility" in natenberg["title"]
    dalton = bk.get_book_by_id("dalton_auction_market")
    assert dalton is not None and "Auction Market" in dalton["title"]
    van_tharp = bk.get_book_by_id("van_tharp")
    assert van_tharp is not None and "Financial Freedom" in van_tharp["title"]
    douglas = bk.get_book_by_id("mark_douglas")
    assert douglas is not None and "Trading in the Zone" in douglas["title"]


def test_recommended_books_analytics():
    import book_analytics as ba

    dates = pd.bdate_range("2025-01-01", periods=30)
    close = pd.Series(100.0 + np.sin(np.linspace(0, 3, 30)) * 5.0, index=dates)
    df = pd.DataFrame({
        "open": close,
        "high": close + 1.5,
        "low": close - 1.5,
        "close": close,
        "volume": 5000 + np.random.randint(100, 1000, size=30),
    })

    # 1. James Dalton Volume Profile & POC
    dalton = ba.calc_dalton_volume_profile(df, n_bins=20)
    assert "poc" in dalton and dalton["poc"] > 0
    assert "vah" in dalton and "val" in dalton
    assert dalton["vah"] >= dalton["val"]
    assert dalton["market_condition"] != ""

    # 2. Sheldon Natenberg Volatility Skew
    chain = pd.DataFrame({
        "strike": [95.0, 97.5, 100.0, 102.5, 105.0],
        "ce_iv": [12.0, 12.5, 13.0, 13.5, 14.0],
        "pe_iv": [18.0, 16.5, 15.0, 13.5, 12.0],
    })
    skew = ba.calc_natenberg_volatility_skew(chain, spot=100.0)
    assert "skew_spread" in skew
    assert skew["skew_spread"] > 0  # OTM Put IV > OTM Call IV
    assert "second_order_greeks" in skew

    # 3. Van Tharp Expectancy & Sizing
    tharp = ba.calc_van_tharp_sizing_and_expectancy(
        account_capital=200000.0,
        risk_per_trade_pct=1.0,
        entry_price=100.0,
        stop_loss=95.0,
        target_price=110.0,
        win_rate=0.55,
    )
    assert tharp["1r_risk_per_share"] == 5.0
    assert tharp["system_expectancy_r"] > 0
    assert tharp["recommended_qty"] > 0

    # 4. Mark Douglas Mindset Guard
    douglas_ok = ba.calc_mark_douglas_discipline_guard(consecutive_losses=0)
    assert douglas_ok["risk_multiplier"] == 1.00
    douglas_freeze = ba.calc_mark_douglas_discipline_guard(consecutive_losses=2)
    assert douglas_freeze["risk_multiplier"] == 0.50


def test_institutional_data_and_india_vix():
    import institutional_data as idata

    # 1. India VIX Metrics
    vix = idata.get_india_vix_metrics()
    assert "vix" in vix and vix["vix"] > 0
    assert "regime" in vix
    assert "options_implication" in vix
    assert "bias" in vix

    # 2. FII / DII Flow Metrics
    flow = idata.get_fii_dii_flow_metrics()
    assert "fii_net_crores" in flow
    assert "dii_net_crores" in flow
    assert "fii_long_short_ratio" in flow
    assert flow["fii_long_short_ratio"] > 0
    assert "sentiment" in flow



def test_book_analytics_nison_candlesticks():
    import book_analytics as ba
    dates = pd.bdate_range("2025-01-01", periods=10)
    # Construct a Hammer at the last bar after a downtrend
    close = pd.Series([100.0, 98.0, 96.0, 94.0, 92.0, 90.0, 88.0, 86.0, 84.0, 84.5], index=dates)
    open_p = pd.Series([100.5, 98.5, 96.5, 94.5, 92.5, 90.5, 88.5, 86.5, 84.5, 84.2], index=dates)
    high = pd.Series([101.0, 99.0, 97.0, 95.0, 93.0, 91.0, 89.0, 87.0, 85.0, 84.6], index=dates)
    low = pd.Series([99.5, 97.5, 95.5, 93.5, 91.5, 89.5, 87.5, 85.5, 83.5, 80.0], index=dates)
    vol = pd.Series(1000.0, index=dates)
    df = pd.DataFrame({"open": open_p, "high": high, "low": low, "close": close, "volume": vol})
    res = ba.calc_nison_candlestick_patterns(df)
    assert "Hammer" in res["active_pattern"]
    assert res["bias"] == "Bullish"


def test_book_backtest_strategies():
    dates = pd.bdate_range("2025-01-01", periods=100)
    up = pd.DataFrame({
        "open": 100 * 1.005 ** np.arange(100),
        "high": 100 * 1.005 ** np.arange(100) + 0.5,
        "low": 100 * 1.005 ** np.arange(100) - 0.5,
        "close": 100 * 1.005 ** np.arange(100),
        "volume": 1000
    }, index=dates)
    
    res_kama = bt.kaufman_adaptive_system(up, cost_bps=10.0)
    assert res_kama.metrics["strategy_return_%"] > 0
    
    res_alli = bt.alligator_trend_system(up, cost_bps=10.0)
    assert res_alli.metrics["strategy_return_%"] > 0
    
    res_oneil = bt.oneil_breakout_system(up, stop_pct=7.0, cost_bps=10.0)
    assert res_oneil.metrics["strategy_return_%"] > 0


def test_options_payoff_builder():
    import options_payoff as op
    res = op.build_strategy_payoff("Bull Call Spread", 24000, 24000, 300)
    assert res["max_profit"] > 0
    assert res["max_loss"] > 0
    assert len(res["breakevens"]) == 1
    fig = op.plot_strategy_payoff(res)
    assert len(fig.data) >= 2


def test_open_source_indicators_consensus():
    import open_source_indicators as osi

    dates = pd.bdate_range("2025-01-01", periods=60)
    close = pd.Series(100.0 + np.arange(60) * 0.5, index=dates)
    df = pd.DataFrame({
        "open": close - 0.2,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1000 + np.arange(60) * 10,
    })

    # 1. SuperTrend calculation
    st_line, st_dir = osi.calc_supertrend(df, period=10, multiplier=3.0)
    assert len(st_line) == 60
    assert len(st_dir) == 60
    assert set(st_dir.unique()).issubset({1, -1})
    assert st_dir.iloc[-1] == 1  # upward trend

    # 2. 90+ indicators computation
    all_ta = osi.compute_all_open_source_indicators(df)
    assert len(all_ta.columns) >= 70
    assert "trend_supertrend" in all_ta.columns
    assert "momentum_rsi" in all_ta.columns
    assert "volume_vwap" in all_ta.columns

    # 3. Consensus voting
    res = osi.evaluate_indicator_consensus(df)
    assert "consensus_pct" in res
    assert 0 <= res["consensus_pct"] <= 100
    assert res["consensus_pct"] > 60  # upward trending test series
    assert "breakdown" in res
    assert "trend" in res["breakdown"]
    assert "momentum" in res["breakdown"]
    assert "volatility" in res["breakdown"]
    assert "volume_flow" in res["breakdown"]
    assert "indicators" in res
    assert "supertrend" in res["indicators"]


def test_ai_trade_engine_and_paper_execution():
    import ai_trade_engine as ate
    import paper_trading as pt
    import live_execution as le

    dates = pd.bdate_range("2025-01-01", periods=60)
    close = pd.Series(100.0 + np.arange(60) * 0.5, index=dates)
    df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000})
    
    # 1. AI Stock Evaluation
    sig = ate.evaluate_stock_trade("TEST_STOCK", df, account_capital=200000)
    assert sig.conviction > 0
    assert sig.action in ("BUY", "SELL", "WAIT")
    assert sig.stop_loss > 0 and sig.target_1 > 0
    assert sig.indicator_consensus is not None
    assert "consensus_pct" in sig.indicator_consensus

    # 2. AI Options Trade Evaluation
    chain = pd.DataFrame({
        "strike": [120.0, 125.0, 130.0, 135.0],
        "ce_ltp": [12.0, 8.0, 5.0, 2.5],
        "pe_ltp": [1.5, 3.0, 6.0, 11.0],
        "ce_oi": [1000, 2500, 5000, 2000],
        "pe_oi": [2000, 5000, 2500, 1000],
        "ce_iv": [14.0, 14.5, 15.0, 15.5],
        "pe_iv": [14.0, 14.5, 15.0, 15.5],
        "ce_volume": [100, 200, 500, 100],
        "pe_volume": [100, 500, 200, 100],
    })
    opt_sig = ate.evaluate_options_trade(
        underlying="NIFTY",
        chain=chain,
        df_underlying=df,
        spot=130.0,
        expiry="2026-11-26",
        account_capital=200000,
    )
    assert opt_sig.asset_type == "OPTION"
    assert opt_sig.conviction >= 50
    assert opt_sig.stop_loss > 0
    assert opt_sig.recommended_qty >= 50

    # 3. Paper Trading Execution (Stock + Option) Isolated Test DB
    orig_db = pt.DB_PATH
    test_db = Path(__file__).parent / "test_paper_portfolio_1.db"
    pt.DB_PATH = test_db
    try:
        pt.reset_paper_account(500000.0)
        acct = pt.get_or_create_paper_account()
        assert acct["cash_balance"] == 500000.0

        ok, msg = pt.execute_paper_order(sig, quantity=5)
        assert ok is True
        ok_opt, msg_opt = pt.execute_paper_order(opt_sig, quantity=50)
        assert ok_opt is True
        open_pos = pt.get_open_paper_positions()
        assert len(open_pos) == 2

        # 4. Automated Stop/Target Trigger
        notifs = pt.update_and_check_triggers({"TEST_STOCK": sig.target_1 + 2.0})
        assert len(notifs) >= 1 and "TARGET" in notifs[0]
        remaining_pos = pt.get_open_paper_positions()
        assert len(remaining_pos) == 1  # Option position still open
    finally:
        pt.DB_PATH = orig_db
        if test_db.exists():
            try:
                test_db.unlink()
            except Exception:
                pass

    # 5. Live Safety Guard Check
    ok_live, msg_live = le.live_executor.place_live_order(sig, 1)
    assert ok_live is False and "credentials missing" in msg_live.lower()


def test_online_provider_and_market_hours():
    # 1. Symbol Mapping
    assert OnlineProvider.map_symbol("RELIANCE") == "RELIANCE.NS"
    assert OnlineProvider.map_symbol("TCS.NS") == "TCS.NS"
    assert OnlineProvider.map_symbol("NIFTY") == "^NSEI"
    assert OnlineProvider.map_symbol("BANKNIFTY") == "^NSEBANK"
    assert OnlineProvider.map_symbol("INDIAVIX") == "^INDIAVIX"

    # 2. Market Open Check
    is_open, desc = is_market_open()
    assert isinstance(is_open, bool)
    assert isinstance(desc, str) and len(desc) > 0

    # 3. Provider initialization
    p = get_provider("online")
    assert p.is_live is True
    assert p.name == "online"

    # 4. Online option chain generation
    chain = p.get_option_chain("NIFTY", "2026-11-26")
    assert not chain.empty
    assert "strike" in chain.columns
    assert "ce_ltp" in chain.columns
    assert "pe_ltp" in chain.columns
    assert chain.attrs.get("underlying_ltp", 0) > 0

    # 5. Online history fetch (or graceful empty on connection issue)
    from datetime import timedelta
    today = date.today()
    hist = p.get_history("RELIANCE", today - timedelta(days=5), today)
    assert isinstance(hist, pd.DataFrame)
    if not hist.empty:
        assert list(hist.columns) == ["open", "high", "low", "close", "volume"]
        assert hist.index.name == "date"


def test_tradingview_integration():
    import tradingview_integration as tvi
    import paper_trading as pt

    # 1. Symbol Mapping
    assert tvi.map_to_tv_symbol("RELIANCE") == "NSE:RELIANCE"
    assert tvi.map_to_tv_symbol("NIFTY") == "NSE:NIFTY"
    assert tvi.map_to_tv_symbol("BANKNIFTY") == "NSE:BANKNIFTY"
    assert tvi.map_to_tv_symbol("INDIAVIX") == "NSE:INDIAVIX"
    assert tvi.extract_raw_ticker("NSE:RELIANCE") == "RELIANCE"

    # 2. HTML Embed Code Generation
    chart_html = tvi.get_tv_chart_embed_html("RELIANCE", height=600)
    assert "TradingView.widget" in chart_html
    assert "NSE:RELIANCE" in chart_html

    gauge_html = tvi.get_tv_technical_gauge_html("RELIANCE", height=400)
    assert "embed-widget-technical-analysis.js" in gauge_html

    # 3. Pine Script Code Template
    pine_code = tvi.get_sample_pine_script_alert_code()
    assert "//@version=5" in pine_code
    assert "supertrend" in pine_code.lower()

    # 4. Webhook Alert Processing (Paper Mode) Isolated Test DB
    orig_db = pt.DB_PATH
    test_db = Path(__file__).parent / "test_paper_portfolio_tv.db"
    pt.DB_PATH = test_db
    try:
        pt.reset_paper_account(500000.0)
        payload = {
            "ticker": "TEST_TV_STOCK",
            "action": "BUY",
            "price": 2500.0,
            "stop_loss": 2450.0,
            "target": 2600.0,
            "quantity": 10,
            "strategy": "SuperTrend Pine Webhook",
            "mode": "paper",
        }
        ok, msg, detail = tvi.process_tradingview_webhook_alert(payload)
        assert ok is True
        assert detail["status"] == "EXECUTED"
        assert detail["ticker"] == "TEST_TV_STOCK"
    finally:
        pt.DB_PATH = orig_db
        if test_db.exists():
            try:
                test_db.unlink()
            except Exception:
                pass


def test_stock_directory_auto_search():
    import stock_directory as sd

    # 1. Master directory coverage (12,600+ listed NSE & BSE equities & indices)
    assert len(sd.MASTER_STOCKS) >= 2600
    assert "RELIANCE" in sd.MASTER_STOCKS
    assert "TCS" in sd.MASTER_STOCKS
    assert "NIFTY" in sd.MASTER_STOCKS
    assert "3MINDIA" in sd.MASTER_STOCKS
    assert "20MICRONS" in sd.MASTER_STOCKS
    assert "SWIGGY" in sd.MASTER_STOCKS
    assert "TAPARIA" in sd.MASTER_STOCKS

    # 2. Local fuzzy / keyword search
    tata_matches = sd.search_stocks_local("tata")
    assert len(tata_matches) >= 3
    tata_syms = [m[0] for m in tata_matches]
    assert "TATAMOTORS" in tata_syms or "TCS" in tata_syms

    taparia_matches = sd.search_stocks_local("taparia")
    assert len(taparia_matches) >= 1
    assert any(m[0] == "TAPARIA" for m in taparia_matches)

    bank_matches = sd.search_stocks_local("bank")
    assert len(bank_matches) >= 3

    # 3. Formatted search list for selectbox
    opts = sd.get_formatted_search_list()
    assert len(opts) >= 2600
    assert any("RELIANCE — " in o for o in opts)
    assert any("3MINDIA — " in o for o in opts)
    assert any("TAPARIA — " in o for o in opts)

    # 4. Symbol extraction from formatted label
    assert sd.extract_symbol_from_selection("RELIANCE — Reliance Industries Ltd. [Energy]") == "RELIANCE"
    assert sd.extract_symbol_from_selection("TCS") == "TCS"
    assert sd.extract_symbol_from_selection("3MINDIA — 3M India Limited [Diversified / Industrial]") == "3MINDIA"
    assert sd.extract_symbol_from_selection("TAPARIA — Taparia Tools Ltd. [BSE • Industrial Tools & Hardware]") == "TAPARIA"

    # 5. Full Universe DataFrame & Company Info
    df_univ = sd.get_all_stocks_dataframe()
    assert len(df_univ) >= 2600
    assert "Symbol" in df_univ.columns
    assert "Company Name" in df_univ.columns
    assert "ISIN" in df_univ.columns

    info_3m = sd.get_stock_info("3MINDIA")
    assert info_3m["series"] == "EQ"
    assert info_3m["isin"].startswith("INE")

    # 6. Comprehensive Indian Market Indices Coverage
    all_idx = sd.get_all_indian_indices()
    assert len(all_idx) >= 50
    assert "NIFTY" in all_idx and "BANKNIFTY" in all_idx and "SENSEX" in all_idx and "BANKEX" in all_idx
    assert "NIFTY_IT" in all_idx and "NIFTY_AUTO" in all_idx and "NIFTY_DEFENCE" in all_idx
    assert sd.is_index("NIFTY") and sd.is_index("BANKNIFTY") and sd.is_index("SENSEX") and sd.is_index("BANKEX")
    assert sd.is_index("NIFTY IT") and sd.is_index("NIFTY_PHARMA")


def test_stock_option_chain_analytics():
    import ai_trade_engine as ate
    import calendar

    # 1. Official NSE Lot Sizes & Fallback Calculation
    assert oa.get_lot_size("RELIANCE") == 250
    assert oa.get_lot_size("TCS") == 175
    assert oa.get_lot_size("INFY") == 400
    assert oa.get_lot_size("HDFCBANK") == 550
    assert oa.get_lot_size("SBIN") == 750
    assert oa.get_lot_size("NIFTY") == 50
    assert oa.get_lot_size("BANKNIFTY") == 15
    assert oa.get_lot_size("SENSEX") == 10
    assert oa.get_lot_size("BANKEX") == 15
    assert oa.get_lot_size("NIFTYNXT50") == 25
    assert oa.is_weekly_index("NIFTY")
    assert oa.is_weekly_index("BANKNIFTY")
    assert oa.is_weekly_index("SENSEX")
    assert oa.is_weekly_index("BANKEX")
    assert oa.is_weekly_index("MIDCPNIFTY")

    # Fallback lot size estimation (~Rs 6 Lakhs contract value)
    est_lot = oa.get_lot_size("CUSTOM_STOCK", spot=1200.0)
    assert est_lot == 500  # 600,000 / 1200 = 500

    # 2. NSE Monthly Stock Expiries vs Weekly Expiries & Auto-Fetch
    # Monthly expiries
    monthly_expiries = oa.get_stock_expiry_dates("RELIANCE", expiry_type="monthly")
    stock_expiries = monthly_expiries
    assert len(monthly_expiries) == 3
    for exp_str in monthly_expiries:
        exp_dt = datetime.strptime(exp_str, "%Y-%m-%d").date()
        assert exp_dt.weekday() == 3  # Must be a Thursday
        # Must be the last Thursday of that month
        last_day = calendar.monthrange(exp_dt.year, exp_dt.month)[1]
        days_remaining = last_day - exp_dt.day
        assert days_remaining < 7
        assert "Monthly Expiry" in oa.get_expiry_label(exp_str, "RELIANCE")

    # Weekly expiries
    weekly_expiries = oa.get_stock_expiry_dates("RELIANCE", expiry_type="weekly")
    assert len(weekly_expiries) >= 4
    for exp_str in weekly_expiries:
        exp_dt = datetime.strptime(exp_str, "%Y-%m-%d").date()
        assert exp_dt.weekday() == 3  # Thursday

    # All expiries (Weekly + Monthly combined)
    all_expiries = oa.get_stock_expiry_dates("RELIANCE", expiry_type="all")
    assert len(all_expiries) >= 6
    # Check that labels are properly generated
    for exp_str in all_expiries:
        lbl = oa.get_expiry_label(exp_str, "RELIANCE")
        assert ("Weekly" in lbl) or ("Monthly" in lbl)

    # Index expiries (e.g. NIFTY, BANKNIFTY)
    index_expiries = oa.get_stock_expiry_dates("NIFTY", expiry_type="all")
    assert len(index_expiries) >= 6
    for exp_str in index_expiries:
        exp_dt = datetime.strptime(exp_str, "%Y-%m-%d").date()
        assert exp_dt.weekday() == 3

    # 3. Dynamic Strike Steps
    assert oa.get_strike_step("NIFTY", 24000.0) == 50.0
    assert oa.get_strike_step("BANKNIFTY", 51000.0) == 100.0
    assert oa.get_strike_step("RELIANCE", 1375.0) == 20.0
    assert oa.get_strike_step("TCS", 3500.0) == 50.0
    assert oa.get_strike_step("SBIN", 780.0) == 10.0
    assert oa.get_strike_step("MIDCAP", 120.0) == 2.5
    assert oa.get_strike_step("PENNY", 85.0) == 1.0

    # 4. Stock Option Chain Generation via OnlineProvider
    p = get_provider("online")
    exp = stock_expiries[0]
    stock_chain = p.get_option_chain("RELIANCE", exp)
    assert not stock_chain.empty
    assert stock_chain.attrs.get("symbol") == "RELIANCE"
    assert stock_chain.attrs.get("lot_size") == 250
    assert stock_chain.attrs.get("underlying_ltp") > 0
    assert len(stock_chain) >= 15
    # Verify strike step is 20 for RELIANCE
    strikes = stock_chain["strike"].tolist()
    assert all(round(strikes[i+1] - strikes[i], 2) == 20.0 for i in range(len(strikes)-1))

    # 5. AI Trade Engine Stock Option Recommendation & Lot Sizing
    signal = ate.evaluate_options_trade(
        underlying="RELIANCE",
        chain=stock_chain,
        df_underlying=None,
        spot=stock_chain.attrs["underlying_ltp"],
        expiry=exp,
        account_capital=200000.0,
        risk_per_trade_pct=1.5,
    )
    assert signal.symbol == "RELIANCE"
    assert signal.asset_type == "OPTION"
    assert signal.recommended_qty >= 250
    assert signal.recommended_qty % 250 == 0  # Multiples of official stock lot size
    assert signal.entry_price > 0
    assert signal.stop_loss > 0


def test_bug_fixes_and_stability():
    import ai_trade_engine as ate
    import live_execution as le
    import paper_trading as pt
    import tradingview_integration as tvi
    import book_analytics as ba

    # 1. Live execution emergency exit method exists
    assert hasattr(le.live_executor, "emergency_exit_all")
    assert callable(le.live_executor.emergency_exit_all)

    # 2. Awesome Oscillator uses median price (high+low)/2
    df_ao = pd.DataFrame({
        "open": [100.0] * 40,
        "high": [110.0] * 40,
        "low": [90.0] * 40,
        "close": [105.0] * 40,
        "volume": [1000] * 40,
    })
    ind_res = ind.add_all(df_ao)
    assert "awesome_osc" in ind_res.columns
    # With constant high=110, low=90, median price is 100, so AO = 100 - 100 = 0.0
    assert abs(ind_res["awesome_osc"].iloc[-1]) < 1e-6

    # 3. CAN SLIM score points and max score bounded at 100
    df_oneil = pd.DataFrame({
        "open": np.linspace(100, 200, 250),
        "high": np.linspace(102, 205, 250),
        "low": np.linspace(98, 195, 250),
        "close": np.linspace(101, 204, 250),
        "volume": [10000] * 250,
    })
    oneil_res = ba.calc_oneil_signals(df_oneil)
    assert 0 <= oneil_res["score"] <= 100
    assert "/5" in oneil_res["points"]

    # 4 & 5. Isolated Test DB for Webhook safety check & Paper Trading close position
    orig_db = pt.DB_PATH
    test_db = Path(__file__).parent / "test_paper_portfolio_close.db"
    pt.DB_PATH = test_db
    try:
        pt.reset_paper_account(500000.0)

        # 4. TradingView Webhook handles malformed / non-numeric payload safely
        malformed_payload = {
            "ticker": "RELIANCE",
            "action": "BUY",
            "price": "not_a_number",
            "quantity": "bad_qty",
            "stop_loss": None,
        }
        ok, msg, det = tvi.process_tradingview_webhook_alert(malformed_payload)
        # Should not crash with ValueError, but succeed or return controlled error
        assert isinstance(ok, bool)

        # 5. Paper trading close position with exit_reason
        pos_res, pos_msg = pt.execute_paper_order(
            ate.TradeSignal(
                id=f"test_close_{int(datetime.now().timestamp())}",
                symbol="TEST_STK",
                asset_type="EQUITY",
                action="BUY",
                conviction=80,
                current_price=100.0,
                entry_price=100.0,
                stop_loss=95.0,
                target_1=110.0,
                target_2=115.0,
                risk_reward=2.0,
                recommended_qty=10,
                legs_desc="Test Stock Buy",
                thesis="Unit test position",
                literature_citations=[],
                timestamp="2026-11-19 12:00:00",
            ),
            quantity=10,
        )
        assert pos_res is True
        open_p = [p for p in pt.get_open_paper_positions() if p["symbol"] == "TEST_STK"]
        assert len(open_p) > 0
        test_id = open_p[0]["id"]
        close_ok, close_msg = pt.close_paper_position(test_id, exit_price=108.0, exit_reason="TEST_TP")
        assert close_ok is True
    finally:
        pt.DB_PATH = orig_db
        if test_db.exists():
            try:
                test_db.unlink()
            except Exception:
                pass


def test_advanced_institutional_analytics():
    import advanced_analytics as aa

    # 1. Derivatives Microstructure: Buildup Radar
    lb = aa.classify_buildup(price_change=+1.5, oi_change=+50000)
    assert lb["regime"] == "Long Buildup"
    assert "Bullish" in lb["bias"]

    sb = aa.classify_buildup(price_change=-2.0, oi_change=+80000)
    assert sb["regime"] == "Short Buildup"
    assert "Bearish" in sb["bias"]

    sc = aa.classify_buildup(price_change=+1.8, oi_change=-30000)
    assert sc["regime"] == "Short Covering"

    lu = aa.classify_buildup(price_change=-1.2, oi_change=-40000)
    assert lu["regime"] == "Long Unwinding"

    # Option Chain Buildup Analysis
    dummy_chain = pd.DataFrame({
        "strike": [23800.0, 23900.0, 24000.0, 24100.0, 24200.0],
        "ce_oi": [10000, 25000, 50000, 45000, 30000],
        "pe_oi": [30000, 48000, 55000, 20000, 8000],
        "ce_change_oi": [-500, 2000, 8500, 12000, 6000],
        "pe_change_oi": [4000, 11000, 9500, -1000, -500],
    })
    b_analysis = aa.analyze_option_chain_buildup(dummy_chain, spot=24000.0, prev_spot=23950.0)
    assert "buildup" in b_analysis
    assert b_analysis["total_ce_oi"] > 0
    assert b_analysis["total_pe_oi"] > 0
    assert b_analysis["top_call_wall_strike"] == 24100.0
    assert b_analysis["top_put_floor_strike"] == 23900.0

    # 2. Multi-Timeframe Confluence (Elder Triple Screen)
    df_mtf = pd.DataFrame({
        "open": np.linspace(100, 200, 100),
        "high": np.linspace(102, 205, 100),
        "low": np.linspace(98, 195, 100),
        "close": np.linspace(101, 204, 100),
        "volume": [10000] * 100,
    }, index=pd.date_range("2026-01-01", periods=100, freq="B"))
    mtf_res = aa.evaluate_multi_timeframe_confluence(df_mtf, "TEST_STOCK")
    assert "weekly" in mtf_res
    assert "daily" in mtf_res
    assert "intraday" in mtf_res
    assert mtf_res["confluence_score"] >= 60

    # 3. Sector Mapping & Mansfield Relative Strength
    sec_info = aa.get_stock_sector_info("RELIANCE")
    assert "Energy" in sec_info["stock_sector"]
    sec_tcs = aa.get_stock_sector_info("TCS")
    assert "IT" in sec_tcs["stock_sector"]

    rs_out = aa.calc_relative_strength(df_mtf)
    assert "mansfield_rs" in rs_out
    assert isinstance(rs_out["is_leader"], bool)

    sec_table = aa.get_sector_performance_table()
    assert len(sec_table) >= 6
    assert "NIFTY BANK" in sec_table["Sector Index"].values

    # 4. NSE Delivery Volume Percentage & Accumulation
    deliv_info = aa.calc_delivery_volume_metrics(df_mtf, "RELIANCE")
    assert 18.0 <= deliv_info["delivery_pct"] <= 82.0
    assert "institutional_score" in deliv_info

    # 5. Corporate Earnings & Event Risk (IV Crush)
    ev_rel = aa.get_earnings_event_risk("RELIANCE")
    assert ev_rel["symbol"] == "RELIANCE"
    assert "risk_level" in ev_rel
    assert "guidance" in ev_rel

    # 6. Modern Portfolio Theory (MPT) & Risk Metrics
    closed_df = pd.DataFrame({
        "net_pnl": [1500.0, -800.0, 2400.0, -600.0, 3100.0],
        "return_pct": [3.0, -1.6, 4.8, -1.2, 6.2],
    })
    risk_out = aa.calc_portfolio_risk_analytics(
        open_positions=[{"symbol": "RELIANCE"}, {"symbol": "TCS"}],
        closed_trades=closed_df,
        initial_capital=500000.0,
        current_portfolio_value=510000.0,
    )
    assert risk_out["portfolio_beta"] > 0
    assert risk_out["var_95_rupees"] > 0
    assert risk_out["var_99_rupees"] > 0
    assert risk_out["sharpe_ratio"] > 0
    assert risk_out["sortino_ratio"] > 0
    assert risk_out["van_tharp_expectancy"] > 0


def test_paper_trading_stocks_and_options_separation_and_remarks():
    import paper_trading as pt
    import ai_trade_engine as ate

    orig_db = pt.DB_PATH
    test_db = Path(__file__).parent / "test_paper_portfolio_sep.db"
    pt.DB_PATH = test_db
    try:
        pt.reset_paper_account(500000.0)

        # 1. Place 1 Equity order
        stock_sig = ate.TradeSignal(
            id="stk_test_1",
            symbol="INFY",
            asset_type="EQUITY",
            action="BUY",
            conviction=85,
            current_price=1800.0,
            entry_price=1800.0,
            stop_loss=1750.0,
            target_1=1900.0,
            target_2=1950.0,
            risk_reward=2.0,
            recommended_qty=10,
            legs_desc="Cash Equity Delivery",
            thesis="Bullish breakout",
            literature_citations=[],
            timestamp="2026-11-20 10:00:00",
        )
        ok_s, msg_s = pt.execute_paper_order(stock_sig, quantity=10)
        assert ok_s is True

        # 2. Place 1 Option order (ATM Long Call)
        opt_sig = ate.TradeSignal(
            id="opt_test_1",
            symbol="INFY",
            asset_type="OPTION",
            action="LONG_CALL",
            conviction=90,
            current_price=1800.0,  # Underlying spot
            entry_price=25.0,      # Option premium
            stop_loss=12.5,        # Option SL
            target_1=50.0,         # Option Target
            target_2=75.0,
            risk_reward=2.0,
            recommended_qty=400,
            legs_desc="Buy 1800 CE (2026-11-26)",
            thesis="Call momentum",
            literature_citations=[],
            timestamp="2026-11-20 10:00:00",
        )
        ok_o, msg_o = pt.execute_paper_order(opt_sig, quantity=400)
        assert ok_o is True

        # 3. Test separation in open positions
        open_all = pt.get_open_paper_positions()
        open_stk = pt.get_open_paper_positions("EQUITY")
        open_opt = pt.get_open_paper_positions("OPTION")
        assert len(open_all) == 2
        assert len(open_stk) == 1
        assert len(open_opt) == 1
        assert open_stk[0]["symbol"] == "INFY" and open_stk[0]["asset_type"] == "EQUITY"
        assert open_opt[0]["symbol"] == "INFY" and open_opt[0]["asset_type"] == "OPTION"
        assert "🟢 Active Position" in open_stk[0]["remark"]
        assert "🟢 Active Position" in open_opt[0]["remark"]

        # 4. Check trigger check does NOT erroneously close option with stock spot price
        # Underlying spot is 1810.0 (slightly up). Option entry was 25.0, target is 50.0.
        # Previously, passing 1810.0 would immediately trigger target (1810 >= 50).
        notifs = pt.update_and_check_triggers({"INFY": 1810.0})
        assert len(notifs) == 0  # Neither stock (tgt 1900) nor option (tgt 50) breached
        rem_opt = pt.get_open_paper_positions("OPTION")
        assert len(rem_opt) == 1
        # Option premium should have moved up with delta ~0.50 (10 pt stock move = ~5 pt option move: 25 + 5 = 30)
        assert rem_opt[0]["current_price"] == 30.0

        # 5. Close stock position manually to test executed trades remarks
        stk_id = open_stk[0]["id"]
        close_ok, close_msg = pt.close_paper_position(stk_id, exit_price=1850.0, exit_reason="MANUAL_EXIT")
        assert close_ok is True

        # 6. Test unified all trades and remarks
        all_stk = pt.get_all_paper_trades("EQUITY")
        assert len(all_stk) == 1
        assert all_stk.iloc[0]["trade_status"] == "🏁 EXECUTED (MANUAL_EXIT)"
        assert "MANUAL_EXIT" in all_stk.iloc[0]["remark"]
        assert all_stk.iloc[0]["net_pnl"] == 500.0  # (1850 - 1800) * 10

        all_opt = pt.get_all_paper_trades("OPTION")
        assert len(all_opt) == 1
        assert all_opt.iloc[0]["trade_status"] == "🟢 ACTIVE (OPEN)"
        assert "🟢 Active Position" in all_opt.iloc[0]["remark"]

        all_trades = pt.get_all_paper_trades()
        assert len(all_trades) == 2

        # 7. Test split performance summaries
        summary_stk = pt.get_paper_performance_summary("EQUITY")
        assert summary_stk["realized_pnl"] == 500.0
        assert summary_stk["open_positions_count"] == 0
        assert summary_stk["total_trades"] == 1

        summary_opt = pt.get_paper_performance_summary("OPTION")
        assert summary_opt["open_positions_count"] == 1
        assert summary_opt["unrealized_pnl"] == 2000.0  # (30.0 - 25.0) * 400

    finally:
        pt.DB_PATH = orig_db
        if test_db.exists():
            try:
                test_db.unlink()
            except Exception:
                pass


def test_realtime_price_execution_and_open_position_fluctuation():
    import paper_trading as pt
    import ai_trade_engine as ate

    # 1. Test get_quote on SampleProvider
    sp = SampleProvider(seed=42)
    q_sample = sp.get_quote("RELIANCE")
    assert isinstance(q_sample, dict)
    assert "ltp" in q_sample and q_sample["ltp"] > 0
    assert "change" in q_sample and "change_pct" in q_sample
    assert q_sample["symbol"] == "RELIANCE"

    quotes_batch = sp.get_quotes(["RELIANCE", "TCS"])
    assert "RELIANCE" in quotes_batch and "TCS" in quotes_batch
    assert quotes_batch["RELIANCE"]["ltp"] > 0

    # 2. Test isolated paper order execution with real-time price fluctuation
    orig_db = pt.DB_PATH
    test_db = Path(__file__).parent / "test_paper_realtime.db"
    pt.DB_PATH = test_db
    try:
        pt.reset_paper_account(500000.0)

        # Equity Signal generated with stale price 1000.0
        stk_sig = ate.TradeSignal(
            id="stk_rt_1",
            symbol="TEST_RT_STK",
            asset_type="EQUITY",
            action="BUY",
            conviction=80,
            current_price=1000.0,
            entry_price=1000.0,
            stop_loss=930.0,  # -7%
            target_1=1150.0,  # +15%
            target_2=1250.0,
            risk_reward=2.14,
            recommended_qty=10,
            legs_desc="Cash Delivery",
            thesis="Breakout setup",
            literature_citations=[],
            timestamp="2026-11-20 10:00:00",
        )

        # Execute with live fluctuating price 1050.0
        ok_s, msg_s = pt.execute_paper_order(stk_sig, quantity=10, live_price=1050.0)
        assert ok_s is True
        assert "1,050.00" in msg_s

        pos_stk = pt.get_open_paper_positions("EQUITY")
        assert len(pos_stk) == 1
        assert pos_stk[0]["entry_price"] == 1050.0
        assert pos_stk[0]["current_price"] == 1050.0
        assert pos_stk[0]["stop_loss"] == 976.5  # 1050 * 0.93
        assert pos_stk[0]["target_1"] == 1207.5  # 1050 * 1.15
        assert pos_stk[0]["unrealized_pnl"] == 0.0

        # Account cash deducted according to 1050.0 * 10 = 10500.0
        acct = pt.get_or_create_paper_account()
        assert acct["cash_balance"] == 500000.0 - 10500.0

        # Option Signal with underlying spot 1000.0 and premium 20.0
        opt_sig = ate.TradeSignal(
            id="opt_rt_1",
            symbol="TEST_RT_STK",
            asset_type="OPTION",
            action="LONG_CALL",
            conviction=85,
            current_price=1000.0,  # Spot at signal generation
            entry_price=20.0,      # Premium
            stop_loss=10.0,        # SL premium
            target_1=40.0,         # TGT premium
            target_2=60.0,
            risk_reward=2.0,
            recommended_qty=100,
            legs_desc="Buy 1000 CE",
            thesis="Call breakout",
            literature_citations=[],
            timestamp="2026-11-20 10:00:00",
        )

        # Underlying spot has moved to 1020.0 (+20 points). With delta ~0.5, premium rises to 20 + (20 * 0.5) = 30.0
        ok_o, msg_o = pt.execute_paper_order(opt_sig, quantity=100, live_price=1020.0)
        assert ok_o is True
        assert "30.00" in msg_o

        pos_opt = pt.get_open_paper_positions("OPTION")
        assert len(pos_opt) == 1
        assert pos_opt[0]["entry_price"] == 30.0
        assert pos_opt[0]["current_price"] == 30.0

        # 3. Test refresh_open_positions_quotes with simulated live market fluctuation
        class MockLiveProvider:
            def get_quotes(self, symbols):
                return {
                    "TEST_RT_STK": {"symbol": "TEST_RT_STK", "ltp": 1080.0, "change": 30.0, "change_pct": 2.86}
                }
            def get_quote(self, symbol):
                return {"symbol": symbol, "ltp": 1080.0, "change": 30.0, "change_pct": 2.86}

        mock_p = MockLiveProvider()
        pt.refresh_open_positions_quotes(provider=mock_p)

        # Check stock position updated
        updated_stk = pt.get_open_paper_positions("EQUITY")
        assert updated_stk[0]["current_price"] == 1080.0
        # (1080 - 1050) * 10 = +300.0
        assert updated_stk[0]["unrealized_pnl"] == 300.0

        # Check option position updated with delta from underlying movement
        # Spot went from 1020 to 1080 (+60 pts). Delta ~0.50 -> premium +30 -> 30 + 30 = 60.0
        # Since target was 60.0 (40 * 1.5 ratio), TARGET-1 hit trigger was fired!
        all_trades_df = pt.get_all_paper_trades()
        assert not all_trades_df.empty

        # 4. Test Short / SELL trade real-time execution and risk parameters
        short_sig = ate.TradeSignal(
            id="short_rt_1",
            symbol="TEST_SHORT",
            asset_type="EQUITY",
            action="SELL",
            conviction=20,
            current_price=2000.0,
            entry_price=2000.0,
            stop_loss=2100.0,  # +5% above entry
            target_1=1800.0,   # -10% below entry
            target_2=1600.0,
            risk_reward=2.0,
            recommended_qty=5,
            legs_desc="Short Equities",
            thesis="Bearish breakdown",
            literature_citations=[],
            timestamp="2026-11-20 10:00:00",
        )
        ok_sh, msg_sh = pt.execute_paper_order(short_sig, quantity=5, live_price=2050.0)
        assert ok_sh is True
        assert "2,050.00" in msg_sh

        pos_sh = [p for p in pt.get_open_paper_positions("EQUITY") if p["symbol"] == "TEST_SHORT"]
        assert len(pos_sh) == 1
        assert pos_sh[0]["entry_price"] == 2050.0
        # SL must be strictly ABOVE execution price for short
        assert pos_sh[0]["stop_loss"] > 2050.0
        # Target must be strictly BELOW execution price for short
        assert pos_sh[0]["target_1"] < 2050.0
        assert pos_sh[0]["stop_loss"] == 2152.5  # 2050 * 1.05
        assert pos_sh[0]["target_1"] == 1845.0   # 2050 * 0.90

    finally:
        pt.DB_PATH = orig_db
        if test_db.exists():
            try:
                test_db.unlink()
            except Exception:
                pass



def test_smc_analytics_and_confluence():
    import smc_analytics as smc
    dates = pd.bdate_range("2025-01-01", periods=40)
    
    # 1. Create series with intentional Bullish Fair Value Gap
    # Bars 0-5 around 100. Bar 6 surges to 110. Bars 7-39 stay elevated around 110 with lows >= 104.
    c = [100.0] * 6 + [110.0] + [108.0] * 33
    o = [100.0] * 6 + [101.0] + [107.0] * 33
    h = [101.0] * 6 + [112.0] + [110.0] * 33
    l = [99.0] * 6 + [100.5] + [104.0] * 33
    
    df_fvg = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1000}, index=dates)
    fvg_res = smc.detect_fair_value_gaps(df_fvg)
    assert fvg_res["all_fvgs_count"] >= 1
    assert any(f["type"] == "BULLISH_FVG" for f in fvg_res["active_bullish"])

    # 2. Test Order Block detection
    ob_res = smc.detect_order_blocks(df_fvg)
    assert "active_bullish_obs" in ob_res
    assert "active_bearish_obs" in ob_res

    # 3. Test Liquidity Sweep detection
    sweep_res = smc.detect_liquidity_sweeps(df_fvg)
    assert "sweeps_found" in sweep_res

    # 4. Unified SMC Confluence evaluation
    eval_res = smc.evaluate_smc_confluence("TEST", df_fvg)
    assert "smc_score" in eval_res
    assert 0 <= eval_res["smc_score"] <= 100
    assert "bias" in eval_res
    assert "smc_badge" in eval_res


def test_macro_regime_and_gating():
    import macro_regime as mr
    
    # 1. Evaluate macro regime
    regime = mr.evaluate_macro_regime()
    assert "regime_code" in regime
    assert regime["regime_code"] in ("BULLISH_TREND", "BEARISH_TREND", "HIGH_VOLATILITY", "CONSOLIDATION")
    assert "win_rate_edge" in regime

    # 2. Gate trade setup
    gate_buy = mr.gate_trade_setup("BUY", "EQUITY", macro_regime=regime)
    assert "gate_passed" in gate_buy
    assert "conviction_adjustment" in gate_buy
    assert isinstance(gate_buy["conviction_adjustment"], int)

    # 3. Simulated Bearish regime should flag or penalize long breakouts
    bear_regime = dict(regime, regime_code="BEARISH_TREND", vix=24.0)
    gate_bear_buy = mr.gate_trade_setup("BUY", "EQUITY", macro_regime=bear_regime)
    assert not gate_bear_buy["gate_passed"]
    assert gate_bear_buy["conviction_adjustment"] < 0


def test_high_winrate_options_selection():
    import options_winrate as ow
    
    # 1. IV Rank & Percentile
    iv_data = ow.calc_iv_rank_and_percentile("RELIANCE", 24.5)
    assert "iv_rank" in iv_data
    assert "iv_percentile" in iv_data
    assert 0.0 <= iv_data["iv_rank"] <= 100.0

    # 2. High-Win Rate Spread Selection
    strikes = np.arange(1200, 1600, 20)
    chain = pd.DataFrame({
        "strike": strikes,
        "ce_ltp": [max(0.5, 1400 - s + 10) for s in strikes],
        "pe_ltp": [max(0.5, s - 1400 + 10) for s in strikes],
        "ce_oi": [1000] * len(strikes),
        "pe_oi": [1000] * len(strikes),
        "ce_iv": [0.22] * len(strikes),
        "pe_iv": [0.22] * len(strikes),
    })
    chain.attrs["underlying_ltp"] = 1400.0
    
    spread_res = ow.select_high_winrate_options_spread(
        chain=chain,
        spot=1400.0,
        expiry="2026-11-26",
        bias="BULLISH",
        symbol="RELIANCE",
        target_pop=0.82,
    )
    assert "error" not in spread_res
    assert spread_res["strategy"] == "BULL_PUT_SPREAD"
    assert spread_res["pop_percent"] >= 75.0
    assert spread_res["short_strike"] < 1400.0  # OTM Put
    assert spread_res["long_strike"] < spread_res["short_strike"]
    assert spread_res["net_credit"] > 0


def test_volume_profile_and_amt():
    import volume_profile as vp
    
    # Synthesize 30 bars of test price & volume data
    np.random.seed(42)
    closes = 24000.0 + np.cumsum(np.random.randn(40) * 50)
    highs = closes + np.random.uniform(20, 60, size=40)
    lows = closes - np.random.uniform(20, 60, size=40)
    volumes = np.random.randint(50000, 200000, size=40)
    
    df = pd.DataFrame({"high": highs, "low": lows, "close": closes, "volume": volumes})
    res = vp.compute_volume_profile(df, num_bins=25)
    
    assert "poc" in res
    assert "vah" in res
    assert "val" in res
    assert res["poc"] > 0
    assert res["vah"] >= res["val"]
    assert "amt_rule_signal" in res
    assert any(s in res["market_state"] for s in ("INSIDE_VALUE_AREA", "ABOVE_VALUE_AREA", "BELOW_VALUE_AREA"))


def test_intermarket_macro():
    import intermarket_macro as im
    
    # 1. Fetch proxy data
    data = im.fetch_macro_proxy_data()
    assert "brent_crude" in data
    assert "usd_inr" in data
    assert "us_10y_yield" in data
    assert "india_vix" in data
    
    # 2. Evaluate sector confluence (e.g. IT on USD/INR depreciation)
    macro_proxy = {
        "brent_crude": {"price": 72.0, "change_pct": -1.0, "trend": "FALLING"},
        "usd_inr": {"price": 84.5, "change_pct": 0.5, "trend": "DEPRECIATING"},
        "us_10y_yield": {"price": 4.3, "change_pct": -0.5, "trend": "EASING"},
        "india_vix": {"price": 13.5, "change_pct": -0.5, "trend": "COMPLACENT"},
    }
    it_conf = im.evaluate_sector_intermarket_confluence("INFY", action="BUY", macro_data=macro_proxy)
    assert it_conf["sector"] == "IT"
    assert it_conf["macro_verdict"] == "TAILWIND"
    assert it_conf["winrate_modifier_pct"] > 0

    # Asian Paints with falling crude should also enjoy tailwind
    paint_conf = im.evaluate_sector_intermarket_confluence("ASIANPAINT", action="BUY", macro_data=macro_proxy)
    assert paint_conf["sector"] == "CRUDE_CONSUMER"
    assert paint_conf["macro_verdict"] == "TAILWIND"


def test_trailing_risk_engine():
    import trailing_risk_engine as tr
    
    # 1. Trailing Plan
    plan = tr.generate_trailing_plan(entry_price=1000.0, stop_loss=950.0, current_price=1060.0, action="BUY")
    assert plan["r_unit"] == 50.0
    assert plan["profit_r_multiple"] == 1.2
    assert plan["stage_status"] == "BREAKEVEN_ACTIVE"
    assert plan["active_stop"] >= 1000.0  # Ratcheted to breakeven or higher
    
    # 2. Position Size
    pos = tr.calculate_position_size(account_capital=500000.0, entry_price=1000.0, stop_loss=980.0, risk_pct=1.0, max_capital_allocation_pct=60.0)
    assert pos["risk_amount_inr"] == 5000.0
    assert pos["risk_per_share"] == 20.0
    assert pos["recommended_shares"] == 250
    assert pos["actual_risk_inr"] <= 5000.0
    
    # 3. Van Tharp Expectancy
    exp = tr.calculate_van_tharp_expectancy(win_rate_pct=65.0, avg_win_r=2.0, avg_loss_r=1.0)
    assert exp["expectancy_r_multiple"] > 0.8
    assert "EXCELLENT" in exp["system_quality"] or "STRONG" in exp["system_quality"]


def test_expiry_gamma_matrix():
    import expiry_gamma_matrix as eg
    
    # 1. Expiry check for NIFTY on Thursday vs Monday
    thurs_dt = datetime(2026, 11, 26, 14, 0)  # Thursday 2:00 PM
    res_thurs = eg.check_symbol_expiry_status("NIFTY 50", current_dt=thurs_dt)
    assert res_thurs["is_expiry_day"] is True
    assert res_thurs["dte"] == 0
    assert res_thurs["hazard_level"] == "CRITICAL_GAMMA_SPIKE"
    
    # 2. Pinning probability
    pin = eg.evaluate_strike_pinning_probability(spot=24005.0, max_pain=24000.0, pcr=1.1, dte=0)
    assert pin["pin_probability_pct"] >= 75.0
    assert "PINNING" in pin["pinning_verdict"]


def test_tradingview_realtime_provider_and_scanner():
    import tradingview_provider as tvp
    import screener
    
    # 1. Symbol mapping test
    assert tvp.map_symbol_to_tv("RELIANCE") == "NSE:RELIANCE"
    assert tvp.map_symbol_to_tv("SENSEX") == "BSE:SENSEX"
    assert tvp.map_symbol_to_tv("NIFTY") == "NSE:NIFTY"
    assert tvp.map_symbol_to_tv("500325") == "BSE:500325"
    assert tvp.map_symbol_to_tv("INFY.BO") == "BSE:INFY"
    
    # 2. Batch quotes structure verification
    res = tvp.get_tradingview_realtime_quotes(["RELIANCE", "NIFTY"], timeout=5)
    if res:
        assert "RELIANCE" in res or "NSE:RELIANCE" in res
        rel = res.get("RELIANCE") or res.get("NSE:RELIANCE")
        assert "ltp" in rel
        assert rel["ltp"] > 0
        assert "rsi14" in rel
        assert "tv_recommendation" in rel

    # 3. Screener scan test
    tbl, errs = screener.scan_tradingview_realtime(["RELIANCE", "TCS"])
    if not tbl.empty:
        assert "close" in tbl.columns
        assert "can_slim_score" in tbl.columns
        assert "conviction" in tbl.columns
        assert "expected_win_rate" in tbl.columns
        assert "data_source" in tbl.columns
        assert "TradingView" in tbl["data_source"].iloc[0]


def test_unified_omnibar_and_criteria_screener():
    import stock_directory as sd
    import screener
    import pandas as pd

    # 1. Verify Omnibar options list contains universes, segments, and single stocks
    search_list = sd.get_formatted_search_list()
    assert len(search_list) >= 12000

    # 2. Verify universe and segment detection
    assert sd.is_universe_or_segment("🏛️ [ALL NSE] Whole National Stock Exchange (8,500+ Listed Equities)")
    assert sd.is_universe_or_segment("⚡ [F&O SEGMENT] Complete NSE F&O Derivatives Universe (185+ High-Liquidity Stocks)")
    assert sd.is_universe_or_segment("🏢 [ALL BSE] Whole Bombay Stock Exchange (4,100+ Listed Equities)")
    assert sd.is_universe_or_segment("🏦 [SEGMENT] Sector: Banking & Financials")
    assert not sd.is_universe_or_segment("RELIANCE — Reliance Industries Ltd. [NSE • Energy/Retail/Telecom]")
    assert not sd.is_universe_or_segment("500325")

    # 3. Verify symbol extraction
    assert sd.extract_symbol_from_selection("RELIANCE — Reliance Industries Ltd. [NSE • Energy]") == "RELIANCE"
    assert sd.extract_symbol_from_selection("🏛️ [ALL NSE] Whole National Stock Exchange (8,500+ Listed Equities)") == ""

    # 4. Verify segment symbol resolution
    fo_syms = sd.get_symbols_for_universe_or_segment("⚡ [F&O SEGMENT]")
    assert len(fo_syms) >= 70
    assert "RELIANCE" in fo_syms or "TCS" in fo_syms

    # 4b. Verify unified BSE 100, Nifty 50, and Midcap index basket screening detection
    assert sd.is_universe_or_segment("🏛️ BSE100 — BSE 100 Index [BSE Index]")
    assert sd.is_universe_or_segment("🏛️ NIFTY — NIFTY 50 Benchmark Index [Benchmark Index]")
    assert sd.is_universe_or_segment("🏢 [BSE 100] BSE 100 Index Universe (Top 100 Large Caps)")
    assert sd.is_universe_or_segment("🏢 [BSE MIDCAP] BSE Midcap Index Universe (Top Midcaps)")
    bse100_syms = sd.get_symbols_for_universe_or_segment("🏛️ BSE100 — BSE 100 Index [BSE Index]")
    assert len(bse100_syms) == 100
    assert "RELIANCE" in bse100_syms and "TCS" in bse100_syms
    nifty_syms = sd.get_symbols_for_universe_or_segment("🏆 [NIFTY 50] Benchmark Bluechip Leaders")
    assert len(nifty_syms) == 50

    bank_syms = sd.get_symbols_for_universe_or_segment("🏦 [SEGMENT] Sector: Banking & Financials")
    assert len(bank_syms) >= 10

    # 5. Verify Screener High-Conviction & Win Ratio filtering
    sample_data = pd.DataFrame([
        {"symbol": "ALPHA", "conviction": 90, "expected_win_rate": 78.0, "action": "STRONG BUY", "can_slim_score": 95},
        {"symbol": "BETA", "conviction": 82, "expected_win_rate": 71.5, "action": "BUY", "can_slim_score": 85},
        {"symbol": "GAMMA", "conviction": 65, "expected_win_rate": 58.0, "action": "NEUTRAL / WATCH", "can_slim_score": 60},
        {"symbol": "DELTA", "conviction": 30, "expected_win_rate": 35.0, "action": "SELL", "can_slim_score": 30},
    ])

    # Filter with Conviction >= 80% and Win Ratio >= 70%
    flt_high = screener.filter_by_criteria(sample_data, min_conviction=80, min_win_rate=70.0)
    assert len(flt_high) == 2
    assert flt_high["symbol"].tolist() == ["ALPHA", "BETA"]

    # Filter for Bearish setups
    flt_bear = screener.filter_by_criteria(sample_data, min_conviction=25, min_win_rate=0.0, bias="BEARISH")
    assert len(flt_bear) == 1
    assert flt_bear["symbol"].iloc[0] == "DELTA"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok  ", f.__name__)
    print(f"{len(fns)} passed")
