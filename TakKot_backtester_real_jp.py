from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import pandas as pd
except ImportError:
    print("Missing dependency: pandas")
    print("Install with: python -m pip install pandas yfinance matplotlib")
    sys.exit(1)

try:
    import yfinance as yf
except ImportError:
    print("Missing dependency: yfinance")
    print("Install with: python -m pip install pandas yfinance matplotlib")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
BLUE = "\033[34m"


# =============================================================================
# PARSE YOUR CONSTITUENT FILE FORMAT DIRECTLY
# =============================================================================

class HistoricalSP500Constituents:
    """
    Parses a text file where each data row looks like:
        1996-01-02,"AAL,AAMRQ,AAPL,..."
    """

    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Constituent file not found: {path}")

        self.dates: list[dt.date] = []
        self.members_by_date: list[list[str]] = []
        self._parse()

    def _normalize_ticker(self, t: str) -> str:
        return t.strip().upper().replace(".", "-")

    def _parse(self) -> None:
        text = self.path.read_text(encoding="utf-8", errors="ignore")
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        for line in lines:
            # skip header lines like:
            # ##S@P500 Constituent Data 01/01/1996 - 01/14/2026
            if not line or not line[0].isdigit():
                continue

            # expected:
            # 1996-01-02,"AAL,AAMRQ,AAPL,..."
            try:
                date_part, tickers_part = line.split(",", 1)
            except ValueError:
                continue

            date_part = date_part.strip()
            try:
                d = dt.datetime.strptime(date_part, "%Y-%m-%d").date()
            except ValueError:
                continue

            tickers_part = tickers_part.strip()
            if tickers_part.startswith('"') and tickers_part.endswith('"'):
                tickers_part = tickers_part[1:-1]

            raw_tickers = tickers_part.split(",")
            cleaned = []
            seen = set()

            for t in raw_tickers:
                t2 = self._normalize_ticker(t)
                if t2 and t2 not in seen:
                    seen.add(t2)
                    cleaned.append(t2)

            if cleaned:
                self.dates.append(d)
                self.members_by_date.append(cleaned)

        if not self.dates:
            raise RuntimeError("No valid constituent rows parsed from file.")

    def get_members_on(self, asof_date: dt.date) -> list[str]:
        """
        Returns membership for the latest file date <= asof_date.
        """
        idx = bisect_right(self.dates, asof_date) - 1
        if idx < 0:
            return []
        return self.members_by_date[idx]

    def build_union_for_period(self, start_date: dt.date, end_date: dt.date) -> list[str]:
        union = set()
        for d, members in zip(self.dates, self.members_by_date):
            if start_date <= d <= end_date:
                union.update(members)
        return sorted(union)


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass
class Bar:
    date: dt.date
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class BNFParams:
    ma_period: int = 25
    dev_threshold: float = 0.08
    rsi_period: int = 14
    rsi_level: float = 40.0
    use_rsi: bool = False
    use_bollinger: bool = False
    use_vol_spike: bool = False
    vol_mult: float = 1.5
    require_reversal: bool = False
    stop_loss: float = 0.05
    take_profit: float = 0.10
    max_hold_days: int = 6
    initial_capital: float = 100_000.0

    # half Kelly controls
    kelly_lookback: int = 30
    min_kelly_trades: int = 10
    max_kelly_frac: float = 0.25
    half_kelly_mult: float = 0.50
    fallback_risk: float = 0.02


@dataclass
class Trade:
    stock_id: int
    ticker: str
    entry_date: dt.date
    exit_date: dt.date
    entry_price: float
    exit_price: float
    shares: int
    pnl_pct: float
    pnl_dollar: float
    days_held: int
    exit_reason: str
    ma_dev_pct: float
    rsi_at_entry: Optional[float]
    kelly_frac: Optional[float]


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: list[tuple[dt.date, float]]
    bnh_curve: list[tuple[dt.date, float]]
    initial_cap: float
    final_cap: float
    total_return: float
    cagr: float
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: Optional[float]
    max_drawdown: float
    sharpe: float
    sortino: float
    n_trades: int
    n_days: int
    years: float


# =============================================================================
# DOWNLOAD HELPERS
# =============================================================================

def download_ohlcv(
    tickers: list[str],
    start: dt.datetime,
    end: dt.datetime,
) -> pd.DataFrame:
    data = yf.download(
        tickers=tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        actions=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    if data is None or len(data) == 0:
        raise RuntimeError("No data downloaded from Yahoo Finance.")
    return data


def download_panel_for_backtest(
    tickers: list[str],
    start_date: dt.date,
    end_date: dt.date,
    extra_lookback_days: int = 252 + 60,
) -> dict[str, pd.DataFrame]:
    start = dt.datetime.combine(start_date - dt.timedelta(days=extra_lookback_days), dt.time.min)
    end = dt.datetime.combine(end_date + dt.timedelta(days=5), dt.time.min)
    raw = download_ohlcv(tickers, start, end)
    panel: dict[str, pd.DataFrame] = {}

    if isinstance(raw.columns, pd.MultiIndex):
        available = set(raw.columns.get_level_values(0))
        for t in tickers:
            if t not in available:
                continue
            df = raw[t].copy()
            needed = ["Open", "High", "Low", "Close", "Volume"]
            if not all(c in df.columns for c in needed):
                continue
            df = df[needed].dropna()
            df = df[df["Volume"] > 0]
            if len(df) >= 80:
                panel[t] = df
    else:
        if len(tickers) == 1:
            df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
            df = df[df["Volume"] > 0]
            if len(df) >= 80:
                panel[tickers[0]] = df

    if not panel:
        raise RuntimeError("No usable OHLCV data downloaded.")
    return panel


# =============================================================================
# INDICATORS
# =============================================================================

def sma(bars: list[Bar], n: int) -> list[Optional[float]]:
    out = [None] * len(bars)
    running = 0.0
    for i, b in enumerate(bars):
        running += b.close
        if i >= n:
            running -= bars[i - n].close
        if i >= n - 1:
            out[i] = running / n
    return out


def rsi_series(bars: list[Bar], n: int = 14) -> list[Optional[float]]:
    out = [None] * len(bars)
    if len(bars) < n + 1:
        return out

    gains = 0.0
    losses = 0.0
    for i in range(1, n + 1):
        d = bars[i].close - bars[i - 1].close
        if d > 0:
            gains += d
        else:
            losses -= d

    avg_gain = gains / n
    avg_loss = losses / n

    for i in range(n, len(bars)):
        if i > n:
            d = bars[i].close - bars[i - 1].close
            gain = max(d, 0.0)
            loss = max(-d, 0.0)
            avg_gain = ((n - 1) * avg_gain + gain) / n
            avg_loss = ((n - 1) * avg_loss + loss) / n

        if avg_loss == 0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - 100.0 / (1.0 + rs)

    return out


def bollinger_series(
    bars: list[Bar],
    n: int = 20,
    k: float = 2.0,
) -> list[Optional[tuple[float, float, float]]]:
    out = [None] * len(bars)
    for i in range(n - 1, len(bars)):
        window = [b.close for b in bars[i - n + 1:i + 1]]
        mu = sum(window) / n
        sd = math.sqrt(sum((x - mu) ** 2 for x in window) / n)
        out[i] = (mu + k * sd, mu, mu - k * sd)
    return out


def avg_vol_series(bars: list[Bar], n: int = 20) -> list[Optional[float]]:
    out = [None] * len(bars)
    for i in range(n, len(bars)):
        out[i] = sum(b.volume for b in bars[i - n:i]) / n
    return out


def _precompute(bars: list[Bar], p: BNFParams) -> dict:
    return {
        "ma": sma(bars, p.ma_period),
        "rs": rsi_series(bars, p.rsi_period),
        "bb": bollinger_series(bars) if p.use_bollinger else [None] * len(bars),
        "av": avg_vol_series(bars) if p.use_vol_spike else [None] * len(bars),
    }


# =============================================================================
# KELLY
# =============================================================================

def compute_half_kelly_fraction(trades: list[Trade], params: BNFParams) -> float:
    if len(trades) < params.min_kelly_trades:
        return params.fallback_risk

    sample = trades[-params.kelly_lookback:]
    wins = [t for t in sample if t.pnl_pct > 0]
    losses = [t for t in sample if t.pnl_pct <= 0]

    if not wins or not losses:
        return params.fallback_risk

    p = len(wins) / len(sample)
    avg_win = sum(t.pnl_pct for t in wins) / len(wins)
    avg_loss = abs(sum(t.pnl_pct for t in losses) / len(losses))

    if avg_win <= 0 or avg_loss <= 0:
        return params.fallback_risk

    b = avg_win / avg_loss
    full_kelly = p - (1.0 - p) / b
    full_kelly = max(0.0, min(full_kelly, params.max_kelly_frac))
    return full_kelly * params.half_kelly_mult


# =============================================================================
# DYNAMIC HISTORICAL UNIVERSE
# =============================================================================

def realized_vol_from_df(df: pd.DataFrame, asof_date: dt.date, lookback_days: int) -> Optional[float]:
    sub = df[df.index.date <= asof_date]
    if len(sub) < lookback_days + 1:
        return None

    closes = sub["Close"].tail(lookback_days + 1)
    rets = closes.pct_change().dropna()
    if len(rets) < int(lookback_days * 0.7):
        return None

    vol = rets.std(ddof=0) * math.sqrt(252)
    if pd.isna(vol) or vol <= 0:
        return None
    return float(vol)


def build_master_dates(panel: dict[str, pd.DataFrame], start_date: dt.date, end_date: dt.date) -> list[dt.date]:
    all_dates = set()

    for df in panel.values():
        for idx in df.index:
            d = idx.date()
            if start_date <= d <= end_date:
                all_dates.add(d)

    dates = sorted(all_dates)
    if len(dates) < 20:
        raise RuntimeError("Too few dates for backtest.")
    return dates

def make_aligned_bar_series(panel: dict[str, pd.DataFrame], dates: list[dt.date]) -> dict[str, list[Optional[Bar]]]:
    out: dict[str, list[Optional[Bar]]] = {}
    date_map = {d: i for i, d in enumerate(dates)}

    for t, df in panel.items():
        series = [None] * len(dates)
        for idx, row in df.iterrows():
            d = idx.date()
            j = date_map.get(d)
            if j is None:
                continue
            series[j] = Bar(
                date=d,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=int(row["Volume"]),
            )
        out[t] = series

    return out


def build_dynamic_universe_schedule(
    panel: dict[str, pd.DataFrame],
    constituents: HistoricalSP500Constituents,
    dates: list[dt.date],
    top_n: int,
    vol_lookback_days: int,
    rebalance_every_days: int,
) -> dict[dt.date, set[str]]:
    schedule: dict[dt.date, set[str]] = {}
    current_selection: set[str] = set()

    for i, d in enumerate(dates):
        if i % rebalance_every_days == 0:
            members = constituents.get_members_on(d)
            ranked: list[tuple[str, float]] = []

            for t in members:
                df = panel.get(t)
                if df is None:
                    continue
                vol = realized_vol_from_df(df, d, vol_lookback_days)
                if vol is not None:
                    ranked.append((t, vol))

            ranked.sort(key=lambda x: x[1], reverse=True)
            current_selection = set(t for t, _ in ranked[:top_n])

        schedule[d] = set(current_selection)

    return schedule


# =============================================================================
# BACKTEST
# =============================================================================

def run_backtest_dynamic_universe(
    panel: dict[str, pd.DataFrame],
    params: BNFParams,
    constituents: HistoricalSP500Constituents,
    top_n: int,
    vol_lookback_days: int,
    rebalance_every_days: int,
    start_date: dt.date,
    end_date: dt.date,
) -> BacktestResult:
    dates = build_master_dates(panel, start_date=start_date, end_date=end_date)
    bars_by_ticker = make_aligned_bar_series(panel, dates)

    per_ticker_precomp: dict[str, dict] = {}
    per_ticker_fullbars: dict[str, list[Bar]] = {}

    for t, series in bars_by_ticker.items():
        compact = [b for b in series if b is not None]
        if len(compact) >= 80:
            per_ticker_fullbars[t] = compact
            per_ticker_precomp[t] = _precompute(compact, params)

    universe_schedule = build_dynamic_universe_schedule(
        panel=panel,
        constituents=constituents,
        dates=dates,
        top_n=top_n,
        vol_lookback_days=vol_lookback_days,
        rebalance_every_days=rebalance_every_days,
    )

    cap = params.initial_capital
    trades: list[Trade] = []
    equity_curve: list[tuple[dt.date, float]] = []
    positions: dict[str, dict] = {}

    compact_lookup: dict[str, dict[dt.date, int]] = {}
    for t, bars in per_ticker_fullbars.items():
        compact_lookup[t] = {b.date: i for i, b in enumerate(bars)}

    first_valid_date = None
    first_valid_names = None
    for d in dates:
        names = universe_schedule[d]
        usable = [t for t in names if t in compact_lookup and d in compact_lookup[t]]
        if usable:
            first_valid_date = d
            first_valid_names = usable
            break

    if first_valid_date is None or first_valid_names is None:
        raise RuntimeError("Could not form initial historical universe.")

    bnh_start_prices = {}
    for t in first_valid_names:
        idx = compact_lookup[t][first_valid_date]
        bnh_start_prices[t] = per_ticker_fullbars[t][idx].close

    bnh_curve: list[tuple[dt.date, float]] = []

    for d in dates:
        current_universe = universe_schedule[d]

        open_value = 0.0
        for t, pos in positions.items():
            idx_map = compact_lookup.get(t, {})
            if d in idx_map:
                bar = per_ticker_fullbars[t][idx_map[d]]
                open_value += pos["shares"] * bar.close

        equity_curve.append((d, cap + open_value))

        to_remove = []
        for t, pos in positions.items():
            idx_map = compact_lookup.get(t, {})
            if d not in idx_map:
                continue

            compact_i = idx_map[d]
            bars = per_ticker_fullbars[t]
            bar = bars[compact_i]
            held = (d - pos["entry_date"]).days
            reason = None
            exit_price = None

            if bar.low <= pos["stop"]:
                exit_price = pos["stop"]
                reason = "Stop Loss"
            elif bar.high >= pos["target"]:
                exit_price = pos["target"]
                reason = "Take Profit"
            elif held >= params.max_hold_days:
                exit_price = bar.close
                reason = "Time Stop"
            elif t not in current_universe:
                exit_price = bar.close
                reason = "Universe Exit"

            if reason is not None:
                pnl_dollar = pos["shares"] * (exit_price - pos["entry_price"])
                cap += pnl_dollar + pos["cost"]

                trades.append(
                    Trade(
                        stock_id=0,
                        ticker=t,
                        entry_date=pos["entry_date"],
                        exit_date=d,
                        entry_price=pos["entry_price"],
                        exit_price=exit_price,
                        shares=pos["shares"],
                        pnl_pct=(exit_price / pos["entry_price"] - 1.0) * 100.0,
                        pnl_dollar=pnl_dollar,
                        days_held=(d - pos["entry_date"]).days,
                        exit_reason=reason,
                        ma_dev_pct=pos["ma_dev"],
                        rsi_at_entry=pos["rsi_entry"],
                        kelly_frac=pos["kelly_frac"],
                    )
                )
                to_remove.append(t)

        for t in to_remove:
            positions.pop(t, None)

        for t in sorted(current_universe):
            if t in positions:
                continue
            if t not in compact_lookup:
                continue
            if d not in compact_lookup[t]:
                continue

            bars = per_ticker_fullbars[t]
            i = compact_lookup[t][d]

            warmup = max(params.ma_period, params.rsi_period + 1, 21)
            if i < warmup:
                continue

            pre = per_ticker_precomp[t]
            bar = bars[i]
            ma_val = pre["ma"][i]
            rs_val = pre["rs"][i]
            bb_val = pre["bb"][i]
            av_val = pre["av"][i]

            if ma_val is None:
                continue

            dev = (bar.close - ma_val) / ma_val

            c_ma = dev <= -params.dev_threshold
            c_rsi = (not params.use_rsi) or (rs_val is not None and rs_val <= params.rsi_level)
            c_bb = (not params.use_bollinger) or (bb_val is not None and bar.close <= bb_val[2])
            c_vol = (not params.use_vol_spike) or (av_val is not None and bar.volume >= av_val * params.vol_mult)
            c_rev = (not params.require_reversal) or (bar.close > bar.open)

            if not (c_ma and c_rsi and c_bb and c_vol and c_rev):
                continue

            stop_p = bar.close * (1.0 - params.stop_loss)
            target_p = bar.close * (1.0 + params.take_profit)
            risk_per_share = bar.close - stop_p
            if risk_per_share <= 0:
                continue

            half_kelly_frac = compute_half_kelly_fraction(trades, params)
            dollar_risk_budget = cap * half_kelly_frac
            shares = int(dollar_risk_budget / risk_per_share)
            cost = shares * bar.close

            if shares < 1 or cost > cap:
                continue

            cap -= cost
            positions[t] = {
                "entry_date": d,
                "entry_price": bar.close,
                "stop": stop_p,
                "target": target_p,
                "shares": shares,
                "cost": cost,
                "ma_dev": dev * 100.0,
                "rsi_entry": rs_val,
                "kelly_frac": half_kelly_frac,
            }

        usable_bnh = [t for t in bnh_start_prices if t in compact_lookup and d in compact_lookup[t]]
        if usable_bnh:
            val = params.initial_capital / len(usable_bnh) * sum(
                per_ticker_fullbars[t][compact_lookup[t][d]].close / bnh_start_prices[t]
                for t in usable_bnh
            )
        else:
            val = params.initial_capital
        bnh_curve.append((d, val))

    final_date = dates[-1]
    for t, pos in list(positions.items()):
        if final_date not in compact_lookup[t]:
            continue
        bar = per_ticker_fullbars[t][compact_lookup[t][final_date]]
        pnl_dollar = pos["shares"] * (bar.close - pos["entry_price"])
        cap += pnl_dollar + pos["cost"]

        trades.append(
            Trade(
                stock_id=0,
                ticker=t,
                entry_date=pos["entry_date"],
                exit_date=final_date,
                entry_price=pos["entry_price"],
                exit_price=bar.close,
                shares=pos["shares"],
                pnl_pct=(bar.close / pos["entry_price"] - 1.0) * 100.0,
                pnl_dollar=pnl_dollar,
                days_held=(final_date - pos["entry_date"]).days,
                exit_reason="End",
                ma_dev_pct=pos["ma_dev"],
                rsi_at_entry=pos["rsi_entry"],
                kelly_frac=pos["kelly_frac"],
            )
        )

    if equity_curve:
        equity_curve[-1] = (equity_curve[-1][0], cap)

    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]

    win_rate = len(wins) / len(trades) * 100.0 if trades else 0.0
    avg_win_pct = sum(t.pnl_pct for t in wins) / len(wins) if wins else 0.0
    avg_loss_pct = sum(t.pnl_pct for t in losses) / len(losses) if losses else 0.0

    gross_win = sum(t.pnl_dollar for t in wins)
    gross_loss = abs(sum(t.pnl_dollar for t in losses))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else None

    peak = params.initial_capital
    max_drawdown = 0.0
    for _, eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_drawdown:
            max_drawdown = dd

    eq_vals = [v for _, v in equity_curve]
    rets = [
        (eq_vals[i] - eq_vals[i - 1]) / eq_vals[i - 1]
        for i in range(1, len(eq_vals))
        if eq_vals[i - 1] > 0
    ]

    if rets:
        mu = sum(rets) / len(rets)
        sd = math.sqrt(sum((r - mu) ** 2 for r in rets) / len(rets))
        downside = [r for r in rets if r < 0]
        downside_sd = math.sqrt(sum(r * r for r in downside) / len(downside)) if downside else 0.0
    else:
        mu = 0.0
        sd = 0.0
        downside_sd = 0.0

    sharpe = (mu / sd) * math.sqrt(252) if sd > 0 else 0.0
    sortino = (mu / downside_sd) * math.sqrt(252) if downside_sd > 0 else 0.0

    years_calc = max(len(dates) / 252.0, 1e-9)
    cagr = ((cap / params.initial_capital) ** (1.0 / years_calc) - 1.0) * 100.0 if cap > 0 else -100.0

    return BacktestResult(
        trades=sorted(trades, key=lambda t: (t.entry_date, t.ticker)),
        equity_curve=equity_curve,
        bnh_curve=bnh_curve,
        initial_cap=params.initial_capital,
        final_cap=cap,
        total_return=(cap / params.initial_capital - 1.0) * 100.0,
        cagr=cagr,
        win_rate=win_rate,
        avg_win_pct=avg_win_pct,
        avg_loss_pct=avg_loss_pct,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown * 100.0,
        sharpe=sharpe,
        sortino=sortino,
        n_trades=len(trades),
        n_days=len(dates),
        years=years_calc,
    )


# =============================================================================
# REPORT / CHART
# =============================================================================

def _c(val: float, fmt: str = "+.2f", suf: str = "%") -> str:
    s = f"{val:{fmt}}{suf}"
    return f"{GREEN}{s}{RESET}" if val >= 0 else f"{RED}{s}{RESET}"


def print_report(result: BacktestResult) -> None:
    sep = "─" * 88
    bnh_ret = (result.bnh_curve[-1][1] / result.initial_cap - 1.0) * 100.0 if result.bnh_curve else 0.0

    print(f"\n{CYAN}{BOLD}BNF / Takashi Kotegawa — Historical Constituents Backtest{RESET}")
    print(f"{DIM}Uses your date-by-line constituent text file directly{RESET}\n")
    print(sep)
    print(f"{BOLD}Performance{RESET}")
    print(f"Total return           : {_c(result.total_return)}   (Buy & Hold: {_c(bnh_ret)})")
    print(f"CAGR                   : {_c(result.cagr)}")
    print(f"Final equity           : ${result.final_cap:,.0f}")
    print(f"Sharpe                 : {BLUE}{result.sharpe:.2f}{RESET}")
    print(f"Sortino                : {BLUE}{result.sortino:.2f}{RESET}")
    print(f"Max drawdown           : {RED}{result.max_drawdown:.2f}%{RESET}")
    print(sep)
    print(f"{BOLD}Trade Statistics{RESET}")
    print(f"Total trades           : {result.n_trades}")
    print(f"Trades / year          : {result.n_trades / max(result.years, 1e-9):.1f}")
    print(f"Win rate               : {YELLOW}{result.win_rate:.1f}%{RESET}")
    print(f"Average win            : {GREEN}+{result.avg_win_pct:.2f}%{RESET}")
    print(f"Average loss           : {RED}{result.avg_loss_pct:.2f}%{RESET}")
    pf_s = f"{result.profit_factor:.2f}" if result.profit_factor is not None else "—"
    print(f"Profit factor          : {YELLOW}{pf_s}{RESET}")
    print(sep)


def save_chart(result: BacktestResult, path: str = "bnf_backtest.png") -> None:
    if not HAS_MPL:
        print(f"{YELLOW}matplotlib not installed; skipping chart.{RESET}")
        return

    dates = [d for d, _ in result.equity_curve]
    eq = [v for _, v in result.equity_curve]
    bnh = [v for _, v in result.bnh_curve]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(dates, eq, label="Strategy")
    ax.plot(dates, bnh, label="Buy & Hold", linestyle="--")
    ax.set_title("BNF Strategy vs Buy & Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    ax.legend()

    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BNF backtester using historical S&P 500 constituent text file directly"
    )

    p.add_argument(
        "--constituent-file",
        type=str,
        default=r"C:\Users\24GHi\OneDrive\Desktop\Quant Trading Strategies\S&P500_ConstituentData_1996_2026.py",
        help="Path to historical constituent text file",
    )
    p.add_argument("--start-date", type=str, default=None, help="Backtest start date YYYY-MM-DD")
    p.add_argument("--end-date", type=str, default=None, help="Backtest end date YYYY-MM-DD")
    p.add_argument("--years", type=float, default=10.0, help="Fallback trailing years if start/end not provided")
    p.add_argument("--stocks", type=int, default=200, help="Top volatile names to keep")
    p.add_argument("--vol-lookback", type=int, default=63)
    p.add_argument("--rebalance-every", type=int, default=5)

    p.add_argument("--ma", type=int, default=25)
    p.add_argument("--dev", type=float, default=0.05)
    p.add_argument("--rsi-per", type=int, default=14)
    p.add_argument("--rsi-lvl", type=float, default=50.0)
    p.add_argument("--no-rsi", action="store_true")
    p.add_argument("--bb", action="store_true")
    p.add_argument("--vol", action="store_true")
    p.add_argument("--vol-mult", type=float, default=1.5)
    p.add_argument("--no-rev", action="store_true")

    p.add_argument("--sl", type=float, default=0.04)
    p.add_argument("--tp", type=float, default=0.6)
    p.add_argument("--hold", type=int, default=3)
    p.add_argument("--capital", type=float, default=100_000.0)

    p.add_argument("--kelly-lookback", type=int, default=30)
    p.add_argument("--min-kelly-trades", type=int, default=10)
    p.add_argument("--max-kelly-frac", type=float, default=0.25)
    p.add_argument("--half-kelly-mult", type=float, default=0.50)
    p.add_argument("--fallback-risk", type=float, default=0.02)

    p.add_argument("--no-chart", action="store_true")
    p.add_argument("--chart-file", type=str, default="bnf_backtest.png")

    return p.parse_args()

def resolve_backtest_window(start_date_str: Optional[str], end_date_str: Optional[str], years: Optional[float]) -> tuple[dt.date, dt.date]:
    if start_date_str is not None and end_date_str is not None:
        start_date = dt.datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = dt.datetime.strptime(end_date_str, "%Y-%m-%d").date()
    elif years is not None:
        end_date = dt.datetime.now().date()
        start_date = end_date - dt.timedelta(days=int(years * 365.25))
    else:
        # final fallback so bare script execution still works
        end_date = dt.datetime.now().date()
        start_date = end_date - dt.timedelta(days=int(10 * 365.25))

    if start_date >= end_date:
        raise ValueError("start date must be before end date")

    return start_date, end_date

def main() -> None:
    args = parse_args()

    params = BNFParams(
        ma_period=args.ma,
        dev_threshold=args.dev,
        rsi_period=args.rsi_per,
        rsi_level=args.rsi_lvl,
        use_rsi=not args.no_rsi,
        use_bollinger=args.bb,
        use_vol_spike=args.vol,
        vol_mult=args.vol_mult,
        require_reversal=not args.no_rev,
        stop_loss=args.sl,
        take_profit=args.tp,
        max_hold_days=args.hold,
        initial_capital=args.capital,
        kelly_lookback=args.kelly_lookback,
        min_kelly_trades=args.min_kelly_trades,
        max_kelly_frac=args.max_kelly_frac,
        half_kelly_mult=args.half_kelly_mult,
        fallback_risk=args.fallback_risk,
    )

    print(f"{CYAN}Parsing historical constituent text file...{RESET}")
    constituents = HistoricalSP500Constituents(args.constituent_file)

    start_date, end_date = resolve_backtest_window(args.start_date, args.end_date, args.years)

    ticker_union = constituents.build_union_for_period(start_date, end_date)
    if not ticker_union:
        raise RuntimeError("No historical tickers found in requested backtest period.")

    print(f"{GREEN}Historical ticker union size: {len(ticker_union)}{RESET}")

    print(f"{CYAN}Downloading OHLCV panel...{RESET}")
    panel = download_panel_for_backtest(
        ticker_union,
        start_date=start_date,
        end_date=end_date,
    )

    print(f"{CYAN}Running backtest...{RESET}")
    result = run_backtest_dynamic_universe(
        panel=panel,
        params=params,
        constituents=constituents,
        top_n=args.stocks,
        vol_lookback_days=args.vol_lookback,
        rebalance_every_days=args.rebalance_every,
        start_date=start_date,
        end_date=end_date,
    )

    print_report(result)

    if not args.no_chart:
        save_chart(result, args.chart_file)
        print(f"{GREEN}Saved chart to {args.chart_file}{RESET}")


if __name__ == "__main__":
    main()