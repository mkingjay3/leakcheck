"""What each kind of lookahead bias is worth, in Sharpe ratio.

finder.py says a line is a leak. This says what believing that line would
have cost you. It runs one honest strategy and five variants of it, each
differing from the control by exactly one line, over ten years of daily SPY
closes pinned in the development corpus, and reports the backtest each one
produces beside the finding leakcheck reports for it.

The strategies in cost_cases/ are ordinary pandas-shaped code. finder.py
reads them statically, as it reads anything; cost.py executes them against
the standard-library Series in cost_cases/series.py, so the flagged line and
the inflated Sharpe come from the same source text and the repository still
needs nothing installed.

    python3 cost.py                 the table
    python3 cost.py --svg           also write docs/cost.svg
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import random
import statistics
import sys

import finder

CASES_DIR = "cost_cases"
PRICES = "corpus/algorithmic-trading-with-python/data/SPY.csv"
CHART = os.path.join("docs", "cost.svg")

TRADING_DAYS = 252

# The same range reading supports two opposite strategies, and a leak does not
# have to help either of them. Running both is the difference between
# measuring what lookahead bias does and picking the direction that flatters
# the point: in the freqtrade exercises this is modelled on, the comparison
# direction is itself a tuned parameter, so neither is the "real" one.
ENTRY_RULES = [
    ("dip-buying", 0.30, lambda depth, level: depth < level),
    ("breakout", 0.70, lambda depth, level: depth > level),
]

# A feed with no holes never exercises a fill, so bars are dropped before the
# strategies see them. Seeded, so the table is the same on every machine.
GAP_RATE = 0.03
GAP_SEED = 7

# control first, then in the order the rules appear in finder.RULES
ORDER = [
    "clean",
    "leak_backward_fill",
    "leak_negative_shift",
    "leak_centered_window",
    "leak_whole_series_stat",
    "gap_whole_series_scalar",
]

COLORS = {
    "clean": "#3d9970",
    "leak_backward_fill": "#ff851b",
    "leak_negative_shift": "#e0384f",
    "leak_centered_window": "#9b59b6",
    "leak_whole_series_stat": "#2b90d9",
    "gap_whole_series_scalar": "#00b3b3",
}


def load_closes(path: str) -> tuple[list[str], list[float]]:
    """Dates and closing prices, oldest first."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row["date"] for row in rows], [float(row["close"]) for row in rows]


def punch_gaps(closes: list[float]) -> list[float | None]:
    """The same prices with a seeded ~3% of bars missing."""
    rng = random.Random(GAP_SEED)
    return [None if rng.random() < GAP_RATE else price for price in closes]


def load_strategy(name: str):
    path = os.path.join(CASES_DIR, name + ".py")
    spec = importlib.util.spec_from_file_location("cost_case_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def backtest(positions, returns: list[float | None]) -> list[float]:
    """Daily strategy returns from a position series and the true returns.

    The position is lagged one bar, so a signal produced on day i is only
    held from day i+1. That lag is the engine refusing to look ahead on the
    strategy's behalf: whatever excess return a variant shows has to come
    from its own leaked line, not from this function.
    """
    held = positions.shift(1)
    daily = []
    for position, ret in zip(held.values, returns):
        daily.append(ret if position and ret is not None else 0.0)
    return daily


def equity_curve(daily: list[float]) -> list[float]:
    equity = []
    total = 1.0
    for ret in daily:
        total = total * (1.0 + ret)
        equity.append(total)
    return equity


def metrics(daily: list[float], equity: list[float]) -> dict:
    years = len(daily) / TRADING_DAYS
    deviation = statistics.pstdev(daily)
    sharpe = 0.0
    if deviation > 0:
        sharpe = statistics.fmean(daily) / deviation * math.sqrt(TRADING_DAYS)

    peak = equity[0]
    drawdown = 0.0
    for value in equity:
        peak = max(peak, value)
        drawdown = min(drawdown, value / peak - 1.0)

    return {
        "sharpe": sharpe,
        "cagr": equity[-1] ** (1.0 / years) - 1.0,
        "drawdown": drawdown,
        "exposure": sum(1 for ret in daily if ret != 0.0) / len(daily),
        "final": equity[-1],
    }


def leakcheck_verdict(name: str) -> str:
    """What finder.py says about this strategy file, as one short phrase."""
    found = finder.analyze_file(os.path.join(CASES_DIR, name + ".py"), quiet=True)
    if found is None or not found.findings:
        return "-"

    first = finder.sort_findings(found.findings)[0]
    return f"{first.rule} L{first.line} ({finder.describe_reach(first.reach)})"


def render_svg(dates: list[str], panels: dict[str, dict[str, list[float]]],
               path: str) -> None:
    """Equity curves as a hand-built SVG, one stacked panel per entry rule.

    Written out directly rather than through a plotting library, so the chart
    in README.md is produced by the same dependency-free repository it
    documents. Colours are mid-tones and the background is left transparent,
    so it reads on a light or a dark page.
    """
    width = 900
    left, right, top, bottom = 58, 214, 30, 30
    panel_height = 240
    gap = 34
    height = top + len(panels) * panel_height + gap * (len(panels) - 1) + bottom
    plot_width = width - left - right
    step = max(1, len(dates) // 700)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="system-ui,sans-serif">',
        '<style>text{fill:#888}.grid{stroke:#8884;stroke-width:1}'
        '.title{fill:#aaa;font-weight:600}</style>',
    ]

    for panel, (label, curves) in enumerate(panels.items()):
        origin = top + panel * (panel_height + gap)
        values = [value for curve in curves.values() for value in curve]
        low, high = math.log(min(values)), math.log(max(values))
        span = high - low or 1.0

        def x_at(index):
            return left + plot_width * index / max(len(dates) - 1, 1)

        def y_at(value, origin=origin, low=low, span=span):
            return origin + panel_height * (1 - (math.log(value) - low) / span)

        parts.append(f'<text x="{left}" y="{origin - 9}" font-size="12" '
                     f'class="title">{label}</text>')

        for multiple in [0.2, 0.5, 1, 2, 3, 5, 10]:
            if not min(values) <= multiple <= max(values):
                continue
            y = y_at(multiple)
            parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" '
                         f'x2="{left + plot_width}" y2="{y:.1f}"/>')
            parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="10" '
                         f'text-anchor="end">{multiple:g}x</text>')

        seen = set()
        for index in range(0, len(dates), step):
            year = dates[index][:4]
            if year in seen or int(year) % 2:
                continue
            seen.add(year)
            x = x_at(index)
            parts.append(f'<line class="grid" x1="{x:.1f}" y1="{origin}" '
                         f'x2="{x:.1f}" y2="{origin + panel_height}"/>')
            parts.append(f'<text x="{x:.1f}" y="{origin + panel_height + 15}" '
                         f'font-size="10" text-anchor="middle">{year}</text>')

        for position, name in enumerate(ORDER):
            curve = curves[name]
            points = " ".join(f"{x_at(i):.1f},{y_at(curve[i]):.1f}"
                              for i in range(0, len(curve), step))
            thickness = 2.4 if name == "clean" else 1.4
            dash = ' stroke-dasharray="4 3"' if name.startswith("gap_") else ""
            parts.append(f'<polyline fill="none" stroke="{COLORS[name]}" '
                         f'stroke-width="{thickness}" points="{points}"{dash}/>')

            y = origin + 12 + position * 18
            parts.append(f'<line x1="{left + plot_width + 14}" y1="{y}" '
                         f'x2="{left + plot_width + 34}" y2="{y}" '
                         f'stroke="{COLORS[name]}" stroke-width="{thickness}"{dash}/>')
            parts.append(f'<text x="{left + plot_width + 40}" y="{y + 4}" '
                         f'font-size="10">{name} {curve[-1]:.2f}x</text>')

    parts.append(f'<text x="{left}" y="{height - 8}" font-size="10">'
                 f'SPY {dates[0][:4]}-{dates[-1][:4]}, growth of 1 unit, log scale. '
                 f'Each variant differs from clean by one line; the solid green '
                 f'line is the leak-free control.</text>')
    parts.append("</svg>")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cost.py",
        description="Measure what each class of lookahead bias is worth.")
    parser.add_argument("--svg", action="store_true",
                        help=f"also write the equity curves to {CHART}")
    args = parser.parse_args(argv)

    if not os.path.exists(PRICES):
        print(f"missing {PRICES}", file=sys.stderr)
        print("run: python3 fetch_corpora.py", file=sys.stderr)
        return 2

    sys.path.insert(0, CASES_DIR)
    from series import Series

    dates, closes = load_closes(PRICES)
    feed = Series(punch_gaps(closes))
    returns = Series(closes).pct_change().values

    depths = {name: load_strategy(name).depth(feed) for name in ORDER}
    verdicts = {name: leakcheck_verdict(name) for name in ORDER}

    print(f"{len(dates)} daily bars, {dates[0]} to {dates[-1]}, "
          f"{round(GAP_RATE * 100)}% of bars missing from the feed")
    print(f"every variant differs from clean.py by one line; the entry rule is "
          f"identical across a block")

    panels = {}
    for label, level, decide in ENTRY_RULES:
        curves = {}
        rows = []
        for name in ORDER:
            daily = backtest(decide(depths[name], level), returns)
            curve = equity_curve(daily)
            curves[name] = curve
            rows.append((name, metrics(daily, curve), verdicts[name]))
        panels[label] = curves

        baseline = rows[0][1]["sharpe"]
        print()
        print(f"{label}  (hold while depth "
              f"{'<' if label == 'dip-buying' else '>'} {level})")
        print(f"  {'strategy':<26}{'leakcheck':<38}{'Sharpe':>7}{'CAGR':>8}"
              f"{'maxDD':>8}{'final':>8}{'vs clean':>10}")
        print("  " + "-" * 105)
        for name, result, verdict in rows:
            ratio = "-"
            if name != "clean" and baseline > 0:
                ratio = f"{result['sharpe'] / baseline:+.1f}x"
            print(f"  {name:<26}{verdict:<38}{result['sharpe']:>7.2f}"
                  f"{result['cagr'] * 100:>7.1f}%{result['drawdown'] * 100:>7.1f}%"
                  f"{result['final']:>7.1f}x{ratio:>10}")

    if args.svg:
        render_svg(dates, panels, CHART)
        print()
        print(f"wrote {CHART}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
