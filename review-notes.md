

## What changed in finder.py

1. **Full-series stats** — `.mean()`, `.std()`, `.max()`, `.min()`,
   `.sum()`, `.median()`, `.var()` flagged as `low` confidence when
   called directly on a column rather than after `.rolling()`,
   `.expanding()`, or `.ewm()`. One number computed over the whole
   series and used as a threshold means early rows get judged against
   data from years later. First pattern where `confidence` does
   something other than say "high".
2. **Sorting** — findings print high-confidence first, not by line
   number.
3. **Directory input** — `finder.py <dir>` walks it with `os.walk`,
   skipping `venv`, `site-packages`, `.git`, `node_modules`. No more
   glob patterns that miss nested folders.
4. **`--quiet`** — suppresses the per-line output, prints only the
   summary. Read/parse errors still print in quiet mode; a silently
   skipped file seemed worse than a slightly noisy `--quiet`.

## First baseline

Corpus grew from 3 repos to 5 (see `baseline.txt` for the list and
full counts). Command: `python3 finder.py corpus`.

590 files walked, 160 findings (4 high, 156 low). Counts by pattern
are in `baseline.txt`, not duplicated here — that file is meant to be
the number to diff against next time, this file is the judgment calls.

## Manual review — opened 7 flagged lines by hand

**5 low-confidence (the new pattern), sampled across the output:**

| file:line | pattern | verdict |
|---|---|---|
| `Awesome Oscillator backtest.py:285` | `std` | **false positive** — `np.std(...)` inside a `stats()` function computing Sharpe ratios *after* the backtest, for reporting, not fed back into earlier decisions |
| `Oil Money RUB.py:182` | `std` | **false positive** — code explicitly splits `train`/`test` per year first; `np.std(train['rub'] - ...)` runs only on the train slice, not the whole series |
| `dynamic_breakout_ii.py:48` | `std` | **false positive, and instructive** — `np.std(df_hist.Close[-lookback_days:])` is windowed, just via manual slicing instead of `.rolling()`, which the detector doesn't recognize as windowed |
| `deep_learning.py:650` | `sum` | **false positive** — `np.sum(...)` computing prediction accuracy on train/valid/test splits, unrelated to price data or trade timing |
| `portfolio.py:335` | `mean` | **false positive** — `average_active_trades` property, a post-hoc portfolio statistic, not a trading input |

**0 out of 5 were real leaks.** All 5 are syntactically correct matches
(a stat method called without a preceding `.rolling()`/`.expanding()`)
but wrong in intent. Two false-positive shapes showed up, both inside
the first five samples:
- **post-hoc reporting** — Sharpe ratio, accuracy, portfolio stats
  computed after the fact, never fed back into an earlier decision
  (3 of 5)
- **windowed by hand** — bounded to a slice or a train split the
  detector can't see, because it only recognizes `.rolling()` /
  `.expanding()` / `.ewm()` as "already windowed" (2 of 5)

Low confidence earned — a `low` label with a 0/5 hit rate on real
leaks is doing its job, not failing at it.

**2 high-confidence `shift` findings, new this run, both in
`corpus/QuantResearch/backtest/turtle.py:62-63`:**

```python
abs((df_hist.Close.iloc[-15:].shift(-1) - df_hist.High.iloc[-15:-1]).dropna())
```

Also false positives, a third shape — `df_hist` is already bounded to
known history at decision time, so `.shift(-1)` reindexes *within* an
already-past 15-row slice rather than pulling in data that didn't
exist yet. (Possibly a real bug in their ATR formula — next-close
instead of prev-close — but that's a formula error, not a temporal
leak.) The two `shift(-1)` findings from `Shooting Star backtest.py`
found in the earlier run are unaffected by this and still stand as
confirmed real.

Three failure shapes now on record for when the false-positive rate
comes up again: bounded-by-slicing (shift), post-hoc reporting
(stats), and windowed-by-hand (stats).

