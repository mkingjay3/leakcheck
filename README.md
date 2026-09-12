# leakcheck

Finds lookahead bias in pandas/numpy backtest code by reading the AST. No
imports, no execution — it parses files and looks for four specific shapes.

Status: working, narrow, and measured. Not a general leak detector, and not
trying to become one — see [Known gaps](#known-gaps).

## The problem it solves

A backtest leaks when a decision made at row *i* uses data from rows after
*i*. There are a lot of ways to do that. leakcheck looks for four, chosen
because each one is decidable from syntax alone:

| | pattern | why it leaks |
|---|---|---|
| high | `bfill()` / `backfill()` | fills a gap with the next known value, which is in the future |
| high | `shift(-n)` | pulls rows from ahead of the current one |
| high | `rolling(center=True)` | centres the window, so half of it is future rows |
| low | a whole-series statistic combined back into its own series | every row's value depends on every row, including later ones |

That last one is the interesting case and the reason the project exists:

```python
df = (df - df.min()) / (df.max() - df.min())
```

`min()` and `max()` are taken over the whole column, then broadcast back
across it. Row 0's normalized value depends on a high that may not happen
for another two years. It reads as ordinary preprocessing and it is the
bug that the freqtrade repo's own planted-bias exercises are built on.

The tool flags an aggregate **only** when the series it summarises also
appears on its own in the same expression. `np.mean(r) / np.std(r)` — a
scalar summary — never touches `r` directly, so it stays quiet.

## Usage

```
python3 finder.py <file_or_directory> ...
python3 finder.py --quiet corpus        # summary only
```

```
$ python3 finder.py sample.py
sample.py:7 [high] bfill backward fill pulls future values into earlier rows
sample.py:14 [high] shift shift(-1) pulls 1 future rows backward
sample.py:24 [low] min min() over the whole series is combined back into that series
sample.py:24 [low] max max() over the whole series is combined back into that series

2 high-confidence findings
2 low-confidence findings
2 shift calls, 0 unreadable (0%)
```

Directory walks skip `venv`, `site-packages`, `.git` and `node_modules`.
Files that can't be read or parsed say so rather than being skipped
silently. `shift` calls whose argument can't be read off the syntax are
counted in the summary, so the blind spot has a number attached.

## What it does not flag

Suppressed on purpose, each with a test case:

- anything already bounded — `rolling`, `expanding`, `ewm`, a literal
  slice like `close[-30:]`, or a name assigned once from one of those
- `groupby(...).sum()` and `mean(axis=1)` — aggregates across categories
  or across columns within a row, not down the time axis
- scalar summaries: Sharpe ratios, accuracy figures, anything computed
  from a series without being pushed back into it

## Measurements

Corpus: 5 trading repos, 590 `.py` files (see `baseline.txt` for the list;
`corpus/` is gitignored). 11 findings — 2 high, 9 low.

Every one of the 11 has been read by hand and recorded in `audit.json`
(`python3 audit.py report`): 11 leaks, 0 false positives. That number is
not independent — this is the corpus the rule was developed against.

The independent check is
`corpus/freqtrade-strategies/user_data/strategies/lookahead_bias/`, four
strategies shipped with deliberately planted lookahead bias and a readme
naming each bug. leakcheck catches 3 of the 4. It misses `wtc`, which
leaks through an sklearn `MinMaxScaler` rather than a pandas aggregate.

```
python3 run_tests.py     # 31 cases, precision and recall per tier
python3 audit.py report  # hand-verified verdicts on the corpus findings
```

## Known gaps

A whole-series statistic parked in a scalar and applied a line later is a
real leak and is **not** detected:

```python
vol = df['returns'].std()
df['zscore'] = df['returns'] / vol
```

Detecting it needs to know whether a subscript target is a DataFrame or a
dict — `df['x'] = ...` and `d['x'] = ...` are the same syntax. The version
that tried anyway produced 26 false positives on the corpus and caught no
real leak, so it was removed on 2026/09/12.

Those five shapes live in `tests/cases/gap_full_series_*.py` and
`run_tests.py` prints them on every run, so the 100% above them can't be
mistaken for "catches everything". Also out of scope: leaks through
scalers or other library calls, and bounding that happens inside a
function call the tool can't see through.

`notes.txt` is the running log — every change, the measurement before and
after, and the reasoning for what was left alone. `review-notes.md` is an
earlier session writeup, kept for the two conclusions it got wrong.
