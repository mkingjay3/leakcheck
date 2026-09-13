# leakcheck

A static checker for lookahead bias in pandas/numpy backtest code. It parses
files and looks at the syntax tree; it never imports or runs anything.

A backtest leaks when a decision made at row *i* uses data from rows after
*i*. There are many ways to write that bug. leakcheck looks for four, picked
because each one can be settled by looking at the code rather than by running
it:

- `bfill()` / `backfill()`, which fills a gap with the next known value
- `shift(-n)`, which pulls rows from ahead of the current one
- `rolling(center=True)`, where half the window is future rows
- a whole-series statistic combined arithmetically with the series it came
  from, or used to fill rows, which is the one the project is really about

The fourth looks like this:

```python
df = (df - df.min()) / (df.max() - df.min())
```

`min()` and `max()` run over the whole column and then get broadcast back
across it, so row 0's normalized value depends on a high that might not
arrive for another two years. It reads like ordinary preprocessing. It is
also the bug behind three of the four planted-lookahead exercises that ship
with freqtrade-strategies, which is what I used as ground truth.

The arithmetic is the whole test. `np.mean(r) / np.std(r)` is a scalar
summary and never touches `r` directly, so it stays quiet, and so does
`{"min": s.min(), "p5": s.quantile(.05)}`, which mentions `s` twice and
combines nothing.

## Running it

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

It exits 1 when it finds something and 0 when it doesn't, so it can gate a
commit hook or a CI job. Directory walks skip `venv`, `site-packages`,
`.git` and `node_modules`. Files it can't read or parse say so instead of
being skipped quietly, and `shift` calls whose argument it can't read off
the syntax get counted in the summary, so that blind spot has a number on it.

To silence a line, put `# leakcheck: ignore` on it. The case that needs this
is a machine learning label:

```python
df['target'] = df['close'].shift(-1)  # leakcheck: ignore
```

Looking ahead is the entire point of a label, and it is written exactly the
same way as the bug, so no rule can separate them. A reader has to say so.

## How well it works

Two corpora. The first is five trading repos, 590 files, which is what the
rules were developed against; the second is seven more repos the tool had
never been run on until the rules were finished. Both are listed in
`baseline.txt` and both are gitignored.

I read every finding on both by hand and recorded a verdict and a one-line
reason for each, so the numbers are checkable by someone who didn't run it:

```
                    findings  leaks  verdicts in
development corpus        11     11  audit.json
held-out corpus           38     18  holdout-audit.json

  python3 audit.py report
  AUDIT_CORPUS=holdout AUDIT_FILE=holdout-audit.json python3 audit.py report
```

11 out of 11 on the corpus it was built against and 18 out of 38 on code it
had never seen. The second number is the real one, and the gap between them
is the most useful thing I learned building this. Held out, the low tier is
16 of 27 and the high tier is 2 of 11 - the high tier was perfect on the
development corpus and its main failure, `shift(-1)` building an ML label,
simply did not occur there.

The held-out run also paid for itself twice. Reading its first 89 findings
showed the broadcast rule was counting any mention of the series, including
`s.quantile(.05)` receivers and `len(s)`, which is what the arithmetic
requirement above fixed. And the real leaks it turned up in `ta` were being
caught for the wrong reason, which is why `fill_value=` has its own rule.
Neither problem was visible on the corpus the rules were written against.

For recall there is independent ground truth:
`corpus/freqtrade-strategies/user_data/strategies/lookahead_bias/` ships four
strategies with deliberately planted lookahead bias and a readme naming each
bug. leakcheck catches three. It misses `wtc`, which leaks through an sklearn
`MinMaxScaler` rather than a pandas aggregate.

```
python3 run_tests.py     # 33 cases, precision and recall per tier
```

## What it won't catch

A whole-series statistic parked in a scalar and applied a line later is a
real leak and is not detected:

```python
vol = df['returns'].std()
df['zscore'] = df['returns'] / vol
```

Catching it means knowing whether a subscript target is a DataFrame or a
dict, since `df['x'] = ...` and `d['x'] = ...` are the same syntax. The
version that tried anyway produced 26 false positives on the development
corpus and caught no real leak, so I removed it. Those five shapes are kept
as `tests/cases/gap_full_series_*.py` and `run_tests.py` prints them on every
run, so the 100% above them can't be read as "catches everything".

Also not caught, all seen in the held-out census: leaks through an sklearn
scaler or another library call, and bounding that happens inside a function
the tool can't see through - `_mad` handed to `rolling().apply()` is bounded
by its caller, and nothing inside the helper says so. It also can't tell a
price column from a vector with no time axis, which is why normalizing
portfolio weights or a fixed kernel (`weights / weights.sum()`) still flags,
and it has no idea whether a file is a strategy or a test fixture.

`notes.txt` is the running log - every change, the measurement before and
after it, and why things were left alone. `review-notes.md` is an earlier
writeup, kept for the two conclusions it got wrong.
