# leakcheck

[![tests](https://github.com/mkingjay3/leakcheck/actions/workflows/tests.yml/badge.svg)](https://github.com/mkingjay3/leakcheck/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)

A static checker for lookahead bias in pandas/numpy backtest code. It parses
files and reads the syntax tree; it never imports or runs anything, and it
depends on nothing outside the standard library.

A backtest leaks when a decision made at row *i* uses data from rows after
*i*. There are many ways to write that bug. leakcheck looks for four, picked
because each one can be settled by looking at the code rather than by running
it:

- `bfill()` / `backfill()`, which fills a gap with the next known value
- `shift(-n)`, which pulls rows from ahead of the current one
- `rolling(center=True)`, where half the window is future rows
- a whole-series statistic put back into the series it came from, which is
  the one the project is really about

The fourth looks like this:

```python
df = (df - df.min()) / (df.max() - df.min())
```

`min()` and `max()` run over the whole column and then get broadcast back
across it, so row 0's normalized value depends on a high that might not
arrive for another two years. It reads like ordinary preprocessing. It is
also the bug behind three of the four planted-lookahead exercises that ship
with freqtrade-strategies, which is what I used as ground truth.

Arithmetic is the test. `np.mean(r) / np.std(r)` is a scalar summary and
never touches `r` directly, so it stays quiet, and so does
`{"min": s.min(), "p5": s.quantile(.05)}`, which mentions `s` twice and
combines nothing. Filling counts too, since
`shift(1, fill_value=close.mean())` writes one number from the whole series
into particular rows.

## Installing it

There is nothing to install. It is one file, it imports only the standard
library, and it runs on Python 3.9 or newer.

```
git clone https://github.com/mkingjay3/leakcheck
python3 leakcheck/finder.py <file_or_directory> ...
```

## Running it

```
$ python3 finder.py sample.py
sample.py:7:27: high  backward-fill
   7 | df['close'] = df['close'].bfill()
     |                           ^^^^^^^ backward fill pulls future values into earlier rows
sample.py:14:32: high  negative-shift
  14 | df['next_close'] = df['close'].shift(-1)
     |                                ^^^^^^^^^ shift(-1) pulls 1 future row backward
sample.py:24:29: low  whole-series-stat
  24 | df['norm'] = (close - close.min()) / (close.max() - close.min())
     |                             ^^^^^ min() over the whole series is combined back into that series
sample.py:24:45: low  whole-series-stat
  24 | df['norm'] = (close - close.min()) / (close.max() - close.min())
     |                                             ^^^^^ max() over the whole series is combined back into that series

scanned 1 file in 0.00s

  2 high      2 low
  2 shift calls, 0 unresolvable (0%)
```

The tier is coloured when the output is a terminal and left plain when it is
redirected, so a piped or logged run has no escape codes in it.

| flag | |
|---|---|
| `--format {text,json,sarif}` | carets for a human, flat json for a script, SARIF 2.1.0 for GitHub |
| `--min-confidence {high,low}` | lowest tier to report; `high` drops the low tier from the output *and* the exit code |
| `--exclude PATTERN` | skip paths matching a glob; repeatable |
| `--quiet` | drop the summary, keep the findings |

It exits 1 when it finds something and 0 when it doesn't, so it can gate a
commit hook or a CI job:

```
python3 finder.py --min-confidence high --quiet src/ || echo "blocked"
```

Directory walks skip `venv`, `site-packages`, `.git` and `node_modules`.
Files it can't read, parse, or walk say so on stderr instead of being skipped
quietly, and the summary counts them: `scanned 589 of 590 files` means one
file was pointed at and never analysed. `shift` calls whose argument can't be
read off the syntax are counted too, so that blind spot has a number on it.

### GitHub code scanning

`--format sarif` emits SARIF 2.1.0, which GitHub reads natively. Upload it
and every finding shows as an inline annotation on the pull request and in
the repository's Security tab:

```yaml
- run: python3 finder.py --format sarif --quiet . > leakcheck.sarif || true
- uses: github/codeql-action/upload-sarif@v3
  with:
    sarif_file: leakcheck.sarif
```

The `|| true` is deliberate: the findings exit code would otherwise skip the
upload step that exists to show those findings. High-confidence findings are
emitted at SARIF level `error` and low-confidence ones at `note`, so the tier
that is allowed to be wrong does not fail a pull request by default. Each
result carries a fingerprint of its rule and source line, so a finding stays
the same finding when edits above it move its line number.

The four rule ids (`backward-fill`, `negative-shift`, `centered-window`,
`whole-series-stat`) are what a consumer groups and suppresses by, and
`run_tests.py` checks the emitted document parses and that its columns
actually select the flagged call, because malformed SARIF uploads exactly as
quietly as valid SARIF and then shows nothing.

To silence a line, put `# leakcheck: ignore` on it. The case that needs this
is a machine learning label:

```python
df['target'] = df['close'].shift(-1)  # leakcheck: ignore
```

Looking ahead is the entire point of a label, and it is written exactly the
same way as the bug, so no rule can separate them. A reader has to say so.

## How well it works

Three corpora, nineteen repos, 1663 files. The first is what the rules were
developed against. The other two were never run until the rules were
finished: one of libraries, one of strategies, split that way because
"unseen code" and "library code" had been confounded in the first held-out
run.

Every finding in all three has been read by hand and given a verdict and a
one-line reason, so the numbers are checkable by someone who didn't run them:

```
                 files   high   low   leaks   verdicts in
corpus             590      2     9   11/11   audit.json
holdout            928     11    25   18/36   holdout-audit.json
strategies         145      0     1    0/1    strategies-audit.json
                                      29/48

python3 audit.py report
AUDIT_CORPUS=holdout AUDIT_FILE=holdout-audit.json python3 audit.py report
```

11 of 11 on the corpus it was built against, and 18 of 37 on code it had
never seen. The second kind of number is the real one, and the gap between
them is the most useful thing I learned building this.

Recall is measured separately, by injecting leaks into real code rather than
by writing more test cases:

```
python3 recall.py        # 153/153 injected leaks caught
```

`recall.py` takes real call sites out of the corpora - a positive `shift`, an
`ffill`, a `rolling` window, a windowed aggregate sitting next to its own
series - mutates each into something that is definitely a leak, and checks
the tool reports it. A miss there is a false negative in code nobody wrote
for this project. It found three, all the same shape
(`returns.mean() / returns.rolling(20).std()`), now fixed and held by a case.

There is independent ground truth too:
`corpus/freqtrade-strategies/user_data/strategies/lookahead_bias/` ships four
strategies with deliberately planted lookahead bias and a readme naming each
bug. leakcheck catches three. It misses `wtc`, which leaks through an sklearn
`MinMaxScaler` rather than a pandas aggregate.

```
python3 run_tests.py     # 37 cases, precision and recall per tier
```

On those 37 cases it is 100% precision and 100% recall, 15 findings, no false
positives and no false negatives. That number is the weakest one on this page
and it is worth saying why: the cases were written to pin the rules down, so
the rules pass them by construction. The corpus verdicts above and the
injected-leak recall are the numbers that were allowed to come out wrong, and
one of them did.

### Reproducing the numbers

The corpora are gitignored clones, but every repo is pinned to an exact
commit in `corpora.json`, along with the finding counts that commit produces:

```
python3 fetch_corpora.py            # clone all nineteen at their pinned commits
python3 fetch_corpora.py --check    # fail if a commit or a count has drifted
```

Without the pin, "590 files, 11 findings" is a claim nobody can check,
including me in a month.

### What the held-out corpora actually bought

Each one broke something the previous corpus had no way to reveal.

The library corpus showed the broadcast rule was counting *any* mention of
the series, so `s.quantile(.05)` receivers and `len(s)` both read as
broadcasts; quantstats alone had 24 of those. It also showed the real leaks
in `ta` were being caught for the wrong reason, which is why `fill_value=`
has its own rule now.

The strategy corpus crashed the tool outright - `RecursionError` on eight
NostalgiaForInfinity files, one of them 79k lines. Then it produced 52
high-confidence findings, every one `rolling(center=True)`, a pattern that
had fired zero times in the previous 1518 files. All 52 were the same correct
idiom: a centered window of 5 followed by `.shift(2)`, where the shift is
exactly the compensation. A pattern can lie dormant across two corpora and
then be the entire output of a third.

## Limitations

Three separate things are wrong with this tool, and all three have numbers.

**It cannot read 19% of the `shift` calls it sees.** On the development
corpus that is 47 of 246. A shift whose amount is a parameter, a computed
expression, or a name assigned more than once is neither cleared nor
flagged; it is counted, and the count is printed at the end of every run.
Those 47 calls are not evidence of anything either way, which is the point of
printing the number next to the findings rather than only the findings.

**Its false positives cluster into six shapes.** All 47 non-leaks across the
three corpora were read by hand and given a shape, so the failure modes are
countable rather than anecdotal:

```
15  post-run reporting     a stat computed after the backtest, for a chart or
                           a summary table, never fed back into a decision
13  no time axis           weights/weights.sum(), a fixed sine kernel, a mean
                           across option contracts at one instant, a Q-table
 9  bounded out of sight   already sliced to a training window, or bounded by
                           the caller: _mad handed to rolling().apply()
 4  fixtures and synthetic data    a price frame built for a unit test
 3  index metadata         min() of a date index, used to truncate a range
 3  deliberate lookahead   shift(-48) building a supervised label
```

The common thread in the first two is that the tool reads syntax and these
distinctions are semantic. `weights / weights.sum()` and
`close / close.mean()` are the same three tokens; one is a portfolio
normalization with no time axis and the other is a leak. Nothing in the
syntax separates them, so this rule cannot get to zero false positives
without also missing real leaks, and the low tier is where that cost is
parked.

**It misses whole classes of leak.** A whole-series statistic parked in a
scalar and applied a line later is a real leak and is not detected:

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

Also not caught, all of it seen in a census: leaks through an sklearn scaler
or another library call, and bounding that happens inside a function the tool
can't see through. And it has no idea whether a file is a strategy or a test
fixture, which is most of why the high tier reads 2 of 11 on held-out code.

None of this is a reason not to run it. It is a reason to read the summary
line rather than only the exit code.

## The files

```
finder.py          the checker
run_tests.py       case suite, with the known gaps printed every run
recall.py          injects leaks into real code and checks they're caught
audit.py           record a hand verdict on every finding in a corpus
fetch_corpora.py   clone the corpora at their pinned commits
corpora.json       the pins, and the counts they produce
notes.txt          the running log: every change, the measurement before and
                   after it, and why things were left alone
baseline.txt       the numbers, per corpus, with dates
review-notes.md    an earlier writeup, kept for the two conclusions it got wrong

.github/workflows/tests.yml
                   the case suite on 3.9/3.11/3.12, a SARIF upload to the
                   Security tab, and the corpora recheck on demand
```

`finder.py` is the whole checker: the rules, the reporting and the CLI in one
file. Every rule is declared in one table at the top that `visit_Call`
dispatches through, so adding a rule is adding an entry there and the method
it names, and the full list of rules can be read in one place rather than
reconstructed from control flow.
