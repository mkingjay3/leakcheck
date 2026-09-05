

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

## The test harness

This session was about building the harness instead of a new detector.
The reason is simple. I measured the tool once by hand, on five low
confidence findings, and got zero real leaks out of five. That is the
most important number in the project so far, and I had no way to check
it again without rereading files one at a time. Every change I made
after that measurement was a claim nobody could verify, including me.

Most of the harness already existed. tests/cases, tests/expected.json,
and a run_tests.py that printed precision and recall. What it did not
do was fold the low confidence tier into those numbers. A low finding
showing up where it should not printed as a line in a failures list,
but never touched precision or recall. So the summary could read one
hundred percent while a known false positive sat quietly below it.
Since the low tier is the one with the zero out of five hit rate, that
is the exact thing a harness should not be able to hide. So most of the
work here was rewriting run_tests.py to track true positives, false
positives, and false negatives separately for each tier, high and low,
and print precision and recall for both.

I also added the third known false positive as its own case file,
tests/cases/fp_shift_bounded_slice.py, matching the turtle.py pattern
where shift(-1) runs on a slice already bounded to known history. It
has an empty expected list, same as the other two false positive
cases, and it fails right now. That is correct. Writing a case for a
bug I already found by hand means the bug cannot quietly stop being
caught. If checkShift changes later and this case starts passing
without anyone touching the windowing logic, I want that to show up in
the test output, not get discovered by rereading corpus output again.

Baseline before any detection logic changed: high tier at 86 percent
precision and 100 percent recall, low tier at 67 percent precision and
100 percent recall, over 21 hand written cases. This number is not the
same claim as zero out of five on the real corpus. The low tier here
has a denominator of six, four true positives and two false positives,
so a couple of cases moving shifts the percentage a lot. The corpus
sample answers what fraction of real output on real trading code is
noise. The case harness answers whether a change to finder.py broke
something already confirmed by hand. Both matter, they are just not
the same measurement, even though both get called precision.

Then I fixed the manual slicing false positive from dynamic_breakout_ii.py,
where np.std(df_hist.Close[-lookback_days:]) gets flagged because the
detector only recognizes rolling, expanding, and ewm as already
windowed, and has no notion that a manual slice does the same job by
hand. checkAggregate used to only look at the method receiver. A call
like series.std() has the series sitting there, but a bare call like
np.std(series) has the receiver as the numpy module name, with the
series sitting in the first argument instead. So np.std wrapped around
anything was invisible to the windowing check even when the argument
was clearly sliced. Now both the receiver and the first positional
argument, when there is one, get checked, and either one being a
rolling call or a literal slice suppresses the finding. Running the
harness again gives low tier precision at 80 percent, up from 67, with
fp_manual_window.py passing and the other two false positives still
correctly failing, since I left post-hoc reporting and the shift
bounded by slicing case alone this round. That delta, 67 to 80 on the
same 21 cases, is the number I could not produce before this session.

I left post-hoc reporting and the shift bounded by slicing case alone
on purpose. Manual slicing was a structural fix, the idea of already
windowed already existed in the code, it just needed to look in the
right place. Post-hoc reporting is a different problem, since it asks
whether a computed statistic ever flows into a comparison or a signal
used later. That is dataflow analysis across variable assignments, not
something readable off one call node, and a shallow pattern match for
it would just be another special case fitted to this corpus. That also
answers the standing question of whether the low confidence tier
should exist. Deleting it throws away four real true positives, and
hiding it behind a flag just moves the noise somewhere the user has to
remember to look. Narrowing it to fire only when the aggregate feeds a
comparison or a signal is the right fix, and it is real work, tracking
whether a value gets compared against or assigned into a later
decision rather than just checking if a stat method ran on something
unbounded. That direction seems right, not just a guess, mainly because
the harness now lets me tell a real improvement from one that only
looks like an improvement. It is a bigger job than this session and
deserves the same before and after treatment as the slicing fix, rather
than getting rushed in at the end.

Small cleanups along the way. self.fpath in LeakFinder was set in
__init__ and never read, since printFindings already takes the path as
its own parameter. Removed the attribute and the constructor argument
that only existed to feed it. Also wrapped the two long lines in
checkShift and checkAggregate, mostly by pulling a nested attribute
access into a short local variable instead of reading it twice inline.

