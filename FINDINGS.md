# Lookahead bias in public backtest code: prevalence, shapes, and cost

A static checker for lookahead bias was built, then pointed at 1,663 files of
public trading code across nineteen repositories. Every finding was read by
hand and given a verdict. Separately, each class of leak the checker knows
about was injected into a working strategy and backtested, to put a number on
what the bug is actually worth.

This document is the result, not the tool. The tool is in `README.md`.

Everything here is reproducible: the corpora are pinned to exact commits in
`corpora.json`, the verdicts are in the audit files, and the cost table comes
from `python3 cost.py`.

---

## 1. What was measured

| | |
|---|---|
| Corpora | 3 (development, held-out libraries, held-out strategies) |
| Repositories | 19, each pinned to a commit |
| Files | 1,663 |
| Findings from the current rules | 48 |
| Confirmed leaks among them | 29 |
| Verdicts recorded in total | 76, every one hand-read with a written reason |
| Of those, non-leaks | 47 |
| Injected-leak recall | 153/153 |

The verdict count is larger than the finding count because it accumulated
across versions of the rules: a finding that an earlier version produced and a
later one stopped producing keeps its verdict in the audit file. That is
deliberate, and it is where the false-positive taxonomy in section 3 comes
from — 47 non-leaks is the record of everything the rules ever got wrong, not
just what they still get wrong.

The three corpora were separated deliberately. The first is what the rules
were developed against. The other two were not run until the rules were
finished, and were split into libraries and strategies because the first
held-out run had confounded "unseen code" with "library code" and the
distinction turned out to matter.

## 2. Prevalence

```
                 files   high   low   confirmed leaks
corpus             590      2     9   11/11
holdout            928     11    25   18/36
strategies         145      0     1    0/1
                                      29/48
```

The number that matters is the second one. On the corpus the rules were
developed against, every finding was a real leak. On code the rules had never
seen, fewer than half were. The gap between those two numbers is the single
most useful thing this project produced, and it is the reason the held-out
corpora exist at all.

Two findings sit underneath that:

**A pattern can lie dormant across two corpora and then be the entire output
of a third.** The strategy corpus produced 52 high-confidence findings, every
one of them `rolling(center=True)` — a pattern that had fired zero times in
the preceding 1,518 files. All 52 were the same correct idiom: a centered
window followed by a compensating `.shift(n // 2)`. The rule now clears that
shape, and all 52 went away.

**The first held-out corpus broke the rule it was testing.** The broadcast
rule had been counting any mention of the series, so `s.quantile(.05)`
receivers and `len(s)` both read as broadcasts. quantstats alone contributed
24 of those.

## 3. The shape of a false positive

All 47 non-leaks were given a shape, so the failure modes are countable rather
than anecdotal:

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

The first two families are 60% of the total and share a cause: the checker
reads syntax, and these distinctions are semantic. `weights / weights.sum()`
and `close / close.mean()` are the same three tokens. One is a portfolio
normalisation with no time axis; the other is a leak. Nothing in the syntax
separates them.

That is a ceiling, not a bug to be fixed. A syntactic rule for this shape
cannot reach zero false positives without also missing real leaks, and the
low-confidence tier is where that cost is deliberately parked.

## 4. What a leak is worth

One honest strategy, five variants each differing by exactly one line, ten
years of daily SPY closes, and two opposite entry rules. The entry rule is
applied outside the strategy files, so no variant is credited or blamed for
the direction it happens to suit. Full table in `README.md`; the results in
one column each:

```
                           dip-buying   breakout
clean                           0.66       0.52     Sharpe
leak_backward_fill              0.52       0.56
leak_negative_shift            -0.90       1.53
leak_centered_window            1.23      -0.84
leak_whole_series_stat          0.47       0.36
gap_whole_series_scalar         0.47       0.36
```

**A leaked backtest is not an inflated backtest. It is an arbitrary one.**
`shift(-1)` nearly triples the Sharpe of one rule and reverses the sign of the
other. The centered window does exactly the opposite. Same leak, same data,
same decade, opposite conclusions. The comfortable version of this bug is
"it makes your results look better"; what it really does is decouple the
result from the strategy, and which way that lands is luck.

This is also why both directions are reported. In the freqtrade exercises this
demo is modelled on, the comparison direction is itself a tuned parameter, so
there is no "real" direction to pick, and picking one would have been choosing
the number that told the better story.

**Reach does not predict cost.** The checker now reports how far into the
future each finding reads. The one-row leak moved the result furthest in both
directions; the two unbounded-reach leaks moved it least. A leak that reaches
one row but reaches it on every row beats a leak that reaches ten years and
only shifts a threshold. The intuitive ordering is wrong, and the reach number
would have led you to it.

**The known gap costs exactly what the caught bug costs.** The checker flags
`(close - close.min()) / (close.max() - close.min())`. It does not flag the
same arithmetic with the statistic bound to a variable first, because deciding
that needs to know whether a subscript target is a DataFrame or a dict. Those
two variants produce identical backtests to the last decimal, in both
directions. The gap is not a rare corner; it is the same bug behind a name.

## 5. What this does not show

One instrument, one decade, one strategy family, two entry rules. The cost
table establishes that these leaks move a result by amounts that dwarf the
effect anyone was trying to measure, and that the direction is unpredictable.
It does not establish a general multiplier, and a different strategy would
reorder it.

The prevalence numbers are drawn from public repositories, which are not a
random sample of trading code: they are the code people chose to publish.

The checker cannot read 19% of the `shift` calls it sees, and those calls are
evidence of nothing in either direction. That number is printed on every run
for the same reason it is printed here.

No head-to-head comparison against a dynamic lookahead checker was run. That
is the obvious next measurement and it is described in `README.md`.

## 6. Reproducing all of it

```
python3 fetch_corpora.py          # clone nineteen repos at pinned commits
python3 fetch_corpora.py --check  # fail if a commit or a count has drifted
python3 run_tests.py              # 37 cases, precision and recall per tier
python3 recall.py                 # 153/153 injected leaks caught
python3 audit.py report           # the hand verdicts, per tier
python3 cost.py --svg             # the cost table and the figure
```
