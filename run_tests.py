import json
import os
import subprocess
import sys

from finder import analyze_file

CASES_DIR = "tests/cases"
EXPECTED_FILE = "tests/expected.json"
CONFIDENCE_TIERS = ["high", "low"]
KNOWN_GAPS_KEY = "_known_gaps"


def load_expected():
    with open(EXPECTED_FILE) as f:
        expected = json.load(f)

    gaps = expected.get(KNOWN_GAPS_KEY, {})
    cases = {}
    for filename, findings in expected.items():
        if not filename.startswith("_"):
            cases[filename] = findings
    return cases, gaps


def findings_as_keyed_dict(findings):
    keyed = {}
    for finding in findings:
        if isinstance(finding, dict):
            line = finding["line"]
            pattern = finding["pattern"]
            conf = finding["conf"]
        else:
            line = finding.line
            pattern = finding.pattern
            conf = finding.conf
        keyed[(line, pattern)] = conf
    return keyed


def run_one_case(filename):
    path = os.path.join(CASES_DIR, filename)
    finder = analyze_file(path, quiet=True)

    if finder is None:
        return None
    return finder.findings


def describe_finding_key(key, conf):
    line, pattern = key
    return f"line {line} [{conf}] {pattern}"


def main():
    expected, known_gaps = load_expected()
    true_positive = {tier: 0 for tier in CONFIDENCE_TIERS}
    false_positive = {tier: 0 for tier in CONFIDENCE_TIERS}
    false_negative = {tier: 0 for tier in CONFIDENCE_TIERS}

    any_case_failed = False

    for filename, expected_findings in sorted(expected.items()):
        actual_findings = run_one_case(filename)

        if actual_findings is None:
            print(f"FAIL {filename}: could not be read or parsed")
            any_case_failed = True
            continue

        actual_keyed = findings_as_keyed_dict(actual_findings)
        expected_keyed = findings_as_keyed_dict(expected_findings)

        actual_keys = set(actual_keyed.keys())
        expected_keys = set(expected_keyed.keys())

        matched_keys = actual_keys & expected_keys
        unexpected_keys = actual_keys - expected_keys   # false positives
        missing_keys = expected_keys - actual_keys      # false negatives

        # a matched finding's confidence is read from what the tool
        # actually produced, since that is the confidence it will be
        # judged on in a real run
        for key in matched_keys:
            conf = actual_keyed[key]
            true_positive[conf] = true_positive[conf] + 1

        for key in unexpected_keys:
            conf = actual_keyed[key]
            false_positive[conf] = false_positive[conf] + 1

        # a missing finding never happened, so there is no "actual"
        # confidence to read; use the confidence expected.json says it
        # should have had
        for key in missing_keys:
            conf = expected_keyed[key]
            false_negative[conf] = false_negative[conf] + 1

        if not unexpected_keys and not missing_keys:
            print(f"pass {filename}")
            continue

        any_case_failed = True
        print(f"FAIL {filename}")
        for key in sorted(unexpected_keys):
            print(f"  unexpected: {describe_finding_key(key, actual_keyed[key])}")
        for key in sorted(missing_keys):
            print(f"  missing:    {describe_finding_key(key, expected_keyed[key])}")

    print()
    print_totals_and_precision(true_positive, false_positive, false_negative, len(expected))
    print()
    print_known_gaps(known_gaps)
    print()
    if not exit_codes_correct():
        any_case_failed = True
    print()
    if not sarif_output_valid():
        any_case_failed = True
    print()
    if not series_shim_correct():
        any_case_failed = True

    sys.exit(1 if any_case_failed else 0)


def print_totals_and_precision(true_positive, false_positive, false_negative, case_count):
    total_tp = sum(true_positive.values())
    total_fp = sum(false_positive.values())
    total_fn = sum(false_negative.values())

    print(f"true positives  {total_tp}")
    print(f"false positives {total_fp}")
    print(f"false negatives {total_fn}")

    if total_tp + total_fp > 0:
        overall_precision = total_tp / (total_tp + total_fp)
        print(f"precision {round(overall_precision * 100)}%")

    if total_tp + total_fn > 0:
        overall_recall = total_tp / (total_tp + total_fn)
        print(f"recall    {round(overall_recall * 100)}%  (over {case_count} cases, "
              f"known gaps excluded - see below)")

    print()
    print("precision by confidence tier:")
    for tier in CONFIDENCE_TIERS:
        tp = true_positive[tier]
        fp = false_positive[tier]

        if tp + fp > 0:
            precision = tp / (tp + fp)
            print(f"  {tier:<5} {round(precision * 100)}%  ({tp} tp, {fp} fp)")
        else:
            print(f"  {tier:<5} n/a  (nothing at this tier was flagged)")


# leaks that are out of scope for this version. They are excluded from the
# numbers above, so print them every run rather than letting a 100% recall
# line stand on its own.
def print_known_gaps(known_gaps):
    print("known gaps (real leaks this version does not detect):")

    now_detected = 0
    for filename, missed in sorted(known_gaps.items()):
        actual = run_one_case(filename)
        described = ", ".join(
            describe_finding_key(key, conf)
            for key, conf in sorted(findings_as_keyed_dict(missed).items())
        )
        print(f"  {filename}  {described}")
        if actual:
            now_detected = now_detected + 1
            print("    now detected - move it back into the expected cases")

    print(f"  {len(known_gaps)} shapes, {now_detected} now detected")


# nothing else here runs finder.py's main(), and a caller gating a commit
# hook on the exit code depends on it
def exit_codes_correct():
    print("cli exit codes:")
    correct = True
    for filename, expected in [("clean_rolling_mean.py", 0), ("leak_bfill.py", 1)]:
        finished = subprocess.run(
            [sys.executable, "finder.py", "--quiet", os.path.join(CASES_DIR, filename)],
            capture_output=True,
        )
        ok = finished.returncode == expected
        correct = correct and ok
        print(f"  {'pass' if ok else 'FAIL'} {filename} exited {finished.returncode}, "
              f"expected {expected}")
    return correct


# A malformed SARIF document uploads to GitHub as silently as a good one and
# then shows nothing, so the shape is checked here rather than discovered from
# an empty Security tab. The column round-trip is the part worth guarding: the
# spec counts from 1 and ends one past the span, and getting either wrong puts
# every annotation on the wrong characters.
def sarif_output_valid():
    print("sarif output:")
    finished = subprocess.run(
        [sys.executable, "finder.py", "--format", "sarif", "--quiet",
         os.path.join(CASES_DIR, "leak_bfill.py")],
        capture_output=True,
        text=True,
    )

    try:
        document = json.loads(finished.stdout)
        run = document["runs"][0]
        declared = {rule["id"] for rule in run["tool"]["driver"]["rules"]}
        result = run["results"][0]
        region = result["locations"][0]["physicalLocation"]["region"]
        snippet = region["snippet"]["text"]
        span = snippet[region["startColumn"] - 1:region["endColumn"] - 1]
    except (ValueError, KeyError, IndexError) as error:
        print(f"  FAIL could not read a result out of it: {error!r}")
        return False

    checks = [
        ("version is 2.1.0", document["version"] == "2.1.0"),
        ("columns are code points", run.get("columnKind") == "unicodeCodePoints"),
        ("rule id is declared", result["ruleId"] in declared),
        ("level is a SARIF level", result["level"] in ("error", "warning", "note")),
        ("uri is repo-relative", not os.path.isabs(
            result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"])),
        ("fingerprint present", bool(result.get("partialFingerprints"))),
        (f"columns select the call, got {span!r}", span == "bfill()"),
    ]

    for description, ok in checks:
        print(f"  {'pass' if ok else 'FAIL'} {description}")
    return all(ok for _, ok in checks)


# Every number cost.py prints rests on this shim behaving like the pandas
# methods the strategies call. If shift or the centered window is off by one,
# the cost table is quietly wrong in a way no other test would notice, so the
# semantics are pinned here against hand-worked expectations.
def series_shim_correct():
    print("series shim:")
    sys.path.insert(0, "cost_cases")
    try:
        from series import Series
    except ImportError as error:
        print(f"  FAIL could not import: {error}")
        return False

    values = Series([1.0, 2.0, None, 4.0, 5.0])
    ramp = Series([1.0, 2.0, 3.0, 4.0, 5.0])

    # the lines below call the very methods finder.py flags, which is the
    # point of them, so they carry the ignore comment a reader would have to
    # add to any deliberate use

    checks = [
        ("shift(1) moves values forward",
         values.shift(1).values == [None, 1.0, 2.0, None, 4.0]),
        ("shift(-1) moves values back",
         values.shift(-1).values == [2.0, None, 4.0, 5.0, None]),  # leakcheck: ignore
        ("ffill carries the last known value",
         values.ffill().values == [1.0, 2.0, 2.0, 4.0, 5.0]),
        ("bfill carries the next known value",
         values.bfill().values == [1.0, 2.0, 4.0, 4.0, 5.0]),  # leakcheck: ignore
        ("min and max skip missing values",
         (values.min(), values.max()) == (1.0, 5.0)),
        ("a trailing window ends on the current row",
         ramp.rolling(3).max().values == [None, None, 3.0, 4.0, 5.0]),
        ("a centered window is the trailing one moved back window // 2",
         ramp.rolling(3, center=True).max().values  # leakcheck: ignore
         == ramp.rolling(3).max().shift(-1).values),  # leakcheck: ignore
        ("a window containing a gap is missing",
         values.rolling(2).max().values == [None, 2.0, None, None, 5.0]),
        ("pct_change is row over previous row",
         ramp.pct_change().values[1] == 1.0),
        ("comparing against a missing value is False",
         (values > 0).values == [True, True, False, True, True]),
    ]

    for description, ok in checks:
        print(f"  {'pass' if ok else 'FAIL'} {description}")
    return all(ok for _, ok in checks)


if __name__ == "__main__":
    main()
