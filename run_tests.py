

import json
import os
import sys

from finder import analyze_file

CASES_DIR = "tests/cases"
EXPECTED_FILE = "tests/expected.json"
CONFIDENCE_TIERS = ["high", "low"]


def load_expected():
    with open(EXPECTED_FILE) as f:
        expected = json.load(f)
    real_cases = {}
    for filename, findings in expected.items():
        if not filename.startswith("_"):
            real_cases[filename] = findings
    return real_cases


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
    expected = load_expected()
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
    print_totals_and_precision(true_positive, false_positive, false_negative)

    sys.exit(1 if any_case_failed else 0)


def print_totals_and_precision(true_positive, false_positive, false_negative):
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
        print(f"recall    {round(overall_recall * 100)}%")

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


if __name__ == "__main__":
    main()
