import json
import os
from finder import analyze_file


CASES_DIR = "tests/cases"
EXPECTED_FILE = "tests/expected.json"


def load_expected():
    with open(EXPECTED_FILE) as fh:
        return json.load(fh)


def run_one(filename):
    """Returns (has_high, has_low) for one test file."""
    path = os.path.join(CASES_DIR, filename)
    finder = analyze_file(path, quiet=True)

    if finder is None:
        return None, None

    confs = {f.conf for f in finder.findings}
    return "high" in confs, "low" in confs


def main():
    expected = load_expected()

    # counters for high-confidence findings
    true_pos = 0
    false_pos = 0
    false_neg = 0

    failures = []

    for filename, want in sorted(expected.items()):
        got_high, got_low = run_one(filename)

        if got_high is None:
            failures.append(f"{filename}: could not analyze")
            continue

        want_high = want["expect_high"]

        if want_high and got_high:
            true_pos += 1
        elif want_high and not got_high:
            false_neg += 1
            failures.append(f"{filename}: expected a high finding, got none")
        elif not want_high and got_high:
            false_pos += 1
            failures.append(f"{filename}: flagged high, shouldn't have")

        if "expect_low" in want and want["expect_low"] != got_low:
            failures.append(
                f"{filename}: low finding expected={want['expect_low']} got={got_low}"
            )

    print(f"{len(expected)} cases")
    print(f"true positives  {true_pos}")
    print(f"false positives {false_pos}")
    print(f"false negatives {false_neg}")

    if true_pos + false_pos > 0:
        precision = true_pos / (true_pos + false_pos)
        print(f"precision {precision:.0%}")

    if true_pos + false_neg > 0:
        recall = true_pos / (true_pos + false_neg)
        print(f"recall    {recall:.0%}")

    if failures:
        print()
        for line in failures:
            print("  " + line)


if __name__ == "__main__":
    main()
