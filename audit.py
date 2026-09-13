"""Hand-audit every finding finder.py reports on a corpus.

A census, not a sample - the point is to be able to say "I read every one"
instead of "a sample suggested". Verdicts and rationales are written to
audit.json so the claim is checkable by someone who didn't run it.

  python3 audit.py list                    every finding, audited or not
  python3 audit.py show <id>               one finding with source context
  python3 audit.py next                    first unaudited finding
  python3 audit.py record <id> <verdict> <shape> "<rationale>"
  python3 audit.py report                  precision per tier + taxonomy

AUDIT_CORPUS picks the tree to scan (default corpus/), AUDIT_FILE the
verdicts to read and write (default audit.json).

verdicts: leak | not_leak | unsure
shape:    short tag, grouped in the report (e.g. report_frame, dict_write)
"""

import json
import os
import sys

from finder import find_py_files, analyze_file

DEFAULT_AUDIT_FILE = "audit.json"
DEFAULT_CORPUS = "corpus"
CONTEXT_BEFORE = 8
CONTEXT_AFTER = 4

VERDICTS = ["leak", "not_leak", "unsure"]


def collect_findings(path):
    findings = []
    for fpath in find_py_files(path):
        finder = analyze_file(fpath, quiet=True)
        if finder is None:
            continue
        for finding in finder.findings:
            findings.append({
                "file": fpath,
                "line": finding.line,
                "pattern": finding.pattern,
                "conf": finding.conf,
                "msg": finding.msg,
            })
    findings.sort(key=lambda f: (f["file"], f["line"], f["pattern"]))
    return findings


def finding_id(finding):
    return f"{finding['file']}:{finding['line']}:{finding['pattern']}"


# a second corpus needs a second verdict file, so both are overridable:
#   AUDIT_CORPUS=holdout AUDIT_FILE=holdout-audit.json python3 audit.py report
def audit_file():
    return os.environ.get("AUDIT_FILE", DEFAULT_AUDIT_FILE)


def load_audit():
    if not os.path.exists(audit_file()):
        return {}
    with open(audit_file()) as f:
        return json.load(f)


def save_audit(audit):
    with open(audit_file(), "w") as f:
        json.dump(audit, f, indent=2, sort_keys=True)
        f.write("\n")


def read_context(fpath, line):
    try:
        with open(fpath, encoding="utf-8") as f:
            lines = f.readlines()
    except (OSError, UnicodeDecodeError):
        return []

    start = max(0, line - 1 - CONTEXT_BEFORE)
    end = min(len(lines), line + CONTEXT_AFTER)
    return [(n + 1, lines[n].rstrip("\n")) for n in range(start, end)]


def print_finding(finding, audit, index=None, total=None):
    fid = finding_id(finding)
    header = fid
    if index is not None:
        header = f"[{index}/{total}] {header}"
    print(header)
    print(f"  {finding['conf']}: {finding['pattern']}() - {finding['msg']}")
    print()

    for number, text in read_context(finding["file"], finding["line"]):
        marker = ">>" if number == finding["line"] else "  "
        print(f"  {marker} {number:>5}  {text}")

    record = audit.get(fid)
    if record:
        print()
        print(f"  VERDICT: {record['verdict']}  [{record['shape']}]")
        print(f"  {record['rationale']}")
    print()


def cmd_list(findings, audit):
    for index, finding in enumerate(findings, start=1):
        fid = finding_id(finding)
        record = audit.get(fid)
        status = f"{record['verdict']:<9} {record['shape']}" if record else "-- unaudited --"
        print(f"{index:>3}  {finding['conf']:<5} {status:<32} {fid}")

    done = sum(1 for f in findings if finding_id(f) in audit)
    print()
    print(f"{done}/{len(findings)} audited")


def cmd_show(findings, audit, args):
    index = int(args[0])
    finding = findings[index - 1]
    print_finding(finding, audit, index, len(findings))


def cmd_next(findings, audit):
    for index, finding in enumerate(findings, start=1):
        if finding_id(finding) not in audit:
            print_finding(finding, audit, index, len(findings))
            return
    print("all findings audited")


def cmd_record(findings, audit, args):
    index, verdict, shape = int(args[0]), args[1], args[2]
    rationale = args[3]

    if verdict not in VERDICTS:
        print(f"verdict must be one of {VERDICTS}")
        return

    finding = findings[index - 1]
    audit[finding_id(finding)] = {
        "verdict": verdict,
        "shape": shape,
        "rationale": rationale,
        "conf": finding["conf"],
        "pattern": finding["pattern"],
    }
    save_audit(audit)
    print(f"recorded {finding_id(finding)}: {verdict} [{shape}]")


def cmd_report(findings, audit):
    tiers = {}
    shapes = {}
    unaudited = 0

    for finding in findings:
        record = audit.get(finding_id(finding))
        if record is None:
            unaudited = unaudited + 1
            continue

        tier = tiers.setdefault(finding["conf"], {"leak": 0, "not_leak": 0, "unsure": 0})
        tier[record["verdict"]] = tier[record["verdict"]] + 1

        if record["verdict"] != "leak":
            shapes[record["shape"]] = shapes.get(record["shape"], 0) + 1

    print("hand-verified precision")
    print()
    print(f"  {'tier':<8} {'findings':>8} {'leaks':>6} {'not leaks':>10} {'unsure':>7}  precision")
    for tier_name in ["high", "low"]:
        counts = tiers.get(tier_name)
        if counts is None:
            continue
        total = counts["leak"] + counts["not_leak"] + counts["unsure"]
        print(f"  {tier_name:<8} {total:>8} {counts['leak']:>6} {counts['not_leak']:>10} "
              f"{counts['unsure']:>7}  {counts['leak']}/{total}")

    if unaudited:
        print()
        print(f"  {unaudited} findings still unaudited - this is a sample, not a census")

    if shapes:
        print()
        print("false-positive shapes")
        print()
        for shape, count in sorted(shapes.items(), key=lambda item: -item[1]):
            print(f"  {count:>3}  {shape}")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    command = args[0]
    path = os.environ.get("AUDIT_CORPUS", DEFAULT_CORPUS)

    findings = collect_findings(path)
    audit = load_audit()

    if command == "list":
        cmd_list(findings, audit)
    elif command == "show":
        cmd_show(findings, audit, args[1:])
    elif command == "next":
        cmd_next(findings, audit)
    elif command == "record":
        cmd_record(findings, audit, args[1:])
    elif command == "report":
        cmd_report(findings, audit)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
