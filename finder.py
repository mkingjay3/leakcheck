import ast
import os
import sys
from collections import namedtuple

Finding = namedtuple("Finding", ["line", "pattern", "msg", "conf"])

RANK = {"high": 0, "low": 1}
# enzo reminder, some pandas use backfill
BACKFILL_METHODS = {"bfill", "backfill"}

AGGREGATE_METHODS = {"mean", "std", "max", "min", "sum", "median", "var"}

# if one of these ran first, an aggregate after it only looks backward by construction
WINDOWED_METHODS = {"rolling", "expanding", "ewm"}

SKIP_DIRS = {"venv", "site-packages", ".git", "node_modules"}


# walks the ast looking for backtest code that peeks at future rows without running it
class LeakFinder(ast.NodeVisitor):

    def __init__(self, fpath):
        self.fpath = fpath
        self.findings = []

        
        self.shift_count = 0
        self.unknown_shifts = 0

    def add_finding(self, line, pattern, message, confidence):
        self.findings.append(Finding(line, pattern, message, confidence))

    def visit_Call(self, node):
        # only method calls to object 
        if not isinstance(node.func, ast.Attribute):
            self.generic_visit(node)
            return

        name = node.func.attr

        if name in BACKFILL_METHODS:
            self.add_finding(node.lineno, name, "backward fill pulls future values into earlier rows", "high")
        elif name == "shift":
            self.check_shift(node)
        elif name == "rolling":
            self.check_rolling(node)
        elif name in AGGREGATE_METHODS:
            self.check_aggregate(node, name)

        # rucurse 
        self.generic_visit(node)

    def check_shift(self, node):
        self.shift_count += 1

        # pandas accepts shift(-1) or shift(periods=-1)
        if node.args:
            arg = node.args[0]
        else:
            arg = next((kw.value for kw in node.keywords if kw.arg == "periods"), None)

        # -1 isn't a single Constant node it's UnaryOp split the - and the 1
        if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
            n = arg.value
        elif isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub) and isinstance(arg.operand, ast.Constant) and isinstance(arg.operand.value, int):
            n = -arg.operand.value
        else:
            n = None

        if n is None:
            # e.g. shift(x) or shift(cfg["lag"]) - can't know the sign without running it
            self.unknown_shifts += 1
            return


        if n < 0:
            self.add_finding(node.lineno, "shift", f"shift({n}) pulls {abs(n)} future rows backward", "high")

    def check_rolling(self, node):
        for kw in node.keywords:
            # only a literal True counts - center=some_flag we can't read, center=1 probably isn't meant as True
            if kw.arg == "center" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                self.add_finding(node.lineno, "rolling", "rolling(center=True) centers the window on future rows", "high")

    def check_aggregate(self, node, name):
        recv = node.func.value
        if isinstance(recv, ast.Call) and isinstance(recv.func, ast.Attribute) and recv.func.attr in WINDOWED_METHODS:
            return
        # could feed a trading decision or just a printout - can't tell from the AST, so low confidence
        self.add_finding(node.lineno, name, f"{name}() over the whole series pulls later rows into earlier decisions", "low")


def sort_findings(findings):
    return sorted(findings, key=lambda f: RANK[f.conf])


def print_findings(fpath, findings):
    for f in sort_findings(findings):
        print(f"{fpath}:{f.line} [{f.conf}] {f.pattern} {f.msg}")


def analyze_file(fpath, quiet=False):
    try:
        with open(fpath, encoding="utf-8") as fh:
            src = fh.read()
    except (OSError, UnicodeDecodeError) as e:
        print(f"could not read {fpath}: {e}")
        return None

    try:
        tree = ast.parse(src, filename=fpath)
    except SyntaxError as e:
        print(f"could not parse {fpath}: {e}")
        return None

    finder = LeakFinder(fpath)
    finder.visit(tree)

    if not quiet:
        print_findings(fpath, finder.findings)

    return finder


def find_py_files(path):
    if os.path.isfile(path):
        return [path]

    found = []
    for d, subdirs, files in os.walk(path):
        # must mutate in place - subdirs = [...] makes a new list os.walk never sees, and pruning stops working
        subdirs[:] = [s for s in subdirs if s not in SKIP_DIRS]
        for fn in files:
            if fn.endswith(".py"):
                found.append(os.path.join(d, fn))
    return found


def print_summary(nfindings, nshifts, nbad):
    print()
    print(f"{nfindings} findings")
    if nshifts > 0:
        pct = 100 * nbad / nshifts
        print(f"{nshifts} shift calls, {nbad} unreadable ({pct:.0f}%)")


def main():
    args = sys.argv[1:]
    quiet = "--quiet" in args
    paths = [a for a in args if a != "--quiet"]

    if not paths:
        print("usage: python finder.py [--quiet] <file_or_directory> ...")
        return

    fpaths = []
    for p in paths:
        fpaths.extend(find_py_files(p))

    nfindings = 0
    nshifts = 0
    nbad = 0

    for fpath in fpaths:
        finder = analyze_file(fpath, quiet=quiet)
        if finder is None:
            continue
        nfindings += len(finder.findings)
        nshifts += finder.shift_count
        nbad += finder.unknown_shifts

    print_summary(nfindings, nshifts, nbad)


if __name__ == "__main__":
    main()
