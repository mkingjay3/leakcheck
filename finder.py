"""Flags four lookahead-bias patterns in pandas/numpy backtest code.

high: bfill/backfill, shift(-n), rolling(center=True) - all three read rows
      that hadn't happened yet, readable off a single call.
low:  a whole-series statistic combined back into the series it summarises,
      e.g. (df - df.min()) / (df.max() - df.min()), which makes every row's
      value depend on every other row.

Anything else is out of scope on purpose - see README.md.
"""

import ast
import os
import sys

RANK = {"high": 0, "low": 1}

BACKFILL_METHODS = ["bfill", "backfill"]
AGGREGATE_METHODS = ["mean", "std", "max", "min", "sum", "median", "var"]
WINDOWED_METHODS = ["rolling", "expanding", "ewm"]

# lambdas and comprehensions bind their own names, so a bare `w` inside one
# is not the same `w` as outside it
NAME_BINDING_NODES = (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

SKIP_DIRS = ["venv", "site-packages", ".git", "node_modules"]


def extract_int_literal(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, int):
            return -node.operand.value
    return None


# a manual slice like df.Close[-30:] bounds the data the same way
# rolling/expanding/ewm do, without calling one of those methods
def is_windowed(node):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return node.func.attr in WINDOWED_METHODS
    return isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice)


# ast doesn't link a node back to its parent, so stamp one on up front
def annotate_parents(tree):
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent


# the series the statistic is computed over: np.std(x) puts it in the first
# argument, x.std() in the receiver
def aggregate_source(node):
    if node.args:
        return node.args[0]
    return node.func.value


def broadcast_root(node):
    """The expression to search for a bare use of the series.

    Stops at the innermost lambda or comprehension, since those bind their
    own names. For a plain assignment only the assigned value counts - the
    `bm_ret` in `bm_ret['x'] = bm_ret.mean(axis=1)` is the write target,
    not a broadcast.
    """
    current = node
    while current is not None:
        if isinstance(current, NAME_BINDING_NODES):
            return current
        if isinstance(current, ast.stmt):
            if isinstance(current, ast.Assign):
                return current.value
            return current
        current = getattr(current, "parent", None)
    return None


def contains_bare(node, source_dump):
    if node is None:
        return False
    # a series mentioned inside another aggregate call is being summarised
    # too, not broadcast: np.mean(r) / np.std(r) never touches r directly
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr in AGGREGATE_METHODS:
            return False
    if isinstance(node, ast.expr) and ast.dump(node) == source_dump:
        return True
    return any(contains_bare(child, source_dump) for child in ast.iter_child_nodes(node))


def broadcasts_over_series(node):
    """True if the aggregate's own series also appears on its own nearby.

    That makes the result a per-row value derived from every row, which is
    the leak itself - it doesn't matter where the result goes afterwards,
    and it often goes out through a return into a caller's column write.
    """
    return contains_bare(broadcast_root(node), ast.dump(aggregate_source(node)))


class Finding:
    def __init__(self, line, pattern, msg, conf):
        self.line = line
        self.pattern = pattern
        self.msg = msg
        self.conf = conf


class LeakFinder(ast.NodeVisitor):

    def __init__(self):
        self.findings = []
        self.shift_count = 0
        self.unknown_shifts = 0
        self.known_ints = {}
        self.windowed_names = set()

    # one finding per (line, pattern): `tib.min()` twice in one normalization
    # is one site, and reporting it twice tells the reader nothing new
    def add_finding(self, line, pattern, message, confidence):
        if any(f.line == line and f.pattern == pattern for f in self.findings):
            return
        self.findings.append(Finding(line, pattern, message, confidence))

    # resolves `name = <int literal>` and `name = <windowed expr>` at the top
    # level of a file only, and only if the name is assigned once - anything
    # more ambiguous is left unknown rather than guessed at
    def collect_top_level_names(self, tree):
        known_ints = {}
        windowed_names = set()
        assigned_twice = set()

        for stmt in tree.body:
            if not isinstance(stmt, ast.Assign):
                continue
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                continue

            int_value = extract_int_literal(stmt.value)
            windowed = is_windowed(stmt.value)
            if int_value is None and not windowed:
                continue

            name = stmt.targets[0].id
            if name in known_ints or name in windowed_names or name in assigned_twice:
                assigned_twice.add(name)
                known_ints.pop(name, None)
                windowed_names.discard(name)
                continue

            if int_value is not None:
                known_ints[name] = int_value
            else:
                windowed_names.add(name)

        self.known_ints = known_ints
        self.windowed_names = windowed_names

    def is_bounded(self, expr):
        if is_windowed(expr):
            return True
        return isinstance(expr, ast.Name) and expr.id in self.windowed_names

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
            if name in BACKFILL_METHODS:
                self.add_finding(node.lineno, name,
                                 "backward fill pulls future values into earlier rows", "high")
            elif name == "shift":
                self.check_shift(node)
            elif name == "rolling":
                self.check_rolling(node)
            elif name in AGGREGATE_METHODS:
                self.check_aggregate(node, name)

        # keep walking so calls nested inside this one still get checked
        self.generic_visit(node)

    def check_shift(self, node):
        self.shift_count = self.shift_count + 1

        # shifting inside an already-bounded slice reindexes within known
        # history rather than reaching past the end of it
        if self.is_bounded(node.func.value):
            return

        # pandas accepts shift(-1) or shift(periods=-1)
        arg = None
        if node.args:
            arg = node.args[0]
        else:
            for keyword in node.keywords:
                if keyword.arg == "periods":
                    arg = keyword.value

        periods = extract_int_literal(arg)
        if periods is None and isinstance(arg, ast.Name):
            periods = self.known_ints.get(arg.id)

        if periods is None:
            self.unknown_shifts = self.unknown_shifts + 1
        elif periods < 0:
            message = f"shift({periods}) pulls {abs(periods)} future rows backward"
            self.add_finding(node.lineno, "shift", message, "high")

    def check_rolling(self, node):
        for keyword in node.keywords:
            # only a literal True counts - center=some_flag can't be read,
            # and center=1 probably isn't meant as True
            if keyword.arg == "center" and isinstance(keyword.value, ast.Constant):
                if keyword.value.value is True:
                    self.add_finding(node.lineno, "rolling",
                                     "rolling(center=True) centers the window on future rows",
                                     "high")

    def check_aggregate(self, node, name):
        if self.is_bounded(aggregate_source(node)):
            return
        if not broadcasts_over_series(node):
            return

        message = f"{name}() over the whole series is combined back into that series"
        self.add_finding(node.lineno, name, message, "low")


def sort_findings(findings):
    return sorted(findings, key=lambda finding: (RANK[finding.conf], finding.line))


def print_findings(fpath, findings):
    for finding in sort_findings(findings):
        print(f"{fpath}:{finding.line} [{finding.conf}] {finding.pattern} {finding.msg}")


def analyze_file(fpath, quiet=False):
    try:
        with open(fpath, encoding="utf-8") as source_file:
            src = source_file.read()
    except (OSError, UnicodeDecodeError) as error:
        print(f"could not read {fpath}: {error}")
        return None

    try:
        tree = ast.parse(src, filename=fpath)
    except SyntaxError as error:
        print(f"could not parse {fpath}: {error}")
        return None

    annotate_parents(tree)

    finder = LeakFinder()
    finder.collect_top_level_names(tree)
    finder.visit(tree)

    if not quiet:
        print_findings(fpath, finder.findings)

    return finder


def find_py_files(path):
    if os.path.isfile(path):
        return [path]

    found = []
    for directory, subdirs, files in os.walk(path):
        subdirs[:] = [subdir for subdir in subdirs if subdir not in SKIP_DIRS]
        for filename in sorted(files):
            if filename.endswith(".py"):
                found.append(os.path.join(directory, filename))
    return found


def print_summary(total_high, total_low, total_shifts, total_unreadable_shifts):
    print()
    print(f"{total_high} high-confidence findings")
    print(f"{total_low} low-confidence findings")
    if total_shifts > 0:
        percent_unreadable = round(100 * total_unreadable_shifts / total_shifts)
        print(f"{total_shifts} shift calls, {total_unreadable_shifts} unreadable ({percent_unreadable}%)")


def main():
    args = sys.argv[1:]
    quiet = "--quiet" in args
    paths = [arg for arg in args if arg != "--quiet"]

    if not paths:
        print("usage: python3 finder.py [--quiet] <file_or_directory> ...")
        return

    fpaths = []
    for path in paths:
        fpaths.extend(find_py_files(path))

    total_high = 0
    total_low = 0
    total_shifts = 0
    total_unreadable_shifts = 0

    for fpath in fpaths:
        finder = analyze_file(fpath, quiet=quiet)
        if finder is None:
            continue
        for finding in finder.findings:
            if finding.conf == "high":
                total_high = total_high + 1
            else:
                total_low = total_low + 1
        total_shifts = total_shifts + finder.shift_count
        total_unreadable_shifts = total_unreadable_shifts + finder.unknown_shifts

    print_summary(total_high, total_low, total_shifts, total_unreadable_shifts)


if __name__ == "__main__":
    main()
