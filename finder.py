import ast
import os
import sys

RANK = {"high": 0, "low": 1}

BACKFILL_METHODS = ["bfill", "backfill"]

AGGREGATE_METHODS = ["mean", "std", "max", "min", "sum", "median", "var"]
WINDOWED_METHODS = ["rolling", "expanding", "ewm"]

SKIP_DIRS = ["venv", "site-packages", ".git", "node_modules"]


class Finding:
    def __init__(self, line, pattern, msg, conf):
        self.line = line
        self.pattern = pattern
        self.msg = msg
        self.conf = conf


# walks the code's syntax tree
class LeakFinder(ast.NodeVisitor):

    def __init__(self, fpath):
        self.fpath = fpath
        self.findings = []
        self.shift_count = 0
        self.unknown_shifts = 0

    def add_finding(self, line, pattern, message, confidence):
        finding = Finding(line, pattern, message, confidence)
        self.findings.append(finding)

    def visit_Call(self, node):
        # only interested in method calls on smth
        if not isinstance(node.func, ast.Attribute):
            self.generic_visit(node)
            return

        name = node.func.attr

        if name in BACKFILL_METHODS:
            message = "backward fill pulls future values into earlier rows"
            self.add_finding(node.lineno, name, message, "high")
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

        # pandas accepts shift(-1) or shift(periods=-1)
        arg = None
        if node.args:
            arg = node.args[0]
        else:
            for keyword in node.keywords:
                if keyword.arg == "periods":
                    arg = keyword.value

        shift_amount = None
        if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
            shift_amount = arg.value
        elif isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
            if isinstance(arg.operand, ast.Constant) and isinstance(arg.operand.value, int):
                shift_amount = -arg.operand.value

        if shift_amount is None:
            self.unknown_shifts = self.unknown_shifts + 1
            return

        if shift_amount < 0:
            rows_pulled = abs(shift_amount)
            message = f"shift({shift_amount}) pulls {rows_pulled} future rows backward"
            self.add_finding(node.lineno, "shift", message, "high")

    def check_rolling(self, node):
        for keyword in node.keywords:
            if keyword.arg == "center":
                # only a literal True counts - center=some_flag we can't read, center=1 probably isn't meant as True
                if isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                    message = "rolling(center=True) centers the window on future rows"
                    self.add_finding(node.lineno, "rolling", message, "high")

    def check_aggregate(self, node, name):
        receiver = node.func.value

        receiver_is_windowed = False
        if isinstance(receiver, ast.Call) and isinstance(receiver.func, ast.Attribute):
            if receiver.func.attr in WINDOWED_METHODS:
                receiver_is_windowed = True

        if receiver_is_windowed:
            return

        # could feed a trading decision or just a printout - can't tell from the AST, so low confidence
        message = f"{name}() over the whole series pulls later rows into earlier decisions"
        self.add_finding(node.lineno, name, message, "low")


def get_confidence_rank(finding):
    return RANK[finding.conf]


def sort_findings(findings):
    return sorted(findings, key=get_confidence_rank)


def print_findings(fpath, findings):
    for finding in sort_findings(findings):
        print(f"{fpath}:{finding.line} [{finding.conf}] {finding.pattern} {finding.msg}")


def analyze_file(fpath, quiet=False):
    try:
        source_file = open(fpath, encoding="utf-8")
        src = source_file.read()
        source_file.close()
    except (OSError, UnicodeDecodeError) as error:
        print(f"could not read {fpath}: {error}")
        return None

    try:
        tree = ast.parse(src, filename=fpath)
    except SyntaxError as error:
        print(f"could not parse {fpath}: {error}")
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
    for directory, subdirs, files in os.walk(path):
        kept_subdirs = []
        for subdir in subdirs:
            if subdir not in SKIP_DIRS:
                kept_subdirs.append(subdir)
        subdirs[:] = kept_subdirs

        for filename in files:
            if filename.endswith(".py"):
                found.append(os.path.join(directory, filename))
    return found


def print_summary(total_findings, total_shifts, total_unreadable_shifts):
    print()
    print(f"{total_findings} findings")
    if total_shifts > 0:
        percent_unreadable = round(100 * total_unreadable_shifts / total_shifts)
        print(f"{total_shifts} shift calls, {total_unreadable_shifts} unreadable ({percent_unreadable}%)")


def main():
    args = sys.argv[1:]
    quiet = "--quiet" in args

    paths = []
    for arg in args:
        if arg != "--quiet":
            paths.append(arg)

    if not paths:
        print("usage: python finder.py [--quiet] <file_or_directory> ...")
        return

    fpaths = []
    for path in paths:
        fpaths.extend(find_py_files(path))

    total_findings = 0
    total_shifts = 0
    total_unreadable_shifts = 0

    for fpath in fpaths:
        finder = analyze_file(fpath, quiet=quiet)
        if finder is None:
            continue
        total_findings = total_findings + len(finder.findings)
        total_shifts = total_shifts + finder.shift_count
        total_unreadable_shifts = total_unreadable_shifts + finder.unknown_shifts

    print_summary(total_findings, total_shifts, total_unreadable_shifts)


if __name__ == "__main__":
    main()
