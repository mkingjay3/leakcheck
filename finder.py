import ast
import sys
from collections import namedtuple


Finding = namedtuple("Finding", ["line", "pattern", "message", "confidence"])


LEAKY_METHODS = {
    "bfill":    "backward fill pulls future values into earlier rows",
    "backfill": "backward fill pulls future values into earlier rows",
}


class LeakFinder(ast.NodeVisitor):

    def __init__(self, file_path):
        self.file_path = file_path
        self.findings = []

        # Counting these so I can check  how often the arg is smth i cant read
        self.shifts_seen = 0
        self.shifts_unresolvable = 0

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Attribute):
            self.generic_visit(node)
            return

        method_name = node.func.attr

        if method_name in LEAKY_METHODS:
            self.findings.append(Finding(
                node.lineno,
                method_name,
                LEAKY_METHODS[method_name],
                "high",
            ))

        elif method_name == "shift":
            self.check_shift(node)

        elif method_name == "rolling":
            self.check_rolling(node)

        self.generic_visit(node)

    def check_shift(self, node):
        self.shifts_seen += 1

        amount_node = self.get_shift_argument(node)

        if amount_node is None:
            return          # bare shift() defaults to 1, which is backward

        value = self.literal_int(amount_node)

        if value is None:
            # shift(x), shift(n - 1), shift(cfg["lag"]) and friends.
            # No way to know the sign without running it
            self.shifts_unresolvable += 1
            return

        if value < 0:
            self.findings.append(Finding(
                node.lineno,
                "shift",
                f"shift({value}) pulls {abs(value)} future rows backward",
                "high",
            ))

    def check_rolling(self, node):
        for kw in node.keywords:
            # center=1 would pass "== True" but isn't someone writing True,
            # so check identity instead of equality.
            if kw.arg == "center" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                self.findings.append(Finding(
                    node.lineno,
                    "rolling",
                    "rolling(center=True) centers the window on future rows",
                    "high",
                ))

    def get_shift_argument(self, node):
        # pandas accepts both shift(-1) and shift(periods=-1)
        if node.args:
            return node.args[0]

        for kw in node.keywords:
            if kw.arg == "periods":
                return kw.value

        return None

    def literal_int(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            operand = node.operand
            if isinstance(operand, ast.Constant) and isinstance(operand.value, int):
                return -operand.value

        return None


def analyze_file(file_path):
    """Check one file. Returns the finder so the caller can read both
    the findings and the shift stats. Returns None on error."""

    try:
        with open(file_path, encoding="utf-8") as source_file:
            source_code = source_file.read()
    except (OSError, UnicodeDecodeError) as error:
        print(f"could not read {file_path}: {error}")
        return None

    try:
        tree = ast.parse(source_code, filename=file_path)
    except SyntaxError as error:
        print(f"could not parse {file_path}: {error}")
        return None

    finder = LeakFinder(file_path)
    finder.visit(tree)

    for f in finder.findings:
        print(f"{file_path}:{f.line}  [{f.confidence}]  {f.pattern}  {f.message}")

    return finder

#Confidnece does nun rn, need kater
file_paths = sys.argv[1:]

if len(file_paths) == 0:
    print("usage: python finder.py <file1.py> <file2.py> ...")
else:
    total_findings = 0
    total_shifts = 0
    total_unresolvable = 0

    for path in file_paths:
        finder = analyze_file(path)
        if finder is None:
            continue
        total_findings += len(finder.findings)
        total_shifts += finder.shifts_seen
        total_unresolvable += finder.shifts_unresolvable

    print()
    print(f"{total_findings} findings")
    if total_shifts:
        pct = 100 * total_unresolvable / total_shifts
        print(f"{total_shifts} shift calls, {total_unresolvable} unreadable ({pct:.0f}%)")
