import ast
import os
import sys

RANK = {"high": 0, "low": 1}

BACKFILL_METHODS = ["bfill", "backfill"]

AGGREGATE_METHODS = ["mean", "std", "max", "min", "sum", "median", "var"]
WINDOWED_METHODS = ["rolling", "expanding", "ewm"]
PLOTTING_METHODS = ["plot", "fill_between", "scatter", "bar", "barh", "hist",
                     "pie", "boxplot", "imshow", "errorbar", "stem", "step"]

SKIP_DIRS = ["venv", "site-packages", ".git", "node_modules"]


def extract_int_literal(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, int):
            return -node.operand.value
    return None


def is_windowed_expr(node):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return node.func.attr in WINDOWED_METHODS
    # a manual slice like df.Close[-30:] bounds the data the same way
    # rolling/expanding/ewm do, just without calling one of those methods
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
        return True
    return False


# ast doesn't link a node back to its parent, so walk the whole tree once
# and stamp one on - lets check_aggregate ask "what statement is this call
# actually part of" without carrying a stack through every visit_* call
def annotate_parents(tree):
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent


# ast.stmt is the common base class for every statement node (Return,
# Assign, If, For, ...), as opposed to ast.expr for expression nodes -
# walking up .parent links until we hit one finds "the statement this
# expression lives inside of"
def enclosing_statement(node):
    current = node
    while current is not None and not isinstance(current, ast.stmt):
        current = getattr(current, "parent", None)
    return current


# groupby().sum() etc. aggregates across a categorical grouping (e.g. by
# country and year), not across time - "future rows leaking into earlier
# decisions" doesn't apply to a result that isn't ordered in time
def is_groupby_result(node):
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "groupby"


# walks up from an aggregate call, within its own statement, looking for
# a plotting call it's an argument to (e.g. ax.fill_between(x, y+std, ...))
# - stops at the enclosing statement, same boundary as enclosing_statement
def is_inside_plot_call(node):
    current = getattr(node, "parent", None)
    while current is not None and not isinstance(current, ast.stmt):
        if isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
            if current.func.attr in PLOTTING_METHODS:
                return True
        current = getattr(current, "parent", None)
    return False


class Finding:
    def __init__(self, line, pattern, msg, conf):
        self.line = line
        self.pattern = pattern
        self.msg = msg
        self.conf = conf


# walks the code's syntax tree
class LeakFinder(ast.NodeVisitor):

    def __init__(self):
        self.findings = []
        self.shift_count = 0
        self.unknown_shifts = 0
        self.known_ints = {}
        self.windowed_names = set()

    def add_finding(self, line, pattern, message, confidence):
        finding = Finding(line, pattern, message, confidence)
        self.findings.append(finding)

    # only resolves name = <int literal> / name = <windowed expr> at the
    # top level of the file, not inside functions/branches/loops, and
    # only if the name is assigned exactly once - anything more ambiguous
    # is left unknown rather than guessed at
    def collect_top_level_assignments(self, tree):
        known_ints = {}
        windowed_names = set()
        seen_more_than_once = set()

        for stmt in tree.body:
            if not isinstance(stmt, ast.Assign):
                continue
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                continue

            int_value = extract_int_literal(stmt.value)
            windowed = is_windowed_expr(stmt.value)
            if int_value is None and not windowed:
                continue

            name = stmt.targets[0].id
            if name in known_ints or name in windowed_names or name in seen_more_than_once:
                seen_more_than_once.add(name)
                known_ints.pop(name, None)
                windowed_names.discard(name)
                continue

            if int_value is not None:
                known_ints[name] = int_value
            else:
                windowed_names.add(name)

        self.known_ints = known_ints
        self.windowed_names = windowed_names

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

        # shifting within an already-bounded slice/window just reindexes
        # inside known history, same reasoning as check_aggregate
        if self.is_already_windowed(node.func.value):
            return

        # pandas accepts shift(-1) or shift(periods=-1)
        arg = None
        if node.args:
            arg = node.args[0]
        else:
            for keyword in node.keywords:
                if keyword.arg == "periods":
                    arg = keyword.value

        shift_amount = extract_int_literal(arg)
        if shift_amount is None and isinstance(arg, ast.Name):
            shift_amount = self.known_ints.get(arg.id)

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
        # np.std(x) puts the series in the first argument instead of the
        # receiver, since the receiver is just the numpy module name
        first_arg = node.args[0] if node.args else None

        if self.is_already_windowed(receiver) or self.is_already_windowed(first_arg):
            return

        # a categorical/cross-sectional aggregate (e.g. by country and
        # year), not a time-ordered one - the leak this tool looks for
        # doesn't apply
        if is_groupby_result(receiver):
            return

        # feeds a plotted band/line, not a trading decision
        if is_inside_plot_call(node):
            return

        # computed and handed straight back out of the function via return,
        # with nothing assigned into a dataframe column in this scope - a
        # report value (Sharpe ratio, accuracy, ...), not a trading input.
        # only catches a direct `return ...mean()...`, not one first
        # assigned to a name and returned on a later line - narrow on
        # purpose, see notes.txt 2026/09/08
        if isinstance(enclosing_statement(node), ast.Return):
            return

        # could feed a trading decision or just a printout - can't tell from the AST, so low confidence
        message = f"{name}() over the whole series pulls later rows into earlier decisions"
        self.add_finding(node.lineno, name, message, "low")

    def is_already_windowed(self, expr):
        if expr is None:
            return False
        if is_windowed_expr(expr):
            return True
        return isinstance(expr, ast.Name) and expr.id in self.windowed_names


def get_confidence_rank(finding):
    return RANK[finding.conf]


def sort_findings(findings):
    return sorted(findings, key=get_confidence_rank)


def print_findings(fpath, findings, show_low):
    for finding in sort_findings(findings):
        if finding.conf == "low" and not show_low:
            continue
        print(f"{fpath}:{finding.line} [{finding.conf}] {finding.pattern} {finding.msg}")


def analyze_file(fpath, quiet=False, show_low=False):
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

    annotate_parents(tree)

    finder = LeakFinder()
    finder.collect_top_level_assignments(tree)
    finder.visit(tree)

    if not quiet:
        print_findings(fpath, finder.findings, show_low)

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


def print_summary(total_high, total_low, show_low, total_shifts, total_unreadable_shifts):
    print()
    print(f"{total_high} high-confidence findings")
    if show_low:
        print(f"{total_low} low-confidence findings")
    else:
        print(f"{total_low} low-confidence findings not shown (pass --low to show them)")
    if total_shifts > 0:
        percent_unreadable = round(100 * total_unreadable_shifts / total_shifts)
        print(f"{total_shifts} shift calls, {total_unreadable_shifts} unreadable ({percent_unreadable}%)")


def main():
    args = sys.argv[1:]
    flags = ("--quiet", "--low")
    quiet = "--quiet" in args
    show_low = "--low" in args

    paths = []
    for arg in args:
        if arg not in flags:
            paths.append(arg)

    if not paths:
        print("usage: python finder.py [--quiet] [--low] <file_or_directory> ...")
        return

    fpaths = []
    for path in paths:
        fpaths.extend(find_py_files(path))

    total_high = 0
    total_low = 0
    total_shifts = 0
    total_unreadable_shifts = 0

    for fpath in fpaths:
        finder = analyze_file(fpath, quiet=quiet, show_low=show_low)
        if finder is None:
            continue
        for finding in finder.findings:
            if finding.conf == "high":
                total_high = total_high + 1
            else:
                total_low = total_low + 1
        total_shifts = total_shifts + finder.shift_count
        total_unreadable_shifts = total_unreadable_shifts + finder.unknown_shifts

    print_summary(total_high, total_low, show_low, total_shifts, total_unreadable_shifts)


if __name__ == "__main__":
    main()


 