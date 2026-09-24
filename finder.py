"""Flags lookahead bias in pandas/numpy backtest code, from the syntax tree
only - nothing here imports or runs the code it reads.

high: bfill/backfill, shift(-n), rolling(center=True). All three read rows
      that hadn't happened yet, and all three are readable off one call.
low:  a whole-series statistic put back into the series it summarises, either
      combined with it arithmetically, as in (df - df.min()) / (df.max() -
      df.min()), or used to fill rows, as in shift(1, fill_value=x.mean()).
      Either way a row ends up depending on every other row.

Anything else is out of scope on purpose, and README.md says what that costs.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import os
import sys
import time
from typing import Iterable, Sequence, TextIO

# real strategy files nest deeply - NostalgiaForInfinityX7.py is 79k lines of
# chained boolean conditions and needs about 3000 frames to walk. Analysing it
# is worth more than the default limit is; anything past this still raises,
# and analyze_file reports that rather than dying mid-run.
sys.setrecursionlimit(5000)

TOOL_NAME = "leakcheck"
TOOL_URL = "https://github.com/mkingjay3/leakcheck"
VERSION = "0.1.0"

RANK = {"high": 0, "low": 1}

ANSI = {"red": "\033[31m", "yellow": "\033[33m", "bold": "\033[1m",
        "dim": "\033[2m", "reset": "\033[0m"}

CONFIDENCE_STYLE = {"high": ("bold", "red"), "low": ("bold", "yellow")}

# None until main decides. Left None, paint() is a no-op, so importing this
# module and calling print_findings never leaks escape codes into a caller's
# output - only the CLI turns colour on.
USE_COLOR = None

BACKFILL_METHODS = ["bfill", "backfill"]
AGGREGATE_METHODS = ["mean", "std", "max", "min", "sum", "median", "var"]
WINDOWED_METHODS = ["rolling", "expanding", "ewm"]

# functions that map over elements, so a series passed through one is still a
# series and can still be broadcast against a statistic
ELEMENTWISE_FUNCS = ["abs", "fabs", "log", "log1p", "log10", "exp", "sqrt",
                     "square", "power", "sign", "clip", "where", "maximum", "minimum"]

# lambdas and comprehensions bind their own names, so a bare `w` inside one
# is not the same `w` as outside it
NAME_BINDING_NODES = (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

# Every rule the tool has, in one table: the calls that trigger it, the
# LeakFinder method that decides whether a given call is really a leak, and the
# text a SARIF consumer shows. Adding a rule is adding an entry here plus the
# method it names - visit_Call dispatches through this table and needs no edit.
#
# The four defects the rules recognise. A Finding's `pattern` is the call that
# triggered it, so `mean()` and `std()` both produce whole-series-stat findings;
# the rule id is the defect class. SARIF consumers and GitHub group and
# suppress findings by rule id, so these ids are the tool's public names and
# have to stay stable even if the internals move around.
RULES = [
    {
        "id": "backward-fill",
        "handler": "check_backfill",
        "patterns": BACKFILL_METHODS,
        "conf": "high",
        "short": "Backward fill pulls future values into earlier rows",
        "full": "bfill() and backfill() fill a gap with the next known value, "
                "which is by definition a value from later in the series. Every "
                "filled row then carries information that did not exist yet at "
                "that row's timestamp. Fill forward instead, or leave the gap.",
    },
    {
        "id": "negative-shift",
        "handler": "check_shift",
        "patterns": ["shift"],
        "conf": "high",
        "short": "A negative shift moves future rows backward onto earlier ones",
        "full": "shift(-n) moves each value n rows earlier, so a row ends up "
                "holding data recorded after it. This is correct only when the "
                "result is a prediction target that never reaches a feature; "
                "mark those lines with `# leakcheck: ignore`.",
    },
    {
        "id": "centered-window",
        "handler": "check_rolling",
        "patterns": ["rolling"],
        "conf": "high",
        "short": "A centered rolling window reads rows ahead of the current one",
        "full": "rolling(n, center=True) puts the current row in the middle of "
                "its window, so half the window is future data. A trailing "
                "window, or a compensating .shift(n // 2) afterwards, keeps the "
                "window behind the row it lands on.",
    },
    {
        "id": "whole-series-stat",
        "handler": "check_aggregate",
        "patterns": AGGREGATE_METHODS,
        "conf": "low",
        "short": "A whole-series statistic is fed back into that same series",
        "full": "A statistic taken over the entire series summarises rows that "
                "had not happened yet. Combining it back into the series, or "
                "using it to fill rows, makes every row depend on every other "
                "row. Compute it over a trailing window, or over training data "
                "only.",
    },
]

RULE_FOR_PATTERN = {pattern: rule["id"] for rule in RULES for pattern in rule["patterns"]}

SKIP_DIRS = ["venv", "site-packages", ".git", "node_modules"]

IGNORE_COMMENT = "leakcheck: ignore"


def extract_int_literal(node: ast.AST | None) -> int | None:
    """The value of an int literal node, negative ones included, else None.

    Only a literal is read. An expression that merely evaluates to an int,
    like `n - 1`, returns None and leaves the caller to treat the value as
    unresolved.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, int):
            return -node.operand.value
    return None


# lines a reader has already judged. The commonest real need is a machine
# learning label - `df['target'] = df['close'].shift(-1)` is deliberate and
# structurally identical to the bug, so no rule can tell them apart.
def ignored_lines(src: str) -> set[int]:
    ignored = set()
    for number, text in enumerate(src.splitlines(), start=1):
        if IGNORE_COMMENT in text:
            ignored.add(number)
    return ignored


# a manual slice like df.Close[-30:] bounds the data the same way
# rolling/expanding/ewm do, without calling one of those methods
def is_windowed(node: ast.AST) -> bool:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return node.func.attr in WINDOWED_METHODS
    return isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice)


def annotate_parents(tree: ast.AST) -> None:
    """Stamp a .parent on every node, since ast does not link back up.

    Several rules decide by looking at what encloses a call, so this has to
    run before a LeakFinder visits the tree.
    """
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent


# the series the statistic is computed over: np.std(x) puts it in the first
# argument, x.std() in the receiver
def aggregate_source(node: ast.Call) -> ast.expr:
    if node.args:
        return node.args[0]
    return node.func.value


def broadcast_root(node: ast.AST) -> ast.AST | None:
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


def rolling_window(node: ast.Call) -> int | None:
    for keyword in node.keywords:
        if keyword.arg == "window":
            return extract_int_literal(keyword.value)
    if node.args:
        return extract_int_literal(node.args[0])
    return None


# A centered window of n reaches n//2 rows ahead. Shifting the result k rows
# forward afterwards lands every value on a row whose window sits entirely
# behind it, so k >= n//2 undoes the lookahead. This is the standard support
# and resistance idiom - rolling(5, center=True).apply(is_pivot).shift(2) -
# and it is correct, not a leak.
def compensating_shift(node: ast.Call, window: int) -> bool:
    current = node
    while True:
        attribute = getattr(current, "parent", None)
        if not isinstance(attribute, ast.Attribute):
            return False
        call = getattr(attribute, "parent", None)
        if not (isinstance(call, ast.Call) and call.func is attribute):
            return False

        if attribute.attr == "shift":
            periods = extract_int_literal(call.args[0]) if call.args else 1
            return periods is not None and periods >= window // 2

        current = call


def warn(message: str) -> None:
    """Report a file that could not be handled, on stderr.

    A file the tool cannot read is not a finding, and putting it on stdout
    would splice plain prose into the middle of a --format json or --format
    sarif document and leave it unparseable. Every such file still gets said
    out loud rather than skipped silently.
    """
    print(message, file=sys.stderr)


def color_enabled(stream: TextIO) -> bool:
    """True if it is safe to write ANSI escapes to this stream.

    A redirected stream gets none, so `leakcheck src > report.txt` stays
    readable, and NO_COLOR (https://no-color.org) and TERM=dumb are honoured
    for the terminals and CI runners that ask for plain text.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return hasattr(stream, "isatty") and stream.isatty()


def paint(text: str, *styles: str) -> str:
    if not USE_COLOR:
        return text
    return "".join(ANSI[style] for style in styles) + text + ANSI["reset"]


# ast reports col_offset as a UTF-8 byte offset, not a character position, so
# any non-ASCII earlier in the line would slide a caret right of its token
def char_column(line: str, byte_column: int) -> int:
    return len(line.encode("utf-8")[:byte_column].decode("utf-8", "replace"))


def call_span(node: ast.Call, lines: Sequence[str]) -> tuple[int, int]:
    """Character columns of the method call to underline, on node.lineno.

    In `df['close'].bfill()` this is `bfill()`. The attribute name ends the
    Attribute node, so it starts exactly len(attr) bytes back from that end
    and no searching of the line is needed. A call split across lines has no
    one-line span to point at, so it underlines from the start of the call
    to the end of its first line instead.
    """
    line = lines[node.lineno - 1] if node.lineno <= len(lines) else ""

    if isinstance(node.func, ast.Attribute) and node.func.end_lineno == node.lineno:
        start_byte = node.func.end_col_offset - len(node.func.attr.encode("utf-8"))
        end_byte = node.end_col_offset if node.end_lineno == node.lineno else None
    else:
        start_byte = node.col_offset
        end_byte = None

    start = char_column(line, start_byte)
    end = char_column(line, end_byte) if end_byte is not None else len(line)
    return start, max(end, start + 1)


def called_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def contains_bare(node: ast.AST | None, source_dump: str) -> bool:
    if node is None:
        return False

    if isinstance(node, ast.expr) and ast.dump(node) == source_dump:
        return True

    # a container doesn't combine what's in it, so a mention in one element
    # of [a, b] says nothing about the other
    if isinstance(node, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
        return False

    if isinstance(node, ast.Call):
        name = called_name(node)
        if name in AGGREGATE_METHODS:
            # a WINDOWED aggregate is still one value per row, so combining a
            # whole-series statistic with one broadcasts over the rows just as
            # a bare series does: returns.mean() / returns.rolling(20).std()
            receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
            if isinstance(receiver, ast.Call) and is_windowed(receiver):
                return ast.dump(receiver.func.value) == source_dump
            # otherwise it is summarised again, not broadcast, the same way
            # np.mean(r) / np.std(r) is only a summary
            return False
        # anything else consuming the series gives back who knows what -
        # len(x) is a count, ulcer_index(x) is a scalar. Only functions that
        # map over elements leave a series that can still be broadcast.
        if name not in ELEMENTWISE_FUNCS:
            return False

    return any(contains_bare(child, source_dump) for child in ast.iter_child_nodes(node))


def fills_rows(node: ast.Call) -> bool:
    """True if the statistic is being used to fill rows.

    shift(r, fill_value=close.mean()) puts the whole-series mean into the
    first r rows, and fillna(df.mean()) puts it wherever data was missing.
    Those rows then depend on every row, which is the same leak as a
    broadcast reached a different way.
    """
    parent = getattr(node, "parent", None)
    if isinstance(parent, ast.keyword):
        if parent.arg == "fill_value":
            return True
        call = getattr(parent, "parent", None)
        return isinstance(call, ast.Call) and called_name(call) == "fillna"
    return isinstance(parent, ast.Call) and called_name(parent) == "fillna"


def broadcasts_over_series(node: ast.Call) -> bool:
    """True if the statistic is combined arithmetically with its own series.

    That makes the result a per-row value derived from every row, which is
    the leak itself - it doesn't matter where the result goes afterwards,
    and it often goes out through a return into a caller's column write.

    Arithmetic is the whole test. `{"min": s.min(), "p5": s.quantile(.05)}`
    mentions s twice and combines nothing; `len(s) > 0 and s.std() > 0`
    likewise. Only a BinOp, UnaryOp or Compare actually applies the one
    number to the many rows, so walk up looking for one of those that also
    holds a bare use of the series.
    """
    source_dump = ast.dump(aggregate_source(node))
    root = broadcast_root(node)

    # a per-row value that another aggregate immediately collapses never
    # escapes as a per-row value. np.sum(np.abs(x - np.mean(x)) > r) is a
    # count, for the same reason np.mean(r) / np.std(r) is only a summary.
    current = getattr(node, "parent", None)
    while current is not None and current is not root:
        if isinstance(current, ast.Call) and called_name(current) in AGGREGATE_METHODS:
            return False
        current = getattr(current, "parent", None)

    current = node
    while current is not None:
        if isinstance(current, (ast.BinOp, ast.UnaryOp, ast.Compare)):
            if contains_bare(current, source_dump):
                return True
        if current is root:
            return False
        current = getattr(current, "parent", None)
    return False


class Finding:
    """One flagged call: where it is, which rule caught it, and why.

    `col` and `end_col` are 0-based character offsets into the source line,
    half-open, so `line[col:end_col]` is the text to underline. `pattern` is
    the method that triggered the finding (`min`, `std`) while `rule` is the
    defect class it belongs to; several patterns share one rule.
    """

    def __init__(self, line: int, col: int, end_col: int,
                 pattern: str, msg: str, conf: str) -> None:
        self.line = line
        self.col = col
        self.end_col = end_col
        self.pattern = pattern
        self.msg = msg
        self.conf = conf

    # the call that triggered this finding is `pattern`; the defect class it
    # belongs to is `rule`, and that is what a SARIF consumer groups by
    @property
    def rule(self) -> str:
        return RULE_FOR_PATTERN.get(self.pattern, self.pattern)


class LeakFinder(ast.NodeVisitor):
    """Walks one file's syntax tree and collects lookahead-bias findings.

    Visit a parsed module with `.visit(tree)`, then read `.findings`, plus
    `.shift_count` and `.unknown_shifts` for the coverage numbers the summary
    reports. Call `annotate_parents(tree)` and `.collect_top_level_names(tree)`
    first: the rules walk upward from a node and resolve names assigned once at
    module level, and neither works without them.

    It deliberately does not follow a value across functions, modules or
    reassignments. A shift amount that arrives as a parameter, or a name
    assigned twice, is left unresolved and counted in `.unknown_shifts` rather
    than guessed at in either direction, which is why that count is printed on
    every run.
    """

    def __init__(self, ignored: Iterable[int] = frozenset(),
                 lines: Iterable[str] = ()) -> None:
        self.ignored = ignored
        self.lines = list(lines)
        self.findings = []
        self.shift_count = 0
        self.unknown_shifts = 0
        self.known_ints = {}
        self.windowed_names = set()

    # one finding per (line, pattern): `tib.min()` twice in one normalization
    # is one site, and reporting it twice tells the reader nothing new
    def add_finding(self, node: ast.Call, pattern: str,
                    message: str, confidence: str) -> None:
        line = node.lineno
        if line in self.ignored:
            return
        if any(f.line == line and f.pattern == pattern for f in self.findings):
            return
        col, end_col = call_span(node, self.lines)
        self.findings.append(Finding(line, col, end_col, pattern, message, confidence))

    # resolves `name = <int literal>` and `name = <windowed expr>` at the top
    # level of a file only, and only if the name is assigned once - anything
    # more ambiguous is left unknown rather than guessed at
    def collect_top_level_names(self, tree: ast.Module) -> None:
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

    def is_bounded(self, expr: ast.AST) -> bool:
        if is_windowed(expr):
            return True
        return isinstance(expr, ast.Name) and expr.id in self.windowed_names

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            handler = HANDLER_FOR_PATTERN.get(node.func.attr)
            if handler is not None:
                handler(self, node, node.func.attr)

        # keep walking so calls nested inside this one still get checked
        self.generic_visit(node)

    def check_backfill(self, node: ast.Call, name: str) -> None:
        self.add_finding(node, name,
                         "backward fill pulls future values into earlier rows", "high")

    def check_shift(self, node: ast.Call, name: str) -> None:
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
            rows = "row" if abs(periods) == 1 else "rows"
            message = f"shift({periods}) pulls {abs(periods)} future {rows} backward"
            self.add_finding(node, "shift", message, "high")

    def check_rolling(self, node: ast.Call, name: str) -> None:
        # only a literal True counts - center=some_flag can't be read, and
        # center=1 probably isn't meant as True
        centered = any(keyword.arg == "center"
                       and isinstance(keyword.value, ast.Constant)
                       and keyword.value.value is True
                       for keyword in node.keywords)
        if not centered:
            return

        window = rolling_window(node)
        if window is not None and compensating_shift(node, window):
            return

        self.add_finding(node, "rolling",
                         "rolling(center=True) centers the window on future rows", "high")

    def check_aggregate(self, node: ast.Call, name: str) -> None:
        if self.is_bounded(aggregate_source(node)):
            return

        if fills_rows(node):
            message = f"{name}() over the whole series is used to fill rows"
        elif broadcasts_over_series(node):
            message = f"{name}() over the whole series is combined back into that series"
        else:
            return

        self.add_finding(node, name, message, "low")


# Resolved after the class body so a handler named in RULES that does not exist
# raises here, on import, rather than on the first file that happens to contain
# that call.
HANDLER_FOR_PATTERN = {
    pattern: getattr(LeakFinder, rule["handler"])
    for rule in RULES
    for pattern in rule["patterns"]
}


def sort_findings(findings: Iterable[Finding]) -> list[Finding]:
    """Findings ordered high confidence first, then by line."""
    return sorted(findings, key=lambda finding: (RANK[finding.conf], finding.line))


def caret_prefix(line: str, column: int) -> str:
    # tabs are kept as tabs so the carets stay under their token whatever
    # width the terminal renders a tab at
    return "".join("\t" if char == "\t" else " " for char in line[:column])


def format_finding(fpath: str, finding: Finding, lines: Sequence[str],
                   number_width: int = 0) -> str:
    """One finding as three lines: a header, the source line, and carets.

    sample.py:7:27: high  backward-fill
       7 | df['close'] = df['close'].bfill()
         |                           ^^^^^^^ backward fill pulls future rows back

    number_width pads the line-number gutter so several findings in one file
    line up under each other.
    """
    style = CONFIDENCE_STYLE[finding.conf]
    header = (f"{paint(fpath, 'bold')}:{finding.line}:{finding.col + 1}: "
              f"{paint(finding.conf, *style)}  {finding.rule}")

    if finding.line > len(lines):
        return f"{header}\n    {finding.msg}"

    source = lines[finding.line - 1]
    number = str(finding.line).rjust(number_width)
    gutter = " " * len(number)
    carets = paint("^" * max(finding.end_col - finding.col, 1), *style)
    bar = paint("|", "dim")

    return "\n".join([
        header,
        f"  {number} {bar} {source}",
        f"  {gutter} {bar} {caret_prefix(source, finding.col)}{carets} {finding.msg}",
    ])


def print_findings(fpath: str, findings: Iterable[Finding],
                   lines: Iterable[str] = ()) -> None:
    """Print each finding with its source line and carets.

    `lines` is the file's source, split; without it the carets cannot be drawn
    and each finding falls back to a header and its message.
    """
    lines = list(lines)
    width = max((len(str(finding.line)) for finding in findings), default=0)
    for finding in sort_findings(findings):
        print(format_finding(fpath, finding, lines, width))


# SARIF levels are a fixed vocabulary. A high-confidence finding is an error;
# a low-confidence one is a note, which keeps GitHub's default pull request
# check from failing on the tier that is allowed to be wrong.
SARIF_LEVELS = {"high": "error", "low": "note"}


def relative_uri(fpath: str) -> str:
    """A repository-relative, forward-slash path for a SARIF location.

    GitHub attaches an annotation only when the uri matches a path in the
    repository, so an absolute path or a Windows separator produces a result
    that uploads cleanly and then appears nowhere.
    """
    try:
        relative = os.path.relpath(fpath, os.getcwd())
    except ValueError:  # a different drive on Windows
        return fpath.replace(os.sep, "/")

    if relative.startswith(".."):
        relative = os.path.abspath(fpath)
    return relative.replace(os.sep, "/")


# lets a consumer follow one finding across commits that move it up or down
# the file, since the line number alone changes when anything above it does
def fingerprint(finding: Finding, source_line: str) -> str:
    payload = "\x00".join([finding.rule, finding.pattern, source_line.strip()])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def sarif_document(results: Sequence[tuple[str, "LeakFinder"]]) -> dict:
    """Findings as a SARIF 2.1.0 run.

    results is a list of (path, finder) pairs. Columns are 1-based and
    endColumn points one past the last character of the span, per the spec;
    columnKind says they are code points, which is what call_span produces
    and what a Python string indexes by.
    """
    rule_index = {rule["id"]: index for index, rule in enumerate(RULES)}
    sarif_results = []

    for fpath, finder in results:
        uri = relative_uri(fpath)
        for finding in sort_findings(finder.findings):
            if finding.line <= len(finder.lines):
                source = finder.lines[finding.line - 1]
            else:
                source = ""

            sarif_results.append({
                "ruleId": finding.rule,
                "ruleIndex": rule_index.get(finding.rule, 0),
                "level": SARIF_LEVELS[finding.conf],
                "message": {"text": finding.msg},
                "locations": [{
                    "physicalLocation": {
                        "artifactLocation": {"uri": uri, "uriBaseId": "%SRCROOT%"},
                        "region": {
                            "startLine": finding.line,
                            "startColumn": finding.col + 1,
                            "endColumn": finding.end_col + 1,
                            "snippet": {"text": source},
                        },
                    },
                }],
                "partialFingerprints": {"leakcheckSourceLine/v1": fingerprint(finding, source)},
            })

    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {
                "name": TOOL_NAME,
                "version": VERSION,
                "informationUri": TOOL_URL,
                "rules": [{
                    "id": rule["id"],
                    "name": rule["id"],
                    "shortDescription": {"text": rule["short"]},
                    "fullDescription": {"text": rule["full"]},
                    "defaultConfiguration": {"level": SARIF_LEVELS[rule["conf"]]},
                    "help": {"text": rule["full"]},
                } for rule in RULES],
            }},
            "columnKind": "unicodeCodePoints",
            "results": sarif_results,
        }],
    }


def json_document(results: Sequence[tuple[str, "LeakFinder"]]) -> list[dict]:
    """Findings as plain JSON, one object per finding.

    Flatter than SARIF and easier to grep or pipe into a script; SARIF is
    the format to hand a tool that already reads SARIF.
    """
    return [
        {
            "path": relative_uri(fpath),
            "line": finding.line,
            "column": finding.col + 1,
            "end_column": finding.end_col + 1,
            "rule": finding.rule,
            "pattern": finding.pattern,
            "confidence": finding.conf,
            "message": finding.msg,
        }
        for fpath, finder in results
        for finding in sort_findings(finder.findings)
    ]


def analyze_source(src: str, fpath: str = "<source>") -> LeakFinder | None:
    """Analyse source text and return the finder holding its findings.

    Returns None if the source will not parse or nests deeper than the
    recursion limit, printing why in both cases; a caller scanning many files
    should skip a None rather than stop. `fpath` is used only in those
    messages and in the findings' locations - nothing is read from disk.
    """
    try:
        tree = ast.parse(src, filename=fpath)
    except SyntaxError as error:
        warn(f"could not parse {fpath}: {error}")
        return None

    lines = src.splitlines()
    finder = LeakFinder(ignored_lines(src), lines)
    try:
        annotate_parents(tree)
        finder.collect_top_level_names(tree)
        finder.visit(tree)
    except RecursionError:
        warn(f"could not walk {fpath}: nested deeper than the recursion limit")
        return None

    return finder


def analyze_file(fpath: str, quiet: bool = False) -> LeakFinder | None:
    """Analyse one file, printing its findings unless `quiet`.

    Returns the finder, or None if the file could not be read, decoded or
    parsed - the reason is printed either way. `quiet` suppresses only the
    findings; the caller still gets them on the returned finder, which is how
    the CLI filters by confidence before anything reaches the terminal.
    """
    try:
        with open(fpath, encoding="utf-8") as source_file:
            src = source_file.read()
    except (OSError, UnicodeDecodeError) as error:
        warn(f"could not read {fpath}: {error}")
        return None

    finder = analyze_source(src, fpath)
    if finder is not None and not quiet:
        print_findings(fpath, finder.findings, finder.lines)

    return finder


def matches_exclude(fpath: str, patterns: Iterable[str]) -> bool:
    """True if a path should be skipped.

    A pattern matches the whole relative path (`corpus/*`), the file name
    (`*_test.py`), or any one directory along the way (`notebooks`), so the
    obvious spelling of an exclusion works without the caller having to know
    which of the three the tool wanted.
    """
    if not patterns:
        return False

    segments = relative_uri(fpath).split("/")
    relative = "/".join(segments)
    for pattern in patterns:
        if fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(segments[-1], pattern):
            return True
        if any(fnmatch.fnmatch(segment, pattern) for segment in segments[:-1]):
            return True
    return False


def find_py_files(path: str, exclude: Iterable[str] = ()) -> list[str]:
    """Every .py file under a path, sorted, minus the excluded ones.

    A file path is returned as itself, so a caller need not care which it was
    given. Directories in SKIP_DIRS are pruned along with anything matching
    `exclude`. Symlinks are not followed and non-.py files are ignored, so a
    notebook holding the same leak is not seen.
    """
    if os.path.isfile(path):
        return [] if matches_exclude(path, exclude) else [path]

    found = []
    for directory, subdirs, files in os.walk(path):
        subdirs[:] = [subdir for subdir in subdirs
                      if subdir not in SKIP_DIRS
                      and not matches_exclude(os.path.join(directory, subdir), exclude)]
        for filename in sorted(files):
            if filename.endswith(".py"):
                fpath = os.path.join(directory, filename)
                if not matches_exclude(fpath, exclude):
                    found.append(fpath)
    return found


def format_duration(seconds: float) -> str:
    return f"{seconds:.1f}s" if seconds >= 1 else f"{seconds:.2f}s"


def print_summary(total_high: int, total_low: int, total_shifts: int,
                  total_unresolvable_shifts: int, file_count: int = 0,
                  found_count: int | None = None, seconds: float = 0.0,
                  stream: TextIO | None = None) -> None:
    """The closing block: what was scanned, what was found, what was skipped.

    Prints "scanned 589 of 590 files" when some file could not be read or
    parsed, so a run that quietly analysed less than it was pointed at says
    so in the place a reader is already looking.

    The unresolvable-shift rate is the tool's own blind spot - a shift whose
    amount is a variable the tool could not resolve is neither cleared nor
    flagged - and printing it every run keeps a low finding count from
    reading as a clean bill of health.
    """
    stream = stream or sys.stdout
    if found_count is None:
        found_count = file_count
    files = "file" if found_count == 1 else "files"
    counted = f"{file_count}" if file_count == found_count else f"{file_count} of {found_count}"

    print(file=stream)
    print(f"scanned {counted} {files} in {format_duration(seconds)}", file=stream)
    print(file=stream)

    high_text = f"{total_high} high".ljust(12)
    low_text = f"{total_low} low"
    if total_high:
        high_text = paint(high_text, *CONFIDENCE_STYLE["high"])
    if total_low:
        low_text = paint(low_text, *CONFIDENCE_STYLE["low"])
    print(f"  {high_text}{low_text}", file=stream)

    if total_shifts > 0:
        percent = round(100 * total_unresolvable_shifts / total_shifts)
        print(f"  {total_shifts} shift calls, {total_unresolvable_shifts} "
              f"unresolvable ({percent}%)", file=stream)


def build_parser() -> argparse.ArgumentParser:
    """The command line parser, kept separate so tests can drive main(argv)."""
    parser = argparse.ArgumentParser(
        prog="leakcheck",
        description="Flag lookahead bias in pandas/numpy backtest code.",
        epilog=(f"add a `# {IGNORE_COMMENT}` comment on a line to silence it.\n"
                "exits 1 when anything was found, 0 when clean, so it can gate "
                "a CI job or a pre-commit hook."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="*", metavar="PATH",
                        help="files or directories to scan")
    parser.add_argument("--format", dest="output_format", default="text",
                        choices=["text", "json", "sarif"],
                        help="text carets (default), flat json, or SARIF 2.1.0 "
                             "for GitHub code scanning")
    parser.add_argument("--min-confidence", default="low", choices=["high", "low"],
                        help="lowest tier to report; 'high' drops the low tier "
                             "from both the output and the exit code "
                             "(default: low, meaning report everything)")
    parser.add_argument("--exclude", action="append", default=[], metavar="PATTERN",
                        help="skip paths matching this glob; repeatable")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress the summary block, leaving only findings")
    parser.add_argument("--version", action="version", version=f"{TOOL_NAME} {VERSION}")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Scan the given paths and report. Returns 1 if anything was found.

    The exit code counts only findings that survive --min-confidence, so a
    run filtered to the high tier stays green on low-tier findings.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # only the text format is for human eyes; escapes in a json or sarif
    # document would break whatever is parsing it
    global USE_COLOR
    USE_COLOR = args.output_format == "text" and color_enabled(sys.stdout)

    if not args.paths:
        parser.print_help()
        return 0

    started = time.monotonic()

    fpaths = []
    for path in args.paths:
        fpaths.extend(find_py_files(path, args.exclude))

    results = []
    total_high = 0
    total_low = 0
    total_shifts = 0
    total_unresolvable_shifts = 0

    for fpath in fpaths:
        finder = analyze_file(fpath, quiet=True)
        if finder is None:
            continue

        # the tier filter drops findings before anything counts or prints them,
        # so --min-confidence high also keeps the exit code from tripping on a
        # low-confidence finding
        finder.findings = [finding for finding in finder.findings
                           if RANK[finding.conf] <= RANK[args.min_confidence]]

        for finding in finder.findings:
            if finding.conf == "high":
                total_high = total_high + 1
            else:
                total_low = total_low + 1
        total_shifts = total_shifts + finder.shift_count
        total_unresolvable_shifts = total_unresolvable_shifts + finder.unknown_shifts
        results.append((fpath, finder))

    if args.output_format == "sarif":
        json.dump(sarif_document(results), sys.stdout, indent=2)
        print()
    elif args.output_format == "json":
        json.dump(json_document(results), sys.stdout, indent=2)
        print()
    else:
        for fpath, finder in results:
            print_findings(fpath, finder.findings, finder.lines)

    if not args.quiet:
        # a summary on stdout would make the json and sarif documents
        # unparseable, so for those it goes to stderr and the document has
        # stdout to itself
        stream = sys.stdout if args.output_format == "text" else sys.stderr
        print_summary(total_high, total_low, total_shifts, total_unresolvable_shifts,
                      len(results), len(fpaths), time.monotonic() - started, stream)

    # nonzero when anything was found, so this can gate a commit or a CI job
    return 1 if total_high or total_low else 0


if __name__ == "__main__":
    sys.exit(main())
