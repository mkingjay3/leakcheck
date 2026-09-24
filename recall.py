"""Measure recall by injecting known leaks into real code.

Precision has three censuses behind it. Recall has had almost nothing: the
hand-written test cases, which I wrote, and four planted-bias strategies,
which is a denominator of four. Neither says whether the suppression rules -
windowing, resolved constants, compensating shifts, reduction to a scalar -
wrongly silence a real leak in code they were never tried against.

So take real call sites out of the corpora, turn each one into a leak that is
definitely a leak, and check the tool reports it. A miss here is a false
negative in code nobody wrote for this project.

  python3 recall.py                      every corpus in corpora.json
  python3 recall.py corpus strategies    named ones
  python3 recall.py --misses             print each miss with its source line

Mutations, all of which produce an unambiguous leak:

  shift(k) -> shift(-k)            a positive shift reindexed to a negative one
  ffill()  -> bfill()              forward fill to backward fill
  rolling(n) -> rolling(n, center=True)
  A.rolling(n).mean() -> A.mean()  only where A also appears bare in the
                                   surrounding arithmetic, so removing the
                                   window leaves a whole-series statistic
                                   broadcast over its own series
"""

import ast
import copy
import json
import os
import sys

import finder

MANIFEST = "corpora.json"
CANDIDATES_PER_FILE = 3

# each trial reparses, deep-copies, mutates and unparses the file, so the cost
# is a few full tree walks apiece. NostalgiaForInfinityX7.py is 3.5MB, which
# turns that into minutes for one site; skip the outliers and say how many
MAX_FILE_BYTES = 400_000


def is_call_to(node, name):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == name)


# Removing the window only produces the shape under test if the receiver also
# appears bare in the same arithmetic expression. Written out here rather than
# imported from finder, so the candidate set doesn't depend on the code being
# measured - and scoped to the enclosing statement, because a bare `series` in
# some other function is a different series.
def receiver_appears_bare(rolling_call):
    source = ast.dump(rolling_call.func.value)
    current = getattr(rolling_call, "parent", None)

    while current is not None and not isinstance(current, ast.stmt):
        if isinstance(current, (ast.BinOp, ast.Compare)):
            for inner in ast.walk(current):
                if inner is rolling_call.func.value:
                    continue
                if isinstance(inner, ast.expr) and ast.dump(inner) == source:
                    return True
        current = getattr(current, "parent", None)
    return False


def negate_shift(node):
    """shift(2) -> shift(-2)"""
    if not is_call_to(node, "shift") or not node.args:
        return None
    periods = finder.extract_int_literal(node.args[0])
    if periods is None or periods <= 0:
        return None

    def apply(clone):
        clone.args[0] = ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=periods))
    return "shift", apply


def backfill(node):
    """ffill() -> bfill()"""
    if not is_call_to(node, "ffill"):
        return None

    def apply(clone):
        clone.func.attr = "bfill"
    return "bfill", apply


def centre_window(node):
    """rolling(n) -> rolling(n, center=True)"""
    if not is_call_to(node, "rolling"):
        return None
    if any(keyword.arg == "center" for keyword in node.keywords):
        return None

    def apply(clone):
        clone.keywords.append(ast.keyword(arg="center", value=ast.Constant(value=True)))
    return "rolling", apply


def unwindow(node):
    """A - A.rolling(n).mean() -> A - A.mean()"""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return None
    if node.func.attr not in finder.AGGREGATE_METHODS:
        return None

    rolling = node.func.value
    if not (is_call_to(rolling, "rolling") and rolling.args):
        return None
    if not receiver_appears_bare(rolling):
        return None

    def apply(clone):
        clone.func.value = clone.func.value.func.value
    return node.func.attr, apply


def candidates(tree):
    found = []
    for node in ast.walk(tree):
        for mutation in (negate_shift(node), backfill(node), centre_window(node),
                         unwindow(node)):
            if mutation is not None:
                found.append((node, mutation[0], mutation[1]))
    return found


def pattern_counts(src, fpath):
    analysed = finder.analyze_source(src, fpath)
    if analysed is None:
        return None
    counts = {}
    for found in analysed.findings:
        counts[found.pattern] = counts.get(found.pattern, 0) + 1
    return counts


# findings are one per (line, pattern), so mutating a site on a line that
# already reports that pattern produces no observable change however well the
# detector works. Those sites are not measurable, so they are not counted.
def reported_sites(src, fpath):
    analysed = finder.analyze_source(src, fpath)
    if analysed is None:
        return set()
    return {(found.line, found.pattern) for found in analysed.findings}


# rebuild from source each time: the mutation is applied to a copy, and
# unparsing a shared tree twice would compound earlier edits. Copy before
# annotating parents, since parent links make the tree cyclic.
def mutate_and_check(src, fpath, index, before):
    tree = ast.parse(src, filename=fpath)
    clone_tree = copy.deepcopy(tree)
    finder.annotate_parents(tree)
    finder.annotate_parents(clone_tree)

    sites = candidates(tree)
    clone_sites = candidates(clone_tree)
    if index >= len(sites) or index >= len(clone_sites):
        return None

    node, pattern, apply = sites[index]
    apply(clone_sites[index][0])
    ast.fix_missing_locations(clone_tree)

    try:
        mutated = ast.unparse(clone_tree)
    except Exception:
        return None

    after = pattern_counts(mutated, fpath)
    if after is None:
        return None

    caught = after.get(pattern, 0) > before.get(pattern, 0)
    return pattern, caught, node.lineno


def run_corpus(directory, show_misses):
    results = {}
    misses = []
    skipped = 0
    unmeasurable = 0

    for fpath in finder.find_py_files(directory):
        try:
            if os.path.getsize(fpath) > MAX_FILE_BYTES:
                skipped = skipped + 1
                continue
            with open(fpath, encoding="utf-8") as f:
                src = f.read()
            tree = ast.parse(src, filename=fpath)
            finder.annotate_parents(tree)
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue

        try:
            sites = candidates(tree)
        except RecursionError:
            continue
        if not sites:
            continue

        before = pattern_counts(src, fpath)
        if before is None:
            continue
        already = reported_sites(src, fpath)

        for index in range(min(len(sites), CANDIDATES_PER_FILE)):
            node, pattern, _ = sites[index]
            if (node.lineno, pattern) in already:
                unmeasurable = unmeasurable + 1
                continue

            try:
                outcome = mutate_and_check(src, fpath, index, before)
            except RecursionError:
                continue
            if outcome is None:
                continue
            pattern, caught, lineno = outcome

            tally = results.setdefault(pattern, [0, 0])
            tally[1] += 1
            if caught:
                tally[0] += 1
            else:
                misses.append(f"{fpath}:{lineno} [{pattern}]")

    print(f"{directory}/")
    total_caught = total_tried = 0
    for pattern in sorted(results):
        caught, tried = results[pattern]
        total_caught += caught
        total_tried += tried
        print(f"  {pattern:<8} {caught:>4}/{tried:<4}  {round(100 * caught / tried)}%")
    if total_tried:
        print(f"  {'all':<8} {total_caught:>4}/{total_tried:<4}  "
              f"{round(100 * total_caught / total_tried)}%")

    if skipped:
        print(f"  {'':<8} {skipped} files over {MAX_FILE_BYTES // 1000}KB not mutated")
    if unmeasurable:
        print(f"  {'':<8} {unmeasurable} sites already reported on their line, "
              f"so a mutation there is unobservable")

    if misses and show_misses:
        print(f"  misses:")
        for miss in misses:
            print(f"    {miss}")
    elif misses:
        print(f"  {len(misses)} missed, pass --misses to list them")
    print()
    return total_caught, total_tried


def main():
    args = sys.argv[1:]
    show_misses = "--misses" in args
    wanted = [arg for arg in args if not arg.startswith("--")]

    with open(MANIFEST) as f:
        groups = list(json.load(f)["groups"])

    caught = tried = 0
    for directory in groups:
        if wanted and directory not in wanted:
            continue
        group_caught, group_tried = run_corpus(directory, show_misses)
        caught += group_caught
        tried += group_tried

    if tried:
        print(f"overall {caught}/{tried} injected leaks caught "
              f"({round(100 * caught / tried)}%)")


if __name__ == "__main__":
    main()
