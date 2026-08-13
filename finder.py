import ast
import sys


# shitty way of thinking for now but common methdo names
# The value is the explanation we print when we find one.
LEAKY_METHODS = {
    "bfill":    "backward fill pulls future values into earlier rows",
    "backfill": "backward fill pulls future values into earlier rows",
}


class LeakFinder(ast.NodeVisitor):
    """Finds method calls that leak future data into the past."""

    def __init__(self, file_path):
        self.file_path = file_path

        # Collect what we find instead of printing straight away, each finding is a pair: (line number, method name)
        #use latr
        self.findings = []

    def visit_Call(self, node):
        is_method_call = isinstance(node.func, ast.Attribute)

        if is_method_call:
            method_name = node.func.attr

            if method_name in LEAKY_METHODS:
                self.findings.append((node.lineno, method_name))

        self.generic_visit(node)


def analyze_file(file_path):
    """Check one file. Returns how many problems were found."""

    try:
        with open(file_path, encoding="utf-8") as source_file:
            source_code = source_file.read()
    except (OSError, UnicodeDecodeError) as error:
        print(f"could not read {file_path}: {error}")
        return 0

    try:
        tree = ast.parse(source_code, filename=file_path)
    except SyntaxError as error:
        print(f"could not parse {file_path}: {error}")
        return 0

    finder = LeakFinder(file_path)
    finder.visit(tree)

    for line_number, method_name in finder.findings:
        explanation = LEAKY_METHODS[method_name]
        print(f"{file_path}:{line_number}  {method_name}()  {explanation}")

    return len(finder.findings)


file_paths = sys.argv[1:]

if len(file_paths) == 0:
    print("usage: python finder.py <file1.py> <file2.py> ...")
else:
    total_findings = 0
    files_checked = 0

    for path in file_paths:
        total_findings = total_findings + analyze_file(path)
        files_checked = files_checked + 1

    print()
    print(f"checked {files_checked} files, found {total_findings} problems")