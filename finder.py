import ast
import sys


class CallVisitor(ast.NodeVisitor):

   #constructor 
    def __init__(self, file_path):
      
        self.file_path = file_path

    def visit_Call(self, node):
        is_method_call = isinstance(node.func, ast.Attribute)

        if is_method_call:
            method_name = node.func.attr
            line_number = node.lineno

            print(f"{self.file_path}:{line_number}  .{method_name}()")
        self.generic_visit(node)


def analyze_file(file_path):

#try catch 
    try:
        with open(file_path, encoding="utf-8") as source_file:
            source_code = source_file.read()
    except (OSError, UnicodeDecodeError) as error:
        print(f"could not read {file_path}: {error}")
        return

    try:
        tree = ast.parse(source_code, filename=file_path)
    except SyntaxError as error:
        print(f"could not parse {file_path}: {error}")
        return

    visitor = CallVisitor(file_path)
    visitor.visit(tree)

file_paths = sys.argv[1:]

if len(file_paths) == 0:
    print("usage: python finder.py <file1.py> <file2.py> ...")
else:
    for path in file_paths:
        analyze_file(path)