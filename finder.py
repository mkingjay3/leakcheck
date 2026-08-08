import ast, sys

class CallVisitor(ast.NodeVisitor):
    def __init__(self, path): self.path = path
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            print(f"{self.path}:{node.lineno}  .{node.func.attr}()")
        self.generic_visit(node)

for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    CallVisitor(path).visit(tree)
