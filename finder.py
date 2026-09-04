#Finds pandas calls that leak future data into past rows
#

import ast
import os
import sys
from pathlib import Path

rank = {"high": 0, "low": 1}

backfillmethods = ["bfill", "backfill"]
aggregatemethods = ["mean", "std", "max", "min", "sum", "median", "var"]
#if one of these ran first, an aggregate after it only looks backward by construction
windowedmethods = ["rolling", "expanding", "ewm"]
skipdirs = ["venv", "site-packages", ".git", "node_modules"]


class Finding:
  def __init__(self, line, pattern, msg, conf):
    self.line = line
    self.pattern = pattern
    self.msg = msg
    self.conf = conf


#walks the code looking for risky pandas calls
class LeakFinder(ast.NodeVisitor):

  def __init__(self, fpath):
    self.fpath = fpath
    self.findings = []
    self.shiftcount = 0
    self.unknownshifts = 0

  def addFinding(self, line, pattern, message, confidence):
    self.findings.append(Finding(line, pattern, message, confidence))

  def visit_Call(self, node):
    #only care about method calls like df.something(), plain functions dont matter here
    if(not isinstance(node.func, ast.Attribute)):
      self.generic_visit(node)
      return

    name = node.func.attr

    if(name in backfillmethods):
      self.addFinding(node.lineno, name, "backward fill pulls future values into earlier rows", "high")
    elif(name == "shift"):
      self.checkShift(node)
    elif(name == "rolling"):
      self.checkRolling(node)
    elif(name in aggregatemethods):
      self.checkAggregate(node, name)

    #keep going so calls nested inside this one still get checked
    self.generic_visit(node)

  def checkShift(self, node):
    self.shiftcount = self.shiftcount + 1

    #pandas accepts shift(-1) or shift(periods=-1)
    arg = None
    if(node.args):
      arg = node.args[0]
    else:
      for kw in node.keywords:
        if(kw.arg == "periods"):
          arg = kw.value

    #-1 isn't a single Constant node, it's a UnaryOp that splits the - and the 1
    shiftamount = None
    if(isinstance(arg, ast.Constant) and isinstance(arg.value, int)):
      shiftamount = arg.value
    elif(isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub)):
      if(isinstance(arg.operand, ast.Constant) and isinstance(arg.operand.value, int)):
        shiftamount = -arg.operand.value

    if(shiftamount is None):
      #cant know the sign without running it, e.g shift(x)
      self.unknownshifts = self.unknownshifts + 1
      return

    if(shiftamount < 0):
      message = f"shift({shiftamount}) pulls {abs(shiftamount)} future rows backward"
      self.addFinding(node.lineno, "shift", message, "high")

  def checkRolling(self, node):
    for kw in node.keywords:
      #only a literal True counts here, center=some_flag we cant read
      if(kw.arg == "center" and isinstance(kw.value, ast.Constant) and kw.value.value is True):
        self.addFinding(node.lineno, "rolling", "rolling(center=True) centers the window on future rows", "high")

  def checkAggregate(self, node, name):
    recv = node.func.value
    if(isinstance(recv, ast.Call) and isinstance(recv.func, ast.Attribute) and recv.func.attr in windowedmethods):
      return
    #could feed a decision or just be a printout, cant tell from the tree so low confidence
    message = f"{name}() over the whole series pulls later rows into earlier decisions"
    self.addFinding(node.lineno, name, message, "low")


def getRank(f):
  return(rank[f.conf])


def sortFindings(findings):
  return(sorted(findings, key=getRank))


def printFindings(fpath, findings):
  for f in sortFindings(findings):
    print(f"{fpath}:{f.line} [{f.conf}] {f.pattern} {f.msg}")


def analyzeFile(fpath, quiet=False):
  try:
    src = Path(fpath).read_text(encoding="utf-8")
  except (OSError, UnicodeDecodeError) as e:
    print(f"could not read {fpath}: {e}")
    return(None)

  try:
    tree = ast.parse(src, filename=fpath)
  except SyntaxError as e:
    print(f"could not parse {fpath}: {e}")
    return(None)

  finder = LeakFinder(fpath)
  finder.visit(tree)

  if(not quiet):
    printFindings(fpath, finder.findings)

  return(finder)


def findPyFiles(path):
  if(os.path.isfile(path)):
    return([path])

  found = []
  for d, subdirs, files in os.walk(path):
    #have to mutate subdirs in place or os.walk keeps going into skipped folders
    kept = []
    for s in subdirs:
      if(s not in skipdirs):
        kept.append(s)
    subdirs[:] = kept

    for fn in files:
      if(fn.endswith(".py")):
        found.append(os.path.join(d, fn))
  return(found)


def printSummary(nfindings, nshifts, nbad):
  print()
  print(f"{nfindings} findings")
  if(nshifts > 0):
    pct = round(100 * nbad / nshifts)
    print(f"{nshifts} shift calls, {nbad} unreadable ({pct}%)")


def main():
  args = sys.argv[1:]
  quiet = "--quiet" in args

  paths = []
  for a in args:
    if(a != "--quiet"):
      paths.append(a)

  if(not paths):
    print("usage: python finder.py [--quiet] <file_or_directory> ...")
    return

  fpaths = []
  for p in paths:
    fpaths.extend(findPyFiles(p))

  nfindings = 0
  nshifts = 0
  nbad = 0

  for fpath in fpaths:
    finder = analyzeFile(fpath, quiet=quiet)
    if(finder is None):
      continue
    nfindings = nfindings + len(finder.findings)
    nshifts = nshifts + finder.shiftcount
    nbad = nbad + finder.unknownshifts

  printSummary(nfindings, nshifts, nbad)


#this file gets imported by run_tests.py too, so main() only runs when finder.py is called directly
if __name__ == "__main__":
  main()
