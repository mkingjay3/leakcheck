#Runs the leak finder against the sample files in tests/cases and checks the results against tests/expected.json
#

import json
import os
from finder import analyzeFile

casesdir = "tests/cases"
expectedfile = "tests/expected.json"


def loadExpected():
  f = open(expectedfile)
  expected = json.load(f)
  f.close()
  return(expected)


def runOne(filename):
  #returns (has_high, has_low) for one test file
  path = os.path.join(casesdir, filename)
  finder = analyzeFile(path, quiet=True)

  if(finder is None):
    return(None, None)

  confs = []
  for finding in finder.findings:
    confs.append(finding.conf)

  hashigh = "high" in confs
  haslow = "low" in confs
  return(hashigh, haslow)


def main():
  expected = loadExpected()

  #counters for high confidence findings
  truepos = 0
  falsepos = 0
  falseneg = 0

  failures = []

  for filename, want in sorted(expected.items()):
    gothigh, gotlow = runOne(filename)

    if(gothigh is None):
      failures.append(f"{filename}: could not analyze")
      continue

    wanthigh = want["expect_high"]

    if(wanthigh and gothigh):
      truepos = truepos + 1
    elif(wanthigh and not gothigh):
      falseneg = falseneg + 1
      failures.append(f"{filename}: expected a high finding, got none")
    elif(not wanthigh and gothigh):
      falsepos = falsepos + 1
      failures.append(f"{filename}: flagged high, shouldn't have")

    if("expect_low" in want):
      if(want["expect_low"] != gotlow):
        failures.append(f"{filename}: low finding expected={want['expect_low']} got={gotlow}")

  print(f"{len(expected)} cases")
  print(f"true positives  {truepos}")
  print(f"false positives {falsepos}")
  print(f"false negatives {falseneg}")

  if(truepos + falsepos > 0):
    precision = truepos / (truepos + falsepos)
    print(f"precision {round(precision * 100)}%")

  if(truepos + falseneg > 0):
    recall = truepos / (truepos + falseneg)
    print(f"recall    {round(recall * 100)}%")

  if(failures):
    print()
    for line in failures:
      print("  " + line)


#no other file imports run_tests.py, so just call main() straight away
main()
