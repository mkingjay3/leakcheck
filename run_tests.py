#Runs the leak finder against the sample files in tests/cases and checks the results against tests/expected.json
#

import json
import os
from finder import analyzeFile

casesdir = "tests/cases"
expectedfile = "tests/expected.json"

tiers = ["high", "low"]


def loadExpected():
  f = open(expectedfile)
  expected = json.load(f)
  f.close()
  return(expected)


def runOne(filename):
  #returns {"high": bool, "low": bool}, whether each tier fired at least once in this file
  path = os.path.join(casesdir, filename)
  finder = analyzeFile(path, quiet=True)

  if(finder is None):
    return(None)

  confs = []
  for finding in finder.findings:
    confs.append(finding.conf)

  got = {}
  for tier in tiers:
    got[tier] = tier in confs
  return(got)


def main():
  expected = loadExpected()

  #true positives, false positives, false negatives, kept separately per tier
  truepos = {tier: 0 for tier in tiers}
  falsepos = {tier: 0 for tier in tiers}
  falseneg = {tier: 0 for tier in tiers}

  failures = []

  for filename, want in sorted(expected.items()):
    got = runOne(filename)

    if(got is None):
      failures.append(f"{filename}: could not analyze")
      continue

    for tier in tiers:
      wanttier = want.get(f"expect_{tier}", False)
      gottier = got[tier]

      if(wanttier and gottier):
        truepos[tier] = truepos[tier] + 1
      elif(wanttier and not gottier):
        falseneg[tier] = falseneg[tier] + 1
        failures.append(f"{filename}: expected a {tier} finding, got none")
      elif(not wanttier and gottier):
        falsepos[tier] = falsepos[tier] + 1
        failures.append(f"{filename}: flagged {tier}, shouldn't have")

  print(f"{len(expected)} cases")
  print()

  for tier in tiers:
    tp = truepos[tier]
    fp = falsepos[tier]
    fn = falseneg[tier]
    print(f"[{tier}]")
    print(f"  true positives  {tp}")
    print(f"  false positives {fp}")
    print(f"  false negatives {fn}")

    if(tp + fp > 0):
      precision = tp / (tp + fp)
      print(f"  precision {round(precision * 100)}%")
    else:
      print("  precision n/a (nothing flagged at this tier)")

    if(tp + fn > 0):
      recall = tp / (tp + fn)
      print(f"  recall    {round(recall * 100)}%")
    else:
      print("  recall    n/a (nothing expected at this tier)")
    print()

  if(failures):
    print("failures:")
    for line in failures:
      print("  " + line)


#no other file imports run_tests.py, so just call main() straight away
main()
