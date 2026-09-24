"""Clone the corpora every number in baseline.txt was measured against.

  python3 fetch_corpora.py            everything in corpora.json
  python3 fetch_corpora.py holdout    one group
  python3 fetch_corpora.py --check    report what's present, whether it sits
                                      on the pinned commit, and whether it
                                      still produces the recorded counts

Each repo is checked out at a pinned commit. Without that the corpora drift
and every recorded count quietly stops meaning anything, which matters more
here than usual because the findings were hand-audited line by line.
"""

import json
import os
import subprocess
import sys

import finder

MANIFEST = "corpora.json"


def load_groups():
    with open(MANIFEST) as f:
        return json.load(f)["groups"]


def run(args, **kwargs):
    return subprocess.run(args, capture_output=True, text=True, **kwargs)


def head_of(path):
    finished = run(["git", "-C", path, "rev-parse", "HEAD"])
    if finished.returncode != 0:
        return None
    return finished.stdout.strip()


# a depth-1 clone can't check out an arbitrary older commit, so fetch that one
# commit directly; falls back to a full clone on servers that refuse it
def clone_at(url, commit, path):
    os.makedirs(path, exist_ok=True)
    for args in (["git", "-C", path, "init", "-q"],
                 ["git", "-C", path, "remote", "add", "origin", url]):
        run(args)

    fetched = run(["git", "-C", path, "fetch", "-q", "--depth", "1", "origin", commit])
    if fetched.returncode != 0:
        run(["git", "-C", path, "fetch", "-q", "origin"])

    checked_out = run(["git", "-C", path, "checkout", "-q", commit])
    return checked_out.returncode == 0, checked_out.stderr.strip()


def fetch_group(directory, group):
    print(f"{directory}/  ({group['role']})")
    for repo in group["repos"]:
        path = os.path.join(directory, repo["name"])
        present = head_of(path)

        if present == repo["commit"]:
            print(f"  ok       {repo['name']}")
            continue
        if present is not None:
            print(f"  WRONG    {repo['name']} is at {present[:12]}, "
                  f"pinned at {repo['commit'][:12]} - remove it to refetch")
            continue

        print(f"  fetching {repo['name']} ...", end=" ", flush=True)
        ok, error = clone_at(repo["url"], repo["commit"], path)
        print("done" if ok else f"FAILED ({error})")


def count_findings(directory):
    counts = {"high": 0, "low": 0}
    for fpath in finder.find_py_files(directory):
        analysed = finder.analyze_file(fpath, quiet=True)
        if analysed is None:
            continue
        for found in analysed.findings:
            counts[found.conf] = counts[found.conf] + 1
    return counts


def check_group(directory, group):
    print(f"{directory}/  ({group['role']})")

    all_pinned = True
    for repo in group["repos"]:
        present = head_of(os.path.join(directory, repo["name"]))
        if present is None:
            state = "missing"
        elif present == repo["commit"]:
            state = "pinned"
        else:
            state = f"drifted to {present[:12]}"
        all_pinned = all_pinned and state == "pinned"
        print(f"  {state:<22} {repo['name']}")

    expected = group.get("expected")
    if not all_pinned or expected is None:
        return all_pinned

    # the corpus being at the right commits only matters because of what it
    # was measured to produce, so check that too
    found = count_findings(directory)
    if found == expected:
        print(f"  findings match          {found['high']} high, {found['low']} low")
        return True

    print(f"  FINDINGS CHANGED        expected {expected['high']} high / "
          f"{expected['low']} low, got {found['high']} high / {found['low']} low")
    print("                          update corpora.json and baseline.txt, or "
          "work out what regressed")
    return False


def main():
    args = sys.argv[1:]
    checking = "--check" in args
    wanted = [arg for arg in args if not arg.startswith("--")]

    groups = load_groups()
    ok = True
    for directory, group in groups.items():
        if wanted and directory not in wanted:
            continue
        if checking:
            ok = check_group(directory, group) and ok
        else:
            fetch_group(directory, group)
        print()

    if not checking:
        print("now: python3 finder.py corpus")
        return 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
