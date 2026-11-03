"""Refuse a tracked file that looks like comment data.

    python tests/check_no_personal_data.py          # scans `git ls-files`; exit 1 on a problem

Comment text, comment ids and author hashes are personal data under UK GDPR (CLAUDE.md
section 10) and must never be committed. `.gitignore` alone does not enforce that: a file
added through GitHub's web "Add files via upload" ignores it, and so does `git add -f`.
This runs in CI on every push and pull request, so such a file is caught on the commit
that adds it.

It checks structure, never content worth quoting, and prints only paths and counts:
  * any tracked file under a data folder (official-data/, data/, archive/, Full Corpus Data/);
  * any tracked spreadsheet, JSONL cache, backup or run record (*.run.json carries the
    example comment ids);
  * any CSV outside tests/ whose header has `body` together with `comment_id` or
    `author_hash` (the corpus schema);
  * any text file outside tests/ holding 3 or more distinct YouTube comment ids;
  * a dashboard export that says it includes verbatim comments.
"""
import csv
import io
import json
import os
import re
import subprocess
import sys

DATA_DIRS = ("official-data/", "data/", "archive/", "Full Corpus Data/")
FORBIDDEN_SUFFIXES = (".jsonl", ".xlsx", ".xls", ".xlsm", ".ods", ".tsv", ".parquet",
                      ".bak", ".run.json")
YOUTUBE_COMMENT_ID = re.compile(r"\bUg[wxyz][A-Za-z0-9_-]{20,}")
MAX_YOUTUBE_IDS = 2

# Files already on main when this check was added. They are comment data and are listed
# here ONLY so the check can go in without failing every build while the decision on
# removing them is made. Remove each line when its file is removed; never add one.
LEGACY_TRACKED = {
    "Full Corpus Data/Summary.md",
    "Full Corpus Data/Temp",
    "Full Corpus Data/full_corpus_jets.csv",
    "Full Corpus Data/full_corpus_jets_exclusions.csv",
    "Full Corpus Data/full_corpus_jets_videos.csv",
    "Full Corpus Data/full_corpus_luigi.csv",
    "Full Corpus Data/full_corpus_luigi_exclusions.csv",
    "Full Corpus Data/full_corpus_luigi_videos.csv",
    "Full Corpus Data/full_corpus_musk.csv",
    "Full Corpus Data/full_corpus_musk_exclusions.csv",
    "Full Corpus Data/full_corpus_musk_videos.csv",
    "Full Corpus Data/full_corpus_propublica.csv",
    "Full Corpus Data/full_corpus_propublica_exclusions.csv",
    "Full Corpus Data/full_corpus_propublica_videos.csv",
}


def _text(path):
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError:
        return None
    if b"\x00" in raw[:4096]:
        return None  # binary (a PDF, an image)
    return raw.decode("utf-8-sig", errors="replace")


def problems_in(root, paths, legacy=frozenset(LEGACY_TRACKED)):
    """[(path, reason)] for every tracked path that looks like comment data."""
    found = []
    for path in sorted(paths):
        norm = path.replace(os.sep, "/")
        if norm in legacy:
            continue
        lower = norm.lower()
        if norm.startswith(DATA_DIRS) or "/official-data/" in norm:
            found.append((norm, "inside a data folder"))
            continue
        if lower.endswith(FORBIDDEN_SUFFIXES):
            found.append((norm, "a spreadsheet, cache, backup or run record"))
            continue
        if norm.startswith("tests/") or (norm.startswith("examples/synthetic_") and lower.endswith(".csv")):
            continue  # synthetic fixtures and the docs' made-up examples live here
        text = _text(os.path.join(root, path))
        if text is None:
            continue
        if lower.endswith(".csv"):
            header = next(csv.reader(io.StringIO(text.split("\n", 1)[0])), [])
            columns = {c.strip().lower() for c in header}
            if "body" in columns and columns & {"comment_id", "author_hash"}:
                found.append((norm, "a CSV with the comment schema (body + comment_id/author_hash)"))
                continue
        ids = set(YOUTUBE_COMMENT_ID.findall(text))
        if len(ids) > MAX_YOUTUBE_IDS:
            found.append((norm, f"{len(ids)} YouTube comment ids"))
            continue
        if norm == "dashboard/data/results.json":
            try:
                verbatim = (json.loads(text).get("provenance") or {}).get(
                    "verbatim_comments_included")
            except ValueError:
                verbatim = None
            if verbatim:
                found.append((norm, "exported with --include-verbatim"))
    return found


def tracked_files(root):
    out = subprocess.run(["git", "-C", root, "ls-files", "-z"], capture_output=True, check=True)
    return [p for p in out.stdout.decode("utf-8").split("\0") if p]


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    found = problems_in(root, tracked_files(root))
    if not found:
        print("No tracked file looks like comment data.")
        return 0
    print("These tracked files look like comment data (personal data: keep them out of git):")
    for path, reason in found:
        print(f"  {path}: {reason}")
    print("Remove them with `git rm --cached <path>` and keep them in official-data/.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
