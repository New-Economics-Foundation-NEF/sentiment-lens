"""Concatenate per-video/per-thread collector CSVs into one master file.

    python merge.py data/luigi_yt.csv data/musk_yt.csv data/jets_yt.csv \\
        --output data/master.csv

Every collector (youtube_api.py, youtube_scrape.py, reddit_scrape.py) writes
the same columns in the same order, so combining them is meant to be exactly
this: read each file, keep every row, write one file. This script only adds
two things a plain concatenation would not: it checks every input actually
has the columns the rest of the pipeline expects before writing anything, and
it drops a row whose comment_id has already been seen rather than silently
double-counting a comment collected twice (e.g. two runs over the same
video, or overlapping search results).

Imports FIELDNAMES from ewl_common rather than keeping a second copy in step
by hand. That used to require EWL_AUTHOR_SALT just to merge already-collected
files, since ewl_common raised on import without it — fixed by moving that
check to the point hash_author() is actually called, which this script never
does.
"""
import argparse
import collections
import csv
import os
import sys

from ewl import corpus as corpus_mod
from ewl.common import FIELDNAMES

# Columns a valid input may predate. See the note where they are checked.
OPTIONAL_COLUMNS = {"detected_lang"}


def cap_per_author(rows, max_per_author, scope="global"):
    """Keep at most `max_per_author` comments from each author_hash.

    `scope="global"` (the default, and what produced the frozen 84,250-row
    corpus) counts an author's comments across every story, so which story loses
    their later comments depends on merge order. `scope="story"` counts within
    each story separately, so merge order cannot matter (it keeps 10 more rows
    on the frozen corpus: 84,260).

    Off by default -- see the call site in main(). Without some cap, one
    prolific commenter's opinion can be counted dozens of times over as if it
    were dozens of independent people, which is a real threat to every
    proportion this project reports: every downstream statistic (harm
    breakdowns, sentiment shares, bootstrap intervals) treats each row as one
    independent observation. Rows are kept in their existing order, so which
    comments survive the cap depends on the corpus's own order, not a fresh
    random draw.

    Blank `author_hash` (deleted/removed-user comments; see
    ewl_common.hash_author) is never capped, since an empty hash does not
    identify one person and capping it would just discard early deleted
    comments in every file.
    """
    if scope not in ("global", "story"):
        raise ValueError(f"scope must be 'global' or 'story', not {scope!r}")
    counts = collections.Counter()
    kept = []
    for row in rows:
        author = (row.get("author_hash") or "").strip()
        key = (row.get("story", ""), author) if scope == "story" else author
        if not author or counts[key] < max_per_author:
            kept.append(row)
            if author:
                counts[key] += 1
    return kept


def drop_duplicate_bodies(rows):
    """Drop rows whose comment body has already been seen IN THE SAME STORY,
    keeping the first.

    Off by default -- see the call site in main(). The comment_id dedupe
    above only catches the same comment collected twice under the same ID
    (e.g. overlapping search results); it does not catch copy-pasted or
    cross-posted text arriving under a different comment_id, which this does.
    Bodies are compared with whitespace collapsed, so trivial formatting
    differences (a doubled space, a stray newline) do not defeat it -- but
    the comparison is otherwise exact, so two genuinely different comments
    that happen to share text (e.g. two people separately typing "this") are
    also merged into one, *within a story*.

    Scoped per story on purpose -- fixed 2026-09-16 from an earlier version
    that tracked `seen_bodies` in one set shared across the whole merge.
    Measured on the real four-story corpus: that version dropped rows purely
    because a short, generic reaction ("Yes", "Lol", a single emoji) already
    appeared under an EARLIER-merged, unrelated story -- independent people
    reacting to different videos about different news events, not the
    copy-paste/cross-post case this function exists to catch. Because
    stories are always merged in the same fixed order, the loss fell
    systematically on whichever story was listed later on the command line,
    worst for the smallest ones. Scoping by story removes that cross-story
    bias while keeping the within-story behaviour (which does still
    legitimately catch copy-paste/cross-posting) unchanged.
    """
    seen_bodies_by_story = collections.defaultdict(set)
    kept = []
    for row in rows:
        body = " ".join((row.get("body") or "").split())
        story = row.get("story") or ""
        seen = seen_bodies_by_story[story]
        if body and body in seen:
            continue
        if body:
            seen.add(body)
        kept.append(row)
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="Per-video/per-thread CSVs to combine")
    parser.add_argument("--output", required=True, help="Path to write the combined CSV")
    parser.add_argument("--no-manifest", action="store_true",
                        help="Skip writing the .manifest.json that freezes this corpus. Only "
                             "for a throwaway file you do not intend to analyse.")
    parser.add_argument("--cap-scope", choices=["global", "story"], default="global",
                        help="--cap-per-author: count an author's comments across all stories "
                             "(global, the default and what built the frozen corpus) or within "
                             "each story. 'story' makes the result independent of merge order.")
    parser.add_argument("--cap-per-author", type=int, default=None,
                        help="Keep at most this many comments per author_hash (see "
                             "cap_per_author's docstring for why). Off by default: a plain "
                             "`ewl-merge` with no flags reproduces exactly what it always did.")
    parser.add_argument("--dedupe-bodies", action="store_true",
                        help="Also drop rows whose comment body duplicates an earlier one IN "
                             "THE SAME STORY (see drop_duplicate_bodies's docstring). Off by "
                             "default, same reasoning as --cap-per-author.")
    args = parser.parse_args()

    seen_ids = set()
    rows = []
    duplicates = 0
    per_file = collections.Counter()

    for path in args.inputs:
        try:
            handle = open(path, newline="", encoding="utf-8-sig")
        except OSError as error:
            sys.exit(f"{path}: {error}")
        with handle:
            reader = csv.DictReader(handle)
            # detected_lang was added after some CSVs were already collected
            # (#23). A file predating it is still perfectly good data, so it is
            # optional here and comes through blank rather than being rejected
            # -- re-collecting a corpus to gain one column nobody had yet would
            # be a silly reason to spend quota.
            required = set(FIELDNAMES) - OPTIONAL_COLUMNS
            missing = required - set(reader.fieldnames or [])
            if missing:
                sys.exit(f"{path}: missing columns {sorted(missing)}. Expected the schema "
                         f"every collector writes: {FIELDNAMES}")
            for row in reader:
                comment_id = (row.get("comment_id") or "").strip()
                if comment_id and comment_id in seen_ids:
                    duplicates += 1
                    continue
                if comment_id:
                    seen_ids.add(comment_id)
                rows.append(row)
                per_file[path] += 1

    if not rows:
        sys.exit("No rows to write: every input was empty or entirely duplicates.")

    # Both off by default: a plain `ewl-merge <files> --output x` reproduces
    # exactly what it always did, and the manifest below records whether
    # either ran on THIS corpus, so a later reader (or a re-merge) does not
    # have to guess -- this replaces the two "uncomment to..." comments that
    # used to sit here, which meant the real corpus (produced by manually
    # uncommenting them) could not be reproduced from the command CLAUDE.md
    # documents, and nothing on disk said the manual step had ever happened.
    n_before_cap = len(rows)
    if args.cap_per_author is not None:
        rows = cap_per_author(rows, max_per_author=args.cap_per_author, scope=args.cap_scope)

    n_before_dedupe = len(rows)
    if args.dedupe_bodies:
        rows = drop_duplicate_bodies(rows)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Merged {len(args.inputs)} files into {args.output}:", file=sys.stderr)
    for path in args.inputs:
        print(f"  {path}: {per_file[path]} rows", file=sys.stderr)
    if duplicates:
        print(f"Dropped {duplicates} row(s) whose comment_id had already been seen "
              f"in an earlier file (the same comment collected more than once).",
              file=sys.stderr)
    if args.cap_per_author is not None:
        print(f"Dropped {n_before_cap - n_before_dedupe} row(s) over the per-author cap "
              f"(max_per_author={args.cap_per_author}) -- these commenters posted more than "
              f"{args.cap_per_author} times each in this corpus.", file=sys.stderr)
    if args.dedupe_bodies:
        print(f"Dropped {n_before_dedupe - len(rows)} row(s) with duplicate comment text seen "
              f"under a different comment_id (copy-pasted or cross-posted comments).",
              file=sys.stderr)
    print(f"Total: {len(rows)} rows written to {args.output}", file=sys.stderr)

    # Freeze it. Collection is irreversible -- yesterday's comments cannot be
    # collected again -- so the corpus is written once and every later analysis
    # can check it has not moved. Without this, re-collecting silently changes
    # what an already published figure refers to.
    if not args.no_manifest:
        # cap_per_author/dedupe_bodies recorded here on purpose: this is
        # exactly the provenance gap that made the real corpus's row count
        # (documented in CLAUDE.md as one number, but on disk as another --
        # see the "dedup fix" investigation) unreproducible from the command
        # CLAUDE.md's own stage-2 example shows. A future reader of this
        # manifest can now see whether either ran, instead of having to
        # reverse-engineer it from row counts and dated backup files.
        path, manifest = corpus_mod.write_manifest(
            args.output, inputs=args.inputs,
            extra={"cap_per_author": args.cap_per_author, "cap_scope": args.cap_scope,
                   "dedupe_bodies": args.dedupe_bodies})
        print(f"\nFrozen: {os.path.basename(path)}", file=sys.stderr)
        print(f"  sha256 {manifest['sha256'][:16]}...  "
              f"{manifest['comments']} comments, {manifest['videos']} videos",
              file=sys.stderr)
        for story, count in manifest["comments_per_story"].items():
            print(f"    {story or '(no story)'}: {count}", file=sys.stderr)
        print("  Analysis commands check this. Re-freeze deliberately if the corpus "
              "is meant to change.", file=sys.stderr)


if __name__ == "__main__":
    main()
