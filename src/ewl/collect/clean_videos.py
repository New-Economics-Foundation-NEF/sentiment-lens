"""Drop every comment belonging to a named list of videos from a collected comments
CSV, before merging.

    ewl-clean-videos data/full_corpus_jets.csv --exclude-file excluded_video_ids.txt \\
        --output data/full_corpus_jets_clean.csv

A video is identified by its 11-character ID, taken from the `source` URL every row
carries, and matched EXACTLY. This used to be a substring test, which hid a real
mistake: a list entry written without its leading hyphen ("dxftQC09H4" for the video
"-dxftQC09H4") still matched, by luck of the substring, and would have silently put
1,471 comments back the day the matching was tightened. An entry that is not a valid
video ID is now refused with a hint, and an ID that matches nothing is reported.

Reports before and after counts per video; keep that output for the methodology notes.
"""
import argparse
import csv
import os
import re
import sys

_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_FROM_URL = (re.compile(r"[?&]v=([A-Za-z0-9_-]{11})"),
             re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})"),
             re.compile(r"/(?:shorts|embed|live)/([A-Za-z0-9_-]{11})"))


def video_id(source):
    """The 11-character video ID in a watch/short/embed URL, or the text itself when it
    already is an ID, or None."""
    text = (source or "").strip()
    if _ID.match(text):
        return text
    for pattern in _FROM_URL:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def check_ids(ids, known_ids=()):
    """Refuse an entry that cannot be a video ID, hinting at the likely real one."""
    bad = [i for i in sorted(ids) if not _ID.match(i)]
    if not bad:
        return
    lines = []
    for entry in bad:
        near = sorted(k for k in known_ids if k.endswith(entry) or k.startswith(entry))
        hint = f" -- did you mean {', '.join(near)}?" if near else ""
        lines.append(f"  {entry!r} is {len(entry)} characters; a video ID is 11{hint}")
    raise SystemExit("These entries are not valid video IDs (a leading '-' is part of an ID, "
                     "and easy to drop in a spreadsheet or a shell):\n" + "\n".join(lines))


def drop_videos(rows, excluded_ids):
    """(kept rows, {video id: comments removed}), matching video IDs exactly."""
    excluded = set(excluded_ids)
    kept, removed = [], {vid: 0 for vid in excluded}
    for row in rows:
        vid = video_id(row.get("source", ""))
        if vid in excluded:
            removed[vid] += 1
        else:
            kept.append(row)
    return kept, removed


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv")
    parser.add_argument("--exclude", default="", help="Comma-separated video IDs to drop")
    parser.add_argument("--exclude-file", help="Text file, one video ID per line")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    excluded_ids = {v.strip() for v in args.exclude.split(",") if v.strip()}
    if args.exclude_file:
        with open(args.exclude_file, encoding="utf-8") as handle:
            excluded_ids |= {line.strip() for line in handle if line.strip()}
    if not excluded_ids:
        sys.exit("No video IDs given -- pass --exclude or --exclude-file.")

    with open(args.input_csv, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    known = {video_id(r.get("source", "")) for r in rows} - {None}
    check_ids(excluded_ids, known)

    kept, removed = drop_videos(rows, excluded_ids)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    print(f"{args.input_csv}: {len(rows)} rows -> {len(kept)} kept, "
          f"{len(rows) - len(kept)} dropped")
    for vid, n in sorted(removed.items()):
        if n:
            print(f"  {vid}: {n} comment(s) removed")
        else:
            print(f"  note: video ID '{vid}' matched no rows in this file (expected when the "
                  f"list covers several stories and this file holds one)", file=sys.stderr)
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
