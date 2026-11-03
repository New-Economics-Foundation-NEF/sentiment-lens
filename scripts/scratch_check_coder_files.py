r"""Check each coder's completed file before computing agreement. Prints counts only, never
comment text.

    python scripts/scratch_check_coder_files.py "coder_a.xlsx" "coder_b.xlsx" "coder_c.xlsx" \
        --core-ids official-data/gold_sample_core_50.csv

For each file it reports: the columns found; rows; blank or repeated comment_ids; how many of
the core comments the coder has, and for those how many have a blank hand_domain or
hand_sentiment; any hand_domain value that is not a category or "none" (a list number such as
"3", a typo, an off-list word); and the spread of sentiment values with any that are not a
valid rating. Then across files: how many core comments EVERY coder has (this is the n that
`ewl-validate interrater` will use), and which core ids are missing from which file.

Exits 0 when nothing needs fixing and 1 otherwise, so it can sit in a checklist.
Needs `pip install -e ".[analysis]"` for .xlsx files.
"""
import argparse
import collections
import difflib
import os
import sys

from ewl.analyse.validate import read_any, read_ids
from ewl.taxonomy import CATEGORY_NAMES

SENTIMENT_WORDS = {"negative", "neutral", "positive", "irrelevant", "uncodeable"}


def clean(value):
    return "" if value is None else str(value).strip()


def domain_problems(value):
    """Tokens in a hand_domain cell that are not a category name or 'none'."""
    text = clean(value).lower()
    if not text:
        return []
    tokens = [t.strip() for t in text.replace(";", ",").split(",") if t.strip()]
    bad = [t for t in tokens if t not in CATEGORY_NAMES and t != "none"]
    if "none" in tokens and len(tokens) > 1:
        bad.append("none+other")
    return bad


def sentiment_ok(value):
    text = clean(value).lower()
    if text in SENTIMENT_WORDS:
        return True
    try:
        number = float(text)
    except ValueError:
        return False
    return number in (-1, 0, 1, 2, 3, 4, 5)


def check_file(path, core_ids):
    """(report dict, set of core ids this file holds with BOTH a domain and a sentiment)."""
    rows = read_any(path)
    report = {"file": os.path.basename(path), "rows": len(rows), "problems": []}
    if not rows:
        report["problems"].append("no rows")
        return report, set()
    cols = set(rows[0].keys())
    missing = [c for c in ("comment_id", "hand_domain", "hand_sentiment") if c not in cols]
    if missing:
        report["problems"].append(f"missing column(s): {', '.join(missing)}")
        return report, set()

    ids = [clean(r.get("comment_id")) for r in rows]
    report["blank_ids"] = sum(1 for i in ids if not i)
    repeats = [c for c, n in collections.Counter(i for i in ids if i).items() if n > 1]
    report["repeated_ids"] = len(repeats)
    if report["blank_ids"]:
        report["problems"].append(f"{report['blank_ids']} row(s) with no comment_id")
    if repeats:
        report["problems"].append(f"{len(repeats)} comment_id(s) appear more than once")

    bad_domain = collections.Counter()
    bad_where = collections.defaultdict(list)     # bad value -> [(sheet row, is core)]
    bad_sent = collections.Counter()
    sent_spread = collections.Counter()
    usable, core_present = set(), set()
    core_blank_dom = core_blank_sent = 0
    for index, row in enumerate(rows):
        cid = clean(row.get("comment_id"))
        dom, sent = clean(row.get("hand_domain")), clean(row.get("hand_sentiment"))
        for tok in domain_problems(dom):
            bad_domain[tok] += 1
            bad_where[tok].append((index + 2, cid in core_ids))   # +2: header row, 1-based
        if sent:
            sent_spread[sent.lower()] += 1
            if not sentiment_ok(sent):
                bad_sent[sent] += 1
        if cid in core_ids:
            core_present.add(cid)
            core_blank_dom += not dom
            core_blank_sent += not sent
            if dom and sent:
                usable.add(cid)
    # A repeated id is ambiguous (which row is the coder's answer?), so it is not usable.
    usable -= set(repeats)
    core_seen = len(core_present)
    report["bad_domain_where"] = dict(bad_where)
    report.update(core_seen=core_seen, core_blank_domain=core_blank_dom,
                  core_blank_sentiment=core_blank_sent, sentiment_spread=dict(sent_spread),
                  bad_domain=dict(bad_domain), bad_sentiment=dict(bad_sent))
    if core_ids and core_seen < len(core_ids):
        report["problems"].append(f"only {core_seen} of {len(core_ids)} core comments present")
    if core_blank_dom:
        report["problems"].append(f"{core_blank_dom} core row(s) with a blank hand_domain")
    if core_blank_sent:
        report["problems"].append(f"{core_blank_sent} core row(s) with a blank hand_sentiment")
    if bad_domain:
        numeric = [t for t in bad_domain if t.isdigit()]
        hint = (" (these look like the 1-5 list numbers: convert to category names)"
                if numeric else "")
        report["problems"].append(
            f"hand_domain has values that are not a category or 'none': "
            f"{dict(bad_domain)}{hint}")
        for value, places in bad_where.items():
            close = difflib.get_close_matches(value.rstrip("#.,;:"), list(CATEGORY_NAMES) + ["none"],
                                              n=1, cutoff=0.75)
            guess = f"; closest valid label: {close[0]!r} (a suggestion, not applied)" if close                 else "; no close valid label (re-read the comment and decide)"
            where = ", ".join(f"row {r}{' (CORE)' if c else ''}" for r, c in places[:8])
            report["problems"].append(f"  {value!r} at sheet {where}{guess}")
    if bad_sent:
        report["problems"].append(f"hand_sentiment has invalid values: {dict(bad_sent)}")
    return report, usable


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="each coder's completed file (.xlsx or .csv)")
    parser.add_argument("--core-ids", default="official-data/gold_sample_core_50.csv",
                        help="the core comment ids (one per line, or a CSV with comment_id)")
    args = parser.parse_args()
    core_ids = set(read_ids(args.core_ids)) if os.path.exists(args.core_ids) else set()
    if not core_ids:
        print(f"NOTE: no core ids read from {args.core_ids}; the core checks are skipped.")

    reports, usable_sets = [], []
    for path in args.files:
        report, usable = check_file(path, core_ids)
        reports.append(report)
        usable_sets.append(usable)
        print(f"\n{report['file']}: {report['rows']} rows")
        if "core_seen" in report:
            print(f"  core comments present: {report['core_seen']}"
                  f"{' of ' + str(len(core_ids)) if core_ids else ''}; "
                  f"blank domain {report['core_blank_domain']}, "
                  f"blank sentiment {report['core_blank_sentiment']}")
            print(f"  sentiment values used: {report['sentiment_spread']}")
        if report["problems"]:
            for problem in report["problems"]:
                print(f"  PROBLEM: {problem}")
        else:
            print("  ok")

    if core_ids and len(usable_sets) > 1:
        common = set.intersection(*usable_sets)
        print(f"\nCore comments that EVERY coder has with a domain and a sentiment: "
              f"{len(common)} of {len(core_ids)}  (this is the n interrater will use)")
        for report, usable in zip(reports, usable_sets):
            gone = core_ids - usable
            if gone:
                print(f"  {report['file']} lacks {len(gone)} of them, e.g. "
                      f"{', '.join(sorted(gone)[:5])}")
    failed = any(r["problems"] for r in reports)
    print("\n" + ("Fix the PROBLEM lines above before computing agreement." if failed
                  else "All files look ready for `ewl-validate interrater`."))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
