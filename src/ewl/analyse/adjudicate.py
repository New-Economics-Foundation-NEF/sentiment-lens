"""Adjudicate the comments several coders coded in common, by majority.

    ewl-adjudicate "coder_a.xlsx" "coder_b.xlsx" "coder_c.xlsx" \
        --core-ids official-data/gold_sample_core_50.csv \
        --review-output official-data/core_review.csv

Step 1 resolves what the majority settles and writes the rest to a review file (with the
comment text and every coder's answer side by side) for the project owner to decide.
Step 2, after the blank `hand_domain` / `hand_sentiment` cells in that file are filled in
for the flagged rows:

    ewl-adjudicate "coder_a.xlsx" "coder_b.xlsx" "coder_c.xlsx" \
        --core-ids official-data/gold_sample_core_50.csv \
        --decisions official-data/core_review.csv \
        --output official-data/core_adjudicated.csv \
        --gold "official-data/Full Gold.csv" --gold-output official-data/Full_Gold_adjudicated.csv

The rules (decided by the project owner, 2026-10-14):

Exactly three coders are required: the rules below are written for 2-of-3.

HARM DOMAIN. Each coder gives a set of categories (an empty set for "none").
  * A majority (2 of 3) for "none" settles it as none.
  * A category tagged by a majority is in the label. A category tagged by only one coder
    is LEFT OUT by default (`--extra-labels ignore`, the default since 2026-10-14: lone
    labels only ever add positives, which moves TPR/FPR). `--extra-labels add` instead
    adds it beside the majority label when that makes at most two labels.
  * No majority at all (for example three different categories), or a majority label
    plus two different lone labels, is flagged for the project owner.
SENTIMENT. Compared at the level of what the model is scored on: negative (1-2), neutral
  (3), positive (4-5), irrelevant (0) and uncodeable (-1). A majority settles it and the
  bucket word (or 0 / -1) is written; three different answers are flagged.
  `--sentiment-rule exact` instead needs two identical values.

Prints counts only. The review file holds comment text: keep it in `official-data/`
(gitignored), never in git. Agreement statistics (`ewl-validate interrater`) must still be
computed from the coders' ORIGINAL files, never from the adjudicated one.
"""
import argparse
import collections
import csv
import json
import os
import sys

from ewl.analyse.validate import (_rating_kind, labels_from_text, read_any, read_ids,
                                  sentiment_bucket)
from ewl.taxonomy import CATEGORY_NAMES

STANCE_WORDS = ("negative", "neutral", "positive")
WRITE_AS = {"irrelevant": "0", "uncodeable": "-1"}


def _majority_needed(n):
    return n // 2 + 1


def domain_set(raw):
    """A hand_domain cell as a set of categories (empty = none). Raises ValueError on
    anything that is not a category name or 'none', or on 'none' mixed with a category."""
    text = (raw or "").strip().lower()
    if not text:
        raise ValueError("blank")
    tokens = [t.strip() for t in text.replace(";", ",").split(",") if t.strip()]
    bad = [t for t in tokens if t not in CATEGORY_NAMES and t != "none"]
    if bad:
        raise ValueError(f"not a category: {bad[0]!r}")
    if "none" in tokens and len(tokens) > 1:
        raise ValueError("'none' together with a category")
    return {t for t in tokens if t != "none"}


def _in_order(categories):
    return [c for c in CATEGORY_NAMES if c in categories]


def resolve_domain(sets, extra_labels="ignore"):
    """(labels, basis, flag). `labels` is a list ([] never; ["none"] for none) or None
    when flagged, in which case `flag` says why."""
    need = _majority_needed(len(sets))
    none_votes = sum(1 for s in sets if not s)
    if none_votes >= need:
        dissent = _in_order(set().union(*sets))
        note = f"; a coder tagged {', '.join(dissent)}" if dissent else ""
        return ["none"], f"majority none ({none_votes} of {len(sets)}){note}", None
    tags = collections.Counter(c for s in sets for c in s)
    majority = _in_order({c for c, k in tags.items() if k >= need})
    extras = _in_order({c for c, k in tags.items() if k < need})
    if not majority:
        detail = "; ".join(f"{c} x{tags[c]}" for c in _in_order(set(tags)))
        return None, "", f"no majority ({detail or 'all none'}; none x{none_votes})"
    if len(majority) > 2:
        return None, "", f"{len(majority)} categories each have a majority (limit is two)"
    if len(majority) == 2:
        note = f"; not kept (limit of two): {', '.join(extras)}" if extras else ""
        return majority, f"majority, two labels{note}", None
    if not extras or extra_labels == "ignore":
        note = f"; left out: {', '.join(extras)}" if extras else ""
        return majority, f"majority{note}", None
    if len(extras) == 1:
        return majority + extras, f"majority plus one dissenter's label ({extras[0]})", None
    return None, "", (f"majority {majority[0]} plus {len(extras)} different lone labels "
                      f"({', '.join(extras)}); limit is two labels")


def sentiment_category(raw):
    """negative / neutral / positive / irrelevant / uncodeable for a rating or its word.
    Raises ValueError for anything else."""
    text = str(raw or "").strip().lower()
    if not text:
        raise ValueError("blank")
    kind = _rating_kind(text)
    if kind != "stance":
        return kind
    bucket = sentiment_bucket(text)
    if bucket in STANCE_WORDS:
        return bucket
    raise ValueError(f"not a rating: {text!r}")


def _exact_key(raw):
    text = str(raw).strip().lower()
    text = {"irrelevant": "0", "uncodeable": "-1"}.get(text, text)
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def resolve_sentiment(values, rule="bucket"):
    """(value, basis, flag): the value to write, or None when flagged."""
    need = _majority_needed(len(values))
    cats = [sentiment_category(v) for v in values]
    if rule == "exact":
        keys = [_exact_key(v) for v in values]
        key, count = collections.Counter(keys).most_common(1)[0]
        if count >= need:
            return key, f"two or more identical values ({count} of {len(values)})", None
        return None, "", "no two identical values (" + ", ".join(sorted(set(keys))) + ")"
    category, count = collections.Counter(cats).most_common(1)[0]
    if count >= need:
        return (WRITE_AS.get(category, category), f"majority {category} ({count} of {len(values)})",
                None)
    return None, "", "no majority (" + ", ".join(sorted(set(cats))) + ")"


def coder_name(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.replace("Gold Set Complete", "").strip(" _#-") or stem


def load_coders(paths, core_ids):
    """[(name, {comment_id: row})] restricted to the core ids. Refuses a coder who lacks
    a core comment, has it twice, or left its domain or sentiment blank."""
    coders = []
    for path in paths:
        rows = read_any(path)
        by_id, problems = {}, []
        for row in rows:
            cid = (row.get("comment_id") or "").strip()
            if cid not in core_ids:
                continue
            if cid in by_id:
                problems.append(f"{cid} appears twice")
                continue
            by_id[cid] = row
        for cid in sorted(core_ids - set(by_id)):
            problems.append(f"{cid} is missing")
        for cid, row in by_id.items():
            for column in ("hand_domain", "hand_sentiment"):
                if not (row.get(column) or "").strip():
                    problems.append(f"{cid} has a blank {column}")
        if problems:
            raise SystemExit(f"{os.path.basename(path)}: cannot adjudicate: "
                             + "; ".join(problems[:6])
                             + (f" (and {len(problems) - 6} more)" if len(problems) > 6 else ""))
        coders.append((coder_name(path), by_id))
    names = [n for n, _ in coders]
    if len(set(names)) != len(names):
        raise SystemExit(f"Two coder files have the same name {names}: give them distinct "
                         f"file names.")
    return coders


def adjudicate(coders, core_ids, sentiment_rule="bucket", extra_labels="ignore"):
    """One record per core comment: resolved values, bases, and flags."""
    if len(coders) != 3:
        raise SystemExit(f"Adjudication needs exactly three coders, got {len(coders)}.")
    records = []
    for cid in sorted(core_ids):
        rows = [by_id[cid] for _, by_id in coders]
        try:
            sets = [domain_set(r.get("hand_domain")) for r in rows]
            values = [r.get("hand_sentiment") for r in rows]
            [sentiment_category(v) for v in values]
        except ValueError as problem:
            raise SystemExit(f"{cid}: {problem}") from None
        labels, domain_basis, domain_flag = resolve_domain(sets, extra_labels)
        value, sentiment_basis, sentiment_flag = resolve_sentiment(values, sentiment_rule)
        first = rows[0]
        records.append({
            "comment_id": cid,
            "story": (first.get("story") or "").strip(),
            "body": first.get("body") or "",
            "domains": [", ".join(_in_order(s)) or "none" for s in sets],
            "sentiments": [str(v).strip() for v in values],
            "hand_domain": ", ".join(labels) if labels else "",
            "domain_basis": domain_basis, "domain_flag": domain_flag or "",
            "hand_sentiment": value if value is not None else "",
            "sentiment_basis": sentiment_basis, "sentiment_flag": sentiment_flag or "",
        })
    return records


def write_csv(path, fieldnames, rows):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_review(path, records, names):
    flagged = [r for r in records if r["domain_flag"] or r["sentiment_flag"]]
    fields = ["comment_id", "story", "body"]
    for name in names:
        fields += [f"{name} domain", f"{name} sentiment"]
    fields += ["domain_flag", "proposed_domain", "sentiment_flag", "proposed_sentiment",
               "hand_domain", "hand_sentiment"]
    rows = []
    for r in flagged:
        row = {"comment_id": r["comment_id"], "story": r["story"], "body": r["body"],
               "domain_flag": r["domain_flag"], "proposed_domain": r["hand_domain"],
               "sentiment_flag": r["sentiment_flag"], "proposed_sentiment": r["hand_sentiment"],
               "hand_domain": "", "hand_sentiment": ""}
        for i, name in enumerate(names):
            row[f"{name} domain"] = r["domains"][i]
            row[f"{name} sentiment"] = r["sentiments"][i]
        rows.append(row)
    write_csv(path, fields, rows)
    return len(rows)


def apply_decisions(records, decisions_path, rule="bucket"):
    """Fill the flagged fields from the review file the owner completed. Refuses a flagged
    field left blank or filled with something that is not a valid answer."""
    decisions = {(row.get("comment_id") or "").strip(): row for row in read_any(decisions_path)}
    unknown = set(decisions) - {r["comment_id"] for r in records}
    if unknown:
        print(f"NOTE: {len(unknown)} row(s) in {os.path.basename(decisions_path)} are not core "
              f"comments and are ignored.", file=sys.stderr)
    problems = []
    for r in records:
        row = decisions.get(r["comment_id"], {})
        for field, flag in (("hand_domain", "domain_flag"), ("hand_sentiment", "sentiment_flag")):
            if not r[flag] and (row.get(field) or "").strip():
                print(f"WARNING: {r['comment_id']} has a {field} decision in the review file "
                      f"but is not flagged for it in this run (settings changed or a coder file "
                      f"corrected?): the decision is ignored.", file=sys.stderr)
        if r["domain_flag"]:
            text = (row.get("hand_domain") or "").strip()
            try:
                labels = _in_order(domain_set(text))
            except ValueError as problem:
                problems.append(f"{r['comment_id']} hand_domain: {problem}")
            else:
                r["hand_domain"] = ", ".join(labels) or "none"
                r["domain_basis"] = "decided by the project owner"
        if r["sentiment_flag"]:
            text = (row.get("hand_sentiment") or "").strip()
            try:
                category = sentiment_category(text)
            except ValueError as problem:
                problems.append(f"{r['comment_id']} hand_sentiment: {problem}")
            else:
                r["hand_sentiment"] = (_exact_key(text) if rule == "exact"
                                       else WRITE_AS.get(category, category))
                r["sentiment_basis"] = "decided by the project owner"
    if problems:
        raise SystemExit("The decisions file is not complete:\n  " + "\n  ".join(problems[:20]))


def settings_path(review_path):
    return review_path + ".settings.json"


def write_settings(review_path, settings):
    with open(settings_path(review_path), "w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2)


def check_settings(decisions_path, settings):
    """Refuse step 2 unless it runs with the settings step 1 wrote the review file under."""
    path = settings_path(decisions_path)
    try:
        with open(path, encoding="utf-8") as handle:
            saved = json.load(handle)
    except (OSError, ValueError):
        raise SystemExit(f"Cannot find the settings step 1 wrote beside the review file "
                         f"({os.path.basename(path)}): run step 1 again.") from None
    if saved != settings:
        raise SystemExit(f"The settings differ from those used in step 1 (step 1: {saved}; now: "
                         f"{settings}): a decision could be ignored or misapplied. Run again "
                         f"with the step 1 settings, or redo step 1.")


def write_resolved(path, records):
    rows = [{"comment_id": r["comment_id"], "hand_domain": r["hand_domain"],
             "hand_sentiment": r["hand_sentiment"], "coder_notes": "adjudicated",
             "story": r["story"], "domain_basis": r["domain_basis"],
             "sentiment_basis": r["sentiment_basis"]} for r in records]
    write_csv(path, ["comment_id", "hand_domain", "hand_sentiment", "coder_notes", "story",
                     "domain_basis", "sentiment_basis"], rows)


def load_gold(gold_path):
    """Read and check the gold file. Separate from apply_to_gold so main() can refuse a bad
    gold file before it writes --output: a refusal used to leave the resolved file behind."""
    with open(gold_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    for column in ("hand_domain", "hand_sentiment"):
        if column not in fields:
            raise SystemExit(f"{gold_path} has no {column} column.")
    if "coder_notes" not in fields:
        fields.append("coder_notes")
    seen = collections.Counter((row.get("comment_id") or "").strip() for row in rows)
    repeated = sum(1 for k, n in seen.items() if k and n > 1)
    if repeated:
        raise SystemExit(f"{gold_path} repeats {repeated} comment_id(s): give it one row per "
                         f"comment before adjudicating into it.")
    return fields, rows


def apply_to_gold(gold_path, out_path, records, gold=None):
    """Copy the gold file with the adjudicated core rows' labels replaced. Counts only.
    `gold` is load_gold's (fields, rows), when the caller has already read it."""
    fields, rows = gold or load_gold(gold_path)
    by_id = {r["comment_id"]: r for r in records}
    present, changed_domain, changed_sentiment = set(), 0, 0
    for row in rows:
        cid = (row.get("comment_id") or "").strip()
        if cid not in by_id:
            continue
        present.add(cid)
        new = by_id[cid]
        old_d = labels_from_text(row.get("hand_domain"))
        if old_d != {c for c in CATEGORY_NAMES if c in new["hand_domain"]}:
            changed_domain += 1
        if (row.get("hand_sentiment") or "").strip() != new["hand_sentiment"]:
            changed_sentiment += 1
        row["hand_domain"] = new["hand_domain"]
        row["hand_sentiment"] = new["hand_sentiment"]
        row["coder_notes"] = "adjudicated"
    write_csv(out_path, fields, rows)
    missing = len(by_id) - len(present)
    print(f"Gold file: {len(present)} adjudicated core rows written into {out_path}; "
          f"domain differed from the old value on {changed_domain}, sentiment on "
          f"{changed_sentiment} (the old sentiment may have been a number where the new one "
          f"is the bucket word).")
    if missing:
        print(f"WARNING: {missing} core comment(s) are not in the gold file.", file=sys.stderr)


def summarise(records):
    d_flag = sum(1 for r in records if r["domain_flag"])
    s_flag = sum(1 for r in records if r["sentiment_flag"])
    either = sum(1 for r in records if r["domain_flag"] or r["sentiment_flag"])
    plus = sum(1 for r in records if "dissenter's label" in r["domain_basis"])
    none_dissent = sum(1 for r in records if r["domain_basis"].startswith("majority none")
                       and "a coder tagged" in r["domain_basis"])
    print(f"{len(records)} core comments.")
    print(f"  Domain: {len(records) - d_flag} settled by majority "
          f"({plus} with a dissenter's second label added, {none_dissent} settled as 'none' "
          f"although one coder tagged a category), {d_flag} flagged.")
    print(f"  Sentiment: {len(records) - s_flag} settled by majority, {s_flag} flagged.")
    print(f"  {either} comment(s) need the project owner's decision.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Adjudicate comments coded by exactly three coders, by majority; flag "
                    "what the majority cannot settle.")
    parser.add_argument("inputs", nargs="+", help="each coder's ORIGINAL completed file "
                                                  "(.xlsx or .csv), exactly three")
    parser.add_argument("--core-ids", required=True, help="the comment_ids all coders coded")
    parser.add_argument("--review-output", help="step 1: the flagged comments, with the text "
                                                "and every coder's answer, for you to decide")
    parser.add_argument("--decisions", help="step 2: the review file with hand_domain and "
                                            "hand_sentiment filled in for the flagged rows")
    parser.add_argument("--output", help="the resolved core rows (comment_id, hand_domain, "
                                         "hand_sentiment, ...); written only when nothing "
                                         "remains flagged. `ewl-validate combine --adjudicated` "
                                         "reads this")
    parser.add_argument("--gold", help="a gold file to copy with the adjudicated labels put in")
    parser.add_argument("--gold-output", help="where to write that copy")
    parser.add_argument("--sentiment-rule", choices=["bucket", "exact"], default="bucket")
    parser.add_argument("--extra-labels", choices=["add", "ignore"], default="ignore",
                        help="ignore (default): a category only one coder tagged is not added. "
                             "add: it is added beside the majority label when that makes at "
                             "most two labels")
    args = parser.parse_args(argv)
    if len(args.inputs) != 3:
        parser.error("adjudication by majority needs exactly three coder files")
    if bool(args.gold) != bool(args.gold_output):
        parser.error("--gold and --gold-output go together")

    core_ids = read_ids(args.core_ids) - {"comment_id"}  # a one-column file reads its header as an id
    coders = load_coders(args.inputs, core_ids)
    names = [n for n, _ in coders]
    records = adjudicate(coders, core_ids, args.sentiment_rule, args.extra_labels)
    summarise(records)
    settings = {"extra_labels": args.extra_labels, "sentiment_rule": args.sentiment_rule}
    if args.decisions:
        check_settings(args.decisions, settings)
        apply_decisions(records, args.decisions, args.sentiment_rule)
        print("Decisions applied: nothing remains flagged.")
    flagged = [r for r in records if r["domain_flag"] or r["sentiment_flag"]]
    if args.decisions:
        flagged = []
    if flagged:
        if not args.review_output:
            sys.exit("Some comments need a decision: pass --review-output to write them out.")
        n = write_review(args.review_output, records, names)
        write_settings(args.review_output, settings)
        print(f"Wrote {n} comment(s) to {args.review_output}. Fill in hand_domain and/or "
              f"hand_sentiment on each (only the field that is flagged), then run again with "
              f"--decisions {args.review_output}.")
        if args.output or args.gold:
            print("Not writing --output / --gold-output yet: flagged comments remain.",
                  file=sys.stderr)
        return
    # The gold file is checked before anything is written, so a refusal leaves no output.
    gold = load_gold(args.gold) if args.gold else None
    if args.output:
        write_resolved(args.output, records)
        print(f"Wrote {args.output}")
    if args.gold:
        apply_to_gold(args.gold, args.gold_output, records, gold)


if __name__ == "__main__":
    main()
