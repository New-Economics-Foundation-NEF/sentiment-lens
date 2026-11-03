"""Deterministic harm-domain classifier, as a check on the LLM labels.

    python keyword_classify.py master.csv > master_keyword.csv

Both the write-up and the methodology note propose the same remedy for the harm
dimension: a rule-based classifier seeded from the report's own vocabulary,
kept editable and open to inspection, with the LLM used as a cross-check and
humans adjudicating the disagreements. This is that classifier.

It has the properties VADER has and the LLM does not: it always returns the
same answer, and you can see the exact word that produced any given label. It
is worse than the LLM at idiom, sarcasm and anything unsaid — which is why the
intended workflow is to run both and hand-code where they disagree, not to pick
one. `validate.py` measures that agreement.

The vocabulary lives in keywords.yml so it can be revised without touching code.
Deliberately no yaml dependency: the file is a flat two-level list.
"""
import argparse
import collections
import csv
import hashlib
import os
import pathlib
import re
import shutil
import sys

from ewl import corpus as corpus_mod

# Windows defaults to cp1252 for stdout when it is redirected to a file, which
# crashes on any comment containing an emoji or non-Latin character.
sys.stdout.reconfigure(encoding="utf-8")

# The vocabulary ships with the package, so it resolves wherever the CLI is run
# from rather than only from the repo root. An explicit --keywords still wins,
# which is how you point at an edited copy without touching the installed one.
DEFAULT_KEYWORDS = pathlib.Path(__file__).resolve().parent.parent / "keywords.yml"

CATEGORIES = ["democracy", "economics", "environment", "social"]
UNCLASSIFIED = "unclassified"


def load_keywords(path):
    """Parse the flat 'category:' / '  - term' structure of keywords.yml."""
    patterns = collections.defaultdict(list)
    category = None
    for line in open(path, encoding="utf-8-sig"):
        line = line.split("#")[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith((" ", "-")) and line.rstrip().endswith(":"):
            category = line.strip()[:-1]
        elif category and line.strip().startswith("-"):
            term = line.strip()[1:].strip().strip('"\'')
            # An inline "#" comment is stripped above, before the term is read,
            # so a hashtag keyword -- entirely plausible in a social-media
            # vocabulary -- arrived here as "" and compiled to r"\b\b", which
            # matches every non-empty comment and would label the whole corpus
            # with that category. Refuse rather than silently do that.
            if not term:
                sys.exit(f"{path}: empty keyword under '{category}'. A '#' starts a comment in "
                         f"this file, so a hashtag term is stripped to nothing; quote it or "
                         f"drop it.")
            # A trailing * is a prefix match; everything else is whole-word.
            if term.endswith("*"):
                regex = r"\b" + re.escape(term[:-1]).replace(r"\ ", r"\s+") + r"\w*"
            else:
                regex = r"\b" + re.escape(term).replace(r"\ ", r"\s+") + r"\b"
            patterns[category].append((term, re.compile(regex, re.I)))
    if set(patterns) - set(CATEGORIES):
        sys.exit(f"Unknown categories in keyword file: {set(patterns) - set(CATEGORIES)}")
    return patterns


def _rank(counts, patterns, body):
    """Rank categories by share of their own vocabulary that matched.

    Two biases used to live in the raw-count ranking this replaces, and both
    inflated whichever category owned the longest term list:

    1. **Vocabulary size.** `democracy` ships 78 terms against `environment`'s
       47. A comment matching 3 democracy terms and 3 environment terms is
       stronger evidence for environment -- it hit a larger share of a smaller
       list -- but the count ranking scored them equal. Dividing by list length
       measures the same evidence per term offered.

    2. **Tie order.** The old key ended in `CATEGORIES.index(c)`, so an exact
       tie was resolved alphabetically: `democracy` won every tie it was in and
       `social` lost every one of its. That is not a tie-break, it is a
       standing preference applied thousands of times in one direction, and it
       silently manufactured part of the democracy share the classifier was
       built to measure. Exact ties are now broken by a hash of the comment
       body, which is stable for a given comment -- so the classifier is still
       deterministic and reruns are identical -- while favouring no category
       across the corpus.

    The cap of two is unchanged; it matches the LLM's output shape.
    """
    def key(category):
        size = len(patterns.get(category, [])) or 1
        rate = counts[category] / size
        # Stable per (comment, category), with no systematic ordering.
        jitter = hashlib.sha256(f"{body}\x00{category}".encode("utf-8")).hexdigest()
        return (-rate, jitter)
    return sorted(counts, key=key)


def classify(body, patterns):
    """Return (labels, evidence). Labels are ranked by the share of each
    category's vocabulary that matched, capped at two to match the LLM's
    output shape. See `_rank` for why it is a share and not a count."""
    hits = {}
    evidence = []
    for category in CATEGORIES:
        matched = [term for term, regex in patterns.get(category, []) if regex.search(body)]
        if matched:
            hits[category] = len(matched)
            evidence.append(f"{category}:{'|'.join(matched[:3])}")
    if not hits:
        return [UNCLASSIFIED], ""
    return _rank(hits, patterns, body)[:2], "; ".join(evidence)


def audit(rows, patterns, min_hits=5):
    """Compare each term's match rate inside vs outside its own category.

    A term matching 30% of `democracy` comments and 30% of everything else
    carries no information however apt it looks -- it is naming the corpus, not
    a harm. That is #13, and this is the check for it.

    **Leave-one-term-out.** Which category a comment belongs to is recomputed
    with the term under test removed from the vocabulary. Without that the
    check cannot work at all: a term that fires is itself what puts the comment
    into the category, so every term scores a near-perfect in-rate against an
    out-rate of ~0 and the audit approves everything.

    Not a theoretical worry. Auditing 1,207 real comments with the word `the`
    inserted into `social` -- the most worthless keyword available, matching
    40% of the corpus -- previously scored it in 79% / out 4% with no flag,
    while `inequal*` and `merit*` were dismissed as too thin. The check was
    exactly inverted: it passed terms matching everything and demoted the
    specific ones. Leaving the term out removes the self-reference, so each
    term is measured against what the REST of the list considers its category.

    Terms under `min_hits` are reported but not judged, the same floor used
    elsewhere for "too few instances to trust the number".
    """
    bodies = [row.get("body") or "" for row in rows]
    bodies = [b for b in bodies if b.strip()]
    index = range(len(bodies))

    # One pass over the corpus: which comments each term matches. Everything
    # after this is set arithmetic rather than re-running regexes per term.
    hits = {}
    for category, terms in patterns.items():
        for term, regex in terms:
            hits[(category, term)] = {i for i in index if regex.search(bodies[i])}

    def labels_without(i, skip):
        """One comment's labels, ignoring a single term. Mirrors classify():
        rank categories by how many of their terms matched, cap at two."""
        counts = {}
        for category, terms in patterns.items():
            n = sum(1 for term, _ in terms
                    if (category, term) != skip and i in hits[(category, term)])
            if n:
                counts[category] = n
        return set(_rank(counts, patterns, bodies[i])[:2])

    rows_out = []
    for category, terms in patterns.items():
        for term, _regex in terms:
            key = (category, term)
            matched = hits[key]
            inside = {i for i in index if category in labels_without(i, key)}
            outside = set(index) - inside
            in_rate = len(matched & inside) / len(inside) if inside else float("nan")
            out_rate = len(matched & outside) / len(outside) if outside else float("nan")
            total = len(matched)
            # Leave-one-term-out cannot judge a term that is the ONLY evidence
            # for its category on the comments it matches: removing it empties
            # the in-group, so in_rate is undefined rather than low. Reporting
            # that as USELESS condemned every perfectly precise keyword -- a
            # single-term category scored in=nan, out=0.33, "should go", when it
            # was 100% precise. Undecidable is its own answer, not a verdict.
            undecidable = in_rate != in_rate
            rows_out.append({
                "category": category, "term": term, "total_hits": total,
                "in_rate": in_rate, "out_rate": out_rate,
                "lift": (in_rate / out_rate) if out_rate else float("inf"),
                "thin": total < min_hits,
                # Judged on the RATIO of the two rates, not their difference.
                # A percentage-point gap cannot work here because the base
                # rates differ by an order of magnitude: on 1,207 comments
                # `free speech` scored 6% in / 1% out (a genuinely specific
                # term) while `the` scored 59% in / 34% out. The point gap
                # calls `the` the better term by 20 points; the ratio puts it
                # at 1.7x against 6x and gets it right.
                "undecidable": undecidable,
                "useless": total >= min_hits and not undecidable
                           and not (in_rate > out_rate),
                "suspect": total >= min_hits and not undecidable
                           and in_rate > out_rate
                           and (in_rate / out_rate if out_rate else float("inf")) < 2.0,
            })
    return rows_out


def cmd_audit(rows, patterns, min_hits):
    report = audit(rows, patterns, min_hits)
    print(f"{'category':<12}{'term':<20}{'hits':>6}{'in%':>8}{'out%':>8}{'lift':>7}  flag",
          file=sys.stderr)
    # Worst first within each category: a term is interesting here when its
    # in-rate fails to clear its out-rate, not when it matches a lot.
    for r in sorted(report, key=lambda r: (r["category"],
                                           r["lift"] if r["lift"] == r["lift"] else 0,
                                           -r["total_hits"])):
        flag = ("too thin" if r["thin"] else "cannot judge" if r["undecidable"]
                else "USELESS" if r["useless"] else "SUSPECT" if r["suspect"] else "")
        in_pct = "  n/a" if r["in_rate"] != r["in_rate"] else f"{r['in_rate']:.0%}"
        out_pct = "  n/a" if r["out_rate"] != r["out_rate"] else f"{r['out_rate']:.0%}"
        lift = "   inf" if r["lift"] == float("inf") else (
            "   n/a" if r["lift"] != r["lift"] else f"{r['lift']:>6.1f}")
        print(f"{r['category']:<12}{r['term']:<20}{r['total_hits']:>6}{in_pct:>8}{out_pct:>8}"
              f"{lift:>7}  {flag}", file=sys.stderr)
    useless = [r for r in report if r["useless"]]
    suspects = [r for r in report if r["suspect"]]
    print(f"\n{len(useless)} USELESS: fires at least as often outside its own category as "
          f"inside it, so it carries no information and should go.", file=sys.stderr)
    print(f"{len(suspects)} SUSPECT: fires under twice as often inside its category as "
          f"outside it.", file=sys.stderr)
    undecidable = [r for r in report if r["undecidable"]]
    if undecidable:
        print(f"{len(undecidable)} could not be judged: the term is the only evidence for its "
              f"category on the comments it matches, so removing it leaves nothing to compare "
              f"against. Not a verdict -- these need the gold set, not more comments.",
              file=sys.stderr)
    print("Judged leave-one-term-out -- each term is scored against what the REST of the list "
          "considers its category. Judging a term against labels it helped produce makes every "
          "term look perfect; see #24. Scoring against hand codes instead is the real "
          "validation; see #33 item 2.", file=sys.stderr)
    print("A term below the hit-count floor is reported but not judged — too few matches to "
          "say anything, same 'thin' logic used everywhere else in this project.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv")
    parser.add_argument("--keywords", default=str(DEFAULT_KEYWORDS),
                        help="Vocabulary file. Defaults to the copy shipped with "
                             "the package.")
    parser.add_argument("--audit", action="store_true",
                        help="Report each term's in-category vs out-of-category match rate "
                             "instead of classifying. See #13.")
    parser.add_argument("--min-hits", type=int, default=5,
                        help="Minimum total matches before a term's rate is judged (--audit only)")
    parser.add_argument("--output", help="CSV file to write to, with the same Windows-safe "
                                          "encoding the collectors use. If omitted, writes to "
                                          "stdout — fine in Git Bash, but PowerShell's '>' "
                                          "redirection still re-encodes as UTF-16, so prefer "
                                          "--output there.")
    args = parser.parse_args()

    # A manifest mismatch here means the corpus moved under this run before it
    # was even labelled -- worth knowing before spending the time to process
    # it, not just at the far end when --strict-corpus checks the output.
    ok, message = corpus_mod.verify(args.input_csv)
    if not ok or "No manifest" in message:
        print(f"NOTE: {message}", file=sys.stderr)

    patterns = load_keywords(args.keywords)
    total_terms = sum(len(v) for v in patterns.values())
    print(f"Loaded {total_terms} terms across {len(patterns)} categories", file=sys.stderr)

    if args.audit:
        with open(args.input_csv, newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        cmd_audit(rows, patterns, args.min_hits)
        return

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out = open(args.output, "w", newline="", encoding="utf-8-sig")
    else:
        # newline="" matters as much as the encoding: csv.writer's own "\r\n"
        # row terminator gets doubled to "\r\r\n" by stdout's default newline
        # translation otherwise, which shows up as a blank row after every one.
        sys.stdout.reconfigure(newline="")
        out = sys.stdout

    extra = [f"kw_{c}" for c in CATEGORIES] + [f"kw_{UNCLASSIFIED}", "kw_domain", "kw_evidence"]
    with open(args.input_csv, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames + [c for c in extra if c not in reader.fieldnames]
        writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        frequencies = collections.Counter()
        processed = 0
        blank = 0
        for row in reader:
            body = row.get("body") or ""
            if not body.strip():
                # Dropped, but counted: the labelled file otherwise has fewer
                # rows than the frozen corpus and its manifest, with nothing
                # anywhere saying why.
                blank += 1
                continue
            labels, evidence = classify(body, patterns)
            frequencies.update(labels)
            processed += 1
            for category in CATEGORIES + [UNCLASSIFIED]:
                row[f"kw_{category}"] = int(category in labels)
            row["kw_domain"] = ", ".join(labels)
            row["kw_evidence"] = evidence
            writer.writerow(row)

    if args.output:
        out.close()
        # Without this, the manifest ewl-merge wrote for the raw corpus does
        # not survive this step -- --strict-corpus on the keyword-passed file
        # always reports "No manifest", not because nothing can be checked,
        # but because nothing here ever wrote one. Freezing the output the
        # same way ewl-merge freezes the corpus closes that gap: a keyword
        # pass that gets silently re-run or hand-edited afterward is now
        # detectable too, not just the raw corpus underneath it.
        corpus_mod.write_manifest(args.output, inputs=[args.input_csv])
        # The documented chain is ewl-classify -> ewl-keywords -> ewl-export-dashboard, and
        # the export reads `<labelled>.run.json` for partial-run detection, the worked
        # examples to leave out of scoring and the --corpus check. ewl-classify wrote that
        # record beside ITS output; without carrying it forward it is lost at this step.
        # Copied unchanged (it holds hashes and settings, no comment text), nothing added.
        record = args.input_csv + ".run.json"
        if os.path.abspath(args.output) != os.path.abspath(args.input_csv):
            if os.path.exists(record):
                shutil.copyfile(record, args.output + ".run.json")
                print(f"Run record copied to {args.output}.run.json", file=sys.stderr)
            elif os.path.exists(args.output + ".run.json"):
                # A record left by an earlier run describes labels this output no longer has.
                os.remove(args.output + ".run.json")
                print(f"Removed a stale {args.output}.run.json: {args.input_csv} has no run "
                      f"record to carry forward.", file=sys.stderr)

    if blank:
        print(f"  {blank} row(s) had no comment text and were not labelled.",
              file=sys.stderr)
    print(f"\nKeyword labels over {processed} comments:", file=sys.stderr)
    for category, count in frequencies.most_common():
        print(f"  {category}: {count} ({count / max(processed, 1):.1%})", file=sys.stderr)
    print("A high unclassified share is expected and is not a failure: it is the "
          "share of the corpus this method cannot speak to, which the LLM hides "
          "by always producing a label.", file=sys.stderr)


if __name__ == "__main__":
    main()
