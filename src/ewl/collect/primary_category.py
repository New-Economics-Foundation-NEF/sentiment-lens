"""Add a `primary_category` column to a keyword-labelled corpus, for use as a
gold-set stratification key: `ewl-validate sample --stratify-by
story,primary_category`.

`kw_domain` (from `ewl-keywords`) is not one label per comment -- `classify()`
returns up to two, ranked by how much of each category's own vocabulary
matched (see keywords.py). Stratifying directly on the raw `kw_domain` string
was tried first and rejected: it treats every label COMBINATION as its own
stratum (19 distinct values on this corpus, not 4), which multiplies cells
far faster than the sample size can support -- see
scripts_scratch_add_sampling_bins.py for the failed attempt this produced
when further crossed with length and time.

`primary_category` collapses a comment to ONE category: the first of
`kw_domain`'s (up to two) comma-separated labels -- i.e. whichever category
the classifier itself ranked higher for that comment, not a fixed priority
order. An earlier version of this script used a fixed order (democracy >
economics > environment > social) to pick a primary label; that was rejected
once the actual ranking mechanism was understood (see `_rank` in
keywords.py). A fixed order would give democracy first claim on every
comment it shares a label with, which is exactly the kind of standing,
one-directional bias keywords.py's own tie-break was rewritten to eliminate
(see the `_rank` docstring). Using the classifier's own first-ranked label
instead adds no new bias: it just picks which of the up-to-two labels
`classify()` already considered the stronger one for that specific comment.

Blank/absent kw_domain (no keyword hits at all) becomes `primary_category =
unclassified`, matching keywords.py's own UNCLASSIFIED label -- this is
deliberately its OWN stratum for sampling, not excluded, so the gold set can
measure the classifier's false-negative rate (see conversation 2026-09-09):
sampling only from already-classified rows can never catch a comment the
keyword matcher missed entirely.

`kw_domain` itself is never modified or dropped -- every row keeps its full
(up to two) labels; `primary_category` is purely an extra sampling key.

    ewl-primary-category data/full_corpus_master_kw.csv \
        --output data/full_corpus_master_kw_primary.csv

(`python scripts_scratch_add_primary_category.py ...` still works and does the same.)
"""
import argparse
import csv
import sys

from ewl import corpus as corpus_mod

sys.stdout.reconfigure(encoding="utf-8")


def primary_category(kw_domain):
    """The classifier's own first-ranked label, or 'unclassified'."""
    raw = (kw_domain or "").strip()
    if not raw:
        return "unclassified"
    return raw.split(",")[0].strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ok, message = corpus_mod.verify(args.input_csv)
    if not ok or "No manifest" in message:
        print(f"NOTE: {message}", file=sys.stderr)

    with open(args.input_csv, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames + ["primary_category"]
        rows = list(reader)

    with open(args.output, "w", newline="", encoding="utf-8-sig") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        counts = {}
        for row in rows:
            category = primary_category(row.get("kw_domain"))
            row["primary_category"] = category
            counts[category] = counts.get(category, 0) + 1
            writer.writerow(row)

    # See keywords.py's identical call -- without it, --strict-corpus on this
    # file always reports "No manifest" regardless of whether it actually
    # changed, because nothing downstream of ewl-merge ever froze it.
    corpus_mod.write_manifest(args.output, inputs=[args.input_csv])

    print(f"Wrote {len(rows)} rows with primary_category to {args.output}", file=sys.stderr)
    for category, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {category}: {count} ({count / len(rows):.1%})", file=sys.stderr)


if __name__ == "__main__":
    main()
