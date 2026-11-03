r"""How each coder used sentiment 0 ("irrelevant"), from the labels alone. Counts only; no
comment text is read or printed.

    python scripts/scratch_zero_usage.py "coder_a.xlsx" "coder_b.xlsx" "coder_c.xlsx" \
        --core-ids official-data/gold_sample_core_50.csv

The brief defined 0 as "nothing to do with the video"; the prompt says "nothing to do with
extreme wealth". Two readings, and a coder's own harm-domain label shows which one they used:

  * a 0 that ALSO carries a harm domain (democracy / economics / environment / social)
    is a comment the coder saw as being about one of the harms but with no stance on wealth:
    the BROAD reading ("no wealth stance");
  * a 0 whose domain is "none" is a comment about nothing relevant: the NARROW reading
    ("off topic / spam / rude reply").

For each coder it reports how many 0s they gave, the share carrying a harm domain, the
share by story, and the typical comment length of their 0s against their other ratings. On
the core comments every coder rated, it then shows how the coders' 0s overlap and which
coder is most like the first file given (the reference), by overlap of their 0s and by
agreement on whether a comment is a 0 at all.

Needs `pip install -e ".[analysis]"` for .xlsx files.
"""
import argparse
import collections
import os
import statistics
import sys

from sentiment_lens.analyse.validate import read_any, read_ids
from sentiment_lens.taxonomy import CATEGORY_NAMES

ZERO = {"0", "irrelevant"}


def clean(value):
    return "" if value is None else str(value).strip()


def is_zero(value):
    return clean(value).lower() in ZERO


def has_harm_domain(value):
    text = clean(value).lower()
    return any(category in text for category in CATEGORY_NAMES)


def coder_name(path):
    name = os.path.splitext(os.path.basename(path))[0]
    for junk in ("Gold Set Complete Kappa", "Gold Set Complete"):
        name = name.replace(junk, "")
    return name.strip(" _#-") or os.path.basename(path)


def usage(rows):
    """Counts describing one coder's use of 0. Pure function over a list of row dicts."""
    rated = [r for r in rows if clean(r.get("hand_sentiment"))]
    zeros = [r for r in rated if is_zero(r.get("hand_sentiment"))]
    others = [r for r in rated if not is_zero(r.get("hand_sentiment"))]
    with_harm = [r for r in zeros if has_harm_domain(r.get("hand_domain"))]
    by_story = collections.Counter()
    story_totals = collections.Counter()
    for r in rated:
        story = clean(r.get("story")) or "(no story column)"
        story_totals[story] += 1
        if is_zero(r.get("hand_sentiment")):
            by_story[story] += 1

    def median_len(items):
        lengths = [len(clean(r.get("body"))) for r in items if clean(r.get("body"))]
        return statistics.median(lengths) if lengths else None

    return {"rated": len(rated), "zeros": len(zeros),
            "zero_share": len(zeros) / len(rated) if rated else 0.0,
            "zeros_with_harm": len(with_harm),
            "zeros_with_harm_share": len(with_harm) / len(zeros) if zeros else 0.0,
            "zero_share_by_story": {s: by_story[s] / n for s, n in sorted(story_totals.items())},
            "median_len_zero": median_len(zeros), "median_len_other": median_len(others)}


def core_comparison(rows_by_coder, core_ids):
    """On core comments every coder rated: each coder's set of 0s, and pairwise overlap."""
    by_id = []
    for rows in rows_by_coder:
        by_id.append({clean(r.get("comment_id")): r for r in rows if clean(r.get("comment_id"))})
    shared = sorted(set.intersection(*[
        {i for i, r in d.items() if clean(r.get("hand_sentiment"))} for d in by_id]) & set(core_ids))
    zero_sets = [{i for i in shared if is_zero(d[i].get("hand_sentiment"))} for d in by_id]
    pairs = {}
    for a in range(len(by_id)):
        for b in range(a + 1, len(by_id)):
            both = len(zero_sets[a] & zero_sets[b])
            only_a = len(zero_sets[a] - zero_sets[b])
            only_b = len(zero_sets[b] - zero_sets[a])
            union = both + only_a + only_b
            agree = sum(1 for i in shared if (i in zero_sets[a]) == (i in zero_sets[b]))
            pairs[(a, b)] = {"both": both, "only_a": only_a, "only_b": only_b,
                             "jaccard": both / union if union else float("nan"),
                             "agree_on_zero": agree / len(shared) if shared else float("nan")}
    return shared, zero_sets, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="each coder's completed file; the FIRST is the reference")
    parser.add_argument("--core-ids", default="official-data/gold_sample_core_50.csv")
    args = parser.parse_args()
    names = [coder_name(p) for p in args.files]
    rows_by_coder = [read_any(p) for p in args.files]
    for p, rows in zip(args.files, rows_by_coder):
        if rows and not {"hand_sentiment", "hand_domain"} <= set(rows[0]):
            sys.exit(f"{os.path.basename(p)} needs hand_sentiment and hand_domain columns.")

    print("HOW EACH CODER USED 0 (irrelevant)\n")
    print(f"{'coder':<12}{'rated':>7}{'zeros':>7}{'share':>8}{'0 + harm domain':>18}{'0 + none':>10}")
    stats = [usage(rows) for rows in rows_by_coder]
    for name, s in zip(names, stats):
        print(f"{name:<12}{s['rated']:>7}{s['zeros']:>7}{s['zero_share']:>8.0%}"
              f"{s['zeros_with_harm']:>11} ({s['zeros_with_harm_share']:.0%})"
              f"{s['zeros'] - s['zeros_with_harm']:>10}")
    print("\n  '0 + harm domain' = a 0 that also names a harm: read as 'about a harm, but no stance on "
          "wealth' (the BROAD reading).\n  '0 + none' = a 0 about nothing relevant (the NARROW reading, "
          "closest to the brief: off topic, spam, rude reply).")

    print("\nShare of each coder's ratings that are 0, by story:")
    stories = sorted({s for st in stats for s in st["zero_share_by_story"]})
    print(f"{'story':<14}" + "".join(f"{n:>10}" for n in names))
    for story in stories:
        print(f"{story:<14}" + "".join(
            f"{st['zero_share_by_story'].get(story, float('nan')):>10.0%}" for st in stats))

    print("\nMedian comment length (characters) of a coder's 0s vs their other ratings:")
    for name, s in zip(names, stats):
        z, o = s["median_len_zero"], s["median_len_other"]
        print(f"  {name:<12} 0s: {z if z is not None else 'n/a'}   others: {o if o is not None else 'n/a'}")

    core_ids = set(read_ids(args.core_ids)) if os.path.exists(args.core_ids) else set()
    if len(rows_by_coder) > 1 and core_ids:
        shared, zero_sets, pairs = core_comparison(rows_by_coder, core_ids)
        print(f"\nCORE COMMENTS EVERY CODER RATED: {len(shared)}")
        print("  0s given on the core: " + ", ".join(f"{n} {len(z)}" for n, z in zip(names, zero_sets)))
        print(f"\n  {'pair':<22}{'both 0':>8}{'only first':>12}{'only second':>13}"
              f"{'overlap':>9}{'agree on 0?':>13}")
        for (a, b), p in pairs.items():
            print(f"  {names[a] + ' / ' + names[b]:<22}{p['both']:>8}{p['only_a']:>12}{p['only_b']:>13}"
                  f"{p['jaccard']:>9.2f}{p['agree_on_zero']:>13.0%}")
        ref = names[0]
        ranked = sorted(((p["jaccard"], names[b]) for (a, b), p in pairs.items() if a == 0),
                        reverse=True)
        print(f"\n  Most like {ref} in which comments get a 0 (overlap of the two sets of 0s): "
              + ", ".join(f"{n} ({j:.2f})" for j, n in ranked))
        print("  overlap = comments both gave 0 / comments either gave 0 (1.00 = identical sets).")
    print("\nRead it with care: these are label patterns, not what the comments said. A coder with "
          "many 0s that carry a harm domain used 0 as 'no wealth stance'. Reading a handful of "
          "disagreements is the check that settles it.")


if __name__ == "__main__":
    main()
