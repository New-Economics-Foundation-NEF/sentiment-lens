"""How closely do the models agree with each other, and with the human coders?

    ewl-agreement --model sonnet=official-data/gold_sonnet.csv \
                  --model opus5=official-data/gold_opus5.csv \
                  --model gemma=official-data/gold_gemma.csv [--no-gold]

Each file is a labelled output of `ewl-classify` or `ewl-gemma` over the SAME gold comments. The
hand codes (`hand_domain`, `hand_sentiment`, the adjudicated gold) are read from the first
file, because the labelled outputs carry the input columns, and are treated as one more
rater called `hand`. Treating each model as a rater beside the coders is the same logic as the
coder-versus-coder agreement figures, so the numbers can be read against them.

Per judgement it prints, counts and statistics only (no comment text):
  * a pairwise Cohen's kappa matrix (models and `hand`);
  * Krippendorff's alpha (nominal) across all the raters, with a 95% interval from resampling
    comments, and across the models alone (without `hand`), so you can see whether the models
    agree with each other more than with the coders;
  * how many comments ALL the models labelled the same way but differently from `hand`
    ("consensus errors"), which is the signature of an error the models share.

Judgements: the presence of each of the four harm categories, and the sentiment answer (one of
negative, neutral, positive, irrelevant, uncodeable). A comment where any rater gave no usable
answer (a parse failure, an unscoreable sentiment) is left out of that judgement and the count
is printed. These are UNWEIGHTED: every gold comment counts once, whatever its sampling
weight, so they describe agreement on this gold set as drawn, not on the corpus.

What it does not tell you: the models share training data and so can share mistakes, so
agreement between models is not correctness; and a high alpha among models with a low alpha
against `hand` points at an instrument or coding question, not at a good model.
"""
import argparse
import collections
import csv
import sys

from ewl import taxonomy
from ewl.analyse import evaluate, validate

CATEGORIES = taxonomy.CATEGORY_NAMES
SENTIMENT_ANSWERS = ("negative", "neutral", "positive", "irrelevant", "uncodeable")


def read_labelled(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "comment_id" not in rows[0] or "domain" not in rows[0]:
        sys.exit(f"{path} is not a labelled file from ewl-classify or ewl-gemma "
                 f"(needs comment_id and domain columns).")
    ids = [(r.get("comment_id") or "").strip() for r in rows]
    if len(set(ids)) != len(ids) or not all(ids):
        sys.exit(f"{path} has blank or repeated comment_id values.")
    return {i: r for i, r in zip(ids, rows)}


def harm_presence(row, category):
    """1 or 0, or None when the model gave no usable answer for the comment."""
    if row.get("domain_parse_failure") == "1":
        return None
    return 1 if row.get(f"domain_{category}") == "1" else 0


def model_sentiment(row):
    label = (row.get("sentiment") or "").strip().lower()
    return label if label in SENTIMENT_ANSWERS else None


def hand_sentiment(row):
    value = evaluate.bucket_hand_sentiment(row.get("hand_sentiment"))
    return value if value in SENTIMENT_ANSWERS else None


def hand_presence(row, category):
    raw = (row.get("hand_domain") or "").strip().lower()
    if not raw:
        return None
    return 1 if category in raw else 0


def pairwise_matrix(raters, names):
    """Cohen's kappa for every pair of raters over the comments both gave an answer for."""
    matrix = {}
    for a in names:
        for b in names:
            if a == b:
                continue
            pairs = [(raters[a][i], raters[b][i]) for i in raters[a]
                     if raters[a][i] is not None and raters[b].get(i) is not None]
            matrix[(a, b)] = (validate.kappa_categorical(pairs), len(pairs))
    return matrix


def print_matrix(matrix, names, title, out):
    width = max(len(n) for n in names) + 2
    print(f"\n{title}: pairwise Cohen's kappa", file=out)
    print(" " * width + "".join(f"{n:>{width}}" for n in names), file=out)
    for a in names:
        cells = []
        for b in names:
            cells.append(f"{'-':>{width}}" if a == b else f"{matrix[(a, b)][0]:>{width}.2f}")
        print(f"{a:<{width}}" + "".join(cells), file=out)


def alpha_line(items, label, n_boot, out):
    if len(items) < 2 or len(items[0]) < 2:
        print(f"  {label}: not enough comments or raters", file=out)
        return
    alpha = validate.krippendorff_alpha(items, "nominal")
    low, high = validate.bootstrap_alpha_ci(items, "nominal", n_boot=n_boot)
    shown = "n/a (no variation)" if alpha != alpha else f"{alpha:.2f}"
    interval = "" if low != low else f" (95% interval {low:.2f} to {high:.2f})"
    print(f"  {label}: alpha {shown}{interval}, {len(items)} comments, {len(items[0])} raters", file=out)


def judgement(title, raters, names, model_names, out, n_boot):
    ids = sorted(set.intersection(*(set(raters[n]) for n in names)))
    complete = [i for i in ids if all(raters[n][i] is not None for n in names)]
    print(f"\n=== {title} ===", file=out)
    print(f"  {len(complete)} of {len(ids)} comments have a usable answer from every rater "
          f"(the rest are left out of the alpha figures)", file=out)
    print_matrix(pairwise_matrix(raters, names), names, title, out)
    alpha_line([[raters[n][i] for n in names] for i in complete], "all raters", n_boot, out)
    if "hand" in names and len(model_names) >= 2:
        complete_models = [i for i in ids if all(raters[n][i] is not None for n in model_names)]
        alpha_line([[raters[n][i] for n in model_names] for i in complete_models],
                   "models only (no hand codes)", n_boot, out)
        hand_ok = [i for i in complete_models if raters["hand"][i] is not None]
        unanimous = [i for i in hand_ok if len({raters[n][i] for n in model_names}) == 1]
        shared_errors = [i for i in unanimous if raters[model_names[0]][i] != raters["hand"][i]]
        print(f"  all {len(model_names)} models gave the same answer on {len(unanimous)} of "
              f"{len(hand_ok)} comments; on {len(shared_errors)} of those they were all different "
              f"from the hand code (shared errors)", file=out)
        agree_hand = collections.Counter(
            sum(1 for n in model_names if raters[n][i] == raters["hand"][i]) for i in hand_ok)
        spread = ", ".join(f"{agree_hand[k]} comments {k} models" for k in range(len(model_names) + 1))
        print(f"  how many models matched the hand code: {spread}", file=out)


def main(argv=None, out=None):
    out = out or sys.stdout
    parser = argparse.ArgumentParser(
        description="Pairwise kappa and Krippendorff's alpha between models and the hand codes, "
                    "on the gold set (counts and statistics only).")
    parser.add_argument("--model", action="append", required=True, metavar="NAME=FILE",
                        help="a labelled gold CSV for one model (repeat for each model)")
    parser.add_argument("--no-gold", action="store_true",
                        help="leave the hand codes out and compare only the models")
    parser.add_argument("--bootstrap", type=int, default=1000, help="resamples for the interval")
    args = parser.parse_args(argv)
    models = []
    for entry in args.model:
        if "=" not in entry:
            sys.exit(f"--model wants NAME=FILE, not {entry!r}")
        name, path = entry.split("=", 1)
        models.append((name.strip(), read_labelled(path)))
    if len({n for n, _ in models}) != len(models):
        sys.exit("Each --model needs its own name.")
    first_ids = set(models[0][1])
    for name, rows in models[1:]:
        if set(rows) != first_ids:
            sys.exit(f"{name} was not labelled over the same comments as {models[0][0]} "
                     f"({len(set(rows) ^ first_ids)} differ). Compare runs over the same gold file.")
    model_names = [n for n, _ in models]
    names = model_names if args.no_gold else model_names + ["hand"]
    print(f"{len(first_ids)} gold comments; raters: {', '.join(names)}. Unweighted.", file=out)

    gold_rows = models[0][1]
    for category in CATEGORIES:
        raters = {n: {i: harm_presence(r, category) for i, r in rows.items()} for n, rows in models}
        if not args.no_gold:
            raters["hand"] = {i: hand_presence(r, category) for i, r in gold_rows.items()}
        judgement(f"harm category: {category}", raters, names, model_names, out, args.bootstrap)
    raters = {n: {i: model_sentiment(r) for i, r in rows.items()} for n, rows in models}
    if not args.no_gold:
        raters["hand"] = {i: hand_sentiment(r) for i, r in gold_rows.items()}
    judgement("sentiment answer (negative / neutral / positive / irrelevant / uncodeable)",
              raters, names, model_names, out, args.bootstrap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
