"""Score classifier arms against a hand-coded gold set and print the scorecard.

    # 1. split the hand-coded file into a retrieval pool and a held-out set
    python evaluate.py split gold_coded.csv --out-prefix gold

    # 2. score one arm
    python evaluate.py score gold_heldout.csv --judge ollama:gemma3:12b

    # 3. run the whole scorecard from #11
    python evaluate.py scorecard gold_heldout.csv --pool gold_pool.csv \
        --arm ollama:gemma3:12b/fixed \
        --arm ollama:gemma3:12b/retrieved \
        --arm keyword/none

This is the step #11 says must come before any classifier is swapped. Without a
baseline number, a change in the output cannot be told apart from a different
set of errors, and the write-up would rest on new labels with no more evidence
than it had for the old ones.

Input is a CSV with `body` and a hand-coded label column (default
`hand_domain`), which is what `validate.py sample` produces once two coders
have filled it in and adjudicated.

An arm is "<judge spec>/<selector>", e.g. `anthropic:claude-haiku-4-5/retrieved`.
"""
import argparse
import collections
import concurrent.futures
import csv
import os
import random
import sys
import time

from sentiment_lens.label import combined  # noqa: F401  -- registers the combined judge/scorer
import sentiment_lens.label.examples as examples_mod
from sentiment_lens.label import instrument
from sentiment_lens.label import judges
import sentiment_lens.label.sentiment as sentiment_mod
from sentiment_lens import taxonomy
from sentiment_lens.analyse import validate as validate_mod

# Windows defaults to cp1252 for stdout when it is redirected to a file, which
# crashes on any comment quoted in a scorecard/sentiment report or a warning
# that contains an emoji or non-Latin character.
sys.stdout.reconfigure(encoding="utf-8")

CATEGORIES = taxonomy.CATEGORY_NAMES


def read_gold(path, label_column="hand_domain", exclude_ids=None):
    """Load hand-coded rows, and complain about labels that mean nothing.

    Membership is a substring test, so a typo ("econimics") or an off-list word
    ("policy") produced an empty gold set -- indistinguishable from a
    deliberate "none", and silently counted as a negative example for all four
    categories, inflating every arm's false positives. Coders working from a
    spreadsheet will produce these, so they are counted and reported rather
    than absorbed.
    """
    rows = []
    unrecognised = collections.Counter()
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            body = (row.get("body") or "").strip()
            raw = (row.get(label_column) or "").strip().lower()
            if not body or not raw:
                continue
            row["_body"] = body
            row["_gold"] = {c for c in CATEGORIES if c in raw}
            if not row["_gold"] and raw != "none":
                unrecognised[raw] += 1
            rows.append(row)
    rows = validate_mod.drop_ids(rows, exclude_ids, "gold rows")
    if not rows:
        sys.exit(f"No hand-coded rows in {path} (need `body` and `{label_column}`).")
    validate_mod.require_gold_weights(rows, path)
    if unrecognised:
        total = sum(unrecognised.values())
        print(f"WARNING: {total} row(s) have a `{label_column}` matching no category and are "
              f"being scored as 'none'. Fix these before quoting anything -- each one is a "
              f"negative example for every category. Values seen: "
              f"{', '.join(repr(v) for v, _ in unrecognised.most_common(6))}", file=sys.stderr)
    return rows


_CAPPED_NOTED = set()


def _is_local_arm(spec):
    """True for an Ollama or Gemma arm, however it is wrapped (`llm:gemma:...`, `combined:ollama:...`).
    Matches the backend name at the start, not anywhere in the text, so an API model that merely has
    'gemma' in its name is not capped."""
    text = str(spec).lower()
    while text.startswith(("llm:", "combined:")):
        text = text.split(":", 1)[1]
    return text.startswith(("ollama", "gemma"))


def effective_concurrency(spec, requested):
    """The concurrency an arm will actually use. A local model (Ollama, Gemma) is compute-bound,
    so extra workers only queue behind each other, and batched inference on a server with
    parallel slots is not guaranteed to give the same answers as one request at a time: those
    arms run one comment at a time whatever was asked, with a note saying so."""
    if requested > 1 and _is_local_arm(spec):
        if spec not in _CAPPED_NOTED:
            _CAPPED_NOTED.add(spec)
            print(f"NOTE: {spec} is a local model, so it runs one comment at a time even with "
                  f"--concurrency {requested}.", file=sys.stderr)
        return 1
    return requested


def concurrency_footnote(results):
    """One line under a table saying what its `mins` column means, when any arm ran in parallel."""
    if not any(r.get("concurrency", 1) > 1 for r in results):
        return None
    arms = ", ".join(f"{r['arm']}: {r.get('concurrency', 1)}" for r in results)
    return ("mins is the wall-clock time at the concurrency shown for each arm (" + arms + "); at "
            "concurrency 1 it is the model's summed call time, so figures at different concurrency "
            "are not comparable.")


def label_in_order(function, items, concurrency=1):
    """`function(item)` for every item, yielded in the order of `items`.

    With concurrency 1 this is a plain lazy loop, exactly the behaviour before the option
    existed. With more, that many calls are in flight at once: the wall-clock cost of an API arm
    is almost all waiting on the network, so this is the difference between a scorecard that
    takes an hour and one that takes minutes. Results are identical either way (each comment is
    labelled on its own and the answers are read back in order); only the clock changes. Keep
    it at 1 for a local ollama judge, which is compute-bound and gains nothing.
    """
    if concurrency <= 1:
        for item in items:
            yield function(item)
        return
    with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
        yield from pool.map(function, items)


def prf(tp, fp, fn):
    """Precision, recall and F1 for one category.

    The distinction that matters: a category the arm gets *completely wrong*
    scores F1 0.0, while a category absent from both the gold set and the
    predictions has no F1 at all (NaN). Collapsing those two into NaN -- as
    this did -- meant `run_arm` dropped total failures from the macro average
    alongside genuinely inapplicable categories, so an arm that got half the
    corpus wrong could score a perfect macro-F1. That is the exact opposite of
    what macro-F1 is for, and of what the scorecard footnote claims it does.
    """
    if tp + fp + fn == 0:
        # Never in the gold set, never predicted. Nothing to score.
        return float("nan"), float("nan"), float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    if tp == 0:
        # Predicted and always wrong, or present and never found. Either way
        # the arm scored nothing on a category that was in play: F1 is 0.
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def run_arm(rows, judge, selector, progress=True, concurrency=1):
    """Label every gold row and return the metrics for one arm.

    Cost, time and call counts are reported as this arm's *delta*, not the
    judge's running totals. Arms can legitimately share one judge object --
    `combined:X` hands the same instance to every arm with matching settings,
    and that sharing is the point of the module -- but the accounting is
    cumulative on that instance, so reading it directly billed arm 2 for
    arm 1's calls as well. The scorecard's cost column is the thing the
    comparison turns on, so it has to be per-arm.
    """
    counts = {c: collections.Counter() for c in CATEGORIES}
    unclassified = unstable = parse_failures = 0
    scored_rows = []   # (video, weight, gold set, predicted set), for the bootstrap
    before_cost = judge.cost_usd()
    before_seconds = judge.seconds
    before_calls = judge.usage["calls"]
    before_hits = judge.usage["cache_hits"]
    before_input, before_output = judge.token_usage()

    concurrency = effective_concurrency(judge.spec, concurrency)
    started = time.monotonic()
    labelled = label_in_order(
        lambda row: judge.label(row["_body"], selector.select(row["_body"]),
                                story=row.get("story")), rows, concurrency)
    for index, (row, (predicted, was_unstable, _ballots)) in enumerate(zip(rows, labelled), 1):
        predicted = set(predicted)
        unstable += was_unstable
        unclassified += taxonomy.UNCLASSIFIED in predicted
        is_parse_failure = taxonomy.PARSE_FAILURE in predicted
        parse_failures += is_parse_failure

        # Horvitz-Thompson weighted, same reasoning and same helper as
        # report._gold_tpr_fpr and validate.cmd_score (see gold_weight's
        # docstring). Confirmed this mattered on a synthetic stratified
        # sample where a classifier's failures were concentrated in the
        # under-sampled (low inclusion_prob, heavily up-weighted) stratum:
        # unweighted recall here read 0.5 while the correctly weighted
        # figure was 0.0. Not a hypothetical gap -- the real gold set's
        # inclusion_prob spans a ~3,000x range, this scorecard's own footer
        # tells the reader to feed its precision/recall/FPR into `slens-report
        # corrected --tpr --fpr`, and that command has no way to know the
        # numbers it was given were never weighted for the draw they came
        # from.
        weight = validate_mod.gold_weight(row)

        # A parse failure is not a genuine "none" -- see taxonomy.PARSE_FAILURE
        # and report._gold_tpr_fpr's own docstring ("Parse failures are
        # excluded, not counted as misses"). Counting it as an implicit
        # negative guess for every category, as this loop did before, deflated
        # recall by the parse-failure rate -- a classifier reliability
        # problem, not a finding about the comments, and already visible on
        # its own via `parse_failures` above and the scorecard's parse-fail
        # column.
        if not is_parse_failure:
            scored_rows.append((row.get("source") or row.get("comment_id") or str(index),
                                weight, set(row["_gold"]), set(predicted)))
            for category in CATEGORIES:
                gold, guess = category in row["_gold"], category in predicted
                counts[category]["tp"] += weight if (gold and guess) else 0.0
                counts[category]["fp"] += weight if (guess and not gold) else 0.0
                counts[category]["fn"] += weight if (gold and not guess) else 0.0
                # True negatives: needed for FPR (fp / (fp + tn)), which report.py's
                # classifier-error correction needs alongside recall (TPR). FPR is
                # not "1 - precision" -- precision conditions on what the classifier
                # guessed, FPR conditions on what is genuinely NOT the category --
                # so it needs its own count rather than being derived from tp/fp/fn.
                counts[category]["tn"] += weight if ((not gold) and (not guess)) else 0.0
        if progress and index % 25 == 0:
            print(f"    {index}/{len(rows)}", file=sys.stderr)

    per_category, f1s = {}, []
    for category in CATEGORIES:
        c = counts[category]
        precision, recall, f1 = prf(c["tp"], c["fp"], c["fn"])
        fpr = c["fp"] / (c["fp"] + c["tn"]) if (c["fp"] + c["tn"]) else float("nan")
        per_category[category] = {
            "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
            "tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "tn": c["tn"],
        }
        if f1 == f1:  # skip NaN, i.e. a category absent from both gold and guesses
            f1s.append(f1)

    total = len(rows)
    after_input, after_output = judge.token_usage()
    return {
        "arm": f"{judge.spec}/{selector.describe()}",
        "n": total,
        "per_category": per_category,
        "macro_f1": sum(f1s) / len(f1s) if f1s else float("nan"),
        "unclassified": unclassified / total,
        "parse_failure": parse_failures / total,
        # At judge.votes == 1 there is nothing for a vote to disagree with,
        # so this is 0 by construction rather than by measurement -- reported
        # as NaN (renders as "—" via fmt/print_scorecard) so it reads as
        # "not measured", not "confirmed stable". See classify.py's matching
        # note on domain_unstable.
        "unstable": (unstable / total) if judge.votes > 1 else float("nan"),
        "votes": judge.votes,
        "deterministic": judge.deterministic,
        "cost_usd": judge.cost_usd() - before_cost,
        # With calls in parallel the judge's own total adds up every call's duration, which is
        # more than the time that passed; report the clock instead.
        "concurrency": concurrency,
        "seconds": (time.monotonic() - started) if concurrency > 1
                   else judge.seconds - before_seconds,
        "calls": judge.usage["calls"] - before_calls,
        "cache_hits": judge.usage["cache_hits"] - before_hits,
        "input_tokens": after_input - before_input,
        "output_tokens": after_output - before_output,
        "_scored_rows": scored_rows,
    }


def bootstrap_scorecard(result, n_boot=1000, seed=42, alpha=0.05):
    """95% intervals on each category's precision, recall and F1, resampling VIDEOS.

    Comments from one video are correlated, so resampling comments would give
    intervals that are too narrow; the weights are the same Horvitz-Thompson
    weights the point estimates use. Returns {category: {"precision": (lo, hi),
    "recall": (lo, hi), "f1": (lo, hi)}}, with NaN where a category never occurs
    in a draw's gold or guesses. The count of distinct videos is returned too,
    because an interval from a handful of videos should not be trusted.
    """
    scored = result.get("_scored_rows") or []
    by_video = collections.defaultdict(list)
    for video, weight, gold, guess in scored:
        by_video[video].append((weight, gold, guess))
    videos = list(by_video)
    rng = random.Random(seed)
    draws = {c: {"precision": [], "recall": [], "f1": []} for c in CATEGORIES}
    if len(videos) < 2:
        return {c: {m: (float("nan"), float("nan")) for m in ("precision", "recall", "f1")}
                for c in CATEGORIES}, len(videos)
    for _ in range(n_boot):
        sample = [row for v in (rng.choice(videos) for _ in videos) for row in by_video[v]]
        for category in CATEGORIES:
            tp = fp = fn = 0.0
            for weight, gold, guess in sample:
                g, p = category in gold, category in guess
                tp += weight if (g and p) else 0.0
                fp += weight if (p and not g) else 0.0
                fn += weight if (g and not p) else 0.0
            precision, recall, f1 = prf(tp, fp, fn)
            for name, value in (("precision", precision), ("recall", recall), ("f1", f1)):
                if value == value:
                    draws[category][name].append(value)
    out = {}
    for category in CATEGORIES:
        out[category] = {}
        for name, values in draws[category].items():
            if len(values) < n_boot * 0.5:
                out[category][name] = (float("nan"), float("nan"))
                continue
            values.sort()
            out[category][name] = (values[int((alpha / 2) * len(values))],
                                   values[min(int((1 - alpha / 2) * len(values)), len(values) - 1)])
    return out, len(videos)


def print_intervals(results, n_boot, seed=42, out=None):
    out = out or sys.stdout
    print(f"\n95% intervals (clustered by video, {n_boot} resamples, weighted like the point "
          f"estimates). An interval from few videos is fragile: the count is shown.", file=out)
    print("\n| arm | videos | category | precision | recall | F1 |", file=out)
    print("|---|---|---|---|---|---|", file=out)
    for r in results:
        intervals, n_videos = bootstrap_scorecard(r, n_boot=n_boot, seed=seed)
        for category in CATEGORIES:
            cells = []
            for name in ("precision", "recall", "f1"):
                lo, hi = intervals[category][name]
                point = r["per_category"][category][name]
                cells.append("—" if lo != lo else f"{point:.2f} ({lo:.2f}–{hi:.2f})")
            print(f"| {r['arm']} | {n_videos} | {category} | " + " | ".join(cells) + " |",
                  file=out)


def build(arm, pool_csv, votes, k, label_column, cache_path, default_selector="fixed"):
    """Turn "<judge>/<selector>" into a (judge, selector) pair.

    An arm may omit the selector, in which case `default_selector` applies --
    which is what --selector sets. It used to be hardcoded here, so
    `scorecard --arm stub --selector none` silently ran `fixed`.
    """
    spec, _, selector_name = arm.rpartition("/")
    if not spec:
        spec, selector_name = arm, default_selector
    judge = judges.create(spec, votes=votes, cache_path=cache_path)
    # n= and seed= were omitted, so a `fixed_random` scorecard arm always drew
    # n=10/seed=42 while `slens-classify --selector fixed_random --k 25` ran 25:
    # the arm you score was not the arm you ship.
    selector = examples_mod.create(
        selector_name, pool_csv=pool_csv, k=k, n=k, label_column=label_column)
    return judge, selector


def fmt(value, places=2):
    return "—" if value != value else f"{value:.{places}f}"


def fmt_pct(value):
    return "—" if value != value else f"{value:.0%}"


def cost(usd):
    """Anything local is free; a fraction of a cent should not print as $0.00."""
    if usd == 0:
        return "free"
    return f"${usd:.2f}" if usd >= 0.01 else "<$0.01"


def print_scorecard(results, out=None):
    """Markdown, so it can be pasted straight into the issue or the methods
    section.

    `out` defaults at call time, not at import: a default of `sys.stdout` in the
    signature is bound once when the module loads and then ignores any later
    redirection of stdout.
    """
    out = out or sys.stdout
    print("\n| arm | n | democracy P | democracy R | macro-F1 | uncls | parse-fail | unstable | "
          "reproducible | cost | mins | in tok | out tok |", file=out)
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|", file=out)
    for r in results:
        dem = r["per_category"]["democracy"]
        print(f"| `{r['arm']}` | {r['n']} | {fmt(dem['precision'])} | {fmt(dem['recall'])} | "
              f"{fmt(r['macro_f1'])} | {r['unclassified']:.0%} | {r.get('parse_failure', 0):.0%} | "
              f"{fmt_pct(r['unstable'])} | "
              f"{'yes' if r['deterministic'] else 'cache only'} | "
              f"{cost(r['cost_usd'])} | "
              f"{r['seconds'] / 60:.1f} | "
              f"{r.get('input_tokens', 0):,} | {r.get('output_tokens', 0):,} |", file=out)

    _note = concurrency_footnote(results)
    if _note:
        print(f"\n{_note}", file=out)

    print("\nPer-category F1:\n", file=out)
    print("| arm | " + " | ".join(CATEGORIES) + " |", file=out)
    print("|---" * (len(CATEGORIES) + 1) + "|", file=out)
    for r in results:
        cells = " | ".join(fmt(r["per_category"][c]["f1"]) for c in CATEGORIES)
        print(f"| `{r['arm']}` | {cells} |", file=out)

    # `slens-report corrected` refuses to run without --tpr and --fpr, and its
    # own error message says to get them "from evaluate.py's scorecard" --
    # but recall (TPR) was the only one of the two ever printed here. FPR was
    # computed in run_arm() the whole time and simply never shown, so the one
    # command told you where to find a number the other command never
    # actually gave you.
    print("\nPer-category FPR (recall is the democracy R column above; both feed "
          "`slens-report corrected --tpr --fpr`):\n", file=out)
    print("| arm | " + " | ".join(CATEGORIES) + " |", file=out)
    print("|---" * (len(CATEGORIES) + 1) + "|", file=out)
    for r in results:
        cells = " | ".join(fmt(r["per_category"][c]["fpr"]) for c in CATEGORIES)
        print(f"| `{r['arm']}` | {cells} |", file=out)

    print("\n`democracy` is broken out because #2 is the finding at stake. Macro-F1 "
          "averages over categories so a win on a common one cannot hide a failure "
          "on a rare one. A high unclassified rate is information, not a bug: it is "
          "the share of the corpus the arm declines to speak to. `unstable` shows "
          "— (not 0%) for any arm run at votes=1 -- with one vote there is "
          "nothing to disagree with, so the rate cannot be measured, only assumed.",
          file=out)


SENTIMENTS = ["negative", "neutral", "positive"]

# Distinct from SENTIMENTS on purpose: 0 is not a fourth point on the 1-5
# scale, it is a coder saying the comment has no stance to rate at all (spam,
# "nice video, subscribed!", or purely factual with no evident feeling) --
# see #63. Feeding it through the (value - 3) / 2 transform below would read
# as -1.5, i.e. "more negative than very negative", which is not what a
# coder means by it. It is deliberately not scored as a fourth class the
# sentiment arms are judged against either: no arm is currently asked to
# predict "irrelevant", only negative/neutral/positive, so a 0-rated comment
# is excluded from the sentiment exercise rather than counted as a miss.
IRRELEVANT = "irrelevant"

# A second, deliberately different exclusion from IRRELEVANT: the comment
# *might* carry a real stance, but the coder cannot judge it -- it is in a
# language they do not read, or is genuinely incomprehensible (garbled text,
# a bare emoji string with no clear valence, an inside joke with no legible
# meaning). Conflating this with IRRELEVANT would hide a real limitation: a
# story whose comment section skews non-English would show as having a lot
# of "no stance" content, when the honest description is "we could not code
# a stance either way" -- a coverage gap, not a finding that the comments
# there are apathetic.
UNCODEABLE = "uncodeable"


def bucket_hand_sentiment(raw):
    """Map a `hand_sentiment` cell to one of SENTIMENTS, IRRELEVANT,
    UNCODEABLE, or "" if it's none of those (blank, a typo, or out of the
    -1-5 range).

    Coders may fill this column with the three words directly, `0` (or the
    word "irrelevant") for a comment with no sentiment to rate, `-1` (or the
    word "uncodeable") for a comment they cannot judge at all -- a different
    language, or genuinely incomprehensible text -- or a 1-5 scale (1=very
    negative ... 5=very positive) matching sentiment.py's SCALE. A 1-5 value
    is bucketed with the exact same (value - 3) / 2 -> to_label transform the
    LLM scorers use on their own rating, so a coder's "4" and a scorer's
    rating of 4 land on the same side of the same threshold -- the kappa is
    comparing two labels built the same way, not two different cuts.
    """
    raw = (raw or "").strip().lower()
    if raw in SENTIMENTS:
        return raw
    if raw == IRRELEVANT:
        return IRRELEVANT
    if raw == UNCODEABLE:
        return UNCODEABLE
    try:
        scale_value = float(raw)
    except ValueError:
        return ""
    if scale_value == 0:
        return IRRELEVANT
    if scale_value == -1:
        return UNCODEABLE
    if not 1 <= scale_value <= 5:
        return ""
    return sentiment_mod.to_label((scale_value - 3) / 2)


def sentiment_report(rows, scorer, weights=None, concurrency=1):
    """Score one sentiment arm, returning everything needed to judge it.

    `weights` (one per row) are the gold rows' Horvitz-Thompson weights
    (validate.gold_weight), the same weights run_arm applies to the domain
    scorecard. Accuracy, the majority baseline and per-class precision/recall/F1
    are weighted, so they describe the corpus rather than the stratified draw;
    without them the sentiment table described the draw while the domain table
    beside it described the corpus. None (the default) weighs every row 1. The
    raw counts (`n`, `gold_n`, `tp`/`fp`/`fn`, the confusion matrix) stay
    integers, for a reader to sanity-check.

    Cost is this arm's delta, for the same reason run_arm reports one: under
    `combined:X` every arm in `cmd_sentiment` is built with the same settings
    and therefore shares one judge, so reading the cumulative total billed arm
    2 for arm 1. Without this the scorecard reported per-arm cost while the
    sentiment table reported cumulative cost under the same column heading.

    Accuracy alone is not enough and never was. VADER scored 38% on 50
    hand-coded comments -- which reads as "weak" until you notice that always
    guessing "negative" scores 58% on the same rows, at which point it reads as
    "worse than not trying" (#14). So the majority-class baseline is computed
    from the same gold labels and reported beside it.

    The confusion matrix is what turns "weak" into "inverted": VADER's failure
    was 14 of 29 genuinely negative comments scored positive, because "support"
    and emphatic punctuation are positive in its lexicon. A fat off-diagonal
    negative->positive cell is a different problem from noise, and only the
    matrix distinguishes them.
    """
    before_cost = scorer.cost_usd() or 0.0
    before_seconds = scorer.seconds
    before_input, before_output = scorer.token_usage()
    concurrency = effective_concurrency(scorer.describe(), concurrency)
    started = time.monotonic()
    results = list(label_in_order(lambda row: scorer.score(row[0], story=row[2]), rows,
                                  concurrency))
    scored = [(result, gold, len(body.split()))
              for result, (body, gold, _story) in zip(results, rows)]
    weights = list(weights) if weights is not None else [1.0] * len(rows)
    if len(weights) != len(rows):
        raise ValueError(f"{len(weights)} weights for {len(rows)} rows")
    n = sum(weights) or 1
    correct = sum(w for (r, g, _), w in zip(scored, weights) if r["label"] == g)

    counts = collections.Counter(gold for _, gold, _ in rows)
    weighted_counts = collections.Counter()
    for (_, gold, _), w in zip(rows, weights):
        weighted_counts[gold] += w
    majority_label, majority_n = (weighted_counts.most_common(1)[0] if weighted_counts
                                  else ("", 0))

    matrix = {g: {p: 0 for p in SENTIMENTS} for g in SENTIMENTS}
    for result, gold, _ in scored:
        if gold in matrix and result["label"] in matrix[gold]:
            matrix[gold][result["label"]] += 1

    # A hand-coded negative/neutral/positive comment that the scorer itself
    # calls irrelevant or uncodeable (sentiment.py's LLMScorer can answer
    # either now, matching the same two sentinels a human coder can use) is
    # not a fourth confusion-matrix cell -- it's a different KIND of miss
    # ("this comment has no stance to rate", when a human coder gave it one)
    # from an ordinary wrong-class error, and folding it into the matrix's
    # off-diagonal the way the loop above already skips it would hide that
    # distinction. It still counts against `correct`/`accuracy` above like
    # any other miss -- this just makes the miss visible and countable
    # instead of silently vanishing from the confusion matrix with no trace.
    off_scale = collections.Counter(
        r["label"] for r, _, _ in scored if r["label"] not in SENTIMENTS)

    per_class = {}
    for target in SENTIMENTS:
        tp = sum(1 for r, g, _ in scored if r["label"] == target and g == target)
        fp = sum(1 for r, g, _ in scored if r["label"] == target and g != target)
        fn = sum(1 for r, g, _ in scored if r["label"] != target and g == target)
        w_tp = sum(w for (r, g, _), w in zip(scored, weights)
                   if r["label"] == target and g == target)
        w_fp = sum(w for (r, g, _), w in zip(scored, weights)
                   if r["label"] == target and g != target)
        w_fn = sum(w for (r, g, _), w in zip(scored, weights)
                   if r["label"] != target and g == target)
        precision, recall, f1 = prf(w_tp, w_fp, w_fn)
        per_class[target] = {"tp": tp, "fp": fp, "fn": fn,
                             "precision": precision, "recall": recall, "f1": f1,
                             "gold_n": counts.get(target, 0)}

    return {
        "arm": scorer.describe(),
        "n": len(scored),
        "ht_weighted": any(w != 1.0 for w in weights),
        "accuracy": correct / n,
        "majority_class": majority_label,
        "majority_baseline": majority_n / n,
        "beats_baseline": (correct / n) > (majority_n / n),
        "per_class": per_class,
        "confusion": matrix,
        "off_scale": dict(off_scale),
        # abs(None) crashes -- a row the scorer called irrelevant/uncodeable
        # carries value=None (see sentiment.py's LLMScorer.score), not a
        # stance value on [-1, 1], so it has nothing to correlate with length.
        "r_length": pearson([(words, abs(r["value"])) for r, _, words in scored
                             if r["value"] is not None]),
        # A share of ROWS, not of weight: `n` above is the Horvitz-Thompson total, so dividing by
        # it gave 0.001 for a gold set where every comment was unstable and weighed 1,000.
        "unstable_rate": sum(1 for r, _, _ in scored if r.get("unstable")) / (len(scored) or 1),
        # None when the scorer does not say. At 1 vote nothing can disagree with itself, so the
        # rate is 0 by construction and the table shows a dash, as the domain scorecard does.
        "votes": getattr(scorer, "votes", None),
        "cost_usd": (scorer.cost_usd() or 0.0) - before_cost,
        "concurrency": concurrency,
        "seconds": (time.monotonic() - started) if concurrency > 1
                   else scorer.seconds - before_seconds,
        "input_tokens": scorer.token_usage()[0] - before_input,
        "output_tokens": scorer.token_usage()[1] - before_output,
    }


def special_case_report(rows, scorer, expected_label, weights=None, concurrency=1):
    """How often does an arm's own irrelevant/uncodeable answer agree with
    the same call a human coder made? `rows` are (body, gold, story) tuples
    where every `gold` is already `expected_label` (IRRELEVANT or UNCODEABLE)
    -- cmd_sentiment is the only caller, once per label.

    Deliberately not folded into sentiment_report's 3-way scorecard as a
    fourth class: IRRELEVANT/UNCODEABLE are categorical outs, not points on
    the negative-neutral-positive axis (see sentiment.py's LLMScorer.score,
    which keeps them out of its own -1..1 stance value for the same reason),
    so mixing them into that confusion matrix would answer two different
    questions -- "what stance?" and "does this even have a stance to rate?"
    -- with one blended number instead of each on its own terms.
    """
    if not rows:
        return None
    concurrency = effective_concurrency(scorer.describe(), concurrency)
    weights = list(weights) if weights is not None else [1.0] * len(rows)
    labels = [result["label"] for result in label_in_order(
        lambda row: scorer.score(row[0], story=row[2]), rows, concurrency)]
    predicted = collections.Counter(labels)
    matched = predicted.get(expected_label, 0)
    # Recall is Horvitz-Thompson weighted, like sentiment_report; `matched` stays a count.
    weighted_matched = sum(w for label, w in zip(labels, weights) if label == expected_label)
    return {"n": len(rows), "matched": matched,
            "recall": weighted_matched / (sum(weights) or 1),
            "predicted": dict(predicted)}


def cmd_sentiment(args):
    """Score sentiment arms against a hand-coded `hand_sentiment` column.

    Reported separately from the harm domains because #9 is a separate claim:
    that VADER's compound score is length-dependent and so confounds the
    per-story comparison. Accuracy is only half the question — the other half
    is whether a scorer's output correlates with comment length, which is
    printed here as well.
    """
    rows = []
    irrelevant_rows, uncodeable_rows = [], []
    # Horvitz-Thompson weights, parallel to the three lists above (see sentiment_report).
    weights, irrelevant_weights, uncodeable_weights = [], [], []
    all_rows = []
    invalid = 0
    # Comments shown to the model as worked examples are left out: each would be scored
    # against its own answer (the same rule the domain scorecard already applies).
    excluded = _exclude_ids(args) or set()
    left_out = 0
    with open(args.input_csv, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            body = (row.get("body") or "").strip()
            if not body:
                continue
            if excluded and validate_mod.clean_id(row.get("comment_id")) in excluded:
                left_out += 1
                continue
            all_rows.append(row)
            gold = bucket_hand_sentiment(row.get("hand_sentiment"))
            story = row.get("story")
            if gold == IRRELEVANT:
                irrelevant_rows.append((body, gold, story))
                irrelevant_weights.append(row)
            elif gold == UNCODEABLE:
                uncodeable_rows.append((body, gold, story))
                uncodeable_weights.append(row)
            elif not gold:
                invalid += 1
            else:
                rows.append((body, gold, story))
                weights.append(row)
    # The same refusal read_gold applies to the domain scorecard: a file mixing blank and
    # recorded inclusion probabilities, or holding one outside (0, 1], cannot be weighted.
    weighted = validate_mod.require_gold_weights(all_rows, args.input_csv) if all_rows else False
    weights = [validate_mod.gold_weight(r) for r in weights]
    irrelevant_weights = [validate_mod.gold_weight(r) for r in irrelevant_weights]
    uncodeable_weights = [validate_mod.gold_weight(r) for r in uncodeable_weights]
    if left_out:
        print(f"  {left_out} comment(s) left out of scoring: shown to the model as worked examples "
              f"(or listed in --exclude-ids).", file=sys.stderr)
    if irrelevant_rows:
        print(f"  {len(irrelevant_rows)} comment(s) marked irrelevant (hand_sentiment=0) -- kept "
              f"out of the main negative/neutral/positive scorecard below (that is still a "
              f"3-way exercise), but each arm is separately checked against them further down, "
              f"now that an LLM arm can answer 'irrelevant' too instead of only ever predicting "
              f"a stance for everything it sees.", file=sys.stderr)
    if uncodeable_rows:
        print(f"  {len(uncodeable_rows)} comment(s) marked uncodeable (hand_sentiment=-1, e.g. a "
              f"different language) -- same treatment as irrelevant above: out of the main "
              f"scorecard, checked separately further down.", file=sys.stderr)
    if invalid:
        print(f"  WARNING: {invalid} comment(s) had an unrecognised hand_sentiment value "
              f"(blank, a typo, or outside -1 to 5) and were excluded. Check the file before "
              f"trusting these numbers.", file=sys.stderr)
    if not rows:
        sys.exit(f"No rows with a `hand_sentiment` of {'/'.join(SENTIMENTS)} (or a 1-5 scale value) "
                  f"in {args.input_csv}.")

    print(f"\n| arm | n | accuracy | baseline | beats it? | "
          + " | ".join(f"{s} F1" for s in SENTIMENTS)
          + " | r(length) | unstable | cost | mins | in tok | out tok |")
    print("|---" * (len(SENTIMENTS) + 11) + "|")

    reports = []
    for arm in args.arm:
        try:
            scorer = sentiment_mod.create(arm, votes=args.votes, cache_path=args.cache)
        except Exception as error:
            print(f"   skipped {arm}: {type(error).__name__}: {error}", file=sys.stderr)
            continue
        print(f"=> {arm}", file=sys.stderr)
        try:
            rep = sentiment_report(rows, scorer, weights,
                                   concurrency=getattr(args, "concurrency", 1))
        except Exception as error:
            print(f"   failed: {type(error).__name__}: {str(error)[:150]}", file=sys.stderr)
            continue
        reports.append(rep)
        cells = " | ".join(fmt(rep["per_class"][s]["f1"]) for s in SENTIMENTS)
        print(f"| `{rep['arm']}` | {rep['n']} | {rep['accuracy']:.0%} | "
              f"{rep['majority_baseline']:.0%} (`{rep['majority_class']}`) | "
              f"{'yes' if rep['beats_baseline'] else '**NO**'} | {cells} | "
              f"{fmt(rep['r_length'])} | "
              f"{'—' if rep['votes'] is not None and rep['votes'] <= 1 else format(rep['unstable_rate'], '.0%')} | "
              f"{cost(rep['cost_usd'])} | {rep['seconds'] / 60:.1f} | "
              f"{rep['input_tokens']:,} | {rep['output_tokens']:,} |")
        if rep["off_scale"]:
            off = ", ".join(f"{n} called {label}" for label, n in rep["off_scale"].items())
            print(f"   {sum(rep['off_scale'].values())} of the {rep['n']} hand-coded "
                  f"negative/neutral/positive comments above were scored {off} instead of a "
                  f"stance -- counted as wrong in accuracy/F1 above, same as any other miss, "
                  f"but shown here rather than left indistinguishable from a wrong-class error "
                  f"in the confusion matrix.", file=sys.stderr)
        for expected_label, special_rows, special_weights in (
                (IRRELEVANT, irrelevant_rows, irrelevant_weights),
                (UNCODEABLE, uncodeable_rows, uncodeable_weights)):
            special = special_case_report(special_rows, scorer, expected_label, special_weights,
                                          concurrency=getattr(args, "concurrency", 1))
            if special is None:
                continue
            others = {label: n for label, n in special["predicted"].items()
                      if label != expected_label}
            other_str = f" (called {others} instead)" if others else ""
            print(f"   {expected_label}: of {special['n']} hand-coded {expected_label} "
                  f"comment(s), this arm also called {special['matched']} {expected_label} "
                  f"({special['recall']:.0%} recall){other_str}.", file=sys.stderr)

    _note = concurrency_footnote(reports)
    if _note:
        print(f"\n{_note}")
    if weighted and reports:
        print("\nAccuracy, baseline and F1 are Horvitz-Thompson weighted by `inclusion_prob`, like "
              "the domain scorecard, so they describe the corpus rather than the stratified draw. "
              "`n` and the confusion matrix are raw counts.")

    for rep in reports:
        print(f"\nConfusion matrix -- {rep['arm']} (rows = hand codes, columns = scorer):",
              file=sys.stderr)
        print("  " + " " * 12 + "".join(f"{s:>10}" for s in SENTIMENTS), file=sys.stderr)
        for gold in SENTIMENTS:
            row = rep["confusion"][gold]
            print(f"  {gold:<12}" + "".join(f"{row[p]:>10}" for p in SENTIMENTS), file=sys.stderr)
        worst = max(((g, p, rep["confusion"][g][p]) for g in SENTIMENTS for p in SENTIMENTS
                     if g != p), key=lambda t: t[2], default=None)
        if worst and worst[2]:
            print(f"  Largest off-diagonal cell: {worst[2]} comment(s) coded {worst[0]} were "
                  f"scored {worst[1]}.", file=sys.stderr)

def pearson(pairs):
    n = len(pairs)
    if n < 2:
        return float("nan")
    mx = sum(x for x, _ in pairs) / n
    my = sum(y for _, y in pairs) / n
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den = (sum((x - mx) ** 2 for x, _ in pairs) * sum((y - my) ** 2 for _, y in pairs)) ** 0.5
    return num / den if den else float("nan")


def split_by_video(rows, pool_fraction, seed=42):
    """Whole videos go to one side, so the pool and the held-out set share no thread.

    A comment and its replies, or two comments under one video, are correlated:
    splitting by comment lets the pool "teach" the held-out set about the same
    thread, which flatters the held-out scores. Videos are shuffled (seeded) and
    added to the pool until it holds `pool_fraction` of the rows. Category balance
    is not forced -- it is reported, and a category missing from a side is warned
    about, because with few videos that can happen.
    """
    by_video = collections.defaultdict(list)
    for row in rows:
        by_video[row.get("source") or row.get("comment_id") or ""].append(row)
    videos = sorted(by_video)
    random.Random(seed).shuffle(videos)
    target = pool_fraction * len(rows)
    pool, heldout = [], []
    for video in videos:
        (pool if len(pool) < target else heldout).extend(by_video[video])
    for name, part in (("pool", pool), ("held-out", heldout)):
        counts = collections.Counter(c for r in part for c in r["_gold"])
        print(f"  {name}: {len(part)} rows from "
              f"{len({r.get('source') for r in part})} videos; per category "
              f"{', '.join(f'{c} {counts[c]}' for c in CATEGORIES)}", file=sys.stderr)
        missing = [c for c in CATEGORIES if counts[c] == 0
                   and any(c in r["_gold"] for r in rows)]
        if missing:
            print(f"WARNING: no {', '.join(missing)} comment in the {name} set. Try another "
                  f"--seed, or accept it and say so.", file=sys.stderr)
    return pool, heldout


def cmd_split(args):
    """Stratify by gold label so both halves carry every category, and
    `democracy` in particular is not concentrated in one of them."""
    rows = read_gold(args.input_csv, args.label_column, exclude_ids=_exclude_ids(args))
    random.seed(args.seed)
    pool, heldout = [], []
    if getattr(args, "split_by", "comment") == "video":
        pool, heldout = split_by_video(rows, args.pool_fraction, args.seed)
    else:
        strata = collections.defaultdict(list)
        for row in rows:
            strata[tuple(sorted(row["_gold"])) or ("none",)].append(row)
        for _key, group in sorted(strata.items()):
            random.shuffle(group)
            cut = round(len(group) * args.pool_fraction)
            # Never let a stratum contribute to only one side if it can fill both.
            if len(group) > 1:
                cut = min(max(cut, 1), len(group) - 1)
            pool += group[:cut]
            heldout += group[cut:]

    # A ragged row gives csv.DictReader a None key for the overflow, so guard
    # rather than raising AttributeError three lines into writing output.
    if any(f is None for f in rows[0]):
        sys.exit(f"{args.input_csv}: a row has more fields than the header. Check for an "
                 f"unquoted comma in a comment body.")
    fieldnames = [f for f in rows[0] if not f.startswith("_")]
    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    for name, part in (("pool", pool), ("heldout", heldout)):
        path = f"{args.out_prefix}_{name}.csv"
        # utf-8-sig, not utf-8: these files are meant to be opened directly in
        # Excel (e.g. to spot-check the pool), and Excel needs the BOM to
        # auto-detect UTF-8 rather than garbling non-ASCII text.
        with open(path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(part)
        print(f"Wrote {len(part)} rows to {path}", file=sys.stderr)
    print("\nThe held-out set must never be added to the retrieval pool. A comment "
          "that retrieves its own gold label as its own example scores near-perfect "
          "and measures nothing.", file=sys.stderr)


def cmd_score(args):
    rows = read_gold(args.input_csv, args.label_column, exclude_ids=_exclude_ids(args))
    judge, selector = build(args.judge, args.pool, args.votes, args.k,
                            args.label_column, args.cache)
    print(f"Scoring {judge.spec} / {selector.describe()} on {len(rows)} comments",
          file=sys.stderr)
    print_scorecard([run_arm(rows, judge, selector, concurrency=getattr(args, "concurrency", 1))])


def non_fixed_arms(arms, default_selector="fixed"):
    """Arms that would not show the model the 25 fixed examples.

    Benchmark policy: every arm is compared on the same fixed examples, so a
    difference between arms is a difference between models. The keyword judge
    ignores examples altogether, so its selector label is irrelevant.
    """
    offenders = []
    for arm in arms:
        spec, _, name = arm.rpartition("/")
        if not spec:
            spec, name = arm, default_selector
        if spec.split(":", 1)[0] == "keyword":
            continue
        if name != "fixed":
            offenders.append(arm if "/" in arm else f"{arm}/{name}")
    return offenders


def _exclude_ids(args):
    """Comments to leave out of scoring: any --exclude-ids file, plus every comment
    loaded as a fixed example (those are in the prompt, so they would be scored
    against their own labels)."""
    path = getattr(args, "exclude_ids", None)
    ids = set(validate_mod.read_ids(path)) if path else set()
    ids |= set(taxonomy.FIXED_EXAMPLES_SOURCE.get("ids", []))
    return ids or None


def cmd_scorecard(args):
    offenders = non_fixed_arms(args.arm, getattr(args, "selector", "fixed"))
    if offenders and not getattr(args, "allow_non_fixed", False):
        sys.exit("Benchmark arms must all use the 25 fixed examples, so a difference between "
                 f"them is a difference between models. Not fixed: {', '.join(offenders)}. "
                 "Use `<judge>/fixed`, or pass --allow-non-fixed for an exploratory run "
                 "that is not part of the comparison.")
    rows = read_gold(args.input_csv, args.label_column, exclude_ids=_exclude_ids(args))
    results = []
    for arm in args.arm:
        print(f"\n=> {arm}", file=sys.stderr)
        # One unavailable or failing backend (no key, no ollama, no embeddings,
        # no credit, a rate limit part-way through) must not throw away the arms
        # that did run. The run is guarded as well as the construction: an API
        # arm can fail on call 200 of 300, which is exactly when losing the
        # other arms' results hurts most.
        try:
            judge, selector = build(arm, args.pool, args.votes, args.k,
                                    args.label_column, args.cache,
                                    getattr(args, "selector", "fixed"))
        except Exception as error:
            print(f"   skipped: {type(error).__name__}: {error}", file=sys.stderr)
            continue
        try:
            results.append(run_arm(rows, judge, selector, concurrency=getattr(args, "concurrency", 1)))
        except KeyboardInterrupt:
            raise
        except Exception as error:
            print(f"   failed after {judge.usage['calls']} calls: "
                  f"{type(error).__name__}: {str(error)[:200]}", file=sys.stderr)
            if judge.cost_usd():
                print(f"   (spent ${judge.cost_usd():.2f} before failing)", file=sys.stderr)
    if not results:
        sys.exit("No arms ran.")
    print_scorecard(results)
    if getattr(args, "bootstrap", 0):
        print_intervals(results, args.bootstrap, seed=args.seed)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["split", "score", "scorecard", "sentiment"])
    parser.add_argument("input_csv")
    parser.add_argument("--judge", default="ollama:gemma3:12b",
                        help=f"score only. One of: {', '.join(judges.available())}")
    parser.add_argument("--arm", action="append", default=[],
                        help="scorecard only, repeatable: '<judge>/<selector>'")
    parser.add_argument("--pool", help="Hand-coded CSV used as the retrieval pool")
    parser.add_argument("--selector", default="fixed",
                        help=f"Default selector when an arm omits one: "
                             f"{', '.join(examples_mod.available())}")
    parser.add_argument("--allow-non-fixed", action="store_true",
                        help="scorecard only: permit an arm whose selector is not `fixed`. "
                             "Benchmark comparisons use the 25 fixed examples for every arm; "
                             "this is for exploratory runs outside that comparison.")
    parser.add_argument("--k", type=int, default=12, help="Examples to retrieve")
    parser.add_argument("--votes", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="scorecard, sentiment and score: how many comments to label at once "
                             "(default 1, as before). An API arm spends its time waiting on the "
                             "network, so this turns a scorecard measured in hours into one "
                             "measured in minutes; the results are identical, only the clock "
                             "changes. Local (Ollama or Gemma) arms always run at 1. Stay well inside "
                             "your provider's rate limits (check the Limits page first).")
    parser.add_argument("--label-column", default="hand_domain")
    parser.add_argument("--cache", help="JSONL response cache; makes a rerun free and exact")
    parser.add_argument("--pool-fraction", type=float, default=0.7, help="split only")
    parser.add_argument("--out-prefix", default="gold",
                        help="split only: writes <prefix>_pool.csv and <prefix>_heldout.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-ids",
                        help="scorecard, sentiment, split: a file of comment_ids (one per line, or "
                             "a CSV with a comment_id column) to leave out of scoring -- the "
                             "gold comments shown to the model as worked examples, which would "
                             "otherwise be scored against their own labels.")
    parser.add_argument("--bootstrap", type=int, default=0, metavar="N",
                        help="scorecard only: also print 95%% intervals on each category's "
                             "precision, recall and F1, from N resamples of whole videos "
                             "(try 1000). Off by default.")
    parser.add_argument("--split-by", choices=["comment", "video"], default="comment",
                        help="split only: 'video' keeps every video's comments on one side, so "
                             "the pool and held-out set share no thread (the default, 'comment', "
                             "is the original behaviour).")
    instrument.add_args(parser)
    args = parser.parse_args()
    # The two commands that ask the sentiment question refuse a ratings-less examples file.
    instrument.apply(args, needs_ratings=args.command in ("sentiment", "scorecard"))

    if args.command == "sentiment":
        if not args.arm:
            sys.exit(f"sentiment needs at least one --arm, e.g. 'vader' or "
                     f"'llm:anthropic:claude-haiku-4-5'")
        cmd_sentiment(args)
    elif args.command == "score":
        args.judge = args.judge if "/" in args.judge else f"{args.judge}/{args.selector}"
        cmd_score(args)
    elif args.command == "scorecard":
        if not args.arm:
            sys.exit("scorecard needs at least one --arm")
        cmd_scorecard(args)
    else:
        cmd_split(args)


if __name__ == "__main__":
    main()
