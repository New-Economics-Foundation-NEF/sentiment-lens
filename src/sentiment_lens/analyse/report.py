"""Confidence intervals for classified proportions, via bootstrap resampling.

    # pooled proportion of comments carrying the democracy label, with a 95% CI
    python report.py observed master_classified.csv --column domain_democracy

    # the same, broken out per story rather than pooled
    python report.py observed master_classified.csv --column domain_democracy --by story

    # one aggregate figure across all four stories, weighted equally per story
    # (not by raw comment count -- see the module docstring below)
    python report.py aggregate master_classified.csv --column domain_democracy

    # classifier-error-corrected estimate for one category, using precision/
    # recall from a scored gold set (evaluate.py's run_arm output)
    python report.py corrected master_classified.csv --column domain_democracy \\
        --tpr 0.85 --fpr 0.10

Three things this module does, and one thing it deliberately does not do yet.

1. Bootstrap by video, not by comment
-----------------------------------
Comments under one video are not independent observations -- same audience,
same thread, people replying to each other. A comment-level bootstrap
reproduces a naively narrow interval; resampling videos (identified by the
`source` column, the video URL every collector writes) with replacement, and
taking every comment that belongs to each resampled video, is the correct
unit of resampling. This is why every function below takes a `cluster_by`
column (default `"source"`) rather than resampling rows directly.

2. Equal-story weighting for any aggregate figure
--------------------------------------------------
"Overall sentiment towards wealth-related harms" or "the most dominant
topics" collapse four story-level numbers into one, and how you collapse
them is a choice: weight by raw comment count (the four stories were not
sampled proportional to anything, so this mostly reports whichever story
happened to generate the most comments), or weight each story equally (the
convention that fits a small set of deliberately-chosen illustrative cases,
per the project's design). `aggregate_ci` below always does the latter --
each story's own rate is computed first, and the four are then averaged with
equal weight, before any bootstrapping happens. This applies ONLY when
collapsing across the four stories. It must never be applied across harm
categories: a category's prevalence is the finding, not an artefact of
collection, so category proportions are reported as observed, never
equalised.

3. Classifier-error correction (Rogan-Gladen)
-----------------------------------------------
A classified corpus reports what the classifier said, not necessarily the
truth. Given a category's true positive rate (TPR, i.e. recall) and false
positive rate (FPR) from a hand-coded gold set:

    corrected_rate = (observed_rate - FPR) / (TPR - FPR)

`corrected_estimate` computes this as a point estimate -- always reported
alongside the observed rate and the observed rate's own bootstrap interval,
never in place of them. The corrected estimate does not get an interval by
default: propagating uncertainty in TPR/FPR requires resampling the gold set
too, which is unstable exactly where it matters most (a thin category, with
few genuinely-positive gold instances, can have its resampled TPR swing to a
nonsensical value purely by chance -- see `corrected_ci_bootstrap`'s
docstring). `min_genuine_positives` gates this: below the threshold, report
"too rare in the gold set to correct reliably" rather than a number built on
an unstable base.

What this module does NOT do: correct for classifier error with a formal
propagated interval by default. `corrected_ci_bootstrap` provides that
capability, but it should only be invoked once a category's genuine-positive
count in the gold set is checked and found sufficient (floor 10 effective, flagged LOW EVIDENCE below 50) --
call `corrected_ci_bootstrap` only after inspecting `genuine_positive_count`.

`corrected_estimate` and `corrected_ci_bootstrap` both use ONE pooled TPR/FPR
per category, applied identically to every story. That assumes the
classifier makes the same kind of mistakes at the same rate regardless of
which story a comment is from -- plausible, but unmeasured by default, since
nothing about a pooled TPR/FPR can tell you whether it does. If a story's
gold set is large enough to check (`min_gold_per_story` in
`corrected_estimate_by_story`), that function nudges the pooled correction
per story by that story's own overall accuracy relative to the corpus-wide
average -- cheaper than a full per-story-per-category correction, which a
~250-comment gold set is usually too thin to support in every cell, but not
free of assumptions either: see its docstring for what it does not catch.

No numpy: resampling a few thousand iterations over a few hundred to a few
thousand rows is fast enough in pure Python, and it keeps this module usable
anywhere the rest of the pipeline already runs, with no new dependency.
"""
import argparse
import collections
import csv
import random
import statistics
import sys

from sentiment_lens import corpus as corpus_mod
from sentiment_lens.analyse import validate as validate_mod

# Windows defaults to cp1252 for stdout, which mangles the en-dash in every
# "lo%-hi%" interval this module prints into a "?" -- the same fix already
# applied to validate.py/evaluate.py/classify.py/keywords.py, just missed here.
sys.stdout.reconfigure(encoding="utf-8")

_FALSY = {"", "0", "0.0", "false", "no"}

# How much weighted evidence the gold set needs before a corrected figure is given at all.
# Counted as EFFECTIVE genuine positives (Horvitz-Thompson weights, Kish effective size), not rows.
#   MIN_EFFECTIVE_POSITIVES      the floor: below it the correction is refused. Lowered from 50 to 10
#                                on 2026-10-13: with the final 718-row gold set every category had
#                                fewer than 50 effective positives even pooled (12 to 30), so at 50
#                                no corrected figure could be produced.
#   WELL_SUPPORTED_POSITIVES     at or above this the figure is well supported; between the two it is
#                                reported but flagged LOW EVIDENCE, and its interval will be wide.
# The 5% degenerate-resample rule and the two-video gold minimum are unchanged and still refuse an
# interval that cannot be trusted.
MIN_EFFECTIVE_POSITIVES = 10
WELL_SUPPORTED_POSITIVES = 50


def low_evidence(effective_positives):
    """True when a figure resting on this many effective positives needs the LOW EVIDENCE caveat."""
    return effective_positives < WELL_SUPPORTED_POSITIVES


def evidence_tag(effective_positives):
    """The caveat printed on the same line as a corrected figure ('' when well supported)."""
    if not low_evidence(effective_positives):
        return ""
    return f" [LOW EVIDENCE: {effective_positives:.1f} effective positives]"

# Two denominators for a domain share (dual statistics).
#   all         every comment the classifier ANSWERED (a parse failure is not an answer).
#               Includes the comments it judged to engage none of the four harms, so
#               "democracy 12%" is a share of the whole corpus. The default, and what every
#               earlier figure used.
#   classified  only the comments the classifier tagged with at least one harm
#               (domain_unclassified is not 1, and not a parse failure). "democracy 30%"
#               is then a share of the comments the classifier thought engaged a harm.
#               A comment can carry two labels, so these shares can sum to over 100%.
BASES = ("all", "classified")
BASE_LABELS = {
    "all": "all answered comments (includes comments the classifier tagged with no harm)",
    "classified": "comments the classifier tagged with at least one harm "
                  "(unclassified and parse failures left out)",
}
_CONSTANT_IN_CLASSIFIED = ("domain_unclassified", "domain_parse_failure")


def restrict_to_base(rows, base="all", what="rows", column=None):
    """The rows a figure in `base` is computed over. `all` returns `rows` untouched.

    `classified` keeps rows the classifier tagged with at least one harm. The same
    restriction must be applied to the gold rows as to the corpus rows: the correction
    (Rogan-Gladen, PPI) is only valid when both sides describe the same population, so a
    corrected `classified` figure is the true share among comments THE CLASSIFIER tagged,
    not the share among comments that are truly about a harm.
    """
    if base not in BASES:
        raise ValueError(f"base must be one of {BASES}, not {base!r}")
    if base == "all":
        return rows
    if column in _CONSTANT_IN_CLASSIFIED:
        raise ValueError(f"{column} is 0 by construction in the 'classified' base; "
                         f"report it in the 'all' base.")
    rows = list(rows)
    if rows and "domain_unclassified" not in rows[0]:
        raise ValueError(f"The 'classified' base needs a domain_unclassified column in the "
                         f"{what} (the classifier's own 'none' flag), and it is missing. For a "
                         f"gold file, run it through slens-classify or slens-validate join first.")
    return [r for r in rows
            if r.get("domain_unclassified") != "1" and r.get("domain_parse_failure") != "1"]


def _as_float(value, column="?"):
    """A row's raw string value as a number for bootstrapping.

    Handles the two shapes this pipeline actually produces: a 0/1 domain flag
    written as text ("1", "0", ""), and a genuine numeric column like
    `sentiment_value`.

    Anything else raises. It used to return 1.0, which silently turned a
    column of the wrong shape into a 100% rate with a zero-width interval --
    `classify.py` writes `sentiment` as positive/negative/neutral and the
    argparse help advertises `--column sentiment_value`, one character away
    and both present, so asking for the wrong one produced a
    publication-ready "100.0% (95% CI 100.0%-100.0%)" and exit 0.
    """
    text = (value or "").strip()
    if text.lower() in _FALSY:
        return 0.0
    try:
        return float(text)
    except ValueError:
        raise SystemExit(
            f"Column {column!r} contains {text!r}, which is neither a number nor a "
            f"0/1 flag. This column is not a proportion and cannot be summarised as "
            f"one -- did you mean a `domain_*` column, or `sentiment_value` rather "
            f"than `sentiment`?")


def _require_column(rows, column):
    """Fail loudly on a column that is not in the file.

    csv.DictReader returns None for a missing key, which _as_float used to read
    as falsy and report as a flat 0.0% with a zero-width CI -- a typo in
    --column produced a confident, entirely fictional figure and exit 0.
    """
    if rows and column not in rows[0]:
        raise SystemExit(
            f"No column {column!r} in the input. Available: "
            f"{', '.join(sorted(k for k in rows[0] if k and not k.startswith('_')))}")


def _clusters(rows, cluster_by):
    """{cluster_id: [row, ...]} -- e.g. every comment under one video."""
    groups = collections.defaultdict(list)
    for row in rows:
        key = (row.get(cluster_by) or "").strip() or "(unknown)"
        groups[key].append(row)
    return groups


def _resample_clusters(cluster_ids, rng):
    """One bootstrap draw: resample cluster IDs with replacement, same count."""
    n = len(cluster_ids)
    return [cluster_ids[rng.randrange(n)] for _ in range(n)]


def bootstrap_ci(values, statistic=statistics.fmean, n_boot=2000, alpha=0.05, seed=42):
    """Bootstrap (observed, lo, hi) for `statistic` over a plain list of values.

    Row-level, not cluster-aware -- used internally once a cluster-level
    resample has already produced one number per iteration. Reproducible for
    a given input: the resampling is random, the result should not be.
    """
    if not values:
        return float("nan"), float("nan"), float("nan")
    observed = statistic(values)
    if len(values) == 1:
        return observed, observed, observed
    rng = random.Random(seed)
    n = len(values)
    resampled = sorted(statistic([values[rng.randrange(n)] for _ in range(n)])
                       for _ in range(n_boot))
    lo_index = max(int((alpha / 2) * n_boot), 0)
    hi_index = min(int((1 - alpha / 2) * n_boot), n_boot - 1)
    return observed, resampled[lo_index], resampled[hi_index]


_DOMAIN_FLAG_PREFIX = "domain_"
_NOT_A_RATE = {"domain_parse_failure", "domain_unstable", "domain_judge", "domain_raw_votes",
               "domain"}


def _population(rows, column):
    """The rows a rate over `column` is a share OF.

    * `sentiment_value` is blank for a comment with no stance to rate (irrelevant,
      uncodeable, parse failure, no consensus). A blank is not a 0.0: it is left
      out, so the mean is over comments that were actually rated.
    * A `domain_<category>` flag is a share of comments the classifier ANSWERED.
      A parse failure is not an answer, so it leaves the denominator, the same way
      it leaves the gold-set TPR/FPR; both sides of the correction then describe
      the same population. (`domain_parse_failure` itself is a share of all rows.)
    """
    if column == "sentiment_value" or column.startswith("sentiment_value"):
        return [r for r in rows if (r.get(column) or "").strip() != ""]
    if column.startswith(_DOMAIN_FLAG_PREFIX) and column not in _NOT_A_RATE:
        return [r for r in rows if r.get("domain_parse_failure") != "1"]
    return rows


def _cluster_rate(rows, column):
    values = [_as_float(row.get(column), column) for row in _population(rows, column)]
    return statistics.fmean(values) if values else float("nan")


MAX_UNUSABLE_DRAW_FRACTION = 0.05


def _percentile_interval(resampled, alpha, label=""):
    """Percentile interval over the usable bootstrap draws.

    A draw is unusable (NaN) when every video it resampled has no comment in the
    rate's population -- e.g. only videos whose comments were all irrelevant, for
    `sentiment_value`. NaN cannot be ordered, so sorting a list that holds one
    leaves it in arbitrary order and the "percentiles" become meaningless. Those
    draws are dropped and counted; above MAX_UNUSABLE_DRAW_FRACTION the interval is
    refused (NaN, NaN) rather than reported, the same rule corrected_ci_bootstrap
    applies to its degenerate draws.
    """
    usable = sorted(v for v in resampled if v == v)
    dropped = len(resampled) - len(usable)
    if dropped:
        print(f"NOTE: {label}: {dropped} of {len(resampled)} bootstrap draw(s) had no comment "
              f"in the rate's population and were dropped.", file=sys.stderr)
    if not usable or dropped > MAX_UNUSABLE_DRAW_FRACTION * len(resampled):
        if usable:
            print(f"WARNING: {label}: more than {MAX_UNUSABLE_DRAW_FRACTION:.0%} of draws were "
                  f"unusable; no interval reported.", file=sys.stderr)
        return float("nan"), float("nan")
    n = len(usable)
    return (usable[max(int((alpha / 2) * n), 0)],
            usable[min(int((1 - alpha / 2) * n), n - 1)])


def column_ci(rows, column, group_by=None, cluster_by="source", n_boot=2000, seed=42, alpha=0.05,
              base="all"):
    """Bootstrap CI for the mean of `column`, resampling by `cluster_by`
    (default: video), overall or per `group_by` value (e.g. story).

    Returns {group_key: (n_comments, n_clusters, observed, lo, hi)}, with
    group_key "(pooled)" when `group_by` is not given. "Pooled" here is
    naive/count-weighted -- whichever group has the most comments (and
    videos) dominates. For an aggregate across stories weighted equally
    instead, use `aggregate_ci`.
    """
    _require_column(rows, column)
    unrestricted = rows
    rows = restrict_to_base(rows, base, "corpus", column)
    if group_by:
        groups = collections.defaultdict(list)
        # Every group present BEFORE the base restriction is kept, so a group with no
        # tagged comment is shown as n=0 rather than vanishing from the table.
        for row in unrestricted:
            groups[(row.get(group_by) or "").strip() or "(blank)"]
        for row in rows:
            groups[(row.get(group_by) or "").strip() or "(blank)"].append(row)
    else:
        groups = {"(pooled)": list(rows)}

    results = {}
    for key, group_rows in groups.items():
        if not group_rows and group_by:
            results[key] = (0, 0, float("nan"), float("nan"), float("nan"))
            continue
        clusters = _clusters(group_rows, cluster_by)
        cluster_ids = list(clusters)
        observed = _cluster_rate(group_rows, column)
        if len(cluster_ids) <= 1:
            results[key] = (len(group_rows), len(cluster_ids), observed, observed, observed)
            continue
        rng = random.Random(seed)
        resampled = []
        for _ in range(n_boot):
            drawn_ids = _resample_clusters(cluster_ids, rng)
            drawn_rows = [row for cid in drawn_ids for row in clusters[cid]]
            resampled.append(_cluster_rate(drawn_rows, column))
        lo, hi = _percentile_interval(resampled, alpha, label=f"{column} [{key}]")
        results[key] = (len(group_rows), len(cluster_ids), observed, lo, hi)
    return results


def aggregate_ci(rows, column, story_column="story", cluster_by="source",
                 n_boot=2000, seed=42, alpha=0.05, base="all"):
    """Equal-per-story aggregate CI: average the four story-level rates with
    equal weight, then bootstrap that average by resampling videos WITHIN
    each story (so every story always contributes on every iteration).

    This is the "aggregate statistics" figure -- see the module docstring for
    why equal weighting, not comment-count weighting, is the right default
    for a small set of deliberately-chosen stories. Only ever apply this
    across stories; never across harm categories.
    """
    _require_column(rows, column)
    all_stories = {(row.get(story_column) or "").strip() or "(blank)" for row in rows}
    rows = restrict_to_base(rows, base, "corpus", column)
    by_story = collections.defaultdict(list)
    for row in rows:
        by_story[(row.get(story_column) or "").strip() or "(blank)"].append(row)
    stories = sorted(by_story)
    missing = sorted(all_stories - set(stories))
    if missing:
        # An equal-per-story average over fewer stories is a different quantity, and
        # quietly reporting it next to the all-base one would be misleading.
        raise ValueError(f"No equal-per-story aggregate on the {base!r} base: no comment in "
                         f"{', '.join(missing)} is in that base, so the average would cover "
                         f"{len(stories)} of {len(all_stories)} stories.")
    if not stories:
        return float("nan"), float("nan"), float("nan"), {}

    story_clusters = {story: _clusters(by_story[story], cluster_by) for story in stories}
    story_observed = {story: _cluster_rate(by_story[story], column) for story in stories}
    observed = statistics.fmean(story_observed.values())

    rng = random.Random(seed)
    resampled = []
    for _ in range(n_boot):
        story_means = []
        for story in stories:
            clusters = story_clusters[story]
            cluster_ids = list(clusters)
            if not cluster_ids:
                continue
            drawn_ids = _resample_clusters(cluster_ids, rng)
            drawn_rows = [row for cid in drawn_ids for row in clusters[cid]]
            story_means.append(_cluster_rate(drawn_rows, column))
        # A story whose resample holds no rated comment has no mean on this draw. Averaging
        # the other stories alone would quietly re-weight the aggregate, so the whole draw
        # is marked unusable instead (and counted by _percentile_interval).
        if story_means:
            resampled.append(float("nan") if any(m != m for m in story_means)
                             else statistics.fmean(story_means))
    lo, hi = _percentile_interval(resampled, alpha, label=f"{column} [equal-per-story]")
    return observed, lo, hi, story_observed


def corrected_estimate(observed_rate, tpr, fpr):
    """Rogan-Gladen point correction: true_rate ~= (observed - FPR) / (TPR - FPR).

    Returns None (not a number) in three cases, all of which mean "too
    unreliable to correct" and should be reported as such rather than as a
    figure:

    * TPR and FPR too close -- the denominator is near zero, so any observed
      rate produces an extreme result.
    * **TPR below FPR.** Rogan-Gladen is only defined for TPR > FPR: a
      classifier that fires more often on genuine negatives than on genuine
      positives carries no usable signal about prevalence. The guard used to
      be `abs(denom) < 0.05`, which rejected near-zero denominators but let
      negative ones through, and the sign-flipped result usually lands inside
      [0, 1] so the range check below did not catch it either. Observed 0.30
      with TPR 0.20 / FPR 0.80 returned a confident 0.833; TPR 0.10 / FPR 0.30
      returned -0.0, which passes `0.0 <= value` on a negative zero. This
      fires precisely when a thin category's gold-set TPR comes out below its
      FPR, which is the case most in need of an honest "cannot say".
    * A result outside [0, 1], treated as unreliable rather than clipped,
      since clipping would hide exactly the signal that the correction is not
      trustworthy here.
    """
    denom = tpr - fpr
    # One-sided: NaN, inverted (TPR < FPR), or indistinguishable.
    if denom != denom or denom < 0.05:
        return None
    value = (observed_rate - fpr) / denom
    if not (0.0 <= value <= 1.0):
        return None
    return value


def _scoreable_gold(row, gold_label_column):
    """False for a gold row that cannot be scored: no hand code yet, or no machine label at all.

    A blank hand code is "not coded", never "none"; four blank domain_* cells mean the row was
    never labelled (a partial `slens-validate join`). Read as zeros, either one scores as a
    classifier miss or a hand-negative and moves TPR, FPR and the PPI rectifier. The commands
    leave such rows out up front and say how many; this keeps a hand-assembled file honest
    when it reaches the maths directly.
    """
    return (bool((row.get(gold_label_column) or "").strip())
            and not validate_mod.machine_unlabelled(row, "domain_"))


def _gold_tpr_fpr(rows, category, gold_label_column):
    """Sensitivity and false-positive rate of the classifier on a gold set.

    Horvitz-Thompson weighted. `slens-validate sample --stratify-by` draws a
    gold set that deliberately over-samples rare categories and records each
    row's `inclusion_prob`; every consumer then counted rows unweighted, so
    the TPR estimated on that sample was the TPR *of the stratified draw*,
    not of the corpus. Because the stratifier is a machine label correlated
    with the classifier's own guess, the two differ -- and `corrected_estimate`
    divides by (TPR - FPR), so the error propagates straight into every
    corrected rate.

    A row with no recorded probability is weighted 1, which reproduces the
    old unweighted behaviour for an unstratified sample. That is the correct
    default: a simple random sample has equal inclusion probabilities, and
    any constant cancels in these ratios.

    Module-level (not nested in `corrected_ci_bootstrap`) so `story_accuracy_
    ratios` and `corrected_estimate_by_story` can share it rather than
    re-deriving the same weighting logic.
    """
    tp = fp = fn = tn = 0.0
    for row in rows:
        # Raises on a probability outside (0, 1]: a malformed gold file is
        # refused, not silently counted at weight 1 (see validate.gold_weight).
        weight = validate_mod.gold_weight(row)
        # Parse failures are excluded, not counted as misses -- see
        # validate.cmd_score. They deflate TPR, and this function's output is
        # the denominator of corrected_estimate.
        if row.get("domain_parse_failure") == "1" or not _scoreable_gold(row, gold_label_column):
            continue
        gold = category in (row.get(gold_label_column) or "").lower()
        guess = row.get(f"domain_{category}") == "1"
        tp += weight if (gold and guess) else 0.0
        fp += weight if (guess and not gold) else 0.0
        fn += weight if (gold and not guess) else 0.0
        tn += weight if (not gold and not guess) else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return tpr, fpr


def kish_effective_n(weights):
    """Kish effective sample size, (sum of weights)^2 / (sum of squared weights); 0.0 for none."""
    weights = list(weights)
    sum_w2 = sum(w * w for w in weights)
    return (sum(weights) ** 2) / sum_w2 if sum_w2 else 0.0


def rescale_unclassified_weights(gold_rows, corpus_path, out=None):
    """Weights sensitivity: scale the gold rows that stand for each story's UNCLASSIFIED stratum so
    that their Horvitz-Thompson weights sum to that stratum's size in `corpus_path`.

    On the real gold set the unclassified strata (the 20 rows drawn with the story-by-category
    stratification plus the 80-row top-up per story) have weights summing to only about 0.7 of the
    strata's real size, while the other strata sum close to theirs. This is a post-stratification
    sensitivity analysis, NOT a correction: it assumes the strata sizes in `corpus_path` are right,
    and it does not say why the stored weights are short. Rows of other strata are left as they
    are. `corpus_path` is a frozen corpus with `story` and `primary_category` columns (the file the
    gold set was drawn from). The scale factors are computed once on the whole gold set, so inside
    a bootstrap every resample keeps the same factors.

    The unclassified strata are the rows whose `sampling_stratum` is the story itself (the top-up)
    or `<story>/unclassified`. Returns new row dicts (the input is not changed) and prints the
    factor used for each story.
    """
    out = out or sys.stderr
    sizes = collections.Counter()
    with open(corpus_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for needed in ("story", "primary_category"):
            if needed not in (reader.fieldnames or []):
                sys.exit(f"--rescale-unclassified {corpus_path}: no {needed!r} column. Pass the "
                         f"frozen corpus that has primary_category (the file the gold set was "
                         f"drawn from).")
        for row in reader:
            if (row.get("primary_category") or "").strip() == "unclassified":
                sizes[(row.get("story") or "").strip()] += 1
    in_stratum = lambda row: (row.get("sampling_stratum") or "").strip() in (
        (row.get("story") or "").strip(), (row.get("story") or "").strip() + "/unclassified")
    totals = collections.defaultdict(float)
    for row in gold_rows:
        if in_stratum(row):
            totals[(row.get("story") or "").strip()] += 1.0 / float(row["inclusion_prob"])
    factors = {}
    for story, size in sorted(sizes.items()):
        if story in totals and totals[story] > 0:
            factors[story] = size / totals[story]
            print(f"weights sensitivity: {story} unclassified stratum: gold weights sum to "
                  f"{totals[story]:,.0f} against {size:,} corpus comments; scaled by "
                  f"{factors[story]:.2f}", file=out)
    if not factors:
        sys.exit("--rescale-unclassified: no gold rows belong to an unclassified stratum "
                 "(sampling_stratum '<story>' or '<story>/unclassified'), so there is nothing to scale.")
    rescaled = []
    for row in gold_rows:
        copy = dict(row)
        story = (row.get("story") or "").strip()
        if in_stratum(row) and story in factors:
            copy["inclusion_prob"] = repr(float(row["inclusion_prob"]) / factors[story])
        rescaled.append(copy)
    return rescaled


def error_row_counts(gold_rows, category, gold_label_column):
    """Raw and effective (Kish) sizes of the gold rows the correction's FPR and TPR rest on.

    The effective-positives gate looks only at the hand-coded positives. FPR = FP / (FP + TN)
    leans just as hard on the gold NEGATIVES, and on the handful of false positives among
    them: one heavily weighted false positive can be most of the numerator while the raw
    count looks healthy. Returns {name: (raw rows, effective rows)} for false positives,
    false negatives and gold negatives. Same rows `_gold_tpr_fpr` uses.
    """
    groups = {"false positives": [], "false negatives": [], "gold negatives": []}
    for row in gold_rows:
        if row.get("domain_parse_failure") == "1" or not _scoreable_gold(row, gold_label_column):
            continue
        weight = validate_mod.gold_weight(row)
        gold = category in (row.get(gold_label_column) or "").lower()
        guess = row.get(f"domain_{category}") == "1"
        if guess and not gold:
            groups["false positives"].append(weight)
        if gold and not guess:
            groups["false negatives"].append(weight)
        if not gold:
            groups["gold negatives"].append(weight)
    return {name: (len(w), kish_effective_n(w)) for name, w in groups.items()}


def error_row_summary(gold_rows, category, gold_label_column):
    """One printed line (not a gate) naming how much weighted evidence sits behind FPR and TPR."""
    counts = error_row_counts(gold_rows, category, gold_label_column)
    parts = [f"{name} {raw} raw / {eff:.1f} effective" for name, (raw, eff) in counts.items()]
    return ("Weighted evidence behind TPR/FPR for " + repr(category) + ": " + "; ".join(parts)
            + ". A few effective false positives or negatives means one or two heavily "
              "weighted rows can move the corrected figure (not a gate; read it beside the "
              "effective positives).")


def corrected_ci_bootstrap(corpus_rows, column, gold_rows, gold_label_column,
                           category, genuine_positive_count, cluster_by="source",
                           min_genuine_positives=MIN_EFFECTIVE_POSITIVES, n_boot=2000, seed=42,
                           alpha=0.05, max_discard_fraction=0.05, min_gold_clusters=2, base="all", details=None):
    """Fully-propagated interval on the corrected estimate: resamples the
    corpus AND the gold set together, both by video, recomputing TPR/FPR and
    the correction fresh on every iteration.

    `genuine_positive_count` should be the EFFECTIVE (Horvitz-Thompson-
    weighted, Kish-style) positive count from `weighted_effective_positive_
    count`, not a raw row count -- a stratified gold draw can carry many raw
    positive rows that are worth far fewer effective ones once their weights
    are accounted for (see that function's own docstring for a measured
    example). Passing a raw count here lets a category through the gate that
    the weighted design cannot actually support.

    The gold set is not exempt from clustering just because it is a sample
    used for measurement rather than for the headline figure -- it is drawn
    from the same corpus, so gold comments from the same video are exactly as
    correlated as any other comments from that video. Resampling gold rows
    individually (the earlier version of this function) understates how much
    TPR/FPR could vary under a different draw, in precisely the same way a
    comment-level bootstrap of the corpus once understated the observed
    rate's interval -- narrower than honest, in the same direction, for the
    same reason.

    Do not call this without checking `genuine_positive_count` first --
    below `min_genuine_positives`, a resampled gold set can by chance draw
    few or none of a rare category's true instances, sending TPR to 0, 1, or
    undefined purely by luck of the draw, which is exactly the degenerate
    case `corrected_estimate` already guards against for the point estimate.
    With few genuine positives this happens often enough that a large
    fraction of iterations would need discarding, and the resulting interval
    would say more about which comments happened to be hand-coded than
    about the correction itself. `genuine_positive_count` is still a comment
    count, not a video count -- a caller drawing this threshold tighter
    should check how many distinct videos those genuine positives actually
    come from, since a handful of comments concentrated in one or two videos
    is weaker evidence than the same count spread across many.

    `details`, if a dict is passed, is filled in (the return tuple is unchanged, so existing
    callers keep working): `effective_positives` (the count the gate used, after any base
    restriction) and `low_evidence` (True below WELL_SUPPORTED_POSITIVES).

    Returns (point_corrected, lo, hi, iterations_used, iterations_discarded)
    or None -- when `genuine_positive_count` is below the threshold, when the
    gold set has fewer than `min_gold_clusters` distinct clusters (one cluster
    resamples to itself, so the interval would be zero-width), when the
    observed TPR/FPR are themselves degenerate so there is no point estimate,
    or when more than `max_discard_fraction` of iterations had to be thrown
    away. Percentiles over the surviving iterations alone would be an interval
    conditioned on the correction having worked, which is narrowest precisely
    where it is least trustworthy.
    """
    # `base` restricts BOTH the corpus and the gold rows to the same population. Pass a
    # `genuine_positive_count` computed on the same base (weighted_effective_positive_count
    # takes the same argument).
    corpus_rows = restrict_to_base(corpus_rows, base, "corpus", column)
    gold_rows = restrict_to_base(gold_rows, base, "gold rows", column)
    # On a restricted base, gate on the evidence this call actually uses: recounted after the
    # restriction, and never more than the caller passed, so a count computed on the 'all' base
    # cannot let a 'classified' figure through. (On 'all' the caller's count is trusted as before.)
    if base != "all":
        genuine_positive_count = min(
            genuine_positive_count,
            weighted_effective_positive_count(gold_rows, category, gold_label_column))
    if genuine_positive_count < min_genuine_positives:
        return None
    if details is not None:
        details["effective_positives"] = genuine_positive_count
        details["low_evidence"] = low_evidence(genuine_positive_count)

    clusters = _clusters(corpus_rows, cluster_by)
    cluster_ids = list(clusters)
    observed = _cluster_rate(corpus_rows, column)

    gold_clusters = _clusters(gold_rows, cluster_by)
    gold_cluster_ids = list(gold_clusters)
    # Cluster-resampling the gold set is right -- gold comments from one video
    # are as correlated as any others from it -- but it fails silently and in
    # the worst direction when the gold set has no cluster column. `_clusters`
    # buckets a missing key under "(unknown)", so every row lands in ONE
    # cluster, that cluster resamples to itself on every draw, TPR/FPR never
    # move, and the interval collapses to zero width: point == lo == hi, a
    # correction claiming infinite precision.
    #
    # This is not hypothetical. `slens-validate sample` writes nine columns and
    # `source` is not among them, so the file the coders hand back has no
    # cluster column at all. `slens-validate join` restores it, and since #50
    # `score` and --scored both refuse an unjoined file -- but that guarantee
    # is incidental (join happens to carry `source`), and this is a library
    # function reachable from anywhere. Refuse explicitly instead of relying
    # on a caller three modules away having done the right thing.
    if len(gold_cluster_ids) < min_gold_clusters:
        return None

    obs_tpr, obs_fpr = _gold_tpr_fpr(gold_rows, category, gold_label_column)
    point = corrected_estimate(observed, obs_tpr, obs_fpr)

    rng = random.Random(seed)
    resampled, discarded = [], 0
    for _ in range(n_boot):
        drawn_ids = _resample_clusters(cluster_ids, rng) if len(cluster_ids) > 1 else cluster_ids
        drawn_rows = [row for cid in drawn_ids for row in clusters[cid]]
        resampled_observed = _cluster_rate(drawn_rows, column)

        gold_drawn_ids = (_resample_clusters(gold_cluster_ids, rng)
                          if len(gold_cluster_ids) > 1 else gold_cluster_ids)
        resampled_gold = [row for cid in gold_drawn_ids for row in gold_clusters[cid]]
        tpr, fpr = _gold_tpr_fpr(resampled_gold, category, gold_label_column)
        corrected = corrected_estimate(resampled_observed, tpr, fpr)
        if corrected is None:
            discarded += 1
            continue
        resampled.append(corrected)

    if not resampled:
        return None
    # Taking percentiles over the survivors alone conditions the interval on
    # the correction having succeeded, which makes it systematically too narrow
    # in exactly the regime the guard exists for -- a degenerate TPR-FPR is not
    # a missing observation, it is the data saying the correction is not
    # identified here. Refuse rather than report a confident-looking interval
    # built from the iterations that happened to behave.
    if discarded > max_discard_fraction * n_boot:
        return None
    # `point` is None when the OBSERVED tpr/fpr are themselves degenerate. An
    # interval without a point estimate was being returned as if it were a
    # result; there is nothing to report in that case.
    if point is None:
        return None
    resampled.sort()
    n = len(resampled)
    lo = resampled[max(int((alpha / 2) * n), 0)]
    hi = resampled[min(int((1 - alpha / 2) * n), n - 1)]
    return point, lo, hi, len(resampled), discarded


def weighted_effective_positive_count(gold_rows, category, gold_label_column, base="all"):
    """Kish effective sample size of the gold rows hand-coded positive for `category`.

    `corrected_ci_bootstrap` used to be gated on a plain row count of genuine
    positives -- e.g. 80 for democracy. That count is not what the bootstrap
    above actually resamples with: every row also carries a Horvitz-Thompson
    weight (`1 / inclusion_prob`), and a stratified draw can concentrate its
    positives in a handful of heavily over-sampled strata, whose rows each
    carry a small weight. 80 raw positive rows sitting almost entirely inside
    one over-sampled stratum are worth far fewer than 80 independent,
    equally-weighted observations for the purpose of pinning down TPR -- Kish's
    effective-n formula, (sum of weights)^2 / (sum of squared weights), is
    exactly the number of equally-weighted rows that same total weight and
    variance is "worth". On this project's real gold set, democracy's 80 raw
    positives were worth about 13 effective ones -- comfortably clearing the
    old raw-count gate of 50, nowhere near clearing it once weighted.

    Returns 0.0 if there are no genuine positives to weight at all.

    With `base="classified"` only gold rows the classifier tagged with a harm count, so most
    of the top-up rows (drawn from the unclassified stratum) drop out and the figure falls.
    That is the honest size of the evidence for a corrected `classified` share.
    """
    gold_rows = restrict_to_base(gold_rows, base, "gold rows")
    weights = []
    for row in gold_rows:
        if category not in (row.get(gold_label_column) or "").lower():
            continue
        weight = validate_mod.gold_weight(row)
        if not _scoreable_gold(row, gold_label_column):
            continue
        weights.append(weight)
    if not weights:
        return 0.0
    sum_w = sum(weights)
    sum_w2 = sum(w * w for w in weights)
    return (sum_w ** 2) / sum_w2 if sum_w2 else 0.0


def _weighted_rectifier(gold_rows, column, category, gold_label_column):
    """PPI's rectifier term: the Horvitz-Thompson-weighted mean machine error.

    `mean_gold(hand_label - machine_label)`, weighted exactly the way
    `_gold_tpr_fpr` already weights TPR/FPR, and for the same reason -- a
    stratified gold draw over- and under-represents strata relative to the
    corpus, so an unweighted average over it describes the draw, not the
    corpus. Parse-failure rows are excluded, matching `_gold_tpr_fpr`: a
    comment the classifier could not parse is not one it got wrong, it is one
    it did not answer, and counting it as an error here would inflate the
    rectifier for the same reason it would deflate TPR there.

    Returns None only when every gold row is excluded (no weight to average
    over) -- unlike Rogan-Gladen's TPR/FPR, there is no ratio here that can
    itself be degenerate.
    """
    total_w = total_wd = 0.0
    for row in gold_rows:
        if row.get("domain_parse_failure") == "1" or not _scoreable_gold(row, gold_label_column):
            continue
        weight = validate_mod.gold_weight(row)
        gold = 1.0 if category in (row.get(gold_label_column) or "").lower() else 0.0
        guess = 1.0 if row.get(column) == "1" else 0.0
        total_w += weight
        total_wd += weight * (gold - guess)
    if total_w == 0:
        return None
    return total_wd / total_w


def _ppi_unclamped(corpus_rows, column, gold_rows, category, gold_label_column):
    """The PPI estimate before the [0, 1] clamp (None when there is nothing to average over)."""
    observed = _cluster_rate(corpus_rows, column)
    if observed != observed:
        return None
    rectifier = _weighted_rectifier(gold_rows, column, category, gold_label_column)
    if rectifier is None:
        return None
    return observed + rectifier


def ppi_point_estimate(corpus_rows, column, gold_rows, category, gold_label_column):
    """Prediction-powered (PPI) point estimate for one category's true prevalence.

        theta_ppi = mean_corpus(machine_label) + mean_gold(hand_label - machine_label)

    both terms Horvitz-Thompson weighted where a weight is available. This is
    the alternative to `corrected_estimate` (Rogan-Gladen) raised during the
    statistical review: instead of summarising the gold set into TPR/FPR first
    and dividing by their difference, PPI uses each gold comment's own
    (machine, hand) pair directly as a bias correction on top of the corpus's
    own machine-labelled rate (Angelopoulos et al., 2023).

    This does not remove the need for more gold data in a genuinely thin
    stratum -- the rectifier term above is itself an average over the gold
    set, and its sampling variance still depends on how many, and how evenly
    weighted, positives (and negatives) that set carries, exactly as
    Rogan-Gladen's TPR/FPR do. What PPI buys instead is efficiency: it never
    divides by a difference that can shrink to nothing or invert sign, so it
    cannot blow up or flip the way `corrected_estimate` can when TPR and FPR
    are close -- and, measured on comparably-sized calibration sets elsewhere,
    it can produce a visibly narrower interval from the same data. Clamped to
    [0, 1] (there is no unstable denominator here to blame an out-of-range
    value on, unlike Rogan-Gladen) rather than returned as None.

    Returns None only when the corpus rows or the gold rectifier have nothing
    to average over.
    """
    raw = _ppi_unclamped(corpus_rows, column, gold_rows, category, gold_label_column)
    return None if raw is None else _clamp01(raw)


def ppi_ci_bootstrap(corpus_rows, column, gold_rows, gold_label_column, category,
                     effective_positive_count, cluster_by="source",
                     min_genuine_positives=MIN_EFFECTIVE_POSITIVES, n_boot=2000, seed=42,
                     alpha=0.05, min_gold_clusters=2, base="all", details=None):
    """Fully-propagated interval on the PPI point estimate above.

    Mirrors `corrected_ci_bootstrap`'s resampling design exactly -- corpus AND
    gold resampled together, by cluster (video) -- for the same reason: gold
    comments from one video are as correlated as any other comments from it.
    Gated the same way too, on `effective_positive_count` (pass
    `weighted_effective_positive_count`'s result, not a raw row count -- see
    its own docstring for why) and on `min_gold_clusters`.

    Unlike `corrected_ci_bootstrap`, there is no `max_discard_fraction` guard:
    the PPI point estimate is a weighted mean difference, which is always
    defined once at least one gold row survives a resample (no ratio, no
    vanishing denominator), so "degenerate iteration" is not a failure mode
    here the way it is for Rogan-Gladen.

    PPI clamps every estimate to [0, 1]. A clamped draw is not a failure here, but it makes a
    bound that sits at 0 or 1 an artefact of the clamp, not a measured limit. `details`, if a
    dict is passed, is filled in (the return tuple is unchanged): `effective_positives`,
    `low_evidence`, `draws` (iterations used), `below_0` and `above_1` (draws clamped).

    Returns (point, lo, hi, iterations_used) or None -- below the effective-
    positive threshold, with too few gold clusters to resample meaningfully,
    or if every single resampled iteration somehow failed to produce an
    estimate (an empty resampled gold or corpus draw).
    """
    corpus_rows = restrict_to_base(corpus_rows, base, "corpus", column)
    gold_rows = restrict_to_base(gold_rows, base, "gold rows", column)
    if base != "all":  # see corrected_ci_bootstrap: recount on the restricted gold rows
        effective_positive_count = min(
            effective_positive_count,
            weighted_effective_positive_count(gold_rows, category, gold_label_column))
    if effective_positive_count < min_genuine_positives:
        return None
    if details is not None:
        details["effective_positives"] = effective_positive_count
        details["low_evidence"] = low_evidence(effective_positive_count)

    clusters = _clusters(corpus_rows, cluster_by)
    cluster_ids = list(clusters)
    gold_clusters = _clusters(gold_rows, cluster_by)
    gold_cluster_ids = list(gold_clusters)
    if len(gold_cluster_ids) < min_gold_clusters:
        return None

    point = ppi_point_estimate(corpus_rows, column, gold_rows, category, gold_label_column)
    if point is None:
        return None

    rng = random.Random(seed)
    resampled = []
    below_0 = above_1 = 0
    for _ in range(n_boot):
        drawn_ids = _resample_clusters(cluster_ids, rng) if len(cluster_ids) > 1 else cluster_ids
        drawn_rows = [row for cid in drawn_ids for row in clusters[cid]]

        gold_drawn_ids = (_resample_clusters(gold_cluster_ids, rng)
                          if len(gold_cluster_ids) > 1 else gold_cluster_ids)
        resampled_gold = [row for cid in gold_drawn_ids for row in gold_clusters[cid]]

        estimate = _ppi_unclamped(drawn_rows, column, resampled_gold, category, gold_label_column)
        if estimate is not None:
            below_0 += estimate < 0.0
            above_1 += estimate > 1.0
            resampled.append(_clamp01(estimate))

    if not resampled:
        return None
    resampled.sort()
    n = len(resampled)
    lo = resampled[max(int((alpha / 2) * n), 0)]
    hi = resampled[min(int((1 - alpha / 2) * n), n - 1)]
    if details is not None:
        details.update(draws=n, below_0=below_0, above_1=above_1)
    return point, lo, hi, len(resampled)


def effective_clusters(rows, cluster_by="source"):
    """How many equally-sized clusters (videos) this many rows is worth: 1 / sum of
    squared shares. 30 videos where one holds half the comments is worth about 3.9,
    and an interval resting on that is much shakier than "30 videos" suggests."""
    sizes = [len(g) for g in _clusters(rows, cluster_by).values()]
    total = sum(sizes)
    if not total:
        return 0.0
    return 1.0 / sum((n / total) ** 2 for n in sizes)


def corrected_by_story(rows, column, gold_rows, gold_label_column, category,
                       cluster_by="source", min_genuine_positives=MIN_EFFECTIVE_POSITIVES, n_boot=2000, seed=42,
                       max_discard_fraction=0.05, min_gold_clusters=2, story_column="story",
                       base="all"):
    """Rogan-Gladen corrected prevalence with a 95% interval, FOR EACH STORY.

    Each story's corpus rows and gold rows are corrected on their own, with the same
    corpus-plus-gold video bootstrap and the same refusal gates as the pooled figure.
    Most story-level cells will be refused with a thin gold set; the reason is
    returned, because "no number, and why" is the result, not an error.

    Returns a list of dicts, one per story: story, n_comments, videos,
    effective_videos, gold_rows, raw_positives, effective_positives, point, lo, hi,
    and `refused` (None, or the reason).
    """
    stories = sorted({(r.get(story_column) or "").strip() or "(blank)" for r in rows})
    rows = restrict_to_base(rows, base, "corpus", column)
    gold_rows = restrict_to_base(gold_rows, base, "gold rows", column)
    out = []
    for story in stories:
        s_rows = [r for r in rows if ((r.get(story_column) or "").strip() or "(blank)") == story]
        s_gold = [r for r in gold_rows
                  if ((r.get(story_column) or "").strip() or "(blank)") == story]
        raw = sum(1 for r in s_gold
                  if category in (r.get(gold_label_column) or "").lower())
        effective = weighted_effective_positive_count(s_gold, category, gold_label_column)
        gold_clusters = len(_clusters(s_gold, cluster_by)) if s_gold else 0
        entry = {"story": story, "n_comments": len(s_rows),
                 "videos": len(_clusters(s_rows, cluster_by)),
                 "effective_videos": effective_clusters(s_rows, cluster_by),
                 "gold_rows": len(s_gold), "raw_positives": raw,
                 "effective_positives": effective, "point": None, "lo": None, "hi": None,
                 "refused": None,
                 "low_evidence": effective < WELL_SUPPORTED_POSITIVES}
        if not s_rows:
            entry["refused"] = "no tagged comments in this story"
        elif not s_gold:
            entry["refused"] = "no gold rows for this story"
        elif effective < min_genuine_positives:
            entry["refused"] = (f"only {effective:.1f} effective positives in this story's gold "
                                f"rows ({raw} raw), needs {min_genuine_positives}")
        elif gold_clusters < min_gold_clusters:
            entry["refused"] = (f"gold rows come from {gold_clusters} video(s), needs "
                                f"{min_gold_clusters}")
        else:
            result = corrected_ci_bootstrap(
                s_rows, column, s_gold, gold_label_column, category, effective,
                cluster_by=cluster_by, min_genuine_positives=min_genuine_positives,
                n_boot=n_boot, seed=seed, max_discard_fraction=max_discard_fraction,
                min_gold_clusters=min_gold_clusters)
            if result is None:
                entry["refused"] = ("the correction was degenerate (TPR and FPR too close, or "
                                    "too many bootstrap draws unusable)")
            else:
                entry["point"], entry["lo"], entry["hi"] = result[0], result[1], result[2]
        out.append(entry)
    return out


def _clamp01(value):
    return min(max(value, 0.0), 1.0)


def story_accuracy_ratios(gold_rows, categories, story_column="story",
                          gold_label_column="hand_domain", base="all"):
    """Each story's overall labelling accuracy, relative to the grand mean.

    "Overall" means every (comment, category) pair pooled across all four
    categories for that story -- not accuracy on one category specifically,
    which is exactly the cell too thin to measure reliably on its own (a
    thin story's rare category might have single-digit genuine positives).
    Pooling across categories first, then taking one ratio per story, is the
    cheap half of an additive main-effects model: it assumes a story's effect
    on accuracy is roughly constant across categories (no story x category
    interaction), which is a real, disclosed assumption, not a proven fact --
    see `corrected_estimate_by_story`'s docstring for what it does not catch.

    Returns {story: ratio}. A ratio below 1 means the classifier is less
    accurate on that story than the corpus-wide average; at 1, no adjustment.
    A story with no gold rows gets no entry (nothing to compute a ratio
    from), and callers should treat that as "unadjusted", not "ratio 1.0
    measured" -- the two look the same numerically but mean different things.
    """
    gold_rows = restrict_to_base(gold_rows, base, "gold rows")
    by_story = collections.defaultdict(list)
    for row in gold_rows:
        by_story[(row.get(story_column) or "").strip() or "(blank)"].append(row)

    def accuracy(rows):
        """Weighted share of (row, category) cells the classifier got right.

        Two bugs lived here. The gold column was hard-coded to "hand_domain"
        while the caller accepts and honours `gold_label_column` -- so under
        any other column name every `gold` lookup returned False, "accuracy"
        silently became "share of machine guesses that are 0", and the ratios
        collapsed toward 1.0 with no error raised.

        And the rows were counted unweighted, one function over from
        `_gold_tpr_fpr`, whose whole docstring is about why that is wrong:
        under a stratified draw this measured the accuracy of the *sample*
        rather than of the corpus.
        """
        correct = total = 0.0
        for row in rows:
            if not _scoreable_gold(row, gold_label_column):
                continue
            weight = validate_mod.gold_weight(row)
            for category in categories:
                gold = category in (row.get(gold_label_column) or "").lower()
                guess = row.get(f"domain_{category}") == "1"
                correct += weight if gold == guess else 0.0
                total += weight
        return correct / total if total else float("nan")

    grand_mean = accuracy(gold_rows)
    if grand_mean != grand_mean or grand_mean == 0:
        return {}
    ratios = {}
    for story, rows in by_story.items():
        story_accuracy = accuracy(rows)
        if story_accuracy == story_accuracy:  # not NaN
            ratios[story] = story_accuracy / grand_mean
    return ratios


def corrected_estimate_by_story(corpus_rows, column, gold_rows, gold_label_column,
                                category, categories, story_column="story",
                                min_gold_per_story=20,
                                min_gold_positives_per_story=15, base="all"):
    """Per-story corrected point estimate: one pooled category-level TPR/FPR
    (as `corrected_estimate` already uses), nudged per story by that story's
    own overall-accuracy ratio -- the additive compromise between "one
    correction factor for the whole corpus" (blind to story) and "a separate
    TPR/FPR per story per category" (needs far more gold data than a ~250-
    comment set can support in every cell).

    What this does NOT catch: a genuine interaction, where a story is
    uniquely weak on ONE specific category rather than uniformly weaker or
    stronger across all of them -- e.g. a story fine on democracy/economics/
    social but specifically bad at environment, because environment is rare
    enough within that story that the classifier has few good local examples
    to draw on. An additive ratio spreads that story's overall accuracy
    evenly across its categories, so a narrow, category-specific weak spot
    would be under-corrected, not caught. Treat this as a real improvement
    over a single pooled correction, not a substitute for the eyeball check
    (comparing a story's accuracy on its rare categories against its common
    ones) recommended alongside it.

    Returns {story: corrected_value_or_None}. A story is None when it has
    fewer than `min_gold_per_story` gold rows -- too little basis to trust
    even the ratio adjustment for that story -- reported as "too rare in the
    gold set to correct reliably", the same phrase used elsewhere for this
    situation, rather than silently falling back to the unadjusted pooled
    figure.
    """
    # Kept for the diagnostic it honestly is -- "is this story harder than the
    # corpus average?" -- and no longer used to adjust the correction itself.
    corpus_rows = restrict_to_base(corpus_rows, base, "corpus", column)
    gold_rows = restrict_to_base(gold_rows, base, "gold rows", column)
    ratios = story_accuracy_ratios(gold_rows, categories, story_column,
                                   gold_label_column)

    gold_by_story = collections.defaultdict(list)
    for row in gold_rows:
        gold_by_story[(row.get(story_column) or "").strip() or "(blank)"].append(row)
    gold_counts = collections.Counter({k: len(v) for k, v in gold_by_story.items()})

    by_story_corpus = collections.defaultdict(list)
    for row in corpus_rows:
        by_story_corpus[(row.get(story_column) or "").strip() or "(blank)"].append(row)

    results = {}
    for story, rows in by_story_corpus.items():
        if gold_counts.get(story, 0) < min_gold_per_story:
            results[story] = None
            continue
        # Scaling BOTH rates by one accuracy ratio was the original approach,
        # and it is not recoverable by flipping a sign. It gives
        #
        #     (observed - r*FPR) / (r*(TPR - FPR))
        #
        # so a story with below-average accuracy (r < 1) has its prevalence
        # revised UP and its assumed false-positive rate revised DOWN --
        # backwards on both counts. Measured, pooled TPR 0.85 / FPR 0.10 at an
        # observed 0.50: r=1.0 gives 0.533, r=0.8 gives 0.700. The worse the
        # story, the higher the answer.
        #
        # A single scalar cannot fix this, because it carries no information
        # about WHICH way a story fails. A story that misses true positives
        # needs a lower TPR; a story that fires on true negatives needs a
        # higher FPR; both score the same "accuracy". A story with perfect
        # recall and 40 false positives in 60 negatives is genuinely
        # uncorrectable -- its Rogan-Gladen numerator is negative -- and the
        # ratio version reported a confident 0.70 for it.
        #
        # So: measure the story's own TPR/FPR when its gold cell can carry
        # them, and refuse when it cannot.
        story_gold = gold_by_story.get(story, [])
        positives = sum(1 for row in story_gold
                        if category in (row.get(gold_label_column) or "").lower())
        if positives < min_gold_positives_per_story:
            results[story] = None
            continue
        story_tpr, story_fpr = _gold_tpr_fpr(story_gold, category, gold_label_column)
        story_observed = _cluster_rate(rows, column)
        results[story] = corrected_estimate(story_observed, story_tpr, story_fpr)
    return results


def print_table(rows_dict, headers, out=None):
    out = out or sys.stdout
    print("| " + " | ".join(headers) + " |", file=out)
    print("|" + "---|" * len(headers), file=out)
    for key in sorted(rows_dict, key=str):
        print(f"| {key} | " + " | ".join(str(v) for v in rows_dict[key]) + " |", file=out)


def _run_command(args, rows, base):
    """One command, on one base. Prints exactly what it printed before when base is 'all'."""
    if args.command == "observed":
        results = column_ci(rows, args.column, args.by, args.cluster_by, args.n_boot, args.seed,
                            base=base)
        by_group = collections.defaultdict(list)
        for row in restrict_to_base(rows, base, "corpus", args.column):
            by_group[(row.get(args.by) or "").strip() or "(blank)"].append(row)
        table = {k: ((n_c, n_v, f"{effective_clusters(by_group.get(k, []), args.cluster_by):.1f}",
                      f"{obs:.1%}", f"{lo:.1%}–{hi:.1%}") if n_c
                     else (0, 0, "0.0", "no tagged comments", "—"))
                for k, (n_c, n_v, obs, lo, hi) in results.items()}
        print_table(table, ["group", "n_comments", "n_videos", "eff_videos", "observed", "95% CI"])
        print("eff_videos = how many equally-sized videos the group's comments are worth; "
              "far below n_videos means a few videos dominate and the interval is fragile.",
              file=sys.stderr)
    elif args.command == "aggregate":
        try:
            observed, lo, hi, per_story = aggregate_ci(rows, args.column, "story",
                                                        args.cluster_by, args.n_boot, args.seed,
                                                        base=base)
        except ValueError as problem:
            print(f"Equal-per-story aggregate: not reported -- {problem}")
            return
        print(f"Equal-per-story aggregate: {observed:.1%} (95% CI {lo:.1%}–{hi:.1%})")
        print("Per-story observed rate (unweighted input to the average above):", file=sys.stderr)
        for story, rate in sorted(per_story.items()):
            print(f"  {story}: {rate:.1%}", file=sys.stderr)
    elif args.command == "corrected":
        if args.tpr is None or args.fpr is None:
            sys.exit("corrected needs --tpr and --fpr (from evaluate.py's scorecard)")
        if base != "all":
            sys.exit("corrected --tpr/--fpr: those rates describe all answered comments, so they "
                     "cannot correct a figure on another base. Use corrected-bootstrap with --gold, "
                     "which restricts the gold rows to the same base.")
        results = column_ci(rows, args.column, None, args.cluster_by, args.n_boot, args.seed)
        _, _, observed, lo, hi = results["(pooled)"]
        corrected = corrected_estimate(observed, args.tpr, args.fpr)
        print(f"Observed: {observed:.1%} (95% CI {lo:.1%}–{hi:.1%})")
        if corrected is None:
            print("Corrected: too rare/unreliable to correct (TPR and FPR too close, "
                  "or the correction fell outside 0-100%)")
        else:
            print(f"Corrected (point estimate, not itself interval-estimated): {corrected:.1%} "
                  f"[UNGATED: no evidence check is possible from --tpr/--fpr alone]")
    else:
        # corrected-bootstrap / ppi-bootstrap: the interval "corrected" above
        # explicitly does not give you -- resamples the corpus AND the gold
        # set together, by video, so the reported range reflects both sources
        # of uncertainty rather than just the corpus's.
        if not args.gold:
            sys.exit(f"{args.command} needs --gold (a CSV with both domain_* and the hand "
                     "codes -- run the gold set through slens-classify first, not slens-validate "
                     "sample's blind output)")
        category = args.category
        if not category:
            if not args.column.startswith("domain_"):
                sys.exit("--category not given and --column doesn't look like domain_<category> "
                         "to derive it from -- pass --category explicitly")
            category = args.column[len("domain_"):]

        with open(args.gold, newline="", encoding="utf-8-sig") as handle:
            gold_rows = list(csv.DictReader(handle))
        if not gold_rows:
            sys.exit(f"No rows in {args.gold}.")
        validate_mod.require_gold_weights(gold_rows, args.gold)
        # A blank machine label reads as "not tagged", so a gold file whose --column is
        # blank on every row (what a join that matched nothing writes) gives TPR 0.
        if not any((row.get(args.column) or "").strip() for row in gold_rows):
            sys.exit(f"{args.gold} has no machine labels in {args.column} (the column is "
                     f"missing or blank on every row). Label the gold file itself with the "
                     f"same instrument as the corpus, or join it to a labelled file that "
                     f"contains its comments.")

        # Rows the scoring cannot use: no hand code yet (blank is not "none"), or no machine
        # label at all (a join that could not match them). Counted, never scored as misses.
        gold_rows = validate_mod.drop_blank_hand(gold_rows, args.gold_label_column)
        gold_rows = validate_mod.drop_unlabelled(gold_rows, "domain_")
        if not gold_rows:
            sys.exit(f"No scoreable gold rows left in {args.gold}.")
        # The worked examples named in the run record beside the gold file, plus --exclude-ids.
        # Resolved once, so --base both does not print the same lines twice.
        if not hasattr(args, "_example_ids"):
            args._example_ids = validate_mod.example_ids_for_gold(
                args.gold, getattr(args, "exclude_ids", None),
                getattr(args, "no_example_exclusion", False))
        example_ids = args._example_ids
        gold_rows = validate_mod.drop_ids(gold_rows, example_ids, "gold rows")
        if getattr(args, "rescale_unclassified", None):
            print("WEIGHTS SENSITIVITY (not the headline): the unclassified strata's gold weights "
                  "are scaled to the corpus stratum sizes.", file=sys.stderr)
            gold_rows = rescale_unclassified_weights(gold_rows, args.rescale_unclassified)
        gold_rows = restrict_to_base(gold_rows, base, "gold rows", args.column)
        if not gold_rows:
            sys.exit(f"No gold rows left in the {base!r} base.")
        raw_positive_count = sum(
            1 for row in gold_rows
            if category in (row.get(args.gold_label_column) or "").lower())
        effective_positive_count = weighted_effective_positive_count(
            gold_rows, category, args.gold_label_column)
        print(f"{raw_positive_count} raw genuine positive(s) for {category!r} in {args.gold}, "
              f"worth {effective_positive_count:.1f} effective (Horvitz-Thompson-weighted) "
              f"one(s) -- needs at least {args.min_genuine_positives} effective", file=sys.stderr)
        print(error_row_summary(gold_rows, category, args.gold_label_column), file=sys.stderr)
        if args.min_genuine_positives <= effective_positive_count < WELL_SUPPORTED_POSITIVES:
            print(f"LOW EVIDENCE: {effective_positive_count:.1f} effective positives is below "
                  f"{WELL_SUPPORTED_POSITIVES}. Any corrected figure and interval below rest on "
                  f"thin evidence: report them with that caveat, and expect the interval to be "
                  f"wide.", file=sys.stderr)

        if args.command == "corrected-by-story":
            table = corrected_by_story(
                rows, args.column, gold_rows, args.gold_label_column, category,
                cluster_by=args.cluster_by, min_genuine_positives=args.min_genuine_positives,
                n_boot=args.n_boot, seed=args.seed,
                max_discard_fraction=args.max_discard_fraction,
                min_gold_clusters=args.min_gold_clusters, base=base)
            print(f"\nCorrected prevalence of {category!r}, by story (Rogan-Gladen, corpus and "
                  f"gold resampled together by video):")
            print(f"{'story':<12}{'comments':>9}{'videos':>8}{'eff.vid':>9}{'gold':>6}"
                  f"{'eff.pos':>9}   corrected (95% CI)")
            for e in table:
                right = (f"{e['point']:.1%} ({e['lo']:.1%}-{e['hi']:.1%})"
                         + ("  [LOW EVIDENCE]" if e.get("low_evidence") else "")
                         if e["refused"] is None else f"no estimate: {e['refused']}")
                print(f"{e['story']:<12}{e['n_comments']:>9}{e['videos']:>8}"
                      f"{e['effective_videos']:>9.1f}{e['gold_rows']:>6}"
                      f"{e['effective_positives']:>9.1f}   {right}")
            print("\n`eff.vid` is how many equally-sized videos the story's comments are worth; "
                  "a small number means the interval is fragile whatever the video count. A "
                  "story with no estimate should be reported as observed and uncorrected.",
                  file=sys.stderr)
            return
        details = {}
        if args.command == "corrected-bootstrap":
            result = corrected_ci_bootstrap(
                rows, args.column, gold_rows, args.gold_label_column, category,
                effective_positive_count, cluster_by=args.cluster_by,
                min_genuine_positives=args.min_genuine_positives, n_boot=args.n_boot,
                seed=args.seed, max_discard_fraction=args.max_discard_fraction,
                min_gold_clusters=args.min_gold_clusters, base=base, details=details)

            if result is None:
                print("Corrected-bootstrap (Rogan-Gladen): refused -- too few effective "
                      "positives in --gold, too few distinct clusters in --gold, the observed "
                      "TPR/FPR were degenerate, or too many bootstrap iterations went "
                      "degenerate to trust the survivors. Not a bug: this is the guard "
                      "working, not silently reporting a number nobody can stand behind.")
            else:
                point, lo, hi, used, discarded = result
                tag = evidence_tag(details.get("effective_positives", effective_positive_count))
                print(f"Corrected (Rogan-Gladen point estimate): {point:.1%}{tag}")
                print(f"Corrected 95% CI (clustered, corpus + gold resampled together): "
                      f"{lo:.1%}–{hi:.1%}{tag}")
                print(f"({used} iteration(s) used, {discarded} discarded as degenerate)",
                      file=sys.stderr)
        else:
            result = ppi_ci_bootstrap(
                rows, args.column, gold_rows, args.gold_label_column, category,
                effective_positive_count, cluster_by=args.cluster_by,
                min_genuine_positives=args.min_genuine_positives, n_boot=args.n_boot,
                seed=args.seed, min_gold_clusters=args.min_gold_clusters, base=base,
                details=details)

            if result is None:
                print("PPI-bootstrap: refused -- too few effective positives in --gold, or too "
                      "few distinct clusters in --gold to resample meaningfully.")
            else:
                point, lo, hi, used = result
                tag = evidence_tag(details.get("effective_positives", effective_positive_count))
                print(f"Corrected (prediction-powered point estimate): {point:.1%}{tag}")
                print(f"Corrected 95% CI (clustered, corpus + gold resampled together): "
                      f"{lo:.1%}–{hi:.1%}{tag}")
                # PPI clamps to [0, 1]; a bound sitting at 0 or 1 because of that is not a
                # measured limit, so say how many draws were clamped, next to the interval.
                draws = details.get("draws", used)
                if lo == 0.0 and details.get("below_0"):
                    print(f"  lower bound truncated at 0 ({details['below_0']} of {draws} "
                          f"draws below 0)")
                if hi == 1.0 and details.get("above_1"):
                    print(f"  upper bound truncated at 1 ({details['above_1']} of {draws} "
                          f"draws above 1)")
                print(f"({used} iteration(s) used)", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["observed", "aggregate", "corrected",
                                            "corrected-bootstrap", "ppi-bootstrap",
                                            "corrected-by-story"])
    parser.add_argument("--strict-corpus", action="store_true",
                        help="Refuse to run if the corpus no longer matches its manifest. "
                             "These are the figures that reach the paper, so this is the "
                             "command most worth setting it on.")
    parser.add_argument("input_csv")
    parser.add_argument("--column", required=True,
                        help="Binary/numeric column, e.g. domain_democracy or sentiment_value")
    parser.add_argument("--by", default="story", help="Group-by column for 'observed'")
    parser.add_argument("--cluster-by", default="source",
                        help="Resampling unit column (default: source, i.e. video)")
    parser.add_argument("--tpr", type=float, help="corrected: recall from the gold set")
    parser.add_argument("--fpr", type=float, help="corrected: false positive rate from the gold set")
    parser.add_argument("--base", choices=["all", "classified", "both"], default="all",
                        help="Which denominator the share is over. 'all' (default, as before): "
                             "every answered comment, including those the classifier tagged with "
                             "no harm. 'classified': only comments tagged with at least one harm. "
                             "'both': print the two side by side. The gold rows are restricted to "
                             "the same base for any corrected figure.")
    parser.add_argument("--rescale-unclassified", metavar="FROZEN_CORPUS",
                        help="corrected-bootstrap/ppi-bootstrap/corrected-by-story: a SENSITIVITY "
                             "run. Scale the gold weights of each story's unclassified strata so "
                             "they sum to that stratum's size in FROZEN_CORPUS (a corpus with "
                             "story and primary_category, e.g. the frozen master). Run it beside "
                             "the same command without the flag and report both; it is not a "
                             "correction and does not explain why the stored weights are short.")
    parser.add_argument("--exclude-ids", metavar="FILE",
                        help="corrected-bootstrap/ppi-bootstrap/corrected-by-story: leave these gold "
                             "comments out (a CSV with a comment_id column; the --fixed-examples "
                             "file works). Comments shown to the model as worked examples must "
                             "not feed the correction: each would be scored against its own "
                             "answer.")
    parser.add_argument("--no-example-exclusion", action="store_true",
                        help="corrected-bootstrap/ppi-bootstrap/corrected-by-story: silence the "
                             "warning printed when no worked-example ids are known (no run "
                             "record beside --gold, no --exclude-ids). Use it only when none of "
                             "the gold comments was shown to the model as an example.")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gold",
                        help="corrected-bootstrap: a gold CSV carrying both the machine labels "
                             "(domain_*) and the hand codes -- i.e. one that has been through "
                             "slens-classify, not the blind file slens-validate sample produces")
    parser.add_argument("--gold-label-column", default="hand_domain",
                        help="corrected-bootstrap: the hand-coded column in --gold (default hand_domain)")
    parser.add_argument("--category",
                        help="corrected-bootstrap: the harm category, e.g. democracy. Defaults to "
                             "--column with a leading 'domain_' stripped, when --column is shaped "
                             "that way (domain_democracy -> democracy)")
    parser.add_argument("--min-genuine-positives", type=int, default=MIN_EFFECTIVE_POSITIVES,
                        help="corrected-bootstrap/ppi-bootstrap: refuse below this many "
                             "EFFECTIVE (Horvitz-Thompson-weighted, Kish-style) genuine "
                             "positives in --gold (default 10; figures under 50 are flagged "
                             "LOW EVIDENCE), not a raw row count -- see "
                             "weighted_effective_positive_count's docstring for why a raw count "
                             "overstates how much a stratified gold set actually supports")
    parser.add_argument("--max-discard-fraction", type=float, default=0.05,
                        help="corrected-bootstrap: refuse rather than report if more than this "
                             "share of bootstrap iterations go degenerate (default 0.05)")
    parser.add_argument("--min-gold-clusters", type=int, default=2,
                        help="corrected-bootstrap: refuse if --gold has fewer distinct "
                             "--cluster-by values than this (default 2) -- one cluster resamples "
                             "to itself and would report a zero-width interval")
    args = parser.parse_args()

    # These figures reach the paper, so say plainly whether the corpus behind
    # them is still the one that was frozen.
    ok, message = corpus_mod.verify(args.input_csv, strict=args.strict_corpus)
    print(f"{'' if ok else 'WARNING: '}{message}", file=sys.stderr)

    with open(args.input_csv, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        sys.exit(f"No rows in {args.input_csv}.")

    bases = list(BASES) if args.base == "both" else [args.base]
    for base in bases:
        if args.base != "all":
            print(f"\n=== Base: {BASE_LABELS[base]} ===")
        try:
            _run_command(args, rows, base)
        except ValueError as problem:
            sys.exit(str(problem))
    print("\nEvery interval above covers sampling error only (which comments/videos were "
          "collected), not classifier error, unless produced via corrected_ci_bootstrap.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
