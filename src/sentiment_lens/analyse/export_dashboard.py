"""Build the single JSON file the QA dashboard reads.

    slens-export-dashboard labelled.csv --output dashboard/data/results.json \
        --gold gold.csv --scored scored.csv

This is an internal QA dashboard, so the point of it is diagnostics: whether
the labels can be trusted, not what they say. The gates below exist because a
dashboard makes numbers look finished, and these particular numbers are not
finished until a gold set says they are.

Refuses to write when:

* the corpus has changed since it was frozen (its manifest no longer matches),
  because every figure would then refer to data that no longer exists;
* intervals would be comment-level rather than clustered by video -- comments
  within a video are not independent, and a comment-level bootstrap understates
  the interval by roughly the square root of the design effect;
* `--allow-unvalidated` is not passed and no hand-coded scores were supplied,
  because a proportion from a classifier nobody has checked is not a result.

`provenance` records what produced the file, so a screenshot can be traced back
to a run rather than being taken on trust.
"""
import argparse
import collections
import csv
import datetime
import json
import os
import sys

from sentiment_lens import corpus as corpus_mod
from sentiment_lens import taxonomy as taxonomy_mod
from sentiment_lens.analyse import evaluate as evaluate_mod
from sentiment_lens.analyse import report as report_mod
from sentiment_lens.analyse import validate as validate_mod
from sentiment_lens.label import sentiment as sentiment_mod
from sentiment_lens.taxonomy import CATEGORIES, CATEGORY_NAMES, FIXED_EXAMPLES

SENTIMENTS = ["negative", "neutral", "positive"]


def _read(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def composition(rows):
    """What the corpus is made of, before anything is claimed about it."""
    per_story = collections.defaultdict(lambda: {"comments": 0, "videos": set(),
                                                 "first": None, "last": None})
    for row in rows:
        s = per_story[row.get("story", "")]
        s["comments"] += 1
        if row.get("source"):
            s["videos"].add(row["source"])
        stamp = (row.get("timestamp") or "")[:10]
        if stamp:
            s["first"] = min(s["first"] or stamp, stamp)
            s["last"] = max(s["last"] or stamp, stamp)
    return [{"story": k, "comments": v["comments"], "videos": len(v["videos"]),
             "first_comment": v["first"], "last_comment": v["last"]}
            for k, v in sorted(per_story.items())]


def per_video(rows):
    """Comments kept per video, biggest first.

    The point of charting this is that it is uneven: a handful of videos
    usually supply most of the corpus, which is the concrete form of the
    clustering problem. A reader who sees one bar dominating understands the
    design effect without being told it.
    """
    counts = collections.Counter()
    story_of = {}
    for row in rows:
        key = row.get("source") or "(unknown)"
        counts[key] += 1
        story_of.setdefault(key, row.get("story", ""))
    return [{"video": k.rsplit("=", 1)[-1], "story": story_of[k], "comments": n}
            for k, n in counts.most_common()]


def proportions(rows, cluster_by="source", n_boot=2000):
    """Per-story rate and clustered interval for every harm and sentiment."""
    out = []
    for name, column in [(c, f"domain_{c}") for c in CATEGORY_NAMES]:
        if not rows or column not in rows[0]:
            continue
        for group, (n, videos, rate, lo, hi) in report_mod.column_ci(
                rows, column, group_by="story", cluster_by=cluster_by, n_boot=n_boot).items():
            out.append({"kind": "harm", "label": name, "story": group,
                        "n": n, "videos": videos, "rate": rate, "lo": lo, "hi": hi})
    # `sentiment` can now also be `irrelevant`/`uncodeable` (sentiment.py's
    # LLMScorer, once it sees real classified output -- VADER never produces
    # either, which is why this was dormant under the old scorer). Those rows
    # have no stance to be negative/neutral/positive about; counting them in
    # the denominator while never in a numerator silently understated every
    # share once real LLM sentiment output is exported. Confirmed: 10
    # negative + 10 irrelevant rows reported "negative: 50%" here, when the
    # true share among comments that could be rated at all is 100%.
    codeable = [r for r in rows if r.get("sentiment") not in sentiment_mod.NOT_A_STANCE]
    for label in SENTIMENTS:
        marked = [dict(r, _s="1" if r.get("sentiment") == label else "0") for r in codeable]
        if not marked:
            continue
        for group, (n, videos, rate, lo, hi) in report_mod.column_ci(
                marked, "_s", group_by="story", cluster_by=cluster_by, n_boot=n_boot).items():
            out.append({"kind": "sentiment", "label": label, "story": group,
                        "n": n, "videos": videos, "rate": rate, "lo": lo, "hi": hi})
    # The comments the stance shares above leave out, as their own figures and over
    # ALL comments: how many had no stance to rate (irrelevant), could not be read
    # (uncodeable), got no readable answer from the model (parse_failure) or no
    # agreed one across votes (no_consensus). Without these the three stance shares
    # look like they describe every comment.
    if any("sentiment" in r for r in rows):
        for label in sentiment_mod.NOT_A_STANCE:
            marked = [dict(r, _s="1" if r.get("sentiment") == label else "0") for r in rows]
            for group, (n, videos, rate, lo, hi) in report_mod.column_ci(
                    marked, "_s", group_by="story", cluster_by=cluster_by,
                    n_boot=n_boot).items():
                out.append({"kind": "sentiment_not_rated", "label": label, "story": group,
                            "n": n, "videos": videos, "rate": rate, "lo": lo, "hi": hi})
    return out


def proportions_classified(rows, cluster_by="source", n_boot=2000):
    """The harm shares again, over the comments the classifier tagged with at least one
    harm (dual statistics; see report.restrict_to_base). Empty when the export has no
    `domain_unclassified` column, so an old file is never silently widened.

    Sentiment is not repeated: this is the domain question's second denominator.
    """
    if not rows or "domain_unclassified" not in rows[0]:
        return []
    out = []
    for name in CATEGORY_NAMES:
        column = f"domain_{name}"
        if column not in rows[0]:
            continue
        for group, (n, videos, rate, lo, hi) in report_mod.column_ci(
                rows, column, group_by="story", cluster_by=cluster_by, n_boot=n_boot,
                base="classified").items():
            # A story with no tagged comment is kept, as n = 0 with no figures (not NaN,
            # which is not valid JSON), so it cannot vanish from the table unremarked.
            none_tagged = n == 0
            out.append({"kind": "harm", "base": "classified", "label": name, "story": group,
                        "n": n, "videos": videos,
                        "rate": None if none_tagged else rate,
                        "lo": None if none_tagged else lo,
                        "hi": None if none_tagged else hi,
                        **({"note": "no tagged comments"} if none_tagged else {})})
    return out


MIN_VIDEOS_FOR_INTERVAL = 8
"""Below this many clusters the bootstrap cannot estimate between-video
variance, and the interval it returns is noise -- often *narrower* than the
naive one, which reads as extra precision. The Diagnostics tab already said so
in prose; `mark_estimability` makes the Results tab act on it."""


def mark_estimability(props, effects):
    """Stamp each proportion with whether its interval means anything.

    The dashboard used to headline whichever harm rate was highest, with its
    95% interval, while `design_effects` separately recorded an inflation ratio
    below 1.0 for that same cell -- the signature of too few videos to measure
    clustering at all. Two tabs of one page disagreeing about the same number.
    Joining them here means the renderer cannot forget to check.
    """
    ratio = {(e["label"], e["story"]): e.get("inflation") for e in effects}
    for row in props:
        inflation = ratio.get((row["label"], row["story"]))
        too_few = (row.get("videos") or 0) < MIN_VIDEOS_FOR_INTERVAL
        # A ratio at or below 1 means clustering apparently made the interval
        # no wider, which for clustered data is not a finding but a failure.
        unmeasurable = inflation is not None and inflation <= 1.0
        row["interval_estimable"] = not (too_few or unmeasurable)
        # Two independent reasons, reported separately. A row with 20 videos
        # failing on inflation was being described as "only 20 videos".
        row["interval_reason"] = ("few_clusters" if too_few
                                  else "deflated" if unmeasurable else None)
        row["interval_note"] = (
            None if row["interval_estimable"] else
            f"{row.get('videos')} videos: too few clusters to estimate the "
            f"between-video variance, so this interval is not meaningful"
            if too_few else
            "clustering did not widen this interval, which means the "
            "between-video variance could not be measured rather than that "
            "there is none")
    return props


MIN_GOLD_FOR_VALIDATED = 200
"""The gold-set size the project design calls for. Below it the page may not
call itself validated, however the flag is set."""


NAIVE_CLUSTER = "comment_id"


def design_effects(rows, cluster_by="source", base="all"):
    """How much clustering widens each interval, as a ratio.

    The single most useful diagnostic on this dashboard: it says how much of
    the apparent precision is real. A design effect of 4 means the effective
    sample is a quarter of the comment count.
    """
    out = []
    for name in CATEGORY_NAMES:
        column = f"domain_{name}"
        if not rows or column not in rows[0]:
            continue
        # Each comment its own cluster == the naive, comments-are-independent
        # interval. Passing cluster_by=None instead collapses every row into a
        # single cluster, resamples that one cluster, and reports a zero-width
        # interval -- which reads as infinite precision.
        naive = report_mod.column_ci(rows, column, group_by="story",
                                     cluster_by="comment_id", n_boot=800, base=base)
        clustered = report_mod.column_ci(rows, column, group_by="story",
                                        cluster_by=cluster_by, n_boot=800, base=base)
        for group in clustered:
            if group not in naive:
                continue
            wn = naive[group][4] - naive[group][3]
            wc = clustered[group][4] - clustered[group][3]
            out.append({"label": name, "story": group,
                        "naive_width": wn, "clustered_width": wc,
                        "inflation": (wc / wn) if wn > 0 else None})
    return out


def _multi_label_share(rows):
    """Share of comments carrying more than one harm label.

    The dashboard's note explaining why the per-story rates do not sum to 100%
    asserted the gap was "mostly not multi-labelling" and pointed at the
    unclassified rate. On the committed corpus the harm rates summed to 55.4%
    against 43.6% classified -- so multi-labelling accounted for ~12 points, a
    fifth of all labels issued, not a rounding detail. Measure it rather than
    asserting it.
    """
    n = len(rows) or 1
    multi = 0
    for row in rows:
        hits = sum(1 for c in CATEGORY_NAMES if row.get(f"domain_{c}") == "1")
        multi += hits > 1
    return multi / n


def _classified_figures(rows):
    """What the classified base is made of: the share of comments it covers and, within it,
    the share carrying more than one harm. {} when the file has no domain_unclassified."""
    if not rows or "domain_unclassified" not in rows[0]:
        return {}
    tagged = report_mod.restrict_to_base(rows, "classified", "corpus")
    return {"classified_n": len(tagged),
            "classified_share": len(tagged) / len(rows),
            "multi_label_rate_classified": _multi_label_share(tagged) if tagged else None}


def per_story_rates(rows):
    """The unclassified and multi-label shares, per story.

    Both were exported as single corpus-wide scalars and then quoted inside a
    per-story table note, so no row on that table could be reconciled against
    the number printed under it.
    """
    by_story = collections.defaultdict(list)
    for row in rows:
        by_story[row.get("story", "")].append(row)
    return [{"story": story,
             "n": len(group),
             "unclassified_rate": sum(1 for r in group
                                      if r.get("domain_unclassified") == "1") / len(group),
             "parse_failure_rate": sum(1 for r in group
                                       if r.get("domain_parse_failure") == "1") / len(group),
             "multi_label_rate": _multi_label_share(group),
             **_classified_figures(group)}
            for story, group in sorted(by_story.items())]


def _votes_used(rows, column):
    """The vote count a run was labelled at, from classify.py's domain_votes /
    sentiment_votes columns. Missing on older exports (pre-dates those
    columns) -- treated as >1 rather than as unmeasurable, so an old file
    keeps reporting the rate it always did instead of losing it retroactively.
    """
    for row in rows:
        raw = row.get(column)
        if raw not in (None, ""):
            try:
                return int(raw)
            except ValueError:
                continue
    return 2


def unclassified_and_instability(rows):
    n = len(rows) or 1
    # At votes=1 there is nothing for a vote to disagree with, so the
    # instability rate is 0 by construction, not by measurement -- a reader
    # seeing "0% unstable" would reasonably read that as "confirmed
    # reproducible", which votes=1 cannot confirm (see classify.py). Reported
    # as null rather than 0 so the dashboard can say "not measured" instead
    # of implying a stability finding that was never actually tested.
    domain_rate = (sum(1 for r in rows if r.get("domain_unstable") == "1") / n
                   if _votes_used(rows, "domain_votes") > 1 else None)
    sentiment_rate = (sum(1 for r in rows if r.get("sentiment_unstable") == "1") / n
                      if _votes_used(rows, "sentiment_votes") > 1 else None)
    return {
        "unclassified_rate": sum(1 for r in rows if r.get("domain_unclassified") == "1") / n,
        # The share the model gave no readable answer for. Kept apart from
        # `unclassified` (a clean "none"): the coverage claim is 1 - both.
        "parse_failure_rate": sum(1 for r in rows if r.get("domain_parse_failure") == "1") / n,
        "multi_label_rate": _multi_label_share(rows),
        **_classified_figures(rows),
        "domain_unstable_rate": domain_rate,
        "sentiment_unstable_rate": sentiment_rate,
    }


def per_category_scores(scored_rows):
    """Precision, recall and F1 against hand codes -- the validation tab.

    Horvitz-Thompson weighted, for the same reason `report._gold_tpr_fpr` is:
    `slens-validate sample --stratify-by` over-samples rare categories on
    purpose and records `inclusion_prob` so the result can be read back onto
    the corpus. Counting rows unweighted makes these figures describe the
    stratified draw instead -- and they are the numbers the dashboard's
    Validation tab publishes, which every proportion on Results is meant to be
    read against. Measured on the audit corpus, `social` recall moved 0.89 ->
    0.76 once the weights were applied.

    The raw counts are reported alongside, because a weighted "tp" of 37.4 is
    not a count of anything and a reader needs the integer to sanity-check it.
    """
    if not scored_rows:
        return []
    # Rows a human has not coded, or that carry no machine label, cannot be scored: blank hand
    # reads as hand-negative and blank machine cells as misses. `main` leaves them out and
    # counts them; this keeps a direct call honest too.
    scored_rows = [r for r in scored_rows
                   if (r.get("hand_domain") or "").strip()
                   and not validate_mod.machine_unlabelled(r, "domain_")]
    if not scored_rows:
        return []
    validate_mod.require_gold_weights(scored_rows, "the scored gold file")
    out = []
    for name in CATEGORY_NAMES:
        tp = fp = fn = tn = 0.0
        raw_tp = raw_fp = raw_fn = raw_tn = 0
        for row in scored_rows:
            # A comment the classifier could not parse is one it did not answer,
            # not one it got wrong: leave it out of TPR/FPR (as report and
            # validate.cmd_score do) and report how many there were.
            if row.get("domain_parse_failure") == "1":
                continue
            weight = validate_mod.gold_weight(row)
            hand = name in (row.get("hand_domain") or "").lower()
            machine = row.get(f"domain_{name}") == "1"
            tp += weight if (hand and machine) else 0.0
            fp += weight if (machine and not hand) else 0.0
            fn += weight if (hand and not machine) else 0.0
            tn += weight if (not hand and not machine) else 0.0
            raw_tp += hand and machine
            raw_fp += machine and not hand
            raw_fn += hand and not machine
            raw_tn += not hand and not machine
        # prf lives in evaluate, not report. This was guarded with hasattr,
        # which silently produced null precision/recall for every category --
        # the exact "plausible output, no error" failure this dashboard is
        # meant to expose.
        precision, recall, f1 = evaluate_mod.prf(tp, fp, fn)
        out.append({"label": name,
                    # Integer counts for the table; the weighted versions are
                    # what precision/recall/F1 are computed from.
                    "tp": raw_tp, "fp": raw_fp, "fn": raw_fn, "tn": raw_tn,
                    "weighted_tp": round(tp, 3), "weighted_fp": round(fp, 3),
                    "weighted_fn": round(fn, 3), "weighted_tn": round(tn, 3),
                    "ht_weighted": any(r.get("inclusion_prob") for r in scored_rows),
                    "precision": precision, "recall": recall, "f1": f1,
                    "gold_positives": raw_tp + raw_fn,
                    "parse_failure_rows_excluded": sum(
                        1 for r in scored_rows if r.get("domain_parse_failure") == "1")})
    return out


def worked_examples(rows, scored_rows, per_category=3, max_chars=260):
    """A few real comments per harm, and the machine/human disagreements.

    The most conspicuous thing peer reports carry that this one did not. Demos
    prints example tweets per class with a machine-versus-human disagreement
    column; ISD, Mozilla and Pew all quote examples. A reader cannot judge
    whether "democracy" means what they think it means from a proportion alone.

    Disagreements are shown deliberately rather than only clean hits: an error
    analysis is what turns a precision figure into something a reader can
    picture, and hiding the misses is how a codebook stops being auditable.
    """
    hand = {r.get("comment_id"): r for r in (scored_rows or [])}
    agree, disagree = collections.defaultdict(list), []
    for row in rows:
        body = " ".join((row.get("body") or "").split())
        if not body:
            continue
        snippet = body if len(body) <= max_chars else body[:max_chars - 1] + "\u2026"
        machine = {c for c in CATEGORY_NAMES if row.get(f"domain_{c}") == "1"}
        gold_row = hand.get(row.get("comment_id"))
        if gold_row:
            gold = {c for c in CATEGORY_NAMES
                    if c in (gold_row.get("hand_domain") or "").lower()}
            if gold != machine:
                disagree.append({"comment": snippet, "story": row.get("story", ""),
                                 "machine": sorted(machine) or ["none"],
                                 "human": sorted(gold) or ["none"]})
                continue
        for c in machine:
            if len(agree[c]) < per_category:
                agree[c].append({"comment": snippet, "story": row.get("story", ""),
                                 "checked_by_hand": bool(gold_row)})
    return {"by_category": {c: agree.get(c, []) for c in CATEGORY_NAMES},
            "disagreements": disagree[:8]}


def _clean(value):
    """JSON has no NaN. Anything unrepresentable becomes null, so a reader sees
    a gap rather than a number that survived a failed computation."""
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _record_field(record, *path):
    """A nested field of a run record, or None when any step is absent."""
    for key in path:
        if not isinstance(record, dict):
            return None
        record = record.get(key)
    return record


def instrument_differences(corpus_record, gold_record):
    """Compare the instrument that labelled the corpus with the one that labelled the gold set.

    Returns (differences, not_comparable), each a list of readable lines. The gold figures
    (TPR, FPR, precision, recall) describe the instrument that labelled the gold file; beside
    corpus proportions made by a different judge, scorer, prompt, example set or switch they
    describe a classifier that never labelled the corpus. Compared: the domain judge spec,
    the sentiment scorer, the prompt hashes (domain, domain per story, sentiment per story),
    the fixed examples (hash, or the synthetic text) and the switch tags.

    Only a field present in BOTH records with different values is a difference. A field absent
    from either (an older record, a record from another route) is "not comparable", so the
    caller can warn without refusing a run that may be perfectly consistent. Per-story prompt
    hashes are compared on the stories both records have, since a record only lists the
    stories its run saw.
    """
    differences, not_comparable = [], []

    def whole(label, *path, optional=False):
        a, b = _record_field(corpus_record, *path), _record_field(gold_record, *path)
        if optional and a is None and b is None:
            return                      # a field neither record carries (e.g. a digest from an API run)
        if a is None or b is None:
            not_comparable.append(label)
        elif a != b:
            differences.append(f"{label}: corpus {a!r} vs gold {b!r}")

    whole("domain judge", "domain_judge", "spec")
    whole("votes", "domain_judge", "votes")
    whole("model build (weights digest)", "model_build", "digest", optional=True)
    whole("sentiment scorer", "sentiment_scorer")
    whole("domain prompt hash", "prompts_sha256", "domain")
    whole("instrument switches (tags)", "instrument_switches", "tags")
    for label, key in (("domain prompt hash for story", "domain_by_story"),
                       ("sentiment prompt hash for story", "sentiment_by_story")):
        a = _record_field(corpus_record, "prompts_sha256", key)
        b = _record_field(gold_record, "prompts_sha256", key)
        if not a or not b:
            not_comparable.append(label.replace(" for story", "es per story"))
            continue
        for story in sorted(set(a) & set(b)):
            if a[story] != b[story]:
                differences.append(f"{label} {story!r}: corpus {str(a[story])[:12]}... vs "
                                   f"gold {str(b[story])[:12]}...")

    def examples(record):
        fixed = _record_field(record, "fixed_examples")
        if not fixed:
            return None
        return fixed.get("sha256") or json.dumps(fixed.get("examples"), sort_keys=True)

    a, b = examples(corpus_record), examples(gold_record)
    if a is None or b is None:
        not_comparable.append("fixed examples")
    elif a != b:
        differences.append("fixed examples: the two runs showed the model different worked "
                           "examples")
    return differences, not_comparable


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labelled_csv", help="A corpus that has been through slens-classify")
    parser.add_argument("--output", required=True, help="Where to write results.json")
    parser.add_argument("--exclude-ids", metavar="FILE",
                        help="Leave these gold comments out of the validation figures (a CSV "
                             "with a comment_id column, e.g. the --fixed-examples file). The "
                             "comments the run record lists as fixed examples are left out "
                             "automatically.")
    parser.add_argument("--scored", help="Hand-coded rows joined to their machine labels, "
                                         "for the validation tab")
    parser.add_argument("--exclusions", help="The exclusion table slens-collect wrote")
    parser.add_argument("--video-list", help="The video list slens-collect wrote")
    parser.add_argument("--interrater", help="JSON from a coder-agreement run, if you have one")
    parser.add_argument("--cluster-by", default="source",
                        help="Column identifying the cluster. Leave as `source` (the video): "
                             "clustering by anything finer understates every interval.")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--allow-unvalidated", action="store_true",
                        help="Write even with no hand-coded scores. The dashboard then says so "
                             "on every tab -- these are the numbers of a classifier nobody has "
                             "checked, which is not the same as a result.")
    parser.add_argument("--run-record", metavar="FILE",
                        help="The run record (.run.json) for the labelled file, when it is not "
                             "at <labelled_csv>.run.json -- e.g. after the file was renamed. "
                             "slens-keywords copies the record beside its output, so the "
                             "classify -> keywords -> export chain finds it without this.")
    parser.add_argument("--allow-instrument-mismatch", action="store_true",
                        help="Export even when the gold file's run record shows it was labelled "
                             "by a different judge, scorer, prompt, example set or switch than "
                             "the corpus. The validation figures then describe a different "
                             "classifier from the one that labelled the corpus; the mismatch "
                             "is recorded in provenance.")
    parser.add_argument("--corpus",
                        help="The frozen corpus the labels came from. Checked against the hash in "
                             "the run record beside the labelled file; the export is refused "
                             "if they differ.")
    parser.add_argument("--include-verbatim", action="store_true",
                        help="Put real comment text in results.json: the worked examples and the "
                             "machine-versus-human disagreement table. OFF by default, so a "
                             "published export quotes no one; a verbatim comment can be searched "
                             "for and traced to its author.")
    parser.add_argument("--status", default="provisional",
                        choices=["provisional", "validated"],
                        help="Stamped on the dashboard. `validated` needs --scored.")
    args = parser.parse_args()

    ok, message = corpus_mod.verify(args.labelled_csv)
    if not ok:
        raise SystemExit(f"Refusing to export.\n{message}")
    if "No manifest" in message:
        # classify.py and report.py both surface this; discarding it here meant
        # an unfrozen corpus exported silently and the dashboard showed only
        # "(not frozen)" buried in a provenance list.
        print(f"NOTE: {message}", file=sys.stderr)

    # `slens-classify` writes a run record beside its output. Its input hash is what ties
    # these labels to a frozen corpus: the labelled file itself is never frozen, so
    # the check above can only ever say "no manifest".
    run_record = None
    run_path = args.run_record or args.labelled_csv + ".run.json"
    if args.run_record and not os.path.exists(run_path):
        raise SystemExit(f"--run-record file not found: {run_path}")
    if os.path.exists(run_path):
        with open(run_path, encoding="utf-8") as handle:
            run_record = json.load(handle)
    # A progress file from `slens-batches check --merge-to --partial` names the whole corpus as
    # its input (the labels did come from it) but holds only the batches finished so far. It
    # must never pass for the whole corpus.
    partial = (run_record or {}).get("partial")
    if partial and args.corpus:
        raise SystemExit(
            f"Refusing to export: {args.labelled_csv} is a PARTIAL progress file "
            f"({partial.get('comments_included', '?')} of "
            f"{partial.get('comments_in_corpus', '?')} comments, "
            f"{partial.get('batches_done', '?')} of {partial.get('batches_total', '?')} batches), "
            f"not the labelled corpus. Finish the batches and merge without --partial.")
    if partial:
        print(f"WARNING: {args.labelled_csv} is a PARTIAL progress file "
              f"({partial.get('batches_done', '?')} of {partial.get('batches_total', '?')} "
              f"batches). Every figure describes those batches only; the export is marked "
              f"partial in its provenance.", file=sys.stderr)
    if args.corpus:
        if run_record is None:
            raise SystemExit(f"--corpus given but there is no run record at {run_path} to "
                             f"compare it with. Re-run slens-classify, or drop --corpus.")
        actual = corpus_mod.content_hash(args.corpus)
        recorded = (run_record.get("input") or {}).get("sha256")
        if actual != recorded:
            raise SystemExit(
                f"Refusing to export: {args.corpus} (sha256 {actual[:16]}...) is not the corpus "
                f"these labels were made from (run record: {str(recorded)[:16]}...). The "
                f"corpus changed after labelling, or this is the wrong file.")
        print(f"Corpus check: {args.corpus} matches the corpus the labels were made from.",
              file=sys.stderr)
    elif run_record is None:
        print("NOTE: no run record beside the labelled file, so the labels cannot be tied to "
              "a frozen corpus. Pass --corpus once slens-classify has written one.",
              file=sys.stderr)

    rows = _read(args.labelled_csv)
    if not rows:
        raise SystemExit(f"{args.labelled_csv} has no rows.")

    # The gate has to check what the column *does*, not what it is called.
    # Rejecting only "none" let `--cluster-by comment_id` through, which is a
    # per-row unique key: the bootstrap then resamples individual comments,
    # producing exactly the comment-level interval this refuses to publish --
    # and, because design_effects used the same column for its naive arm, the
    # dashboard would have reported an inflation of 1.0 everywhere, asserting
    # that clustering costs nothing precisely when it had been switched off.
    if args.cluster_by in ("", "none", "None"):
        raise SystemExit(
            "Refusing to export with clustering disabled. Comments within a video are not "
            "independent; a comment-level interval understates the width by roughly the "
            "square root of the design effect.")
    distinct = len({(r.get(args.cluster_by) or "").strip() for r in rows})
    if args.cluster_by not in rows[0]:
        raise SystemExit(f"No column {args.cluster_by!r} to cluster by. Available: "
                         f"{', '.join(sorted(k for k in rows[0] if k))}")
    if distinct > 0.9 * len(rows):
        raise SystemExit(
            f"Refusing to cluster by {args.cluster_by!r}: it has {distinct} distinct values "
            f"across {len(rows)} comments, so it identifies individual comments rather than "
            f"grouping them. Bootstrapping over it gives the comment-level interval this "
            f"command exists to avoid. Cluster by the video (`source`).")

    scored = _read(args.scored) if args.scored else []
    # The gold file's own run record (slens-classify writes one beside what it labels): the
    # instrument that produced the gold labels, and the worked examples it showed.
    gold_record = None
    instrument_mismatch = []
    if args.scored:
        gold_path = args.scored + ".run.json"
        if os.path.exists(gold_path):
            with open(gold_path, encoding="utf-8") as handle:
                gold_record = json.load(handle)
        if gold_record is None or run_record is None:
            missing = " and ".join(name for name, rec in (("the labelled file's", run_record),
                                                          (f"{args.scored}'s", gold_record))
                                   if rec is None)
            print(f"WARNING: {missing} run record missing, so nothing ties the instrument "
                  f"that labelled the gold set to the one that labelled the corpus. The "
                  f"validation figures are only valid for the corpus if both were labelled the "
                  f"same way (same judge, scorer, prompts, examples and switches).",
                  file=sys.stderr)
        else:
            instrument_mismatch, not_comparable = instrument_differences(run_record, gold_record)
            if not_comparable:
                print(f"WARNING: could not compare {', '.join(not_comparable)}: absent from "
                      f"one of the two run records. Everything that could be compared is "
                      f"reported below.", file=sys.stderr)
            if instrument_mismatch and not args.allow_instrument_mismatch:
                raise SystemExit(
                    "Refusing to export: the gold file was labelled by a different instrument "
                    "from the corpus, so its precision, recall and TPR/FPR describe a "
                    "classifier that did not label the corpus. Differences:\n  - "
                    + "\n  - ".join(instrument_mismatch)
                    + "\n\nRe-label one of them so they match, or pass "
                      "--allow-instrument-mismatch to export anyway (recorded in provenance).")
            if instrument_mismatch:
                print("WARNING: instrument mismatch between corpus and gold (allowed): "
                      + "; ".join(instrument_mismatch), file=sys.stderr)
    # Comments shown to the model as worked examples must not be scored: each would be
    # scored against its own answer. The run record names them; --exclude-ids adds more.
    record_examples = (run_record or {}).get("fixed_examples") or {}
    record_ids = set(record_examples.get("ids") or [])
    record_ids |= set(((gold_record or {}).get("fixed_examples") or {}).get("ids") or [])
    # The built-in examples are synthetic: no gold comment can be among them.
    builtin_examples = str(record_examples.get("source") or "").startswith("built-in")
    file_ids = set()
    if args.exclude_ids:
        if not os.path.exists(args.exclude_ids):
            raise SystemExit(f"--exclude-ids file not found: {args.exclude_ids}")
        file_ids = validate_mod.read_ids(args.exclude_ids) - {"comment_id"}
    example_ids = record_ids | file_ids
    excluded_examples = 0
    if scored and example_ids:
        kept = [r for r in scored if (r.get("comment_id") or "").strip() not in example_ids]
        excluded_examples = len(scored) - len(kept)
        if excluded_examples:
            print(f"  left {excluded_examples} gold comment(s) out of the validation "
                  f"figures: they were shown to the model as worked examples.", file=sys.stderr)
        scored_ids = {(r.get("comment_id") or "").strip() for r in scored}
        if file_ids and not (file_ids & scored_ids):
            print(f"WARNING: none of the {len(file_ids)} id(s) in {args.exclude_ids} matches a gold "
                  f"row, so nothing was left out. Does the file have a `comment_id` column?",
                  file=sys.stderr)
        if record_ids and not (record_ids & scored_ids):
            print(f"WARNING: none of the {len(record_ids)} worked-example id(s) in the run record "
                  f"matches a gold row, so none was left out on its account. Was the gold set "
                  f"drawn from the same comments as the examples file?", file=sys.stderr)
        scored = kept
    elif scored and not builtin_examples:
        print("WARNING: no worked-example ids are known (the run record does not list "
              "fixed examples and --exclude-ids was not given), so any gold comment that was "
              "shown to the model as an example is being scored against its own answer. "
              "Pass --exclude-ids with the fixed-examples file.", file=sys.stderr)
    if not scored and not args.allow_unvalidated:
        raise SystemExit(
            "Refusing to export without hand-coded scores (--scored). A proportion from a "
            "classifier nobody has checked is not a result, and a dashboard makes it look "
            "like one. Pass --allow-unvalidated to publish it as explicitly provisional.")
    if args.status == "validated" and not scored:
        raise SystemExit("--status validated needs --scored.")
    if scored and not any((r.get("hand_domain") or "").strip() for r in scored):
        raise SystemExit(
            f"{args.scored} has {len(scored)} rows but no hand_domain values. Every category "
            f"would score precision 0.0 -- which reads as 'the classifier got everything "
            f"wrong' rather than 'nobody has coded this yet'.")
    # The mirror of the above, and far more dangerous because the file LOOKS
    # complete: hand codes present, machine labels absent. `slens-validate
    # sample` strips domain_* on purpose so the coder cannot see the answer,
    # so the file the coders hand back never has them. Every comparison then
    # reads False and the export publishes tp=0, recall=0.0 for every category
    # -- under `--status validated`, next to a Results tab full of numbers.
    # The header alone is not enough: a join that matched no comment still writes the
    # domain_* columns, blank on every row, and those score exactly like absent ones.
    if scored and not validate_mod.has_machine_labels(scored, "domain_"):
        raise SystemExit(
            f"{args.scored} has hand codes but no machine labels (no domain_* columns, or "
            f"only blank ones), so every category would "
            f"score tp=0 and recall=0.0 by construction -- not because the classifier failed, "
            f"but because there is nothing here to compare against.\n\n"
            f"Join the hand codes back onto the labelled corpus first:\n\n"
            f"    slens-validate join {args.scored} --labelled {args.labelled_csv} "
            f"--output scored.csv\n\n"
            f"or, if the gold comments were left out of the labelled corpus (as they are for "
            f"the Gemma batches), label the gold file itself with the same instrument and pass "
            f"that output as --scored.\n")

    # Rows that cannot be scored are left out and counted: a blank hand_domain is "not coded",
    # not "none" (it read as hand-negative), and a gold row with no machine labels at all (a
    # partial join) was never labelled (it read as a miss in every category). Both move TPR,
    # FPR, precision and recall, and both used to count toward the 200-row validated floor.
    scored = validate_mod.drop_blank_hand(scored, "hand_domain", "gold rows")
    scored = validate_mod.drop_unlabelled(scored, "domain_", "gold rows")
    if args.scored and not scored:
        raise SystemExit(f"{args.scored} has no scoreable gold rows (hand-coded and carrying "
                         f"machine labels) after leaving out the unusable ones.")

    # A floor, because "--status validated" was previously honoured on any
    # non-empty gold set: the committed export claimed validation on 24
    # hand-coded comments, three of whose four categories the dashboard's own
    # threshold calls too thin to judge.
    validation_status = args.status
    if args.status == "validated" and len(scored) < MIN_GOLD_FOR_VALIDATED:
        print(f"  --status validated asked for, but the gold set is "
              f"{len(scored)} comments and the floor is {MIN_GOLD_FOR_VALIDATED}. "
              f"Publishing as provisional.", file=sys.stderr)
        validation_status = "provisional"
    if partial and validation_status == "validated":
        print("  --status validated asked for, but this is a partial progress file. "
              "Publishing as provisional.", file=sys.stderr)
        validation_status = "provisional"

    # Every video in the corpus, so the list cannot describe one story as if it
    # were all of them. Anything from --video-list (the search query that found
    # a video, which the corpus does not record) is merged on top.
    from_corpus = {}
    for row in rows:
        key = row.get("source") or ""
        if not key:
            continue
        v = from_corpus.setdefault(key, {"video_id": key.rsplit("=", 1)[-1],
                                         "story": row.get("story", ""),
                                         "comments_kept": 0, "search_query": ""})
        v["comments_kept"] += 1
    for extra in (_read(args.video_list) if args.video_list else []):
        vid = (extra.get("video_id") or "").strip()
        for v in from_corpus.values():
            if v["video_id"] == vid:
                v["search_query"] = extra.get("search_query", "")
    video_rows = sorted(from_corpus.values(), key=lambda v: -v["comments_kept"])

    # An exclusion table is written per collection run, so one file covers one
    # story. Say so rather than let it read as the whole corpus.
    exclusion_rows = _read(args.exclusions) if args.exclusions else []
    ledger_stories = {r.get("story") for r in rows}
    exclusions_partial = bool(exclusion_rows) and len(ledger_stories) > 1
    # Computed before the payload so the estimability join can see both.
    props = proportions(rows, args.cluster_by, args.n_boot)
    effects = design_effects(rows, args.cluster_by)
    props_classified = proportions_classified(rows, args.cluster_by, args.n_boot)
    # The tagged-base table gets the same "clustering did not widen this interval" check.
    props_classified = mark_estimability(
        props_classified,
        design_effects(rows, args.cluster_by, base="classified") if props_classified else [])


    payload = {
        "provenance": {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "corpus": os.path.basename(args.labelled_csv),
            "corpus_manifest": {k: v for k, v in
                                (corpus_mod.read_manifest(args.labelled_csv) or {}).items()
                                if k != "salt_fingerprint"} or None,
            "judge": rows[0].get("domain_judge", "(unknown)"),
            "sentiment_scorer": rows[0].get("sentiment_scorer", "(unknown)"),
            # One field, not two. The committed export carried
            # status="provisional" AND validated=true, so the banner said "not
            # validated against a two-coder gold set" directly above a tab
            # rendering one. `validated` is gone; `status` is derived and can
            # only say "validated" when the gold set can actually carry it.
            "status": validation_status,
            "gold_comments": len(scored),
            "worked_examples_excluded": {
                "n": excluded_examples, "example_ids_known": len(example_ids),
                "source": ", ".join(name for name, ids in (("run record", record_ids),
                                                          ("--exclude-ids", file_ids)) if ids) or None},
            "min_gold_for_validated": MIN_GOLD_FOR_VALIDATED,
            "stories": sorted({r.get("story", "") for r in rows}),
            "deterministic_judge": all(r.get("domain_unstable", "") in ("", "0")
                                       for r in rows) and
                                  rows[0].get("domain_judge", "").startswith("keyword"),
            "unclassified_rate": sum(1 for r in rows
                                     if r.get("domain_unclassified") == "1") / len(rows),
            "parse_failure_rate": sum(1 for r in rows
                                      if r.get("domain_parse_failure") == "1") / len(rows),
            "verbatim_comments_included": bool(args.include_verbatim),
            "classified_from": run_record.get("input") if run_record else None,
            "run_record_found": run_record is not None,
            "gold_run_record_found": gold_record is not None,
            "instrument_mismatch": instrument_mismatch or None,
            "partial_run": partial or None,
            "min_videos_for_interval": MIN_VIDEOS_FOR_INTERVAL,
            "cluster_by": args.cluster_by,
            "n_boot": args.n_boot,
        },
        # The definitions are the text the model actually sees, so publishing
        # them is publishing the instrument. Pew publishes its verbatim prompt;
        # a category name alone tells a reader nothing about where the boundary
        # was drawn.
        "codebook": {"categories": dict(CATEGORIES),
                     # What the domain question is shown about each story, under the switch
                     # in force (facts only, or nothing).
                     "domain_instrument": {
                         "story_context_mode": taxonomy_mod.configure_domain_context(),
                         "story_context": {story: taxonomy_mod.domain_story_context(story)
                                           for story in sentiment_mod.STORY_CONTEXT}},
                     # Real (file-sourced) examples are raw YouTube comments: their ids point
                     # straight at the comment and its author, so by default only a count and
                     # a checksum are published. The ids and labels come out only with
                     # --include-verbatim, the same switch that releases comment text.
                     "examples_shown_to_the_model": (
                         ([dict({"comment_id": cid, "label": lab},
                                **({"sentiment": rating, "story": ex_story}
                                   if run_record["fixed_examples"].get("ratings") else {}))
                           for cid, lab, rating, ex_story in zip(
                              run_record["fixed_examples"]["ids"],
                              run_record["fixed_examples"]["labels"],
                              run_record["fixed_examples"].get("ratings")
                              or [None] * len(run_record["fixed_examples"]["ids"]),
                              run_record["fixed_examples"].get("stories")
                              or [None] * len(run_record["fixed_examples"]["ids"]))]
                          if args.include_verbatim else [])
                         if (run_record or {}).get("fixed_examples", {}).get("source") == "file"
                         else [{"comment": t, "label": l} for t, l in FIXED_EXAMPLES]),
                     "example_text_withheld": (run_record or {}).get(
                         "fixed_examples", {}).get("source") == "file",
                     "examples_withheld_summary": (
                         {"n": len(run_record["fixed_examples"].get("ids") or []),
                          "sha256": run_record["fixed_examples"].get("sha256")}
                         if (run_record or {}).get("fixed_examples", {}).get("source") == "file"
                         else None),
                     # The sentiment instrument verbatim, under the switches in force
                     # when this was exported: the prompt, the scale and each story's
                     # context paragraph are all part of what the model saw.
                     "sentiment_instrument": {
                         "prompt_template": sentiment_mod.PROMPT,
                         "scale": {str(k): v for k, v in sentiment_mod.scale_for().items()},
                         "story_context": {story: sentiment_mod.story_context_for(story)
                                           for story in sentiment_mod.STORY_CONTEXT},
                         "no_story_context": sentiment_mod.NO_STORY_CONTEXT,
                         "switches": dict(sentiment_mod._CONFIG),
                     }},
        # Real comments are left out unless asked for. The empty shape is kept so the page and
        # any older reader still work; `withheld` says why it is empty.
        "examples": (worked_examples(rows, scored) if args.include_verbatim
                     else {"by_category": {c: [] for c in CATEGORY_NAMES},
                           "disagreements": [], "withheld": True}),
        "composition": composition(rows),
        "per_video": per_video(rows),
        "proportions": mark_estimability(props, effects),
        "proportions_classified": props_classified,
        "design_effects": effects,
        "rates": unclassified_and_instability(rows),
        "per_story_rates": per_story_rates(rows),
        "validation": per_category_scores(scored),
        "exclusions": exclusion_rows,
        "exclusions_partial": exclusions_partial,
        "videos": video_rows,
        "interrater": json.load(open(args.interrater)) if args.interrater else None,
    }

    interrater = payload["interrater"]
    if isinstance(interrater, dict) and (interrater.get("statistic") == "krippendorff_alpha"
                                         or (interrater.get("n_coders") or 2) > 2):
        print("WARNING: --interrater is a multi-coder (Krippendorff alpha) file, but the "
              "dashboard's coder-agreement table is written for two coders: its coder A / "
              "coder B / both columns show a dash, its verdict pills read Fleiss' kappa "
              "as if it were Cohen's, and its text still says two coders. The alpha "
              "figures are in results.json (interrater.domain, interrater.sentiment) but "
              "the page does not draw them: report them from the terminal output or the "
              "JSON.", file=sys.stderr)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(_clean(payload), handle, indent=2)
        handle.write("\n")

    print(f"Wrote {args.output}", file=sys.stderr)
    print(f"  status={validation_status}  gold={len(scored)}  "
          f"{len(rows)} comments, clustered by {args.cluster_by}", file=sys.stderr)
    if not scored:
        print("  NOTE: no hand-coded scores. The dashboard will mark every figure "
              "provisional, because that is what they are.", file=sys.stderr)


if __name__ == "__main__":
    main()
