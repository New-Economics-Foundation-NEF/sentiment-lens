"""Label each comment with a sentiment score and an LLM harm domain.

    # unchanged default: the local model, the five fixed examples, VADER sentiment
    python classify.py master.csv --judge ollama:gemma3:12b > master_classified.csv

    # retrieved examples from a hand-coded pool
    python classify.py master.csv --selector retrieved --pool gold_pool.csv > out.csv

    # a different judge entirely; nothing else about the run changes
    python classify.py master.csv --judge anthropic:claude-haiku-4-5 --cache cache.jsonl > out.csv

    # whichever sentiment scorer won evaluate.py sentiment, not VADER
    python classify.py master.csv --sentiment llm:anthropic:claude-haiku-4-5 --cache cache.jsonl > out.csv

Findings that motivated the first rewrite
-----------------------------------------
Re-running the pilot classifier over the same 530 Musk/Reddit comments three
times produced three different answers: runs 2 and 3 agreed on only 65% of
comments, runs 1 and 2 on 50%. A number quoted from any single run is therefore
not reproducible.

Worse, `democracy` was never once assigned across the ~4,600 labelled rows in
the pilot. `gemma3:1b` returned the word "policy" — a category that does not
exist — and `get_close_matches(..., cutoff=0)` accepted any string however
unrelated, so an off-list answer was quietly rewritten as one of the four
rather than flagged.

What this version adds
----------------------
The model, the examples and the taxonomy are now three separate modules, so an
arm of the #11 scorecard can vary one of them and hold the others fixed:

* `judges.py` — the model. A new backend is one `_sample` method; voting,
  parsing, caching and usage accounting are shared, so two arms differ only in
  the model and not in how their answers were handled.
* `examples.py` — the few-shot examples. `fixed` is the five hand-written
  anchors; `retrieved` finds the nearest hand-coded comments to the one being
  classified.
* `taxonomy.py` — the categories, the prompt and the parser, defined once.

Defaults are unchanged, so this file still does what it did before unless a
flag says otherwise. What it cannot do is tell you whether the labels are
*correct* — that needs a hand-coded gold set and `evaluate.py`.
"""
import argparse
import collections
import concurrent.futures
import csv
import os
import sys
import time

from sentiment_lens.label import combined  # noqa: F401  -- registers the combined judge/scorer
import sentiment_lens.label.examples as examples_mod
from sentiment_lens.label import instrument
from sentiment_lens.label import judges
import sentiment_lens.label.sentiment as sentiment_mod
from sentiment_lens import corpus as corpus_mod
from sentiment_lens import taxonomy

# Windows defaults to cp1252 for stdout when it is redirected to a file, which
# crashes on any comment containing an emoji or non-Latin character.
sys.stdout.reconfigure(encoding="utf-8")

CATEGORIES = taxonomy.CATEGORIES
UNCLASSIFIED = taxonomy.UNCLASSIFIED
PARSE_FAILURE = taxonomy.PARSE_FAILURE


def warn_if_pool_overlaps(input_csv, pool_csv):
    """Refuse to silently score comments that are in their own retrieval pool.

    A comment that retrieves its own hand-coded label as one of its own
    few-shot examples is being shown the answer. It scores near-perfect and
    measures nothing, and nothing in the output looks wrong — you get a
    plausible CSV with excellent metrics. Measured on a 50-comment sample: the
    in-pool comments scored F1 1.00 in every category, against 0.77 for the
    same model on comments it had not seen.

    This is fine when labelling a fresh corpus, which is the normal case — the
    pool is the gold set and the corpus is everything else. It is only a
    problem when the two overlap, so warn rather than refuse.
    """
    def ids(path):
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if "comment_id" not in (reader.fieldnames or []):
                return None
            return {(row.get("comment_id") or "").strip()
                    for row in reader} - {""}

    try:
        input_ids, pool_ids = ids(input_csv), ids(pool_csv)
    except (OSError, csv.Error):
        return
    if input_ids is None or pool_ids is None:
        # This guard is what stops a comment retrieving its own gold answer
        # (F1 1.00 vs 0.77 -- see the docstring above); silently skipping it
        # would let that failure mode through with no visible sign anything
        # is wrong, which is worse than a noisy warning about a missing column.
        missing = input_csv if input_ids is None else pool_csv
        print(f"WARNING: {missing} has no comment_id column, so the pool/input overlap "
              f"check could not run. If any comment in {pool_csv} is also being labelled "
              f"here, it will retrieve its own hand-coded answer as an example and score "
              f"near-perfectly while testing nothing -- this would not be visible in the "
              f"output. Add comment_id to both files before trusting this run's accuracy.",
              file=sys.stderr)
        return
    overlap = input_ids & pool_ids
    if overlap:
        print(f"WARNING: {len(overlap)} of the comments being labelled are also in the "
              f"retrieval pool {pool_csv}. Those comments will retrieve their own gold "
              f"label as an example, so any accuracy measured on them is meaningless. "
              f"Use evaluate.py split, and keep the held-out half out of the pool.",
              file=sys.stderr)


def write_run_manifest(path, args, judge, scorer, selector, summary, partial=None):
    """A record of exactly what produced a labelled file, written beside it.

    The labels are only reproducible together with the model, the exact prompts, the
    settings and the response cache; a run whose labels are ever quoted needs all of
    them on file. No comment text goes in it: only hashes, counts and settings.
    """
    import datetime
    import hashlib
    import json
    import platform

    def digest(text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    versions = {"python": platform.python_version()}
    for package in ("anthropic", "openai", "google-genai", "ollama"):
        try:
            from importlib import metadata
            versions[package] = metadata.version(package)
        except Exception:
            pass
    sentiment_prompts = getattr(scorer, "_prompts", {}) or {}
    cache_path = None if args.no_cache else args.cache
    cache_lines = None
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as handle:
            cache_lines = sum(1 for _ in handle)
    manifest = {
        "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "versions": versions,
        "input": {"path": args.input_csv, "sha256": corpus_mod.content_hash(args.input_csv)},
        "output": args.output,
        "domain_judge": {"spec": judge.spec, "model": getattr(judge, "model", None),
                         "votes": judge.votes, "selector": selector.describe(),
                         "reply_limit": getattr(judge, "reply_limit", None),
                         "pool": args.pool, "k": args.k, "seed": args.seed},
        "model_build": getattr(judge, "preflight_info", None),
        "sentiment_scorer": scorer.describe(),
        "instrument_switches": {"sentiment": sentiment_mod.configure(),
                                "domain_context": taxonomy.configure_domain_context(),
                                "tags": (sentiment_mod.config_tag() + combined.config_tag()
                                         + taxonomy.domain_context_tag())},
        "prompts_sha256": {
            "domain": digest(taxonomy.build_prompt(selector.select(""))),
            "domain_by_story": {story: digest(taxonomy.build_prompt(selector.select(""), story))
                                for story in sorted(sentiment_mod.STORY_CONTEXT)},
            "sentiment_by_story": {story: digest(text)
                                   for story, text in sorted(sentiment_prompts.items())},
        },
        # Built-in examples are synthetic, so their text is recorded. Examples loaded from a
        # file are real comments: only their ids, labels and a hash go in the record.
        "fixed_examples": (
            {"source": "built-in (synthetic)",
             "examples": [{"comment": t, "label": l} for t, l in taxonomy.FIXED_EXAMPLES]}
            if taxonomy.FIXED_EXAMPLES_SOURCE.get("source") != "file"
            else {k: v for k, v in taxonomy.FIXED_EXAMPLES_SOURCE.items() if k != "path"}
            | {"file": os.path.basename(taxonomy.FIXED_EXAMPLES_SOURCE["path"])}),
        "cache": {"path": cache_path, "lines": cache_lines},
        "result": summary,
    }
    if partial:
        # The same key `slens-batches check --partial` writes: `slens-export-dashboard --corpus`
        # refuses a record that carries it, so a run stopped early is never exported as the
        # whole labelled corpus.
        manifest["partial"] = partial
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    print(f"Run record written to {path}", file=sys.stderr)


def refuse_empty_cache(cache_path, keep=False):
    """A cache holding empty replies would replay a truncated or refused answer for free
    (#82). Refuse, unless told they are known and harmless."""
    if not cache_path:
        return
    stored_empty = judges._Cache.shared(cache_path).empty_replies()
    if stored_empty and not keep:
        raise SystemExit(
            f"{cache_path} holds {stored_empty} empty cached repl"
            f"{'y' if stored_empty == 1 else 'ies'}: the mark of a truncated or refused "
            f"answer stored by an older version, which a rerun would replay as a free "
            f"failure. Remove those lines (they contain \"reply\": \"\"), or pass "
            f"--keep-empty-cached-replies to proceed anyway.")


EXTRA_COLUMNS = ([f"domain_{name}" for name in CATEGORIES]
                 + [f"domain_{UNCLASSIFIED}", f"domain_{PARSE_FAILURE}"]
                 + ["domain", "domain_unstable", "domain_votes", "domain_judge",
                    "domain_raw_votes",
                    "sentiment", "sentiment_value", "sentiment_scorer",
                    "sentiment_parse_failure", "sentiment_no_consensus",
                    "sentiment_unstable", "sentiment_votes", "sentiment_raw_votes",
                    "compound", "pos", "neu", "neg"])


def annotate_row(row, labels, unstable, ballots, result, domain_judge, sentiment_scorer,
                 domain_votes, sentiment_votes):
    """Write one comment's labels into `row` as the output columns.

    Module-level so every entry point that labels comments (`slens-classify`, and
    `slens-gemma`) writes exactly the same columns: anything downstream (join, score,
    evaluate, the dashboard export) reads these names.
    """
    for name in list(CATEGORIES) + [UNCLASSIFIED, PARSE_FAILURE]:
        row[f"domain_{name}"] = int(name in labels)
    row["domain"] = ", ".join(labels)
    row["domain_unstable"] = int(unstable)
    # At votes=1 there is nothing for a vote to disagree with, so
    # domain_unstable is 0 for every row by construction, not because
    # anything was measured stable -- record how many votes actually
    # ran so a reader (and export_dashboard) can tell the difference
    # between "measured 0% unstable" and "not measurable at this vote
    # count" rather than reading the second as the first.
    row["domain_votes"] = domain_votes
    # Which arm produced this row. Without it, two CSVs labelled by
    # different judges are indistinguishable once they leave the shell.
    row["domain_judge"] = domain_judge
    # The individual per-vote label sets, before majority resolution --
    # previously computed and then discarded. Needed to measure the
    # judge's own self-consistency across votes on the same
    # statistical footing as human coder-vs-coder kappa (see
    # validate.py's cmd_selfconsistency), which can't be done from
    # `domain_unstable` alone -- that's a single yes/no flag, not
    # enough to compute an agreement statistic across N votes.
    # "|"-separated per vote, comma-separated within a vote, so a
    # blank/"none" ballot and a genuinely blank cell stay distinguishable.
    row["domain_raw_votes"] = "|".join(",".join(b) or "none" for b in ballots)

    row["sentiment"] = result["label"]
    row["sentiment_value"] = result["value"]
    row["sentiment_scorer"] = sentiment_scorer
    row["sentiment_unstable"] = int(result["unstable"])
    # A scorer that calls no model (VADER, the stub) has `votes = 0` for the speed report, but it
    # still gives one score per comment, so the column says 1, as it did before #132.
    row["sentiment_votes"] = max(1, sentiment_votes)
    row["sentiment_parse_failure"] = int(result["label"] == sentiment_mod.PARSE_FAILURE)
    row["sentiment_no_consensus"] = int(result["label"] == sentiment_mod.NO_CONSENSUS)
    # Same reasoning as domain_raw_votes above. Only a list for the
    # LLM scorer (VADER/stub's "raw" is a dict or None) -- guarded so
    # this stays blank rather than crashing for those.
    raw = result.get("raw")
    if isinstance(raw, list):
        # A vote with no readable rating is written as "parse_failure" (the numeric readers of
        # this column skip non-numeric tokens).
        row["sentiment_raw_votes"] = ",".join(
            "parse_failure" if r is None else str(r) for r in raw)
    # VADER-specific detail, kept for anything already reading these
    # columns; blank for a non-VADER scorer, which has no such split.
    if isinstance(raw, dict) and "pos" in raw:
        row["compound"] = raw["compound"]
        row["pos"] = raw["pos"]
        row["neu"] = raw["neu"]
        row["neg"] = raw["neg"]


DEFAULT_PROJECTIONS = (693, 83532)   # the gold set scored, and the batching corpus


def speed_report(latencies, calls_per_comment, concurrency, sizes, wall_seconds=None, comments=None):
    """Lines describing how fast a run is and how long other sizes would take.

    Based on the MEDIAN live call and the concurrency, not on the wall clock: a computer that
    sleeps, or a network that drops, mid-run adds hours to the clock (a 30-comment trial once
    reported 58 hours) without making the calls any slower. The wall clock is still shown, and
    flagged when it is far above what the calls account for."""
    import statistics
    if not latencies:
        return ["Speed: every reply came from the cache, so there is nothing to time."]
    median = statistics.median(latencies)
    per_comment = median * calls_per_comment / max(concurrency, 1)
    lines = [f"Speed: a typical call takes {median:.1f} s (median of {len(latencies)} live calls); "
             f"at {calls_per_comment} calls per comment and --concurrency {concurrency} that is "
             f"about {per_comment:.1f} s per comment."]
    for size in sizes:
        lines.append(f"  {size:,} comments would take about {format_duration(per_comment * size)} "
                     f"at this speed.")
    if wall_seconds and comments:
        expected = per_comment * comments
        if wall_seconds > max(4 * expected, expected + 120):
            lines.append(f"  This run's wall clock ({format_duration(wall_seconds)} for {comments} "
                         f"comments) is far above that: the computer probably slept or the network "
                         f"stalled while it ran. Ignore the wall-clock figure.")
        else:
            lines.append(f"  Wall clock for this run: {format_duration(wall_seconds)}.")
    lines.append("  (Straight scaling. Rate limits and retries can slow a large run, and replies "
                 "already in the cache are instant.)")
    return lines


def format_duration(seconds):
    """A duration for people: '45 s', '2 min 05 s', '1 h 02 min'."""
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return (f"{hours} h {minutes:02d} min" if hours else
            f"{minutes} min {secs:02d} s" if minutes else f"{secs} s")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv")
    parser.add_argument("--judge", default="ollama:gemma3:12b",
                        help=f"'<backend>:<model>'. Backends: {', '.join(judges.available())}. "
                             f"The 1b model used in the pilot was too small to hold four "
                             f"categories apart; prefer 12b or larger.")
    parser.add_argument("--selector", default="fixed",
                        help=f"Few-shot examples: {', '.join(examples_mod.available())}")
    parser.add_argument("--pool", help="Hand-coded CSV to retrieve examples from")
    parser.add_argument("--k", type=int, default=None,
                        help="Examples to retrieve (--selector retrieved), or to draw "
                             "(--selector fixed_random).")
    parser.add_argument("--seed", type=int, default=42,
                        help="--selector fixed_random only: which random N-example draw to "
                             "freeze for the whole run. Same seed always picks the same "
                             "examples from a given pool.")
    parser.add_argument("--votes", type=int, default=3,
                        help="Samples per comment; labels need a majority to be kept.")
    parser.add_argument("--fuzzy", action="store_true",
                        help="Restore the old behaviour of fuzzy-matching off-list answers.")
    parser.add_argument("--cache-only", action="store_true",
                        help="Never call the model: use only replies already in --cache, and stop "
                             "at the first comment that has none. For replaying answers collected "
                             "by slens-anthropic-batch, so a gap can never become a full-price call.")
    parser.add_argument("--keep-empty-cached-replies", action="store_true",
                        help="Proceed although the cache holds empty replies (default: refuse).")
    parser.add_argument("--cache", default="data/judge_cache.jsonl",
                        help="JSONL response cache, on by default. A long run meets transient "
                             "failures; with a cache a restart replays completed calls for "
                             "free and only pays for what is left. Without one, a crash at "
                             "hour three of a four-hour run loses every call. --no-cache "
                             "disables it.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable the response cache. Only sensible for a one-off trial "
                             "run you do not intend to repeat.")
    parser.add_argument("--label-column", default="hand_domain")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="How many comments to label at once. The wall-clock cost of an "
                             "API run is almost entirely waiting on the network, so this is "
                             "the difference between a run measured in hours and one measured "
                             "in minutes: 10,000 comments at 3 votes is ~33h sequentially and "
                             "roughly 33/N with --concurrency N. Output row order is "
                             "unchanged. Keep it at 1 for a local ollama judge, which is "
                             "compute-bound rather than network-bound and gains nothing.")
    parser.add_argument("--strict-corpus", action="store_true",
                        help="Refuse to run if the corpus no longer matches the manifest "
                             "slens-merge wrote. Worth setting for any run whose labels get "
                             "quoted: a corpus that changed after a figure was produced means "
                             "the figure refers to data that no longer exists.")
    parser.add_argument("--limit", type=int, help="Only process the first N rows (for a trial run).")
    parser.add_argument("--max-cost", type=float, default=None, metavar="USD",
                        help="Stop once this run's own spend (domain judge plus sentiment scorer, "
                             "live calls only: replies replayed from the cache cost nothing) passes "
                             "USD. No new comment is started; those already in flight finish, so "
                             "the final figure can be a little over. What is done is written, and "
                             "the run exits non-zero with a message. Re-run the same command to "
                             "continue (the cache supplies everything already answered); the cap "
                             "applies to each run separately. Default: no cap.")
    parser.add_argument("--project-to", type=int, action="append", metavar="N",
                        help="after the run, print how long N comments would take at this run's "
                             "speed (repeatable). Default: "
                             + " and ".join(f"{n:,}" for n in DEFAULT_PROJECTIONS)
                             + " comments (the gold set and the batching corpus; see "
                             "DEFAULT_PROJECTIONS).")
    parser.add_argument("--sentiment", default="vader",
                        help=f"'vader', or 'llm:<judge spec>' e.g. 'llm:anthropic:claude-haiku-4-5'. "
                             f"Whichever arm won evaluate.py sentiment. Available: "
                             f"{', '.join(sentiment_mod.available())}")
    parser.add_argument("--output", help="CSV file to write to, with the same Windows-safe "
                                          "encoding the collectors use. If omitted, writes to "
                                          "stdout — fine in Git Bash, but PowerShell's '>' "
                                          "redirection still re-encodes as UTF-16, so prefer "
                                          "--output there.")
    instrument.add_args(parser)
    args = parser.parse_args()
    if args.max_cost is not None and not args.max_cost >= 0:
        parser.error("--max-cost must be zero or more (USD).")
    # A run that asks the sentiment question of a model (an `llm:` or `combined:` scorer, or a
    # combined judge) shows the examples' ratings in its prompt, so an examples file without
    # them would silently send a different sentiment prompt from the one slens-gemma and the
    # gold-set scorecard use (both refuse it). VADER and the stub ask no model.
    instrument.apply(args, needs_ratings=(
        args.sentiment.startswith(("llm:", "combined:")) or args.judge.startswith("combined:")))

    # Caching is on unless refused: the claim that a stopped run "restarts for
    # free" is only true when there is a cache to restart from.
    cache_path = None if args.no_cache else args.cache
    if cache_path:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    refuse_empty_cache(cache_path, args.keep_empty_cached_replies)
    # Check the corpus is the one that was frozen, before spending anything
    # labelling it. Not strict by default: a corpus collected before manifests
    # existed is still good data, and refusing it would be a false gate.
    ok, message = corpus_mod.verify(args.input_csv, strict=args.strict_corpus)
    if not ok or "No manifest" in message:
        print(f"NOTE: {message}", file=sys.stderr)

    try:
        judge = judges.create(args.judge, votes=args.votes, fuzzy=args.fuzzy, cache_path=cache_path)
    except (judges.ExternalProcessorNotApproved, judges.PriceUnavailable,
            judges.SettingRefused) as error:
        sys.exit(str(error))              # a clear message before anything is sent
    examples_mod.refuse_pool_flags_for_fixed(args.selector, args.pool, args.k)
    k = 12 if args.k is None else args.k
    selector = examples_mod.create(args.selector, pool_csv=args.pool, k=k, n=k,
                                   seed=args.seed, label_column=args.label_column)
    try:
        scorer = sentiment_mod.create(args.sentiment, votes=args.votes, cache_path=cache_path,
                                      fuzzy=args.fuzzy)
    except (judges.ExternalProcessorNotApproved, judges.PriceUnavailable,
            judges.SettingRefused) as error:
        sys.exit(str(error))
    if args.cache_only:
        if not cache_path:
            sys.exit("--cache-only needs a cache (drop --no-cache).")
        judge.cache_only = True
        if getattr(scorer, "_judge", None) is not None:
            scorer._judge.cache_only = True
    if args.pool:
        warn_if_pool_overlaps(args.input_csv, args.pool)
    print(f"Judge: {judge.spec} | examples: {selector.describe()} | votes: {judge.votes} | "
          f"sentiment: {scorer.describe()}", file=sys.stderr)

    # Prompt caching has a steep, model-dependent minimum that a short trial
    # pool can fall well under with no error at all -- see CLAUDE.md §7 and
    # AnthropicJudge._cacheable_system. This is a rough offline estimate
    # (~4 chars/token, no tokenizer, no API call), not the exact count
    # `client.messages.count_tokens` would give -- but it turns "won't
    # necessarily kick in, confirm before assuming either way" into something
    # that fires automatically instead of only when someone remembers to
    # check by hand.
    # Every prompt this run will send, each checked against the minimum of the
    # model that actually receives it. The domain prompt used to be the only
    # one checked; the sentiment prompts are the ones nearest the line.
    _inner_scorer_judge = getattr(scorer, "_judge", None)
    # The longest of the per-story domain prompts is the one that must clear the minimum.
    _checks = [(judge, {"domain": max(
        (taxonomy.build_prompt(selector.select(""), story) for story in sentiment_mod.STORY_CONTEXT),
        key=len)})]
    if _inner_scorer_judge is not None and _inner_scorer_judge is not judge:
        _checks.append((_inner_scorer_judge,
                        {f"sentiment/{story}": text
                         for story, text in getattr(scorer, "_prompts", {}).items()}))
    for _model_judge, _prompts in _checks:
        for _label, _est, _min, _verdict in judges.cache_report(
                getattr(_model_judge, "model", ""), _prompts):
            if _verdict == "below":
                print(f"WARNING: {_label} prompt is ~{_est} tokens by estimate, under "
                      f"{_model_judge.model}'s {_min}-token cache minimum -- it likely will not "
                      f"cache at all (no error, just no discount and no 'Prompt cache' savings). "
                      f"Rough estimate, not a tokenizer count: confirm with "
                      f"scripts_scratch_count_prompt_tokens.py.", file=sys.stderr)
            elif _verdict == "borderline":
                print(f"NOTE: {_label} prompt is ~{_est} tokens by estimate, only just over "
                      f"{_model_judge.model}'s {_min}-token cache minimum. The estimate can be "
                      f"~15% out either way, so it may not cache. Confirm with "
                      f"scripts_scratch_count_prompt_tokens.py before a long run.",
                      file=sys.stderr)

    extra = EXTRA_COLUMNS

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

    # The number of comments this run will label, so progress can say "of N" and how long is
    # left: a long corpus run otherwise shows only a count going up. One quick pass over the
    # input (rows with no comment text are not labelled, so they are not counted).
    with open(args.input_csv, newline="", encoding="utf-8-sig") as _counting:
        total_to_label = sum(1 for _r in csv.DictReader(_counting) if (_r.get("body") or "").strip())
    if args.limit:
        total_to_label = min(total_to_label, args.limit)
    run_started = time.monotonic()

    def _hms(seconds):
        seconds = int(max(seconds, 0))
        hours, rest = divmod(seconds, 3600)
        minutes, secs = divmod(rest, 60)
        return f"{hours} h {minutes:02d} min" if hours else f"{minutes} min {secs:02d} s"

    # Under `combined:X` the judge and the scorer are backed by the SAME inner judge, so adding
    # the two would count every dollar twice (see the end-of-run summary).
    shared_backend = getattr(scorer, "_judge", None) is judge

    def running_spend():
        """This run's spend so far, from the live usage counters (cached replies are free)."""
        return (judge.cost_usd() or 0.0) + (0.0 if shared_backend else (scorer.cost_usd() or 0.0))

    capped = [False]       # set once the spend passes --max-cost; nothing new is started after

    frequencies = collections.Counter()
    processed = unstable_count = 0
    _labelling_started = None
    sentiment_labels = collections.Counter()
    blank = [0]

    try:
        with open(args.input_csv, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames + [c for c in extra if c not in reader.fieldnames]
            if "story" not in reader.fieldnames:
                # Not a failure -- a file can legitimately have no story -- but every domain and
                # sentiment prompt is then built WITHOUT the story background the corpus run (and
                # the gold-set scorecard) gives each comment, which changes both the labels and
                # the cache keys, and used to happen with nothing in the log to say so.
                print(f"WARNING: {args.input_csv} has no `story` column, so every comment is "
                      f"labelled with the story-free prompts (no story background, no story "
                      f"stance guidance). That is a different instrument from a run on the "
                      f"corpus or the gold set, whose rows carry their story: the labels are not "
                      f"comparable with theirs and share no cache entries. Add a `story` column "
                      f"({', '.join(sorted(sentiment_mod.STORY_CONTEXT))}) if that is not "
                      f"intended.", file=sys.stderr)
            writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            out.flush()

            def label_row(row):
                """Everything that touches the model for one comment.

                Returned rather than written, so the writer stays on one thread and
                output order matches input order regardless of which comment
                finishes first — a reordered CSV would be a silent diff against
                every previous run.
                """
                # Stripped for the prompt and the cache key (the row written out keeps the
                # text exactly as collected): the gold-set scoring strips, so a padded comment
                # in the corpus now has the same cache key as the same comment in the gold set.
                body = row["body"].strip()
                story = row.get("story")
                labels, unstable, ballots = judge.label(body, selector.select(body), story=story)
                return row, labels, unstable, ballots, scorer.score(body, story=story)

            def finish(row, labels, unstable, ballots, result):
                nonlocal processed, unstable_count
                frequencies.update(labels)
                unstable_count += unstable
                processed += 1

                annotate_row(row, labels, unstable, ballots, result,
                             f"{judge.spec}/{selector.describe()}", scorer.describe(),
                             judge.votes, getattr(scorer, "votes", 1))
                sentiment_labels[result["label"]] += 1

                writer.writerow(row)
                # Every row reaches the file as it is finished, so a crash or a kill leaves a
                # valid checkpoint (complete rows, in input order) rather than whatever the
                # buffer happened to hold.
                out.flush()
                if args.max_cost is not None and not capped[0] and running_spend() > args.max_cost:
                    capped[0] = True
                if processed % 100 == 0:
                    # Elapsed time and an estimate of what is left (from the pace so far, so it
                    # is rough, and optimistic while cached replies are being replayed).
                    elapsed = time.monotonic() - run_started
                    line = (f"  {processed:,} of {total_to_label:,} comments labelled "
                            f"({processed / max(total_to_label, 1):.0%}), {_hms(elapsed)} elapsed")
                    if processed >= 200 and total_to_label > processed:
                        line += f", about {_hms(elapsed / processed * (total_to_label - processed))} left"
                    if args.max_cost is not None:
                        line += f", ${running_spend():.2f} of the ${args.max_cost:.2f} cap spent"
                    print(line, file=sys.stderr)

            def _has_body(row):
                if (row.get("body") or "").strip():
                    return True
                # Counted rather than silently skipped -- see keywords.py. Without
                # this the labelled CSV is shorter than the frozen corpus and the
                # manifest, and nothing explains the gap.
                blank[0] += 1
                return False

            wanted = (row for row in reader if _has_body(row))
            if args.limit:
                import itertools
                wanted = itertools.islice(wanted, args.limit)

            def _finish_or_diagnose(row, get_result):
                """Call get_result() and finish(), but on failure say which row and
                how much survives.

                An unhandled error (a permanently-failing comment, an exhausted
                quota) still stops the run -- that is deliberate, see judges.py's
                PERMANENT list -- but it used to do so with a bare traceback that
                named neither the row nor how much of the run was already safe.
                Everything written so far is already durable (each row is flushed
                as it is written, and the response cache is append-only), so a
                restart of the same command reprocesses free cache hits up to
                this point and continues from here -- but only if the operator
                knows where "here" was. `get_result` is a thunk, not a value,
                specifically so the exception (which happens inside label_row, or
                inside Future.result() re-raising it) is caught here rather than
                escaping while the caller's arguments are still being evaluated.
                """
                try:
                    finish(*get_result())
                except judges.CacheMiss:
                    print(f"\n--cache-only: comment_id={row.get('comment_id', '?')!r} has a reply "
                          f"missing from the cache, so nothing was called. {processed} comment(s) "
                          f"were written to {args.output or 'stdout'} before this one; treat that "
                          f"file as INCOMPLETE. Collect the missing replies first (for a batch run: "
                          f"`slens-anthropic-batch status`).", file=sys.stderr)
                    raise SystemExit(2)
                except Exception:
                    print(f"\nFAILED on comment_id={row.get('comment_id', '?')!r} after "
                          f"{processed} comment(s) already written to {args.output or 'stdout'} "
                          f"(each row is flushed as it is written) -- those are safe. Re-running "
                          f"this same command will reuse the cache for everything up to here and "
                          f"pick up from this row.",
                          file=sys.stderr)
                    raise

            _labelling_started = time.monotonic()
            if args.concurrency > 1:
                # A window of submitted comments, about 4x the workers, written strictly from the
                # head so the output is in input order. The old look-ahead (`concurrency`, and a
                # new comment only once the oldest had finished) let ONE slow call idle every
                # other worker: measured, concurrency 8 with 3% of calls 20x slower ran at 39%
                # of the ideal rate. A bounded window still keeps a 30,000-row corpus from
                # having every future in memory at once.
                window = 4 * args.concurrency
                pool = concurrent.futures.ThreadPoolExecutor(args.concurrency)
                pending = collections.deque()

                def cancel_unstarted():
                    # The executor starts work in submission order, so what has not started is
                    # always the tail of the queue: cancelling it leaves a clean prefix.
                    for _row, _future in pending:
                        _future.cancel()

                def write_ready(block_at):
                    """Write finished comments from the head of the queue, in order: while the
                    queue holds at least `block_at` (waiting for the head to finish), or the
                    head has already finished."""
                    while pending and (len(pending) >= block_at or pending[0][1].done()):
                        _row, _future = pending.popleft()
                        if _future.cancelled():
                            pending.clear()          # only the cancelled tail was left
                            return
                        _finish_or_diagnose(_row, _future.result)
                        if capped[0]:
                            cancel_unstarted()

                try:
                    for row in wanted:
                        pending.append((row, pool.submit(label_row, row)))
                        write_ready(window)
                        if capped[0]:
                            break
                        if any(f.done() and not f.cancelled() and f.exception() is not None
                               for _r, f in pending):
                            # A worker failed. Start nothing more; the comments queued before
                            # it are written, then the drain below reaches it and stops the run.
                            cancel_unstarted()
                            break
                    write_ready(1)
                except BaseException:
                    # Ctrl+C, a failing comment, a cache-only miss: do not wait for the queue.
                    cancel_unstarted()
                    raise
                finally:
                    pool.shutdown(wait=True)
            else:
                for row in wanted:
                    _finish_or_diagnose(row, lambda row=row: label_row(row))
                    if capped[0]:
                        break
    finally:
        # Closed on every exit, so a crash does not leave the output half-flushed and locked.
        if args.output:
            out.close()
        else:
            out.flush()

    _labelling_seconds = (time.monotonic() - _labelling_started) if _labelling_started else 0.0
    matches = getattr(selector, "self_matches", 0)
    if matches:
        print(f"\n{matches} comment(s) found themselves in the retrieval pool and were "
              f"shown the next-ranked example instead. Expected when relabelling the "
              f"gold set; unexpected otherwise.", file=sys.stderr)
    if blank[0]:
        print(f"\n{blank[0]} row(s) had no comment text and were not labelled.",
              file=sys.stderr)
    print(f"\nLabelled {processed} comments with {judge.spec} ({judge.votes} vote(s) each)",
          file=sys.stderr)
    if processed:
        _scorer_judge = getattr(scorer, "_judge", None)
        _latencies = list(getattr(judge, "latencies", []))
        if _scorer_judge is not None and _scorer_judge is not judge:
            _latencies += list(getattr(_scorer_judge, "latencies", []))
        # Under `combined:X` one call per vote answers both questions, so a comment costs
        # `votes` calls, not `votes` for the domain plus `votes` for the sentiment.
        # A scorer that shares the judge adds no calls; one with its own model adds `votes`;
        # VADER and the stub add none.
        _calls_per_comment = judge.votes + (0 if _scorer_judge is judge
                                            else getattr(scorer, "votes", 0))
        for _line in speed_report(_latencies, _calls_per_comment,
                                  args.concurrency, args.project_to or DEFAULT_PROJECTIONS,
                                  _labelling_seconds, processed):
            print(_line, file=sys.stderr)
    _thinking = judge.thinking_setting() if hasattr(judge, "thinking_setting") else None
    if _thinking:
        print(f"Thinking setting the service accepted: {_thinking}.", file=sys.stderr)
    print("Domain frequencies:", file=sys.stderr)
    for name, count in frequencies.most_common():
        print(f"  {name}: {count} ({count / max(processed, 1):.1%} of comments)", file=sys.stderr)
    if processed:
        unclassified_rate = frequencies[UNCLASSIFIED] / processed
        parse_failure_rate = frequencies[PARSE_FAILURE] / processed
        # At --votes 1 there is only ever one ballot per comment, so
        # Judge.resolve() returns unstable=False by construction (nothing to
        # disagree with) -- unstable_count is always 0 here, not because the
        # model was checked and found stable, but because no check happened
        # at all. Printing "0.0%" the same way a genuine --votes >= 2
        # measurement would print it read as "checked, and stable" to anyone
        # scanning the log, with nothing distinguishing the two cases.
        if judge.votes >= 2:
            unstable_str = f"{unstable_count / processed:.1%}"
        else:
            unstable_str = "not measured (--votes 1)"
        print(f"Unclassified: {unclassified_rate:.1%} | "
              f"Parse failures: {parse_failure_rate:.1%} | "
              f"Unstable across votes: {unstable_str}", file=sys.stderr)
        _s_total = sum(sentiment_labels.values())
        if _s_total:
            _s_pf = sentiment_labels[sentiment_mod.PARSE_FAILURE] / _s_total
            _s_nc = sentiment_labels[sentiment_mod.NO_CONSENSUS] / _s_total
            print(f"Sentiment: parse failures {_s_pf:.1%} | no consensus across votes "
                  f"{_s_nc:.1%} | irrelevant "
                  f"{sentiment_labels[sentiment_mod.IRRELEVANT] / _s_total:.1%} | uncodeable "
                  f"{sentiment_labels[sentiment_mod.UNCODEABLE] / _s_total:.1%}",
                  file=sys.stderr)
            if _s_pf > 0.02:
                print(f"WARNING: {_s_pf:.1%} of sentiment replies could not be read as a rating "
                      f"at all. They are labelled `parse_failure` and left out of every stance "
                      f"share and mean; they are not neutral.", file=sys.stderr)
        if parse_failure_rate > 0.02:
            print(f"WARNING: {parse_failure_rate:.1%} of replies could not be read as a valid "
                  f"answer at all (empty, garbled, or off-format) -- this is a classifier "
                  f"reliability problem, not a finding about the comments. It was previously "
                  f"hidden inside the unclassified count.", file=sys.stderr)
        # Replies that stopped early (token limit, refusal, blocked) were used as
        # empty answers and NOT written to the cache, so a rerun asks again.
        _seen, _unc = set(), collections.Counter()
        _named_backends = []
        for _name, _owner in (("domain", judge), ("sentiment", getattr(scorer, "_judge", None))):
            _owner = getattr(_owner, "_inner", _owner)
            if _owner is None or id(_owner) in _seen:
                continue
            _seen.add(id(_owner))
            _named_backends.append((_name, _owner))
            for _key, _count in getattr(_owner, "usage", {}).items():
                if _key.startswith("uncacheable_"):
                    _unc[_key[len("uncacheable_"):]] += _count
        _backends = [_o for _n, _o in _named_backends]
        if _unc:
            print(f"WARNING: {sum(_unc.values())} reply(ies) stopped early and were treated as "
                  f"unreadable, not cached: "
                  f"{', '.join(f'{k} x{v}' for k, v in _unc.most_common())}. A rerun will ask "
                  f"the model again for these.", file=sys.stderr)
        # Real requests against comments x votes: a run that limped through retries (rate limits,
        # dropped connections, replies cut off at the token limit -- each retry a full-price
        # call that is never cached) should not look like one that went straight through.
        _attempts = sum(o.usage["attempts"] for o in _backends)
        _retries = sum(o.usage["retries"] for o in _backends)
        _live_calls = sum(o.usage["calls"] for o in _backends)
        _blocked = _unc.get("blocked", 0)
        if _attempts:
            print(f"Requests: {_attempts:,} attempt(s) for {_live_calls:,} live call(s), "
                  f"{_retries:,} of them retries (rate limits, network errors, truncated replies; "
                  f"every retry is billed).", file=sys.stderr)
        if _blocked:
            print(f"WARNING: {_blocked} reply(ies) were blocked by the provider's content policy "
                  f"for that one comment. They are labelled as parse failures, not cached, and "
                  f"the run went on; look at them by hand.", file=sys.stderr)
        # Under `combined:X` the judge and the scorer are backed by the SAME
        # inner judge -- which the shared-settings fix now guarantees rather
        # than leaves to chance -- so adding the two reported exactly twice the
        # real spend, and printed it as two separate line items.
        judge_cost = judge.cost_usd() or 0.0
        scorer_cost = 0.0 if shared_backend else (scorer.cost_usd() or 0.0)
        total_cost = judge_cost + scorer_cost
        if total_cost:
            print(f"Spend: ${total_cost:.2f}" + ("" if shared_backend else
                  f" (domain judge ${judge_cost:.2f}, "
                  f"sentiment scorer ${scorer_cost:.2f})") +
                  (" (one combined judge serves both halves)" if shared_backend else ""),
                  file=sys.stderr)
            # Reported so a caching-enabled run's actual effect is visible,
            # not assumed: cost_usd() already prices in cache writes/reads,
            # but says nothing on its own about whether caching did anything
            # -- a 0% hit rate and a 40%-saved run can show similar totals for
            # unrelated reasons (a longer prompt, a smaller corpus, ...).
            saved = (judge.cache_savings_usd() or 0.0) + (
                0.0 if shared_backend else (scorer.cache_savings_usd() or 0.0))
            judge_written, judge_read = judge.cache_tokens()
            scorer_written, scorer_read = (0, 0) if shared_backend else scorer.cache_tokens()
            written, read = judge_written + scorer_written, judge_read + scorer_read
            if read or written:
                print(f"Prompt cache: {written:,} token(s) written, {read:,} read back "
                      f"-- an estimated ${saved:.2f} saved vs. paying full input price for "
                      f"the {read:,} read tokens.", file=sys.stderr)
            else:
                print("Prompt cache: no reads or writes recorded -- either this judge doesn't "
                      "support caching, or every prompt was below the minimum cacheable length.",
                      file=sys.stderr)
        if unclassified_rate > 0.1 or (judge.votes >= 2 and unstable_count / processed > 0.1):
            print("WARNING: these labels are not stable enough to quote as proportions. "
                  "Check the model size and the category definitions first.", file=sys.stderr)
        print("Stability is not accuracy. Run evaluate.py against a hand-coded gold set "
              "before quoting any of these labels in the write-up.", file=sys.stderr)
        if capped[0]:
            _resume = (f"Re-run the same command to continue: every reply already in "
                       f"{cache_path} is reused for free, so it picks up where this stopped "
                       f"(the cap applies to each run's own spend; raise it or it stops again)."
                       if cache_path else
                       "This run used --no-cache, so a re-run would pay for everything again: "
                       "use a --cache file to be able to continue.")
            print(f"\nSTOPPED FOR THE COST CAP: this run's spend,${total_cost:.2f}, passed "
                  f"--max-cost ${args.max_cost:.2f} after {processed:,} of {total_to_label:,} "
                  f"comments. Nothing new was started; the comments already in flight were "
                  f"finished and everything done is in {args.output or 'stdout'}, which is "
                  f"INCOMPLETE. {_resume}", file=sys.stderr)
        if args.output:
            write_run_manifest(
                args.output + ".run.json", args, judge, scorer, selector,
                {"comments": processed, "domain_frequencies": dict(frequencies),
                 "unstable": unstable_count,
                 "sentiment_labels": dict(sentiment_labels),
                 "cost_usd": round(total_cost, 4),
                 "labelling_seconds": round(_labelling_seconds, 1),
                 "thinking_setting": (judge.thinking_setting()
                                      if hasattr(judge, "thinking_setting") else None),
                 # Every backend that made calls, once each: a sentiment scorer with its own
                 # judge (llm:...) makes as many calls as the domain judge, and its refusals and
                 # cut-off replies count too. Reading the domain judge alone halved `calls` and
                 # dropped the sentiment half's refusals from the record.
                 "calls": sum(o.usage["calls"] for o in _backends),
                 "calls_by_backend": {name: o.usage["calls"] for name, o in _named_backends},
                 "uncacheable": dict(_unc),
                 "blocked": _blocked,
                 "attempts": _attempts, "retries": _retries,
                 "attempts_by_backend": {name: o.usage["attempts"] for name, o in _named_backends},
                 "retries_by_backend": {name: o.usage["retries"] for name, o in _named_backends},
                 "max_cost_usd": args.max_cost,
                 "stopped_for_cost_cap": bool(capped[0])},
                partial=({"reason": f"stopped at --max-cost ${args.max_cost:.2f}",
                          "comments_included": processed, "comments_in_corpus": total_to_label}
                         if capped[0] else None))
    if capped[0]:
        sys.exit(3)


if __name__ == "__main__":
    main()
