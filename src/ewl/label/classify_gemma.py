"""Label comments for harm domain AND sentiment with one model only: Gemma 2 27B
Instruct (Q4_0), run locally through Ollama. Nothing leaves the machine.

    ewl-gemma --check                                   # is everything set up? (~2 min the first time)
    ewl-gemma corpus.csv --output t.csv --sample 60 --project-to 1000   # measure the speed (~3 min)
    ewl-gemma corpus.csv --output t.csv --sample 5000                   # the 5,000-comment trial
    ewl-gemma comments.csv --output labelled.csv        # label a whole file; safe to stop and re-run

Setup, hardware requirements and troubleshooting: CLAUDE.md section 12. Running the whole
corpus in batches of 1,000: docs/GEMMA_FULL_CORPUS_BATCHES.md and `ewl-batches`.

The instrument switches (`--fixed-examples FILE`, `--story-rules`, `--stance-object`,
`--combined-domain-context`) work exactly as they do on `ewl-classify`. Pass the same ones
for every batch: the run record notes them, and `ewl-batches check` refuses to merge batches
labelled under different instruments.

What this is, and is not
------------------------
There is deliberately no `--judge` flag. This script can only ever run the model
named in `ewl.label.gemma_local.MODEL_TAG`, and it refuses to start unless Ollama
reports that model as Q4_0. It uses the project's own prompts and parsers
(`taxonomy.build_prompt` with the 25 `fixed` examples, `sentiment.PROMPT` with the
per-story context, `taxonomy.parse_labels`, `sentiment.resolve_ratings`) and writes
the same columns as `ewl-classify`, so its output feeds `ewl-validate join`,
`ewl-report` and `ewl-export-dashboard` unchanged. It exists so a Gemma run cannot
differ from an Opus or Sonnet run in anything but the model.

Two passes, not one
-------------------
`ewl-classify` asks the domain question and then the sentiment question for each
comment in turn. That alternates between two different system prompts, and a local
model can only skip re-reading a prompt when the next request begins with the same
text as the last one. Alternating defeats that and makes every call re-read a
1,000-1,500 token prompt. This script instead labels every comment for domain first
(one prompt, identical for every comment), then every comment for sentiment, grouped
by story (four prompts, each used for a whole run of comments). The output is
written once both passes finish, in the input's original order.

Choosing the comments: --sample, not --limit
--------------------------------------------
The corpus file is ordered by story, so `--limit N` takes the first N rows, which
are all one story. `--sample N` instead draws N comments at random, spread as
evenly as possible across the stories (a story with too few passes its share to
the others), so both the timing and the labels reflect the whole corpus. The same
`--seed` always draws the same comments.

Measuring the speed
-------------------
Each real model call is timed (replies replayed from the cache are not). The first
call of each prompt is reported separately as start-up, because it includes loading
the 16 GB model and reading that prompt once; every later call is the steady-state
rate. `--project-to N` turns that into a projected time for N comments. Use a
`--sample` of at least 40-60 comments: with fewer, the steady-state figure rests on
too few calls to trust.

Stopping and resuming
---------------------
Every reply is appended to the cache file (`--cache`, default data/gemma_cache.jsonl)
the moment it is received. If the run stops for any reason, run the same command
again: everything already answered is replayed from the cache in seconds and the run
continues from there. The cache is also the reproducibility record for this arm, so
keep it with the results.

Cost: none. Time: measured and reported at the end of every run.
"""
import argparse
import collections
import csv
import os
import random
import statistics
import sys
import time

import types

from ewl import taxonomy
from ewl.label import classify as classify_mod
from ewl.label import examples as examples_mod
from ewl.label import instrument, judges, gemma_local
from ewl.label import sentiment as sentiment_mod

DEFAULT_CACHE = "data/gemma_cache.jsonl"

# Below this many steady-state calls per prompt type, a projection is a guess.
MIN_CALLS_FOR_PROJECTION = 20

# A steady-state call slower than this means the prompt is very probably being
# re-read every time (a reused prompt should cost roughly 0.5-1.5 s for this
# model on a 64 GB Apple Silicon Mac; re-reading a ~1,200-1,450 token prompt costs
# several seconds more).
SLOW_CALL_SECONDS = 3.0

# Two invented comments, one per prompt shape, for `--check`. Not corpus text.
_SMOKE_COMMENTS = [
    ("He bought the platform to silence his critics.", "musk"),
    ("Lovely video, thanks for posting.", "jets"),
]


def read_rows(path, limit=None):
    """(fieldnames, rows, blank_count) for every row with comment text, up to `limit`.

    Blank-body rows are skipped and counted by the caller, as `ewl-classify` does.
    """
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "body" not in reader.fieldnames:
            sys.exit(f"{path} has no `body` column to label.")
        fieldnames = list(reader.fieldnames)
        rows, blank = [], 0
        for row in reader:
            if not (row.get("body") or "").strip():
                blank += 1
                continue
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return fieldnames, rows, blank


def stratified_sample(rows, n, seed=42):
    """`n` rows chosen at random, spread as evenly as possible across stories.

    Each story gets an equal share; a story with fewer rows than its share gives
    the difference to the others. The rows come back in their original order, and
    the same seed always gives the same rows, and a larger `n` with the same seed
    returns a superset (see below). If `n` is at least the number of rows, every row
    is returned.
    """
    if n >= len(rows):
        return list(rows)
    groups = collections.defaultdict(list)
    for index, row in enumerate(rows):
        groups[row.get("story") or ""].append(index)
    quota = {story: 0 for story in groups}
    open_stories = sorted(groups)
    remaining = n
    while remaining and open_stories:
        share = max(remaining // len(open_stories), 1)
        for story in list(open_stories):
            take = min(share, len(groups[story]) - quota[story], remaining)
            quota[story] += take
            remaining -= take
            if quota[story] == len(groups[story]):
                open_stories.remove(story)
            if not remaining:
                break
    # One shuffled order PER STORY, each from its own seed, and the sample is the first
    # `quota` of it. The quotas never shrink as `n` grows (each story's share only goes
    # up, or stays at its size), so every sample is a prefix of the next: `--sample 1000`
    # contains every comment of `--sample 500` for the same seed. A run can then be grown
    # in batches (500, then 1,000, then 2,000) with no earlier labelling wasted, and the
    # comments are still a simple random sample of each story at every stage. (They used
    # to be drawn from one shared generator, so a bigger `n` re-drew almost everything.)
    chosen = []
    for story in sorted(groups):
        order = list(groups[story])
        random.Random(f"{seed}:{story}").shuffle(order)
        chosen += order[:quota[story]]
    return [rows[index] for index in sorted(chosen)]


def _progress(name, done, total, started):
    if done % 25 and done != total:
        return
    elapsed = time.monotonic() - started
    per = elapsed / done
    remaining = per * (total - done)
    print(f"  {name}: {done}/{total}   {per:.1f} s per comment so far, "
          f"about {remaining / 60:.0f} min left", file=sys.stderr)


def _run_pass(name, indices, rows, work, usage, group_of):
    """Run `work(row)` for each index in order, timing every REAL model call.

    Returns (results, timings). `timings` holds (group, seconds, tokens_read) for
    each comment that reached the model; a comment answered from the cache made no
    call (`usage["calls"]` did not move) and is left out, so a resumed run does not
    make the model look instant. On failure, says which comment and that nothing
    already answered is lost.
    """
    results, timings = {}, []
    started = time.monotonic()
    for done, index in enumerate(indices, 1):
        row = rows[index]
        calls_before, read_before = usage["calls"], usage["input_tokens"]
        began = time.monotonic()
        try:
            results[index] = work(row)
        except Exception:
            print(f"\nFAILED in the {name} pass on comment_id={row.get('comment_id', '?')!r} "
                  f"after {done - 1} comment(s) in this pass. Everything already answered is "
                  f"saved in the cache: fix the cause (is Ollama still running?) and run the "
                  f"same command again to carry on from here.", file=sys.stderr)
            raise
        if usage["calls"] > calls_before:
            timings.append((group_of(row), time.monotonic() - began,
                            usage["input_tokens"] - read_before))
        _progress(name, done, len(indices), started)
    return results, timings


def steady_state(timings):
    """Split timings into start-up and steady state.

    The first call of each group (the domain prompt; each story's sentiment prompt)
    pays for loading the model and reading that prompt once, so it is start-up.
    Returns a dict: `startup` (seconds, summed), `calls` (steady-state count),
    `mean`, `median` and `tokens` (mean tokens Ollama reported reading per call),
    the last three None if there are no steady-state calls yet.
    """
    seen, startup, seconds, tokens = set(), 0.0, [], []
    for group, took, read in timings:
        if group in seen:
            seconds.append(took)
            tokens.append(read)
        else:
            seen.add(group)
            startup += took
    return {
        "startup": startup,
        "calls": len(seconds),
        "mean": statistics.fmean(seconds) if seconds else None,
        "median": statistics.median(seconds) if seconds else None,
        "tokens": statistics.fmean(tokens) if tokens else None,
    }


def format_duration(seconds):
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def report_timing(domain, sentiment, project_to, domain_prompt_tokens, file=None):
    """Print the measured speed and, if asked, a projection. Returns the projected
    seconds (or None). `domain` and `sentiment` are `steady_state` dicts."""
    file = file or sys.stderr
    print("\nSpeed (real model calls only; replies replayed from the cache are not counted):",
          file=file)
    for label, part in (("domain   ", domain), ("sentiment", sentiment)):
        if part["mean"] is None:
            print(f"  {label}: too few calls to measure a steady rate yet", file=file)
            continue
        print(f"  {label}: {part['mean']:.2f} s per call (median {part['median']:.2f}, "
              f"{part['calls']} calls after the first), Ollama read ~{part['tokens']:.0f} "
              f"tokens per call", file=file)
    startup = domain["startup"] + sentiment["startup"]
    print(f"  start-up : {format_duration(startup)} for the first call of each prompt "
          f"(loading the model, and reading each prompt once)", file=file)
    print(f"  the domain prompt is ~{domain_prompt_tokens} tokens: if the 'tokens per call' "
          f"figures above are close to that, either Ollama is not reusing the prompt between "
          f"calls or this Ollama version reports the full prompt length; the seconds per call "
          f"are the figure to trust.", file=file)

    if domain["mean"] is None or sentiment["mean"] is None:
        return None
    per_comment = domain["mean"] + sentiment["mean"]
    print(f"  => {per_comment:.2f} s per comment for both questions.", file=file)
    slowest = max(domain["mean"], sentiment["mean"])
    if slowest > SLOW_CALL_SECONDS:
        print(f"  WARNING: {slowest:.1f} s per call is slow. A prompt that is being reused "
              f"takes roughly 0.5-1.5 s here; re-reading a ~1,200-1,450 token prompt on every call "
              f"takes several seconds more. Check `ollama ps` shows 100% GPU, that "
              f"OLLAMA_NUM_PARALLEL is 1, and that nothing else is using the memory.",
              file=file)
    if project_to:
        thin = min(domain["calls"], sentiment["calls"]) < MIN_CALLS_FOR_PROJECTION
        projected = startup + project_to * per_comment
        print(f"  => {project_to:,} comments would take about {format_duration(projected)} "
              f"(start-up + {project_to:,} x {per_comment:.2f} s)"
              + ("  -- TENTATIVE: fewer than "
                 f"{MIN_CALLS_FOR_PROJECTION} steady-state calls per prompt; use a larger "
                 f"--sample" if thin else "."), file=file)
        return projected
    return None


def _check_prompt_sizes(scorer, judge):
    """Say how big the prompts are against the context window, before any comment is read.

    The estimate is characters / 3, deliberately pessimistic. Gemma 2 keeps full attention
    only over a 4,096-token sliding window, and prompt reuse is least reliable beyond it, so
    a prompt near that size is flagged; one that cannot leave room for a comment is refused.
    """
    prompts = {f"domain/{story}": taxonomy.build_prompt(examples_mod.create("fixed").select(""), story)
               for story in sentiment_mod.STORY_CONTEXT}
    prompts.update({f"sentiment/{story}": text
                    for story, text in getattr(scorer, "_prompts", {}).items()})
    sizes = {name: len(text) // 3 for name, text in prompts.items()}
    biggest_name, biggest = max(sizes.items(), key=lambda item: item[1])
    print(f"Longest prompt: {biggest_name}, about {biggest:,} tokens (pessimistic estimate); "
          f"context window {judge.num_ctx:,}.", file=sys.stderr)
    if biggest > judge.num_ctx - 1500:
        sys.exit(f"The {biggest_name} prompt (about {biggest:,} tokens) leaves no room for a "
                 f"comment in the {judge.num_ctx:,}-token window. Ollama would silently drop "
                 f"the start of the prompt. Shorten the examples.")
    if biggest > 3500:
        print(f"WARNING: the {biggest_name} prompt is near Gemma 2's 4,096-token attention "
              f"window; reuse of the prompt between calls may not hold. Watch the speed "
              f"figures at the end of a --sample 60 run.", file=sys.stderr)


def run_check(judge, selector, scorer):
    """Send two invented comments through both prompts and show what came back."""
    failed = False
    print("\nSmoke test (first call includes loading the model, which can take a minute or two):",
          file=sys.stderr)
    for body, story in _SMOKE_COMMENTS:
        started = time.monotonic()
        domain_prompt = taxonomy.build_prompt(selector.select(body))
        domain_raw = judge.sample(body, domain_prompt, 0)
        domain_seconds = time.monotonic() - started
        labels = taxonomy.parse_labels(domain_raw)

        started = time.monotonic()
        sentiment_raw = scorer._judge.sample(body, scorer._prompts[story], 0)
        sentiment_seconds = time.monotonic() - started
        result = scorer.score(body, story=story)

        print(f"\n  comment ({story}): {body}")
        print(f"    domain    raw reply: {domain_raw!r}  ->  {labels}   ({domain_seconds:.1f} s)")
        print(f"    sentiment raw reply: {sentiment_raw!r}  ->  {result['label']} "
              f"(value {result['value']})   ({sentiment_seconds:.1f} s)")
        if taxonomy.PARSE_FAILURE in labels:
            failed = True
            print("    WARNING: the domain reply could not be read as labels. If it is a "
                  "sentence, the model is ignoring 'answer with the labels only'.")
    print("\nIf both replies above look sensible, the setup works.", file=sys.stderr)
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", nargs="?", help="A CSV with a `body` column (and `story`).")
    parser.add_argument("--output", help="Where to write the labelled CSV (required to label).")
    parser.add_argument("--check", action="store_true",
                        help="Check the setup and send two invented comments through the model, "
                             "then stop.")
    parser.add_argument("--sample", type=int, metavar="N",
                        help="Label N comments drawn at random, spread evenly across stories. "
                             "Use this, not --limit, on the corpus file (which is ordered by "
                             "story).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Which random draw --sample makes (default 42). Same seed, same "
                             "comments.")
    parser.add_argument("--limit", type=int,
                        help="Only label the first N comments of the file, in file order.")
    parser.add_argument("--project-to", type=int, metavar="N",
                        help="After the run, project how long N comments would take from the "
                             "measured speed.")
    parser.add_argument("--cache", default=DEFAULT_CACHE,
                        help=f"JSONL response cache (default {DEFAULT_CACHE}). It is what makes "
                             f"a stopped run resumable and this arm reproducible; keep it.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Do not cache replies. Only for a throwaway trial.")
    parser.add_argument("--keep-empty-cached-replies", action="store_true",
                        help="Proceed although the cache holds empty replies (default: refuse).")
    instrument.add_args(parser)
    parser.add_argument("--host",
                        help="Ollama server address if it is not on this machine (sets "
                             "OLLAMA_HOST), e.g. http://192.168.1.20:11434.")
    args = parser.parse_args(argv)

    if not args.check and not (args.input_csv and args.output):
        parser.error("give an input CSV and --output, or use --check")
    if args.sample and args.limit:
        parser.error("use --sample or --limit, not both")
    if args.host:
        os.environ["OLLAMA_HOST"] = args.host

    cache_path = None if (args.no_cache or args.check) else args.cache
    if cache_path and os.path.dirname(cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    classify_mod.refuse_empty_cache(cache_path, args.keep_empty_cached_replies)
    # The switches change the prompt, and the prompt is part of the cache key, so a run under
    # one setting never reuses an answer given under another. Applied before any scorer exists.
    # Refuses an examples file without ratings unless --sentiment-examples off.
    instrument.apply(args, needs_ratings=True)

    # The corpus arm is the frozen temperature-0 instrument. A temperature set in the environment
    # (EWL_GEMMA_TEMPERATURE, for a stability measurement through ewl-classify) would silently
    # change it, so this command refuses it rather than label a batch under another setting.
    _temperature = os.environ.get("EWL_GEMMA_TEMPERATURE", "").strip()
    if _temperature:
        try:
            _is_zero = float(_temperature) == 0.0
        except ValueError:
            _is_zero = False
        if not _is_zero:
            sys.exit(f"EWL_GEMMA_TEMPERATURE={_temperature} is set, but ewl-gemma is the frozen "
                     f"temperature-0 arm and does not take it. Unset it for this command (a "
                     f"stability measurement goes through ewl-classify --judge gemma, with its "
                     f"own cache file).")

    # Constructing the judge IS the readiness check (server reachable, model
    # pulled, quantisation Q4_0 -- see GemmaJudge.__init__), so a problem
    # surfaces here, before any comment is read, with the fix in the message.
    try:
        judge = judges.create("gemma", cache_path=cache_path)
        selector = examples_mod.create("fixed")
        scorer = sentiment_mod.create("llm:gemma", votes=1, cache_path=cache_path)
    except gemma_local.GemmaUnavailable as error:
        sys.exit(f"Gemma is not ready.\n\n{error}")
    info = judge.preflight_info
    print(f"Model ready: {info['model']} ({info['parameter_size']}, {info['quantisation']}), "
          f"temperature {info.get('temperature', 0):g}, {judge.votes} vote(s)", file=sys.stderr)
    if taxonomy.FIXED_EXAMPLES_SOURCE.get("source") == "file":
        has_ratings = bool(taxonomy.FIXED_EXAMPLES_SOURCE.get("ratings"))
        print(f"Examples: {len(taxonomy.FIXED_EXAMPLES_SOURCE.get('ids', []))} real gold comments "
              f"from {os.path.basename(taxonomy.FIXED_EXAMPLES_SOURCE['path'])} "
              f"(sha256 {taxonomy.FIXED_EXAMPLES_SOURCE['sha256'][:12]}...), "
              + ("with ratings and stories" if has_ratings
                 else "NO ratings: the sentiment prompt has no worked examples"),
              file=sys.stderr)
    else:
        print("WARNING: using the BUILT-IN fixed examples (5 hand-written, 20 AI-drafted "
              "placeholders), not the final 25. Anything labelled now is provisional and will "
              "have to be redone: pass --fixed-examples FILE once the final set exists.",
              file=sys.stderr)
    _check_prompt_sizes(scorer, judge)

    if args.check:
        return run_check(judge, selector, scorer)

    fieldnames, rows, blank = read_rows(args.input_csv, args.limit)
    if not rows:
        sys.exit(f"No rows with comment text in {args.input_csv}.")
    if args.sample:
        rows = stratified_sample(rows, args.sample, args.seed)
        per_story = collections.Counter(row.get("story") or "(none)" for row in rows)
        print(f"Sample of {len(rows)} comments (seed {args.seed}): "
              + ", ".join(f"{story} {count}" for story, count in sorted(per_story.items())),
              file=sys.stderr)
    print(f"Labelling {len(rows)} comment(s) with {judge.spec} | examples: "
          f"{selector.describe()} | 2 passes (domain, then sentiment)", file=sys.stderr)
    # Redoing a labelled file: the old output and its record go first. The new record is
    # written before the rename below, so a stop in between would otherwise leave it beside
    # the OLD labels, and `ewl-batches check` would vouch for them.
    for stale in (args.output, args.output + ".run.json"):
        if os.path.exists(stale):
            os.remove(stale)

    started = time.monotonic()
    everything = range(len(rows))
    # Grouped by story in BOTH passes, so each story's prompt is used for a whole run of
    # consecutive comments (the sort is stable: input order within a story). The domain
    # prompts share everything but their last paragraph, so this costs little either way.
    by_story = sorted(everything, key=lambda i: rows[i].get("story") or "")
    domain, domain_timings = _run_pass(
        "domain", by_story, rows,
        lambda row: judge.label(row["body"], selector.select(row["body"]),
                                story=row.get("story")),
        judge.usage, lambda row: "domain")
    sentiment, sentiment_timings = _run_pass(
        "sentiment", by_story, rows,
        lambda row: scorer.score(row["body"], story=row.get("story")),
        scorer._judge.usage, lambda row: row.get("story") or "")
    elapsed = time.monotonic() - started

    domain_label = f"{judge.spec}/{selector.describe()}"
    frequencies = collections.Counter()
    for index, row in enumerate(rows):
        labels, unstable, ballots = domain[index]
        frequencies.update(labels)
        classify_mod.annotate_row(row, labels, unstable, ballots, sentiment[index],
                                  domain_label, scorer.describe(), judge.votes,
                                  getattr(scorer, "votes", 1))

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    columns = fieldnames + [c for c in classify_mod.EXTRA_COLUMNS if c not in fieldnames]
    # Written under a temporary name and renamed only AFTER the run record exists, so an
    # output file never appears without its record (a stop in between leaves the batch
    # pending, and the same command resumes from the cache).
    pending_output = args.output + ".partial"
    with open(pending_output, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    print(f"\nLabelled {total} comment(s) with {judge.spec} -> {args.output}", file=sys.stderr)
    if blank:
        print(f"{blank} row(s) had no comment text and were not labelled.", file=sys.stderr)
    print("Domain frequencies:", file=sys.stderr)
    for name, count in frequencies.most_common():
        print(f"  {name}: {count} ({count / total:.1%} of comments)", file=sys.stderr)
    parse_failure_rate = frequencies[taxonomy.PARSE_FAILURE] / total
    print(f"Unclassified: {frequencies[taxonomy.UNCLASSIFIED] / total:.1%} | "
          f"Parse failures: {parse_failure_rate:.1%} | Unstable across votes: "
          f"not measured (one vote, temperature 0)", file=sys.stderr)
    sentiment_labels = collections.Counter(result["label"] for result in sentiment.values())
    sentiment_failure_rate = (sentiment_labels[sentiment_mod.PARSE_FAILURE]
                              + sentiment_labels[sentiment_mod.NO_CONSENSUS]) / total
    print(f"Sentiment: parse failures or no consensus {sentiment_failure_rate:.1%} | "
          f"irrelevant {sentiment_labels[sentiment_mod.IRRELEVANT] / total:.1%} | uncodeable "
          f"{sentiment_labels[sentiment_mod.UNCODEABLE] / total:.1%}", file=sys.stderr)
    print(f"Time: {format_duration(elapsed)} for {total} comment(s) "
          f"({elapsed / total:.1f} s per comment, both questions). Cost: none (local).",
          file=sys.stderr)
    report_timing(steady_state(domain_timings), steady_state(sentiment_timings),
                  args.project_to, judges.estimate_tokens(taxonomy.build_prompt(
                      selector.select(""))))

    truncated_bodies = (judge.usage["truncated_bodies"]
                        + scorer._judge.usage["truncated_bodies"])
    truncated_replies = (judge.usage["truncated_replies"]
                         + scorer._judge.usage["truncated_replies"])
    if truncated_bodies:
        print(f"NOTE: {truncated_bodies} comment(s) were too long for the context window and "
              f"were cut before labelling.", file=sys.stderr)
    if truncated_replies:
        print(f"NOTE: {truncated_replies} reply(ies) hit the {gemma_local.NUM_PREDICT}-token "
              f"limit, which usually means the model wrote a sentence instead of labels.",
              file=sys.stderr)
    if parse_failure_rate > 0.02:
        print(f"WARNING: {parse_failure_rate:.1%} of domain replies could not be read as a "
              f"valid answer at all. This is a classifier reliability problem, not a finding "
              f"about the comments. Run `ewl-gemma --check` and read the raw replies.",
              file=sys.stderr)
    print("Stability is not accuracy. Run evaluate.py against a hand-coded gold set before "
          "quoting any of these labels in the write-up.", file=sys.stderr)
    # The run record `ewl-batches check` reads: what input, which instrument, which prompts.
    record_args = types.SimpleNamespace(
        input_csv=args.input_csv, output=args.output, pool=None, k=None, seed=None,
        no_cache=bool(args.no_cache), cache=args.cache)
    classify_mod.write_run_manifest(
        args.output + ".run.json", record_args, judge, scorer, selector,
        {"comments": total, "domain_frequencies": dict(frequencies),
         "sentiment_labels": dict(sentiment_labels), "sample": args.sample,
         "limit": args.limit, "blank_body_rows": blank, "seconds": round(elapsed, 1),
         "calls": judge.usage["calls"] + scorer._judge.usage["calls"]})
    os.replace(pending_output, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
