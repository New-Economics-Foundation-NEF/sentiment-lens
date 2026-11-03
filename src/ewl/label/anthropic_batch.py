"""Label a corpus with Claude through Anthropic's Message Batches API (half price, slower).

    ewl-anthropic-batch plan   corpus.csv --sample 200 --seed 7        # dry run: counts and cost, no API call
    ewl-anthropic-batch submit corpus.csv --out-dir official-data/batch_trial --sample 200 --seed 7
    ewl-anthropic-batch status official-data/batch_trial               # progress, and (--check-cache) gaps
    ewl-anthropic-batch fetch  official-data/batch_trial               # download finished batches into the cache
    ewl-anthropic-batch replay official-data/batch_trial --output labelled.csv
    ewl-anthropic-batch compare labelled.csv live_labelled.csv          # batch vs a live run

What this is
------------
Not a new judge: a different way of SENDING the requests `ewl-classify` would make, so the
labels, the cache and every downstream tool are the same. The cache spec stays
`anthropic:<model>`, so an answer collected here is replayed under exactly the key a live
call would have written, and anything missing can still be filled in live.

1. `submit` walks the corpus through the real labelling code (`judge.label`, `scorer.score`)
   with a collector in place of the API. A reply already in the cache is skipped (so running
   it again sends only the gaps); a missing one becomes a batch request whose `custom_id` is
   its cache key. Requests are sent batch by batch as they are built, never all held at once.
2. `fetch` downloads finished batches and writes every normal answer into the cache, by the
   same rules as the live path: a refusal or a reply cut off at the token limit is NOT cached.
   Its key, reason and reply text (never the comment) go to `batch_uncacheable.jsonl`, so
   `replay` uses it as the live path would and `submit` does not pay for it again (unless
   `--retry-uncacheable`). A batch whose results cannot be downloaded does not stop the
   others; `fetch --abandon <id>` gives one up for good, and its requests become gaps.
3. `replay` labels the corpus from the cache alone (it can never call the API) and writes the
   same columns and run record as `ewl-classify`.

A batch is recorded in the manifest BEFORE it is sent, and the SDK's own retries are off for
that call: a lost reply to `create` must not become a second, unrecorded, billed batch. If
`create` fails, `status`, `fetch` and `submit` first check Anthropic's list of batches and
adopt whatever was created (or drop the placeholder if nothing was).

The manifest also records the judges' cache specs and the prompt hashes; `submit`,
`status --check-cache` and `replay` refuse when they no longer match, since a changed prompt
(or EWL_ANTHROPIC_EFFORT) changes every cache key and would silently send everything again.

Use a SEPARATE cache file for the batch trial from any live comparison run (the default is
`<out-dir>/batch_cache.jsonl`). If a live run fills the same cache first, `submit` finds
everything already answered and sends nothing.

Cost: batch usage is billed at half the list price. `plan` estimates it from the prompt sizes;
`fetch` reports it from the real token counts, beside the same tokens at list price. Requests
that error, expire or are cancelled are not billed.

Limits (Anthropic's): 100,000 requests or 256 MB per batch, whichever comes first; batches
expire after 24 hours; results are kept for 29 days after creation (download them in that
window; once they are in the cache they are yours). The defaults here are well inside the
limits (50,000 requests, 150 MB). Comment text goes to Anthropic, exactly as with live calls,
and Anthropic holds the batch and its results for up to 29 days.

The manifest (`batch_manifest.json`) holds batch ids, counts, hashes and the settings, never
comment text. The subset file written by `--sample` DOES contain comment text: keep it in
`official-data/`.
"""
import argparse
import collections
import csv
import datetime
import json
import os
import random
import sys
import types

from ewl import corpus as corpus_mod
from ewl import taxonomy
from ewl.label import classify as classify_mod
from ewl.label import examples as examples_mod
from ewl.label import instrument, judges
from ewl.label import sentiment as sentiment_mod

MANIFEST = "batch_manifest.json"
UNCACHEABLE = "batch_uncacheable.jsonl"   # key, reason and reply of refused / cut-off replies
CREATE_TIMEOUT_SECONDS = 600         # the longest one `batches.create` call may take (set on it)
RECONCILE_SLACK_SECONDS = 600        # clock difference (ours vs Anthropic's) allowed when matching
# A lost batch can only have been created between the placeholder and the end of its `create`
# call. Until that window has passed (plus slack) a batch that is not listed yet may still
# appear: wait, do not drop.
PLACEHOLDER_GRACE_SECONDS = CREATE_TIMEOUT_SECONDS + RECONCILE_SLACK_SECONDS
DEFAULT_MAX_REQUESTS = 50_000        # Anthropic's limit is 100,000
DEFAULT_MAX_MB = 150                 # Anthropic's limit is 256 MB
BATCH_DISCOUNT = 0.5                 # batch usage is billed at 50% of list price
CACHE_WRITE_MULTIPLIER = {"5m": 1.25, "1h": 2.0}   # of the input price (prompt-caching page)
EST_OUTPUT_TOKENS = 8                # a label, or a number
sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------- rows and subset
def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "body" not in reader.fieldnames:
            sys.exit(f"{path} has no `body` column to label.")
        return list(reader.fieldnames), list(reader)


def write_rows(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stratified_sample(rows, n, seed):
    """`n` rows at random, spread as evenly as possible across stories, in their original
    order. Each story has its own seeded order, so a bigger `n` with the same seed contains
    every comment of a smaller one."""
    if n >= len(rows):
        return list(rows)
    groups = collections.defaultdict(list)
    for index, row in enumerate(rows):
        groups[row.get("story") or ""].append(index)
    quota = {story: 0 for story in groups}
    open_stories, remaining = sorted(groups), n
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
    chosen = []
    for story in sorted(groups):
        order = list(groups[story])
        random.Random(f"{seed}:{story}").shuffle(order)
        chosen += order[:quota[story]]
    return [rows[i] for i in sorted(chosen)]


def _has_text(row):
    return bool((row.get("body") or "").strip())


# ---------------------------------------------------------------- the labelling context
SETTINGS = ("judge", "sentiment", "votes", "ttl", "cache", "fixed_examples", "story_rules",
            "stance_object", "combined_domain_context", "domain_context", "sentiment_examples")


def settings_from(args):
    return {name: getattr(args, name, None) for name in SETTINGS}


def build_context(settings):
    """The judge, scorer and example selector, built exactly as `ewl-classify` builds them."""
    args = types.SimpleNamespace(**settings)
    # Every comment is asked the sentiment question too (the scorer must be llm:anthropic:...),
    # so an examples file without ratings is refused, as ewl-gemma and the scorecard refuse it.
    instrument.apply(args, needs_ratings=True)
    cache = settings["cache"]
    try:
        judge = judges.create(settings["judge"], votes=settings["votes"], cache_path=cache)
    except judges.PriceUnavailable as error:
        sys.exit(str(error))                  # an unpriced model: nothing is planned or sent
    if not isinstance(judge, judges.AnthropicJudge):
        sys.exit("ewl-anthropic-batch only works with an anthropic: judge "
                 f"(got {settings['judge']!r}).")
    selector = examples_mod.create("fixed")
    scorer = sentiment_mod.create(settings["sentiment"], votes=settings["votes"], cache_path=cache)
    inner = getattr(scorer, "_judge", None)
    if not isinstance(inner, judges.AnthropicJudge) or inner.model != judge.model:
        sys.exit("The sentiment scorer must be llm:anthropic:<the same model as --judge> "
                 f"(got {settings['sentiment']!r}).")
    ttl = None if settings["ttl"] == "5m" else settings["ttl"]
    judge.batch_ttl = inner.batch_ttl = ttl
    return types.SimpleNamespace(judge=judge, scorer=scorer, selector=selector,
                                 settings=settings)


def _label_row(ctx, row):
    story = row.get("story")
    body = row["body"].strip()      # as ewl-classify: the cache key and the prompt use the stripped text
    ctx.judge.label(body, ctx.selector.select(body), story=story)
    ctx.scorer.score(body, story=story)


def plan_rows(ctx, rows, on_request):
    """Run every comment through the real labelling code with the collectors in place of the
    API. `on_request(key, params)` is called for each reply that is not already cached.
    Returns per-comment missing-request counts, in row order (comments with text only)."""
    ctx.judge.cache_only = ctx.scorer._judge.cache_only = False
    counts = []
    state = {"n": 0}

    def counting(key, params):
        state["n"] += 1
        on_request(key, params)

    ctx.judge.collector = ctx.scorer._judge.collector = counting
    try:
        for row in rows:
            if not _has_text(row):
                continue
            state["n"] = 0
            _label_row(ctx, row)
            counts.append(state["n"])
    finally:
        ctx.judge.collector = ctx.scorer._judge.collector = None
    return counts


# ---------------------------------------------------------------- cost
def estimate_tokens(params):
    chars = len(params["system"][0]["text"]) + len(params["messages"][0]["content"])
    return chars / 4, chars / 3        # (typical, pessimistic)


def cost_from_usage(judge, usage, ttl, batch=True):
    """Dollar cost of real token counts. `batch` applies the 50% batch discount."""
    rate_in, rate_out = judge.price_per_mtok
    write_mult = CACHE_WRITE_MULTIPLIER["5m" if ttl in (None, "5m") else ttl]
    total = (usage.get("input_tokens", 0) / 1e6 * rate_in
             + usage.get("output_tokens", 0) / 1e6 * rate_out
             + usage.get("cache_creation_input_tokens", 0) / 1e6 * rate_in * write_mult
             + usage.get("cache_read_input_tokens", 0) / 1e6 * rate_in * judge.CACHE_READ_MULTIPLIER)
    return total * (BATCH_DISCOUNT if batch else 1.0)


# ---------------------------------------------------------------- manifest
def manifest_path(out_dir):
    return os.path.join(out_dir, MANIFEST)


def load_manifest(out_dir):
    path = manifest_path(out_dir)
    if not os.path.exists(path):
        sys.exit(f"No {MANIFEST} in {out_dir}: run `ewl-anthropic-batch submit` first.")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_manifest(out_dir, manifest):
    os.makedirs(out_dir, exist_ok=True)
    temp = manifest_path(out_dir) + ".tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
        handle.write("\n")
    os.replace(temp, manifest_path(out_dir))


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _as_time(value):
    """A manifest timestamp (ISO text) or an SDK one (datetime), as an aware datetime."""
    if not isinstance(value, datetime.datetime):
        value = datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return value if value.tzinfo else value.replace(tzinfo=datetime.timezone.utc)


def _live_batches(manifest):
    """The batches still to fetch: recorded, not fetched, not abandoned."""
    return [b for b in manifest["batches"] if not b["fetched"] and not b.get("abandoned")]


def reconcile(manifest, client, out_dir):
    """Settle the batches whose `create` call never returned an id.

    The entry is written before the call, so a dropped connection or a crash leaves it here
    with no id. The batch may or may not exist: Anthropic's list of batches says which. Every
    unrecorded batch created since the placeholder, with the same number of requests, is
    adopted (more than one means the same requests were billed more than once, and is said
    so); none means it was never created, and the placeholder is dropped so its requests
    count as gaps. A placeholder younger than PLACEHOLDER_GRACE_SECONDS is left alone, since
    a batch that new may not be listed yet.

    The match is bounded on BOTH sides: a batch can only come from this placeholder if it was
    created while its `create` call could still have been running (CREATE_TIMEOUT_SECONDS),
    give or take the clock slack. Without the upper bound, any later same-size batch (from
    another folder: every full batch is DEFAULT_MAX_REQUESTS) would be adopted as this one."""
    placeholders = [b for b in manifest["batches"] if not b.get("id")]
    if not placeholders:
        return
    slack = datetime.timedelta(seconds=RECONCILE_SLACK_SECONDS)
    oldest = min(_as_time(b["created_at"]) for b in placeholders) - slack
    try:
        listed = []
        for batch in client.messages.batches.list(limit=100):      # newest first
            if _as_time(batch.created_at) < oldest:
                break
            listed.append(batch)
    except Exception as error:
        sys.exit(f"{len(placeholders)} batch(es) in {out_dir} were being sent when the last "
                 f"run stopped, and Anthropic's list of batches could not be read to see "
                 f"whether they exist ({type(error).__name__}: {error}). Nothing was sent; "
                 f"try again.")
    recorded = {b["id"] for b in manifest["batches"] if b.get("id")}
    now = datetime.datetime.now(datetime.timezone.utc)
    window = datetime.timedelta(seconds=CREATE_TIMEOUT_SECONDS)
    for entry in placeholders:
        since = _as_time(entry["created_at"]) - slack
        until = _as_time(entry["created_at"]) + window + slack
        found = [b for b in reversed(listed)
                 if b.id not in recorded and since <= _as_time(b.created_at) <= until
                 and _request_total(b.request_counts) == entry["requests"]]
        if found:
            entry["id"] = found[0].id
            entry["adopted"] = True
            recorded.add(found[0].id)
            for extra in found[1:]:
                manifest["batches"].append(dict(entry, id=extra.id, duplicate_of=found[0].id))
                recorded.add(extra.id)
            print(f"  adopted batch {found[0].id} ({entry['requests']:,} requests), sent by a "
                  f"run that stopped before recording it"
                  + (f"; {len(found) - 1} more with the same requests were also created and "
                     f"billed, and are adopted too: {', '.join(b.id for b in found[1:])}"
                     if len(found) > 1 else "") + ".", file=sys.stderr)
        elif now - _as_time(entry["created_at"]) < datetime.timedelta(
                seconds=PLACEHOLDER_GRACE_SECONDS):
            sys.exit(f"A batch of {entry['requests']:,} requests was being sent at "
                     f"{entry['created_at']} and is not in Anthropic's list yet. Wait a few "
                     f"minutes and run this again, so it is not sent twice.")
        else:
            manifest["batches"].remove(entry)
            print(f"  the batch of {entry['requests']:,} requests being sent at "
                  f"{entry['created_at']} was never created: its requests are gaps, and the "
                  f"next `submit` sends them.", file=sys.stderr)
    save_manifest(out_dir, manifest)


def _request_total(counts):
    return sum(getattr(counts, field, 0) or 0
               for field in ("processing", "succeeded", "errored", "canceled", "expired"))


# ---------------------------------------------------------------- refused / cut-off replies
def uncacheable_path(out_dir):
    return os.path.join(out_dir, UNCACHEABLE)


def load_uncacheable(out_dir):
    """{cache key: (reason, reply)} for replies that came back refused or cut off. The last
    line for a key wins, so a retried key that failed again carries its latest reply."""
    path = uncacheable_path(out_dir) if out_dir else None
    known = {}
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue            # a truncated final line from an interrupted fetch
                known[record["key"]] = (record["reason"], record["reply"])
    return known


def use_uncacheable(ctx, known):
    """Hand the judges the recorded refusals and cut-off replies (see Judge._cached_sample)."""
    ctx.judge.known_uncacheable = ctx.scorer._judge.known_uncacheable = known or None


def _uncacheable_used(ctx):
    return ctx.judge.usage["uncacheable_replayed"] + ctx.scorer._judge.usage["uncacheable_replayed"]


# ---------------------------------------------------------------- the instrument guard
def _shared(record, other):
    """`record` restricted to the keys `other` also has (a dict), else `record` itself."""
    if isinstance(record, dict) and isinstance(other, dict):
        return {k: v for k, v in record.items() if k in other}
    return record


def instrument_record(ctx):
    """What the cache keys depend on beyond the flags: each judge's cache spec (the model,
    and EWL_ANTHROPIC_EFFORT) and the prompt text."""
    return {"cache_specs": {"domain": ctx.judge.cache_spec,
                            "sentiment": ctx.scorer._judge.cache_spec},
            "prompts_sha256": prompt_hashes(ctx)}


def check_instrument(manifest, ctx):
    """Refuse when the prompts or cache specs differ from the ones the folder was started
    with: every cache key would differ, so `submit` would quietly send everything again and
    `replay` would find nothing."""
    now = instrument_record(ctx)
    # Only what both sides record: a folder started by an earlier version of this tool has no
    # cache specs and fewer prompt hashes, and that is not a change.
    changed = [name for name in now if name in manifest
               and _shared(manifest[name], now[name]) != _shared(now[name], manifest[name])]
    if changed:
        names = {"cache_specs": "judges' cache specs", "prompts_sha256": "prompts"}
        sys.exit(f"The {' and '.join(names[n] for n in changed)} differ from the ones "
                 f"this folder was started with (a prompt edit, a different --fixed-examples "
                 f"file, or EWL_ANTHROPIC_EFFORT set differently). Every cache key would change. "
                 f"Restore them, or use a new --out-dir.")


# ---------------------------------------------------------------- submit / plan
class Submitter:
    """Collects requests and sends them batch by batch as the size limits are reached."""

    def __init__(self, ctx, out_dir, manifest, max_requests, max_bytes, dry_run):
        self.ctx, self.out_dir, self.manifest = ctx, out_dir, manifest
        self.max_requests, self.max_bytes, self.dry_run = max_requests, max_bytes, dry_run
        self.pending, self.pending_bytes, self.seen = [], 0, set()
        self.requests = self.duplicates = self.batches = 0
        self.tokens = self.tokens_pessimistic = 0.0
        # The system prompt's share (the part a prompt-cache write is charged on), characters / 3.
        self.system_tokens_pessimistic = self.max_system_tokens = 0.0

    def add(self, key, params):
        if key in self.seen:
            self.duplicates += 1       # an identical request (same comment text in the same story)
            return
        self.seen.add(key)
        size = len(json.dumps(params)) + 120
        if self.pending and (len(self.pending) >= self.max_requests
                             or self.pending_bytes + size > self.max_bytes):
            self.flush()
        self.pending.append({"custom_id": key, "params": params})
        self.pending_bytes += size
        self.requests += 1
        typical, pessimistic = estimate_tokens(params)
        self.tokens += typical
        self.tokens_pessimistic += pessimistic
        system = len(params["system"][0]["text"]) / 3
        self.system_tokens_pessimistic += system
        self.max_system_tokens = max(self.max_system_tokens, system)

    def flush(self):
        if not self.pending:
            return
        self.batches += 1
        entry = {"id": None, "requests": len(self.pending), "bytes_estimate": self.pending_bytes,
                 "created_at": _now(), "fetched": False}
        if not self.dry_run:
            # On disk BEFORE the call: if the reply to `create` is lost, the next run finds
            # this placeholder and checks Anthropic's list (`reconcile`) instead of sending the
            # same requests again.
            self.manifest["batches"].append(entry)
            save_manifest(self.out_dir, self.manifest)
            # No SDK retries here. The SDK would resend the whole batch after a dropped
            # connection, with no idempotency key: if the first attempt had been accepted, that
            # is a second billed batch that nothing records.
            client = self.ctx.judge._client.with_options(max_retries=0,
                                                         timeout=CREATE_TIMEOUT_SECONDS)
            try:
                batch = client.messages.batches.create(requests=self.pending)
            except Exception as error:
                status = getattr(error, "status_code", None)
                if isinstance(status, int) and 400 <= status < 500 and status != 408:
                    # A definite refusal (bad request, auth, too large, rate limit): no batch
                    # was created, so the placeholder goes and nothing is left blocking.
                    self.manifest["batches"].remove(entry)
                    save_manifest(self.out_dir, self.manifest)
                    print(f"\n  Anthropic refused the batch ({status}); none was created.",
                          file=sys.stderr)
                    raise
                print(f"\n  sending a batch of {entry['requests']:,} requests failed; it may or "
                      f"may not have been created. Run `ewl-anthropic-batch status "
                      f"{self.out_dir}`: it checks Anthropic's list of batches and records it, "
                      f"or drops it, before anything is sent again.", file=sys.stderr)
                raise
            entry["id"] = batch.id
            save_manifest(self.out_dir, self.manifest)     # the id is on disk before anything else
            print(f"  submitted batch {batch.id}: {entry['requests']:,} requests", file=sys.stderr)
        self.pending, self.pending_bytes = [], 0


def cmd_submit(args, dry_run=False):
    fieldnames, rows = read_rows(args.input_csv)
    effective = args.input_csv
    subset_note = subset_new = None
    if args.sample or args.limit:
        if args.sample and args.limit:
            sys.exit("use --sample or --limit, not both")
        chosen = (stratified_sample([r for r in rows if _has_text(r)], args.sample, args.seed)
                  if args.sample else [r for r in rows if _has_text(r)][:args.limit])
        subset_note = {"sample": args.sample, "limit": args.limit, "seed": args.seed,
                       "from": os.path.basename(args.input_csv), "rows": len(chosen)}
        rows = chosen
        if not dry_run:
            effective = os.path.join(args.out_dir, "subset.csv")
            # Written beside, not over, the subset an earlier submit used: a run refused below
            # (a different --sample or --seed) must leave that folder's subset.csv as it was,
            # or the folder could no longer be replayed.
            subset_new = effective + ".new"
            write_rows(subset_new, fieldnames, rows)
    try:
        return _submit(args, dry_run, fieldnames, rows, effective, subset_note, subset_new)
    finally:
        if subset_new and os.path.exists(subset_new):
            os.remove(subset_new)


def _submit(args, dry_run, fieldnames, rows, effective, subset_note, subset_new):
    cache = args.cache or os.path.join(args.out_dir or ".", "batch_cache.jsonl")
    settings = settings_from(args)
    settings["cache"] = cache
    if not dry_run:
        classify_mod.refuse_empty_cache(cache, False)
    ctx = build_context(settings)
    taxonomy_tags = instrument_tags()

    max_cost = getattr(args, "max_cost", None)
    if max_cost is not None and not dry_run:
        # Replies already known as refusals or cut-off replies are not sent again, so the
        # estimate must know them too.
        use_uncacheable(ctx, {} if args.retry_uncacheable else load_uncacheable(args.out_dir))
        refuse_over_cap(ctx, rows, max_cost, args)

    manifest = None
    if not dry_run:
        manifest = (load_manifest(args.out_dir) if os.path.exists(manifest_path(args.out_dir))
                    else None)
        sha = corpus_mod.content_hash(subset_new or effective)
        if manifest is not None:
            reconcile(manifest, ctx.judge._client, args.out_dir)
            if manifest["input"]["sha256"] != sha:
                sys.exit(f"{args.out_dir} was started for a different input. Use a new --out-dir.")
            if manifest["settings"] != settings:
                sys.exit(f"{args.out_dir} was started with different settings (model, votes, "
                         f"instrument, cache). Use a new --out-dir, or the same flags.")
            check_instrument(manifest, ctx)
            unfetched = [b["id"] for b in _live_batches(manifest)]
            if unfetched:
                sys.exit(f"{len(unfetched)} earlier batch(es) are not fetched yet "
                         f"({unfetched[0]}...). Run `ewl-anthropic-batch fetch {args.out_dir}` "
                         f"first, or this would send the same requests twice.")
        if subset_new:
            os.replace(subset_new, effective)
        if manifest is None:
            manifest = {"created_at": _now(), "settings": settings, "tags": taxonomy_tags,
                        "input": {"path": effective, "sha256": sha, "rows": len(rows),
                                  "original": os.path.basename(args.input_csv)},
                        "subset": subset_note, "batches": []}
            manifest.update(instrument_record(ctx))
            save_manifest(args.out_dir, manifest)
    # Refusals and cut-off replies already collected are used as they are, as the live path
    # would, not bought again -- unless asked (a larger token budget, say).
    known = {} if args.retry_uncacheable else load_uncacheable(args.out_dir)
    use_uncacheable(ctx, known)
    submitter = Submitter(ctx, args.out_dir, manifest, args.max_batch_requests,
                          int(args.max_batch_mb * 1024 * 1024), dry_run)
    comments = [r for r in rows if _has_text(r)]
    print(f"Planning {len(comments):,} comment(s) with {ctx.judge.spec} | examples: "
          f"{ctx.selector.describe()} | votes {args.votes} | cache {cache}", file=sys.stderr)
    counts = plan_rows(ctx, comments, submitter.add)
    submitter.flush()
    estimate_report(ctx, submitter, counts, len(comments), dry_run)
    if _uncacheable_used(ctx):
        print(f"{_uncacheable_used(ctx):,} earlier refusal(s) or cut-off repl(ies) are NOT sent "
              f"again (`replay` uses them as the live path would). --retry-uncacheable sends "
              f"them again.", file=sys.stderr)
    if not dry_run:
        manifest["planned"] = {"at": _now(), "requests": submitter.requests,
                               "comments": len(comments), "batches": submitter.batches}
        save_manifest(args.out_dir, manifest)
        if submitter.requests:
            print(f"\nSubmitted {submitter.requests:,} request(s) in {submitter.batches} "
                  f"batch(es). Most batches finish within an hour. Then:\n"
                  f"  ewl-anthropic-batch fetch {args.out_dir}", file=sys.stderr)
        else:
            print("\nNothing to send: every reply is already in the cache.", file=sys.stderr)
    return 0


def instrument_tags():
    return (sentiment_mod.config_tag() + taxonomy.domain_context_tag())


def prompt_hashes(ctx):
    import hashlib

    def digest(text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {"domain_by_story": {s: digest(taxonomy.build_prompt(ctx.selector.select(""), s))
                                for s in sorted(sentiment_mod.STORY_CONTEXT)},
            "sentiment_by_story": {s: digest(t) for s, t in
                                   sorted(getattr(ctx.scorer, "_prompts", {}).items())},
            # The prompts a comment with no recognised story gets.
            "domain_no_story": digest(taxonomy.build_prompt(ctx.selector.select(""), None)),
            "sentiment_no_story": digest(getattr(ctx.scorer, "_fallback_prompt", ""))}


def estimate_costs(ctx, submitter):
    """(typical, pessimistic) BATCH cost of what `submitter` has collected: the two figures
    `estimate_report` prints. The pessimistic one (characters / 3) is what a cap is held to."""
    usage = {"input_tokens": submitter.tokens, "output_tokens": submitter.requests * EST_OUTPUT_TOKENS}
    pess = {"input_tokens": submitter.tokens_pessimistic, "output_tokens": usage["output_tokens"]}
    ttl = ctx.settings["ttl"]
    return cost_from_usage(ctx.judge, usage, ttl), cost_from_usage(ctx.judge, pess, ttl)


def refuse_over_cap(ctx, rows, max_cost, args):
    """With --max-cost: plan everything once WITHOUT sending (a dry-run submitter) and stop if
    the estimate is over the cap, before a manifest, a subset file or a single batch exists.

    Held to the pessimistic estimate (characters / 3), which is the figure `plan` prints as the
    top of its range. The prompt-cache write worst case (shown by `plan` for a model that can
    cache these prompts) can be higher still, so a cap near the estimate is a cap on the
    estimate, not a ceiling on the bill."""
    probe = Submitter(ctx, args.out_dir, None, args.max_batch_requests,
                      int(args.max_batch_mb * 1024 * 1024), dry_run=True)
    # Planning counts uncacheable replies replayed from the record; keep the real run's counters.
    saved = (ctx.judge.usage.copy(), ctx.scorer._judge.usage.copy())
    try:
        plan_rows(ctx, [r for r in rows if _has_text(r)], probe.add)
        probe.flush()
    finally:
        for owner, usage in zip((ctx.judge, ctx.scorer._judge), saved):
            owner.usage.clear()
            owner.usage.update(usage)
    if not probe.requests:
        return
    low, high = estimate_costs(ctx, probe)
    # The ceiling is the worst case, not the estimate: a model that can cache the system prompt
    # is charged a cache WRITE (2x the input price at --ttl 1h) on every request that finds no
    # cached copy, and hits inside a batch are best effort. Hold the cap to that figure.
    ttl = ctx.settings["ttl"]
    minimum = judges.cache_minimum_for(ctx.judge.model)
    if minimum is not None and probe.max_system_tokens >= minimum:
        system = probe.system_tokens_pessimistic
        high = max(high, cost_from_usage(ctx.judge, {
            "input_tokens": probe.tokens_pessimistic - system,
            "cache_creation_input_tokens": system,
            "output_tokens": probe.requests * EST_OUTPUT_TOKENS}, ttl))
    print(f"--max-cost ${max_cost:,.2f}: estimated BATCH cost ${low:,.2f} to ${high:,.2f} "
          f"(the top is the worst case, every request writing the prompt cache) for "
          f"{probe.requests:,} request(s).", file=sys.stderr)
    if high > max_cost:
        sys.exit(f"Refusing to submit: the pessimistic estimate, ${high:,.2f}, is over --max-cost "
                 f"${max_cost:,.2f}. Nothing was sent and nothing was written. Raise the cap, "
                 f"or send fewer comments (--sample / --limit), or run `plan` for the full "
                 f"estimate.")


def estimate_report(ctx, submitter, counts, n_comments, dry_run):
    done = sum(1 for c in counts if c == 0)
    print(f"\n{n_comments:,} comment(s): {done:,} already fully in the cache, "
          f"{n_comments - done:,} need at least one request.", file=sys.stderr)
    print(f"{submitter.requests:,} request(s) to send"
          + (f" ({submitter.duplicates:,} identical ones counted once)" if submitter.duplicates else "")
          + f", in {submitter.batches} batch(es).", file=sys.stderr)
    if not submitter.requests:
        return
    usage = {"input_tokens": submitter.tokens, "output_tokens": submitter.requests * EST_OUTPUT_TOKENS}
    pess = {"input_tokens": submitter.tokens_pessimistic, "output_tokens": usage["output_tokens"]}
    ttl = ctx.settings["ttl"]
    print(f"Estimated input about {submitter.tokens:,.0f} tokens (characters / 4; "
          f"{submitter.tokens_pessimistic:,.0f} if characters / 3), output about "
          f"{usage['output_tokens']:,.0f}.", file=sys.stderr)
    low = cost_from_usage(ctx.judge, usage, ttl)
    high = cost_from_usage(ctx.judge, pess, ttl)
    print(f"Estimated BATCH cost ${low:,.2f} to ${high:,.2f}  (list price "
          f"${low * 2:,.2f} to ${high * 2:,.2f}); per comment about "
          f"${low / max(n_comments - done, 1):.4f}. No prompt-cache discount is assumed: it is "
          f"best-effort inside batches, and Haiku 4.5's 4,096-token minimum is above these "
          f"prompts. The real figure comes from `fetch`.", file=sys.stderr)
    # The figures above charge the system prompt as plain input. A model that can cache it
    # charges a cache WRITE instead whenever a request finds no cached copy -- 2x the input
    # price at 1h -- and hits inside a batch are best effort. So the high figure above is not
    # a ceiling: this one is (every request writing the cache).
    minimum = judges.cache_minimum_for(ctx.judge.model)
    if minimum is not None and submitter.max_system_tokens >= minimum:
        system = submitter.system_tokens_pessimistic
        worst = cost_from_usage(ctx.judge, {
            "input_tokens": submitter.tokens_pessimistic - system,
            "cache_creation_input_tokens": system,
            "output_tokens": usage["output_tokens"]}, ttl)
        print(f"If every request writes the prompt cache (no hits), up to ${worst:,.2f} at "
              f"BATCH price: a cache write costs {CACHE_WRITE_MULTIPLIER[ttl or '5m']}x the "
              f"input price at --ttl {ttl or '5m'} (2x at 1h, 1.25x at 5m).", file=sys.stderr)
    if dry_run:
        print("(dry run: nothing was sent and nothing was written.)", file=sys.stderr)


# ---------------------------------------------------------------- status
def _client_for(manifest):
    judge = judges.create(manifest["settings"]["judge"], votes=1,
                          cache_path=manifest["settings"]["cache"])
    return judge


def cmd_status(args):
    manifest = load_manifest(args.out_dir)
    judge = _client_for(manifest)
    reconcile(manifest, judge._client, args.out_dir)
    s = manifest["settings"]
    print(f"{manifest['input']['rows']:,} comment(s) planned from {manifest['input']['original']}"
          + (f" (sample, seed {manifest['subset']['seed']})" if manifest.get("subset") and
             manifest["subset"].get("sample") else "") + f" | {s['judge']} | votes {s['votes']}.")
    for batch in manifest["batches"]:
        line = f"  {batch['id']}  {batch['requests']:,} requests  "
        if batch["fetched"]:
            line += (f"FETCHED {batch['fetched_at']}: " + ", ".join(
                f"{k} {v}" for k, v in sorted(batch["counts"].items())))
        elif batch.get("abandoned"):
            line += f"ABANDONED {batch['abandoned_at']}: its requests count as gaps"
        else:
            live = judge._client.messages.batches.retrieve(batch["id"])
            c = live.request_counts
            line += (f"{live.processing_status}: processing {c.processing}, succeeded "
                     f"{c.succeeded}, errored {c.errored}, expired {c.expired}, "
                     f"cancelled {c.canceled}")
        print(line)
    if not manifest["batches"]:
        print("  no batches sent (everything was already cached).")
    if args.check_cache:
        report_gaps(manifest, args.out_dir)
    return 0


def report_gaps(manifest, out_dir):
    """Re-plan against the cache: a comment is done only when every question and vote has a
    reply, cached or recorded as a refusal or cut off (which `replay` uses as live does)."""
    ctx = build_context(manifest["settings"])
    _, rows = read_rows(manifest["input"]["path"])
    check_input(manifest)
    check_instrument(manifest, ctx)
    use_uncacheable(ctx, load_uncacheable(out_dir))
    comments = [r for r in rows if _has_text(r)]
    sink = types.SimpleNamespace(add=lambda key, params: None)
    counts = plan_rows(ctx, comments, sink.add)
    done = sum(1 for c in counts if c == 0)
    print(f"\nComments complete (every reply cached or recorded): {done:,} of {len(comments):,}; "
          f"{len(comments) - done:,} incomplete ({sum(counts):,} replies missing).")
    if _uncacheable_used(ctx):
        print(f"{_uncacheable_used(ctx):,} of the replies are refusals or cut off, recorded and "
              f"used as the live path uses them (`submit --retry-uncacheable` asks again).")
    causes = collections.Counter()
    for batch in manifest["batches"]:
        if batch["fetched"]:
            for k, v in (batch.get("not_cached") or {}).items():
                causes[k] += v
    if causes:
        print("Why replies were not cached (from the fetched batches): "
              + ", ".join(f"{k} {v}" for k, v in sorted(causes.items())))
    pending = _live_batches(manifest)
    if pending:
        print(f"{len(pending)} batch(es) are not fetched yet, so some of the gaps may still arrive.")
    if len(comments) - done:
        print("Run `ewl-anthropic-batch submit` again with the same flags to send only the gaps.")


def check_input(manifest):
    sha = corpus_mod.content_hash(manifest["input"]["path"])
    if sha != manifest["input"]["sha256"]:
        sys.exit(f"{manifest['input']['path']} has changed since the batches were submitted "
                 f"(its hash no longer matches the manifest).")


# ---------------------------------------------------------------- fetch
def cmd_fetch(args):
    manifest = load_manifest(args.out_dir)
    judge = _client_for(manifest)
    client = judge._client
    if args.abandon:
        return abandon(manifest, args.out_dir, args.abandon, client, cancel=args.cancel)
    reconcile(manifest, client, args.out_dir)
    ttl = manifest["settings"]["ttl"]
    fetched_any, failed = False, []
    for batch in manifest["batches"]:
        if batch.get("abandoned") or (batch["fetched"] and not args.refetch):
            continue
        # One batch that cannot be read (results gone after 29 days, a network failure) must
        # not stop the others: every batch left unfetched blocks the next `submit`.
        try:
            fetched = fetch_one(client, judge, batch, args.out_dir)
        except Exception as error:
            failed.append(batch["id"])
            print(f"  {batch['id']}: could not be fetched ({type(error).__name__}: {error}). "
                  f"The others carry on. Run `fetch` again later; if its results are gone for "
                  f"good (Anthropic keeps them 29 days), `ewl-anthropic-batch fetch "
                  f"{args.out_dir} --abandon {batch['id']}` turns its requests into gaps.",
                  file=sys.stderr)
            continue
        if fetched:
            save_manifest(args.out_dir, manifest)
            fetched_any = True
    summarise_fetch(manifest, judge, ttl)
    if fetched_any:
        print(f"\nNext: `ewl-anthropic-batch status {args.out_dir} --check-cache` to count "
              f"complete comments, then `replay`.", file=sys.stderr)
    return 1 if failed else 0


def fetch_one(client, judge, batch, out_dir):
    """Download one ended batch into the cache. False if it has not ended yet."""
    live = client.messages.batches.retrieve(batch["id"])
    if live.processing_status != "ended":
        c = live.request_counts
        print(f"  {batch['id']}: still {live.processing_status} (processing {c.processing}, "
              f"succeeded {c.succeeded}); try again later.", file=sys.stderr)
        return False
    counts, usage, not_cached, errors = (collections.Counter(), collections.Counter(),
                                         collections.Counter(), collections.Counter())
    with open(uncacheable_path(out_dir), "a", encoding="utf-8") as refused:
        for item in client.messages.batches.results(batch["id"]):
            outcome = item.result
            counts[outcome.type] += 1
            if outcome.type == "succeeded":
                message = outcome.message
                for field in ("input_tokens", "output_tokens", "cache_creation_input_tokens",
                              "cache_read_input_tokens"):
                    usage[field] += getattr(message.usage, field, 0) or 0
                reply = judge._reply_from(message)
                if isinstance(reply, judges.UncacheableReply):
                    not_cached[f"uncacheable_{reply.reason}"] += 1
                    # Billed, and an answer of a kind: kept (key, reason and the model's reply,
                    # never the comment) so `replay` uses it as the live path does and `submit`
                    # does not buy it again.
                    refused.write(json.dumps({"key": item.custom_id, "reason": reply.reason,
                                              "reply": str(reply)}) + "\n")
                else:
                    judge._cache.put(item.custom_id, reply)
            elif outcome.type == "errored":
                kind = getattr(getattr(getattr(outcome, "error", None), "error", None), "type",
                               "unknown")
                errors[kind] += 1
                not_cached[f"errored_{kind}"] += 1
            else:                                   # expired / canceled: never run, not billed
                not_cached[outcome.type] += 1
    batch.update({"fetched": True, "fetched_at": _now(), "counts": dict(counts),
                  "usage": dict(usage), "not_cached": dict(not_cached),
                  "errors": dict(errors)})
    print(f"  {batch['id']}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
          file=sys.stderr)
    return True


def abandon(manifest, out_dir, batch_id, client, cancel=False):
    """Give up on a batch that can never be fetched, so it stops blocking `submit`."""
    entry = next((b for b in manifest["batches"] if b.get("id") == batch_id), None)
    if entry is None:
        sys.exit(f"No batch {batch_id} in {out_dir}.")
    if entry["fetched"]:
        sys.exit(f"{batch_id} is already fetched; there is nothing to abandon.")
    # Only a batch that has stopped. One still running keeps being processed and billed, and
    # the next `submit` would send its requests again: paid for twice.
    try:
        status = client.messages.batches.retrieve(batch_id).processing_status
    except Exception as error:
        if not any(m in f"{type(error).__name__} {error}".lower()
                   for m in ("not_found", "notfound", "not found", "404")):
            sys.exit(f"Could not check {batch_id} ({type(error).__name__}: {error}); nothing "
                     f"was abandoned. Try again.")
        status = "gone"
    if status not in ("ended", "gone"):
        if cancel and status == "in_progress":
            client.messages.batches.cancel(batch_id)
            print(f"  {batch_id} cancel requested. Requests it already processed are billed. "
                  f"Run `fetch` once it has ended: what it finished goes into the cache, and "
                  f"the rest are gaps for the next `submit`.", file=sys.stderr)
            return 0
        sys.exit(f"{batch_id} is still {status}: it would keep running, and be billed, while "
                 f"`submit` sent its requests again. Wait for it to end and `fetch` it, or stop "
                 f"it with `fetch {out_dir} --abandon {batch_id} --cancel` and fetch it once "
                 f"it has ended.")
    entry.update({"abandoned": True, "abandoned_at": _now()})
    save_manifest(out_dir, manifest)
    print(f"  {batch_id} abandoned: its {entry['requests']:,} requests are gaps, and the next "
          f"`submit` sends them again. If it ran, it was billed all the same.", file=sys.stderr)
    return 0


def summarise_fetch(manifest, judge, ttl):
    done = [b for b in manifest["batches"] if b["fetched"]]
    if not done:
        return
    counts, usage, not_cached = collections.Counter(), collections.Counter(), collections.Counter()
    for b in done:
        counts.update(b["counts"])
        usage.update(b["usage"])
        not_cached.update(b.get("not_cached") or {})
    total = sum(counts.values())
    print(f"\nFetched {len(done)} of {len(manifest['batches'])} batch(es), {total:,} request(s): "
          + ", ".join(f"{k} {v:,}" for k, v in sorted(counts.items())), file=sys.stderr)
    if total:
        # Succeeded replies that were refusals or cut off are billed but not usable.
        unusable = (total - counts["succeeded"]
                    + sum(v for k, v in not_cached.items() if k.startswith("uncacheable_")))
        print(f"Not processed or not usable: {unusable:,} of {total:,} "
              f"({', '.join(f'{k} {v}' for k, v in sorted(not_cached.items())) or 'none'}).",
              file=sys.stderr)
    batch_cost = cost_from_usage(judge, usage, ttl, batch=True)
    list_cost = cost_from_usage(judge, usage, ttl, batch=False)
    print(f"Tokens: input {usage['input_tokens']:,}, output {usage['output_tokens']:,}, cache "
          f"written {usage['cache_creation_input_tokens']:,}, cache read "
          f"{usage['cache_read_input_tokens']:,}.", file=sys.stderr)
    print(f"Cost at BATCH price ${batch_cost:,.4f}  (the same tokens at list price "
          f"${list_cost:,.4f})"
          + (f"; per succeeded request ${batch_cost / counts['succeeded']:.5f}"
             if counts["succeeded"] else "") + ".", file=sys.stderr)


# ---------------------------------------------------------------- replay
def cmd_replay(args):
    manifest = load_manifest(args.out_dir)
    check_input(manifest)
    ctx = build_context(manifest["settings"])
    check_instrument(manifest, ctx)
    # A refusal or cut-off reply is used as the live path uses it (refusal -> parse failure, a
    # cut-off reply's partial text as written), so those comments are labelled, not left out.
    use_uncacheable(ctx, load_uncacheable(args.out_dir))
    fieldnames, rows = read_rows(manifest["input"]["path"])
    ctx.judge.cache_only = ctx.scorer._judge.cache_only = True
    kept, incomplete, blank, uncacheable_used = [], 0, 0, 0
    for row in rows:
        if not _has_text(row):
            blank += 1
            continue
        story, body = row.get("story"), row["body"].strip()    # as in `_label_row` and ewl-classify
        before = _uncacheable_used(ctx)
        try:
            labels, unstable, ballots = ctx.judge.label(body, ctx.selector.select(body), story=story)
            result = ctx.scorer.score(body, story=story)
        except judges.CacheMiss:
            incomplete += 1
            continue
        uncacheable_used += _uncacheable_used(ctx) - before
        kept.append((row, labels, unstable, ballots, result))
    if incomplete and not args.partial:
        print(f"{incomplete:,} of {incomplete + len(kept):,} comment(s) are missing at least one "
              f"reply, so nothing was written. See `ewl-anthropic-batch status {args.out_dir} "
              f"--check-cache`, or pass --partial to write only the complete ones.",
              file=sys.stderr)
        return 1
    judge, scorer, selector = ctx.judge, ctx.scorer, ctx.selector
    domain_label = f"{judge.spec}/{selector.describe()}"
    frequencies, sentiment_labels = collections.Counter(), collections.Counter()
    out_rows = []
    for row, labels, unstable, ballots, result in kept:
        frequencies.update(labels)
        sentiment_labels[result["label"]] += 1
        classify_mod.annotate_row(row, labels, unstable, ballots, result, domain_label,
                                  scorer.describe(), judge.votes, getattr(scorer, "votes", 1))
        out_rows.append(row)
    columns = fieldnames + [c for c in classify_mod.EXTRA_COLUMNS if c not in fieldnames]
    write_rows(args.output, columns, out_rows)
    total = len(out_rows)
    print(f"Labelled {total:,} comment(s) from the cache -> {args.output}", file=sys.stderr)
    if incomplete:
        print(f"{incomplete:,} incomplete comment(s) were LEFT OUT (--partial).", file=sys.stderr)
    if uncacheable_used:
        print(f"{uncacheable_used:,} repl(ies) were refusals or cut off: used as the live path "
              f"uses them, and counted in the rates below.", file=sys.stderr)
    if total:
        pf = frequencies[taxonomy.PARSE_FAILURE] / total
        spf = (sentiment_labels[sentiment_mod.PARSE_FAILURE]
               + sentiment_labels[sentiment_mod.NO_CONSENSUS]) / total
        print(f"Unclassified {frequencies[taxonomy.UNCLASSIFIED] / total:.1%} | domain parse "
              f"failures {pf:.1%} | sentiment parse failures or no consensus {spf:.1%} | "
              f"irrelevant {sentiment_labels[sentiment_mod.IRRELEVANT] / total:.1%}.",
              file=sys.stderr)
        if pf > 0.02:
            print("WARNING: more than 2% of domain replies could not be read as an answer.",
                  file=sys.stderr)
    record_args = types.SimpleNamespace(
        input_csv=manifest["input"]["path"], output=args.output, pool=None, k=None, seed=None,
        no_cache=False, cache=manifest["settings"]["cache"])
    classify_mod.write_run_manifest(
        args.output + ".run.json", record_args, judge, scorer, selector,
        {"comments": total, "domain_frequencies": dict(frequencies),
         "sentiment_labels": dict(sentiment_labels), "blank_body_rows": blank,
         "incomplete_comments_left_out": incomplete,
         "refused_or_cut_off_replies_used": uncacheable_used,
         "collected_by": "ewl-anthropic-batch",
         "batches": [{"id": b["id"], "requests": b["requests"], "counts": b.get("counts"),
                      "usage": b.get("usage"), "abandoned": bool(b.get("abandoned"))}
                     for b in manifest["batches"]],
         "live_calls": judge.usage["calls"] - judge.usage["uncacheable_planned"]})
    return 0


# ---------------------------------------------------------------- compare two labelled files
def cmd_compare(args):
    """How far two labelled files (say, a batch replay and a live run on the same comments)
    agree, comment by comment. Two runs of the same model at one vote differ a little by
    chance alone (about 1 to 3 comments in 100 in earlier trials), so a small gap is the noise
    floor, not a batch effect."""
    def load(path):
        _, rows = read_rows(path)
        return {(r.get("comment_id") or "").strip(): r for r in rows}
    a, b = load(args.file_a), load(args.file_b)
    shared = sorted(set(a) & set(b))
    if not shared:
        sys.exit("The two files have no comment_id in common.")
    print(f"{len(shared):,} comment(s) in both ({len(set(a) - set(b)):,} only in A, "
          f"{len(set(b) - set(a)):,} only in B).")

    def labels(row):
        return frozenset(x.strip() for x in (row.get("domain") or "").split(",") if x.strip())
    same_domain = sum(1 for k in shared if labels(a[k]) == labels(b[k]))
    same_value = sum(1 for k in shared if (a[k].get("sentiment_value") or "") == (b[k].get("sentiment_value") or ""))
    same_kind = sum(1 for k in shared if (a[k].get("sentiment") or "") == (b[k].get("sentiment") or ""))
    n = len(shared)
    print(f"Domain labels identical:            {same_domain:,} ({same_domain / n:.1%})")
    print(f"Sentiment rating identical:         {same_value:,} ({same_value / n:.1%})")
    print(f"Sentiment class (neg/neu/pos/...) identical: {same_kind:,} ({same_kind / n:.1%})")
    for category in taxonomy.CATEGORY_NAMES:
        col = f"domain_{category}"
        agree = sum(1 for k in shared if (a[k].get(col) or "0") == (b[k].get(col) or "0"))
        print(f"  {category:<12} flag agrees on {agree / n:.1%}")
    for name, rows in (("A", a), ("B", b)):
        pf = sum(1 for k in shared if rows[k].get("domain_parse_failure") == "1") / n
        spf = sum(1 for k in shared if rows[k].get("sentiment_parse_failure") == "1"
                  or rows[k].get("sentiment_no_consensus") == "1") / n
        print(f"{name}: domain parse failures {pf:.1%}, sentiment failures {spf:.1%}")
    print("Two runs of the same model at one vote differ a little by chance (about 1 to 3 in 100 "
          "in earlier trials), so a gap of that size is not evidence against the batch route.")
    return 0


# ---------------------------------------------------------------- command line
def add_common(parser):
    parser.add_argument("--judge", default="anthropic:claude-haiku-4-5",
                        help="the model, as anthropic:<model>")
    parser.add_argument("--sentiment", default=None,
                        help="default llm:<the same model as --judge>")
    parser.add_argument("--votes", type=int, default=1)
    parser.add_argument("--ttl", choices=["5m", "1h"], default="1h",
                        help="prompt-cache lifetime on the batch requests. 1h (default) is what "
                             "Anthropic suggests for batches; 5m is the live default. Cache "
                             "writes cost 2x the input price at 1h, 1.25x at 5m.")
    parser.add_argument("--cache", default=None,
                        help="response cache (default <out-dir>/batch_cache.jsonl). Use a "
                             "SEPARATE file from any live run.")
    parser.add_argument("--sample", type=int, metavar="N",
                        help="only N comments, spread evenly across stories (use this, not "
                             "--limit, on the corpus file)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, help="only the first N comments (one story, on the corpus)")
    parser.add_argument("--max-batch-requests", type=int, default=DEFAULT_MAX_REQUESTS)
    parser.add_argument("--max-batch-mb", type=float, default=DEFAULT_MAX_MB)
    parser.add_argument("--retry-uncacheable", action="store_true",
                        help="send again the replies that came back refused or cut off (by "
                             "default they are used as the live path uses them, not re-bought)")
    instrument.add_args(parser)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="dry run: counts and an estimated cost; sends nothing")
    p.add_argument("input_csv")
    p.add_argument("--out-dir", default=None)
    add_common(p)

    p = sub.add_parser("submit", help="send the missing requests as batches")
    p.add_argument("input_csv")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-cost", type=float, default=None, metavar="USD",
                   help="refuse to send anything if the estimated BATCH cost (the pessimistic end "
                        "of the range `plan` prints) is over USD. Checked before any batch, "
                        "manifest or subset file is written. Default: no cap.")
    add_common(p)

    p = sub.add_parser("status", help="progress of the batches; --check-cache counts gaps")
    p.add_argument("out_dir")
    p.add_argument("--check-cache", action="store_true")

    p = sub.add_parser("fetch", help="download finished batches into the cache")
    p.add_argument("out_dir")
    p.add_argument("--refetch", action="store_true", help="fetch batches already fetched")
    p.add_argument("--abandon", metavar="BATCH_ID",
                   help="give up on a batch that can never be fetched (its results are gone): "
                        "its requests become gaps that the next submit sends again. Only a "
                        "batch that has ended (or no longer exists)")
    p.add_argument("--cancel", action="store_true",
                   help="with --abandon on a batch still running: cancel it instead")

    p = sub.add_parser("replay", help="label the corpus from the cache alone")
    p.add_argument("out_dir")
    p.add_argument("--output", required=True)
    p.add_argument("--partial", action="store_true",
                   help="write the complete comments even if some are missing a reply")

    p = sub.add_parser("compare", help="how far two labelled files agree, comment by comment")
    p.add_argument("file_a")
    p.add_argument("file_b")

    args = parser.parse_args(argv)
    if args.command in ("plan", "submit"):
        if args.sentiment is None:
            args.sentiment = "llm:" + args.judge
        if args.votes < 1:
            parser.error("--votes must be at least 1")
        if getattr(args, "max_cost", None) is not None and not args.max_cost >= 0:
            parser.error("--max-cost must be zero or more (USD).")
        return cmd_submit(args, dry_run=(args.command == "plan"))
    return {"status": cmd_status, "fetch": cmd_fetch, "replay": cmd_replay,
            "compare": cmd_compare}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
