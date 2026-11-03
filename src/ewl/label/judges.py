"""Pluggable judges: one interface, many models behind it.

    from ewl.label.judges import create
    judge = create("ollama:gemma3:12b")
    judge = create("anthropic:claude-haiku-4-5")
    judge = create("keyword")

    labels, unstable, ballots = judge.label("They bought the whole election.", examples)

The point of this module is that the scorecard in #11 compares arms that differ
in exactly one respect. If each backend brought its own prompt, its own voting
rule and its own parser, a difference in the results would not be attributable
to the model. So the base class owns everything except the single call that
produces one raw reply, and each backend supplies only that.

**Adding a backend is one method.** Subclass `Judge`, implement `_sample`,
decorate with `@register("name")`. Voting, parsing, caching, usage accounting
and the prompt are inherited and identical across arms.

Specs are "<backend>:<model>". Everything after the first colon is the model,
so `ollama:gemma3:12b` works.
"""
import collections
import concurrent.futures
import hashlib
import json
import os
import re
import random
import threading
import time

import sys

from ewl import taxonomy

_REGISTRY = {}


def register(name):
    def wrap(cls):
        _REGISTRY[name] = cls
        cls.backend = name
        return cls
    return wrap


def _load_optional_backends():
    """Backends that live in their own module register themselves on import. Imported
    on demand, not at start-up, so `import ewl.label.judges` never needs the ollama
    package."""
    from ewl.label import gemma_local  # noqa: F401  (registers "gemma")


def create(spec, **kwargs):
    """Build a judge from a "<backend>:<model>" spec."""
    backend, _, model = spec.partition(":")
    if backend not in _REGISTRY:
        _load_optional_backends()
    if backend not in _REGISTRY:
        raise ValueError(f"Unknown backend {backend!r}. Available: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[backend](model or None, **kwargs)


def available():
    _load_optional_backends()
    return sorted(_REGISTRY)


class CacheMiss(RuntimeError):
    """A judge in cache-only mode was asked for a reply that is not in the cache.

    Raised instead of making a live call, so a replay of batch-collected answers can never
    quietly fall back to full-price API calls for whatever is missing."""

    def __init__(self, key):
        super().__init__(f"reply not in the cache (key {str(key)[:12]}...)")
        self.key = key


class UncacheableReply(str):
    """A reply that must be used for this call but never written to the cache.

    The cache replays whatever it holds, forever, without calling the API. A
    reply that stopped early (token limit, safety refusal, a blocked response)
    is not an answer, and caching it would make every later run -- including one
    with a bigger token budget -- replay the failure for free. The text is
    returned (usually empty, so it parses as a parse failure and is counted as
    one); `reason` says why it was not stored.
    """
    reason = "uncacheable"


def uncacheable(text, reason):
    reply = UncacheableReply(text)
    reply.reason = reason
    return reply


class ThinkingUnavailable(RuntimeError):
    """The model would not accept any way of turning thinking OFF. Sending no `thinking` field
    would run it with thinking ON: a different instrument from every other arm, whose answers
    get cut off at the reply limit. So the run stops unless a thinking-on arm was chosen."""


class ExternalProcessorNotApproved(RuntimeError):
    """Comment text would be sent to a provider that has not been approved for this project.
    Raised before any request is made. (A RuntimeError, not SystemExit, so a scorecard skips
    that one arm with this message instead of ending the whole run.)"""


class PriceUnavailable(RuntimeError):
    """A model has no price on record (or the price variable is malformed). Raised when the
    judge is built, before any comment is sent."""


class SettingRefused(RuntimeError):
    """A setting asked for through the environment cannot be used with this model (for example a
    reasoning effort on a model that does not reason). Raised when the judge is built, before
    anything is sent; a scorecard skips that arm with this message and `ewl-classify` exits with it."""


def require_external_processor(name):
    """Refuse to send comment text to provider `name` unless it has been approved.

    The gold set and the corpus are unpublished comment data, and each provider that receives
    them is a further external processor needing data-protection sign-off (#100, C3). Nothing
    reaches OpenAI or Google unless the environment variable EWL_EXTERNAL_PROCESSORS lists the
    provider (comma-separated, for example "openai,google"), to be set only once the sign-off is
    recorded. A local model (Ollama, Gemma) sends nothing anywhere and is not gated."""
    approved = {part.strip().lower()
                for part in os.environ.get("EWL_EXTERNAL_PROCESSORS", "").split(",") if part.strip()}
    if name.lower() not in approved:
        raise ExternalProcessorNotApproved(
            f"Running this model sends comment text to {name}, which has not been approved as an "
            f"external processor for this project (data-protection sign-off, #100 C3). Nothing "
            f"was sent. Once the sign-off is recorded, set EWL_EXTERNAL_PROCESSORS={name} "
            f"(several providers: comma-separated, for example openai,google).")


class ReasoningEffortRejected(RuntimeError):
    """The model refused the `reasoning_effort` setting the run sent. The setting changes how
    the model answers, so the run stops with the fix named; it never carries on with another
    setting. (The text contains "invalid_request" so the retry logic treats it as permanent.)"""


class TruncatedReply(RuntimeError):
    """The model stopped before writing a usable answer (e.g. its token budget went
    on reasoning). Retried like a transient failure; if it keeps happening the
    comment gets an uncacheable empty reply instead of aborting the run."""


class Judge:
    """Base judge. Subclasses implement `_sample` and nothing else.

    `votes` samples the model N times and keeps labels a majority agree on.
    That is what makes a stochastic backend reproducible enough to quote, and
    `unstable` reports when the votes disagreed so the rate can be published
    alongside the labels rather than discovered by a referee.
    """

    # Whether repeated runs of this backend can be expected to agree exactly.
    # Reported in the scorecard, because an accuracy win that costs
    # reproducibility is a trade the write-up has to make consciously.
    deterministic = False

    def __init__(self, model=None, votes=3, fuzzy=False, cache_path=None):
        self.model = model or self.default_model
        self.votes = max(1, votes)
        self.fuzzy = fuzzy
        self.usage = collections.Counter()   # input/output tokens, calls, cache_hits
        self.seconds = 0.0
        self.slept = 0.0                     # retry backoff: waiting, not modelling
        self._cache = _Cache.shared(cache_path) if cache_path else None
        # Guards `usage`, `seconds`, and the cache. classify.py can dispatch
        # comments across a thread pool (--concurrency), and Counter increments
        # and a shared file handle are not atomic; without this the token
        # totals and the cost figure would quietly under-report.
        self._lock = threading.Lock()

    @property
    def cache_spec(self):
        """Fully identifies what produced a reply, for the cache key.

        Distinct from `spec`, which is shortened for display. Two vocabularies
        with the same basename -- data/v1/keywords.yml and data/v2/keywords.yml
        -- have the same `spec` and must not share cache entries, or one run is
        silently served the other's labels.
        """
        return f"{self.backend}:{self.model}" if self.model else self.backend

    @property
    def spec(self):
        """Identifies the arm, and goes in the scorecard and in domain_judge.

        The keyword judge's model is now an absolute path to the packaged
        vocabulary, which turned every scorecard row into
        `keyword:/Users/.../src/ewl/keywords.yml/none`. The file name carries
        all the information (which vocabulary), the directory none of it, so
        only the name is shown -- an explicit --keywords still shows its own
        name, which is the case where the distinction matters.
        """
        if not self.model:
            return self.backend
        model = self.model
        if os.sep in str(model) or "/" in str(model):
            model = os.path.basename(str(model))
        return f"{self.backend}:{model}"

    def _sample(self, body, prompt, vote):
        """Return one raw reply for `body`. The only method a backend defines.

        `vote` is the 0-based sample index, so backends with a seed can vary it
        and still be reproducible.
        """
        raise NotImplementedError

    def sample(self, body, prompt, vote=0):
        """One raw reply to an arbitrary prompt.

        Exposed so other labelling tasks can reuse these backends, and with
        them the response cache, the usage accounting and the cost model,
        rather than growing a second parallel set of model wrappers. See
        `sentiment.py`, which scores sentiment through the same judges that
        assign harm domains.
        """
        return self._cached_sample(body, prompt, vote)

    def label(self, body, examples, story=None):
        """Return (labels, unstable, ballots) for one comment.

        `ballots` -- the individual per-vote label sets, before majority
        resolution -- used to come back nowhere; only the collapsed
        (labels, unstable) survived. Needed to compare the LLM's own
        self-consistency across votes against human coder-vs-coder agreement
        on the same statistical footing (kappa across votes vs. kappa across
        coders) -- see validate.py's cmd_selfconsistency, the only consumer
        that needs the raw ballots rather than just the resolved labels.

        `story` selects the factual background paragraph that follows the
        examples (taxonomy.build_prompt; "facts" by default, switched off with
        --domain-context none). The comment is still classified on what IT is
        about, not the story: the paragraph is background, as it was for the
        hand coders.
        """
        prompt = taxonomy.build_prompt(examples, story)
        ballots = []
        for vote in range(self.votes):
            ballots.append(taxonomy.parse_labels(self._cached_sample(body, prompt, vote), self.fuzzy))

        labels, unstable = self.resolve(ballots)
        return labels, unstable, ballots

    def resolve(self, ballots):
        """Majority vote over per-sample label sets -> (labels, unstable).

        Extracted so any judge producing its own ballots -- combined.py parses
        the domain out of a reply that also carries the sentiment -- applies
        exactly this rule rather than a second copy that could drift.
        """
        if len(ballots) == 1:
            return ballots[0], False

        tally = collections.Counter(label for ballot in ballots for label in ballot)
        threshold = len(ballots) // 2 + 1
        agreed = [label for label, count in tally.most_common() if count >= threshold][:2]
        unstable = any(set(b) != set(ballots[0]) for b in ballots[1:])
        # No majority is not a finding. It used to return UNCLASSIFIED, which
        # asserts "the model read this and judged it engages none of the four
        # harms" -- manufactured out of three votes that simply disagreed. At
        # --votes 3 that turns every split decision into a substantive claim,
        # and it hides the comment from the parse-failure rate as well.
        return (agreed or [taxonomy.PARSE_FAILURE]), unstable

    # When True, a reply that is not in the cache raises CacheMiss instead of calling the
    # model. Set by `ewl-classify --cache-only` and by the batch replay.
    cache_only = False
    # {cache key: (reason, reply text)} for replies the batch route got back as a refusal or
    # cut off at the token limit (anthropic_batch.py). Empty everywhere else.
    known_uncacheable = None

    def _cache_key(self, body, prompt, vote):
        """The response-cache key: (spec, prompt, comment, vote), hashed. One definition,
        used by the live path and by the batch planner, so a batch answer is replayed under
        exactly the key a live call would have written."""
        return hashlib.sha256(
            "\x00".join([self.cache_spec, prompt, body, str(vote)]).encode("utf-8")).hexdigest()

    def _cached_sample(self, body, prompt, vote):
        """Cache on (spec, prompt, comment, vote).

        The prompt is part of the key because retrieved examples differ per
        comment: without it, switching the example selector would silently
        serve labels produced under the old one. A cached run costs nothing and
        reproduces exactly, which is how an API arm stays re-runnable even
        though the endpoint behind it is not frozen.
        """
        if self._cache is None:
            if self.cache_only:
                raise CacheMiss(None)
            return self._timed_sample(body, prompt, vote)
        # Whether or not a reply is cached, an uncacheable one is counted (in
        # _timed_sample), so the run summary can report it.
        key = self._cache_key(body, prompt, vote)
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self.usage["cache_hits"] += 1
                return hit
        known = self.known_uncacheable.get(key) if self.known_uncacheable else None
        if known is not None:
            # A refusal or cut-off reply collected by ewl-anthropic-batch: used for this call
            # exactly as the live path uses one (refusal -> parse failure, a cut-off reply's
            # partial text as written), never cached, and not asked for again.
            reason, text = known
            with self._lock:
                self.usage[f"uncacheable_{reason}"] += 1
                self.usage["uncacheable_replayed"] += 1
            return uncacheable(text, reason)
        if self.cache_only:
            raise CacheMiss(key)
        # Sampled outside the lock: the whole point of concurrency is that the slow part -- the
        # network call -- overlaps. But only ONE call per key at a time: a second worker that
        # wants the same (spec, prompt, comment, vote) while the first is still waiting on the
        # model waits for that reply instead of calling again. Without this, duplicate comments
        # in a parallel run each got their own (different) reply while the cache kept only one,
        # so the labels returned did not match what the cache replays; a serial run would have
        # made the second one a cache hit. Now parallel, serial and replayed runs agree.
        with self._lock:
            # Look at the cache again under the SAME lock that registers the call: a worker that
            # finished (cache written) between the first look and this one would otherwise be
            # missed, and this worker would call again and return a different reply.
            hit = self._cache.get(key)
            if hit is not None:
                self.usage["cache_hits"] += 1
                return hit
            inflight = self.__dict__.setdefault("_inflight_calls", {})
            pending = inflight.get(key)
            owner = pending is None
            if owner:
                pending = concurrent.futures.Future()
                inflight[key] = pending
        if not owner:
            reply = pending.result()           # raises what the first caller raised
            with self._lock:
                self.usage["shared_inflight"] += 1
            return reply
        try:
            reply = self._timed_sample(body, prompt, vote)
            if not isinstance(reply, UncacheableReply):
                with self._lock:
                    self._cache.put(key, reply)
            pending.set_result(reply)
            return reply
        except BaseException as error:
            pending.set_exception(error)
            raise
        finally:
            with self._lock:
                self.__dict__.get("_inflight_calls", {}).pop(key, None)
            # Drop this frame's reference to the Future: when a call failed, the Future holds the
            # exception, whose traceback holds this frame, and the cycle would keep every open
            # file in the calling frames alive until the garbage collector ran (on Windows that
            # blocks deleting a half-written output file).
            pending = None

    # A long unattended run meets transient failures: rate limits, gateway
    # errors, a dropped connection. Retrying is the difference between losing
    # an overnight run at hour three and pausing for a few seconds.
    max_attempts = 5
    # A truncated or empty reply is retried too, but each retry is a full-price call that is
    # never cached (and a comment that truncates once often truncates again), so it gets fewer
    # tries than a transient network failure: at most this many retries, then the comment is
    # given an uncacheable empty reply (a parse failure) and the run goes on.
    max_truncation_retries = 2
    # Substrings identifying a failure worth retrying. Matched on the text
    # rather than on exception classes so this works across every backend
    # without judges.py importing any of their SDKs.
    RETRYABLE = ("rate_limit", "429", "overloaded", "529", "500", "502", "503", "504",
                 "timeout", "timed out", "connection", "temporarily unavailable",
                 # Not an HTTP/network failure -- a reply that came back truncated
                 # or empty because the model spent its whole token budget on
                 # invisible reasoning before writing an answer (see OpenAIJudge
                 # and GeminiJudge's _sample). Retrying is the right response: a
                 # fresh attempt is a fresh, independent draw, not a repeat of the
                 # same failure, and the alternative -- caching the empty reply as
                 # a valid answer -- is worse than paying for a retry.
                 "truncated", "empty reply")
    # Checked FIRST, and never retried. Substring matching on the whole error
    # text is loose enough that a permanent failure can contain a transient
    # word -- "model claude-429 not found", "your connection string is
    # invalid", "request timeout policy forbids this" all matched RETRYABLE.
    # A wrong API key would then burn five attempts and ~90s of backoff on
    # every comment before failing, turning an instant, obvious error into a
    # very slow one.
    PERMANENT = ("authentication", "invalid_request", "invalid api key",
                 "permission", "not_found", "not found", "unauthorized",
                 "forbidden", "insufficient_quota", "credit balance")

    def _is_retryable(self, error):
        text = f"{type(error).__name__} {error}".lower()
        if any(marker in text for marker in self.PERMANENT):
            return False
        # A bare status code ("500") must not match inside a longer number: "model
        # limit 4500 tokens" is not a server error.
        return any(re.search(rf"(?<!\d){re.escape(marker)}(?!\d)", text)
                   if marker.isdigit() else marker in text
                   for marker in self.RETRYABLE)

    def _with_retries(self, call, label):
        """Retry a transient failure with exponential backoff and jitter.

        Jitter matters with --concurrency: without it, N workers rate-limited at
        the same moment all sleep the same interval and retry in lockstep,
        reproducing the burst that caused the limit.
        """
        truncation_retries = 0
        for attempt in range(1, self.max_attempts + 1):
            try:
                with self._lock:
                    # Real requests, as distinct from usage["calls"], which
                    # counts one per (comment, vote) however many attempts it
                    # took. A run that limped through retries should not look
                    # identical to one that went straight through.
                    self.usage["attempts"] += 1
                return call()
            except Exception as error:
                if attempt == self.max_attempts or not self._is_retryable(error):
                    raise
                if isinstance(error, TruncatedReply):
                    if truncation_retries >= self.max_truncation_retries:
                        raise
                    truncation_retries += 1
                delay = min(2 ** (attempt - 1), 30) * (1 + random.random())
                with self._lock:
                    self.usage["retries"] += 1
                    self.slept += delay
                    tls = self.__dict__.setdefault("_tls", threading.local())
                tls.slept = getattr(tls, "slept", 0.0) + delay
                print(f"  {label}: {type(error).__name__} on attempt {attempt}/"
                      f"{self.max_attempts}, retrying in {delay:.1f}s", file=sys.stderr)
                time.sleep(delay)

    def _thread_slept(self):
        """Backoff this THREAD has slept so far. `self.slept` is shared by every worker, so
        reading it around a call would subtract other workers' sleeps from this call's time."""
        tls = self.__dict__.get("_tls")
        if tls is None:
            with self._lock:
                tls = self.__dict__.setdefault("_tls", threading.local())
        return getattr(tls, "slept", 0.0)

    def _timed_sample(self, body, prompt, vote):
        started = time.monotonic()
        slept_before = self._thread_slept()
        try:
            reply = self._with_retries(lambda: self._sample(body, prompt, vote), self.spec)
            if isinstance(reply, UncacheableReply):
                with self._lock:
                    self.usage[f"uncacheable_{reply.reason}"] += 1
            return reply
        except TruncatedReply:
            # Retries exhausted. One comment the model keeps truncating must not
            # abort an 84,000-comment run (or throw away a whole scorecard arm):
            # give it an empty reply, which parses as a parse failure and is
            # counted, and do not cache it.
            with self._lock:
                self.usage["uncacheable_truncated"] += 1
            return uncacheable("", "truncated")
        finally:
            with self._lock:
                # Sum of time spent in calls, so with concurrency > 1 this
                # exceeds wall-clock. Reported as "model time", not elapsed --
                # and backoff sleeps are subtracted, since up to ~90s of
                # waiting per comment billed as model time would make the
                # scorecard's `mins` column describe the network rather than
                # the run.
                spent = time.monotonic() - started - (self._thread_slept() - slept_before)
                self.seconds += spent
                self.usage["calls"] += 1
                # One entry per live call, for a median: robust to a few stalls (a computer
                # that slept mid-run, a network drop), which a wall-clock total is not.
                latencies = self.__dict__.setdefault("latencies", [])
                if len(latencies) < 50000:
                    latencies.append(spent)

    # Anthropic's published multipliers on the base input rate: a cache WRITE
    # costs slightly more than an ordinary input token (the model has to
    # process it once to store it), a cache READ costs a small fraction of
    # one (it skips reprocessing the prefix entirely). Applied uniformly here
    # rather than per-backend, since any backend that reports these usage
    # fields (currently only AnthropicJudge -- see its _sample) means the
    # same thing by them.
    CACHE_WRITE_MULTIPLIER = 1.25
    CACHE_READ_MULTIPLIER = 0.1

    def cost_usd(self):
        """Estimated spend. Zero for anything running locally.

        `input_tokens` alone understates cost once prompt caching is in play:
        Anthropic bills cache-written and cache-read tokens separately from
        plain input tokens, at their own rates, and `response.usage.input_tokens`
        does not include them. Ignoring `cache_creation_input_tokens` /
        `cache_read_input_tokens` here would make a caching-enabled run look
        cheaper than it is by simply not counting a real fraction of the
        spend, rather than by actually saving anything.
        """
        rate_in, rate_out = self.price_per_mtok
        return (self.usage["input_tokens"] / 1e6 * rate_in
                + self.usage["output_tokens"] / 1e6 * rate_out
                + self.usage["cache_creation_input_tokens"] / 1e6 * rate_in * self.CACHE_WRITE_MULTIPLIER
                + self.usage["cache_read_input_tokens"] / 1e6 * rate_in * self.CACHE_READ_MULTIPLIER)

    def cache_tokens(self):
        """(cache_creation_input_tokens, cache_read_input_tokens). Same shape
        as sentiment.Scorer.cache_tokens(), so classify.py can report on a
        judge and a scorer uniformly regardless of which one actually
        supports caching."""
        return (self.usage["cache_creation_input_tokens"], self.usage["cache_read_input_tokens"])

    def token_usage(self):
        """(input_tokens, output_tokens) -- the plain, non-cache counts
        cost_usd() is built from. Same shape as sentiment.Scorer.token_usage(),
        added so the scorecard can report raw token counts, not just the
        derived dollar figure -- evaluate.run_arm/sentiment_report read this
        as a before/after delta the same way they already do for seconds."""
        return (self.usage["input_tokens"], self.usage["output_tokens"])

    def cache_savings_usd(self):
        """How much `cache_read_input_tokens` actually saved vs. paying full
        input price for the same tokens -- the figure worth reporting
        alongside cost, since `cost_usd` alone doesn't say whether caching is
        doing anything (a run with a 0% cache hit rate and a run saving 40%
        can report similar-looking total costs for unrelated reasons)."""
        rate_in, _ = self.price_per_mtok
        read = self.usage["cache_read_input_tokens"]
        return read / 1e6 * rate_in * (1 - self.CACHE_READ_MULTIPLIER)

    default_model = None
    price_per_mtok = (0.0, 0.0)


_cache_registry = {}
_cache_registry_lock = threading.Lock()


class _Cache:
    """Append-only JSONL cache. Deliberately dumb: greppable, diffable, and
    safe to delete.

    One instance per PATH is shared process-wide -- see `shared()` below, the
    only way one of these should now be constructed. classify.py passes the
    SAME --cache path to both the domain judge and the sentiment LLMScorer's
    inner judge (they are two separate Judge objects), and each Judge's own
    `self._lock` only ever protected that one instance's view of the file --
    it gave no mutual exclusion between the two. Confirmed by measurement:
    two Judge instances pointing at one cache file, 8 threads each making
    ~300 calls, silently lost about 1% of entries to the file with no error
    and no corruption -- just missing rows. That is a real problem here
    specifically, since the response cache is this project's only
    reproducibility mechanism for an API arm now that `temperature` is not
    settable (see the determinism note above `_ANTHROPIC_PRICES`): a lost
    entry means a "reproducible" re-run quietly makes a fresh, possibly
    different API call for that one comment instead. Sharing the instance
    (and moving the lock inside it, rather than trusting each caller's own
    lock) closes this: every Judge that names the same path now serialises
    through the one lock that actually guards that file.
    """

    def __init__(self, path):
        self.path = path
        self.entries = {}
        self._lock = threading.Lock()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a truncated final line from an interrupted run
                    self.entries[record["key"]] = record["reply"]
        # A run killed in the middle of a write leaves a last line with no newline. The next
        # `put` would append its record onto that fragment, making one unreadable line, so the
        # first reply saved after a resume would be skipped on the following load (and asked
        # of the model again). Start the next record on a line of its own instead.
        self._needs_newline = False
        if os.path.exists(path) and os.path.getsize(path):
            with open(path, "rb") as handle:
                handle.seek(-1, os.SEEK_END)
                self._needs_newline = handle.read(1) != b"\n"

    @classmethod
    def shared(cls, path):
        """The one _Cache for this path in this process. Always use this,
        never _Cache(path) directly -- see the class docstring for why."""
        path = str(path)
        with _cache_registry_lock:
            cache = _cache_registry.get(path)
            if cache is None:
                cache = cls(path)
                _cache_registry[path] = cache
            return cache

    def get(self, key):
        with self._lock:
            return self.entries.get(key)

    def empty_replies(self):
        """How many stored replies are empty. An empty reply is the signature of a
        truncated or refused answer stored before early-stopped replies stopped being
        cached (#82): replaying it for free would make a rerun repeat the failure."""
        with self._lock:
            return sum(1 for reply in self.entries.values() if reply == "")

    def put(self, key, reply):
        with self._lock:
            # Another thread (this instance or, now that instances are
            # shared by path, formerly a *different* instance on the same
            # file) may have already written this exact key while this
            # thread was making its own (identical, wasted but harmless --
            # see _cached_sample) call. Skip the write rather than appending
            # a second, redundant line for the same key.
            if key in self.entries:
                return
            self.entries[key] = reply
            with open(self.path, "a", encoding="utf-8") as handle:
                if self._needs_newline:
                    handle.write("\n")
                    self._needs_newline = False
                handle.write(json.dumps({"key": key, "reply": reply}) + "\n")


@register("keyword")
class KeywordJudge(Judge):
    """The rule-based classifier, wrapped in the judge interface.

    Not a real use of the interface so much as the floor of the scorecard: if
    the model arms cannot beat a keyword list, the LLM approach needs
    rethinking, and that is worth knowing before spending anything on tokens.
    Ignores the examples — it has no prompt.
    """

    deterministic = True
    # Resolved from the package rather than the working directory.
    default_model = None

    def __init__(self, model=None, **kwargs):
        super().__init__(model, **kwargs)
        from ewl.label import keywords as keyword_classify
        if not self.model:
            self.model = str(keyword_classify.DEFAULT_KEYWORDS)
        self._patterns = keyword_classify.load_keywords(self.model)
        self._classify = keyword_classify.classify
        self._unclassified = keyword_classify.UNCLASSIFIED
        self.votes = 1  # voting over a deterministic function is a waste

    def _sample(self, body, prompt, vote):
        labels, _evidence = self._classify(body, self._patterns)
        # keyword_classify.UNCLASSIFIED is the literal string "unclassified",
        # but taxonomy.parse_labels() -- which every reply from every judge is
        # read through -- only recognises the word "none" as a clean, confident
        # null; anything else off-list (including "unclassified" itself) sets
        # saw_garbage and returns PARSE_FAILURE. Confirmed on the real 84,250-
        # comment corpus: every genuine "matches none of the four categories"
        # verdict from this classifier was being recorded as a PARSE_FAILURE
        # (58.3% of the corpus) rather than UNCLASSIFIED (0%) -- exactly the
        # "classifier reliability problem, not a finding about the comments"
        # false alarm PARSE_FAILURE exists to distinguish from a genuine null,
        # and it silently excluded every one of those rows from
        # report._gold_tpr_fpr (which drops parse failures rather than
        # counting them as misses) whenever this judge is scored.
        if labels == [self._unclassified]:
            return "none"
        return ", ".join(labels)


@register("ollama")
class OllamaJudge(Judge):
    """Local model through Ollama. Free, private, and seed-stable."""

    deterministic = True
    default_model = "gemma3:12b"

    def __init__(self, model=None, host=None, **kwargs):
        super().__init__(model, **kwargs)
        import ollama
        self._client = ollama.Client(host=host) if host else ollama.Client()

    def _sample(self, body, prompt, vote):
        response = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f'Classify this comment: "{body}"'},
            ],
            # Fixed seed per vote index: reproducible across runs, while still
            # giving the votes something to disagree about.
            options={"temperature": 0, "seed": 42 + vote, "num_predict": 16},
        )
        with self._lock:
            self.usage["input_tokens"] += response.get("prompt_eval_count", 0) or 0
            self.usage["output_tokens"] += response.get("eval_count", 0) or 0
        return response["message"]["content"]


# Determinism note. Earlier drafts of this module pinned temperature=0 for the
# API judges. That is no longer possible: `temperature` was removed from the
# Messages API for the current Claude models, and anthropic 1.x drops it from
# the SDK signature altogether, so there is no parameter to set on any model.
# Reproducibility for an API arm therefore rests entirely on the response cache
# in `Judge._cached_sample` — which is why `deterministic` is False here and why
# the scorecard reports it as a column. If an API arm's labels are quoted in the
# write-up, its cache file is a required artefact, not a convenience.
#
# Anthropic list prices, USD per million tokens. Update alongside the model.
_ANTHROPIC_PRICES = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),  # was (3.00, 15.00) -- stale; corrected against
                                       # Anthropic's current published pricing, 2026-09.
    "claude-opus-5": (5.00, 25.00),
    # From Anthropic's batch-pricing table (2026-10-13; batch price is half of list):
    # Opus 5.5 batch $2/$10 -> list $4/$20, Sonnet 5.5 batch $1/$5 -> list $2/$10. Without an
    # entry, "claude-opus-5-5" matched the Opus 5 prefix and was priced at $5/$25.
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-fable-5": (10.00, 50.00),
}


_DATED_SNAPSHOT = re.compile(r"(?P<base>.+)-\d{8}")


def _anthropic_price_overrides():
    return _price_overrides("EWL_ANTHROPIC_PRICES", "claude-model")


def price_for(model):
    """List price for `model`, or a loud refusal (PriceUnavailable).

    Module-level rather than only a property, so it is testable without
    constructing an AnthropicJudge -- which imports the SDK, and CI does not
    install it.

    The old fallback was (0.0, 0.0) for anything absent, which is the worst
    possible default: an unpriced model reports the run as free.
    `claude-fable-5` was missing and is the most expensive model in the
    roster, and classify.py gates its whole summary on `if total_cost:` -- so
    at zero cost the Spend line AND the Prompt cache line both vanish, and a
    trial would report neither its cost nor whether caching engaged.

    A dated snapshot id (claude-opus-5-20260115: the base id and exactly eight digits) resolves
    to its base model, since pricing does not change with the snapshot. Nothing else is matched
    by prefix: a new model such as claude-sonnet-5-7 used to be silently priced as Sonnet 5.
    It must be in the table, or have its price supplied for the run with
    EWL_ANTHROPIC_PRICES="model=input,output" (USD per million tokens, from Anthropic's page).
    """
    supplied = _anthropic_price_overrides()   # also validates the variable, even for a known model
    if model in _ANTHROPIC_PRICES:
        if model in supplied:
            _note_table_price_once("EWL_ANTHROPIC_PRICES", model)
        return _ANTHROPIC_PRICES[model]
    if model in supplied:
        return supplied[model][:2]
    snapshot = _DATED_SNAPSHOT.fullmatch(model or "")
    if snapshot:
        base = snapshot.group("base")
        if base in _ANTHROPIC_PRICES:
            return _ANTHROPIC_PRICES[base]
        if base in supplied:
            return supplied[base][:2]
    raise PriceUnavailable(
        f"No price on record for Anthropic model {model!r}, and refusing to assume it "
        f"is free (or to price it as a similarly named model) -- an unpriced run reports "
        f"$0.00, and classify.py gates its summary on a non-zero cost, so the spend and cache "
        f"lines both disappear. Add it to _ANTHROPIC_PRICES in judges.py, or supply it for this "
        f"run with EWL_ANTHROPIC_PRICES=\"{model}=<input>,<output>\" (USD per million tokens, "
        f"from Anthropic's pricing page). Known: {', '.join(sorted(_ANTHROPIC_PRICES))}.")


# Anthropic's minimum cacheable prefix length, in tokens -- published, and
# confirmed to matter here: a combined system prompt under this size does not
# cache at all (no error, no cost change from an uncached call -- see
# AnthropicJudge._cacheable_system). Haiku's minimum is the highest of any
# current model, and Haiku is also this project's default judge.
_CACHE_MINIMUM_TOKENS = {
    "claude-haiku-4-5": 4096,
    "claude-sonnet-5": 1024,
    "claude-opus-5": 512,
    "claude-fable-5": 512,
}


def cache_minimum_for(model):
    """Anthropic's minimum cacheable prefix length for `model`, in tokens, or
    None if unknown (an unrecognised model is not this function's job to
    refuse -- price_for already does that where it matters)."""
    if model in _CACHE_MINIMUM_TOKENS:
        return _CACHE_MINIMUM_TOKENS[model]
    for base, minimum in _CACHE_MINIMUM_TOKENS.items():
        if model.startswith(base + "-"):
            return minimum
    return None


def estimate_tokens(text):
    """A rough, offline estimate -- about 4 characters per token for English
    prose, no API call and no tokenizer dependency required.

    Deliberately approximate: the point is to catch a prompt that is
    obviously, drastically short of the cache minimum (a few hundred tokens
    against Haiku's 4096) before spending anything, not to replace an exact
    check. `client.messages.count_tokens` against the real judge and pool is
    still the authoritative source once one exists to check (see CLAUDE.md
    §7) -- this just means the gap is visible with no setup at all, rather
    than only discoverable by remembering to run that check by hand.
    """
    return len(text) // 4


# An estimate this far above a model's minimum is still treated as borderline.
# `estimate_tokens` assumes 4 characters per token; real prose runs from roughly
# 3.5 to 4.7, so a prompt whose estimate is 1,100 tokens can genuinely be 935.
# 1.2x covers that spread: at 4.7 characters per token a true count is 0.85x the
# estimate, and 1 / 0.85 is about 1.18.
CACHE_BORDERLINE_FACTOR = 1.2


def cache_report(model, prompts):
    """[(label, estimated_tokens, minimum, verdict)] for each named prompt.

    `verdict` is "below" (estimated under the model's cache minimum: it will
    not cache), "borderline" (estimated over it by less than the estimate's own
    error: it might not), or "ok". Empty if the model's minimum is unknown.

    A prompt under the minimum is not an error: nothing fails, and every call
    is billed at the full input price instead of the 0.1x cache-read price.
    That is why this is worth reporting up front. On 84,000 comments the gap
    is roughly 10x on one prompt's input tokens.
    """
    minimum = cache_minimum_for(model)
    if minimum is None:
        return []
    report = []
    for label, text in prompts.items():
        estimate = estimate_tokens(text)
        if estimate < minimum:
            verdict = "below"
        elif estimate < minimum * CACHE_BORDERLINE_FACTOR:
            verdict = "borderline"
        else:
            verdict = "ok"
        report.append((label, estimate, minimum, verdict))
    return report


@register("anthropic")
class AnthropicJudge(Judge):
    """Claude through the official SDK.

    Reads ANTHROPIC_API_KEY from the environment; never takes a key as an
    argument, so a key cannot reach a log line or the shell history.

    Defaults to Haiku: this is four-way classification of short comments, the
    cheapest current model is the right starting point, and the scorecard
    exists to say whether a larger one earns its cost.
    """

    deterministic = False
    default_model = "claude-haiku-4-5"

    def __init__(self, model=None, **kwargs):
        super().__init__(model, **kwargs)
        # An unpriced model (or a malformed EWL_ANTHROPIC_PRICES) is refused here, before
        # anything is sent, as the OpenAI and Gemini judges do.
        price_for(self.model)
        import anthropic
        # A call that gets no reply must not hold a worker for the SDK's default of ten minutes:
        # on a flaky connection a few percent of calls hang, and at concurrency 3 to 12 that
        # stalls the whole run. EWL_ANTHROPIC_TIMEOUT (seconds, default 60) abandons such a call,
        # and the retry layer below sends it again (and counts it). The SDK's own silent retries
        # are turned off so every attempt is visible in the run's attempt and retry counts.
        raw_timeout = os.environ.get("EWL_ANTHROPIC_TIMEOUT", "").strip()
        try:
            timeout = float(raw_timeout) if raw_timeout else 60.0
            if timeout <= 0:
                raise ValueError
        except ValueError:
            raise SettingRefused(f"EWL_ANTHROPIC_TIMEOUT={raw_timeout!r} is not a number of seconds "
                                 f"above zero.") from None
        self.request_timeout = timeout
        self._client = anthropic.Anthropic(timeout=timeout, max_retries=0)
        self._sdk = anthropic
        # Opt-in, off by default (no change to any request): `low` keeps a model that
        # reasons by default from spending its answer budget on thinking (#82). Set
        # EWL_ANTHROPIC_EFFORT=low|medium|high. Sent as an extra body field, so an
        # older SDK does not reject it; NOT confirmed against a live call, so run the
        # 20-comment trial before relying on it. It is part of the cache key.
        effort = os.environ.get("EWL_ANTHROPIC_EFFORT", "").strip().lower()
        if effort and effort not in ("low", "medium", "high"):
            raise SystemExit(f"EWL_ANTHROPIC_EFFORT={effort!r}: use low, medium or high "
                             f"(or leave it unset).")
        self.effort = effort or None
        # Opt-in, off by default: let a model that cannot have thinking switched off run with
        # thinking ON (no `thinking` field, a larger reply limit). That is a different
        # instrument, so its replies get their own cache keys and it must be disclosed.
        wanted = os.environ.get("EWL_ANTHROPIC_ALLOW_THINKING", "").strip().lower() in (
            "1", "true", "yes")
        # It means "let a model that cannot have thinking turned off run with thinking ON", so
        # it takes effect only for such a model: one already known not to accept an 'off'
        # setting, or one that turns out to refuse every 'off' spelling and so settles on
        # sending none. For any other model (Sonnet 5.5, Haiku, ...) it would change the reply
        # limit and the cache key while thinking stayed off, so it has no effect there, and the
        # run record shows the limit actually used.
        self._allow_thinking_requested = wanted

    @property
    def allow_thinking(self):
        """True only when this judge really is a thinking-ON arm (see __init__)."""
        if not getattr(self, "_allow_thinking_requested", False):
            return False
        return (self._thinking_cannot_be_off()
                or getattr(self, "_thinking_override", None) == self.THINKING_OMIT)

    @allow_thinking.setter
    def allow_thinking(self, value):
        self._allow_thinking_requested = bool(value)

    @property
    def reply_limit(self):
        """The `max_tokens` this judge sends (256, or 1,024 for a thinking-on arm), for the run
        record."""
        return 1024 if getattr(self, "allow_thinking", False) else 256

    @property
    def cache_spec(self):
        base = super().cache_spec
        effort = getattr(self, "effort", None)
        base = f"{base}|effort={effort}" if effort else base
        return f"{base}|thinking=allowed" if getattr(self, "allow_thinking", False) else base

    @property
    def CACHE_READ_MULTIPLIER(self):
        """Cache reads cost 0.1x the input price, but 0.05x on Opus 5.5 and Sonnet 5.5
        (Anthropic's prompt-caching page, 2026-10-13)."""
        return 0.05 if str(getattr(self, "model", "")).startswith(
            ("claude-opus-5-5", "claude-sonnet-5-5")) else 0.1

    @property
    def price_per_mtok(self):
        """See `price_for`."""
        return price_for(self.model)

    def _cacheable_system(self, prompt, ttl=None):
        """Wrap the whole system prompt as a single cache-annotated block.

        REVISED 2026-09: this used to split the prompt into two tiers -- the
        corpus-wide-constant `taxonomy.STABLE_PREFIX` and the per-comment
        retrieved examples -- on the theory that the stable part could be
        cached once for the entire run. Measured on a real trial run, that
        theory was wrong in a way that mattered: with the trial's small pool
        (k capped to 10), the stable prefix alone came to ~1,331 characters
        (~330 tokens) and the examples block ~1,300 characters (~325 tokens)
        -- both under Anthropic's minimum cacheable prefix length, so BOTH
        tiers silently failed to cache at all (confirmed: `Prompt cache: no
        reads or writes recorded` on every run). Worse, `STABLE_PREFIX` is a
        fixed ~330 tokens regardless of how large the real gold pool ends up
        being, so splitting it out would likely never clear the minimum even
        at full scale -- this was not just a trial-data artefact.

        Caching the whole prompt as one block roughly doubles the size in
        contention (~660 tokens combined on the trial pool), giving it more
        of a chance to clear the minimum -- but the minimum itself is
        STEEP and varies enormously by model, confirmed against Anthropic's
        published table: 512 tokens on Claude Opus 5 and the Fable/Mythos
        family, 1024 on Claude Sonnet 5, but a full **4096 tokens on Haiku
        4.5** -- the highest of any current model. A combined ~660-800 token
        prompt (this project's typical size with the small trial pool)
        cannot clear Haiku's minimum at all, and still falls short of
        Sonnet's; only a real, much larger gold pool (the actual ~400-row
        hand-coded set, not this trial's 11 rows) retrieving longer/more
        examples might push the prompt past 1024 (Sonnet) or 4096 (Haiku) --
        confirm with `client.messages.count_tokens` once that pool exists
        rather than assuming either way.

        Still captures the win that was directly measured to matter when it
        does clear the minimum: one comment's `--votes` calls send this
        exact same prompt 3 times over (the selector runs once per comment,
        not once per vote -- see Judge.label), so votes 2 and 3 can read it
        back cheaply instead of reprocessing it. A prompt too short to clear
        the minimum simply doesn't cache -- no error, no cost change from an
        uncached call, since a cache write that doesn't happen is just
        billed as a plain input token instead.
        """
        control = {"type": "ephemeral"}
        if ttl:
            control["ttl"] = ttl   # "1h": the cache lifetime a long batch needs (default is 5m)
        return [{"type": "text", "text": prompt, "cache_control": control}]

    # Models where omitting `thinking` entirely runs adaptive thinking ON by
    # default (confirmed live 2026-09: a 20-token budget spent 19 of them on
    # invisible thinking before the model could write an answer, hitting
    # max_tokens with nothing left). Haiku 4.5 is NOT in this set -- it
    # defaults to no thinking when the parameter is omitted, and explicitly
    # disabling is not a documented, safe request shape for its older-style
    # thinking config, so it is left alone rather than risk a 400.
    THINKING_ON_BY_DEFAULT = {"claude-sonnet-5", "claude-opus-5"}

    # How "no thinking" is spelled. Sonnet 5.5 rejects {"type": "disabled"} with a 400 (confirmed
    # live 2026-10-17: 'send "thinking": {"type": "between_tools"} instead'); the 5 generation
    # accepts "disabled". The 5.5 prefixes are listed first so a 5.5 id never matches a 5 base.
    # Opus 5.5 rejects {"type": "between_tools"} ("is not supported for this model", no hint,
    # confirmed live 2026-10-17), so it keeps the default spelling. If any model rejects the
    # spelling it is sent, `_sample` tries the next one (the type the 400 names if it names one,
    # otherwise each of THINKING_ALTERNATIVES in turn) and remembers what worked.
    THINKING_OFF_BY_PREFIX = (("claude-sonnet-5-5", {"type": "between_tools"}),)
    THINKING_OFF_DEFAULT = {"type": "disabled"}
    THINKING_OMIT = {"type": "omit"}      # a marker: send no `thinking` field at all
    THINKING_ALTERNATIVES = ({"type": "disabled"}, {"type": "between_tools"})
    # Models that rejected BOTH ways of turning thinking off in a live trial (Opus 5.5,
    # 2026-10-17: "disabled" and "between_tools" both refused). Sending nothing would run them
    # with thinking ON, so they stop with a clear message unless EWL_ANTHROPIC_ALLOW_THINKING=1.
    THINKING_CANNOT_BE_OFF = ("claude-opus-5-5",)

    def _thinking_cannot_be_off(self):
        return any(self.model == prefix or self.model.startswith(prefix + "-")
                   for prefix in self.THINKING_CANNOT_BE_OFF)

    def _thinking_unavailable(self, tried):
        return ThinkingUnavailable(
            f"{self.model} would not accept any way of turning thinking off "
            f"({', '.join(str(t) for t in tried) or 'none tried'}). Sending no thinking setting "
            f"would run it with thinking ON, which changes the instrument and cuts answers off "
            f"at the reply limit. Stopping. If a thinking-on arm is what you want, set "
            f"EWL_ANTHROPIC_ALLOW_THINKING=1 (larger reply limit, its own cache keys; disclose "
            f"it in the methods); otherwise leave this model out of the comparison.")

    def _thinking_off_config(self):
        """The `thinking` value that turns thinking off for this model."""
        override = getattr(self, "_thinking_override", None)
        if override:
            return override
        if self._thinking_cannot_be_off():
            if getattr(self, "allow_thinking", False):
                return self.THINKING_OMIT
            raise self._thinking_unavailable([])
        for prefix, config in self.THINKING_OFF_BY_PREFIX:
            if self.model == prefix or self.model.startswith(prefix + "-"):
                return config
        return self.THINKING_OFF_DEFAULT

    @staticmethod
    def _thinking_fix_from(error):
        """The `thinking` value the service asks for in a 400 about it, or None. The message
        reads like: send "thinking": {"type": "between_tools"} instead of {"type": "disabled"}."""
        import re
        match = re.search(r'"thinking":\s*\{"type":\s*"(\w+)"\}', str(error))
        return {"type": match.group(1)} if match else None

    def thinking_setting(self):
        """What this run sent for thinking, for the run record: 'disabled', 'between_tools',
        'omitted (no thinking field)', or None for a model that is never sent one."""
        if not self._thinking_on_by_default():
            return None
        try:
            config = self._thinking_off_config()
        except ThinkingUnavailable:
            return "cannot be turned off (run refused)"
        return ("thinking ON (no thinking field, allowed by EWL_ANTHROPIC_ALLOW_THINKING)"
                if config == self.THINKING_OMIT else config["type"])

    def _thinking_on_by_default(self):
        """Whether `self.model` is one of THINKING_ON_BY_DEFAULT, matching a
        dated snapshot id (claude-sonnet-5-20260115) the same way `price_for`
        already matches one -- an exact `in` check against the base names
        silently let every dated id through with thinking left on, since
        `--judge anthropic:claude-sonnet-5-20260115` never equals
        `"claude-sonnet-5"`."""
        if self.model in self.THINKING_ON_BY_DEFAULT:
            return True
        return any(self.model.startswith(base + "-") for base in self.THINKING_ON_BY_DEFAULT)

    def _build_request(self, body, prompt, ttl=None, for_batch=False):
        """The Messages request for one comment, as a dict. The live path sends it with
        `messages.create(**request)`; the batch path puts the very same dict in a batch, so a
        batched answer is for exactly the request a live call would have made.

        `for_batch` only changes how the optional effort setting travels: the SDK's
        `extra_body` is not part of a batch request's params, so it becomes a top-level field.
        """
        request = {
            "model": self.model,
            "max_tokens": self.reply_limit,
            "system": self._cacheable_system(prompt, ttl),
            "messages": [{"role": "user", "content": f'Classify this comment: "{body}"'}],
        }
        if self._thinking_on_by_default():
            # Classification is a short, structured, single-label answer --
            # extended reasoning buys nothing here, and leaving thinking on
            # by default means a variable, uncontrolled share of `max_tokens`
            # goes to invisible reasoning before the label is written, which
            # can silently truncate the actual answer to nothing (parsed as
            # `unclassified`) and always inflates output-token cost.
            config = self._thinking_off_config()
            if config != self.THINKING_OMIT:
                request["thinking"] = config
        effort = getattr(self, "effort", None)
        if effort:
            if for_batch:
                request["output_config"] = {"effort": effort}
            else:
                request["extra_body"] = {"output_config": {"effort": effort}}
        return request

    @staticmethod
    def _reply_from(response):
        """The reply text from a Message, or an UncacheableReply when it is not an answer.

        A safety decline is not a label, and a reply that stopped at the token limit is not
        an answer. Neither crashes a long run on one comment, and both are flagged
        uncacheable, so a rerun (with a larger budget, say) asks again instead of replaying
        the failure for free. A refusal is returned empty (parse failure); a truncated reply
        keeps whatever it wrote (used for this call, but never stored). Shared by the live
        path and the batch download.
        """
        if response.stop_reason == "refusal":
            return uncacheable("", "refusal")
        text = "".join(block.text for block in response.content if block.type == "text")
        if response.stop_reason not in ("end_turn", "stop_sequence"):
            return uncacheable(text, str(response.stop_reason))
        if not text.strip():
            # A normal stop with nothing written is not an answer either. Cached, it would be
            # replayed as a parse failure for ever, and `refuse_empty_cache` would then stop the
            # next run (and the next batch `submit`, which has no override) until the line was
            # removed by hand. Same rule as the OpenAI and Gemini backends.
            return uncacheable("", "empty")
        return text

    def _record_usage(self, usage):
        with self._lock:
            self.usage["input_tokens"] += usage.input_tokens
            self.usage["output_tokens"] += usage.output_tokens
            # Both default to 0 (not present at all) on a response that had
            # nothing to write or read from cache, e.g. the very first call
            # ever made, or a prompt too short to qualify -- never missing.
            self.usage["cache_creation_input_tokens"] += getattr(
                usage, "cache_creation_input_tokens", 0) or 0
            self.usage["cache_read_input_tokens"] += getattr(
                usage, "cache_read_input_tokens", 0) or 0

    # Set by the batch planner (anthropic_batch.py): called with (cache key, batch request)
    # for every reply that is NOT already cached, instead of calling the API.
    collector = None
    batch_ttl = None

    def _sample(self, body, prompt, vote):
        if self.collector is not None:
            self.collector(self._cache_key(body, prompt, vote),
                           self._build_request(body, prompt, self.batch_ttl, for_batch=True))
            return uncacheable("", "planned")
        while True:
            request = self._build_request(body, prompt)
            sent = request.get("thinking", self.THINKING_OMIT)   # what THIS request carried
            try:
                response = self._client.messages.create(**request)
                break
            except Exception as error:
                # A 400 about the thinking setting means this model wants "thinking off"
                # spelled another way: use the spelling the message names, or else the next
                # alternative not yet rejected, and remember what works for every later call.
                # Several comments run at once, so what was REJECTED is recorded from the
                # request that failed (not from the current setting, which another thread may
                # already have changed). Any other error (a credit balance, a bad request) is
                # raised as before.
                text = str(error)
                if "invalid_request" not in text or "thinking" not in text.lower():
                    raise
                # A model that is not sent a thinking field (Haiku, anything not known to think
                # by default) has no setting to correct, and a request that already carried no
                # thinking field has none to correct either: retrying would repeat the identical
                # request for ever.
                if not self._thinking_on_by_default() or sent == self.THINKING_OMIT:
                    raise
                with self._lock:
                    rejected = self.__dict__.setdefault("_thinking_rejected", [])
                    if sent not in rejected:
                        rejected.append(sent)
                    if self._thinking_off_config() in rejected:   # nobody has fixed it yet
                        fix = self._thinking_fix_from(error)
                        # Only a spelling that turns thinking OFF may be adopted: a hint such as
                        # {"type": "adaptive"} would silently switch thinking on.
                        if fix not in self.THINKING_ALTERNATIVES:
                            fix = None
                        if not fix or fix in rejected:
                            fix = next((c for c in self.THINKING_ALTERNATIVES
                                        if c not in rejected), None)
                        if fix is None:
                            if getattr(self, "_allow_thinking_requested", False):
                                fix = self.THINKING_OMIT
                            else:
                                raise self._thinking_unavailable(rejected) from error
                        self._thinking_override = fix
        self._record_usage(response.usage)
        return self._reply_from(response)


# OpenAI list prices, USD per million tokens: (input, output). Confirmed
# against developers.openai.com/api/docs/pricing 2026-09. GPT-5.6-Sol's exact
# API model-string is NOT confirmed here (the pricing page shows it by
# display name only) -- check OpenAI's own model list before passing
# --judge openai:<id>, and add its rate below before trusting a run against it.
_OPENAI_PRICES = {
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
}

# OpenAI's automatic-caching minimum, per its own docs -- same shape as
# Anthropic's _CACHE_MINIMUM_TOKENS above, kept separate rather than merged
# into one table since the two providers' minimums are not comparable
# per-model (this one applies uniformly across the current GPT-5 line).
OPENAI_CACHE_MINIMUM_TOKENS = 1024


def _price_overrides(variable, example):
    """Prices you supply for models not in a provider's table, from an environment variable
    holding "model=input,output[,cached];model2=..." (USD per million tokens, read off the
    provider's own pricing page; the optional third figure is the cached-input price). Used for
    cost reporting only; it never changes a label. A malformed value is refused (before anything
    is sent) rather than skipped. Returns {model: (input, output, cached or None)}."""
    raw = os.environ.get(variable, "").strip()
    found = {}
    for entry in filter(None, (part.strip() for part in raw.split(";"))):
        try:
            name, rates = entry.split("=", 1)
            figures = [float(x) for x in rates.split(",")]
            if not name.strip() or len(figures) not in (2, 3) or any(f < 0 or f != f for f in figures):
                raise ValueError
            found[name.strip()] = (figures[0], figures[1], figures[2] if len(figures) == 3 else None)
        except ValueError:
            raise PriceUnavailable(
                f"{variable} entry {entry!r} is not 'model=input,output' or "
                f"'model=input,output,cached' (USD per million tokens), for example "
                f"{variable}=\"{example}=1.25,10.00\".") from None
    return found


def _openai_price_overrides():
    return _price_overrides("EWL_OPENAI_PRICES", "gpt-5")


_TABLE_PRICE_NOTED = set()


def _note_table_price_once(variable, model):
    """Say once per run that a price in the environment is ignored because the model is in the
    price table (the price is read on every cost call, so printing each time floods the log)."""
    if (variable, model) not in _TABLE_PRICE_NOTED:
        _TABLE_PRICE_NOTED.add((variable, model))
        print(f"NOTE: {variable} gives a price for {model}, which is already in the price table; "
              f"the table's price is used.", file=sys.stderr)


def openai_price_for(model):
    """List price for an OpenAI `model`, or a loud refusal -- same reasoning
    as price_for() above: an unpriced model silently reporting $0.00 is worse
    than refusing to run. A price supplied in EWL_OPENAI_PRICES is used when the model is not
    in the table, so a new model can be run once its price has been read off OpenAI's own page."""
    supplied = _openai_price_overrides()      # also validates the variable, even for a known model
    if model in _OPENAI_PRICES:
        if model in supplied:
            _note_table_price_once("EWL_OPENAI_PRICES", model)
        return _OPENAI_PRICES[model]
    if model in supplied:
        return supplied[model][:2]
    raise PriceUnavailable(
        f"No price on record for OpenAI model {model!r}, and refusing to assume it is "
        f"free. Add it to _OPENAI_PRICES in judges.py, or supply it for this run with "
        f"EWL_OPENAI_PRICES=\"{model}=<input>,<output>\" (USD per million tokens, from OpenAI's "
        f"pricing page). Known: {', '.join(sorted(_OPENAI_PRICES))}.")


def openai_cached_ratio(model):
    """Cached-input price as a fraction of the input price: the figure supplied in
    EWL_OPENAI_PRICES if there is one, else 10% (what the older GPT-5 models charge; some newer
    models charge less, for example 5%, so supply the cached price for them)."""
    supplied = _openai_price_overrides().get(model)
    if model in _OPENAI_PRICES:
        return 0.1        # a model in the table keeps the table's rule; the variable is ignored
    if supplied and supplied[2] is not None and supplied[0]:
        return supplied[2] / supplied[0]
    return 0.1


@register("openai")
class OpenAIJudge(Judge):
    """GPT through the official SDK, for the same comparison.

    Defaults to gpt-5-mini, not gpt-4o-mini -- the old default predated the
    entire GPT-5 line and would silently benchmark a retired model.
    """

    deterministic = False
    default_model = "gpt-5-mini"
    DEFAULT_REASONING_EFFORT = "minimal"

    def __init__(self, model=None, **kwargs):
        super().__init__(model, **kwargs)
        # Before anything else: no data-protection sign-off, nothing is built or sent; an
        # unpriced model or a malformed price variable is refused here too, not after the
        # comments have gone.
        require_external_processor("openai")
        openai_price_for(self.model)
        # How much hidden reasoning a reasoning model may spend before answering. The default
        # is what every earlier run used; EWL_OPENAI_REASONING_EFFORT=low|medium|high (or
        # whatever the model accepts) changes it, goes into the cache key and the run record,
        # and is never silently swapped for another value if the model refuses it.
        raw_effort = os.environ.get("EWL_OPENAI_REASONING_EFFORT", "").strip().lower()
        if raw_effort and not self._is_reasoning_model():
            raise SettingRefused(
                f"EWL_OPENAI_REASONING_EFFORT={raw_effort!r} is set but {self.model} is not a "
                f"reasoning model, so the setting would be ignored. Unset it for this model.")
        self.reasoning_effort = raw_effort or self.DEFAULT_REASONING_EFFORT
        from openai import OpenAI
        self._client = OpenAI()

    @property
    def spec(self):
        """The arm's name. A non-default reasoning effort is part of it (`openai:gpt-6.1-sol@low`),
        so a scorecard row or an output column says which setting produced it."""
        base = super().spec
        if self._is_reasoning_model() and self._effort() != self.DEFAULT_REASONING_EFFORT:
            return f"{base}@{self._effort()}"
        return base

    @property
    def reply_limit(self):
        """`max_completion_tokens` sent: hidden reasoning is spent out of the same allowance as
        the answer, so a higher effort gets more room (512 for minimal, none or low)."""
        return {"medium": 2048, "high": 4096}.get(self._effort(), 512) if self._is_reasoning_model() else 512

    @property
    def CACHE_READ_MULTIPLIER(self):
        return openai_cached_ratio(self.model)

    def _effort(self):
        return getattr(self, "reasoning_effort", None) or self.DEFAULT_REASONING_EFFORT

    @property
    def cache_spec(self):
        """The default effort keeps the key it always had; any other value gets its own, so
        answers given under one effort are never replayed under another."""
        base = super().cache_spec
        if self._is_reasoning_model() and self._effort() != self.DEFAULT_REASONING_EFFORT:
            return f"{base}|reasoning_effort={self._effort()}"
        return base

    def thinking_setting(self):
        """What this run sent for hidden reasoning, for the run record; None for a model that
        is never sent the setting."""
        return f"reasoning_effort={self._effort()}" if self._is_reasoning_model() else None

    @property
    def price_per_mtok(self):
        return openai_price_for(self.model)

    def _is_reasoning_model(self):
        """GPT-5 family and the o-series reason before answering; others do not."""
        name = (self.model or "").lower()
        # gpt-6 and the o-series reason too; a model not recognised here would be sent no effort
        # setting and could spend its short answer limit on hidden reasoning.
        return name.startswith(("gpt-5", "gpt-6")) or name[:2] in ("o1", "o3", "o4")

    def _sample(self, body, prompt, vote):
        # GPT-5-family models reject BOTH of these on the Chat Completions
        # API with a 400 error: "max_tokens" (must be "max_completion_tokens"
        # instead -- confirmed against OpenAI's own current docs/community
        # reports, not memory) and any non-default "temperature" (GPT-5 is a
        # reasoning model and only accepts the implicit default). Confirmed
        # this was a live break on gpt-5-mini, this class's own default
        # model -- every call would 400 immediately. max_completion_tokens is
        # accepted by both the current and older model generations, so this
        # is not conditional on which model is configured; temperature is
        # simply dropped rather than pinned, matching the project's existing
        # position (see the determinism note above _ANTHROPIC_PRICES) that
        # reproducibility for an API arm rests on the response cache, not on
        # decode determinism.
        extra = {}
        if self._is_reasoning_model():
            # Only reasoning models accept this; a non-reasoning model answers a
            # request carrying it with a 400, which would stop the run.
            extra["reasoning_effort"] = self._effort()
        try:
            response = self._create(extra, body, prompt)
        except Exception as error:
            if "reasoning_effort" in str(error) and "400" in f"{getattr(error, 'status_code', '')} {error}":
                raise ReasoningEffortRejected(
                    f"{self.model} rejected reasoning_effort={self._effort()!r} (invalid_request): "
                    f"{error}. The run stops rather than carrying on with a different setting. "
                    f"Set EWL_OPENAI_REASONING_EFFORT to a value this model accepts.") from error
            if self._is_policy_block(error):
                # This one comment (or the prompt built round it) was refused by OpenAI's
                # content policy. It must not end an 84,000-comment run: it gets an empty,
                # uncacheable reply (a parse failure, counted as `blocked`), as a model
                # refusal does. Any other 400 -- a bad setting above all -- still stops the run.
                return uncacheable("", "blocked")
            raise
        choice = response.choices[0]
        text = choice.message.content or ""
        return self._finish(response, choice, text)

    # Markers of a 400 that is about THIS comment's content, not about the request's settings.
    # A bare "invalid_request" is deliberately not one: every settings error carries it.
    POLICY_BLOCK_MARKERS = ("invalid_prompt", "content_policy", "flagged")

    @classmethod
    def _is_policy_block(cls, error):
        text = f"{getattr(error, 'status_code', '')} {error}".lower()
        return ("400" in text
                and "reasoning_effort" not in text
                and any(marker in text for marker in cls.POLICY_BLOCK_MARKERS))

    def _create(self, extra, body, prompt):
        return self._client.chat.completions.create(
            model=self.model,
            # 256 gave no headroom once a reasoning model spends part of its
            # budget on invisible reasoning tokens before writing the visible
            # answer -- confirmed possible to exhaust entirely, leaving an
            # empty reply with no error. 512 leaves room for that; reasoning
            # is also turned down explicitly, since a short four-way
            # classification of one sentence does not need real deliberation.
            max_completion_tokens=self.reply_limit,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f'Classify this comment: "{body}"'},
            ],
            **extra,
        )

    def _finish(self, response, choice, text):
        """Record the spend, then turn the response into a reply (or an uncacheable one)."""
        if response.usage:
            # OpenAI's automatic caching has no write premium to track --
            # cost_usd()'s CACHE_WRITE_MULTIPLIER only ever applies to
            # cache_creation_input_tokens, which this backend never sets, so
            # leaving it at 0 (the Counter default) is correct, not an
            # omission. `prompt_tokens` is the TOTAL input (cached tokens
            # included, unlike Anthropic's input_tokens, which already
            # excludes them) -- subtracting cached_tokens before storing
            # keeps this backend's usage in the same shape cost_usd() expects,
            # rather than double-billing the cached portion at full price.
            # Recorded BEFORE the truncation check below, not after: a
            # truncated reply still spent real tokens and real money, and
            # raising on truncation must not make that spend vanish from the
            # cost total.
            details = getattr(response.usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) or 0
            with self._lock:
                self.usage["input_tokens"] += response.usage.prompt_tokens - cached
                self.usage["cache_read_input_tokens"] += cached
                self.usage["output_tokens"] += response.usage.completion_tokens
        if getattr(choice.message, "refusal", None):
            return uncacheable("", "refusal")
        if choice.finish_reason == "content_filter":
            return uncacheable("", "content_filter")
        if choice.finish_reason == "length" and text.strip():
            # A partial answer is used for this call but never stored.
            return uncacheable(text, "length")
        if choice.finish_reason == "length" and not text.strip():
            # The reasoning budget (or the 512-token cap) was used up before
            # any visible answer was written. Returning "" here would parse as
            # a confident "unclassified" and then get cached as if it were a
            # real answer -- indistinguishable from the model genuinely
            # judging the comment engages none of the four harms. Raising
            # instead sends this through the same retry path as a transient
            # network failure (see RETRYABLE's "truncated" marker): a fresh
            # attempt is a real second chance, not a repeat of a config error.
            raise TruncatedReply(
                f"{self.spec}: reply truncated (finish_reason=length) with no usable text")
        if not text.strip():
            return uncacheable("", "empty")
        return text


# Google list prices, USD per million tokens: (input, output), <=200k-token
# prompts. Confirmed against ai.google.dev/gemini-api/docs/pricing 2026-09.
_GOOGLE_PRICES = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.8-flash": (0.75, 3.75),       # rises to 1.50 / 7.50 on 1 January 2027
    "gemini-3.1-pro": (2.00, 12.00),
    "gemini-3.1-pro-preview": (2.00, 12.00),  # the id a key actually lists (22 October 2026)
    "gemini-4-argon": (2.00, 10.00),          # rises to 4.00 / 20.00 on 1 December 2026
}
# Thinking tokens are billed at the output rate (Google's pricing page, read 22 October 2026).

# Confirmed against ai.google.dev/gemini-api/docs/caching 2026-09: uniform
# across the current Gemini 3.x line (Pro and Flash alike), for implicit
# (automatic) caching -- the only kind this backend uses.
GOOGLE_CACHE_MINIMUM_TOKENS = 4096


def google_price_for(model):
    """List price for a Gemini `model`, or a loud refusal -- same reasoning
    as price_for() above. A price supplied in EWL_GOOGLE_PRICES is used for a model not in the
    table, so a new model can be run once its price has been read off Google's own page."""
    supplied = _price_overrides("EWL_GOOGLE_PRICES", "gemini-model")   # validates the variable
    if model in _GOOGLE_PRICES:
        if model in supplied:
            _note_table_price_once("EWL_GOOGLE_PRICES", model)
        return _GOOGLE_PRICES[model]
    if model in supplied:
        return supplied[model][:2]
    raise PriceUnavailable(
        f"No price on record for Google model {model!r}, and refusing to assume it is "
        f"free. Add it to _GOOGLE_PRICES in judges.py, or supply it for this run with "
        f"EWL_GOOGLE_PRICES=\"{model}=<input>,<output>\" (USD per million tokens, from Google's "
        f"pricing page). Known: {', '.join(sorted(_GOOGLE_PRICES))}.")


@register("google")
class GeminiJudge(Judge):
    """Gemini through the official google-genai SDK.

    NEW backend, unverified against a real call as of when this was written --
    the google-genai SDK's exact response attribute names (particularly on
    UsageMetadata: prompt_token_count / cached_content_token_count, and
    whichever of candidates_token_count / output_token_count this SDK version
    uses) were confirmed from Google's docs, not from actually running a
    request. Run one real comment through this backend and print
    `response.usage_metadata` before trusting its cost_usd() figures, the
    same way count_tokens is used to confirm Anthropic's cache behaviour
    elsewhere in this project rather than assumed.

    Reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment via the
    SDK's own zero-argument Client() -- never takes a key as an argument, for
    the same reason AnthropicJudge doesn't.

    Requires the google-genai package: pip install -e ".[google]".
    """

    deterministic = False
    default_model = "gemini-3.1-pro-preview"

    # Google's thinking page (read 22 October 2026) names `thinking_level` as the control for the
    # Gemini 3.x and 4 models, lists no "off" setting, and says every model listed thinks by
    # default (3.1 pro at high; 3.8 flash and 4 argon at medium; 3.5 flash-lite at minimal).
    # So there is no way to turn thinking off here: by default nothing is sent and each model
    # runs at its own default level, recorded as such; EWL_GEMINI_THINKING_LEVEL sends a level
    # (and keys the cache on it). Thinking tokens are billed as output and count against the
    # reply limit, so the limit is larger than for a model that does not think.
    THINKING_LEVELS = ("minimal", "low", "medium", "high")
    REPLY_LIMITS = {"minimal": 1024, "low": 1024, "medium": 2048, "high": 4096}
    DEFAULT_REPLY_LIMIT = 2048
    # The level each model thinks at when nothing is sent, and the levels it accepts (Google's
    # thinking table, read 23 October 2026). A model not listed is let through unchecked; the
    # first call then says if the level is wrong.
    MODEL_DEFAULT_LEVEL = {"gemini-3.1-pro-preview": "high", "gemini-3.1-pro": "high",
                           "gemini-3.8-flash": "medium", "gemini-4-argon": "medium",
                           "gemini-3.5-flash-lite": "minimal"}
    MODEL_ACCEPTED_LEVELS = {"gemini-3.1-pro-preview": ("low", "medium", "high"),
                             "gemini-3.1-pro": ("low", "medium", "high"),
                             "gemini-3.8-flash": ("low", "medium", "high"),
                             "gemini-4-argon": ("low", "medium", "high")}

    def __init__(self, model=None, **kwargs):
        super().__init__(model, **kwargs)
        # Before anything else: no sign-off, nothing is built or sent; an unpriced model or a
        # malformed price variable is refused here too, before any comment goes.
        require_external_processor("google")
        google_price_for(self.model)
        if os.environ.get("EWL_GEMINI_THINKING_BUDGET", "").strip():
            raise SettingRefused("EWL_GEMINI_THINKING_BUDGET is no longer used: Google's documentation "
                             "names thinking_level (minimal, low, medium, high) as the control for "
                             "these models. Unset it, or set EWL_GEMINI_THINKING_LEVEL.")
        level = os.environ.get("EWL_GEMINI_THINKING_LEVEL", "").strip().lower()
        if level and level not in self.THINKING_LEVELS:
            raise SettingRefused(f"EWL_GEMINI_THINKING_LEVEL={level!r}: use one of "
                                 f"{', '.join(self.THINKING_LEVELS)}, or leave it unset for the "
                                 f"model's own default.")
        accepted = self.MODEL_ACCEPTED_LEVELS.get(self.model)
        if level and accepted and level not in accepted:
            raise SettingRefused(f"{self.model} accepts only the thinking levels "
                                 f"{', '.join(accepted)}; EWL_GEMINI_THINKING_LEVEL={level!r} "
                                 f"would be rejected on the first call.")
        self.thinking_level = level or None
        from google import genai
        self._client = genai.Client()
        self._types = __import__("google.genai.types", fromlist=["types"])

    @property
    def spec(self):
        """The arm's name. A thinking level set by hand is part of it (`google:gemini-3.8-flash@low`),
        so a scorecard row or an output column says which setting produced it."""
        level = getattr(self, "thinking_level", None)
        return f"{super().spec}@{level}" if level else super().spec

    @property
    def cache_spec(self):
        """Unset keeps the plain key; a level set by hand gets its own, so answers given under
        one level are never replayed under another."""
        base = super().cache_spec
        level = getattr(self, "thinking_level", None)
        return f"{base}|thinking_level={level}" if level else base

    def thinking_setting(self):
        """What this run sent for thinking, for the run record."""
        level = getattr(self, "thinking_level", None)
        return f"thinking_level={level}" if level else "model default (thinking on; nothing sent)"

    @property
    def reply_limit(self):
        """`max_output_tokens`: thinking is spent out of the same allowance as the answer."""
        level = getattr(self, "thinking_level", None) or self.MODEL_DEFAULT_LEVEL.get(self.model)
        return self.REPLY_LIMITS.get(level, self.DEFAULT_REPLY_LIMIT)

    @property
    def price_per_mtok(self):
        return google_price_for(self.model)

    def _sample(self, body, prompt, vote):
        config_kwargs = dict(
            system_instruction=prompt,
            # 256 gave no headroom against invisible thinking tokens eating
            # the whole budget before any visible text was written (see the
            # thinking_config below, and OpenAIJudge's equivalent fix) -- 512
            # leaves margin even where thinking cannot be turned off.
            max_output_tokens=self.reply_limit,
            temperature=0,
        )
        level = getattr(self, "thinking_level", None)
        if level:
            config_kwargs["thinking_config"] = self._types.ThinkingConfig(thinking_level=level.upper())
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=f'Classify this comment: "{body}"',
                config=self._types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as error:
            # Only when a thinking level WAS sent: a thinking-related 400 with nothing sent is
            # not a setting to correct and is raised as it is.
            if level and "thinking" in str(error).lower() and "400" in f"{getattr(error, 'code', '')} {error}":
                raise ThinkingUnavailable(
                    f"{self.model} rejected {self.thinking_setting()} (invalid_request): {error}. "
                    f"The run stops rather than carrying on with a different thinking setting. "
                    f"Set EWL_GEMINI_THINKING_LEVEL to a level this model accepts (a new --cache "
                    f"is not needed: the level is part of the cache key), or leave it unset.") from error
            raise
        text = getattr(response, "text", None) or ""

        finish_reason = None
        candidates = getattr(response, "candidates", None)
        if candidates:
            finish_reason = getattr(candidates[0], "finish_reason", None)
        else:
            # No candidate at all: if Google blocked the prompt itself, say so (a block is not
            # the same as the model answering nothing).
            feedback = getattr(response, "prompt_feedback", None)
            if getattr(feedback, "block_reason", None):
                return uncacheable("", "blocked")

        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            cached = getattr(usage, "cached_content_token_count", 0) or 0
            # Field name genuinely uncertain between SDK versions -- try both
            # rather than silently recording 0 output tokens if one is wrong.
            output_tokens = (getattr(usage, "candidates_token_count", None)
                              or getattr(usage, "output_token_count", 0) or 0)
            # Invisible reasoning tokens are billed as output tokens by Google
            # but were never added here -- a mocked test measured this as
            # under-reporting true cost by roughly 2.2x. Added in whenever the
            # SDK reports them (0 when thinking is disabled above, or on a
            # model/SDK version that doesn't surface the field at all).
            thoughts_tokens = getattr(usage, "thoughts_token_count", 0) or 0
            output_tokens += thoughts_tokens
            # Same shape as OpenAIJudge: prompt_token_count is the TOTAL
            # input (cached portion included), so subtract it out before
            # storing -- cost_usd() expects input_tokens to already exclude
            # whatever is being billed separately as a cache read.
            # Recorded BEFORE the truncation check below -- a truncated reply
            # still spent real (billable) tokens, and raising on truncation
            # must not make that spend vanish from the cost total.
            with self._lock:
                self.usage["input_tokens"] += prompt_tokens - cached
                self.usage["cache_read_input_tokens"] += cached
                self.usage["output_tokens"] += output_tokens

        if finish_reason is not None and "MAX_TOKENS" in str(finish_reason) and not text.strip():
            # The token budget (thinking, if not disabled above, plus the
            # visible answer) was exhausted before any visible text was
            # written. `getattr(response, "text", None) or ""` used to let
            # this fall through silently as an empty string, which then
            # parsed as a confident "unclassified" and got cached as if it
            # were a real answer -- indistinguishable from the model
            # genuinely finding none of the four harms. Raising instead sends
            # this through the same retry path as a transient network failure
            # (see RETRYABLE's "truncated" marker).
            raise TruncatedReply(
                f"{self.spec}: reply truncated (finish_reason=MAX_TOKENS) with no usable text")
        if finish_reason is not None and "STOP" not in str(finish_reason):
            # MAX_TOKENS with partial text, SAFETY, RECITATION, ...: not an answer.
            return uncacheable(text, str(finish_reason).split(".")[-1].lower())
        if not text.strip():
            return uncacheable("", "empty")
        return text


@register("stub")
class StubJudge(Judge):
    """Always answers `social`. Exists so the harness can be exercised end to
    end with no model, no network and no key — see test_harness.py."""

    deterministic = True
    default_model = "constant"

    def _sample(self, body, prompt, vote):
        return "social"


@register("manual")
class ManualJudge(Judge):
    """No API call at all: prints the exact prompt this backend would have
    sent, for you to paste into a chat interface (e.g. claude.ai) by hand,
    then reads back whatever you paste as the reply.

    For a dry run of the pipeline before paying for API access -- same
    prompt, same parsing, same downstream tools (`ewl-classify --judge
    manual`, `ewl-evaluate scorecard --arm manual/none`, or
    `--sentiment llm:manual` for sentiment, since LLMScorer builds on this
    same registry). Keep `--votes 1`: re-pasting one comment three times for
    a "majority vote" defeats the point of a quick manual check, and nothing
    stops you doing it if you want to.

    Printed to stderr, not stdout, since classify.py writes the labelled CSV
    to stdout when --output is omitted -- the prompt must not land in it.
    """

    deterministic = False
    default_model = "chat-relay"

    def _sample(self, body, prompt, vote):
        print("\n" + "=" * 70, file=sys.stderr)
        print("Paste everything below the line into your chat, then paste its "
              "reply back here and press Enter.", file=sys.stderr)
        print("-" * 70, file=sys.stderr)
        print(prompt, file=sys.stderr)
        print(f"\nComment: {body}", file=sys.stderr)
        print("-" * 70, file=sys.stderr)
        print("Reply: ", end="", file=sys.stderr, flush=True)
        return input()
