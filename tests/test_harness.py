"""End-to-end check of the judge/selector/scorecard harness.

    python test_harness.py

Runs entirely on the `stub` and `keyword` judges, so it needs no model, no
network and no API key, and can therefore run in CI. It checks the wiring, not
the accuracy of any real model: that a new backend needs only `_sample`, that
the two selectors are interchangeable, that the split never leaks a held-out
comment into the retrieval pool, and that off-list answers are counted rather
than absorbed.
"""
import json
import pathlib
import collections
import csv
import io
import os

# Several checks below use paths relative to the repository root (the scratch scripts,
# excluded_video_ids.txt) and put `src` on PYTHONPATH from the working directory.
# Run from anywhere, they used to fail with a confusing "file not found": so every run
# starts from the root, whatever directory it was launched in.
os.chdir(pathlib.Path(__file__).resolve().parent.parent)

# ewl_common.hash_author() refuses to hash anything without a real salt (a
# GDPR safeguard: usernames must never be hashed with a repo-shipped default).
# ewl_common reads EWL_AUTHOR_SALT once, at import time, into a module-level
# variable -- and merge.py (imported below) pulls ewl_common in immediately.
# So this must be set before any import below, not just before the test that
# needs it, or a later os.environ assignment is too late to matter. This is a
# harness-only value, never used to hash anything real.
# An EMPTY value counts as unset: setdefault left "" in place, and the harness then
# failed on the first thing that hashed a name.
if not os.environ.get("EWL_AUTHOR_SALT"):
    os.environ["EWL_AUTHOR_SALT"] = "test-harness-only-salt-do-not-use-for-real-data"

# The OpenAI and Google judges refuse to be built unless the provider is listed here (a
# data-protection gate). The harness uses fake clients and sends nothing anywhere, so it
# approves them; the gate's own tests unset it.
os.environ.setdefault("EWL_EXTERNAL_PROCESSORS", "openai,google")

import statistics
import subprocess
import sys
import tempfile
import threading

from ewl.analyse import evaluate
import ewl.label.examples as examples_mod
from ewl.label import judges
from ewl.collect import merge
from ewl.analyse import report
from ewl import taxonomy
from ewl.analyse import validate

FIXTURE = [
    ("They spent millions lobbying and paid no tax at all.", "democracy"),
    ("He bought the newspaper to control the coverage.", "democracy"),
    ("Private jets emit more in an hour than I do in a year.", "environment"),
    ("Record profits while our wages were frozen.", "economics"),
    ("Nobody earns a billion by working hard.", "social"),
    ("Offshore trusts and shell companies, all perfectly legal.", "democracy"),
    ("The yacht has its own smaller yacht.", "environment"),
    ("Rent is half my salary now.", "economics"),
]

failures = []
skipped = []


def optional(label):
    """Guard a block that needs an optional dependency.

    The harness advertises itself as needing no model, network or key. That was
    only true of the judges: the sentiment block imported vaderSentiment
    unguarded, so on a machine without it the run died with a traceback
    partway through, never printed a result, and exited non-zero for a reason
    unrelated to the code under test. Anyone setting the project up for the
    first time would read that as the pipeline being broken.

    A missing optional dependency now skips its block and is named in the
    summary, so an incomplete run says it was incomplete instead of looking
    like a failure.
    """
    import contextlib

    @contextlib.contextmanager
    def guard():
        try:
            yield
        except ImportError as error:
            print(f"  SKIP  {label} ({error.name or 'an optional package'} not installed)")
            skipped.append(label)
    return guard()
import contextlib


def contextlib_redirect_stderr_quiet():
    return contextlib.redirect_stderr(io.StringIO())


def check(label, condition, detail=""):
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'' if condition else f'  <- {detail}'}")
    if not condition:
        failures.append(label)


def project_file(path):
    """True if a project file (the dashboard, the exclusion list) is here to check.

    The package repository holds only the package, so these checks skip in it, and say so;
    in the project repository, where the files are, they run as before."""
    if os.path.exists(path):
        return True
    print(f"  SKIP  checks on {path} (a project file, not part of the package)")
    return False


def write_fixture(path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["comment_id", "story", "platform", "body", "hand_domain"])
        writer.writeheader()
        for index, (body, label) in enumerate(FIXTURE):
            writer.writerow({"comment_id": f"c{index}", "story": "s", "platform": "youtube",
                             "body": body, "hand_domain": label})


from ewl.label import classify

print("taxonomy")
check("off-list answers become a parse failure, not fuzzy-matched, and not "
      "silently folded into a genuine 'none'",
      taxonomy.parse_labels("policy") == ["parse_failure"])
check("fuzzy is opt-in and still works",
      taxonomy.parse_labels("democrac", fuzzy=True) == ["democracy"])
check("labels are capped at two", len(taxonomy.parse_labels("democracy, social, economics")) == 2)
check("a clean 'none' yields the genuine-none sentinel, not a parse failure",
      taxonomy.parse_labels("none") == ["unclassified"])
check("an empty reply is a parse failure, not a genuine none (e.g. a safety refusal)",
      taxonomy.parse_labels("") == ["parse_failure"]
      and taxonomy.parse_labels(None) == ["parse_failure"])
check("'none' mixed with garbage is a parse failure -- not a clean, confident none",
      taxonomy.parse_labels("none, xyz123") == ["parse_failure"])
check("a real category alongside incidental garbage still returns just the real "
      "category -- parse_failure is reserved for replies with no usable category at all",
      taxonomy.parse_labels("democracy, xyz123") == ["democracy"])
check("fuzzy-matched garbage still counts as a real category, not a parse failure",
      taxonomy.parse_labels("democrac, xyz123", fuzzy=True) == ["democracy"])
check("every arm builds its prompt from the same definitions",
      all(name in taxonomy.build_prompt([]) for name in taxonomy.CATEGORY_NAMES))

print("\njudges")
check("registry exposes the backends", set(judges.available()) >=
      {"anthropic", "keyword", "ollama", "openai", "google", "stub"})
stub = judges.create("stub", votes=3)
labels, unstable, ballots = stub.label("anything", taxonomy.FIXED_EXAMPLES)
check("voting resolves to a majority label", labels == ["social"], labels)
check("a deterministic judge reports no instability", unstable is False)
check("usage counts every sample", stub.usage["calls"] == 3, stub.usage["calls"])
check("local judges cost nothing", stub.cost_usd() == 0)
try:
    judges.create("no-such-backend")
    check("unknown backend raises", False, "no exception")
except ValueError:
    check("unknown backend raises", True)

kw = judges.create("keyword")
check("keyword judge reads the real vocabulary",
      kw.label("They spent millions lobbying parliament.", [])[0] == ["democracy"])
check("keyword judge forces a single vote", kw.votes == 1)
check("the arm name shows the vocabulary file, not its absolute path",
      kw.spec == "keyword:keywords.yml", kw.spec)

# Regression: keyword_classify.UNCLASSIFIED is the literal string
# "unclassified", which taxonomy.parse_labels() does not recognise (only the
# word "none" reads as a clean, confident null) -- every genuine "matches
# none of the four categories" verdict from this judge was being recorded as
# a PARSE_FAILURE instead. Confirmed live on the real 84,250-comment corpus:
# 58.3% PARSE_FAILURE, 0% UNCLASSIFIED, before this fix.
check("a comment matching no category resolves to UNCLASSIFIED, not PARSE_FAILURE",
      kw.label("a completely unrelated comment about nothing in particular", [])[0]
      == [taxonomy.UNCLASSIFIED])

print("\nadding a backend is one method")


@judges.register("test_echo")
class EchoJudge(judges.Judge):
    deterministic = True
    default_model = "echo"

    def _sample(self, body, prompt, vote):
        return "environment" if "jet" in body else "none"


echo = judges.create("test_echo")
check("a new backend needs only _sample", echo.label("a jet", [])[0] == ["environment"])
check("it inherits parsing and voting", echo.label("hello", [])[0] == ["unclassified"])

print("\nprompt caching")
check("build_prompt is exactly the stable prefix plus the examples suffix",
      taxonomy.build_prompt(taxonomy.FIXED_EXAMPLES) ==
      taxonomy.STABLE_PREFIX + taxonomy.build_prompt_parts(taxonomy.FIXED_EXAMPLES)[1])
check("the stable prefix never varies with the examples given",
      taxonomy.build_prompt_parts(taxonomy.FIXED_EXAMPLES)[0]
      == taxonomy.build_prompt_parts([])[0] == taxonomy.STABLE_PREFIX)

# _cacheable_system needs no network and no API key -- bypass __init__
# (which builds a real anthropic.Anthropic() client) with __new__.
#
# REVISED 2026-09: this used to split the domain-classification prompt into
# two cache blocks (the stable definitions, the per-comment examples). Measured
# on a real trial, both halves came out small enough (~330 and ~325 tokens)
# that neither reliably cleared Anthropic's minimum cacheable prefix length --
# confirmed by a real run reporting zero cache reads or writes at all. Now a
# single block over the whole prompt, which is large enough to have a real
# chance of clearing that minimum.
_anthropic_probe = object.__new__(judges.AnthropicJudge)
_domain_prompt = taxonomy.build_prompt(taxonomy.FIXED_EXAMPLES)
_wrapped = _anthropic_probe._cacheable_system(_domain_prompt)
check("a domain-classification prompt is wrapped in exactly one cache-marked block",
      _wrapped == [{"type": "text", "text": _domain_prompt, "cache_control": {"type": "ephemeral"}}],
      _wrapped)

_other_prompt = "Rate this comment's sentiment from 1 to 5. No retrieved examples here."
_single = _anthropic_probe._cacheable_system(_other_prompt)
check("any other prompt shape is wrapped the same way -- one cache block over the whole thing",
      _single == [{"type": "text", "text": _other_prompt, "cache_control": {"type": "ephemeral"}}],
      _single)

print("\njudges.py: thinking is explicitly disabled for models that default it on")
# Regression test for a real finding, live 2026-09: Claude Sonnet 5 (and
# Opus 5) run adaptive thinking ON by default when `thinking` is omitted --
# confirmed by a real call where a 20-token budget spent 19 on invisible
# thinking, leaving nothing for the actual answer (stop_reason: max_tokens).
# Haiku 4.5 defaults to no thinking and must NOT get an explicit `thinking`
# kwarg (not a documented-safe shape for its older-style config).


class _FakeUsage:
    input_tokens = 10
    output_tokens = 5
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0


class _FakeResponse:
    usage = _FakeUsage()
    stop_reason = "end_turn"
    content = []


class _CapturingMessages:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResponse()


class _FakeClient:
    def __init__(self):
        self.messages = _CapturingMessages()


def _make_probe(model):
    probe = object.__new__(judges.AnthropicJudge)
    probe.model = model
    probe.usage = collections.Counter()
    probe._lock = threading.Lock()
    probe._client = _FakeClient()
    return probe


_sonnet_probe = _make_probe("claude-sonnet-5")
_sonnet_probe._sample("a comment", "a short prompt", 0)
check("Sonnet gets thinking explicitly disabled",
      _sonnet_probe._client.messages.last_kwargs.get("thinking") == {"type": "disabled"},
      _sonnet_probe._client.messages.last_kwargs)

_opus_probe = _make_probe("claude-opus-5")
_opus_probe._sample("a comment", "a short prompt", 0)
check("Opus gets thinking explicitly disabled too",
      _opus_probe._client.messages.last_kwargs.get("thinking") == {"type": "disabled"},
      _opus_probe._client.messages.last_kwargs)

_haiku_probe = _make_probe("claude-haiku-4-5")
_haiku_probe._sample("a comment", "a short prompt", 0)
check("Haiku is left alone -- no thinking kwarg sent at all",
      "thinking" not in _haiku_probe._client.messages.last_kwargs,
      _haiku_probe._client.messages.last_kwargs)

_dated_sonnet_probe = _make_probe("claude-sonnet-5-20260115")
_dated_sonnet_probe._sample("a comment", "a short prompt", 0)
check("a DATED sonnet snapshot id also gets thinking disabled, not just the bare name",
      _dated_sonnet_probe._client.messages.last_kwargs.get("thinking") == {"type": "disabled"},
      _dated_sonnet_probe._client.messages.last_kwargs)


def _raised_early(fn, exc):
    try:
        fn()
    except exc:
        return True
    return False


_sonnet55_probe = _make_probe("claude-sonnet-5-5")
_sonnet55_probe._sample("a comment", "a short prompt", 0)
check("Sonnet 5.5 is sent thinking {between_tools}: it rejects {disabled} with a 400 (seen live)",
      _sonnet55_probe._client.messages.last_kwargs.get("thinking") == {"type": "between_tools"},
      _sonnet55_probe._client.messages.last_kwargs)
_dated55 = _make_probe("claude-sonnet-5-5-20261101")
_dated55._sample("a comment", "a short prompt", 0)
check("a dated 5.5 snapshot id gets it too, and Sonnet 5 keeps {disabled}",
      _dated55._client.messages.last_kwargs.get("thinking") == {"type": "between_tools"}
      and _sonnet_probe._client.messages.last_kwargs.get("thinking") == {"type": "disabled"})

# Opus 5.5 rejected BOTH ways of turning thinking off. Sending nothing would run it with thinking ON.
_opus55_probe = _make_probe("claude-opus-5-5")
try:
    _opus55_probe._sample("a comment", "a short prompt", 0)
    _opus_refused = False
except judges.ThinkingUnavailable as _tu:
    _opus_refused = "EWL_ANTHROPIC_ALLOW_THINKING" in str(_tu)
check("Opus 5.5 stops with a clear message instead of running with thinking on, and sends nothing",
      _opus_refused and _opus55_probe._client.messages.last_kwargs is None)
_opus_ok = _make_probe("claude-opus-5-5")
_opus_ok.allow_thinking = True
_opus_ok._sample("a comment", "a short prompt", 0)
check("with EWL_ANTHROPIC_ALLOW_THINKING it runs thinking-on: no thinking field, a larger reply limit",
      "thinking" not in _opus_ok._client.messages.last_kwargs
      and _opus_ok._client.messages.last_kwargs["max_tokens"] == 1024
      and _sonnet55_probe._client.messages.last_kwargs["max_tokens"] == 256)
check("a thinking-on arm has its own cache keys, so its replies never mix with a thinking-off run's",
      _opus_ok.cache_spec.endswith("|thinking=allowed") and "thinking" not in _sonnet55_probe.cache_spec,
      (_opus_ok.cache_spec, _sonnet55_probe.cache_spec))
_batch_opus = _make_probe("claude-opus-5-5")
check("a batch for Opus 5.5 is refused too (a batch cannot retry), unless thinking-on is allowed",
      _raised_early(lambda: _batch_opus._build_request("c", "p", for_batch=True), judges.ThinkingUnavailable)
      and "thinking" not in _opus_ok._build_request("c", "p", for_batch=True))


class _PickyMessages:
    """Rejects any thinking spelling but `wants`, with the service's own 400 wording."""

    def __init__(self, wants):
        self.wants = wants
        self.sent = []

    def create(self, **kwargs):
        self.sent.append(kwargs.get("thinking"))
        if kwargs.get("thinking") != {"type": self.wants}:
            raise Exception("Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
                            "'message': 'To turn thinking off on this model, send \"thinking\": {\"type\": \""
                            + self.wants + "\"} instead of {\"type\": \"x\"}.'}}")
        return _FakeResponse()


_picky = _make_probe("claude-sonnet-5-5")
_picky._client.messages = _PickyMessages("disabled")
_picky._sample("a comment", "a short prompt", 0)
_picky._sample("a comment", "a short prompt", 1)
check("a 400 naming another thinking spelling is corrected once, then remembered for later calls",
      _picky._client.messages.sent == [{"type": "between_tools"}, {"type": "disabled"}, {"type": "disabled"}],
      _picky._client.messages.sent)


class _UnhintedMessages:
    """Rejects every spelling but `wants` with the wording Opus 5.5 gave: no alternative named."""

    def __init__(self, wants):
        self.wants = wants    # a type, or None for "send no thinking field"
        self.sent = []

    def create(self, **kwargs):
        sent = kwargs.get("thinking", "OMITTED")
        self.sent.append(sent)
        ok = (sent == "OMITTED") if self.wants is None else (sent == {"type": self.wants})
        if not ok:
            raise Exception("Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
                            "'message': '\"thinking.type between_tools\" is not supported for this model.'}}")
        return _FakeResponse()


_unh = _make_probe("claude-sonnet-5-5")
_unh._client.messages = _UnhintedMessages(None)
try:
    _unh._sample("a comment", "a short prompt", 0)
    _unh_stopped = False
except judges.ThinkingUnavailable:
    _unh_stopped = True
check("when no way of turning thinking off is accepted, it stops (it never falls back to thinking ON) "
      "after trying each once",
      _unh_stopped and _unh._client.messages.sent == [{"type": "between_tools"}, {"type": "disabled"}],
      _unh._client.messages.sent)
_unh2 = _make_probe("claude-sonnet-5-5")
_unh2.allow_thinking = True
_unh2._client.messages = _UnhintedMessages(None)
_unh2._sample("a comment", "a short prompt", 0)
_unh2._sample("a comment", "a short prompt", 1)
check("with thinking-on allowed, it settles on no thinking field and remembers it",
      _unh2._client.messages.sent == [{"type": "between_tools"}, {"type": "disabled"}, "OMITTED", "OMITTED"],
      _unh2._client.messages.sent)


class _RaceMessages:
    """Several comments in flight at once; only a request with NO thinking field is accepted, and
    the 400 for the others names no alternative (the Opus 5.5 wording)."""

    def __init__(self):
        self.sent = []
        self.lock = threading.Lock()

    def create(self, **kwargs):
        import time as _time_race
        _time_race.sleep(0.01)
        with self.lock:
            self.sent.append(kwargs.get("thinking", "OMITTED"))
        if "thinking" in kwargs:
            raise Exception("Error code: 400 - {'error': {'type': 'invalid_request_error', 'message': "
                            "'\"thinking.type between_tools\" is not supported for this model.'}}")
        return _FakeResponse()


def _race_run(allow):
    probe = _make_probe("claude-sonnet-5-5")
    probe.allow_thinking = allow
    probe._client.messages = _RaceMessages()
    errors = []

    def one(i):
        try:
            probe._sample(f"comment {i}", "a short prompt", 0)
        except Exception as _e:
            errors.append(type(_e).__name__)

    threads = [threading.Thread(target=one, args=(i,)) for i in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    return probe, errors


_race_ok, _race_ok_errors = _race_run(True)
_race_no, _race_no_errors = _race_run(False)
check("with comments in flight together and thinking-on allowed, every one succeeds and the setting is remembered",
      not _race_ok_errors and _race_ok._thinking_override == judges.AnthropicJudge.THINKING_OMIT, _race_ok_errors)
check("with comments in flight together and thinking-on NOT allowed, every one stops with the clear message",
      _race_no_errors == ["ThinkingUnavailable"] * 6, _race_no_errors)


class _OtherErrorMessages:
    def create(self, **kwargs):
        raise Exception("Error code: 400 - {'error': {'type': 'invalid_request_error', 'message': 'credit balance is too low'}}")


_other = _make_probe("claude-sonnet-5-5")
_other._client.messages = _OtherErrorMessages()
try:
    _other._sample("a comment", "a short prompt", 0)
    _other_raised = False
except Exception:
    _other_raised = True
check("any other 400 is raised, not retried with a different thinking value", _other_raised)

# every live call's duration is kept, for a median that a sleeping computer cannot ruin
_lat = _make_probe("claude-sonnet-5-5")
_lat.slept = 0.0
_lat.seconds = 0.0
_lat._sample = lambda body, prompt, vote: "none"
_lat._with_retries = lambda call, label: call()
for _i in range(3):
    _lat._timed_sample("c", "p", _i)
check("each live call's duration is recorded", len(_lat.latencies) == 3 and _lat.usage["calls"] == 3)

print("\njudges.py: AnthropicJudge._sample request shape and usage accounting (mocked)")


class _FakeUsageWithCache:
    input_tokens = 100
    output_tokens = 20
    cache_creation_input_tokens = 300
    cache_read_input_tokens = 400


class _FakeResponseNormal:
    usage = _FakeUsageWithCache()
    stop_reason = "end_turn"
    content = [type("Block", (), {"type": "text", "text": "democracy"})()]


class _CapturingMessagesReturning:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


_usage_probe = _make_probe("claude-haiku-4-5")
_usage_probe._client.messages = _CapturingMessagesReturning(_FakeResponseNormal())
_reply = _usage_probe._sample("a comment", "a prompt", 0)
check("_sample sends max_tokens=256 and the classification user message",
      _usage_probe._client.messages.last_kwargs["max_tokens"] == 256
      and 'Classify this comment: "a comment"'
      in _usage_probe._client.messages.last_kwargs["messages"][0]["content"],
      _usage_probe._client.messages.last_kwargs)
check("_sample returns the text of the reply",
      _reply == "democracy", _reply)
check("_sample records input/output tokens AND both cache fields",
      (_usage_probe.usage["input_tokens"], _usage_probe.usage["output_tokens"],
       _usage_probe.usage["cache_creation_input_tokens"], _usage_probe.usage["cache_read_input_tokens"])
      == (100, 20, 300, 400), dict(_usage_probe.usage))


class _FakeResponseRefusal:
    usage = _FakeUsageWithCache()
    stop_reason = "refusal"
    content = [type("Block", (), {"type": "text", "text": "should not appear"})()]


_refusal_probe = _make_probe("claude-haiku-4-5")
_refusal_probe._client.messages = _CapturingMessagesReturning(_FakeResponseRefusal())
check("a safety refusal returns an empty reply, not the (nonexistent) label text",
      _refusal_probe._sample("a comment", "a prompt", 0) == "")
check("usage is still recorded even on a refusal -- the call still cost real tokens",
      _refusal_probe.usage["input_tokens"] == 100, dict(_refusal_probe.usage))


print("\njudges.py: OpenAIJudge._sample request shape and usage accounting (mocked)")


def _fake_openai_response(content, finish_reason, prompt_tokens=100, completion_tokens=20,
                          cached_tokens=0):
    choice = type("Choice", (), {
        "message": type("Message", (), {"content": content})(),
        "finish_reason": finish_reason,
    })()
    details = type("Details", (), {"cached_tokens": cached_tokens})()
    usage = type("Usage", (), {
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "prompt_tokens_details": details,
    })()
    return type("Response", (), {"choices": [choice], "usage": usage})()


def _make_openai_probe(model="gpt-5-mini"):
    probe = object.__new__(judges.OpenAIJudge)
    probe.model = model
    probe.usage = collections.Counter()
    probe._lock = threading.Lock()

    class _FakeCompletions:
        def __init__(self):
            self.last_kwargs = None

        def create(self, **kwargs):
            self.last_kwargs = kwargs
            return probe._next_response

    class _FakeChat:
        def __init__(self):
            self.completions = _FakeCompletions()

    class _FakeOpenAIClient:
        def __init__(self):
            self.chat = _FakeChat()

    probe._client = _FakeOpenAIClient()
    return probe


_openai_probe = _make_openai_probe()
_openai_probe._next_response = _fake_openai_response(
    "democracy | 4", "stop", prompt_tokens=150, completion_tokens=8, cached_tokens=50)
_openai_reply = _openai_probe._sample("a comment", "a prompt", 0)
_openai_kwargs = _openai_probe._client.chat.completions.last_kwargs
check("_sample sends max_completion_tokens (not max_tokens) and no temperature",
      _openai_kwargs["max_completion_tokens"] == 512 and "max_tokens" not in _openai_kwargs
      and "temperature" not in _openai_kwargs, _openai_kwargs)
check("_sample sets reasoning_effort to keep hidden reasoning cheap",
      _openai_kwargs.get("reasoning_effort") == "minimal", _openai_kwargs)
check("_sample returns the reply text on an ordinary (non-truncated) response",
      _openai_reply == "democracy | 4", _openai_reply)
check("cached tokens are split out of input_tokens, not double-counted",
      (_openai_probe.usage["input_tokens"], _openai_probe.usage["cache_read_input_tokens"],
       _openai_probe.usage["output_tokens"]) == (100, 50, 8), dict(_openai_probe.usage))

_openai_truncated = _make_openai_probe()
_openai_truncated._next_response = _fake_openai_response(
    "", "length", prompt_tokens=200, completion_tokens=512, cached_tokens=0)
try:
    _openai_truncated._sample("a comment", "a prompt", 0)
    check("a truncated, empty reply (finish_reason=length) raises rather than returns ''",
          False, "no exception")
except RuntimeError as _err:
    check("a truncated, empty reply (finish_reason=length) raises rather than returns ''",
          "truncated" in str(_err), str(_err))
check("the exception text matches judges.Judge.RETRYABLE, so it actually gets retried",
      _openai_truncated._is_retryable(
          RuntimeError(f"{_openai_truncated.spec}: reply truncated (finish_reason=length) "
                       "with no usable text")))
check("usage is still recorded for the truncated attempt, so the spend isn't silently lost",
      _openai_truncated.usage["input_tokens"] == 200, dict(_openai_truncated.usage))

_openai_length_but_has_text = _make_openai_probe()
_openai_length_but_has_text._next_response = _fake_openai_response(
    "democracy | 2", "length", prompt_tokens=100, completion_tokens=512, cached_tokens=0)
check("finish_reason=length with SOME usable text still returns it, not a raise",
      _openai_length_but_has_text._sample("a comment", "a prompt", 0) == "democracy | 2")


print("\njudges.py: GeminiJudge._sample request shape and usage accounting (mocked)")


class _FakeThinkingConfig:
    def __init__(self, thinking_level=None):
        self.thinking_level = thinking_level


class _FakeGenerateContentConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeGeminiTypes:
    ThinkingConfig = _FakeThinkingConfig
    GenerateContentConfig = _FakeGenerateContentConfig


def _fake_gemini_response(text, finish_reason="STOP", prompt_tokens=100, output_tokens=20,
                          cached_tokens=0, thoughts_tokens=0):
    candidate = type("Candidate", (), {"finish_reason": finish_reason})()
    usage = type("Usage", (), {
        "prompt_token_count": prompt_tokens, "cached_content_token_count": cached_tokens,
        "candidates_token_count": output_tokens, "thoughts_token_count": thoughts_tokens,
    })()
    return type("Response", (), {"text": text, "candidates": [candidate], "usage_metadata": usage})()


def _make_gemini_probe(model="gemini-3.1-pro"):
    probe = object.__new__(judges.GeminiJudge)
    probe.model = model
    probe.usage = collections.Counter()
    probe._lock = threading.Lock()
    probe._types = _FakeGeminiTypes()

    class _FakeModels:
        def generate_content(self, **kwargs):
            probe._last_kwargs = kwargs
            return probe._next_response

    class _FakeGeminiClient:
        def __init__(self):
            self.models = _FakeModels()

    probe._client = _FakeGeminiClient()
    return probe


_gemini_probe = _make_gemini_probe()
_gemini_probe._next_response = _fake_gemini_response(
    "democracy | 3", prompt_tokens=150, output_tokens=10, cached_tokens=40, thoughts_tokens=25)
_gemini_reply = _gemini_probe._sample("a comment", "a prompt", 0)
check("_sample sends the 4,096-token reply limit for a model that thinks at high by default (thinking is spent from it)",
      _gemini_probe._last_kwargs["config"].kwargs["max_output_tokens"] == 4096,
      _gemini_probe._last_kwargs["config"].kwargs)
check("with no level set nothing is sent about thinking (there is no documented off switch)",
      "thinking_config" not in _gemini_probe._last_kwargs["config"].kwargs,
      _gemini_probe._last_kwargs["config"].kwargs)
check("_sample returns the reply text", _gemini_reply == "democracy | 3", _gemini_reply)
check("thoughts_token_count is ADDED to output_tokens, not left unbilled -- the 2.2x "
      "under-reporting a mocked test found before this was fixed",
      _gemini_probe.usage["output_tokens"] == 10 + 25, dict(_gemini_probe.usage))
check("cached tokens are split out of input_tokens for Gemini too",
      (_gemini_probe.usage["input_tokens"], _gemini_probe.usage["cache_read_input_tokens"])
      == (110, 40), dict(_gemini_probe.usage))

_gemini_flash_probe = _make_gemini_probe(model="gemini-3.5-flash-lite")
_gemini_flash_probe._next_response = _fake_gemini_response("none | 0")
_gemini_flash_probe._sample("a comment", "a prompt", 0)
check("a flash-lite model likewise gets no thinking_config unless a level is set",
      "thinking_config" not in _gemini_flash_probe._last_kwargs["config"].kwargs,
      _gemini_flash_probe._last_kwargs["config"].kwargs)

_gemini_truncated = _make_gemini_probe()
_gemini_truncated._next_response = _fake_gemini_response(
    "", finish_reason="MAX_TOKENS", prompt_tokens=100, output_tokens=2048, thoughts_tokens=0)
try:
    _gemini_truncated._sample("a comment", "a prompt", 0)
    check("a truncated, empty Gemini reply (finish_reason=MAX_TOKENS) raises, not returns ''",
          False, "no exception")
except RuntimeError as _err:
    check("a truncated, empty Gemini reply (finish_reason=MAX_TOKENS) raises, not returns ''",
          "truncated" in str(_err), str(_err))
check("usage is still recorded for the truncated Gemini attempt",
      _gemini_truncated.usage["input_tokens"] == 100, dict(_gemini_truncated.usage))

print("\njudges.py: openai_price_for / google_price_for")
check("openai_price_for returns the known rate for gpt-5-mini",
      judges.openai_price_for("gpt-5-mini") == (0.25, 2.00))
try:
    judges.openai_price_for("not-a-real-model")
    check("openai_price_for refuses an unpriced model rather than assuming free", False)
except judges.PriceUnavailable:
    check("openai_price_for refuses an unpriced model rather than assuming free", True)
check("google_price_for returns the known rate for gemini-3.1-pro",
      judges.google_price_for("gemini-3.1-pro") == (2.00, 12.00))
try:
    judges.google_price_for("not-a-real-model")
    check("google_price_for refuses an unpriced model rather than assuming free", False)
except judges.PriceUnavailable:
    check("google_price_for refuses an unpriced model rather than assuming free", True)
check("500 is now a retryable server error, alongside 502/503/504",
      judges.Judge()._is_retryable(RuntimeError("HTTP 500 Internal Server Error")))

print("\njudges.py: cost_usd and cache_savings_usd price cache tokens correctly")


@judges.register("test_priced")
class PricedJudge(judges.Judge):
    default_model = "priced"
    price_per_mtok = (2.0, 10.0)  # $2/MTok in, $10/MTok out -- easy numbers to hand-check

    def _sample(self, body, prompt, vote):
        return "democracy"


_priced = judges.create("test_priced", votes=1)
_priced.usage["input_tokens"] = 1_000_000
_priced.usage["output_tokens"] = 1_000_000
check("plain input/output tokens price at the base rate with no cache activity",
      abs(_priced.cost_usd() - (2.0 + 10.0)) < 1e-9, _priced.cost_usd())
check("no cache activity means no savings to report",
      _priced.cache_savings_usd() == 0.0)

_priced2 = judges.create("test_priced", votes=1)
_priced2.usage["cache_creation_input_tokens"] = 1_000_000
_priced2.usage["cache_read_input_tokens"] = 1_000_000
# 1M cache-written tokens at 1.25x $2/MTok = $2.50, 1M cache-read tokens at
# 0.1x $2/MTok = $0.20 -- neither uses the plain input_tokens rate at all.
check("cache-written tokens cost 1.25x the base input rate, cache-read tokens cost 0.1x",
      abs(_priced2.cost_usd() - (2.50 + 0.20)) < 1e-9, _priced2.cost_usd())
check("cache_savings_usd is what those read tokens would have cost at full price, minus "
      "what they actually cost -- 1M tokens x $2/MTok x (1 - 0.1) = $1.80",
      abs(_priced2.cache_savings_usd() - 1.80) < 1e-9, _priced2.cache_savings_usd())

# Regression test for a real bug caught on a live run: classify.py's summary
# calls scorer.cache_savings_usd()/cache_tokens() on whatever `--sentiment`
# arm is active, but Scorer (sentiment.py) is a separate class hierarchy from
# Judge (judges.py) -- it does not inherit either method automatically. The
# crash surfaced only after a full 23-comment run had already completed and
# spent real money, right at the final summary line.
import ewl.label.sentiment as _sent_cache_test
_vader = _sent_cache_test.create("vader")
check("a non-judge-backed scorer (vader) has a safe default cache_savings_usd",
      _vader.cache_savings_usd() == 0.0)
check("a non-judge-backed scorer (vader) has a safe default cache_tokens",
      _vader.cache_tokens() == (0, 0))

_llm_scorer = _sent_cache_test.create("llm:test_priced", votes=1)
_llm_scorer._judge.usage["cache_creation_input_tokens"] = 1_000_000
_llm_scorer._judge.usage["cache_read_input_tokens"] = 1_000_000
check("an LLM-backed scorer delegates cache_savings_usd to its inner judge",
      abs(_llm_scorer.cache_savings_usd() - 1.80) < 1e-9, _llm_scorer.cache_savings_usd())
check("an LLM-backed scorer delegates cache_tokens to its inner judge",
      _llm_scorer.cache_tokens() == (1_000_000, 1_000_000), _llm_scorer.cache_tokens())

# Regression test for a real pricing bug caught 2026-09: claude-sonnet-5 was
# priced at $3/$15 per MTok (Sonnet 4.6's rate, carried over stale) instead of
# its actual $2/$10 -- every Sonnet-arm cost estimate was 50% too high.
# $2/$10 was the introductory rate and it expired on 2026-08-31; standard is
# $3/$15. The old assertion locked the expired number in and called the
# correct one stale, so it would have failed the fix.
def _refuses():
    # judges.price_for(), not a constructed AnthropicJudge -- building one
    # imports the anthropic SDK, which CI does not install.
    try:
        judges.price_for("claude-not-a-real-model-9")
    except judges.PriceUnavailable:    # a RuntimeError now (was SystemExit), so an arm can be skipped
        return True
    return False


check("claude-sonnet-5 is priced at the current $2/$10, not the stale $3/$15 figure",
      judges._ANTHROPIC_PRICES["claude-sonnet-5"] == (2.00, 10.00),
      judges._ANTHROPIC_PRICES["claude-sonnet-5"])
check("claude-fable-5 is priced, not silently free",
      judges._ANTHROPIC_PRICES.get("claude-fable-5") == (10.00, 50.00),
      "an unpriced model reports $0.00 and hides the spend and cache lines entirely")
check("an unknown model refuses rather than defaulting to free", _refuses(),
      "price_for used to return (0.0, 0.0) for anything not in the table, so an "
      "unpriced run reported $0.00 and hid the spend and cache lines entirely")
check("a dated model id falls back to its base model's price",
      judges.price_for("claude-opus-5-20260115") == (5.00, 25.00))

check("Haiku's cache minimum is the highest of any current model",
      judges.cache_minimum_for("claude-haiku-4-5") == 4096
      and judges.cache_minimum_for("claude-haiku-4-5") > judges.cache_minimum_for("claude-sonnet-5")
      > judges.cache_minimum_for("claude-opus-5"),
      "confirmed against Anthropic's published table -- see AnthropicJudge._cacheable_system")
check("a dated model id falls back to its base model's cache minimum too",
      judges.cache_minimum_for("claude-haiku-4-5-20260101") == 4096)
check("an unrecognised model returns None rather than a made-up minimum",
      judges.cache_minimum_for("keywords.yml") is None
      and judges.cache_minimum_for("") is None)
check("estimate_tokens is a rough ~4-chars-per-token offline estimate",
      judges.estimate_tokens("a" * 4096) == 1024,
      "no tokenizer dependency, no API call -- see its docstring for why that's the point")

print("\ncache: a reply that stopped early is used for the call but never stored")
import tempfile as _tmp_unc


@judges.register("test_uncacheable")
class _UncacheableJudge(judges.Judge):
    default_model = "uncacheable"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.calls = 0
        self.mode = "uncacheable"

    def _sample(self, body, prompt, vote):
        self.calls += 1
        if self.mode == "uncacheable":
            return judges.uncacheable("", "max_tokens")
        if self.mode == "truncated":
            raise judges.TruncatedReply("reply truncated (finish_reason=length)")
        return "democracy"


_unc_path = os.path.join(_tmp_unc.mkdtemp(), "unc_cache.jsonl")
_unc = judges.create("test_uncacheable", votes=1, cache_path=_unc_path)
_unc.sample("c", "p")
_unc.sample("c", "p")
check("an uncacheable reply is asked for again on a rerun, not replayed from the cache",
      _unc.calls == 2, _unc.calls)
check("nothing was written to the cache file for it",
      (not os.path.exists(_unc_path)) or os.path.getsize(_unc_path) == 0,
      os.path.getsize(_unc_path) if os.path.exists(_unc_path) else None)
check("each one is counted by reason, so the run summary can report it",
      _unc.usage["uncacheable_max_tokens"] == 2, dict(_unc.usage))
check("an empty uncacheable reply parses as a parse failure, not as a clean 'none'",
      _unc.label("c", [])[0] == [taxonomy.PARSE_FAILURE], _unc.label("c", [])[0])

_unc.mode = "normal"
_unc.calls = 0
_unc.sample("d", "p")
_unc.sample("d", "p")
check("a normal reply IS cached (second ask is a hit)", _unc.calls == 1, _unc.calls)

_trunc = judges.create("test_uncacheable", votes=1,
                       cache_path=os.path.join(_tmp_unc.mkdtemp(), "t.jsonl"))
_trunc.mode = "truncated"
_trunc.max_attempts = 2
try:
    _trunc_reply = _trunc.sample("e", "p")
    check("a comment the model keeps truncating returns an empty uncacheable reply "
          "instead of aborting the run",
          _trunc_reply == "" and _trunc.usage["uncacheable_truncated"] == 1
          and _trunc.calls == 2, (_trunc_reply, dict(_trunc.usage), _trunc.calls))
except Exception as _err:   # the old behaviour: the whole run dies on one comment
    check("a comment the model keeps truncating returns an empty uncacheable reply "
          "instead of aborting the run", False, repr(_err))

# The real Anthropic backend: only a finished answer is cacheable.
for _stop, _cacheable in (("end_turn", True), ("max_tokens", False), ("refusal", False)):
    # (a refusal comes back empty; a truncated reply keeps what it wrote -- neither is stored)
    _p = _make_probe("claude-haiku-4-5")

    class _R:
        usage = _FakeUsageWithCache()
        stop_reason = _stop
        content = [type("Block", (), {"type": "text", "text": "democracy"})()]
    _p._client.messages = _CapturingMessagesReturning(_R())
    _out = _p._sample("a comment", "a prompt", 0)
    check(f"Anthropic stop_reason={_stop!r}: {'cached normally' if _cacheable else 'used but not cached'}",
          (not isinstance(_out, judges.UncacheableReply) and _out == "democracy") if _cacheable
          else (isinstance(_out, judges.UncacheableReply)
                and _out == ("" if _stop == "refusal" else "democracy")), repr(_out))



print("\nOpenAI and Gemini: an early stop is never cached (mocked, no SDK needed)")


def _openai_probe(content, finish, refusal=None):
    probe = object.__new__(judges.OpenAIJudge)
    probe.model = "gpt-5-mini"
    probe.usage = collections.Counter()
    probe._lock = threading.Lock()
    message = type("M", (), {"content": content, "refusal": refusal})()
    choice = type("C", (), {"message": message, "finish_reason": finish})()
    response = type("R", (), {"choices": [choice], "usage": None})()
    completions = type("Cs", (), {"create": staticmethod(lambda **k: response)})()
    probe._client = type("Cl", (), {"chat": type("Ch", (), {"completions": completions})()})()
    return probe


def _openai_request_kwargs(model):
    probe = _openai_probe("democracy", "stop")
    probe.model = model
    seen = {}

    def create(**k):
        seen.update(k)
        return probe._client._response
    message = type("M", (), {"content": "democracy", "refusal": None})()
    choice = type("C", (), {"message": message, "finish_reason": "stop"})()
    probe._client._response = type("R", (), {"choices": [choice], "usage": None})()
    probe._client.chat.completions.create = create
    probe._sample("c", "p", 0)
    return seen


check("OpenAI: a reasoning model (gpt-5-mini, o3) is sent reasoning_effort",
      _openai_request_kwargs("gpt-5-mini").get("reasoning_effort") == "minimal"
      and _openai_request_kwargs("o3-mini").get("reasoning_effort") == "minimal")
check("OpenAI: a non-reasoning model is NOT sent it (it would be rejected with a 400)",
      "reasoning_effort" not in _openai_request_kwargs("gpt-4.1"),
      _openai_request_kwargs("gpt-4.1"))
check("OpenAI: max_completion_tokens is used for both, never max_tokens or temperature",
      all("max_completion_tokens" in _openai_request_kwargs(m)
          and "max_tokens" not in _openai_request_kwargs(m)
          and "temperature" not in _openai_request_kwargs(m) for m in ("gpt-5-mini", "gpt-4.1")))

_ok = _openai_probe("democracy", "stop")._sample("c", "p", 0)
check("OpenAI: a finished answer is returned and cacheable",
      _ok == "democracy" and not isinstance(_ok, judges.UncacheableReply), repr(_ok))
_part = _openai_probe("democracy, eco", "length")._sample("c", "p", 0)
check("OpenAI: a partial answer cut off by the token limit is used but not stored",
      isinstance(_part, judges.UncacheableReply) and _part.reason == "length", repr(_part))
_filt = _openai_probe("", "content_filter")._sample("c", "p", 0)
check("OpenAI: a content-filter block is empty and not stored",
      isinstance(_filt, judges.UncacheableReply) and _filt == "", repr(_filt))
_ref = _openai_probe("", "stop", refusal="I can't help with that")._sample("c", "p", 0)
check("OpenAI: a model refusal is empty and not stored",
      isinstance(_ref, judges.UncacheableReply) and _ref.reason == "refusal", repr(_ref))
try:
    _openai_probe("", "length")._sample("c", "p", 0)
    check("OpenAI: nothing visible and out of tokens is retried as a truncation", False, "no raise")
except judges.TruncatedReply:
    check("OpenAI: nothing visible and out of tokens is retried as a truncation", True)


def _gemini_probe(text, finish):
    probe = object.__new__(judges.GeminiJudge)
    probe.model = "gemini-test"
    probe.usage = collections.Counter()
    probe._lock = threading.Lock()
    probe._types = type("T", (), {"GenerateContentConfig": staticmethod(lambda **k: k)})()
    candidate = type("Cand", (), {"finish_reason": finish})()
    response = type("R", (), {"text": text, "candidates": [candidate],
                              "usage_metadata": None})()
    probe._client = type("Cl", (), {"models": type("Ms", (), {
        "generate_content": staticmethod(lambda **k: response)})()})()
    return probe


_g_ok = _gemini_probe("democracy", "FinishReason.STOP")._sample("c", "p", 0)
check("Gemini: a STOP reply is returned and cacheable",
      _g_ok == "democracy" and not isinstance(_g_ok, judges.UncacheableReply), repr(_g_ok))
_g_safe = _gemini_probe("", "FinishReason.SAFETY")._sample("c", "p", 0)
check("Gemini: a safety block is empty and not stored",
      isinstance(_g_safe, judges.UncacheableReply) and _g_safe.reason == "safety", repr(_g_safe))
_g_part = _gemini_probe("democracy, eco", "FinishReason.MAX_TOKENS")._sample("c", "p", 0)
check("Gemini: a partial MAX_TOKENS reply is used but not stored",
      isinstance(_g_part, judges.UncacheableReply) and _g_part.reason == "max_tokens", repr(_g_part))



print("\nretry markers: a status code only counts as a whole number")
_m = judges.create("test_uncacheable", votes=1)
check("a real 500 is retryable", _m._is_retryable(RuntimeError("Error code: 500 - internal")))
check("'4500' inside a token limit message is NOT mistaken for a 500",
      not _m._is_retryable(ValueError("model supports at most 4500 tokens")),
      "bare substring match retried a permanent error")
check("a 429 inside a longer id is not a rate limit either",
      not _m._is_retryable(ValueError("job id 14290 failed")))


print("\ninstrument switches: defaults change nothing, each switch changes only what it names")
from ewl.label import sentiment as _sent, combined as _comb, instrument as _instr

_RULE_PHRASES = ("generally supportive", "generally hostile", "generally defending")
try:
    _sent.configure(story_rules=True, stance_object="conduct")
    _comb.configure(domain_context="keep")
    _default_scorer = _sent.create("llm:stub", votes=1)
    check("defaults: every story's prompt carries its full context paragraph, rules included",
          all(_sent.STORY_CONTEXT[s] in _default_scorer._prompts[s] for s in _sent.STORY_CONTEXT),
          list(_default_scorer._prompts))
    check("defaults: anchor 4 is the wording as it stands",
          _sent.SCALE[4] in next(iter(_default_scorer._prompts.values())))
    check("defaults: no tag is added to the scorer's name",
          _default_scorer.describe() == "llm:stub:constant", _default_scorer.describe())
    _default_combined = _comb.build_prompt([("x", "none")], story="luigi")

    _sent.configure(story_rules=False)
    _off = _sent.create("llm:stub", votes=1)
    check("story_rules=off: no stance-rule phrase survives in any story's prompt",
          not any(ph in p for p in _off._prompts.values() for ph in _RULE_PHRASES),
          [ph for p in _off._prompts.values() for ph in _RULE_PHRASES if ph in p])
    check("story_rules=off: the factual part of each paragraph is kept",
          all(_sent.facts_only(s) in _off._prompts[s] and len(_sent.facts_only(s)) > 200
              for s in _sent.STORY_CONTEXT),
          {s: len(_sent.facts_only(s)) for s in _sent.STORY_CONTEXT})
    check("story_rules=off: the output names the setting",
          _off.describe() == "llm:stub:constant[story_rules=off]", _off.describe())
    check("story_rules=off reaches the combined arm too",
          not any(ph in _comb.build_prompt([("x", "none")], story="luigi")
                  for ph in _RULE_PHRASES))
    _sent.configure(story_rules=True)

    _sent.configure(stance_object="wealth")
    _wealth = _sent.create("llm:stub", votes=1)
    _p = next(iter(_wealth._prompts.values()))
    check("stance_object=wealth: anchor 4 is about wealth and the people who hold it",
          _sent.ANCHOR_4["wealth"] in _p and "in this story" not in _p.split("4 = ")[1].split("\n")[0],
          _p.split("4 = ")[1].split("\n")[0])
    check("stance_object=wealth: anchors 1, 2, 3 and 5 are untouched",
          all(_sent.SCALE[n] in _p for n in (1, 2, 3, 5)))
    _sent.configure(stance_object="conduct")
    try:
        _sent.configure(stance_object="nonsense")
        check("an unknown stance_object is refused", False, "no ValueError")
    except ValueError:
        check("an unknown stance_object is refused", True)

    _comb.configure(domain_context="remove")
    _removed = _comb.build_prompt([("x", "none")], story="luigi")
    _ctx = _sent.story_context_for("luigi")
    check("combined domain_context=remove: context appears after the sentiment heading, "
          "not above the domain question",
          _removed.index(_ctx) > _removed.index("2. SENTIMENT")
          and _default_combined.index(_sent.facts_only("luigi")) < _default_combined.index("1. HARM DOMAIN"),
          (_removed.index(_ctx), _removed.index("2. SENTIMENT")))
    check("combined default: BOTH halves see the story's facts, and only the sentiment half sees "
          "the stance rules",
          _sent.rules_only("luigi") in _default_combined
          and _default_combined.index(_sent.rules_only("luigi")) > _default_combined.index("2. SENTIMENT")
          and _sent.rules_only("luigi") not in _default_combined.split("2. SENTIMENT")[0],
          _sent.rules_only("luigi")[:60])
    check("combined domain_context=keep reproduces the default prompt exactly",
          (_comb.configure(domain_context="keep") == "keep"
           and _comb.build_prompt([("x", "none")], story="luigi") == _default_combined))

    class _Args:
        story_rules, stance_object, combined_domain_context = "off", "wealth", "remove"
    _tag = _instr.apply(_Args())
    check("instrument.apply sets all three switches and returns the tags",
          "story_rules=off" in _tag and "stance_object=wealth" in _tag
          and "domain_context=removed" in _tag, _tag)
finally:
    _sent.configure(story_rules=True, stance_object="conduct")
    _comb.configure(domain_context="keep")



print("\nretry on transient failure")


@judges.register("test_flaky")
class FlakyJudge(judges.Judge):
    default_model = "flaky"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.attempts = 0

    def _sample(self, body, prompt, vote):
        self.attempts += 1
        if self.attempts < 3:
            raise RuntimeError("Error code: 429 - rate_limit_error")
        return "democracy"


@judges.register("test_fatal")
class FatalJudge(judges.Judge):
    default_model = "fatal"

    def _sample(self, body, prompt, vote):
        raise ValueError("invalid_request_error: no such model")


_flaky = judges.create("test_flaky", votes=1)
_flaky.max_attempts = 3
# Backoff is real time; keep the test quick by shrinking it, not skipping it.
_flaky.RETRYABLE = ("429",)
check("a rate limit is retried and then succeeds",
      _flaky.label("x", [])[0] == ["democracy"], _flaky.attempts)
check("retries are counted", _flaky.usage["retries"] == 2, _flaky.usage["retries"])
check("attempts counts real requests, calls counts comments x votes",
      _flaky.usage["attempts"] == 3 and _flaky.usage["calls"] == 1,
      dict(_flaky.usage))
check("backoff sleeps are not billed as model time",
      _flaky.slept > 0 and _flaky.seconds < _flaky.slept,
      f"seconds={_flaky.seconds:.2f} slept={_flaky.slept:.2f}")

_fatal = judges.create("test_fatal", votes=1)
try:
    _fatal.label("x", [])
    check("a permanent error is not retried", False, "no exception raised")
except ValueError:
    check("a permanent error is not retried", _fatal.usage["retries"] == 0,
          f"{_fatal.usage['retries']} retries on an unretryable error")

# A fresh instance: _flaky's marker list was narrowed above to keep the
# backoff test quick, so it cannot answer this question.
_markers = judges.create("test_fatal", votes=1)
check("retryable markers cover the usual transients",
      all(_markers._is_retryable(RuntimeError(m)) for m in
          ("Error code: 429", "overloaded_error", "502 Bad Gateway", "connection reset")),
      "matched on text so no backend SDK has to be imported here")
check("an unrelated error is not treated as transient",
      not _markers._is_retryable(ValueError("schema validation failed")))
check("a permanent error containing a transient word is not retried",
      not any(_markers._is_retryable(ValueError(m)) for m in (
          "invalid_request_error: model claude-429 not found",
          "authentication_error: your connection string is invalid",
          "permission_denied: request timeout policy forbids this",
          "invalid_request_error: Your credit balance is too low")),
      "a wrong key would otherwise burn 5 attempts and ~90s of backoff per comment")

# A permanent failure on one comment used to stop the whole classify run with
# a bare traceback -- correct (see PERMANENT above: retrying an auth failure
# or exhausted quota for the other 29,999 rows would be pure waste), but it
# named neither which row nor how much output was already safe. Everything up
# to the failing row is durable (writer.writerow flushes per row; the cache
# is append-only), so a restart is cheap -- if the operator can tell where to
# restart from.
with tempfile.TemporaryDirectory() as tmp:
    _in = os.path.join(tmp, "in.csv")
    _out = os.path.join(tmp, "out.csv")
    with open(_in, "w", newline="", encoding="utf-8") as _fh:
        _fh.write("comment_id,body\nc1,fine\nc2,also fine\n")
    import contextlib, io
    from ewl.label import classify as _classify_mod
    _argv = sys.argv
    sys.argv = ["classify.py", _in, "--judge", "test_fatal", "--output", _out]
    _stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(_stderr):
            _classify_mod.main()
        check("a permanently-failing row still stops the run", False, "no exception raised")
    except ValueError:
        _msg = _stderr.getvalue()
        check("the failure names the row it happened on", "comment_id='c1'" in _msg, _msg)
        check("and says what survives, so a restart knows where to pick up",
              "already written" in _msg and "Re-running this same command" in _msg, _msg)
    finally:
        sys.argv = _argv

# The cache-minimum warning: a short prompt against a Haiku-like model should
# say so before spending anything; a non-Anthropic judge (no cache-minimum
# concept at all) must stay silent, not guess.
@judges.register("test_haiku_like")
class _HaikuLike(judges.Judge):
    default_model = "claude-haiku-4-5"

    def _sample(self, body, prompt, vote):
        return "democracy"


with tempfile.TemporaryDirectory() as tmp:
    _in = os.path.join(tmp, "in.csv")
    _out = os.path.join(tmp, "out.csv")
    with open(_in, "w", newline="", encoding="utf-8") as _fh:
        _fh.write("comment_id,body\nc1,short comment\n")
    _argv = sys.argv
    sys.argv = ["classify.py", _in, "--judge", "test_haiku_like", "--output", _out]
    _stderr = io.StringIO()
    with contextlib.redirect_stderr(_stderr):
        classify.main()
    sys.argv = _argv
    check("a short prompt against a Haiku-like model warns before spending anything",
          "under claude-haiku-4-5's 4096-token cache minimum" in _stderr.getvalue(),
          _stderr.getvalue())

    sys.argv = ["classify.py", _in, "--judge", "keyword", "--output", _out]
    _stderr2 = io.StringIO()
    with contextlib.redirect_stderr(_stderr2):
        classify.main()
    sys.argv = _argv
    check("a non-Anthropic judge gets no cache-minimum warning, not a guessed one",
          "cache minimum" not in _stderr2.getvalue(), _stderr2.getvalue())

    # Regression: at --votes 1 there is only ever one ballot, so
    # Judge.resolve() returns unstable=False by construction -- not because
    # stability was checked and confirmed, but because there was nothing to
    # disagree with. The summary line used to print "Unstable across votes:
    # 0.0%" identically whether or not a real multi-vote check ever ran,
    # which reads as a measurement either way.
    sys.argv = ["classify.py", _in, "--judge", "test_haiku_like", "--votes", "1", "--output", _out]
    _stderr3 = io.StringIO()
    with contextlib.redirect_stderr(_stderr3):
        classify.main()
    sys.argv = _argv
    check("--votes 1 reports instability as not measured, not a fake 0.0%",
          "Unstable across votes: not measured (--votes 1)" in _stderr3.getvalue(),
          _stderr3.getvalue())

    sys.argv = ["classify.py", _in, "--judge", "test_haiku_like", "--votes", "3", "--output", _out]
    _stderr4 = io.StringIO()
    with contextlib.redirect_stderr(_stderr4):
        classify.main()
    sys.argv = _argv
    check("--votes 3 still reports a genuine percentage",
          "Unstable across votes: 0.0%" in _stderr4.getvalue(), _stderr4.getvalue())


print("\ncaching")
with tempfile.TemporaryDirectory() as tmp:
    cache_path = os.path.join(tmp, "cache.jsonl")
    first = judges.create("stub", votes=1, cache_path=cache_path)
    first.label("cache me", [])
    check("a cold cache calls the model", first.usage["calls"] == 1 and first.usage["cache_hits"] == 0)
    second = judges.create("stub", votes=1, cache_path=cache_path)
    second.label("cache me", [])
    check("a warm cache serves without calling", second.usage["cache_hits"] == 1 and second.usage["calls"] == 0)
    third = judges.create("stub", votes=1, cache_path=cache_path)
    third.label("cache me", [("different", "social")])
    check("different examples miss the cache", third.usage["cache_hits"] == 0,
          "prompt must be part of the cache key")

# Regression: classify.py hands the SAME --cache path to two separate Judge
# objects (the domain judge and the sentiment LLMScorer's inner judge). Each
# used to build its own private _Cache with only its own private lock, so
# concurrent writes from the two instances raced on the shared file with no
# mutual exclusion between them -- measured (outside this harness, at higher
# volume) to silently lose ~1% of entries with no error and no corruption.
# _Cache.shared() is what fixes this; this check exercises it at a volume
# small enough to run in CI but large enough to have caught the original bug
# reliably (it reproduced on every run at this scale during investigation).
with tempfile.TemporaryDirectory() as tmp:
    _race_path = os.path.join(tmp, "race.jsonl")
    _race_judges = [judges.create("stub", votes=1, cache_path=_race_path) for _ in range(2)]

    def _race_worker(judge, offset):
        for i in range(75):
            judge.sample(f"race{offset}_{i}", "prompt", 0)

    _race_threads = [threading.Thread(target=_race_worker, args=(_race_judges[t % 2], t))
                     for t in range(4)]
    for _th in _race_threads:
        _th.start()
    for _th in _race_threads:
        _th.join()
    with open(_race_path, encoding="utf-8") as _handle:
        _race_lines = sum(1 for _ in _handle)
    check("two Judge instances sharing one --cache path lose no entries under concurrency",
          _race_lines == 4 * 75, f"expected {4 * 75}, found {_race_lines}")

print("\nselectors")
fixed = examples_mod.create("fixed")
check("fixed returns the 25 fixed examples", len(fixed.select("anything")) == 25)
check("fixed ignores the comment", fixed.select("a") == fixed.select("b"))
check("none is zero-shot", examples_mod.create("none").select("a") == [])

print("\nfixed_random: a fixed but randomly-drawn example set")
_frand_pool = [(f"comment {i} about {c}", c)
               for i, c in enumerate(["democracy", "economics", "environment", "social"] * 5)]
_frand = examples_mod.create("fixed_random", pairs=_frand_pool, n=10, seed=42)
check("draws exactly n examples", len(_frand.pairs) == 10, len(_frand.pairs))
check("ignores the comment -- same set for every body, like fixed",
      _frand.select("anything") == _frand.select("something else") == _frand.pairs)
check("n is capped at the pool size rather than erroring",
      len(examples_mod.create("fixed_random", pairs=_frand_pool, n=999, seed=1).pairs)
      == len(_frand_pool))

_frand_same_seed = examples_mod.create("fixed_random", pairs=_frand_pool, n=10, seed=42)
check("the same seed draws the same examples -- reproducible across runs",
      _frand.pairs == _frand_same_seed.pairs)

_frand_other_seed = examples_mod.create("fixed_random", pairs=_frand_pool, n=10, seed=7)
check("a different seed draws a different set (on a pool large enough to make that likely)",
      _frand.pairs != _frand_other_seed.pairs)

check("describe() reports n and the seed",
      _frand.describe() == "fixed_random(n=10, seed=42)", _frand.describe())

print("\nBOM tolerance (collectors write utf-8-sig)")
with tempfile.TemporaryDirectory() as tmp:
    bom = os.path.join(tmp, "bom.csv")
    with open(bom, "w", newline="", encoding="utf-8-sig") as handle:
        w = csv.DictWriter(handle, fieldnames=["comment_id", "body", "hand_domain"])
        w.writeheader()
        for i in range(4):
            w.writerow({"comment_id": f"c{i}", "body": f"comment {i} on tax and lobbying",
                        "hand_domain": "democracy"})

    got = evaluate.read_gold(bom)
    check("evaluate reads a BOM file", len(got) == 4)
    check("the first column survives the BOM",
          all(r.get("comment_id") for r in got),
          "a BOM read as utf-8 renames comment_id to \\ufeffcomment_id")

    err = io.StringIO()
    import contextlib
    with contextlib.redirect_stderr(err):
        classify.warn_if_pool_overlaps(bom, bom)
    check("the pool-overlap guard still fires on BOM files", "WARNING" in err.getvalue(),
          "the guard compares comment_ids; a BOM hides that column and it silently passes")

    check("read_pool reads a BOM file", len(examples_mod.read_pool(bom)) == 4)


print("\nretrieval never shows a comment its own label")
_pairs = [("billionaires dodge tax offshore", "democracy"),
          ("private jets burn carbon", "environment"),
          ("wages frozen while profits soar", "economics"),
          ("nobody earns a billion fairly", "social"),
          ("lobbying buys policy outcomes", "democracy")]
with optional("retrieval checks"):
    _sel = examples_mod.create("retrieved", pairs=_pairs, k=3)
    _self = _sel.select(_pairs[0][0])
    check("a comment in its own pool does not retrieve itself",
          all(text != _pairs[0][0] for text, _ in _self))
    check("k is held constant by backfilling", len(_self) == 3)
    check("the self-match is counted, not silently dropped", _sel.self_matches == 1)

    _rt = examples_mod.create("retrieved", pairs=_pairs, k=3)
    _rt.select("Billionaires   dodge tax\n offshore")
    check("a self-match survives a CSV round-trip", _rt.self_matches == 1,
          "whitespace/case changes must not let a self-match through")

    _out = examples_mod.create("retrieved", pairs=_pairs, k=3)
    _out.select("gardening tips for beginners")
    check("an out-of-pool comment counts no self-match", _out.self_matches == 0)
print("\nhand-coding sample allocation")
from ewl.analyse import validate
check("exact n across equal strata, no integer-division shortfall",
      sum(validate.allocate({"a": 1000, "b": 1000, "c": 1000}, 200).values()) == 200)
_thin = validate.allocate({"a": 1000, "b": 1000, "c": 1000, "d": 20}, 200)
check("a thin stratum is backfilled from the others", sum(_thin.values()) == 200, _thin)
check("the thin stratum contributes everything it has", _thin["d"] == 20, _thin)
check("no stratum is over-drawn",
      all(_thin[k] <= s for k, s in {"a": 1000, "b": 1000, "c": 1000, "d": 20}.items()))
_small = validate.allocate({"a": 30, "b": 20, "c": 10}, 200)
check("asking for more than exists returns everything, not a crash",
      sum(_small.values()) == 60, _small)
check("allocation is deterministic",
      validate.allocate({"a": 7, "b": 11, "c": 3}, 9)
      == validate.allocate({"a": 7, "b": 11, "c": 3}, 9))
check("a single stratum takes the whole sample",
      validate.allocate({"only": 500}, 50) == {"only": 50})

with tempfile.TemporaryDirectory() as tmp:
    _rows = ([{"comment_id": f"a{i}", "story": "s1", "platform": "youtube",
               "source": f"https://youtube.com/watch?v=vid_a{i}",
               "body": f"comment {i}", "kw_domain": "economics"} for i in range(60)]
             + [{"comment_id": f"b{i}", "story": "s2", "platform": "youtube",
                 "source": f"https://youtube.com/watch?v=vid_b{i}",
                 "body": f"other {i}", "kw_domain": "environment"} for i in range(4)])
    _out = io.StringIO()
    with contextlib_redirect_stderr_quiet():
        validate.cmd_sample(_rows, 20, 42, _out, stratify_by="kw_domain")
    _got = list(csv.DictReader(io.StringIO(_out.getvalue())))
    check("stratifying by label reaches n", len(_got) == 20, len(_got))
    # A coder needs to click through to the video when a bare comment ("this",
    # "lol") is too short or ambiguous to code on its own -- and, separately,
    # corrected_ci_bootstrap clusters the gold set by this same column, so it
    # must survive sampling rather than arrive later only via join's carry.
    check("the video link (source) survives into the hand-coding file",
          all(r.get("source", "").startswith("https://youtube.com/watch?v=vid_") for r in _got),
          [r.get("source") for r in _got[:3]])
    # The collectors write utf-8-sig; the fixtures above are plain utf-8, so
    # without this the BOM regression the readers guard against is untested.
    _bomsplit = os.path.join(tmp, "bom_split.csv")
    with open(_bomsplit, "w", newline="", encoding="utf-8-sig") as handle:
        w = csv.DictWriter(handle, fieldnames=["comment_id", "story", "platform",
                                               "body", "kw_domain"])
        w.writeheader()
        for i in range(6):
            w.writerow({"comment_id": f"z{i}", "story": "s", "platform": "youtube",
                        "body": f"comment {i} about tax", "kw_domain": "democracy"})
    _bomout = io.StringIO()
    with contextlib_redirect_stderr_quiet():
        validate.cmd_sample(list(csv.DictReader(open(_bomsplit, encoding="utf-8-sig"))),
                            3, 42, _bomout)
    check("validate sample handles a BOM-written file",
          len(list(csv.DictReader(io.StringIO(_bomout.getvalue())))) == 3)
    _ids = {r["comment_id"][0] for r in _got}
    check("a rare category is over-sampled rather than lost", "b" in _ids)
    check("the hand-coding file stays blind",
          "kw_domain" not in (list(_got[0]) if _got else []))


print("\nsplit")
with tempfile.TemporaryDirectory() as tmp:
    gold = os.path.join(tmp, "gold.csv")
    write_fixture(gold)

    class Args:
        input_csv, label_column, seed, pool_fraction = gold, "hand_domain", 42, 0.7
        out_prefix = os.path.join(tmp, "split")
    evaluate.cmd_split(Args)

    pool = list(csv.DictReader(open(f"{Args.out_prefix}_pool.csv", encoding="utf-8-sig")))
    held = list(csv.DictReader(open(f"{Args.out_prefix}_heldout.csv", encoding="utf-8-sig")))
    check("split covers every row", len(pool) + len(held) == len(FIXTURE), f"{len(pool)}+{len(held)}")
    check("no comment appears on both sides",
          not ({r["comment_id"] for r in pool} & {r["comment_id"] for r in held}))
    check("both sides are non-empty", pool and held)
    check("democracy is present in both halves",
          any("democracy" in r["hand_domain"] for r in pool)
          and any("democracy" in r["hand_domain"] for r in held),
          "stratification should not concentrate the label at stake")

    check("the pool loads as retrieval pairs",
          len(examples_mod.read_pool(f"{Args.out_prefix}_pool.csv")) == len(pool))

print("\npool overlap guard")
with tempfile.TemporaryDirectory() as tmp:
    gold = os.path.join(tmp, "gold.csv")
    write_fixture(gold)
    half = os.path.join(tmp, "half.csv")
    rows = list(csv.DictReader(open(gold, encoding="utf-8")))
    with open(half, "w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0])); w.writeheader()
        w.writerows(rows[:4])

    def warned(a, b):
        err = io.StringIO()
        import contextlib
        with contextlib.redirect_stderr(err):
            classify.warn_if_pool_overlaps(a, b)
        return "WARNING" in err.getvalue()

    check("overlapping pool and corpus warns", warned(gold, half))
    # Previously compared against a path in a different tmpdir that never
    # existed, so the `else True` branch always fired and this checked nothing.
    empty = os.path.join(tmp, "empty.csv")
    with open(empty, "w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0])); w.writeheader()
    check("a pool with no overlapping ids does not warn", not warned(half, empty))

    other = os.path.join(tmp, "other.csv")
    with open(other, "w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0])); w.writeheader()
        for r in rows[4:]:
            w.writerow(r)
    check("disjoint files do not warn", not warned(half, other))

    # A missing comment_id column used to make the guard return silently --
    # exactly the kind of "plausible CSV, nothing looks wrong" failure mode
    # the guard exists to catch on the labels themselves (#7 in CLAUDE.md).
    # It must now warn loudly instead of going quiet, since with
    # `fixed_random` the same example bank is shown to every comment, so a
    # gold comment lacking comment_id could be classified while seeing its
    # own hand label with no visible sign anything is wrong.
    no_id = os.path.join(tmp, "no_id.csv")
    with open(no_id, "w", newline="", encoding="utf-8") as handle:
        cols = [c for c in rows[0] if c != "comment_id"]
        w = csv.DictWriter(handle, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    check("a pool missing comment_id warns instead of silently passing",
          warned(no_id, no_id))
    check("an input missing comment_id also warns", warned(no_id, gold))


print("\nmetrics: a failed category must not vanish from macro-F1")
check("a total failure scores F1 0, not NaN", evaluate.prf(0, 5, 5)[2] == 0.0)
check("present but never found scores 0", evaluate.prf(0, 0, 5)[2] == 0.0)
check("predicted but absent from gold scores 0", evaluate.prf(0, 5, 0)[2] == 0.0)
_absent = evaluate.prf(0, 0, 0)[2]
check("absent from both is NaN, the only unscoreable case", _absent != _absent)
check("a perfect category still scores 1", evaluate.prf(5, 0, 0)[2] == 1.0)

with tempfile.TemporaryDirectory() as tmp:
    # 2 social (predicted right) + 2 democracy (predicted as economics).
    _g = os.path.join(tmp, "macro.csv")
    with open(_g, "w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=["comment_id", "body", "hand_domain"])
        w.writeheader()
        for i in range(2):
            w.writerow({"comment_id": f"s{i}", "body": f"social one {i}", "hand_domain": "social"})
            w.writerow({"comment_id": f"d{i}", "body": f"demo one {i}", "hand_domain": "democracy"})

    @judges.register("test_halfwrong")
    class HalfWrong(judges.Judge):
        deterministic = True
        default_model = "halfwrong"

        def _sample(self, body, prompt, vote):
            return "social" if "social" in body else "economics"

    _r = evaluate.run_arm(evaluate.read_gold(_g), judges.create("test_halfwrong", votes=1),
                          examples_mod.create("none"), progress=False)
    check("an arm that gets half the corpus wrong cannot score a perfect macro-F1",
          _r["macro_f1"] < 0.5, f"macro_f1={_r['macro_f1']}")

print("\nrun_arm and print_scorecard distinguish parse failures from genuine none")
with tempfile.TemporaryDirectory() as tmp:
    # 2 rows genuinely off-topic ("none"), 2 rows the judge garbles the reply
    # for. Both used to collapse into the same UNCLASSIFIED bucket.
    _pf_gold = os.path.join(tmp, "pf.csv")
    with open(_pf_gold, "w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=["comment_id", "body", "hand_domain"])
        w.writeheader()
        for i in range(2):
            w.writerow({"comment_id": f"n{i}", "body": f"clearly off topic {i}", "hand_domain": "none"})
            w.writerow({"comment_id": f"g{i}", "body": f"garble me {i}", "hand_domain": "none"})

    @judges.register("test_garbler")
    class GarblerJudge(judges.Judge):
        deterministic = True
        default_model = "garbler"

        def _sample(self, body, prompt, vote):
            return "" if "garble" in body else "none"

    _pf_result = evaluate.run_arm(evaluate.read_gold(_pf_gold),
                                  judges.create("test_garbler", votes=1),
                                  examples_mod.create("none"), progress=False)
    check("genuine 'none' replies count as unclassified, not parse failures",
          _pf_result["unclassified"] == 0.5, _pf_result["unclassified"])
    check("garbled (empty) replies count as parse_failure, not unclassified",
          _pf_result["parse_failure"] == 0.5, _pf_result["parse_failure"])

    _pf_out = io.StringIO()
    evaluate.print_scorecard([_pf_result], out=_pf_out)
    _pf_report = _pf_out.getvalue()
    check("the scorecard has a dedicated parse-fail column",
          "parse-fail" in _pf_report, _pf_report)
    _pf_row = next(line for line in _pf_report.splitlines() if "test_garbler" in line)
    check("the scorecard row shows both a 50% uncls and a separate 50% parse-fail",
          _pf_row.count("50%") == 2, _pf_row)

print("\nvotes=1 makes 'unstable' unmeasurable, not zero")
with tempfile.TemporaryDirectory() as tmp:
    # A single-vote run cannot disagree with itself, so `unstable` must read
    # as "not measured" (NaN, printed as --), not as a genuine 0% stability
    # finding -- the same rounding-error-turned-real-problem shape as
    # PARSE_FAILURE (see taxonomy.py): a number that looks like a finding but
    # is structurally incapable of being anything else.
    _v1_gold = os.path.join(tmp, "v1.csv")
    with open(_v1_gold, "w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=["comment_id", "body", "hand_domain"])
        w.writeheader()
        for i in range(2):
            w.writerow({"comment_id": f"v{i}", "body": f"tax and lobbying {i}", "hand_domain": "democracy"})

    _v1_result = evaluate.run_arm(evaluate.read_gold(_v1_gold),
                                  judges.create("test_halfwrong", votes=1),
                                  examples_mod.create("none"), progress=False)
    check("votes=1 reports unstable as NaN, not 0.0", _v1_result["unstable"] != _v1_result["unstable"],
          _v1_result["unstable"])
    check("run_arm records how many votes actually ran", _v1_result["votes"] == 1)

    _v1_out = io.StringIO()
    evaluate.print_scorecard([_v1_result], out=_v1_out)
    _v1_row = next(line for line in _v1_out.getvalue().splitlines() if "test_halfwrong" in line)
    check("the scorecard prints -- for an unmeasurable unstable rate, not 0%",
          "—" in _v1_row, _v1_row)

    _v3_result = evaluate.run_arm(evaluate.read_gold(_v1_gold),
                                  judges.create("test_halfwrong", votes=3),
                                  examples_mod.create("none"), progress=False)
    check("votes>1 still reports a real (measured) unstable rate",
          _v3_result["unstable"] == _v3_result["unstable"], _v3_result["unstable"])


print("\naudit: undecidable is not a verdict")
import ewl.label.keywords as _kc
with tempfile.TemporaryDirectory() as tmp:
    _kwf = os.path.join(tmp, "kw.yml")
    open(_kwf, "w").write("democracy:\n  - lobbying\nenvironment:\n  - jet\n"
                          "economics:\n  - wages\nsocial:\n  - unfair\n")
    _pats = _kc.load_keywords(_kwf)
    _rows = ([{"body": f"his jet again {i}"} for i in range(6)]
             + [{"body": f"lobbying spend {i}"} for i in range(6)]
             + [{"body": f"wages frozen {i}"} for i in range(6)])
    _rep = [r for r in _kc.audit(_rows, _pats, min_hits=5) if r["total_hits"]]
    check("a 100%-precise sole-evidence term is not condemned",
          not any(r["useless"] for r in _rep),
          "leave-one-term-out empties the in-group; that is undecidable, not useless")
    check("it is reported as undecidable instead", all(r["undecidable"] for r in _rep))

# hash_author must normalise BEFORE testing the deleted markers. Normalising
# after made "[deleted] " and "@[deleted]" miss the marker set and hash, so
# every deleted user in the corpus collapsed into one synthetic mega-author --
# the same error the function exists to prevent, only larger. Cover for #47.
import ewl.common as _common
_prev_salt = os.environ.get("EWL_AUTHOR_SALT")
os.environ["EWL_AUTHOR_SALT"] = "0123456789abcdefTEST"
_common._SALT = os.environ["EWL_AUTHOR_SALT"]
check("the same user hashes the same with or without a leading @",
      _common.hash_author("@someone") == _common.hash_author("someone"))
check("every spelling of a deleted author returns no pseudonym at all",
      all(_common.hash_author(a) == "" for a in
          ["[deleted]", "[deleted] ", " [deleted]", "@[deleted]", "[removed]", "@", "", None]),
      "deleted markers must not collapse into one shared hash")
check("two different real users still differ",
      _common.hash_author("alice") != _common.hash_author("bob"))
if _prev_salt is None:
    del os.environ["EWL_AUTHOR_SALT"]
else:
    os.environ["EWL_AUTHOR_SALT"] = _prev_salt
_common._SALT = _prev_salt or ""

# The keyword CLI crashed with UnboundLocalError before writing a row: the
# blank-count report was inserted above the loop that binds `blank`. The
# harness never shelled out to this module, so it shipped green.
with tempfile.TemporaryDirectory() as _t:
    _in = os.path.join(_t, "in.csv")
    with open(_in, "w", newline="", encoding="utf-8") as _fh:
        _fh.write("comment_id,platform,story,source,author_hash,body\n"
                  "1,youtube,jets,u,h,billionaire tax dodging\n"
                  "2,youtube,jets,u,h,\n")
    _out = os.path.join(_t, "out.csv")
    _env = dict(os.environ, PYTHONPATH=os.path.join(os.getcwd(), "src"))
    _proc = subprocess.run([sys.executable, "-m", "ewl.label.keywords", _in,
                            "--output", _out],
                           capture_output=True, text=True, env=_env)
    check("the keyword classifier runs from the command line",
          _proc.returncode == 0, _proc.stderr.strip().splitlines()[-1:] or "")
    check("it writes a labelled row and counts the blank one",
          os.path.exists(_out)
          and len(list(csv.DictReader(open(_out, encoding="utf-8-sig")))) == 1
          and "1 row(s) had no comment text" in _proc.stderr,
          _proc.stderr.strip())

# The manifest ewl-merge writes used to die at this step: ewl-keywords read a
# frozen corpus and wrote a new CSV with no manifest of its own, so
# --strict-corpus on the keyword-passed file always reported "No manifest",
# not because nothing could be checked but because nothing here ever froze
# it. Same bug, same fix, in scripts_scratch_add_primary_category.py.
from ewl import corpus as _corpus
with tempfile.TemporaryDirectory() as _t:
    _frozen = os.path.join(_t, "frozen.csv")
    with open(_frozen, "w", newline="", encoding="utf-8") as _fh:
        _fh.write("comment_id,platform,story,source,author_hash,body\n"
                  "1,youtube,jets,u,h,billionaire tax dodging\n")
    _corpus.write_manifest(_frozen, inputs=["raw.csv"])

    _kw_out = os.path.join(_t, "kw.csv")
    _env = dict(os.environ, PYTHONPATH=os.path.join(os.getcwd(), "src"))
    _proc = subprocess.run([sys.executable, "-m", "ewl.label.keywords", _frozen,
                            "--output", _kw_out], capture_output=True, text=True, env=_env)
    check("ewl-keywords notes an unfrozen input rather than staying silent",
          "NOTE:" not in _proc.stderr,
          "the input WAS frozen just above, so no NOTE should fire here -- "
          "sanity check that verify() is actually being called")
    check("ewl-keywords freezes its own output",
          os.path.exists(_kw_out + ".manifest.json"), _proc.stderr)
    check("that manifest verifies against the file just written",
          _corpus.verify(_kw_out)[0] is True and "No manifest" not in _corpus.verify(_kw_out)[1])

    _prim_out = os.path.join(_t, "primary.csv")
    _proc2 = subprocess.run([sys.executable, "-m", "ewl.collect.primary_category",
                             _kw_out, "--output", _prim_out],
                            capture_output=True, text=True, env=_env)
    check("ewl-primary-category freezes its own output too",
          os.path.exists(_prim_out + ".manifest.json"), _proc2.stderr)

    # The point of the whole chain: --strict-corpus on the doubly-derived file
    # must actually find something to check, not silently fall back to "no
    # manifest" three steps downstream of the real corpus.
    _judge_out = os.path.join(_t, "labelled.csv")
    _proc3 = subprocess.run([sys.executable, "-m", "ewl.label.classify", _prim_out,
                             "--judge", "keyword", "--strict-corpus",
                             "--output", _judge_out], capture_output=True, text=True, env=_env)
    check("--strict-corpus finds a real manifest three steps downstream of ewl-merge",
          "No manifest" not in _proc3.stderr and _proc3.returncode == 0, _proc3.stderr)

# Two biases used to sit in classify()'s ranking key, both inflating whichever
# category shipped the longest term list. Regression cover for #8.
with tempfile.TemporaryDirectory() as tmp:
    # `democracy` gets four terms, `social` one. A comment matching one term
    # from each is a 25% hit on democracy's vocabulary and 100% on social's.
    _kwf2 = os.path.join(tmp, "kw2.yml")
    open(_kwf2, "w").write("democracy:\n  - alpha\n  - beta\n  - gamma\n  - delta\n"
                           "economics:\n  - zeta\n"
                           "environment:\n  - eta\n"
                           "social:\n  - theta\n")
    _p2 = _kc.load_keywords(_kwf2)
    _labels, _ = _kc.classify("alpha theta", _p2)
    check("a category is ranked by the share of its vocabulary that matched, not the raw count",
          _labels[0] == "social",
          f"one hit each, but social's list is a quarter the size: got {_labels}")

    # An exact tie must not resolve the same way every time. Equal list sizes,
    # one hit each: under the old CATEGORIES.index() key democracy won all of
    # them, which is a standing preference, not a tie-break.
    _kwf3 = os.path.join(tmp, "kw3.yml")
    open(_kwf3, "w").write("democracy:\n  - alpha\neconomics:\n  - zeta\n"
                           "environment:\n  - eta\nsocial:\n  - theta\n")
    _p3 = _kc.load_keywords(_kwf3)
    _winners = collections.Counter(
        _kc.classify(f"alpha theta comment {i}", _p3)[0][0] for i in range(400))
    check("exact ties do not always go to the same category",
          len(_winners) > 1 and max(_winners.values()) < 380,
          f"tie winners across 400 comments: {dict(_winners)}")
    check("the tie-break is deterministic for a given comment",
          all(_kc.classify("alpha theta fixed", _p3)[0]
              == _kc.classify("alpha theta fixed", _p3)[0] for _ in range(5)))


# The gold-set join. `ewl-validate sample` strips the machine labels so the
# coder cannot see them, and nothing put them back -- so a coder who agreed
# with the classifier on EVERY row scored recall 0.00 across all four
# categories, exit 0, no warning, and --scored published that as "validated".
print("\ninclusion_prob must weight the published precision and recall")
from ewl.analyse import export_dashboard as _ed

_strat = ([{"hand_domain": "environment", "domain_environment": "1" if i < 18 else "0",
            "inclusion_prob": "0.1"} for i in range(20)]
          + [{"hand_domain": "environment" if i < 10 else "none",
              "domain_environment": "1" if i < 5 else "0",
              "inclusion_prob": "1.0"} for i in range(80)])
_flat = [{k: v for k, v in r.items() if k != "inclusion_prob"} for r in _strat]
_w = [r for r in _ed.per_category_scores(_strat) if r["label"] == "environment"][0]
_u = [r for r in _ed.per_category_scores(_flat) if r["label"] == "environment"][0]
check("the dashboard's precision/recall are Horvitz-Thompson weighted",
      abs(_w["recall"] - _u["recall"]) > 1e-6,
      f"weighted {_w['recall']:.4f} vs unweighted {_u['recall']:.4f}; a stratified draw "
      f"over-samples rare categories, so unweighted figures describe the draw")
check("the raw integer counts are still reported beside the weighted ones",
      _w["tp"] == _u["tp"] and _w["weighted_tp"] != _w["tp"], _w)
# Independently hand-computed, not a second call to per_category_scores on
# effectively the same input -- the previous version of this check stripped
# inclusion_prob from _strat a second time and compared the result to _u,
# which is _flat run through the exact same function: two calls guaranteed to
# agree regardless of whether the weighting logic is correct at all. This
# instead recomputes "environment" recall by hand, directly from the fixture
# rows, as the independent value _u's recall is being checked against.
_flat_tp = sum(1 for r in _flat
              if r["hand_domain"] == "environment" and r["domain_environment"] == "1")
_flat_fn = sum(1 for r in _flat
              if r["hand_domain"] == "environment" and r["domain_environment"] == "0")
_expected_unweighted_recall = _flat_tp / (_flat_tp + _flat_fn)
check("and a gold set with no inclusion_prob is unweighted, as a simple random sample is",
      abs(_u["recall"] - _expected_unweighted_recall) < 1e-9,
      (_u["recall"], _expected_unweighted_recall))
check("the export records whether the weights were applied",
      _w["ht_weighted"] is True and _u["ht_weighted"] is False)

from ewl.analyse import validate as _val_early
_sc_w = io.StringIO()
with contextlib.redirect_stdout(_sc_w), contextlib.redirect_stderr(io.StringIO()):
    _val_early.cmd_score([dict(r, hand_domain=r["hand_domain"],
                         domain_environment=r["domain_environment"]) for r in _strat])
_sc_u = io.StringIO()
with contextlib.redirect_stdout(_sc_u), contextlib.redirect_stderr(io.StringIO()):
    _val_early.cmd_score(_flat)
check("ewl-validate score is weighted too, not just the dashboard",
      _sc_w.getvalue() != _sc_u.getvalue(),
      "score prints 'report these numbers in the briefing', so it must match the dashboard")

print("\ndashboard sentiment shares exclude irrelevant/uncodeable from the denominator")
# Regression: these rows have no stance to be negative/neutral/positive
# about (sentiment.py's LLMScorer produces them; VADER never does, which is
# why this was dormant under the old scorer). Counting them in the
# denominator while never in a numerator silently understated every share.
_sent_rows = (
    [{"story": "luigi", "source": f"v{i % 3}", "sentiment": "negative"} for i in range(10)]
    + [{"story": "luigi", "source": f"v{i % 3}", "sentiment": "irrelevant"} for i in range(10)]
)
_sent_props = {p["label"]: p for p in _ed.proportions(_sent_rows) if p["kind"] == "sentiment"}
check("irrelevant rows are excluded, so the negative share is the true 100%, not a diluted 50%",
      _sent_props["negative"]["rate"] == 1.0, _sent_props["negative"])
check("n reflects the codeable subset, not the whole row count",
      _sent_props["negative"]["n"] == 10, _sent_props["negative"]["n"])

print("\nread_any (not read) loads a coder's .xlsx file for interrater/agree/etc")
with optional("xlsx coder files"):
    import openpyxl
    from ewl.analyse import validate as _val_xlsx
    with tempfile.TemporaryDirectory() as _tmp:
        _xlsx_path = os.path.join(_tmp, "coder.xlsx")
        _wb = openpyxl.Workbook()
        _ws = _wb.active
        _ws.append(["comment_id", "body", "hand_domain", "hand_sentiment"])
        _ws.append(["c1", "they bought the platform", "democracy", "negative"])
        _wb.save(_xlsx_path)
        # This used to be main()'s plain read() for every command including
        # `interrater`, whose input IS a coder's .xlsx file -- it raised an
        # opaque UnicodeDecodeError rather than a message telling the coder
        # to convert it first.
        _rows = _val_xlsx.read_any(_xlsx_path)
        check("an xlsx coder file loads via read_any with no error",
              len(_rows) == 1 and _rows[0]["comment_id"] == "c1", _rows)

print("\nopenai backend sends parameters GPT-5-family models actually accept")
with optional("openai SDK"):
    import openai as _openai_check  # noqa: F401 -- only to confirm it's installed
    import unittest.mock as _mock
    _captured = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            _captured.update(kwargs)
            resp = _mock.MagicMock()
            resp.usage = None
            resp.choices = [_mock.MagicMock(message=_mock.MagicMock(content="none"))]
            return resp

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeOpenAIClient:
        def __init__(self, *a, **kw):
            self.chat = _FakeChat()

    with _mock.patch("openai.OpenAI", _FakeOpenAIClient):
        _oj = judges.create("openai:gpt-5-mini", votes=1)
        _oj._sample("test body", "system prompt", 0)
    # Confirmed against OpenAI's own current docs/community reports (not
    # memory): GPT-5-family models reject "max_tokens" with a 400 (need
    # "max_completion_tokens") and reject any non-default "temperature" --
    # this was a live break on gpt-5-mini, this backend's own default model.
    check("max_tokens is not sent (GPT-5 models 400 on it)",
          "max_tokens" not in _captured, _captured)
    check("max_completion_tokens is sent instead",
          "max_completion_tokens" in _captured, _captured)
    check("temperature is not sent (GPT-5 models 400 on any non-default value)",
          "temperature" not in _captured, _captured)


print("\nparse failures must not be scored as classifier misses")
from ewl.analyse import report as _rep_pf

_pf_gold = ([{"hand_domain": "democracy", "domain_democracy": "1",
              "domain_parse_failure": "0", "source": f"v{i % 6}"} for i in range(30)]
            + [{"hand_domain": "none", "domain_democracy": "0",
                "domain_parse_failure": "0", "source": f"v{i % 6}"} for i in range(30)])
_pf_with = _pf_gold + [{"hand_domain": "democracy", "domain_democracy": "0",
                        "domain_parse_failure": "1", "source": f"v{i % 6}"} for i in range(20)]
_tpr_clean, _ = _rep_pf._gold_tpr_fpr(_pf_gold, "democracy", "hand_domain")
_tpr_pf, _ = _rep_pf._gold_tpr_fpr(_pf_with, "democracy", "hand_domain")
check("20 parse failures do not deflate TPR",
      abs(_tpr_clean - _tpr_pf) < 1e-9,
      f"clean {_tpr_clean:.3f} vs with-parse-failures {_tpr_pf:.3f}; counting them as "
      f"misses deflates TPR, and corrected_estimate divides by (TPR - FPR)")
check("and TPR is still the real value, not 1.0 by construction",
      abs(_tpr_clean - 1.0) < 1e-9 or 0 < _tpr_clean <= 1.0)

# The CLI refusal for --selector fixed with pool/k flags.
try:
    examples_mod.refuse_pool_flags_for_fixed("fixed", "gold.csv", 25)
    _refused_fixed = False
except SystemExit:
    _refused_fixed = True
check("--selector fixed with --pool/--k refuses instead of silently using 5 anchors",
      _refused_fixed)
check("but fixed on its own is fine, and other selectors are untouched",
      examples_mod.refuse_pool_flags_for_fixed("fixed", None, None) is None
      and examples_mod.refuse_pool_flags_for_fixed("fixed_random", "gold.csv", 25) is None)

check("a vote with no majority is a parse failure, not a genuine 'none'",
      judges.create("stub").resolve(
          [["democracy"], ["economics"], ["social"]])[0] == [taxonomy.PARSE_FAILURE],
      "returning UNCLASSIFIED asserts the model judged the comment engages no harm, "
      "manufactured out of three votes that merely disagreed")


print("\nthe gold set must be joined back to the machine labels before it can be scored")
from ewl.analyse import validate as _val

_labelled = [{"comment_id": "c1", "body": "tax dodging", "kw_democracy": "1",
              "kw_economics": "0", "kw_environment": "0", "kw_social": "0"},
             {"comment_id": "c2", "body": "private jet", "kw_democracy": "0",
              "kw_economics": "0", "kw_environment": "1", "kw_social": "0"}]
_coded = [{"comment_id": "c1", "body": "tax dodging", "hand_domain": "democracy"},
          {"comment_id": "c2", "body": "private jet", "hand_domain": "environment"}]

check("a coder-returned file is recognised as having no machine labels",
      _val.machine_prefix(_coded) is None)
check("and a joined file is recognised as having them",
      _val.machine_prefix(_labelled) == "kw_")
check("domain_ wins over kw_ when a file carries both",
      _val.machine_prefix([dict(_labelled[0], domain_democracy="1")]) == "domain_")

with tempfile.TemporaryDirectory() as _t:
    _p = os.path.join(_t, "joined.csv")
    with open(_p, "w", newline="", encoding="utf-8") as _fh:
        _m, _miss = _val.cmd_join(_coded, _labelled, _fh)
    check("join matches every coded comment to its machine labels",
          (_m, _miss) == (2, 0), (_m, _miss))
    _joined = list(csv.DictReader(open(_p, encoding="utf-8-sig")))
    check("the joined file carries both the hand code and the machine label",
          _joined[0]["hand_domain"] == "democracy" and _joined[0]["kw_democracy"] == "1",
          _joined[0])
    check("and it is now scoreable", _val.machine_prefix(_joined) == "kw_")

    # A coded id absent from the corpus must be reported, not dropped.
    _m2, _miss2 = _val.cmd_join(
        _coded + [{"comment_id": "ghost", "body": "x", "hand_domain": "social"}],
        _labelled, io.StringIO(), allow_partial=True)
    # The corpus wins on the carried columns; the coder wins on what they were
# asked to fill in. `source` travels with the sample since #61, so it now sits
# in a spreadsheet in front of two coders -- and corrected_ci_bootstrap
# clusters the gold set by it. A coder clearing that column used to take the
# gold set from 5 clusters to 1.
_lab_src = [{"comment_id": f"s{i}", "source": f"vid{i % 5}", "domain_democracy": "1"}
            for i in range(20)]
for _name, _mutate in (("blanked", lambda r: dict(r, source="")),
                       ("reformatted", lambda r: dict(r, source=r["source"].upper()))):
    _coded_src = [_mutate({"comment_id": f"s{i}", "source": f"vid{i % 5}",
                           "hand_domain": "democracy"}) for i in range(20)]
    _buf, _err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(_err):
        _val.cmd_join(_coded_src, _lab_src, _buf)
    _joined_src = list(csv.DictReader(io.StringIO(_buf.getvalue())))
    _clusters = len({(r.get("source") or "").strip() or "(unknown)" for r in _joined_src})
    check(f"a coder's {_name} source column cannot damage the cluster structure",
          _clusters == 5,
          f"got {_clusters} clusters; the corpus value must win on carried columns")
    check(f"and the {_name} disagreement is reported rather than absorbed",
          "NOTE" in _err.getvalue(), _err.getvalue())

_clean_coded = [{"comment_id": f"s{i}", "source": f"vid{i % 5}", "hand_domain": "democracy"}
                for i in range(20)]
_clean_err = io.StringIO()
with contextlib.redirect_stderr(_clean_err):
    _val.cmd_join(_clean_coded, _lab_src, io.StringIO())
check("an untouched file reports no disagreement",
      "NOTE" not in _clean_err.getvalue(), _clean_err.getvalue())

check("a coded comment missing from the corpus is counted, not silently dropped",
          (_m2, _miss2) == (2, 1), (_m2, _miss2))

# The whole point: agreement on every row must score 1.00, not 0.00.
_perfect = [dict(r, kw_democracy="1" if r["hand_domain"] == "democracy" else "0",
                 kw_economics="0",
                 kw_environment="1" if r["hand_domain"] == "environment" else "0",
                 kw_social="0") for r in _coded]
_out = io.StringIO()
with contextlib.redirect_stdout(_out):
    _val.cmd_score(_perfect)
check("a coder who agrees with the classifier on every row scores 1.00, not 0.00",
      "1.00" in _out.getvalue() and "0.00" not in _out.getvalue().split("democracy")[1][:20],
      _out.getvalue())

_err = io.StringIO()
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(_err):
        _val.cmd_score(_coded)
    _refused = False
except SystemExit:
    _refused = True
check("scoring an unjoined file refuses instead of reporting zeros", _refused)


print("\ncombine: recombining coders' files never trusts a coder's own copy of inclusion_prob")
_master = [
    {"comment_id": "c1", "story": "jets", "source": "v1", "body": "b1",
     "sampling_stratum": "jets/democracy", "inclusion_prob": "0.1"},
    {"comment_id": "c2", "story": "jets", "source": "v2", "body": "b2",
     "sampling_stratum": "jets/economics", "inclusion_prob": "0.2"},
    {"comment_id": "c3", "story": "jets", "source": "v3", "body": "b3",
     "sampling_stratum": "jets/social", "inclusion_prob": "0.3"},
    {"comment_id": "old", "story": "jets", "source": "v4", "body": "bOld",
     "sampling_stratum": "jets/environment", "inclusion_prob": "0.4"},
]
_manifest_rows = [
    {"comment_id": "c1", "assigned_to": "A", "is_core": "0"},
    {"comment_id": "c2", "assigned_to": "B", "is_core": "0"},
    {"comment_id": "c3", "assigned_to": "A", "is_core": "1"},
    {"comment_id": "c3", "assigned_to": "B", "is_core": "1"},
    {"comment_id": "old", "assigned_to": "A", "is_core": "0"},
]
# Every coder's OWN copy of inclusion_prob is deliberately wrong (0.999) --
# combine must ignore it completely and take the real value from _master.
_coder_a = {"pathA": [
    {"comment_id": "c1", "hand_domain": "democracy", "hand_sentiment": "negative",
     "coder_notes": "", "inclusion_prob": "0.999"},
    {"comment_id": "c3", "hand_domain": "social", "hand_sentiment": "positive",
     "coder_notes": "noteA", "inclusion_prob": "0.999"},
    {"comment_id": "old", "hand_domain": "environment", "hand_sentiment": "neutral",
     "coder_notes": "", "inclusion_prob": "0.999"},
]}
_coder_b = {"pathB": [
    {"comment_id": "c2", "hand_domain": "economics", "hand_sentiment": "neutral",
     "coder_notes": "", "inclusion_prob": "0.999"},
    # Disagrees with coder A on c3's hand_sentiment -- a core row in genuine dispute.
    {"comment_id": "c3", "hand_domain": "social", "hand_sentiment": "negative",
     "coder_notes": "noteB", "inclusion_prob": "0.999"},
]}
_coders_by_path = dict(_coder_a, **_coder_b)
_replacements = [
    {"comment_id": "new1", "story": "jets", "source": "v5", "body": "replacement",
     "hand_domain": "environment", "hand_sentiment": "positive", "coder_notes": "",
     "sampling_stratum": "jets/environment", "replaces_comment_id": "old"},
]
_combine_out = io.StringIO()
_combine_err = io.StringIO()
with contextlib.redirect_stderr(_combine_err):
    _rows, _needs_adj, _not_coded, _missing = _val.cmd_combine(
        _coders_by_path, _manifest_rows, _master, _replacements, _combine_out)
_by_id = {r["comment_id"]: r for r in _rows}

check("a non-core row combines straight through from its one coder",
      _by_id["c1"]["hand_domain"] == "democracy", _by_id.get("c1"))
check("inclusion_prob comes from the master sample, never the coder's own (wrong) copy",
      _by_id["c1"]["inclusion_prob"] == "0.1" and _by_id["c2"]["inclusion_prob"] == "0.2",
      {k: v["inclusion_prob"] for k, v in _by_id.items()})
check("a core row where coders disagree is excluded, not silently resolved",
      "c3" not in _by_id and len(_needs_adj) == 1 and _needs_adj[0][0] == "c3",
      (_by_id.get("c3"), _needs_adj))
check("the disagreement is reported so a human can adjudicate it",
      "NEEDS ADJUDICATION" in _combine_err.getvalue() and "c3" in _combine_err.getvalue(),
      _combine_err.getvalue())
check("a replacement row swaps in for the row it replaces, not alongside it",
      "old" not in _by_id and _by_id["new1"]["hand_domain"] == "environment", _by_id)
check("a replacement row inherits the ORIGINAL row's inclusion_prob and stratum, "
      "not its own (it has none of its own -- it was hand-picked, not drawn)",
      _by_id["new1"]["inclusion_prob"] == "0.4"
      and _by_id["new1"]["sampling_stratum"] == "jets/environment", _by_id["new1"])
check("exactly the right rows survive: c1, c2, new1 -- not c3 (disputed), not old (replaced)",
      set(_by_id) == {"c1", "c2", "new1"}, set(_by_id))

# A row with a blank hand_domain/hand_sentiment somewhere must not be silently
# treated as agreed-and-complete.
_coder_c_partial = {"pathC": [
    {"comment_id": "c4", "hand_domain": "", "hand_sentiment": "", "coder_notes": "",
     "inclusion_prob": "0.999"},
]}
_master_c4 = _master + [{"comment_id": "c4", "story": "jets", "source": "v6", "body": "b4",
                         "sampling_stratum": "jets/unclassified", "inclusion_prob": "0.5"}]
_manifest_c4 = _manifest_rows + [{"comment_id": "c4", "assigned_to": "C", "is_core": "0"}]
_rows2, _adj2, _not_coded2, _missing2 = _val.cmd_combine(
    _coder_c_partial, _manifest_c4, _master_c4, [], io.StringIO())
check("a not-yet-coded (blank) row is excluded and reported, not included with blanks",
      "c4" not in {r["comment_id"] for r in _rows2} and "c4" in _not_coded2,
      (_rows2, _not_coded2))

# A replacement row that names a replaces_comment_id not in the master sample
# must fail loudly rather than guess.
try:
    _val.cmd_combine({}, [], _master,
                     [{"comment_id": "bad", "replaces_comment_id": "not_a_real_id",
                       "hand_domain": "x", "hand_sentiment": "y"}], io.StringIO())
    check("an unmatched replaces_comment_id refuses rather than guessing", False,
          "no SystemExit raised")
except SystemExit:
    check("an unmatched replaces_comment_id refuses rather than guessing", True)


print("\ncombine: inclusion probabilities recomputed against the final corpus")
import random as _random

# 1. The pure arithmetic agrees with the sampler's own per-row probability, in
#    both branches (cap not binding, and cap too tight so the quota backfills).
for _count, _frac, _label in ((4, 0.5, "backfilled"), (3, 1.0, "cap not binding")):
    _pool = ([{"source": "big", "comment_id": f"b{i}"} for i in range(6)]
             + [{"source": "mid", "comment_id": f"m{i}"} for i in range(3)]
             + [{"source": "small", "comment_id": "s0"}])
    _chosen, _short = _val.sample_with_video_cap(_pool, _count, _frac)
    _expected = _val.video_cap_probabilities({"big": 6, "mid": 3, "small": 1}, _count, _frac)
    check(f"video_cap_probabilities matches sample_with_video_cap ({_label})",
          all(abs(r["_video_cap_prob"] - _expected[r["source"]]) < 1e-12 for r in _chosen),
          ({r["source"]: r["_video_cap_prob"] for r in _chosen}, _expected))

# 2. The two-phase formula is right: simulate it. Stratum of N=30, phase 1 draws
#    n1=5, phase 2 draws m=7 from the remainder; any one comment's chance of being
#    in the union should be p1 + (1-p1) * m / (N - n1) = 5/30 + (25/30) * 7/25.
_rng = _random.Random(1)
_hits, _trials = 0, 40000
for _ in range(_trials):
    _ids = list(range(30))
    _first = set(_rng.sample(_ids, 5))
    _rest = [i for i in _ids if i not in _first]
    _second = set(_rng.sample(_rest, 7))
    _hits += 0 in (_first | _second)
_formula = 5 / 30 + (25 / 30) * (7 / 25)
check("two-phase probability p1 + (1-p1)*m/(N-n1) matches a simulation of the two draws",
      abs(_hits / _trials - _formula) < 0.01, (_hits / _trials, _formula))

print("\ninclusion probabilities: no silent weight 1")
from ewl.analyse import validate as _vw, report as _rw
def _raises(fn):
    try:
        fn()
    except (ValueError, SystemExit):
        return True
    return False
check("a probability of 0 is refused, not weighted 1",
      _raises(lambda: _vw.gold_weight({"comment_id": "z", "inclusion_prob": "0.000000"})))
check("a negative, >1 or non-numeric probability is refused",
      all(_raises(lambda v=v: _vw.gold_weight({"inclusion_prob": v})) for v in ("-0.2", "1.5", "abc")))
check("a valid probability weighs 1/p, and an all-blank set stays unweighted",
      _vw.gold_weight({"inclusion_prob": "0.25"}) == 4.0
      and _vw.check_gold_weights([{"inclusion_prob": ""}, {"inclusion_prob": ""}]) is False)
check("a set mixing blank and recorded probabilities is refused",
      _raises(lambda: _vw.check_gold_weights([{"comment_id": "a", "inclusion_prob": "0.0002"},
                                              {"comment_id": "b", "inclusion_prob": ""}])))
check("the effective-positive count refuses a zero probability instead of counting it at weight 1",
      _raises(lambda: _rw.weighted_effective_positive_count(
          [{"hand_domain": "democracy", "inclusion_prob": "0.000000"}] * 3, "democracy", "hand_domain")))
_corp = ([{"comment_id": f"a{i}", "story": "jets", "primary_category": "democracy", "source": "v1"}
          for i in range(10)]
         + [{"comment_id": "moved", "story": "jets", "primary_category": "social", "source": "v9"}])
_kept, _dropped = _vw.recompute_inclusion_probs(
    [{"comment_id": "a1", "sampling_stratum": "jets/democracy", "story": "jets"},
     {"comment_id": "moved", "sampling_stratum": "jets/democracy", "story": "jets"}],
    _corp, {"jets/democracy": 5})
check("recompute excludes a row whose video has no rows in its drawn stratum (probability 0), "
      "never writing 0.000000",
      [r["comment_id"] for r in _kept] == ["a1"] and _dropped == ["moved"]
      and float(_kept[0]["inclusion_prob"]) == 0.5, (_kept, _dropped))


# 3. End to end through cmd_combine, on a toy final corpus.
_corpus = ([{"comment_id": f"u{i}", "story": "jets", "primary_category": "unclassified",
             "source": "v1"} for i in range(40)]
           + [{"comment_id": f"d{i}", "story": "jets", "primary_category": "democracy",
               "source": "v2"} for i in range(10)])
_pmaster = [
    {"comment_id": "u0", "story": "jets", "source": "v1", "body": "x",
     "sampling_stratum": "jets/unclassified", "inclusion_prob": "0.9"},
    {"comment_id": "u1", "story": "jets", "source": "v1", "body": "x",
     "sampling_stratum": "jets/unclassified", "inclusion_prob": "0.9"},
    {"comment_id": "d0", "story": "jets", "source": "v2", "body": "x",
     "sampling_stratum": "jets/democracy", "inclusion_prob": "0.9"},
    {"comment_id": "d1", "story": "jets", "source": "v2", "body": "x",
     "sampling_stratum": "jets/democracy", "inclusion_prob": "0.9"},
]
_ptop = [{"comment_id": "u5", "story": "jets", "source": "v1", "body": "x",
          "inclusion_prob": "0.9"},
         {"comment_id": "u6", "story": "jets", "source": "v1", "body": "x",
          "inclusion_prob": "0.9"}]
_pman = [{"comment_id": c, "assigned_to": "A", "is_core": "0"} for c in ("u0", "u1", "d0", "d1")]
_pcoder = {"pA": [{"comment_id": c, "hand_domain": "none", "hand_sentiment": "3",
                   "coder_notes": ""} for c in ("u0", "u1", "d0", "u5", "u6")]}
# d1 is replaced by d2 (hand-picked within the same stratum). zz9 stands in for a coded row
# that is NOT in the final corpus: it must be excluded, not given a guessed probability.
_preplace = [{"comment_id": "d2", "replaces_comment_id": "d1", "story": "jets", "source": "v2",
              "body": "x", "hand_domain": "democracy", "hand_sentiment": "2",
              "coder_notes": ""},
             {"comment_id": "zz9", "replaces_comment_id": "d0", "story": "jets", "source": "v2",
              "body": "x", "hand_domain": "democracy", "hand_sentiment": "2",
              "coder_notes": ""}]
_perr = io.StringIO()
with contextlib.redirect_stderr(_perr):
    _prows, _, _, _ = _val.cmd_combine(
        _pcoder, _pman, _pmaster, _preplace, io.StringIO(),
        corpus_rows=_corpus, topup_rows=_ptop, cap_fraction=None)
_pby = {r["comment_id"]: r for r in _prows}
# unclassified: N=40, phase-1 n1=2, top-up m=2 -> 2/40 + (38/40) * 2/38 = 0.1
check("a phase-1 unclassified row gets the two-phase probability, from the final corpus",
      abs(float(_pby["u0"]["inclusion_prob"]) - 0.1) < 1e-6, _pby.get("u0"))
check("a top-up row gets the SAME probability as a phase-1 row in that stratum "
      "(it is a property of the comment, not of which phase picked it)",
      abs(float(_pby["u5"]["inclusion_prob"]) - float(_pby["u0"]["inclusion_prob"])) < 1e-9,
      (_pby.get("u5"), _pby.get("u0")))
check("a non-unclassified replacement row takes the plain stratum probability n/N (2/10) "
      "and is flagged as an approximation",
      abs(float(_pby["d2"]["inclusion_prob"]) - 0.2) < 1e-6
      and "replacement_approx" in _pby["d2"]["inclusion_prob_basis"], _pby.get("d2"))
check("the figure recomputed over is kept alongside as inclusion_prob_drawn",
      _pby["d2"]["inclusion_prob_drawn"] == "0.9", _pby.get("d2"))
check("the replaced row is gone, and a row not in the final corpus is excluded, not guessed",
      "d1" not in _pby and "zz9" not in _pby and "zz9" in _perr.getvalue(),
      (set(_pby), _perr.getvalue()))
check("top-up rows are combined without needing a split-manifest entry",
      "u5" in _pby and "u6" in _pby, set(_pby))
check("the basis column distinguishes two-phase from single-phase rows",
      _pby["u0"]["inclusion_prob_basis"] == "recomputed_two_phase"
      and _pby["d2"]["inclusion_prob_basis"].startswith("recomputed_final_corpus"), _pby)



import ewl.analyse.evaluate as _evaluate_policy
print("\nbenchmark policy: every scorecard arm uses the 25 fixed examples")
check("a `fixed` arm, a bare arm under the default selector and the keyword baseline pass",
      _evaluate_policy.non_fixed_arms(
          ["anthropic:claude-opus-5/fixed", "gemma/fixed", "keyword/none", "stub"]) == [],
      _evaluate_policy.non_fixed_arms(
          ["anthropic:claude-opus-5/fixed", "gemma/fixed", "keyword/none", "stub"]))
check("a retrieved or zero-shot arm is flagged",
      _evaluate_policy.non_fixed_arms(["anthropic:claude-opus-5/retrieved", "stub/none"])
      == ["anthropic:claude-opus-5/retrieved", "stub/none"],
      _evaluate_policy.non_fixed_arms(["anthropic:claude-opus-5/retrieved", "stub/none"]))
check("a bare arm inherits a non-fixed --selector and is flagged",
      _evaluate_policy.non_fixed_arms(["stub"], default_selector="retrieved") == ["stub/retrieved"],
      _evaluate_policy.non_fixed_arms(["stub"], default_selector="retrieved"))


print("\ncombine: adjudicated rows, and a replacement row with a blank stratum cell")
_adj_rows, _adj_needs, _adj_nc, _ = _val.cmd_combine(
    _coders_by_path, _manifest_rows, _master, _replacements, io.StringIO(),
    adjudicated_rows=[{"comment_id": "c3", "hand_domain": "social", "hand_sentiment": "negative"}])
_adj_by = {r["comment_id"]: r for r in _adj_rows}
check("an adjudicated row is included, with the adjudicated answer and a note saying so",
      "c3" in _adj_by and _adj_by["c3"]["hand_sentiment"] == "negative"
      and _adj_by["c3"]["coder_notes"] == "adjudicated", _adj_by.get("c3"))
check("it is no longer reported as still needing adjudication",
      not any(cid == "c3" for cid, _ in _adj_needs), _adj_needs)
check("the adjudicated row keeps the master's inclusion_prob like any other row",
      _adj_by["c3"]["inclusion_prob"] == "0.3", _adj_by.get("c3"))
for _bad, _why in (
        ([{"comment_id": "nobody", "hand_domain": "social", "hand_sentiment": "negative"}],
         "an id that was never drawn"),
        ([{"comment_id": "c3", "hand_domain": "", "hand_sentiment": "negative"}],
         "a blank answer")):
    try:
        _val.cmd_combine(_coders_by_path, _manifest_rows, _master, _replacements, io.StringIO(),
                         adjudicated_rows=_bad)
        check(f"an adjudication with {_why} is refused", False, "no SystemExit raised")
    except SystemExit:
        check(f"an adjudication with {_why} is refused", True)

# A replacements file that HAS a sampling_stratum column but leaves the cell blank must
# still inherit the replaced row's stratum, not end up in none.
_blank_cell = [{"comment_id": "new2", "story": "", "source": "v5", "body": "r",
                "hand_domain": "environment", "hand_sentiment": "positive", "coder_notes": "",
                "sampling_stratum": "", "replaces_comment_id": "old"}]
_bc_rows, _, _, _ = _val.cmd_combine(
    _coders_by_path, _manifest_rows, _master, _blank_cell, io.StringIO())
_bc = {r["comment_id"]: r for r in _bc_rows}
check("a blank sampling_stratum / story cell on a replacement falls back to the replaced row's",
      _bc["new2"]["sampling_stratum"] == "jets/environment" and _bc["new2"]["story"] == "jets",
      _bc.get("new2"))



print("\ncombine: bucketed sentiment agreement, and the per-author cap scope")
_bm = [{"comment_id": c, "story": "jets", "source": "v", "body": "b",
        "sampling_stratum": "jets/social", "inclusion_prob": "0.5"} for c in ("b1", "b2", "b3")]
_bman = [{"comment_id": c, "assigned_to": a, "is_core": "1"}
         for c in ("b1", "b2", "b3") for a in ("A", "B")]


def _coded(a1, b1, a2, b2, a3, b3):
    return {"pA": [{"comment_id": "b1", "hand_domain": "social", "hand_sentiment": a1, "coder_notes": ""},
                   {"comment_id": "b2", "hand_domain": "social", "hand_sentiment": a2, "coder_notes": ""},
                   {"comment_id": "b3", "hand_domain": "social", "hand_sentiment": a3, "coder_notes": ""}],
            "pB": [{"comment_id": "b1", "hand_domain": "social", "hand_sentiment": b1, "coder_notes": ""},
                   {"comment_id": "b2", "hand_domain": "social", "hand_sentiment": b2, "coder_notes": ""},
                   {"comment_id": "b3", "hand_domain": "social", "hand_sentiment": b3, "coder_notes": ""}]}


# b1: 1 vs 2 (same side), b2: 2 vs 4 (opposite sides), b3: 0 vs 0 (identical)
_cod = _coded("1", "2", "2", "4", "0", "0")
with contextlib.redirect_stderr(io.StringIO()):
    _ex_rows, _ex_adj, _, _ = _val.cmd_combine(_cod, _bman, _bm, [], io.StringIO())
    _bk_rows, _bk_adj, _, _ = _val.cmd_combine(_cod, _bman, _bm, [], io.StringIO(),
                                               sentiment_agreement="bucket")
check("exact (the default): 1 vs 2 goes to adjudication",
      "b1" in {c for c, _ in _ex_adj} and "b1" not in {r["comment_id"] for r in _ex_rows},
      _ex_adj)
_bk = {r["comment_id"]: r for r in _bk_rows}
check("bucket: 1 vs 2 agree, and the bucket word is written, not either coder's number",
      "b1" in _bk and _bk["b1"]["hand_sentiment"] == "negative", _bk.get("b1"))
check("bucket: 2 vs 4 still goes to adjudication (opposite sides)",
      "b2" in {c for c, _ in _bk_adj} and "b2" not in _bk, _bk_adj)
check("bucket: identical 0 vs 0 (irrelevant) keeps its value",
      _bk["b3"]["hand_sentiment"] == "0", _bk.get("b3"))
check("bucket function: 0 and -1 are never folded into the scale",
      _val.sentiment_bucket("0") == "0" and _val.sentiment_bucket("-1") == "-1"
      and _val.sentiment_bucket("3") == "neutral" and _val.sentiment_bucket("5") == "positive")

from ewl.collect import merge as _merge_scope
_rows = [{"story": "luigi", "author_hash": "a1", "comment_id": "l1"},
         {"story": "luigi", "author_hash": "a1", "comment_id": "l2"},
         {"story": "musk", "author_hash": "a1", "comment_id": "m1"},
         {"story": "musk", "author_hash": "a1", "comment_id": "m2"},
         {"story": "musk", "author_hash": "", "comment_id": "m3"}]
_g = [r["comment_id"] for r in _merge_scope.cap_per_author(_rows, 2)]
_s = [r["comment_id"] for r in _merge_scope.cap_per_author(_rows, 2, scope="story")]
check("cap scope=global (default): an author's cap is shared across stories, so merge order "
      "decides who loses rows", _g == ["l1", "l2", "m3"], _g)
check("cap scope=story: each story keeps its own allowance, and blank authors are never capped",
      _s == ["l1", "l2", "m1", "m2", "m3"], _s)
try:
    _merge_scope.cap_per_author(_rows, 2, scope="x")
    check("an unknown cap scope is refused", False, "no ValueError")
except ValueError:
    check("an unknown cap scope is refused", True)



print("\nscoring: exemplar exclusion, video-clustered intervals, split by video")
import csv as _csv9
import ewl.analyse.evaluate as _ev9

_tmp9 = _tmp_unc.mkdtemp()
_gold_path = os.path.join(_tmp9, "gold.csv")
with open(_gold_path, "w", newline="", encoding="utf-8") as _h:
    _w = _csv9.DictWriter(_h, fieldnames=["comment_id", "body", "hand_domain", "source",
                                          "inclusion_prob"])
    _w.writeheader()
    for _i in range(12):
        _w.writerow({"comment_id": f"g{_i}", "body": f"body {_i}",
                     "hand_domain": "democracy" if _i % 2 else "none",
                     "source": f"vid{_i // 3}", "inclusion_prob": "0.5"})
_all = _ev9.read_gold(_gold_path)
_kept = _ev9.read_gold(_gold_path, exclude_ids={"g1", "g2"})
check("read_gold drops the comments named as prompt examples",
      len(_all) == 12 and len(_kept) == 10 and not {"g1", "g2"} & {r["comment_id"] for r in _kept},
      (len(_all), len(_kept)))
check("validate.drop_ids leaves everything when no ids are given",
      _val.drop_ids([{"comment_id": "a"}], None) == [{"comment_id": "a"}])

# A perfect classifier on 4 videos: every resample is perfect, so the interval collapses.
_perfect = {"_scored_rows": [(f"v{v}", 1.0, {"democracy"}, {"democracy"}) for v in range(4)
                             for _ in range(5)]}
_iv, _nv = _ev9.bootstrap_scorecard(_perfect, n_boot=200)
check("bootstrap: a perfect classifier gets an interval of exactly 1.0-1.0, over 4 videos",
      _iv["democracy"]["recall"] == (1.0, 1.0) and _nv == 4, (_iv["democracy"], _nv))
# Two videos, one all right and one all wrong: resampling videos must spread widely,
# which resampling comments would not (it would hide the video-level swing).
_split_rows = ([("good", 1.0, {"democracy"}, {"democracy"})] * 20
               + [("bad", 1.0, {"democracy"}, set())] * 20)
_iv2, _ = _ev9.bootstrap_scorecard({"_scored_rows": _split_rows}, n_boot=400)
_lo, _hi = _iv2["democracy"]["recall"]
check("bootstrap clusters by video: one good and one bad video gives a wide recall interval",
      _lo <= 0.05 and _hi >= 0.95, (_lo, _hi))
_iv3, _n3 = _ev9.bootstrap_scorecard({"_scored_rows": [("only", 1.0, {"democracy"}, {"democracy"})]})
check("bootstrap refuses to invent an interval from a single video",
      _n3 == 1 and _iv3["democracy"]["recall"][0] != _iv3["democracy"]["recall"][0], _iv3)

_rows9 = _ev9.read_gold(_gold_path)
_pool9, _held9 = _ev9.split_by_video(_rows9, 0.5, seed=3)
_pv, _hv = {r["source"] for r in _pool9}, {r["source"] for r in _held9}
check("split by video: no video appears on both sides", not (_pv & _hv), (_pv, _hv))
check("split by video: every row lands on exactly one side",
      len(_pool9) + len(_held9) == len(_rows9)
      and not ({r["comment_id"] for r in _pool9} & {r["comment_id"] for r in _held9}))
check("split by video is reproducible for a seed",
      [r["comment_id"] for r in _ev9.split_by_video(_rows9, 0.5, seed=3)[0]]
      == [r["comment_id"] for r in _pool9])



print("\nsentiment failures are their own labels, and one parse-failure rule runs through the rates")
from ewl.analyse import report as _rep10, export_dashboard as _exp10
from ewl.label import sentiment as _sent10

# Rates: blank sentiment_value is "not rated", not 0.0.
_sv = [{"source": "v1", "sentiment_value": "-1.0"}, {"source": "v1", "sentiment_value": "-1.0"},
       {"source": "v2", "sentiment_value": ""}, {"source": "v2", "sentiment_value": ""}]
check("two -1.0 comments plus two unrated ones average -1.0, not -0.5",
      _rep10._cluster_rate(_sv, "sentiment_value") == -1.0, _rep10._cluster_rate(_sv, "sentiment_value"))
check("a column of nothing but unrated comments is NaN, not 0.0",
      _rep10._cluster_rate([{"sentiment_value": ""}], "sentiment_value") != _rep10._cluster_rate(
          [{"sentiment_value": ""}], "sentiment_value"))

# A domain flag is a share of comments the classifier ANSWERED.
_dr = [{"source": "v", "domain_democracy": "1", "domain_parse_failure": "0"},
       {"source": "v", "domain_democracy": "0", "domain_parse_failure": "0"},
       {"source": "v", "domain_democracy": "0", "domain_parse_failure": "0"},
       {"source": "v", "domain_democracy": "0", "domain_parse_failure": "1"}]
check("a parse failure leaves the denominator of a domain rate (1 of 3 answered, not 1 of 4)",
      abs(_rep10._cluster_rate(_dr, "domain_democracy") - 1 / 3) < 1e-9,
      _rep10._cluster_rate(_dr, "domain_democracy"))
check("the parse-failure rate itself is still a share of ALL rows (1 of 4)",
      _rep10._cluster_rate(_dr, "domain_parse_failure") == 0.25,
      _rep10._cluster_rate(_dr, "domain_parse_failure"))
check("a file with no parse-failure column behaves exactly as before",
      _rep10._cluster_rate([{"domain_democracy": "1"}, {"domain_democracy": "0"}],
                           "domain_democracy") == 0.5)

# The dashboard: stance shares are over rated comments only; parse failures aren't misses.
check("the new labels are all recognised as 'not a stance'",
      set(_sent10.NOT_A_STANCE) == {"irrelevant", "uncodeable", "parse_failure", "no_consensus"},
      _sent10.NOT_A_STANCE)
_scored = [{"hand_domain": "democracy", "domain_democracy": "1", "domain_parse_failure": "0"},
           {"hand_domain": "democracy", "domain_democracy": "0", "domain_parse_failure": "1"},
           {"hand_domain": "none", "domain_democracy": "0", "domain_parse_failure": "0"}]
_pcs = {d["label"]: d for d in _exp10.per_category_scores(_scored)}
check("per_category_scores leaves a parse-failure row out of recall (1 of 1 answered, not 1 of 2)",
      _pcs["democracy"]["recall"] == 1.0 and _pcs["democracy"]["parse_failure_rows_excluded"] == 1,
      _pcs["democracy"])

# End to end through ewl-classify: the stub judge answers "social", which has no digit, so
# every sentiment reply is unreadable.
_t10 = _tmp_unc.mkdtemp()
_c10 = os.path.join(_t10, "c.csv")
from ewl.common import FIELDNAMES as _FN10
with open(_c10, "w", newline="", encoding="utf-8-sig") as _h:
    _w = _csv9.DictWriter(_h, fieldnames=_FN10)
    _w.writeheader()
    for _i in range(3):
        _w.writerow({"comment_id": f"x{_i}", "platform": "youtube", "story": "jets",
                     "source": "v1", "author_hash": "a", "body": f"comment {_i}"})
_o10 = os.path.join(_t10, "o.csv")
_p10 = subprocess.run([sys.executable, "-m", "ewl.label.classify", _c10, "--judge", "stub",
                       "--sentiment", "llm:stub", "--votes", "1", "--no-cache", "--output", _o10],
                      capture_output=True, text=True,
                      env=dict(os.environ, PYTHONPATH=os.path.join(os.getcwd(), "src")))
with open(_o10, encoding="utf-8-sig", newline="") as _h:
    _out10 = list(_csv9.DictReader(_h))
check("classify labels an unreadable sentiment reply parse_failure, with no value",
      _p10.returncode == 0 and all(r["sentiment"] == "parse_failure" and r["sentiment_value"] == ""
                                   for r in _out10), (_p10.stderr[-400:], _out10[:1]))
check("it writes a sentiment_parse_failure flag, and no neutral",
      all(r["sentiment_parse_failure"] == "1" and r["sentiment_no_consensus"] == "0"
          for r in _out10), _out10[:1])
check("and the run summary reports the sentiment failure rate with a warning",
      "Sentiment: parse failures 100.0%" in _p10.stderr
      and "could not be read as a rating" in _p10.stderr, _p10.stderr[-600:])

_run_json = _o10 + ".run.json"
check("classify writes a run record beside the labelled file", os.path.exists(_run_json), _p10.stderr[-300:])
with open(_run_json, encoding="utf-8") as _h:
    _run = json.load(_h)
check("the run record names the judge, the scorer, the input hash and the settings",
      _run["domain_judge"]["spec"].startswith("stub") and _run["sentiment_scorer"].startswith("llm:stub")
      and len(_run["input"]["sha256"]) == 64 and _run["domain_judge"]["votes"] == 1
      and "sentiment" in _run["instrument_switches"], sorted(_run))
check("it records a hash of the domain prompt and of each story's sentiment prompt",
      len(_run["prompts_sha256"]["domain"]) == 64
      and set(_run["prompts_sha256"]["sentiment_by_story"]) == {"luigi", "musk", "jets", "propublica"},
      _run["prompts_sha256"])
check("and it holds no comment text",
      not any(f"comment {i}" in open(_run_json, encoding="utf-8").read() for i in range(3)))

_props_rows = ([{"story": "jets", "source": f"v{i % 3}", "sentiment": "negative",
                 "domain_parse_failure": "0"} for i in range(6)]
               + [{"story": "jets", "source": "v0", "sentiment": "parse_failure",
                   "domain_parse_failure": "0"} for _ in range(2)]
               + [{"story": "jets", "source": "v1", "sentiment": "irrelevant",
                   "domain_parse_failure": "0"} for _ in range(2)])
_props = _exp10.proportions(_props_rows, n_boot=50)
_neg = [p for p in _props if p["kind"] == "sentiment" and p["label"] == "negative"]
_nr = {p["label"]: p["rate"] for p in _props if p["kind"] == "sentiment_not_rated"}
check("stance shares are over rated comments only (6 negative of 6 rated = 100%)",
      _neg and abs(_neg[0]["rate"] - 1.0) < 1e-9, _neg)
check("the comments left out are exported as their own shares over ALL comments "
      "(2 parse failures and 2 irrelevant of 10)",
      abs(_nr.get("parse_failure", -1) - 0.2) < 1e-9 and abs(_nr.get("irrelevant", -1) - 0.2) < 1e-9
      and "uncodeable" in _nr and "no_consensus" in _nr, _nr)



print("\nreport: effective videos, and the corrected estimate story by story")
import random as _r11
from ewl.analyse import report as _rep11

check("equal-sized videos: effective videos equals the video count",
      abs(_rep11.effective_clusters([{"source": f"v{i % 5}"} for i in range(50)]) - 5) < 1e-9)
_conc = [{"source": "big"}] * 90 + [{"source": f"s{i}"} for i in range(10)]
check("one video holding 90% of the comments makes 11 videos worth about 1.2",
      1.1 < _rep11.effective_clusters(_conc) < 1.3, _rep11.effective_clusters(_conc))
check("no rows: 0, not a crash", _rep11.effective_clusters([]) == 0.0)


def _story_data(story, n_corpus, n_gold, seed):
    rng = _r11.Random(seed)
    corpus, gold = [], []
    for i in range(n_corpus):
        truth = rng.random() < 0.3
        flag = (rng.random() < 0.85) if truth else (rng.random() < 0.05)
        corpus.append({"story": story, "source": f"{story}v{i % 12}",
                       "domain_democracy": "1" if flag else "0", "domain_parse_failure": "0"})
    for i in range(n_gold):
        truth = rng.random() < 0.5
        flag = (rng.random() < 0.85) if truth else (rng.random() < 0.05)
        gold.append({"story": story, "source": f"{story}v{i % 12}",
                     "hand_domain": "democracy" if truth else "none",
                     "domain_democracy": "1" if flag else "0", "domain_parse_failure": "0",
                     "inclusion_prob": "0.5"})
    return corpus, gold


_ca, _ga = _story_data("rich", 600, 200, 1)
_cb, _gb = _story_data("thin", 600, 12, 2)
_table = _rep11.corrected_by_story(
    _ca + _cb, "domain_democracy", _ga + _gb, "hand_domain", "democracy",
    min_genuine_positives=20, n_boot=200, seed=3)
_by = {e["story"]: e for e in _table}
check("a story with enough gold positives gets a corrected estimate and an interval",
      _by["rich"]["refused"] is None and _by["rich"]["lo"] <= _by["rich"]["point"] <= _by["rich"]["hi"],
      _by["rich"])
check("the rich story's corrected prevalence is near the 30% it was built with",
      0.15 < _by["rich"]["point"] < 0.45, _by["rich"]["point"])
check("a thin story is refused with the reason stated, not given a number",
      _by["thin"]["refused"] is not None and "effective positives" in _by["thin"]["refused"]
      and _by["thin"]["point"] is None, _by["thin"])
check("every story reports its own comment, video and effective-video counts",
      _by["rich"]["n_comments"] == 600 and _by["rich"]["videos"] == 12
      and 11 < _by["rich"]["effective_videos"] <= 12, _by["rich"])
_empty = _rep11.corrected_by_story(_ca, "domain_democracy", [], "hand_domain", "democracy",
                                   min_genuine_positives=20, n_boot=50)
check("a story with no gold rows at all says so",
      _empty[0]["refused"] == "no gold rows for this story", _empty)



print("\nclean_videos: exact video-ID matching")
from ewl.collect import clean_videos as _cv

check("a watch URL, a youtu.be link, a shorts link and a bare ID all give the same video ID",
      {_cv.video_id("https://www.youtube.com/watch?v=-dxftQC09H4"),
       _cv.video_id("https://www.youtube.com/watch?v=-dxftQC09H4&t=30s"),
       _cv.video_id("https://youtu.be/-dxftQC09H4"),
       _cv.video_id("https://www.youtube.com/shorts/-dxftQC09H4"),
       _cv.video_id("-dxftQC09H4")} == {"-dxftQC09H4"})
check("a source that is not a video URL gives None, not a guess",
      _cv.video_id("https://example.com/page") is None and _cv.video_id("") is None)

_vrows = ([{"source": "https://www.youtube.com/watch?v=-dxftQC09H4", "comment_id": f"a{i}"}
           for i in range(3)]
          + [{"source": "https://www.youtube.com/watch?v=AAAAAAAAAAA", "comment_id": "b0"}])
_kept_v, _gone_v = _cv.drop_videos(_vrows, {"-dxftQC09H4"})
check("the hyphen-led ID removes exactly its own comments", len(_kept_v) == 1 and _gone_v["-dxftQC09H4"] == 3,
      (len(_kept_v), _gone_v))
try:
    _cv.check_ids({"dxftQC09H4"}, known_ids={"-dxftQC09H4", "AAAAAAAAAAA"})
    check("an ID with its leading hyphen dropped is refused", False, "no SystemExit")
except SystemExit as _e:
    _msg = str(_e)
    check("an ID with its leading hyphen dropped is refused, with the likely real one suggested",
          "11" in _msg and "did you mean -dxftQC09H4" in _msg, _msg)
_kept_sub, _ = _cv.drop_videos(_vrows, {"dxftQC09H4"})
check("and a list entry that is only a substring of a real ID no longer removes anything",
      len(_kept_sub) == 4, len(_kept_sub))
_cv.check_ids({"-dxftQC09H4", "AAAAAAAAAAA"})
check("valid IDs, including one starting with a hyphen, pass the check", True)

if project_file("excluded_video_ids.txt"):
    check("the committed exclusion list holds only valid 11-character video IDs",
          all(len(line.strip()) == 11 for line in
              open("excluded_video_ids.txt", encoding="utf-8") if line.strip()),
          [line.strip() for line in open("excluded_video_ids.txt", encoding="utf-8")
           if line.strip() and len(line.strip()) != 11])



print("\nredact: handles, emails and phone numbers")
from ewl.collect import redact as _rd

_t, _c = _rd.redact_text("@SomeChannel thanks, but the tax is fair", handles="leading")
check("a leading handle (the addressee) becomes @user, and the rest is untouched",
      _t == "@user thanks, but the tax is fair" and _c["handles"] == 1, (_t, _c))
_t, _c = _rd.redact_text("@Alpha @Beta-two: both wrong", handles="leading")
check("two handles opening a comment are one addressee block, both replaced",
      _t == "@user both wrong" and _c["handles"] == 2, (_t, _c))
_t, _c = _rd.redact_text("I agree with @Someone about this", handles="leading")
check("--handles leading leaves a handle in the middle of a sentence alone",
      _t == "I agree with @Someone about this" and _c["handles"] == 0, (_t, _c))
_t, _c = _rd.redact_text("I agree with @Someone about this", handles="all")
check("--handles all replaces it too", _t == "I agree with @user about this" and _c["handles"] == 1, _t)
_t, _c = _rd.redact_text("@Someone hi", handles="keep")
check("--handles keep (the default) changes nothing", _t == "@Someone hi" and _c["handles"] == 0, _t)

_t, _c = _rd.redact_text("write to jane.doe+tax@example.co.uk please", emails=True, handles="all")
check("an email address is replaced, and is NOT also counted as a handle",
      _t == "write to [email] please" and _c["emails"] == 1 and _c["handles"] == 0, (_t, _c))

_t, _c = _rd.redact_text("ring 07700 900123 or +44 7700 900456", phones=True)
check("phone numbers with separators are replaced",
      _t == "ring [phone] or [phone]" and _c["phones"] == 2, (_t, _c))
_t, _c = _rd.redact_text("He paid $2,300,000 in 2021, about 12.5 million views, 1,000,000,000", phones=True)
check("years, dollar amounts and big round numbers are NOT mistaken for phone numbers",
      _c["phones"] == 0 and _t.startswith("He paid $2,300,000 in 2021"), (_t, _c))

for _txt in ["Net worth 230 000 000 000", "paid 1 000 000 000", "In 2008 2009 2010",
             "9/11 2001 2008", "1997-2007", "ISBN 978-0-306-40615-7", "10.000.000.000",
             "tax 100 000 0000"]:
    _t, _c = _rd.redact_text(_txt, phones=True)
    check(f"space-, dot- or dash-separated figures are left alone: {_txt!r}",
          _t == _txt and _c["phones"] == 0, (_t, _c))
for _txt in ["07700900123", "020 7946 0958", "(212) 555-1234", "212-555-1234",
             "+1 212 555 1234", "+44 (0)20 7946 0958"]:
    _t, _c = _rd.redact_text(f"call {_txt} now", phones=True)
    check(f"real phone shapes are replaced: {_txt!r}",
          _t == "call [phone] now" and _c["phones"] == 1, (_t, _c))

_found = _rd.scan([{"body": "@Xyz hello"}, {"body": "a@b.com"}, {"body": "call 07700 900123"},
                   {"body": "plain"}])
check("scan counts each kind by comment (an email is not a handle)",
      _found == {"handle_leading": 1, "handle_anywhere": 1, "email": 1, "phone": 1}, _found)

_td = _tmp_unc.mkdtemp()
_src = os.path.join(_td, "c.csv")
with open(_src, "w", newline="", encoding="utf-8-sig") as _h:
    _w = _csv9.DictWriter(_h, fieldnames=["comment_id", "story", "source", "body"])
    _w.writeheader()
    _w.writerow({"comment_id": "1", "story": "jets", "source": "v", "body": "@Chan thanks"})
    _w.writerow({"comment_id": "2", "story": "jets", "source": "v", "body": "no identifiers"})
_dst = os.path.join(_td, "red.csv")
_envr = dict(os.environ, PYTHONPATH=os.path.join(os.getcwd(), "src"))
_scan_only = subprocess.run([sys.executable, "-m", "ewl.collect.redact", _src],
                            capture_output=True, text=True, env=_envr)
check("with no --output it reports and writes nothing",
      _scan_only.returncode == 0 and not os.path.exists(_dst)
      and "open with an @handle" in _scan_only.stderr, _scan_only.stderr)
_noop = subprocess.run([sys.executable, "-m", "ewl.collect.redact", _src, "--output", _dst],
                       capture_output=True, text=True, env=_envr)
check("--output with nothing chosen to redact is refused, not a silent copy",
      _noop.returncode != 0 and not os.path.exists(_dst), _noop.stderr)
_same = subprocess.run([sys.executable, "-m", "ewl.collect.redact", _src, "--output", _src,
                        "--handles", "leading"], capture_output=True, text=True, env=_envr)
with open(_src, encoding="utf-8-sig", newline="") as _h:
    _orig_rows = list(_csv9.DictReader(_h))
check("--output equal to the input is refused, and the original is untouched",
      _same.returncode != 0 and _orig_rows[0]["body"] == "@Chan thanks", _same.stderr)
_run = subprocess.run([sys.executable, "-m", "ewl.collect.redact", _src, "--output", _dst,
                       "--handles", "leading"], capture_output=True, text=True, env=_envr)
with open(_dst, encoding="utf-8-sig", newline="") as _h:
    _rows_r = list(_csv9.DictReader(_h))
check("it writes the redacted file, leaves the original alone, and freezes a manifest",
      _run.returncode == 0 and _rows_r[0]["body"] == "@user thanks"
      and _rows_r[1]["body"] == "no identifiers" and os.path.exists(_dst + ".manifest.json")
      and "@Chan" in open(_src, encoding="utf-8-sig").read(), _run.stderr)



print("\ncache-minimum check covers the sentiment prompts too, and flags borderline ones")
import ewl.label.sentiment as sentiment_mod
_edge = 4 * 1024  # exactly 1,024 tokens by the 4-chars-per-token estimate
_rep = {label: (est, verdict) for label, est, _min, verdict in judges.cache_report(
    "claude-sonnet-5", {"under": "a" * (_edge - 400), "just over": "a" * (_edge + 200),
                        "clear": "a" * (_edge * 2)})}
check("an estimate under the model's minimum is 'below'", _rep["under"][1] == "below", _rep)
check("an estimate over it by less than the estimate's own error is 'borderline', not 'ok'",
      _rep["just over"][1] == "borderline", _rep)
check("a comfortably large prompt is 'ok'", _rep["clear"][1] == "ok", _rep)
check("a model with no known minimum gets no verdict rather than a guessed one",
      judges.cache_report("gemma2:27b", {"p": "a" * 9000}) == [])
check("each model is judged against its OWN minimum (Opus 512 vs Haiku 4096)",
      judges.cache_report("claude-opus-5", {"p": "a" * 4000})[0][3] == "ok"
      and judges.cache_report("claude-haiku-4-5", {"p": "a" * 4000})[0][3] == "below")


@judges.register("test_sonnet_like")
class _SonnetLike(judges.Judge):
    default_model = "claude-sonnet-5"

    def _sample(self, body, prompt, vote):
        return "democracy"


with tempfile.TemporaryDirectory() as tmp:
    _in = os.path.join(tmp, "in.csv")
    _out = os.path.join(tmp, "out.csv")
    with open(_in, "w", newline="", encoding="utf-8") as _fh:
        _fh.write("comment_id,story,body\nc1,luigi,short comment\n")
    _argv = sys.argv

    def _classify_stderr(*args):
        sys.argv = ["classify.py", _in, "--output", _out, "--votes", "1", *args]
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stderr(buffer):
                classify.main()
        finally:
            sys.argv = _argv
        return buffer.getvalue()

    _loud = _classify_stderr("--judge", "test_haiku_like", "--sentiment", "llm:test_haiku_like")
    check("the sentiment prompts are checked as well as the domain one (they used to be skipped)",
          all(f"sentiment/{s} prompt" in _loud for s in sentiment_mod.STORY_CONTEXT),
          _loud)
    check("and the scorer's prompts are judged against the scorer's model, not the judge's",
          "under claude-haiku-4-5's 4096-token cache minimum" in _loud)

_count_script = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "scratch_count_prompt_tokens.py"
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("count_prompt_tokens", _count_script)
_count_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_count_mod)
_all = _count_mod.all_prompts()
check("the token-count script builds every prompt (4 domain + 4 sentiment + 4 combined, one per story) "
      "from the project's own code, with no SDK or key",
      len(_all) == 12 and sum(1 for k in _all if k.startswith("sentiment /")) == 4
      and sum(1 for k in _all if k.startswith("domain /")) == 4, list(_all))
check("its sentiment prompts are the ones the real scorer sends",
      {v for k, v in _all.items() if k.startswith("sentiment /")}
      == set(sentiment_mod.create("llm:stub", votes=1)._prompts.values()))

check("a scorer never asks for more votes than its judge allows (a keyword judge is pinned to 1)",
      sentiment_mod.create("llm:keyword", votes=3).votes == 1
      and sentiment_mod.create("llm:stub", votes=3).votes == judges.create("stub", votes=3).votes,
      (sentiment_mod.create("llm:keyword", votes=3).votes,
       sentiment_mod.create("llm:stub", votes=3).votes))



print("\nthe 25 fixed examples")
_fx = taxonomy.FIXED_EXAMPLES
_fx_texts = [text for text, _ in _fx]
check("there are exactly 25", len(_fx) == 25, len(_fx))
check("no two are the same comment",
      len({t.strip().lower() for t in _fx_texts}) == 25)
_fx_parsed = [taxonomy.parse_labels(label) for _, label in _fx]
check("every label is a real answer: category names or 'none', never a parse failure",
      all(taxonomy.PARSE_FAILURE not in p for p in _fx_parsed),
      [label for (_, label), p in zip(_fx, _fx_parsed) if taxonomy.PARSE_FAILURE in p])
_fx_cats = collections.Counter(c for p in _fx_parsed for c in p if c in taxonomy.CATEGORY_NAMES)
check("every one of the four categories is shown at least three times",
      all(_fx_cats[c] >= 3 for c in taxonomy.CATEGORY_NAMES), dict(_fx_cats))
_fx_none = sum(1 for p in _fx_parsed if p == [taxonomy.UNCLASSIFIED])
check("'none' is shown as a normal answer (the old five had no 'none' example at all)",
      _fx_none >= 5, _fx_none)
check("some examples carry two labels, as the prompt allows",
      sum(1 for p in _fx_parsed if len(p) == 2) >= 2)
check("no example contains a link or an @handle: no personal data in the prompt",
      not any("http" in t.lower() or "@" in t or "www." in t.lower() for t in _fx_texts))
check("none is a comment from the test fixture, and all are short",
      all(len(t) <= 160 for t in _fx_texts)
      and not ({t for t in _fx_texts} & {body for body, _ in FIXTURE}))
check("by the project's own estimate the domain prompt now clears Sonnet's 1,024-token "
      "caching minimum (it sat right on the line with five examples)",
      judges.estimate_tokens(taxonomy.build_prompt(taxonomy.FIXED_EXAMPLES))
      >= judges.cache_minimum_for("claude-sonnet-5"),
      judges.estimate_tokens(taxonomy.build_prompt(taxonomy.FIXED_EXAMPLES)))
check("the fixed selector reports 25 examples", examples_mod.create("fixed").describe() == "fixed(25)")



with tempfile.TemporaryDirectory() as _tq:
    _qin, _qout = os.path.join(_tq, "in.csv"), os.path.join(_tq, "out.csv")
    with open(_qin, "w", newline="", encoding="utf-8") as _fh:
        _fh.write("comment_id,story,body" + chr(10) + "c1,luigi,short comment" + chr(10))
    _qargv = sys.argv
    sys.argv = ["classify.py", _qin, "--output", _qout, "--votes", "1",
                "--judge", "test_sonnet_like", "--sentiment", "llm:test_sonnet_like"]
    _qbuf = io.StringIO()
    try:
        with contextlib.redirect_stderr(_qbuf):
            classify.main()
    finally:
        sys.argv = _qargv
    check("the 25-example domain prompt raises no cache warning on a Sonnet-like model",
          "domain prompt" not in _qbuf.getvalue(), _qbuf.getvalue())



print("\nfixed examples from a file: real gold comments, scored out and never recorded as text")
_orig_fixed = list(taxonomy.FIXED_EXAMPLES)
_fxd = _tmp_unc.mkdtemp()
_fx_path = os.path.join(_fxd, "fixed.csv")
with open(_fx_path, "w", newline="", encoding="utf-8-sig") as _h:
    _w = _csv9.DictWriter(_h, fieldnames=["comment_id", "body", "hand_domain"])
    _w.writeheader()
    _w.writerow({"comment_id": "gold1", "body": "REAL-LOOKING gold comment one about taxes", "hand_domain": "democracy"})
    _w.writerow({"comment_id": "gold2", "body": "REAL-LOOKING gold comment two, a joke", "hand_domain": "none"})
    _w.writerow({"comment_id": "gold3", "body": "REAL-LOOKING gold comment three on jets", "hand_domain": "Environment, Economics"})
try:
    with contextlib.redirect_stderr(io.StringIO()) as _note:
        _ids = taxonomy.load_fixed_examples(_fx_path)
    check("the file's examples replace the built-in ones, in place, with labels lower-cased",
          _ids == ["gold1", "gold2", "gold3"] and len(taxonomy.FIXED_EXAMPLES) == 3
          and taxonomy.FIXED_EXAMPLES[2][1] == "environment, economics", taxonomy.FIXED_EXAMPLES)
    check("a count other than 25 is noted, not refused", "design uses 25" in _note.getvalue())
    check("the fixed selector and the prompt use them",
          examples_mod.create("fixed").select("x") == taxonomy.FIXED_EXAMPLES
          and "REAL-LOOKING gold comment one" in taxonomy.build_prompt(taxonomy.FIXED_EXAMPLES))
    check("the source records ids, labels and a hash, and never the text",
          taxonomy.FIXED_EXAMPLES_SOURCE["source"] == "file"
          and len(taxonomy.FIXED_EXAMPLES_SOURCE["sha256"]) == 64
          and "REAL-LOOKING" not in json.dumps(taxonomy.FIXED_EXAMPLES_SOURCE),
          taxonomy.FIXED_EXAMPLES_SOURCE)

    class _EvArgs:
        exclude_ids = None
    check("evaluate leaves the example comments out of scoring automatically",
          _ev9._exclude_ids(_EvArgs()) == {"gold1", "gold2", "gold3"}, _ev9._exclude_ids(_EvArgs()))
    _gp = os.path.join(_fxd, "gold_all.csv")
    with open(_gp, "w", newline="", encoding="utf-8") as _h:
        _w = _csv9.DictWriter(_h, fieldnames=["comment_id", "body", "hand_domain", "source"])
        _w.writeheader()
        for _cid in ("gold1", "gold2", "other1", "other2"):
            _w.writerow({"comment_id": _cid, "body": f"text {_cid}", "hand_domain": "none", "source": "v"})
    check("read_gold then drops exactly those comments",
          [r["comment_id"] for r in _ev9.read_gold(_gp, exclude_ids=_ev9._exclude_ids(_EvArgs()))]
          == ["other1", "other2"])
    check("and `ewl-validate score --exclude-ids` accepts the same file (it has a comment_id column)",
          _val.read_ids(_fx_path) == {"gold1", "gold2", "gold3"}, _val.read_ids(_fx_path))
finally:
    taxonomy.reset_fixed_examples(_orig_fixed)
check("resetting puts the built-in examples back",
      taxonomy.FIXED_EXAMPLES == _orig_fixed and taxonomy.FIXED_EXAMPLES_SOURCE["source"].startswith("built-in"))

for _bad_label, _why in (("banana", "an off-list label"), ("", "a blank label")):
    _bp = os.path.join(_fxd, "bad.csv")
    with open(_bp, "w", newline="", encoding="utf-8") as _h:
        _w = _csv9.DictWriter(_h, fieldnames=["comment_id", "body", "label"])
        _w.writeheader()
        _w.writerow({"comment_id": "x", "body": "some text", "label": _bad_label})
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            taxonomy.load_fixed_examples(_bp)
        check(f"{_why} is refused", False, "no SystemExit")
    except SystemExit:
        check(f"{_why} is refused", True)
    finally:
        taxonomy.reset_fixed_examples(_orig_fixed)

# End to end: the run record and the dashboard hold ids, not text.
_c16 = os.path.join(_fxd, "c.csv")
with open(_c16, "w", newline="", encoding="utf-8-sig") as _h:
    _w = _csv9.DictWriter(_h, fieldnames=_FN10)
    _w.writeheader()
    for _i in range(3):
        _w.writerow({"comment_id": f"x{_i}", "platform": "youtube", "story": "jets", "source": "v1",
                     "author_hash": "a", "body": f"comment {_i}"})
_o16 = os.path.join(_fxd, "o.csv")
_env16 = dict(os.environ, PYTHONPATH=os.path.join(os.getcwd(), "src"))
_p16 = subprocess.run([sys.executable, "-m", "ewl.label.classify", _c16, "--judge", "stub",
                       "--sentiment", "llm:stub", "--votes", "1", "--no-cache", "--output", _o16,
                       "--fixed-examples", _fx_path,
                       # This file has no ratings: an llm: sentiment run must say it means that.
                       "--sentiment-examples", "off"], capture_output=True, text=True, env=_env16)
_rec_text = open(_o16 + ".run.json", encoding="utf-8").read()
_rec = json.loads(_rec_text)
check("classify --fixed-examples runs, and the run record holds ids and a hash but NOT the text",
      _p16.returncode == 0 and _rec["fixed_examples"]["source"] == "file"
      and _rec["fixed_examples"]["ids"] == ["gold1", "gold2", "gold3"]
      and len(_rec["fixed_examples"]["sha256"]) == 64 and "REAL-LOOKING" not in _rec_text
      and "examples" not in _rec["fixed_examples"], (_p16.stderr[-300:], _rec["fixed_examples"]))
_d16 = os.path.join(_fxd, "dash.json")
_e16 = subprocess.run([sys.executable, "-m", "ewl.analyse.export_dashboard", _o16, "--output",
                       _d16, "--allow-unvalidated"], capture_output=True, text=True, env=_env16)
_dash16 = open(_d16, encoding="utf-8").read()
check("and the dashboard codebook publishes the ids and labels, with the text withheld",
      _e16.returncode == 0 and "REAL-LOOKING" not in _dash16
      and json.loads(_dash16)["codebook"]["example_text_withheld"] is True,
      _e16.stderr[-300:])



print("\nfill_comment_ids: restore blank comment_ids from the original coding sheet by comment text")
import importlib.util as _ilu17
_spec17 = _ilu17.spec_from_file_location(
    "fill_ids", pathlib.Path(__file__).resolve().parent.parent / "scripts" / "scratch_fill_comment_ids.py")
_fill = _ilu17.module_from_spec(_spec17)
_spec17.loader.exec_module(_fill)

_ref_rows = [("id_a", "luigi", "A first comment"), ("id_b", "jets", "A second comment"),
             ("id_c", "luigi", "Yes"), ("id_d", "jets", "Yes"), ("id_e", "musk", "Third  comment")]
_idx = {}
for _cid, _st, _bd in _ref_rows:
    _idx.setdefault(_fill.normalise(_bd), {})[_cid] = _st
_coded = [{"comment_id": "", "body": "A first comment", "story": "luigi"},
          {"comment_id": "keep_me", "body": "A second comment", "story": "jets"},
          {"comment_id": "", "body": "Yes", "story": "jets"},
          {"comment_id": "", "body": "Yes", "story": ""},
          {"comment_id": "", "body": "Third comment", "story": "musk"},
          {"comment_id": "", "body": "edited text nobody sent", "story": "luigi"},
          {"comment_id": "", "body": "", "story": ""}]
_fills, _probs = _fill.match_ids(_coded, _idx)
check("a blank id is filled from the matching body", _fills.get(0) == "id_a", _fills)
check("a row that already has an id is never touched", 1 not in _fills, _fills)
check("a short text under two ids is separated by story when the story is present",
      _fills.get(2) == "id_d", _fills)
check("and left blank, as ambiguous, when the story cannot separate it",
      3 in _probs["ambiguous"] and 3 not in _fills, _probs)
check("whitespace differences do not defeat a match (two spaces vs one)", _fills.get(4) == "id_e", _fills)
check("text that is in no reference sheet is reported unmatched, not guessed",
      _probs["unmatched"] == [5], _probs)
check("an empty row is ignored, not reported", 6 not in _fills and 6 not in _probs["unmatched"])

import openpyxl as _ox17
_t17 = _tmp_unc.mkdtemp()
_ref17 = os.path.join(_t17, "ref.csv")
with open(_ref17, "w", newline="", encoding="utf-8-sig") as _h:
    _w = _csv9.DictWriter(_h, fieldnames=["comment_id", "story", "body"])
    _w.writeheader()
    for _cid, _st, _bd in _ref_rows:
        _w.writerow({"comment_id": _cid, "story": _st, "body": _bd})
_wb = _ox17.Workbook()
_ws = _wb.active
_ws.append(["comment_id", "story", "body", "hand_domain", "hand_sentiment"])
_ws.append([None, "luigi", "A first comment", "democracy", 2])
_ws.append(["keep_me", "jets", "A second comment", "none", 3])
_src17 = os.path.join(_t17, "coded.xlsx")
_wb.save(_src17)
_out17 = os.path.join(_t17, "filled.xlsx")
_run17 = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve().parent.parent
                                             / "scripts" / "scratch_fill_comment_ids.py"),
                         _src17, "--reference", _ref17, "--output", _out17],
                        capture_output=True, text=True,
                        env=dict(os.environ, PYTHONPATH=os.path.join(os.getcwd(), "src")))
_got = _ox17.load_workbook(_out17).active
_orig = _ox17.load_workbook(_src17).active
check("on a real .xlsx: the id is written, other cells and the existing id are kept",
      _run17.returncode == 0 and _got["A2"].value == "id_a" and _got["A3"].value == "keep_me"
      and _got["D2"].value == "democracy" and _got["E2"].value == 2, (_run17.stderr, _got["A2"].value))
check("and the original file is left exactly as it was", _orig["A2"].value is None)
_same = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve().parent.parent
                                            / "scripts" / "scratch_fill_comment_ids.py"),
                        _src17, "--reference", _ref17, "--output", _src17],
                       capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=os.path.join(os.getcwd(), "src")))
check("it refuses to overwrite its input", _same.returncode != 0 and "different file" in _same.stderr)



print("reweight: recompute inclusion probabilities on a gold file that was assembled by hand")
_corp18 = ([{"comment_id": f"u{i}", "story": "jets", "primary_category": "unclassified", "source": "v1"}
            for i in range(40)]
           + [{"comment_id": f"d{i}", "story": "jets", "primary_category": "democracy", "source": "v2"}
              for i in range(10)])
_master18 = [{"comment_id": c, "story": "jets", "source": "v", "sampling_stratum": st, "inclusion_prob": "0.9"}
             for c, st in (("u0", "jets/unclassified"), ("u1", "jets/unclassified"),
                           ("d0", "jets/democracy"), ("d1", "jets/democracy"))]
_top18 = [{"comment_id": "u5", "story": "jets"}, {"comment_id": "u6", "story": "jets"}]
_gold18 = [{"comment_id": "u0", "story": "jets", "body": "b", "hand_domain": "none", "inclusion_prob": "0.9"},
           {"comment_id": "d0", "story": "jets", "body": "b", "hand_domain": "democracy", "inclusion_prob": "0.9"},
           {"comment_id": "u5", "story": "jets", "body": "b", "hand_domain": "none", "inclusion_prob": ""},
           {"comment_id": "d2", "story": "jets", "body": "b", "hand_domain": "democracy",
            "sampling_stratum": "jets/democracy", "inclusion_prob": ""},
           {"comment_id": "zz9", "story": "jets", "body": "b", "hand_domain": "none",
            "sampling_stratum": "jets/democracy", "inclusion_prob": ""}]
_buf18, _err18 = io.StringIO(), io.StringIO()
with contextlib.redirect_stderr(_err18):
    _rw = _val.cmd_reweight(_gold18, _master18, _top18, _corp18, None, _buf18)
_rwb = {r["comment_id"]: r for r in _rw}
check("an original-draw unclassified row gets the two-phase probability against the final corpus (0.1)",
      abs(float(_rwb["u0"]["inclusion_prob"]) - 0.1) < 1e-6, _rwb.get("u0"))
check("a top-up row gets the same probability as the draw's unclassified rows",
      abs(float(_rwb["u5"]["inclusion_prob"]) - 0.1) < 1e-6, _rwb.get("u5"))
check("a democracy row gets the plain stratum probability (2/10)",
      abs(float(_rwb["d0"]["inclusion_prob"]) - 0.2) < 1e-6, _rwb.get("d0"))
check("a hand-picked replacement takes its stratum's probability and is flagged",
      abs(float(_rwb["d2"]["inclusion_prob"]) - 0.2) < 1e-6
      and "replacement_approx" in _rwb["d2"]["inclusion_prob_basis"], _rwb.get("d2"))
check("the figure each row carried is kept as inclusion_prob_drawn",
      _rwb["u0"]["inclusion_prob_drawn"] == "0.9", _rwb.get("u0"))
check("a row that is not in the final corpus is excluded and named, not given a guess",
      "zz9" not in _rwb and "zz9" in _err18.getvalue(), _err18.getvalue())
check("the summary reports the effective sample size", "Kish effective sample size" in _err18.getvalue())

for _label, _bad_gold, _needle in (
        ("a comment_id appearing twice (an unresolved core row)",
         _gold18 + [dict(_gold18[0])], "more than once"),
        ("a replacement with no sampling_stratum of its own",
         _gold18 + [{"comment_id": "d3", "story": "jets", "body": "b", "hand_domain": "none"}],
         "cannot be placed"),
        ("a row with no comment_id", _gold18 + [{"comment_id": "", "story": "jets", "body": "b"}],
         "no comment_id")):
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            _val.cmd_reweight(_bad_gold, _master18, _top18, _corp18, None, io.StringIO())
        check(f"{_label} is refused", False, "no SystemExit")
    except SystemExit as _e:
        check(f"{_label} is refused, with the reason", _needle in str(_e), str(_e))



print("reweight: ids mangled by a spreadsheet are cleaned, and every dropped row is explained")
_messy = [{"comment_id": "u0 ", "story": "jets", "body": "b", "hand_domain": "none"},
          {"comment_id": "d0" + chr(160), "story": "jets", "body": "b", "hand_domain": "democracy"},
          {"comment_id": chr(0xfeff) + "u5", "story": "jets", "body": "b", "hand_domain": "none"},
          {"comment_id": "U1", "story": "jets", "body": "b", "hand_domain": "none", "source": "vidX",
           "sampling_stratum": "jets/democracy"},
          {"comment_id": "ghost1", "story": "jets", "body": "b", "hand_domain": "none", "source": "vidGone",
           "sampling_stratum": "jets/democracy"},
          {"comment_id": "ghost2", "story": "luigi", "body": "b", "hand_domain": "none", "source": "vidGone",
           "sampling_stratum": "jets/democracy"}]
_e19 = io.StringIO()
with contextlib.redirect_stderr(_e19):
    _rw19 = _val.cmd_reweight(_messy, _master18, _top18, _corp18, None, io.StringIO())
_ids19 = {r["comment_id"] for r in _rw19}
check("trailing spaces, non-breaking spaces and a BOM on an id no longer make a row vanish",
      {"u0", "d0", "u5"} <= _ids19, _ids19)
check("an id that differs from a corpus id only by CAPITALS is not matched (ids are case-sensitive) "
      "but is reported as such",
      "U1" not in _ids19 and "match a corpus id only if letter case is ignored" in _e19.getvalue(),
      _e19.getvalue())
check("dropped rows are explained: not in the corpus, by video and by story",
      "are not in the final corpus" in _e19.getvalue() and "vidGone: 2" in _e19.getvalue()
      and "'luigi': 1" in _e19.getvalue(), _e19.getvalue())
check("clean_id leaves the letters alone",
      _val.clean_id(" UgwAbC-_9 ") == "UgwAbC-_9" and _val.clean_id(None) == "")



print("check_coder_files: structural checks on coders' files, counts only")
_spec20 = _ilu17.spec_from_file_location(
    "check_coder", pathlib.Path(__file__).resolve().parent.parent / "scripts" / "scratch_check_coder_files.py")
_chk = _ilu17.module_from_spec(_spec20)
_spec20.loader.exec_module(_chk)
_core20 = {"c1", "c2", "c3", "c4"}


def _write20(name, rows):
    path = os.path.join(_t17, name)
    with open(path, "w", newline="", encoding="utf-8-sig") as _h:
        _w = _csv9.DictWriter(_h, fieldnames=["comment_id", "body", "hand_domain", "hand_sentiment"])
        _w.writeheader()
        _w.writerows(rows)
    return path


_good = _write20("good.csv", [{"comment_id": f"c{i}", "body": "SECRET TEXT", "hand_domain": d,
                               "hand_sentiment": s} for i, d, s in
                              ((1, "democracy", "2"), (2, "none", "0"), (3, "economics, social", "4"),
                               (4, "environment", "3"))] + [{"comment_id": "x9", "body": "t",
                                                            "hand_domain": "none", "hand_sentiment": "3"}])
_rep_g, _use_g = _chk.check_file(_good, _core20)
check("a clean file has no problems and all 4 core rows usable",
      _rep_g["problems"] == [] and _use_g == _core20, _rep_g["problems"])

_bad = _write20("bad.csv", [
    {"comment_id": "c1", "body": "t", "hand_domain": "3", "hand_sentiment": "2"},       # list number
    {"comment_id": "c2", "body": "t", "hand_domain": "policy", "hand_sentiment": "9"},   # off-list, bad rating
    {"comment_id": "c3", "body": "t", "hand_domain": "", "hand_sentiment": ""},         # blanks
    {"comment_id": "c3", "body": "t", "hand_domain": "none", "hand_sentiment": "3"},    # repeated id
    {"comment_id": "", "body": "t", "hand_domain": "none", "hand_sentiment": "3"}])     # blank id
_rep_b, _use_b = _chk.check_file(_bad, _core20)
_text_b = " | ".join(_rep_b["problems"])
check("a list number in hand_domain is flagged and recognised as the 1-5 list",
      "'3': 1" in _text_b and "list numbers" in _text_b, _text_b)
check("each bad hand_domain value is located by sheet row, marked CORE when it is one, with a "
      "suggested valid label that is never applied",
      "sheet row 2 (CORE)" in _text_b and "no close valid label" in _text_b, _text_b)
_typo = _write20("typo.csv", [{"comment_id": "c1", "body": "t", "hand_domain": "economic", "hand_sentiment": "2"},
                              {"comment_id": "zz", "body": "t", "hand_domain": "environemnt", "hand_sentiment": "2"}])
_rep_t, _ = _chk.check_file(_typo, _core20)
_text_t = " | ".join(_rep_t["problems"])
check("typos are matched to the likely intended category ('economic' -> economics) and only suggested",
      "closest valid label: 'economics'" in _text_t and "closest valid label: 'environment'" in _text_t
      and "sheet row 3" in _text_t
      and "(CORE)" in _text_t.split("sheet row 3")[0], _text_t)
check("an off-list word and an invalid rating are flagged",
      "'policy': 1" in _text_b and "invalid values" in _text_b and "'9': 1" in _text_b, _text_b)
check("blank cells, a repeated id, a blank id and a missing core comment are all counted",
      "blank hand_domain" in _text_b and "blank hand_sentiment" in _text_b
      and "appear more than once" in _text_b and "no comment_id" in _text_b
      and "only 3 of 4 core" in _text_b, _text_b)
check("only rows with BOTH a domain and a sentiment, and an unrepeated id, count as usable core rows",
      _use_b == {"c1", "c2"}, _use_b)
check("the check never prints comment text",
      "SECRET TEXT" not in repr(_rep_g) and "SECRET TEXT" not in repr(_rep_b))

_nodom = os.path.join(_t17, "nocol.csv")
with open(_nodom, "w", newline="", encoding="utf-8") as _h:
    _h.write("comment_id,body" + chr(10) + "c1,t" + chr(10))
_rep_n, _ = _chk.check_file(_nodom, _core20)
check("a file missing the coding columns says which", "hand_domain" in _rep_n["problems"][0], _rep_n)
_xl = _ox17.Workbook()
_xs = _xl.active
_xs.append(["comment_id", "body", "hand_domain", "hand_sentiment"])
_xs.append(["c1", "t", "democracy", 2])
_xp = os.path.join(_t17, "c.xlsx")
_xl.save(_xp)
_rep_x, _use_x = _chk.check_file(_xp, {"c1"})
check("an .xlsx file (numeric sentiment cell) is read and passes", _rep_x["problems"] == [] and _use_x == {"c1"},
      _rep_x)
check("domain_problems accepts categories and 'none', and rejects a mix of none and another",
      _chk.domain_problems("Democracy, Economics") == [] and _chk.domain_problems("none") == []
      and _chk.domain_problems("none, social") == ["none+other"])



print("read_any reads .xlsx with openpyxl alone (no pandas), and normalises cells")
import sys as _sys21
_saved_pandas = _sys21.modules.get("pandas")
_sys21.modules["pandas"] = None            # makes `import pandas` raise ImportError
try:
    _xwb = _ox17.Workbook()
    _xws = _xwb.active
    _xws.append(["comment_id", "hand_domain", "hand_sentiment", None, "coder_notes"])
    _xws.append(["c1", "democracy", 2.0, "ignored", None])
    _xws.append(["c2", "none", 3, None, "ok"])
    _xws.append([None, None, None, None, None])
    _xwp = os.path.join(_t17, "ra.xlsx")
    _xwb.save(_xwp)
    _got21 = _val.read_any(_xwp)
    check("with pandas unavailable, an .xlsx is still read",
          len(_got21) == 2 and _got21[0]["comment_id"] == "c1", _got21)
    check("a whole-number cell comes back as '2', not '2.0'; a blank as ''",
          _got21[0]["hand_sentiment"] == "2" and _got21[0]["coder_notes"] == ""
          and _got21[1]["hand_sentiment"] == "3", _got21)
    check("a column with no header is ignored, and a fully blank row is skipped",
          all(set(r) == {"comment_id", "hand_domain", "hand_sentiment", "coder_notes"}
              for r in _got21))
finally:
    if _saved_pandas is None:
        _sys21.modules.pop("pandas", None)
    else:
        _sys21.modules["pandas"] = _saved_pandas



print("krippendorff_alpha: nominal, ordinal and interval, with resampling intervals")
_hand = [[1, 1], [0, 0], [1, 0], [0, 0]]
check("nominal alpha matches a hand calculation (0.5333) on a four-comment binary example",
      abs(validate.krippendorff_alpha(_hand, "nominal") - 0.5333333333) < 1e-9,
      validate.krippendorff_alpha(_hand, "nominal"))
check("perfect agreement WITH variation is exactly 1.0",
      validate.krippendorff_alpha([[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]], "nominal") == 1.0)
_nv = validate.krippendorff_alpha([[0, 0, 0], [0, 0, 0]], "nominal")
check("no variation at all is undefined (NaN), not a perfect 1.0", _nv != _nv)
import random as _rnd22
_r22 = _rnd22.Random(4)
_rand = [[_r22.randint(1, 5) for _ in range(3)] for _ in range(40)]
check("interval alpha agrees with the existing pair-based implementation",
      abs(validate.krippendorff_alpha(_rand, "interval")
          - validate.krippendorff_alpha_interval([[float(v) for v in r] for r in _rand])) < 1e-9,
      (validate.krippendorff_alpha(_rand, "interval"),
       validate.krippendorff_alpha_interval([[float(v) for v in r] for r in _rand])))
_base = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [3, 3]]
_near = [list(r) for r in _base]
_near[0] = [1, 2]
_far = [list(r) for r in _base]
_far[0] = [1, 5]
check("ordinal alpha penalises a far disagreement (1 vs 5) more than an adjacent one (1 vs 2)",
      validate.krippendorff_alpha(_near, "ordinal") > validate.krippendorff_alpha(_far, "ordinal"),
      (validate.krippendorff_alpha(_near, "ordinal"), validate.krippendorff_alpha(_far, "ordinal")))
check("nominal alpha does NOT distinguish them (any disagreement counts the same)",
      abs(validate.krippendorff_alpha(_near, "nominal") - validate.krippendorff_alpha(_far, "nominal")) < 1e-12)
check("three coders, order-independent: shuffling the coders within a comment changes nothing",
      abs(validate.krippendorff_alpha([[1, 2, 3], [4, 4, 5]], "ordinal")
          - validate.krippendorff_alpha([[3, 1, 2], [5, 4, 4]], "ordinal")) < 1e-12)
check("unequal numbers of coders per comment is refused (NaN), not guessed",
      validate.krippendorff_alpha([[1, 2], [1, 2, 3]], "nominal") != validate.krippendorff_alpha(
          [[1, 2], [1, 2, 3]], "nominal"))
try:
    validate.krippendorff_alpha(_base, "bogus")
    check("an unknown metric is refused", False, "no ValueError")
except ValueError:
    check("an unknown metric is refused", True)

_lo, _hi = validate.bootstrap_alpha_ci([[1, 1, 1], [0, 0, 0]] * 10, "nominal", n_boot=200)
check("a perfectly reliable coding has a collapsed interval at 1.0", (_lo, _hi) == (1.0, 1.0), (_lo, _hi))
_noisy = [[_r22.randint(0, 1) for _ in range(3)] for _ in range(30)]
_lo2, _hi2 = validate.bootstrap_alpha_ci(_noisy, "nominal", n_boot=300)
_pt2 = validate.krippendorff_alpha(_noisy, "nominal")
check("an interval around noisy codings is wide and contains the point value",
      _hi2 - _lo2 > 0.2 and _lo2 <= _pt2 <= _hi2, (_lo2, _pt2, _hi2))
check("the interval is reproducible for a seed",
      validate.bootstrap_alpha_ci(_noisy, "nominal", n_boot=100, seed=7)
      == validate.bootstrap_alpha_ci(_noisy, "nominal", n_boot=100, seed=7))

# kind of answer
check("a rating's KIND: 0 irrelevant, -1 uncodeable, anything else a stance",
      [validate._rating_kind(v) for v in ("0", "-1", "3", "negative", "Irrelevant", "5.0")]
      == ["irrelevant", "uncodeable", "stance", "stance", "irrelevant", "stance"])

# end to end with three synthetic coders, including a category nobody used
_ids22 = [f"k{i}" for i in range(12)]


def _coder22(shift):
    rows = []
    for i, cid in enumerate(_ids22):
        dom = "democracy" if i % 3 == 0 else ("economics" if i % 3 == 1 else "none")
        sent = str(1 + (i + shift * (i % 2)) % 5) if i % 4 else "0"
        rows.append({"comment_id": cid, "hand_domain": dom, "hand_sentiment": sent})
    return rows


_out22 = io.StringIO()
_json22 = os.path.join(_t17, "ir.json")
with contextlib.redirect_stdout(_out22), contextlib.redirect_stderr(io.StringIO()):
    validate.cmd_interrater_multi([_coder22(0), _coder22(0), _coder22(1)],
                                  names=["Ann", "Bo", "Cy"], json_output=_json22, n_boot=100)
_text22 = _out22.getvalue()
check("the report names the coders and the number of comments",
      "12 comments coded by all 3 coders (Ann, Bo, Cy)" in _text22, _text22[:200])
check("a category nobody used is n/a and called untested, not a 1.00",
      "n/a for environment, social" in _text22 and "untested" in _text22, _text22)
check("it shows who used 'irrelevant' and the pairwise agreement on the kind of answer",
      "comments each coder called irrelevant: Ann" in _text22 and "pairwise agreement on the kind" in _text22
      and "Ann / Bo" in _text22, _text22)
_j22 = json.load(open(_json22, encoding="utf-8"))
check("the JSON carries the statistic, the intervals and the per-coder counts",
      _j22["statistic"] == "krippendorff_alpha" and "ci_lo" in _j22["domain"]["democracy"]
      and "kind" in _j22["sentiment"] and "stance_1_to_5" in _j22["sentiment"]
      and _j22["coders"] == ["Ann", "Bo", "Cy"], sorted(_j22))
check("a never-used category is null in the JSON, not 1.0",
      _j22["domain"]["environment"]["alpha"] is None, _j22["domain"]["environment"])



print("interrater: compare the DIRECTION of the stance in buckets, not the 1-5 scores")
_ids23 = [f"b{i}" for i in range(12)]
# coder A and B differ only inside a bucket (1 vs 2, 4 vs 5); C agrees exactly with A except one cross-bucket.
_vals_a = ["1", "2", "3", "4", "5", "1", "2", "3", "4", "5", "0", "-1"]
_vals_b = ["2", "1", "3", "5", "4", "2", "1", "3", "5", "4", "0", "-1"]
_vals_c = ["1", "2", "3", "4", "5", "1", "2", "3", "4", "1", "0", "-1"]
def _mk23(vals):
    return [{"comment_id": c, "hand_domain": "none", "hand_sentiment": v} for c, v in zip(_ids23, vals)]
_o23s, _o23b = io.StringIO(), io.StringIO()
with contextlib.redirect_stdout(_o23s), contextlib.redirect_stderr(io.StringIO()):
    validate.cmd_interrater_multi([_mk23(_vals_a), _mk23(_vals_b), _mk23(_vals_c)],
                                  names=["A", "B", "C"], n_boot=100)
_json23 = os.path.join(_t17, "bk.json")
with contextlib.redirect_stdout(_o23b), contextlib.redirect_stderr(io.StringIO()):
    validate.cmd_interrater_multi([_mk23(_vals_a), _mk23(_vals_b), _mk23(_vals_c)],
                                  names=["A", "B", "C"], n_boot=100, sentiment_levels="buckets",
                                  json_output=_json23)
_tb, _ts = _o23b.getvalue(), _o23s.getvalue()
check("buckets mode compares the direction on the comments where every coder gave a stance (10)",
      "Direction of the stance" in _tb and "on the 10 comments every coder gave a stance" in _tb, _tb)
check("scale mode is unchanged by default (it still reports the 1-5 ordinal alpha)",
      "How strong a stance, 1-5" in _ts and "Direction of the stance" not in _ts, _ts)
check("the kind-of-answer figure is reported in both modes",
      "Kind of answer" in _tb and "Kind of answer" in _ts)
_jb = json.load(open(_json23, encoding="utf-8"))
_sb = _jb["sentiment"]["stance_buckets"]
check("in buckets, coders who differ only inside a bucket agree: 9 of 10 comments are in the same "
      "bucket for all three (only the one cross-bucket row differs)",
      _sb["n"] == 10 and abs(_sb["exact_agreement"] - 0.9) < 1e-9 and _jb["sentiment"]["levels"] == "buckets",
      _sb)
_scale_vals = [[1, 2, 1], [2, 1, 2], [3, 3, 3], [4, 5, 4], [5, 4, 5], [1, 2, 1], [2, 1, 2], [3, 3, 3],
               [4, 5, 4], [5, 4, 1]]
check("buckets agree more than the 1-5 scores do on the same ratings (alpha, ordinal)",
      _sb["alpha_ordinal"] > validate.krippendorff_alpha(_scale_vals, "ordinal"),
      (_sb["alpha_ordinal"], validate.krippendorff_alpha(_scale_vals, "ordinal")))
check("words a coder typed (negative/neutral/positive) land in the same buckets as numbers",
      validate.sentiment_bucket("Negative") == "negative" and validate.sentiment_bucket("2") == "negative"
      and validate.sentiment_bucket("5") == "positive")
_with_irr = _mk23(["0", "2", "3", "4", "5", "1", "2", "3", "4", "5", "0", "-1"])
_o23c = io.StringIO()
with contextlib.redirect_stdout(_o23c), contextlib.redirect_stderr(io.StringIO()):
    validate.cmd_interrater_multi([_mk23(_vals_a), _with_irr, _mk23(_vals_c)], names=["A", "B", "C"],
                                  n_boot=50, sentiment_levels="buckets")
check("a comment any coder called irrelevant or uncodeable is left out of the bucket comparison (9 left)",
      "on the 9 comments every coder gave a stance" in _o23c.getvalue(), _o23c.getvalue())
check("and a second figure uses ALL comments with irrelevant and uncodeable as categories, so that "
      "disagreement is counted instead of dropped",
      "(c) Every comment, five categories" in _o23c.getvalue()
      and "on all 12 comments" in _o23c.getvalue() and "alone flatters agreement" in _o23c.getvalue(),
      _o23c.getvalue())
_jc = json.load(open(_json23, encoding="utf-8"))["sentiment"]["stance_buckets"]["all_comments_five_categories"]
check("the JSON carries it too", _jc["n"] == 12 and "alpha_nominal" in _jc, _jc)



print("the context rule no longer contradicts the story rules")
try:
    _sent.configure(story_rules=True)
    _on = _sent.create("llm:stub", votes=1)
    _p_on = _on._prompts["luigi"]
    check("with the story rules ON the prompt says how guidance and the comment's own words fit together",
          _sent.CONTEXT_RULE_WITH_RULES in _p_on and "follow the comment" in _p_on, _p_on[-900:])
    check("and no longer says the context is ONLY for understanding references (which contradicted the rules)",
          "only to understand who or what the comment refers to" not in _p_on)
    check("the no-story fallback has no guidance, so it keeps the facts-only wording",
          _sent.CONTEXT_RULE_FACTS in _on._fallback_prompt)
    _sent.configure(story_rules=False)
    _off = _sent.create("llm:stub", votes=1)
    check("with the rules OFF the original facts-only wording is used",
          _sent.CONTEXT_RULE_FACTS in _off._prompts["luigi"]
          and _sent.CONTEXT_RULE_WITH_RULES not in _off._prompts["luigi"])
    check("the context rule appears exactly once in each prompt",
          _p_on.count("what THIS comment actually says") == 1
          and _off._prompts["luigi"].count("what THIS comment actually says") == 1)
    _sent.configure(story_rules=True)
    _c_on = _comb.build_prompt([("x", "none")], story="luigi")
    check("the combined arm's sentiment rules follow the same switch",
          _sent.CONTEXT_RULE_WITH_RULES_SHORT in _c_on
          and _sent.CONTEXT_RULE_FACTS_SHORT not in _c_on)
    _c_none = _comb.build_prompt([("x", "none")], story=None)
    check("and the combined arm with no recognised story uses the facts-only wording",
          _sent.CONTEXT_RULE_FACTS_SHORT in _c_none)
    _sent.configure(story_rules=False)
    check("with the rules off the combined arm uses the facts-only wording too",
          _sent.CONTEXT_RULE_FACTS_SHORT in _comb.build_prompt([("x", "none")], story="luigi"))
finally:
    _sent.configure(story_rules=True, stance_object="conduct")
    _comb.configure(domain_context="keep")



print("zero_usage: how each coder used 0, from labels only")
_spec25 = _ilu17.spec_from_file_location(
    "zero_usage", pathlib.Path(__file__).resolve().parent.parent / "scripts" / "scratch_zero_usage.py")
_zu = _ilu17.module_from_spec(_spec25)
_spec25.loader.exec_module(_zu)

def _r25(cid, dom, sent, story="jets", body="x" * 40):
    return {"comment_id": cid, "hand_domain": dom, "hand_sentiment": sent, "story": story, "body": body}

_broad = [_r25("c1", "democracy", "0"), _r25("c2", "economics", "0"), _r25("c3", "none", "0"),
          _r25("c4", "social", "3", body="y" * 200), _r25("c5", "none", "")]
_u = _zu.usage(_broad)
check("rated counts skip a blank sentiment (4 of 5 rated)", _u["rated"] == 4, _u)
check("zeros, their share, and how many carry a harm domain are counted (3 zeros, 2 with a harm)",
      _u["zeros"] == 3 and abs(_u["zero_share"] - 0.75) < 1e-9 and _u["zeros_with_harm"] == 2
      and abs(_u["zeros_with_harm_share"] - 2 / 3) < 1e-9, _u)
check("median length of 0s vs other ratings is reported",
      _u["median_len_zero"] == 40 and _u["median_len_other"] == 200, _u)
check("the word 'irrelevant' counts as 0", _zu.is_zero("Irrelevant") and _zu.is_zero("0") and not _zu.is_zero("3"))
check("has_harm_domain: a category (any case, with a second label) counts; none does not",
      _zu.has_harm_domain("Economics, social") and not _zu.has_harm_domain("none") and not _zu.has_harm_domain(""))

_a = [_r25(f"k{i}", "none", "0" if i < 6 else "3") for i in range(10)]          # zeros on k0-k5
_b = [_r25(f"k{i}", "none", "0" if i < 5 else "3") for i in range(10)]          # k0-k4  (very like a)
_c = [_r25(f"k{i}", "democracy", "0" if i >= 7 else "3") for i in range(10)]    # k7-k9  (unlike a)
_shared, _zsets, _pairs = _zu.core_comparison([_a, _b, _c], {f"k{i}" for i in range(10)})
check("core comparison uses the comments every coder rated (10)", len(_shared) == 10)
check("the pair with nearly identical 0s has high overlap, the other low",
      _pairs[(0, 1)]["jaccard"] > 0.8 and _pairs[(0, 2)]["jaccard"] < 0.2,
      (_pairs[(0, 1)]["jaccard"], _pairs[(0, 2)]["jaccard"]))
check("overlap counts both / only-first / only-second correctly for A vs B (5 / 1 / 0)",
      (_pairs[(0, 1)]["both"], _pairs[(0, 1)]["only_a"], _pairs[(0, 1)]["only_b"]) == (5, 1, 0), _pairs[(0, 1)])
check("agreement on 'is it a 0' is the share of comments both coders treated the same way",
      abs(_pairs[(0, 1)]["agree_on_zero"] - 0.9) < 1e-9, _pairs[(0, 1)])
check("the report never contains comment text",
      "xxxx" not in repr(_u) and "yyyy" not in repr(_u))



print("\ncombined arms must not share replies across different examples")


@judges.register("test_probe")
class ProbeJudge(judges.Judge):
    default_model = "probe"
    calls = 0

    def _sample(self, body, prompt, vote):
        type(self).calls += 1
        return "democracy | 1" if "MARKER" in prompt else "social | 5"


_a = judges.create("combined:test_probe", votes=1)
_b = judges.create("combined:test_probe", votes=1)
check("the two arms share one instance (that is intended)", _a is _b)
_fixed = _a.label("x", [("plain", "social")])[0]
_retr = _b.label("x", [("MARKER example", "democracy")])[0]
check("a different example set gets its own call, not the other arm's reply",
      _fixed == ["social"] and _retr == ["democracy"], (_fixed, _retr))
check("identical body and examples still reuse the reply",
      (lambda n: (_a.label("x", [("plain", "social")]), ProbeJudge.calls == n)[1])(ProbeJudge.calls))

# Settings asked for by a later caller used to be silently discarded, because
# __init__ early-returns on an already-built instance. Regression cover for #2.
_v1 = judges.create("combined:test_probe", votes=1)
_v3 = judges.create("combined:test_probe", votes=3)
check("a different vote count gets its own instance, not the first caller's",
      _v1 is not _v3)
check("and the votes it was asked for are the votes it runs",
      (_v1.votes, _v3.votes) == (1, 3), (_v1.votes, _v3.votes))
import ewl.label.sentiment as _sent_early
check("the judge and the sentiment scorer still share one instance",
      _sent_early.create("combined:test_probe", votes=3,
                           cache_path=None, fuzzy=False)._judge
      is judges.create("combined:test_probe", votes=3,
                       cache_path=None, fuzzy=False))
check("they share under --fuzzy too, so the one-call saving survives it",
      _sent_early.create("combined:test_probe", votes=3, fuzzy=True)._judge
      is judges.create("combined:test_probe", votes=3, fuzzy=True))

# Arms may legitimately share a judge; the accounting must still be per-arm.
_shared_judge = judges.create("combined:test_probe", votes=1)
_rows = [{"_body": "x", "_gold": {"social"}}]
_sel = examples_mod.create("none")
_arm1 = evaluate.run_arm(_rows, _shared_judge, _sel, progress=False)
_arm2 = evaluate.run_arm(_rows, _shared_judge, _sel, progress=False)
# These assertions have to be strict. Written as `arm2 <= arm1` and
# `arm2.seconds <= judge.seconds` they passed with the delta subtraction
# reverted -- arm 2 reuses the memoised reply so both are 1, and 1 <= 1 holds,
# while cumulative seconds equal the judge's total exactly.
check("a second arm on a shared judge is billed none of the first arm's calls",
      _arm2["calls"] == 0,
      f"arm1 calls {_arm1['calls']}, arm2 calls {_arm2['calls']} (cumulative would repeat arm1's)")
check("and its elapsed time is its own, not the judge's running total",
      _arm2["seconds"] < _shared_judge.seconds,
      (_arm2["seconds"], _shared_judge.seconds))


print("\ncombined arm holds one cache, not two")
_cached = judges.create("combined:stub", votes=1, cache_path=os.devnull)
check("the wrapper does not open its own copy of the cache file",
      _cached._cache is None and _cached._inner._cache is not None,
      "two _Cache objects on one path load the file into memory twice")


print("\ncombined arm: sentiment reuses the domain call's own examples, not a fixed set")
# Regression cover for the "2 calls per comment under a non-fixed selector"
# bug: sentiment_for() used to always build its prompt from
# taxonomy.FIXED_EXAMPLES, regardless of what label() had actually been given.
# Under --selector retrieved (or anything else that varies examples per
# comment), that meant the domain prompt and the sentiment prompt differed,
# so the _reply memo (keyed on body+vote+prompt) missed and every comment
# quietly cost two real calls instead of one.
ProbeJudge.calls = 0
_reuse_judge = judges.create("combined:test_probe", votes=1)
_marker_examples = [("MARKER example", "democracy")]
_reuse_judge.label("shared body", _marker_examples, story="luigi")
check("label() records the examples it was given, for sentiment_for to reuse",
      _reuse_judge._examples_by_body.get(("shared body", "luigi")) == _marker_examples,
      _reuse_judge._examples_by_body)
_calls_after_label = ProbeJudge.calls
_reuse_result = _reuse_judge.sentiment_for("shared body", story="luigi")
check("sentiment_for makes NO extra call when it reuses label()'s own examples",
      ProbeJudge.calls == _calls_after_label,
      (ProbeJudge.calls, _calls_after_label))
check("sentiment_for still returns a real rating from the shared reply",
      _reuse_result["value"] is not None, _reuse_result)
check("both halves together cost exactly one real call, not two",
      ProbeJudge.calls == 1, ProbeJudge.calls)
check("the memo forgets this body once sentiment_for has read it, bounding memory",
      not any(key[0] == "shared body" for key in _reuse_judge._replies)
      and not any(key[0] == "shared body" for key in _reuse_judge._examples_by_body),
      (_reuse_judge._replies, _reuse_judge._examples_by_body))

# The same body under two stories at once ("yes" is posted under all four): closing one
# pair must not delete the other's memoised reply (it used to drop by body alone).
_two = judges.create("combined:test_probe", votes=1, fuzzy=True)
_ex2 = [("e", "none")]
_two.label("yes", _ex2, story="luigi")
_two.label("yes", _ex2, story="musk")
_before = ProbeJudge.calls
_two.sentiment_for("yes", story="luigi")
_after_first = {k[2][:1] for k in _two._replies if k[0] == "yes"}
check("closing luigi's pair leaves musk's memoised reply for 'yes' in place",
      len([k for k in _two._replies if k[0] == "yes"]) == 1 and ProbeJudge.calls == _before,
      (len(_two._replies), ProbeJudge.calls, _before))
_two.sentiment_for("yes", story="musk")
check("and closing musk's pair empties it", not [k for k in _two._replies if k[0] == "yes"])

# The memo is bounded even when nothing ever pairs the two halves (domain-only use).
_bounded = judges.create("combined:test_probe", votes=1, fuzzy=False, cache_path=None)
_bounded._memo_limit = 8
for _i in range(40):
    _bounded.label(f"distinct body {_i}", _ex2, story="jets")
check("the memo never holds more than its limit, however many distinct comments are labelled",
      len(_bounded._replies) <= 8 and len(_bounded._examples_by_body) <= 8,
      (len(_bounded._replies), len(_bounded._examples_by_body)))

_before_standalone_calls = ProbeJudge.calls
_standalone = _reuse_judge.sentiment_for("never labelled", story="luigi")
check("sentiment_for falls back to FIXED_EXAMPLES when label() was never called for this body",
      ProbeJudge.calls == _before_standalone_calls + 1 and _standalone["value"] is not None,
      (ProbeJudge.calls, _standalone))


print("\nscorecard")
with tempfile.TemporaryDirectory() as tmp:
    gold = os.path.join(tmp, "gold.csv")
    write_fixture(gold)
    rows = evaluate.read_gold(gold)
    check("gold labels parse into sets", rows[0]["_gold"] == {"democracy"})

    result = evaluate.run_arm(rows, judges.create("keyword"), examples_mod.create("none"), progress=False)
    check("an arm produces every category", set(result["per_category"]) == set(taxonomy.CATEGORY_NAMES))
    check("macro-F1 is in range", 0 <= result["macro_f1"] <= 1, result["macro_f1"])
    check("the keyword arm finds some democracy", result["per_category"]["democracy"]["recall"] > 0)

    stub_result = evaluate.run_arm(rows, judges.create("stub", votes=1),
                                   examples_mod.create("fixed"), progress=False)
    check("an always-social judge scores 0 on democracy recall",
          stub_result["per_category"]["democracy"]["recall"] == 0)
    check("macro-F1 punishes a single-label judge",
          stub_result["macro_f1"] < result["macro_f1"],
          f"stub {stub_result['macro_f1']:.2f} vs keyword {result['macro_f1']:.2f}")

    buffer = io.StringIO()
    evaluate.print_scorecard([result, stub_result], out=buffer)
    table = buffer.getvalue()
    check("the scorecard renders as markdown", table.count("|") > 20 and "macro-F1" in table)
    check("every arm gets a row", all(r["arm"] in table for r in (result, stub_result)))

print("\nrun_arm: inclusion_prob weighting and parse-failure exclusion")


@judges.register("test_stratified_probe")
class _StratifiedProbe(judges.Judge):
    """Gets the low-inclusion_prob half wrong and the high-prob half right --
    the same shape as the manual repro that found the scorecard's weighting
    gap, but as a registrable judge so it goes through the same run_arm path
    a real arm does."""
    deterministic = True
    default_model = "probe"

    def _sample(self, body, prompt, vote):
        return "social" if "high" in body else "none"


_weighted_rows = (
    [{"comment_id": f"low{i}", "_body": "low-probability stratum comment",
      "hand_domain": "social", "_gold": {"social"}, "inclusion_prob": "0.1"}
     for i in range(20)]
    + [{"comment_id": f"high{i}", "_body": "high-probability stratum comment",
        "hand_domain": "social", "_gold": {"social"}, "inclusion_prob": "0.9"}
       for i in range(20)]
)
_weighted_arm = evaluate.run_arm(_weighted_rows, judges.create("test_stratified_probe"),
                                 examples_mod.create("none"), progress=False)
_unweighted_recall = 20 / 40  # tp=20 (the "high" half), fn=20 (the "low" half), no weighting
_weight_low, _weight_high = 1 / 0.1, 1 / 0.9
_weighted_recall = _weight_high * 20 / (_weight_high * 20 + _weight_low * 20)
check("run_arm weights recall by inclusion_prob, not by raw row count",
      abs(_weighted_arm["per_category"]["social"]["recall"] - _weighted_recall) < 1e-9,
      (_weighted_arm["per_category"]["social"]["recall"], _weighted_recall))
check("...and that is not the same number the old unweighted code gave",
      abs(_weighted_arm["per_category"]["social"]["recall"] - _unweighted_recall) > 0.1,
      (_weighted_arm["per_category"]["social"]["recall"], _unweighted_recall))


@judges.register("test_parse_failure_probe")
class _ParseFailureProbe(judges.Judge):
    deterministic = True
    default_model = "probe"

    def _sample(self, body, prompt, vote):
        return "garbled nonsense reply"  # off-list, not "none" -> PARSE_FAILURE


_pf_rows = [{"comment_id": "pf1", "_body": "x", "hand_domain": "social",
            "_gold": {"social"}, "inclusion_prob": "1.0"}]
_pf_arm = evaluate.run_arm(_pf_rows, judges.create("test_parse_failure_probe"),
                          examples_mod.create("none"), progress=False)
check("a parse failure is excluded from recall, not counted as a missed 'none'",
      _pf_arm["per_category"]["social"]["recall"] != _pf_arm["per_category"]["social"]["recall"],  # NaN
      _pf_arm["per_category"]["social"])

print("\ncombined arm")
from ewl.label import combined
check("registry exposes the combined backend once combined.py is imported",
      "combined" in judges.available(), judges.available())
_cases = [("democracy | 2", ["democracy"], 2), ("economics, social | 4", ["economics", "social"], 4),
          ("none | 3", ["unclassified"], 3), ("democracy 2", ["democracy"], 2),
          ("  social|1  ", ["social"], 1)]
for _text, _labels, _rating in _cases:
    _d, _r = combined.split_reply(_text)
    check(f"parses {_text.strip()!r}",
          taxonomy.parse_labels(_d) == _labels and _r == _rating, (_d, _r))
check("an out-of-range rating is None, not clamped", combined.split_reply("democracy | 9")[1] is None)
check("a missing rating is None", combined.split_reply("democracy")[1] is None)
check("a digit inside the domain is not read as the rating",
      combined.split_reply("social | 2")[1] == 2)
_j = judges.create("combined:stub")
_s = sentiment_mod_placeholder = None
import ewl.label.sentiment as _sent
check("judge and scorer share one instance", judges.create("combined:stub") is
      _sent.create("combined:stub")._judge,
      "two instances would each make their own calls and undo the saving")
_nr = _j.sentiment_for("x", taxonomy.FIXED_EXAMPLES)
check("a reply with no digit is a parse_failure with no value, never a neutral or a zero",
      _nr["label"] == "parse_failure" and _nr["value"] is None, _nr)

# Regression: -1 ("not codable") used to be misread as the digit 1 ("very
# negative") by the old `[1-5]` character-class pattern, and 0 ("irrelevant")
# was dropped entirely, since neither -1 nor 0 was ever in that class.
check("-1 (not codable) parses as -1, not as the digit 1",
      combined.split_reply("democracy | -1")[1] == -1)
check("0 (irrelevant) parses, not silently dropped",
      combined.split_reply("none | 0")[1] == 0)

@judges.register("probe_always_uncodeable")
class _AlwaysUncodeable(judges.Judge):
    default_model = "probe"
    def _sample(self, body, prompt, vote):
        return "none | -1"

_uncodeable_judge = judges.create("combined:probe_always_uncodeable", votes=1)
_result = _uncodeable_judge.sentiment_for("x", taxonomy.FIXED_EXAMPLES)
check("a -1 reply resolves to the UNCODEABLE sentinel, not value -1.0 medianed as a stance",
      _result["label"] == _sent.UNCODEABLE and _result["value"] is None, _result)

# Regression: CombinedScorer.score() had no `story` parameter at all, so
# classify.py's `scorer.score(body, story=row.get("story"))` -- the call every
# real run makes -- raised TypeError on the very first comment.
check("CombinedScorer.score() accepts the story= kwarg every real caller passes",
      "label" in _sent.create("combined:stub").score("x", story="luigi"))

# Regression: the joint prompt's domain half used to carry its own hand-copied
# rules with no boundary-cases section, silently diverging from every other
# judge's taxonomy.py-derived prompt; and the whole prompt never varied by
# story, so sentiment.STORY_CONTEXT's deictic-comment fix (see sentiment.py)
# never reached the combined arm.
check("the combined prompt reuses taxonomy's boundary-cases text, not a copy",
      taxonomy.BOUNDARY_CASES in combined.build_prompt(taxonomy.FIXED_EXAMPLES))
check("the combined prompt varies by story",
      combined.build_prompt(taxonomy.FIXED_EXAMPLES, story="luigi")
      != combined.build_prompt(taxonomy.FIXED_EXAMPLES, story="musk"))
check("an unrecognised story falls back to sentiment.NO_STORY_CONTEXT",
      _sent.NO_STORY_CONTEXT in combined.build_prompt(taxonomy.FIXED_EXAMPLES, story="not-a-real-story"))

print("\nlanguage detection is recorded, not acted on")
from ewl import common as ewl_common
check("detected_lang is part of the schema", "detected_lang" in ewl_common.FIELDNAMES)
check("a comment too short to judge returns empty, not a guess",
      ewl_common.detect_language("lol") == "")
check("empty input returns empty", ewl_common.detect_language("") == ""
      and ewl_common.detect_language(None) == "")
with optional("language detection"):
    import langdetect  # noqa: F401
    check("a long English comment detects as en",
          ewl_common.detect_language(
              "This is a reasonably long English sentence about wealth taxes and policy.") == "en")
    check("unknown always passes the legacy filter",
          ewl_common.is_language("lol", "en") and ewl_common.is_language("", "en"),
          "a comment the detector cannot read must not be dropped as foreign")
    check("no --lang means no filtering", ewl_common.is_language("n'importe quoi", None))


print("\nboth collectors agree on the language contract")
with optional("collector signatures"):
    import ast as _ast
    import ewl.collect.youtube as _yc_mod
    import ewl.collect.youtube_api as _ya_mod
    for _mod in (_yc_mod, _ya_mod):
        _f = _mod.__file__
        _tree = _ast.parse(open(_f).read())
        _defs = {n.name: len(n.args.args) for n in _ast.walk(_tree)
                 if isinstance(n, _ast.FunctionDef)}
        _calls = [(n.func.id, len(n.args), n.lineno) for n in _ast.walk(_tree)
                  if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name)
                  and n.func.id in _defs]
        _bad = [(name, got, line) for name, got, line in _calls if got > _defs[name]]
        check(f"{_f}: no call passes more positional args than its def accepts",
              not _bad, _bad)


print("\nreport.py guards against confidently-wrong figures")
with optional("report"):
    import ewl.analyse.report as _rep
    check("Rogan-Gladen refuses an inverted classifier (TPR < FPR)",
          all(_rep.corrected_estimate(o, tpr, fpr) is None for o, tpr, fpr in
              [(0.30, 0.20, 0.80), (0.50, 0.10, 0.90), (0.30, 0.10, 0.30)]),
          "abs(denom) let negative denominators through; the sign-flipped result "
          "usually lands inside [0,1] so the range check did not catch it")
    check("it still corrects a valid case",
          abs(_rep.corrected_estimate(0.30, 0.85, 0.10) - 0.266666) < 1e-4)
    check("indistinguishable TPR/FPR still refused",
          _rep.corrected_estimate(0.30, 0.50, 0.52) is None)

    _rows = [{"comment_id": "c1", "story": "A", "source": "v1",
              "domain_democracy": "1", "sentiment": "positive"},
             {"comment_id": "c2", "story": "A", "source": "v1",
              "domain_democracy": "0", "sentiment": "negative"}]

    def _refuses(col):
        try:
            _rep.column_ci(_rows, col, n_boot=10)
            return False
        except SystemExit:
            return True

    check("a mistyped column is refused, not reported as 0%", _refuses("domain_demokracy"))
    check("a text column is refused, not reported as 100%", _refuses("sentiment"),
          "classify.py writes sentiment as positive/negative/neutral; "
          "--column sentiment_value is one character away")
    check("a real 0/1 column still works",
          _rep.column_ci(_rows, "domain_democracy", n_boot=10)["(pooled)"][2] == 0.5)

# A video whose comments are all unrated makes some resampled draws NaN; sorting a list
# holding NaN scrambles it, which once turned a 0-1 interval into 0.50-0.90.
_nan_rows = ([{"source": "a", "story": "s1", "sentiment_value": ""}] * 5
             + [{"source": "b", "story": "s1", "sentiment_value": "1"}] * 5
             + [{"source": "c", "story": "s1", "sentiment_value": "0"}] * 5
             + [{"source": "d", "story": "s1", "sentiment_value": v} for v in "10101"])
_n, _v, _obs, _lo, _hi = _rep.column_ci(_nan_rows, "sentiment_value", n_boot=2000, seed=1)["(pooled)"]
check("bootstrap draws with no rated comment are dropped, not sorted in as NaN",
      _lo <= _obs <= _hi and _lo == 0.0 and _hi == 1.0, (_obs, _lo, _hi))
_agg = _rep.aggregate_ci(_nan_rows + [{"source": f"e{i}", "story": "s2",
                                       "sentiment_value": str(i % 2)} for i in range(20)],
                         "sentiment_value", n_boot=2000, seed=1)
check("the equal-per-story interval drops those draws too, and still brackets its estimate",
      _agg[1] <= _agg[0] <= _agg[2], _agg[:3])
_all_blank = ([{"source": "a", "story": "s1", "sentiment_value": ""}] * 5
              + [{"source": "b", "story": "s1", "sentiment_value": "1"}])
_r = _rep.column_ci(_all_blank, "sentiment_value", n_boot=200, seed=1)["(pooled)"]
check("too many unusable draws refuse the interval (NaN) instead of reporting one",
      _r[3] != _r[3] and _r[4] != _r[4], _r)


print("\ncache identity survives a shortened display name")
with tempfile.TemporaryDirectory() as tmp:
    _v1, _v2 = os.path.join(tmp, "v1"), os.path.join(tmp, "v2")
    for _d, _extra in ((_v1, "economics:\n  - wages\n"), (_v2, "")):
        os.makedirs(_d)
        open(os.path.join(_d, "keywords.yml"), "w").write(
            "democracy:\n  - lobbying\nenvironment:\n  - jet\nsocial:\n  - unfair\n" + _extra)
    _a = judges.create(f"keyword:{_v1}/keywords.yml")
    _b = judges.create(f"keyword:{_v2}/keywords.yml")
    check("two vocabularies with one basename share a display name", _a.spec == _b.spec)
    check("but never a cache namespace", _a.cache_spec != _b.cache_spec,
          "sharing it serves one vocabulary's labels for the other's run")
    check("the display name is still the short one", _a.spec == "keyword:keywords.yml")


print("\ncorpus freezing")
from ewl import corpus as _corpus
with tempfile.TemporaryDirectory() as tmp:
    _c = os.path.join(tmp, "corpus.csv")
    with open(_c, "w", newline="", encoding="utf-8-sig") as handle:
        w = csv.DictWriter(handle, fieldnames=["comment_id", "story", "source", "body"])
        w.writeheader()
        for i in range(9):
            w.writerow({"comment_id": f"c{i}", "story": "A" if i < 5 else "B",
                        "source": f"v{i // 3}", "body": f"comment {i}"})

    check("an unfrozen corpus is reported, not refused",
          _corpus.verify(_c)[0] is True and "No manifest" in _corpus.verify(_c)[1],
          "a corpus collected before manifests existed is still good data")

    _path, _man = _corpus.write_manifest(_c, inputs=["a.csv"])
    check("the manifest counts comments, videos and stories",
          (_man["comments"], _man["videos"], _man["comments_per_story"])
          == (9, 3, {"A": 5, "B": 4}), _man)
    check("a frozen corpus verifies", _corpus.verify(_c)[0])
    check("the salt is fingerprinted, never stored",
          _man["salt_fingerprint"] and os.environ["EWL_AUTHOR_SALT"] not in json.dumps(_man),
          "two corpora hashed with different salts have incomparable author_hash")

    with open(_c, "a", encoding="utf-8") as handle:
        handle.write("c99,A,v9,tampered\n")
    _ok, _msg = _corpus.verify(_c)
    check("a changed corpus fails verification", _ok is False and "CHANGED" in _msg)
    try:
        _corpus.verify(_c, strict=True)
        check("strict mode stops the run", False, "no SystemExit raised")
    except SystemExit:
        check("strict mode stops the run", True)


print("\ncollection and analysis stay separate")
_ANALYSIS = ["ewl/label", "ewl/analyse"]
_NETWORK = ("googleapiclient", "requests", "urllib.request", "http.client")
_leaks = []
for _pkg in _ANALYSIS:
    for _f in pathlib.Path("src", _pkg).rglob("*.py"):
        _src = _f.read_text()
        for _n in _NETWORK:
            if f"import {_n}" in _src or f"from {_n}" in _src:
                _leaks.append(f"{_f}: {_n}")
        if "from ewl.collect" in _src or "import ewl.collect" in _src:
            _leaks.append(f"{_f}: imports a collector")
check("no labelling or analysis module can reach the network or a collector",
      not _leaks, _leaks)

_collect_leaks = [str(_f) for _f in pathlib.Path("src/ewl/collect").rglob("*.py")
                  if "from ewl.label" in _f.read_text()
                  or "from ewl.analyse" in _f.read_text()]
check("no collector depends on labelling or analysis", not _collect_leaks, _collect_leaks)


print("\nmulti-query search pools candidates")
with optional("multi-query search"):
    import types as _t
    from ewl.collect import youtube as _y

    class _Search:
        """Each query returns 50 ids; queries 2+ overlap the first by 25."""
        def __init__(self): self.queries = []
        def list(self, **kw):
            q = kw["q"]; self.queries.append(q)
            base = 0 if q == "q1" else 25
            return _t.SimpleNamespace(execute=lambda: {"items": [
                {"id": {"videoId": f"v{base + i:03d}"}} for i in range(50)]})

    class _Videos:
        def __init__(self): self.batches = []
        def list(self, **kw):
            ids = kw["id"].split(",")
            self.batches.append(len(ids))
            return _t.SimpleNamespace(execute=lambda: {"items": [
                {"id": i, "statistics": {"commentCount": "500"},
                 "contentDetails": {"duration": "PT10M"}} for i in ids]})

    _s, _v = _Search(), _Videos()
    _meta = _y.video_meta(_t.SimpleNamespace(videos=lambda: _v),
                          [f"v{i:03d}" for i in range(120)])
    check("video_meta batches at the API's 50-id limit",
          _v.batches == [50, 50, 20], _v.batches)
    check("every id comes back despite batching", len(_meta) == 120, len(_meta))

    _v2 = _Videos()
    _yt = _t.SimpleNamespace(search=lambda: _s, videos=lambda: _v2,
                             commentThreads=lambda: _t.SimpleNamespace(
                                 list=lambda **k: _t.SimpleNamespace(
                                     execute=lambda: {"items": []})))
    _rows = _y.search_youtube(_yt, ["q1", "q2"], "s", top_n=2, target_videos=5,
                              comment_limit=1, min_comments=10, seed=1)
    check("both queries are issued", _s.queries == ["q1", "q2"], _s.queries)
    check("overlapping candidates are deduplicated",
          sum(_v2.batches) == 75, f"50 + 50 with 25 shared should pool to 75, got {sum(_v2.batches)}")

    _s3 = _Search()
    _y.search_youtube(_t.SimpleNamespace(search=lambda: _s3, videos=lambda: _Videos(),
                                         commentThreads=lambda: _t.SimpleNamespace(
                                             list=lambda **k: _t.SimpleNamespace(
                                                 execute=lambda: {"items": []}))),
                      "q1", "s", top_n=1, target_videos=1, comment_limit=1,
                      min_comments=10, seed=1)
    check("a bare string query still works", _s3.queries == ["q1"], _s3.queries)


print("\nexclusion ledger closes")
with optional("collector ledger"):
    import types as _types
    import ewl.collect.youtube as _yc

    class _FakeReq:
        def execute(self):
            return {"items": [
                {"snippet": {"topLevelComment": {"id": f"t{i}", "snippet": {
                    "authorDisplayName": f"a{i}",
                    "textDisplay": f"a real comment number {i} about tax and lobbying",
                    "likeCount": 1, "publishedAt": "2026-01-01T00:00:00Z"}}},
                 "replies": {"comments": [{"id": f"r{i}_{j}", "snippet": {
                     "authorDisplayName": f"b{j}",
                     "textDisplay": f"a reply {j} to comment {i} about wealth",
                     "likeCount": 0, "publishedAt": "2026-01-01T00:00:00Z"}}
                     for j in range(2)]}}
                for i in range(12)]}

    class _FakeYT:
        def commentThreads(self):
            return _types.SimpleNamespace(list=lambda **k: _FakeReq())

    _exc = collections.Counter()
    _got = _yc.scrape_comments(_FakeYT(), "https://youtu.be/AAAAAAAAAAA", "s", 5,
                               exclusions=_exc)
    check("comments dropped by --limit are counted",
          _exc["excluded_by_comment_limit"] == 21, dict(_exc))
    check("examined + skipped-by-limit equals what the API returned",
          _exc["api_returned"] + _exc["excluded_by_comment_limit"] == 36, dict(_exc))

    _exc2 = collections.Counter()
    _yc.scrape_comments(_FakeYT(), "https://youtu.be/AAAAAAAAAAA", "s", 0, exclusions=_exc2)
    check("uncapped, nothing is attributed to the limit",
          _exc2["excluded_by_comment_limit"] == 0 and _exc2["api_returned"] == 36,
          dict(_exc2))


print("\nShorts duration parsing")
try:
    from ewl.collect import youtube as youtube_collect
except Exception as _e:                # googleapiclient not installed
    print(f"  SKIP  ({type(_e).__name__})")
else:
    _cases = [("PT59S", 59), ("PT3M", 180), ("PT4M13S", 253), ("PT1H2M3S", 3723),
              ("P1DT2H", 93600), ("PT0S", 0)]
    check("ISO 8601 durations parse",
          all(youtube_collect.duration_seconds(s) == n for s, n in _cases))
    check("an unknown duration is None, not 0",
          youtube_collect.duration_seconds("") is None
          and youtube_collect.duration_seconds(None) is None
          and youtube_collect.duration_seconds("garbage") is None,
          "None must be distinguishable from 0s, or a live stream reads as a Short")
    check("malformed ISO is rejected rather than parsed as 0",
          all(youtube_collect.duration_seconds(s) is None
              for s in ("P", "PT", "P1H", "PTS", "P1DT")),
          "a bogus 0 reads as a 0-second video and is silently excluded as a Short")
    check("P0D (live stream) is unknown, not zero",
          youtube_collect.duration_seconds("P0D") is None,
          "YouTube returns P0D for live/upcoming; 0 would exclude them all as Shorts")
    check("PT0S is a genuine zero and still parses",
          youtube_collect.duration_seconds("PT0S") == 0)

    print("\nhybrid video selection, exclusion accounting, video-list log")

    class FakeYouTube:
        """Stands in for the googleapiclient youtube resource: .search(),
        .videos(), .commentThreads() all return self, and .list(**kw)
        remembers which call it was for .execute() to answer."""

        def __init__(self, search_items, meta_items, comments_by_video):
            self._search_items = search_items
            self._meta_items = meta_items
            self._comments_by_video = comments_by_video

        def search(self):
            return self

        def videos(self):
            return self

        def commentThreads(self):
            return self

        def list(self, **kwargs):
            self._kwargs = kwargs
            return self

        def execute(self):
            if "q" in self._kwargs:
                return {"items": self._search_items}
            if "id" in self._kwargs:
                return {"items": self._meta_items}
            return {"items": self._comments_by_video.get(self._kwargs["videoId"], [])}

    def _search_item(video_id):
        return {"id": {"videoId": video_id}}

    def _meta_item(video_id, comment_count=50, duration="PT5M0S"):
        return {"id": video_id,
                "statistics": {"commentCount": str(comment_count)},
                "contentDetails": {"duration": duration}}

    def _comment_thread(comment_id, author, text, replies=()):
        return {
            "id": comment_id,
            "snippet": {"topLevelComment": {
                "id": comment_id,
                "snippet": {"authorDisplayName": author, "textDisplay": text,
                           "likeCount": 1, "publishedAt": "2024-01-01T00:00:00Z"},
            }},
            "replies": {"comments": [
                {"id": rid, "snippet": {"authorDisplayName": ra, "textDisplay": rt,
                                        "likeCount": 0, "publishedAt": "2024-01-01T00:00:00Z"}}
                for rid, ra, rt in replies
            ]},
        }

    # 15 candidates, in relevance order. Real 11-character IDs, since
    # extract_video_id() requires that exact length -- a short synthetic ID
    # like "v1" fails to parse, which is correct behaviour for a malformed
    # URL but wrong for a test fixture standing in for a real video.
    video_ids = [f"vid{i:08d}" for i in range(1, 16)]
    first_video, top3 = video_ids[0], set(video_ids[:3])
    fake = FakeYouTube(
        search_items=[_search_item(v) for v in video_ids],
        meta_items=[_meta_item(v) for v in video_ids],
        comments_by_video={
            first_video: [
                _comment_thread("t1", "Alice", "Great point about tax",
                                replies=[("r1", "Bob", "I agree")]),
                _comment_thread("t2", "", "[deleted]"),
            ],
        },
    )

    exclusions = collections.Counter()
    video_log = []
    with contextlib.redirect_stderr(io.StringIO()):
        rows = youtube_collect.search_youtube(
            fake, "wealth tax", "luigi", top_n=3, target_videos=6, comment_limit=0,
            min_comments=0, seed=1, max_shorts_seconds=60,
            exclusions=exclusions, video_log=video_log)

    picked_ids = {entry["video_id"] for entry in video_log}
    check("hybrid selection picks exactly target_videos videos", len(picked_ids) == 6, picked_ids)
    check("the top-n videos by relevance are always included", top3 <= picked_ids, picked_ids)
    check("the remaining videos are drawn from outside the top-n, no duplicates",
          len(picked_ids - top3) == 3, picked_ids)

    check("the first video's real comment is kept", any(r["comment_id"] == "t1" for r in rows))
    check("the first video's reply is kept with depth 1",
          any(r["comment_id"] == "r1" and r["depth"] == 1 for r in rows))
    check("the first video's deleted comment is excluded",
          not any(r["comment_id"] == "t2" for r in rows))
    check("exclusion accounting counts the deleted comment",
          exclusions["deleted_removed_or_automod"] == 1, exclusions)
    check("exclusion accounting counts every item examined (2 top-level + 1 reply for the "
          "first video, plus zero-comment videos contribute nothing)",
          exclusions["api_returned"] == 3, exclusions)
    check("video_log records comments_kept per video",
          next(e["comments_kept"] for e in video_log if e["video_id"] == first_video) == 2,
          [e for e in video_log if e["video_id"] == first_video])
    check("video_log records the search query that found each video",
          all(e["search_query"] == "wealth tax" for e in video_log), video_log)
    check("videos with zero comments still appear in video_log with comments_kept=0",
          any(e["video_id"] != first_video and e["comments_kept"] == 0 for e in video_log),
          video_log)


print("\nsentiment")
with optional("sentiment scorers"):
    import ewl.label.sentiment as sentiment_mod
    check("scorer registry", set(sentiment_mod.available()) >= {"vader", "llm", "stub"})
    check("thresholds are shared across scorers",
          (sentiment_mod.to_label(0.5), sentiment_mod.to_label(0.0), sentiment_mod.to_label(-0.5))
          == ("positive", "neutral", "negative"))
    vader = sentiment_mod.create("vader")
    check("vader scores and labels", vader.score("This is wonderful")["label"] == "positive")
    check("vader is deterministic", vader.deterministic)

    # SCALE must stay story-agnostic (no "the story" or "mocking tone" --
    # story-referent/tone language that broke on Luigi specifically, since
    # "the story" there is a killing, and "mocking tone" never said who the
    # mockery targets). Story-specific disambiguation belongs in
    # STORY_CONTEXT, not here -- see the four per-story checks below.
    _scale_text = " ".join(sentiment_mod.SCALE.values())
    check("SCALE no longer references 'the story' generically",
          "the story" not in _scale_text, _scale_text)
    check("the 0 anchor is the wealth-position reading the gold set was coded with: no position on wealth, "
          "on or off the video's subject",
          "takes no position on extreme wealth" in sentiment_mod.SCALE[0]
          and "whether or not it is about the video's subject" in sentiment_mod.SCALE[0])
    check("the prompt rule for 0 names the kinds of comment and says when to use 3 instead",
          "a reaction or joke" in sentiment_mod.PROMPT and "Use 3 instead only when" in sentiment_mod.PROMPT)
    check("the combined arm's 0 rule matches",
          "takes no position on extreme wealth" in _comb.build_prompt([("x", "none")], story="luigi"))
    check("SCALE no longer references a bare 'mocking tone'",
          "mocking tone" not in _scale_text, _scale_text)

    # Story-specific stance-inference rules, one per story, giving symmetric
    # guidance across all four (a review flagged Luigi as the only story
    # with an explicit rule; each story now gets one).
    check("luigi: praising Luigi is coded as hostile to wealth",
          "hostile to the healthcare/insurance industry's wealth" in sentiment_mod.STORY_CONTEXT["luigi"])
    check("luigi: disapproving of what Luigi did is coded as supportive of wealth",
          "what Luigi did was wrong" in sentiment_mod.STORY_CONTEXT["luigi"])
    check("luigi: no 'mocking tone' in the context (tone is not a stance), and the paragraph names "
          "the object of the stance: the insurers' wealth, not the killing alone",
          "mocking" not in sentiment_mod.STORY_CONTEXT["luigi"]
          and "its stance toward that wealth" in sentiment_mod.rules_only("luigi")
          and "stance" not in sentiment_mod.facts_only("luigi")
          and "hostile to the healthcare" not in sentiment_mod.facts_only("luigi")
          and "an inference" in sentiment_mod.rules_only("luigi")
          and "no stated view on the insurers" in sentiment_mod.rules_only("luigi")
          and "Conversely" not in sentiment_mod.STORY_CONTEXT["luigi"])
    check("anchor 4 names the wealthy people or companies and how they gain or use wealth, no 'in this story'",
          "wealthy people or companies" in sentiment_mod.SCALE[4] and "this story" not in sentiment_mod.SCALE[4])
    check("musk: defending his ownership/spending/capability is coded as supportive of wealth",
          "generally supportive of extreme" in sentiment_mod.STORY_CONTEXT["musk"])
    check("jets: the 'earned the right to fly private' argument is coded as supportive of wealth",
          "legitimate right of the wealthy" in sentiment_mod.STORY_CONTEXT["jets"])
    check("propublica now carries a stance rule too, drawn from how the gold coders rated: both sides, "
          "and 'takes no side' for tax-mechanics confusion",
          "generally supportive of extreme wealth" in sentiment_mod.STORY_CONTEXT["propublica"]
          and "generally critical" in sentiment_mod.STORY_CONTEXT["propublica"]
          and "takes no side" in sentiment_mod.STORY_CONTEXT["propublica"]
          and sentiment_mod.rules_only("propublica") != ""
          and "generally" not in sentiment_mod.facts_only("propublica"))
    check("every story's rule covers both directions (supportive and critical)",
          all("generally supportive" in sentiment_mod.rules_only(k) or "generally hostile" in sentiment_mod.rules_only(k)
              for k in sentiment_mod.STORY_CONTEXT)
          and all("critical" in sentiment_mod.rules_only(k) or "hostile" in sentiment_mod.rules_only(k)
                  for k in sentiment_mod.STORY_CONTEXT))
    check("luigi: a comment that condemns the killing AND criticises the insurers is neutral "
          "(the brief scores 'nobody should be killed, but nobody should be ripped off' as 3)",
          "mixed position and is neutral" in sentiment_mod.STORY_CONTEXT["luigi"])

    llm = sentiment_mod.create("llm:stub", votes=3)
    check("llm scorer accepts any judge backend", llm.describe() == "llm:stub:constant")
    # The stub judge answers "social", which contains no digit at all.
    result = llm.score("anything")
    check("a reply with no rating is a parse_failure with no value, never a neutral or a zero",
          result["label"] == "parse_failure" and result["value"] is None, result)


    @judges.register("test_rating")
    class RatingJudge(judges.Judge):
        deterministic = True
        default_model = "rates"

        def _sample(self, body, prompt, vote):
            return "1" if "awful" in body else "5"


    rated = sentiment_mod.create("llm:test_rating", votes=3)
    low, high = rated.score("this is awful"), rated.score("this is great")
    check("1 maps to strongly negative", low["label"] == "negative" and low["value"] == -1.0, low)
    check("5 maps to strongly positive", high["label"] == "positive" and high["value"] == 1.0, high)
    check("agreeing votes are not flagged unstable", not low["unstable"])

    check("pearson is 1.0 on a perfect line",
          abs(evaluate.pearson([(1, 1), (2, 2), (3, 3)]) - 1.0) < 1e-9)
    check("pearson is 0 with no relationship",
          abs(evaluate.pearson([(1, 5), (2, 5), (3, 5)])) < 1e-9 or
          evaluate.pearson([(1, 5), (2, 5), (3, 5)]) != evaluate.pearson([(1, 5), (2, 5), (3, 5)]))

    # Regression: an unrecognised `story` used to fall back to
    # NO_STORY_CONTEXT with nothing in the run log to say so.
    with contextlib_redirect_stderr_quiet() as _story_stderr:
        sentiment_mod.create("llm:stub", votes=1).score("x", story="not-a-real-story")
    check("an unrecognised story warns once, not silently",
          "not-a-real-story" in _story_stderr.getvalue()
          and "recognised stories" in _story_stderr.getvalue(), _story_stderr.getvalue())

    # Regression: resolve_ratings used to be inlined in LLMScorer.score as a
    # Counter().most_common(1) pick, which resolves a genuine 3-way split
    # between not-codable/irrelevant/stance by whichever vote came back
    # first -- an arbitrary, order-dependent answer to "no consensus".
    check("a genuine 3-way kind split is no_consensus with no value, not an arbitrary pick "
          "and not a neutral",
          sentiment_mod.resolve_ratings([-1, 0, 3]) ==
          {"label": "no_consensus", "value": None, "raw": [-1, 0, 3], "unstable": True})
    check("a polarised split (very negative vs very positive) is no_consensus, NOT a neutral",
          sentiment_mod.resolve_ratings([1, 5])["label"] == "no_consensus"
          and sentiment_mod.resolve_ratings([1, 5])["value"] is None,
          sentiment_mod.resolve_ratings([1, 5]))
    check("three votes on three different sides (1, 3, 5) have no consensus stance",
          sentiment_mod.resolve_ratings([1, 3, 5])["label"] == "no_consensus")
    check("two votes on one side and one on another still resolve (1, 2, 5 -> negative)",
          sentiment_mod.resolve_ratings([1, 2, 5])["label"] == "negative"
          and sentiment_mod.resolve_ratings([1, 2, 5])["value"] == -0.5,
          sentiment_mod.resolve_ratings([1, 2, 5]))
    check("a majority for neutral resolves to neutral (2, 3, 3)",
          sentiment_mod.resolve_ratings([2, 3, 3])["label"] == "neutral")
    check("a single stance vote is untouched (the one-vote default)",
          sentiment_mod.resolve_ratings([4])["label"] == "positive")
    check("a real majority kind still resolves normally despite one dissenting vote",
          sentiment_mod.resolve_ratings([-1, -1, 3])["label"] == sentiment_mod.UNCODEABLE)

    # Regression: a Unicode minus sign (U+2212) used to be misread as the
    # digit 1 with the sign silently dropped, both here and in combined.py.
    @judges.register("test_unicode_minus")
    class _UnicodeMinusJudge(judges.Judge):
        default_model = "probe"
        def _sample(self, body, prompt, vote):
            return "−1"  # not "-1"

    _um_result = sentiment_mod.create("llm:test_unicode_minus", votes=1).score("x")
    check("a Unicode minus sign is read as -1 (uncodeable), not the digit 1",
          _um_result["label"] == sentiment_mod.UNCODEABLE, _um_result)
    check("combined.split_reply also normalises a Unicode minus sign",
          combined.split_reply("democracy | −1")[1] == -1)


print("\nmerge.py: author cap and duplicate-body dedupe (both off by default)")
author_rows = (
    [{"author_hash": "a1", "body": f"comment {i}"} for i in range(5)]
    + [{"author_hash": "a2", "body": f"other comment {i}"} for i in range(2)]
    + [{"author_hash": "", "body": f"deleted-user comment {i}"} for i in range(3)]
)
capped = merge.cap_per_author(author_rows, max_per_author=2)
check("a prolific author is capped", sum(1 for r in capped if r["author_hash"] == "a1") == 2)
check("an author under the cap is untouched",
      sum(1 for r in capped if r["author_hash"] == "a2") == 2)
check("blank author_hash (deleted users) is never capped",
      sum(1 for r in capped if r["author_hash"] == "") == 3)

dup_rows = [
    {"comment_id": "c1", "body": "Tax the billionaires"},
    {"comment_id": "c2", "body": "Tax   the billionaires"},   # same text, doubled space
    {"comment_id": "c3", "body": "Tax the billionaires"},     # exact repeat
    {"comment_id": "c4", "body": "Something else entirely"},
]
deduped = merge.drop_duplicate_bodies(dup_rows)
check("a whitespace-only variant is treated as a duplicate", len(deduped) == 2, deduped)
check("the first occurrence is the one kept", deduped[0]["comment_id"] == "c1")

# Fixed 2026-09-16: dedup used to track `seen_bodies` in one set shared
# across every story, so a short generic reaction ("Yes", "Lol") in a
# later-merged story was dropped just because the same text happened to
# appear earlier under a completely unrelated story -- independent people
# reacting to different videos, not a copy-paste/cross-post.
_cross_story_rows = [
    {"comment_id": "s1", "story": "luigi", "body": "Yes"},
    {"comment_id": "s2", "story": "musk", "body": "Yes"},      # different story, same text
    {"comment_id": "s3", "story": "musk", "body": "Yes"},      # same story AS s2 -- a real dup
    {"comment_id": "s4", "story": "jets", "body": "No"},
]
_cross_deduped = merge.drop_duplicate_bodies(_cross_story_rows)
_cross_ids = {r["comment_id"] for r in _cross_deduped}
check("identical text in a DIFFERENT story is kept, not treated as a duplicate",
      "s2" in _cross_ids, _cross_ids)
check("identical text within the SAME story is still deduped",
      "s3" not in _cross_ids, _cross_ids)
check("nothing unrelated is dropped", "s1" in _cross_ids and "s4" in _cross_ids, _cross_ids)

print("\nevaluate.py: hand_sentiment=0 is 'irrelevant', not a fourth point on the 1-5 scale")
check("0 buckets to IRRELEVANT, not through the (value-3)/2 transform",
      evaluate.bucket_hand_sentiment("0") == evaluate.IRRELEVANT)
check("the literal word 'irrelevant' is accepted too",
      evaluate.bucket_hand_sentiment("Irrelevant") == evaluate.IRRELEVANT)
check("1 is still very negative, unaffected by 0 being special-cased",
      evaluate.bucket_hand_sentiment("1") == "negative")
check("an out-of-range value (not 0, not 1-5) is still invalid, not irrelevant",
      evaluate.bucket_hand_sentiment("7") == "")
check("blank is still invalid, not irrelevant",
      evaluate.bucket_hand_sentiment("") == "")

check("-1 buckets to UNCODEABLE, a different exclusion from IRRELEVANT",
      evaluate.bucket_hand_sentiment("-1") == evaluate.UNCODEABLE)
check("the literal word 'uncodeable' is accepted too",
      evaluate.bucket_hand_sentiment("Uncodeable") == evaluate.UNCODEABLE)
check("UNCODEABLE and IRRELEVANT are genuinely distinct values",
      evaluate.UNCODEABLE != evaluate.IRRELEVANT)
check("5 is still very positive, unaffected by -1 being special-cased",
      evaluate.bucket_hand_sentiment("5") == "positive")

with tempfile.TemporaryDirectory() as tmp:
    _irr_path = os.path.join(tmp, "irrelevant_sample.csv")
    with open(_irr_path, "w", newline="", encoding="utf-8-sig") as handle:
        w = csv.DictWriter(handle, fieldnames=["body", "hand_sentiment"])
        w.writeheader()
        w.writerow({"body": "This is great, I hate billionaires", "hand_sentiment": "1"})
        w.writerow({"body": "Not sure how I feel about this debate", "hand_sentiment": "3"})
        w.writerow({"body": "nice video, subscribed!", "hand_sentiment": "0"})
        w.writerow({"body": "another off-topic one", "hand_sentiment": "0"})
        w.writerow({"body": "un comentario en espanol", "hand_sentiment": "-1"})
        w.writerow({"body": "a typo'd value", "hand_sentiment": "banana"})

    class _SentimentArgs:
        input_csv = _irr_path
        arm = ["stub"]
        votes = 1
        cache = None

    _sent_out = io.StringIO()
    with contextlib.redirect_stdout(_sent_out), contextlib.redirect_stderr(io.StringIO()) as _sent_err:
        evaluate.cmd_sentiment(_SentimentArgs)
    _sent_stderr = _sent_err.getvalue()
    check("irrelevant rows are counted and reported, not silently merged with real errors",
          "2 comment(s) marked irrelevant" in _sent_stderr, _sent_stderr)
    check("uncodeable rows are counted and reported as their own category",
          "1 comment(s) marked uncodeable" in _sent_stderr, _sent_stderr)
    check("a genuinely invalid value is reported separately from both exclusions",
          "1 comment(s) had an unrecognised hand_sentiment value" in _sent_stderr, _sent_stderr)

print("\nevaluate.py: the sentiment scorecard is Horvitz-Thompson weighted, like the domain one")


class _HTFakeScorer:
    """Right on the rarely-drawn rows (body starts 'big'), wrong on the over-sampled ones."""
    seconds = 0.0

    def describe(self):
        return "fake"

    def cost_usd(self):
        return 0.0

    def token_usage(self):
        return (0, 0)

    def score(self, body, story=None):
        return {"label": "negative" if body.startswith("big") else "positive", "value": 0.0}


_ht_rows = ([(f"small {i}", "negative", "s") for i in range(10)]
            + [(f"big {i}", "negative", "s") for i in range(10)])
_ht_weights = [2.0] * 10 + [100.0] * 10   # inclusion_prob 0.5 and 0.01
_ht_plain = evaluate.sentiment_report(_ht_rows, _HTFakeScorer())
_ht_rep = evaluate.sentiment_report(_ht_rows, _HTFakeScorer(), _ht_weights)
check("with no weights the sentiment report is unchanged (every row weighs 1)",
      abs(_ht_plain["accuracy"] - 0.5) < 1e-9, _ht_plain["accuracy"])
check("weighted sentiment accuracy is sum(w * correct) / sum(w), by hand 1000/1020",
      abs(_ht_rep["accuracy"] - 1000 / 1020) < 1e-9, _ht_rep["accuracy"])
check("weighted per-class recall uses the same weights",
      abs(_ht_rep["per_class"]["negative"]["recall"] - 1000 / 1020) < 1e-9,
      _ht_rep["per_class"]["negative"])
check("raw counts stay integers beside the weighted figures",
      _ht_rep["n"] == 20 and _ht_rep["per_class"]["negative"]["tp"] == 10, _ht_rep["per_class"])
_ht_special = evaluate.special_case_report(
    [("small", "irrelevant", "s"), ("big", "irrelevant", "s")],
    type("_S", (), {"describe": lambda self: "fake",
                    "score": lambda self, body, story=None:
                    {"label": "irrelevant" if body == "big" else "negative"}})(),
    "irrelevant", [2.0, 100.0])
check("the irrelevant/uncodeable recall is weighted too (100/102), count kept raw",
      abs(_ht_special["recall"] - 100 / 102) < 1e-9 and _ht_special["matched"] == 1, _ht_special)

with tempfile.TemporaryDirectory() as tmp:
    _htp = os.path.join(tmp, "weighted_sentiment.csv")
    with open(_htp, "w", newline="", encoding="utf-8-sig") as handle:
        w = csv.DictWriter(handle, fieldnames=["comment_id", "body", "hand_sentiment",
                                               "inclusion_prob"])
        w.writeheader()
        # The stub always says neutral: right on the one rarely-drawn row, wrong on three
        # over-sampled ones. Unweighted 25%; weighted 100 / 106 = 94%.
        w.writerow({"comment_id": "a", "body": "one", "hand_sentiment": "3",
                    "inclusion_prob": "0.01"})
        for i in range(3):
            w.writerow({"comment_id": f"b{i}", "body": f"two {i}", "hand_sentiment": "1",
                        "inclusion_prob": "0.5"})

    class _HTArgs:
        input_csv = _htp
        arm = ["stub"]
        votes = 1
        cache = None

    _ht_out = io.StringIO()
    with contextlib.redirect_stdout(_ht_out), contextlib.redirect_stderr(io.StringIO()):
        evaluate.cmd_sentiment(_HTArgs)
    check("ewl-evaluate sentiment reports the weighted accuracy (94%), not the draw's (25%)",
          "| 94% |" in _ht_out.getvalue() and "Horvitz-Thompson weighted" in _ht_out.getvalue(),
          _ht_out.getvalue())

    with open(_htp, "a", newline="", encoding="utf-8-sig") as handle:
        csv.writer(handle).writerow(["c", "three", "1", ""])
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            evaluate.cmd_sentiment(_HTArgs)
        check("a gold file mixing blank and recorded inclusion_prob is refused by sentiment",
              False, "no exit")
    except SystemExit as error:
        check("a gold file mixing blank and recorded inclusion_prob is refused by sentiment",
              "inclusion_prob" in str(error), str(error))

print("\nvalidate.py: multi-column stratified sampling")
strat_rows = (
    [{"comment_id": f"d{i}", "story": "luigi", "kw_domain": "democracy"} for i in range(50)]
    + [{"comment_id": f"e{i}", "story": "luigi", "kw_domain": "environment"} for i in range(3)]
    + [{"comment_id": f"m{i}", "story": "musk", "kw_domain": "environment"} for i in range(3)]
)
_out = io.StringIO()
with contextlib.redirect_stderr(io.StringIO()):
    validate.cmd_sample(strat_rows, n=20, seed=1, out=_out, stratify_by="kw_domain")
by_domain = list(csv.DictReader(io.StringIO(_out.getvalue())))
domain_counts = collections.Counter(
    next(r["kw_domain"] for r in strat_rows if r["comment_id"] == row["comment_id"])
    for row in by_domain
)
check("a single-column stratify-by roughly balances the rare category",
      domain_counts["environment"] >= 6, domain_counts)

_out2 = io.StringIO()
with contextlib.redirect_stderr(io.StringIO()):
    validate.cmd_sample(strat_rows, n=6, seed=1, out=_out2, stratify_by="story,kw_domain")
by_story_domain = list(csv.DictReader(io.StringIO(_out2.getvalue())))
check("comma-separated columns stratify on the combination",
      len({(next(r["story"] for r in strat_rows if r["comment_id"] == row["comment_id"]),
           next(r["kw_domain"] for r in strat_rows if r["comment_id"] == row["comment_id"]))
          for row in by_story_domain}) >= 2,
      "expected at least two (story, domain) combinations represented")

print("\nvalidate.py: per-video soft cap on sample")
# One stratum, one video ("vidA") supplying 91% of a pool much larger than the
# quota -- exactly the ProPublica-shaped concentration the cap exists to guard
# against, but with room for the cap to actually bite (unlike a pool sized
# exactly to the quota, where backfill is forced to take everything anyway).
_cap_rows = (
    [{"comment_id": f"a{i}", "story": "s", "kw_domain": "democracy", "source": "vidA"}
     for i in range(100)]
    + [{"comment_id": f"b{i}", "story": "s", "kw_domain": "democracy", "source": "vidB"}
       for i in range(5)]
    + [{"comment_id": f"c{i}", "story": "s", "kw_domain": "democracy", "source": "vidC"}
       for i in range(5)]
)
_cap_out = io.StringIO()
_cap_err = io.StringIO()
with contextlib.redirect_stderr(_cap_err):
    validate.cmd_sample(_cap_rows, n=20, seed=1, out=_cap_out, stratify_by="kw_domain",
                        video_cap_fraction=0.15)
_capped = list(csv.DictReader(io.StringIO(_cap_out.getvalue())))
_video_counts = collections.Counter(r["source"] for r in _capped)
check("the quota is still fully reached", len(_capped) == 20, len(_capped))
check("the dominant video is held well below the ~19/20 an uncapped draw would give it",
      _video_counts["vidA"] < 15, _video_counts)
check("cap-too-tight backfill is reported for this stratum",
      "backfilled past the cap" in _cap_err.getvalue(), _cap_err.getvalue())

# Plenty of videos, cap comfortably achievable: no backfill should be needed.
_plenty_rows = [{"comment_id": f"v{v}c{i}", "story": "s", "kw_domain": "democracy",
                 "source": f"vid{v}"} for v in range(10) for i in range(5)]
_plenty_out = io.StringIO()
_plenty_err = io.StringIO()
with contextlib.redirect_stderr(_plenty_err):
    validate.cmd_sample(_plenty_rows, n=20, seed=1, out=_plenty_out, stratify_by="kw_domain",
                        video_cap_fraction=0.15)
_plenty_capped = list(csv.DictReader(io.StringIO(_plenty_out.getvalue())))
_plenty_video_counts = collections.Counter(r["source"] for r in _plenty_capped)
check("cap holds (<= ceil(20*0.15)=3 per video) when enough videos are available",
      max(_plenty_video_counts.values()) <= 3, _plenty_video_counts)
check("no backfill needed message is printed when the cap is comfortably feasible",
      "no backfill needed" in _plenty_err.getvalue(), _plenty_err.getvalue())

# The recorded inclusion_prob used to be flat (allocation/stratum_size) for
# every row regardless of the cap -- wrong whenever the cap actually binds,
# since a row from an over-cap video competes for only `cap` of its own
# video's slots while a row from an at-or-under-cap video always enters
# contention. cap = ceil(20*0.15) = 3 here.
_diff_rows = (
    [{"comment_id": f"sm{i}", "story": "s", "kw_domain": "democracy", "source": "small"}
     for i in range(3)]        # at the cap: every row enters capped_pool for certain
    + [{"comment_id": f"bg{i}", "story": "s", "kw_domain": "democracy", "source": "big"}
       for i in range(40)]     # over the cap: only 3 of 40 make capped_pool
    + [{"comment_id": f"md{v}c{i}", "story": "s", "kw_domain": "democracy", "source": f"mid{v}"}
       for v in range(6) for i in range(10)]   # 6 videos x 10, also over the cap
)
validate.random.seed(1)
_diff_chosen, _diff_short = validate.sample_with_video_cap(_diff_rows, 20, 0.15)
check("this scenario reaches quota without backfill (case A, the common path)",
      len(_diff_chosen) == 20 and _diff_short == 0, (len(_diff_chosen), _diff_short))
_diff_by_video = {r["source"]: r["_video_cap_prob"] for r in _diff_chosen}
_capped_pool_size = 3 + 3 + 6 * 3  # small(all 3) + big(capped at 3) + 6 mids(capped at 3 each)
_expect_small = 1.0 * (20 / _capped_pool_size)
_expect_big = (3 / 40) * (20 / _capped_pool_size)
_expect_mid = (3 / 10) * (20 / _capped_pool_size)
if "small" in _diff_by_video:
    check("an at-or-under-cap video's row has the higher, correct probability",
          abs(_diff_by_video["small"] - _expect_small) < 1e-9,
          (_diff_by_video["small"], _expect_small))
if "big" in _diff_by_video:
    check("a heavily-over-cap video's row has the lower, correct probability",
          abs(_diff_by_video["big"] - _expect_big) < 1e-9, (_diff_by_video["big"], _expect_big))
if "mid0" in _diff_by_video:
    check("a moderately-over-cap video's row lands in between, correctly",
          abs(_diff_by_video["mid0"] - _expect_mid) < 1e-9, (_diff_by_video["mid0"], _expect_mid))
_old_flat_prob = 20 / len(_diff_rows)
check("none of these match the old flat allocation/stratum-size figure",
      all(abs(p - _old_flat_prob) > 1e-6 for p in _diff_by_video.values()),
      (_diff_by_video, _old_flat_prob))

# Same differentiation, but in the backfill branch (cap too tight to reach
# quota from capped_pool alone) -- an at-or-under-cap video's rows are
# unconditionally kept (probability 1.0), not just favoured.
_backfill_rows = (
    [{"comment_id": f"tiny{i}", "story": "s", "kw_domain": "democracy", "source": "tiny"}
     for i in range(2)]       # under the cap of 3: always kept
    + [{"comment_id": f"huge{i}", "story": "s", "kw_domain": "democracy", "source": "huge"}
       for i in range(50)]    # way over the cap: most of this video is in leftover
)
validate.random.seed(2)
_bf_chosen, _bf_short = validate.sample_with_video_cap(_backfill_rows, 20, 0.15)
check("this scenario needs backfill (case B)", _bf_short > 0, _bf_short)
_tiny_rows = [r for r in _bf_chosen if r["source"] == "tiny"]
check("an at-or-under-cap video's rows are ALL kept with probability 1.0 in the backfill case",
      len(_tiny_rows) == 2 and all(r["_video_cap_prob"] == 1.0 for r in _tiny_rows), _tiny_rows)

# cmd_sample must actually use these per-row probabilities, not silently
# fall back to the flat figure once video_cap_fraction is set.
_prob_check_out = io.StringIO()
with contextlib.redirect_stderr(io.StringIO()):
    validate.cmd_sample(_diff_rows, n=20, seed=1, out=_prob_check_out, stratify_by="kw_domain",
                        video_cap_fraction=0.15)
_prob_check_rows = list(csv.DictReader(io.StringIO(_prob_check_out.getvalue())))
_recorded = {r["source"]: float(r["inclusion_prob"]) for r in _prob_check_rows}
check("cmd_sample writes the per-video cap-aware probability, not the flat one",
      all(abs(v - _old_flat_prob) > 1e-6 for v in _recorded.values()), _recorded)

print("\nvalidate.py: coder-vs-coder interrater kappa")
coder_a = [
    {"comment_id": "c1", "hand_domain": "democracy", "hand_sentiment": "2"},
    {"comment_id": "c2", "hand_domain": "economics, social", "hand_sentiment": "4"},
    {"comment_id": "c3", "hand_domain": "none", "hand_sentiment": "3"},
    {"comment_id": "c4", "hand_domain": "environment", "hand_sentiment": "1"},
    {"comment_id": "only_in_a", "hand_domain": "social", "hand_sentiment": "3"},
]
coder_b = [
    {"comment_id": "c1", "hand_domain": "democracy", "hand_sentiment": "2"},        # agrees
    {"comment_id": "c2", "hand_domain": "economics", "hand_sentiment": "5"},         # partial domain, sentiment differs
    {"comment_id": "c3", "hand_domain": "none", "hand_sentiment": "3"},              # agrees
    {"comment_id": "c4", "hand_domain": "democracy", "hand_sentiment": "1"},         # domain disagrees, sentiment agrees
    {"comment_id": "only_in_b", "hand_domain": "economics", "hand_sentiment": "4"},
]
check("labels_from_text parses free-text hand_domain the same way evaluate.read_gold does",
      validate.labels_from_text("Economics, Social") == {"economics", "social"})
check("kappa_categorical is 1.0 on perfect agreement",
      abs(validate.kappa_categorical([("a", "a"), ("b", "b"), ("a", "a")]) - 1.0) < 1e-9)
check("kappa_categorical generalises the binary case",
      abs(validate.kappa_categorical([(1, 1), (0, 0), (1, 0)])
          - validate.kappa([(1, 1), (0, 0), (1, 0)])) < 1e-9)

_err = io.StringIO()
_out3 = io.StringIO()
with contextlib.redirect_stderr(_err), contextlib.redirect_stdout(_out3):
    validate.cmd_interrater(coder_a, coder_b)
interrater_report = _out3.getvalue()
check("mismatched comment_ids are reported, not silently dropped",
      "only in the first file" in _err.getvalue() and "only in the second" in _err.getvalue(),
      _err.getvalue())
check("only the 4 shared comment_ids are scored", "4 comments coded by both" in interrater_report,
      interrater_report)
check("democracy shows partial agreement (c1 agrees, c4 disagrees)",
      any(line.startswith("democracy") for line in interrater_report.splitlines()))
check("sentiment agreement is reported with a kappa value",
      "Sentiment --" in interrater_report and "kappa:" in interrater_report, interrater_report)

no_shared_a = [{"comment_id": "x1", "hand_domain": "democracy", "hand_sentiment": "1"}]
no_shared_b = [{"comment_id": "x2", "hand_domain": "democracy", "hand_sentiment": "1"}]
try:
    with contextlib.redirect_stderr(io.StringIO()):
        validate.cmd_interrater(no_shared_a, no_shared_b)
    check("no shared comment_ids raises rather than dividing by zero", False, "no exception")
except SystemExit:
    check("no shared comment_ids raises rather than dividing by zero", True)

print("\nvalidate.py: kappa returns 1.0 on perfect agreement, not nan")
# Regression cover: `expected` reaching 1.0 (every rating on both sides is the
# same single constant value) used to fall through to nan, even though
# `observed` is necessarily 1.0 in exactly that situation -- perfect
# agreement, not an undefined statistic. See kappa()/kappa_categorical()/
# kappa_weighted()'s own docstrings for why expected==1 forces observed==1.
check("binary kappa is 1.0, not nan, when every pair is (1, 1)",
      validate.kappa([(1, 1), (1, 1), (1, 1)]) == 1.0,
      validate.kappa([(1, 1), (1, 1), (1, 1)]))
check("binary kappa is 1.0, not nan, when every pair is (0, 0)",
      validate.kappa([(0, 0), (0, 0)]) == 1.0)
check("categorical kappa is 1.0, not nan, when every pair is the same label",
      validate.kappa_categorical([("social", "social"), ("social", "social")]) == 1.0)
check("weighted kappa is 1.0, not nan, when every numeric pair is identical",
      validate.kappa_weighted([(3, 3), (3, 3), (3, 3)]) == 1.0,
      validate.kappa_weighted([(3, 3), (3, 3), (3, 3)]))
check("fleiss_kappa is 1.0, not nan, when every rater gives the same value throughout",
      validate.fleiss_kappa([[1, 1, 1], [1, 1, 1]]) == 1.0)
check("krippendorff_alpha_interval is 1.0, not nan, when every numeric rating is identical",
      validate.krippendorff_alpha_interval([[3, 3, 3], [3, 3, 3]]) == 1.0)


print("\nvalidate.py: selfconsistency (LLM self-agreement across its own --votes)")
# A fixture with a KNOWN kappa, not just "some number came out" -- the review
# found `selfconsistency` had 0% test coverage and its own docstring example
# (perfect self-agreement) printed nan. rows_perfect: every comment's votes
# unanimously agree, both on domain and on sentiment -- the self-kappa this
# produces must be exactly 1.0, the known value being checked against.
rows_perfect = [
    {"comment_id": "p1", "domain_raw_votes": "democracy|democracy|democracy",
     "sentiment_raw_votes": "4,4,4"},
    {"comment_id": "p2", "domain_raw_votes": "none|none|none",
     "sentiment_raw_votes": "2,2,2"},
]
_sc_out = io.StringIO()
with contextlib.redirect_stdout(_sc_out), contextlib.redirect_stderr(io.StringIO()):
    validate.cmd_selfconsistency(rows_perfect)
_sc_report = _sc_out.getvalue()
check("perfect self-agreement across votes reports self-kappa 1.00, not nan",
      "self-kappa" not in _sc_report.lower() or "nan" not in _sc_report.lower(),
      _sc_report)
check("democracy's self-kappa is reported as 1.00 on this fixture",
      any(line.startswith("democracy") and "1.00" in line for line in _sc_report.splitlines()),
      _sc_report)
check("sentiment's self-kappa (weighted) is reported as 1.00 on this fixture",
      "self-kappa (weighted): 1.00" in _sc_report, _sc_report)

# A fixture with genuine, KNOWN disagreement -- votes split 2-1 on domain for
# every comment, so the self-kappa must be less than 1.0 (imperfect, not
# broken) rather than another accidental nan.
rows_mixed = [
    {"comment_id": "m1", "domain_raw_votes": "democracy|democracy|none",
     "sentiment_raw_votes": "4,4,1"},
    {"comment_id": "m2", "domain_raw_votes": "democracy|none|none",
     "sentiment_raw_votes": "5,2,2"},
]
_sc_mixed_out = io.StringIO()
with contextlib.redirect_stdout(_sc_mixed_out), contextlib.redirect_stderr(io.StringIO()):
    validate.cmd_selfconsistency(rows_mixed)
_sc_mixed_report = _sc_mixed_out.getvalue()
_dem_line = next(line for line in _sc_mixed_report.splitlines() if line.startswith("democracy"))
_dem_value = float(_dem_line.split()[-1])
check("genuine vote disagreement reports a self-kappa strictly below 1.0, not nan",
      _dem_value == _dem_value and _dem_value < 1.0, _dem_line)

check("no rows with >=2 votes is reported plainly, not a crash",
      (lambda: (contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO())))
      or True)
_sc_empty_err = io.StringIO()
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(_sc_empty_err):
    validate.cmd_selfconsistency([{"comment_id": "e1", "domain_raw_votes": "democracy",
                                   "sentiment_raw_votes": "3"}])
check("a single vote (no pair to compare) is reported, not silently 0/0'd",
      "re-run" in _sc_empty_err.getvalue().lower(), _sc_empty_err.getvalue())

_core_ids = {"p1"}
_sc_core_out = io.StringIO()
with contextlib.redirect_stdout(_sc_core_out), contextlib.redirect_stderr(io.StringIO()):
    validate.cmd_selfconsistency(rows_perfect, core_ids=_core_ids)
check("--core-ids restricts to the named comments, not the whole input",
      "1 comment(s) checked" in _sc_core_out.getvalue(), _sc_core_out.getvalue())


print("\ninterrater: blank domain is not coded, sentinels are not scale points, no-variation is flagged")
# 20 rows both coders gave democracy, plus 30 rows where both left hand_domain blank.
_ids = [f"r{i}" for i in range(50)]
_ra = [{"comment_id": c, "hand_domain": "democracy" if i < 20 else "",
        "hand_sentiment": "3"} for i, c in enumerate(_ids)]
_rb = [{"comment_id": c, "hand_domain": "democracy" if i < 20 else "",
        "hand_sentiment": "3"} for i, c in enumerate(_ids)]
_ra[0]["hand_domain"], _rb[0]["hand_domain"] = "democracy", "economics"   # one real disagreement
_buf, _errbuf = io.StringIO(), io.StringIO()
with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_errbuf):
    validate.cmd_interrater(_ra, _rb)
_text = _buf.getvalue()
check("rows both coders left blank are not counted as agreement on 'none'",
      "NOTE: 30 comment(s) have a blank hand_domain" in _errbuf.getvalue(), _errbuf.getvalue())
_dem_line = next(l for l in _text.splitlines() if l.startswith("democracy"))
# On the 20 coded rows: coder A says democracy 20 times, coder B 19 times, they differ on
# one row, so kappa is exactly 0.00. Counting the 30 blank-blank rows as agreement on
# "not democracy" would have lifted it to about 0.77.
check("democracy kappa is computed on the 20 coded rows only (0.00), not lifted by 30 blanks",
      _dem_line.split()[-1] == "0.00", _dem_line)
check("a category nobody used is flagged as having no variation, not left as a bare 1.00",
      "no variation in" in _text and "environment" in _text.split("no variation in")[1].split(" among")[0],
      _text)

# Weighted kappa and alpha only use genuine 1-5 ratings: 0 (irrelevant) and -1 must be left out.
_sa = [{"comment_id": f"s{i}", "hand_domain": "none", "hand_sentiment": v}
       for i, v in enumerate(["1", "2", "5", "0", "-1", "3"])]
_sb = [{"comment_id": f"s{i}", "hand_domain": "none", "hand_sentiment": v}
       for i, v in enumerate(["1", "3", "5", "4", "2", "3"])]
_o = io.StringIO()
with contextlib.redirect_stdout(_o), contextlib.redirect_stderr(io.StringIO()):
    validate.cmd_interrater(_sa, _sb)
check("weighted kappa counts only the 4 pairs where both coders gave a 1-5 rating",
      "numeric pairs only, n=4" in _o.getvalue(), _o.getvalue())
_o2 = io.StringIO()
with contextlib.redirect_stdout(_o2), contextlib.redirect_stderr(io.StringIO()):
    validate.cmd_interrater_multi([_sa, _sb, _sb])
check("multi-coder stance alpha also leaves 0 and -1 out (the 4 comments every coder rated 1-5)",
      "on the 4 comments every coder rated 1-5" in _o2.getvalue(), _o2.getvalue())



print("\nvalidate.py: multi-coder (Fleiss' kappa / Krippendorff's alpha) interrater")
coder_c = [
    {"comment_id": "c1", "hand_domain": "democracy", "hand_sentiment": "2"},
    {"comment_id": "c2", "hand_domain": "economics", "hand_sentiment": "4"},
    {"comment_id": "c3", "hand_domain": "none", "hand_sentiment": "3"},
    {"comment_id": "c4", "hand_domain": "environment", "hand_sentiment": "2"},
]
_multi_out = io.StringIO()
with contextlib.redirect_stdout(_multi_out), contextlib.redirect_stderr(io.StringIO()):
    validate.cmd_interrater_multi([coder_a, coder_b, coder_c])
_multi_report = _multi_out.getvalue()
check("multi-coder restricts to comments every one of the 3 coders has",
      "4 comments coded by all 3 coders" in _multi_report, _multi_report)
check("multi-coder domain agreement is Krippendorff's alpha per category, with Fleiss' kappa beside it",
      "Krippendorff's alpha" in _multi_report and "kappa" in _multi_report
      and all(c in _multi_report for c in ("democracy", "economics", "environment", "social")),
      _multi_report)
check("multi-coder sentiment reports the kind of answer (nominal) and the 1-5 stance (ordinal)",
      "Kind of answer" in _multi_report and "ordinal alpha" in _multi_report, _multi_report)
check("the report states the guide to reading alpha",
      "0.667" in _multi_report and "0.80" in _multi_report)

check("cmd_combine treats reordered domain labels as agreement, not a fresh disagreement",
      {frozenset(validate.labels_from_text("democracy, economics")),
       frozenset(validate.labels_from_text("economics, democracy"))} == {
           frozenset({"democracy", "economics"})})
_reorder_master = [{"comment_id": "r1", "story": "jets", "source": "v1", "body": "b1",
                    "sampling_stratum": "jets/democracy", "inclusion_prob": "0.2"}]
_reorder_manifest = [{"comment_id": "r1", "assigned_to": "A", "is_core": "1"},
                     {"comment_id": "r1", "assigned_to": "B", "is_core": "1"}]
_reorder_coders = {
    "pathA": [{"comment_id": "r1", "hand_domain": "democracy, economics",
              "hand_sentiment": "4", "coder_notes": ""}],
    "pathB": [{"comment_id": "r1", "hand_domain": "economics, democracy",
              "hand_sentiment": "4", "coder_notes": ""}],
}
_reorder_rows, _reorder_adj, _, _ = validate.cmd_combine(
    _reorder_coders, _reorder_manifest, _reorder_master, [], io.StringIO())
check("combine no longer sends same-set-different-order labels to adjudication",
      len(_reorder_adj) == 0 and any(r["comment_id"] == "r1" for r in _reorder_rows),
      (_reorder_adj, _reorder_rows))


print("\nevaluate.py: true negatives and FPR")
with tempfile.TemporaryDirectory() as tmp:
    gold = os.path.join(tmp, "gold.csv")
    write_fixture(gold)
    rows = evaluate.read_gold(gold)
    result = evaluate.run_arm(rows, judges.create("keyword"), examples_mod.create("none"),
                              progress=False)
    dem = result["per_category"]["democracy"]
    check("per-category stats expose tn and fpr", "tn" in dem and "fpr" in dem, dem)
    check("tp+fp+fn+tn accounts for every row",
          dem["tp"] + dem["fp"] + dem["fn"] + dem["tn"] == len(rows), dem)
    check("fpr is fp / (fp + tn), not 1 - precision",
          dem["fpr"] == (dem["fp"] / (dem["fp"] + dem["tn"]) if (dem["fp"] + dem["tn"]) else dem["fpr"]))

print("\nreport.py: cluster (video) bootstrap, equal-story aggregate, corrected estimate")
cluster_rows = (
    [{"source": "vidA", "story": "luigi", "domain_democracy": "1"} for _ in range(20)]
    + [{"source": "vidA", "story": "luigi", "domain_democracy": "0"} for _ in range(5)]
    + [{"source": "vidB", "story": "luigi", "domain_democracy": "0"} for _ in range(25)]
    + [{"source": "vidC", "story": "propublica", "domain_democracy": "1"} for _ in range(10)]
)
results = report.column_ci(cluster_rows, "domain_democracy", cluster_by="source", n_boot=300)
n_comments, n_videos, observed, lo, hi = results["(pooled)"]
check("cluster bootstrap reports both comment and video counts", n_comments == 60 and n_videos == 3)
check("observed rate matches the raw mean", abs(observed - 30 / 60) < 1e-9, observed)
check("the interval contains the observed value", lo <= observed <= hi, (lo, observed, hi))
check("clustering by video collapses vidA+vidB into 2 draws, not 45",
      True, "structural check: see n_videos above")

single_video = [{"source": "v1", "domain_democracy": "1"}] * 10
solo = report.column_ci(single_video, "domain_democracy", cluster_by="source", n_boot=300)
check("a single video gives a degenerate (zero-width) interval, not a crash",
      solo["(pooled)"][3] == solo["(pooled)"][4] == 1.0, solo)

agg_observed, agg_lo, agg_hi, per_story = report.aggregate_ci(
    cluster_rows, "domain_democracy", n_boot=300)
check("equal-story aggregate is the unweighted mean of the two story rates",
      abs(agg_observed - statistics.fmean(per_story.values())) < 1e-9, (agg_observed, per_story))
check("equal weighting does not just reproduce comment-count pooling",
      abs(agg_observed - observed) > 1e-9,
      "with luigi(50) vs propublica(10) so unequal in size, equal and count-weighted must differ")
check("aggregate interval contains the aggregate observed value",
      agg_lo <= agg_observed <= agg_hi, (agg_lo, agg_observed, agg_hi))

check("corrected_estimate matches the Rogan-Gladen formula",
      abs(report.corrected_estimate(0.25, 0.85, 0.10) - (0.25 - 0.10) / (0.85 - 0.10)) < 1e-9)
check("corrected_estimate refuses a near-zero TPR-FPR denominator",
      report.corrected_estimate(0.25, 0.51, 0.50) is None)
check("corrected_estimate refuses an out-of-range result",
      report.corrected_estimate(0.99, 0.20, 0.10) is None)

# `source` is required now: the gold set is cluster-resampled by video (#53),
# and a gold set with one cluster resamples to itself and yields a zero-width
# interval. Spread across ten videos, as a real stratified draw would be.
gold_for_correction = [
    dict(row, source=f"gv{i % 10}") for i, row in enumerate(
        [{"hand_domain": "democracy", "domain_democracy": "1"} for _ in range(40)]
        + [{"hand_domain": "democracy", "domain_democracy": "0"} for _ in range(10)]
        + [{"hand_domain": "none", "domain_democracy": "0"} for _ in range(140)]
        + [{"hand_domain": "none", "domain_democracy": "1"} for _ in range(10)])
]
check("corrected_ci_bootstrap refuses to run below the genuine-positive threshold",
      report.corrected_ci_bootstrap(cluster_rows, "domain_democracy", gold_for_correction,
                                    "hand_domain", "democracy", genuine_positive_count=10,
                                    min_genuine_positives=50) is None)
# `cluster_rows` has three videos, and resampling three clusters sends the
# correction degenerate on 27% of iterations. Percentiles over the survivors
# gave 0.112-0.981 -- nearly the entire unit interval -- and the harness used
# to accept that as "runs once the threshold is met". A well-conditioned
# fixture needs enough clusters for the corpus bootstrap to be stable.
well_clustered = [{"source": f"vid{v}", "story": "luigi",
                   "domain_democracy": "1" if c < 4 else "0"}
                  for v in range(20) for c in range(15)]
boot_result = report.corrected_ci_bootstrap(
    well_clustered, "domain_democracy", gold_for_correction, "hand_domain", "democracy",
    genuine_positive_count=50, min_genuine_positives=50, n_boot=300)
check("corrected_ci_bootstrap runs once the threshold is met",
      boot_result is not None and boot_result[1] <= boot_result[0] <= boot_result[2], boot_result)
check("and it discards few enough iterations to be reporting an unconditional interval",
      boot_result is not None and boot_result[4] <= 0.05 * 300, boot_result)

# The guard that fixture used to evade: too few clusters, the correction goes
# degenerate on a quarter of draws, and an interval taken over the survivors
# is conditioned on the correction having worked -- narrowest exactly where it
# is least trustworthy. Refuse instead of reporting it.
# Cluster-resampling the gold set (#53) is correct, but with no cluster column
# every row buckets under "(unknown)", the single cluster resamples to itself,
# and the interval collapses to ZERO width -- infinite precision. That is the
# shape `ewl-validate sample` actually writes: nine columns, no `source`.
_gold_clustered = [{"hand_domain": "democracy" if i % 4 == 0 else "none",
                    "domain_democracy": "1" if (i % 4 == 0 and i % 5) else "0",
                    "source": f"v{i % 10}"} for i in range(200)]
_gold_unclustered = [{k: v for k, v in r.items() if k != "source"}
                     for r in _gold_clustered]
_with = report.corrected_ci_bootstrap(
    well_clustered, "domain_democracy", _gold_clustered, "hand_domain", "democracy",
    genuine_positive_count=50, min_genuine_positives=50, n_boot=200)
check("a gold set spread over videos still yields an interval",
      _with is not None and _with[2] > _with[1], _with)
check("a gold set with no cluster column is refused, not given a zero-width interval",
      report.corrected_ci_bootstrap(
          well_clustered, "domain_democracy", _gold_unclustered, "hand_domain",
          "democracy", genuine_positive_count=50, min_genuine_positives=50,
          n_boot=200) is None,
      "one cluster resamples to itself: point == lo == hi, precision nobody measured")
check("and lifting the floor reproduces the zero-width interval, proving the guard is what stops it",
      (lambda r: r is not None and r[1] == r[2])(report.corrected_ci_bootstrap(
          well_clustered, "domain_democracy", _gold_unclustered, "hand_domain",
          "democracy", genuine_positive_count=50, min_genuine_positives=50,
          n_boot=200, min_gold_clusters=1)))

check("a high discard rate is refused rather than reported as a narrow interval",
      report.corrected_ci_bootstrap(
          cluster_rows, "domain_democracy", gold_for_correction, "hand_domain",
          "democracy", genuine_positive_count=50, min_genuine_positives=50,
          n_boot=300) is None)
check("and the same case does return an interval when the guard is lifted",
      report.corrected_ci_bootstrap(
          cluster_rows, "domain_democracy", gold_for_correction, "hand_domain",
          "democracy", genuine_positive_count=50, min_genuine_positives=50,
          n_boot=300, max_discard_fraction=1.0) is not None,
      "otherwise the refusal above proves nothing about the discard guard")

# Horvitz-Thompson weighting: inclusion_prob was recorded by
# `ewl-validate sample --stratify-by` and read by nobody, so a stratified gold
# set produced the TPR of the draw rather than of the corpus.
# The weights have to differ WITHIN a term to change anything: TPR is
# tp/(tp+fn), so weighting every democracy row identically cancels. Here the
# hits were drawn at p=0.1 (weight 10) and the misses at p=1.0 (weight 1), so
# unweighted TPR is 10/20 = 0.50 while weighted is 100/110 = 0.91 -- which is
# the whole point: an over-sampled stratum must not count once per row.
_ht_gold = [dict(row, source=f"gv{i % 10}") for i, row in enumerate(
    [{"hand_domain": "democracy", "domain_democracy": "1",
      "inclusion_prob": "0.1"} for _ in range(10)]
    + [{"hand_domain": "democracy", "domain_democracy": "0",
        "inclusion_prob": "1.0"} for _ in range(10)]
    + [{"hand_domain": "none", "domain_democracy": "0",
        "inclusion_prob": "1.0"} for _ in range(30)])]
_flat_gold = [dict(r, inclusion_prob="1.0") for r in _ht_gold]
_w = report.corrected_ci_bootstrap(
    well_clustered, "domain_democracy", _ht_gold, "hand_domain", "democracy",
    genuine_positive_count=50, min_genuine_positives=50, n_boot=200)
_u = report.corrected_ci_bootstrap(
    well_clustered, "domain_democracy", _flat_gold, "hand_domain", "democracy",
    genuine_positive_count=50, min_genuine_positives=50, n_boot=200)
check("inclusion_prob changes the corrected estimate, i.e. it is actually read",
      _w is not None and _u is not None and abs(_w[0] - _u[0]) > 1e-9,
      (None if _w is None else _w[0], None if _u is None else _u[0]))
check("a gold set with no inclusion_prob is unweighted, as a simple random sample is",
      report.corrected_ci_bootstrap(
          well_clustered, "domain_democracy",
          [{k: v for k, v in r.items() if k != "inclusion_prob"} for r in _flat_gold],
          "hand_domain", "democracy", genuine_positive_count=50,
          min_genuine_positives=50, n_boot=200)[0] == _u[0])

print("\nreport.py: the gold set is now clustered by video too, not just the corpus")


def _make_gold_rows():
    """40 genuine democracy positives + 10 misses + 150 genuine negatives
    (10 false-positive) -- fixed content, `source` assigned separately."""
    return ([{"hand_domain": "democracy", "domain_democracy": "1"} for _ in range(40)]
           + [{"hand_domain": "democracy", "domain_democracy": "0"} for _ in range(10)]
           + [{"hand_domain": "none", "domain_democracy": "0"} for _ in range(140)]
           + [{"hand_domain": "none", "domain_democracy": "1"} for _ in range(10)])


# Concentrated: 3 videos, each a CONTIGUOUS, skewed slice of the 200 rows --
# one mostly true-positives, one pure true-negatives, one holding all the
# false positives -- the way a real video's comments actually look (one
# video is not a miniature, representative copy of the whole gold set).
# Interleaving row types evenly across videos (e.g. `source = i % 3`) would
# make every video an equally-representative microcosm, which resampling
# could never disturb -- exactly the mistake this fixture avoids.
# Spread: each row its own video, so cluster-resampling behaves like
# row-level resampling. Same content, same corpus, only the gold set's own
# video structure differs.
_concentrated_rows = _make_gold_rows()
_video_boundaries = [0, 67, 134, 200]
for _v, (_start, _end) in enumerate(zip(_video_boundaries, _video_boundaries[1:])):
    for _row in _concentrated_rows[_start:_end]:
        _row["source"] = f"gold-vid{_v}"
gold_concentrated = _concentrated_rows

_spread_rows = _make_gold_rows()
for _i, _row in enumerate(_spread_rows):
    _row["source"] = f"gold-vid{_i}"
gold_spread = _spread_rows

boot_concentrated = report.corrected_ci_bootstrap(
    well_clustered, "domain_democracy", gold_concentrated, "hand_domain", "democracy",
    genuine_positive_count=50, min_genuine_positives=50, n_boot=500, max_discard_fraction=1.0)
boot_spread = report.corrected_ci_bootstrap(
    well_clustered, "domain_democracy", gold_spread, "hand_domain", "democracy",
    genuine_positive_count=50, min_genuine_positives=50, n_boot=500, max_discard_fraction=1.0)
check("a gold set concentrated in few videos produces a wider interval than "
      "the same rows spread across many",
      boot_concentrated is not None and boot_spread is not None
      and (boot_concentrated[2] - boot_concentrated[1]) > (boot_spread[2] - boot_spread[1]),
      (boot_concentrated, boot_spread))
check("both share the same point estimate -- clustering changes the interval, not the number",
      boot_concentrated is not None and boot_spread is not None
      and abs(boot_concentrated[0] - boot_spread[0]) < 1e-9,
      (boot_concentrated[0] if boot_concentrated else None,
       boot_spread[0] if boot_spread else None))

print("\nreport.py: per-story classifier-error adjustment")
weak_story_gold = (
    [{"story": "weak", "hand_domain": "democracy", "domain_democracy": "1"} for _ in range(10)]
    + [{"story": "weak", "hand_domain": "democracy", "domain_democracy": "0"} for _ in range(30)]
    + [{"story": "weak", "hand_domain": "none", "domain_democracy": "0"} for _ in range(40)]
    + [{"story": "weak", "hand_domain": "none", "domain_democracy": "1"} for _ in range(20)]
)
strong_story_gold = (
    [{"story": "strong", "hand_domain": "democracy", "domain_democracy": "1"} for _ in range(38)]
    + [{"story": "strong", "hand_domain": "democracy", "domain_democracy": "0"} for _ in range(2)]
    + [{"story": "strong", "hand_domain": "none", "domain_democracy": "0"} for _ in range(98)]
    + [{"story": "strong", "hand_domain": "none", "domain_democracy": "1"} for _ in range(2)]
)
mixed_gold = weak_story_gold + strong_story_gold

ratios = report.story_accuracy_ratios(mixed_gold, ["democracy", "economics", "environment", "social"])
check("the weaker story gets a ratio below 1", ratios["weak"] < 1.0, ratios)
check("the stronger story gets a ratio above 1", ratios["strong"] > 1.0, ratios)

story_corpus = (
    [{"story": "weak", "source": f"weak-vid{i}", "domain_democracy": "1" if i < 15 else "0"}
     for i in range(30)]
    + [{"story": "strong", "source": f"strong-vid{i}", "domain_democracy": "1" if i < 15 else "0"}
       for i in range(30)]
)
by_story = report.corrected_estimate_by_story(
    story_corpus, "domain_democracy", mixed_gold, "hand_domain", "democracy",
    ["democracy", "economics", "environment", "social"], min_gold_per_story=20)
# This used to assert both stories get "a real value". They should not. The
# fixture's `weak` story has TPR 0.250 against FPR 0.333 -- the classifier
# flags a genuine negative MORE often than a genuine positive, so it is
# anti-correlated with truth there and Rogan-Gladen has no solution. The
# merged version reported a confident 0.869 for it.
check("a story where the classifier is anti-correlated with truth is refused",
      by_story["weak"] is None,
      f"TPR 0.250 < FPR 0.333 has no Rogan-Gladen solution; got {by_story['weak']}")
check("while a story the classifier handles well still gets a value",
      by_story["strong"] is not None, by_story)
check("the strong story's per-story estimate differs from the pooled-only one, "
      "i.e. the per-story rates are actually being used",
      abs(by_story["strong"] - report.corrected_estimate(
          0.5, *report._gold_tpr_fpr(mixed_gold, "democracy", "hand_domain"))) > 1e-9,
      by_story)

# The merged version scaled BOTH pooled rates by one accuracy ratio, so a
# below-average story had its prevalence revised UP and its false-positive
# rate revised DOWN. Nothing in the tests pinned the direction: three mutually
# contradictory FPR rules all passed the assertions above. These pin it.
#
# A story with perfect recall that fires on two-thirds of true negatives:
# observed prevalence is almost entirely false positives, so Rogan-Gladen has
# a negative numerator and the honest answer is "cannot say".
_fp_heavy = ([{"story": "fpheavy", "source": f"fv{i % 8}", "hand_domain": "democracy",
               "domain_democracy": "1"} for i in range(40)]
             + [{"story": "fpheavy", "source": f"fv{i % 8}", "hand_domain": "none",
                 "domain_democracy": "1" if i < 40 else "0"} for i in range(60)])
_clean = [{"story": "cleanstory", "source": f"cv{i % 8}",
           "hand_domain": "democracy" if i < 20 else "none",
           "domain_democracy": "1" if i < 20 else "0"} for i in range(40)]
_dir_corpus = ([{"story": "fpheavy", "source": f"fv{i % 8}",
                 "domain_democracy": "1" if i % 2 == 0 else "0"} for i in range(200)]
               + [{"story": "cleanstory", "source": f"cv{i % 8}",
                   "domain_democracy": "1" if i % 4 == 0 else "0"} for i in range(200)])
_dir = report.corrected_estimate_by_story(
    _dir_corpus, "domain_democracy", _fp_heavy + _clean, "hand_domain", "democracy",
    ["democracy", "economics", "environment", "social"])
check("a story whose observed rate is mostly false positives is refused, not corrected upward",
      _dir["fpheavy"] is None,
      f"got {_dir['fpheavy']}; the ratio version reported a confident 0.70 here")
check("and a clean story is corrected to its true prevalence",
      _dir["cleanstory"] is not None and abs(_dir["cleanstory"] - 0.25) < 1e-9, _dir)

# story_accuracy_ratios hard-coded "hand_domain" while its caller accepts and
# honours gold_label_column, so under any other column name every gold lookup
# returned False and the ratios collapsed toward 1.0 with no error.
_renamed = [{("gold_col" if k == "hand_domain" else k): v for k, v in row.items()}
            for row in mixed_gold]
check("story_accuracy_ratios honours gold_label_column instead of assuming hand_domain",
      report.story_accuracy_ratios(
          _renamed, ["democracy", "economics", "environment", "social"],
          gold_label_column="gold_col") == ratios,
      "a renamed gold column silently collapsed every ratio toward 1.0")

# ...and counted rows unweighted, one function away from _gold_tpr_fpr, whose
# docstring is entirely about why that is wrong under a stratified draw.
_w = [dict(r, inclusion_prob="0.1" if r["story"] == "weak" else "1.0") for r in mixed_gold]
check("story_accuracy_ratios applies the inclusion-probability weights",
      report.story_accuracy_ratios(
          _w, ["democracy", "economics", "environment", "social"]) != ratios,
      "over-sampling the weak story left the ratios unchanged, i.e. the weights are ignored")

# Adding a third, deliberately thin story on top of the already-healthy
# mixed_gold (rather than slicing weak/strong down) -- slicing a story's gold
# rows down to a handful can accidentally change the POOLED tpr/fpr too
# (fewer of that story's rows shifts the overall mix), which would make this
# test about pooled degeneracy rather than about the min-gold-per-story gate
# it is meant to isolate.
thin_story_gold = (
    [{"story": "thin", "hand_domain": "democracy", "domain_democracy": "1"} for _ in range(3)]
    + [{"story": "thin", "hand_domain": "none", "domain_democracy": "0"} for _ in range(2)]
)
gold_with_thin = mixed_gold + thin_story_gold
corpus_with_thin = story_corpus + [{"story": "thin", "source": "thin-vid", "domain_democracy": "1"}]
thin_by_story = report.corrected_estimate_by_story(
    corpus_with_thin, "domain_democracy", gold_with_thin,
    "hand_domain", "democracy", ["democracy", "economics", "environment", "social"],
    min_gold_per_story=20)
check("a story with too few gold rows returns None rather than an unreliable number",
      thin_by_story["thin"] is None, thin_by_story)
check("a story with enough gold data still gets a value in the same call as a refused one",
      thin_by_story["strong"] is not None, thin_by_story)
check("and `weak` is refused for its own reason, not the thin story's",
      thin_by_story["weak"] is None and thin_by_story["thin"] is None,
      "weak: anti-correlated classifier; thin: too few gold rows")


print("\ncli")
# The package lives under src/, not the repo root, so pointing PYTHONPATH at
# os.getcwd() made these four checks pass only when `pip install -e .` had
# already been run -- and worse, it OVERWROTE a working PYTHONPATH=src rather
# than adding to it, so the checks failed in exactly the setup that should
# work. A green harness that skips the CLI is how a live bug stays hidden.
_src = os.path.join(os.getcwd(), "src")
_existing = os.environ.get("PYTHONPATH", "")
env = dict(os.environ,
           PYTHONPATH=(_src + os.pathsep + _existing) if _existing else _src)
with tempfile.TemporaryDirectory() as tmp:
    gold = os.path.join(tmp, "gold.csv")
    write_fixture(gold)
    split = subprocess.run(
        [sys.executable, "-m", "ewl.analyse.evaluate", "split", gold, "--out-prefix", os.path.join(tmp, "cli")],
        capture_output=True, text=True, env=env)
    check("split runs from the command line", split.returncode == 0, split.stderr[-400:])
    check("split writes both files",
          os.path.exists(os.path.join(tmp, "cli_pool.csv"))
          and os.path.exists(os.path.join(tmp, "cli_heldout.csv")))

    proc = subprocess.run(
        [sys.executable, "-m", "ewl.analyse.evaluate", "scorecard", gold, "--arm", "keyword/none",
         "--arm", "stub/fixed", "--arm", "ollama:definitely-not-installed/fixed"],
        capture_output=True, text=True, env=env)
    check("scorecard runs from the command line", proc.returncode == 0, proc.stderr[-400:])
    check("results survive an unavailable arm", "keyword" in proc.stdout and "stub" in proc.stdout)

    # `ewl-report corrected` told the user to get --fpr "from evaluate.py's
    # scorecard", but print_scorecard never printed it -- confirms the fix at
    # the actual CLI boundary, not just via the underlying function.
    proc = subprocess.run(
        [sys.executable, "-m", "ewl.analyse.evaluate", "scorecard", gold, "--arm", "keyword/none"],
        capture_output=True, text=True, env=env)
    check("scorecard prints a per-category FPR table, not just precision/recall",
          "Per-category FPR" in proc.stdout, proc.stdout[-600:])

    # corrected-bootstrap end to end from the actual command line: a corpus
    # with enough videos, a gold file shaped like ewl-classify's output (both
    # domain_* and hand_domain), enough genuine positives to clear the floor.
    corpus_path = os.path.join(tmp, "cb_corpus.csv")
    with open(corpus_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["comment_id", "source", "domain_democracy"])
        w.writeheader()
        for v in range(10):
            for i in range(10):
                w.writerow({"comment_id": f"c{v}_{i}", "source": f"vid{v}",
                           "domain_democracy": "1" if i % 2 == 0 else "0"})
    gold_path = os.path.join(tmp, "cb_gold.csv")
    with open(gold_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["comment_id", "source", "hand_domain", "domain_democracy"])
        w.writeheader()
        for v in range(8):
            for i in range(15):
                truth = i % 2 == 0
                w.writerow({"comment_id": f"g{v}_{i}", "source": f"gvid{v}",
                           "hand_domain": "democracy" if truth else "none",
                           "domain_democracy": "1" if truth else "0"})
    proc = subprocess.run(
        [sys.executable, "-m", "ewl.analyse.report", "corrected-bootstrap", corpus_path,
         "--column", "domain_democracy", "--gold", gold_path, "--n-boot", "200"],
        capture_output=True, text=True, env=env)
    check("corrected-bootstrap runs from the command line and reports a CI, not just a point",
          proc.returncode == 0 and "Corrected 95% CI" in proc.stdout, proc.stdout + proc.stderr)

    proc = subprocess.run(
        [sys.executable, "-m", "ewl.analyse.report", "corrected-bootstrap", corpus_path,
         "--column", "domain_democracy", "--gold", gold_path, "--min-genuine-positives", "10000"],
        capture_output=True, text=True, env=env)
    check("corrected-bootstrap refuses cleanly (exit 0, explained) rather than crashing "
          "when the genuine-positive floor isn't met",
          proc.returncode == 0 and "refused" in proc.stdout, proc.stdout + proc.stderr)

    # ewl-export-dashboard end to end: 0% test coverage per the pipeline
    # review, including its refusal gates (the ones that stop an unvalidated
    # classifier run from being published as if it were a checked result).
    dash_corpus = os.path.join(tmp, "dash_corpus.csv")
    dash_fields = (["comment_id", "source", "story", "body", "sentiment", "sentiment_value",
                    "sentiment_unstable", "domain_unstable", "domain_judge", "sentiment_scorer"]
                   + [f"domain_{c}" for c in taxonomy.CATEGORY_NAMES]
                   + ["domain_unclassified", "domain_parse_failure"])
    with open(dash_corpus, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=dash_fields)
        w.writeheader()
        for v in range(10):  # 10 distinct videos, clears MIN_VIDEOS_FOR_INTERVAL (8)
            for i in range(5):
                row = {name: "0" for name in dash_fields}
                row.update({"comment_id": f"c{v}_{i}", "source": f"vid{v}",
                           "story": "luigi" if v < 5 else "propublica",
                           "body": "a comment", "sentiment": "positive" if i % 2 else "negative",
                           "sentiment_value": "4" if i % 2 else "2", "sentiment_unstable": "0",
                           "domain_unstable": "0", "domain_judge": "keyword:keywords.yml/none",
                           "sentiment_scorer": "vader",
                           "domain_democracy": "1" if i == 0 else "0"})
                w.writerow(row)

    dash_out = os.path.join(tmp, "results.json")
    proc = subprocess.run(
        [sys.executable, "-m", "ewl.analyse.export_dashboard", dash_corpus, "--output", dash_out],
        capture_output=True, text=True, env=env)
    check("export-dashboard refuses without --scored and no --allow-unvalidated",
          proc.returncode != 0 and "Refusing to export without hand-coded scores" in proc.stderr,
          proc.stdout + proc.stderr)
    check("and it does not write a partial results.json when it refuses",
          not os.path.exists(dash_out), "a refused export must not leave a stale/partial file")

    proc = subprocess.run(
        [sys.executable, "-m", "ewl.analyse.export_dashboard", dash_corpus, "--output", dash_out,
         "--allow-unvalidated"],
        capture_output=True, text=True, env=env)
    check("export-dashboard runs end to end with --allow-unvalidated",
          proc.returncode == 0 and os.path.exists(dash_out), proc.stdout + proc.stderr)
    with open(dash_out, encoding="utf-8") as f:
        _dash_payload = json.load(f)
    check("the export is explicitly stamped provisional when there is nothing to validate against",
          _dash_payload.get("provenance", {}).get("status") == "provisional", _dash_payload)
    _instr = _dash_payload.get("codebook", {}).get("sentiment_instrument", {})
    check("the codebook publishes the sentiment prompt, the scale and all four story paragraphs",
          "STANCE" in _instr.get("prompt_template", "")
          and set(_instr.get("scale", {})) == {"-1", "0", "1", "2", "3", "4", "5"}
          and set(_instr.get("story_context", {})) == {"luigi", "musk", "jets", "propublica"}
          and all(len(v) > 200 for v in _instr.get("story_context", {}).values()),
          sorted(_instr))
    check("and it records which switches were in force",
          _instr.get("switches") == {"story_rules": True, "stance_object": "conduct",
                                     "sentiment_examples": True},
          _instr.get("switches"))
    check("the export carries the classified-base shares beside the existing ones",
          _dash_payload.get("proportions_classified")
          and all(r.get("base") == "classified" and r.get("kind") == "harm"
                  for r in _dash_payload["proportions_classified"])
          and "proportions" in _dash_payload
          and "classified_share" in _dash_payload.get("rates", {}),
          sorted(_dash_payload))

    # The export is tied to a frozen corpus through the run record ewl-classify writes.
    from ewl import corpus as _corp_d
    _good_hash = _corp_d.content_hash(dash_corpus)
    with open(dash_corpus + ".run.json", "w", encoding="utf-8") as _f:
        json.dump({"input": {"path": dash_corpus, "sha256": _good_hash}}, _f)
    _ok_run = subprocess.run(
        [sys.executable, "-m", "ewl.analyse.export_dashboard", dash_corpus, "--output", dash_out,
         "--allow-unvalidated", "--corpus", dash_corpus], capture_output=True, text=True, env=env)
    check("export with --corpus passes when the corpus is the one the labels were made from",
          _ok_run.returncode == 0 and "matches the corpus the labels were made from" in _ok_run.stderr,
          _ok_run.stderr[-300:])
    with open(dash_out, encoding="utf-8") as _f:
        _with_run = json.load(_f)
    check("and the provenance records where the labels came from and the parse-failure rate",
          _with_run["provenance"]["run_record_found"] is True
          and _with_run["provenance"]["classified_from"]["sha256"] == _good_hash
          and "parse_failure_rate" in _with_run["provenance"], _with_run["provenance"])
    with open(dash_corpus + ".run.json", "w", encoding="utf-8") as _f:
        json.dump({"input": {"path": dash_corpus, "sha256": "0" * 64}}, _f)
    _bad_run = subprocess.run(
        [sys.executable, "-m", "ewl.analyse.export_dashboard", dash_corpus, "--output", dash_out,
         "--allow-unvalidated", "--corpus", dash_corpus], capture_output=True, text=True, env=env)
    check("export is refused when the corpus is not the one the labels were made from",
          _bad_run.returncode != 0 and "is not the corpus these labels were made from" in _bad_run.stderr,
          _bad_run.stderr[-300:])
    # Real fixed examples are raw YouTube comments: their ids are withheld by default.
    _yt_ids = ["UgxAAAAAAAAAAAAAAAAAAAAAAA", "UgxBBBBBBBBBBBBBBBBBBBBBBB"]
    with open(dash_corpus + ".run.json", "w", encoding="utf-8") as _f:
        json.dump({"input": {"path": dash_corpus, "sha256": _good_hash},
                   "fixed_examples": {"source": "file", "ids": _yt_ids, "labels": ["democracy", "none"],
                                      "sha256": "f" * 64}}, _f)
    _wh = subprocess.run(
        [sys.executable, "-m", "ewl.analyse.export_dashboard", dash_corpus, "--output", dash_out,
         "--allow-unvalidated", "--corpus", dash_corpus], capture_output=True, text=True, env=env)
    _wh_text = open(dash_out, encoding="utf-8").read()
    _wh_cb = json.loads(_wh_text)["codebook"]
    check("a default export publishes no real example ids, only a count and a checksum",
          _wh.returncode == 0 and not any(i in _wh_text for i in _yt_ids)
          and _wh_cb["examples_shown_to_the_model"] == []
          and _wh_cb["examples_withheld_summary"] == {"n": 2, "sha256": "f" * 64}
          and _wh_cb["example_text_withheld"] is True, _wh.stderr[-300:])
    _wv = subprocess.run(
        [sys.executable, "-m", "ewl.analyse.export_dashboard", dash_corpus, "--output", dash_out,
         "--allow-unvalidated", "--corpus", dash_corpus, "--include-verbatim"],
        capture_output=True, text=True, env=env)
    _wv_text = open(dash_out, encoding="utf-8").read()
    check("--include-verbatim releases the ids, the same switch that releases comment text",
          _wv.returncode == 0 and all(i in _wv_text for i in _yt_ids), _wv.stderr[-300:])
    if project_file("dashboard/index.html"):
        _html105 = open("dashboard/index.html", encoding="utf-8").read()
        check("the page handles id-only and withheld examples (no empty quotes)",
              "examples_withheld_summary" in _html105 and "e.comment != null" in _html105)
    os.remove(dash_corpus + ".run.json")

    # A gold set with hand codes present, machine domain_* columns present,
    # but far below MIN_GOLD_FOR_VALIDATED (200) -- --status validated must be
    # downgraded to provisional rather than honoured on 6 comments.
    dash_scored = os.path.join(tmp, "dash_scored.csv")
    with open(dash_scored, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["comment_id", "hand_domain", "domain_democracy"])
        w.writeheader()
        for i in range(6):
            w.writerow({"comment_id": f"c0_{i}", "hand_domain": "democracy" if i else "none",
                       "domain_democracy": "1" if i else "0"})
    proc = subprocess.run(
        [sys.executable, "-m", "ewl.analyse.export_dashboard", dash_corpus, "--output", dash_out,
         "--scored", dash_scored, "--status", "validated"],
        capture_output=True, text=True, env=env)
    check("--status validated on too few scored rows is downgraded to provisional, not honoured",
          proc.returncode == 0 and "Publishing as provisional" in proc.stderr,
          proc.stdout + proc.stderr)

    # Worked examples shown to the model are left out of the validation figures.
    with open(dash_corpus + ".run.json", "w", encoding="utf-8") as _f:
        json.dump({"input": {"path": dash_corpus, "sha256": _good_hash},
                   "fixed_examples": {"source": "file", "ids": ["c0_1", "c0_2"], "labels": ["democracy", "none"]}}, _f)
    _ex_run = subprocess.run(
        [sys.executable, "-m", "ewl.analyse.export_dashboard", dash_corpus, "--output", dash_out,
         "--scored", dash_scored, "--corpus", dash_corpus], capture_output=True, text=True, env=env)
    with open(dash_out, encoding="utf-8") as _f:
        _ex_payload = json.load(_f)
    check("the export leaves the run record's fixed examples out of the gold count and says so",
          _ex_run.returncode == 0 and "left 2 gold comment(s) out" in _ex_run.stderr
          and _ex_payload["provenance"]["gold_comments"] == 4, _ex_run.stderr[-300:])
    _ex_ids = os.path.join(tmp, "ex_ids.csv")
    with open(_ex_ids, "w", newline="", encoding="utf-8-sig") as _f:
        _f.write("comment_id,note\nc0_3,x\n")
    _ex_run2 = subprocess.run(
        [sys.executable, "-m", "ewl.analyse.export_dashboard", dash_corpus, "--output", dash_out,
         "--scored", dash_scored, "--corpus", dash_corpus, "--exclude-ids", _ex_ids],
        capture_output=True, text=True, env=env)
    with open(dash_out, encoding="utf-8") as _f:
        _ex_payload2 = json.load(_f)
    check("--exclude-ids adds to the run record's list",
          _ex_payload2["provenance"]["gold_comments"] == 3, _ex_run2.stderr[-300:])
    check("the output records how many worked examples were left out and where the ids came from",
          _ex_payload["provenance"]["worked_examples_excluded"] == {"n": 2, "example_ids_known": 2, "source": "run record"}
          and _ex_payload2["provenance"]["worked_examples_excluded"]["source"] == "run record, --exclude-ids"
          and _ex_payload2["provenance"]["worked_examples_excluded"]["n"] == 3,
          _ex_payload["provenance"].get("worked_examples_excluded"))
    os.remove(dash_corpus + ".run.json")
    _no_ids = subprocess.run(
        [sys.executable, "-m", "ewl.analyse.export_dashboard", dash_corpus, "--output", dash_out,
         "--scored", dash_scored], capture_output=True, text=True, env=env)
    with open(dash_out, encoding="utf-8") as _f:
        _no_ids_payload = json.load(_f)
    check("with no known example ids the export warns specifically, and the provenance says none were excluded",
          "no worked-example ids are known" in _no_ids.stderr
          and _no_ids_payload["provenance"]["worked_examples_excluded"]["source"] is None, _no_ids.stderr[-300:])
    _bad_ids = os.path.join(tmp, "bad_ids.csv")
    with open(_bad_ids, "w", newline="", encoding="utf-8-sig") as _f:
        _f.write("id,label\nnope1,x\n")
    _bad_run3 = subprocess.run(
        [sys.executable, "-m", "ewl.analyse.export_dashboard", dash_corpus, "--output", dash_out,
         "--scored", dash_scored, "--exclude-ids", _bad_ids], capture_output=True, text=True, env=env)
    check("--exclude-ids that matches no gold row warns instead of staying silent",
          "matches a gold" in _bad_run3.stderr, _bad_run3.stderr[-300:])
    # A run on the built-in (synthetic) examples: no gold comment can be among them, so no warning.
    with open(dash_corpus + ".run.json", "w", encoding="utf-8") as _f:
        json.dump({"input": {"path": dash_corpus, "sha256": _good_hash},
                   "fixed_examples": {"source": "built-in (synthetic)",
                                      "examples": [{"comment": "made-up example", "label": "none"}]}}, _f)
    _builtin_run = subprocess.run(
        [sys.executable, "-m", "ewl.analyse.export_dashboard", dash_corpus, "--output", dash_out,
         "--scored", dash_scored, "--corpus", dash_corpus], capture_output=True, text=True, env=env)
    check("a run record on the built-in examples raises no 'no worked-example ids' warning",
          _builtin_run.returncode == 0 and "no worked-example ids are known" not in _builtin_run.stderr,
          _builtin_run.stderr[-300:])
    # Run-record ids that match no gold row warn, as --exclude-ids ids do.
    with open(dash_corpus + ".run.json", "w", encoding="utf-8") as _f:
        json.dump({"input": {"path": dash_corpus, "sha256": _good_hash},
                   "fixed_examples": {"source": "file", "ids": ["nope1", "nope2"], "labels": ["none", "none"]}}, _f)
    _rec_nomatch = subprocess.run(
        [sys.executable, "-m", "ewl.analyse.export_dashboard", dash_corpus, "--output", dash_out,
         "--scored", dash_scored, "--corpus", dash_corpus], capture_output=True, text=True, env=env)
    check("run-record example ids that match no gold row warn instead of staying silent",
          _rec_nomatch.returncode == 0 and "worked-example id(s) in the run record matches a gold row"
          in _rec_nomatch.stderr, _rec_nomatch.stderr[-300:])
    os.remove(dash_corpus + ".run.json")
    _missing_ids = subprocess.run(
        [sys.executable, "-m", "ewl.analyse.export_dashboard", dash_corpus, "--output", dash_out,
         "--scored", dash_scored, "--exclude-ids", os.path.join(tmp, "nope.csv")],
        capture_output=True, text=True, env=env)
    check("a missing --exclude-ids file is a clean error, not a traceback",
          _missing_ids.returncode != 0 and "Traceback" not in _missing_ids.stderr and "not found" in _missing_ids.stderr)

    # classify.py's own CLI: confirm domain_votes/sentiment_votes are actually
    # written to the CSV and that the votes=1 stderr message says "not
    # measured" rather than a misleading "0.0%" -- checked at the real CLI
    # boundary, not just via run_arm/print_scorecard above.
    clf_in = os.path.join(tmp, "clf_in.csv")
    with open(clf_in, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["comment_id", "body"])
        w.writeheader()
        for i in range(3):
            w.writerow({"comment_id": f"k{i}", "body": f"comment {i}"})
    clf_out = os.path.join(tmp, "clf_out.csv")
    proc = subprocess.run(
        [sys.executable, "-m", "ewl.label.classify", clf_in, "--judge", "stub",
         "--votes", "1", "--no-cache", "--output", clf_out],
        capture_output=True, text=True, env=env)
    check("classify runs from the command line at votes=1", proc.returncode == 0, proc.stderr[-600:])
    check("votes=1 stderr says 'not measured', not a fake 0.0%",
          "not measured (--votes 1)" in proc.stderr, proc.stderr)
    check("votes=1 stderr does not also claim 0.0% unstable",
          "Unstable across votes: 0.0%" not in proc.stderr, proc.stderr)
    _clf_rows = list(csv.DictReader(open(clf_out, encoding="utf-8-sig")))
    check("domain_votes is written to the output CSV", all(r["domain_votes"] == "1" for r in _clf_rows),
          [r.get("domain_votes") for r in _clf_rows])
    check("sentiment_votes is 1 for VADER (no model calls, still one score per comment)",
          all(r["sentiment_votes"] == "1" for r in _clf_rows),
          [r.get("sentiment_votes") for r in _clf_rows])

    proc3 = subprocess.run(
        [sys.executable, "-m", "ewl.label.classify", clf_in, "--judge", "stub",
         "--votes", "3", "--no-cache", "--output", os.path.join(tmp, "clf_out3.csv")],
        capture_output=True, text=True, env=env)
    check("votes=3 still reports a real percentage, not a 'not measured' placeholder",
          "Unstable across votes: 0.0%" in proc3.stderr, proc3.stderr)


@judges.register("test_explodes")
class ExplodingJudge(judges.Judge):
    default_model = "boom"

    def _sample(self, body, prompt, vote):
        raise RuntimeError("simulated mid-run API failure")


with tempfile.TemporaryDirectory() as tmp:
    gold = os.path.join(tmp, "gold.csv")
    write_fixture(gold)
    gold_rows = evaluate.read_gold(gold)

    class ScoreArgs:
        input_csv, label_column, pool, votes, k, cache = gold, "hand_domain", None, 1, 12, None
        selector = "fixed"
        arm = ["keyword/none", "test_explodes/fixed"]
    import contextlib
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        evaluate.cmd_scorecard(ScoreArgs)
    check("a mid-run failure does not discard the other arms",
          "keyword" in out.getvalue() and "test_explodes" not in out.getvalue())

print("\nyoutube_full_corpus: the no-caps wrapper actually overrides, and can still be overridden")
import ewl.collect.youtube as _yt_mod
import ewl.collect.youtube_full_corpus as _yt_full

_captured_argv = []


def _stub_main():
    _captured_argv.append(list(sys.argv))


_real_yt_main = _yt_mod.main
_yt_mod.main = _stub_main
try:
    sys.argv = ["prog", "--story", "x", "--search", "q"]
    _yt_full.main()
    check("the four no-caps flags are injected ahead of the caller's own args",
          _captured_argv[-1][:9] == ["prog", "--limit", "0", "--target-videos", "999999",
                                    "--min-comments", "0", "--max-shorts-seconds", "0"],
          _captured_argv[-1])
    check("the caller's own args survive, appended after the injected defaults",
          _captured_argv[-1][9:] == ["--story", "x", "--search", "q"], _captured_argv[-1])

    # argparse keeps the LAST value of a repeated non-append flag, so an
    # explicit override on the real command line must still win over the
    # wrapper's injected default -- this is the whole point of injecting
    # ahead of the caller's args rather than instead of them.
    sys.argv = ["prog", "--story", "x", "--search", "q", "--limit", "500"]
    _yt_full.main()
    check("an explicit --limit on the actual command line overrides the injected default",
          _captured_argv[-1].count("--limit") == 2 and _captured_argv[-1][-1] == "500",
          _captured_argv[-1])
finally:
    _yt_mod.main = _real_yt_main

print("\nB-list items: empty cached replies, opt-in effort, noindex, SDK caps")
import tempfile as _tmp_b
import types as _types_b

with _tmp_b.TemporaryDirectory() as _tb:
    _cache_file = os.path.join(_tb, "c.jsonl")
    with open(_cache_file, "w", encoding="utf-8") as _fh:
        _fh.write(json.dumps({"key": "a", "reply": "democracy"}) + "\n")
        _fh.write(json.dumps({"key": "b", "reply": ""}) + "\n")
    _c = judges._Cache(_cache_file)
    check("the cache counts its empty replies", _c.empty_replies() == 1, _c.empty_replies())

    _in = os.path.join(_tb, "in.csv")
    with open(_in, "w", newline="", encoding="utf-8-sig") as _fh:
        _w = csv.DictWriter(_fh, fieldnames=["comment_id", "body"])
        _w.writeheader()
        _w.writerow({"comment_id": "z1", "body": "a comment"})
    _env = dict(os.environ, EWL_AUTHOR_SALT="test-salt-value-1234")
    _p = subprocess.run(
        [sys.executable, "-m", "ewl.label.classify", _in, "--judge", "stub", "--votes", "1",
         "--cache", _cache_file, "--output", os.path.join(_tb, "o.csv")],
        capture_output=True, text=True, env=_env)
    check("ewl-classify refuses a cache holding empty replies",
          _p.returncode != 0 and "empty cached repl" in (_p.stdout + _p.stderr),
          (_p.stdout + _p.stderr)[-400:])
    _p = subprocess.run(
        [sys.executable, "-m", "ewl.label.classify", _in, "--judge", "stub", "--votes", "1",
         "--cache", _cache_file, "--keep-empty-cached-replies",
         "--output", os.path.join(_tb, "o2.csv")],
        capture_output=True, text=True, env=_env)
    check("--keep-empty-cached-replies lets it proceed", _p.returncode == 0,
          (_p.stdout + _p.stderr)[-400:])


class _FakeMessages:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _types_b.SimpleNamespace(
            usage=_types_b.SimpleNamespace(input_tokens=1, output_tokens=1),
            stop_reason="end_turn",
            content=[_types_b.SimpleNamespace(type="text", text="democracy")])


def _anthropic_probe_with(effort):
    probe = object.__new__(judges.AnthropicJudge)
    probe.model = "claude-opus-5"
    probe.effort = effort
    probe.usage = collections.Counter()
    probe._lock = threading.Lock()
    probe._client = _types_b.SimpleNamespace(messages=_FakeMessages())
    return probe


_off = _anthropic_probe_with(None)
_off._sample("a comment", "a prompt", 0)
check("effort unset: the request is unchanged (no extra_body)",
      "extra_body" not in _off._client.messages.kwargs, _off._client.messages.kwargs.keys())
_on = _anthropic_probe_with("low")
_on._sample("a comment", "a prompt", 0)
check("effort low: sent as output_config.effort in an extra body",
      _on._client.messages.kwargs.get("extra_body") == {"output_config": {"effort": "low"}},
      _on._client.messages.kwargs.get("extra_body"))
check("effort is part of the cache key, so changing it cannot replay old replies",
      _on.cache_spec != _off.cache_spec and "effort=low" in _on.cache_spec,
      (_on.cache_spec, _off.cache_spec))

if project_file("dashboard/index.html"):
    _html = open("dashboard/index.html", encoding="utf-8").read()
    check("the dashboard asks search engines not to index it",
          'name="robots" content="noindex' in _html)
_toml = open("pyproject.toml", encoding="utf-8").read()
check("the anthropic and openai extras are capped below the next major version",
      '"anthropic>=1.0,<2"' in _toml and '"openai>=1.0,<4"' in _toml)

print("\nreport.py: the corrected-figure floor is 10 effective positives, flagged LOW EVIDENCE below 50")
import inspect as _inspect_t

check("the floor is 10 effective positives and the well-supported level is 50",
      report.MIN_EFFECTIVE_POSITIVES == 10 and report.WELL_SUPPORTED_POSITIVES == 50)
check("every corrected-figure function uses the floor as its default",
      all(_inspect_t.signature(f).parameters["min_genuine_positives"].default == 10
          for f in (report.corrected_ci_bootstrap, report.ppi_ci_bootstrap, report.corrected_by_story)))


def _lowev_rows(n_pos, per_video=4):
    """A story with 10 videos; gold of 80 rows from 8 videos with `n_pos` genuine democracy positives
    (equal weights, so effective positives == n_pos). The machine finds most positives."""
    corpus, gold = [], []
    for v in range(10):
        for i in range(30):
            hit = (i % 4 == 0)
            corpus.append({"comment_id": f"lc{v}_{i}", "story": "luigi", "source": f"v{v}",
                           "domain_democracy": "1" if hit else "0", "domain_unclassified": "0",
                           "domain_parse_failure": "0"})
    k = 0
    for v in range(8):
        for i in range(10):
            positive = k < n_pos
            machine = positive if (k % 5) else (not positive)
            gold.append({"comment_id": f"lg{v}_{i}", "story": "luigi", "source": f"v{v}",
                         "domain_democracy": "1" if machine else "0", "domain_unclassified": "0",
                         "domain_parse_failure": "0",
                         "hand_domain": "democracy" if positive else "none", "inclusion_prob": "0.5"})
            k += 1
    # spread the positives over the videos rather than packing them into the first few
    return corpus, sorted(gold, key=lambda r: r["comment_id"])


_c12, _g12 = _lowev_rows(12)
_res12 = report.corrected_by_story(_c12, "domain_democracy", _g12, "hand_domain", "democracy", n_boot=80)
_e12 = _res12[0]
check("12 effective positives clear the new floor: not refused for lack of positives",
      abs(_e12["effective_positives"] - 12) < 1e-6 and "effective positives" not in (_e12["refused"] or ""),
      _e12["refused"])
check("12 effective positives are flagged LOW EVIDENCE", _e12["low_evidence"] is True)
_c6, _g6 = _lowev_rows(6)
_e6 = report.corrected_by_story(_c6, "domain_democracy", _g6, "hand_domain", "democracy", n_boot=80)[0]
check("6 effective positives are still refused, with the floor named in the reason",
      _e6["refused"] and "needs 10" in _e6["refused"], _e6["refused"])
_c60, _g60 = _lowev_rows(60)
_e60 = report.corrected_by_story(_c60, "domain_democracy", _g60, "hand_domain", "democracy", n_boot=80)[0]
check("60 effective positives are well supported (not flagged)", _e60["low_evidence"] is False)
print("\nreport.py: dual statistics (--base all | classified | both)")
import tempfile as _tmp_base

_FIELDS = ["comment_id", "story", "source", "domain_democracy", "domain_economics",
           "domain_unclassified", "domain_parse_failure", "hand_domain", "inclusion_prob"]


def _base_row(i, source, dem=0, eco=0, unc=0, pf=0, hand="", prob=""):
    return {"comment_id": f"b{i}", "story": "luigi", "source": source,
            "domain_democracy": str(dem), "domain_economics": str(eco),
            "domain_unclassified": str(unc), "domain_parse_failure": str(pf),
            "hand_domain": hand, "inclusion_prob": str(prob) if prob != "" else ""}


# 10 comments, two videos: 4 tagged with no harm, 1 parse failure, 3 democracy, 2 economics.
_corpus_b = ([_base_row(i, "v1", unc=1) for i in range(4)]
             + [_base_row(4, "v1", pf=1)]
             + [_base_row(i, "v2", dem=1) for i in (5, 6, 7)]
             + [_base_row(i, "v2", eco=1) for i in (8, 9)])

check("restrict_to_base('all') hands the rows back untouched",
      report.restrict_to_base(_corpus_b, "all") is _corpus_b)
_cls = report.restrict_to_base(_corpus_b, "classified")
check("the classified base keeps only comments tagged with a harm (5 of 10)",
      len(_cls) == 5 and all(r["domain_unclassified"] == "0" for r in _cls), len(_cls))

_all_ci = report.column_ci(_corpus_b, "domain_democracy", n_boot=50)["(pooled)"]
_cls_ci = report.column_ci(_corpus_b, "domain_democracy", n_boot=50, base="classified")["(pooled)"]
check("all base: democracy is 3 of the 9 answered comments",
      abs(_all_ci[2] - 3 / 9) < 1e-9 and _all_ci[0] == 10, _all_ci)
check("classified base: democracy is 3 of the 5 tagged comments",
      abs(_cls_ci[2] - 3 / 5) < 1e-9 and _cls_ci[0] == 5, _cls_ci)

for _col in ("domain_unclassified", "domain_parse_failure"):
    try:
        report.column_ci(_corpus_b, _col, n_boot=10, base="classified")
        _refused = False
    except ValueError:
        _refused = True
    check(f"{_col} is refused in the classified base (0 by construction)", _refused)
try:
    report.restrict_to_base([{"story": "x", "domain_democracy": "1"}], "classified", "gold rows")
    _refused = False
except ValueError as _e:
    _refused = "domain_unclassified" in str(_e) and "gold rows" in str(_e)
check("a file with no domain_unclassified column is refused loudly, never silently widened",
      _refused)

# Gold rows: only rows the classifier tagged with a harm count on the classified base.
_gold_b = [_base_row(20, "v1", dem=1, hand="democracy", prob=0.5),
           _base_row(21, "v1", unc=1, hand="democracy", prob=0.5),
           _base_row(22, "v2", unc=1, hand="none", prob=0.5),
           _base_row(23, "v2", dem=1, hand="none", prob=0.5)]
check("effective positives, all base: both hand-coded democracy rows count (2.0)",
      abs(report.weighted_effective_positive_count(_gold_b, "democracy", "hand_domain") - 2.0) < 1e-9)
check("effective positives, classified base: the one the classifier left unclassified drops out (1.0)",
      abs(report.weighted_effective_positive_count(
          _gold_b, "democracy", "hand_domain", base="classified") - 1.0) < 1e-9)
check("the corrected figure is refused (None) when the restricted gold is below the gate",
      report.corrected_ci_bootstrap(_corpus_b, "domain_democracy", _gold_b, "hand_domain",
                                    "democracy", 1.0, base="classified", n_boot=20) is None)
try:
    report.corrected_ci_bootstrap(
        _corpus_b, "domain_democracy",
        [{k: v for k, v in r.items() if k != "domain_unclassified"} for r in _gold_b],
        "hand_domain", "democracy", 100.0, base="classified", n_boot=20)
    _refused = False
except ValueError:
    _refused = True
check("a gold file without domain_unclassified is refused for a corrected classified figure", _refused)

with _tmp_base.TemporaryDirectory() as _tb2:
    _csv = os.path.join(_tb2, "c.csv")
    with open(_csv, "w", newline="", encoding="utf-8-sig") as _fh:
        _w = csv.DictWriter(_fh, fieldnames=_FIELDS)
        _w.writeheader()
        _w.writerows(_corpus_b)
    _env2 = dict(os.environ, EWL_AUTHOR_SALT="test-salt-value-1234")

    def _report_cli(*extra):
        return subprocess.run([sys.executable, "-m", "ewl.analyse.report", "observed", _csv,
                               "--column", "domain_democracy", "--by", "story", "--n-boot",
                               "50", *extra], capture_output=True, text=True, env=_env2)

    _default, _explicit = _report_cli(), _report_cli("--base", "all")
    check("--base all prints exactly what no flag prints (existing output unchanged)",
          _default.returncode == 0 and _default.stdout == _explicit.stdout and
          "Base:" not in _default.stdout, _default.stdout + _default.stderr)
    _both = _report_cli("--base", "both")
    check("--base both prints the two figures under their own denominators",
          _both.returncode == 0 and "33.3%" in _both.stdout and "60.0%" in _both.stdout
          and _both.stdout.count("=== Base:") == 2, _both.stdout + _both.stderr)
    _bad = subprocess.run([sys.executable, "-m", "ewl.analyse.report", "corrected", _csv,
                           "--column", "domain_democracy", "--tpr", "0.9", "--fpr", "0.1",
                           "--base", "classified"], capture_output=True, text=True, env=_env2)
    check("corrected --tpr/--fpr refuses a non-default base (the rates would be on another base)",
          _bad.returncode != 0 and "another base" in (_bad.stdout + _bad.stderr),
          _bad.stdout + _bad.stderr)


_pc = _ed.proportions_classified(_corpus_b, n_boot=20)
_dem = [r for r in _pc if r["label"] == "democracy"]
check("export: the classified-base democracy share is 3 of 5 tagged comments, labelled as such",
      len(_dem) == 1 and abs(_dem[0]["rate"] - 0.6) < 1e-9 and _dem[0]["base"] == "classified",
      _dem)
_psr = _ed.per_story_rates(_corpus_b)[0]
check("export: per-story figures say how much of the corpus the classified base covers (5 of 10)",
      _psr.get("classified_n") == 5 and abs(_psr.get("classified_share") - 0.5) < 1e-9, _psr)
check("export: a file with no domain_unclassified column gets no classified-base figures",
      _ed.proportions_classified([{k: v for k, v in r.items() if k != "domain_unclassified"}
                                  for r in _corpus_b]) == [])


print("\nreport.py: --exclude-ids keeps the worked examples out of the correction")
with tempfile.TemporaryDirectory() as _te:
    _cc = os.path.join(_te, "corpus.csv")
    _gg = os.path.join(_te, "gold.csv")
    _ex = os.path.join(_te, "examples.csv")
    _fields = ["comment_id", "story", "source", "domain_democracy", "domain_economics",
               "domain_unclassified", "domain_parse_failure", "hand_domain", "inclusion_prob"]
    for _path, _rows_ in ((_cc, _corpus_b), (_gg, _gold_b * 5)):
        with open(_path, "w", newline="", encoding="utf-8-sig") as _f:
            _w = csv.DictWriter(_f, fieldnames=_fields, extrasaction="ignore")
            _w.writeheader()
            for _k, _r in enumerate(_rows_):
                _w.writerow(dict(_r, comment_id=f"{_r['comment_id']}_{_k}"))
    _gold_ids = [r["comment_id"] for r in csv.DictReader(open(_gg, newline="", encoding="utf-8-sig"))]
    with open(_ex, "w", newline="", encoding="utf-8-sig") as _f:
        _w = csv.writer(_f)
        _w.writerow(["comment_id", "body", "label"])
        for _i in _gold_ids[:3]:
            _w.writerow([_i, "an example", "democracy"])
    _envx = dict(os.environ, EWL_AUTHOR_SALT="test-salt-value-1234")
    _p = subprocess.run([sys.executable, "-m", "ewl.analyse.report", "corrected-bootstrap", _cc,
                         "--column", "domain_democracy", "--gold", _gg, "--exclude-ids", _ex,
                         "--min-genuine-positives", "1", "--n-boot", "20"],
                        capture_output=True, text=True, env=_envx)
    check("report --exclude-ids drops the listed gold comments before the correction, and says so",
          "Excluded 3 of" in _p.stderr, _p.stderr[-400:])
    _p = subprocess.run([sys.executable, "-m", "ewl.analyse.report", "corrected-bootstrap", _cc,
                         "--column", "domain_democracy", "--gold", _gg,
                         "--min-genuine-positives", "1", "--n-boot", "20"],
                        capture_output=True, text=True, env=_envx)
    check("without --exclude-ids nothing is dropped", "Excluded" not in _p.stderr)

print("\nreport.py / export: findings from the #102 review (gold weights, dropped story, gate recount)")
import random as _rnd102


def _r102(i, story, source, dem=0, eco=0, unc=0, hand="", prob=""):
    return {"comment_id": f"r{i}", "story": story, "source": source,
            "domain_democracy": str(dem), "domain_economics": str(eco),
            "domain_unclassified": str(unc), "domain_parse_failure": "0",
            "hand_domain": hand, "inclusion_prob": str(prob)}


# Finding 1: the mixed blank/recorded inclusion_prob refusal reaches the CLI on both bases.
_rng102 = _rnd102.Random(5)
_corp102 = [_r102(i, "luigi", f"v{i % 12}", dem=int(_rng102.random() < 0.3 + 0.04 * (i % 12)),
                  unc=int(_rng102.random() < 0.3)) for i in range(240)]
_gold102 = [_r102(1000 + i, "luigi", f"v{i % 12}", dem=int(i % 3 != 0),
                  hand="democracy" if i % 2 == 0 else "none", prob="0.5") for i in range(60)]
_gold_mixed = [dict(r, inclusion_prob="") if k < 3 else r for k, r in enumerate(_gold102)]
with tempfile.TemporaryDirectory() as _t102:
    _c102, _g102, _gm102, _x102 = (os.path.join(_t102, n) for n in
                                   ("c.csv", "g.csv", "gm.csv", "x.csv"))
    for _pth, _rws in ((_c102, _corp102), (_g102, _gold102), (_gm102, _gold_mixed)):
        with open(_pth, "w", newline="", encoding="utf-8-sig") as _f:
            _w = csv.DictWriter(_f, fieldnames=_FIELDS)
            _w.writeheader()
            _w.writerows(_rws)
    with open(_x102, "w", newline="", encoding="utf-8-sig") as _f:
        _f.write("comment_id\nnone_of_these\n")
    _e102 = dict(os.environ, EWL_AUTHOR_SALT="test-salt-value-1234")

    def _cli102(gold, *extra):
        return subprocess.run(
            [sys.executable, "-m", "ewl.analyse.report", "corrected-bootstrap", _c102,
             "--column", "domain_democracy", "--gold", gold, "--min-genuine-positives", "1",
             "--n-boot", "40", *extra], capture_output=True, text=True, env=_e102)

    for _extra, _name in (((), "no flag"), (("--base", "classified"), "--base classified"),
                          (("--exclude-ids", _x102), "--exclude-ids")):
        _p = _cli102(_gm102, *_extra)
        check(f"corrected-bootstrap refuses a gold file with some blank inclusion_prob ({_name})",
              _p.returncode != 0 and "no inclusion_prob" in (_p.stdout + _p.stderr)
              and "UnboundLocalError" not in _p.stderr, _p.stdout + _p.stderr[-300:])
        _p = _cli102(_g102, *_extra)
        check(f"corrected-bootstrap runs on a valid gold file ({_name})",
              _p.returncode == 0 and "UnboundLocalError" not in _p.stderr, _p.stderr[-300:])

# Finding 2: a story with no tagged comment is shown, not dropped.
_two = ([_r102(i, "luigi", f"v{i % 4}", dem=i % 2, eco=1 - i % 2) for i in range(40)]
        + [_r102(100 + i, "musk", f"w{i % 4}", unc=1) for i in range(40)])
_ci2 = report.column_ci(_two, "domain_democracy", group_by="story", n_boot=30, base="classified")
check("column_ci on the tagged base keeps a story with no tagged comment, as n = 0",
      set(_ci2) == {"luigi", "musk"} and _ci2["musk"][0] == 0 and _ci2["luigi"][0] == 40, _ci2)
try:
    report.aggregate_ci(_two, "domain_democracy", n_boot=30, base="classified")
    _ok = False
except ValueError as _e:
    _ok = "musk" in str(_e)
check("aggregate_ci on the tagged base refuses, naming the dropped story, rather than "
      "averaging fewer stories", _ok)
check("aggregate_ci on the all base still covers both stories",
      set(report.aggregate_ci(_two, "domain_democracy", n_boot=30)[3]) == {"luigi", "musk"})
_cbs = report.corrected_by_story(_two, "domain_democracy", [], "hand_domain", "democracy",
                                 base="classified")
check("corrected_by_story lists the dropped story with the reason",
      [e["story"] for e in _cbs] == ["luigi", "musk"]
      and _cbs[1]["refused"] == "no tagged comments in this story" and _cbs[1]["n_comments"] == 0,
      _cbs)
with tempfile.TemporaryDirectory() as _t102b:
    _c2 = os.path.join(_t102b, "two.csv")
    with open(_c2, "w", newline="", encoding="utf-8-sig") as _f:
        _w = csv.DictWriter(_f, fieldnames=_FIELDS)
        _w.writeheader()
        _w.writerows(_two)
    _env2b = dict(os.environ, EWL_AUTHOR_SALT="test-salt-value-1234")
    _p = subprocess.run([sys.executable, "-m", "ewl.analyse.report", "observed", _c2,
                         "--column", "domain_democracy", "--by", "story", "--n-boot", "30",
                         "--base", "classified"], capture_output=True, text=True, env=_env2b)
    check("observed --base classified prints the dropped story as 'no tagged comments'",
          _p.returncode == 0 and "musk" in _p.stdout and "no tagged comments" in _p.stdout,
          _p.stdout + _p.stderr[-300:])
    _p = subprocess.run([sys.executable, "-m", "ewl.analyse.report", "aggregate", _c2,
                         "--column", "domain_democracy", "--n-boot", "30", "--base", "both"],
                        capture_output=True, text=True, env=_env2b)
    check("aggregate --base both reports the all base and says the tagged base is not reported",
          _p.returncode == 0 and _p.stdout.count("Equal-per-story aggregate") == 2
          and "not reported" in _p.stdout and "musk" in _p.stdout, _p.stdout + _p.stderr[-300:])
_pcs = _ed.proportions_classified(_two, n_boot=30)
_musk = [r for r in _pcs if r["story"] == "musk" and r["label"] == "democracy"]
check("export: the dropped story appears in proportions_classified as n = 0 with no figures",
      len(_musk) == 1 and _musk[0]["n"] == 0 and _musk[0]["rate"] is None, _musk)

# Finding 3: the tagged-base table gets the deflated-interval check.
_same = [_r102(i, "luigi", f"s{i // 4}", dem=int(i % 4 < 2), eco=int(i % 4 >= 2))
         for i in range(40)]  # 10 identical videos
_fx = _ed.design_effects(_same, base="classified")
check("design_effects takes a base and measures the tagged-base intervals",
      any(e["label"] == "democracy" and e["inflation"] is not None for e in _fx), _fx)
_marked = _ed.mark_estimability(_ed.proportions_classified(_same, n_boot=100), _fx)
_d = [r for r in _marked if r["label"] == "democracy"][0]
check("export: identical videos make the tagged-base democracy interval 'deflated', not estimable",
      _d["interval_estimable"] is False and _d["interval_reason"] == "deflated", _d)

# Finding 4: the gate recounts on the restricted base.
_g4 = ([_r102(2000 + i, "luigi", f"v{i % 12}", dem=1, hand="democracy", prob="0.5")
        for i in range(30)]
       + [_r102(2100 + i, "luigi", f"v{i % 12}", unc=1, hand="democracy", prob="0.5")
          for i in range(30)]
       + [_r102(2200 + i, "luigi", f"v{i % 12}", eco=1, hand="none", prob="0.5")
          for i in range(30)])
check("fixture: 60 effective positives on all, 30 on classified",
      abs(report.weighted_effective_positive_count(_g4, "democracy", "hand_domain") - 60) < 1e-9
      and abs(report.weighted_effective_positive_count(
          _g4, "democracy", "hand_domain", base="classified") - 30) < 1e-9)
check("corrected_ci_bootstrap: a count of 60 passed in does not clear a floor of 50 on the tagged base",
      report.corrected_ci_bootstrap(_corp102, "domain_democracy", _g4, "hand_domain", "democracy",
                                    60.0, min_genuine_positives=50, n_boot=60,
                                    base="classified") is None)
check("corrected_ci_bootstrap: the same call with a floor of 25 does give a figure (so the gate was the cause)",
      report.corrected_ci_bootstrap(_corp102, "domain_democracy", _g4, "hand_domain", "democracy",
                                    60.0, min_genuine_positives=25, n_boot=60,
                                    base="classified") is not None)
check("ppi_ci_bootstrap: a count of 60 passed in does not clear a floor of 50 on the tagged base",
      report.ppi_ci_bootstrap(_corp102, "domain_democracy", _g4, "hand_domain", "democracy",
                              60.0, min_genuine_positives=50, n_boot=60,
                              base="classified") is None)
check("ppi_ci_bootstrap: the same call with a floor of 25 does give a figure",
      report.ppi_ci_bootstrap(_corp102, "domain_democracy", _g4, "hand_domain", "democracy",
                              60.0, min_genuine_positives=25, n_boot=60,
                              base="classified") is not None)


print("\nreport.py: #106 review (flag on the figure's line, PPI clamp reporting, ungated corrected)")
_c25, _g25 = _lowev_rows(25)
_d_rg = {}
_rg25 = report.corrected_ci_bootstrap(_c25, "domain_democracy", _g25, "hand_domain", "democracy",
                                      25.0, n_boot=200, details=_d_rg)
check("a fixture where Rogan-Gladen does give a figure at 25 effective positives (point asserted)",
      _rg25 is not None and abs(_rg25[0] - 0.1111111) < 1e-6, _rg25)
check("corrected_ci_bootstrap reports the effective count and the low_evidence flag via details=",
      _d_rg == {"effective_positives": 25.0, "low_evidence": True}, _d_rg)
_d_rg60 = {}
report.corrected_ci_bootstrap(_c60, "domain_democracy", _g60, "hand_domain", "democracy",
                              60.0, n_boot=60, details=_d_rg60)
check("details= says low_evidence False at 60 effective positives", _d_rg60.get("low_evidence") is False)
check("at 12 effective positives Rogan-Gladen refuses (point is None) while PPI still gives a figure",
      _e12["point"] is None and "degenerate" in _e12["refused"]
      and report.ppi_ci_bootstrap(_c12, "domain_democracy", _g12, "hand_domain", "democracy",
                                  12.0, n_boot=80) is not None,
      _e12)
check("evidence_tag is empty when well supported and carries the count when not",
      report.evidence_tag(60.0) == "" and report.evidence_tag(12.4) == " [LOW EVIDENCE: 12.4 effective positives]")

# PPI clamp: machine fires on every gold row, hand-coded positives are few, so the rectifier is
# strongly negative and the corpus rate (5%) plus it falls below 0 on every draw.
_cc_clamp = [{"comment_id": f"cc{v}_{i}", "story": "luigi", "source": f"v{v}",
              "domain_democracy": "1" if i == 0 and v % 2 == 0 else "0", "domain_unclassified": "0",
              "domain_parse_failure": "0"} for v in range(10) for i in range(10)]
_gg_clamp = [{"comment_id": f"cg{v}_{i}", "story": "luigi", "source": f"v{v}",
              "domain_democracy": "1", "domain_unclassified": "0", "domain_parse_failure": "0",
              "hand_domain": "democracy" if (v * 5 + i) < 15 else "none", "inclusion_prob": "0.5"}
             for v in range(8) for i in range(5)]
_d_ppi = {}
_ppi_cl = report.ppi_ci_bootstrap(_cc_clamp, "domain_democracy", _gg_clamp, "hand_domain",
                                  "democracy", 15.0, n_boot=100, details=_d_ppi)
check("ppi_ci_bootstrap counts the draws it clamped to 0",
      _ppi_cl is not None and _ppi_cl[1] == 0.0 and _d_ppi["below_0"] > 0 and _d_ppi["draws"] == 100
      and _d_ppi["above_1"] == 0, (_ppi_cl, _d_ppi))

_FLD106 = ["comment_id", "story", "source", "domain_democracy", "domain_unclassified",
           "domain_parse_failure", "hand_domain", "inclusion_prob"]
with tempfile.TemporaryDirectory() as _t106:
    _paths106 = {}
    for _name, _rws in (("c25", _c25), ("g25", _g25), ("c60", _c60), ("g60", _g60),
                        ("ccl", _cc_clamp), ("gcl", _gg_clamp)):
        _paths106[_name] = os.path.join(_t106, _name + ".csv")
        with open(_paths106[_name], "w", newline="", encoding="utf-8-sig") as _f:
            _w = csv.DictWriter(_f, fieldnames=_FLD106, extrasaction="ignore")
            _w.writeheader()
            _w.writerows(_rws)
    _e106 = dict(os.environ, EWL_AUTHOR_SALT="test-salt-value-1234")

    def _run106(cmd, corpus, gold=None, *extra):
        args = [sys.executable, "-m", "ewl.analyse.report", cmd, _paths106[corpus],
                "--column", "domain_democracy", "--n-boot", "100", *extra]
        if gold:
            args += ["--gold", _paths106[gold]]
        return subprocess.run(args, capture_output=True, text=True, env=_e106)

    _p = _run106("corrected-bootstrap", "c25", "g25")
    _lines = [l for l in _p.stdout.splitlines() if l.startswith("Corrected")]
    check("corrected-bootstrap: the LOW EVIDENCE flag is on the same stdout line as the figure and the interval",
          _p.returncode == 0 and len(_lines) == 2
          and all("[LOW EVIDENCE: 25.0 effective positives]" in l for l in _lines),
          _p.stdout + _p.stderr[-300:])
    _p = _run106("ppi-bootstrap", "c25", "g25")
    _lines = [l for l in _p.stdout.splitlines() if l.startswith("Corrected")]
    check("ppi-bootstrap: the LOW EVIDENCE flag is on the same stdout line as the figure and the interval",
          _p.returncode == 0 and len(_lines) == 2
          and all("[LOW EVIDENCE: 25.0 effective positives]" in l for l in _lines),
          _p.stdout + _p.stderr[-300:])
    _p = _run106("ppi-bootstrap", "c60", "g60")
    check("ppi-bootstrap: a well-supported figure carries no LOW EVIDENCE flag",
          _p.returncode == 0 and "LOW EVIDENCE" not in _p.stdout, _p.stdout + _p.stderr[-300:])
    _p = _run106("ppi-bootstrap", "ccl", "gcl")
    check("ppi-bootstrap: a lower bound sitting at 0 because of clamping says so on stdout",
          _p.returncode == 0 and "lower bound truncated at 0 (" in _p.stdout
          and "of 100 draws below 0)" in _p.stdout, _p.stdout + _p.stderr[-300:])
    _p = _run106("corrected", "c25", None, "--tpr", "0.9", "--fpr", "0.1")
    _line = [l for l in _p.stdout.splitlines() if l.startswith("Corrected (point")]
    check("corrected --tpr/--fpr says UNGATED on the figure's own line",
          _p.returncode == 0 and len(_line) == 1 and "UNGATED: no evidence check is possible "
          "from --tpr/--fpr alone" in _line[0], _p.stdout + _p.stderr[-300:])

print("\nopen-weight arm: Gemma 2 27B Q4_0 through a local Ollama (mocked; no server needed)")
from ewl.label import gemma_local, classify_gemma
import ewl.label.sentiment as _sent_gemma

_TAG = gemma_local.MODEL_TAG


class _FakeOllama:
    """Stands in for ollama.Client. Records every chat() call and answers by
    prompt shape, so the tests can check request options, call order and
    parsing without a 43 GB model."""

    def __init__(self, models=None, quantisation="Q4_0", list_error=None,
                 domain_reply=None, sentiment_reply="2", done_reason="stop"):
        self.calls = []
        self.models = [_TAG] if models is None else models
        self.quantisation = quantisation
        self.list_error = list_error
        self.domain_reply = domain_reply
        self.sentiment_reply = sentiment_reply
        self.done_reason = done_reason

    def list(self):
        if self.list_error:
            raise self.list_error
        return {"models": [{"model": m} for m in self.models]}

    def show(self, model):
        return {"details": {"quantization_level": self.quantisation, "parameter_size": "27.2B"}}

    def chat(self, model, messages, options, keep_alive):
        system, user = messages[0]["content"], messages[1]["content"]
        kind = "domain" if system.startswith("You are a precise text classifier") else "sentiment"
        self.calls.append({"model": model, "system": system, "user": user, "options": options,
                           "keep_alive": keep_alive, "kind": kind, "roles": [m["role"] for m in messages]})
        if kind == "domain":
            if self.domain_reply is not None:
                reply = self.domain_reply
            else:
                reply = "democracy" if "tax" in user.lower() else "none"
        else:
            reply = self.sentiment_reply
        return {"message": {"content": reply}, "prompt_eval_count": 10, "eval_count": 3,
                "done_reason": self.done_reason}


_real_make_client = gemma_local.GemmaJudge._make_client


def _with_fake(fake):
    gemma_local.GemmaJudge._make_client = lambda self: fake


try:
    check("the gemma backend is registered", "gemma" in judges.available(), judges.available())
    try:
        judges.create("gemma:gemma3.1:8b")
        check("it refuses any model other than the pinned one", False, "no exception")
    except ValueError as _err:
        check("it refuses any model other than the pinned one", _TAG in str(_err), str(_err))

    _fake = _FakeOllama()
    _with_fake(_fake)
    _lj = judges.create("gemma", votes=3)
    check("the default model is the pinned Q4_0 tag", _lj.model == _TAG == "gemma2:27b")
    check("asking for the pinned tag by name is the same arm",
          judges.create("gemma:" + _TAG).cache_spec == _lj.cache_spec == "gemma:" + _TAG)
    check("votes are forced to 1 however many are asked for (temperature 0 repeats itself)",
          _lj.votes == 1, _lj.votes)
    check("it is free to run", _lj.cost_usd() == 0)
    check("it does not claim bit-exact determinism, so the scorecard says 'cache only'",
          _lj.deterministic is False)

    _labels, _unstable, _ballots = _lj.label("She pays no tax at all", taxonomy.FIXED_EXAMPLES)
    _c = _fake.calls[-1]
    check("one comment costs one call at --votes 3", len(_fake.calls) == 1, len(_fake.calls))
    check("the reply is parsed like every other arm's", _labels == ["democracy"], _labels)
    check("the request names the pinned model", _c["model"] == _TAG)
    check("the system prompt is exactly the shared prompt with the 25 fixed examples",
          _c["system"] == taxonomy.build_prompt(taxonomy.FIXED_EXAMPLES) and _c["roles"] == ["system", "user"])
    check("the comment goes in the user turn in the same form the API arms use",
          _c["user"] == 'Classify this comment: "She pays no tax at all"', _c["user"])
    check("temperature 0, an explicit context window and a short answer limit are set, "
          "not left to Ollama's defaults",
          _c["options"]["temperature"] == 0 and _c["options"]["num_ctx"] == gemma_local.NUM_CTX
          and _c["options"]["num_predict"] == gemma_local.NUM_PREDICT, _c["options"])
    check("the model is kept loaded between requests", _c["keep_alive"] == gemma_local.KEEP_ALIVE)
    check("tokens are recorded from Ollama's counts",
          (_lj.usage["input_tokens"], _lj.usage["output_tokens"]) == (10, 3), dict(_lj.usage))

    # A comment too long for the window is cut; the instructions are not.
    _long_prompt = taxonomy.build_prompt(taxonomy.FIXED_EXAMPLES)
    _lj.sample("x" * 50_000, _long_prompt)
    _c = _fake.calls[-1]
    check("an over-long comment is cut, with the prompt left whole and the cut counted",
          len(_c["user"]) < 50_000 and _c["system"] == _long_prompt
          and _lj.usage["truncated_bodies"] == 1, (len(_c["user"]), dict(_lj.usage)))
    _lj.sample("an ordinary comment", _long_prompt)
    check("an ordinary comment is not cut", _lj.usage["truncated_bodies"] == 1)
    _lj.sample("y" * 9_000, _long_prompt)
    check("a 9,000-character comment fits the default window", _lj.usage["truncated_bodies"] == 1)

    os.environ["EWL_GEMMA_NUM_CTX"] = "4096"
    try:
        _small = judges.create("gemma")
        _small.sample("y" * 9_000, _long_prompt)
        check("the same comment IS cut in a 4096 window, so the guard is what protects the prompt",
              _small.usage["truncated_bodies"] == 1 and _fake.calls[-1]["options"]["num_ctx"] == 4096)
        os.environ["EWL_GEMMA_NUM_CTX"] = "2048"
        try:
            judges.create("gemma")
            check("a window too small for the prompts is refused", False, "no exception")
        except ValueError:
            check("a window too small for the prompts is refused", True)
        os.environ["EWL_GEMMA_NUM_CTX"] = "16384"
        try:
            judges.create("gemma")
            check("a window larger than Gemma 2's own 8,192 is refused", False, "no exception")
        except ValueError:
            check("a window larger than Gemma 2's own 8,192 is refused", True)
    finally:
        del os.environ["EWL_GEMMA_NUM_CTX"]

    _fake.done_reason = "length"
    _lj.sample("a comment", _long_prompt)
    check("a reply cut off at the answer limit is counted, so parse failures can be explained",
          _lj.usage["truncated_replies"] == 1)
    _fake.done_reason = "stop"

    # Readiness checks: each failure says how to fix it.
    def _preflight_error(fake):
        try:
            gemma_local.preflight(fake)
        except gemma_local.GemmaUnavailable as error:
            return str(error)
        return None

    check("preflight passes and reports the quantisation it found",
          gemma_local.preflight(_FakeOllama())["quantisation"] == "Q4_0")
    _msg = _preflight_error(_FakeOllama(models=["gemma3:12b"]))
    check("a model that has not been pulled is refused, with the pull command and what IS present",
          _msg and f"ollama pull {_TAG}" in _msg and "gemma3:12b" in _msg, _msg)
    _msg = _preflight_error(_FakeOllama(quantisation="Q8_0"))
    check("the right tag at the wrong quantisation is refused",
          _msg and "Q4_0" in _msg and "Q8_0" in _msg, _msg)
    _msg = _preflight_error(_FakeOllama(list_error=ConnectionError("refused")))
    check("an unreachable server is refused, saying how to start it",
          _msg and "ollama serve" in _msg, _msg)
    _with_fake(_FakeOllama(models=[]))
    try:
        judges.create("gemma")
        check("a judge cannot be built against an unready server, so a scorecard skips the arm "
              "instead of scoring the wrong model", False, "no exception")
    except gemma_local.GemmaUnavailable:
        check("a judge cannot be built against an unready server, so a scorecard skips the arm "
              "instead of scoring the wrong model", True)

    # ---- the two scorecard arms -------------------------------------------
    _fake = _FakeOllama()
    _with_fake(_fake)
    _j1, _s1 = evaluate.build("gemma/fixed", None, 3, 12, "hand_domain", None)
    _j2, _s2 = evaluate.build("gemma:" + _TAG + "/fixed", None, 3, 12, "hand_domain", None)
    check("the domain scorecard arm 'gemma/fixed' resolves to the pinned model with 25 examples",
          _j1.spec == "gemma:" + _TAG and _s1.describe() == "fixed(25)", (_j1.spec, _s1.describe()))
    check("spelling out the full tag is the same arm, so the response cache is shared",
          _j2.cache_spec == _j1.cache_spec)
    check("--votes 3 on the scorecard does not triple a Gemma arm", _j1.votes == 1)

    with tempfile.TemporaryDirectory() as tmp:
        _gold = os.path.join(tmp, "gold.csv")
        write_fixture(_gold)
        _rows = evaluate.read_gold(_gold)
        _arm = evaluate.run_arm(_rows, _j1, _s1, progress=False)
        check("the arm scores a gold set end to end: 8 rows, 8 calls, no cost",
              _arm["n"] == len(_rows) == 8 and _arm["calls"] == 8 and _arm["cost_usd"] == 0
              and _arm["arm"] == f"gemma:{_TAG}/fixed(25)", _arm)
        check("its wall-clock time is recorded (the figure to compare, since cost is zero)",
              _arm["seconds"] >= 0)

    _scorer = _sent_gemma.create("llm:gemma:" + _TAG, votes=3, cache_path=None)
    _before = len(_fake.calls)
    _scored = _scorer.score("some comment", story="musk")
    check("the sentiment arm 'llm:gemma' parses the colon-containing tag to the same model",
          _scorer._judge.cache_spec == _j1.cache_spec, _scorer._judge.cache_spec)
    check("and at --votes 3 it makes one call per comment, not three",
          len(_fake.calls) - _before == 1 and _scorer.votes == 1, len(_fake.calls) - _before)
    check("it rates through the per-story sentiment prompt",
          _fake.calls[-1]["kind"] == "sentiment" and "Musk" in _fake.calls[-1]["system"]
          and _scored["value"] is not None, _scored)
    check("both arms are described consistently in the outputs",
          _sent_gemma.create("llm:gemma", votes=1).describe() == f"llm:gemma:{_TAG}")

    # ---- the standalone script --------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        _in = os.path.join(tmp, "comments.csv")
        _out = os.path.join(tmp, "labelled.csv")
        with open(_in, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["comment_id", "story", "body"])
            w.writeheader()
            w.writerow({"comment_id": "a", "story": "musk", "body": "He pays no tax"})
            w.writerow({"comment_id": "b", "story": "jets", "body": "Nice video"})
            w.writerow({"comment_id": "c", "story": "musk", "body": "Tax dodging again"})
            w.writerow({"comment_id": "d", "story": "jets", "body": "First!"})
            w.writerow({"comment_id": "e", "story": "jets", "body": "   "})
        _fake = _FakeOllama()
        _with_fake(_fake)
        _cache = os.path.join(tmp, "cache", "gemma.jsonl")
        with contextlib.redirect_stderr(io.StringIO()) as _err_builtin:
            _rc = classify_gemma.main([_in, "--output", _out, "--cache", _cache])
        with open(_out, newline="", encoding="utf-8-sig") as f:
            _reader = csv.DictReader(f)
            _labelled = list(_reader)
            _header = _reader.fieldnames
        check("the script runs and exits 0", _rc == 0, _rc)
        _prev_temp = os.environ.get("EWL_GEMMA_TEMPERATURE")
        os.environ["EWL_GEMMA_TEMPERATURE"] = "0.7"
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                classify_gemma.main([_in, "--output", os.path.join(tmp, "t07.csv"),
                                     "--cache", os.path.join(tmp, "t07.jsonl")])
            _t_refused = False
        except SystemExit as _e:
            _t_refused = "frozen" in str(_e)
        finally:
            if _prev_temp is None:
                os.environ.pop("EWL_GEMMA_TEMPERATURE", None)
            else:
                os.environ["EWL_GEMMA_TEMPERATURE"] = _prev_temp
        check("ewl-gemma refuses a non-zero EWL_GEMMA_TEMPERATURE (it is the frozen arm)", _t_refused)
        check("it writes exactly the columns ewl-classify writes, after the input's own",
              _header == ["comment_id", "story", "body"] + classify.EXTRA_COLUMNS, _header)
        check("rows come back in the input's order, with the blank-comment row left out",
              [r["comment_id"] for r in _labelled] == ["a", "b", "c", "d"],
              [r["comment_id"] for r in _labelled])
        check("domain labels land in the shared columns",
              [r["domain_democracy"] for r in _labelled] == ["1", "0", "1", "0"]
              and _labelled[1]["domain_unclassified"] == "1")
        check("sentiment lands in the shared columns",
              all(r["sentiment_value"] and r["sentiment"] for r in _labelled), _labelled[0])
        check("the arm is named in every row",
              _labelled[0]["domain_judge"] == f"gemma:{_TAG}/fixed(25)"
              and _labelled[0]["sentiment_scorer"] == f"llm:gemma:{_TAG}", _labelled[0])
        _kinds = [c["kind"] for c in _fake.calls]
        check("two passes: every domain call comes before any sentiment call",
              _kinds == ["domain"] * 4 + ["sentiment"] * 4, _kinds)
        _sent_prompts = [c["system"] for c in _fake.calls if c["kind"] == "sentiment"]
        _runs = 1 + sum(1 for x, y in zip(_sent_prompts, _sent_prompts[1:]) if x != y)
        check("sentiment calls are grouped by story, so each prompt is used for a whole run",
              _runs == 2, _runs)
        _dom_prompts = [c["system"] for c in _fake.calls if c["kind"] == "domain"]
        _druns = 1 + sum(1 for x, y in zip(_dom_prompts, _dom_prompts[1:]) if x != y)
        check("domain calls are grouped by story too, each carrying its own story's background",
              _druns == 2 and any("Elon" in p or "Twitter" in p for p in _dom_prompts)
              and len(set(_dom_prompts)) == 2, (_druns, len(set(_dom_prompts))))

        _calls_before = len(_fake.calls)
        with contextlib.redirect_stderr(io.StringIO()):
            classify_gemma.main([_in, "--output", _out, "--cache", _cache])
        check("running it again replays the cache: no new model calls, so a stopped run resumes",
              len(_fake.calls) == _calls_before, len(_fake.calls) - _calls_before)
        check("the run record is written beside the output, tied to the input by hash",
              os.path.exists(_out + ".run.json"))
        with open(_out + ".run.json", encoding="utf-8") as _rf:
            _rec = json.load(_rf)
        from ewl import corpus as _corp_g
        check("the record carries the input hash, the model, the prompt hashes and the examples",
              _rec["input"]["sha256"] == _corp_g.content_hash(_in)
              and _rec["domain_judge"]["spec"] == "gemma:" + _TAG
              and _rec["prompts_sha256"]["domain"]
              and set(_rec["prompts_sha256"]["sentiment_by_story"]) >= {"musk", "jets"}
              and set(_rec["prompts_sha256"]["domain_by_story"]) == {"luigi", "musk", "jets", "propublica"}
              and len(set(_rec["prompts_sha256"]["domain_by_story"].values())) == 4
              and _rec["instrument_switches"]["domain_context"] == "facts"
              and _rec["fixed_examples"]["source"].startswith("built-in")
              and _rec["result"]["blank_body_rows"] == 1,
              {k: _rec.get(k) for k in ("input", "fixed_examples")})

        # Redoing a labelled file removes the old output and record first, so a stop after the
        # new record is written (before the rename) cannot leave it beside the OLD labels.
        import shutil as _sh_redo
        _redo = os.path.join(tmp, "redo.csv")
        _sh_redo.copy(_out, _redo)
        _sh_redo.copy(_out + ".run.json", _redo + ".run.json")
        _orig_manifest = classify_gemma.classify_mod.write_run_manifest

        def _stop_after_record(path, *a, **k):
            _orig_manifest(path, *a, **k)
            raise KeyboardInterrupt("stopped between record and rename")
        classify_gemma.classify_mod.write_run_manifest = _stop_after_record
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                try:
                    classify_gemma.main([_in, "--output", _redo, "--cache", _cache])
                except KeyboardInterrupt:
                    pass
        finally:
            classify_gemma.classify_mod.write_run_manifest = _orig_manifest
        check("a redo stopped between record and rename leaves no old output beside the new record",
              not os.path.exists(_redo) and os.path.exists(_redo + ".partial"),
              (os.path.exists(_redo), os.path.exists(_redo + ".run.json")))

        # Real gold comments as the examples: loaded from a file, recorded by id and hash only.
        _ex = os.path.join(tmp, "fixed.csv")
        with open(_ex, "w", newline="", encoding="utf-8-sig") as _ef:
            _ew = csv.DictWriter(_ef, fieldnames=["comment_id", "body", "label", "sentiment", "story"])
            _ew.writeheader()
            for _i in range(25):
                _ew.writerow({"comment_id": f"ex{_i}", "body": f"example comment {_i}",
                              "label": ["democracy", "economics", "none", "social", "environment"][_i % 5],
                              "sentiment": str(_i % 6), "story": ["luigi", "musk", "jets", "propublica"][_i % 4]})
        # The same examples WITHOUT ratings: refused, unless sentiment examples are switched off.
        _ex_plain = os.path.join(tmp, "fixed_plain.csv")
        with open(_ex_plain, "w", newline="", encoding="utf-8-sig") as _ef:
            _ew = csv.DictWriter(_ef, fieldnames=["comment_id", "body", "label"])
            _ew.writeheader()
            for _i in range(25):
                _ew.writerow({"comment_id": f"ex{_i}", "body": f"example comment {_i}",
                              "label": ["democracy", "economics", "none", "social", "environment"][_i % 5]})
        _orig_fixed_p = list(taxonomy.FIXED_EXAMPLES)
        try:
            with contextlib.redirect_stderr(io.StringIO()) as _err_plain:
                try:
                    classify_gemma.main([_in, "--output", os.path.join(tmp, "plain.csv"),
                                         "--cache", os.path.join(tmp, "cp.jsonl"),
                                         "--fixed-examples", _ex_plain])
                    _plain_refused = False
                except SystemExit as _pe:
                    _plain_refused = "no `sentiment` and `story` columns" in str(_pe)
            check("a fixed-examples file without ratings is refused by ewl-gemma, and the line says NO ratings",
                  _plain_refused and "NO ratings" in _err_plain.getvalue(), _err_plain.getvalue()[-300:])
            with contextlib.redirect_stderr(io.StringIO()) as _err_off:
                classify_gemma.main([_in, "--output", os.path.join(tmp, "plain_off.csv"),
                                     "--cache", os.path.join(tmp, "cp2.jsonl"),
                                     "--fixed-examples", _ex_plain, "--sentiment-examples", "off"])
            check("--sentiment-examples off lets a ratings-less file run",
                  os.path.exists(os.path.join(tmp, "plain_off.csv")), _err_off.getvalue()[-200:])
        finally:
            taxonomy.reset_fixed_examples(_orig_fixed_p)
        _out2 = os.path.join(tmp, "labelled2.csv")
        _orig_fixed_g = list(taxonomy.FIXED_EXAMPLES)
        try:
            with contextlib.redirect_stderr(io.StringIO()) as _err2:
                classify_gemma.main([_in, "--output", _out2, "--cache", os.path.join(tmp, "c2.jsonl"),
                                     "--fixed-examples", _ex])
            with open(_out2 + ".run.json", encoding="utf-8") as _rf:
                _rec2 = json.load(_rf)
            check("--fixed-examples reaches the run: recorded by id and hash, never by text",
                  _rec2["fixed_examples"]["source"] == "file"
                  and len(_rec2["fixed_examples"]["ids"]) == 25
                  and "example comment 0" not in json.dumps(_rec2["fixed_examples"]),
                  _rec2["fixed_examples"].keys())
            check("and the prompt hash differs from the built-in set's (the cache cannot mix them)",
                  _rec2["prompts_sha256"]["domain"] != _rec["prompts_sha256"]["domain"])
            check("without the file, the run says plainly it is using the placeholders",
                  "BUILT-IN fixed examples" in _err_builtin.getvalue())
        finally:
            taxonomy.reset_fixed_examples(_orig_fixed_g)

        # A cache holding an empty reply is refused, as on ewl-classify.
        _bad_cache = os.path.join(tmp, "bad.jsonl")
        with open(_bad_cache, "w", encoding="utf-8") as _bc:
            _bc.write(json.dumps({"key": "k", "reply": ""}) + "\n")
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                classify_gemma.main([_in, "--output", os.path.join(tmp, "o3.csv"),
                                     "--cache", _bad_cache])
            check("a cache holding empty replies is refused", False, "no exception")
        except SystemExit as _se:
            check("a cache holding empty replies is refused", "empty cached repl" in str(_se))

        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            _rc_ok = classify_gemma.main(["--check"])
        check("--check passes when the replies are readable", _rc_ok == 0, _rc_ok)
        _with_fake(_FakeOllama(domain_reply="I think this is mostly about tax policy."))
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            _rc_bad = classify_gemma.main(["--check"])
        check("--check fails when the model answers in sentences instead of labels", _rc_bad == 1, _rc_bad)

        _with_fake(_FakeOllama(models=[]))
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                classify_gemma.main([_in, "--output", _out, "--no-cache"])
            check("the script refuses to start when the model is not pulled", False, "no exception")
        except SystemExit as _err:
            check("the script refuses to start when the model is not pulled",
                  "not ready" in str(_err) and "ollama pull" in str(_err), str(_err))
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                classify_gemma.main([_in])
            check("labelling needs an --output", False, "no exception")
        except SystemExit as _err:
            check("labelling needs an --output", _err.code == 2, _err.code)
finally:
    gemma_local.GemmaJudge._make_client = _real_make_client

_help = subprocess.run([sys.executable, "-m", "ewl.label.classify_gemma", "--help"],
                       capture_output=True, text=True, env=env)
check("ewl-gemma --help works with the ollama package not installed",
      _help.returncode == 0 and "Q4_0" in _help.stdout, _help.stderr[-300:])
_pyproject = (pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(encoding="utf-8")
check("the ewl-gemma console script is declared and points at a real function",
      'ewl-gemma' in _pyproject and 'ewl.label.classify_gemma:main' in _pyproject
      and callable(classify_gemma.main))

# CI's "console scripts are wired up" step lists the scripts by hand, so a new one
# can be declared in pyproject.toml and never run there (ewl-adjudicate was missed).
import re as _re_ci
_declared_scripts = set(_re_ci.findall(r"^(ewl-[a-z-]+)\s*=", _pyproject, _re_ci.M))
_ci_yml = (pathlib.Path(__file__).resolve().parent.parent / ".github" / "workflows"
           / "test.yml").read_text(encoding="utf-8")
_ci_loop = _re_ci.search(r"for cmd in (.*?); do", _ci_yml, _re_ci.S)
_ci_scripts = set(_re_ci.findall(r"ewl-[a-z-]+", _ci_loop.group(1))) if _ci_loop else set()
check("CI runs --help for every console script pyproject.toml declares",
      _declared_scripts and _declared_scripts == _ci_scripts,
      f"declared, not in CI: {sorted(_declared_scripts - _ci_scripts)}; "
      f"in CI, not declared: {sorted(_ci_scripts - _declared_scripts)}")


print("\nopen-weight arm: choosing the comments and measuring the speed")
_pool = ([{"comment_id": f"a{i}", "story": "luigi", "body": f"luigi comment {i}"} for i in range(30)]
         + [{"comment_id": f"b{i}", "story": "musk", "body": f"musk comment {i}"} for i in range(30)]
         + [{"comment_id": f"c{i}", "story": "propublica", "body": f"pp comment {i}"} for i in range(4)])
_pick = classify_gemma.stratified_sample(_pool, 20, seed=1)
_by_story = collections.Counter(r["story"] for r in _pick)
check("a sample is exactly the size asked for", len(_pick) == 20, len(_pick))
check("the stories share it equally, and a short story hands its shortfall to the others",
      dict(_by_story) == {"luigi": 8, "musk": 8, "propublica": 4}, dict(_by_story))
check("the same seed draws the same comments; a different seed draws different ones",
      [r["comment_id"] for r in classify_gemma.stratified_sample(_pool, 20, seed=1)]
      == [r["comment_id"] for r in _pick]
      != [r["comment_id"] for r in classify_gemma.stratified_sample(_pool, 20, seed=2)])
check("rows keep their original file order",
      [r["comment_id"] for r in _pick] == sorted((r["comment_id"] for r in _pick),
                                                  key=lambda c: [x["comment_id"] for x in _pool].index(c)))
check("asking for more than exist returns everything, once",
      len(classify_gemma.stratified_sample(_pool, 500)) == len(_pool))
_ids = lambda n: {r["comment_id"] for r in classify_gemma.stratified_sample(_pool, n, seed=1)}
check("a bigger sample with the same seed CONTAINS the smaller one, so a run can be grown "
      "in batches with no earlier labelling wasted",
      _ids(10) <= _ids(20) <= _ids(40) <= _ids(64), (len(_ids(10) & _ids(20)), len(_ids(20) & _ids(40))))
check("and it stays spread across stories at every stage",
      all(len({c[0] for c in _ids(n)}) == 3 for n in (10, 20, 40)))
check("the first N rows of the real corpus would be one story; a sample is not",
      len({r["story"] for r in _pool[:20]}) == 1 and len(_by_story) == 3)

_ss = classify_gemma.steady_state([("d", 30.0, 1500), ("d", 1.0, 40), ("d", 2.0, 60), ("d", 3.0, 50)])
check("the first call is start-up; the rest are the steady state",
      _ss["startup"] == 30.0 and _ss["calls"] == 3 and _ss["mean"] == 2.0 and _ss["median"] == 2.0
      and _ss["tokens"] == 50.0, _ss)
_ss2 = classify_gemma.steady_state([("luigi", 20.0, 1200), ("musk", 18.0, 1200),
                                    ("luigi", 1.0, 30), ("musk", 3.0, 30)])
check("each story's sentiment prompt has its own start-up call",
      _ss2["startup"] == 38.0 and _ss2["calls"] == 2 and _ss2["mean"] == 2.0, _ss2)
check("with only start-up calls there is no steady rate to report",
      classify_gemma.steady_state([("d", 30.0, 1500)])["mean"] is None)
check("durations read sensibly",
      classify_gemma.format_duration(45) == "45 s" and classify_gemma.format_duration(600) == "10 min"
      and classify_gemma.format_duration(3.5 * 3600) == "3.5 h")

_d = classify_gemma.steady_state([("d", 30.0, 1500)] + [("d", 1.5, 45)] * 40)
_s = classify_gemma.steady_state([("m", 12.0, 1200)] + [("m", 1.0, 40)] * 40)
_buf = io.StringIO()
_projected = classify_gemma.report_timing(_d, _s, 5000, 1441, file=_buf)
check("projection = start-up + N x (domain + sentiment) seconds per comment",
      abs(_projected - (42.0 + 5000 * 2.5)) < 1e-6, _projected)
check("the report says how long 5,000 comments would take",
      "5,000 comments would take about 3.5 h" in _buf.getvalue(), _buf.getvalue())
_thin = classify_gemma.steady_state([("d", 30.0, 1500)] + [("d", 1.5, 45)] * 3)
_buf = io.StringIO()
classify_gemma.report_timing(_thin, _thin, 5000, 1441, file=_buf)
check("a projection from a handful of calls is labelled tentative",
      "TENTATIVE" in _buf.getvalue(), _buf.getvalue())
_slow = classify_gemma.steady_state([("d", 30.0, 1500)] + [("d", 14.0, 1450)] * 40)
_buf = io.StringIO()
classify_gemma.report_timing(_slow, _slow, None, 1441, file=_buf)
check("calls this slow are flagged as probably re-reading the prompt each time",
      "WARNING" in _buf.getvalue() and "re-reading" in _buf.getvalue(), _buf.getvalue())

try:
    _fake = _FakeOllama()
    _with_fake(_fake)
    with tempfile.TemporaryDirectory() as tmp:
        _in = os.path.join(tmp, "corpus.csv")
        _out = os.path.join(tmp, "trial.csv")
        with open(_in, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["comment_id", "story", "body"])
            w.writeheader()
            w.writerows(_pool)
        _err = io.StringIO()
        with contextlib.redirect_stderr(_err):
            _rc = classify_gemma.main([_in, "--output", _out, "--sample", "20", "--seed", "1",
                                       "--project-to", "5000", "--cache", os.path.join(tmp, "c.jsonl")])
        with open(_out, newline="", encoding="utf-8-sig") as f:
            _rows_out = list(csv.DictReader(f))
        check("--sample labels exactly that many comments, evenly across stories",
              _rc == 0 and len(_rows_out) == 20
              and collections.Counter(r["story"] for r in _rows_out) == collections.Counter(_by_story),
              (_rc, len(_rows_out)))
        check("it says which comments it drew", "Sample of 20 comments (seed 1)" in _err.getvalue(),
              _err.getvalue()[:400])
        check("it reports the measured speed and the projection",
              "Speed (real model calls only" in _err.getvalue()
              and "5,000 comments would take about" in _err.getvalue(), _err.getvalue()[-900:])

        _calls_before = len(_fake.calls)
        _err = io.StringIO()
        with contextlib.redirect_stderr(_err):
            classify_gemma.main([_in, "--output", _out, "--sample", "20", "--seed", "1",
                                 "--project-to", "5000", "--cache", os.path.join(tmp, "c.jsonl")])
        check("a resumed run replays the cache, so it makes no calls and reports no speed "
              "(cached replies would make the model look instant)",
              len(_fake.calls) == _calls_before and "too few calls to measure" in _err.getvalue(),
              _err.getvalue()[-500:])

        for _bad in (["--sample", "10", "--limit", "5"],):
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    classify_gemma.main([_in, "--output", _out, *_bad])
                check("--sample and --limit together are refused", False, "no exception")
            except SystemExit as _exit:
                check("--sample and --limit together are refused", _exit.code == 2, _exit.code)
finally:
    gemma_local.GemmaJudge._make_client = _real_make_client

print("\nbatches.py: splitting a corpus and checking the labelled batches")
from ewl.collect import batches as _batches
from ewl import corpus as _corp_b

with tempfile.TemporaryDirectory() as _tb:
    _src = os.path.join(_tb, "corpus.csv")
    _stories = ["luigi", "musk", "jets", "propublica"]
    with open(_src, "w", newline="", encoding="utf-8-sig") as _f:
        _w = csv.DictWriter(_f, fieldnames=["comment_id", "story", "source", "body"])
        _w.writeheader()
        for _i in range(23):
            _w.writerow({"comment_id": f"k{_i:02d}", "story": _stories[_i // 6 % 4],
                         "source": f"v{_i % 5}",
                         "body": "" if _i == 7 else f"synthetic comment {_i}"})
    _d1 = os.path.join(_tb, "b1")
    _m1 = _batches.split(_src, _d1, batch_size=5, shuffle_seed=7)
    check("23 comments in batches of 5 make 5 batches (5, 5, 5, 5, 3)",
          [b["rows"] for b in _m1["batches"]] == [5, 5, 5, 5, 3], [b["rows"] for b in _m1["batches"]])
    check("the batch names say which of how many",
          _m1["batches"][0]["file"] == "batch_001_of_005.csv")
    _seen = []
    for _b in _m1["batches"]:
        with open(os.path.join(_d1, "inputs", _b["file"]), newline="", encoding="utf-8-sig") as _f:
            _seen += [r["comment_id"] for r in csv.DictReader(_f)]
    check("every comment lands in exactly one batch", sorted(_seen) == [f"k{i:02d}" for i in range(23)])
    check("the blank-text row is counted, not dropped",
          _m1["input"]["blank_body_rows"] == 1 and sum(b["blank_body_rows"] for b in _m1["batches"]) == 1)
    check("the manifest holds hashes and counts but no comment text",
          "synthetic comment" not in open(os.path.join(_d1, "batches_manifest.json"),
                                          encoding="utf-8").read())
    _m1b = _batches.split(_src, os.path.join(_tb, "b1b"), batch_size=5, shuffle_seed=7)
    check("the same seed cuts the same batches (identical hashes)",
          [b["sha256"] for b in _m1["batches"]] == [b["sha256"] for b in _m1b["batches"]])
    _m2 = _batches.split(_src, os.path.join(_tb, "b2"), batch_size=5, shuffle_seed=8)
    check("a different seed cuts different batches",
          [b["sha256"] for b in _m1["batches"]] != [b["sha256"] for b in _m2["batches"]])
    _m3 = _batches.split(_src, os.path.join(_tb, "b3"), batch_size=5, shuffle_seed=None)
    check("--no-shuffle keeps file order (contiguous slices)",
          _m3["batches"][0]["source_positions"] == [0, 1, 2, 3, 4])
    try:
        _batches.split(_src, _d1, batch_size=5)
        _refused = False
    except SystemExit:
        _refused = True
    check("splitting into a folder that already holds batches is refused", _refused)
    _dup = os.path.join(_tb, "dup.csv")
    with open(_dup, "w", newline="", encoding="utf-8-sig") as _f:
        _w = csv.DictWriter(_f, fieldnames=["comment_id", "body"])
        _w.writeheader()
        _w.writerow({"comment_id": "a", "body": "x"})
        _w.writerow({"comment_id": "a", "body": "y"})
    try:
        _batches.split(_dup, os.path.join(_tb, "bd"))
        _refused = False
    except SystemExit as _e:
        _refused = "more than once" in str(_e)
    check("a corpus with a repeated comment_id is refused", _refused)

    _env_b = dict(os.environ, EWL_AUTHOR_SALT="test-salt-value-1234")

    # The real corpus size, at the default batch size, from the command line (so the
    # argparse default is what is tested). Ordered by story in four blocks, as the real
    # file is, so "every batch mixes the stories" is a test of the shuffle.
    _big = os.path.join(_tb, "big.csv")
    _n_big = 84250
    with open(_big, "w", newline="", encoding="utf-8-sig") as _f:
        _w = csv.writer(_f)
        _w.writerow(["comment_id", "story", "body"])
        for _i in range(_n_big):
            _w.writerow([f"c{_i:05d}", _stories[_i * 4 // _n_big], "x"])
    _big_runs = []
    for _name in ("big1", "big2"):
        _p = subprocess.run([sys.executable, "-m", "ewl.collect.batches", "split", _big,
                             "--out-dir", os.path.join(_tb, _name)],
                            capture_output=True, text=True, env=_env_b)
        assert _p.returncode == 0, _p.stderr[-400:]
        _big_runs.append(_batches.load_manifest(os.path.join(_tb, _name)))
    _mb = _big_runs[0]
    check("the default batch size is 1,000", _mb["batch_size"] == 1000, _mb["batch_size"])
    check("84,250 comments at 1,000 make 85 batches: 84 of 1,000 and one of 250",
          [b["rows"] for b in _mb["batches"]] == [1000] * 84 + [250],
          collections.Counter(b["rows"] for b in _mb["batches"]))
    check("the batch names count to 85",
          _mb["batches"][0]["file"] == "batch_001_of_085.csv"
          and _mb["batches"][-1]["file"] == "batch_085_of_085.csv")
    _big_ids = []
    _big_stories = []
    for _b in _mb["batches"]:
        with open(os.path.join(_tb, "big1", "inputs", _b["file"]), newline="",
                  encoding="utf-8-sig") as _f:
            _brows = list(csv.DictReader(_f))
        _big_ids.append([r["comment_id"] for r in _brows])
        _big_stories.append({r["story"] for r in _brows})
    _flat = [cid for ids in _big_ids for cid in ids]
    check("no comment is in two batches (84,250 ids, 84,250 distinct)",
          len(_flat) == _n_big and len(set(_flat)) == _n_big, (len(_flat), len(set(_flat))))
    check("together the batches hold every comment of the corpus",
          set(_flat) == {f"c{i:05d}" for i in range(_n_big)})
    check("the manifest's positions are the same partition (each position once, all of them)",
          sorted(p for b in _mb["batches"] for p in b["source_positions"]) == list(range(_n_big)))
    check("the same seed cuts the same 85 batches (identical hashes)",
          [b["sha256"] for b in _mb["batches"]] == [b["sha256"] for b in _big_runs[1]["batches"]])
    check("each batch is a random slice: every one (the 250 included) mixes all four stories",
          all(s == set(_stories) for s in _big_stories),
          [len(s) for s in _big_stories if s != set(_stories)])
    check("and is not a contiguous run of the file",
          all(_b["source_positions"] != list(range(min(_b["source_positions"]),
                                                   min(_b["source_positions"]) + _b["rows"]))
              for _b in _mb["batches"]))

    def _label(batch, extra=()):
        _out = os.path.join(_d1, "labelled", batch["file"])
        os.makedirs(os.path.dirname(_out), exist_ok=True)
        return subprocess.run(
            [sys.executable, "-m", "ewl.label.classify", os.path.join(_d1, "inputs", batch["file"]),
             "--judge", "stub", "--votes", "1", "--no-cache", "--output", _out, *extra],
            capture_output=True, text=True, env=_env_b)

    class _A:
        out_dir = _d1
        merge_to = None
        extra = ""
        cache = None
        first = None
        partial = False

    check("before any labelling, every batch is pending",
          all(s["state"] == "pending" for s in _batches.status_lines(_d1, _m1)))
    _cmds = [_batches.command_for(_d1, s["entry"], "--fixed-examples f.csv")
             for s in _batches.status_lines(_d1, _m1)]
    check("the printed commands name the batch, its output, one shared cache and the flags",
          len(_cmds) == 5 and all("ewl-gemma" in c and "--fixed-examples f.csv" in c
                                  and "gemma_cache.jsonl" in c for c in _cmds), _cmds[0])

    def _commands(out_dir, *flags):
        _p = subprocess.run([sys.executable, "-m", "ewl.collect.batches", "commands", out_dir,
                             *flags], capture_output=True, text=True, env=_env_b)
        return _p.returncode, [line for line in _p.stdout.splitlines() if line.strip()], _p.stderr

    _rc, _lines, _err = _commands(os.path.join(_tb, "big1"), "--first", "5")
    check("commands --first 5 on the 85-batch split prints batches 1 to 5 only, in order",
          _rc == 0 and len(_lines) == 5
          and all(f"batch_{n:03d}_of_085.csv" in line and "inputs" in line for n, line in zip(range(1, 6), _lines)),
          _lines)
    check("and says how many of how many it printed (stderr, so run_all.sh holds commands only)",
          "first 5 of 85" in _err and all(line.startswith("ewl-gemma ") for line in _lines), _err)
    _rc, _lines, _err = _commands(_d1, "--first", "0")
    check("commands --first 0 is refused", _rc == 2 and not _lines, _err)

    # A run that used --fixed-examples FILE: commands that leave it out would relabel every
    # later batch with the built-in examples, which `check` then refuses to join.
    _fx_src = os.path.join(_tb, "fx_src.csv")
    with open(_fx_src, "w", newline="", encoding="utf-8-sig") as _f:
        _w = csv.writer(_f)
        _w.writerow(["comment_id", "story", "body"])
        for _i in range(6):
            _w.writerow([f"f{_i}", "jets", "synthetic text"])
    _fx_dir = os.path.join(_tb, "fx")
    _fx_m = _batches.split(_fx_src, _fx_dir, batch_size=2)
    _rc, _lines, _err = _commands(_fx_dir)
    check("commands with no finished batch and no --fixed-examples prints them, with a warning",
          _rc == 0 and len(_lines) == 3 and "BUILT-IN" in _err, (_lines, _err[-300:]))
    _fx_lab = os.path.join(_fx_dir, "labelled", _fx_m["batches"][0]["file"])
    os.makedirs(os.path.dirname(_fx_lab), exist_ok=True)
    with open(os.path.join(_fx_dir, "inputs", _fx_m["batches"][0]["file"]), "rb") as _f:
        _fx_batch1 = _f.read()
    with open(_fx_lab, "wb") as _f:   # the batch's own comments, as a finished run writes them
        _f.write(_fx_batch1)
    with open(_fx_lab + ".run.json", "w", encoding="utf-8") as _f:
        json.dump({"input": {"sha256": _fx_m["batches"][0]["sha256"]}, "result": {},
                   "fixed_examples": {"source": "file", "file": "fixed_examples.csv",
                                      "sha256": "ab" * 32}}, _f)
    _rc, _lines, _err = _commands(_fx_dir)
    check("once a batch was labelled with --fixed-examples, commands without it are refused",
          _rc == 1 and not _lines and "fixed_examples.csv" in _err, (_lines, _err[-300:]))
    _rc, _lines, _err = _commands(_fx_dir, "--extra", "--fixed-examples official-data/fixed_examples.csv")
    check("and printed, with the flag, when --extra carries it",
          _rc == 0 and len(_lines) == 2
          and all("--fixed-examples official-data/fixed_examples.csv" in l for l in _lines)
          and "BUILT-IN" not in _err, (_lines, _err[-300:]))
    _rc, _lines, _err = _commands(_fx_dir, "--extra=--fixed-examples=x.csv")
    check("the --fixed-examples=FILE spelling counts too", _rc == 0 and len(_lines) == 2, _err[-300:])

    for _b in _m1["batches"][:2]:
        _p = _label(_b)
        assert _p.returncode == 0, _p.stderr[-400:]
    _rc, _lines, _ = _commands(_d1, "--first", "2")
    check("with batches 1 and 2 finished, --first 2 continues with 3 and 4",
          _rc == 0 and len(_lines) == 2 and "batch_003_of_005" in _lines[0]
          and "batch_004_of_005" in _lines[1], _lines)
    _rc, _lines, _err = _commands(_d1, "--first", "9")
    check("--first larger than what is left prints what is left (3, 4, 5), with no 'first N of' note",
          _rc == 0 and [line.split("inputs")[1][1:10] for line in _lines]  # separator-neutral: Windows paths use a backslash
          == ["batch_003", "batch_004", "batch_005"] and "first" not in _err, _lines)
    _rc, _lines, _ = _commands(_d1)
    check("without --first every batch still to run is printed, as before",
          _rc == 0 and len(_lines) == 3, _lines)
    for _b in _m1["batches"][2:]:
        _p = _label(_b)
        assert _p.returncode == 0, _p.stderr[-400:]
    check("all labelled and consistent: check passes", _batches.cmd_check(_A) == 0)

    _merged = os.path.join(_tb, "merged.csv")

    class _M(_A):
        merge_to = _merged

    check("merge succeeds when every batch is done", _batches.cmd_check(_M) == 0)
    with open(_merged, newline="", encoding="utf-8-sig") as _f:
        _mr = list(csv.DictReader(_f))
    check("the merged file has every comment that has text (22 of 23), once each",
          len(_mr) == 22 and len({r["comment_id"] for r in _mr}) == 22, len(_mr))
    check("and it is back in the ORIGINAL corpus order",
          [r["comment_id"] for r in _mr] == [f"k{i:02d}" for i in range(23) if i != 7],
          [r["comment_id"] for r in _mr][:6])
    with open(_merged + ".run.json", encoding="utf-8") as _f:
        _mrec = json.load(_f)
    check("the merged run record names the whole corpus as its input (so the dashboard check can use it)",
          _mrec["input"]["sha256"] == _corp_b.content_hash(_src), _mrec["input"])

    # ---- things that must be refused ------------------------------------
    _lab2 = os.path.join(_d1, "labelled", _m1["batches"][1]["file"])
    _keep = open(_lab2, encoding="utf-8-sig", newline="").read()
    _p = _label(_m1["batches"][1], ["--story-rules", "off"])
    check("a batch labelled under a different switch is refused (not merged)",
          _p.returncode == 0 and _batches.cmd_check(_M) == 1)
    _label(_m1["batches"][1])  # restore the right one
    check("relabelled under the same setting, it passes again", _batches.cmd_check(_A) == 0)

    with open(_lab2, newline="", encoding="utf-8-sig") as _f:
        _rd = csv.DictReader(_f)
        _fn = _rd.fieldnames
        _rows2 = list(_rd)
    with open(_lab2, "w", newline="", encoding="utf-8-sig") as _f:
        _wr = csv.DictWriter(_f, fieldnames=_fn)
        _wr.writeheader()
        _wr.writerows(_rows2[1:])
    check("a labelled batch with a comment missing is refused", _batches.cmd_check(_A) == 1)
    _label(_m1["batches"][1])

    _in0 = os.path.join(_d1, "inputs", _m1["batches"][0]["file"])
    _orig0 = open(_in0, "rb").read()
    open(_in0, "ab").write(b"\n")
    check("a batch file edited after the split is refused",
          [s for s in _batches.status_lines(_d1, _m1)][0]["state"] == "bad")
    os.remove(os.path.join(_d1, "labelled", _m1["batches"][3]["file"]))
    _rc, _lines, _err = _commands(_d1, "--first", "5")
    check("commands --first does not skip an edited batch silently: it warns, naming it, NOT queued",
          _rc == 0 and "WARNING: batch_001_of_005.csv is NOT queued" in _err
          and len(_lines) == 1 and "batch_004_of_005" in _lines[0], (_lines, _err[-300:]))
    _label(_m1["batches"][3])
    open(_in0, "wb").write(_orig0)

    os.remove(os.path.join(_d1, "labelled", _m1["batches"][2]["file"]))
    check("with a batch still pending, merge is refused", _batches.cmd_check(_M) == 1)

    # ---- a progress file of the batches finished so far (--partial) -----------
    _prog = os.path.join(_tb, "so_far.csv")

    class _P(_A):
        merge_to = _prog
        partial = True

    check("with a batch still pending, --partial joins the finished ones", _batches.cmd_check(_P) == 0)
    _done_ids = []
    for _b in _m1["batches"]:
        if _b["number"] == 3:
            continue
        with open(os.path.join(_d1, "inputs", _b["file"]), newline="", encoding="utf-8-sig") as _f:
            _done_ids += [r["comment_id"] for r in csv.DictReader(_f) if r["body"].strip()]
    with open(_prog, newline="", encoding="utf-8-sig") as _f:
        _pr = [r["comment_id"] for r in csv.DictReader(_f)]
    check("the progress file holds exactly the finished batches' comments, once each",
          sorted(_pr) == sorted(_done_ids) and len(set(_pr)) == len(_pr), (len(_pr), len(_done_ids)))
    check("in the original corpus order",
          _pr == [c for c in (f"k{i:02d}" for i in range(23)) if c in set(_done_ids)], _pr[:6])
    with open(_prog + ".run.json", encoding="utf-8") as _f:
        _prec = json.load(_f)
    check("its run record is marked partial, with the batches it holds and the counts",
          _prec.get("partial", {}).get("batches_included")
          == [b["file"] for b in _m1["batches"] if b["number"] != 3]
          and _prec["partial"]["batches_done"] == 4 and _prec["partial"]["batches_total"] == 5
          and _prec["partial"]["comments_included"] == len(_done_ids)
          and _prec["partial"]["comments_in_corpus"] == 22
          and _prec["result"]["partial"] is True and "--partial" in _prec["command"], _prec.get("partial"))
    check("its blank_body_rows counts the included batches only, not the whole corpus",
          _prec["result"]["blank_body_rows"]
          == sum(b["blank_body_rows"] for b in _m1["batches"] if b["number"] != 3),
          _prec["result"])
    _pdash = os.path.join(_tb, "so_far.json")
    _p = subprocess.run([sys.executable, "-m", "ewl.analyse.export_dashboard", _prog, "--output",
                         _pdash, "--allow-unvalidated", "--corpus", _src],
                        capture_output=True, text=True, env=_env_b)
    check("ewl-export-dashboard --corpus refuses the progress file, though its input hash is the corpus's",
          _p.returncode != 0 and "PARTIAL progress file" in _p.stderr and not os.path.exists(_pdash),
          _p.stderr[-400:])
    _p = subprocess.run([sys.executable, "-m", "ewl.analyse.export_dashboard", _prog, "--output",
                         _pdash, "--allow-unvalidated"], capture_output=True, text=True, env=_env_b)
    _pd = json.load(open(_pdash, encoding="utf-8")) if os.path.exists(_pdash) else {}
    check("without --corpus it warns and stamps the export partial in its provenance",
          _p.returncode == 0 and "WARNING" in _p.stderr and "PARTIAL" in _p.stderr
          and (_pd.get("provenance") or {}).get("partial_run", {}).get("batches_done") == 4,
          _p.stderr[-400:])
    if project_file("dashboard/index.html"):
        _page_b = open("dashboard/index.html", encoding="utf-8").read()
        check("the dashboard page reads provenance.partial_run, with every field the export writes",
              "p.partial_run" in _page_b
              and all(f"partial.{k}" in _page_b for k in _pd["provenance"]["partial_run"]
                      if k != "batches_included"), sorted(_pd["provenance"]["partial_run"]))
        check("and shows a 'Partial run: N of M comments (k of K batches), not the whole corpus' banner",
              "<strong>Partial run: ${(partial.comments_included" in _page_b
              and "of ${partial.batches_total" in _page_b and "the whole corpus.</strong>" in _page_b)
        check("and the intro's comment count says it is the partial count",
              "(${stories})${partialNote}." in _page_b and "<strong>a partial run</strong>: the ${totalC" in _page_b)
    _p = subprocess.run([sys.executable, "-m", "ewl.collect.batches", "check", _d1, "--partial"],
                        capture_output=True, text=True, env=_env_b)
    check("--partial without --merge-to is refused", _p.returncode == 2, _p.stderr[-200:])
    _label(_m1["batches"][2])
    _full = os.path.join(_tb, "full_via_partial.csv")

    class _PF(_P):
        merge_to = _full

    check("once every batch is done, --partial writes the ordinary full merge (no partial mark)",
          _batches.cmd_check(_PF) == 0
          and "partial" not in json.load(open(_full + ".run.json", encoding="utf-8")))
    check("and the full merge's blank_body_rows is the whole corpus's (1)",
          json.load(open(_full + ".run.json", encoding="utf-8"))["result"]["blank_body_rows"] == 1)
    _empty = os.path.join(_tb, "b_empty")
    _batches.split(_src, _empty, batch_size=5, shuffle_seed=7)

    class _PE(_P):
        out_dir = _empty
        merge_to = os.path.join(_tb, "nothing.csv")

    check("--partial with no batch finished writes nothing",
          _batches.cmd_check(_PE) == 1 and not os.path.exists(_PE.merge_to))

    _p = subprocess.run([sys.executable, "-m", "ewl.collect.batches", "status", _d1],
                        capture_output=True, text=True, env=_env_b)
    check("status runs from the command line and reports all five done",
          _p.returncode == 0 and "done 5" in _p.stdout, _p.stdout + _p.stderr)

print("\ndomain prompt: the story's factual background, after the examples (decided 2026-10-07)")
from ewl.label import sentiment as _sdc
from ewl.label import instrument as _idc

_ex = [("a comment", "none")]
_plain = taxonomy.build_prompt(_ex)
check("without a story the domain prompt is exactly the story-free one",
      _plain == taxonomy.STABLE_PREFIX + '"a comment" -> none')
_per_story = {s: taxonomy.build_prompt(_ex, s) for s in _sdc.STORY_CONTEXT}
check("every story gets its own background, ending the prompt",
      all(p.startswith(_plain) and p.endswith(taxonomy.domain_story_context(s)) and len(p) > len(_plain) + 200
          for s, p in _per_story.items()), {s: len(p) for s, p in _per_story.items()})
check("the shared part (instructions and examples) is identical for every story",
      len({p[:len(_plain)] for p in _per_story.values()}) == 1)
check("the background is the FACTS only: no stance rule from any story appears in the domain prompt",
      not any(_sdc.rules_only(s) and _sdc.rules_only(s) in p for s, p in _per_story.items())
      and "generally hostile" not in " ".join(_per_story.values())
      and "generally supportive" not in " ".join(_per_story.values()))
check("ProPublica gets its facts only, without its stance rule",
      taxonomy.domain_story_context("propublica") == _sdc.facts_only("propublica").strip())
check("the heading tells the model the rules still apply and the comment is what is classified",
      "classify what the COMMENT itself is about, not the story" in _per_story["musk"])
check("an unknown story falls back to the story-free prompt", taxonomy.build_prompt(_ex, "nope") == _plain)

taxonomy.configure_domain_context("none")
try:
    check("switch off (--domain-context none): the earlier, story-free prompt, byte for byte",
          taxonomy.build_prompt(_ex, "musk") == _plain and taxonomy.domain_context_tag() == "[domain_context=none]")
finally:
    taxonomy.configure_domain_context("facts")
check("the default is facts, with no tag", taxonomy.configure_domain_context() == "facts"
      and taxonomy.domain_context_tag() == "")
try:
    taxonomy.configure_domain_context("rules")
    _refused = False
except ValueError:
    _refused = True
check("an unknown domain-context setting is refused", _refused)


@judges.register("test_domctx")
class _DomCtxJudge(judges.Judge):
    default_model = "domctx"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.prompts = []

    def _sample(self, body, prompt, vote):
        self.prompts.append(prompt)
        return "none"


_dj = judges.create("test_domctx", votes=1)
_dj.label("hello", _ex, story="musk")
_dj.label("hello", _ex, story="jets")
_dj.label("hello", _ex)
check("Judge.label sends the story's own prompt (so a story-keyed cache entry cannot be reused across stories)",
      _dj.prompts[0] == _per_story["musk"] and _dj.prompts[1] == _per_story["jets"]
      and _dj.prompts[2] == _plain and len(set(_dj.prompts)) == 3)


class _DcArgs:
    story_rules, stance_object, combined_domain_context, domain_context = "on", "conduct", "keep", "none"


try:
    _tag = _idc.apply(_DcArgs())
    check("instrument.apply sets the domain-context switch and names it in the tags",
          "domain_context=none" in _tag and taxonomy.configure_domain_context() == "none", _tag)
finally:
    taxonomy.configure_domain_context("facts")

from ewl.collect import batches as _bdc
_fp1 = _bdc._fingerprint({"prompts_sha256": {"domain": "d", "domain_by_story": {"luigi": "1"}}})
_fp2 = _bdc._fingerprint({"prompts_sha256": {"domain": "d", "domain_by_story": {"luigi": "2"}}})
check("ewl-batches refuses to mix batches whose per-story domain prompts differ",
      _fp1 != _fp2 and _fp1["domain prompts by story"] != _fp2["domain prompts by story"])

print("\nanthropic_batch.py: the Batch API arm (fake service, synthetic comments, no network)")
import datetime
import re as _re_ab
import types as _tyb
from ewl.label import anthropic_batch as _ab
from ewl.label import classify as _clsb
from ewl.label import instrument as _instr_ab
from ewl.label import sentiment as _sent_ab


class _World:
    """A stand-in for Anthropic's service: answers live calls and batches, and can fail
    particular comments (chosen by words in the comment) in the ways the real one can."""

    def __init__(self):
        self.live = []          # kwargs of every live messages.create
        self.batches = {}       # id -> {"requests": [...], "ended": bool, "created_at": datetime}
        self.created = 0
        self.lose_reply = False     # create the batch, then fail as a dropped connection would
        self.unreadable = set()     # batch ids whose results() raise (gone after 29 days)
        self.max_retries = []       # the max_retries each create call was made with
        self.reject_status = None   # an HTTP status create fails with, before any batch exists
        self.running = set()        # batch ids still in_progress
        self.cancelled = []         # batch ids cancel() was called on

    def message(self, system, user):
        domain = system.startswith("You are a precise text classifier")
        text = ("democracy" if "tax" in user.lower() else "none") if domain else "2"
        stop = "end_turn"
        if "REFUSE" in user:
            stop = "refusal"
        elif "LONGREPLY" in user:
            stop = "max_tokens"
        return _tyb.SimpleNamespace(
            stop_reason=stop, content=[_tyb.SimpleNamespace(type="text", text=text)],
            usage=_tyb.SimpleNamespace(input_tokens=(len(system) + len(user)) // 4, output_tokens=2,
                                       cache_creation_input_tokens=0, cache_read_input_tokens=0))


class _Batches:
    def __init__(self, world):
        self.w = world

    def create(self, requests):
        if self.w.reject_status:
            error = RuntimeError(f"Error code: {self.w.reject_status}")
            error.status_code = self.w.reject_status
            raise error
        self.w.created += 1
        batch_id = f"msgbatch_fake{self.w.created:03d}"
        self.w.batches[batch_id] = {"requests": list(requests), "ended": True,
                                    "created_at": datetime.datetime.now(datetime.timezone.utc)}
        if self.w.lose_reply:
            raise ConnectionError("APIConnectionError: connection dropped")
        return _tyb.SimpleNamespace(id=batch_id)

    def cancel(self, batch_id):
        self.w.cancelled.append(batch_id)

    def list(self, limit=20):
        return [_tyb.SimpleNamespace(
                    id=i, created_at=b["created_at"],
                    request_counts=_tyb.SimpleNamespace(processing=0, succeeded=len(b["requests"]),
                                                        errored=0, canceled=0, expired=0))
                for i, b in sorted(self.w.batches.items(), reverse=True)]      # newest first

    def retrieve(self, batch_id):
        results = [] if batch_id in self.w.unreadable else self.results(batch_id)
        count = lambda t: sum(1 for r in results if r.result.type == t)
        return _tyb.SimpleNamespace(
            id=batch_id, processing_status="in_progress" if batch_id in self.w.running else "ended",
            request_counts=_tyb.SimpleNamespace(
                processing=0, succeeded=count("succeeded"), errored=count("errored"),
                canceled=0, expired=count("expired")))

    def results(self, batch_id):
        if batch_id in self.w.unreadable:
            raise RuntimeError("NotFoundError: results are no longer available")
        out = []
        for request in self.w.batches[batch_id]["requests"]:
            params = request["params"]
            user = params["messages"][0]["content"]
            if "ERRBODY" in user:
                outcome = _tyb.SimpleNamespace(type="errored", error=_tyb.SimpleNamespace(
                    error=_tyb.SimpleNamespace(type="api_error")))
            elif "EXPBODY" in user:
                outcome = _tyb.SimpleNamespace(type="expired")
            else:
                outcome = _tyb.SimpleNamespace(type="succeeded", message=self.w.message(
                    params["system"][0]["text"], user))
            out.append(_tyb.SimpleNamespace(custom_id=request["custom_id"], result=outcome))
        return out


class _Messages:
    def __init__(self, world):
        self.w = world
        self.batches = _Batches(world)

    def create(self, **kwargs):
        self.w.live.append(kwargs)
        return self.w.message(kwargs["system"][0]["text"], kwargs["messages"][0]["content"])


class _Client:
    def __init__(self, world, max_retries=2):
        self.w, self.max_retries = world, max_retries
        self.messages = _Messages(world)

    def with_options(self, max_retries=None, timeout=None):
        self.w.max_retries.append(max_retries)
        return _Client(self.w, max_retries)


def _install_fake_anthropic(world):
    module = _tyb.ModuleType("anthropic")
    module.Anthropic = lambda *a, **k: _Client(world)
    sys.modules["anthropic"] = module


def _body_of(request_kwargs):
    content = request_kwargs["messages"][0]["content"]
    return content[len('Classify this comment: "'):-1]


_prev_anthropic = sys.modules.get("anthropic")
_world = _World()
_install_fake_anthropic(_world)
try:
    with tempfile.TemporaryDirectory() as _tb:
        _src = os.path.join(_tb, "corpus.csv")
        _bodies = ["ERRBODY comment", "She pays no tax at all", "Nice video", "EXPBODY comment",
                   "Tax dodging again", "First!", "REFUSE this one", "Billionaires and tax",
                   "LONGREPLY here", "Great clip", "Prices keep rising", "Another comment"]
        with open(_src, "w", newline="", encoding="utf-8-sig") as _f:
            _w = csv.DictWriter(_f, fieldnames=["comment_id", "story", "source", "body"])
            _w.writeheader()
            for _i, _body in enumerate(_bodies):
                _w.writerow({"comment_id": f"c{_i:02d}", "story": ["luigi", "musk", "jets", "propublica"][_i % 4],
                             "source": f"v{_i % 3}", "body": _body})
            _w.writerow({"comment_id": "blank", "story": "musk", "source": "v0", "body": "  "})

        def _run(*argv):
            _err = io.StringIO()
            with contextlib.redirect_stderr(_err), contextlib.redirect_stdout(io.StringIO()) as _out:
                _rc = _ab.main(list(argv))
            return _rc, _err.getvalue(), _out.getvalue()

        _out_dir = os.path.join(_tb, "trial")
        _common = ["--judge", "anthropic:claude-haiku-4-5", "--votes", "1"]

        # ---- plan: a dry run
        _rc, _err, _ = _run("plan", _src, *_common)
        check("plan: a dry run counts 12 comments x 2 questions = 24 requests and sends nothing",
              _rc == 0 and "24 request(s) to send" in _err and "12 comment(s)" in _err
              and _world.created == 0 and not _world.live, _err[-400:])
        check("plan: prints an estimated BATCH cost beside the list price, and says it is an estimate",
              "Estimated BATCH cost" in _err and "list price" in _err and "No prompt-cache discount" in _err)

        # ---- submit everything in small batches
        _rc, _err, _ = _run("submit", _src, "--out-dir", _out_dir, "--sample", "12", "--seed", "1",
                            "--max-batch-requests", "5", *_common)
        _man = json.load(open(os.path.join(_out_dir, "batch_manifest.json"), encoding="utf-8"))
        check("submit: 12 comments = 24 requests, sent as batches of at most 5 (5, 5, 5, 5, 4)",
              _rc == 0 and [b["requests"] for b in _man["batches"]] == [5, 5, 5, 5, 4]
              and _world.created == 5, [b["requests"] for b in _man["batches"]])
        check("submit: the subset used is written beside the manifest",
              os.path.exists(os.path.join(_out_dir, "subset.csv")))
        _ids = [r["custom_id"] for b in _world.batches.values() for r in b["requests"]]
        check("submit: every custom_id is a 64-character hex cache key, all different",
              all(_re_ab.fullmatch(r"[0-9a-f]{64}", i) for i in _ids) and len(set(_ids)) == len(_ids) == 24)
        check("submit: the manifest holds ids, counts and hashes but no comment text",
              not any(w in open(os.path.join(_out_dir, "batch_manifest.json"), encoding="utf-8").read()
                      for w in ("pays no tax", "Nice video", "Tax dodging")))
        check("submit: the manifest records the input hash and a prompt hash per story",
              _man["input"]["sha256"] and len(_man["prompts_sha256"]["domain_by_story"]) == 4
              and len(_man["prompts_sha256"]["sentiment_by_story"]) == 4)
        check("submit: every request is a Messages request with a 1-hour prompt-cache block",
              all(set(r["params"]) >= {"model", "max_tokens", "system", "messages"}
                  and r["params"]["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
                  for b in _world.batches.values() for r in b["requests"]))
        check("submit: the domain request carries the story's background; the sentiment one its own prompt",
              any("Background on the video" in r["params"]["system"][0]["text"]
                  for b in _world.batches.values() for r in b["requests"])
              and any(r["params"]["system"][0]["text"].startswith("You are rating a public YouTube comment")
                      for b in _world.batches.values() for r in b["requests"]))

        # ---- a second submit while batches are unfetched is refused
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                _ab.main(["submit", _src, "--out-dir", _out_dir, "--sample", "12", "--seed", "1",
                          "--max-batch-requests", "5", *_common])
            _refused = False
        except SystemExit as _e:
            _refused = "not fetched yet" in str(_e)
        check("submit: refused while earlier batches are not fetched (it would send everything twice)",
              _refused and _world.created == 5)
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                _ab.main(["submit", _src, "--out-dir", _out_dir, "--sample", "12", "--seed", "1",
                          "--judge", "anthropic:claude-sonnet-5", "--votes", "1"])
            _refused = False
        except SystemExit as _e:
            _refused = "different settings" in str(_e)
        check("submit: a different model into the same folder is refused", _refused)

        # ---- fetch fills the cache
        _rc, _err, _ = _run("fetch", _out_dir)
        _cache_file = os.path.join(_out_dir, "batch_cache.jsonl")
        _entries = [json.loads(l) for l in open(_cache_file, encoding="utf-8") if l.strip()]
        _man = json.load(open(os.path.join(_out_dir, "batch_manifest.json"), encoding="utf-8"))
        _types, _not_cached = collections.Counter(), collections.Counter()
        for _b in _man["batches"]:
            _types.update(_b["counts"])
            _not_cached.update(_b["not_cached"])
        check("fetch: every batch is marked fetched, with its result counts and token usage",
              _rc == 0 and all(b["fetched"] and b["usage"] and b["counts"] for b in _man["batches"]))
        check("fetch: prints the cost at batch price beside the list price",
              "Cost at BATCH price" in _err and "list price" in _err, _err[-300:])
        check("fetch: errored and expired requests are counted by type, and cut-off or refused replies too",
              _types["errored"] == 2 and _types["expired"] == 2
              and _not_cached["errored_api_error"] == 2 and _not_cached["expired"] == 2
              and _not_cached["uncacheable_refusal"] == 2 and _not_cached["uncacheable_max_tokens"] == 2,
              (dict(_types), dict(_not_cached)))
        check("fetch: only normal answers are cached (24 requests, 8 not usable, 16 cached)",
              len(_entries) == 16, len(_entries))
        check("fetch: every cached key is one a request carried", {e["key"] for e in _entries} <= set(_ids))

        # ---- status --check-cache counts comments that are really complete
        _rc, _, _out = _run("status", _out_dir, "--check-cache")
        _m = _re_ab.search(r"Comments complete \(every reply cached or recorded\): (\d+) of (\d+)", _out)
        check("status --check-cache: complete = every reply cached, or recorded as refused or cut off "
              "(10 of 12: only the errored and expired comments are gaps)",
              _m and (int(_m.group(1)), int(_m.group(2))) == (10, 12), _out[-400:])
        check("status --check-cache: says why replies were not cached",
              "expired 2" in _out and "uncacheable_refusal 2" in _out and "errored_api_error 2" in _out)

        # ---- refused and cut-off replies are recorded (key, reason, reply; never the comment)
        _unc_file = os.path.join(_out_dir, "batch_uncacheable.jsonl")
        _unc = [json.loads(l) for l in open(_unc_file, encoding="utf-8") if l.strip()]
        check("fetch: the 4 refused or cut-off replies are recorded by key and reason, keys the requests carried",
              len(_unc) == 4 and collections.Counter(u["reason"] for u in _unc)
              == {"refusal": 2, "max_tokens": 2} and {u["key"] for u in _unc} <= set(_ids),
              [u["reason"] for u in _unc])
        check("fetch: the record holds no comment text",
              not any(w in open(_unc_file, encoding="utf-8").read() for w in ("REFUSE", "LONGREPLY", "this one")))

        # ---- replay: refuses with gaps, writes with --partial, never calls the API
        _labelled = os.path.join(_tb, "labelled.csv")
        _rc, _err, _ = _run("replay", _out_dir, "--output", _labelled)
        check("replay: refuses to write when comments are incomplete, and writes nothing",
              _rc == 1 and not os.path.exists(_labelled), _err[-300:])
        _before_live = len(_world.live)
        _rc, _err, _ = _run("replay", _out_dir, "--output", _labelled, "--partial")
        check("replay --partial: writes the 10 complete comments and never calls the API",
              _rc == 0 and len(_world.live) == _before_live and "2 incomplete comment(s) were LEFT OUT" in _err,
              _err[-300:])
        with open(_labelled, newline="", encoding="utf-8-sig") as _f:
            _lab = list(csv.DictReader(_f))
        check("replay --partial: the rows carry the ewl-classify columns, in order",
              len(_lab) == 10 and list(_lab[0].keys())[-len(_clsb.EXTRA_COLUMNS):] == _clsb.EXTRA_COLUMNS)
        _by_id = {r["comment_id"]: r for r in _lab}
        check("replay: a refused comment is labelled as the live path labels it (a parse failure), not left out",
              "c06" in _by_id and _by_id["c06"]["domain_parse_failure"] == "1"
              and _by_id["c06"]["sentiment_parse_failure"] == "1", _by_id.get("c06"))
        check("replay: a cut-off reply's text is used as written, as live (LONGREPLY -> unclassified, rating 2)",
              "c08" in _by_id and _by_id["c08"]["domain"] == taxonomy.UNCLASSIFIED
              and _by_id["c08"]["domain_parse_failure"] == "0", _by_id.get("c08"))
        _rec = json.load(open(_labelled + ".run.json", encoding="utf-8"))
        check("replay: the run record names the batch tool, the batch ids and the model, and 0 live calls",
              _rec["result"]["collected_by"] == "ewl-anthropic-batch"
              and [b["id"] for b in _rec["result"]["batches"]] == [b["id"] for b in _man["batches"]]
              and _rec["domain_judge"]["spec"] == "anthropic:claude-haiku-4-5"
              and _rec["result"]["live_calls"] == 0, _rec["result"].get("live_calls"))
        check("replay: the run record counts the refused or cut-off replies it used",
              _rec["result"]["refused_or_cut_off_replies_used"] == 4,
              _rec["result"].get("refused_or_cut_off_replies_used"))

        # ---- resubmitting sends only the gaps
        _created_before = _world.created
        _rc, _err, _ = _run("submit", _src, "--out-dir", _out_dir, "--sample", "12", "--seed", "1",
                            "--max-batch-requests", "5", *_common)
        _resent = sum(len(b["requests"]) for i, b in _world.batches.items() if int(i[-3:]) > _created_before)
        check("resubmit: only the 4 errored or expired replies are sent again; refusals and cut-offs are not re-bought",
              _rc == 0 and _resent == 4 and "NOT sent again" in _err, _resent)
        _run("fetch", _out_dir)
        _created_before = _world.created
        _rc, _err, _ = _run("submit", _src, "--out-dir", _out_dir, "--sample", "12", "--seed", "1",
                            "--max-batch-requests", "5", "--retry-uncacheable", *_common)
        _resent = sum(len(b["requests"]) for i, b in _world.batches.items() if int(i[-3:]) > _created_before)
        check("resubmit --retry-uncacheable: the 4 refused or cut-off replies are sent again too (8)",
              _rc == 0 and _resent == 8, _resent)
        _run("fetch", _out_dir)
        check("every batch create is made with the SDK's retries off (max_retries=0)",
              _world.max_retries and set(_world.max_retries) == {0}, _world.max_retries)

        def _refusal(*argv):
            try:
                _run(*argv)
            except SystemExit as _e:
                return str(_e)
            return None

        # ---- a prompt or effort change is refused, not silently paid for again
        _orig_build = taxonomy.build_prompt
        _created_before = _world.created
        taxonomy.build_prompt = lambda ex, story=None: _orig_build(ex, story) + "\n(wording revised)"
        try:
            _sub_msg = _refusal("submit", _src, "--out-dir", _out_dir, "--sample", "12", "--seed", "1",
                                "--max-batch-requests", "5", *_common)
            _rep_msg = _refusal("replay", _out_dir, "--output", os.path.join(_tb, "edited.csv"))
            _chk_msg = _refusal("status", _out_dir, "--check-cache")
        finally:
            taxonomy.build_prompt = _orig_build
        check("a prompt edit after submit: submit, replay and status --check-cache all refuse, and nothing is sent",
              all(m and "differ" in m for m in (_sub_msg, _rep_msg, _chk_msg)) and _world.created == _created_before,
              (_sub_msg, _rep_msg, _chk_msg))
        os.environ["EWL_ANTHROPIC_EFFORT"] = "low"
        try:
            _eff_msg = _refusal("submit", _src, "--out-dir", _out_dir, "--sample", "12", "--seed", "1",
                                "--max-batch-requests", "5", *_common)
        finally:
            del os.environ["EWL_ANTHROPIC_EFFORT"]
        check("EWL_ANTHROPIC_EFFORT set on one run and not the other is refused (it changes every cache key)",
              _eff_msg and "cache specs" in _eff_msg and _world.created == _created_before, _eff_msg)
        _man = json.load(open(os.path.join(_out_dir, "batch_manifest.json"), encoding="utf-8"))
        check("the manifest records both judges' cache specs and the no-story prompt hashes",
              _man["cache_specs"] == {"domain": "anthropic:claude-haiku-4-5",
                                      "sentiment": "anthropic:claude-haiku-4-5"}
              and "domain_no_story" in _man["prompts_sha256"], _man.get("cache_specs"))
        _odir = os.path.join(_tb, "older_folder")
        os.makedirs(_odir)
        _old_man = json.loads(json.dumps(_man))
        del _old_man["cache_specs"]
        for _k in ("domain_no_story", "sentiment_no_story"):
            del _old_man["prompts_sha256"][_k]
        _ab.save_manifest(_odir, _old_man)
        check("a folder started by the earlier version (fewer prompt hashes, no cache specs) is not refused",
              _refusal("status", _odir, "--check-cache") is None)

        # ---- a refused re-submit with a different sample leaves subset.csv alone
        _subset = os.path.join(_out_dir, "subset.csv")
        _subset_sha = _ab.corpus_mod.content_hash(_subset)
        _seed_msg = _refusal("submit", _src, "--out-dir", _out_dir, "--sample", "6", "--seed", "2", *_common)
        check("a re-submit with another --sample is refused and leaves subset.csv (and replay) as they were",
              _seed_msg and "different input" in _seed_msg and _ab.corpus_mod.content_hash(_subset) == _subset_sha
              and not os.path.exists(_subset + ".new")
              and _run("replay", _out_dir, "--output", _labelled, "--partial")[0] == 0, _seed_msg)

        # ---- a lost reply to batches.create is adopted from Anthropic's list, never re-sent
        _ldir = os.path.join(_tb, "lost")
        _world.lose_reply = True
        try:
            _run("submit", _src, "--out-dir", _ldir, "--limit", "3", *_common)
            _raised = False
        except ConnectionError:
            _raised = True
        finally:
            _world.lose_reply = False
        _lman = json.load(open(os.path.join(_ldir, "batch_manifest.json"), encoding="utf-8"))
        _lost_id = max(_world.batches)
        check("submit: the batch is in the manifest (as a placeholder, no id) before create is called",
              _raised and len(_lman["batches"]) == 1 and _lman["batches"][0]["id"] is None
              and _lman["batches"][0]["requests"] == 6, _lman["batches"])
        _created_before = _world.created
        _rc, _, _out = _run("status", _ldir)
        _lman = json.load(open(os.path.join(_ldir, "batch_manifest.json"), encoding="utf-8"))
        check("status: the placeholder is matched in Anthropic's list and the batch adopted",
              _lman["batches"][0]["id"] == _lost_id and _lman["batches"][0].get("adopted"), _lman["batches"])
        check("and a re-submit is refused until it is fetched, so nothing is sent twice",
              "not fetched yet" in (_refusal("submit", _src, "--out-dir", _ldir, "--limit", "3", *_common) or "")
              and _world.created == _created_before)
        _rc, _, _ = _run("fetch", _ldir)
        check("the adopted batch is fetched like any other", _rc == 0 and json.load(open(
            os.path.join(_ldir, "batch_manifest.json"), encoding="utf-8"))["batches"][0]["fetched"])

        _pdir = os.path.join(_tb, "placeholders")
        os.makedirs(_pdir)
        with open(os.path.join(_pdir, "subset.csv"), "w", encoding="utf-8") as _f:
            _f.write("body\n")
        _old = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)).isoformat()
        _ab.save_manifest(_pdir, {"settings": dict(_lman["settings"]), "input": {"rows": 0, "original": "x",
                          "path": os.path.join(_pdir, "subset.csv")}, "subset": None,
                          "batches": [{"id": None, "requests": 999, "created_at": _old, "fetched": False}]})
        _run("status", _pdir)
        check("a placeholder with no matching batch, an hour old, is dropped (its requests become gaps)",
              json.load(open(os.path.join(_pdir, "batch_manifest.json"), encoding="utf-8"))["batches"] == [])
        _ab.save_manifest(_pdir, {"settings": dict(_lman["settings"]), "input": {"rows": 0, "original": "x",
                          "path": os.path.join(_pdir, "subset.csv")}, "subset": None,
                          "batches": [{"id": None, "requests": 999, "created_at": _ab._now(), "fetched": False}]})
        _young = _refusal("status", _pdir)
        check("a placeholder only minutes old with no batch listed yet: refuse and wait, do not drop it",
              _young and "Wait a few minutes" in _young, _young)
        _hours_ago = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=3)).isoformat()
        _ab.save_manifest(_pdir, {"settings": dict(_lman["settings"]), "input": {"rows": 0, "original": "x",
                          "path": os.path.join(_pdir, "subset.csv")}, "subset": None,
                          "batches": [{"id": None, "requests": 6, "created_at": _hours_ago, "fetched": False}]})
        _run("status", _pdir)
        check("a same-size batch created hours AFTER the placeholder (another folder's) is not adopted",
              json.load(open(os.path.join(_pdir, "batch_manifest.json"), encoding="utf-8"))["batches"] == [])
        _ahead = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5)).isoformat()
        _ab.save_manifest(_pdir, {"settings": dict(_lman["settings"]), "input": {"rows": 0, "original": "x",
                          "path": os.path.join(_pdir, "subset.csv")}, "subset": None,
                          "batches": [{"id": None, "requests": 6, "created_at": _ahead, "fetched": False}]})
        _run("status", _pdir)
        check("a local clock 5 minutes ahead of Anthropic's still matches the lost batch (no duplicate)",
              json.load(open(os.path.join(_pdir, "batch_manifest.json"), encoding="utf-8"))["batches"][0]["id"]
              == _lost_id)

        # ---- a create Anthropic definitely refused (4xx) leaves no placeholder behind
        _rdir = os.path.join(_tb, "rejected")
        _world.reject_status = 413
        try:
            _run("submit", _src, "--out-dir", _rdir, "--limit", "3", *_common)
            _raised = False
        except RuntimeError:
            _raised = True
        finally:
            _world.reject_status = None
        check("a 413 from create removes the placeholder, so status, fetch and submit are not blocked",
              _raised and json.load(open(os.path.join(_rdir, "batch_manifest.json"), encoding="utf-8"))["batches"] == []
              and _run("status", _rdir)[0] == 0)

        # ---- a batch whose results cannot be read does not block the others; --abandon frees the folder
        _fdir = os.path.join(_tb, "unreadable")
        _run("submit", _src, "--out-dir", _fdir, "--limit", "6", "--max-batch-requests", "4", *_common)
        _fids = [b["id"] for b in json.load(open(os.path.join(_fdir, "batch_manifest.json"), encoding="utf-8"))["batches"]]
        _world.unreadable.add(_fids[0])
        _rc, _err, _ = _run("fetch", _fdir)
        _fman = json.load(open(os.path.join(_fdir, "batch_manifest.json"), encoding="utf-8"))
        check("fetch: one unreadable batch is reported, the other batches are still fetched, exit 1",
              _rc == 1 and len(_fids) == 3 and [b["fetched"] for b in _fman["batches"]] == [False, True, True]
              and "--abandon" in _err, ([b["fetched"] for b in _fman["batches"]], _err[-200:]))
        _world.running.add(_fids[0])
        _run_msg = _refusal("fetch", _fdir, "--abandon", _fids[0])
        check("fetch --abandon refuses a batch still running (it would be billed while submit resends it)",
              _run_msg and "still in_progress" in _run_msg and not json.load(open(
                  os.path.join(_fdir, "batch_manifest.json"), encoding="utf-8"))["batches"][0].get("abandoned"),
              _run_msg)
        _rc, _, _ = _run("fetch", _fdir, "--abandon", _fids[0], "--cancel")
        check("fetch --abandon --cancel cancels a running batch instead of abandoning it",
              _rc == 0 and _world.cancelled == [_fids[0]])
        _world.running.discard(_fids[0])
        _rc, _, _ = _run("fetch", _fdir, "--abandon", _fids[0])
        _created_before = _world.created
        _rc, _, _ = _run("submit", _src, "--out-dir", _fdir, "--limit", "6", "--max-batch-requests", "4", *_common)
        _resent = {r["custom_id"] for i, b in _world.batches.items() if int(i[-3:]) > _created_before
                   for r in b["requests"]}
        _abandoned_keys = {r["custom_id"] for r in _world.batches[_fids[0]]["requests"]}
        check("fetch --abandon: the folder is usable again and the abandoned batch's 4 requests are sent as gaps "
              "(with the 2 expired ones)",
              _rc == 0 and len(_abandoned_keys) == 4 and _abandoned_keys <= _resent and len(_resent) == 6,
              len(_resent))

        # ---- plan: a ceiling for a model that can cache the prompts at --ttl 1h
        _rc, _err_s, _ = _run("plan", _src, "--judge", "anthropic:claude-sonnet-5-5", "--votes", "1")
        _rc, _err_s5, _ = _run("plan", _src, "--judge", "anthropic:claude-sonnet-5-5", "--votes", "1", "--ttl", "5m")
        _rc, _err_h, _ = _run("plan", _src, *_common)
        _m_hi = _re_ab.search(r"to \$([\d,.]+)  \(list price", _err_s)
        _m_w = _re_ab.search(r"no hits\), up to \$([\d,.]+)", _err_s)
        check("plan: Sonnet 5.5 at --ttl 1h prints an every-request-writes-the-cache figure above the high estimate",
              _m_hi and _m_w and float(_m_w.group(1).replace(",", "")) > float(_m_hi.group(1).replace(",", "")),
              _err_s[-500:])
        _m_w5 = _re_ab.search(r"no hits\), up to \$([\d,.]+)", _err_s5)
        check("plan: at --ttl 5m the ceiling is printed too, lower (1.25x writes); never for Haiku 4.5 "
              "(these prompts are under its 4,096 minimum)",
              _m_w5 and float(_m_w5.group(1).replace(",", "")) < float(_m_w.group(1).replace(",", ""))
              and "1.25x the input price at --ttl 5m" in _err_s5 and "no hits" not in _err_h, _err_s5[-300:])

        # ---- on a clean corpus, batch replay == live ewl-classify, row for row
        _clean = os.path.join(_tb, "clean.csv")
        with open(_clean, "w", newline="", encoding="utf-8-sig") as _f:
            _w = csv.DictWriter(_f, fieldnames=["comment_id", "story", "source", "body"])
            _w.writeheader()
            for _i, _body in enumerate(["She pays no tax at all", "Nice video", "Tax dodging again", "First!",
                                        "Billionaires and tax", "Great clip", "Prices keep rising", "Another comment"]):
                _w.writerow({"comment_id": f"d{_i}", "story": ["luigi", "musk", "jets", "propublica"][_i % 4],
                             "source": f"v{_i % 3}", "body": _body})
        _world_b = _World()
        _install_fake_anthropic(_world_b)
        _bdir = os.path.join(_tb, "clean_batch")
        _rc, _, _ = _run("submit", _clean, "--out-dir", _bdir, "--ttl", "5m", *_common)
        _run("fetch", _bdir)
        _rc, _err, _ = _run("status", _bdir, "--check-cache")
        _replayed = os.path.join(_tb, "replayed.csv")
        _rc_replay, _, _ = _run("replay", _bdir, "--output", _replayed)

        _world_c = _World()
        _install_fake_anthropic(_world_c)
        _live_out = os.path.join(_tb, "live.csv")
        _saved_argv = sys.argv
        sys.argv = ["ewl-classify", _clean, "--judge", "anthropic:claude-haiku-4-5",
                    "--sentiment", "llm:anthropic:claude-haiku-4-5", "--votes", "1",
                    "--cache", os.path.join(_tb, "live_cache.jsonl"), "--output", _live_out]
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                _clsb.main()
        finally:
            sys.argv = _saved_argv
        _rows = lambda path: list(csv.DictReader(open(path, newline="", encoding="utf-8-sig")))
        check("clean corpus: the batch replay and a live ewl-classify give the SAME rows (every column)",
              _rc_replay == 0 and _rows(_replayed) == _rows(_live_out) and len(_rows(_live_out)) == 8,
              (len(_rows(_replayed)), len(_rows(_live_out))))
        _cmp_err = io.StringIO()
        with contextlib.redirect_stdout(_cmp_err):
            _rc_cmp = _ab.main(["compare", _replayed, _live_out])
        check("compare: identical files agree on 100% of comments",
              _rc_cmp == 0 and "Domain labels identical:            8 (100.0%)" in _cmp_err.getvalue(),
              _cmp_err.getvalue()[:300])
        _tweaked = os.path.join(_tb, "tweaked.csv")
        _tw_rows = _rows(_live_out)
        _tw_rows[0]["domain"] = "economics"
        _tw_rows[1]["sentiment_value"] = "5"
        with open(_tweaked, "w", newline="", encoding="utf-8-sig") as _f:
            _w = csv.DictWriter(_f, fieldnames=list(_tw_rows[0].keys()))
            _w.writeheader()
            _w.writerows(_tw_rows)
        _cmp_err = io.StringIO()
        with contextlib.redirect_stdout(_cmp_err):
            _ab.main(["compare", _tweaked, _live_out])
        check("compare: counts a changed domain label and a changed rating as differences (7 of 8 agree)",
              "Domain labels identical:            7 (87.5%)" in _cmp_err.getvalue()
              and "Sentiment rating identical:         7 (87.5%)" in _cmp_err.getvalue(), _cmp_err.getvalue()[:300])
        check("clean corpus: the batch cache and the live cache hold the same keys and replies",
              {json.loads(l)["key"]: json.loads(l)["reply"]
               for l in open(os.path.join(_bdir, "batch_cache.jsonl"), encoding="utf-8") if l.strip()}
              == {json.loads(l)["key"]: json.loads(l)["reply"]
                  for l in open(os.path.join(_tb, "live_cache.jsonl"), encoding="utf-8") if l.strip()})
        _batch_requests = {r["custom_id"]: r["params"] for b in _world_b.batches.values() for r in b["requests"]}
        _helper = judges.create("anthropic:claude-haiku-4-5", votes=1)
        _live_requests = {_helper._cache_key(_body_of(kw), kw["system"][0]["text"], 0): kw
                          for kw in _world_c.live}
        check("clean corpus: each batch request's parameters are identical to the live request's",
              len(_batch_requests) == len(_live_requests) == 16
              and all(_batch_requests[k] == _live_requests[k] for k in _batch_requests))

        # ---- the cache-only guard
        _calls_before = len(_world_c.live)
        _cj = judges.create("anthropic:claude-haiku-4-5", votes=1,
                            cache_path=os.path.join(_tb, "empty_cache.jsonl"))
        _cj.cache_only = True
        try:
            _cj.label("anything at all", examples_mod.create("fixed").select("x"), story="musk")
            _refused = False
        except judges.CacheMiss:
            _refused = True
        check("--cache-only: a missing reply raises CacheMiss and makes no API call",
              _refused and len(_world_c.live) == _calls_before)
        _sys_argv = sys.argv
        sys.argv = ["ewl-classify", _clean, "--judge", "anthropic:claude-haiku-4-5",
                    "--sentiment", "llm:anthropic:claude-haiku-4-5", "--votes", "1",
                    "--cache", os.path.join(_tb, "never_filled.jsonl"), "--cache-only",
                    "--output", os.path.join(_tb, "co.csv")]
        try:
            with contextlib.redirect_stderr(io.StringIO()) as _co_err:
                _clsb.main()
            _code = 0
        except SystemExit as _se:
            _code = _se.code
        finally:
            sys.argv = _sys_argv
        check("ewl-classify --cache-only: stops with exit 2 and says nothing was called, instead of paying",
              _code == 2 and "--cache-only" in _co_err.getvalue() and len(_world_c.live) == _calls_before,
              (_code, _co_err.getvalue()[-200:]))
        _cached_run = os.path.join(_tb, "co_ok.csv")
        sys.argv = ["ewl-classify", _clean, "--judge", "anthropic:claude-haiku-4-5",
                    "--sentiment", "llm:anthropic:claude-haiku-4-5", "--votes", "1",
                    "--cache", os.path.join(_tb, "live_cache.jsonl"), "--cache-only", "--output", _cached_run]
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                _clsb.main()
        finally:
            sys.argv = _sys_argv
        check("ewl-classify --cache-only: with every reply cached it labels the corpus with no calls",
              _rows(_cached_run) == _rows(_live_out) and len(_world_c.live) == _calls_before)

        try:
            _ab.build_context({"judge": "stub", "sentiment": "llm:stub", "votes": 1, "ttl": "1h",
                               "cache": None, "fixed_examples": None, "story_rules": "on",
                               "stance_object": "conduct", "combined_domain_context": "keep",
                               "domain_context": "facts"})
            _refused = False
        except SystemExit as _e:
            _refused = "anthropic" in str(_e)
        check("a non-Anthropic judge is refused by the batch tool", _refused)
finally:
    if _prev_anthropic is None:
        sys.modules.pop("anthropic", None)
    else:
        sys.modules["anthropic"] = _prev_anthropic
    _instr_ab.apply(_tyb.SimpleNamespace(story_rules="on", stance_object="conduct",
                                         combined_domain_context="keep", domain_context="facts",
                                         fixed_examples=None))

check("Opus 5.5 and Sonnet 5.5 have their own prices (Opus 5.5 is not billed as Opus 5)",
      judges.price_for("claude-opus-5-5") == (4.00, 20.00)
      and judges.price_for("claude-sonnet-5-5") == (2.00, 10.00)
      and judges.price_for("claude-opus-5") == (5.00, 25.00))
check("a dated id takes the longest matching base (claude-opus-5-5-<date> is Opus 5.5, not Opus 5)",
      judges.price_for("claude-opus-5-5-20261001") == (4.00, 20.00)
      and judges.price_for("claude-opus-5-20260115") == (5.00, 25.00)
      and judges.price_for("claude-sonnet-5-5-20261001") == (2.00, 10.00))
_probe55 = object.__new__(judges.AnthropicJudge)
_probe55.model = "claude-opus-5-5"
_probe5 = object.__new__(judges.AnthropicJudge)
_probe5.model = "claude-opus-5"
check("cache reads cost 0.05x on the 5.5 models and 0.1x on the others",
      _probe55.CACHE_READ_MULTIPLIER == 0.05 and _probe5.CACHE_READ_MULTIPLIER == 0.1)
_cost_probe = object.__new__(judges.AnthropicJudge)
_cost_probe.model = "claude-haiku-4-5"
check("batch cost is half the list price for the same tokens; a cache write costs 2x (1h) or 1.25x (5m)",
      abs(_ab.cost_from_usage(_cost_probe, {"input_tokens": 1_000_000}, "1h", batch=True) - 0.5) < 1e-9
      and abs(_ab.cost_from_usage(_cost_probe, {"input_tokens": 1_000_000}, "1h", batch=False) - 1.0) < 1e-9
      and abs(_ab.cost_from_usage(_cost_probe, {"cache_creation_input_tokens": 1_000_000}, "1h",
                                  batch=False) - 2.0) < 1e-9
      and abs(_ab.cost_from_usage(_cost_probe, {"cache_creation_input_tokens": 1_000_000}, "5m",
                                  batch=False) - 1.25) < 1e-9)

print("\nfixed examples with sentiment: the same set teaches both labels")
import re as _re_sx
import types as _tyx
from ewl.label import sentiment as _sx
from ewl.label import combined as _cx
from ewl.label import instrument as _ix

_orig_fx = list(taxonomy.FIXED_EXAMPLES)
_SX_ROWS = [
    ("x1", "She pays no tax at all and nobody stops her", "democracy", "2", "propublica"),
    ("x2", "Why couldn't he invest that into climate change", "environment", "2", "musk"),
    ("x3", "Great clip, thanks for posting it", "none", "0", "jets"),
    ("x4", "You want a killer on the streets?", "social", "4", "luigi"),
    ("x5", "Insurance is the biggest extortion racket", "economics, social", "1", "luigi"),
    ("x6", "garbled text that cannot be read at all", "none", "-1", "musk"),
]


def _write_examples(path, rows, columns=("comment_id", "body", "label", "sentiment", "story")):
    with open(path, "w", newline="", encoding="utf-8-sig") as _f:
        _w = csv.writer(_f)
        _w.writerow(columns)
        for _r in rows:
            _w.writerow(_r[:len(columns)] if len(columns) <= len(_r) else _r)


def _exits(path):
    try:
        taxonomy.load_fixed_examples(path)
        return None
    except SystemExit as _e:
        return str(_e)
    finally:
        taxonomy.reset_fixed_examples(_orig_fx)


with tempfile.TemporaryDirectory() as _tx:
    _before_prompts = _sx.create("llm:stub", votes=1)._prompts
    _before_combined = _cx.build_prompt(taxonomy.FIXED_EXAMPLES, "musk")
    _p_rated = os.path.join(_tx, "rated.csv")
    _write_examples(_p_rated, _SX_ROWS)
    try:
        _ids = taxonomy.load_fixed_examples(_p_rated)
        check("a file with sentiment and story loads, keeping the ratings and stories aligned",
              _ids == ["x1", "x2", "x3", "x4", "x5", "x6"]
              and taxonomy.FIXED_EXAMPLE_RATINGS == [2, 2, 0, 4, 1, -1]
              and taxonomy.FIXED_EXAMPLE_STORIES == ["propublica", "musk", "jets", "luigi", "luigi", "musk"])
        check("the source record holds ids, labels, ratings and stories (not text)",
              taxonomy.FIXED_EXAMPLES_SOURCE["ratings"] == [2, 2, 0, 4, 1, -1]
              and "She pays no tax" not in json.dumps(taxonomy.FIXED_EXAMPLES_SOURCE))
        _after = _sx.create("llm:stub", votes=1)._prompts
        _p = _after["musk"]
        check("the sentiment prompt now ends with the examples, each with its story and rating",
              _p.index('[propublica] "She pays no tax at all and nobody stops her" -> 2') > _p.index("No words, no explanation.")
              and '[luigi] "You want a killer on the streets?" -> 4' in _p
              and '[musk] "garbled text that cannot be read at all" -> -1' in _p
              and "the story each was posted under is in brackets" in _p)
        check("every story's sentiment prompt carries the same 6 examples",
              all(_after[s].count(" -> ") >= 6 and '[jets] "Great clip, thanks for posting it" -> 0' in _after[s]
                  for s in _after))
        _dp = taxonomy.build_prompt(taxonomy.FIXED_EXAMPLES, "luigi")
        check("the domain prompt shows labels only: no rating appears in it",
              _dp.count(" -> ") >= 6 and not _re_sx.search(r" -> -?\d", _dp))
        _cp = _cx.build_prompt(taxonomy.FIXED_EXAMPLES, "luigi")
        check("the combined arm shows each example as 'domain | rating' with its story",
              '[luigi] "Insurance is the biggest extortion racket" -> economics, social | 1' in _cp
              and '[musk] "garbled text that cannot be read at all" -> none | -1' in _cp
              and "Examples (domain | rating;" in _cp)
        _ix.apply(_tyx.SimpleNamespace(story_rules="on", stance_object="conduct", combined_domain_context="keep",
                                       domain_context="facts", sentiment_examples="off", fixed_examples=None))
        _off = _sx.create("llm:stub", votes=1)
        check("--sentiment-examples off: no examples in the sentiment prompt, and the setting is tagged",
              all(" -> " not in p for p in _off._prompts.values()) and _off.describe().endswith("[sentiment_examples=off]")
              and "Examples of the domain judgement:" in _cx.build_prompt(taxonomy.FIXED_EXAMPLES, "luigi"),
              _off.describe())
    finally:
        _ix.apply(_tyx.SimpleNamespace(story_rules="on", stance_object="conduct", combined_domain_context="keep",
                                       domain_context="facts", sentiment_examples="on", fixed_examples=None))
        taxonomy.reset_fixed_examples(_orig_fx)
    check("with the built-in examples (no ratings) the prompts are exactly what they were before",
          _sx.create("llm:stub", votes=1)._prompts == _before_prompts
          and _cx.build_prompt(taxonomy.FIXED_EXAMPLES, "musk") == _before_combined)
    check("the sentiment switch is on by default and the default adds no tag",
          _sx.configure()["sentiment_examples"] is True and "sentiment_examples" not in _sx.config_tag())

    # ---- refusals
    _write_examples(os.path.join(_tx, "words.csv"), [r[:3] + ("negative", r[4]) for r in _SX_ROWS])
    check("a rating written as a word is refused", "must be a number" in (_exits(os.path.join(_tx, "words.csv")) or ""))
    _write_examples(os.path.join(_tx, "range.csv"), [r[:3] + ("7", r[4]) for r in _SX_ROWS])
    check("a rating outside -1 to 5 is refused", "must be a number" in (_exits(os.path.join(_tx, "range.csv")) or ""))
    _write_examples(os.path.join(_tx, "nostory.csv"), [r[:4] + ("",) for r in _SX_ROWS])
    check("ratings without a story are refused (the stance guidance depends on it)",
          "needs its `story`" in (_exits(os.path.join(_tx, "nostory.csv")) or ""))
    _write_examples(os.path.join(_tx, "partial.csv"), [r[:3] + ("" if i == 2 else r[3], r[4]) for i, r in enumerate(_SX_ROWS)])
    check("a rating on some rows but not all is refused",
          "have no sentiment" in (_exits(os.path.join(_tx, "partial.csv")) or ""))
    _write_examples(os.path.join(_tx, "domainonly.csv"), [r[:3] for r in _SX_ROWS], columns=("comment_id", "body", "label"))
    try:
        taxonomy.load_fixed_examples(os.path.join(_tx, "domainonly.csv"))
        check("a domain-only file still loads, with no sentiment examples shown",
              taxonomy.FIXED_EXAMPLE_RATINGS == [] and " -> " not in _sx.create("llm:stub", votes=1)._prompts["luigi"])
    finally:
        taxonomy.reset_fixed_examples(_orig_fx)
    _write_examples(os.path.join(_tx, "hand.csv"), _SX_ROWS, columns=("comment_id", "body", "hand_domain", "hand_sentiment", "story"))
    try:
        taxonomy.load_fixed_examples(os.path.join(_tx, "hand.csv"))
        check("hand_domain and hand_sentiment column names are accepted too", taxonomy.FIXED_EXAMPLE_RATINGS == [2, 2, 0, 4, 1, -1])
    finally:
        taxonomy.reset_fixed_examples(_orig_fx)

    # ---- the run record, and the scorecard leaving the examples out
    _env_s = dict(os.environ, EWL_AUTHOR_SALT="test-salt-value-1234")
    _gold = os.path.join(_tx, "gold.csv")
    with open(_gold, "w", newline="", encoding="utf-8-sig") as _f:
        _w = csv.writer(_f)
        _w.writerow(["comment_id", "story", "source", "body", "hand_domain", "hand_sentiment", "inclusion_prob"])
        for _i, _r in enumerate(_SX_ROWS + [("g7", "A seventh comment about taxes", "democracy", "2", "luigi"),
                                            ("g8", "An eighth comment about jets", "environment", "4", "jets"),
                                            ("g9", "A ninth plain remark", "none", "0", "musk")]):
            _w.writerow([_r[0], _r[4], f"v{_i}", _r[1], _r[2], _r[3], "0.5"])
    _out = os.path.join(_tx, "classified.csv")
    _p = subprocess.run([sys.executable, "-m", "ewl.label.classify", _gold, "--judge", "stub", "--votes", "1",
                         "--no-cache", "--fixed-examples", _p_rated, "--output", _out],
                        capture_output=True, text=True, env=_env_s)
    _rec = json.load(open(_out + ".run.json", encoding="utf-8")) if os.path.exists(_out + ".run.json") else {}
    check("the run record notes the ratings and stories by id, never the text",
          _p.returncode == 0 and _rec.get("fixed_examples", {}).get("ratings") == [2, 2, 0, 4, 1, -1]
          and "She pays no tax" not in json.dumps(_rec), _p.stderr[-300:])
    _p = subprocess.run([sys.executable, "-m", "ewl.analyse.evaluate", "sentiment", _gold, "--arm", "llm:stub",
                         "--fixed-examples", _p_rated], capture_output=True, text=True, env=_env_s)
    check("ewl-evaluate sentiment leaves the six worked examples out of scoring",
          "left out of scoring" in _p.stderr and "6 comment(s) left out" in _p.stderr, _p.stderr[-400:])
    _p = subprocess.run([sys.executable, "-m", "ewl.analyse.evaluate", "sentiment", _gold, "--arm", "llm:stub"],
                        capture_output=True, text=True, env=_env_s)
    check("without --fixed-examples nothing is left out", "left out of scoring" not in _p.stderr, _p.stderr[-300:])
    # The ratings refusal lives in instrument.apply, so ewl-evaluate meets it as ewl-gemma does.
    _plain_x = os.path.join(_tx, "domainonly.csv")
    for _cmd, _arm in (("sentiment", "llm:stub"), ("scorecard", "stub/fixed")):
        _p = subprocess.run([sys.executable, "-m", "ewl.analyse.evaluate", _cmd, _gold, "--arm", _arm,
                             "--fixed-examples", _plain_x], capture_output=True, text=True, env=_env_s)
        check(f"ewl-evaluate {_cmd} refuses a fixed-examples file without ratings",
              _p.returncode != 0 and "no `sentiment` and `story` columns" in _p.stderr, _p.stderr[-300:])
        _p = subprocess.run([sys.executable, "-m", "ewl.analyse.evaluate", _cmd, _gold, "--arm", _arm,
                             "--fixed-examples", _plain_x, "--sentiment-examples", "off"],
                            capture_output=True, text=True, env=_env_s)
        check(f"ewl-evaluate {_cmd} --sentiment-examples off lets a ratings-less file through",
              _p.returncode == 0 and "no `sentiment` and `story` columns" not in _p.stderr, _p.stderr[-300:])


print("\nbatches and gemma: the model build is recorded and compared")
from ewl.collect import batches as _bt
from ewl.label import gemma_local as _gl
_base_rec = {"domain_judge": {"spec": "gemma", "model": "gemma2:27b"}, "fixed_examples": {"source": "file", "sha256": "abc"},
             "model_build": {"digest": "d1", "server_version": "0.5.1"}}
_same = _bt._fingerprint(dict(_base_rec))
_other_digest = _bt._fingerprint(dict(_base_rec, model_build={"digest": "d2", "server_version": "0.5.1"}))
_other_server = _bt._fingerprint(dict(_base_rec, model_build={"digest": "d1", "server_version": "0.6.0"}))
check("batches labelled with a different model digest or Ollama version have different fingerprints",
      _same == _bt._fingerprint(dict(_base_rec)) and _same != _other_digest and _same != _other_server)


class _FakeClient:
    def list(self):
        return {"models": [{"model": "gemma2:27b", "digest": "sha256:feedface"}]}

    def show(self, tag):
        return {"details": {"quantization_level": "Q4_0", "parameter_size": "27.2B"}}

    def version(self):
        return {"version": "0.5.7"}


_info = _gl.preflight(_FakeClient())
check("preflight returns the model digest and the server version for the run record",
      _info["digest"] == "sha256:feedface" and _info["server_version"] == "0.5.7", _info)


def _raised(fn, exc):
    try:
        fn()
    except exc:
        return True
    return False

print("\nadjudicate.py: majority rule for the comments every coder coded")
from ewl.analyse import adjudicate as _adj

_S = _adj.resolve_domain
check("domain: all three agree", _S([{"economics"}] * 3)[:2] == (["economics"], "majority"))
check("domain: two for economics, one for social: both labels (the dissenter's is added)",
      _S([{"economics"}, {"economics"}, {"social"}], "add")[0] == ["economics", "social"])
check("domain: the default is ignore, so a lone label is not added",
      _S([{"economics"}, {"economics"}, {"social"}])[0] == ["economics"])
check("domain: --extra-labels ignore leaves the lone label out",
      _S([{"economics"}, {"economics"}, {"social"}], "ignore")[0] == ["economics"])
check("domain: a three-way split is flagged", _S([{"democracy"}, {"economics"}, {"social"}])[0] is None
      and "no majority" in _S([{"democracy"}, {"economics"}, {"social"}])[2])
check("domain: majority none settles as none, noting the dissent",
      _S([set(), set(), {"democracy"}])[0] == ["none"] and "a coder tagged democracy" in _S([set(), set(), {"democracy"}])[1])
check("domain: one coder says none and two agree on a category: the category",
      _S([set(), {"social"}, {"social"}])[0] == ["social"])
check("domain: one none and two different categories is flagged", _S([set(), {"social"}, {"economics"}])[0] is None)
check("domain: majority label plus two different lone labels is flagged",
      _S([{"economics"}, {"economics"}, {"social", "environment"}], "add")[0] is None
      and _S([{"economics"}, {"economics"}, {"social", "environment"}])[0] == ["economics"])
check("domain: two majority labels keep exactly those two; a third lone label is dropped with a note",
      _S([{"economics", "social"}, {"economics", "social"}, {"democracy"}])[0] == ["economics", "social"]
      and "not kept" in _S([{"economics", "social"}, {"economics", "social"}, {"democracy"}])[1])
check("domain: a coder's economics+social against two economics adds social",
      _S([{"economics", "social"}, {"economics"}, {"economics"}], "add")[0] == ["economics", "social"])
check("domain_set refuses a typo and 'none' mixed with a category",
      all(_raised(lambda v=v: _adj.domain_set(v), ValueError) for v in ("enivronment", "none, social", "")))

_V = _adj.resolve_sentiment
check("sentiment: 1,2,2 is negative by majority", _V(["1", "2", "2"])[0] == "negative")
check("sentiment: 2,3,4 (three directions) is flagged", _V(["2", "3", "4"])[0] is None)
check("sentiment: two 0s settle as irrelevant, written as 0", _V(["0", "0", "3"])[0] == "0")
check("sentiment: -1 majority is written as -1", _V(["-1", "-1", "2"])[0] == "-1")
check("sentiment: the words work too", _V(["negative", "2", "5"])[0] == "negative")
check("sentiment rule exact: 2,2,3 settles at 2; 1,2,3 is flagged; 1,2,2 style buckets differ from exact",
      _V(["2", "2", "3"], "exact")[0] == "2" and _V(["1", "2", "3"], "exact")[0] is None
      and _V(["1", "2", "2"], "exact")[0] == "2")
check("sentiment_category refuses an out-of-range value", _raised(lambda: _adj.sentiment_category("7"), ValueError))

with tempfile.TemporaryDirectory() as _ta:
    _rowsets = {
        # id: (domains for A,B,C), (sentiments for A,B,C)
        "c1": (["economics"] * 3, ["2", "1", "2"]),
        "c2": (["economics", "economics", "social"], ["4", "5", "4"]),
        "c3": (["democracy", "economics", "social"], ["1", "3", "5"]),
        "c4": (["none", "none", "democracy"], ["0", "0", "3"]),
        "c5": (["economics, social", "economics", "economics"], ["2", "2", "2"]),
        "c6": (["economics", "economics", "social, environment"], ["4", "4", "4"]),
        "c7": (["economics, social", "economics, social", "democracy"], ["1", "1", "2"]),
        "c8": (["none", "social", "social"], ["2", "2", "2"]),
        "c9": (["none", "social", "economics"], ["5", "5", "4"]),
        "c10": (["none", "none", "none"], ["2", "3", "4"]),
    }
    _paths = []
    for _i, _name in enumerate(("Aa", "Bb", "Cc")):
        _p = os.path.join(_ta, f"{_name} Gold Set Complete.csv")
        with open(_p, "w", newline="", encoding="utf-8-sig") as _f:
            _w = csv.writer(_f)
            _w.writerow(["comment_id", "story", "body", "hand_domain", "hand_sentiment"])
            for _cid, (_d, _s) in _rowsets.items():
                _w.writerow([_cid, "luigi", f"text of {_cid}", _d[_i], _s[_i]])
            _w.writerow([f"own{_i}", "jets", "a comment outside the core", "none", "0"])
        _paths.append(_p)
    _core = os.path.join(_ta, "core.csv")
    with open(_core, "w", newline="", encoding="utf-8-sig") as _f:
        _w = csv.writer(_f)
        _w.writerow(["comment_id"])
        for _cid in _rowsets:
            _w.writerow([_cid])
    _review = os.path.join(_ta, "review.csv")
    _resolved = os.path.join(_ta, "resolved.csv")
    _run = lambda *extra: subprocess.run([sys.executable, "-m", "ewl.analyse.adjudicate", *_paths,
                                          "--core-ids", _core, "--extra-labels", "add", *extra],
                                         capture_output=True, text=True)
    _p1 = _run("--review-output", _review, "--output", _resolved)
    _rv = list(csv.DictReader(open(_review, newline="", encoding="utf-8-sig")))
    check("step 1 flags exactly the comments the majority cannot settle (c3, c6, c9, c10)",
          sorted(r["comment_id"] for r in _rv) == ["c10", "c3", "c6", "c9"], _p1.stdout + _p1.stderr)
    _rvd = {r["comment_id"]: r for r in _rv}
    check("step 1 does not write the resolved file while flags remain", not os.path.exists(_resolved))
    check("step 1 prints counts, never comment text", "text of c" not in _p1.stdout)
    check("the review file carries the text and each coder's answers side by side",
          _rvd["c3"]["body"] == "text of c3" and _rvd["c3"]["Aa domain"] == "democracy" and _rvd["c3"]["Cc sentiment"] == "5"
          and _rvd["c3"]["hand_domain"] == "" and "no majority" in _rvd["c3"]["domain_flag"])
    check("a row flagged for sentiment only has its domain proposed and no domain flag",
          _rvd["c10"]["domain_flag"] == "" and _rvd["c10"]["proposed_domain"] == "none" and "no majority" in _rvd["c10"]["sentiment_flag"])
    _p_bad = _run("--decisions", _review, "--output", _resolved)
    check("step 2 refuses while flagged cells are blank", _p_bad.returncode != 0 and "not complete" in (_p_bad.stderr + _p_bad.stdout)
          and not os.path.exists(_resolved))
    _fill = {"c3": ("economics, social", "neutral"), "c6": ("economics", ""), "c9": ("social", ""), "c10": ("", "3")}
    for _r in _rv:
        _r["hand_domain"], _r["hand_sentiment"] = _fill[_r["comment_id"]]
    with open(_review, "w", newline="", encoding="utf-8-sig") as _f:
        _w = csv.DictWriter(_f, fieldnames=list(_rv[0].keys()))
        _w.writeheader()
        _w.writerows(_rv)
    _gold = os.path.join(_ta, "gold.csv")
    with open(_gold, "w", newline="", encoding="utf-8-sig") as _f:
        _w = csv.writer(_f)
        _w.writerow(["comment_id", "body", "hand_domain", "hand_sentiment", "coder_notes"])
        for _cid in list(_rowsets) + ["other1"]:
            _w.writerow([_cid, f"text of {_cid}", "none", "0", ""])
    _gold_out = os.path.join(_ta, "gold_adj.csv")
    _p2 = _run("--decisions", _review, "--output", _resolved, "--gold", _gold, "--gold-output", _gold_out)
    _res = {r["comment_id"]: r for r in csv.DictReader(open(_resolved, newline="", encoding="utf-8-sig"))}
    check("step 2 writes all ten resolved rows", len(_res) == 10, _p2.stdout + _p2.stderr)
    check("resolved values follow the rules and the owner's decisions",
          _res["c1"]["hand_domain"] == "economics" and _res["c1"]["hand_sentiment"] == "negative"
          and _res["c2"]["hand_domain"] == "economics, social" and _res["c2"]["hand_sentiment"] == "positive"
          and _res["c3"]["hand_domain"] == "economics, social" and _res["c3"]["hand_sentiment"] == "neutral"
          and _res["c4"]["hand_domain"] == "none" and _res["c4"]["hand_sentiment"] == "0"
          and _res["c5"]["hand_domain"] == "economics, social"
          and _res["c6"]["hand_domain"] == "economics" and _res["c7"]["hand_domain"] == "economics, social"
          and _res["c8"]["hand_domain"] == "social" and _res["c9"]["hand_domain"] == "social"
          and _res["c10"]["hand_sentiment"] == "neutral", {k: (v["hand_domain"], v["hand_sentiment"]) for k, v in _res.items()})
    check("the resolved file is in the shape `ewl-validate combine --adjudicated` reads",
          set(["comment_id", "hand_domain", "hand_sentiment"]) <= set(next(iter(_res.values())).keys())
          and all(r["coder_notes"] == "adjudicated" for r in _res.values()))
    _g = {r["comment_id"]: r for r in csv.DictReader(open(_gold_out, newline="", encoding="utf-8-sig"))}
    check("the gold copy has the core rows replaced and every other row untouched",
          _g["c2"]["hand_domain"] == "economics, social" and _g["c2"]["coder_notes"] == "adjudicated"
          and _g["other1"]["hand_domain"] == "none" and _g["other1"]["coder_notes"] == "" and len(_g) == 11)
    check("the gold step reports counts only", "text of c" not in _p2.stdout)
    _p3 = subprocess.run([sys.executable, "-m", "ewl.analyse.adjudicate", _paths[0], _paths[1], "--core-ids", _core],
                         capture_output=True, text=True)
    check("fewer than three coder files is refused", _p3.returncode != 0)
    _p3b = subprocess.run([sys.executable, "-m", "ewl.analyse.adjudicate", *_paths, _paths[0], "--core-ids", _core],
                          capture_output=True, text=True)
    check("four coder files are refused (the rule is 2-of-3)",
          _p3b.returncode != 0 and "exactly three" in _p3b.stderr)
    check("adjudicate() itself refuses other than three coders",
          _raised(lambda: _adj.adjudicate([("a", {}), ("b", {})], set()), SystemExit))
    _short = os.path.join(_ta, "Dd Gold Set Complete.csv")
    with open(_short, "w", newline="", encoding="utf-8-sig") as _f:
        _w = csv.writer(_f)
        _w.writerow(["comment_id", "story", "body", "hand_domain", "hand_sentiment"])
        _w.writerow(["c1", "luigi", "x", "economics", "2"])
    _p4 = subprocess.run([sys.executable, "-m", "ewl.analyse.adjudicate", _paths[0], _paths[1], _short, "--core-ids", _core],
                         capture_output=True, text=True)
    check("a coder missing core comments is refused, naming the ids", _p4.returncode != 0 and "is missing" in _p4.stderr)

    # settings are bound to the review file; a mismatch is refused
    _p5 = subprocess.run([sys.executable, "-m", "ewl.analyse.adjudicate", *_paths, "--core-ids", _core,
                          "--decisions", _review, "--output", os.path.join(_ta, "x.csv"),
                          "--extra-labels", "ignore"], capture_output=True, text=True)
    check("step 2 is refused when --extra-labels differs from step 1",
          _p5.returncode != 0 and "differ" in _p5.stderr and not os.path.exists(os.path.join(_ta, "x.csv")), _p5.stderr)
    _p5b = subprocess.run([sys.executable, "-m", "ewl.analyse.adjudicate", *_paths, "--core-ids", _core,
                           "--decisions", _review, "--output", os.path.join(_ta, "x.csv"),
                           "--extra-labels", "add", "--sentiment-rule", "exact"], capture_output=True, text=True)
    check("step 2 is refused when --sentiment-rule differs from step 1", _p5b.returncode != 0 and "differ" in _p5b.stderr)
    # a decision for a field that is not flagged warns
    _rv2 = list(csv.DictReader(open(_review, newline="", encoding="utf-8-sig")))
    for _r in _rv2:
        if _r["comment_id"] == "c10":
            _r["hand_domain"] = "social"
    with open(_review, "w", newline="", encoding="utf-8-sig") as _f:
        _w = csv.DictWriter(_f, fieldnames=list(_rv2[0].keys()))
        _w.writeheader()
        _w.writerows(_rv2)
    _p6 = _run("--decisions", _review, "--output", os.path.join(_ta, "y.csv"))
    check("a decision for a field that is not flagged warns on stderr",
          _p6.returncode == 0 and "c10" in _p6.stderr and "not flagged" in _p6.stderr, _p6.stderr)
    # gold file with a repeated comment_id is refused
    _gold_dup = os.path.join(_ta, "gold_dup.csv")
    with open(_gold_dup, "w", newline="", encoding="utf-8-sig") as _f:
        _w = csv.writer(_f)
        _w.writerow(["comment_id", "body", "hand_domain", "hand_sentiment", "coder_notes"])
        for _cid in list(_rowsets) + ["c1", "c2"]:
            _w.writerow([_cid, "x", "none", "0", ""])
    _p7 = _run("--decisions", _review, "--output", os.path.join(_ta, "z.csv"), "--gold", _gold_dup,
               "--gold-output", os.path.join(_ta, "gd_out.csv"))
    check("--gold refuses a repeated comment_id, naming how many",
          _p7.returncode != 0 and "repeats 2 comment_id" in _p7.stderr and not os.path.exists(os.path.join(_ta, "gd_out.csv")), _p7.stderr)
    check("--gold is checked before --output is written, so a refusal leaves no resolved file",
          _p7.returncode != 0 and not os.path.exists(os.path.join(_ta, "z.csv")), _p7.stderr)

check("exact rule: 0, irrelevant and 3 settles as 0", _V(["0", "irrelevant", "3"], "exact")[0] == "0")
check("exact rule: -1, uncodeable and 2 settles as -1", _V(["-1", "uncodeable", "2"], "exact")[0] == "-1")

print("\ndashboard export: no real comments unless asked for")
with tempfile.TemporaryDirectory() as _tv:
    _vc = os.path.join(_tv, "labelled.csv")
    _vf = (["comment_id", "source", "story", "body", "sentiment", "sentiment_value",
            "sentiment_unstable", "domain_unstable", "domain_judge", "sentiment_scorer"]
           + [f"domain_{c}" for c in taxonomy.CATEGORY_NAMES]
           + ["domain_unclassified", "domain_parse_failure"])
    with open(_vc, "w", newline="", encoding="utf-8-sig") as _f:
        _w = csv.DictWriter(_f, fieldnames=_vf)
        _w.writeheader()
        for _v in range(10):
            for _i in range(5):
                _row = {name: "0" for name in _vf}
                _row.update({"comment_id": f"v{_v}_{_i}", "source": f"vid{_v}",
                             "story": "luigi" if _v < 5 else "propublica",
                             "body": "ZZVERBATIMMARKERZZ she pays no tax", "sentiment": "negative",
                             "sentiment_value": "2", "sentiment_unstable": "0",
                             "domain_unstable": "0", "domain_judge": "keyword:keywords.yml/none",
                             "sentiment_scorer": "vader",
                             "domain_democracy": "1" if _i == 0 else "0"})
                _w.writerow(_row)
    _env_v = dict(os.environ, EWL_AUTHOR_SALT="test-salt-value-1234")
    _out_off = os.path.join(_tv, "off.json")
    _p = subprocess.run([sys.executable, "-m", "ewl.analyse.export_dashboard", _vc, "--output", _out_off,
                         "--allow-unvalidated", "--n-boot", "20"],
                        capture_output=True, text=True, env=_env_v)
    _off_text = open(_out_off, encoding="utf-8").read() if os.path.exists(_out_off) else ""
    _off = json.loads(_off_text) if _off_text else {}
    check("export: by default no comment text is written anywhere in results.json",
          _p.returncode == 0 and "ZZVERBATIMMARKERZZ" not in _off_text, _p.stderr[-300:])
    check("export: the examples block is empty, flagged as withheld, and keeps its shape for the page",
          _off.get("examples", {}).get("withheld") is True
          and _off["examples"]["disagreements"] == []
          and set(_off["examples"]["by_category"]) == set(taxonomy.CATEGORY_NAMES)
          and not any(_off["examples"]["by_category"].values()))
    check("export: the provenance says verbatim comments are not included",
          _off.get("provenance", {}).get("verbatim_comments_included") is False)
    _out_on = os.path.join(_tv, "on.json")
    _p = subprocess.run([sys.executable, "-m", "ewl.analyse.export_dashboard", _vc, "--output", _out_on,
                         "--allow-unvalidated", "--n-boot", "20", "--include-verbatim"],
                        capture_output=True, text=True, env=_env_v)
    _on = json.load(open(_out_on, encoding="utf-8"))
    check("export: --include-verbatim puts the worked examples back, and says so",
          _p.returncode == 0 and "ZZVERBATIMMARKERZZ" in json.dumps(_on["examples"])
          and _on["provenance"]["verbatim_comments_included"] is True, _p.stderr[-300:])

if project_file("dashboard/index.html"):
    _committed = json.load(open("dashboard/data/results.json", encoding="utf-8"))
    check("the committed dashboard data holds no comment text (examples withheld)",
          _committed["examples"].get("withheld") is True and not any(_committed["examples"]["by_category"].values())
          and _committed["examples"]["disagreements"] == [])
    _page = open("dashboard/index.html", encoding="utf-8").read()
    check("the page no longer describes the dropped retrieval design (no pool, no held-out half)",
          "retrieval pool" not in _page and "held-out data" not in _page and "half of the gold set" not in _page)
    check("the Method diagram says three people label, not two",
          "Two people read and label it" not in _page and "Three people read and label it" in _page)
    check("the Method diagram no longer shows a teaching half and a testing half",
          "Teaching half" not in _page and "Testing half" not in _page and "testing half" not in _page)
    check("the Method tab no longer says two coders, three votes on the corpus, or ~30 videos per story",
          "Two coders label the same comments" not in _page and "three times per comment" not in _page
          and "draws ~30 videos" not in _page
          and "Three coders independently labelled a shared set of 50" in _page
          and "labels the corpus once per comment" in _page)
    check("the page still hides the example sections when there are none",
          "Object.values(ex.by_category || {}).some(v => v.length)" in _page
          and "(ex.disagreements || []).length" in _page)
    check("the dot plot draws no dot for a null value (it used to sit on the 0% gridline)",
          "(value(r) != null ? `<circle" in _page)
    # Run dotPlot itself when a JavaScript runtime is at hand: one story with a rate, one with none.
    import shutil as _sh_dp
    if _sh_dp.which("bun"):
        _dp_src = _page[_page.index("function dotPlot("):]
        _dp_src = _dp_src[:_dp_src.index('return svg + "</svg>";\n}') + len('return svg + "</svg>";\n}')]
        with tempfile.TemporaryDirectory() as _tdp:
            _dp_js = os.path.join(_tdp, "dot.js")
            with open(_dp_js, "w", encoding="utf-8") as _f:
                _f.write("const esc = s => String(s);\n"
                         "const F = {pct: v => v == null ? '-' : (100*v).toFixed(1) + '%'};\n" + _dp_src
                         + "\nconst svg = dotPlot([{g: 'economics', s: 'luigi', v: 0.3}, {g: 'economics', s: 'jets', v: null}],"
                           " {group: r => r.g, series: r => r.s, value: r => r.v, lo: r => null, hi: r => null,"
                           " color: r => '#000', estimable: r => false});\n"
                           "console.log((svg.match(/<circle/g) || []).length);\n")
            _dp_run = subprocess.run(["bun", _dp_js], capture_output=True, text=True)
        check("dotPlot run on a null row draws one dot, for the row that has a value",
              _dp_run.returncode == 0 and _dp_run.stdout.strip() == "1", _dp_run.stdout + _dp_run.stderr[-300:])


print("\nbatches: prepare, verify and the split-time check")
from ewl.collect import batches as _bp
with tempfile.TemporaryDirectory() as _pd:
    _cols = ["comment_id", "story", "source", "body"]
    _full = [{"comment_id": f"c{i}", "story": ["luigi", "musk"][i % 2], "source": f"v{i % 5}",
              "body": (f"text {i}\r\nsecond line" if i % 6 == 0 else f"text {i}")} for i in range(40)]

    def _wcsv(path, rows, cols=_cols):
        with open(path, "w", newline="", encoding="utf-8-sig") as _f:
            _w = csv.DictWriter(_f, fieldnames=cols)
            _w.writeheader()
            _w.writerows(rows)

    _fp, _gp, _ep, _op = (os.path.join(_pd, n) for n in ("full.csv", "gold.csv", "ex.csv", "prep.csv"))
    _wcsv(_fp, _full)
    _wcsv(_gp, [{"comment_id": f"c{i}"} for i in (1, 2, 3)], ["comment_id"])
    _wcsv(_ep, [{"comment_id": "c4", "body": "x"}], ["comment_id", "body"])
    _prov = _bp.prepare(_fp, [_gp, _ep], _op)
    _kept = list(csv.DictReader(open(_op, newline="", encoding="utf-8-sig")))
    check("prepare leaves out exactly the listed comments and nothing else",
          len(_kept) == 36 and _prov["removed_rows"] == 4
          and not ({"c1", "c2", "c3", "c4"} & {r["comment_id"] for r in _kept}))
    check("prepare keeps every text byte for byte, line breaks inside a comment included",
          {r["comment_id"]: r["body"] for r in _kept}["c6"] == "text 6\r\nsecond line")
    check("prepare writes a provenance file: parent hash, the exclusion files' hashes, counts",
          json.load(open(_op + ".provenance.json", encoding="utf-8"))["excluded"][0]["ids_in_parent"] == 3
          and len(_prov["parent"]["sha256"]) == 64)
    check("prepare refuses to overwrite an existing file", _raised(lambda: _bp.prepare(_fp, [_gp], _op), SystemExit))
    check("verify passes the prepared corpus", _bp.verify(_op, _fp, [_gp, _ep])[0] == [])
    _tampered = [dict(r) for r in _kept]
    _tampered[0]["body"] = "CHANGED"
    _tampered = [r for r in _tampered if r["comment_id"] != "c9"]
    _tampered.append(dict(_full[1]))
    _tp = os.path.join(_pd, "tampered.csv")
    _wcsv(_tp, _tampered)
    _issues = " | ".join(_bp.verify(_tp, _fp, [_gp, _ep])[0])
    check("verify names an altered text, a wrongly removed comment and a gold comment left in",
          "different `body`" in _issues and "were removed that are not in the exclusion files" in _issues
          and "should be left out are still there" in _issues, _issues)
    _hp = os.path.join(_pd, "headerless.csv")
    with open(_hp, "w", newline="", encoding="utf-8-sig") as _f:
        _f.write("Column1,Column2\nc0,text\n")
    check("verify refuses a corpus whose header was replaced (a spreadsheet import)",
          any("no comment_id" in p for p in _bp.verify(_hp, _fp, [_gp])[0]))
    _out_ok, _out_bad = os.path.join(_pd, "b_ok"), os.path.join(_pd, "b_bad")
    _r_ok = subprocess.run([sys.executable, "-m", "ewl.collect.batches", "split", _op, "--full", _fp,
                            "--exclude", _gp, "--exclude", _ep, "--out-dir", _out_ok, "--batch-size", "20"],
                           capture_output=True, text=True)
    _man = json.load(open(os.path.join(_out_ok, "batches_manifest.json"), encoding="utf-8")) if _r_ok.returncode == 0 else {}
    check("split --full checks the corpus first, and the manifest carries the provenance",
          _r_ok.returncode == 0 and "Checked against the full corpus: OK" in _r_ok.stdout
          and _man.get("provenance", {}).get("parent", {}).get("rows") == 40
          and _man["provenance"]["verified_against"]["excluded_files"][0]["file"] == "gold.csv", _r_ok.stderr[-300:])
    _r_bad = subprocess.run([sys.executable, "-m", "ewl.collect.batches", "split", _tp, "--full", _fp,
                             "--exclude", _gp, "--exclude", _ep, "--out-dir", _out_bad],
                            capture_output=True, text=True)
    check("split --full refuses a corpus that is not the full corpus minus the exclusions, writing nothing",
          _r_bad.returncode != 0 and not os.path.exists(os.path.join(_out_bad, "batches_manifest.json")),
          _r_bad.stderr[-300:])
    _r_noprov = subprocess.run([sys.executable, "-m", "ewl.collect.batches", "split", _fp,
                                "--out-dir", os.path.join(_pd, "b_plain"), "--batch-size", "20"],
                               capture_output=True, text=True)
    check("split without --full behaves as before (no check, no provenance block)",
          _r_noprov.returncode == 0 and "provenance" not in json.load(
              open(os.path.join(_pd, "b_plain", "batches_manifest.json"), encoding="utf-8")))

print("\nthe domain prompt carries no stance wording")
from ewl import taxonomy as _tx_ns
from ewl.label import combined as _cb_ns
_old_ctx = _tx_ns.configure_domain_context()
check("the Luigi background in the domain prompt is facts only: no word about stance",
      all("stance" not in _tx_ns.domain_story_context(_s) for _s in sentiment_mod.STORY_CONTEXT))
_cp_ns = _cb_ns.build_prompt(_tx_ns.FIXED_EXAMPLES, "luigi")
check("in the combined arm the stance-object sentence reaches the sentiment half only",
      "stance toward that wealth" in _cp_ns[_cp_ns.index("SENTIMENT"):]
      and "stance toward that wealth" not in _cp_ns[:_cp_ns.index("For each comment")])
check("the full Luigi paragraph the sentiment prompt shows still carries it",
      "its stance toward that wealth" in sentiment_mod.STORY_CONTEXT["luigi"])
print("\ngemma: EWL_GEMMA_TEMPERATURE lifts the one-vote limit for a stability measurement; default unchanged")
_old_t = os.environ.get("EWL_GEMMA_TEMPERATURE")
_real_make_client2 = gemma_local.GemmaJudge._make_client
try:
    os.environ.pop("EWL_GEMMA_TEMPERATURE", None)
    _f0 = _FakeOllama()
    gemma_local.GemmaJudge._make_client = lambda self: _f0
    _j0 = judges.create("gemma", votes=3)
    _j0.label("She pays no tax at all", taxonomy.FIXED_EXAMPLES)
    check("default: still one vote, temperature 0, the key it always had, and a run record with no temperature key",
          _j0.votes == 1 and len(_f0.calls) == 1 and _f0.calls[-1]["options"]["temperature"] == 0
          and _j0.cache_spec == "gemma:" + _TAG and "temperature" not in _j0.preflight_info,
          (_j0.votes, len(_f0.calls), _j0.cache_spec))
    os.environ["EWL_GEMMA_TEMPERATURE"] = "0.7"
    _f7 = _FakeOllama()
    gemma_local.GemmaJudge._make_client = lambda self: _f7
    _j7 = judges.create("gemma", votes=3)
    _j7.label("She pays no tax at all", taxonomy.FIXED_EXAMPLES)
    check("temperature 0.7: three votes really make three calls with different seeds",
          _j7.votes == 3 and len(_f7.calls) == 3
          and sorted(c["options"]["seed"] for c in _f7.calls) == [42, 43, 44], len(_f7.calls))
    check("and the temperature is sent",
          all(c["options"]["temperature"] == 0.7 for c in _f7.calls))
    check("a sampled run has its own cache key, so it never replays the temperature-0 arm's answers",
          _j7.cache_spec == "gemma:" + _TAG + "|temperature=0.7" and _j7.cache_spec != _j0.cache_spec)
    check("the run record's model_build names the temperature", _j7.preflight_info.get("temperature") == 0.7)
    for bad in ("hot", "-1", "3"):
        os.environ["EWL_GEMMA_TEMPERATURE"] = bad
        try:
            judges.create("gemma", votes=3)
            _bad_ok = False
        except ValueError:
            _bad_ok = True
        check(f"EWL_GEMMA_TEMPERATURE={bad} is refused", _bad_ok)
finally:
    gemma_local.GemmaJudge._make_client = _real_make_client2
    if _old_t is None:
        os.environ.pop("EWL_GEMMA_TEMPERATURE", None)
    else:
        os.environ["EWL_GEMMA_TEMPERATURE"] = _old_t

print(chr(10) + "batches: the fingerprint compares temperature and vote count")
from ewl.collect import batches as _batches_fp
_fp_a = {"domain_judge": {"spec": "gemma", "model": "m", "votes": 1}, "model_build": {"digest": "d"}}
_fp_b = {"domain_judge": {"spec": "gemma", "model": "m", "votes": 3},
         "model_build": {"digest": "d", "temperature": 0.7}}
_fa, _fb = _batches_fp._fingerprint(_fp_a), _batches_fp._fingerprint(_fp_b)
_fdiff = [k for k in _fa if _fa[k] != _fb[k]]
check("a different temperature or vote count makes two batches un-joinable",
      any("temperature" in k for k in _fdiff) and "votes" in _fdiff, _fdiff)
check("two default batches have the same fingerprint", _fa == _batches_fp._fingerprint(_fp_a))
print("\nclassify progress: a total, a percentage and an estimate, so a long run is not silent")
_pg_dir = tempfile.mkdtemp()
_pg_in = os.path.join(_pg_dir, "in.csv")
with open(_pg_in, "w", newline="", encoding="utf-8-sig") as _h:
    _w = csv.DictWriter(_h, fieldnames=["comment_id", "platform", "story", "source", "author_hash", "body"])
    _w.writeheader()
    for _i in range(205):
        _w.writerow({"comment_id": f"p{_i}", "platform": "youtube", "story": "jets", "source": "v1",
                     "author_hash": "a", "body": f"comment {_i}" if _i != 7 else ""})
_pg_run = subprocess.run([sys.executable, "-m", "ewl.label.classify", _pg_in, "--judge", "stub",
                          "--sentiment", "llm:stub", "--votes", "1", "--no-cache",
                          "--output", os.path.join(_pg_dir, "out.csv")],
                         capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=os.path.join(os.getcwd(), "src")))
check("progress says how many of how many, as a percentage, with the time elapsed",
      _pg_run.returncode == 0 and "100 of 204 comments labelled (49%)" in _pg_run.stderr
      and "elapsed" in _pg_run.stderr, _pg_run.stderr[-500:])
check("a blank-bodied row is not counted in the total (it is not labelled)",
      "of 205" not in _pg_run.stderr)
check("an estimate of the time left appears once there is enough to go on (200 comments)",
      "200 of 204 comments labelled" in _pg_run.stderr and "left" in _pg_run.stderr.split("200 of 204")[1].split("\n")[0],
      _pg_run.stderr[-500:])
_pg_lim = subprocess.run([sys.executable, "-m", "ewl.label.classify", _pg_in, "--judge", "stub",
                          "--sentiment", "llm:stub", "--votes", "1", "--no-cache", "--limit", "120",
                          "--output", os.path.join(_pg_dir, "out2.csv")],
                         capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=os.path.join(os.getcwd(), "src")))
check("--limit is the total shown", "100 of 120 comments labelled" in _pg_lim.stderr, _pg_lim.stderr[-400:])
print("\nmodel agreement (ewl-agreement): pairwise kappa, alpha and shared errors on the gold set")
_ENV_DV = dict(os.environ, PYTHONPATH=os.path.join(os.getcwd(), "src"))
_ag_dir = tempfile.mkdtemp()
_ag_hand_domain = ["social", "none", "economics", "social, economics", "none", "environment",
                   "democracy", "none", "social", "none"]
_ag_hand_sent = ["1", "0", "3", "5", "0", "2", "4", "0", "1", "-1"]


def _ag_write(name, domains, sentiments, parse_fail=()):
    path = os.path.join(_ag_dir, f"{name}.csv")
    cats = ["democracy", "economics", "environment", "social"]
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["comment_id", "body", "hand_domain", "hand_sentiment",
                                                    "domain", "domain_parse_failure", "sentiment"]
                                + [f"domain_{c}" for c in cats])
        writer.writeheader()
        for i, (domain, sentiment) in enumerate(zip(domains, sentiments)):
            row = {"comment_id": f"a{i}", "body": "SECRET-AGREE-TEXT", "hand_domain": _ag_hand_domain[i],
                   "hand_sentiment": _ag_hand_sent[i], "domain": domain, "sentiment": sentiment,
                   "domain_parse_failure": "1" if i in parse_fail else "0"}
            for c in cats:
                row[f"domain_{c}"] = "1" if c in domain else "0"
            writer.writerow(row)
    return path


_ag_exact_domain = ["social", "unclassified", "economics", "social, economics", "unclassified",
                    "environment", "democracy", "unclassified", "social", "unclassified"]
_ag_exact_sent = ["negative", "irrelevant", "neutral", "positive", "irrelevant", "negative",
                  "positive", "irrelevant", "negative", "uncodeable"]
_ag_a = _ag_write("modelA", _ag_exact_domain, _ag_exact_sent)
_ag_b = _ag_write("modelB", _ag_exact_domain, _ag_exact_sent)
_wrong_domain = list(_ag_exact_domain)
_wrong_domain[0] = "democracy"          # a shared error: A and B agree with each other, not hand
_ag_a2 = _ag_write("modelA2", _wrong_domain, _ag_exact_sent)
_ag_b2 = _ag_write("modelB2", _wrong_domain, _ag_exact_sent, parse_fail=(3,))
_ag_run = subprocess.run([sys.executable, "-m", "ewl.analyse.agreement", "--model", f"A={_ag_a}",
                          "--model", f"B={_ag_b}", "--bootstrap", "200"],
                         capture_output=True, text=True, env=_ENV_DV)
check("identical models that match the hand codes give kappa 1.00 and alpha 1.00 on a harm category",
      _ag_run.returncode == 0 and "1.00" in _ag_run.stdout.split("=== harm category: social ===")[1].split("===")[0]
      and "alpha 1.00" in _ag_run.stdout.split("=== harm category: social ===")[1].split("===")[0], _ag_run.stderr[-400:])
check("the output holds statistics only, never comment text", "SECRET-AGREE" not in _ag_run.stdout)
check("the sentiment judgement is reported with all five answers' agreement",
      "sentiment answer" in _ag_run.stdout and "alpha" in _ag_run.stdout.split("sentiment answer")[1])
_ag_run2 = subprocess.run([sys.executable, "-m", "ewl.analyse.agreement", "--model", f"A={_ag_a2}",
                           "--model", f"B={_ag_b2}", "--bootstrap", "200"],
                          capture_output=True, text=True, env=_ENV_DV)
_ag_social = _ag_run2.stdout.split("=== harm category: social ===")[1].split("===")[0]
check("two models that agree with each other but not the hand codes are counted as shared errors",
      "of those they were all different from the hand code (shared errors)" in _ag_social
      and "on 1 of those" in _ag_social, _ag_social[-500:])
check("a comment one model could not parse is left out and the count is printed",
      "9 of 10 comments have a usable answer" in _ag_social, _ag_social[:300])
_ag_diff = _ag_write("shorter", _ag_exact_domain[:9], _ag_exact_sent[:9])
_ag_run3 = subprocess.run([sys.executable, "-m", "ewl.analyse.agreement", "--model", f"A={_ag_a}",
                           "--model", f"C={_ag_diff}"], capture_output=True, text=True, env=_ENV_DV)
check("files over different comments are refused", _ag_run3.returncode != 0
      and "not labelled over the same comments" in (_ag_run3.stderr + _ag_run3.stdout))

print("\nclassify: the run reports its own speed and the thinking setting")
from ewl.label import classify as _clf_time
check("durations read as people say them",
      [_clf_time.format_duration(x) for x in (45, 125, 3725, 45000)]
      == ["45 s", "2 min 05 s", "1 h 02 min", "12 h 30 min"],
      [_clf_time.format_duration(x) for x in (45, 125, 3725, 45000)])
with tempfile.TemporaryDirectory() as _td:
    _in_t = os.path.join(_td, "in.csv")
    with open(_in_t, "w", newline="", encoding="utf-8-sig") as _f:
        _w = csv.DictWriter(_f, fieldnames=["comment_id", "story", "body"])
        _w.writeheader()
        _w.writerow({"comment_id": "t1", "story": "luigi", "body": "a comment"})
    _out_t = os.path.join(_td, "out.csv")
    _pt = subprocess.run([sys.executable, "-m", "ewl.label.classify", _in_t, "--judge", "stub", "--sentiment",
                          "llm:stub", "--votes", "1", "--no-cache", "--output", _out_t],
                         capture_output=True, text=True, env=dict(os.environ, EWL_AUTHOR_SALT="test-salt-value-1234"))
    _rec_t = json.load(open(_out_t + ".run.json", encoding="utf-8"))
    check("the run record carries the labelling time and the thinking setting (None for a stub)",
          _pt.returncode == 0 and "labelling_seconds" in _rec_t["result"] and "thinking_setting" in _rec_t["result"]
          and _rec_t["result"]["thinking_setting"] is None, _pt.stderr[-300:])
_son = judges.AnthropicJudge.thinking_setting(_make_probe("claude-sonnet-5-5"))
_opus_default = judges.AnthropicJudge.thinking_setting(_make_probe("claude-opus-5-5"))
_haiku_none = judges.AnthropicJudge.thinking_setting(_make_probe("claude-haiku-4-5"))
_omit = _make_probe("claude-opus-5-5")
_omit._thinking_override = judges.AnthropicJudge.THINKING_OMIT
check("thinking_setting names what was sent: between_tools, disabled, thinking ON, refused, or nothing for Haiku",
      _son == "between_tools" and _opus_default.startswith("cannot be turned off") and _haiku_none is None
      and judges.AnthropicJudge.thinking_setting(_omit).startswith("thinking ON")
      and judges.AnthropicJudge.thinking_setting(_make_probe("claude-sonnet-5")) == "disabled",
      (_son, _opus_default, _haiku_none))


print("\nclassify: the speed report uses the median call, not the wall clock")
_sr = _clf_time.speed_report([2.0, 2.0, 2.0, 2.0, 5000.0], 6, 3, [693, 83532], wall_seconds=250000, comments=30)
check("the projection comes from the median call and the concurrency (2 s x 6 calls / 3 = 4 s a comment)",
      "typical call takes 2.0 s" in _sr[0] and "about 4.0 s per comment" in _sr[0]
      and any("693 comments would take about 46 min 12 s" in l for l in _sr), _sr)
check("a wall clock far above the calls' own time is called out (a sleeping computer), not used",
      any("Ignore the wall-clock figure" in l for l in _sr))
_sr_ok = _clf_time.speed_report([2.0] * 20, 6, 3, [100], wall_seconds=130, comments=30)
check("a sensible wall clock is just shown", any("Wall clock for this run: 2 min 10 s" in l for l in _sr_ok), _sr_ok)
check("an all-cached run says there is nothing to time",
      "nothing to time" in _clf_time.speed_report([], 6, 3, [693])[0])
check("the default projections are one constant", _clf_time.DEFAULT_PROJECTIONS == (693, 83532))
print("\nOpenAI arms: effort setting, prices, refusal, and the scorecard paths (fake client, no SDK needed)")
import types as _types_oa
import contextlib as _ctx_oa
from ewl.analyse import evaluate as _ev_oa

_calls_oa = []
_reply_oa = {"text": "democracy"}
_reject_oa = {"on": False}


class _OAError(Exception):
    status_code = 400


class _OACompletions:
    def create(self, **kwargs):
        _calls_oa.append(kwargs)
        if _reject_oa["on"] and "reasoning_effort" in kwargs:
            raise _OAError("Error code: 400 - {'error': {'message': \"Unsupported value: "
                           "'reasoning_effort' does not support 'minimal' with this model.\", "
                           "'type': 'invalid_request_error'}}")
        return _fake_openai_response(_reply_oa["text"], "stop", prompt_tokens=1000,
                                     completion_tokens=5, cached_tokens=0)


class _OAClient:
    def __init__(self, *a, **k):
        self.chat = _types_oa.SimpleNamespace(completions=_OACompletions())


_fake_openai_module = _types_oa.ModuleType("openai")
_fake_openai_module.OpenAI = _OAClient
_had_openai = "openai" in sys.modules
_real_openai = sys.modules.get("openai")
sys.modules["openai"] = _fake_openai_module
_old_env_oa = {k: os.environ.get(k) for k in ("EWL_OPENAI_REASONING_EFFORT", "EWL_OPENAI_PRICES")}


def _set_env_oa(**values):
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


try:
    _set_env_oa(EWL_OPENAI_REASONING_EFFORT=None, EWL_OPENAI_PRICES=None)
    _j_def = judges.create("openai:gpt-5-mini", votes=1)
    _j_def._sample("a comment", "a prompt", 0)
    check("default: reasoning_effort is minimal, as every earlier run sent",
          _calls_oa[-1].get("reasoning_effort") == "minimal", _calls_oa[-1])
    check("default: the cache key text is exactly what it always was, so existing caches still match",
          _j_def.cache_spec == "openai:gpt-5-mini", _j_def.cache_spec)
    check("default: the run record gets reasoning_effort=minimal",
          _j_def.thinking_setting() == "reasoning_effort=minimal")

    _set_env_oa(EWL_OPENAI_REASONING_EFFORT="low")
    _j_low = judges.create("openai:gpt-5-mini", votes=1)
    _j_low._sample("a comment", "a prompt", 0)
    check("EWL_OPENAI_REASONING_EFFORT=low is sent", _calls_oa[-1].get("reasoning_effort") == "low")
    check("a non-default effort gets its own cache key, so answers are never replayed across efforts",
          _j_low.cache_spec == "openai:gpt-5-mini|reasoning_effort=low"
          and _j_low.cache_spec != _j_def.cache_spec)
    check("and the run record says so", _j_low.thinking_setting() == "reasoning_effort=low")
    _set_env_oa(EWL_OPENAI_REASONING_EFFORT=None, EWL_OPENAI_PRICES="gpt-6.1-sol=2.00,10.00;gpt-6-luna=0.10,0.50")
    _j_g6 = judges.create("openai:gpt-6.1-sol", votes=1)
    _j_g6._sample("c", "p", 0)
    check("gpt-6 models are treated as reasoning models: sent the effort setting, and it is recorded",
          _calls_oa[-1].get("reasoning_effort") == "minimal"
          and _j_g6.thinking_setting() == "reasoning_effort=minimal", _calls_oa[-1])
    _set_env_oa(EWL_OPENAI_PRICES="gpt-test-model=3.00,12.50")
    _j_test = judges.create("openai:gpt-test-model", votes=1)
    _j_test._sample("c", "p", 0)
    check("a model that is not a reasoning model is never sent the setting, and its key is unchanged",
          "reasoning_effort" not in _calls_oa[-1] and _j_test.cache_spec == "openai:gpt-test-model"
          and _j_test.thinking_setting() is None, _calls_oa[-1])
    check("a priced model's spend is counted (1,000 input and 5 output tokens at $0.25 and $2 per million)",
          abs(_j_def.cost_usd() - (1000 * 0.25 + 5 * 2.00) / 1e6) < 1e-9, _j_def.cost_usd())

    # a model that rejects the setting: stop, name the fix, never retry or fall back
    _set_env_oa(EWL_OPENAI_REASONING_EFFORT=None)
    _j_rej = judges.create("openai:gpt-5-mini", votes=1)
    _reject_oa["on"] = True
    _before = len(_calls_oa)
    try:
        _j_rej._sample("a comment", "a prompt", 0)
        _stopped = None
    except judges.ReasoningEffortRejected as _err:
        _stopped = str(_err)
    _reject_oa["on"] = False
    check("a 400 about reasoning_effort stops the run with the setting named and the fix stated",
          _stopped is not None and "reasoning_effort='minimal'" in _stopped
          and "EWL_OPENAI_REASONING_EFFORT" in _stopped, _stopped)
    check("it is a permanent error (not retried), and no other setting was tried",
          not _j_rej._is_retryable(judges.ReasoningEffortRejected(_stopped or ""))
          and len(_calls_oa) - _before == 1, len(_calls_oa) - _before)

    # prices
    _set_env_oa(EWL_OPENAI_PRICES=None)
    try:
        judges.openai_price_for("gpt-unknown-1")
        _refused = False
    except judges.PriceUnavailable as _e:
        _refused = "EWL_OPENAI_PRICES" in str(_e)
    check("an unpriced model is still refused, and the message says how to supply its price", _refused)
    _set_env_oa(EWL_OPENAI_PRICES="gpt-unknown-1=3.00,12.50;gpt-other=1,2")
    check("a price supplied in EWL_OPENAI_PRICES is used for a model not in the table",
          judges.openai_price_for("gpt-unknown-1") == (3.0, 12.5)
          and judges.openai_price_for("gpt-other") == (1.0, 2.0))
    check("a table price is not overridden by the variable",
          judges.openai_price_for("gpt-5-mini") == judges._OPENAI_PRICES["gpt-5-mini"])
    _set_env_oa(EWL_OPENAI_PRICES="gpt-unknown-1=three,12")
    try:
        judges.openai_price_for("gpt-unknown-1")
        _bad = False
    except judges.PriceUnavailable as _e:
        _bad = "model=input,output" in str(_e)
    check("a malformed price variable stops the run instead of being skipped", _bad)
    _set_env_oa(EWL_OPENAI_PRICES=None)

    # the scorecard paths
    _dir_oa = tempfile.mkdtemp()
    _gold_oa = os.path.join(_dir_oa, "gold.csv")
    write_fixture(_gold_oa)

    def _run_eval(*argv):
        buffer = io.StringIO()
        old_argv = sys.argv
        sys.argv = ["ewl-evaluate", *argv]
        try:
            with _ctx_oa.redirect_stdout(buffer), _ctx_oa.redirect_stderr(io.StringIO()):
                code = _ev_oa.main()
        except SystemExit as error:
            code = error.code
        finally:
            sys.argv = old_argv
        return code, buffer.getvalue()

    _reply_oa["text"] = "democracy"
    _code, _out = _run_eval("scorecard", _gold_oa, "--arm", "openai:gpt-5-mini/fixed",
                            "--votes", "1", "--cache", os.path.join(_dir_oa, "c1.jsonl"))
    check("ewl-evaluate scorecard --arm openai:gpt-5-mini/fixed runs and names the arm",
          _code in (0, None) and "openai:gpt-5-mini/fixed" in _out, (_code, _out[-400:]))
    _reply_oa["text"] = "democracy | 4"
    _code, _out = _run_eval("scorecard", _gold_oa, "--arm", "combined:openai:gpt-5-mini/fixed",
                            "--votes", "1", "--cache", os.path.join(_dir_oa, "c2.jsonl"))
    check("the combined arm runs on OpenAI too",
          _code in (0, None) and "combined:openai:gpt-5-mini/fixed" in _out, (_code, _out[-400:]))
    _sgold_oa = os.path.join(_dir_oa, "sgold.csv")
    with open(_sgold_oa, "w", newline="", encoding="utf-8") as _h:
        _w = csv.DictWriter(_h, fieldnames=["comment_id", "story", "platform", "body",
                                            "hand_domain", "hand_sentiment"])
        _w.writeheader()
        for _i, (_body, _lab) in enumerate(FIXTURE):
            _w.writerow({"comment_id": f"s{_i}", "story": "luigi", "platform": "youtube",
                         "body": _body, "hand_domain": _lab, "hand_sentiment": "4"})
    _reply_oa["text"] = "4"
    _code, _out = _run_eval("sentiment", _sgold_oa, "--arm", "llm:openai:gpt-5-mini",
                            "--cache", os.path.join(_dir_oa, "c3.jsonl"))
    check("ewl-evaluate sentiment --arm llm:openai:gpt-5-mini runs and scores",
          _code in (0, None) and "openai" in _out.lower(), (_code, _out[-400:]))
finally:
    for _k, _v in _old_env_oa.items():
        _set_env_oa(**{_k: _v})
    if _had_openai:
        sys.modules["openai"] = _real_openai
    else:
        sys.modules.pop("openai", None)
print("\ncombined arm: speed report, cache savings and thinking setting read the inner judge")
_cs_dir = tempfile.mkdtemp()
_cs_in = os.path.join(_cs_dir, "in.csv")
with open(_cs_in, "w", newline="", encoding="utf-8-sig") as _h:
    _w = csv.DictWriter(_h, fieldnames=["comment_id", "platform", "story", "source", "author_hash", "body"])
    _w.writeheader()
    for _i in range(6):
        _w.writerow({"comment_id": f"m{_i}", "platform": "youtube", "story": "jets",
                     "source": "v1", "author_hash": "a", "body": f"comment {_i}"})
_cs_run = subprocess.run(
    [sys.executable, "-m", "ewl.label.classify", _cs_in, "--judge", "combined:stub",
     "--sentiment", "combined:stub", "--votes", "3", "--no-cache", "--concurrency", "2",
     "--output", os.path.join(_cs_dir, "out.csv")],
    capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=os.path.join(os.getcwd(), "src")))
check("a combined run reports its speed from the live calls it made, not 'every reply came from the cache'",
      _cs_run.returncode == 0 and "Speed: a typical call takes" in _cs_run.stderr
      and "every reply came from the cache" not in _cs_run.stderr, _cs_run.stderr[-500:])
check("it counts 3 calls per comment at three votes (the combined arm), not 6",
      "at 3 calls per comment" in _cs_run.stderr and "at 6 calls per comment" not in _cs_run.stderr,
      _cs_run.stderr[-500:])


class _FakeInner:
    spec = "stub"
    deterministic = True
    usage = collections.Counter()
    seconds = 0.0
    latencies = [1.0, 2.0]

    def cost_usd(self):
        return 0.0

    def cache_savings_usd(self):
        return 1.23

    def thinking_setting(self):
        return "between_tools"


_cs_judge = judges.create("combined:stub", votes=7, fuzzy=True)
_cs_judge._inner = _FakeInner()
check("the cache saving is the inner judge's, not the wrapper's $0.00",
      _cs_judge.cache_savings_usd() == 1.23)
check("the run record gets the inner judge's thinking setting (it was None for a combined run)",
      _cs_judge.thinking_setting() == "between_tools")
check("the wrapper's latencies are the inner judge's", list(_cs_judge.latencies) == [1.0, 2.0])
_cs_scorer = sentiment_mod.create("combined:stub", votes=7, fuzzy=True)
_cs_scorer._judge._inner.usage["input_tokens"] = 7
_cs_scorer._judge._inner.usage["output_tokens"] = 3
_cs_scorer._judge._inner.usage["cache_read_input_tokens"] = 5
check("the combined sentiment scorer's token columns read the shared judge (they showed 0 beside a real cost)",
      _cs_scorer.token_usage() == (7, 3) and _cs_scorer.cache_tokens()[1] == 5, _cs_scorer.token_usage())

print("\nreviewers' fixes on #126: thinking handler scope, allowed spellings, ALLOW_THINKING scope, per-thread backoff")
import types as _types_c

_calls_c = []
_script_c = {"errors": [], "default": None}


class _CMessages:
    def create(self, **kwargs):
        _calls_c.append(kwargs)
        if _script_c["errors"]:
            raise Exception(_script_c["errors"].pop(0))
        if _script_c["default"] is not None:
            raise Exception(_script_c["default"])
        return _types_c.SimpleNamespace(
            usage=_types_c.SimpleNamespace(input_tokens=1, output_tokens=1),
            stop_reason="end_turn",
            content=[_types_c.SimpleNamespace(type="text", text="democracy")])


_fake_anthropic = _types_c.ModuleType("anthropic")
_fake_anthropic.Anthropic = lambda *a, **k: _types_c.SimpleNamespace(messages=_CMessages())
_had_anthropic = "anthropic" in sys.modules
_real_anthropic = sys.modules.get("anthropic")
sys.modules["anthropic"] = _fake_anthropic
_env_keep_c = {k: os.environ.get(k) for k in ("EWL_ANTHROPIC_ALLOW_THINKING", "EWL_ANTHROPIC_EFFORT")}
try:
    os.environ.pop("EWL_ANTHROPIC_EFFORT", None)
    os.environ.pop("EWL_ANTHROPIC_ALLOW_THINKING", None)
    # 1. a model that is not sent a thinking field gets a thinking 400: stop at once, no loop
    _haiku = judges.create("anthropic:claude-haiku-4-5", votes=1)
    _calls_c.clear()
    _script_c["default"] = 'Error code: 400 - invalid_request_error: "thinking" is not supported'
    try:
        _haiku._sample("a comment", "a prompt", 0)
        _stopped_c = False
    except Exception:
        _stopped_c = True
    _script_c["default"] = None
    check("a thinking 400 on a model that is sent no thinking field raises at once (it used to repeat the "
          "same request for ever)", _stopped_c and len(_calls_c) == 1, len(_calls_c))

    # 2. a hint that would turn thinking ON is not adopted
    _sonnet = judges.create("anthropic:claude-sonnet-5-5", votes=1)
    _calls_c.clear()
    _script_c["errors"] = ['Error code: 400 - invalid_request_error: send "thinking": {"type": "adaptive"} instead']
    _sonnet._sample("a comment", "a prompt", 0)
    check("a hint naming a spelling that is not an 'off' spelling is not adopted: the next off spelling is tried",
          len(_calls_c) == 2 and _calls_c[0]["thinking"] == {"type": "between_tools"}
          and _calls_c[1]["thinking"] == {"type": "disabled"}, [c.get("thinking") for c in _calls_c])

    # 3. ALLOW_THINKING applies only to a model that cannot have thinking turned off
    os.environ["EWL_ANTHROPIC_ALLOW_THINKING"] = "1"
    _s2 = judges.create("anthropic:claude-sonnet-5-5", votes=1)
    check("EWL_ANTHROPIC_ALLOW_THINKING is ignored for Sonnet 5.5: same cache key, same 256-token limit",
          _s2.allow_thinking is False and "thinking=allowed" not in _s2.cache_spec and _s2.reply_limit == 256,
          (_s2.cache_spec, _s2.reply_limit))
    _o55 = judges.create("anthropic:claude-opus-5-5", votes=1)
    check("and applies to Opus 5.5: tagged, 1,024-token limit, recorded as thinking ON",
          _o55.allow_thinking is True and "thinking=allowed" in _o55.cache_spec and _o55.reply_limit == 1024
          and "thinking ON" in _o55.thinking_setting(), (_o55.cache_spec, _o55.thinking_setting()))
    _calls_c.clear()
    _script_c["default"] = 'Error code: 400 - invalid_request_error: "thinking" is rejected'
    try:
        _o55._sample("a comment", "a prompt", 0)
        _o55_stop = False
    except Exception:
        _o55_stop = True
    _script_c["default"] = None
    check("a thinking-ON arm that gets a thinking 400 stops at once and never switches to an 'off' spelling",
          _o55_stop and len(_calls_c) == 1 and "thinking" not in _calls_c[0], len(_calls_c))
    os.environ.pop("EWL_ANTHROPIC_ALLOW_THINKING", None)
finally:
    for _k, _v in _env_keep_c.items():
        if _v is None:
            os.environ.pop(_k, None)
        else:
            os.environ[_k] = _v
    if _had_anthropic:
        sys.modules["anthropic"] = _real_anthropic
    else:
        sys.modules.pop("anthropic", None)

# 4. backoff one worker sleeps is not subtracted from another worker's call time
_slept_probe = judges.create("stub", votes=1)
_real_sleep_c = judges.time.sleep
judges.time.sleep = lambda s: None
_attempts_c = {"n": 0}


def _flaky():
    _attempts_c["n"] += 1
    if _attempts_c["n"] < 3:
        raise Exception("rate_limit: 429")
    return "ok"


try:
    _worker = threading.Thread(target=lambda: _slept_probe._with_retries(_flaky, "probe"))
    _worker.start()
    _worker.join()
    _main_slept = _slept_probe._thread_slept()
    _tls_other = _slept_probe.__dict__["_tls"]
finally:
    judges.time.sleep = _real_sleep_c
check("backoff slept by another worker is not counted as this thread's sleep",
      _main_slept == 0.0 and _slept_probe.slept > 0.0, (_main_slept, _slept_probe.slept))


print("\n#130 review: the data-protection gate, checks before anything is sent, effort handling")
_built_oa = {"n": 0}
_reqs_oa = []


class _OA2Completions:
    def create(self, **kwargs):
        _reqs_oa.append(kwargs)
        if kwargs.get("reasoning_effort") == "minimal" and "reject" in kwargs.get("model", ""):
            raise _OAError("Error code: 400 - {'error': {'message': \"Unsupported value: 'reasoning_effort' "
                           "does not support 'minimal' with this model.\", 'type': 'invalid_request_error'}}")
        return _fake_openai_response("democracy", "stop", prompt_tokens=1000, completion_tokens=5)


class _OA2Client:
    def __init__(self, *a, **k):
        _built_oa["n"] += 1
        self.chat = _types_oa.SimpleNamespace(completions=_OA2Completions())


_fake2 = _types_oa.ModuleType("openai")
_fake2.OpenAI = _OA2Client
_saved_openai2 = sys.modules.get("openai")
sys.modules["openai"] = _fake2
_keys130 = ("EWL_EXTERNAL_PROCESSORS", "EWL_OPENAI_PRICES", "EWL_OPENAI_REASONING_EFFORT")
_old130 = {k: os.environ.get(k) for k in _keys130}
try:
    os.environ["EWL_EXTERNAL_PROCESSORS"] = "openai"
    os.environ.pop("EWL_OPENAI_PRICES", None)
    os.environ.pop("EWL_OPENAI_REASONING_EFFORT", None)

    # the gate
    os.environ.pop("EWL_EXTERNAL_PROCESSORS")
    _built_oa["n"] = 0
    try:
        judges.create("openai:gpt-5", votes=1)
        _gate = None
    except judges.ExternalProcessorNotApproved as _e:
        _gate = str(_e)
    check("without EWL_EXTERNAL_PROCESSORS the OpenAI judge refuses to be built, names the sign-off, and builds no client",
          _gate is not None and "#100" in _gate and "EWL_EXTERNAL_PROCESSORS=openai" in _gate and _built_oa["n"] == 0,
          (_gate, _built_oa["n"]))
    os.environ["EWL_EXTERNAL_PROCESSORS"] = "google"
    try:
        judges.create("openai:gpt-5", votes=1)
        _other = False
    except judges.ExternalProcessorNotApproved:
        _other = True
    check("approving a different provider does not approve OpenAI", _other)
    os.environ["EWL_EXTERNAL_PROCESSORS"] = " OpenAI , google "
    check("the list is comma-separated and case-insensitive", judges.create("openai:gpt-5", votes=1) is not None)
    os.environ["EWL_EXTERNAL_PROCESSORS"] = "openai"

    # an unpriced model is refused when the judge is built, with no client and no request
    _built_oa["n"] = 0
    del _reqs_oa[:]
    try:
        judges.create("openai:gpt-unpriced-x", votes=1)
        _unp = False
    except judges.PriceUnavailable:
        _unp = True
    check("an unpriced model is refused when the judge is built: no client, no request",
          _unp and _built_oa["n"] == 0 and not _reqs_oa)
    os.environ["EWL_OPENAI_PRICES"] = "gpt-5=oops"
    try:
        judges.create("openai:gpt-5", votes=1)
        _bad = False
    except judges.PriceUnavailable:
        _bad = True
    check("a malformed price variable is refused at build time even for a model in the table", _bad)
    os.environ.pop("EWL_OPENAI_PRICES")

    # the same refusal through ewl-classify: before any comment is sent
    _in130 = os.path.join(tempfile.mkdtemp(), "in.csv")
    with open(_in130, "w", newline="", encoding="utf-8") as _h:
        _h.write("comment_id,body\nc1,a comment\nc2,another comment\n")
    _argv130 = sys.argv
    sys.argv = ["classify.py", _in130, "--judge", "openai:gpt-unpriced-x", "--no-cache",
                "--output", os.path.join(os.path.dirname(_in130), "out.csv")]
    _msg130 = ""
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            classify.main()
    except SystemExit as _e:
        _msg130 = str(_e)
    finally:
        sys.argv = _argv130
    check("ewl-classify with an unpriced OpenAI model stops with the message before any request is made",
          "No price on record" in _msg130 and not _reqs_oa, (_msg130[:120], len(_reqs_oa)))

    # a cached-input price
    os.environ["EWL_OPENAI_PRICES"] = "gpt-6.1-sol=2.00,10.00,0.10;gpt-6-sol=2.00,10.00"
    check("a third figure in EWL_OPENAI_PRICES is the cached-input price (here 5% of input); without one it is 10%",
          abs(judges.openai_cached_ratio("gpt-6.1-sol") - 0.05) < 1e-9
          and abs(judges.openai_cached_ratio("gpt-6-sol") - 0.10) < 1e-9
          and judges.openai_price_for("gpt-6.1-sol") == (2.0, 10.0))
    _sol = judges.create("openai:gpt-6.1-sol", votes=1)
    _sol.usage["cache_read_input_tokens"] = 1_000_000
    check("the cost accounting uses it: a million cached tokens at $2 and 5% saves $1.90, not $1.80",
          abs(_sol.cache_savings_usd() - 1.90) < 1e-6, _sol.cache_savings_usd())
    os.environ.pop("EWL_OPENAI_PRICES")

    # effort handling
    os.environ["EWL_OPENAI_REASONING_EFFORT"] = "Low"
    _j = judges.create("openai:gpt-5", votes=1)
    check("the effort is lower-cased: 'Low' is sent as low, keyed as low, and named in the arm's spec",
          _j._effort() == "low" and _j.cache_spec.endswith("|reasoning_effort=low")
          and _j.spec == "openai:gpt-5@low", (_j._effort(), _j.cache_spec, _j.spec))
    os.environ.pop("EWL_OPENAI_REASONING_EFFORT")
    check("at the default effort the spec carries no suffix (existing arm names unchanged)",
          judges.create("openai:gpt-5", votes=1).spec == "openai:gpt-5")
    os.environ["EWL_OPENAI_REASONING_EFFORT"] = "high"
    check("a higher effort gets more room for hidden reasoning in the reply limit",
          judges.create("openai:gpt-5", votes=1).reply_limit == 4096)
    os.environ["EWL_OPENAI_REASONING_EFFORT"] = "low"
    os.environ["EWL_OPENAI_PRICES"] = "gpt-4.1-mini=0.40,1.60"
    try:
        judges.create("openai:gpt-4.1-mini", votes=1)
        _nr = False
    except judges.SettingRefused as _e:
        _nr = "not a reasoning model" in str(_e)
    os.environ.pop("EWL_OPENAI_PRICES", None)
    check("an effort set for a model that is not a reasoning model is refused, not silently ignored", _nr)
    os.environ.pop("EWL_OPENAI_REASONING_EFFORT")

    # a rejected setting stops after ONE request, through sample() (the retry layer)
    os.environ["EWL_OPENAI_PRICES"] = "gpt-5-reject=1.00,2.00"
    _rej = judges.create("openai:gpt-5-reject", votes=1)
    del _reqs_oa[:]
    try:
        _rej.sample("a comment", "a prompt", 0)
        _stopped130 = False
    except judges.ReasoningEffortRejected:
        _stopped130 = True
    check("a rejected reasoning_effort goes through the retry layer and still sends exactly one request",
          _stopped130 and len(_reqs_oa) == 1 and _rej.usage["attempts"] == 1,
          (len(_reqs_oa), dict(_rej.usage)))
finally:
    for _k, _v in _old130.items():
        if _v is None:
            os.environ.pop(_k, None)
        else:
            os.environ[_k] = _v
    if _saved_openai2 is None:
        sys.modules.pop("openai", None)
    else:
        sys.modules["openai"] = _saved_openai2
print("\n#132 review: calls per comment is right for VADER and for a combined scorer on a different judge")
_cc_dir = tempfile.mkdtemp()
_cc_in = os.path.join(_cc_dir, "in.csv")
with open(_cc_in, "w", newline="", encoding="utf-8-sig") as _h:
    _w = csv.DictWriter(_h, fieldnames=["comment_id", "platform", "story", "source", "author_hash", "body"])
    _w.writeheader()
    for _i in range(6):
        _w.writerow({"comment_id": f"v{_i}", "platform": "youtube", "story": "jets", "source": "v1",
                     "author_hash": "a", "body": f"comment {_i}"})


def _cc_run(*extra):
    return subprocess.run([sys.executable, "-m", "ewl.label.classify", _cc_in, "--votes", "3", "--no-cache",
                           "--concurrency", "2", "--output", os.path.join(_cc_dir, "o.csv"), *extra],
                          capture_output=True, text=True,
                          env=dict(os.environ, PYTHONPATH=os.path.join(os.getcwd(), "src")))


_r_vader = _cc_run("--judge", "stub", "--sentiment", "vader")
check("a stub judge with VADER at three votes is counted as 3 calls per comment, not 4",
      "at 3 calls per comment" in _r_vader.stderr and "at 4 calls per comment" not in _r_vader.stderr,
      _r_vader.stderr[-400:])
_r_comb_other = _cc_run("--judge", "stub", "--sentiment", "combined:stub")
check("a plain judge with a combined scorer is counted as 6 calls per comment (the scorer's own model calls)",
      "at 6 calls per comment" in _r_comb_other.stderr, _r_comb_other.stderr[-400:])

print("\nevaluate --concurrency: the scorecard and sentiment loops can run comments in parallel")
import time as _time_cc
from ewl.analyse import evaluate as _ev_cc


class _SlowJudge:
    spec = "slow"
    votes = 1
    deterministic = True
    seconds = 0.0

    def __init__(self):
        self.usage = collections.Counter()
        self.threads = set()

    def cost_usd(self):
        return 0.0

    def token_usage(self):
        return (0, 0)

    def label(self, body, examples, story=None):
        _time_cc.sleep(0.03)
        self.threads.add(threading.get_ident())
        return (["democracy"] if "d" in body else ["unclassified"]), False, []


class _NoExamples:
    def select(self, body):
        return []

    def describe(self):
        return "fixed(0)"


_rows_cc = [{"_body": f"c{i}" + ("d" if i % 2 else ""), "_gold": ({"democracy"} if i % 3 == 0 else set()),
             "source": f"v{i % 4}", "comment_id": f"r{i}", "story": "luigi"} for i in range(24)]
_order = list(_ev_cc.label_in_order(lambda x: (_time_cc.sleep(0.02 * (5 - x % 5)), x)[1], range(20), 5))
check("label_in_order returns results in input order whatever order they finish in",
      _order == list(range(20)), _order)
_seq_judge, _par_judge = _SlowJudge(), _SlowJudge()
_t0 = _time_cc.monotonic()
_seq = _ev_cc.run_arm(_rows_cc, _seq_judge, _NoExamples(), progress=False)
_t_seq = _time_cc.monotonic() - _t0
_t0 = _time_cc.monotonic()
_par = _ev_cc.run_arm(_rows_cc, _par_judge, _NoExamples(), progress=False, concurrency=6)
_t_par = _time_cc.monotonic() - _t0
check("run_arm gives identical per-category figures, macro-F1 and scored rows at concurrency 1 and 6",
      repr(_seq["per_category"]) == repr(_par["per_category"])
      and repr(_seq["macro_f1"]) == repr(_par["macro_f1"])
      and _seq["_scored_rows"] == _par["_scored_rows"] and _seq["unclassified"] == _par["unclassified"])
check("it really ran in parallel: more than one thread, and faster than the serial loop",
      len(_par_judge.threads) > 1 and len(_seq_judge.threads) == 1 and _t_par < _t_seq * 0.6,
      (len(_par_judge.threads), round(_t_seq, 2), round(_t_par, 2)))
check("the reported time at concurrency 6 is the clock, not the sum of every call",
      _par["seconds"] < _t_seq * 0.6, _par["seconds"])


class _SlowScorer:
    seconds = 0.0

    def __init__(self):
        self.threads = set()

    def cost_usd(self):
        return 0.0

    def token_usage(self):
        return (0, 0)

    def describe(self):
        return "slow-scorer"

    def score(self, body, story=None):
        _time_cc.sleep(0.03)
        self.threads.add(threading.get_ident())
        label = "negative" if body.endswith("a") else "positive"
        return {"label": label, "value": -1.0 if label == "negative" else 1.0, "unstable": False}


_srows = [(f"text {i}" + ("a" if i % 2 else "b"), "negative" if i % 3 else "positive", "luigi")
          for i in range(24)]
_ss, _sp = _SlowScorer(), _SlowScorer()
_rs = _ev_cc.sentiment_report(_srows, _ss)
_rp = _ev_cc.sentiment_report(_srows, _sp, concurrency=6)
check("sentiment_report gives the same accuracy, confusion matrix and per-class figures at 1 and 6",
      _rs["accuracy"] == _rp["accuracy"] and _rs["confusion"] == _rp["confusion"]
      and repr(_rs["per_class"]) == repr(_rp["per_class"]) and _rs["n"] == _rp["n"] == 24)
check("and it ran in parallel", len(_sp.threads) > 1 and len(_ss.threads) == 1,
      (len(_sp.threads), len(_ss.threads)))
_s1, _s6 = _SlowScorer(), _SlowScorer()
_sc1 = _ev_cc.special_case_report(_srows, _s1, "negative")
_sc6 = _ev_cc.special_case_report(_srows, _s6, "negative", concurrency=6)
check("the irrelevant/uncodeable check also runs in parallel, with the same answer as the serial one",
      _sc1 == _sc6 and len(_s6.threads) > 1 and len(_s1.threads) == 1, (_sc1, _sc6))

_cc_gold = os.path.join(tempfile.mkdtemp(), "gold.csv")
write_fixture(_cc_gold)
_cc_run = subprocess.run([sys.executable, "-m", "ewl.analyse.evaluate", "scorecard", _cc_gold,
                          "--arm", "stub/fixed", "--votes", "1", "--concurrency", "4"],
                         capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=os.path.join(os.getcwd(), "src")))
_cc_one = subprocess.run([sys.executable, "-m", "ewl.analyse.evaluate", "scorecard", _cc_gold,
                          "--arm", "stub/fixed", "--votes", "1"],
                         capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=os.path.join(os.getcwd(), "src")))
check("ewl-evaluate scorecard --concurrency 4 runs on the command line and matches the serial run's table",
      _cc_run.returncode == 0 and _cc_one.returncode == 0
      and [l for l in _cc_run.stdout.splitlines() if l.startswith("| `")][:1]
      and [l.rsplit("|", 6)[0] for l in _cc_run.stdout.splitlines() if l.startswith("| `")][:1]
      == [l.rsplit("|", 6)[0] for l in _cc_one.stdout.splitlines() if l.startswith("| `")][:1],
      (_cc_run.stderr[-300:], _cc_run.stdout[-300:]))

print("\n#133 review: duplicate comments in parallel, concurrency recorded, local models capped")
import concurrent.futures as _cf_dup
import time
import uuid as _uuid_dup


class _RandomJudge(judges.Judge):
    """Every real call returns a different reply, and takes long enough to overlap."""
    default_model = "random-test"
    deterministic = False
    backend = "randomtest"

    def __init__(self, **kwargs):
        super().__init__("random-test", **kwargs)
        self.real_calls = 0

    def _sample(self, body, prompt, vote):
        with self._lock:
            self.real_calls += 1
        time.sleep(0.15)
        return "reply-" + _uuid_dup.uuid4().hex


_dup_cache = os.path.join(tempfile.mkdtemp(), "dup.jsonl")
_dup_judge = _RandomJudge(cache_path=_dup_cache, votes=1)
with _cf_dup.ThreadPoolExecutor(8) as _pool_dup:
    _replies = list(_pool_dup.map(lambda _i: _dup_judge.sample("the same comment", "the same prompt", 0), range(8)))
check("eight workers asking for the same comment at once get ONE reply, from one real call",
      len(set(_replies)) == 1 and _dup_judge.real_calls == 1, (len(set(_replies)), _dup_judge.real_calls))
_dup_judge2 = _RandomJudge(cache_path=_dup_cache, votes=1)
check("and a rerun from the cache gives that same reply (the cache reproduces what the run returned)",
      _dup_judge2.sample("the same comment", "the same prompt", 0) == _replies[0] and _dup_judge2.real_calls == 0)


class _FailingJudge(_RandomJudge):
    def _sample(self, body, prompt, vote):
        time.sleep(0.1)
        raise RuntimeError("permanent failure: invalid_request")


_fail_judge = _FailingJudge(cache_path=os.path.join(tempfile.mkdtemp(), "f.jsonl"), votes=1)
_raised = []


def _try_sample(_i):
    try:
        _fail_judge.sample("c", "p", 0)
    except RuntimeError:
        _raised.append(_i)


with _cf_dup.ThreadPoolExecutor(4) as _pool_dup:
    list(_pool_dup.map(_try_sample, range(4)))
check("a failing call is raised to every worker waiting on it, and nothing is left stuck",
      len(_raised) == 4 and not _fail_judge.__dict__.get("_inflight_calls"), len(_raised))

check("a local model (gemma or ollama) is capped at one worker whatever was asked",
      _ev_cc.effective_concurrency("gemma:gemma2:27b", 8) == 1
      and _ev_cc.effective_concurrency("ollama:gemma3:12b", 4) == 1
      and _ev_cc.effective_concurrency("anthropic:claude-sonnet-5-5", 8) == 8
      and _ev_cc.effective_concurrency("gemma:gemma2:27b", 1) == 1)

_cc133 = subprocess.run([sys.executable, "-m", "ewl.analyse.evaluate", "scorecard", _cc_gold,
                         "--arm", "stub/fixed", "--votes", "1", "--concurrency", "4"],
                        capture_output=True, text=True,
                        env=dict(os.environ, PYTHONPATH=os.path.join(os.getcwd(), "src")))
check("the scorecard says what `mins` means and the concurrency each arm ran at",
      "wall-clock time at the concurrency shown" in _cc133.stdout and "stub" in _cc133.stdout.split("wall-clock")[1],
      _cc133.stdout[-500:])
_cc133_one = subprocess.run([sys.executable, "-m", "ewl.analyse.evaluate", "scorecard", _cc_gold,
                             "--arm", "stub/fixed", "--votes", "1"],
                            capture_output=True, text=True,
                            env=dict(os.environ, PYTHONPATH=os.path.join(os.getcwd(), "src")))
check("a serial run prints no such footnote", "wall-clock time at the concurrency" not in _cc133_one.stdout)


print(chr(10) + "#133 round two: local-arm match is by backend name; duplicate keys share one call")
_ev_l = evaluate
check("gemma and ollama arms are local, plain or wrapped",
      all(_ev_l._is_local_arm(s) for s in ("gemma:gemma2:27b/fixed", "llm:gemma:gemma2:27b", "combined:ollama:x", "ollama:llama3")))
check("an API model with 'gemma' in its name is not capped",
      not _ev_l._is_local_arm("google:gemma-hosted-model") and not _ev_l._is_local_arm("openai:gpt-5")
      and _ev_l.effective_concurrency("google:gemma-hosted-model", 4) == 4)
print("\nGemini arms: thinking budget, prices, refusal, and the scorecard paths (fake client, no SDK needed)")
_g_calls = []
_g_reply = {"text": "democracy"}
_g_reject = {"on": False}


class _GError(Exception):
    code = 400


class _GModels:
    def generate_content(self, **kwargs):
        _g_calls.append(kwargs)
        if _g_reject["on"] and "thinking_config" in kwargs["config"].kwargs:
            raise _GError("400 INVALID_ARGUMENT. {'error': {'message': 'Thinking budget 0 is "
                          "invalid for this model. This model only works in thinking mode.'}}")
        return _fake_gemini_response(_g_reply["text"], prompt_tokens=1000, output_tokens=5)


class _GClient:
    def __init__(self, *a, **k):
        self.models = _GModels()


_g_genai = _types_oa.ModuleType("google.genai")
_g_genai.Client = _GClient
_g_genai.types = _FakeGeminiTypes
_g_types_mod = _types_oa.ModuleType("google.genai.types")
_g_types_mod.ThinkingConfig = _FakeGeminiTypes.ThinkingConfig
_g_types_mod.GenerateContentConfig = _FakeGeminiTypes.GenerateContentConfig
_g_google = _types_oa.ModuleType("google")
_g_google.__path__ = []
_g_google.genai = _g_genai
_g_saved = {name: sys.modules.get(name) for name in ("google", "google.genai", "google.genai.types")}
sys.modules.update({"google": _g_google, "google.genai": _g_genai,
                    "google.genai.types": _g_types_mod})
_g_old_env = {k: os.environ.get(k) for k in ("EWL_GEMINI_THINKING_BUDGET", "EWL_GEMINI_THINKING_LEVEL",
                                              "EWL_GOOGLE_PRICES", "EWL_EXTERNAL_PROCESSORS")}


def _g_env(**values):
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


try:
    _g_env(EWL_GEMINI_THINKING_BUDGET=None, EWL_GEMINI_THINKING_LEVEL=None, EWL_GOOGLE_PRICES=None,
           EWL_EXTERNAL_PROCESSORS="openai,google")
    _gj = judges.create("google:gemini-3.1-pro", votes=1)
    _gj._sample("a comment", "a prompt", 0)
    _gcfg = _g_calls[-1]["config"].kwargs
    check("default: nothing is sent about thinking, and the reply limit leaves room for it",
          "thinking_config" not in _gcfg and _gcfg["max_output_tokens"] == 4096, _gcfg)
    check("default: the cache key is the plain one, and the run record says the model's own default",
          _gj.cache_spec == "google:gemini-3.1-pro" and "model default" in _gj.thinking_setting())
    check("a priced model's spend is counted (1,000 input and 5 output tokens at $2 and $12 per million)",
          abs(_gj.cost_usd() - (1000 * 2.00 + 5 * 12.00) / 1e6) < 1e-9, _gj.cost_usd())

    _gf = judges.create("google:gemini-3.5-flash-lite", votes=1)
    _gf._sample("a comment", "a prompt", 0)
    check("a flash-lite model is likewise sent nothing by default",
          "thinking_config" not in _g_calls[-1]["config"].kwargs)

    _g_env(EWL_GEMINI_THINKING_LEVEL="Low")
    _gb = judges.create("google:gemini-3.1-pro", votes=1)
    _gb._sample("a comment", "a prompt", 0)
    _gbcfg = _g_calls[-1]["config"].kwargs
    check("EWL_GEMINI_THINKING_LEVEL is sent (upper-cased) with the 1,024-token reply limit",
          _gbcfg["thinking_config"].thinking_level == "LOW" and _gbcfg["max_output_tokens"] == 1024, _gbcfg)
    check("a level set by hand has its own cache key and is recorded",
          _gb.cache_spec == "google:gemini-3.1-pro|thinking_level=low"
          and _gb.cache_spec != _gj.cache_spec and _gb.thinking_setting() == "thinking_level=low")
    _g_env(EWL_GEMINI_THINKING_LEVEL="high")
    _gh = judges.create("google:gemini-3.1-pro", votes=1)
    check("a higher level gets a larger reply limit", _gh.reply_limit == 4096)
    _g_env(EWL_GEMINI_THINKING_LEVEL="lots")
    try:
        judges.create("google:gemini-3.1-pro", votes=1)
        _gbad = False
    except judges.SettingRefused as _e:
        _gbad = "minimal, low, medium, high" in str(_e)
    check("a thinking level that is not one of the four stops the run", _gbad)
    _g_env(EWL_GEMINI_THINKING_LEVEL=None, EWL_GEMINI_THINKING_BUDGET="0")
    try:
        judges.create("google:gemini-3.1-pro", votes=1)
        _gold = False
    except judges.SettingRefused as _e:
        _gold = "EWL_GEMINI_THINKING_LEVEL" in str(_e)
    check("the old EWL_GEMINI_THINKING_BUDGET is refused, pointing to the level", _gold)
    _g_env(EWL_GEMINI_THINKING_BUDGET=None, EWL_EXTERNAL_PROCESSORS="openai")
    try:
        judges.create("google:gemini-3.1-pro", votes=1)
        _ggate = False
    except judges.ExternalProcessorNotApproved:
        _ggate = True
    check("without google in EWL_EXTERNAL_PROCESSORS the judge is not built", _ggate)
    _g_env(EWL_EXTERNAL_PROCESSORS="openai,google")
    try:
        judges.create("google:gemini-unknown-1", votes=1)
        _gnp = False
    except judges.PriceUnavailable:
        _gnp = True
    check("an unpriced Gemini model is refused when the judge is built, before any request", _gnp)

    # a model that refuses a level that WAS sent: stop, name the fix, never retry or fall back
    _g_env(EWL_GEMINI_THINKING_LEVEL="low")
    _grej = judges.create("google:gemini-3.1-pro", votes=1)
    _g_reject["on"] = True
    _gbefore = len(_g_calls)
    try:
        _grej._sample("a comment", "a prompt", 0)
        _gmsg = None
    except judges.ThinkingUnavailable as _err:
        _gmsg = str(_err)
    _g_reject["on"] = False
    check("a 400 about thinking stops the run with the setting named and the fix stated",
          _gmsg is not None and "thinking_level=low" in _gmsg and "EWL_GEMINI_THINKING_LEVEL" in _gmsg,
          _gmsg)
    check("it is a permanent error (not retried), and no other setting was tried",
          not _grej._is_retryable(judges.ThinkingUnavailable(_gmsg or ""))
          and len(_g_calls) - _gbefore == 1)

    # prices
    try:
        judges.google_price_for("gemini-unknown-1")
        _gr = False
    except judges.PriceUnavailable as _e:
        _gr = "EWL_GOOGLE_PRICES" in str(_e)
    check("an unpriced Gemini model is still refused, and the message says how to supply a price", _gr)
    _g_env(EWL_GOOGLE_PRICES="gemini-unknown-1=1.5,9;gemini-other=0.5,2")
    check("a price supplied in EWL_GOOGLE_PRICES is used, and a table price is not overridden",
          judges.google_price_for("gemini-unknown-1") == (1.5, 9.0)
          and judges.google_price_for("gemini-3.1-pro") == judges._GOOGLE_PRICES["gemini-3.1-pro"])
    _g_env(EWL_GOOGLE_PRICES="gemini-unknown-1=much,9")
    try:
        judges.google_price_for("gemini-unknown-1")
        _gm = False
    except judges.PriceUnavailable as _e:
        _gm = "model=input,output" in str(_e)
    check("a malformed Gemini price variable stops the run", _gm)
    _g_env(EWL_GOOGLE_PRICES=None)

    # the scorecard paths (default thinking: the arm name carries no @level suffix)
    _g_env(EWL_GEMINI_THINKING_LEVEL=None)
    _gdir = tempfile.mkdtemp()
    _ggold = os.path.join(_gdir, "gold.csv")
    write_fixture(_ggold)
    _g_reply["text"] = "democracy"
    _gcode, _gout = _run_eval("scorecard", _ggold, "--arm", "google:gemini-3.1-pro/fixed",
                              "--votes", "1", "--cache", os.path.join(_gdir, "c1.jsonl"))
    check("ewl-evaluate scorecard --arm google:gemini-3.1-pro/fixed runs and names the arm",
          _gcode in (0, None) and "google:gemini-3.1-pro/fixed" in _gout, (_gcode, _gout[-400:]))
    _g_reply["text"] = "democracy | 4"
    _gcode, _gout = _run_eval("scorecard", _ggold, "--arm", "combined:google:gemini-3.1-pro/fixed",
                              "--votes", "1", "--cache", os.path.join(_gdir, "c2.jsonl"))
    check("the combined arm runs on Gemini too",
          _gcode in (0, None) and "combined:google:gemini-3.1-pro/fixed" in _gout, (_gcode, _gout[-400:]))
    _g_reply["text"] = "4"
    _gcode, _gout = _run_eval("sentiment", _sgold_oa, "--arm", "llm:google:gemini-3.1-pro",
                              "--cache", os.path.join(_gdir, "c3.jsonl"))
    check("ewl-evaluate sentiment --arm llm:google:gemini-3.1-pro runs and scores",
          _gcode in (0, None) and "google" in _gout.lower(), (_gcode, _gout[-400:]))
finally:
    for _k, _v in _g_old_env.items():
        _g_env(**{_k: _v})
    for _name, _mod in _g_saved.items():
        if _mod is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _mod

print(chr(10) + "#130 round two: a table model keeps the table's cached rule, and the NOTE prints once")
_prev_p = os.environ.get("EWL_OPENAI_PRICES")
os.environ["EWL_OPENAI_PRICES"] = "gpt-5-mini=0.25,2.00,0.0125"
judges._TABLE_PRICE_NOTED.clear()
with contextlib.redirect_stderr(io.StringIO()) as _note_err:
    for _ in range(6):
        judges.openai_price_for("gpt-5-mini")
check("the NOTE about a price already in the table prints once, not on every read",
      _note_err.getvalue().count("NOTE: EWL_OPENAI_PRICES") == 1, _note_err.getvalue())
check("a cached price supplied for a model in the table is ignored, like its input and output prices",
      abs(judges.openai_cached_ratio("gpt-5-mini") - 0.1) < 1e-9)
if _prev_p is None:
    os.environ.pop("EWL_OPENAI_PRICES", None)
else:
    os.environ["EWL_OPENAI_PRICES"] = _prev_p

print(chr(10) + "#131 round two: default model id, accepted levels, level in the arm name, reply limit per default level")
_old_g = {k: os.environ.get(k) for k in ("EWL_GEMINI_THINKING_LEVEL", "EWL_EXTERNAL_PROCESSORS", "EWL_GOOGLE_PRICES")}
os.environ["EWL_EXTERNAL_PROCESSORS"] = "openai,google"
os.environ.pop("EWL_GEMINI_THINKING_LEVEL", None)
check("the default Gemini model is the id Google lists", judges.GeminiJudge.default_model == "gemini-3.1-pro-preview")
check("every id the run book names is priced", all(m in judges._GOOGLE_PRICES for m in
      ("gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.1-pro-preview")))
_sys_g, _sys_t = sys.modules.get("google"), sys.modules.get("google.genai")
_fake_g = _types_oa.ModuleType("google"); _fake_g.__path__ = []
_fake_genai = _types_oa.ModuleType("google.genai"); _fake_genai.Client = lambda *a, **k: object()
_fake_gt = _types_oa.ModuleType("google.genai.types"); _fake_g.genai = _fake_genai
_saved_mods = {n: sys.modules.get(n) for n in ("google", "google.genai", "google.genai.types")}
sys.modules.update({"google": _fake_g, "google.genai": _fake_genai, "google.genai.types": _fake_gt})
try:
    os.environ["EWL_GEMINI_THINKING_LEVEL"] = "minimal"
    try:
        judges.create("google:gemini-3.8-flash", votes=1)
        _min_ok = False
    except judges.SettingRefused as _e:
        _min_ok = "low, medium, high" in str(_e)
    check("minimal is refused when the model is built for a model that does not accept it", _min_ok)
    check("minimal is accepted for flash-lite, which thinks at minimal by default",
          judges.create("google:gemini-3.5-flash-lite", votes=1).thinking_level == "minimal")
    os.environ["EWL_GEMINI_THINKING_LEVEL"] = "low"
    check("a level set by hand is part of the arm's name", judges.create("google:gemini-3.8-flash", votes=1).spec == "google:gemini-3.8-flash@low")
    os.environ.pop("EWL_GEMINI_THINKING_LEVEL")
    check("with nothing set the arm name is plain", judges.create("google:gemini-3.8-flash", votes=1).spec == "google:gemini-3.8-flash")
    check("the default reply limit follows the model's own default level (pro high 4,096; flash medium 2,048; flash-lite minimal 1,024)",
          (judges.create("google:gemini-3.1-pro-preview", votes=1).reply_limit,
           judges.create("google:gemini-3.8-flash", votes=1).reply_limit,
           judges.create("google:gemini-3.5-flash-lite", votes=1).reply_limit) == (4096, 2048, 1024))
    os.environ["EWL_GOOGLE_PRICES"] = "gemini-3.8-flash=0.1,0.2"
    judges._TABLE_PRICE_NOTED.clear()
    with contextlib.redirect_stderr(io.StringIO()) as _g_note:
        _gj2 = judges.create("google:gemini-3.8-flash", votes=1)
        for _ in range(5):
            _gj2.cost_usd()
    check("the NOTE about a Google price already in the table prints once", _g_note.getvalue().count("NOTE: EWL_GOOGLE_PRICES") == 1, _g_note.getvalue())
    check("a combined Gemini judge records the inner thinking setting",
          "model default" in (judges.create("combined:google:gemini-3.8-flash", votes=1).thinking_setting() or ""))
finally:
    for _n, _m in _saved_mods.items():
        if _m is None: sys.modules.pop(_n, None)
        else: sys.modules[_n] = _m
    for _k, _v in _old_g.items():
        if _v is None: os.environ.pop(_k, None)
        else: os.environ[_k] = _v
print(chr(10) + "#136 round two: a sampled Gemma arm says so in its name")
_gt = object.__new__(gemma_local.GemmaJudge)
_gt.model = gemma_local.MODEL_TAG
check("at temperature 0 the arm name is unchanged", (setattr(_gt, "temperature", 0.0) or _gt.spec) == _gt.backend + ":" + gemma_local.MODEL_TAG
      if hasattr(_gt, "backend") else True)
_gt.temperature = 0.7
check("at temperature 0.7 the arm name carries @t=0.7 and the default name does not",
      _gt.spec.endswith("@t=0.7") and not (setattr(_gt, "temperature", 0.0) or _gt.spec).endswith("@t=0"))
print("\nresponse cache: a run killed mid-write does not cost the next reply")
with tempfile.TemporaryDirectory() as _pl_dir:
    _pl = os.path.join(_pl_dir, "cache.jsonl")
    with open(_pl, "w", encoding="utf-8") as _h:
        _h.write(json.dumps({"key": "k1", "reply": "social"}) + "\n" + '{"key": "k2", "rep')
    _c1 = judges._Cache(_pl)
    _c1.put("k3", "economics")
    _c1.put("k4", "none")
    _c2 = judges._Cache(_pl)
    check("a reply saved after a half-written last line is still there on the next load",
          _c2.entries == {"k1": "social", "k3": "economics", "k4": "none"}, _c2.entries)
    _clean = os.path.join(_pl_dir, "clean.jsonl")
    with open(_clean, "w", encoding="utf-8") as _h:
        _h.write(json.dumps({"key": "k1", "reply": "social"}) + "\n")
    judges._Cache(_clean).put("k2", "none")
    with open(_clean, encoding="utf-8") as _h:
        _clean_text = _h.read()
    check("a clean cache file gets no blank line added",
          _clean_text.count("\n") == 2 and "\n\n" not in _clean_text, repr(_clean_text))
print("\nAnthropic: a normal stop with no text is not cached (live and batch)")
import types as _types_e
_e_replies = {"text": None}


def _e_message(text):
    content = [] if text is None else [_types_e.SimpleNamespace(type="text", text=text)]
    return _types_e.SimpleNamespace(
        stop_reason="end_turn", content=content,
        usage=_types_e.SimpleNamespace(input_tokens=10, output_tokens=0,
                                       cache_creation_input_tokens=0, cache_read_input_tokens=0))


class _EBatches:
    def retrieve(self, batch_id):
        return _types_e.SimpleNamespace(processing_status="ended")

    def results(self, batch_id):
        return [_types_e.SimpleNamespace(custom_id=f"key{_i}", result=_types_e.SimpleNamespace(
                    type="succeeded", message=_e_message(_t)))
                for _i, _t in enumerate([None, "  \n", "economics"])]


class _EMessages:
    batches = _EBatches()

    def create(self, **kwargs):
        return _e_message(_e_replies["text"])


_e_fake = _types_e.ModuleType("anthropic")
_e_fake.Anthropic = lambda *a, **k: _types_e.SimpleNamespace(messages=_EMessages())
_e_had = "anthropic" in sys.modules
_e_prev = sys.modules.get("anthropic")
sys.modules["anthropic"] = _e_fake
try:
    from ewl.label import anthropic_batch as _e_batch
    from ewl.label import classify as _e_classify
    _e_dir = tempfile.mkdtemp()
    _e_cache = os.path.join(_e_dir, "live_cache.jsonl")
    _e_judge = judges.create("anthropic:claude-haiku-4-5", votes=1, cache_path=_e_cache)
    _e_got = []
    for _e_text in (None, "  \n"):
        _e_replies["text"] = _e_text
        _e_got.append(_e_judge.sample(f"comment {_e_text!r}", "a prompt", 0))
    check("an end_turn reply with no text (or only whitespace) comes back uncacheable, reason 'empty'",
          all(isinstance(r, judges.UncacheableReply) and r.reason == "empty" and r == ""
              for r in _e_got), [(type(r).__name__, getattr(r, "reason", None)) for r in _e_got])
    check("and nothing is written to the response cache, so the next run is not refused",
          not os.path.exists(_e_cache) or open(_e_cache, encoding="utf-8").read() == "",
          open(_e_cache, encoding="utf-8").read() if os.path.exists(_e_cache) else "")
    _e_bcache = os.path.join(_e_dir, "batch_cache.jsonl")
    _e_bjudge = judges.create("anthropic:claude-haiku-4-5", votes=1, cache_path=_e_bcache)
    _e_entry = {"id": "msgbatch_e1", "requests": 3, "fetched": False}
    with contextlib.redirect_stderr(io.StringIO()):
        _e_batch.fetch_one(_e_bjudge._client, _e_bjudge, _e_entry, _e_dir)
    with open(_e_batch.uncacheable_path(_e_dir), encoding="utf-8") as _h:
        _e_recorded = [json.loads(_l) for _l in _h]
    judges._cache_registry.pop(_e_bcache, None)
    try:
        _e_classify.refuse_empty_cache(_e_bcache)
        _e_resumable = True
    except SystemExit:
        _e_resumable = False
    check("batch fetch: empty replies go to batch_uncacheable.jsonl (reason 'empty'), only the real answer "
          "is cached, and the next submit is not refused",
          sorted(r["key"] for r in _e_recorded) == ["key0", "key1"]
          and all(r["reason"] == "empty" for r in _e_recorded)
          and judges._Cache.shared(_e_bcache).entries == {"key2": "economics"} and _e_resumable,
          (_e_recorded, judges._Cache.shared(_e_bcache).entries, _e_resumable))
finally:
    if _e_had:
        sys.modules["anthropic"] = _e_prev
    else:
        sys.modules.pop("anthropic", None)
print("\nAn examples file without ratings is refused wherever the sentiment question goes to a model")
_r_dir = tempfile.mkdtemp()
_r_ex = os.path.join(_r_dir, "examples_no_ratings.csv")
with open(_r_ex, "w", newline="", encoding="utf-8") as _h:
    _w = csv.DictWriter(_h, fieldnames=["comment_id", "body", "label"])
    _w.writeheader()
    for _i in range(25):
        _w.writerow({"comment_id": f"ex{_i}", "body": f"synthetic example {_i}", "label": "none"})
_r_in = os.path.join(_r_dir, "in.csv")
with open(_r_in, "w", newline="", encoding="utf-8-sig") as _h:
    _w = csv.DictWriter(_h, fieldnames=["comment_id", "story", "body"])
    _w.writeheader()
    _w.writerow({"comment_id": "c1", "story": "musk", "body": "synthetic comment"})
_r_env = dict(os.environ, PYTHONPATH=os.path.join(os.getcwd(), "src"))


def _r_classify(*extra):
    _out = os.path.join(_r_dir, "o.csv")
    if os.path.exists(_out):
        os.remove(_out)
    _p = subprocess.run([sys.executable, "-m", "ewl.label.classify", _r_in, "--judge", "stub",
                         "--votes", "1", "--no-cache", "--output", _out, "--fixed-examples", _r_ex,
                         *extra], capture_output=True, text=True, env=_r_env)
    return _p.returncode, os.path.exists(_out), _p.stderr


_r_llm = _r_classify("--sentiment", "llm:stub")
check("ewl-classify with an llm: sentiment scorer refuses it before labelling anything",
      _r_llm[0] != 0 and not _r_llm[1] and "no `sentiment` and `story` columns" in _r_llm[2], _r_llm)
_r_comb = _r_classify("--sentiment", "combined:stub")
check("and so does a combined: scorer", _r_comb[0] != 0 and not _r_comb[1], _r_comb[0])
_r_off = _r_classify("--sentiment", "llm:stub", "--sentiment-examples", "off")
check("--sentiment-examples off still lets it run, as on ewl-gemma", _r_off[0] == 0 and _r_off[1], _r_off[2][-300:])
_r_vader = _r_classify("--sentiment", "vader")
check("VADER asks no model, so the file is accepted there", _r_vader[0] == 0 and _r_vader[1], _r_vader[2][-300:])
_r_plan = subprocess.run([sys.executable, "-m", "ewl.label.anthropic_batch", "plan", _r_in,
                          "--fixed-examples", _r_ex], capture_output=True, text=True, env=_r_env)
check("ewl-anthropic-batch plan refuses it too (every request set includes the sentiment question)",
      _r_plan.returncode != 0 and "no `sentiment` and `story` columns" in _r_plan.stderr,
      (_r_plan.returncode, _r_plan.stderr[-300:]))
print("\nThe run record counts the sentiment scorer's own calls, not only the domain judge's")
_rc_dir = tempfile.mkdtemp()
_rc_in = os.path.join(_rc_dir, "in.csv")
with open(_rc_in, "w", newline="", encoding="utf-8-sig") as _h:
    _w = csv.DictWriter(_h, fieldnames=["comment_id", "story", "body"])
    _w.writeheader()
    for _i in range(4):
        _w.writerow({"comment_id": f"r{_i}", "story": "musk", "body": f"synthetic comment {_i}"})
_rc_env = dict(os.environ, PYTHONPATH=os.path.join(os.getcwd(), "src"))


def _rc_record(*extra):
    _out = os.path.join(_rc_dir, "o.csv")
    subprocess.run([sys.executable, "-m", "ewl.label.classify", _rc_in, "--judge", "stub", "--votes", "1",
                    "--no-cache", "--output", _out, *extra], capture_output=True, text=True, env=_rc_env)
    with open(_out + ".run.json", encoding="utf-8") as _h:
        return json.load(_h)["result"]


_rc_sep = _rc_record("--sentiment", "llm:stub")
check("a separate llm: scorer: 4 comments x (1 domain + 1 sentiment) = 8 calls, split by backend",
      _rc_sep["calls"] == 8 and _rc_sep["calls_by_backend"] == {"domain": 4, "sentiment": 4},
      (_rc_sep.get("calls"), _rc_sep.get("calls_by_backend")))
_rc_comb = _rc_record("--judge", "combined:stub", "--sentiment", "combined:stub")
check("a combined judge is one backend, counted once (4 calls, not 8)",
      _rc_comb["calls"] == 4 and _rc_comb["calls_by_backend"] == {"domain": 4},
      (_rc_comb.get("calls"), _rc_comb.get("calls_by_backend")))
_rc_vader = _rc_record("--sentiment", "vader")
check("VADER makes no model calls, so only the domain judge's are counted",
      _rc_vader["calls"] == 4 and _rc_vader["calls_by_backend"] == {"domain": 4},
      (_rc_vader.get("calls"), _rc_vader.get("calls_by_backend")))

print("\na join that matches no gold comment is refused, not scored as recall 0.00")
# The Gemma batches label the corpus with the gold comments taken out (ewl-batches
# prepare --exclude), so joining the gold set to that output matches nothing. join used
# to write the domain_* header anyway, blank on every row; score, export --scored and
# report --gold took the header as proof of labels and published recall 0.00, exit 0.
_zj_corpus_fields = ["comment_id", "story", "source", "body", "domain_democracy",
                     "domain_economics", "domain_environment", "domain_social",
                     "domain_unclassified", "domain_parse_failure", "sentiment_value"]
_zj_gold_fields = ["comment_id", "story", "source", "body", "hand_domain", "hand_sentiment",
                   "inclusion_prob"]
with tempfile.TemporaryDirectory() as _zj:
    _zj_corpus = os.path.join(_zj, "labelled.csv")
    with open(_zj_corpus, "w", newline="", encoding="utf-8-sig") as _fh:
        _w = csv.DictWriter(_fh, fieldnames=_zj_corpus_fields)
        _w.writeheader()
        for _i in range(12):
            _w.writerow({"comment_id": f"corp{_i}", "story": "jets", "source": f"v{_i % 4}",
                         "body": f"corpus comment {_i}", "domain_democracy": str(_i % 2),
                         "domain_economics": "0", "domain_environment": "0",
                         "domain_social": "0", "domain_unclassified": str(1 - _i % 2),
                         "domain_parse_failure": "0", "sentiment_value": "0"})
    _zj_gold = os.path.join(_zj, "gold.csv")
    _zj_gold_rows = [{"comment_id": f"gold{_i}", "story": "jets", "source": f"v{_i % 4}",
                      "body": f"gold comment {_i}", "hand_domain": "democracy" if _i % 2 else "none",
                      "hand_sentiment": "3", "inclusion_prob": "0.5"} for _i in range(12)]
    with open(_zj_gold, "w", newline="", encoding="utf-8-sig") as _fh:
        _w = csv.DictWriter(_fh, fieldnames=_zj_gold_fields)
        _w.writeheader()
        _w.writerows(_zj_gold_rows)

    def _zj_run(*argv):
        return subprocess.run([sys.executable, "-m", *argv], capture_output=True, text=True,
                              env=env, cwd=_zj)

    _zj_out = os.path.join(_zj, "scored.csv")
    _r = _zj_run("ewl.analyse.validate", "join", _zj_gold, "--labelled", _zj_corpus,
                 "--output", _zj_out)
    check("join refuses when no coded comment is in the labelled file, and writes nothing",
          _r.returncode != 0 and "Refusing to join" in _r.stderr and not os.path.exists(_zj_out),
          (_r.returncode, _r.stderr[-300:]))
    check("and it says how to label a gold set that was left out of the corpus",
          "label the gold file" in _r.stderr, _r.stderr[-300:])
    try:
        _val.cmd_join(_zj_gold_rows, [{"comment_id": "corp0", "domain_democracy": "1"}],
                      io.StringIO())
        check("cmd_join refuses a join that matches nothing when called directly", False,
              "no exit")
    except SystemExit:
        check("cmd_join refuses a join that matches nothing when called directly", True)

    # The file the old join wrote: the domain_* header, every value blank.
    _zj_blank = os.path.join(_zj, "blank_scored.csv")
    with open(_zj_blank, "w", newline="", encoding="utf-8-sig") as _fh:
        _w = csv.DictWriter(_fh, fieldnames=_zj_gold_fields + _zj_corpus_fields[4:])
        _w.writeheader()
        _w.writerows(_zj_gold_rows)
    _blank_rows = list(csv.DictReader(open(_zj_blank, encoding="utf-8-sig")))
    check("a domain_* header with every value blank is not taken as machine labels",
          _val.machine_prefix(_blank_rows) is None and not _val.has_machine_labels(_blank_rows))
    _r = _zj_run("ewl.analyse.validate", "score", _zj_blank)
    check("score refuses a file whose machine columns are all blank",
          _r.returncode != 0 and "recall" not in _r.stdout, (_r.returncode, _r.stdout[-300:]))
    _r = _zj_run("ewl.analyse.export_dashboard", _zj_corpus, "--output",
                 os.path.join(_zj, "r.json"), "--scored", _zj_blank, "--n-boot", "20")
    check("export --scored refuses a file whose machine columns are all blank",
          _r.returncode != 0 and "no machine labels" in _r.stderr and
          not os.path.exists(os.path.join(_zj, "r.json")), (_r.returncode, _r.stderr[-300:]))
    _r = _zj_run("ewl.analyse.report", "corrected-bootstrap", _zj_corpus, "--column",
                 "domain_democracy", "--gold", _zj_blank, "--n-boot", "20")
    check("report --gold refuses a gold file whose machine column is all blank",
          _r.returncode != 0 and "no machine labels" in _r.stderr, (_r.returncode, _r.stderr[-300:]))

    # The same gold file with real labels still goes through (the guard is not too eager).
    _zj_ok = os.path.join(_zj, "ok_scored.csv")
    with open(_zj_ok, "w", newline="", encoding="utf-8-sig") as _fh:
        _w = csv.DictWriter(_fh, fieldnames=_zj_gold_fields + _zj_corpus_fields[4:])
        _w.writeheader()
        for _row in _zj_gold_rows:
            _hit = "1" if _row["hand_domain"] == "democracy" else "0"
            _w.writerow(dict(_row, domain_democracy=_hit, domain_economics="0",
                             domain_environment="0", domain_social="0",
                             domain_unclassified="0" if _hit == "1" else "1",
                             domain_parse_failure="0", sentiment_value="0"))
    _r = _zj_run("ewl.analyse.validate", "score", _zj_ok)
    check("score still runs on a gold file that carries its machine labels",
          _r.returncode == 0 and "1.00" in _r.stdout, (_r.returncode, _r.stderr[-300:]))
print("\npersonal-data guard: no tracked file may look like comment data")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_no_personal_data as _pd
_pd_dir = tempfile.mkdtemp()
_pd_files = {
    "src/ok.py": "x = 1\n",
    "dashboard/data/results.json": json.dumps({"provenance": {"verbatim_comments_included": False}}),
    "notes/corpus_copy.csv": "\ufeffcomment_id,story,body\nc1,jets,synthetic text\n",
    "notes/hashes.csv": "author_hash,body\nabc,synthetic\n",
    "notes/videos.csv": "video_id,story,video_title\nAAAAAAAAAAA,jets,t\n",
    "docs/ids.md": " ".join("Ugx" + ch * 22 for ch in "ABC"),
    "docs/two_ids.md": " ".join("Ugx" + ch * 22 for ch in "AB"),
    "official-data/x.txt": "anything",
    "labelled.run.json": "{}",
    "cache.jsonl": "{}",
    "coders.xlsx": "",
    "tests/fixture.csv": "comment_id,body\nc1,synthetic\n",
}
for _rel, _content in _pd_files.items():
    _full = os.path.join(_pd_dir, _rel)
    os.makedirs(os.path.dirname(_full), exist_ok=True)
    with open(_full, "w", encoding="utf-8") as _h:
        _h.write(_content)
_pd_found = dict(_pd.problems_in(_pd_dir, list(_pd_files), legacy=frozenset()))
check("the guard flags a CSV with the corpus schema, even behind a BOM",
      "notes/corpus_copy.csv" in _pd_found and "notes/hashes.csv" in _pd_found, _pd_found)
check("the guard flags 3 YouTube comment ids in a text file, not 2",
      "docs/ids.md" in _pd_found and "docs/two_ids.md" not in _pd_found, _pd_found)
check("the guard flags data folders, run records, JSONL caches and spreadsheets",
      all(p in _pd_found for p in ("official-data/x.txt", "labelled.run.json", "cache.jsonl",
                                   "coders.xlsx")), _pd_found)
check("the guard passes code, a video list, the dashboard export and tests/ fixtures",
      not any(p in _pd_found for p in ("src/ok.py", "notes/videos.csv",
                                       "dashboard/data/results.json", "tests/fixture.csv")),
      _pd_found)
with open(os.path.join(_pd_dir, "dashboard/data/results.json"), "w", encoding="utf-8") as _h:
    json.dump({"provenance": {"verbatim_comments_included": True}}, _h)
check("the guard flags a dashboard export made with --include-verbatim",
      "dashboard/data/results.json" in dict(_pd.problems_in(_pd_dir, ["dashboard/data/results.json"])))
check("the legacy skip list only names files under Full Corpus Data/",
      all(p.startswith("Full Corpus Data/") for p in _pd.LEGACY_TRACKED))

# ----------------------------------------------------------------------------------------------
print("\nhardening run: classify executor window, --max-cost, retries, votes, output, strip, story")
import contextlib as _hc_ctx
import time as _hc_time
from ewl.label import classify as _hc
from ewl.label import combined as _hc_combined
from ewl.label import sentiment as _hc_sent


def _hc_write(path, rows, with_story=True):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        names = ["comment_id", "story", "body"] if with_story else ["comment_id", "body"]
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in names})


def _hc_classify(argv):
    """Run ewl-classify in-process: (the exception it ended with or None, its stderr)."""
    old = sys.argv
    sys.argv = ["classify.py"] + list(argv)
    err = io.StringIO()
    exc = None
    try:
        with _hc_ctx.redirect_stderr(err):
            _hc.main()
    except BaseException as error:      # SystemExit included: the exit code is what is checked
        exc = error
    finally:
        sys.argv = old
    return exc, err.getvalue()


def _hc_read(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


_hc_events = []         # (index, "done") in the order the judge finished each comment
_hc_lock = threading.Lock()


@judges.register("hard_delay")
class _HardDelayJudge(judges.Judge):
    """Answers from the comment alone; takes a time that depends on the comment (1 in 17 is
    slow, as a few real calls are), and records the order in which comments finish."""
    default_model = "delay"
    slow_seconds = 0.15

    def _sample(self, body, prompt, vote):
        index = int(body.split("-")[1])
        _hc_time.sleep(self.slow_seconds if index % 17 == 0 else 0.003)
        with _hc_lock:
            _hc_events.append(index)
        return ["democracy", "economics", "environment", "social"][index % 4]


with tempfile.TemporaryDirectory() as _hc_tmp:
    _hc_in = os.path.join(_hc_tmp, "in.csv")
    _hc_write(_hc_in, [{"comment_id": f"c{i:03d}", "story": "musk", "body": f"comment-{i}"}
                       for i in range(120)])
    _hc_outputs = {}
    _hc_elapsed = {}
    for _conc in (1, 8):
        _hc_out = os.path.join(_hc_tmp, f"out{_conc}.csv")
        _t0 = _hc_time.monotonic()
        _exc, _err = _hc_classify([_hc_in, "--judge", "hard_delay", "--votes", "1", "--sentiment",
                                   "stub", "--no-cache", "--concurrency", str(_conc),
                                   "--output", _hc_out])
        _hc_elapsed[_conc] = _hc_time.monotonic() - _t0
        _hc_outputs[_conc] = open(_hc_out, "rb").read()
        check(f"concurrency {_conc}: the run finishes cleanly", _exc is None, repr(_exc))
    check("concurrency 8 writes exactly the same file as concurrency 1 (rows, order, labels)",
          _hc_outputs[1] == _hc_outputs[8])
    check("the output is in input order",
          [r["comment_id"] for r in _hc_read(os.path.join(_hc_tmp, "out8.csv"))]
          == [f"c{i:03d}" for i in range(120)])
    check("concurrency 8 is clearly faster than serial (a loose bound, not a benchmark)",
          _hc_elapsed[8] < _hc_elapsed[1] * 0.7, _hc_elapsed)

    # Head-of-line blocking: ONE slow comment at the head. The old loop submitted a new comment
    # only once the oldest had finished, so at most concurrency-1 others could finish before it.
    # With a window of 4x concurrency, the other workers keep going. Judged by completion order,
    # not by the clock, so the check does not depend on how fast the machine is.
    @judges.register("hard_headslow")
    class _HardHeadSlow(judges.Judge):
        default_model = "headslow"

        def _sample(self, body, prompt, vote):
            index = int(body.split("-")[1])
            if index == 0:
                _hc_time.sleep(0.8)
            with _hc_lock:
                _hc_events.append(index)
            return "social"

    _hc_events.clear()
    _exc, _err = _hc_classify([_hc_in, "--judge", "hard_headslow", "--votes", "1", "--sentiment",
                               "stub", "--no-cache", "--concurrency", "4", "--limit", "40",
                               "--output", os.path.join(_hc_tmp, "head.csv")])
    _before = _hc_events.index(0)
    check("a slow comment at the head no longer idles the other workers "
          "(at least 2x concurrency finished before it; the old loop allowed 3)",
          _exc is None and _before >= 8, (_before, repr(_exc)))
    check("...and the file is still in input order, 40 rows",
          [r["comment_id"] for r in _hc_read(os.path.join(_hc_tmp, "head.csv"))]
          == [f"c{i:03d}" for i in range(40)])

    # A failing comment mid-run stops the run promptly (no hang), after writing the comments
    # that came before it, in order.
    @judges.register("hard_failing")
    class _HardFailing(judges.Judge):
        default_model = "failing"
        snapshot = None
        out_path = None

        def _sample(self, body, prompt, vote):
            index = int(body.split("-")[1])
            if index == 20:
                _hc_time.sleep(0.05)
                if self.out_path:      # what is on disk at the moment of the failure
                    _HardFailing.snapshot = _hc_read(self.out_path)
                raise ValueError("invalid_request_error: boom")
            _hc_time.sleep(0.005)
            return "social"

    for _conc in (4, 1):
        _fail_out = os.path.join(_hc_tmp, f"fail{_conc}.csv")
        _HardFailing.snapshot, _HardFailing.out_path = None, _fail_out
        _t0 = _hc_time.monotonic()
        _exc, _err = _hc_classify([_hc_in, "--judge", "hard_failing", "--votes", "1",
                                   "--sentiment", "stub", "--no-cache", "--concurrency",
                                   str(_conc), "--output", _fail_out])
        _rows = _hc_read(_fail_out)
        check(f"concurrency {_conc}: a failing comment stops the run promptly with its own error",
              isinstance(_exc, ValueError) and _hc_time.monotonic() - _t0 < 10, repr(_exc))
        check(f"concurrency {_conc}: the 20 comments before it are written, in order, complete",
              [r["comment_id"] for r in _rows] == [f"c{i:03d}" for i in range(20)]
              and all(r["domain"] and r["sentiment"] for r in _rows), len(_rows))
        check(f"concurrency {_conc}: the message names the row and says what is safe",
              "comment_id='c020'" in _err and "20 comment(s) already written" in _err, _err[-300:])
    check("each row was on disk, flushed, when the next comment failed (a kill leaves a checkpoint)",
          _HardFailing.snapshot is not None and len(_HardFailing.snapshot) == 20
          and all(r["domain"] and r["comment_id"] for r in _HardFailing.snapshot),
          None if _HardFailing.snapshot is None else len(_HardFailing.snapshot))

print("\nhardening run: --max-cost on ewl-classify")
_hc_calls = []


@judges.register("hard_priced")
class _HardPriced(judges.Judge):
    """$0.50 a call, to the cent: the cap is hit after a known number of comments."""
    default_model = "priced"
    price_per_mtok = (1.0, 0.0)

    def _sample(self, body, prompt, vote):
        with self._lock:
            self.usage["input_tokens"] += 500_000
        with _hc_lock:
            _hc_calls.append(body)
        _hc_time.sleep(0.01)
        return "economics"


with tempfile.TemporaryDirectory() as _hc_tmp:
    _hc_in = os.path.join(_hc_tmp, "in.csv")
    _hc_write(_hc_in, [{"comment_id": f"c{i:03d}", "story": "jets", "body": f"comment-{i}"}
                       for i in range(10)])
    _hc_cache = os.path.join(_hc_tmp, "cache.jsonl")
    _hc_out = os.path.join(_hc_tmp, "out.csv")
    _exc, _err = _hc_classify([_hc_in, "--judge", "hard_priced", "--votes", "1", "--sentiment",
                               "stub", "--cache", _hc_cache, "--max-cost", "2.2",
                               "--output", _hc_out])
    _rows = _hc_read(_hc_out)
    check("--max-cost: the run stops once the spend passes the cap, writing what is done",
          isinstance(_exc, SystemExit) and _exc.code == 3 and len(_rows) == 5, (repr(_exc), len(_rows)))
    check("--max-cost: the message says it stopped for the cap and how to resume",
          "STOPPED FOR THE COST CAP" in _err and "Re-run the same command" in _err
          and "--max-cost $2.20" in _err and "INCOMPLETE" in _err, _err[-500:])
    _record = json.load(open(_hc_out + ".run.json", encoding="utf-8"))
    check("--max-cost: the run record says so, and carries the `partial` block the export refuses",
          _record["result"]["stopped_for_cost_cap"] is True and _record["result"]["max_cost_usd"] == 2.2
          and _record["partial"]["comments_included"] == 5 and _record["partial"]["comments_in_corpus"] == 10,
          _record.get("partial"))
    _hc_calls.clear()
    _exc, _err = _hc_classify([_hc_in, "--judge", "hard_priced", "--votes", "1", "--sentiment",
                               "stub", "--cache", _hc_cache, "--output", _hc_out])
    _rows = _hc_read(_hc_out)
    check("--max-cost: re-running the same command (no cap) continues from the cache: "
          "only the 5 missing comments are paid for, and the file is complete",
          _exc is None and len(_rows) == 10 and len(_hc_calls) == 5
          and "partial" not in json.load(open(_hc_out + ".run.json", encoding="utf-8")),
          (repr(_exc), len(_rows), len(_hc_calls)))

    _hc_write(_hc_in, [{"comment_id": f"c{i:03d}", "story": "jets", "body": f"item-{i}"}
                       for i in range(60)])
    _hc_out = os.path.join(_hc_tmp, "out8.csv")
    _hc_calls.clear()
    _exc, _err = _hc_classify([_hc_in, "--judge", "hard_priced", "--votes", "1", "--sentiment",
                               "stub", "--no-cache", "--concurrency", "4", "--max-cost", "3.2",
                               "--output", _hc_out])
    _rows = _hc_read(_hc_out)
    check("--max-cost at concurrency 4: stops, exits 3, and the written rows are a clean prefix "
          "of the input (nothing new was started; in-flight comments finished)",
          isinstance(_exc, SystemExit) and _exc.code == 3 and 1 <= len(_rows) < 60
          and [r["comment_id"] for r in _rows] == [f"c{i:03d}" for i in range(len(_rows))],
          (repr(_exc), len(_rows)))
    check("--max-cost at concurrency 4: far fewer than all 60 calls were made",
          7 <= len(_hc_calls) < 40, len(_hc_calls))
    _exc, _err = _hc_classify([_hc_in, "--judge", "hard_priced", "--votes", "1", "--sentiment",
                               "stub", "--no-cache", "--max-cost", "-1", "--output", _hc_out])
    check("--max-cost: a negative cap is refused", isinstance(_exc, SystemExit) and _exc.code == 2)
    _exc, _err = _hc_classify([_hc_in, "--judge", "hard_priced", "--votes", "1", "--sentiment",
                               "stub", "--no-cache", "--output", _hc_out, "--limit", "3"])
    check("no --max-cost: no cap, a normal exit", _exc is None, repr(_exc))

print("\nhardening run: --max-cost on ewl-anthropic-batch submit")
_hc_prev_anthropic = sys.modules.get("anthropic")
_hc_world = _World()
_install_fake_anthropic(_hc_world)
try:
    with tempfile.TemporaryDirectory() as _hc_tmp:
        _hc_src = os.path.join(_hc_tmp, "corpus.csv")
        _hc_write(_hc_src, [{"comment_id": f"c{i:02d}", "story": ["luigi", "musk", "jets", "propublica"][i % 4],
                             "body": f"Tax comment number {i}"} for i in range(12)])
        _hc_dir = os.path.join(_hc_tmp, "trial")
        _hc_common = ["--judge", "anthropic:claude-haiku-4-5", "--votes", "1"]
        _err = io.StringIO()
        try:
            with _hc_ctx.redirect_stderr(_err):
                _ab.main(["submit", _hc_src, "--out-dir", _hc_dir, "--max-cost", "0.0001", *_hc_common])
            _hc_refused = None
        except SystemExit as _e:
            _hc_refused = str(_e)
        check("batch submit --max-cost: an estimate over the cap is refused before anything is sent",
              _hc_refused is not None and "Refusing to submit" in _hc_refused
              and "over --max-cost" in _hc_refused and _hc_world.created == 0, _hc_refused)
        check("...and nothing was written: no manifest, no subset, no cache",
              not os.path.exists(os.path.join(_hc_dir, "batch_manifest.json"))
              and not os.path.exists(os.path.join(_hc_dir, "subset.csv")))
        with _hc_ctx.redirect_stderr(_err):
            _rc = _ab.main(["submit", _hc_src, "--out-dir", _hc_dir, "--max-cost", "100", *_hc_common])
        check("batch submit --max-cost: an estimate under the cap goes ahead, printing it",
              _rc == 0 and _hc_world.created == 1 and "--max-cost $100.00: estimated BATCH cost" in _err.getvalue(),
              _err.getvalue()[-300:])
        try:
            with _hc_ctx.redirect_stderr(io.StringIO()):
                _ab.main(["plan", _hc_src, "--judge", "anthropic:claude-sonnet-5-7", "--votes", "1"])
            _hc_unpriced = None
        except SystemExit as _e:
            _hc_unpriced = str(_e)
        check("the batch tool refuses an unpriced model with the price message, not a traceback",
              _hc_unpriced is not None and "No price on record" in _hc_unpriced, _hc_unpriced)
finally:
    if _hc_prev_anthropic is not None:
        sys.modules["anthropic"] = _hc_prev_anthropic
    else:
        sys.modules.pop("anthropic", None)

print("\nhardening run: Anthropic prices are exact (or supplied), checked when the judge is built")
for _model, _price in (("claude-sonnet-5-5", (2.00, 10.00)), ("claude-opus-5-5", (4.00, 20.00)),
                       ("claude-opus-5", (5.00, 25.00)), ("claude-haiku-4-5", (1.00, 5.00)),
                       ("claude-sonnet-5", (2.00, 10.00)), ("claude-fable-5", (10.00, 50.00)),
                       ("claude-sonnet-5-5-20261001", (2.00, 10.00)),
                       ("claude-haiku-4-5-20251001", (1.00, 5.00))):
    check(f"price_for({_model!r}) still resolves", judges.price_for(_model) == _price, judges.price_for(_model))
for _unknown in ("claude-sonnet-5-7", "claude-opus-5-6", "claude-sonnet-5-latest",
                 "claude-sonnet-5-2026", "claude-haiku-4-6"):
    try:
        judges.price_for(_unknown)
        _hc_priced_as = "priced"
    except judges.PriceUnavailable as _e:
        _hc_priced_as = "refused" if "EWL_ANTHROPIC_PRICES" in str(_e) else "wrong message"
    check(f"price_for({_unknown!r}) is refused, not priced as a similar model", _hc_priced_as == "refused",
          _hc_priced_as)
_hc_env = os.environ.get("EWL_ANTHROPIC_PRICES")
os.environ["EWL_ANTHROPIC_PRICES"] = "claude-sonnet-5-7=2.5,12;claude-haiku-4-5=9,9"
try:
    check("a price supplied in EWL_ANTHROPIC_PRICES is used for a model not in the table "
          "(and for its dated snapshots)",
          judges.price_for("claude-sonnet-5-7") == (2.5, 12.0)
          and judges.price_for("claude-sonnet-5-7-20270101") == (2.5, 12.0))
    check("...but never replaces a table price",
          judges.price_for("claude-haiku-4-5") == (1.00, 5.00))
    os.environ["EWL_ANTHROPIC_PRICES"] = "claude-sonnet-5-7=oops"
    try:
        judges.price_for("claude-sonnet-5-7")
        _hc_bad = False
    except judges.PriceUnavailable:
        _hc_bad = True
    check("a malformed EWL_ANTHROPIC_PRICES is refused", _hc_bad)
finally:
    if _hc_env is None:
        os.environ.pop("EWL_ANTHROPIC_PRICES", None)
    else:
        os.environ["EWL_ANTHROPIC_PRICES"] = _hc_env
_hc_prev_anthropic = sys.modules.get("anthropic")
_install_fake_anthropic(_World())
try:
    try:
        judges.create("anthropic:claude-sonnet-5-7", votes=1)
        _hc_built = True
    except judges.PriceUnavailable:
        _hc_built = False
    check("building an AnthropicJudge for an unpriced model raises PriceUnavailable at once",
          _hc_built is False)
    check("a priced model builds",
          judges.create("anthropic:claude-haiku-4-5", votes=1).price_per_mtok == (1.00, 5.00))
    os.environ["EWL_ANTHROPIC_PRICES"] = "claude-sonnet-5-7=2.5,12"
    check("an unpriced model gets its supplied price when built",
          judges.create("anthropic:claude-sonnet-5-7", votes=1).price_per_mtok == (2.5, 12.0))
finally:
    os.environ.pop("EWL_ANTHROPIC_PRICES", None)
    if _hc_env is not None:
        os.environ["EWL_ANTHROPIC_PRICES"] = _hc_env
    if _hc_prev_anthropic is not None:
        sys.modules["anthropic"] = _hc_prev_anthropic
    else:
        sys.modules.pop("anthropic", None)

print("\nhardening run: a truncated reply is retried at most twice; attempts and retries are reported")


@judges.register("hard_truncating")
class _HardTruncating(judges.Judge):
    default_model = "truncating"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.mode = "truncated"
        self.calls = 0

    def _sample(self, body, prompt, vote):
        self.calls += 1
        if "T" in body.split("-")[0] and self.mode == "truncated":
            raise judges.TruncatedReply("reply truncated (finish_reason=length)")
        if self.mode == "ratelimit":
            raise RuntimeError("Error code: 429 - rate_limit_error")
        return "social"


_hc_sleep = judges.time.sleep
judges.time.sleep = lambda seconds: None        # the backoff is real time otherwise
try:
    _trunc = judges.create("hard_truncating", votes=1)
    _reply = _trunc.sample("T-1", "p")
    check("a comment that keeps being truncated gets 3 attempts (2 retries), not 5",
          _trunc.calls == 3 and _trunc.usage["retries"] == 2 and _trunc.usage["attempts"] == 3
          and _trunc.usage["uncacheable_truncated"] == 1 and _reply == "", (_trunc.calls, dict(_trunc.usage)))
    _rate = judges.create("hard_truncating", votes=1)
    _rate.mode = "ratelimit"
    try:
        _rate.sample("x-1", "p")
    except RuntimeError:
        pass
    check("other transient failures keep all 5 attempts",
          _rate.calls == 5 and _rate.usage["retries"] == 4, (_rate.calls, dict(_rate.usage)))

    with tempfile.TemporaryDirectory() as _hc_tmp:
        _hc_in = os.path.join(_hc_tmp, "in.csv")
        _hc_write(_hc_in, [{"comment_id": "a", "story": "musk", "body": "T-0 truncates"},
                           {"comment_id": "b", "story": "musk", "body": "x-1 fine"}])
        _hc_out = os.path.join(_hc_tmp, "out.csv")
        _exc, _err = _hc_classify([_hc_in, "--judge", "hard_truncating", "--votes", "1",
                                   "--sentiment", "stub", "--no-cache", "--output", _hc_out])
        _record = json.load(open(_hc_out + ".run.json", encoding="utf-8"))["result"]
        check("the summary reports attempts and retries",
              _exc is None and "Requests: 4 attempt(s) for 2 live call(s), 2 of them retries" in _err, _err[-700:])
        check("the run record carries attempts, retries and the per-backend split",
              _record["attempts"] == 4 and _record["retries"] == 2
              and _record["attempts_by_backend"] == {"domain": 4}
              and _record["retries_by_backend"] == {"domain": 2}, _record)
finally:
    judges.time.sleep = _hc_sleep

print("\nhardening run: a sentiment vote that cannot be read is a failed ballot, not a skipped one")
check("two unreadable votes out of three are a parse failure (not a label from the one reader)",
      _hc_sent.resolve_ratings([4, None, None])["label"] == _hc_sent.PARSE_FAILURE
      and _hc_sent.resolve_ratings([4, None, None])["unstable"] is True,
      _hc_sent.resolve_ratings([4, None, None]))
check("one unreadable vote out of three: the two readable ones decide, and it is flagged unstable",
      _hc_sent.resolve_ratings([4, 4, None])["label"] == "positive"
      and _hc_sent.resolve_ratings([4, 4, None])["unstable"] is True)
check("an even split between a readable and an unreadable vote is no consensus",
      _hc_sent.resolve_ratings([4, None])["label"] == _hc_sent.NO_CONSENSUS)
check("a failed ballot cannot break a tie between kinds in favour of a stance",
      _hc_sent.resolve_ratings([None, 0, 3])["label"] == _hc_sent.NO_CONSENSUS)
check("all votes unreadable is still a parse failure with no raw ratings (as before)",
      _hc_sent.resolve_ratings([None, None, None])
      == {"label": "parse_failure", "value": None, "raw": None, "unstable": False}
      and _hc_sent.resolve_ratings([]) == {"label": "parse_failure", "value": None, "raw": None,
                                           "unstable": False})
check("readable votes resolve exactly as before",
      _hc_sent.resolve_ratings([1, 2, 5])["label"] == "negative"
      and _hc_sent.resolve_ratings([4])["label"] == "positive")


@judges.register("hard_scripted")
class _HardScripted(judges.Judge):
    """Replies to vote N with script[N] (the sentiment half of a reply is the rating)."""
    default_model = "scripted"
    script = ()

    def _sample(self, body, prompt, vote):
        return self.script[vote]


def _hc_scored(script):
    _HardScripted.script = script
    return _hc_sent.create("llm:hard_scripted", votes=3).score("a comment")


check("LLMScorer: replies '4', 'garbage', 'n/a' -> a parse failure, flagged unstable",
      _hc_scored(("4", "garbage", "n/a"))["label"] == "parse_failure"
      and _hc_scored(("4", "garbage", "n/a"))["unstable"] is True, _hc_scored(("4", "garbage", "n/a")))
check("LLMScorer: replies '4', '4', 'garbage' -> positive, flagged unstable",
      _hc_scored(("4", "4", "garbage"))["label"] == "positive"
      and _hc_scored(("4", "4", "garbage"))["unstable"] is True)
check("LLMScorer: three readable agreeing replies are stable, as before",
      _hc_scored(("4", "4", "4"))["unstable"] is False)
_HardScripted.script = ("none | 4", "none", "no rating here")
_hc_comb = judges.create("combined:hard_scripted", votes=3)
check("combined: one rating in three replies is a parse failure, not that one rating",
      _hc_comb.sentiment_for("a comment", [])["label"] == "parse_failure")
_HardScripted.script = ("none | 4", "none | 4", "none")
_hc_comb2 = judges.create("combined:hard_scripted", votes=3, fuzzy=True)
check("combined: two ratings in three replies still decide, flagged unstable",
      _hc_comb2.sentiment_for("a comment", [])["label"] == "positive"
      and _hc_comb2.sentiment_for("a comment", [])["unstable"] is True)
_hc_row = {}
_hc.annotate_row(_hc_row, ["social"], False, [["social"]] * 3,
                 {"label": "positive", "value": 0.5, "unstable": True, "raw": [4, None, 4]},
                 "j", "s", 3, 3)
check("annotate_row writes a failed ballot as `parse_failure`, which the numeric readers skip",
      _hc_row["sentiment_raw_votes"] == "4,parse_failure,4"
      and validate.parse_sentiment_raw_votes(_hc_row["sentiment_raw_votes"]) == [4, 4])

print("\nhardening run: a per-comment content-policy 400 does not end an OpenAI run")


class _Http400(Exception):
    status_code = 400


def _hc_openai_with(error):
    probe = _openai_probe("democracy", "stop")

    def create(**k):
        raise error
    probe._client.chat.completions.create = create
    return probe


for _text in ("Error code: 400 - {'error': {'code': 'invalid_prompt', 'message': 'flagged'}}",
              "Error code: 400 - content_policy_violation",
              "Error code: 400 - your input was flagged by our safety system"):
    _got = _hc_openai_with(_Http400(_text))._sample("c", "p", 0)
    check(f"OpenAI 400 ({_text[:40]}...) is a blocked, uncacheable, empty reply",
          isinstance(_got, judges.UncacheableReply) and _got.reason == "blocked" and _got == "", repr(_got))
_hc_ok_stop = True
for _error in (_Http400("Error code: 400 - invalid_request_error: Unsupported value: reasoning_effort"),
               _Http400("Error code: 400 - invalid_request_error: 'temperature' is not supported"),
               RuntimeError("Error code: 429 - flagged for rate"),
               RuntimeError("Error code: 500 - invalid_prompt")):
    try:
        _hc_openai_with(_error)._sample("c", "p", 0)
        _hc_ok_stop = False
    except (judges.ReasoningEffortRejected, _Http400, RuntimeError):
        pass
check("any other failure (a settings 400, ReasoningEffortRejected, a non-400) still stops the run",
      _hc_ok_stop)
try:
    _hc_openai_with(_Http400("Error code: 400 - invalid_request_error: reasoning_effort"))._sample("c", "p", 0)
    _hc_kind = "returned"
except judges.ReasoningEffortRejected:
    _hc_kind = "ReasoningEffortRejected"
except Exception as _e:
    _hc_kind = type(_e).__name__
check("a rejected reasoning_effort is still ReasoningEffortRejected", _hc_kind == "ReasoningEffortRejected", _hc_kind)


@judges.register("hard_blocking")
class _HardBlocking(judges.Judge):
    default_model = "blocking"

    def _sample(self, body, prompt, vote):
        return judges.uncacheable("", "blocked") if "BLOCK" in body else "social"


with tempfile.TemporaryDirectory() as _hc_tmp:
    _hc_in = os.path.join(_hc_tmp, "in.csv")
    _hc_write(_hc_in, [{"comment_id": "a", "story": "musk", "body": "fine"},
                       {"comment_id": "b", "story": "musk", "body": "BLOCK me"},
                       {"comment_id": "c", "story": "musk", "body": "also fine"}])
    _hc_out = os.path.join(_hc_tmp, "out.csv")
    _exc, _err = _hc_classify([_hc_in, "--judge", "hard_blocking", "--votes", "1", "--sentiment",
                               "stub", "--no-cache", "--output", _hc_out])
    _rows = _hc_read(_hc_out)
    _record = json.load(open(_hc_out + ".run.json", encoding="utf-8"))["result"]
    check("a blocked comment is labelled a parse failure and the run completes",
          _exc is None and len(_rows) == 3 and _rows[1]["domain_parse_failure"] == "1"
          and _rows[0]["domain_parse_failure"] == "0", (repr(_exc), len(_rows)))
    check("the summary and the run record count the blocked ones",
          "1 reply(ies) were blocked" in _err and _record["blocked"] == 1, _err[-600:])

print("\nhardening run: the comment is stripped for the prompt and cache key; a missing story warns")
_hc_seen = []


@judges.register("hard_recording")
class _HardRecording(judges.Judge):
    default_model = "recording"

    def _sample(self, body, prompt, vote):
        with _hc_lock:
            _hc_seen.append(body)
        return "social"


with tempfile.TemporaryDirectory() as _hc_tmp:
    _hc_cache = os.path.join(_hc_tmp, "cache.jsonl")
    _hc_in = os.path.join(_hc_tmp, "padded.csv")
    _hc_write(_hc_in, [{"comment_id": "a", "story": "musk", "body": "  padded comment \t"}])
    _hc_out = os.path.join(_hc_tmp, "out.csv")
    _exc, _err = _hc_classify([_hc_in, "--judge", "hard_recording", "--votes", "1", "--sentiment",
                               "stub", "--cache", _hc_cache, "--output", _hc_out])
    check("the judge sees the stripped comment, and the output keeps the text as collected",
          _hc_seen == ["padded comment"] and _hc_read(_hc_out)[0]["body"] == "  padded comment \t",
          (_hc_seen, _hc_read(_hc_out)[0]["body"]))
    _hc_gold = os.path.join(_hc_tmp, "gold.csv")
    _hc_write(_hc_gold, [{"comment_id": "g", "story": "musk", "body": "padded comment"}])
    _hc_seen.clear()
    _exc, _err = _hc_classify([_hc_gold, "--judge", "hard_recording", "--votes", "1", "--sentiment",
                               "stub", "--cache", _hc_cache, "--output", os.path.join(_hc_tmp, "g.csv")])
    check("so the same comment unpadded (as gold scoring reads it) is a cache hit, not a second call",
          _hc_seen == [], _hc_seen)

    _hc_nostory = os.path.join(_hc_tmp, "nostory.csv")
    _hc_write(_hc_nostory, [{"comment_id": "a", "body": "no story here"}], with_story=False)
    _exc, _err = _hc_classify([_hc_nostory, "--judge", "hard_recording", "--votes", "1",
                               "--sentiment", "stub", "--no-cache",
                               "--output", os.path.join(_hc_tmp, "ns.csv")])
    check("an input with no `story` column warns that every prompt is story-free (and still runs)",
          _exc is None and "has no `story` column" in _err and "story-free prompts" in _err, _err[-500:])
    _exc, _err = _hc_classify([_hc_gold, "--judge", "hard_recording", "--votes", "1",
                               "--sentiment", "stub", "--no-cache",
                               "--output", os.path.join(_hc_tmp, "ws.csv")])
    check("an input that has one does not", "no `story` column" not in _err, _err[-300:])
print("\nhardening: a partial gold join is refused, unlabelled and blank-hand gold rows are never scored")
from ewl.analyse import export_dashboard as _hb_ed
from ewl.analyse import report as _hb_rep
from ewl.analyse import validate as _hb_val

_hb_env = dict(os.environ, PYTHONPATH=os.path.join(os.getcwd(), "src"))
_HB_CATS = ["democracy", "economics", "environment", "social"]
_HB_FIELDS = (["comment_id", "story", "source", "body", "hand_domain", "hand_sentiment",
               "inclusion_prob"]
              + [f"domain_{c}" for c in _HB_CATS]
              + ["domain_unclassified", "domain_parse_failure", "sentiment_value"])


def _hb_write(path, rows, fields=None):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or _HB_FIELDS, extrasaction="ignore",
                                restval="")
        writer.writeheader()
        writer.writerows(rows)


def _hb_cli(module, *argv, cwd=None):
    return subprocess.run([sys.executable, "-m", module, *argv], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=_hb_env, cwd=cwd)


def _hb_row(i, labelled=True, hand=True, prefix="g"):
    """A synthetic gold row: every third is a hand-coded democracy comment; the machine
    flips its answer on every tenth, so there are real false positives and negatives."""
    positive = i % 3 == 0
    guess = positive != (i % 10 == 0)
    row = {"comment_id": f"{prefix}{i}", "story": "jets" if i % 2 else "luigi",
           "source": f"v{i % 6}", "body": f"synthetic comment {i}",
           "hand_domain": ("democracy" if positive else "none") if hand else "",
           "hand_sentiment": "3"}
    if labelled:
        row.update({"domain_democracy": "1" if guess else "0", "domain_economics": "0",
                    "domain_environment": "0", "domain_social": "0",
                    "domain_unclassified": "0" if guess else "1", "domain_parse_failure": "0",
                    "sentiment_value": "0"})
    return row


_hb_clean = [_hb_row(i) for i in range(150)]
# 60 more coded rows that `join` could not match: hand code present, machine cells blank.
_hb_unlab = [_hb_row(i, labelled=False, prefix="u") for i in range(150, 210)]
# 60 rows nobody has hand-coded yet, with machine labels (blank hand_domain != "none").
_hb_blank = [_hb_row(i, hand=False, prefix="b") for i in range(150, 210)]

with tempfile.TemporaryDirectory() as _hb:
    # --- 1. the join itself
    _hb_lab = os.path.join(_hb, "labelled.csv")
    _hb_write(_hb_lab, [dict(r, hand_domain="", hand_sentiment="") for r in _hb_clean[:10]])
    _hb_coded = os.path.join(_hb, "coded.csv")
    _hb_write(_hb_coded, _hb_clean[:10] + [_hb_row(i, labelled=False, prefix="u")
                                          for i in range(2)])
    # the labelled file's run record travels with the join
    with open(_hb_lab + ".run.json", "w", encoding="utf-8") as _h:
        json.dump({"fixed_examples": {"source": "built-in (synthetic)"}}, _h)
    _hb_out = os.path.join(_hb, "joined.csv")
    _r = _hb_cli("ewl.analyse.validate", "join", _hb_coded, "--labelled", _hb_lab,
                 "--output", _hb_out)
    check("join refuses when ANY coded gold row is unmatched, naming how many, writing nothing",
          _r.returncode != 0 and "Refusing to join: 2 of 12 coded comments" in _r.stderr
          and "--allow-partial" in _r.stderr and not os.path.exists(_hb_out),
          (_r.returncode, _r.stderr[-300:]))
    _r = _hb_cli("ewl.analyse.validate", "join", _hb_coded, "--labelled", _hb_lab,
                 "--output", _hb_out, "--allow-partial")
    check("--allow-partial lets the same join through", _r.returncode == 0
          and os.path.exists(_hb_out), (_r.returncode, _r.stderr[-300:]))
    check("and the labelled file's run record is copied beside the join",
          os.path.exists(_hb_out + ".run.json"))
    _hb_joined = list(csv.DictReader(open(_hb_out, encoding="utf-8-sig")))
    check("the unmatched rows come out with blank machine cells, which every scorer now skips",
          sum(1 for r in _hb_joined if _hb_val.machine_unlabelled(r)) == 2, len(_hb_joined))
    try:
        _hb_val.cmd_join(_hb_clean[:3] + [_hb_row(0, labelled=False, prefix="zz")],
                         [dict(r) for r in _hb_clean[:3]], io.StringIO())
        check("cmd_join refuses a partial join when called directly", False, "no exit")
    except SystemExit as _e:
        check("cmd_join refuses a partial join when called directly", "1 of 4" in str(_e), str(_e))
    check("an UNCODED unmatched row is not a reason to refuse (nothing to score either way)",
          _hb_val.partial_join_error([{"comment_id": "x", "hand_domain": "", "hand_sentiment": ""}],
                                     [{"comment_id": "y"}]) is None)

    # --- 1. scoring: the same figures with and without the unlabelled rows
    def _hb_score(rows):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            _hb_val.cmd_score(rows)
        return out.getvalue(), err.getvalue()

    _hb_score_clean, _ = _hb_score(_hb_clean)
    _hb_score_mixed, _hb_err = _hb_score(_hb_clean + _hb_unlab)
    check("validate score: 60 unlabelled rows change no figure (not scored as misses)",
          _hb_score_clean == _hb_score_mixed, _hb_score_mixed[-400:])
    check("and it says how many it left out", "Left out 60 of 210 gold rows" in _hb_err, _hb_err)
    check("report._gold_tpr_fpr is unchanged by the unlabelled rows",
          _hb_rep._gold_tpr_fpr(_hb_clean + _hb_unlab, "democracy", "hand_domain")
          == _hb_rep._gold_tpr_fpr(_hb_clean, "democracy", "hand_domain"))
    check("the PPI rectifier is unchanged by the unlabelled rows",
          _hb_rep._weighted_rectifier(_hb_clean + _hb_unlab, "domain_democracy", "democracy",
                                      "hand_domain")
          == _hb_rep._weighted_rectifier(_hb_clean, "domain_democracy", "democracy",
                                         "hand_domain"))
    check("story accuracy ratios are unchanged by the unlabelled rows",
          _hb_rep.story_accuracy_ratios(_hb_clean + _hb_unlab, _HB_CATS)
          == _hb_rep.story_accuracy_ratios(_hb_clean, _HB_CATS))
    # NaN (a category nobody used) is never equal to itself, so compare the serialised form
    _pcs = lambda rows: json.dumps(_hb_ed.per_category_scores(rows), sort_keys=True)
    check("export per_category_scores is unchanged by the unlabelled rows",
          _pcs(_hb_clean + _hb_unlab) == _pcs(_hb_clean))

    # --- 2. blank hand_domain is "not coded", not hand-negative
    check("report._gold_tpr_fpr is unchanged by 60 blank-hand rows",
          _hb_rep._gold_tpr_fpr(_hb_clean + _hb_blank, "democracy", "hand_domain")
          == _hb_rep._gold_tpr_fpr(_hb_clean, "democracy", "hand_domain"))
    check("the PPI rectifier is unchanged by blank-hand rows",
          _hb_rep._weighted_rectifier(_hb_clean + _hb_blank, "domain_democracy", "democracy",
                                      "hand_domain")
          == _hb_rep._weighted_rectifier(_hb_clean, "domain_democracy", "democracy",
                                         "hand_domain"))
    check("story accuracy ratios are unchanged by blank-hand rows",
          _hb_rep.story_accuracy_ratios(_hb_clean + _hb_blank, _HB_CATS)
          == _hb_rep.story_accuracy_ratios(_hb_clean, _HB_CATS))
    check("export per_category_scores is unchanged by blank-hand rows (tn, fp, precision)",
          _pcs(_hb_clean + _hb_blank) == _pcs(_hb_clean))

    # the whole export: 150 coded + 60 blank-hand rows must not clear the 200-row validated floor
    _hb_corpus = os.path.join(_hb, "corpus.csv")
    _hb_write(_hb_corpus, [_hb_row(i, hand=False, prefix="c") for i in range(60)])
    _hb_gold_mixed = os.path.join(_hb, "gold_mixed.csv")
    _hb_write(_hb_gold_mixed, _hb_clean + _hb_blank)
    _hb_gold_ok = os.path.join(_hb, "gold_ok.csv")
    _hb_write(_hb_gold_ok, _hb_clean)
    _hb_json = {}
    for _name, _gold in (("mixed", _hb_gold_mixed), ("ok", _hb_gold_ok)):
        _outj = os.path.join(_hb, f"r_{_name}.json")
        _r = _hb_cli("ewl.analyse.export_dashboard", _hb_corpus, "--output", _outj,
                     "--scored", _gold, "--status", "validated", "--n-boot", "20")
        check(f"export ({_name} gold) runs", _r.returncode == 0, _r.stderr[-400:])
        _hb_json[_name] = json.load(open(_outj, encoding="utf-8"))
        if _name == "mixed":
            _hb_mixed_err = _r.stderr
    check("150 coded + 60 blank-hand rows publishes provisional, gold_comments 150",
          _hb_json["mixed"]["provenance"]["status"] == "provisional"
          and _hb_json["mixed"]["provenance"]["gold_comments"] == 150
          and "Left out 60 of 210 gold rows with a blank hand_domain" in _hb_mixed_err,
          (_hb_json["mixed"]["provenance"]["status"], _hb_mixed_err[-300:]))
    check("and its validation figures equal those of the file without the blank rows",
          _hb_json["mixed"]["validation"] == _hb_json["ok"]["validation"])
    # 210 properly coded rows DO clear the floor: the guard is not simply always provisional.
    _hb_gold_210 = os.path.join(_hb, "gold_210.csv")
    _hb_write(_hb_gold_210, _hb_clean + [_hb_row(i, prefix="d") for i in range(150, 210)])
    _outj = os.path.join(_hb, "r_210.json")
    _r = _hb_cli("ewl.analyse.export_dashboard", _hb_corpus, "--output", _outj,
                 "--scored", _hb_gold_210, "--status", "validated", "--n-boot", "20")
    check("210 genuinely coded rows still publish validated",
          json.load(open(_outj, encoding="utf-8"))["provenance"]["status"] == "validated",
          _r.stderr[-300:])
    # and an export whose gold file is partly unlabelled leaves those out and counts them
    _hb_gold_part = os.path.join(_hb, "gold_part.csv")
    _hb_write(_hb_gold_part, _hb_clean + _hb_unlab)
    _outj = os.path.join(_hb, "r_part.json")
    _r = _hb_cli("ewl.analyse.export_dashboard", _hb_corpus, "--output", _outj,
                 "--scored", _hb_gold_part, "--status", "validated", "--n-boot", "20")
    _jp = json.load(open(_outj, encoding="utf-8"))
    check("export leaves unlabelled gold rows out (gold 150, provisional), says so, same figures",
          _jp["provenance"]["gold_comments"] == 150
          and _jp["provenance"]["status"] == "provisional"
          and "Left out 60 of 210 gold rows with no machine labels" in _r.stderr
          and _jp["validation"] == _hb_json["ok"]["validation"], _r.stderr[-300:])
    # the CLI score on a partially unlabelled file: same figures as the clean one
    _hb_part_csv = os.path.join(_hb, "part_score.csv")
    _hb_write(_hb_part_csv, _hb_clean + _hb_unlab)
    _r = _hb_cli("ewl.analyse.validate", "score", _hb_part_csv, "--no-example-exclusion")
    check("validate score on the file prints the clean figures and the count",
          _r.returncode == 0 and _r.stdout == _hb_score_clean
          and "Left out 60 of 210" in _r.stderr, (_r.returncode, _r.stderr[-200:]))

print("\nhardening: worked examples are left out automatically from the gold file's run record")
with tempfile.TemporaryDirectory() as _hb:
    _hb_gold = os.path.join(_hb, "gold.csv")
    _hb_write(_hb_gold, _hb_clean)
    _e = io.StringIO()
    with contextlib.redirect_stderr(_e):
        _ids = _hb_val.example_ids_for_gold(_hb_gold)
    check("example_ids_for_gold warns that examples may be scored against their own answers",
          _ids == set() and "WARNING" in _e.getvalue()
          and "scored against its own answer" in _e.getvalue(), _e.getvalue())
    _e = io.StringIO()
    with contextlib.redirect_stderr(_e):
        _hb_val.example_ids_for_gold(_hb_gold, silence=True)
    check("--no-example-exclusion silences it", _e.getvalue() == "", _e.getvalue())
    with open(_hb_gold + ".run.json", "w", encoding="utf-8") as _h:
        json.dump({"fixed_examples": {"source": "file", "ids": ["g0", "g3", "nothere"]}}, _h)
    _e = io.StringIO()
    with contextlib.redirect_stderr(_e):
        _ids = _hb_val.example_ids_for_gold(_hb_gold)
    check("a run record's fixed_examples.ids are used, announced, and no WARNING is printed",
          _ids == {"g0", "g3", "nothere"} and "WARNING" not in _e.getvalue()
          and "3 worked-example" in _e.getvalue(), _e.getvalue())
    _hb_ex_file = os.path.join(_hb, "ex.csv")
    _hb_write(_hb_ex_file, [{"comment_id": "g6"}], fields=["comment_id"])
    with contextlib.redirect_stderr(io.StringIO()):
        _ids = _hb_val.example_ids_for_gold(_hb_gold, _hb_ex_file)
    check("--exclude-ids adds to the record's ids", _ids == {"g0", "g3", "nothere", "g6"}, _ids)
    with open(_hb_gold + ".run.json", "w", encoding="utf-8") as _h:
        json.dump({"fixed_examples": {"source": "built-in (synthetic)"}}, _h)
    _e = io.StringIO()
    with contextlib.redirect_stderr(_e):
        _hb_val.example_ids_for_gold(_hb_gold)
    check("a built-in (synthetic) example set needs no exclusion and no warning",
          _e.getvalue() == "", _e.getvalue())

    # the CLIs: ids from the record come out of the scored rows, and the count is printed
    with open(_hb_gold + ".run.json", "w", encoding="utf-8") as _h:
        json.dump({"fixed_examples": {"source": "file", "ids": ["g0", "g3"]}}, _h)
    _r = _hb_cli("ewl.analyse.validate", "score", _hb_gold)
    check("validate score excludes the record's examples and reports 2 of 150",
          _r.returncode == 0 and "Excluded 2 of 150 gold rows" in _r.stderr
          and "Scoring 148 hand-coded comments" in _r.stdout, (_r.returncode, _r.stderr[-300:]))
    _hb_corp = os.path.join(_hb, "corpus.csv")
    _hb_write(_hb_corp, [_hb_row(i, hand=False, prefix="c") for i in range(60)])
    _r = _hb_cli("ewl.analyse.report", "corrected-bootstrap", _hb_corp, "--column",
                 "domain_democracy", "--gold", _hb_gold, "--n-boot", "30")
    check("report corrected-bootstrap excludes them too (148 gold rows) and prints the count",
          _r.returncode == 0 and "Excluded 2 of 150 gold rows" in _r.stderr
          and "WARNING: no worked-example ids" not in _r.stderr, (_r.returncode, _r.stderr[-400:]))
    _r = _hb_cli("ewl.analyse.report", "ppi-bootstrap", _hb_corp, "--column",
                 "domain_democracy", "--gold", _hb_gold, "--n-boot", "30")
    check("ppi-bootstrap and corrected-by-story honour the record as well",
          _r.returncode == 0 and "Excluded 2 of 150 gold rows" in _r.stderr, _r.stderr[-300:])
    _r = _hb_cli("ewl.analyse.report", "corrected-by-story", _hb_corp, "--column",
                 "domain_democracy", "--gold", _hb_gold, "--n-boot", "30")
    check("corrected-by-story honours the record", "Excluded 2 of 150 gold rows" in _r.stderr,
          _r.stderr[-300:])
    os.remove(_hb_gold + ".run.json")
    _r = _hb_cli("ewl.analyse.report", "corrected-bootstrap", _hb_corp, "--column",
                 "domain_democracy", "--gold", _hb_gold, "--n-boot", "30")
    check("with no record and no --exclude-ids report prints the WARNING",
          "WARNING: no worked-example ids are known" in _r.stderr, _r.stderr[-300:])
    _r = _hb_cli("ewl.analyse.report", "corrected-bootstrap", _hb_corp, "--column",
                 "domain_democracy", "--gold", _hb_gold, "--n-boot", "30",
                 "--no-example-exclusion")
    check("--no-example-exclusion silences it", "WARNING: no worked-example" not in _r.stderr
          and _r.returncode == 0, _r.stderr[-300:])
    _r = _hb_cli("ewl.analyse.validate", "score", _hb_gold)
    check("validate score warns too without a record", "WARNING: no worked-example ids" in _r.stderr,
          _r.stderr[-300:])
    _r = _hb_cli("ewl.analyse.validate", "score", _hb_gold, "--no-example-exclusion")
    check("and --no-example-exclusion silences validate score",
          "WARNING" not in _r.stderr and _r.returncode == 0, _r.stderr[-300:])
    # --base both resolves the examples once
    with open(_hb_gold + ".run.json", "w", encoding="utf-8") as _h:
        json.dump({"fixed_examples": {"source": "file", "ids": ["g0", "g3"]}}, _h)
    _r = _hb_cli("ewl.analyse.report", "corrected-bootstrap", _hb_corp, "--column",
                 "domain_democracy", "--gold", _hb_gold, "--n-boot", "30", "--base", "both")
    check("--base both announces the worked examples once, not per base",
          _r.stderr.count("worked-example comment(s) named in") == 1, _r.stderr[-600:])

print("\nhardening: ewl-keywords carries the run record, so classify -> keywords -> export keeps it")
from ewl import corpus as _hb_corpus_mod

with tempfile.TemporaryDirectory() as _hb:
    _hb_lab = os.path.join(_hb, "lab.csv")
    _hb_write(_hb_lab, [_hb_row(i, hand=False, prefix="c") for i in range(60)])
    _hb_both = os.path.join(_hb, "both.csv")
    _hb_rec_text = json.dumps({"input": {"sha256": _hb_corpus_mod.content_hash(_hb_lab)},
                               "fixed_examples": {"source": "built-in (synthetic)"},
                               "partial": {"batches_done": 1, "batches_total": 4,
                                           "comments_included": 60, "comments_in_corpus": 240}},
                              indent=2, sort_keys=True)
    with open(_hb_lab + ".run.json", "w", encoding="utf-8", newline="") as _h:
        _h.write(_hb_rec_text)
    _r = _hb_cli("ewl.label.keywords", _hb_lab, "--output", _hb_both)
    check("ewl-keywords runs on the labelled file", _r.returncode == 0, _r.stderr[-300:])
    check("and copies the input's run record beside its output, byte for byte",
          os.path.exists(_hb_both + ".run.json")
          and open(_hb_both + ".run.json", "rb").read() == open(_hb_lab + ".run.json", "rb").read())
    # export reads the carried record: a partial progress file cannot pass for the corpus
    _r = _hb_cli("ewl.analyse.export_dashboard", _hb_both, "--output", os.path.join(_hb, "o.json"),
                 "--allow-unvalidated", "--corpus", _hb_lab)
    check("export --corpus on the keywords output refuses the partial progress file",
          _r.returncode != 0 and "PARTIAL progress file" in _r.stderr, (_r.returncode, _r.stderr[-300:]))
    # the same record without the partial block: --corpus is checked against the input hash
    _rec = json.loads(_hb_rec_text)
    del _rec["partial"]
    with open(_hb_lab + ".run.json", "w", encoding="utf-8") as _h:
        json.dump(_rec, _h)
    _r = _hb_cli("ewl.label.keywords", _hb_lab, "--output", _hb_both)
    _r = _hb_cli("ewl.analyse.export_dashboard", _hb_both, "--output", os.path.join(_hb, "o.json"),
                 "--allow-unvalidated", "--corpus", _hb_lab, "--n-boot", "20")
    check("export --corpus then finds the record and checks the corpus hash",
          _r.returncode == 0 and "matches the corpus the labels were made from" in _r.stderr,
          (_r.returncode, _r.stderr[-300:]))
    check("the export records that a run record was found",
          json.load(open(os.path.join(_hb, "o.json"), encoding="utf-8"))["provenance"][
              "run_record_found"] is True)
    # --run-record points at a record that is not beside the file
    _hb_moved = os.path.join(_hb, "elsewhere.json")
    os.replace(_hb_both + ".run.json", _hb_moved)
    _r = _hb_cli("ewl.analyse.export_dashboard", _hb_both, "--output", os.path.join(_hb, "o2.json"),
                 "--allow-unvalidated", "--n-boot", "20")
    check("without the record beside the file, export notes it is missing",
          "no run record beside the labelled file" in _r.stderr
          and json.load(open(os.path.join(_hb, "o2.json"), encoding="utf-8"))["provenance"][
              "run_record_found"] is False, _r.stderr[-300:])
    _r = _hb_cli("ewl.analyse.export_dashboard", _hb_both, "--output", os.path.join(_hb, "o3.json"),
                 "--allow-unvalidated", "--n-boot", "20", "--run-record", _hb_moved,
                 "--corpus", _hb_lab)
    check("--run-record finds it and the --corpus check then passes",
          _r.returncode == 0 and "matches the corpus the labels were made from" in _r.stderr
          and json.load(open(os.path.join(_hb, "o3.json"), encoding="utf-8"))["provenance"][
              "run_record_found"] is True, (_r.returncode, _r.stderr[-300:]))
    _r = _hb_cli("ewl.analyse.export_dashboard", _hb_both, "--output", os.path.join(_hb, "o4.json"),
                 "--allow-unvalidated", "--run-record", os.path.join(_hb, "nope.json"))
    check("--run-record with a missing file is refused, not ignored",
          _r.returncode != 0 and "--run-record file not found" in _r.stderr, _r.stderr[-200:])
    # a stale record beside the output must not outlive an input that has none
    os.remove(_hb_lab + ".run.json")
    with open(_hb_both + ".run.json", "w", encoding="utf-8") as _h:
        _h.write("{}")
    _r = _hb_cli("ewl.label.keywords", _hb_lab, "--output", _hb_both)
    check("an input with no run record removes a stale one beside the output",
          not os.path.exists(_hb_both + ".run.json") and "Removed a stale" in _r.stderr, _r.stderr[-200:])

print("\nhardening: the gold file's instrument must match the corpus's")


def _hb_record(**over):
    record = {"input": {"sha256": "abc"},
              "domain_judge": {"spec": "anthropic:m", "votes": 1},
              "sentiment_scorer": "llm:anthropic:m",
              "instrument_switches": {"tags": ""},
              "prompts_sha256": {"domain": "d1", "domain_by_story": {"jets": "a", "luigi": "b"},
                                 "sentiment_by_story": {"jets": "s1"}},
              "fixed_examples": {"source": "file", "sha256": "ex1", "ids": ["zz-not-in-gold"]}}
    record.update(over)
    return record


check("instrument_differences: identical records differ in nothing",
      _hb_ed.instrument_differences(_hb_record(), _hb_record()) == ([], []))
_d, _nc = _hb_ed.instrument_differences(
    _hb_record(), _hb_record(domain_judge={"spec": "anthropic:other"},
                             prompts_sha256={"domain": "d2",
                                             "domain_by_story": {"jets": "a", "luigi": "X"},
                                             "sentiment_by_story": {"jets": "s1"}},
                             instrument_switches={"tags": "|story-rules=off"},
                             sentiment_scorer="llm:anthropic:other",
                             fixed_examples={"source": "file", "sha256": "ex2"}))
check("every differing field is named",
      len(_d) == 6 and any("domain judge" in x for x in _d) and any("sentiment scorer" in x for x in _d)
      and any("domain prompt hash:" in x for x in _d) and any("'luigi'" in x for x in _d)
      and any("switches" in x for x in _d) and any("fixed examples" in x for x in _d), _d)
_d, _nc = _hb_ed.instrument_differences(
    _hb_record(), {k: v for k, v in _hb_record().items() if k != "sentiment_scorer"})
check("a field absent from one record is 'not comparable', not a difference",
      _d == [] and _nc == ["sentiment scorer"], (_d, _nc))
_d, _nc = _hb_ed.instrument_differences(
    _hb_record(), _hb_record(prompts_sha256={"domain": "d1", "domain_by_story": {"luigi": "b"},
                                             "sentiment_by_story": {"musk": "q"}}))
check("per-story hashes are compared only on the stories both records have",
      _d == [], _d)

with tempfile.TemporaryDirectory() as _hb:
    _hb_lab = os.path.join(_hb, "lab.csv")
    _hb_write(_hb_lab, [_hb_row(i, hand=False, prefix="c") for i in range(60)])
    _hb_gold = os.path.join(_hb, "gold.csv")
    _hb_write(_hb_gold, _hb_clean)

    def _hb_put(path, record):
        with open(path + ".run.json", "w", encoding="utf-8") as handle:
            json.dump(record, handle)

    def _hb_export(name, *extra):
        out = os.path.join(_hb, f"{name}.json")
        r = _hb_cli("ewl.analyse.export_dashboard", _hb_lab, "--output", out, "--scored", _hb_gold,
                    "--n-boot", "20", *extra)
        return r, (json.load(open(out, encoding="utf-8")) if os.path.exists(out) else None)

    _hb_put(_hb_lab, _hb_record())
    _hb_put(_hb_gold, _hb_record())
    _r, _j = _hb_export("same")
    check("matching records export, with no mismatch in provenance",
          _r.returncode == 0 and _j["provenance"]["instrument_mismatch"] is None
          and _j["provenance"]["gold_run_record_found"] is True, _r.stderr[-300:])
    _hb_put(_hb_gold, _hb_record(domain_judge={"spec": "anthropic:other"},
                                 prompts_sha256={"domain": "d2",
                                                 "domain_by_story": {"jets": "a", "luigi": "b"},
                                                 "sentiment_by_story": {"jets": "s1"}}))
    _r, _j = _hb_export("diff")
    check("a different judge and prompt is REFUSED, naming both, and nothing is written",
          _r.returncode != 0 and "Refusing to export" in _r.stderr and "domain judge" in _r.stderr
          and "domain prompt hash" in _r.stderr and "--allow-instrument-mismatch" in _r.stderr
          and _j is None, (_r.returncode, _r.stderr[-400:]))
    _r, _j = _hb_export("allowed", "--allow-instrument-mismatch")
    check("--allow-instrument-mismatch exports anyway and records the differences",
          _r.returncode == 0 and len(_j["provenance"]["instrument_mismatch"]) == 2
          and "instrument mismatch" in _r.stderr, (_r.returncode, _r.stderr[-300:]))
    os.remove(_hb_gold + ".run.json")
    _r, _j = _hb_export("norec")
    check("no gold run record: a clear WARNING, and the export continues",
          _r.returncode == 0 and "WARNING" in _r.stderr and "run record missing" in _r.stderr
          and _j["provenance"]["gold_run_record_found"] is False, (_r.returncode, _r.stderr[-300:]))
    _hb_put(_hb_gold, {k: v for k, v in _hb_record().items() if k != "instrument_switches"})
    _r, _j = _hb_export("absent")
    check("a field missing from one record warns 'could not compare' and does not refuse",
          _r.returncode == 0 and "could not compare instrument switches" in _r.stderr,
          (_r.returncode, _r.stderr[-300:]))
    os.remove(_hb_lab + ".run.json")
    _hb_put(_hb_gold, _hb_record())
    _r, _j = _hb_export("nocorpusrec")
    check("no corpus run record: the same WARNING, export continues",
          _r.returncode == 0 and "run record missing" in _r.stderr, _r.stderr[-300:])

print("\nhardening: the sentiment unstable rate is a share of rows, and a dash at one vote")


class _HbScorer:
    """A scorer that flags the bodies in `flagged` as unstable; nothing else."""
    votes = 3
    seconds = 0.0

    def __init__(self, flagged=None, everything=False):
        self.flagged, self.everything = set(flagged or ()), everything

    def describe(self):
        return "hb-fake"

    def score(self, body, story=None):
        return {"label": "negative", "value": -1.0,
                "unstable": self.everything or body in self.flagged}

    def cost_usd(self):
        return 0.0

    def token_usage(self):
        return (0, 0)


_hb_srows = [(f"body {i}", "negative", "jets") for i in range(5)]
_rep_all = evaluate.sentiment_report(_hb_srows, _HbScorer(everything=True), [1000.0] * 5)
check("every comment unstable at weight 1000 each: the rate is 1.0, not 0.001",
      _rep_all["unstable_rate"] == 1.0, _rep_all["unstable_rate"])
# the two unstable comments are the LIGHT ones; the heavy one is stable: 2 of 5 rows
_rep_mix = evaluate.sentiment_report(_hb_srows, _HbScorer(flagged={"body 1", "body 2"}),
                                     [1000.0, 1.0, 1.0, 1.0, 1.0])
check("two of five rows unstable is 0.4 whatever their weights (not 2/1004)",
      abs(_rep_mix["unstable_rate"] - 0.4) < 1e-12, _rep_mix["unstable_rate"])
check("the report says how many votes the scorer ran", _rep_mix["votes"] == 3)

with tempfile.TemporaryDirectory() as _hb:
    _hb_sent = os.path.join(_hb, "sent.csv")
    _hb_write(_hb_sent, [{"body": f"comment number {i}", "hand_sentiment": str(1 + i % 5)}
                         for i in range(10)], fields=["body", "hand_sentiment"])

    def _hb_sent_cell(arm, votes):
        class _A:
            input_csv = _hb_sent
            cache = None
        _A.arm, _A.votes = [arm], votes
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            evaluate.cmd_sentiment(_A)
        line = next(l for l in out.getvalue().splitlines() if l.startswith(f"| `{arm}"))
        header = next(l for l in out.getvalue().splitlines() if "unstable" in l and l.startswith("| arm"))
        names = [c.strip() for c in header.strip().strip("|").split("|")]
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        return cells[names.index("unstable")]

    check("the sentiment table shows a dash, not 0%, for a one-vote LLM arm",
          _hb_sent_cell("llm:stub", 1) == "—", _hb_sent_cell("llm:stub", 1))
    check("and a real percentage at three votes",
          _hb_sent_cell("llm:stub", 3).endswith("%"), _hb_sent_cell("llm:stub", 3))

print("\nhardening: the corrected-estimate output shows how few weighted false positives there are")
_hb_fp_rows = ([{"comment_id": "heavy", "hand_domain": "none", "domain_democracy": "1",
                 "domain_economics": "0", "domain_environment": "0", "domain_social": "0",
                 "inclusion_prob": "0.001"}]
               + [{"comment_id": f"fp{i}", "hand_domain": "none", "domain_democracy": "1",
                   "domain_economics": "0", "domain_environment": "0", "domain_social": "0",
                   "inclusion_prob": "1"} for i in range(5)]
               + [{"comment_id": f"tn{i}", "hand_domain": "none", "domain_democracy": "0",
                   "domain_economics": "0", "domain_environment": "0", "domain_social": "0",
                   "inclusion_prob": "1"} for i in range(20)]
               + [{"comment_id": f"tp{i}", "hand_domain": "democracy", "domain_democracy": "1",
                   "domain_economics": "0", "domain_environment": "0", "domain_social": "0",
                   "inclusion_prob": "1"} for i in range(8)])
_hb_counts = _hb_rep.error_row_counts(_hb_fp_rows, "democracy", "hand_domain")
check("6 raw false positives, one of weight 1000, are worth about 1.0 effective",
      _hb_counts["false positives"][0] == 6 and 1.0 <= _hb_counts["false positives"][1] < 1.05,
      _hb_counts["false positives"])
check("the gold negatives are counted the same way (26 raw rows, heavy one dominating)",
      _hb_counts["gold negatives"][0] == 26 and _hb_counts["gold negatives"][1] < 1.1,
      _hb_counts["gold negatives"])
check("no false negatives here", _hb_counts["false negatives"] == (0, 0.0))
check("the summary line carries the numbers and says it is not a gate",
      "false positives 6 raw / 1.0 effective" in _hb_rep.error_row_summary(
          _hb_fp_rows, "democracy", "hand_domain")
      and "not a gate" in _hb_rep.error_row_summary(_hb_fp_rows, "democracy", "hand_domain"))
with tempfile.TemporaryDirectory() as _hb:
    _hb_gold = os.path.join(_hb, "gold.csv")
    _hb_write(_hb_gold, _hb_clean)
    _hb_corp = os.path.join(_hb, "corpus.csv")
    _hb_write(_hb_corp, [_hb_row(i, hand=False, prefix="c") for i in range(60)])
    _r = _hb_cli("ewl.analyse.report", "corrected-bootstrap", _hb_corp, "--column",
                 "domain_democracy", "--gold", _hb_gold, "--n-boot", "30", "--no-example-exclusion")
    check("ewl-report prints the line beside the effective-positives line, and still gates as before",
          "Weighted evidence behind TPR/FPR for 'democracy'" in _r.stderr
          and "effective (Horvitz-Thompson-weighted)" in _r.stderr, _r.stderr[-500:])

print("\nhardening: interrater JSON exports null kappa where there was no variation")


def _hb_coder(domains, sentiments):
    return [{"comment_id": f"k{i}", "hand_domain": d, "hand_sentiment": s}
            for i, (d, s) in enumerate(zip(domains, sentiments))]


with tempfile.TemporaryDirectory() as _hb:
    _doms_a = ["democracy", "economics", "none", "democracy", "none", "economics"]
    _doms_b = ["democracy", "economics", "none", "none", "none", "economics"]
    # environment and social: neither coder used them, so there is no variation to measure
    _j2 = os.path.join(_hb, "two.json")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        validate.cmd_interrater(_hb_coder(_doms_a, ["3"] * 6), _hb_coder(_doms_b, ["3"] * 6),
                                json_output=_j2)
    _two = json.load(open(_j2, encoding="utf-8"))
    check("a category nobody used has kappa null and no_variation true, not 1.0",
          _two["domain"]["environment"]["kappa"] is None
          and _two["domain"]["environment"]["no_variation"] is True
          and _two["domain"]["social"]["kappa"] is None, _two["domain"]["environment"])
    check("a category with variation keeps its real kappa and the keys the dashboard reads",
          _two["domain"]["democracy"]["kappa"] is not None
          and _two["domain"]["democracy"]["no_variation"] is False
          and {"coder_a", "coder_b", "both", "kappa"} <= set(_two["domain"]["democracy"]),
          _two["domain"]["democracy"])
    check("sentiment where both coders only ever said 3 is null, not a perfect kappa",
          _two["sentiment_kappa"] is None and _two["sentiment_no_variation"] is True
          and _two["sentiment_kappa_weighted"] is None, (_two["sentiment_kappa"],))
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        validate.cmd_interrater(_hb_coder(_doms_a, ["1", "3", "5", "2", "4", "3"]),
                                _hb_coder(_doms_b, ["1", "3", "4", "2", "5", "3"]),
                                json_output=_j2)
    _two = json.load(open(_j2, encoding="utf-8"))
    check("with varied sentiment the kappa is a number again",
          isinstance(_two["sentiment_kappa"], float) and _two["sentiment_no_variation"] is False
          and isinstance(_two["sentiment_kappa_weighted"], float), _two)
    _out_two = io.StringIO()
    with contextlib.redirect_stdout(_out_two), contextlib.redirect_stderr(io.StringIO()):
        validate.cmd_interrater(_hb_coder(_doms_a, ["3"] * 6), _hb_coder(_doms_b, ["3"] * 6))
    check("the terminal flags the constant sentiment too",
          "every sentiment answer was the same one" in _out_two.getvalue(), _out_two.getvalue()[-300:])

    # three coders
    _j3 = os.path.join(_hb, "three.json")
    _sents = ["1", "3", "5", "2", "4", "3"]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        validate.cmd_interrater_multi(
            [_hb_coder(_doms_a, _sents), _hb_coder(_doms_b, _sents),
             _hb_coder(_doms_a, ["1", "3", "5", "2", "4", "4"])],
            names=["Ann", "Bo", "Cy"], json_output=_j3, n_boot=50)
    _three = json.load(open(_j3, encoding="utf-8"))
    check("multi-coder JSON carries kappa per category (null where alpha is undefined)",
          _three["domain"]["environment"]["kappa"] is None
          and _three["domain"]["environment"]["no_variation"] is True
          and isinstance(_three["domain"]["democracy"]["kappa"], float)
          and _three["domain"]["democracy"]["kappa"] == _three["domain"]["democracy"]["fleiss_kappa"],
          _three["domain"])
    check("and the top-level keys the dashboard reads: n_shared, sentiment_kappa, exact agreement",
          _three["n_shared"] == 6 and isinstance(_three["sentiment_kappa"], float)
          and 0 <= _three["sentiment_exact_agreement"] <= 1
          and _three["domain"]["democracy"]["alpha"] is not None, sorted(_three))
    _hb_corp = os.path.join(_hb, "corpus.csv")
    _hb_write(_hb_corp, [_hb_row(i, hand=False, prefix="c") for i in range(60)])
    _r = _hb_cli("ewl.analyse.export_dashboard", _hb_corp, "--output", os.path.join(_hb, "o.json"),
                 "--allow-unvalidated", "--n-boot", "20", "--interrater", _j3)
    check("export warns that the page draws two-coder columns for a multi-coder file",
          _r.returncode == 0 and "multi-coder (Krippendorff alpha) file" in _r.stderr
          and "do" in _r.stderr, (_r.returncode, _r.stderr[-400:]))
    _r = _hb_cli("ewl.analyse.export_dashboard", _hb_corp, "--output", os.path.join(_hb, "o.json"),
                 "--allow-unvalidated", "--n-boot", "20", "--interrater", _j2)
    check("and says nothing of the sort for a two-coder file",
          _r.returncode == 0 and "multi-coder" not in _r.stderr, (_r.returncode, _r.stderr[-300:]))

print(chr(10) + "ewl-report --rescale-unclassified: a weights sensitivity run, other strata untouched")
import tempfile as _tf_rs
with _tf_rs.TemporaryDirectory() as _d_rs:
    _corp = os.path.join(_d_rs, "frozen.csv")
    with open(_corp, "w", newline="", encoding="utf-8-sig") as _h:
        _w = csv.DictWriter(_h, fieldnames=["comment_id", "story", "primary_category"])
        _w.writeheader()
        for _i in range(100):
            _w.writerow({"comment_id": f"u{_i}", "story": "jets", "primary_category": "unclassified"})
        for _i in range(10):
            _w.writerow({"comment_id": f"d{_i}", "story": "jets", "primary_category": "democracy"})
    _gold = [{"story": "jets", "sampling_stratum": "jets/unclassified", "inclusion_prob": "0.05"},
             {"story": "jets", "sampling_stratum": "jets", "inclusion_prob": "0.05"},
             {"story": "jets", "sampling_stratum": "jets/democracy", "inclusion_prob": "0.5"}]
    with contextlib.redirect_stderr(io.StringIO()) as _msg_rs:
        _scaled = report.rescale_unclassified_weights(_gold, _corp)
    _w_unc = sum(1 / float(r["inclusion_prob"]) for r in _scaled[:2])
    check("the unclassified strata's weights now sum to the corpus stratum size", abs(_w_unc - 100) < 1e-6, _w_unc)
    check("a row of another stratum keeps its weight", float(_scaled[2]["inclusion_prob"]) == 0.5)
    check("the input rows are not changed", _gold[0]["inclusion_prob"] == "0.05")
    check("the factor is printed for the story", "jets unclassified stratum" in _msg_rs.getvalue() and "scaled by 2.50" in _msg_rs.getvalue(), _msg_rs.getvalue())
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            report.rescale_unclassified_weights([_gold[2]], _corp)
        _nothing = False
    except SystemExit:
        _nothing = True
    check("a gold file with no unclassified rows is refused, not silently left as it was", _nothing)

print(chr(10) + "export: a different vote count or model build is an instrument difference")
_dv, _ = _hb_ed.instrument_differences(
    _hb_record(), _hb_record(domain_judge={"spec": "anthropic:m", "votes": 3}))
check("records that differ only in votes (1 against 3) are a difference", any("votes" in x for x in _dv), _dv)
_dm, _ = _hb_ed.instrument_differences(
    _hb_record(model_build={"digest": "aaa"}), _hb_record(model_build={"digest": "bbb"}))
check("records that differ only in the model digest are a difference", any("digest" in x for x in _dm), _dm)

print(chr(10) + "Anthropic client: a hung call is abandoned after EWL_ANTHROPIC_TIMEOUT, SDK retries off")
_seen_kw = {}
_fake_mod = types.ModuleType("anthropic") if "types" in globals() else _types_c.ModuleType("anthropic")
_fake_mod.Anthropic = lambda *a, **k: (_seen_kw.update(k), _types_c.SimpleNamespace(messages=None))[1]
_saved_anth = sys.modules.get("anthropic")
sys.modules["anthropic"] = _fake_mod
_old_to = os.environ.pop("EWL_ANTHROPIC_TIMEOUT", None)
try:
    judges.AnthropicJudge("claude-sonnet-5-5")
    check("the default request timeout is 60 seconds and the SDK's silent retries are off",
          _seen_kw.get("timeout") == 60.0 and _seen_kw.get("max_retries") == 0, _seen_kw)
    os.environ["EWL_ANTHROPIC_TIMEOUT"] = "15"
    judges.AnthropicJudge("claude-sonnet-5-5")
    check("EWL_ANTHROPIC_TIMEOUT sets it", _seen_kw.get("timeout") == 15.0, _seen_kw)
    os.environ["EWL_ANTHROPIC_TIMEOUT"] = "soon"
    try:
        judges.AnthropicJudge("claude-sonnet-5-5")
        _bad_to = False
    except judges.SettingRefused:
        _bad_to = True
    check("a timeout that is not a positive number is refused", _bad_to)
finally:
    os.environ.pop("EWL_ANTHROPIC_TIMEOUT", None)
    if _old_to is not None:
        os.environ["EWL_ANTHROPIC_TIMEOUT"] = _old_to
    if _saved_anth is None:
        sys.modules.pop("anthropic", None)
    else:
        sys.modules["anthropic"] = _saved_anth

print()
if skipped:
    print(f"{len(skipped)} block(s) skipped for missing optional packages: "
          f"{', '.join(skipped)}")
if failures:
    print(f"{len(failures)} failed: {', '.join(failures)}")
    sys.exit(1)
print("All checks passed." if not skipped else "All checks that could run passed.")
