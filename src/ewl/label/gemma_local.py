"""The open-weight arm: Gemma 2 27B (instruct, Q4_0 quantisation), run locally.

    from ewl.label.judges import create
    judge = create("gemma")          # the only model this backend will run

Everything runs on the researcher's own machine through Ollama. No comment text
leaves the laptop, which is a data-protection advantage over every API arm, and
there is no per-token bill (the scorecard's cost column reads "free"; wall-clock
time is the figure to compare).

Why this is its own module and not just `--judge ollama:<tag>`
---------------------------------------------------------------
`OllamaJudge` will run whatever tag it is given. For a comparison that has to
say "this is Gemma 2 27B at Q4_0 and nothing else", that is the wrong default: a
typo, or a tag that resolves to something else, would silently benchmark a
different model and still print a plausible scorecard. This backend therefore

* accepts exactly one model tag (`MODEL_TAG`) and raises on any other,
* checks at construction that the server is reachable, that the tag has been
  pulled, and that Ollama reports its quantisation as Q4_0 (`preflight`), and
* sets the request options that matter for a local model on a laptop explicitly
  instead of trusting Ollama's defaults (see `NUM_CTX`, `NUM_PREDICT`).

The spec string is part of the response-cache key, so the standalone script
(`ewl-gemma`), the domain scorecard arm (`gemma/fixed`) and the sentiment arm
(`llm:gemma`) all resolve to this same `MODEL_TAG` and share cache entries.

Behaviour worth knowing
-----------------------
* `votes` is always 1. The call is made at temperature 0, so a second and third
  vote would repeat the first, at three times the wall-clock cost, and would make
  any self-consistency figure trivially 1.0. The scorecard and `LLMScorer` both
  respect this (see `LLMScorer.__init__`).
  **The one exception is a deliberate stability measurement.** Setting
  `EWL_GEMMA_TEMPERATURE` to a value above 0 (for example 0.7) lifts the one-vote
  limit, so `--votes 3` really makes three calls whose seeds differ (42, 43, 44) and
  the stability columns mean something. That is a different instrument from the
  frozen temperature-0 arm, so it gets its own cache key (`|temperature=0.7`) and
  the run record's `model_build` says so; use a separate cache file for it and never
  the corpus run's cache. The default is unchanged.
* `deterministic` is False even at temperature 0 with a fixed seed. Metal
  kernels, and Ollama reusing an earlier prompt's cached prefix, are not
  guaranteed bit-identical from run to run, so the response cache is the
  reproducibility mechanism here exactly as it is for the API arms. If labels
  from this arm are quoted, its cache file is a required artefact.
* Token counts in the scorecard are what Ollama reports as *newly evaluated*.
  When a prompt prefix is reused they are far smaller than the prompt's real
  length, so they are not comparable with API-billed token counts.
* Gemma 2 has no separate system role. Ollama's template puts the system text at
  the start of the first user turn, so the instructions still come first and the
  comment last, which is what lets a repeated prompt prefix be reused.
* The model's context window is 8,192 tokens, and that is also the default here.
  The comment is truncated (and counted in `usage["truncated_bodies"]`) if it
  could not fit. Ollama's own behaviour on an over-long prompt is to drop the
  START of it, which would silently remove the instructions and leave a
  plausible-looking label; truncating the comment instead keeps the
  instructions intact.
* Unverified: Gemma 2 uses sliding-window attention, and reuse of a cached prompt
  prefix is more delicate with that than with plain attention. The prompts here
  (about 1,500 tokens) sit well inside the 4,096-token window, which should keep
  reuse working, but that is an expectation, not a measurement.
  `ewl-gemma --sample 60 --project-to N` shows whether it holds: a prompt being
  re-read on every call is flagged as slow.
"""
import os

from ewl.label import judges

# The one model this backend runs. Confirmed on ollama.com/library/gemma2/tags
# on 2026-09-30: `gemma2:27b` is the instruct model, 16 GB, with the same hash as
# `gemma2:27b-instruct-q4_0` (so it is Q4_0, not the Q4_K_M of the larger
# 27b-instruct-q4_K_M tag). That page is the source; `preflight` re-checks the
# quantisation on the machine itself, so a tag that later resolved to something
# else would be refused, not used.
MODEL_TAG = "gemma2:27b"
REQUIRED_QUANTISATION = "Q4_0"

# Context window, in tokens: Gemma 2's maximum. Ollama's default is small enough
# that the 25-example domain prompt (~1,450 tokens) plus a long comment can
# overflow it, and an overflow is silent (see the module docstring). 8192 leaves
# room for the longest allowed YouTube comment (10,000 characters) on top of the
# biggest prompt, at a KV-cache cost of very roughly 3 GB. Override with
# EWL_GEMMA_NUM_CTX (values above the model's own 8,192 are refused).
NUM_CTX = 8192
_MIN_NUM_CTX = 4096
_MAX_NUM_CTX = 8192

# Longest useful reply is "environment, social" or a single digit; 32 leaves
# margin for a model that prefixes its answer, without letting a chatty reply
# run on.
NUM_PREDICT = 32

# How long Ollama keeps the 16 GB model in memory after the last request. Its
# default is 5 minutes, which is shorter than some pauses in a long run and
# would mean reloading 16 GB. Override with EWL_GEMMA_KEEP_ALIVE.
KEEP_ALIVE = "4h"

# Deliberately pessimistic characters-per-token for noisy YouTube text (English
# prose is nearer 4; emoji and non-Latin scripts are far worse).
_CHARS_PER_TOKEN = 2.5


class GemmaUnavailable(RuntimeError):
    """The local Gemma model cannot be used, with the fix in the message."""


def _field(obj, name, default=None):
    """Read `name` from a dict or an object. The ollama package has returned
    both shapes across versions."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _new_client():
    try:
        import ollama
    except ImportError as error:
        raise GemmaUnavailable(
            "The `ollama` Python package is not installed. Run: pip install -e \".[ollama]\""
        ) from error
    # No host argument: the client reads OLLAMA_HOST from the environment, and
    # falls back to the local server.
    return ollama.Client()


def preflight(client=None):
    """Confirm the local Gemma model can actually be used, or say how to fix it.

    Checks, in order: the Ollama server is reachable; `MODEL_TAG` is present
    locally; Ollama reports it as Q4_0. Returns a small dict describing what
    was found, and raises `GemmaUnavailable` otherwise.
    """
    client = client if client is not None else _new_client()
    try:
        listing = client.list()
    except Exception as error:
        raise GemmaUnavailable(
            f"Cannot reach the Ollama server ({type(error).__name__}: {error}). "
            f"Start it with `ollama serve` (or open the Ollama app), then try again."
        ) from error

    names = set()
    digest = None
    for entry in _field(listing, "models", []) or []:
        name = _field(entry, "model") or _field(entry, "name")
        if name:
            names.add(name)
            if name == MODEL_TAG:
                digest = _field(entry, "digest")
    if MODEL_TAG not in names:
        raise GemmaUnavailable(
            f"{MODEL_TAG} has not been pulled on this machine. Run:\n"
            f"    ollama pull {MODEL_TAG}\n"
            f"(about 16 GB). Models present now: {', '.join(sorted(names)) or '(none)'}.")

    details = _field(client.show(MODEL_TAG), "details")
    quantisation = _field(details, "quantization_level")
    if quantisation != REQUIRED_QUANTISATION:
        raise GemmaUnavailable(
            f"{MODEL_TAG} is present but Ollama reports its quantisation as "
            f"{quantisation!r}, not {REQUIRED_QUANTISATION!r}. Refusing to run: this arm is "
            f"defined as Gemma 2 27B Instruct at Q4_0 and nothing else. Remove it "
            f"(`ollama rm {MODEL_TAG}`) and pull it again.")
    return {"model": MODEL_TAG, "quantisation": quantisation,
            "parameter_size": _field(details, "parameter_size"),
            # Which weights and which Ollama build did the labelling. A re-pull of the model
            # or an Ollama update between batches would otherwise mix two builds in one
            # merged file with no trace; `ewl-batches check` compares both.
            "digest": digest, "server_version": _server_version(client)}


def _server_version(client):
    """The Ollama server's version string, or None when it cannot be read."""
    try:
        version = getattr(client, "version", None)
        if callable(version):
            reply = version()
            return str(_field(reply, "version") or reply)
    except Exception:
        pass
    try:
        # The ollama client's own HTTP session: no second network path in this module.
        http = getattr(client, "_client", None)
        if http is not None:
            return http.get("/api/version").json().get("version")
    except Exception:
        pass
    return None


@judges.register("gemma")
class GemmaJudge(judges.Judge):
    """Gemma 2 27B (instruct, Q4_0) through a local Ollama server."""

    deterministic = False
    default_model = MODEL_TAG

    def __init__(self, model=None, **kwargs):
        if model and model != MODEL_TAG:
            raise ValueError(
                f"The gemma backend runs exactly one model, {MODEL_TAG}, and was asked for "
                f"{model!r}. Use `--judge gemma` (or `gemma/fixed` as a scorecard arm). To try "
                f"any other local model, use the general `ollama:<model>` backend instead.")
        # EWL_GEMMA_TEMPERATURE (default 0, the frozen arm). Above 0 the votes can differ, so
        # the one-vote limit is lifted; at 0 repeat votes add nothing (module docstring).
        raw_temperature = os.environ.get("EWL_GEMMA_TEMPERATURE", "").strip()
        try:
            self.temperature = float(raw_temperature) if raw_temperature else 0.0
        except ValueError:
            raise ValueError(f"EWL_GEMMA_TEMPERATURE={raw_temperature!r} is not a number "
                             f"(0 is the frozen setting; try 0.7 for a stability measurement).")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"EWL_GEMMA_TEMPERATURE={self.temperature} is outside 0 to 2.")
        if self.temperature == 0.0:
            kwargs["votes"] = 1
        super().__init__(model, **kwargs)
        self.num_ctx = int(os.environ.get("EWL_GEMMA_NUM_CTX", NUM_CTX))
        if not _MIN_NUM_CTX <= self.num_ctx <= _MAX_NUM_CTX:
            raise ValueError(
                f"EWL_GEMMA_NUM_CTX={self.num_ctx} is outside {_MIN_NUM_CTX}-{_MAX_NUM_CTX}. "
                f"The prompts alone need about 2,100 tokens, and a smaller window makes Ollama "
                f"silently drop the start of the prompt; Gemma 2's own window is "
                f"{_MAX_NUM_CTX}, so a larger one is not usable.")
        self.keep_alive = os.environ.get("EWL_GEMMA_KEEP_ALIVE", KEEP_ALIVE)
        self._client = self._make_client()
        self.preflight_info = preflight(self._client)
        if self.temperature > 0:
            # Only added when it differs from the frozen setting, so a default run record is
            # exactly what it was.
            self.preflight_info = dict(self.preflight_info, temperature=self.temperature)

    @property
    def spec(self):
        """The arm's name. A non-zero temperature is part of it (`gemma:gemma2:27b@t=0.7`), so a
        scorecard row, an output column and the start-up line say which setting produced it. At the
        default temperature the name is exactly what it always was."""
        temperature = getattr(self, "temperature", 0.0)
        return f"{super().spec}@t={temperature:g}" if temperature > 0 else super().spec

    @property
    def cache_spec(self):
        """The default keeps the key it always had; a non-zero temperature gets its own, so
        sampled answers are never replayed as if they were the temperature-0 arm's."""
        base = super().cache_spec
        temperature = getattr(self, "temperature", 0.0)
        return f"{base}|temperature={temperature:g}" if temperature > 0 else base

    def _make_client(self):
        return _new_client()

    def _fit_body(self, body, prompt):
        """The comment, cut to fit the context window if it could not (see the
        module docstring for why the comment is cut rather than the prompt)."""
        reserved_tokens = len(prompt) / 3 + NUM_PREDICT + 128
        max_chars = int(max(self.num_ctx - reserved_tokens, 256) * _CHARS_PER_TOKEN)
        if len(body) <= max_chars:
            return body
        with self._lock:
            self.usage["truncated_bodies"] += 1
        return body[:max_chars]

    def _sample(self, body, prompt, vote):
        body = self._fit_body(body, prompt)
        response = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f'Classify this comment: "{body}"'},
            ],
            options={
                "temperature": getattr(self, "temperature", 0),
                "seed": 42 + vote,
                "num_predict": NUM_PREDICT,
                "num_ctx": self.num_ctx,
            },
            keep_alive=self.keep_alive,
        )
        with self._lock:
            self.usage["input_tokens"] += response.get("prompt_eval_count", 0) or 0
            self.usage["output_tokens"] += response.get("eval_count", 0) or 0
            # A reply cut off at NUM_PREDICT is almost always a model ignoring
            # "answer with the labels only"; counted so a high parse-failure
            # rate can be traced to it rather than guessed at.
            if response.get("done_reason") == "length":
                self.usage["truncated_replies"] += 1
        return response["message"]["content"]
