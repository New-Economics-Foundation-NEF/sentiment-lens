"""Pluggable example selectors: what few-shot examples does a comment get?

    from sentiment_lens.label.examples import create
    selector = create("fixed")
    selector = create("retrieved", pool_csv="gold_pool.csv", k=12)
    shown = selector.select("They bought the whole election.")

The second variable in the #11 scorecard. Held separate from the judge so the
two can be crossed: same model with fixed vs retrieved examples isolates
whether retrieval helps, which a bundled change could not tell you.

`retrieved` is the idea in #11 — for each comment, find the nearest labelled
comments in a human-coded pool and show those. A comment about offshore trusts
gets shown other tax and lobbying comments and their labels, rather than the
same five generic examples every time. This is the boundary the pilot got
wrong, and where nearest-neighbour examples should help most.

At a few hundred rows a brute-force cosine over a matrix is instant, so there
is no index and no FAISS dependency.
"""
import csv
import random

from sentiment_lens import taxonomy

_REGISTRY = {}


def register(name):
    def wrap(cls):
        _REGISTRY[name] = cls
        cls.name = name
        return cls
    return wrap


def create(name, **kwargs):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown selector {name!r}. Available: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[name](**kwargs)


def available():
    return sorted(_REGISTRY)


def _normalise(text):
    """Whitespace- and case-insensitive comparison, so a comment that survived
    a round-trip through CSV still matches its own pool entry."""
    return " ".join((text or "").split()).casefold()


def read_pool(path, label_column="hand_domain"):
    """Load a hand-coded CSV into (comment, label) pairs.

    Reads the file `validate.py sample` produces once humans have filled in
    `hand_domain`, so the retrieval pool and the hand-coding workflow are the
    same artefact rather than two things to keep in step.
    """
    pairs = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            body = (row.get("body") or "").strip()
            raw = (row.get(label_column) or "").strip().lower()
            if not body or not raw:
                continue
            labels = [p for p in taxonomy.parse_labels(raw)
                      if p not in (taxonomy.UNCLASSIFIED, taxonomy.PARSE_FAILURE)]
            pairs.append((body, ", ".join(labels) if labels else "none"))
    if not pairs:
        raise ValueError(f"No usable rows in {path}: need a `body` and a `{label_column}` column.")
    return pairs


class Selector:
    def select(self, body):
        raise NotImplementedError

    def describe(self):
        return self.name


@register("fixed")
class FixedSelector(Selector):
    """The fixed set (taxonomy.FIXED_EXAMPLES, 25 synthetic examples). The incumbent, and the control arm."""

    def __init__(self, pairs=None, **_ignored):
        # Silently ignores pool/k/n, by design -- it has no pool. The refusal
        # lives in the CLI (see `refuse_pool_flags_for_fixed`), because only
        # argparse can tell an explicitly-typed --k from its default, and
        # callers pass the default unconditionally.
        self.pairs = list(pairs or taxonomy.FIXED_EXAMPLES)

    def select(self, body):
        return self.pairs

    def describe(self):
        return f"fixed({len(self.pairs)})"


@register("fixed_random")
class RandomFixedSelector(Selector):
    """N examples drawn once at random from a hand-coded pool, then frozen
    for the rest of the run -- every comment sees the exact same N examples.

    Distinct from both existing selectors: `fixed` is hand-picked (the five
    curated FIXED_EXAMPLES), `retrieved` picks different, comment-relevant
    examples every time. This one is neither -- a random, but reproducible
    (seeded), draw from real hand-coded data, used unchanged for every
    comment. That "unchanged for every comment" property is the whole point:
    because the resulting prompt is byte-identical across the entire corpus,
    not just within one comment's --votes calls, it is corpus-wide cacheable
    by AnthropicJudge._cacheable_system, the same way `retrieved` can only
    ever be cached per-comment. Whether that is worth the accuracy cost of
    giving up per-comment relevance is a separate question `slens-evaluate
    scorecard` can answer -- this selector exists to make that comparison
    possible, not to claim it is better.
    """

    def __init__(self, pool_csv=None, pairs=None, n=10, seed=42,
                 label_column="hand_domain", **_ignored):
        if pairs is None:
            if not pool_csv:
                raise ValueError("fixed_random selector needs pool_csv= or pairs=")
            pairs = read_pool(pool_csv, label_column)
        pool = list(pairs)
        n = min(n, len(pool))
        self.pairs = random.Random(seed).sample(pool, n)
        self.seed = seed

    def select(self, body):
        return self.pairs

    def describe(self):
        return f"fixed_random(n={len(self.pairs)}, seed={self.seed})"


@register("retrieved")
class RetrievedSelector(Selector):
    """k nearest labelled examples by embedding cosine similarity."""

    def __init__(self, pool_csv=None, pairs=None, k=12,
                 model="sentence-transformers/all-MiniLM-L6-v2",
                 label_column="hand_domain", **_ignored):
        if pairs is None:
            if not pool_csv:
                raise ValueError("retrieved selector needs pool_csv= or pairs=")
            pairs = read_pool(pool_csv, label_column)
        self.pairs = list(pairs)
        # One slot is reserved: a comment that is in its own pool has only
        # len(pairs) - 1 examples available to it, and capping k at len(pairs)
        # would hand it one example fewer than every other comment — a quiet
        # inconsistency in the thing being measured.
        self.k = min(k, max(len(self.pairs) - 1, 1))
        self.model_name = model
        self.self_matches = 0

        from sentence_transformers import SentenceTransformer
        self._encoder = SentenceTransformer(model)
        # normalize_embeddings makes cosine similarity a plain dot product.
        self._matrix = self._encoder.encode(
            [text for text, _ in self.pairs],
            normalize_embeddings=True, show_progress_bar=False,
        )

    def select(self, body):
        query = self._encoder.encode([body], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self._matrix @ query
        # Descending, so we can drop an exact self-match and backfill from the
        # next-most-similar example without losing k. A comment being
        # classified can end up in its own pool when a hand-coded corpus is
        # relabelled; showing it its own gold answer would score it perfectly
        # while testing nothing.
        #
        # Compared on normalised text rather than raw equality: a comment that
        # has been through a CSV round-trip can come back with collapsed
        # newlines or altered spacing and no longer match its own pool entry
        # character-for-character, which would let the self-match slip through
        # silently — the one failure this guard exists to prevent.
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        target = _normalise(body)
        chosen = []
        for i in ranked:
            if _normalise(self.pairs[i][0]) == target:
                self.self_matches += 1
                continue
            chosen.append(i)
            if len(chosen) == self.k:
                break
        # Most similar last, closest to the comment being classified — the
        # position models attend to most.
        return [self.pairs[i] for i in reversed(chosen)]

    def describe(self):
        return f"retrieved(k={self.k}, {self.model_name.split('/')[-1]})"


@register("none")
class NoSelector(Selector):
    """Zero-shot: definitions only, no examples. Shows how much of the
    performance is the taxonomy wording rather than the examples."""

    def __init__(self, **_ignored):
        pass

    def select(self, body):
        return []


def refuse_pool_flags_for_fixed(selector, pool_csv, k_given):
    """Refuse `--selector fixed` combined with pool/k flags, at the CLI.

    `FixedSelector` has no pool, so `--selector fixed --k 25 --pool gold.csv`
    used to run the five hardcoded anchors and report `fixed(5)` -- a wrong
    method rather than a wrong flag, and five examples is also under every
    model's minimum cacheable prefix, so the run silently loses caching too.

    Checked here rather than in the selector because callers pass `k` with its
    default on every call; only argparse knows whether the user typed it.
    """
    if selector != "fixed":
        return
    asked = [name for name, given in (("--pool", pool_csv), ("--k", k_given)) if given]
    if asked:
        raise SystemExit(
            f"`--selector fixed` does not draw from a pool, so {' and '.join(asked)} "
            f"would be ignored and you would silently get the "
            f"{len(taxonomy.FIXED_EXAMPLES)} hardcoded anchors. Use "
            f"`--selector fixed_random --pool <gold.csv> --k <n>` to draw a fixed set "
            f"of that size from the gold pool.")
