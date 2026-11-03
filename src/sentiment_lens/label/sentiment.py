"""Pluggable sentiment scorers, so VADER can be compared against alternatives.

    from sentiment_lens.label.sentiment import create
    scorer = create("vader")
    scorer = create("llm:anthropic:claude-haiku-4-5")
    result = scorer.score("Nobody works a billion dollars hard.")
    # {"label": "negative", "value": -0.42, "raw": ...}

Motivated by #9. VADER's compound score is length-dependent, and the pipeline
uses it to compare sentiment between stories whose comments differ in length.
Measured on 50 comments from a wealth-tax video:

    words       n   mean |compound|   scored neutral
    0-5         9             0.197              67%
    5-15       15             0.440              13%
    15-40      15             0.468               7%
    40+        11             0.657               9%

    Pearson r(word count, |compound|) = +0.32

The mechanism is not subtle. Repeating one sentence verbatim, which cannot
change its sentiment, moves the score most of the way to the maximum:

    "This is outrageous and unfair."   x1  ->  -0.727
                                       x2  ->  -0.904
                                       x4  ->  -0.973
                                       x8  ->  -0.993

compound normalises a *sum* of valences by a constant, so more sentiment-
bearing words means a larger magnitude regardless of intensity. A story whose
comments run long will therefore look more polarised than one whose comments
run short, even when feeling is identical — which is exactly the comparison the
briefing makes.

An LLM asked for a fixed rating has no such artefact: the scale is bounded by
the instruction rather than accumulated from token counts. It should also
read sarcasm, which VADER cannot, and which the README flags as bearing hardest
on the UnitedHealthcare material where ironic approval reads as positive.

Whether it is actually *better* is an empirical question, and needs hand-coded
sentiment labels exactly as the harm domains do. This module makes the
comparison runnable; it does not settle it.

REVISED 2026-09: the prompt used to ask the model to rate "the feeling the
comment EXPRESSES" -- expressed emotion, not stance toward extreme wealth. On
that design "eat the rich" scored negative for being hostile in tone, and a
"positive" share was not an approval share (CLAUDE.md flagged this as an open
decision -- see its "Open decisions" section history). PROMPT below instead
asks for the comment's STANCE on extreme wealth: does it approve of, or
condemn, how wealth is being used. See PROMPT's own rules for the worked
examples that motivated the change (tone and stance can point opposite ways).
"""
import collections
import re
import statistics
import sys

_REGISTRY = {}

# -1 and 0 are not points on the 1-5 stance scale -- they are the same two
# categorical outs evaluate.bucket_hand_sentiment gives a human coder
# (IRRELEVANT/UNCODEABLE there). Kept as separate string constants here,
# mirrored rather than imported, since sentiment.py is the module evaluate.py
# imports from and not the reverse. Letting the LLM itself answer with these
# (rather than only ever emitting a 1-5 stance rating) is the decision
# CLAUDE.md's sentiment-prompt-rewrite section originally left open.
IRRELEVANT = "irrelevant"
UNCODEABLE = "uncodeable"
# Two more labels that are NOT stances and never a "neutral": the model gave no
# readable answer at all, or its votes split with no majority kind. They used to
# be saved as a real neutral (0.0), which put every failure into the neutral
# share and the mean stance.
PARSE_FAILURE = "parse_failure"
NO_CONSENSUS = "no_consensus"
NOT_A_STANCE = (IRRELEVANT, UNCODEABLE, PARSE_FAILURE, NO_CONSENSUS)

# The scale a judge is asked to rate against. -1/0 are categorical, matching
# hand-coding; 1-5 is the comment's STANCE on extreme wealth (does it approve
# or condemn how wealth is being used), not the emotional tone of its
# wording -- see PROMPT below for why that distinction matters ("eat the
# rich" is hostile in tone but its stance is unambiguously negative on 1-5).
SCALE = {
    -1: "not codable — a language barrier or something else makes the comment genuinely incomprehensible, not just off-topic",
    0: "irrelevant — takes no position on extreme wealth, whether or not it is about the video's subject: a reaction or joke, a reply to someone else, a question, a practical or factual remark about the topic, a personal anecdote, a general political comment that is not about wealth, spam",
    1: "very negative — hates the idea of extreme wealth, may be demanding action against it",
    2: "negative — broadly against extreme wealth, expresses dissatisfaction, but not as strongly as 1",
    3: "neutral — engages with the topic but shows balance or does not pick a side",
    4: "positive — likes the idea of extreme wealth, or defends the wealthy people or companies involved, or how they gain or use their wealth",
    5: "very positive — actively defends, supports, or celebrates the idea of extreme wealth",
}
# Both anchors above USED to reference "the story" and "a mocking tone" --
# tone and story-referent language, not stance language, and both broke on
# Luigi specifically: "the story" there is a killing, so approving of the
# killing could read as scoring positive toward wealth, and "a mocking tone"
# said nothing about WHO the mockery targets, when that is exactly what
# determines the stance (mocking the CEO reads very differently from mocking
# Luigi). Reworded to be pure stance-on-wealth language that means the same
# thing regardless of which story a comment sits under -- the story-specific
# disambiguation this used to smuggle in (what counts as approving of THIS
# story's wealth) now lives entirely in STORY_CONTEXT below, one paragraph
# per story, so this dict never has to vary by story to stay correct.

# Per-story context, one paragraph per `story` value in the corpus (matches
# the 4 keys slens-collect uses via --story). Shown to the judge alongside the
# comment so a terse, deictic comment ("Free Luigi") can be resolved at all --
# without it the model sees only the bare comment text and has nothing to
# resolve such a comment against (see the conversation this was designed in:
# a bare "Free Luigi" is unreadable without knowing which Luigi). Purely
# factual/descriptive -- no framing of how a reader "should" feel about the
# story -- so the PROMPT's own rule below (don't guess a stance from the
# story alone) has something real to enforce against.
STORY_CONTEXT = {
    "luigi": (
        "This comment was posted on a video about: the killing of UnitedHealthcare CEO "
        "Brian Thompson in December 2024, for which Luigi Mangione was arrested and charged. "
        "The story prompted extensive public debate about US healthcare costs, insurance "
        "claim denials, corporate accountability, and vigilantism, with reactions ranging "
        "from condemnation of the killing to sympathy rooted in anger at the healthcare "
        "industry. Comments referencing 'Luigi' mean Luigi Mangione. "
        "The wealth at issue is that of the health insurance industry and its "
        "executives; the killing is the event being discussed, and a comment's stance is "
        "its stance toward that wealth. "
        "A reaction to the killing is used as a signal of that stance, following the "
        "coders' brief, although it is an inference: comments defending or praising Luigi "
        "are generally hostile to the healthcare/insurance industry's wealth, not "
        "supportive of it, regardless of how calm or approving their own tone is; comments "
        "whose main point is that what Luigi did was wrong, or that the CEO should not have "
        "been killed, with no stated view on the insurers, are generally supportive of that "
        "wealth; but a comment that says so AND criticises the insurers or the healthcare "
        "system takes a mixed position and is neutral. Anything a comment says explicitly "
        "about the insurers, health care or wealth takes priority over this."
    ),
    "musk": (
        "This comment was posted on a video about: Elon Musk's purchase of Twitter (now X) "
        "for approximately $44 billion in October 2022, after a months-long acquisition "
        "process. Commentary covers free speech and content moderation on the platform, "
        "Musk's management and layoffs, his political views and influence, and the broader "
        "question of a single wealthy individual controlling a major public communication "
        "platform. Comments may reference the platform as 'Twitter' or 'X' interchangeably, "
        "and may discuss Musk's other companies (Tesla, SpaceX) as part of the same "
        "wealth-and-influence discussion. Comments arguing that Musk's ownership defends "
        "democracy or free speech, that he has the right to spend his money as he chooses, "
        "or that praise his business capabilities, are generally supportive of extreme "
        "wealth, regardless of how measured or matter-of-fact their tone is. Comments "
        "criticising his control of the platform, his management or his political "
        "influence, or objecting to one wealthy individual owning it, are generally "
        "critical of extreme wealth, however calmly they are put."
    ),
    "jets": (
        "This comment was posted on a video about: the use of private jets by celebrities "
        "and the ultra-wealthy, including tracking of individual flights and their carbon "
        "emissions. Commentary covers the environmental impact of private aviation, the "
        "fairness of luxury consumption amid climate concerns, and specific celebrities' jet "
        "usage and public statements about it. A comment naming a specific celebrity is "
        "usually about that person's own jet use, not a general statement about celebrities "
        "as a class. Comments arguing that flying privately is a legitimate right of the "
        "wealthy -- earned through income, effort, talent, or simply a matter of convenience "
        "-- are generally supportive of extreme wealth, regardless of how calmly they "
        "express it. Comments objecting to the emissions, the hypocrisy or the sense of "
        "entitlement in flying privately are generally critical of extreme wealth, "
        "however calmly they are put."
    ),
    "propublica": (
        "This comment was posted on a video about: ProPublica's 2021 'Secret IRS Files' "
        "investigation, which used leaked tax data to report on how little federal income "
        "tax America's wealthiest individuals (including Jeff Bezos, Elon Musk, and Warren "
        "Buffett) pay relative to their wealth growth, and the legal strategies used to "
        "minimise their tax bills. Comments may reference specific named billionaires from "
        "this reporting, general 'loopholes', or the IRS/tax system itself as the target of "
        "praise or criticism. Comments arguing that the tax paid is legitimate or not a "
        "scandal (for example that unrealised gains are not income), that the wealthy "
        "earned or deserve their position, that taxing them would backfire, or that their "
        "charity or spending makes up for it, are generally supportive of extreme wealth; "
        "comments condemning the loopholes or the tax system as rigged for the rich, or "
        "calling for wealth taxes or stricter enforcement, are generally critical of it; "
        "a comment that only asks how the tax rules work, or is confused by them, takes no "
        "side."
    ),
}

# --- Switches for the open instrument decisions -----------------------------
# Each default below reproduces the wording as it stands, byte for byte, so
# nothing changes until a switch is flipped. A switch changes the prompt, and
# the prompt is part of the cache key, so a flipped run never reuses an answer
# given under the other wording.
#
#   story_rules   True  = the context paragraphs include their stance rules
#                         ("comments arguing X are generally supportive of ...")
#                 False = factual context only: who and what the video is about
#   stance_object "conduct" = anchor 4 as it stands ("... defends the wealthy
#                             people or companies involved, or how they gain or use their wealth")
#                 "wealth"  = anchor 4 about extreme wealth itself and the
#                             people who hold it, in line with anchors 1 and 5
#                             (DRAFT wording, for the decision)
_CONFIG = {"story_rules": True, "stance_object": "conduct", "sentiment_examples": True}

# Where each story's stance rule begins inside its context paragraph. Cutting
# there leaves the factual part. (Luigi's rule starts mid-sentence, so its
# factual part is closed with a full stop.)
STORY_RULE_MARKERS = {
    "luigi": " The wealth at issue is that of the health insurance industry",
    "musk": " Comments arguing that Musk's ownership",
    "jets": " Comments arguing that flying privately",
    "propublica": " Comments arguing that the tax paid is legitimate",
}

ANCHOR_4 = {
    "conduct": SCALE[4],
    "wealth": "positive \u2014 approves of extreme wealth, or of the people who hold it",
}


def configure(story_rules=None, stance_object=None, sentiment_examples=None):
    """Set the instrument switches. Returns the active settings."""
    if story_rules is not None:
        _CONFIG["story_rules"] = bool(story_rules)
    if sentiment_examples is not None:
        _CONFIG["sentiment_examples"] = bool(sentiment_examples)
    if stance_object is not None:
        if stance_object not in ANCHOR_4:
            raise ValueError(f"stance_object must be one of {sorted(ANCHOR_4)}, "
                             f"not {stance_object!r}")
        _CONFIG["stance_object"] = stance_object
    return dict(_CONFIG)


def facts_only(story):
    """A story's context paragraph without its stance rule(s)."""
    text = STORY_CONTEXT[story]
    marker = STORY_RULE_MARKERS.get(story)
    if marker is None or marker not in text:
        return text
    head = text.split(marker, 1)[0].rstrip()
    return head if head.endswith(".") else head + "."


def rules_only(story):
    """The stance rule(s) of a story's context paragraph, as a sentence of their own, or ""
    when the story has none (ProPublica) -- the part `facts_only` leaves out."""
    text = STORY_CONTEXT.get(story, "")
    marker = STORY_RULE_MARKERS.get(story)
    if marker is None or marker not in text:
        return ""
    rule = (marker + text.split(marker, 1)[1]).lstrip("; ").strip()
    return rule[:1].upper() + rule[1:]


def story_context_for(story):
    """The context paragraph shown for `story` under the current switches."""
    if story not in STORY_CONTEXT:
        return NO_STORY_CONTEXT
    return STORY_CONTEXT[story] if _CONFIG["story_rules"] else facts_only(story)


def scale_for():
    """SCALE with anchor 4 chosen by the `stance_object` switch."""
    scale = dict(SCALE)
    scale[4] = ANCHOR_4[_CONFIG["stance_object"]]
    return scale


def config_tag():
    """"" at the defaults; otherwise a short tag naming what was switched, so an
    output file says which wording produced it."""
    parts = []
    if not _CONFIG["story_rules"]:
        parts.append("story_rules=off")
    if _CONFIG["stance_object"] != "conduct":
        parts.append(f"stance_object={_CONFIG['stance_object']}")
    if not _CONFIG["sentiment_examples"]:
        parts.append("sentiment_examples=off")
    return f"[{','.join(parts)}]" if parts else ""


# The rule about how to use the story context. When the context paragraph is facts only, the
# model must not infer a stance from it. When the paragraph ALSO carries guidance on what
# common positions mean (the story rules), "use it only to understand references" would
# contradict that guidance, so the rule is worded to say how the two fit: the guidance is the
# usual reading, the comment's own words win. (The coders' brief worked the same way: worked
# examples of how each position scores, plus "go with your gut" for the rest.)
CONTEXT_RULE_FACTS = (
    "- Use the context above only to understand who or what the comment refers to, not to "
    "guess a comment's stance from the story alone. Plenty of comments on any of these "
    "stories disagree with the majority reaction -- rate what THIS comment actually says, "
    "not what a typical comment on this story tends to say.")
CONTEXT_RULE_WITH_RULES = (
    "- Use the context above to understand who or what the comment refers to and what the "
    "common positions on this story usually mean for the stance (the guidance it gives). "
    "Rate what THIS comment actually says: do not infer a stance from the story alone, and "
    "where the comment's own words clearly say something different from the usual reading, "
    "or take a mixed position, follow the comment.")
CONTEXT_RULE_FACTS_SHORT = (
    "- Use the context above only to understand who/what the comment refers to, not to "
    "guess a stance from the story alone -- rate what THIS comment actually says.")
CONTEXT_RULE_WITH_RULES_SHORT = (
    "- Use the context above to understand who/what the comment refers to and what the "
    "common positions on this story usually mean for the stance (the guidance it gives); "
    "rate what THIS comment actually says -- where its own words differ from the usual "
    "reading, or take a mixed position, follow the comment.")


def context_rule_for(has_rules=None, short=False):
    """The wording of the context rule: with the story rules, or for facts only. `has_rules`
    defaults to the current `story_rules` switch."""
    if has_rules is None:
        has_rules = _CONFIG["story_rules"]
    if short:
        return CONTEXT_RULE_WITH_RULES_SHORT if has_rules else CONTEXT_RULE_FACTS_SHORT
    return CONTEXT_RULE_WITH_RULES if has_rules else CONTEXT_RULE_FACTS


PROMPT = """You are rating a public YouTube comment's STANCE on extreme wealth for a research study — not the emotional tone of its language, and not whether you personally agree with it.

{story_context}

First check whether the comment can be rated at all, then rate its stance on this scale:
{scale}

Rules:
- Use -1 only when the comment itself cannot be judged — it is in a language you cannot read, or the text is genuinely incomprehensible. Do not use -1 just because a comment is unusual, rude, or hard to categorise; if you can read it, rate it.
- Use 0 when the comment is comprehensible but takes no position on extreme wealth, whether or not it is about the video's subject: a reaction or joke, a reply to another commenter, a question, a practical or factual remark about the topic, a personal anecdote, a general political comment that is not about wealth, spam, or a comment with no evident subject at all. Use 3 instead only when the comment does engage with wealth, tax or accountability and is balanced or picks no side.
- For everything else, rate the comment's stance on extreme wealth (1-5), not the emotional intensity of its wording. Tone and stance are not the same thing: a calm comment can still take a strong stance ("He earned it fair and square, no issue here" is positive even with no exclamation marks), and an angry or mocking comment can still be positive in stance if the anger targets critics of wealth rather than wealth itself ("People who have a problem with this are just envious of those who have worked harder" is positive despite the dismissive tone). The reverse also happens: a comment can be calm, even approving in tone, while its stance on wealth is negative ("It's sweet that he donates so much, shows how broken the tax system must be that he can afford to" is negative -- the warmth is directed at the donor, not at the system that let them accumulate that much).
- Read sarcasm and irony as what they mean, not what they say. "Eat the rich" is very negative (1), not neutral or positive.
{context_rule}
- A comment can shift topic partway through. Rate the part that actually engages with extreme wealth; if it only gestures at the topic before moving on to something unrelated, that's 0, not a guess based on the passing phrase.
- A comment that quotes or directly replies to another comment (starting with "@username", or restating someone else's words to disagree with them) should be rated on the stance the WRITER is taking, which is often the opposite of whatever is being quoted -- rate the argument being made, not the quoted material in isolation.
- A very short comment -- a few words, an emoji, a single phrase -- can still carry a clear stance. Do not default to 3 just because there is little text to go on; use -1 only if it is genuinely unreadable, and 0 only if it takes no position on wealth, not merely because it is brief.
- 3 is for a comment that engages with the topic but takes no side, or says the issue is complicated or unresolved — not a default for anything unclear. If it's unclear because the comment is off-topic, that's 0; if unclear because it can't be read, that's -1.
- Length is not intensity. A short blunt comment and a long measured one can express the same stance at the same strength.
- Answer with a single value from this list: -1, 0, 1, 2, 3, 4, 5. No words, no explanation.{examples_block}"""

# No story context available -- an unrecognised or missing `story` value
# (e.g. a trial run over hand-picked examples that were never assigned one
# of the four real story keys). Keeps LLMScorer.score usable without one,
# rather than crashing or silently mis-describing an unrelated comment.
# Public (no leading underscore) because combined.py's joint prompt needs the
# same fallback text for its own sentiment half.
NO_STORY_CONTEXT = (
    "This comment's source story is not specified. Rate it on the comment's own "
    "words alone, without assuming which of the four wealth-related stories it responds to."
)


def register(name):
    def wrap(cls):
        _REGISTRY[name] = cls
        cls.name = name
        return cls
    return wrap


def create(spec, **kwargs):
    """"vader", or "llm:<judge spec>" e.g. "llm:anthropic:claude-haiku-4-5"."""
    kind, _, rest = spec.partition(":")
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown scorer {kind!r}. Available: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[kind](rest or None, **kwargs)


def available():
    return sorted(_REGISTRY)


def to_label(value):
    """Shared thresholds on [-1, 1], so every scorer's labels mean the same."""
    return "positive" if value >= 0.05 else "negative" if value <= -0.05 else "neutral"


def resolve_ratings(ratings):
    """Majority-of-KIND resolution for a list of raw -1..5 ratings.

    Shared by LLMScorer.score and combined.CombinedJudge.sentiment_for -- the
    same question, asked two different ways, so it should be resolved one way
    rather than two copies that can silently drift apart (the way
    combined.py's copy already had, before this was factored out: it
    medianed every returned digit as if it were always a real 1-5 stance,
    which silently mishandled any -1/0 answer).

    -1/0 are categorical outs, not points on the -1..1 stance scale, so they
    must be resolved by which KIND of answer most votes agree on (not
    codable / irrelevant / a real stance), not by naively medianing every
    number together: medianing "irrelevant" (0) against a stance rating would
    silently launder a real disagreement about what KIND of answer this is
    into a fake stance number, and 0 fed through the (value - 3) / 2
    transform reads as -1.5 -- "more negative than very negative".

    Also handles what a naive Counter().most_common(1) tie-break does not: a
    genuine split with no majority KIND (e.g. one vote each of not-codable,
    irrelevant, and a real stance) used to be resolved by whichever vote
    happened to come back first in the ballot list -- an arbitrary,
    order-dependent answer to a question that is properly "no consensus",
    the same principle judges.Judge.resolve() already applies to domain
    labels ("no majority is not a finding").
    """
    if not ratings or all(r is None for r in ratings):
        # No valid rating came back at all (empty reply, refusal, off-format text).
        # Not a neutral stance and not a zero: its own label, no value.
        return {"label": PARSE_FAILURE, "value": None, "raw": None,
                "unstable": len(set(ratings)) > 1}

    # A `None` is a vote whose reply could not be read as a rating. It is a ballot like any
    # other (the domain judge counts an unreadable vote as a parse-failure ballot): dropping it
    # let one readable vote out of three decide the label, with nothing flagged. Here it is its
    # own kind, so two unreadable votes out of three make the comment a parse failure, an even
    # split is no consensus, and any unreadable vote makes the comment unstable.
    unstable = len(set(ratings)) > 1

    def kind(rating):
        return ("failed" if rating is None else
                "uncodeable" if rating == -1 else "irrelevant" if rating == 0 else "stance")

    kind_counts = collections.Counter(kind(r) for r in ratings)
    top_count = max(kind_counts.values())
    tied_kinds = [k for k, count in kind_counts.items() if count == top_count]
    if len(tied_kinds) > 1:
        # No majority kind -- e.g. votes split evenly across not-codable,
        # irrelevant and a real stance. Genuinely undecided: its own label and no
        # value, flagged unstable, never a neutral stance.
        return {"label": NO_CONSENSUS, "value": None, "raw": ratings, "unstable": True}
    majority_kind = tied_kinds[0]

    if majority_kind == "failed":
        return {"label": PARSE_FAILURE, "value": None, "raw": ratings, "unstable": unstable}
    if majority_kind == "uncodeable":
        return {"label": UNCODEABLE, "value": None, "raw": ratings, "unstable": unstable}
    if majority_kind == "irrelevant":
        return {"label": IRRELEVANT, "value": None, "raw": ratings, "unstable": unstable}

    stance_ratings = [r for r in ratings if r is not None and r >= 1]
    if len(stance_ratings) > 1:
        # Votes that agree on "a stance" but land on opposite sides (very negative
        # against very positive) have a median that sounds like a neutral and is
        # not one. Same rule as the domain judge's "no majority is not a finding":
        # a side (1-2 negative, 3 neutral, 4-5 positive) needs MORE than half the
        # stance votes, or the comment has no consensus stance.
        sides = collections.Counter("neg" if r <= 2 else "neu" if r == 3 else "pos"
                                    for r in stance_ratings)
        if max(sides.values()) * 2 <= len(stance_ratings):
            return {"label": NO_CONSENSUS, "value": None, "raw": ratings, "unstable": True}
    rating = statistics.median(stance_ratings)
    # 1..5 -> -1..1, so the value is directly comparable with VADER's
    # compound and the existing thresholds apply unchanged.
    return {"label": to_label((rating - 3) / 2), "value": (rating - 3) / 2,
            "raw": ratings, "unstable": unstable}


class Scorer:
    deterministic = False
    # Model calls this scorer makes per comment (one per vote). Zero for a scorer that calls no
    # model (VADER, the stub), so the speed report does not charge a call that never happens.
    votes = 0

    def score(self, body, story=None):
        raise NotImplementedError

    def describe(self):
        return self.name

    cost_usd = staticmethod(lambda: 0.0)
    cache_savings_usd = staticmethod(lambda: 0.0)
    # Cumulative wall-clock seconds spent in real calls (0.0 for anything
    # running locally/deterministically, same reasoning as cost_usd). A
    # plain attribute rather than a method so LLMScorer can just point it at
    # its inner judge's own running total -- see cmd_sentiment in evaluate.py,
    # which reads this before/after scoring an arm's rows, the same way
    # run_arm already does for judge.seconds.
    seconds = 0.0

    def cache_tokens(self):
        """(cache_creation_input_tokens, cache_read_input_tokens). 0, 0 for
        any scorer that doesn't call a caching-aware judge -- VADER, stub,
        and anything backed by a non-Anthropic judge all land here."""
        return 0, 0

    def token_usage(self):
        """(input_tokens, output_tokens) -- the plain, non-cache token counts
        cost_usd() is built from. 0, 0 for anything that doesn't call a real
        judge backend, same reasoning as cache_tokens()."""
        return 0, 0


@register("vader")
class VaderScorer(Scorer):
    """The incumbent. Kept as the baseline any replacement has to beat."""

    deterministic = True

    def __init__(self, _unused=None, **_ignored):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self._analyzer = SentimentIntensityAnalyzer()

    def score(self, body, story=None):
        # VADER has no notion of story context -- accepted and ignored so
        # every scorer shares one call signature (see LLMScorer.score).
        polarity = self._analyzer.polarity_scores(body)
        return {"label": to_label(polarity["compound"]), "value": polarity["compound"],
                "raw": polarity, "unstable": False}


def example_ratings():
    """[(story, comment, rating)] for the fixed examples, or [] when there are none to show:
    the built-in set has no ratings (shown only from a real file that carries them), or the
    `sentiment_examples` switch is off."""
    if not _CONFIG["sentiment_examples"]:
        return []
    from sentiment_lens import taxonomy
    if not taxonomy.FIXED_EXAMPLE_RATINGS:
        return []
    return [(story, text, rating) for (text, _label), rating, story in zip(
        taxonomy.FIXED_EXAMPLES, taxonomy.FIXED_EXAMPLE_RATINGS, taxonomy.FIXED_EXAMPLE_STORIES)]


def sentiment_examples_block():
    """The worked examples at the end of the sentiment prompt: "" unless real hand-coded
    ratings were loaded (`--fixed-examples FILE` with `sentiment` and `story` columns) and the
    switch is on. The same 25 comments the domain prompt shows, with their stance rating."""
    items = example_ratings()
    if not items:
        return ""
    lines = "\n".join(f'[{story}] "{text}" -> {rating}' for story, text, rating in items)
    return ("\n\nExamples of how comments were rated by hand (the story each was posted under is "
            "in brackets, and the guidance above is for this comment's story only):\n" + lines)


@register("llm")
class LLMScorer(Scorer):
    """A -1..5 rating from any registered judge backend."""

    def __init__(self, judge_spec=None, votes=3, cache_path=None, **_ignored):
        from sentiment_lens.label import judges
        self._judge = judges.create(judge_spec or "anthropic:claude-haiku-4-5",
                                    votes=votes, cache_path=cache_path)
        # Never more votes than the judge itself allows. A judge that pins its
        # own vote count to 1 (the keyword judge, a local model at temperature 0
        # -- repeating a deterministic call adds no information) would otherwise
        # still be called `votes` times here, since this loop is independent of
        # the judge's own `votes`, and each repeat is a full extra call with a
        # different cache key. For every other judge `judge.votes == votes`, so
        # this changes nothing.
        self.votes = max(1, min(votes, self._judge.votes))
        self.deterministic = self._judge.deterministic
        # One fully-assembled prompt per story (plus a fallback for an
        # unrecognised/missing story), not one prompt built lazily per call --
        # every comment in the same story gets the exact same prompt STRING,
        # which is what lets a caching-aware judge (AnthropicJudge) recognise
        # repeats and serve them from cache. Building it per-call from the
        # same inputs would produce identical text anyway, but pre-building
        # once avoids paying that formatting cost 84,000+ times over.
        scale_text = "\n".join(f"  {n} = {text}" for n, text in scale_for().items())
        self._config_tag = config_tag()
        # A story paragraph with no rules in it (ProPublica) still sits under the matching rule
        # when the switch is on: the rule only says how to read guidance IF the context gives
        # any, so it is harmless there. The no-story fallback has no guidance at all.
        self._prompts = {
            story: PROMPT.format(scale=scale_text, story_context=story_context_for(story),
                                 context_rule=context_rule_for(),
                                 examples_block=sentiment_examples_block())
            for story in STORY_CONTEXT
        }
        self._fallback_prompt = PROMPT.format(scale=scale_text, story_context=NO_STORY_CONTEXT,
                                              context_rule=context_rule_for(has_rules=False),
                                              examples_block=sentiment_examples_block())
        # Warned about once per unrecognised value, not once per comment --
        # a mistyped/renamed `story` (`--story` on the collector is free-text
        # with no `choices=` restriction) used to fall back to
        # NO_STORY_CONTEXT with nothing in the run log to say so, silently
        # degrading every sentiment rating for that story. Not live on the
        # real four-story corpus (checked: its `story` values match
        # STORY_CONTEXT's keys exactly) but a future collection run (e.g. a
        # Shorts re-collection, or a fifth story) could hit it with no warning.
        self._warned_stories = set()

    _VALID_RATINGS = (-1, 0, 1, 2, 3, 4, 5)

    def score(self, body, story=None):
        if story not in self._prompts and story not in self._warned_stories:
            self._warned_stories.add(story)
            print(f"WARNING: story {story!r} is not one of the {len(STORY_CONTEXT)} "
                  f"recognised stories ({', '.join(sorted(STORY_CONTEXT))}) -- this comment, "
                  f"and any other with the same story value, gets no factual context, which "
                  f"silently weakens its sentiment rating (see STORY_CONTEXT's docstring for "
                  f"why that context matters for a terse, deictic comment). Check for a typo "
                  f"in --story at collection time. Reported once per story value, not once per "
                  f"comment.", file=sys.stderr)
        prompt = self._prompts.get(story, self._fallback_prompt)
        ratings = []
        for vote in range(self.votes):
            reply = self._judge.sample(body, prompt, vote)
            # A model occasionally answers with a Unicode minus sign (U+2212,
            # "−1") rather than an ASCII hyphen; normalised first so it
            # is not misread as the digit 1 with the sign silently dropped.
            reply = (reply or "").replace("−", "-")
            # Match the whole number token, not a single character -- the old
            # r"[1-5]" pattern would misread a "-1" reply as "1" (the digit
            # sits right there in the string) and would also accept the "1"
            # inside a stray "15", rather than rejecting the whole token as
            # invalid. Validating against _VALID_RATINGS after extracting the
            # full number closes both.
            match = re.search(r"-?\d+", reply)
            # An unreadable vote is kept as a failed ballot (None), not skipped: see
            # resolve_ratings.
            ratings.append(int(match.group())
                           if match and int(match.group()) in self._VALID_RATINGS else None)
        return resolve_ratings(ratings)

    def cost_usd(self):
        return self._judge.cost_usd()

    def cache_savings_usd(self):
        return self._judge.cache_savings_usd()

    def cache_tokens(self):
        return (self._judge.usage["cache_creation_input_tokens"],
                self._judge.usage["cache_read_input_tokens"])

    @property
    def seconds(self):
        """Delegates to the inner judge's own running total, read live (a
        property, not a value captured once at __init__) so a before/after
        delta around a batch of scoring calls -- see evaluate.sentiment_report,
        which needs this the same way run_arm already uses judge.seconds --
        actually measures that batch."""
        return self._judge.seconds

    def token_usage(self):
        return (self._judge.usage["input_tokens"], self._judge.usage["output_tokens"])

    def describe(self):
        return f"llm:{self._judge.spec}{self._config_tag}"


@register("stub")
class StubScorer(Scorer):
    """Always neutral. For exercising the harness with no model."""

    deterministic = True

    def __init__(self, _unused=None, **_ignored):
        pass

    def score(self, body, story=None):
        return {"label": "neutral", "value": 0.0, "raw": None, "unstable": False}
