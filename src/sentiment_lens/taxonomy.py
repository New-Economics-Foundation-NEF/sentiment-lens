"""The harm taxonomy, shared by every classifier and every judge.

Both `classify.py` and `keyword_classify.py` previously defined their own copy
of the category list. They agreed, but nothing enforced that they agreed, and a
category added to one would have silently skewed any comparison between them.
One definition, imported everywhere, so a change to the taxonomy cannot leave
half the pipeline behind.

The definitions are not decoration: they are the text the model actually sees,
and the pilot showed that abstract one-liners were not enough to pull
`democracy` out of comments about lobbying, donations and tax.
"""

# Condensed from the seven domains in "Risks of Extreme Wealth".
CATEGORIES = {
    "democracy": (
        "capture of public institutions by the rich: political donations and lobbying, "
        "tax avoidance and offshore arrangements, media ownership and control of the "
        "narrative, courts and police treating the wealthy differently, oligarchy"
    ),
    "economics": (
        "markets, monopoly and competition, wages and the cost of living, corporate "
        "power and profits, prices, financial crises, who owns what share of the economy, "
        "hollowing out of public services, government bailing out large private firms"
    ),
    "environment": (
        "emissions and carbon footprints, private jets, yachts and luxury consumption, "
        "fossil fuel financing, climate and ecological damage"
    ),
    "social": (
        "fairness, merit and desert, resentment and division between rich and ordinary "
        "people, health, housing, education, inequality of gender and race, societal collapse"
    ),
}

CATEGORY_NAMES = list(CATEGORIES)
UNCLASSIFIED = "unclassified"
# Distinct from UNCLASSIFIED: UNCLASSIFIED now means "the model explicitly and
# cleanly said none of the categories apply" -- a genuine, informative answer.
# PARSE_FAILURE means the reply could not be read as a valid answer at all
# (empty, garbled, off-format, a safety refusal) -- a classifier error, not a
# finding about the comment. Before this, both collapsed to the same label,
# which was a rounding error while "unclassified" was a small share of the
# corpus but became a real problem once it turned out to be 50-64% of it: a
# hidden parse-failure rate inside that number would make "most comments
# aren't about these harms" partly an artefact of the classifier rather than
# a fact about the comments.
PARSE_FAILURE = "parse_failure"

# Used when no retrieval pool is available: the `fixed` selector, i.e. the same
# 25 examples for every comment in a run. They anchor the boundary the pilot got
# wrong (a comment about tax or media control under an environment story is
# democracy, not environment) and, deliberately, show "none" as a normal answer
# -- most of the real corpus is none, and the earlier five-example set contained
# no "none" example at all.
#
# PROVENANCE, read before changing or trusting these: the first five are the
# original hand-written anchors; the other twenty were written by an AI
# assistant (Claude Sonnet 5.5, 2026-09-30) as placeholders and labelled by it
# against the definitions and boundary cases above. They are synthetic. NONE is
# a real comment, and none was taken from the corpus or the gold set, on
# purpose: a held-out gold comment that also appears in the prompt would be
# shown its own answer (the leakage `classify.warn_if_pool_overlaps` guards
# against for retrieved examples), and real comment text has no place in
# committed code. They have NOT been reviewed by a human coder. Review them, and
# replace any you disagree with, before the main run -- and disclose the
# AI-drafted examples in the methodology.
#
# Shape of the set: 5 democracy, 4 economics, 3 environment, 4 social, 2 with
# two labels, 7 none. Categories are interleaved rather than grouped, so no
# label sits in a run that a model could pick up as a pattern.
FIXED_EXAMPLES = [
    ("He bought the platform so he could control what people are allowed to say about him.", "democracy"),
    ("First!", "none"),
    ("One jet trip puts out more carbon than I will in my whole life.", "environment"),
    ("They posted record profits and still froze our wages.", "economics"),
    ("Nobody works a billion dollars hard. Nobody.", "social"),
    ("She paid a lower tax rate than her cleaner and it's completely legal. That's the problem.", "democracy"),
    ("Great video, just subscribed.", "none"),
    ("One company crushes every small shop and then buys whatever is left. That is not competition.", "economics"),
    ("A yacht that size burns thousands of litres of fuel just sitting in the harbour.", "environment"),
    ("He didn't earn that. His family handed him a head start and he got lucky.", "social"),
    ("The judge gave him a slap on the wrist. Anyone else would have got years.", "democracy"),
    ("Private jets for the ultra-rich while the rest of us are told to cut back on flights to save the planet.", "environment, social"),
    ("Anyone else watching this in 2025?", "none"),
    ("Jets are fine, my problem is that he pays next to nothing in tax on them.", "democracy"),
    ("Bailing out billion-dollar companies while public services get cut is completely backwards.", "economics"),
    ("Meanwhile kids in my town go to school hungry.", "social"),
    ("Say what you want, the man gets things done. Respect.", "none"),
    ("The leaked IRS files show billionaires paying a true tax rate of almost nothing. Where is the enforcement?", "democracy"),
    ("Funding new oil and gas projects while telling us to recycle. Unreal.", "environment"),
    ("lol this comment section is wild 😂", "none"),
    ("They lobby to gut environmental rules so they can keep polluting for free.", "democracy, environment"),
    ("Prices keep going up and the company keeps posting record profits. Funny how that works.", "economics"),
    ("It's us against them and they know it. Society is falling apart because of people like this.", "social"),
    ("Does anyone know what song is playing at 3:20?", "none"),
    ("I wonder how many billionaires actually watch this channel.", "none"),
]

# Where the fixed examples came from. The built-in set is synthetic. When real gold
# comments are used instead (`--fixed-examples FILE`), only their ids and a hash are ever
# recorded, never their text: real comment text does not belong in committed code,
# run records or the published codebook.
FIXED_EXAMPLES_SOURCE = {"source": "built-in (synthetic)"}

# The hand-coded stance rating (-1 to 5) and the story of each fixed example, aligned with
# FIXED_EXAMPLES, when they were loaded from a file that carries them. Empty for the built-in
# set, which has no ratings: the sentiment prompt shows examples only from real, hand-coded
# ratings, never invented ones.
FIXED_EXAMPLE_RATINGS = []
FIXED_EXAMPLE_STORIES = []
EXAMPLE_STORIES = ("luigi", "musk", "jets", "propublica")
VALID_RATINGS = (-1, 0, 1, 2, 3, 4, 5)


def load_fixed_examples(path):
    """Replace FIXED_EXAMPLES with the examples in a CSV, and return their comment_ids.

        comment_id,body,label                       domain examples only
        comment_id,body,label,sentiment,story       domain AND sentiment examples

    The label may also be in a `hand_domain` column and the rating in `hand_sentiment`.
    When the rating is present it must be on EVERY row, as a number from -1 to 5 (words are
    refused), and so must the `story` the comment was posted under: the stance guidance
    differs by story ("condemning the killing" is supportive for Luigi, "defending free
    speech" for Musk), so an example shown without its story could teach the wrong reading.
    The sentiment prompt and the combined arm show the ratings; the domain prompt shows only
    the labels. Without ratings, no sentiment examples are shown.

    Meant for real, hand-coded gold comments. Keep the file in a gitignored folder
    (`official-data/`): it holds comment text. The comments are shown to the model in
    the prompt for every comment, so they must be left out of scoring (each would be
    scored against its own label); `slens-evaluate` does that automatically from the ids
    returned here, and `slens-validate score --exclude-ids FILE` and
    `slens-report --exclude-ids FILE` accept the same file.

    Refuses a row with no id, no text, a duplicate, or a label that is not a clean answer
    (category names or "none"). Warns when the count is not 25.
    """
    import csv
    import hashlib
    import json
    import sys

    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No rows in {path}.")
    pairs, ids, labels, ratings, stories = [], [], [], [], []
    seen_ids, seen_bodies = set(), set()
    for n, row in enumerate(rows, 2):
        cid = (row.get("comment_id") or "").strip()
        body = (row.get("body") or "").strip()
        label = (row.get("label") or row.get("hand_domain") or "").strip()
        if not cid or not body or not label:
            raise SystemExit(f"{path} row {n}: needs comment_id, body and a label "
                             f"(`label` or `hand_domain`).")
        if cid in seen_ids or body.lower() in seen_bodies:
            raise SystemExit(f"{path} row {n}: a duplicate example ({cid}).")
        if PARSE_FAILURE in parse_labels(label):
            raise SystemExit(f"{path} row {n}: {label!r} is not a clean answer: use one or "
                             f"two of {', '.join(CATEGORY_NAMES)}, or none.")
        raw_rating = (row.get("sentiment") or row.get("hand_sentiment") or "").strip()
        rating = None
        if raw_rating:
            try:
                number = float(raw_rating)
            except ValueError:
                number = None
            if number is None or number != int(number) or int(number) not in VALID_RATINGS:
                raise SystemExit(f"{path} row {n}: the sentiment {raw_rating!r} must be a number "
                                 f"from -1 to 5 (not a word such as 'negative').")
            rating = int(number)
        seen_ids.add(cid)
        seen_bodies.add(body.lower())
        pairs.append((body, label.lower()))
        ids.append(cid)
        labels.append(label.lower())
        ratings.append(rating)
        stories.append((row.get("story") or "").strip().lower())
    rated = [r is not None for r in ratings]
    if any(rated) and not all(rated):
        raise SystemExit(f"{path}: {rated.count(False)} example(s) have no sentiment. Give every "
                         f"example a rating, or none of them.")
    if all(rated):
        bad = [(n, s) for n, s in enumerate(stories, 2) if s not in EXAMPLE_STORIES]
        if bad:
            raise SystemExit(f"{path} row {bad[0][0]}: a sentiment example needs its `story` "
                             f"(one of {', '.join(EXAMPLE_STORIES)}), not {bad[0][1]!r}: the stance "
                             f"guidance depends on the story.")
    else:
        ratings, stories = [], []
    if len(pairs) != 25:
        print(f"NOTE: {len(pairs)} fixed examples loaded from {path}; the design uses 25.",
              file=sys.stderr)
    material = [list(item) for item in zip(ids, [b for b, _ in pairs], labels)]
    if ratings:
        material = [m + [r, s] for m, r, s in zip(material, ratings, stories)]
    digest = hashlib.sha256(json.dumps(material, ensure_ascii=False).encode("utf-8")).hexdigest()
    FIXED_EXAMPLES[:] = pairs          # in place, so every module that imported the list sees it
    FIXED_EXAMPLE_RATINGS[:] = ratings
    FIXED_EXAMPLE_STORIES[:] = stories
    FIXED_EXAMPLES_SOURCE.clear()
    source = {"source": "file", "path": path, "n": len(pairs),
              "ids": ids, "labels": labels, "sha256": digest}
    if ratings:
        source.update({"ratings": ratings, "stories": stories})
    FIXED_EXAMPLES_SOURCE.update(source)
    return ids


def reset_fixed_examples(original):
    """Put the built-in examples back (used by the harness)."""
    FIXED_EXAMPLES[:] = original
    FIXED_EXAMPLE_RATINGS[:] = []
    FIXED_EXAMPLE_STORIES[:] = []
    FIXED_EXAMPLES_SOURCE.clear()
    FIXED_EXAMPLES_SOURCE.update({"source": "built-in (synthetic)"})


BOUNDARY_CASES = """Boundary cases (these are the mistakes a naive read most often makes):
- A comment that names a specific policy lever -- tax rate, a bailout, lobbying,
  campaign donations, a court ruling, media ownership -- is democracy or
  economics even when it appears under an environment or social story. The
  story a comment sits under is not evidence for what the comment itself is
  about.
- "The rich"/"billionaires"/"wealthy" appearing in a comment is not on its own
  evidence for "social". Ask what specifically is being said about them: a
  complaint about their carbon footprint is environment; about their tax rate
  or political donations is democracy; about their market power or wages paid
  to workers is economics. "Social" is for complaints about fairness, desert,
  and the gap between rich and ordinary people as such -- resentment, envy,
  "one rule for them", "they don't deserve it" -- not a general keyword match
  on wealth.
- A comment about a private jet, yacht, or other luxury purchase is
  environment when the complaint is about emissions or ecological harm, but
  economics when the complaint is about frivolous spending or market
  distortion with no environmental angle, and social when the complaint is
  about fairness/entitlement with no material harm named at all.
- A comment can genuinely belong to two categories when it raises two
  distinct harms in the same breath (e.g. tax avoidance and deregulation); it
  should get only one when a second category is merely implied by the first
  without being separately stated.
- When a comment could plausibly fit more than one category and you are
  genuinely unsure, pick the two categories that best fit rather than
  guessing at a single one -- the two-label limit exists so a genuinely
  dual-topic comment does not have to be forced into one bucket, not so an
  uncertain classifier can hedge on comments that really do have one clear
  reading.
- A comment that is purely about a person's character, a joke, a reply to
  another commenter, or a remark on the video's production quality is "none",
  even if it appears under one of these stories and even if a word like
  "billionaire" or "rich" appears in passing."""

# Split out from _STABLE_INSTRUCTIONS (not just decoration -- see BOUNDARY_CASES
# above) so combined.py's joint domain+sentiment prompt can reuse the exact
# same boundary-case text instead of keeping a second, hand-maintained copy
# that silently drifted from this one (confirmed: it had, until this split).
_STABLE_INSTRUCTIONS = """You are a precise text classifier for a study of public comments about extreme wealth.

Assign the comment to one or two of these categories, or answer "none" if it genuinely engages none of them:
{definitions}

Rules:
- Classify what the COMMENT is about, not the news story it sits under.
- Use exactly the category names above. Never invent a category such as "policy".
- "None" is a normal, expected answer, not a last resort. Jokes, greetings, one-word
  replies, spam, and anything genuinely unrelated to these four harms should all be
  "none" -- do not force a comment into a category it does not fit.
- Answer with a comma-separated list of at most two names, or the single word "none".
  No explanation, no markdown.

{boundary_cases}

Examples:
"""

# Also reused by combined.py, for the same reason BOUNDARY_CASES is exposed
# above: one definitions list, not a second copy that can drift from it.
CATEGORY_DEFINITIONS_TEXT = "\n".join(f"- {name}: {text}" for name, text in CATEGORIES.items())

# The definitions never change at runtime (CATEGORIES is a module-level
# constant), so this prefix is identical across every comment and every vote
# in an entire run -- not just across one comment's votes. Computed once here
# so a judge backend that supports prompt caching (see AnthropicJudge._sample)
# can mark exactly this prefix as its own cache breakpoint, separate from the
# per-comment retrieved examples that follow it. See build_prompt_parts.
STABLE_PREFIX = _STABLE_INSTRUCTIONS.format(
    definitions=CATEGORY_DEFINITIONS_TEXT, boundary_cases=BOUNDARY_CASES)


def build_prompt_parts(examples):
    """(stable, variable) -- the corpus-wide-constant prefix and the
    per-comment retrieved-examples suffix, as two separate strings.

    Split out so a caching-aware backend can mark them as two separate cache
    breakpoints: STABLE_PREFIX is identical for every comment in a run and
    benefits from being cached once, corpus-wide; the examples text is
    identical only across one comment's --votes calls (the selector is run
    once per comment, not once per vote -- see judges.Judge.label), so it
    benefits from a second, narrower cache scoped to that comment.
    """
    shown = "\n".join(f'"{text}" -> {label}' for text, label in examples)
    return STABLE_PREFIX, shown


# Does the domain question see the story context? Decided 2026-10-07: YES, the FACTS only.
# The hand coders knew which video a comment sat under when they assigned its harm domain,
# so the model is shown the same background. The stance rules ("condemning the killing
# counts as supportive") are about sentiment and stay out of the domain prompt. The
# background goes at the END of the prompt, after the examples, so the large shared part
# (instructions and the 25 examples) is identical for every story and a model that reuses
# a repeated prompt prefix (prompt caching, Ollama) can reuse it across stories.
# "none" reproduces the earlier prompt, which never varied by story.
DOMAIN_CONTEXT_MODES = ("facts", "none")
_DOMAIN_CONTEXT = {"mode": "facts"}

DOMAIN_CONTEXT_HEADING = (
    "\n\nBackground on the video the comment you are asked to classify was posted under (context only: the rules "
    "above still apply, and you classify what the COMMENT itself is about, not the story):\n")


def configure_domain_context(mode=None):
    """Set or read whether the domain prompt carries the story's factual background."""
    if mode is not None:
        if mode not in DOMAIN_CONTEXT_MODES:
            raise ValueError(f"domain context must be one of {DOMAIN_CONTEXT_MODES}, not {mode!r}")
        _DOMAIN_CONTEXT["mode"] = mode
    return _DOMAIN_CONTEXT["mode"]


def domain_context_tag():
    """"" at the default; otherwise a short tag naming the setting, for run records."""
    return "" if _DOMAIN_CONTEXT["mode"] == "facts" else "[domain_context=none]"


def domain_story_context(story):
    """The factual background shown to the domain question for `story`, or "" (switch off,
    or a story with no paragraph). Facts only: never the stance rules."""
    if _DOMAIN_CONTEXT["mode"] != "facts":
        return ""
    from sentiment_lens.label import sentiment  # imported here: sentiment imports this module
    if story not in sentiment.STORY_CONTEXT:
        return ""
    return sentiment.facts_only(story).strip()


def build_prompt(examples, story=None):
    """Assemble the system prompt from a list of (comment, label) pairs.

    Every judge builds its prompt through this function, so an arm that scores
    better cannot be doing so because it was quietly given a better prompt. `story`
    adds that story's factual background after the examples (see above); without a
    story, or with the switch off, the prompt is exactly the story-free one.
    """
    stable, shown = build_prompt_parts(examples)
    background = domain_story_context(story)
    return stable + shown + (DOMAIN_CONTEXT_HEADING + background if background else "")


def parse_labels(text, fuzzy=False):
    """Turn a model reply into a list of at most two valid category names.

    A category found in the reply is always returned. Anything off-list
    (a category name that isn't real, stray words, a garbled reply) sets
    `saw_garbage`, and an explicit "none" token sets `saw_none` -- these
    used to both collapse into the same UNCLASSIFIED result, which hid a
    real classifier-error rate inside what looked like a legitimate
    "nothing applies" finding (see PARSE_FAILURE's docstring above). Now:

      - any real category found -> those categories (as before)
      - no category, but the reply was empty or contained anything that
        wasn't a real category, a fuzzy-matched category, or "none"
        -> PARSE_FAILURE (the reply could not be read as a valid answer)
      - no category, and the reply was cleanly and only "none"
        -> UNCLASSIFIED (a genuine, informative answer)
    """
    raw = (text or "").strip()
    if not raw:
        return [PARSE_FAILURE]

    labels = []
    saw_none = False
    saw_garbage = False
    for part in raw.lower().replace("\n", ",").split(","):
        part = part.strip().strip('."-* ')
        if not part:
            continue
        if part == "none":
            saw_none = True
            continue
        if part in CATEGORIES:
            labels.append(part)
            continue
        if fuzzy:
            from difflib import get_close_matches
            match = get_close_matches(part, CATEGORY_NAMES, n=1, cutoff=0.6)
            if match:
                labels.append(match[0])
                continue
        saw_garbage = True

    seen, ordered = set(), []
    for label in labels:
        if label not in seen:
            seen.add(label)
            ordered.append(label)
    if ordered:
        return ordered[:2]
    # A reply mixing "none" with garbage ("none, xyz") is not a clean,
    # confident null answer -- treat it as a parse failure, not a genuine
    # none, so garbage never hides behind a "none" token.
    if saw_garbage:
        return [PARSE_FAILURE]
    if saw_none:
        return [UNCLASSIFIED]
    return [PARSE_FAILURE]
