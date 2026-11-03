"""One model call that returns both the harm domain and the sentiment.

    python classify.py master.csv --judge combined:anthropic:claude-haiku-4-5 \
        --sentiment combined:anthropic:claude-haiku-4-5 --cache cache.jsonl

    python evaluate.py scorecard heldout.csv --pool pool.csv \
        --arm combined:anthropic:claude-haiku-4-5/retrieved

Labelling a comment currently costs six API calls: three votes for the harm
domain, then three more for sentiment. Both read the same comment; they just
ask different questions about it. Asking once for both halves the calls, and
with them the time and the cost -- on the measured rate that takes 10,000
comments from ~3.2 hours and ~$38 to roughly ~1.6 hours and ~$19.

**Whether it is as accurate is an open question, and the point of this module
is to make it testable rather than to assume it.** Joint prompting sometimes
degrades both tasks relative to asking separately: the model has two things to
hold at once and one answer to produce. It may also couple the two outputs --
a comment read as angry being pushed toward `democracy` because the examples
correlate that way -- which separate calls cannot do. Add it as a scorecard arm
and keep it only if `evaluate.py scorecard` and `evaluate.py sentiment` show no
loss against the arms already there.

The two halves are wired to one instance per spec (see `shared`), so `--judge`
and `--sentiment` naming the same spec make one set of calls rather than two.
Without that they would be separate objects and the saving would vanish.
"""
import re
import sys
import threading

from ewl.label import judges
from ewl.label import sentiment
from ewl import taxonomy

SCALE = sentiment.SCALE

# REVISED alongside sentiment.py's stance rewrite (see its module docstring):
# this template used to ask for "a single digit for the feeling the comment
# EXPRESSES" (the old emotion framing) and its domain half carried its own
# hand-copied rules with no boundary-cases section -- both silently drifted
# from every other judge's prompt, which is built through taxonomy.py so that
# "an arm that scores better cannot be doing so because it was quietly given
# a better prompt" (judges.py's own stated invariant). Now built from the same
# taxonomy.CATEGORY_DEFINITIONS_TEXT / taxonomy.BOUNDARY_CASES every standalone
# domain judge uses, and the same STANCE framing / story context / -1/0
# sentinels sentiment.py's LLMScorer uses, so the joint call answers the exact
# same two questions as the two separate calls it replaces.
_TEMPLATE = """You are labelling public comments about extreme wealth for a research study.

{top_context}
For each comment, give TWO things.

1. HARM DOMAIN -- one or two of these categories, or answer "none" if it genuinely
   engages none of them:
{definitions}

Rules for the domain:
- Classify what the COMMENT is about, not the news story it sits under.
- Use exactly the category names above. Never invent a category such as "policy".
- "None" is a normal, expected answer, not a last resort. Jokes, greetings, one-word
  replies, spam, and anything genuinely unrelated to these four harms should all be
  "none" -- do not force a comment into a category it does not fit.

{boundary_cases}

2. SENTIMENT -- the comment's STANCE on extreme wealth, not the emotional tone of its
   wording and not whether you personally agree with it:
{sentiment_context}{scale}

Rules for the sentiment:
- Use -1 only when the comment itself cannot be judged -- a language you cannot read,
  or genuinely incomprehensible text. Do not use -1 just because a comment is unusual,
  rude, or hard to categorise.
- Use 0 when the comment is comprehensible but takes no position on extreme wealth, whether
  or not it is about the video's subject: a reaction or joke, a reply to another commenter, a question, a practical or factual remark about the topic, a personal anecdote, a general political
  comment that is not about wealth, spam. Use 3 instead
  only when it engages with wealth, tax or accountability and is balanced or picks no side.
- Otherwise rate 1-5 on stance, not intensity of wording. Read sarcasm and irony as what
  they mean, not what they say -- "eat the rich" is very negative (1), not neutral.
{context_rule}
- Judge the sentiment on its own. It does not have to agree with the domain.

Answer on ONE line, exactly: domain(s), then a pipe, then the rating -- a single number
from this list: -1, 0, 1, 2, 3, 4, 5. No explanation, no markdown. For example:
  democracy | 2
  economics, social | 4
  none | 0
  none | -1

{examples_heading}
{examples}"""


# Where the story context sits. "keep" (the default) puts it at the top, so the
# DOMAIN question sees it too, which every other arm's domain prompt deliberately
# does not. "remove" moves it down to the sentiment half, so the domain half is
# comparable with the standalone arms. The default reproduces the old text exactly.
_DOMAIN_CONTEXT = {"mode": "keep"}


def configure(domain_context=None):
    """`domain_context`: "keep" or "remove". Returns the active setting."""
    if domain_context is not None:
        if domain_context not in ("keep", "remove"):
            raise ValueError(f"domain_context must be 'keep' or 'remove', not {domain_context!r}")
        _DOMAIN_CONTEXT["mode"] = domain_context
    return _DOMAIN_CONTEXT["mode"]


def config_tag():
    return "" if _DOMAIN_CONTEXT["mode"] == "keep" else "[domain_context=removed]"


def build_prompt(examples, story=None):
    """Same (stable instructions, per-comment examples) split every other
    judge's prompt follows, plus one more per-story variable dimension:
    `story` picks which of sentiment.STORY_CONTEXT's four factual paragraphs
    is shown, exactly as sentiment.LLMScorer does, since this prompt answers
    the same sentiment question. An unrecognised/missing story falls back to
    sentiment.NO_STORY_CONTEXT, the same fallback LLMScorer uses."""
    rated = {text: (rating, ex_story) for ex_story, text, rating in sentiment.example_ratings()}
    if rated and all(text in rated for text, _label in examples):
        # Real hand-coded ratings exist for every example shown: show each as it must be answered,
        # "domain | rating", with the story it came from (the stance guidance depends on it).
        heading = ("Examples (domain | rating; the story each was posted under is in brackets, "
                   "and the guidance above is for this comment's story only):")
        shown = "\n".join(f'[{rated[t][1]}] "{t}" -> {label} | {rated[t][0]}'
                          for t, label in examples)
    else:
        heading = "Examples of the domain judgement:"
        shown = "\n".join(f'"{t}" -> {label}' for t, label in examples)
    if _DOMAIN_CONTEXT["mode"] == "keep":
        # Both halves see the story's FACTS (the hand coders knew the video for both
        # judgements); the stance rules go only to the sentiment half.
        if story in sentiment.STORY_CONTEXT:
            facts = sentiment.facts_only(story)
            rules = sentiment.rules_only(story) if sentiment._CONFIG["story_rules"] else ""
        else:
            facts, rules = sentiment.NO_STORY_CONTEXT, ""
        top_context = facts + "\n"
        sentiment_context = ("Guidance on reading stance for this story: " + rules + "\n") if rules else ""
    else:
        top_context, sentiment_context = "", sentiment.story_context_for(story) + "\n"
    return _TEMPLATE.format(
        top_context=top_context,
        sentiment_context=sentiment_context,
        context_rule=sentiment.context_rule_for(has_rules=(
            sentiment._CONFIG["story_rules"] and story in sentiment.STORY_CONTEXT), short=True),
        definitions=taxonomy.CATEGORY_DEFINITIONS_TEXT,
        boundary_cases=taxonomy.BOUNDARY_CASES,
        scale="\n".join(f"  {n} = {t}" for n, t in sentiment.scale_for().items()),
        examples_heading=heading, examples=shown)


_VALID_RATINGS = (-1, 0, 1, 2, 3, 4, 5)


def split_reply(text):
    """Pull (domain part, rating) out of one reply.

    Tolerant on purpose: a reply that loses the pipe, or answers on two lines,
    still yields whatever it did say rather than discarding both halves. The
    rating is the last standalone token in `_VALID_RATINGS`, so a "the 1%"
    inside a domain name cannot be mistaken for it -- and matching the whole
    numeric token (not a `[1-5]` character class, the previous pattern) closes
    the bug that pattern had: it read a "-1" reply ("not codable") as plain
    "1" ("very negative"), the opposite end of the scale, because the digit
    sits right there in the string; and it silently dropped "0" ("irrelevant")
    entirely, since 0 was never in the character class at all.
    """
    # A model occasionally answers with a Unicode minus sign (U+2212, "−"
    # rather than "-"); normalised first so a "not codable" -1 is not misread
    # as the digit 1 with the sign silently dropped.
    text = (text or "").strip().replace("−", "-")
    domain_part, _, rating_part = text.partition("|")
    if not rating_part:
        # No pipe: take a trailing number if there is one and it's valid,
        # else treat it all as the domain and let the rating come back missing.
        match = re.search(r"-?\d+\s*$", text)
        if match and int(match.group()) in _VALID_RATINGS:
            return text[:match.start()], int(match.group())
        return text, None
    numbers = [int(token) for token in re.findall(r"-?\d+", rating_part)]
    valid = [n for n in numbers if n in _VALID_RATINGS]
    return domain_part, valid[-1] if valid else None


@judges.register("combined")
class CombinedJudge(judges.Judge):
    """A judge whose single reply carries the sentiment too."""

    deterministic = False
    default_model = "anthropic:claude-haiku-4-5"

    @classmethod
    def _share_key(cls, model, kwargs):
        """What has to match for two callers to be given the same object.

        The spec alone is not enough. `__init__` early-returns on an already
        built instance, so anything the second caller asked for was silently
        thrown away: `create('combined:X', votes=3)` after
        `create('combined:X', votes=1)` returned the 1-vote judge and ran one
        vote while reporting three. Every setting that changes behaviour is
        therefore part of the key, and differing settings now get their own
        instance rather than quietly inheriting the first caller's.
        """
        return (model or cls.default_model,
                max(1, kwargs.get("votes", 3)),
                kwargs.get("cache_path"),
                bool(kwargs.get("fuzzy", False)))

    def __new__(cls, model=None, **kwargs):
        """One instance per (spec, settings), per process.

        `--judge combined:X` and `--sentiment combined:X` must reach the *same*
        object or each makes its own calls and the halving does not happen. Done
        here rather than in a factory so it holds however the object is reached
        -- judges.create(), sentiment.create(), or directly. Both callers are
        passed the same votes/cache/fuzzy by classify.py, so they still share.
        """
        key = cls._share_key(model, kwargs)
        with _shared_lock:
            instance = _shared.get(key)
            if instance is None:
                instance = super().__new__(cls)
                instance._ready = False
                _shared[key] = instance
            return instance

    def __init__(self, model=None, **kwargs):
        if getattr(self, "_ready", False):
            return
        spec = model or self.default_model
        # The inner judge owns the cache. Every request routes through
        # inner.sample(), so a cache here would never be read -- it would only
        # load a second full in-memory copy of the same file at startup, which
        # for a 30,000-comment run is 90,000 lines held twice for nothing, and
        # a hazard the day something does read it.
        super().__init__(spec, **{k: v for k, v in kwargs.items() if k != "cache_path"})
        # The backend that does the talking. Its own votes are forced to 1:
        # this class owns the voting, and letting the inner judge vote as well
        # would multiply the calls back up and undo the whole point.
        inner_kwargs = dict(kwargs)
        inner_kwargs["votes"] = 1
        # The inner judge owns the cache. This class routes every request
        # through inner.sample(), so a cache here would never be read -- it
        # would only load a second full in-memory copy of the same file at
        # startup, which for a 30,000-comment run is 90,000 lines held twice
        # for nothing, and a hazard the day something does read it.
        self._inner = judges.create(spec, **inner_kwargs)
        self.deterministic = self._inner.deterministic
        # One reply per (body, vote, prompt) shared between the domain and
        # sentiment halves, so asking for both really is one call and not two.
        # Bounded, not corpus-sized: classify.py's label_row calls label()
        # then sentiment_for() back-to-back for the same body, and
        # sentiment_for() -- the second, closing half of that pair -- forgets
        # this body's entries once it has read them (see `_forget`). Without
        # that, both this dict and `_examples_by_body` below grow for the
        # life of the process: measured at roughly 1GB resident for a
        # combined-arm run over the real ~84,000-comment corpus, since
        # nothing was ever removing an entry once written.
        self._replies = {}
        # The examples `label()` was given for a body, so `sentiment_for()`
        # can build the exact same prompt instead of falling back to a fixed
        # set -- see `sentiment_for`'s docstring for why that fallback was
        # itself a bug under a non-fixed selector.
        self._examples_by_body = {}
        self._memo_lock = threading.Lock()
        # Same reasoning as sentiment.LLMScorer's own `_warned_stories`: an
        # unrecognised `story` silently falls back to no context otherwise,
        # with nothing in the run log to say so.
        self._warned_stories = set()
        self._ready = True

    @property
    def spec(self):
        return f"combined:{self._inner.spec}"

    def _reply(self, body, examples, vote, story=None):
        if story not in sentiment.STORY_CONTEXT and story not in self._warned_stories:
            self._warned_stories.add(story)
            print(f"WARNING: story {story!r} is not one of the "
                  f"{len(sentiment.STORY_CONTEXT)} recognised stories "
                  f"({', '.join(sorted(sentiment.STORY_CONTEXT))}) -- this comment, and any "
                  f"other with the same story value, gets no factual context, which silently "
                  f"weakens its sentiment rating. Check for a typo in --story at collection "
                  f"time. Reported once per story value, not once per comment.",
                  file=sys.stderr)
        # The prompt is part of the key, not just the comment. Without it, two
        # scorecard arms sharing a spec but differing in examples --
        # combined:X/fixed vs combined:X/retrieved, exactly the comparison this
        # module exists to enable -- silently served each other's replies, so
        # the comparison was guaranteed to come back "no difference" and the
        # second arm reported zero calls. `story` is folded into the prompt
        # itself (see build_prompt), so it does not need to be a separate key
        # element -- two different stories already produce two different
        # prompt strings.
        prompt = build_prompt(examples, story)
        key = (body, vote, prompt)
        with self._memo_lock:
            if key in self._replies:
                return self._replies[key]
        text = self._inner.sample(body, prompt, vote)
        with self._memo_lock:
            self._replies[key] = text
            self._trim(self._replies)
        return text

    # Upper bound on memoised entries. The pairing label()-then-sentiment_for()
    # normally empties the memo comment by comment; this bound is for every other
    # way of using the judge (domain only, or sentiment scored separately), where
    # nothing ever forgot an entry and the memo grew with the corpus -- about 1 GB
    # resident over 84,000 comments. Evicting the oldest costs at worst one repeat
    # call (served from the on-disk response cache when there is one).
    _memo_limit = 5000

    def _trim(self, memo):
        while len(memo) > self._memo_limit:
            memo.pop(next(iter(memo)))

    def _forget(self, body, examples=None, story=None):
        """Drop what was memoised for this exact (body, examples, story), once both
        halves have read it. Exact on purpose: the same body can be in flight under
        another story (a short comment like "yes" is posted under all four), and
        dropping by body alone took that other call's entries with it.

        Called from `sentiment_for` -- the second, closing half of the
        label()-then-sentiment_for() pair classify.py always calls in that
        order for one comment (see label_row there) -- not from `label`
        itself, which runs first and still needs its replies to survive
        until the sentiment half asks for them.
        """
        with self._memo_lock:
            if examples is not None:
                prompt = build_prompt(examples, story)
                for vote in range(self.votes):
                    self._replies.pop((body, vote, prompt), None)
            self._examples_by_body.pop((body, story), None)

    def label(self, body, examples, story=None):
        # Recorded so `sentiment_for` can ask the SAME question with the SAME
        # examples if it is called next for this body -- see that method's
        # docstring for why using a different examples set breaks the memo
        # and quietly doubles the call count.
        with self._memo_lock:
            self._examples_by_body[(body, story)] = examples
            self._trim(self._examples_by_body)
        ballots = []
        for vote in range(self.votes):
            domain_part, _rating = split_reply(self._reply(body, examples, vote, story))
            ballots.append(taxonomy.parse_labels(domain_part, self.fuzzy))
        labels, unstable = self.resolve(ballots)
        return labels, unstable, ballots

    def sentiment_for(self, body, examples=None, story=None):
        """The other half of the same replies.

        `examples` defaults to whatever `label()` was last given for this
        exact `body` (see `_examples_by_body`), not a fixed set. It used to
        always fall back to `taxonomy.FIXED_EXAMPLES` regardless of what
        `label()` had actually used -- fine under `--selector fixed`, where
        both halves see the same five examples anyway, but wrong under any
        other selector (`retrieved`, `fixed_random`): the domain call's
        prompt embeds per-comment retrieved examples, the sentiment call's
        prompt embedded the fixed five, `_reply`'s memo key includes the
        prompt, so the two prompts differed and the memo missed -- silently
        making a second real API call for every single comment, exactly the
        two-calls-per-comment cost this module exists to halve. Falls back to
        `taxonomy.FIXED_EXAMPLES` only when nothing was recorded (sentiment
        scored standalone, with no preceding domain pass on this body).

        Resolution shared with sentiment.LLMScorer.score via
        sentiment.resolve_ratings -- one implementation of "-1/0 are
        categorical outs, resolved by majority KIND, not medianed together
        with a real stance" and "no majority kind is genuinely undecided,
        not an arbitrary pick", rather than two copies that can drift apart
        (the old version here medianed every returned digit as if it were
        always a 1-5 stance, and broke a 3-way kind tie by whichever vote
        came back first).
        """
        if examples is None:
            with self._memo_lock:
                examples = self._examples_by_body.get((body, story), taxonomy.FIXED_EXAMPLES)
        ratings = []
        for vote in range(self.votes):
            _domain, rating = split_reply(self._reply(body, examples, vote, story))
            # A reply with no readable rating is a failed ballot (None), not a skipped one: see
            # sentiment.resolve_ratings.
            ratings.append(rating)
        result = sentiment.resolve_ratings(ratings)
        self._forget(body, examples, story)
        return result

    # Accounting lives on the inner judge, which is what actually spends money.
    def cost_usd(self):
        return self._inner.cost_usd()

    @property
    def usage(self):
        return self._inner.usage

    @usage.setter
    def usage(self, value):
        pass

    def cache_savings_usd(self):
        # Priced at the inner judge's rates, not this wrapper's (which has none): reading the
        # wrapper's own price table printed "$0.00 saved" beside hundreds of thousands of
        # cache-read tokens.
        return self._inner.cache_savings_usd()

    def thinking_setting(self):
        """What the inner judge sent for thinking, for the run record (None if it sends none)."""
        inner = getattr(self._inner, "thinking_setting", None)
        return inner() if inner else None

    @property
    def latencies(self):
        """The inner judge's live-call durations, which the speed report is built from. The
        wrapper makes no calls itself, so its own list was always empty and the report said
        "every reply came from the cache" on a run that had spent money."""
        existing = getattr(self._inner, "latencies", None)
        return existing if existing is not None else self._inner.__dict__.setdefault("latencies", [])

    @property
    def seconds(self):
        return self._inner.seconds

    @seconds.setter
    def seconds(self, value):
        pass

    def _sample(self, body, prompt, vote):
        return self._inner.sample(body, prompt, vote)


_shared = {}
_shared_lock = threading.Lock()


def shared(spec, **kwargs):
    """The one CombinedJudge for this spec (see CombinedJudge.__new__)."""
    return CombinedJudge(spec, **kwargs)


@sentiment.register("combined")
class CombinedScorer(sentiment.Scorer):
    """The sentiment half, reading the shared judge's replies."""

    def __init__(self, judge_spec=None, votes=3, cache_path=None, fuzzy=False, **_ignored):
        # `fuzzy` is forwarded even though the sentiment half never reads it:
        # it is part of the sharing key, so dropping it here would hand the
        # scorer a *different* instance from the judge under --fuzzy, and the
        # one-call-serves-both saving this module exists for would be lost.
        self._judge = shared(judge_spec, votes=votes, cache_path=cache_path, fuzzy=fuzzy)
        self.deterministic = self._judge.deterministic
        self.votes = self._judge.votes

    def score(self, body, story=None):
        # No examples passed: classify.py's generic scorer.score(body, story)
        # signature has nowhere to carry them, so `sentiment_for` looks up
        # whatever `label()` (the domain half) was actually given for this
        # body instead -- see its docstring for why hard-coding
        # `taxonomy.FIXED_EXAMPLES` here silently doubled the call count under
        # any selector other than `fixed`. `story` is real, not optional:
        # classify.py and evaluate.py always pass it (every other scorer's
        # `.score(body, story=...)` accepts it too), and without it every
        # combined sentiment call crashed outright.
        return self._judge.sentiment_for(body, story=story)

    def cost_usd(self):
        return self._judge.cost_usd()

    # The tokens are spent by the one shared judge, so the sentiment table's token columns (and
    # any cache line that asks the scorer) read from it. Without these the scorer reported
    # "0 tokens" next to a real cost.
    def token_usage(self):
        return self._judge.token_usage()

    def cache_tokens(self):
        return self._judge.cache_tokens()

    def cache_savings_usd(self):
        return self._judge.cache_savings_usd()

    def describe(self):
        return f"combined:{self._judge._inner.spec}"
