# Judges and scorers

A **judge** assigns the harm domain. A **sentiment scorer** assigns the stance. A **selector**
decides which worked examples the judge sees in its prompt. The three are independent, so you
can change one and hold the others fixed. That is what makes two runs comparable.

```bash
ewl-classify corpus.csv --judge <judge spec> --sentiment <scorer spec> --selector <selector>
```

The defaults are `--judge ollama:gemma3:12b`, `--sentiment vader` and `--selector fixed`. Set
all three explicitly for any real run.

Every judge shares one prompt, one parser, one voting rule and one cache. A backend supplies
only the call that returns a raw reply, so a difference between two arms comes from the model
and not from how its answers were handled.

## Judge specs

A spec is `<backend>:<model>`. Everything after the first colon is the model name, so
`ollama:gemma3:12b` works.

| Spec | What it runs | Needs | Sends comments to |
|---|---|---|---|
| `stub` | Always answers `social`. For tests only. | nothing | nowhere |
| `keyword` | The keyword matcher (`src/ewl/keywords.yml`). Free, deterministic, one vote. `keyword:path/to/file.yml` uses another vocabulary. | nothing | nowhere |
| `manual` | Prints the prompt for you to paste into a chat window, then reads back the reply you paste. For a dry run with no API access. Use `--votes 1`. | nothing | wherever you paste it |
| `ollama:<model>` | Any model on a local Ollama server (default `gemma3:12b`). Sets no context window and runs no checks: do not use it for a real run. | `ollama` extra, Ollama | nowhere |
| `gemma` | Gemma 2 27B Instruct at Q4_0 through Ollama, and nothing else. Checks the server, the model and its quantisation before it starts, sets the context window, and runs one vote at temperature 0. | `ollama` extra, Ollama, `gemma2:27b` pulled | nowhere |
| `anthropic:<model>` | Claude (default `claude-haiku-4-5`). Caches the system prompt with Anthropic's prompt caching. | `anthropic` extra, `ANTHROPIC_API_KEY` | Anthropic |
| `openai:<model>` | GPT (default `gpt-5-mini`). Reasoning effort defaults to `minimal`. | `openai` extra, `OPENAI_API_KEY`, `openai` in `EWL_EXTERNAL_PROCESSORS` | OpenAI |
| `google:<model>` | Gemini (default `gemini-3.1-pro-preview`). Thinking cannot be turned off. The usage field names and cost figures have not been confirmed against a live call. | `google` extra, `GEMINI_API_KEY` or `GOOGLE_API_KEY`, `google` in `EWL_EXTERNAL_PROCESSORS` | Google |
| `combined:<judge spec>` | One call per comment that answers both questions, through the inner judge, for example `combined:anthropic:claude-haiku-4-5`. | as the inner judge | as the inner judge |

A cloud model must have a price, either in the built-in table or from `EWL_ANTHROPIC_PRICES`,
`EWL_OPENAI_PRICES` or `EWL_GOOGLE_PRICES` (see [Installation](installation.md#model-settings)).
An unpriced model, an unapproved provider or a setting the model cannot take is refused when
the judge is built, before any comment is sent.

For `ewl-gemma` and long local runs, see [Long runs and batches](batches.md).

## Sentiment scorer specs

| Spec | What it does |
|---|---|
| `vader` | VADER's compound score, mapped to negative/neutral/positive. It measures the tone of the words, not the stance toward wealth, and has no `irrelevant` or `uncodeable` answer. Use it only as a baseline; never quote it. |
| `llm:<judge spec>` | Asks a judge backend for a stance rating, for example `llm:anthropic:claude-haiku-4-5`, `llm:openai:gpt-5-mini` or `llm:gemma`. `llm` alone uses `anthropic:claude-haiku-4-5`. |
| `combined:<judge spec>` | Takes the stance from the same combined call as the domain. Pass the same spec to `--judge` and `--sentiment`; the two share one set of calls. |
| `stub` | A constant answer, for tests. |

### The stance question

An LLM scorer is asked for the comment's **stance toward extreme wealth** on this scale:

| Rating | Meaning | Label |
|---|---|---|
| `-1` | Not codable (another language, or unreadable) | `uncodeable` |
| `0` | No stance on extreme wealth to rate | `irrelevant` |
| `1`, `2` | Very negative, negative | `negative` |
| `3` | Engages with the topic but takes no side | `neutral` |
| `4`, `5` | Positive, very positive | `positive` |

The prompt carries a short factual paragraph about the comment's story, plus stance guidance
for that story. The [instrument switches](#instrument-switches) change these.

## Selectors

| Selector | What the judge sees | Flags |
|---|---|---|
| `fixed` | The same worked examples for every comment: the 25 built-in synthetic examples, or the file given with `--fixed-examples`. The default. `ewl-evaluate scorecard` refuses any other selector unless `--allow-non-fixed` is passed. | `--fixed-examples FILE` |
| `fixed_random` | `--k` examples drawn once at random from a hand-coded pool, then used for every comment. | `--pool FILE`, `--k N`, `--seed N` |
| `retrieved` | The `--k` nearest hand-coded comments to each comment, by sentence embedding (default 12). A comment never retrieves itself. Needs the `retrieval` extra. | `--pool FILE`, `--k N` |
| `none` | No examples; the category definitions only. | |

`--selector fixed` refuses `--pool` and `--k`, because it would ignore them.

!!! warning "A comment must never see its own answer"
    If a comment in the input is also in the retrieval pool or the examples file, it can be
    shown its own hand-coded label. `ewl-classify` warns when the input overlaps `--pool`.
    Examples from a `--fixed-examples` file are left out of scoring automatically (see
    [Validation](validation.md#worked-examples-are-left-out)).

### The fixed examples file

`--fixed-examples FILE` replaces the built-in examples with a CSV of real hand-coded comments:

```text
comment_id,body,label,sentiment,story
fx01,<comment text>,democracy,2,propublica
fx02,<comment text>,none,0,jets
fx03,<comment text>,"economics, social",1,luigi
```

- `comment_id`, `body` and `label` are required on every row. `label` (or `hand_domain`) is one
  or two category names, or `none`.
- `sentiment` (or `hand_sentiment`) is optional. If any row has one, every row must: a whole
  number from −1 to 5, not a word. Rated examples also need `story`, one of the four story
  labels, because the stance guidance differs by story.
- Without ratings, the stance prompt shows no examples. `ewl-gemma` and the `ewl-evaluate`
  sentiment and scorecard commands refuse an unrated file unless `--sentiment-examples off`.
- Ids and texts must be unique. The design uses 25 examples; another count prints a note.

The file holds comment text: keep it in your git-ignored data folder. Run records and the
dashboard codebook keep only its ids, labels and a hash.

## Instrument switches

These flags change the prompt wording. Each default reproduces the wording as it stands. A
non-default setting is tagged in the `sentiment_scorer` column and recorded in the run record,
and because the prompt is part of the cache key, answers are never shared between settings.
They are accepted by `ewl-classify`, `ewl-evaluate`, `ewl-gemma` and `ewl-anthropic-batch`.

| Flag | Default | Other value |
|---|---|---|
| `--story-rules {on,off}` | `on`: the story paragraph includes its stance guidance | `off`: facts only |
| `--stance-object {conduct,wealth}` | `conduct`: rating 4 means defending the wealthy people or companies involved, or how they gain or use their wealth | `wealth`: rating 4 is about extreme wealth and those who hold it |
| `--sentiment-examples {on,off}` | `on`: show the rated fixed examples in the stance prompt | `off`: no stance examples |
| `--domain-context {facts,none}` | `facts`: the domain prompt ends with the story's factual paragraph | `none`: no story background |
| `--combined-domain-context {keep,remove}` | `keep`: the combined arm's domain question sees the story facts | `remove`: only the stance question sees them |

Decide the instrument before a real run. Changing any of it means labelling everything again.

## Votes

`--votes N` (default 3) asks the model N times per comment and keeps what a majority agrees on.

- **Domain.** A label is kept when more than half the votes carry it, up to two labels. No
  majority gives `parse_failure`, never `unclassified`. `domain_unstable` is 1 when the votes
  differ.
- **Stance.** Votes are grouped by kind (a stance, `irrelevant`, `uncodeable`, unreadable).
  The largest kind wins; a tie between kinds is `no_consensus`. Two unreadable votes out of
  three is a parse failure. Within a stance, a side needs more than half the votes, and the
  value is the median rating.
- The `keyword` judge always runs one vote. The `gemma` judge runs one vote at temperature 0,
  because repeat votes would be identical.
- With one vote, stability is **not measured**: `domain_unstable` is 0 by construction, and the
  scorecard shows `—`. The cache, not voting, is then what makes the labels reproducible.

Voting makes labels reproducible. It does not make them correct.

## Concurrency

`--concurrency N` labels N comments at once. API runs spend most of their time waiting on the
network, so this is the biggest lever on wall-clock time. `ewl-classify` keeps about 4N comments
in flight and still writes rows in input order. Results match a serial run.

- Stay inside your provider's rate limits. Check its limits page before you raise N.
- Keep it at 1 for a local Ollama or Gemma judge, which is limited by your computer, not the
  network. `ewl-evaluate` always runs local arms one at a time.

## Cost cap

`ewl-classify --max-cost USD` stops starting new comments once this run's own live spend passes
USD. Replies from the cache cost nothing and do not count. Comments already in flight finish,
so the final figure can be a little over. The run writes what it has done, prints how to
resume, and exits with status 3. Its run record carries a `partial` block, so
`ewl-export-dashboard --corpus` refuses it. Run the same command again to continue; the cap
applies to each run separately.

`ewl-anthropic-batch submit --max-cost USD` works differently: it plans every request first and
refuses to send anything if the pessimistic cost estimate is over the cap.

Costs are computed from the price tables in `src/ewl/label/judges.py`. Check them against the
provider's current prices before you rely on a projection.

## Thinking and reasoning settings

Some models reason before they answer. That changes the instrument and uses up the reply limit.

- **Anthropic.** Thinking is turned off where the model allows it. If a model refuses every
  way of turning it off, `ewl-classify` and the batch planner stop, unless
  `EWL_ANTHROPIC_ALLOW_THINKING=1` is set; that runs a disclosed thinking-on arm with a larger
  reply limit and its own cache keys. The run record says which setting was used
  (`result.thinking_setting`).
- **OpenAI.** `EWL_OPENAI_REASONING_EFFORT` sets the effort; the default is `minimal`. A model
  that rejects it stops the run rather than falling back.
- **Gemini.** Thinking cannot be turned off. `EWL_GEMINI_THINKING_LEVEL` sets a level.

## Adding a backend

See [Development](development.md#adding-a-judge-backend).
