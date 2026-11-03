# The pipeline

Each stage is one command. Each reads a CSV and writes a new CSV; no stage edits its input.
This page describes the stages in order, with the files and columns each one reads and writes.
[Full run, in order](full-run.md) gives the exact procedure for a real run.

Keep every file this page mentions in a git-ignored local data folder. The project used
`official-data/` (the `.gitignore` already excludes it, as well as `data/`, `*.csv`, `*.jsonl`
and `*.run.json`). Every one of these files except the manifests holds comment text or ids.
See [Data protection](data-protection.md).

## The comment schema

Every collector writes the same columns in the same order:

| Column | Meaning |
|---|---|
| `comment_id` | The platform's id for the comment. Every later stage matches rows on it. |
| `platform` | `youtube` (a Reddit collector exists but is not part of the active route). |
| `story` | The story label given at collection, for example `jets`. |
| `source` | The video. Every interval is clustered on this column. |
| `author_hash` | A salted hash of the author name. The raw name is never written. |
| `body` | The comment text. |
| `upvotes`, `timestamp`, `timestamp_precision`, `depth`, `collected_at` | Metadata. |
| `detected_lang` | The detected language, when the `language` extra is installed. |

Labelling only needs `body`, and uses `story` (to add story background to the prompt),
`comment_id` and `source` when present. An input with no `story` column is labelled with
story-free prompts, and `ewl-classify` warns.

All CSVs are written as UTF-8 with a byte-order mark and read with `utf-8-sig`, so Excel opens
them correctly. If you edit a file in Excel, save it with **CSV UTF-8**, then check the first
line still holds the real column names.

## 1. Collect

```bash
ewl-collect --story jets --search "private jet emissions billionaires" \
    --target-videos 50 --top-n 10 --min-comments 20 --seed 7 \
    --output official-data/jets.csv \
    --video-list-output official-data/jets_videos.csv \
    --exclusion-table-output official-data/jets_exclusions.csv
```

`ewl-collect` uses the YouTube Data API v3. It needs `YOUTUBE_API_KEY` and `EWL_AUTHOR_SALT`.

- **Video selection is hybrid.** It takes the `--top-n` videos by relevance, then tops up at
  random to `--target-videos`. `--search` is repeatable; every query feeds one shared,
  de-duplicated pool before the draw. The same `--seed` and search settings give the same
  videos.
- **Shorts are excluded by default** (`--max-shorts-seconds 60`). The API has no flag for a
  Short, so this is a duration rule, not an exact filter. `0` keeps everything.
- `--video-list-output` records which videos went in. `--exclusion-table-output` records how
  many comments each filter removed.
- Collect one story per run. There is no resume after the API quota runs out: a stopped run
  still writes what it collected and reports the truncation on stderr.

`ewl-collect-full-corpus` takes the same arguments but turns every cap off (all comments, every
video found, no comment minimum and no Shorts filter).

## 2. Clean, merge and freeze

Drop videos that should not be in the corpus (off-topic, or posted before the story):

```bash
ewl-clean-videos official-data/jets.csv --exclude-file excluded_video_ids.txt \
    --output official-data/jets_clean.csv
```

Video ids must match exactly. A mistyped id is refused, with the likely real one suggested.

Then merge every story into one corpus. Always merge, even with one input: later stages
expect the merged file.

```bash
ewl-merge official-data/jets_clean.csv official-data/luigi_clean.csv \
    official-data/musk_clean.csv official-data/propublica_clean.csv \
    --cap-per-author 20 --cap-scope story --dedupe-bodies \
    --output official-data/master.csv
```

- `ewl-merge` removes repeated `comment_id`s, so re-running a story and merging again is safe.
- `--cap-per-author N` keeps at most N comments per `author_hash`. `--cap-scope story` counts
  within each story, so the result does not depend on the order of the inputs.
- `--dedupe-bodies` drops a comment whose text repeats an earlier one in the same story.

### Manifests

`ewl-merge` writes `<corpus>.manifest.json` beside the corpus. It records the file's SHA-256
hash, the row, story and video counts, the input file names and a fingerprint of
`EWL_AUTHOR_SALT` (never the salt). It holds no comment text.

Later commands check the corpus still matches its manifest and print a `NOTE` when it does not,
or when there is no manifest. `--strict-corpus` on `ewl-classify` and `ewl-report` turns a
mismatch into a hard stop. Use it for any run whose figures you will quote. `ewl-keywords` and
`ewl-redact` write a manifest for their own output too.

### Optional: redact

```bash
ewl-redact official-data/master.csv                       # report only
ewl-redact official-data/master.csv --handles leading --emails --phones \
    --output official-data/master_redacted.csv
```

Redact **before** labelling: the cache is keyed on the comment text, and a cloud judge would
already have received the handles. See [Data protection](data-protection.md#redaction).

## 3. Keyword pass and primary category

```bash
ewl-keywords official-data/master.csv --output official-data/master_kw.csv
ewl-primary-category official-data/master_kw.csv --output official-data/master_kw_primary.csv
```

`ewl-keywords` adds `kw_democracy`, `kw_economics`, `kw_environment`, `kw_social` and
`kw_unclassified` (0 or 1), `kw_domain` (up to two labels, strongest first) and `kw_evidence`
(the matched words). The vocabulary is `src/ewl/keywords.yml`; `--keywords FILE` uses another.
`--audit` reports how much each term lifts each category.

`ewl-primary-category` collapses `kw_domain` to its first label in `primary_category`, so each
comment sits in exactly one sampling stratum.

## 4. Draw the gold sample

```bash
ewl-validate sample official-data/master_kw_primary.csv --n 400 --seed 7 \
    --stratify-by story,primary_category --video-cap-fraction 0.15 \
    --output official-data/gold_sample.csv
```

The output has `comment_id`, `story`, `platform`, `source` and `body`, empty `hand_domain`,
`hand_sentiment` and `coder_notes` columns for the coders, and `sampling_stratum` and
`inclusion_prob`. Machine labels are left out, so coders work blind. Keep `inclusion_prob`: the
scores and corrections weight each row by `1 / inclusion_prob`.

`--video-cap-fraction 0.15` stops one video supplying more than 15% of a stratum's draw (a
soft cap: a stratum that cannot fill within the cap is topped up past it, and the output says
so).

Coding, agreement and adjudication are covered in [Validation](validation.md).

## 5. Label

```bash
ewl-classify official-data/master.csv \
    --judge anthropic:claude-haiku-4-5 --sentiment llm:anthropic:claude-haiku-4-5 \
    --selector fixed --fixed-examples official-data/fixed_examples.csv \
    --votes 1 --concurrency 8 --max-cost 50 \
    --cache official-data/judge_cache.jsonl --output official-data/master_labelled.csv
```

[Judges and scorers](judges.md) covers every `--judge`, `--sentiment` and `--selector` value.
For long runs, see [Long runs and batches](batches.md).

Rows with no comment text are skipped and counted. `ewl-classify` adds these columns:

| Column | Meaning |
|---|---|
| `domain_democracy`, `domain_economics`, `domain_environment`, `domain_social` | 1 when the comment carries that domain. |
| `domain_unclassified` | 1 when the model cleanly answered "none". |
| `domain_parse_failure` | 1 when the reply could not be read, or the votes had no majority. A parse failure is left out of every share's denominator; it is not counted as "none". |
| `domain` | The labels as text, for example `democracy, economics`. |
| `domain_unstable` | 1 when the votes disagreed. Always 0 at one vote, where stability is not measured. |
| `domain_votes`, `domain_judge`, `domain_raw_votes` | How many votes ran, the judge and selector, and each vote's answer (`|` between votes). |
| `sentiment` | `negative`, `neutral`, `positive`, `irrelevant` or `uncodeable`. |
| `sentiment_value` | The stance on a −1 to 1 scale. An LLM's 1–5 rating becomes `(rating − 3) / 2`; VADER gives its compound score. Blank when there is no stance to rate. |
| `sentiment_scorer` | The scorer, with a tag for any instrument switch that is not at its default. |
| `sentiment_parse_failure`, `sentiment_no_consensus`, `sentiment_unstable`, `sentiment_votes`, `sentiment_raw_votes` | The same checks for the stance question. |
| `compound`, `pos`, `neu`, `neg` | VADER's own scores. Blank for any other scorer. |

`ewl-gemma` and `ewl-anthropic-batch replay` write the same columns.

### Run records

Every labelling run writes `<output>.run.json` beside its output. It records:

- the command line and Python version;
- the input path and its SHA-256 hash;
- the judge (`domain_judge`: spec, selector, votes, reply limit) and the sentiment scorer;
- the instrument switches and a hash of every prompt, including one per story;
- the worked examples: the text of the built-in synthetic set, or, for a
  `--fixed-examples` file, only the ids, labels and a hash;
- the cache path and size;
- the results: counts, frequencies, calls, attempts, retries, cost and speed, and a `partial`
  block if the run stopped early.

The run record holds no comment text from the input. It does hold comment ids when a real
examples file is used, so it is git-ignored. Later commands read it:

- `ewl-validate join` copies it beside its output, and `ewl-keywords` copies it beside its
  output, so the chain classify → keywords → export keeps it.
- `ewl-validate score` and `ewl-report` read the example ids from the gold file's record and
  leave those comments out of scoring.
- `ewl-export-dashboard --scored` compares the gold file's record with the corpus's record and
  refuses a difference in judge, scorer, prompts, examples or switches (unless
  `--allow-instrument-mismatch`). `--corpus` refuses a corpus that is not the input the record
  names, and refuses a partial run.

The run record, the cache file and the git commit of the code are what make a figure
reproducible. Keep all three for any figure you quote.

## Caches

Every model reply is stored in a JSONL response cache as it arrives. The cache key is a
SHA-256 hash of:

- the judge's **cache spec**: the backend and model, plus any setting that changes the request
  (`|effort=…` and `|thinking=allowed` for Anthropic, `|reasoning_effort=…` for a non-default
  OpenAI effort, `|thinking_level=…` for Gemini, `|temperature=…` for a sampled Gemma run);
- the full **prompt**, which includes the instructions, the worked examples, the story
  background and every instrument switch;
- the **comment text**, with leading and trailing spaces removed (by `ewl-classify` and
  `ewl-anthropic-batch`; `ewl-gemma` does not strip it);
- the **vote number**.

So a changed prompt, example set, switch, model or setting gets new answers; nothing is
replayed under the wrong instrument. A reply that was refused or cut off at the token limit is
used for that run but not cached.

What this gives you:

- **Resuming.** If a run stops (a network drop, the `--max-cost` cap, a closed laptop), run the
  same command again. Answered comments replay from the cache for free and the run carries on.
- **Reproducing.** An API model's answers can change over time. The cache is what makes a
  re-run byte-identical. If you quote labels from an API arm, keep its cache file.

Defaults: `ewl-classify` uses `data/judge_cache.jsonl` relative to where you run it (so set
`--cache` explicitly, into your data folder), `ewl-gemma` uses `data/gemma_cache.jsonl`, and
`ewl-anthropic-batch` uses `<out-dir>/batch_cache.jsonl`. `ewl-evaluate` caches only when
`--cache` is given. `--no-cache` turns caching off.

!!! warning "One cache per instrument"
    Use a fresh cache file for each real run, and a separate one for trials. A sampled Gemma
    run (`EWL_GEMMA_TEMPERATURE` above 0) and a batch trial must not share a cache with the
    corpus run.

`ewl-classify` refuses a cache that holds empty replies (left by an older version) unless
`--keep-empty-cached-replies` is passed. `--cache-only` never calls the model: it stops at the
first comment with no cached reply.

## 6. Validate, correct and export

- [Validation](validation.md): score the labels against the gold set.
- [Correcting prevalence](correction.md): `ewl-report`.
- Export for the dashboard:

    ```bash
    ewl-export-dashboard official-data/master_labelled.csv \
        --output dashboard/data/results.json \
        --corpus official-data/master.csv \
        --scored official-data/gold_labelled.csv
    ```

    `results.json` holds observed shares with clustered intervals, validation figures, the
    codebook (prompts, scale, story paragraphs) and provenance. It holds no comment text unless
    `--include-verbatim` is passed. It does not carry corrected (Rogan-Gladen or PPI) figures;
    those come from `ewl-report`. Without `--scored` it refuses to write unless
    `--allow-unvalidated` is passed. Fewer than 200 scored gold comments can never be stamped
    `validated`, and a story with fewer than 8 videos gets no interval.
