# Long runs and batches

A full corpus can take hours with a cloud model and days with a local one. This page covers
three ways to run one safely:

- `ewl-classify` with a cache, for a live API run that can be stopped and resumed;
- `ewl-gemma` with `ewl-batches`, for the local Gemma model in batches;
- `ewl-anthropic-batch`, for Claude through Anthropic's Message Batches API at half price.

All three use the same prompts, parsers and cache keys, and write the same output columns and
run records. Labels from one route can be checked against another.

## Resuming any run

Every reply goes into the response cache the moment it arrives. If a run stops for any reason
(Ctrl-C, a network drop, the cost cap, a sleeping laptop), run **the same command again**.
Answered comments replay from the cache in seconds and the run carries on from where it
stopped.

This only works if nothing about the instrument changes between runs: the same judge, scorer,
examples file, switches and environment settings, and the same version of the code. Any change
alters the cache key and the comments are asked again.

## Keeping a laptop awake

A sleeping computer stops a run. Keep it on mains power with the lid open, and stop it
sleeping:

- **macOS**: run the command under `caffeinate`, for example
  `caffeinate -dimsu ewl-classify ...` or `caffeinate -dimsu bash -e run_all.sh`.
- **Windows and Linux**: change the power settings so the machine does not sleep while plugged
  in, for the length of the run.

The speed projection `ewl-classify` prints at the end is based on the median live call and the
concurrency, not the wall clock, which a sleep would inflate.

## Local runs: `ewl-gemma`

`ewl-gemma` labels a CSV for both domain and stance with one model only: Gemma 2 27B Instruct
at Q4_0, through a local Ollama server. Nothing leaves the machine and there is no per-token
cost.

```bash
ewl-gemma --check                                              # set-up check, two invented comments
ewl-gemma corpus.csv --output t.csv --sample 60 --project-to 1000   # measure the speed
ewl-gemma corpus.csv --output labelled.csv \
    --fixed-examples official-data/fixed_examples.csv \
    --cache official-data/gemma_cache.jsonl
```

How it differs from `ewl-classify --judge gemma`:

- It has no `--judge` flag. It refuses any other model, or a model that Ollama does not report
  as Q4_0.
- It sets the 8,192-token context window explicitly. Ollama otherwise drops the **start** of an
  over-long prompt, which would remove the instructions and still return a plausible label.
- It asks every domain question first, then every stance question grouped by story. A local
  model only skips re-reading a prompt when the next request starts with the same text, so this
  is much faster than alternating the two questions.
- It runs one vote at temperature 0, and refuses a non-zero `EWL_GEMMA_TEMPERATURE`.
- `--sample N` draws N comments at random, spread across the stories. Use it instead of
  `--limit N` on a corpus file sorted by story, which would take one story only.

The default cache is `data/gemma_cache.jsonl` relative to where you run it; pass `--cache` into
your data folder. Two runs that write the same cache must not run at the same time.

## Batches for a local model: `ewl-batches`

`ewl-batches` cuts a corpus into batches (1,000 comments by default), prints the `ewl-gemma`
command for each, and checks the labelled batches before joining them.

```bash
# 1. The run input: the corpus minus the gold and example comments
ewl-batches prepare official-data/master.csv \
    --exclude official-data/gold.csv --exclude official-data/fixed_examples.csv \
    --output official-data/corpus_for_batching.csv
ewl-batches verify official-data/corpus_for_batching.csv --full official-data/master.csv \
    --exclude official-data/gold.csv --exclude official-data/fixed_examples.csv

# 2. Split it
ewl-batches split official-data/corpus_for_batching.csv --out-dir official-data/batches

# 3. Run the next five batches, unattended (macOS shown)
ewl-batches commands official-data/batches --first 5 \
    --extra "--fixed-examples official-data/fixed_examples.csv" > official-data/run_all.sh
caffeinate -dimsu bash -e official-data/run_all.sh

# 4. See progress, and join
ewl-batches status official-data/batches
ewl-batches check official-data/batches --merge-to official-data/labelled_so_far.csv --partial
ewl-batches check official-data/batches --merge-to official-data/labelled_corpus.csv
```

What each subcommand does:

| Subcommand | What it does |
|---|---|
| `prepare` | Writes the full corpus minus the comments in each `--exclude` file, plus a `.provenance.json` with the parent's hash, the exclusion files' hashes and the counts. |
| `verify` | Checks that a corpus is exactly the full corpus minus the excluded comments, with every `body` and `story` unchanged. Use it on any corpus that was made by hand. |
| `split` | Shuffles once (seed 7 by default) and cuts consecutive slices, so every batch mixes all the stories and every comment is in exactly one batch. A repeated `comment_id` is refused. `--full` and `--exclude` run the `verify` check first. `--no-shuffle` keeps file order. |
| `status` | Lists the batches that are done, pending or need attention. |
| `commands` | Prints one `ewl-gemma` command per unfinished batch, all with the same flags and one shared cache (default `<out_dir>/gemma_cache.jsonl`). `--first N` prints only the next N; run it again later for the next N. A batch that stopped half way is listed again and resumes from the cache. |
| `check` | Verifies every finished batch. It refuses a batch whose run record names a different input, whose rows are not exactly its own comments, that was labelled with `--sample` or `--limit`, or that was labelled under a different model, prompt, example set, switch, temperature or vote count. `--merge-to FILE` writes the joined file in the original corpus order, with a run record naming the whole corpus. |

`--partial` with `--merge-to` joins only the batches finished so far. Its run record carries a
`partial` block. `ewl-export-dashboard` refuses it with `--corpus`, and without `--corpus` it
warns, marks the export as a partial run and never publishes it as validated.

Under `--out-dir`, `batches_manifest.json` holds only hashes and counts. The `inputs/` and
`labelled/` folders hold comment text.

!!! note "The gold set is not in the batches"
    `prepare` leaves the gold comments out, so you cannot `ewl-validate join` the gold set to
    the batch output: it would match nothing, and `join` refuses. Label the gold file itself
    with the same `ewl-gemma` flags and cache instead, and use that output wherever a scored
    gold file is asked for.

## Claude through the Batch API: `ewl-anthropic-batch`

`ewl-anthropic-batch` sends the requests `ewl-classify` would make as Anthropic Message
Batches. Batch usage is billed at half the list price. A batch can take up to 24 hours.

It is not a new judge. The cache spec stays `anthropic:<model>`, so an answer from a batch is
stored under exactly the key a live call would use. Labels from the two routes are
interchangeable, and a gap can be filled with a live run.

```bash
ewl-anthropic-batch plan   corpus.csv --judge anthropic:claude-haiku-4-5 --sample 200 --seed 7
ewl-anthropic-batch submit corpus.csv --judge anthropic:claude-haiku-4-5 --sample 200 --seed 7 \
    --out-dir official-data/batch_trial --max-cost 20
ewl-anthropic-batch status official-data/batch_trial --check-cache
ewl-anthropic-batch fetch  official-data/batch_trial
ewl-anthropic-batch replay official-data/batch_trial --output official-data/batch_labelled.csv
ewl-anthropic-batch compare official-data/batch_labelled.csv official-data/live_labelled.csv
```

| Subcommand | What it does |
|---|---|
| `plan` | A dry run: request count and estimated cost. Sends and writes nothing. |
| `submit` | Runs the corpus through the real labelling code with a collector in place of the API. A reply already in the cache is skipped, so running it again sends only the gaps. Sends batches one at a time as they are built. `--max-cost` refuses before anything is written or sent if the pessimistic estimate is over the cap. Refuses an output folder started with different settings, input, prompts or cache specs, and refuses while earlier batches are not yet fetched. |
| `status` | Progress of each batch. `--check-cache` counts comments whose replies are all cached, and why the others are not. |
| `fetch` | Downloads finished batches into the cache. A refusal or a reply cut off at the token limit is counted and not cached; it is recorded by key in `batch_uncacheable.jsonl` so `replay` uses it and `submit` does not pay for it again (unless `--retry-uncacheable`). `--abandon BATCH_ID` gives up a batch whose results are gone. |
| `replay` | Labels the corpus from the cache alone and never calls the API. Writes the same columns and run record as `ewl-classify`. Refuses while comments are incomplete, unless `--partial`. |
| `compare` | How far two labelled files agree, comment by comment. |

Things to know:

- **Use a separate cache** for a batch run (the default is `<out-dir>/batch_cache.jsonl`). If a
  live run fills the same cache first, `submit` finds everything answered and sends nothing.
- **Download results within 29 days.** Anthropic deletes batch results after that. Once they
  are in the cache they are yours.
- **Cache writes cost more.** A prompt-cache write costs 2× the input price with `--ttl 1h` (the
  default) and 1.25× with `--ttl 5m`.
- The manifest holds batch ids, counts and hashes, never comment text. `subset.csv` (written
  by `--sample`) holds comment text.
- `ewl-classify --cache-only` also replays from the cache and stops at the first missing reply,
  so a gap can never turn into a full-price call.
- Comment text goes to Anthropic exactly as with live calls. See
  [Data protection](data-protection.md).
- Run `plan`, then a small `--sample` trial on your own account, before a full run. Read the
  real token counts and cost that `fetch` prints.
