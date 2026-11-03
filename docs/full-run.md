# Full run, in order

This is the whole route for labelling a real corpus with a cloud model, in order. Paths use
`official-data/` as the git-ignored data folder; use your own. Replace `<JUDGE>`, `<SCORER>`
and the other placeholders with the choices you have written down.

Read [Data protection](data-protection.md) first.

## 1. Freeze the corpus

Collect, clean and merge as described in [The pipeline](pipeline.md):

```bash
ewl-merge official-data/<story>_clean.csv ... \
    --cap-per-author 20 --cap-scope story --dedupe-bodies \
    --output official-data/master.csv
```

`ewl-merge` writes `master.csv.manifest.json`. Export `EWL_AUTHOR_SALT` first, so the manifest
records its fingerprint. Do not change the corpus once the gold set has been drawn from it.

## 2. Build the gold set

1. `ewl-keywords`, then `ewl-primary-category`, on the frozen corpus.
2. `ewl-validate sample --stratify-by story,primary_category --video-cap-fraction 0.15`.
3. At least two coders code the sample independently.
4. `ewl-validate interrater` on the coders' original files.
5. `ewl-adjudicate` for the comments coded in common, then `ewl-validate combine` (or
   `reweight`) against the frozen corpus, so every inclusion probability is right.

See [Validation](validation.md).

## 3. Make the run input

Leave the gold comments and the worked examples out of the corpus you label:

```bash
ewl-batches prepare official-data/master.csv \
    --exclude official-data/gold.csv --exclude official-data/fixed_examples.csv \
    --output official-data/corpus_for_batching.csv
```

Check the count against `corpus_for_batching.csv.provenance.json`. Rows with no comment text
are dropped by the labelling step and counted on the terminal.

## 4. Fix the instrument

Choose, and write down, before anything is labelled:

- the judge and the sentiment scorer;
- the worked examples file (`--fixed-examples`);
- every instrument switch (leave them at their defaults unless you have decided otherwise);
- the number of votes;
- the model's thinking or effort setting, set explicitly in the environment (for example
  `EWL_OPENAI_REASONING_EFFORT=low`) so it is part of the cache key.

The cache is keyed on all of it. A change later means labelling everything again.

For an OpenAI or Gemini judge, set `EWL_EXTERNAL_PROCESSORS` once your data-protection
sign-off is recorded; without it the judge refuses to start. An Anthropic judge is not gated by
that variable, so record its sign-off separately.

**Redaction.** Decide whether to redact. If you do, redact a **copy** of
`corpus_for_batching.csv` and of the gold file (`ewl-redact --handles all --emails --phones
--output ...`), label the copies, and pass the redacted corpus to `--corpus` in step 9. Never
redact the master corpus or the prepared corpus in place. If you do not redact, say in the
methods that handles and URLs are present in the text sent to the provider.

## 5. Pre-flight

Run a few hundred comments with a separate cache:

```bash
ewl-classify official-data/corpus_for_batching.csv --limit 200 \
    --judge <JUDGE> --sentiment <SCORER> --selector fixed \
    --fixed-examples official-data/fixed_examples.csv --votes <N> --concurrency <C> \
    --cache official-data/preflight_cache.jsonl --output official-data/preflight_corpus.csv
```

Do the same for a slice of the gold file, score it (step 7) and export it (step 9). A join
between the two cannot work, because the prepared corpus holds no gold comments.

Read the printed cost, time per comment and parse-failure rate, and the run record. If the
corpus file is sorted by story, `--limit` takes one story only; that is fine for a mechanical
check but not for a speed estimate (`ewl-gemma` and `ewl-anthropic-batch` have `--sample`).

## 6. Label the corpus

```bash
ewl-classify official-data/corpus_for_batching.csv \
    --judge <JUDGE> --sentiment <SCORER> --selector fixed \
    --fixed-examples official-data/fixed_examples.csv --votes <N> --concurrency <C> \
    --max-cost <USD> --cache official-data/<run>_cache.jsonl \
    --output official-data/<run>_corpus.csv
```

Use a fresh cache file for the run. If it stops for the cost cap, a network drop or a sleeping
laptop, run the same command again: the cache continues it. Keep the laptop awake (see
[Long runs and batches](batches.md#keeping-a-laptop-awake)). For a local model use `ewl-gemma`
with `ewl-batches`; for Claude at half price use `ewl-anthropic-batch`.

## 7. Label the gold set with the same instrument

Run `ewl-classify` on the gold file with exactly the same judge, scorer, votes, examples and
switches, into its own output (`official-data/<run>_gold.csv`). `ewl-export-dashboard` compares
the two run records and refuses a mismatch.

## 8. Score the gold set

```bash
ewl-validate score official-data/<run>_gold.csv
```

It leaves out the worked examples named in the gold file's run record. If the examples were
already taken out of the gold file, expect `Excluded 0`; that is correct. Read the counts of
rows left out as unlabelled or uncoded.

## 9. Correct

Run Rogan-Gladen (the lead estimator) and PPI (the check) for each category, on the
denominator chosen for the headline (`--base`):

```bash
ewl-report corrected-by-story official-data/<run>_corpus.csv \
    --column domain_democracy --gold official-data/<run>_gold.csv
ewl-report corrected-bootstrap official-data/<run>_corpus.csv \
    --column domain_democracy --gold official-data/<run>_gold.csv
ewl-report ppi-bootstrap official-data/<run>_corpus.csv \
    --column domain_democracy --gold official-data/<run>_gold.csv
```

Run each a second time with `--rescale-unclassified official-data/master_kw_primary.csv` (the
frozen corpus with `primary_category`) and report both. See
[Correcting prevalence](correction.md).

## 10. Export

```bash
ewl-export-dashboard official-data/<run>_corpus.csv \
    --corpus official-data/corpus_for_batching.csv \
    --scored official-data/<run>_gold.csv \
    --interrater official-data/interrater.json \
    --output dashboard/data/results.json
```

`--corpus` is the **input** file the run record names, not the labelled output. The export
reports a partial run, an instrument mismatch or a missing run record. Do not publish a figure
from a file the export refused.

## 11. Keep the record

For every figure you quote, keep:

- the run record (`<output>.run.json`);
- the cache file;
- the git commit or tag of the code used.

Together these reproduce the labels exactly. All three stay out of git: the cache and the run
record are git-ignored because they hold replies and ids.
