# Quickstart

This page runs the pipeline end to end on two small files of made-up comments that ship with
the package. It needs no API key, no network and no model download. It uses the `keyword`
judge, which matches a word list, so the labels are free and the same every time.

The two files are:

- `examples/synthetic_comments.csv`: 12 invented comments across the four stories. This plays
  the part of the corpus.
- `examples/synthetic_gold.csv`: 24 different invented comments, with hand codes
  (`hand_domain`, `hand_sentiment`) and an inclusion probability for each row. This plays the
  part of the gold set.

Run the commands from the repository root. Write outputs to a folder outside the repository,
so nothing you produce can be committed by mistake:

```bash
OUT=$(mktemp -d)
```

## 1. Label the comments

```bash
slens-classify examples/synthetic_comments.csv --judge keyword --no-cache --output "$OUT/out.csv"
```

`--judge keyword` picks the keyword matcher. `--no-cache` skips the response cache, which a
free, instant judge does not need. Without `--sentiment`, the stance column comes from VADER;
VADER is a placeholder here, not a scorer to quote (see [Judges and scorers](judges.md)).

The summary goes to the terminal (trimmed):

```text
NOTE: No manifest for synthetic_comments.csv -- it was not frozen by slens-merge, so nothing here can confirm it has not changed.
Judge: keyword:keywords.yml | examples: fixed(25) | votes: 1 | sentiment: vader

Labelled 12 comments with keyword:keywords.yml (1 vote(s) each)
...
Domain frequencies:
  democracy: 4 (33.3% of comments)
  social: 3 (25.0% of comments)
  unclassified: 3 (25.0% of comments)
  economics: 2 (16.7% of comments)
  environment: 1 (8.3% of comments)
Unclassified: 25.0% | Parse failures: 0.0% | Unstable across votes: not measured (--votes 1)
...
Stability is not accuracy. Run evaluate.py against a hand-coded gold set before quoting any of these labels in the write-up.
Run record written to out.csv.run.json
```

The `NOTE` about a manifest is expected: the example file was not frozen by `slens-merge`.

Two files are written:

- `out.csv`: the input columns plus the labels. The first rows look like this (some columns
  left out):

    ```text
    comment_id,story,...,domain_democracy,domain_economics,domain_environment,domain_social,domain_unclassified,domain_parse_failure,domain,...,sentiment,sentiment_value,...
    ex001,jets,...,0,0,1,0,0,0,environment,...,negative,-0.2732,...
    ex002,jets,...,0,0,0,1,0,0,social,...,positive,0.3612,...
    ```

- `out.csv.run.json`: the run record. It names the input file and its hash, the judge, the
  sentiment scorer, the worked examples, a hash of every prompt and the result counts. It holds
  no comment text from the input. See [The pipeline](pipeline.md#run-records).

## 2. Label the gold set with the same instrument

To score a labelling run, the gold comments must be labelled by the same judge, scorer and
settings as the corpus:

```bash
slens-classify examples/synthetic_gold.csv --judge keyword --no-cache --output "$OUT/gold_labelled.csv"
```

The output keeps the hand codes and inclusion probabilities beside the machine labels.

## 3. Score the labels against the hand codes

```bash
slens-validate score "$OUT/gold_labelled.csv"
```

```text
Scoring against the LLM labels (domain_*).
Scoring 24 hand-coded comments

domain          precision   recall     F1   (LLM vs hand)
democracy            1.00     0.88   0.93
economics            0.55     0.55   0.55
environment          1.00     0.33   0.50
social               0.00     0.00   0.00
```

The figures are weighted by each row's `inclusion_prob` (see [Validation](validation.md)).
The line says "LLM labels" whichever judge produced the `domain_*` columns. These numbers
describe a keyword list on 24 invented comments; they say nothing about any real data.

To compare several judges at once on the gold set, use the scorecard:

```bash
slens-evaluate scorecard examples/synthetic_gold.csv --arm keyword/none --arm stub/fixed \
    --cache "$OUT/eval_cache.jsonl"
```

## 4. Correct a share for classifier error

`slens-report` takes the labelled corpus and the labelled gold set and estimates the true share
of one harm domain:

```bash
slens-report corrected-bootstrap "$OUT/out.csv" --column domain_democracy \
    --gold "$OUT/gold_labelled.csv"
```

With 24 gold rows it refuses, and says why:

```text
8 raw genuine positive(s) for 'democracy' in gold_labelled.csv, worth 8.0 effective (Horvitz-Thompson-weighted) one(s) -- needs at least 10 effective
...
Corrected-bootstrap (Rogan-Gladen): refused -- too few effective positives in --gold, ...
```

That refusal is the guard working. A real gold set needs at least 10 effective positives per
category, and fewer than 50 is flagged as low evidence. To see the output shape on this toy
data only, lower the floor:

```bash
slens-report corrected-bootstrap "$OUT/out.csv" --column domain_democracy \
    --gold "$OUT/gold_labelled.csv" --min-genuine-positives 3
```

```text
LOW EVIDENCE: 8.0 effective positives is below 50. ...
Corrected (Rogan-Gladen point estimate): 38.1% [LOW EVIDENCE: 8.0 effective positives]
Corrected 95% CI (clustered, corpus + gold resampled together): 7.1%–81.8% [LOW EVIDENCE: 8.0 effective positives]
```

Never lower `--min-genuine-positives` for a figure you intend to report.

## 5. Clean up

```bash
rm -r "$OUT"
```

## Where to go next

- [The pipeline](pipeline.md): every stage, with its input and output files.
- [Judges and scorers](judges.md): how to run a real model.
- [Validation](validation.md) and [Correcting prevalence](correction.md): what the numbers
  mean.
- [Full run, in order](full-run.md): the procedure for a real corpus.
- [Data protection](data-protection.md): read this before you handle real comments.
