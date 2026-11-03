# Validation

A model's labels are not a result until they have been checked against comments that people
coded by hand. This page covers the gold set, how coders' agreement is measured, how
disagreements are settled, and how a labelling run is scored.

## The gold set

The gold set is a sample of the corpus, coded by hand. Draw it with `ewl-validate sample` (see
[The pipeline](pipeline.md#4-draw-the-gold-sample)). Each row has:

| Column | Filled in by | Meaning |
|---|---|---|
| `comment_id`, `story`, `source`, `body` | the sampler | The comment. Coders see `body`, never a machine label. |
| `hand_domain` | coder | One or two of `democracy`, `economics`, `environment`, `social` (comma-separated), or `none`. Blank means "not coded yet", which is not the same as `none`. |
| `hand_sentiment` | coder | `negative`/`neutral`/`positive`, or a rating from 1 (very negative) to 5 (very positive); `0` (or `irrelevant`) for no stance on wealth; `-1` (or `uncodeable`) when the stance cannot be judged. |
| `coder_notes` | coder | Free text. |
| `sampling_stratum`, `inclusion_prob` | the sampler | Which stratum the row was drawn from, and its chance of being drawn. |

Numeric ratings are bucketed the same way as an LLM's: 1–2 negative, 3 neutral, 4–5 positive.
`0` and `-1` are kept out of the three-way score and counted separately.

`examples/synthetic_gold.csv` shows the format with invented comments.

### Weights

A stratified draw over-samples rare strata on purpose. Every score and correction therefore
weights each gold row by `1 / inclusion_prob` (a Horvitz-Thompson weight), so the figures
describe the corpus, not the draw. If every row's `inclusion_prob` is blank, every row weighs 1
(right for a simple random sample). A file where some rows have a probability and others do not
is refused, and so is a value outside (0, 1].

If the gold set was assembled by hand (top-ups, replacement rows), recompute the probabilities
against the final corpus before scoring:

```bash
ewl-validate reweight gold.csv --master gold_sample.csv --topup gold_topup.csv \
    --corpus master_kw_primary.csv --video-cap-fraction 0.15 --output gold_reweighted.csv
```

`ewl-validate combine --corpus ...` does the same while it combines coders' files. A row that is
not in the final corpus is excluded and listed, never given a guessed probability.

## Agreement between coders

Measure how far the coders agree with each other before you compare any coder with a model. If
careful readers cannot agree, no classifier can be held to a higher standard.

```bash
# two coders
ewl-validate interrater coder_a.csv --second-input coder_b.csv --json-output interrater.json

# three or more coders, on the comments they all coded
ewl-validate interrater coder_a.xlsx --second-input coder_b.xlsx --extra-inputs coder_c.xlsx \
    --json-output interrater.json
```

Use each coder's **original** file, never an adjudicated one. `.xlsx` files need the
`analysis` extra. `scripts/scratch_check_coder_files.py` checks coders' files for blank or
repeated ids and off-list values first.

**Two coders** get Cohen's kappa for each harm category and for the stance, on the full 1–5
scale, both plain and quadratic-weighted. The weighted figure counts a one-point disagreement
as smaller than a four-point one; it uses only pairs where both coders gave a number.

**Three or more coders** get Krippendorff's alpha, at the level of measurement each judgement
has, with a 95% interval from resampling comments:

- the harm domain, one present/absent judgement per category: nominal alpha, with Fleiss'
  kappa beside it;
- the kind of stance answer (a stance, `0` or `-1`): nominal alpha;
- the strength of the stance (1–5), on comments every coder rated 1–5: ordinal alpha.
  `--sentiment-levels buckets` compares only the direction (negative/neutral/positive).

Read alpha with Krippendorff's guide: 0.80 or more is reliable; 0.667 to 0.80 supports
tentative conclusions only; below 0.667 is not reliable. A category nobody used shows `n/a`
(alpha is undefined) and must be reported as untested. With a few dozen shared comments the
intervals are wide: report the interval and the counts beside every figure.

`--json-output` writes the figures for the dashboard. A category with no variation exports
`kappa: null` and `no_variation: true`, not 1.0.

## Settling disagreements

`ewl-adjudicate` settles the comments that exactly three coders coded in common, by majority,
in two steps:

```bash
# step 1: settle what the majority can, and write the rest to a review file
ewl-adjudicate coder_a.xlsx coder_b.xlsx coder_c.xlsx --core-ids core_ids.csv \
    --review-output core_review.csv

# step 2: after filling in hand_domain / hand_sentiment for the flagged rows
ewl-adjudicate coder_a.xlsx coder_b.xlsx coder_c.xlsx --core-ids core_ids.csv \
    --decisions core_review.csv --output core_adjudicated.csv \
    --gold gold.csv --gold-output gold_adjudicated.csv
```

- **Domain.** A majority for `none` settles as none. A category two of three coders tagged is
  kept. A category only one coder tagged is left out (`--extra-labels ignore`, the default);
  `--extra-labels add` adds it when that makes at most two labels. A three-way split is
  flagged.
- **Stance.** Compared as negative (1–2), neutral (3), positive (4–5), irrelevant (0) and
  uncodeable (−1). A majority settles it; three different answers are flagged.
  `--sentiment-rule exact` needs two identical values instead.
- Step 1 records its settings; step 2 refuses different ones.
- The review file holds comment text. Keep it in your data folder.

Report the agreement figures from the original files, and say how many comments the majority
settled and how many a person decided.

`ewl-validate combine` then builds one gold file from every coder's file, the split manifest
and the drawn sample (`--coders`, `--manifest`, `--master`, optionally `--adjudicated`,
`--topup`, `--replacements`, `--corpus`).

## Scoring a labelling run

### Label the gold file with the same instrument

The simplest route is to label the gold file itself, with the same judge, scorer, votes,
examples and switches as the corpus run, into its own output:

```bash
ewl-classify gold.csv --judge <same> --sentiment <same> --selector fixed \
    --fixed-examples official-data/fixed_examples.csv --votes <same> \
    --cache official-data/judge_cache.jsonl --output official-data/gold_labelled.csv
```

If the gold comments are inside the labelled corpus, `ewl-validate join` copies their machine
labels across instead:

```bash
ewl-validate join gold.csv --labelled master_labelled.csv --output scored.csv
```

`join` refuses when any coded comment is missing from `--labelled` (`--allow-partial` accepts
that, and the missing rows are then left out of scoring, not counted as misses).

### `ewl-validate score`

```bash
ewl-validate score official-data/gold_labelled.csv
```

It prints weighted precision, recall and F1 for each category. Rows the model could not parse
are left out (a parse failure is not an answer), as are rows with no machine label and rows
with a blank `hand_domain`, and the counts are printed. If there are no `domain_*` columns, it
scores the `kw_*` columns from `ewl-keywords` instead.

### Worked examples are left out

A comment shown to the model as a worked example would be scored against its own label.
`ewl-validate score` and `ewl-report` read the example ids from the gold file's run record
(`fixed_examples.ids`) and leave those rows out automatically. `--exclude-ids FILE` adds more
(a list of ids, or any CSV with a `comment_id` column; the examples file works). When no
example ids are known, and the built-in synthetic examples were not used, a warning says so;
`--no-example-exclusion` silences it when no gold comment was shown as an example.

### `ewl-evaluate`: compare arms

`ewl-evaluate` runs one or more arms directly on a hand-coded file and prints a scorecard.

```bash
ewl-evaluate scorecard gold_heldout.csv \
    --arm keyword/none --arm anthropic:claude-haiku-4-5/fixed \
    --fixed-examples official-data/fixed_examples.csv --cache official-data/eval_cache.jsonl

ewl-evaluate sentiment gold_heldout.csv \
    --arm vader --arm llm:anthropic:claude-haiku-4-5 --cache official-data/eval_cache.jsonl
```

An arm is `<judge spec>/<selector>`. Every arm must use `fixed` (the keyword baseline is
exempt) unless `--allow-non-fixed` is passed.

The domain scorecard columns:

| Column | Meaning |
|---|---|
| `n` | Gold comments scored. |
| `democracy P`, `democracy R` | Precision and recall for democracy, broken out. |
| `macro-F1` | F1 averaged over the four categories, so a win on a common category cannot hide a failure on a rare one. |
| `uncls` | Share answered "none". Information, not a bug. |
| `parse-fail` | Share of replies that could not be read. Kept apart from `uncls`. |
| `unstable` | Share where the votes disagreed; `—` at one vote, where it cannot be measured. |
| `reproducible` | Whether repeated runs of the backend agree exactly. |
| `cost`, `mins`, `in tok`, `out tok` | Spend, time and tokens for the arm. |

Tables of per-category F1 and false-positive rate follow. `--bootstrap N` adds 95% intervals on
each category's precision, recall and F1, resampling whole videos.

The sentiment scorecard reports weighted accuracy against a majority-class baseline, F1 per
class, `r(length)` (the correlation between the score and comment length; near zero is the
target) and a confusion matrix. It also reports, separately, how each arm handled the comments
coded `irrelevant` or `uncodeable`.

Other subcommands:

- `ewl-evaluate split` divides a gold file into a retrieval pool and a held-out set
  (`--split-by video` keeps each video on one side). Never let a held-out comment enter the
  pool.
- `ewl-evaluate score` scores one `--judge`.

`--concurrency N` on `scorecard`, `sentiment` and `score` labels N comments at once; results
match a serial run.

### Other checks

- `ewl-validate agree FILE` gives per-category kappa between the model (`domain_*`) and the
  keyword matcher (`kw_*`) on a labelled corpus: a cheap triage signal, not a score.
- `ewl-validate selfconsistency FILE` measures how far a model agrees with itself across votes
  (it needs a run with `--votes 2` or more), as a kappa that can be read beside the coders'
  kappa. `--core-ids` restricts it to the comments the coders shared; `--compare-json` prints
  it beside an `interrater` JSON file.
- `ewl-agreement --model a=gold_a.csv --model b=gold_b.csv ...` treats each model and the hand
  codes as raters on the same gold comments, and prints pairwise kappa, Krippendorff's alpha
  with and without the hand codes, and "consensus errors" (all models agree with each other and
  disagree with the hand codes). These figures are unweighted. Models can share mistakes, so
  agreement between models is not correctness.

## What the numbers mean

- **Precision** is the share of comments the model tagged with a category that the coders also
  tagged. **Recall** (the true positive rate, TPR) is the share of comments the coders tagged
  that the model also tagged. The **false positive rate** (FPR) is the share of comments the
  coders did not tag that the model did. TPR and FPR feed the correction in
  [Correcting prevalence](correction.md).
- Scores on a gold set that included the worked examples, or examples chosen from the same
  coders, are more optimistic than a test on untouched comments. Say so when you report them.
- A gold set of fewer than 200 scored comments can never be stamped `validated` on the
  dashboard.
