# Correcting prevalence

A labelled corpus says what the model said. If the model misses some comments about a harm and
wrongly tags others, the observed share is biased. `ewl-report` estimates the true share by
measuring the model's errors on the gold set and correcting for them, with an interval that
carries both sources of uncertainty.

`ewl-report` takes a labelled corpus and, for the corrected figures, a labelled gold file
(`--gold`). `--column` names the share, for example `domain_democracy` or `sentiment_value`.

## Subcommands

| Subcommand | What it gives |
|---|---|
| `observed` | The raw share in each group (`--by`, default `story`), with a 95% interval clustered by video. Also prints `eff_videos`. |
| `aggregate` | One figure across the stories, weighting each story equally, with a clustered interval. |
| `corrected` | A Rogan-Gladen point estimate from a TPR and FPR you supply (`--tpr`, `--fpr`). No interval and no evidence check; it prints `UNGATED`. |
| `corrected-bootstrap` | The Rogan-Gladen estimate with a 95% interval, resampling corpus and gold together by video. **The lead estimator.** |
| `ppi-bootstrap` | The prediction-powered (PPI) estimate with a 95% interval, resampled the same way. **The check.** |
| `corrected-by-story` | The Rogan-Gladen estimate and interval for each story on its own, or the reason a story is refused. |

```bash
ewl-report corrected-bootstrap official-data/master_labelled.csv \
    --column domain_democracy --gold official-data/gold_labelled.csv
ewl-report ppi-bootstrap official-data/master_labelled.csv \
    --column domain_democracy --gold official-data/gold_labelled.csv
ewl-report corrected-by-story official-data/master_labelled.csv \
    --column domain_democracy --gold official-data/gold_labelled.csv
```

The category is taken from `--column` (`domain_democracy` → `democracy`), or set with
`--category`. Report a corrected figure beside the observed share and its interval, never in
place of them.

## Rogan-Gladen

For one category, let:

- *p* be the observed share in the corpus (the share the model tagged);
- TPR be the true positive rate on the gold set (the share of hand-coded positives the model
  also tagged; this is recall);
- FPR be the false positive rate on the gold set (the share of hand-coded negatives the model
  tagged).

The corrected share is:

```text
corrected share = (p − FPR) / (TPR − FPR)
```

For example, if the model tags 30% of the corpus, catches 80% of the true positives and wrongly
tags 10% of the true negatives, the corrected share is (0.30 − 0.10) / (0.80 − 0.10) ≈ 0.29.

The estimate is refused (no number is given) when:

- TPR − FPR is below 0.05, or TPR is below FPR: the model carries no usable signal about the
  share;
- the result falls outside 0 to 1, which means the correction is not trustworthy here. It is
  not clipped.

## PPI

Prediction-powered inference uses each gold comment's (hand, machine) pair directly:

```text
PPI share = p + weighted mean over the gold set of (hand label − machine label)
```

where each label is 1 or 0 for the category. It never divides by TPR − FPR, so it cannot blow up
when the two are close. Its estimate is clamped to 0 to 1. When an interval bound sits at 0 or
1, the output says how many resamples were clamped, because that bound is an artefact of the
clamp, not a measured limit.

The two estimators should broadly agree. If they do not, look at the effective counts below
before reporting either.

## Weights

Gold rows are weighted by `1 / inclusion_prob` (Horvitz-Thompson), in TPR, FPR and the PPI
term, so they describe the corpus rather than the stratified draw. The corpus side counts each
answered comment once. Parse failures leave both sides.

The evidence behind a corrected figure is the **effective** number of hand-coded positives: the
Kish effective sample size of the positives' weights, (sum of weights)² / (sum of squared
weights). A positive found in a large, lightly sampled stratum carries a large weight, and a
few such rows are worth far fewer than their raw count. The output prints both, and the raw
and effective counts of false positives, false negatives and gold negatives.

## When a figure is refused or flagged

| Rule | Threshold | Flag |
|---|---|---|
| Effective positives | below 10: refused | `--min-genuine-positives` (default 10) |
| Low evidence | 10 to 49.9: reported with `LOW EVIDENCE` on the same line as the figure | fixed at 50 (`WELL_SUPPORTED_POSITIVES` in `report.py`) |
| Gold videos | fewer than 2 distinct videos in the gold set: refused | `--min-gold-clusters` (default 2) |
| Degenerate resamples | more than 5% of Rogan-Gladen resamples have no valid correction: refused | `--max-discard-fraction` (default 0.05) |

A refusal is a result, not a bug: it means the gold set cannot support a number. Report the
observed share without a correction, and say why. Do not lower a threshold to get a figure you
intend to publish.

## Clustered bootstrap

Comments under one video share an audience and a thread, so they are not independent.
Every interval resamples whole videos (the `source` column; change it with `--cluster-by`, but
anything finer than the video understates the interval). The corrected intervals resample the
corpus **and** the gold set together, both by video, and recompute TPR, FPR and the correction
on every draw. `--n-boot` sets the number of resamples (default 2,000) and `--seed` the random
seed.

## Pooling across stories

`aggregate` weights each story equally: it computes each story's rate, averages the four, and
resamples videos within each story. The stories were chosen as cases, not sampled in proportion
to anything, so a comment-weighted average would mostly reflect whichever story drew the most
comments. Equal weighting applies across stories only, never across harm categories.

`corrected-bootstrap` and `ppi-bootstrap` pool every comment (each comment counts once) and use
one TPR and FPR for the whole corpus. That assumes the model's errors are the same across
stories. `corrected-by-story` drops that assumption by correcting each story on its own, which
needs enough gold rows in each story; with a thin gold set most stories are refused.

## Two denominators: `--base`

| Value | Share of |
|---|---|
| `all` (default) | every comment the model answered, including those it tagged with no harm |
| `classified` | only comments the model tagged with at least one harm |
| `both` | prints the two under their own headings |

On the `classified` base the gold rows are restricted to the same population, so the corrected
figure is the true share among comments the model tagged. Because a comment can carry two
labels, shares on that base can add up to more than 100%. `corrected --tpr/--fpr` refuses the
`classified` base, because those rates describe the other population. A file with no
`domain_unclassified` column is refused for it.

## Weights sensitivity: `--rescale-unclassified`

```bash
ewl-report corrected-bootstrap official-data/master_labelled.csv \
    --column domain_democracy --gold official-data/gold_labelled.csv \
    --rescale-unclassified official-data/master_kw_primary.csv
```

This scales the weights of the gold rows in each story's unclassified stratum (`sampling_stratum`
of `<story>` or `<story>/unclassified`) so that they sum to that stratum's size in the given
corpus, which must have `story` and `primary_category` columns. Other strata are unchanged. It
prints the factor used for each story.

It is a sensitivity run, not a correction: it assumes the corpus strata sizes are right and does
not explain why the stored weights differ. Run each corrected figure with and without it and
report both. It works on `corrected-bootstrap`, `ppi-bootstrap` and `corrected-by-story`.

## Worked examples

The gold comments shown to the model as worked examples are left out of the correction, read
from the gold file's run record. `--exclude-ids FILE` adds more. See
[Validation](validation.md#worked-examples-are-left-out).
