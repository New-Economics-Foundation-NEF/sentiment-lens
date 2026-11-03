# sentiment-lens: social sentiment labelling

`sentiment-lens` labels social-media comments about extreme wealth. For each comment it records a harm
domain (`democracy`, `economics`, `environment` or `social`, up to two, or `unclassified`) and
a stance toward extreme wealth (`negative`, `neutral`, `positive`, `irrelevant` or
`uncodeable`). The labels come from a cloud LLM, a local model or a keyword list. The package
then scores the labels against a hand-coded gold set, corrects the headline shares for the
model's errors (Rogan-Gladen, with PPI as a check), with Horvitz-Thompson weights and intervals
from a bootstrap clustered by video, and exports the results for a dashboard. It was built for
the Extreme Wealth Line project on YouTube comments.

**Documentation:** <https://new-economics-foundation-nef.github.io/sentiment-lens/>

## Install

Python 3.9 or later.

```bash
git clone https://github.com/New-Economics-Foundation-NEF/sentiment-lens.git
cd sentiment-lens
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Optional extras enable the model backends and other features, for example
`pip install -e ".[anthropic]"`. See the
[installation page](https://new-economics-foundation-nef.github.io/sentiment-lens/installation/)
for the full list and the environment variables.

## Quickstart

This runs on invented comments, with no API key and no network:

```bash
OUT=$(mktemp -d)
slens-classify examples/synthetic_comments.csv --judge keyword --no-cache --output "$OUT/out.csv"
```

It writes `out.csv` (the comments with their labels) and `out.csv.run.json` (a record of how
they were produced). The
[quickstart](https://new-economics-foundation-nef.github.io/sentiment-lens/quickstart/)
continues with scoring and correction.

## Data protection

Comment text, comment ids and author hashes are personal data under UK GDPR.

- Never commit them. Keep all comment data in a git-ignored local folder.
  `tests/check_no_personal_data.py` runs in CI and fails the build if a tracked file looks like
  comment data.
- Cloud judges send comment text to the model provider. OpenAI and Google judges refuse to
  start unless `SLENS_EXTERNAL_PROCESSORS` names the provider; the Anthropic judge is not gated.
  Get data-protection sign-off first.
- Collectors need `SLENS_AUTHOR_SALT`, a secret kept outside the repository.

Read the
[data protection page](https://new-economics-foundation-nef.github.io/sentiment-lens/data-protection/)
before you collect or label real comments.

## Tests

```bash
python tests/test_harness.py
python tests/check_no_personal_data.py
```

## Licence

Licence: to be decided by NEF.
