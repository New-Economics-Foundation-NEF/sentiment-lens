# Development

## Layout

```text
src/sentiment_lens/
  common.py        the comment schema, author hashing, language detection
  taxonomy.py      the four harm categories, the domain prompt, the label parser, the built-in examples
  keywords.yml     the keyword vocabulary
  corpus.py        corpus manifests: hash, counts, verify
  collect/         collectors, merge, clean, redact, batches, primary category
  label/           judges, example selectors, sentiment scorers, the combined arm,
                   instrument switches, slens-classify, slens-gemma, slens-anthropic-batch
  analyse/         validate, evaluate, agreement, adjudicate, report, export_dashboard
tests/             the test harness and the personal-data check
scripts/           one-off helpers (see below)
examples/          invented comments for the quickstart
docs/              this site
```

## Running the tests

The harness needs no model, no network and no API key. It uses the `stub` and `keyword`
judges.

```bash
pip install -e ".[analysis,language]"
python tests/test_harness.py
```

It prints one `PASS` or `FAIL` line per check and ends with `All checks that could run
passed.` Blocks that need an optional package that is not installed (for example
`sentence-transformers` or `openai`) print `SKIP` and are named in the summary; that is
expected.

`pytest` runs the same harness as one test:

```bash
pytest tests
```

### Project-file checks that skip here

Fourteen checks, in five blocks, test files that belong to the original project repository
rather than the package: the dashboard page (`dashboard/index.html` and
`dashboard/data/results.json`) and the video exclusion list (`excluded_video_ids.txt`). In this
repository those files do not exist, so each block prints

```text
SKIP  checks on dashboard/index.html (a project file, not part of the package)
```

and the run still passes. In the project repository they run as before.

### The personal-data check

```bash
python tests/check_no_personal_data.py
```

It scans the files git tracks and fails if any looks like comment data. See
[Data protection](data-protection.md#never-commit-it) for what it checks. Run it before every
push.

### Continuous integration

`.github/workflows/test.yml` runs on every push and pull request, on Python 3.9 and 3.12. It
byte-compiles `src` and `tests`, runs the personal-data check, runs the harness, and runs
`--help` on every console script to catch a broken entry point.

`.github/workflows/docs.yml` builds this site with `mkdocs build --strict` on pull requests, and
builds and deploys it to GitHub Pages on a push to `main`.

## Building the docs

```bash
pip install -e ".[docs]"
python docs/gen_cli.py          # regenerate docs/cli.md after changing any command's flags
mkdocs build --strict
```

`docs/cli.md` is generated from each command's `--help` output. Do not edit it by hand: change
the command's argparse help, then run `docs/gen_cli.py` and commit both.

## Adding a judge backend

A backend is one class with one method. The base class `Judge` in `src/sentiment_lens/label/judges.py`
owns the prompt, voting, parsing, the cache and usage accounting, so a new backend only supplies
the call that returns one raw reply.

```python
from sentiment_lens.label.judges import Judge, register


@register("mybackend")
class MyJudge(Judge):
    """One line on what this runs."""

    default_model = "my-model-name"   # used when the spec has no ":<model>"
    deterministic = False             # True only if repeat runs agree exactly

    def _sample(self, body, prompt, vote):
        # prompt: the full system prompt (instructions, examples, story background)
        # body:   the comment text
        # vote:   0-based vote index; use it to vary a seed
        reply = call_my_model(system=prompt, user=f'Classify this comment: "{body}"', seed=42 + vote)
        with self._lock:
            self.usage["input_tokens"] += reply.input_tokens
            self.usage["output_tokens"] += reply.output_tokens
        return reply.text
```

Then `--judge mybackend:<model>`, `--sentiment llm:mybackend:<model>` and
`combined:mybackend:<model>` all work.

Points to get right:

- **Register it where it is imported.** The registry fills when `judges.py` is imported. Put
  the class in `judges.py`, or in its own module imported from `_load_optional_backends()` (as
  `gemma_local.py` is), so `create()` can find it.
- **Import optional SDKs inside `__init__`**, not at the top of the module, so the core install
  still works without them. Add the SDK as an extra in `pyproject.toml`.
- **Refuse early.** Raise `PriceUnavailable`, `SettingRefused` or call
  `require_external_processor("<provider>")` in `__init__`, before anything is sent. A cloud
  provider that receives comment text must be gated this way.
- **Do not cache a non-answer.** Return `uncacheable(text, reason)` for a refusal or a reply cut
  off at the token limit, so it is used once and not replayed forever.
- **Put every request setting in the cache key.** If the backend reads a setting that changes
  the answer (an effort level, a temperature), override the `cache_spec` property to append it,
  as the Anthropic, OpenAI, Gemini and Gemma judges do.
- **Cost.** Set `price_per_mtok` (USD per million input and output tokens) and update
  `self.usage` under `self._lock` so `--max-cost` and the cost columns work. A local backend
  can leave the price at zero.
- **Never take an API key as an argument.** Let the SDK read it from the environment.
- Add harness checks for the new backend, with a fake client, so they run with no network.

## Helper scripts

`scripts/` holds one-off helpers. They are not installed as commands; run them with
`python scripts/<name>.py`. Each prints counts only, never comment text. Reading `.xlsx` files
needs the `analysis` extra.

| Script | What it does |
|---|---|
| `scratch_check_coder_files.py` | Checks coders' completed files before agreement is computed: columns, blank or repeated ids, core coverage, off-list domain values and invalid ratings. Exits 1 if anything needs fixing. |
| `scratch_fill_comment_ids.py` | Fills blank `comment_id`s in a coded sheet by matching `body` against the original coding sheets. Leaves a row blank, and lists it, when the match is missing or ambiguous. |
| `scratch_zero_usage.py` | Shows how each coder used stance `0` (irrelevant): with or without a harm domain, by story, and how coders' zeros overlap. |
| `scratch_count_prompt_tokens.py` | Exact token counts for every prompt against each Claude model's prompt-caching minimum. Needs the `anthropic` extra and `ANTHROPIC_API_KEY`; with `--fixed-examples` it sends the example comments to Anthropic's token-counting endpoint. |
