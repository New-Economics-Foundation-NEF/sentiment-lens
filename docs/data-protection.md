# Data protection

Read this page before you collect or label any real comments.

## What is personal data

Under UK GDPR, the following are personal data:

- **comment text** (`body`): a comment can be searched for and traced to its author;
- **comment ids** (`comment_id`): each one points to a specific person's comment on the
  platform;
- **author hashes** (`author_hash`): a pseudonym is still personal data, and a salted hash can
  be linked across files.

Every file that holds any of these is personal data. That includes the collected and merged
corpus, every labelled output, the gold sample and the coders' files, the review files from
`slens-adjudicate`, the batch inputs and outputs, `subset.csv` from `slens-anthropic-batch`, the
response caches, and a fixed examples file. Run records hold comment ids when a real examples
file is used.

## Never commit it

Keep all of it in a git-ignored local data folder. The project used `official-data/`. The
repository's `.gitignore` excludes `official-data/`, `data/`, `archive/`, `*.csv`, spreadsheets,
`*.jsonl` caches, `*.bak` files and `*.run.json` run records.

`.gitignore` does not stop `git add -f` or a file uploaded through GitHub's web page. So
`tests/check_no_personal_data.py` checks every tracked file, and it runs in CI on every push
and pull request. It fails the build on:

- any tracked file under `official-data/`, `data/`, `archive/` or `Full Corpus Data/`;
- any tracked spreadsheet, JSONL cache, backup or run record;
- any CSV outside `tests/` whose header has `body` together with `comment_id` or
  `author_hash`;
- any text file outside `tests/` with three or more YouTube comment ids;
- a dashboard export made with `--include-verbatim`.

It prints only paths and counts. Run it yourself before you push:

```bash
python tests/check_no_personal_data.py
```

Do not paste comment text, ids or handles into an issue, a pull request, a chat or a model
prompt you control by hand. Report counts and figures instead.

## The author salt

Collectors never write an author's name. They write `author_hash`, a SHA-256 hash of the name
mixed with a secret salt from `SLENS_AUTHOR_SALT`. There is no default: a salt published in the
repository would let anyone hash a guessed name and match it. The collectors stop if the
variable is not set, or is shorter than 16 characters.

- Keep the salt outside the repository, for example in `~/.config/sentiment_lens.env` readable only by
  you.
- Agree one value for a project and never change it. A different salt makes `author_hash`
  values incomparable, so never merge corpora collected under different salts.
- `slens-merge` records a short fingerprint of the salt in the manifest, never the salt itself, so
  a merge across salts can be detected.

## Cloud judges send comment text to a third party

A cloud judge or scorer sends each comment's text, and the worked examples in the prompt, to the
model provider. That provider becomes a further processor of personal data. Get
data-protection sign-off before you use one.

| Backend | Where text goes | What the code does |
|---|---|---|
| `openai:` | OpenAI | Refuses to start unless `SLENS_EXTERNAL_PROCESSORS` lists `openai`. Nothing is sent. |
| `google:` | Google | Refuses to start unless `SLENS_EXTERNAL_PROCESSORS` lists `google`. Nothing is sent. |
| `anthropic:` and `slens-anthropic-batch` | Anthropic | **Not gated.** The code sends text as soon as `ANTHROPIC_API_KEY` is set. Record the sign-off before you run it. Anthropic keeps batch results for 29 days. |
| `ollama:`, `gemma`, `slens-gemma` | nowhere: runs on your machine | Not gated. Make sure `OLLAMA_HOST` (or `--host`) points at a machine you control. |
| `keyword`, `stub` | nowhere | — |
| `manual` | wherever you paste the prompt | You are the processor: do not paste real comments into a service without sign-off. |

Set `SLENS_EXTERNAL_PROCESSORS` (comma-separated, for example `openai,google`) only once the
sign-off is recorded.

`scripts/scratch_count_prompt_tokens.py --fixed-examples FILE` sends the prompts, including the
example comments, to Anthropic's token-counting endpoint.

## Redaction

`slens-redact` finds, and optionally replaces, three kinds of identifier in comment text:

- `@handles` (`--handles leading` for a handle that opens the comment, `--handles all` for every
  one), replaced with `@user` (change it with `--handle-token`);
- email addresses (`--emails`), replaced with `[email]`;
- phone numbers (`--phones`), replaced with `[phone]`.

```bash
slens-redact official-data/master.csv                 # counts only, nothing written
slens-redact official-data/master.csv --handles all --emails --phones \
    --output official-data/master_redacted.csv
```

It writes a new file with its own manifest and never changes the input. Run it **before**
labelling: the cache is keyed on the text, and a cloud judge would already have received the
handles. It is a pattern match, not a guarantee. A name written in prose is not found, and URLs
are not redacted. Read a sample of the output.

## Publishing results

`slens-export-dashboard` writes no comment text by default. The worked examples and the table of
machine-versus-human disagreements are included only with `--include-verbatim`, and the export
records that it was used. The codebook publishes the prompts and, for a real examples file, only
a count and a checksum. Publish figures, not comments.
