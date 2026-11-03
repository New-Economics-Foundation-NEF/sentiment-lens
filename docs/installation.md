# Installation

## Requirements

- Python 3.9 or later.
- About 100 MB of disk space for the core install. The optional `retrieval` extra pulls in
  PyTorch and is much larger.
- For the local Gemma arm: [Ollama](https://ollama.com) and enough memory to run Gemma 2 27B
  (a 16 GB model file).

## Install the package

Clone the repository, make a virtual environment and install the package in editable mode:

```bash
git clone https://github.com/New-Economics-Foundation-NEF/ewl-social-sentiment.git
cd ewl-social-sentiment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

This installs the `ewl` package and its command-line tools (`ewl-classify`, `ewl-report` and
the rest; see the [command reference](cli.md)). The core install is enough to run the
[quickstart](quickstart.md): the keyword judge, the VADER sentiment scorer, validation and
correction all work without any optional extra.

!!! tip "Command not found on Windows"
    pip sometimes puts the console scripts in a `Scripts` folder that is not on `PATH`. The
    install output names the folder. Every command also runs as a module, for example
    `python -m ewl.label.classify` for `ewl-classify`. The module for each command is listed
    under `[project.scripts]` in `pyproject.toml`.

## Optional extras

Each extra enables one part of the package. A missing extra turns off that part only; nothing
else breaks.

| Extra | Install | What it enables |
|---|---|---|
| `anthropic` | `pip install -e ".[anthropic]"` | The `anthropic:<model>` judge, `llm:anthropic:<model>` sentiment scorer and `ewl-anthropic-batch`. |
| `openai` | `pip install -e ".[openai]"` | The `openai:<model>` judge and scorer. |
| `google` | `pip install -e ".[google]"` | The `google:<model>` (Gemini) judge and scorer. |
| `ollama` | `pip install -e ".[ollama]"` | The local `ollama:<model>` and `gemma` judges, and `ewl-gemma`. |
| `retrieval` | `pip install -e ".[retrieval]"` | The `retrieved` example selector (sentence-transformers). The first use downloads a model of about 90 MB with no progress bar. |
| `language` | `pip install -e ".[language]"` | Language detection: the `detected_lang` column and `--lang` on `ewl-collect`. |
| `analysis` | `pip install -e ".[analysis]"` | Reading coders' `.xlsx` files (`ewl-validate interrater`/`combine`, `ewl-adjudicate`, `scripts/`). Also installs matplotlib. |
| `docs` | `pip install -e ".[docs]"` | MkDocs Material, to build this site. |

You can combine extras: `pip install -e ".[anthropic,language]"`.

The `anthropic` extra is capped below version 2 and `openai` below version 4.

## Environment variables

Set secrets and settings in your shell, or in a file outside the repository that you source
before a run (for example `~/.config/ewl.env`, readable only by you). Never commit them and
never paste them into an issue or chat. Variables do not survive a new terminal: set them
again each session.

### Required for collection

| Variable | What it does |
|---|---|
| `EWL_AUTHOR_SALT` | The secret mixed into every author name before it is hashed into `author_hash`. There is no default on purpose: a salt that ships in a public repository is not a secret. The collectors stop if it is not set or is shorter than 16 characters; use a long random string. Agree one value for a whole project and never change it, or old and new `author_hash` values cannot be compared. `ewl-merge` records a fingerprint of it (never the value) in the corpus manifest. |
| `YOUTUBE_API_KEY` | Key for the YouTube Data API v3, used by `ewl-collect`. |

### API keys for cloud judges

The SDKs read these themselves. `ewl` never takes a key as a command-line argument, so a key
cannot reach your shell history or a log.

| Variable | Used by |
|---|---|
| `ANTHROPIC_API_KEY` | `anthropic:` judges and scorers, `ewl-anthropic-batch` |
| `OPENAI_API_KEY` | `openai:` judges and scorers (read by the OpenAI SDK) |
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `google:` judges and scorers (read by the google-genai SDK) |
| `OLLAMA_HOST` | Address of an Ollama server that is not on this machine (`ewl-gemma --host` sets it) |

### Approval to send comments to a provider

| Variable | What it does |
|---|---|
| `EWL_EXTERNAL_PROCESSORS` | A comma-separated list of the cloud providers you have approved to receive comment text, for example `openai,google`. The `openai:` and `google:` judges refuse to start unless their provider is listed, and nothing is sent. Set it only once your data-protection sign-off is recorded. The `anthropic:` judge is **not** gated by this variable. See [Data protection](data-protection.md). |

### Model settings

The settings that change what the model is asked (effort, thinking and temperature) are part
of the response-cache key, so answers given under one setting are never replayed under
another. Timeouts, prices and the Gemma context window and keep-alive are not part of the key.
The run record shows which settings were used.

| Variable | Values | What it does |
|---|---|---|
| `EWL_ANTHROPIC_TIMEOUT` | seconds, default `60` | Abandons an Anthropic call that gets no reply after this long, and retries it. |
| `EWL_ANTHROPIC_EFFORT` | `low`, `medium`, `high`; unset by default | Sends an effort setting with every Anthropic request. Not confirmed against a live call. |
| `EWL_ANTHROPIC_ALLOW_THINKING` | `1`, `true` or `yes` | Lets a Claude model whose thinking cannot be turned off run with thinking on, with a larger reply limit. Has no effect on models where thinking can be turned off. |
| `EWL_ANTHROPIC_PRICES` | `model=input,output` | Price (USD per million tokens) for a Claude model that is not in the built-in table. An unpriced model is refused before anything is sent. |
| `EWL_OPENAI_REASONING_EFFORT` | for example `low`, `medium`, `high`; default `minimal` | Reasoning effort for OpenAI reasoning models. A model that rejects it stops the run. Refused for a model that does not reason. |
| `EWL_OPENAI_PRICES` | `model=input,output[,cached]` | Price for an OpenAI model not in the built-in table. The optional third figure is the cached-input price. |
| `EWL_GEMINI_THINKING_LEVEL` | `minimal`, `low`, `medium`, `high`; unset by default | Thinking level for Gemini. Unset sends nothing and the model uses its own default (thinking on). A model that rejects the level stops the run. |
| `EWL_GEMINI_THINKING_BUDGET` | — | No longer used. The judge refuses to start if it is set. |
| `EWL_GOOGLE_PRICES` | `model=input,output` | Price for a Gemini model not in the built-in table. |
| `EWL_GEMMA_TEMPERATURE` | number from 0 to 2, default `0` | Temperature for the `gemma` judge. At 0 the judge runs one vote only, because repeat votes would be identical. Above 0 the votes differ (seeds 42, 43, 44), which is a different instrument: use its own cache file. `ewl-gemma` refuses a value above 0. |
| `EWL_GEMMA_NUM_CTX` | tokens, 4096 to 8192, default `8192` | Context window sent to Ollama for Gemma. A smaller window makes Ollama drop the start of the prompt. |
| `EWL_GEMMA_KEEP_ALIVE` | Ollama duration, default `4h` | How long Ollama keeps the Gemma model loaded between calls. |

## Check the install

```bash
ewl-classify --help
```

Then run the [quickstart](quickstart.md).
