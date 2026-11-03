"""Exact token counts for every prompt this project sends, against each model's
prompt-caching minimum. Answers "will this prompt actually cache?" with a real
count instead of the character-based estimate.

    python scripts/scratch_count_prompt_tokens.py
    python scripts/scratch_count_prompt_tokens.py --fixed-examples official-data/fixed_examples.csv
    python scripts/scratch_count_prompt_tokens.py --model claude-sonnet-5-5 --model claude-opus-5-5

Needs `pip install -e ".[anthropic]"` and ANTHROPIC_API_KEY set. Anthropic's
token-counting endpoint is not billed (it is rate-limited). Only the fixed prompts are
sent, but with --fixed-examples they CONTAIN the 25 real gold comments, so the comment
text of those 25 does reach Anthropic (the API arms send comments there anyway).

Why this matters: a prompt under the model's minimum still works. Nothing fails
and no warning appears; every call is simply billed at the full input price
instead of about a tenth of it. `slens-classify` prints an estimate-based warning,
but the estimate (4 characters per token) can be ~15% out, and the sentiment
prompts sit close enough to Sonnet's 1,024-token minimum for that to matter.

Each count includes a one-word user message, so the cached system prompt itself
is a few tokens smaller than the figure shown. Anything within ~3% of the
minimum is marked TIGHT for that reason.
"""
import argparse
import sys

from sentiment_lens import taxonomy
from sentiment_lens.label import combined, judges
from sentiment_lens.label import sentiment as sentiment_mod

DEFAULT_MODELS = ["claude-sonnet-5-5", "claude-opus-5-5"]
TIGHT_MARGIN = 1.03


def all_prompts():
    """{label: prompt text}, built by the project's own code, not copied."""
    prompts = {}
    for story in sentiment_mod.STORY_CONTEXT:
        prompts[f"domain / {story}"] = taxonomy.build_prompt(taxonomy.FIXED_EXAMPLES, story)
    # A scorer over the stub judge needs no SDK and no key, and builds exactly the
    # per-story prompts the real LLM scorer sends.
    scorer = sentiment_mod.create("llm:stub", votes=1)
    for story, text in scorer._prompts.items():
        prompts[f"sentiment / {story}"] = text
    for story in sentiment_mod.STORY_CONTEXT:
        prompts[f"combined / {story}"] = combined.build_prompt(taxonomy.FIXED_EXAMPLES, story)
    return prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append",
                        help=f"Model to count for (repeatable). Default: {', '.join(DEFAULT_MODELS)}")
    parser.add_argument("--fixed-examples", metavar="FILE",
                        help="count the prompts as they will be sent, with these fixed examples "
                             "(and their ratings and stories) loaded, not the built-in set")
    args = parser.parse_args()
    models = args.model or DEFAULT_MODELS
    if args.fixed_examples:
        taxonomy.load_fixed_examples(args.fixed_examples)
        print(f"Counting with the {len(taxonomy.FIXED_EXAMPLES)} fixed examples in "
              f"{args.fixed_examples} (ratings: {'yes' if taxonomy.FIXED_EXAMPLE_RATINGS else 'no'}).")

    try:
        import anthropic
    except ImportError:
        sys.exit('The anthropic package is not installed. Run: pip install -e ".[anthropic]"')
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    prompts = all_prompts()
    problems = 0
    for model in models:
        minimum = judges.cache_minimum_for(model)
        print(f"\n{model}  (cache minimum: {minimum if minimum else 'unknown'} tokens)")
        print(f"  {'prompt':<30}{'estimated':>10}{'exact':>8}{'headroom':>10}   verdict")
        for label, text in prompts.items():
            exact = client.messages.count_tokens(
                model=model, system=text,
                messages=[{"role": "user", "content": "x"}]).input_tokens
            if minimum is None:
                verdict, headroom = "no minimum on record", ""
            else:
                headroom = f"{exact - minimum:+d}"
                if exact < minimum:
                    verdict, problems = "WILL NOT CACHE", problems + 1
                elif exact < minimum * TIGHT_MARGIN:
                    verdict, problems = "TIGHT", problems + 1
                else:
                    verdict = "ok"
            print(f"  {label:<30}{judges.estimate_tokens(text):>10}{exact:>8}{headroom:>10}   {verdict}")

    if problems:
        print(f"\n{problems} prompt/model pair(s) will not cache or are too close to call. "
              f"Cached prompts are billed at about a tenth of the input price on every call "
              f"after the first, so this is worth fixing before a long run: add substantive "
              f"content (for sentiment, worked examples are the natural addition), do not pad.")
        return 1
    print("\nEvery prompt clears its model's caching minimum.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
