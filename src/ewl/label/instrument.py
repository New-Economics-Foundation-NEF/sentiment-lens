"""Command-line switches for the instrument decisions that are still open.

Each switch defaults to the wording as it stands, so adding the flag changes
nothing until it is used. Every switch changes the prompt, and the prompt is part
of the response-cache key, so a run under one setting never reuses an answer
given under another. The output names the setting: the `sentiment_scorer`
column carries a tag such as `[story_rules=off]`.

    --story-rules {on,off}
        on  = story context paragraphs include their stance rules (as now)
        off = factual context only
    --stance-object {conduct,wealth}
        conduct = anchor 4 as it stands ("defends the wealthy people or companies involved, or how they gain or use their wealth")
        wealth  = anchor 4 about extreme wealth and the people who hold it
    --combined-domain-context {keep,remove}
        keep   = the combined arm's domain question sees the story's FACTS, and only its
                 sentiment question sees the stance rules (as now)
        remove = the context sits only above the sentiment question
    --sentiment-examples {on,off}
        on  = the sentiment prompt (and the combined arm) show the fixed examples WITH their
              hand-coded ratings and stories, when the examples file carries them (as now)
        off = no sentiment examples: the earlier sentiment prompt, byte for byte
    --domain-context {facts,none}
        facts = the standalone domain prompt ends with the story's factual background
                (never the stance rules), as the hand coders knew the video (as now)
        none  = the earlier domain prompt, which never varied by story
"""
import os
import sys

from ewl import taxonomy
from ewl.label import combined, sentiment


def add_fixed_examples_arg(parser):
    parser.add_argument("--fixed-examples", metavar="FILE",
                        help="CSV (comment_id, body, label) of the examples shown to the model in "
                             "the `fixed` arm, instead of the built-in synthetic set. For real "
                             "hand-coded gold comments: keep the file in a gitignored folder. "
                             "They are left out of scoring automatically (each would be scored "
                             "against its own label), and run records and the dashboard codebook "
                             "hold their ids and a hash, never their text.")


def add_args(parser):
    group = parser.add_argument_group(
        "instrument switches (defaults reproduce the current wording)")
    add_fixed_examples_arg(group)
    group.add_argument("--story-rules", choices=["on", "off"], default="on",
                       help="off = story context is factual only, with no stance rules")
    group.add_argument("--stance-object", choices=["conduct", "wealth"], default="conduct",
                       help="wealth = anchor 4 is about extreme wealth and those who hold it")
    group.add_argument("--sentiment-examples", choices=["on", "off"], default="on",
                       help="off = the sentiment prompt shows no worked examples. With "
                            "--fixed-examples FILE holding `sentiment` and `story` columns, on "
                            "(default) shows the same fixed examples with their ratings")
    group.add_argument("--domain-context", choices=list(taxonomy.DOMAIN_CONTEXT_MODES),
                       default="facts",
                       help="none = the domain prompt carries no story background (the "
                            "earlier wording); facts (default) = it ends with the story's "
                            "factual paragraph, never the stance rules")
    group.add_argument("--combined-domain-context", choices=["keep", "remove"],
                       default="keep",
                       help="remove = the combined arm's domain question does not see the "
                            "story context")


def apply(args, needs_ratings=False):
    """Set the switches from parsed arguments. Call before building any scorer.

    needs_ratings: the command asks the sentiment question (ewl-gemma, `ewl-evaluate
    sentiment` and `scorecard`), so an examples file without ratings is refused here,
    where every such command passes, unless --sentiment-examples off says it is meant.
    """
    if getattr(args, "fixed_examples", None):
        taxonomy.load_fixed_examples(args.fixed_examples)
        source = taxonomy.FIXED_EXAMPLES_SOURCE
        if (needs_ratings and source.get("source") == "file" and not source.get("ratings")
                and getattr(args, "sentiment_examples", "on") != "off"):
            print(f"Examples: {len(source.get('ids', []))} real gold comments from "
                  f"{os.path.basename(source['path'])}, NO ratings", file=sys.stderr)
            sys.exit("This examples file has no `sentiment` and `story` columns, so every "
                     "sentiment question would run WITHOUT worked examples (the pre-14-October "
                     "prompt) and `ewl-batches check` would still pass. Use the final file "
                     "from the project lead, or pass --sentiment-examples off if that is really intended.")
    sentiment.configure(story_rules=(getattr(args, "story_rules", "on") == "on"),
                        stance_object=getattr(args, "stance_object", "conduct"),
                        sentiment_examples=(getattr(args, "sentiment_examples", "on") == "on"))
    combined.configure(domain_context=getattr(args, "combined_domain_context", "keep"))
    taxonomy.configure_domain_context(getattr(args, "domain_context", "facts"))
    return sentiment.config_tag() + combined.config_tag() + taxonomy.domain_context_tag()
