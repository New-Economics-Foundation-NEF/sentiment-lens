"""Labelling: harm domain and sentiment.

Deliberately split into what varies independently, so the scorecard can hold
one fixed while changing another:

    judges     the model behind a label -- a new backend is one `_sample`
    examples   which few-shot examples a comment is shown
    sentiment  the sentiment scorer, VADER or an LLM rating
    combined   one call answering both questions, as a testable arm
    keywords   the deterministic rule-based classifier and its audit
    classify   the CLI that runs a chosen combination over a corpus
"""
