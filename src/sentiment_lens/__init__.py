"""Extreme Wealth Line: collecting and labelling social media comments.

The pipeline runs in three stages, and the package is laid out to match:

    sentiment_lens.collect   YouTube and Reddit collectors, and the merge into one corpus
    sentiment_lens.label     harm-domain and sentiment labelling, and the classifier arms
    sentiment_lens.analyse   scoring against hand codes, and the reported statistics

`sentiment_lens.taxonomy` holds the four harm categories, the prompt and the parser --
defined once, imported by every labelling path. `sentiment_lens.common` holds the CSV
schema and the collection-time helpers both collectors share.

Every stage has a command-line entry point (see pyproject.toml): slens-collect,
slens-merge, slens-classify, slens-keywords, slens-evaluate, slens-validate, slens-report.
"""
__all__ = ["common", "taxonomy", "collect", "label", "analyse"]
