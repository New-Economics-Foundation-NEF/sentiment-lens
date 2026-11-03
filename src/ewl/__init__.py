"""Extreme Wealth Line: collecting and labelling social media comments.

The pipeline runs in three stages, and the package is laid out to match:

    ewl.collect   YouTube and Reddit collectors, and the merge into one corpus
    ewl.label     harm-domain and sentiment labelling, and the classifier arms
    ewl.analyse   scoring against hand codes, and the reported statistics

`ewl.taxonomy` holds the four harm categories, the prompt and the parser --
defined once, imported by every labelling path. `ewl.common` holds the CSV
schema and the collection-time helpers both collectors share.

Every stage has a command-line entry point (see pyproject.toml): ewl-collect,
ewl-merge, ewl-classify, ewl-keywords, ewl-evaluate, ewl-validate, ewl-report.
"""
__all__ = ["common", "taxonomy", "collect", "label", "analyse"]
