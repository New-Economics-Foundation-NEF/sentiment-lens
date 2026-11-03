"""Collectors, and the merge into a single corpus.

Every collector writes the same columns in the same order (`ewl.common.
FIELDNAMES`), so per-video and per-thread files concatenate without
hand-editing. `merge` is the only thing that should do that concatenation: it
validates the schema and drops comments already seen.
"""
