"""pytest support.

`test_harness.py` is a script, not a set of test functions: it runs its ~500 checks at
import time and prints PASS/FAIL, which is why `pytest tests` used to collect nothing
(or import the whole thing at collection). pytest is told to leave it alone here, and
`test_harness_entry.py` runs it as a program and fails if any check does.
"""
collect_ignore = ["test_harness.py"]
