"""Lets `pytest tests` run the harness, as one test that fails if any check fails."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_harness_passes():
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "test_harness.py")],
                          capture_output=True, text=True, env=env, cwd=os.path.dirname(ROOT))
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])
    assert proc.returncode == 0 and "All checks that could run passed." in proc.stdout, tail
