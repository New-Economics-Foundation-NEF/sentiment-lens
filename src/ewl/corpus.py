"""Freezing a collected corpus, and checking one has not moved under you.

Collection and analysis are different kinds of work and this module is the
boundary between them. Collection costs quota, depends on a rate-limited
external API, is slow, and is **irreversible** -- yesterday's comments cannot
be collected again, and a video deleted next week is gone. Analysis is free,
fast, and will be run dozens of times as the classifier changes.

So a corpus is written once and then frozen. `ewl-merge` records a manifest
beside it; every analysis command can check the corpus still matches, and stop
if it does not. Without that, re-collecting silently changes what an already
published figure refers to, and nothing in the output would say so.

The manifest is deliberately small and human-readable: it is evidence, and
evidence nobody can read is not much use.
"""
import csv
import datetime
import hashlib
import json
import os
import sys

MANIFEST_SUFFIX = ".manifest.json"


def manifest_path(corpus_path):
    return str(corpus_path) + MANIFEST_SUFFIX


def content_hash(corpus_path):
    """SHA-256 of the file's bytes, streamed so a 30,000-row corpus does not
    have to be held in memory."""
    digest = hashlib.sha256()
    with open(corpus_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarise(corpus_path):
    """Row, story and video counts, read straight off the corpus."""
    stories, videos, rows = {}, set(), 0
    with open(corpus_path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            stories[row.get("story", "")] = stories.get(row.get("story", ""), 0) + 1
            if row.get("source"):
                videos.add(row["source"])
    return {"comments": rows, "videos": len(videos),
            "comments_per_story": dict(sorted(stories.items()))}


def write_manifest(corpus_path, inputs=None, extra=None):
    """Freeze `corpus_path`: record what it contains and what it hashes to.

    `EWL_AUTHOR_SALT` is fingerprinted, never stored: two corpora hashed with
    different salts have incomparable `author_hash` columns and must not be
    merged, and that is worth being able to detect without putting the secret
    in a file that gets committed.
    """
    salt = os.environ.get("EWL_AUTHOR_SALT", "")
    if not salt:
        # Not an error -- merging already-collected files never needs the
        # salt itself (see the module docstring above `hash_author` in
        # common.py) -- but a manifest frozen without it can never again
        # detect a future accidental merge across the salt-scheme boundary
        # CLAUDE.md warns about, for THIS corpus specifically. Confirmed to
        # happen silently: the real frozen corpus's own manifest carries
        # `salt_fingerprint: null` because EWL_AUTHOR_SALT was not set in
        # the shell that last ran ewl-merge.
        print("NOTE: EWL_AUTHOR_SALT is not set -- this manifest will record "
              "salt_fingerprint: null, which means it can never detect a future "
              "merge across the salt-scheme boundary for this corpus. Harmless if "
              "you are only re-merging already-collected data, but export the "
              "project's real salt first if you want that protection.", file=sys.stderr)
    manifest = {
        "corpus": os.path.basename(str(corpus_path)),
        "frozen_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sha256": content_hash(corpus_path),
        "salt_fingerprint": hashlib.sha256(salt.encode()).hexdigest()[:12] if salt else None,
        "inputs": [os.path.basename(str(p)) for p in (inputs or [])],
    }
    manifest.update(summarise(corpus_path))
    if extra:
        manifest.update(extra)
    path = manifest_path(corpus_path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path, manifest


def read_manifest(corpus_path):
    path = manifest_path(corpus_path)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def verify(corpus_path, strict=False):
    """Check a corpus still matches its manifest.

    Returns (ok, message). A missing manifest is not a failure: corpora
    collected before this existed are still perfectly good data, and refusing
    to analyse them would be a false gate. It is reported, so the absence is
    visible rather than assumed.

    `strict` turns a mismatch into a hard stop. That is the right setting for
    anything whose numbers get published: a corpus that changed after a figure
    was produced means the figure refers to something that no longer exists.
    """
    manifest = read_manifest(corpus_path)
    if manifest is None:
        return True, (f"No manifest for {os.path.basename(str(corpus_path))} -- it was not "
                      f"frozen by ewl-merge, so nothing here can confirm it has not changed.")
    actual = content_hash(corpus_path)
    if actual == manifest.get("sha256"):
        return True, (f"Corpus matches its manifest ({manifest['comments']} comments, "
                      f"{manifest['videos']} videos, frozen {manifest['frozen_at'][:10]}).")
    message = (f"{os.path.basename(str(corpus_path))} has CHANGED since it was frozen on "
               f"{manifest.get('frozen_at', '?')[:10]}.\n"
               f"  manifest sha256: {manifest.get('sha256', '?')[:16]}...\n"
               f"  actual   sha256: {actual[:16]}...\n"
               f"Any figure already produced from this file refers to different data. "
               f"Re-freeze deliberately with ewl-merge, or restore the original.")
    if strict:
        raise SystemExit(message)
    return False, message
