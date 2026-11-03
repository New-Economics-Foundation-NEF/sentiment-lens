"""Split a corpus into batches for a slow local model, and check the labelled batches
came back complete and consistent before joining them.

    ewl-batches prepare full.csv --exclude gold.csv --exclude examples.csv --output corpus.csv
    ewl-batches verify corpus.csv --full full.csv --exclude gold.csv --exclude examples.csv
    ewl-batches split corpus.csv --out-dir official-data/batches        # 1,000 per batch
    ewl-batches status official-data/batches                             # what is done, what is next
    ewl-batches commands official-data/batches --extra "--fixed-examples official-data/fixed.csv"
    ewl-batches commands official-data/batches --first 5 --extra "..."  # only the next 5 batches
    ewl-batches check  official-data/batches                             # verify every finished batch
    ewl-batches check  official-data/batches --merge-to official-data/labelled_corpus.csv
    ewl-batches check  official-data/batches --merge-to official-data/labelled_so_far.csv --partial

Layout under --out-dir (all of it contains comment text except the manifest: keep it out of
git, which `official-data/` already is):

    batches_manifest.json     the input file's hash and row count, and each batch's hash,
                              row count and original positions. No comment text.
    inputs/batch_001_of_085.csv ...    the batches, header intact, same columns as the corpus.
    labelled/batch_001_of_085.csv      what `ewl-gemma` writes for that batch (plus its
                                       .run.json, which `check` reads).

Why the order is shuffled by default
------------------------------------
The corpus file is ordered by story. Cut in file order, the first batches would be one story
and a run that stopped early would describe that story only. Shuffled with a fixed seed, every
batch is a random slice of all four, so finished batches are always a fair sample of the
whole corpus and the same seed always cuts the same batches. The shuffle is done once and the
shuffled list is then cut into consecutive slices, so the batches are a partition: no comment is
in two batches and together they hold every comment exactly once (a corpus with a repeated
`comment_id` is refused at the split). `--no-shuffle` keeps file order.

What `check` refuses
--------------------
* a labelled batch whose run record names a different input than the batch it sits beside
  (the wrong file, or the batch was edited after the manifest was written);
* a batch whose rows are not exactly the batch's own comments (any missing, repeated or
  extra `comment_id`; rows with no comment text are never labelled and are counted apart);
* batches labelled under different instruments: a different model, prompt, example set or
  switch. A merged file must come from one instrument or no figure from it means anything;
* a batch labelled with `--sample` or `--limit`, which labels only part of it.
With `--merge-to`, it writes the joined file in the ORIGINAL corpus order and a run record for
it that names the whole corpus as its input, so `ewl-export-dashboard --corpus` can check it.

Progress file (`--partial`)
---------------------------
`--merge-to FILE --partial` joins the batches finished SO FAR, after the same checks, while
others are still pending. Its run record carries a `partial` block naming the batches included
and the comment counts, and `ewl-export-dashboard` refuses it with `--corpus` (and flags it
without), so a progress file is never taken for the whole corpus. Once every batch is done,
`--partial` makes no difference: the file is the ordinary full merge.
"""
import argparse
import collections
import csv
import json
import os
import random
import sys

from ewl import corpus as corpus_mod

MANIFEST = "batches_manifest.json"
INPUTS, LABELLED = "inputs", "labelled"
sys.stdout.reconfigure(encoding="utf-8")


def _read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_rows(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _has_text(row):
    return bool((row.get("body") or "").strip())


def batch_name(number, total):
    width = max(3, len(str(total)))
    return f"batch_{number:0{width}d}_of_{total:0{width}d}.csv"


def _ids_from(path):
    """The comment_ids in a CSV with a `comment_id` column: a gold file, the fixed examples,
    or any list of ids to leave out. (A collector never imports the analysis code, so a
    spreadsheet must be saved as CSV UTF-8 first.)"""
    if str(path).lower().endswith((".xlsx", ".xlsm", ".xls")):
        raise SystemExit(f"{path} is a spreadsheet: save it as CSV UTF-8 first.")
    _, rows = _read_rows(path)
    ids = {(r.get("comment_id") or "").strip() for r in rows} - {""}
    if not ids:
        raise SystemExit(f"{path} holds no comment_id values (does it have a `comment_id` column?).")
    return ids


def _norm(text):
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def prepare(full_csv, exclude_paths, output):
    """Write `full_csv` minus every comment_id in `exclude_paths` to `output`, and a
    provenance file beside it. Nothing is edited by hand, so nothing can be altered on the way
    (a spreadsheet can change text, dates and line endings without a word)."""
    if os.path.exists(output):
        raise SystemExit(f"{output} already exists: choose a new name, or remove it first.")
    fieldnames, rows = _read_rows(full_csv)
    for column in ("comment_id", "body"):
        if column not in fieldnames:
            raise SystemExit(f"{full_csv} has no `{column}` column.")
    excluded = {}
    for path in exclude_paths:
        excluded[path] = _ids_from(path)
    leave_out = set().union(*excluded.values()) if excluded else set()
    present = {(r.get("comment_id") or "").strip() for r in rows}
    kept = [r for r in rows if (r.get("comment_id") or "").strip() not in leave_out]
    _write_rows(output, fieldnames, kept)
    provenance = {
        "parent": {"file": os.path.basename(full_csv), "sha256": corpus_mod.content_hash(full_csv),
                   "rows": len(rows)},
        "excluded": [{"file": os.path.basename(p), "sha256": corpus_mod.content_hash(p),
                      "ids": len(ids), "ids_in_parent": len(ids & present)}
                     for p, ids in excluded.items()],
        "removed_rows": len(rows) - len(kept),
        "output": {"file": os.path.basename(output), "sha256": corpus_mod.content_hash(output),
                   "rows": len(kept)},
    }
    with open(output + ".provenance.json", "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=1)
        handle.write("\n")
    return provenance


def verify(corpus_csv, full_csv, exclude_paths):
    """Check `corpus_csv` is exactly `full_csv` minus the ids in `exclude_paths`, with every
    comment's text unchanged. Returns (problems, notes); counts and ids only, never text."""
    problems, notes = [], []
    full_cols, full = _read_rows(full_csv)
    cols, rows = _read_rows(corpus_csv)
    for name, c in (("full", full_cols), ("batching", cols)):
        if "comment_id" not in c or "body" not in c:
            problems.append(f"the {name} corpus has no comment_id and/or body column (its first "
                            f"columns read {c[:4]}); a spreadsheet import can replace the header "
                            f"with Column1, Column2...")
    if problems:
        return problems, notes
    if cols != full_cols:
        problems.append(f"the header differs from the full corpus: columns added "
                        f"{[c for c in cols if c not in full_cols]}, missing "
                        f"{[c for c in full_cols if c not in cols]}")
    full_ids = [(r.get("comment_id") or "").strip() for r in full]
    ids = [(r.get("comment_id") or "").strip() for r in rows]
    for name, seq in (("full", full_ids), ("batching", ids)):
        if len(seq) != len(set(seq)):
            problems.append(f"{len(seq) - len(set(seq))} repeated comment_id(s) in the {name} corpus")
        if "" in seq:
            problems.append(f"{seq.count('')} blank comment_id(s) in the {name} corpus")
    leave_out = set().union(*[_ids_from(p) for p in exclude_paths]) if exclude_paths else set()
    expected = leave_out & set(full_ids)
    removed = set(full_ids) - set(ids)
    notes.append(f"full corpus {len(full):,} rows; batching corpus {len(rows):,} rows; to leave out: "
                 f"{len(expected):,} ({len(leave_out) - len(expected)} of the listed ids are not in the "
                 f"full corpus); actually removed {len(removed):,}")
    if set(ids) - set(full_ids):
        problems.append(f"{len(set(ids) - set(full_ids))} comment(s) are not in the full corpus "
                        f"(first ids: {sorted(set(ids) - set(full_ids))[:3]})")
    if removed - expected:
        problems.append(f"{len(removed - expected)} comment(s) were removed that are not in the "
                        f"exclusion files (first ids: {sorted(removed - expected)[:3]})")
    if expected & set(ids):
        problems.append(f"{len(expected & set(ids))} comment(s) that should be left out are still "
                        f"there (first ids: {sorted(expected & set(ids))[:3]})")
    by_id = {(r.get("comment_id") or "").strip(): r for r in full}
    changed, newline_only = {}, 0
    for row in rows:
        base = by_id.get((row.get("comment_id") or "").strip())
        if base is None:
            continue
        for col in full_cols:
            a, b = row.get(col) or "", base.get(col) or ""
            if a != b:
                if _norm(a) == _norm(b):
                    newline_only += 1
                else:
                    changed.setdefault(col, []).append((row.get("comment_id") or "").strip())
    for col in ("body", "story", "comment_id"):
        if col in changed:
            problems.append(f"{len(changed[col])} row(s) have a different `{col}` from the full "
                            f"corpus (first ids: {changed[col][:3]})")
    if newline_only:
        notes.append(f"{newline_only} cell(s) differ only in line endings (harmless for labelling)")
    others = {c: len(v) for c, v in changed.items() if c not in ("body", "story", "comment_id")}
    if others:
        notes.append(f"other columns differ from the full corpus (a spreadsheet reformatting?): {others}")
    return problems, notes


def _provenance_beside(input_csv):
    path = input_csv + ".provenance.json"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    return None


def split(input_csv, out_dir, batch_size=1000, shuffle_seed=7, provenance=None):
    """Cut `input_csv` into batches under `out_dir`. Returns the manifest dict."""
    if batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")
    fieldnames, rows = _read_rows(input_csv)
    for column in ("comment_id", "body"):
        if column not in fieldnames:
            raise SystemExit(f"{input_csv} has no `{column}` column.")
    ids = [(row.get("comment_id") or "").strip() for row in rows]
    if not all(ids):
        raise SystemExit(f"{ids.count('')} row(s) in {input_csv} have no comment_id; the batches "
                         f"are checked against their ids, so every row needs one.")
    repeated = [cid for cid, n in collections.Counter(ids).items() if n > 1]
    if repeated:
        raise SystemExit(f"{len(repeated)} comment_id(s) appear more than once in {input_csv} "
                         f"(first: {repeated[0]!r}). Dedupe the corpus first.")
    if os.path.exists(os.path.join(out_dir, MANIFEST)) or (
            os.path.isdir(os.path.join(out_dir, INPUTS)) and os.listdir(os.path.join(out_dir, INPUTS))):
        raise SystemExit(f"{out_dir} already holds batches. Use an empty folder: re-splitting "
                         f"over finished work would orphan the labelled batches beside it.")

    order = list(range(len(rows)))
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(order)
    total = max(1, -(-len(rows) // batch_size))
    batches = []
    for number in range(1, total + 1):
        positions = sorted(order[(number - 1) * batch_size: number * batch_size]) \
            if shuffle_seed is None else order[(number - 1) * batch_size: number * batch_size]
        chunk = [rows[i] for i in positions]
        name = batch_name(number, total)
        path = os.path.join(out_dir, INPUTS, name)
        _write_rows(path, fieldnames, chunk)
        batches.append({"number": number, "file": name, "rows": len(chunk),
                        "blank_body_rows": sum(1 for row in chunk if not _has_text(row)),
                        "sha256": corpus_mod.content_hash(path),
                        "source_positions": positions})
    manifest = {
        "input": {"file": os.path.basename(input_csv),
                  "sha256": corpus_mod.content_hash(input_csv), "rows": len(rows),
                  "blank_body_rows": sum(1 for row in rows if not _has_text(row))},
        "fieldnames": fieldnames, "batch_size": batch_size,
        "order": "file order" if shuffle_seed is None else f"shuffled, seed {shuffle_seed}",
        "batches": batches,
    }
    if provenance:
        manifest["provenance"] = provenance
    with open(os.path.join(out_dir, MANIFEST), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1)
        handle.write("\n")
    return manifest


def load_manifest(out_dir):
    path = os.path.join(out_dir, MANIFEST)
    if not os.path.exists(path):
        raise SystemExit(f"No {MANIFEST} in {out_dir}: run `ewl-batches split` first.")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _fingerprint(record):
    """What a labelled batch was labelled WITH. Two batches with different fingerprints
    cannot be joined."""
    fixed = record.get("fixed_examples") or {}
    examples = (fixed.get("sha256") if fixed.get("source") == "file"
                else json.dumps(fixed.get("examples"), sort_keys=True))
    return {
        "domain judge": (record.get("domain_judge") or {}).get("spec"),
        "model": (record.get("domain_judge") or {}).get("model"),
        "model build (weights digest, Ollama version, temperature)": json.dumps(
            {k: (record.get("model_build") or {}).get(k)
             for k in ("digest", "server_version", "temperature")},
            sort_keys=True),
        "votes": (record.get("domain_judge") or {}).get("votes"),
        "sentiment scorer": record.get("sentiment_scorer"),
        "domain prompt": (record.get("prompts_sha256") or {}).get("domain"),
        "domain prompts by story": json.dumps(
            (record.get("prompts_sha256") or {}).get("domain_by_story"), sort_keys=True),
        "sentiment prompts": json.dumps((record.get("prompts_sha256") or {}).get(
            "sentiment_by_story"), sort_keys=True),
        "example set": examples,
        "switches": (record.get("instrument_switches") or {}).get("tags"),
    }


def inspect_batch(out_dir, manifest, entry):
    """State of one batch: {'state': pending|done|bad, 'problems': [...], ...}."""
    name = entry["file"]
    input_path = os.path.join(out_dir, INPUTS, name)
    labelled = os.path.join(out_dir, LABELLED, name)
    result = {"entry": entry, "labelled": labelled, "problems": [], "record": None}
    if not os.path.exists(input_path):
        result["state"] = "bad"
        result["problems"].append(f"the batch file {input_path} is missing")
        return result
    if corpus_mod.content_hash(input_path) != entry["sha256"]:
        result["state"] = "bad"
        result["problems"].append("the batch file has changed since it was split "
                                  "(its hash no longer matches the manifest)")
        return result
    if not os.path.exists(labelled):
        result["state"] = "pending"
        return result
    result["state"] = "done"
    record_path = labelled + ".run.json"
    if not os.path.exists(record_path):
        result["problems"].append("no .run.json beside the labelled file (a run that did not "
                                  "finish, or a file from another tool)")
    else:
        with open(record_path, encoding="utf-8") as handle:
            record = json.load(handle)
        result["record"] = record
        if (record.get("input") or {}).get("sha256") != entry["sha256"]:
            result["problems"].append("its run record names a different input than this batch "
                                      "(the wrong file, or the batch was edited)")
        part = record.get("result") or {}
        if part.get("sample") or part.get("limit"):
            result["problems"].append("it was run with --sample or --limit, so it labels only "
                                      "part of the batch")
    _, input_rows = _read_rows(input_path)
    _, labelled_rows = _read_rows(labelled)
    expected = collections.Counter((r.get("comment_id") or "").strip()
                                   for r in input_rows if _has_text(r))
    got = collections.Counter((r.get("comment_id") or "").strip() for r in labelled_rows)
    missing = sum(1 for cid in expected if cid not in got)
    extra = sum(1 for cid in got if cid not in expected)
    repeated = sum(1 for cid, n in got.items() if n > 1)
    if missing or extra or repeated:
        result["problems"].append(f"its comments do not match the batch: {missing} missing, "
                                  f"{extra} not in the batch, {repeated} repeated")
    result["rows"] = labelled_rows
    result["expected_labelled"] = sum(expected.values())
    return result


def status_lines(out_dir, manifest):
    states = [inspect_batch(out_dir, manifest, entry) for entry in manifest["batches"]]
    return states


def cmd_status(args):
    manifest = load_manifest(args.out_dir)
    states = status_lines(args.out_dir, manifest)
    total = len(states)
    counts = collections.Counter(
        "problem" if s["problems"] else s["state"] for s in states)
    print(f"{manifest['input']['rows']:,} comments in {total} batch(es) of up to "
          f"{manifest['batch_size']:,} ({manifest['order']}).")
    print(f"done {counts['done']} | pending {counts['pending']} | need attention "
          f"{counts['problem']}")
    for s in states:
        label = "PROBLEM" if s["problems"] else s["state"]
        print(f"  {s['entry']['file']}  {s['entry']['rows']:>6,} rows   {label}")
        for problem in s["problems"]:
            print(f"      - {problem}")
    pending = [s for s in states if s["state"] == "pending"]
    if pending:
        print(f"\nNext: {pending[0]['entry']['file']}. Run `ewl-batches commands "
              f"{args.out_dir}` to print the exact command.")
    return 1 if counts["problem"] else 0


def command_for(out_dir, entry, extra="", cache=None):
    cache = cache or os.path.join(out_dir, "gemma_cache.jsonl")
    return (f'ewl-gemma "{os.path.join(out_dir, INPUTS, entry["file"])}" '
            f'--output "{os.path.join(out_dir, LABELLED, entry["file"])}" '
            f'--cache "{cache}"' + (f" {extra}" if extra else ""))


def cmd_commands(args):
    manifest = load_manifest(args.out_dir)
    states = status_lines(args.out_dir, manifest)
    pending = [s for s in states if s["state"] == "pending"]
    # A batch with a labelled file but a problem (no run record, wrong comments) must be
    # run again, or `check` will refuse it forever. The same command overwrites its output
    # and resumes from the cache, so it is safe to list.
    redo = [s for s in states if s["state"] == "done" and s["problems"]]
    if not pending and not redo and not any(s["state"] == "bad" for s in states):
        print("Nothing pending.", file=sys.stderr)
        return 0
    todo = sorted(pending + redo, key=lambda s: s["entry"]["file"])
    # Without --fixed-examples, ewl-gemma uses the BUILT-IN examples: a different prompt, so the
    # batch is labelled from scratch under another instrument and `check` refuses to join it,
    # after the hours it took. If the finished batches used an examples file, refuse to print
    # commands that leave it out. (Only the flag's presence can be checked here: the recorded
    # hash is of the parsed examples, which this module does not load.)
    file_examples = next((s["record"].get("fixed_examples") for s in states
                          if s["state"] == "done" and s["record"]
                          and (s["record"].get("fixed_examples") or {}).get("source") == "file"),
                         None)
    if not any(token.split("=")[0] == "--fixed-examples" for token in (args.extra or "").split()):
        if file_examples:
            print(f"Not printing commands: the finished batches were labelled with "
                  f"--fixed-examples {file_examples.get('file', '?')} (sha256 "
                  f"{(file_examples.get('sha256') or '?')[:12]}...), and these commands would "
                  f"leave it out, so every batch would be labelled with the built-in examples "
                  f"instead and `check` would refuse to join it. Pass the same file, e.g.\n"
                  f"    ewl-batches commands {args.out_dir} --extra \"--fixed-examples "
                  f"<path to {file_examples.get('file', 'the examples file')}>\"",
                  file=sys.stderr)
            return 1
        print("# WARNING: no --fixed-examples in --extra, so these commands use the BUILT-IN "
              "examples (provisional). Add --extra \"--fixed-examples FILE\" for the real run.",
              file=sys.stderr)
    # A batch whose input file is missing or was edited cannot be run as it stands. Say so
    # loudly, or `--first 5` would quietly skip it and queue the next one in its place.
    bad = [s for s in states if s["state"] == "bad"]
    for s in bad:
        print(f"# WARNING: {s['entry']['file']} is NOT queued: {'; '.join(s['problems'])}. "
              f"Re-copy the corpus and split into a new, empty folder.", file=sys.stderr)
    # --first N: only the next N in batch order, so a long run can be done a few batches at
    # a time. Pending means not yet finished, so the same command later prints the next N.
    chosen = todo[:args.first] if args.first else todo
    print("# Run these in order. Same --cache and same flags for every batch. If one stops, "
          "run the same line again: it resumes.", file=sys.stderr)
    if len(chosen) < len(todo):
        print(f"# The first {len(chosen)} of {len(todo)} batch(es) still to run. Run "
              f"`ewl-batches commands` again with --first when they are done for the next ones.",
              file=sys.stderr)
    for s in redo:
        if s in chosen:
            print(f"# {s['entry']['file']} has a labelled file with problems "
                  f"({'; '.join(s['problems'])}): running it again replaces the file.",
                  file=sys.stderr)
    for s in chosen:
        print(command_for(args.out_dir, s["entry"], args.extra, args.cache))
    return 0


def cmd_check(args):
    manifest = load_manifest(args.out_dir)
    states = status_lines(args.out_dir, manifest)
    problems = []
    done = [s for s in states if s["state"] == "done"]
    for s in states:
        for problem in s["problems"]:
            problems.append(f"{s['entry']['file']}: {problem}")
        if s["state"] == "bad":
            continue
    # one instrument across every finished batch
    prints = {}
    for s in done:
        if s["record"]:
            prints[s["entry"]["file"]] = _fingerprint(s["record"])
    if prints:
        reference_name, reference = next(iter(prints.items()))
        for name, fingerprint in prints.items():
            differing = [k for k in reference if fingerprint[k] != reference[k]]
            if differing:
                problems.append(f"{name} was labelled under a different instrument from "
                                f"{reference_name} ({', '.join(differing)}). Relabel the "
                                f"odd one out, or all of them, under one setting.")
    pending = [s for s in states if s["state"] == "pending"]
    print(f"{len(done)} of {len(states)} batch(es) finished, {len(pending)} pending.")
    if done:
        parse_fail = sentiment_fail = unclassified = labelled = 0
        for s in done:
            for row in s.get("rows", []):
                labelled += 1
                parse_fail += row.get("domain_parse_failure") == "1"
                unclassified += row.get("domain_unclassified") == "1"
                sentiment_fail += (row.get("sentiment_parse_failure") == "1"
                                   or row.get("sentiment_no_consensus") == "1")
        print(f"Across the finished batches ({labelled:,} comments): domain parse failures "
              f"{parse_fail / labelled:.1%}, sentiment failures {sentiment_fail / labelled:.1%}, "
              f"unclassified {unclassified / labelled:.1%}.")
    if problems:
        print("\nPROBLEMS (nothing was merged):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    if not args.merge_to:
        print("All finished batches check out." + (" Pass --merge-to FILE to join them once "
                                                   "every batch is done (or add --partial for "
                                                   "a progress file of those done so far)."
                                                   if pending else ""))
        return 0
    if pending and not args.partial:
        print(f"\nNot merging: {len(pending)} batch(es) still pending. Add --partial to join "
              f"the {len(done)} finished so far into a progress file.")
        return 1
    if not done:
        print("\nNot merging: no batch is finished yet.")
        return 1
    return _merge(args, manifest, done, partial=bool(pending))


def _merge(args, manifest, done, partial=False):
    by_position = {}
    labelled_columns = None
    for s in done:
        positions = s["entry"]["source_positions"]
        input_rows = _read_rows(os.path.join(args.out_dir, INPUTS, s["entry"]["file"]))[1]
        position_of = {(row.get("comment_id") or "").strip(): pos
                       for row, pos in zip(input_rows, positions)}
        for row in s["rows"]:
            by_position[position_of[(row.get("comment_id") or "").strip()]] = row
        if labelled_columns is None:
            labelled_columns = list(s["rows"][0].keys()) if s["rows"] else manifest["fieldnames"]
    ordered = [by_position[p] for p in sorted(by_position)]
    expected = (sum(s["expected_labelled"] for s in done) if partial
                else manifest["input"]["rows"] - manifest["input"]["blank_body_rows"])
    if len(ordered) != expected:
        print(f"\nNot merging: {len(ordered):,} labelled rows, but {expected:,} were expected.")
        return 1
    _write_rows(args.merge_to, labelled_columns, ordered)
    # A run record for the joined file that names the WHOLE corpus as its input.
    record = json.loads(json.dumps(done[0]["record"]))
    record["input"] = {"path": manifest["input"]["file"], "sha256": manifest["input"]["sha256"]}
    record["output"] = args.merge_to
    record["command"] = "ewl-batches check --merge-to " + args.merge_to + (
        " --partial" if partial else "")
    # blank_body_rows counts the rows of THIS file's batches with no text (for a full merge,
    # the whole corpus's).
    record["result"] = {"comments": len(ordered), "merged_from_batches": len(done),
                        "blank_body_rows": sum(s["entry"]["blank_body_rows"] for s in done)}
    if partial:
        # The input is still the whole corpus (that is what these labels came from), so the
        # record must say loudly that the file is NOT all of it. ewl-export-dashboard reads this.
        record["result"]["partial"] = True
        record["partial"] = {
            "batches_included": [s["entry"]["file"] for s in done],
            "batches_done": len(done), "batches_total": len(manifest["batches"]),
            "comments_included": len(ordered),
            "comments_in_corpus": manifest["input"]["rows"] - manifest["input"]["blank_body_rows"],
        }
    record.pop("cache", None)
    with open(args.merge_to + ".run.json", "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if partial:
        p = record["partial"]
        print(f"\nPROGRESS FILE, NOT THE WHOLE CORPUS: {len(ordered):,} of "
              f"{p['comments_in_corpus']:,} labelled comments, from {len(done)} of "
              f"{p['batches_total']} batches, in the original corpus order -> {args.merge_to}")
        print(f"Run record -> {args.merge_to}.run.json, marked partial with the batches it holds. "
              f"`ewl-export-dashboard --corpus` refuses it; merge without --partial once every "
              f"batch is done.")
        return 0
    print(f"\nMerged {len(ordered):,} labelled comments from {len(done)} batches, in the original "
          f"corpus order -> {args.merge_to}")
    print(f"Run record -> {args.merge_to}.run.json (its input is the full corpus, so "
          f"`ewl-export-dashboard --corpus` can check it).")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="write the full corpus minus the gold/example comments "
                                       "(no spreadsheet involved), with a provenance file")
    p.add_argument("full_csv")
    p.add_argument("--exclude", action="append", required=True, metavar="FILE",
                   help="a file with a comment_id column whose comments are left out (repeatable)")
    p.add_argument("--output", required=True)

    p = sub.add_parser("verify", help="check a corpus is exactly the full corpus minus the "
                                      "excluded comments, with every text unchanged")
    p.add_argument("input_csv")
    p.add_argument("--full", required=True)
    p.add_argument("--exclude", action="append", default=[], metavar="FILE")

    p = sub.add_parser("split", help="cut a corpus into batches")
    p.add_argument("input_csv")
    p.add_argument("--full", metavar="FILE",
                   help="check the corpus against the full corpus before splitting (with --exclude): "
                        "refuses a corpus that is not the full corpus minus the excluded comments")
    p.add_argument("--exclude", action="append", default=[], metavar="FILE",
                   help="with --full: a file whose comment_ids should be absent (repeatable)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=1000,
                   help="comments per batch (default 1,000: about 40 minutes at 2.4 s a comment)")
    p.add_argument("--shuffle-seed", type=int, default=7,
                   help="seed for the shuffle (default 7)")
    p.add_argument("--no-shuffle", action="store_true",
                   help="keep the corpus file's order (batches are then contiguous slices)")

    p = sub.add_parser("status", help="which batches are done, pending or need attention")
    p.add_argument("out_dir")

    p = sub.add_parser("commands", help="print the ewl-gemma command for every pending batch")
    p.add_argument("out_dir")
    p.add_argument("--extra", default="",
                   help='flags added to every command, e.g. "--fixed-examples official-data/fixed.csv"')
    p.add_argument("--cache", help="one shared response cache (default <out_dir>/gemma_cache.jsonl)")
    p.add_argument("--first", type=int, metavar="N",
                   help="print only the first N batches still to run, in order (e.g. 5 for "
                        "batches 1 to 5); run again later for the next N")

    p = sub.add_parser("check", help="verify finished batches; optionally join them")
    p.add_argument("out_dir")
    p.add_argument("--merge-to", metavar="FILE",
                   help="write the joined, labelled corpus here once every batch is done and checks out")
    p.add_argument("--partial", action="store_true",
                   help="with --merge-to: join only the batches finished so far, into a progress "
                        "file whose run record is marked partial (never the whole corpus)")

    args = parser.parse_args(argv)
    if getattr(args, "first", None) is not None and args.first < 1:
        parser.error("--first must be at least 1.")
    if getattr(args, "partial", False) and not args.merge_to:
        parser.error("--partial needs --merge-to FILE (it names the progress file to write).")
    if args.command == "prepare":
        result = prepare(args.full_csv, args.exclude, args.output)
        print(f"{result['parent']['rows']:,} comments in {result['parent']['file']} -> "
              f"{result['output']['rows']:,} in {args.output} ({result['removed_rows']:,} left out).")
        for item in result["excluded"]:
            print(f"  {item['file']}: {item['ids']:,} ids, {item['ids_in_parent']:,} in the full corpus")
        print(f"Send the recipient: {result['output']['rows']:,} rows, sha256 starting "
              f"{result['output']['sha256'][:16]}. Provenance: {args.output}.provenance.json")
        return 0
    if args.command == "verify":
        problems, notes = verify(args.input_csv, args.full, args.exclude)
        for note in notes:
            print("note:", note)
        for problem in problems:
            print("PROBLEM:", problem)
        if problems:
            print("Do not use this corpus.")
            return 1
        print(f"OK: {args.input_csv} is {args.full} minus the excluded comments, every text unchanged "
              f"(sha256 {corpus_mod.content_hash(args.input_csv)[:16]}).")
        return 0
    if args.command == "split":
        provenance = _provenance_beside(args.input_csv)
        if args.exclude and not args.full:
            parser.error("--exclude on split needs --full FILE (the corpus to check against).")
        if args.full:
            problems, notes = verify(args.input_csv, args.full, args.exclude)
            for note in notes:
                print("note:", note)
            if problems:
                raise SystemExit("The corpus is not the full corpus minus the excluded comments:\n  "
                                 + "\n  ".join(problems))
            print("Checked against the full corpus: OK.")
            provenance = dict(provenance or {}, verified_against={
                "full_sha256": corpus_mod.content_hash(args.full),
                "excluded_files": [{"file": os.path.basename(p),
                                    "sha256": corpus_mod.content_hash(p)} for p in args.exclude]})
        if provenance and provenance.get("parent"):
            print(f"Made from {provenance['parent']['file']} (sha256 "
                  f"{provenance['parent']['sha256'][:16]}), {provenance['removed_rows']:,} comments left out.")
        manifest = split(args.input_csv, args.out_dir, args.batch_size,
                         None if args.no_shuffle else args.shuffle_seed, provenance)
        info = manifest["input"]
        print(f"{info['rows']:,} comments ({info['blank_body_rows']:,} with no text, which are "
              f"never labelled) -> {len(manifest['batches'])} batch(es) of up to "
              f"{manifest['batch_size']:,}, {manifest['order']}.")
        for entry in manifest["batches"]:
            print(f"  {entry['file']}  {entry['rows']:,} rows")
        print(f"Input sha256 {info['sha256'][:16]}... recorded in "
              f"{os.path.join(args.out_dir, MANIFEST)}")
        print("The batch files contain comment text: keep them out of git and off shared drives.")
        return 0
    return {"status": cmd_status, "commands": cmd_commands, "check": cmd_check}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
