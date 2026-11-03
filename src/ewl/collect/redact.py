"""Find, and optionally remove, personal identifiers in comment text.

    # Look first: counts only, nothing written.
    ewl-redact data/full_corpus_master.csv

    # Then redact, writing a new frozen corpus (the original is untouched):
    ewl-redact data/full_corpus_master.csv --handles leading --emails --phones \\
        --output data/full_corpus_master_redacted.csv

What it finds
  * @handles. A YouTube reply usually opens with the handle of the person being
    replied to ("@SomeChannel thanks, but..."). The handle is a real, searchable
    account name, and in a thread it can often be matched to the top-level author whose
    pseudonymous `author_hash` sits in the same file, which undoes the pseudonym.
  * Email addresses and phone numbers a commenter typed into the text.

Choices (nothing is decided for you; with no flags it only reports)
  --handles leading   replace only a handle that opens the comment (the addressee)
  --handles all       replace every @handle anywhere in the comment
  --handles keep      leave handles alone (the default)
  --handle-token      what a handle becomes (default "@user"): the comment still reads as a
                      reply to someone, which short replies often depend on, but not whom.
  --emails / --phones replace those with "[email]" / "[phone]"

Run it BEFORE ewl-classify: the judge cache is keyed on the comment text, so redacting
afterwards means paying for the run again, and the handles would already have been
sent to a third-party model provider. The output is frozen with its own manifest, so
the new corpus hash is recorded.

It is a pattern match, not a guarantee: a name written in prose ("my neighbour John
Smith") is not found. Read a sample of the output.
"""
import argparse
import csv
import os
import re
import sys

from ewl import corpus as corpus_mod

HANDLE = re.compile(r"(?<![\w.\-])@[\w.\-]{2,}")
LEADING_HANDLE = re.compile(r"^\s*((?:(?<![\w.\-])@[\w.\-]{2,}[\s,:;]*)+)")
EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
# Phone numbers are matched only in shapes a phone number actually takes, never as "any
# run of digit groups": this corpus argues about money and dates, and "230 000 000 000",
# "1 000 000 000" or "2008 2009 2010" must reach the model unchanged.
#   * international: a leading +, a 1-3 digit country code, then 7-14 more digits
#   * UK national: a leading 0, 10-11 digits in all (07700 900123, 020 7946 0958, 07700900123)
#   * North American: 3-3-4 with a valid area code and exchange (both start 2-9), the area
#     code bracketed or followed by a separator: (212) 555-1234, 212-555-1234, 212 555 1234
_NOT_AFTER = r"(?<![\w$+.,/])(?<!\d[\s.,\-])"
_NOT_BEFORE = r"(?![\w$])(?![\s.,\-]?\d)"
PHONE_PATTERNS = [
    (re.compile(_NOT_AFTER + r"\+\d{1,3}(?:[\s.\-]?\(?\d{1,4}\)?){2,6}" + _NOT_BEFORE), 8, 15),
    (re.compile(_NOT_AFTER + r"0\d{2,4}[\s.\-]?\d{3,4}[\s.\-]?\d{3,4}" + _NOT_BEFORE), 10, 11),
    (re.compile(_NOT_AFTER + r"(?:\(\s*[2-9]\d{2}\s*\)\s*|[2-9]\d{2}[\s.\-])[2-9]\d{2}[\s.\-]\d{4}"
                + _NOT_BEFORE), 10, 10),
]


def _phone_matches(text):
    """Non-overlapping phone-number matches in `text`, as (start, end) spans."""
    spans = []
    for pattern, low, high in PHONE_PATTERNS:
        for match in pattern.finditer(text):
            digits = len(re.sub(r"\D", "", match.group()))
            if not low <= digits <= high:
                continue
            if any(match.start() < end and start < match.end() for start, end in spans):
                continue
            spans.append((match.start(), match.end()))
    return sorted(spans)


def redact_text(text, handles="keep", emails=False, phones=False, handle_token="@user"):
    """Return (redacted text, counts of what was replaced)."""
    counts = {"handles": 0, "emails": 0, "phones": 0}
    text = text or ""
    if emails:
        text, counts["emails"] = EMAIL.subn("[email]", text)
    if phones:
        spans = _phone_matches(text)
        for start, end in reversed(spans):
            text = text[:start] + "[phone]" + text[end:]
        counts["phones"] = len(spans)
    if handles == "all":
        text, counts["handles"] = HANDLE.subn(handle_token, text)
    elif handles == "leading":
        match = LEADING_HANDLE.match(text)
        if match:
            counts["handles"] = len(HANDLE.findall(match.group(1)))
            text = handle_token + " " + text[match.end():].lstrip()
            text = text.rstrip()
    return text, counts


def scan(rows, column="body"):
    """Counts of rows, for the report: {kind: number of rows containing one}."""
    found = {"handle_leading": 0, "handle_anywhere": 0, "email": 0, "phone": 0}
    for row in rows:
        body = row.get(column) or ""
        if LEADING_HANDLE.match(body):
            found["handle_leading"] += 1
        if HANDLE.search(body):
            found["handle_anywhere"] += 1
        if EMAIL.search(body):
            found["email"] += 1
        if _phone_matches(body):
            found["phone"] += 1
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv")
    parser.add_argument("--output", help="Write the redacted corpus here. Without it, only report.")
    parser.add_argument("--column", default="body")
    parser.add_argument("--handles", choices=["keep", "leading", "all"], default="keep")
    parser.add_argument("--handle-token", default="@user")
    parser.add_argument("--emails", action="store_true")
    parser.add_argument("--phones", action="store_true")
    args = parser.parse_args()

    with open(args.input_csv, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if not rows or args.column not in fieldnames:
        sys.exit(f"No {args.column!r} column / rows in {args.input_csv}.")

    if args.output and os.path.abspath(args.output) == os.path.abspath(args.input_csv):
        sys.exit("--output is the input file. Redaction writes a NEW frozen corpus and leaves "
                 "the original untouched, so the two can be compared; choose another path.")

    found = scan(rows, args.column)
    total = len(rows)
    print(f"{args.input_csv}: {total:,} comments", file=sys.stderr)
    print(f"  open with an @handle (the addressee): {found['handle_leading']:,} "
          f"({found['handle_leading'] / total:.1%})", file=sys.stderr)
    print(f"  contain an @handle anywhere:          {found['handle_anywhere']:,} "
          f"({found['handle_anywhere'] / total:.1%})", file=sys.stderr)
    print(f"  contain an email address:             {found['email']:,}", file=sys.stderr)
    print(f"  contain a phone-like number:          {found['phone']:,}", file=sys.stderr)

    if not args.output:
        print("Nothing written (no --output). Choose --handles / --emails / --phones to "
              "redact.", file=sys.stderr)
        return
    if args.handles == "keep" and not args.emails and not args.phones:
        sys.exit("--output given but nothing chosen to redact: pass --handles leading|all, "
                 "--emails and/or --phones.")

    changed = {"handles": 0, "emails": 0, "phones": 0}
    for row in rows:
        new, counts = redact_text(row.get(args.column), args.handles, args.emails, args.phones,
                                  args.handle_token)
        if new != row.get(args.column):
            row[args.column] = new
        for key, value in counts.items():
            changed[key] += value

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    path, _ = corpus_mod.write_manifest(
        args.output, inputs=[args.input_csv],
        extra={"redaction": {"handles": args.handles, "handle_token": args.handle_token,
                             "emails": args.emails, "phones": args.phones,
                             "replaced": changed}})
    print(f"Replaced {changed['handles']:,} handle(s), {changed['emails']:,} email(s), "
          f"{changed['phones']:,} phone number(s). Wrote {args.output} and {path}.",
          file=sys.stderr)
    print("Read a sample of the output: this is a pattern match, not a guarantee. Any "
          "labelling must run on THIS file, and the judge cache is keyed on the text.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
