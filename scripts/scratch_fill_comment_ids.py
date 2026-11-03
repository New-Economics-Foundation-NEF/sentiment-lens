r"""Fill in missing comment_id values in a hand-coded sheet, from the original coding sheets.

    python scripts/scratch_fill_comment_ids.py gold_sample_Jamie_complete.xlsx \
        --reference official-data/gold_sample_to_code_Abhijay.xlsx \
        --output official-data/gold_sample_Jamie_complete_ids.xlsx

For each row whose comment_id is blank, the comment text (`body`) is looked up in the
reference sheet(s) -- the file the comments were originally sent out in, which has the
comment_id beside each body -- and the matching comment_id is written in. Rows that
already have an id are never touched.

A row is left blank, and listed by row number, if its text matches nothing (the text
was edited, or is not in the reference) or matches more than one comment (the same
short text, such as "Yes", appears under more than one id). If the coded sheet has a
`story` column it is used to separate those. Nothing is guessed. The comment text is
never printed, only row numbers and counts.

Reference sheets that work: any `gold_sample_to_code_<coder>.xlsx`, or
`gold_sample_balanced.csv`. The coded sheet needs a `body` column; if the body was not
copied across, copy it from the original sheet first.

Needs `pip install -e ".[analysis]"` (openpyxl) for .xlsx files.
"""
import argparse
import collections
import csv
import os
import sys


def normalise(text):
    """Whitespace collapsed, so a stray double space or a line break does not defeat a match."""
    return " ".join((text or "").split())


def read_reference(paths, body_column="body", id_column="comment_id"):
    """{normalised body: {comment_id: story}} from one or more reference sheets."""
    from sentiment_lens.analyse.validate import read_any
    index = collections.defaultdict(dict)
    for path in paths:
        rows = read_any(path)
        if rows and (body_column not in rows[0] or id_column not in rows[0]):
            sys.exit(f"{path} needs `{body_column}` and `{id_column}` columns.")
        for row in rows:
            cid = (str(row.get(id_column) or "")).strip()
            body = normalise(str(row.get(body_column) or ""))
            if cid and body:
                index[body][cid] = (str(row.get("story") or "")).strip()
    return index


def match_ids(coded_rows, index, body_column="body", id_column="comment_id"):
    """For each coded row (a dict), the id to write, or why not.

    Returns (fills, problems): fills is {row index: id}; problems is
    {"unmatched": [row indices], "ambiguous": [row indices]}.
    """
    fills, problems = {}, {"unmatched": [], "ambiguous": []}
    for i, row in enumerate(coded_rows):
        if (str(row.get(id_column) or "")).strip():
            continue
        body = normalise(str(row.get(body_column) or ""))
        if not body:
            continue
        candidates = index.get(body)
        if not candidates:
            problems["unmatched"].append(i)
            continue
        if len(candidates) == 1:
            fills[i] = next(iter(candidates))
            continue
        story = (str(row.get("story") or "")).strip()
        same_story = [cid for cid, s in candidates.items() if story and s == story]
        if len(same_story) == 1:
            fills[i] = same_story[0]
        else:
            problems["ambiguous"].append(i)
    return fills, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("coded", help="the hand-coded sheet with blank comment_id cells (.xlsx or .csv)")
    parser.add_argument("--reference", nargs="+", required=True,
                        help="sheet(s) holding comment_id and body for the original comments")
    parser.add_argument("--output", required=True, help="where to write the filled sheet (a new file)")
    parser.add_argument("--body-column", default="body")
    parser.add_argument("--id-column", default="comment_id")
    args = parser.parse_args()
    if os.path.abspath(args.output) == os.path.abspath(args.coded):
        sys.exit("--output must be a different file: the original is never overwritten.")

    index = read_reference(args.reference, args.body_column, args.id_column)
    ext = os.path.splitext(args.coded)[1].lower()

    if ext in (".xlsx", ".xlsm"):
        try:
            import openpyxl
        except ImportError:
            sys.exit("Reading .xlsx needs openpyxl: pip install -e \".[analysis]\"")
        wb = openpyxl.load_workbook(args.coded)
        ws = wb.active
        header = [(c.value or "") for c in ws[1]]
        header = [str(h).strip() for h in header]
        if args.body_column not in header:
            sys.exit(f"No `{args.body_column}` column in {args.coded}; copy the comment text "
                     f"across from the original sheet first.")
        if args.id_column not in header:
            ws.cell(row=1, column=len(header) + 1, value=args.id_column)
            header.append(args.id_column)
        id_col = header.index(args.id_column) + 1
        rows = [{h: ws.cell(row=r, column=c + 1).value for c, h in enumerate(header)}
                for r in range(2, ws.max_row + 1)]
        fills, problems = match_ids(rows, index, args.body_column, args.id_column)
        for i, cid in fills.items():
            ws.cell(row=i + 2, column=id_col, value=cid)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        wb.save(args.output)
        sheet_row = lambda i: i + 2
    else:
        with open(args.coded, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames)
            rows = list(reader)
        if args.body_column not in fieldnames:
            sys.exit(f"No `{args.body_column}` column in {args.coded}.")
        if args.id_column not in fieldnames:
            fieldnames.append(args.id_column)
        fills, problems = match_ids(rows, index, args.body_column, args.id_column)
        for i, cid in fills.items():
            rows[i][args.id_column] = cid
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        sheet_row = lambda i: i + 2

    filled_ids = collections.Counter(fills.values())
    print(f"{len(fills)} comment_id(s) filled in; written to {args.output}. "
          f"The original is untouched.")
    already = {str(r.get(args.id_column) or "").strip() for r in rows} - {""}
    clash = sorted(cid for cid in filled_ids if cid in already)
    if clash or any(n > 1 for n in filled_ids.values()):
        print("WARNING: some filled ids appear more than once in the sheet (the same comment "
              "coded twice, or two rows with the same text). Check these ids: "
              f"{', '.join(sorted(set(clash) | {c for c, n in filled_ids.items() if n > 1})[:10])}",
              file=sys.stderr)
    for kind, label in (("unmatched", "text matched nothing in the reference sheet(s) "
                         "(edited, or not in them)"),
                        ("ambiguous", "text matches more than one comment id, and the story "
                         "does not separate them")):
        if problems[kind]:
            shown = ", ".join(str(sheet_row(i)) for i in problems[kind][:25])
            more = f" (+{len(problems[kind]) - 25} more)" if len(problems[kind]) > 25 else ""
            print(f"LEFT BLANK, {label}: sheet row(s) {shown}{more}", file=sys.stderr)


if __name__ == "__main__":
    main()
