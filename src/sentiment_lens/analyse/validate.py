"""Compare harm labels across methods and draw a hand-coding sample.

    # 1. how far do the LLM and keyword labels agree?
    python validate.py agree master_classified_keyword.csv

    # 2. draw a stratified sample for humans to code by hand
    python validate.py sample master_classified_keyword.csv --n 300 > to_hand_code.csv

    # 3. score the automated labels against the returned hand codes
    python validate.py score to_hand_code_completed.csv

    # (between 2 and 3) how well did the two coders agree with each other,
    # on the shared subset they both coded independently?
    python validate.py interrater coder_a_completed.csv --second-input coder_b_completed.csv

Both project documents say the harm dimension cannot be quoted until a
hand-coded sample exists. This is the missing step. Nothing here needs a model
or a network connection.

Expects a file that has been through classify.py (domain_* columns) and
keyword_classify.py (kw_* columns). For `score`, add a `hand_domain` column
containing one or two of democracy/economics/environment/social, or "none".
"""
import argparse
import collections
import csv
import json
import math
import os
import random
import shutil
import statistics
import sys

# Windows defaults to cp1252 for stdout when it is redirected to a file, which
# crashes on any comment containing an emoji or non-Latin character. Every
# command here prints comment text, so this has to be set unconditionally,
# not just for `sample`.
sys.stdout.reconfigure(encoding="utf-8")

CATEGORIES = ["democracy", "economics", "environment", "social"]
# Imported rather than re-spelled: the sentinel moved once already (#68) and a
# second copy of the string here would not have moved with it.
from sentiment_lens.taxonomy import PARSE_FAILURE


def labels(row, prefix):
    got = {c for c in CATEGORIES if row.get(f"{prefix}{c}") == "1"}
    return got


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 1.0


def kappa(pairs):
    """Cohen's kappa on a single binary judgement (does label L apply?).

    Returned `nan` whenever `expected == 1`, even when every pair actually
    agreed -- and `expected` (`pa*pb + (1-pa)*(1-pb)`) can only reach 1 when
    both raters used the SAME single value for every single item (pa == pb
    == 1, or pa == pb == 0), which forces `observed` to be exactly 1 too:
    with both marginals pinned to one constant value, every pair is that
    value on both sides, so there is no way for `a != b` to occur. So
    `expected == 1` is not a genuinely undefined case here, it is "perfect,
    boring agreement" -- e.g. `slens-validate selfconsistency` on a fixture
    where the model gave the identical vote every single time -- and used to
    print `nan` for it instead of the 1.0 that agreement actually deserves.
    """
    n = len(pairs)
    if not n:
        return float("nan")
    observed = sum(1 for a, b in pairs if a == b) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected >= 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def read(path):
    return list(csv.DictReader(open(path, newline="", encoding="utf-8-sig")))


def drop_ids(rows, ids, what="rows"):
    """Rows whose comment_id is not in `ids`, reporting how many were removed.

    For comments shown to a model as worked examples: scored against their own
    hand label they would measure nothing, so they are left out of the scoring.
    """
    ids = set(ids or ())
    if not ids:
        return rows
    kept = [r for r in rows if (r.get("comment_id") or "").strip() not in ids]
    print(f"Excluded {len(rows) - len(kept)} of {len(rows)} {what} named as worked examples "
          f"(--exclude-ids or the run record; comments used as prompt examples are not "
          f"scored).", file=sys.stderr)
    return kept


def read_ids(path):
    """A set of comment_id values from either a plain one-per-line text file
    or any CSV with a comment_id column -- `--core-ids` accepts both, since
    the 50-row core set exists both as a plain list and as a full CSV
    (official-data/gold_sample_core_50.csv) depending on how it was saved."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        first_line = handle.readline()
        handle.seek(0)
        if "," in first_line and "comment_id" in first_line.lower():
            return {(row.get("comment_id") or "").strip()
                    for row in csv.DictReader(handle)} - {""}
        return {line.strip() for line in handle if line.strip()}


def example_ids_for_gold(gold_path, exclude_ids_path=None, silence=False):
    """The worked-example comment_ids to leave out of scoring `gold_path`, from every source.

    The 25 fixed examples are shown to the model with their own hand-coded answers, so a gold
    comment that is also an example would be scored against its own label. Two sources: the
    `<gold_path>.run.json` that `slens-classify` writes beside a labelled file (its
    fixed_examples.ids), and `--exclude-ids FILE`. Both are used when both exist.

    When neither names any, and the examples were not the built-in synthetic set (which no
    gold comment can be among), print a WARNING instead of staying silent; `silence` (the
    `--no-example-exclusion` flag) turns that into a deliberate, quiet choice.
    """
    file_ids = set()
    if exclude_ids_path:
        file_ids = read_ids(exclude_ids_path) - {"comment_id"}
    record_ids, builtin = set(), False
    record_path = f"{gold_path}.run.json"
    if gold_path and os.path.exists(record_path):
        try:
            with open(record_path, encoding="utf-8") as handle:
                examples = (json.load(handle) or {}).get("fixed_examples") or {}
        except (OSError, ValueError):
            examples = {}
        record_ids = set(examples.get("ids") or [])
        builtin = str(examples.get("source") or "").startswith("built-in")
    if record_ids:
        print(f"Leaving out the {len(record_ids)} worked-example comment(s) named in "
              f"{os.path.basename(record_path)}: they were shown to the model with their own "
              f"answers.", file=sys.stderr)
    ids = record_ids | file_ids
    if not ids and not builtin and not silence:
        print("WARNING: no worked-example ids are known (no run record with fixed_examples "
              f"beside {gold_path}, and no --exclude-ids), so any gold comment that was shown "
              "to the model as an example is being scored against its own answer. Pass "
              "--exclude-ids with the fixed-examples file, or --no-example-exclusion if "
              "none of the gold comments is an example.", file=sys.stderr)
    return ids


def _cell_text(value):
    """An Excel cell as the text every command here expects: blank -> "", a whole-number
    float -> "2" (not "2.0": a column with blanks comes back as floats, and "2.0" would
    not equal a coder's "2"), anything else -> str()."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def read_any(path):
    """Like `read`, but also accepts a coder's .xlsx file.

    Coder files are handed out as .xlsx (scripts_scratch_split_gold_set.py) so a coder
    can work in Excel. Read with openpyxl (the `analysis` extra declares it); pandas used
    to be required here and nothing declared it, so the command failed on a clean install
    with "No module named 'pandas'". The import is local so the rest of the CLI stays
    usable with only the core install. Only the first/active sheet is read; columns with no
    header are ignored.
    """
    lower = str(path).lower()
    if lower.endswith((".xlsx", ".xlsm")):
        try:
            import openpyxl
        except ImportError:
            raise SystemExit("Reading .xlsx needs openpyxl: pip install openpyxl "
                             "(or pip install -e \".[analysis]\")")
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        sheet = workbook.active
        rows_iter = sheet.iter_rows(values_only=True)
        header = next(rows_iter, None)
        if header is None:
            return []
        names = [(_cell_text(h)).strip() for h in header]
        records = []
        for values in rows_iter:
            record = {name: _cell_text(values[i]) if i < len(values) else ""
                      for i, name in enumerate(names) if name}
            if any(v.strip() for v in record.values()):
                records.append(record)
        workbook.close()
        return records
    if lower.endswith(".xls"):
        import pandas as pd
        df = pd.read_excel(path)
        return df.where(df.notna(), "").astype(str).to_dict("records")
    return read(path)


def cmd_agree(rows):
    judge = (rows[0].get("domain_judge") or "") if rows else ""
    if judge.startswith("keyword"):
        print("WARNING: this corpus was labelled by the keyword classifier "
              f"({judge}), so the `domain_*` and `kw_*` columns come from the same "
              "method. The agreement below is a classifier compared with itself and will "
              "read as perfect. Re-label with an LLM judge before using this as a "
              "cross-check.\n", file=sys.stderr)

    print(f"{len(rows)} comments\n")
    print(f"{'domain':<14}{'LLM':>8}{'keyword':>10}{'both':>8}{'kappa':>9}")
    for category in CATEGORIES:
        pairs = [(int(category in labels(r, "domain_")), int(category in labels(r, "kw_"))) for r in rows]
        llm = sum(a for a, _ in pairs)
        kw = sum(b for _, b in pairs)
        both = sum(1 for a, b in pairs if a and b)
        print(f"{category:<14}{llm:>8}{kw:>10}{both:>8}{kappa(pairs):>9.2f}")

    overlap = sum(jaccard(labels(r, "domain_"), labels(r, "kw_")) for r in rows) / max(len(rows), 1)
    exact = sum(1 for r in rows if labels(r, "domain_") == labels(r, "kw_")) / max(len(rows), 1)
    print(f"\nMean set overlap: {overlap:.2f}   Exact agreement: {exact:.1%}")
    print("\nKappa below about 0.4 on a domain means the two methods are not measuring "
          "the same thing, and neither figure should be published for it until the "
          "hand-coded sample says which one is closer.")


def labels_from_text(raw):
    """Parse a free-text hand_domain value ("democracy, economics", "none")
    into a set of categories, the same substring-membership rule
    evaluate.read_gold() uses. `agree`'s `labels()` above reads separate
    binary domain_<category>/kw_<category> columns instead -- that shape
    doesn't fit a coder's hand_domain column, which is one free-text field."""
    raw = (raw or "").strip().lower()
    return {c for c in CATEGORIES if c in raw}


def kappa_categorical(pairs):
    """Cohen's kappa for a judgement with any number of possible values, e.g.
    sentiment on a 1-5 scale. `kappa()` above is the 2-category (present/
    absent) special case of this same formula; this is the general version,
    needed because sentiment is one value from several, not a yes/no flag
    per category.

    UNWEIGHTED: a 1-vs-2 disagreement counts exactly the same as a 1-vs-5
    one. That is the right statistic for a genuinely nominal judgement (the
    three-word negative/neutral/positive form, or the categorical -1/0
    not-codable/irrelevant sentinels), but not for the numeric 1-5 scale,
    where CLAUDE.md's own stated design intent is that "a one-point
    disagreement is meaningfully different from a four-point one" -- see
    kappa_weighted below, which is what actually honours that for numeric
    pairs. Kept here, unchanged, rather than silently swapped out, since
    this is also what non-numeric sentiment values still need, and other
    code already reads `sentiment_kappa` built from this function.
    """
    n = len(pairs)
    if not n:
        return float("nan")
    observed = sum(1 for a, b in pairs if a == b) / n
    values = {v for pair in pairs for v in pair}
    freq_a = collections.Counter(a for a, _ in pairs)
    freq_b = collections.Counter(b for _, b in pairs)
    expected = sum((freq_a[v] / n) * (freq_b[v] / n) for v in values)
    # Same reasoning as `kappa` above: `expected == 1` here only happens when
    # every pair used the same single value throughout, which forces
    # `observed == 1` too -- perfect, not undefined.
    if expected >= 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def kappa_weighted(pairs):
    """Quadratic-weighted Cohen's kappa for numeric, ordinal pairs.

    Added so a 1-vs-2 sentiment disagreement is scored as a smaller miss
    than a 1-vs-5 one, which kappa_categorical's unweighted formula (above)
    does not do -- CLAUDE.md says this distinction should matter ("a
    one-point disagreement is meaningfully different from a four-point
    one"), but until this function existed nothing in the codebase actually
    implemented it; kappa_categorical was the only sentiment statistic on
    offer, and it is nominal, not ordinal.

    Standard formula: build the observed and expected confusion matrices
    over the categories actually present in the data, weight cell (i, j) by
    (i - j)^2 normalised to [0, 1] by the widest possible gap, and report
    1 - (weighted observed disagreement / weighted expected disagreement).
    Only meaningful for pairs that are genuinely numeric and ordered --
    callers must filter to numeric-parseable pairs first (see
    cmd_interrater and cmd_selfconsistency, the two callers), not pass this
    the same raw strings kappa_categorical tolerates.
    """
    n = len(pairs)
    if not n:
        return float("nan")
    values = sorted({v for pair in pairs for v in pair})
    if len(values) < 2:
        # Every single vote, on both sides of every pair, was the identical
        # number -- e.g. a self-consistency fixture where the model gave the
        # same rating every time. There is no spread to weight a distance
        # against, but that is "perfect agreement", not "undefined": nothing
        # here ever disagreed, weighted or not. Kept as a distinct branch from
        # `kappa`/`kappa_categorical`'s equivalent case rather than falling
        # through to `spread = 0` below, which would divide by zero instead.
        return 1.0
    spread = (values[-1] - values[0]) ** 2
    freq_a = collections.Counter(a for a, _ in pairs)
    freq_b = collections.Counter(b for _, b in pairs)

    def weight(i, j):
        return ((i - j) ** 2) / spread

    observed = sum(weight(a, b) for a, b in pairs) / n
    expected = sum((freq_a[i] / n) * (freq_b[j] / n) * weight(i, j)
                   for i in values for j in values)
    return 1 - (observed / expected) if expected else float("nan")


def fleiss_kappa(item_ratings):
    """Fleiss' kappa: Cohen's kappa generalised from exactly two raters to any
    number of raters (Cohen's own formula, `kappa`/`kappa_categorical` above,
    is only defined for a pair). Needed once the gold set has more than two
    coders overlapping on the same "core" rows -- five coders' agreement
    cannot be reduced to a single pairwise number without picking one pair
    arbitrarily or averaging pairs together in a way nothing here previously
    did on purpose.

    Same [-1, 1] scale and the same rough interpretation as Cohen's kappa (0 =
    no better than chance, 1 = perfect, below ~0.4 needs reconciling) -- it is
    the standard choice specifically because it stays interpretable on that
    familiar scale rather than requiring a new mental model, unlike
    Krippendorff's alpha, which is more general (handles missing ratings and
    ordinal weighting) but is not a drop-in reading of "kappa below 0.4".

    `item_ratings` is one list per item (comment), each holding that item's
    rating from every coder -- e.g. `[[1, 1, 0, 1, 1], [0, 0, 0, 1, 0], ...]`
    for a binary domain flag across 5 coders on 2 comments, or
    `["positive", "positive", "neutral", "positive", "negative"]`-style string
    ratings for sentiment. Every item must carry the SAME number of ratings
    (Fleiss' formula assumes a constant n raters-per-item) -- filter to
    comments every coder in the comparison actually coded before calling this,
    the same way `cmd_interrater` already restricts a 2-coder comparison to
    `shared_ids`.
    """
    if not item_ratings:
        return float("nan")
    n = len(item_ratings[0])
    if n < 2 or any(len(ratings) != n for ratings in item_ratings):
        return float("nan")
    categories = sorted({v for ratings in item_ratings for v in ratings})
    n_items = len(item_ratings)
    counts_per_item = []
    for ratings in item_ratings:
        counts = collections.Counter(ratings)
        counts_per_item.append([counts.get(cat, 0) for cat in categories])

    per_item_agreement = [
        (sum(n_ij * n_ij for n_ij in row) - n) / (n * (n - 1))
        for row in counts_per_item
    ]
    p_bar = sum(per_item_agreement) / n_items

    totals = [sum(row[j] for row in counts_per_item) for j in range(len(categories))]
    category_shares = [t / (n_items * n) for t in totals]
    p_e = sum(share * share for share in category_shares)

    # Same reasoning as kappa()/kappa_categorical() above: `p_e` can only
    # reach 1 when every single rating, from every rater on every item, was
    # the same one category -- which forces `p_bar` to be 1 too (perfect
    # agreement), not undefined.
    if p_e >= 1:
        return 1.0
    return (p_bar - p_e) / (1 - p_e)


def krippendorff_alpha_interval(item_ratings):
    """Krippendorff's alpha with an interval (squared-distance) metric -- the
    multi-rater counterpart to `kappa_weighted` above, needed because Fleiss'
    kappa (the multi-rater counterpart to plain Cohen's kappa) has no standard
    ordinal/interval-weighted version the way Cohen's kappa does. Krippendorff's
    alpha is the standard statistic that DOES generalise cleanly to any number
    of raters with any difference metric plugged in; squared distance is used
    here for the same reason `kappa_weighted` uses it for two coders -- a
    1-vs-2 sentiment gap is a smaller disagreement than a 1-vs-5 one, and this
    is what actually scores it that way rather than nominal all-or-nothing.

    Only meaningful for genuinely numeric, ordered ratings (the 1-5 sentiment
    scale) -- callers must filter to comments where every coder gave a numeric
    value first, the same restriction `kappa_weighted` already places on its
    own numeric-pairs input.

    `item_ratings` is one list per comment, holding that comment's numeric
    rating from every coder, same shape as `fleiss_kappa`'s input but every
    value a float/int. Every item needs the same number of ratings, same
    restriction as `fleiss_kappa`.

    Standard formula for the constant-raters-per-unit case:
        Do = mean squared difference between two ratings of the SAME comment
             (every ordered pair of coders, within each comment, pooled)
        De = mean squared difference between two ratings drawn from the
             WHOLE pool of ratings (every ordered pair across the entire
             dataset, ignoring which comment they came from)
        alpha = 1 - Do / De
    De is what "the two ratings could have differed by, if agreement were
    pure chance" -- the same role `kappa_weighted`'s expected-disagreement
    term plays for two coders, just computed by pooling every rating instead
    of from two coders' marginal frequency tables.
    """
    if not item_ratings:
        return float("nan")
    n = len(item_ratings[0])
    if n < 2 or any(len(ratings) != n for ratings in item_ratings):
        return float("nan")

    do_sum, do_pairs = 0.0, 0
    for ratings in item_ratings:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                diff = ratings[i] - ratings[j]
                do_sum += diff * diff
                do_pairs += 1
    if do_pairs == 0:
        return float("nan")
    observed = do_sum / do_pairs

    pooled = [v for ratings in item_ratings for v in ratings]
    total = len(pooled)
    de_sum, de_pairs = 0.0, 0
    for i in range(total):
        for j in range(total):
            if i == j:
                continue
            diff = pooled[i] - pooled[j]
            de_sum += diff * diff
            de_pairs += 1
    if not de_pairs:
        return float("nan")
    expected = de_sum / de_pairs
    # `expected` (De) reaching 0 means every rating ever recorded, across
    # every item and every coder, was the identical number -- which forces
    # `observed` (Do) to be 0 too, since there is nothing left to differ.
    # Same reasoning as the kappa functions' expected==1 case above, just
    # phrased in terms of disagreement instead of agreement: zero possible
    # spread is perfect agreement, not an undefined ratio.
    if expected == 0:
        return 1.0
    return 1 - observed / expected


def krippendorff_alpha(item_ratings, metric="nominal"):
    """Krippendorff's alpha for complete data: every item rated by the same number of coders.

    One statistic for every judgement in the study, with the level of measurement stated:
      * "nominal"  -- categories with no order (is a harm category present or not; the KIND
                      of sentiment answer: a stance, irrelevant, or uncodeable);
      * "ordinal"  -- ordered categories with unknown spacing (the 1-5 stance scale: 1 vs 2
                      is a smaller disagreement than 1 vs 5, without assuming the steps are
                      equal);
      * "interval" -- equal steps (squared difference).
    alpha = 1 - observed disagreement / disagreement expected by chance (pooling every rating).
    1 is perfect agreement, 0 is what chance gives, below 0 is worse than chance. Returns NaN
    when there is no variation at all (every rating the same), because chance disagreement is
    then zero and the ratio is undefined: that is "untested", not "perfect".

    Written with the coincidence matrix (Krippendorff 2011), so the ordinal metric uses the
    value frequencies as the definition requires.
    """
    if not item_ratings:
        return float("nan")
    m = len(item_ratings[0])
    if m < 2 or any(len(r) != m for r in item_ratings):
        return float("nan")
    values = sorted({v for r in item_ratings for v in r}, key=lambda v: (str(type(v)), v))
    if len(values) < 2:
        return float("nan")
    index = {v: i for i, v in enumerate(values)}
    k = len(values)
    # coincidence matrix: each ordered pair of coders on one item adds 1 / (m - 1)
    o = [[0.0] * k for _ in range(k)]
    for ratings in item_ratings:
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[index[ratings[i]]][index[ratings[j]]] += 1.0 / (m - 1)
    n_c = [sum(row) for row in o]
    n = sum(n_c)
    if metric == "nominal":
        def delta2(c, d):
            return 0.0 if c == d else 1.0
    elif metric == "interval":
        def delta2(c, d):
            return (values[c] - values[d]) ** 2
    elif metric == "ordinal":
        def delta2(c, d):
            if c == d:
                return 0.0
            lo, hi = min(c, d), max(c, d)
            return (sum(n_c[lo:hi + 1]) - (n_c[lo] + n_c[hi]) / 2.0) ** 2
    else:
        raise ValueError(f"metric must be nominal, ordinal or interval, not {metric!r}")
    observed = sum(o[c][d] * delta2(c, d) for c in range(k) for d in range(k))
    expected = sum(n_c[c] * n_c[d] * delta2(c, d) for c in range(k) for d in range(k))
    if expected == 0:
        return float("nan")
    return 1.0 - (n - 1) * observed / expected


def bootstrap_alpha_ci(item_ratings, metric="nominal", n_boot=2000, seed=42, level=0.95):
    """(lo, hi) percentile interval for alpha, resampling ITEMS (comments) with replacement.

    With 50 comments an interval is wide, and that is the honest picture: report it beside
    the point value. A resample with no variation has no alpha and is skipped; if fewer than
    half the resamples have one, no interval is returned (NaN, NaN).
    """
    if len(item_ratings) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(item_ratings)
    draws = []
    for _ in range(n_boot):
        value = krippendorff_alpha([item_ratings[rng.randrange(n)] for _ in range(n)], metric)
        if value == value:
            draws.append(value)
    if len(draws) < n_boot / 2:
        return float("nan"), float("nan")
    draws.sort()
    tail = (1 - level) / 2
    return draws[int(tail * len(draws))], draws[min(int((1 - tail) * len(draws)), len(draws) - 1)]


def _rating_kind(value):
    """The KIND of sentiment answer: a stance, 'irrelevant' (0) or 'uncodeable' (-1)."""
    text = str(value).strip().lower()
    if text in ("0", "0.0", "irrelevant"):
        return "irrelevant"
    if text in ("-1", "-1.0", "uncodeable"):
        return "uncodeable"
    return "stance"


KRIPPENDORFF_GUIDE = ("Krippendorff's guide: alpha of 0.80 or more is reliable; 0.667 to 0.80 "
                      "supports tentative conclusions only; below 0.667 is not reliable. With "
                      "few comments the interval is wide: read the point value and the interval "
                      "together.")


def cmd_interrater_multi(rows_by_coder, domain_column="hand_domain",
                         sentiment_column="hand_sentiment", json_output=None, names=None,
                         n_boot=2000, sentiment_levels="scale"):
    """Agreement among THREE OR MORE coders who independently coded the same comments.

    The primary statistic is Krippendorff's alpha, with a 95% interval from resampling
    comments, at the level of measurement each judgement has:
      * harm domain: one binary judgement per category (present or not), nominal alpha;
      * sentiment, part 1: the KIND of answer (stance / irrelevant / uncodeable), nominal
        alpha, because 0 and -1 are categories and coders differ most on whether to use them;
      * sentiment, part 2: how strong a stance, ordinal alpha, on the comments where every
        coder gave a 1-5 rating.
    Fleiss' kappa is printed beside the domain alpha for familiarity (they agree closely).
    Restricts to comments every coder coded; a blank cell is not coded, never "none".
    """
    names = list(names) if names else [f"coder {i + 1}" for i in range(len(rows_by_coder))]
    by_id_per_coder = [
        {r["comment_id"]: r for r in rows if r.get("comment_id")}
        for rows in rows_by_coder
    ]
    id_sets = [set(by_id) for by_id in by_id_per_coder]
    shared_ids = sorted(set.intersection(*id_sets)) if id_sets else []
    only_some = set.union(*id_sets) - (set.intersection(*id_sets) if id_sets else set())
    if only_some:
        print(f"WARNING: {len(only_some)} comment_id(s) were not coded by every one of the "
              f"{len(rows_by_coder)} coders -- excluded from the comparison.", file=sys.stderr)
    if not shared_ids:
        sys.exit("No comment_id values shared across every coder file -- nothing to compare.")

    print(f"{len(shared_ids)} comments coded by all {len(rows_by_coder)} coders "
          f"({', '.join(names)})\n")
    print("Primary statistic: Krippendorff's alpha, 95% interval from resampling comments "
          f"({n_boot:,} resamples). Fleiss' kappa shown beside the domain figures.")
    print(KRIPPENDORFF_GUIDE + "\n")

    # Blank hand_domain = not coded, never "none" (see cmd_interrater).
    domain_ids = [cid for cid in shared_ids
                  if all((by_id[cid].get(domain_column) or "").strip()
                         for by_id in by_id_per_coder)]
    if len(domain_ids) < len(shared_ids):
        print(f"NOTE: {len(shared_ids) - len(domain_ids)} comment(s) have a blank "
              f"{domain_column} from at least one coder; left out of the domain figures "
              f"(blank means not coded, not 'none').", file=sys.stderr)
    if not domain_ids:
        sys.exit(f"No comment has a {domain_column} from every coder -- nothing to compare.")

    print("HARM DOMAIN (one present/absent judgement per category; nominal alpha)")
    print(f"{'category':<13}{'any':>5}{'all':>5}{'exact':>8}{'alpha':>8}   {'95% interval':<16}"
          f"{'kappa':>7}")
    per_category, domain_json, no_variation = {}, {}, []
    for category in CATEGORIES:
        ratings = [
            [int(category in labels_from_text(by_id[cid].get(domain_column)))
             for by_id in by_id_per_coder]
            for cid in domain_ids
        ]
        per_category[category] = ratings
        n_any = sum(1 for r in ratings if any(r))
        n_all = sum(1 for r in ratings if all(r))
        exact_cat = sum(1 for r in ratings if len(set(r)) == 1) / len(ratings)
        alpha = krippendorff_alpha(ratings, "nominal")
        lo, hi = bootstrap_alpha_ci(ratings, "nominal", n_boot)
        kappa_f = fleiss_kappa(ratings)
        if alpha != alpha:
            no_variation.append(category)
            alpha_text, ci_text, kappa_text = "n/a", "n/a", "n/a"
        else:
            alpha_text = f"{alpha:.2f}"
            ci_text = "n/a" if lo != lo else f"{lo:.2f} to {hi:.2f}"
            kappa_text = f"{kappa_f:.2f}"
        print(f"{category:<13}{n_any:>5}{n_all:>5}{exact_cat:>8.0%}{alpha_text:>8}   "
              f"{ci_text:<16}{kappa_text:>7}")
        # No variation (alpha undefined): Fleiss' kappa would come out 1.0 or 0, which says
        # nothing, so it is null in the file, as the terminal's "n/a" says. `kappa` is the
        # key the dashboard's coder-agreement table reads; for three or more coders it carries
        # Fleiss' kappa, the multi-coder form of the same statistic.
        kappa_f = None if alpha != alpha else kappa_f
        domain_json[category] = {"alpha": alpha, "ci_lo": lo, "ci_hi": hi,
                                 "fleiss_kappa": kappa_f, "kappa": kappa_f,
                                 "no_variation": alpha != alpha,
                                 "tagged_by_any": n_any,
                                 "tagged_by_all": n_all, "exact_agreement": exact_cat}
    print("any / all = comments at least one / every coder tagged with the category; "
          "exact = comments where all coders agree on it either way.")
    if no_variation:
        print(f"n/a for {', '.join(no_variation)}: every coder gave every comment the same "
              f"answer, so alpha is undefined. The category is untested here, not reliable. "
              f"Report it as untested.")

    exact = sum(
        1 for cid in domain_ids
        if len({frozenset(labels_from_text(by_id[cid].get(domain_column)))
                for by_id in by_id_per_coder}) == 1
    ) / len(domain_ids)
    print(f"\nExact agreement on the full set of domains, all coders: {exact:.1%} "
          f"of {len(domain_ids)} comments")

    # ---- sentiment
    sentiment_rows = [
        [(by_id[cid].get(sentiment_column) or "").strip().lower() for by_id in by_id_per_coder]
        for cid in shared_ids
        if all((by_id[cid].get(sentiment_column) or "").strip() for by_id in by_id_per_coder)
    ]
    sentiment_json = {}
    if not sentiment_rows:
        print("\nSentiment: no comments where every coder filled in a value.", file=sys.stderr)
    else:
        print(f"\nSENTIMENT ({len(sentiment_rows)} comments every coder rated)")
        kinds = [[_rating_kind(v) for v in r] for r in sentiment_rows]
        kind_alpha = krippendorff_alpha(kinds, "nominal")
        k_lo, k_hi = bootstrap_alpha_ci(kinds, "nominal", n_boot)
        kind_counts = collections.Counter(k for r in kinds for k in r)
        kind_exact = sum(1 for r in kinds if len(set(r)) == 1) / len(kinds)
        print(f"  (a) Kind of answer: stance / irrelevant (0) / uncodeable (-1), nominal alpha")
        if kind_alpha == kind_alpha:
            ci = "n/a" if k_lo != k_lo else f"{k_lo:.2f} to {k_hi:.2f}"
            print(f"      alpha {kind_alpha:.2f}  (95% interval {ci});  "
                  f"all coders agree on the kind for {kind_exact:.0%} of comments")
        else:
            print("      n/a: every coder gave every comment the same kind of answer")
        print("      answers given: " + ", ".join(
            f"{name} {kind_counts[name]}" for name in ("stance", "irrelevant", "uncodeable")))
        pair_agreement = {}
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                same = sum(1 for r in kinds if r[a] == r[b]) / len(kinds)
                pair_agreement[f"{names[a]} / {names[b]}"] = same
        print("      pairwise agreement on the kind: " + "; ".join(
            f"{pair} {share:.0%}" for pair, share in pair_agreement.items()))
        per_coder_zero = {names[i]: sum(1 for r in kinds if r[i] == "irrelevant")
                          for i in range(len(names))}
        print("      comments each coder called irrelevant: " + ", ".join(
            f"{n} {c}" for n, c in per_coder_zero.items())
            + "  (a large spread here is a calibration difference between coders)")

        numeric = []
        bucket_json = None
        if sentiment_levels == "buckets":
            # Direction of the stance only: 1-2 negative, 3 neutral, 4-5 positive (or the
            # word the coder wrote). Compared as three ordered classes, on the comments
            # where EVERY coder gave a stance (a comment any coder called irrelevant or
            # uncodeable has no direction to compare; part (a) already covers that).
            order = {"negative": 1, "neutral": 2, "positive": 3}
            coded, per_coder_counts = [], [collections.Counter() for _ in names]
            for ratings in sentiment_rows:
                if any(_rating_kind(v) != "stance" for v in ratings):
                    continue
                buckets = [sentiment_bucket(v) for v in ratings]
                if all(b in order for b in buckets):
                    coded.append([order[b] for b in buckets])
                    for i, b in enumerate(buckets):
                        per_coder_counts[i][b] += 1
            print(f"  (b) Direction of the stance: negative / neutral / positive (the 1-5 scale "
                  f"in three buckets), ordinal alpha, on the {len(coded)} comments every coder "
                  f"gave a stance")
            b_alpha = krippendorff_alpha(coded, "ordinal") if coded else float("nan")
            b_lo, b_hi = bootstrap_alpha_ci(coded, "ordinal", n_boot) if coded else (
                float("nan"), float("nan"))
            if b_alpha == b_alpha:
                ci = "n/a" if b_lo != b_lo else f"{b_lo:.2f} to {b_hi:.2f}"
                same = sum(1 for r in coded if len(set(r)) == 1) / len(coded)
                print(f"      alpha {b_alpha:.2f}  (95% interval {ci});  all coders in the same "
                      f"bucket for {same:.0%} of comments")
            else:
                print("      n/a: too few comments, or every coder gave every comment the "
                      "same bucket (no variation: untested, not perfect)")
            print("      buckets used: " + "; ".join(
                f"{names[i]} (neg {c['negative']}, neu {c['neutral']}, pos {c['positive']})"
                for i, c in enumerate(per_coder_counts)))
            # (c) Every comment, none left out: five categories, so a disagreement about
            # whether a comment is irrelevant counts here instead of dropping the comment.
            five = []
            for ratings in sentiment_rows:
                five.append([_rating_kind(v) if _rating_kind(v) != "stance"
                             else sentiment_bucket(v) for v in ratings])
            f_alpha = krippendorff_alpha(five, "nominal")
            f_lo, f_hi = bootstrap_alpha_ci(five, "nominal", n_boot)
            f_exact = sum(1 for r in five if len(set(r)) == 1) / len(five)
            print(f"  (c) Every comment, five categories (negative / neutral / positive / "
                  f"irrelevant / uncodeable), nominal alpha, on all {len(five)} comments")
            if f_alpha == f_alpha:
                ci = "n/a" if f_lo != f_lo else f"{f_lo:.2f} to {f_hi:.2f}"
                print(f"      alpha {f_alpha:.2f}  (95% interval {ci});  all coders give the "
                      f"same answer for {f_exact:.0%} of comments")
            else:
                print("      n/a: every coder gave every comment the same answer")
            print(f"      (b) leaves out {len(sentiment_rows) - len(coded)} of "
                  f"{len(sentiment_rows)} comments because at least one coder called them "
                  f"irrelevant or uncodeable, and the rest are the clearer ones, so (b) alone "
                  f"flatters agreement. Report (a), (b) and (c) together.")
            bucket_json = {"alpha_ordinal": b_alpha, "ci_lo": b_lo, "ci_hi": b_hi,
                           "all_comments_five_categories": {
                               "alpha_nominal": f_alpha, "ci_lo": f_lo, "ci_hi": f_hi,
                               "n": len(five), "exact_agreement": f_exact},
                           "n": len(coded),
                           "exact_agreement": (sum(1 for r in coded if len(set(r)) == 1)
                                               / len(coded)) if coded else None}
        for ratings in ([] if sentiment_levels == "buckets" else sentiment_rows):
            try:
                numbers = [float(v) for v in ratings]
            except ValueError:
                continue
            if all(1 <= v <= 5 for v in numbers):
                numeric.append(numbers)
        if sentiment_levels != "buckets":
            print(f"  (b) How strong a stance, 1-5, ordinal alpha, on the {len(numeric)} comments "
                  f"every coder rated 1-5")
        ord_alpha = krippendorff_alpha(numeric, "ordinal") if numeric else float("nan")
        o_lo, o_hi = bootstrap_alpha_ci(numeric, "ordinal", n_boot) if numeric else (
            float("nan"), float("nan"))
        if sentiment_levels == "buckets":
            pass
        elif ord_alpha == ord_alpha:
            ci = "n/a" if o_lo != o_lo else f"{o_lo:.2f} to {o_hi:.2f}"
            exact_num = sum(1 for r in numeric if len(set(r)) == 1) / len(numeric)
            within1 = sum(1 for r in numeric if max(r) - min(r) <= 1) / len(numeric)
            print(f"      alpha {ord_alpha:.2f}  (95% interval {ci});  identical for "
                  f"{exact_num:.0%}, within one point for {within1:.0%}")
        else:
            print("      n/a: too few comments, or no variation, to compute it")
        sentiment_fleiss = fleiss_kappa(sentiment_rows)
        if sentiment_levels != "buckets":
            print(f"  Fleiss' kappa on the raw values, for comparison: {sentiment_fleiss:.2f}")
        sentiment_json = {
            "n": len(sentiment_rows),
            "kind": {"alpha": kind_alpha, "ci_lo": k_lo, "ci_hi": k_hi,
                     "exact_agreement": kind_exact, "answers": dict(kind_counts),
                     "pairwise_agreement": pair_agreement, "called_irrelevant": per_coder_zero},
            "stance_1_to_5": {"alpha_ordinal": ord_alpha, "ci_lo": o_lo, "ci_hi": o_hi,
                              "n": len(numeric)},
            "fleiss_kappa_raw_values": sentiment_fleiss,
            "levels": sentiment_levels,
            "stance_buckets": bucket_json,
        }

    if json_output:
        payload = {
            "statistic": "krippendorff_alpha",
            "n_shared": len(shared_ids),
            "n_domain": len(domain_ids),
            "n_coders": len(rows_by_coder),
            "coders": names,
            "domain": domain_json,
            "domain_exact_agreement": exact,
            "sentiment": sentiment_json,
            # kept so anything reading the earlier output still finds these
            "sentiment_n": len(sentiment_rows),
            # The keys dashboard/index.html reads for the coder-agreement section, so a
            # multi-coder file does not leave its sentence blank: Fleiss' kappa on the raw
            # sentiment values and the share of comments all coders rated identically.
            # Null when every coder gave every comment one and the same value (no variation).
            "sentiment_kappa": (None if len({v for r in sentiment_rows for v in r}) < 2
                                else sentiment_json.get("fleiss_kappa_raw_values")),
            "sentiment_exact_agreement": (
                sum(1 for r in sentiment_rows if len(set(r)) == 1) / len(sentiment_rows)
                if sentiment_rows else None),
            "sentiment_fleiss_kappa": sentiment_json.get("fleiss_kappa_raw_values"),
            "sentiment_alpha_ordinal": (sentiment_json.get("stance_1_to_5") or {}).get(
                "alpha_ordinal"),
            "sentiment_alpha_ordinal_n": (sentiment_json.get("stance_1_to_5") or {}).get("n", 0),
        }
        clean = json.loads(json.dumps(payload, default=lambda v: None).replace("NaN", "null"))
        out_dir = os.path.dirname(json_output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(json_output, "w", encoding="utf-8") as handle:
            json.dump(clean, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"\nWrote {json_output}", file=sys.stderr)


def cmd_interrater(rows_a, rows_b, domain_column="hand_domain",
                   sentiment_column="hand_sentiment", json_output=None):
    """Cohen's kappa between two coders' independent codings of the same
    comments.

    `agree` above compares the two *machine* methods (domain_ vs kw_ columns)
    within one file. It cannot check whether two humans agree with each
    other, because that needs two separate files -- each coder works from
    their own copy of the same hand-coding sample -- matched back together by
    comment_id. This is that comparison.

    A comment_id present in only one file cannot be scored and is reported,
    not silently dropped -- if that count is large, the two coders were not
    actually given the same file.
    """
    by_id_a = {r["comment_id"]: r for r in rows_a if r.get("comment_id")}
    by_id_b = {r["comment_id"]: r for r in rows_b if r.get("comment_id")}
    shared_ids = sorted(set(by_id_a) & set(by_id_b))
    only_a, only_b = set(by_id_a) - set(by_id_b), set(by_id_b) - set(by_id_a)
    if only_a or only_b:
        print(f"WARNING: {len(only_a)} comment_id(s) only in the first file, "
              f"{len(only_b)} only in the second -- excluded from the comparison.",
              file=sys.stderr)
    if not shared_ids:
        sys.exit("No shared comment_id values between the two files -- nothing to compare.")

    print(f"{len(shared_ids)} comments coded by both\n")
    # A blank hand_domain is "not coded", never "none": counting it as an empty
    # label set made two coders who both left a cell blank agree that nothing
    # applied, and one blank against a real label a disagreement about nothing.
    # (20 identical rows plus 30 blanks gave kappa 0.48 instead of 1.00.)
    domain_ids = [cid for cid in shared_ids
                  if (by_id_a[cid].get(domain_column) or "").strip()
                  and (by_id_b[cid].get(domain_column) or "").strip()]
    if len(domain_ids) < len(shared_ids):
        print(f"NOTE: {len(shared_ids) - len(domain_ids)} comment(s) have a blank "
              f"{domain_column} from one or both coders; they are left out of the domain "
              f"figures (blank means not coded, not 'none').", file=sys.stderr)
    if not domain_ids:
        sys.exit(f"No comment has a {domain_column} from both coders -- nothing to compare.")
    print(f"{'domain':<14}{'coder A':>9}{'coder B':>9}{'both':>8}{'kappa':>9}")
    per_category = {}
    no_variation = []
    for category in CATEGORIES:
        pairs = [(int(category in labels_from_text(by_id_a[cid].get(domain_column))),
                  int(category in labels_from_text(by_id_b[cid].get(domain_column))))
                 for cid in domain_ids]
        per_category[category] = pairs
        a_count, b_count = sum(a for a, _ in pairs), sum(b for _, b in pairs)
        both = sum(1 for a, b in pairs if a and b)
        flag = ""
        if a_count == b_count and a_count in (0, len(pairs)):
            flag = " *"
            no_variation.append(category)
        print(f"{category:<14}{a_count:>9}{b_count:>9}{both:>8}{kappa(pairs):>9.2f}{flag}")
    if no_variation:
        print(f"* no variation in {', '.join(no_variation)} among these comments (every "
              f"comment labelled the same way by both coders): the 1.00 is agreement on "
              f"nothing in particular, not evidence the category is reliable. Report it as "
              f"untested.")

    exact = sum(1 for cid in domain_ids
               if labels_from_text(by_id_a[cid].get(domain_column))
               == labels_from_text(by_id_b[cid].get(domain_column))) / len(domain_ids)
    overlap = sum(jaccard(labels_from_text(by_id_a[cid].get(domain_column)),
                         labels_from_text(by_id_b[cid].get(domain_column)))
                 for cid in domain_ids) / len(domain_ids)
    print(f"\nDomain -- mean set overlap: {overlap:.2f}   Exact agreement: {exact:.1%}")

    sentiment_pairs = [
        ((by_id_a[cid].get(sentiment_column) or "").strip().lower(),
         (by_id_b[cid].get(sentiment_column) or "").strip().lower())
        for cid in shared_ids
        if (by_id_a[cid].get(sentiment_column) or "").strip()
        and (by_id_b[cid].get(sentiment_column) or "").strip()
    ]
    sentiment_exact = sentiment_kappa = sentiment_kappa_weighted = None
    numeric_sentiment_pairs = []
    # Both coders used one and the same answer on every comment: kappa comes out 1.0 by
    # construction, not as evidence of agreement (same flag as the domain categories).
    sentiment_no_variation = (bool(sentiment_pairs)
                              and len({a for a, _ in sentiment_pairs}
                                      | {b for _, b in sentiment_pairs}) == 1)
    sentiment_weighted_no_variation = False
    if sentiment_pairs:
        sentiment_exact = sum(1 for a, b in sentiment_pairs if a == b) / len(sentiment_pairs)
        sentiment_kappa = kappa_categorical(sentiment_pairs)
        # Weighted kappa needs genuinely numeric, ordered pairs -- a coder
        # writing "negative" instead of a 1-5 number has no distance to
        # anything else on the scale, so those rows are excluded here rather
        # than crashing or being silently coerced to some arbitrary number.
        for a, b in sentiment_pairs:
            try:
                fa, fb = float(a), float(b)
            except ValueError:
                continue
            # Only genuine 1-5 stance ratings have a distance. 0 (irrelevant) and
            # -1 (uncodeable) are categories, not points "below very negative".
            if 1 <= fa <= 5 and 1 <= fb <= 5:
                numeric_sentiment_pairs.append((fa, fb))
        if numeric_sentiment_pairs:
            sentiment_kappa_weighted = kappa_weighted(numeric_sentiment_pairs)
            sentiment_weighted_no_variation = len({v for pair in numeric_sentiment_pairs
                                                   for v in pair}) == 1
        print(f"\nSentiment -- {len(sentiment_pairs)} comment(s) both coders scored, "
              f"exact agreement: {sentiment_exact:.1%}, kappa: {sentiment_kappa:.2f}"
              + (f", weighted kappa (numeric pairs only, n={len(numeric_sentiment_pairs)}): "
                 f"{sentiment_kappa_weighted:.2f}" if numeric_sentiment_pairs else ""))
        if sentiment_no_variation:
            print("  * every sentiment answer was the same one on both sides: that kappa is "
                  "agreement on nothing in particular (null in the JSON), not evidence the "
                  "scale is applied reliably.")
    else:
        print("\nSentiment: no comments where both coders filled in a value.", file=sys.stderr)

    if json_output:
        # The dashboard needs this as data, not as terminal output. Without a
        # machine-readable emitter the coder-agreement figure -- the ceiling on
        # every accuracy claim in the project -- could only ever be retyped by
        # hand into a screenshot, which is how a number stops being evidence.
        payload = {
            "n_shared": len(shared_ids),
            "n_domain": len(domain_ids),
            # Null kappa (and no_variation: true) where both coders gave every comment the same
            # answer: kappa() returns 1.0 there, "agreement on nothing in particular", which the
            # terminal marks with an asterisk and which must not reach the dashboard as a
            # perfect score and a green "agreed" pill.
            "domain": {c: {"coder_a": int(sum(a for a, _ in prs)),
                           "coder_b": int(sum(b for _, b in prs)),
                           "both": int(sum(1 for a, b in prs if a and b)),
                           "no_variation": c in no_variation,
                           "kappa": None if c in no_variation else kappa(prs)}
                       for c, prs in per_category.items()},
            "domain_mean_overlap": overlap,
            "domain_exact_agreement": exact,
            "sentiment_n": len(sentiment_pairs),
            "sentiment_exact_agreement": sentiment_exact,
            "sentiment_kappa": (None if sentiment_pairs and sentiment_no_variation
                                else sentiment_kappa),
            "sentiment_no_variation": bool(sentiment_pairs and sentiment_no_variation),
            "sentiment_kappa_weighted": (None if sentiment_weighted_no_variation
                                         else sentiment_kappa_weighted),
            "sentiment_kappa_weighted_n": len(numeric_sentiment_pairs),
        }
        clean = json.loads(json.dumps(payload, default=lambda v: None).replace("NaN", "null"))
        out_dir = os.path.dirname(json_output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(json_output, "w", encoding="utf-8") as handle:
            json.dump(clean, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"\nWrote {json_output}", file=sys.stderr)

    print("\nKappa below about 0.4 signals the coders (or the category definitions) need "
          "to be reconciled before this label is trusted as ground truth. Adjudicate "
          "disagreements on this shared set before drawing conclusions from either "
          "coder's labels on the rest of the sample.")


def _vote_pairs(votes):
    """Every unordered pair of a comment's own votes -- (a, b) once per pair,
    not (a, b) and (b, a) both, since kappa()/kappa_weighted() treat their
    input as unordered ratings, not "rater 1 vs rater 2" in a fixed order."""
    return [(votes[i], votes[j]) for i in range(len(votes)) for j in range(i + 1, len(votes))]


def parse_domain_raw_votes(raw):
    """"democracy,social|none|democracy" -> [{"democracy","social"}, set(), {"democracy"}].

    Matches classify.py's own serialisation exactly (see finish() there) --
    "|" between votes, "," within a vote, "none" for an empty ballot (not an
    empty string, which would collapse under ''.join/split ambiguity between
    "no votes recorded" and "one vote that found nothing")."""
    raw = (raw or "").strip()
    if not raw:
        return []
    ballots = []
    for part in raw.split("|"):
        part = part.strip()
        if not part or part == "none":
            ballots.append(set())
        else:
            ballots.append({label.strip() for label in part.split(",") if label.strip()})
    return ballots


def parse_sentiment_raw_votes(raw):
    """"4,4,3" -> [4, 4, 3]. Non-numeric tokens are dropped rather than
    raising -- a cache/parse hiccup on one vote should not crash the whole
    self-consistency run over a corpus of thousands of rows."""
    raw = (raw or "").strip()
    if not raw:
        return []
    votes = []
    for token in raw.split(","):
        token = token.strip()
        try:
            votes.append(int(token))
        except ValueError:
            continue
    return votes


def cmd_selfconsistency(rows, core_ids=None, compare_json=None):
    """How much does the LLM disagree with ITSELF across its own --votes, on
    the same statistical footing as cmd_interrater measures how much two
    HUMANS disagree with each other -- an answer to "does the AI vary more
    than the human coders?" that raw unstable-rate percentages can't give,
    since accuracy(kappa) and a raw disagreement rate are different units.

    Needs a classified CSV with domain_raw_votes/sentiment_raw_votes columns
    (classify.py writes these; only populated at --votes >= 2, since there
    is nothing for one vote to disagree with). `core_ids`, if given, is the
    set of comment_id values also coded by every human coder (the "core" 50)
    -- pass it (read from e.g. official-data/gold_sample_core_50.csv) so the
    comparison against cmd_interrater's output is on the SAME comments, not
    an apples-to-oranges comparison of self-consistency measured over the
    whole corpus against human agreement measured over 50 rows.

    `compare_json` is the `--json-output` a prior `slens-validate interrater`
    run wrote, if you have one -- printed alongside this run's figures for a
    direct side-by-side rather than two separate outputs to reconcile by eye.
    """
    if core_ids is not None:
        rows = [r for r in rows if (r.get("comment_id") or "").strip() in core_ids]
    if not rows:
        sys.exit("No rows to check -- the core-ids filter may not match this file's "
                 "comment_id values." if core_ids is not None else "No rows in input.")

    print(f"{len(rows)} comment(s) checked for self-consistency across votes\n")

    # --- domain: pairwise kappa across a comment's own votes, per category,
    # pooled across every comment -- the same pooling cmd_interrater already
    # uses across comments, just extended to vote-pairs within them too.
    domain_pair_counts = 0
    per_category_pairs = {c: [] for c in CATEGORIES}
    votes_per_comment = []
    for row in rows:
        ballots = parse_domain_raw_votes(row.get("domain_raw_votes"))
        if len(ballots) < 2:
            continue
        votes_per_comment.append(len(ballots))
        for a, b in _vote_pairs(ballots):
            domain_pair_counts += 1
            for category in CATEGORIES:
                per_category_pairs[category].append((int(category in a), int(category in b)))

    if domain_pair_counts:
        print(f"Domain -- {domain_pair_counts} vote-pair(s) across {len(votes_per_comment)} "
              f"comment(s) with >=2 votes recorded:")
        print(f"{'domain':<14}{'self-kappa':>12}")
        for category in CATEGORIES:
            print(f"{category:<14}{kappa(per_category_pairs[category]):>12.2f}")
    else:
        print("Domain: no rows with >=2 recorded votes (domain_raw_votes) -- re-run "
              "slens-classify with --votes >= 2 to measure this.", file=sys.stderr)

    # --- sentiment: pairwise weighted kappa (numeric 1-5 votes only) plus
    # mean within-comment SD, the more intuitive "how spread out" figure the
    # user asked for directly. A comment whose votes mix a real 1-5 rating
    # with a -1/0 categorical one is a different kind of instability (the
    # votes disagree on WHAT KIND of answer this is, not just how positive) --
    # counted and reported separately rather than silently averaged into the
    # SD of whichever votes happen to be numeric.
    sentiment_pairs = []
    stance_sds = []
    kind_inconsistent = 0
    sentiment_rows_with_votes = 0
    for row in rows:
        votes = parse_sentiment_raw_votes(row.get("sentiment_raw_votes"))
        if len(votes) < 2:
            continue
        sentiment_rows_with_votes += 1
        stance_votes = [v for v in votes if v >= 1]
        if stance_votes and len(stance_votes) != len(votes):
            kind_inconsistent += 1
        if len(stance_votes) >= 2:
            stance_sds.append(statistics.pstdev(stance_votes))
            sentiment_pairs.extend(_vote_pairs(stance_votes))

    if sentiment_pairs:
        print(f"\nSentiment -- {len(sentiment_pairs)} numeric vote-pair(s) across "
              f"{sentiment_rows_with_votes} comment(s) with >=2 votes recorded:")
        print(f"  self-kappa (weighted): {kappa_weighted(sentiment_pairs):.2f}")
        print(f"  mean within-comment SD: {statistics.mean(stance_sds):.2f} "
              f"(on the -1..5 scale)")
        if kind_inconsistent:
            print(f"  {kind_inconsistent} comment(s) had votes split between a real 1-5 "
                  f"rating and a -1/0 categorical answer (not codable/irrelevant) -- a "
                  f"disagreement about what KIND of answer this is, excluded from the SD "
                  f"above and worth its own look rather than averaged away.", file=sys.stderr)
    else:
        print("\nSentiment: no rows with >=2 numeric votes (sentiment_raw_votes) -- re-run "
              "slens-classify with --votes >= 2 to measure this.", file=sys.stderr)

    if compare_json:
        with open(compare_json, encoding="utf-8") as handle:
            human = json.load(handle)
        print("\n--- Side by side against the human coder-vs-coder run in "
              f"{compare_json} ({human.get('n_shared', '?')} shared comment(s)) ---")
        for category in CATEGORIES:
            human_k = ((human.get("domain") or {}).get(category) or {}).get("kappa")
            self_k = kappa(per_category_pairs[category]) if domain_pair_counts else float("nan")
            print(f"  {category:<14} human kappa: {fmt(human_k)}   LLM self-kappa: {fmt(self_k)}")
        human_sk = human.get("sentiment_kappa_weighted")
        self_sk = kappa_weighted(sentiment_pairs) if sentiment_pairs else float("nan")
        print(f"  {'sentiment':<14} human weighted kappa: {fmt(human_sk)}   "
              f"LLM self weighted kappa: {fmt(self_sk)}")
        print("\nA lower LLM self-kappa than the human figure means the model disagrees with "
              "itself more than two independent people agree with each other on the same "
              "comments -- worth treating as a real reliability problem, not noise, since it "
              "means labels from a single run would not reproduce under a second run either.")


def fmt(value):
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def allocate(sizes, n):
    """Split `n` places across strata as evenly as their sizes allow.

    Two failures this replaces, both silent:

    * Integer division. `n // len(strata)` asking for 200 across 3 strata gave
      66 each and returned 198, two short of what was asked for, with no
      indication anything had been rounded away.
    * No backfill. A stratum smaller than its share was capped at its own size
      and the unused places were simply lost, so one thin story could leave the
      sample well under `n` — the case raised in #19.

    Places freed by a small stratum are redistributed to strata that still have
    capacity, repeatedly, until `n` is reached or every comment is spoken for.
    Returns {key: count}, whose total is min(n, total available).

    `sizes` is {key: available}, and the result is deterministic given it: keys
    are processed in sorted order, so the same corpus and the same `n` always
    allocate the same way.
    """
    allocation = {key: 0 for key in sizes}
    remaining = min(n, sum(sizes.values()))
    while remaining > 0:
        hungry = [k for k in sorted(sizes) if allocation[k] < sizes[k]]
        if not hungry:
            break
        share = max(1, remaining // len(hungry))
        for key in hungry:
            if remaining <= 0:
                break
            take = min(share, sizes[key] - allocation[key], remaining)
            allocation[key] += take
            remaining -= take
    return allocation


def sample_with_video_cap(pool, count, cap_fraction, video_column="source"):
    """Draw `count` rows from `pool`, capping how many can share one video.

    A soft cap, not a hard one: `cap = max(1, ceil(count * cap_fraction))`
    limits how many rows any single video is *allowed into contention for*,
    but if the capped pool can't supply `count` on its own, the shortfall is
    backfilled from the remainder rather than returning fewer rows than asked
    for -- the same "never silently fall short" principle `allocate()` uses
    for strata sizes, applied one level down to video concentration inside a
    single stratum.

    Motivated by the video-concentration diagnostic run on this corpus
    (2026-09-09): one story's comments were 30% one video and 50% just three
    videos, which risks a "representative" gold sample that is mostly one
    video's comment section -- the same correlation the whole video-level
    bootstrap (report.py) exists to correct for. A feasibility scan across
    every story x category stratum found a 10% cap already achievable
    everywhere (the tightest case, propublica/environment, needed only 19
    distinct videos to clear it); 15% is used here for margin above that
    floor, not because tighter was infeasible.

    Each returned row gets a `_video_cap_prob` key: its TRUE marginal
    selection probability under this design, which `cmd_sample` reads
    instead of the flat stratum-wide `allocation/size` figure. Fixed
    2026-09-16 -- that flat figure is only correct when every video's rows
    have an equal chance of selection, which the cap deliberately breaks: a
    row from a video with more than `cap` rows has LOWER true probability
    (it competes for one of only `cap` slots first), while a row from a
    smaller video has HIGHER true probability (all its rows enter
    contention). On the real gold draw (`--video-cap-fraction 0.15`, quota
    20), a row from a video supplying 100 of a 200-row stratum could have a
    true weight around 33x a row from a small video in the same stratum,
    while the old code recorded the identical flat probability for both --
    silently defeating the Horvitz-Thompson correction `_gold_tpr_fpr`,
    `gold_weight` and `per_category_scores` all depend on, in exactly the
    cap-bound strata this function exists to handle.
    """
    if not pool or count <= 0:
        return [], 0
    cap = max(1, math.ceil(count * cap_fraction))
    by_video = collections.defaultdict(list)
    for row in pool:
        by_video[row.get(video_column, "")].append(row)
    video_sizes = {vid: len(group) for vid, group in by_video.items()}
    for group in by_video.values():
        random.shuffle(group)

    capped_pool = []
    for group in by_video.values():
        capped_pool.extend(group[:cap])

    if len(capped_pool) >= count:
        chosen = random.sample(capped_pool, count)
        for row in chosen:
            vsize = video_sizes[row.get(video_column, "")]
            # Every row of an at-or-under-cap video enters capped_pool with
            # certainty; an over-cap video's rows compete for `cap` of its
            # `vsize` slots first.
            p_in_capped_pool = 1.0 if vsize <= cap else cap / vsize
            row["_video_cap_prob"] = p_in_capped_pool * (count / len(capped_pool))
        return chosen, 0

    # Cap too tight for this stratum to reach its quota -- backfill from
    # whatever the cap left on the table, rather than shortchanging the
    # stratum's allocation to protect video diversity it cannot afford.
    chosen = list(capped_pool)
    leftover = []
    for group in by_video.values():
        leftover.extend(group[cap:])
    random.shuffle(leftover)
    still_needed = count - len(chosen)
    backfilled = leftover[:still_needed]
    chosen.extend(backfilled)

    for row in chosen:
        vsize = video_sizes[row.get(video_column, "")]
        if vsize <= cap:
            # Every row of an at-or-under-cap video is in capped_pool, and
            # this branch keeps the whole of capped_pool -- certain to be
            # chosen.
            row["_video_cap_prob"] = 1.0
        else:
            # An over-cap video's row is either one of the `cap` that made
            # capped_pool (certain to be chosen, in this branch) or one of
            # the `vsize - cap` that went to leftover (chosen only if the
            # backfill draw happens to pick it).
            p_in_capped_pool = cap / vsize
            p_in_leftover = (vsize - cap) / vsize
            p_drawn_from_leftover = still_needed / len(leftover) if leftover else 0.0
            row["_video_cap_prob"] = p_in_capped_pool + p_in_leftover * p_drawn_from_leftover
    return chosen, still_needed


def video_cap_probabilities(video_sizes, count, cap_fraction):
    """Marginal selection probability, per video, under `sample_with_video_cap`.

    Pure arithmetic on the pool's shape -- {video: rows in it}, how many rows
    are drawn, the cap fraction -- with no randomness, so a probability can be
    re-derived for any frame (e.g. the final frozen corpus) without redrawing
    anything. Returns {video: probability for one row of that video}. Mirrors
    `sample_with_video_cap`'s two branches exactly (the harness checks the two
    agree on a toy pool); the draw itself is unchanged.
    """
    if not video_sizes or count <= 0:
        return {vid: 0.0 for vid in video_sizes}
    cap = max(1, math.ceil(count * cap_fraction))
    capped_total = sum(min(size, cap) for size in video_sizes.values())
    probs = {}
    if capped_total >= count:
        for vid, size in video_sizes.items():
            p_in_capped = 1.0 if size <= cap else cap / size
            probs[vid] = p_in_capped * (count / capped_total)
        return probs
    leftover_total = sum(max(0, size - cap) for size in video_sizes.values())
    still_needed = count - capped_total
    p_from_leftover = still_needed / leftover_total if leftover_total else 0.0
    for vid, size in video_sizes.items():
        if size <= cap:
            probs[vid] = 1.0
        else:
            probs[vid] = cap / size + ((size - cap) / size) * p_from_leftover
    return probs


UNCLASSIFIED_STRATUM = "unclassified"


def recompute_inclusion_probs(rows, corpus_rows, phase1_counts, topup_counts=None,
                              cap_fraction=None, topup_ids=(), replacement_ids=()):
    """Re-derive every gold row's inclusion probability against the FINAL corpus.

    The probabilities written when the gold set was drawn describe the frame
    that existed then. The corpus was re-frozen afterwards, and the top-up was
    drawn later still, so a coder file's (or the sampler's) figure no longer
    describes how likely a comment was to end up in the gold set. This
    recomputes them, from the corpus the estimates are actually made on.

    Design assumed (the one the project ran):
      * Phase 1 -- per story x primary_category stratum, `phase1_counts[stratum]`
        comments drawn at random, with a soft per-video cap if `cap_fraction`.
      * Phase 2 (top-up) -- `topup_counts[story]` comments drawn at random,
        no cap, from that story's `unclassified` stratum, excluding phase 1's
        picks.

    A comment's probability of being in the gold set at all is then
        p1                                  (strata other than unclassified)
        p1 + (1 - p1) * m / (N - n1)        (unclassified, with a top-up of m)
    where N is the stratum's size in the final corpus and n1 its phase-1
    allocation. The second line is exact for a simple random draw from the
    remainder; it applies to every unclassified comment, including the ones
    that came from phase 1, because it is a property of the comment, not of
    which phase happened to pick it.

    `rows` need `comment_id` and `sampling_stratum`; top-up rows have only the
    story there, and are assigned `<story>/unclassified`. `corpus_rows` need
    `comment_id`, `story`, `primary_category` and `source` (the video).

    Returns (kept_rows, dropped_ids): kept rows carry the new `inclusion_prob`,
    the figure it replaced as `inclusion_prob_drawn`, and `inclusion_prob_basis`.
    A comment absent from the final corpus has no probability under it and is
    dropped, never given a guessed one; so is one whose computed probability is 0.
    """
    corpus_by_id = {}
    video_sizes = collections.defaultdict(lambda: collections.defaultdict(int))
    stratum_sizes = collections.defaultdict(int)
    for crow in corpus_rows:
        stratum = f"{crow.get('story', '')}/{crow.get('primary_category', '')}"
        corpus_by_id[crow.get("comment_id", "")] = (stratum, crow.get("source", ""))
        video_sizes[stratum][crow.get("source", "")] += 1
        stratum_sizes[stratum] += 1

    topup_counts = topup_counts or {}
    topup_ids, replacement_ids = set(topup_ids), set(replacement_ids)
    phase1_probs = {}
    for stratum, count in phase1_counts.items():
        if cap_fraction:
            phase1_probs[stratum] = video_cap_probabilities(
                dict(video_sizes.get(stratum, {})), count, cap_fraction)
        else:
            size = stratum_sizes.get(stratum, 0)
            phase1_probs[stratum] = {vid: (count / size if size else 0.0)
                                     for vid in video_sizes.get(stratum, {})}

    kept, dropped = [], []
    for row in rows:
        cid = row["comment_id"]
        story = row.get("story", "")
        if cid in topup_ids:
            stratum = f"{story}/{UNCLASSIFIED_STRATUM}"
        else:
            stratum = row.get("sampling_stratum", "")
        found = corpus_by_id.get(cid)
        if found is None or stratum not in phase1_probs:
            dropped.append(cid)
            continue
        _, video = found
        p1 = phase1_probs[stratum].get(video, 0.0)
        basis = "recomputed_final_corpus"
        story_of_stratum = stratum.split("/", 1)[0]
        m = topup_counts.get(story_of_stratum, 0)
        if stratum.endswith("/" + UNCLASSIFIED_STRATUM) and m:
            remainder = stratum_sizes.get(stratum, 0) - phase1_counts[stratum]
            p2 = m / remainder if remainder > 0 else 1.0
            prob = p1 + (1.0 - p1) * p2
            basis = "recomputed_two_phase"
        else:
            prob = p1
        if prob <= 0:
            # p1 is 0 when the comment's video has no rows in the drawn stratum of the
            # final corpus (its keyword category changed at the re-freeze, or a replacement
            # came from another stratum). Under this design it could not have been drawn,
            # so there is no weight to give it: exclude it, never write 0.000000, which
            # every weighting consumer would once have turned into weight 1.
            dropped.append(cid)
            continue
        if cid in replacement_ids:
            basis += "+replacement_approx"
        out = dict(row)
        out["inclusion_prob_drawn"] = row.get("inclusion_prob", "")
        out["inclusion_prob"] = f"{min(prob, 1.0):.6f}"
        out["inclusion_prob_basis"] = basis
        kept.append(out)
    return kept, dropped


def cmd_sample(rows, n, seed, out, stratify_by=None, video_cap_fraction=None):
    """Draw a stratified sample for hand coding.

    Stratifies by story and platform by default, so no story dominates the way
    the UnitedHealthcare material dominates the corpus.

    `--stratify-by <col1>[,<col2>...]` stratifies on one or more existing
    columns instead. A single column, e.g. `kw_domain` or `domain` from a
    previous classifier run, is the remedy for the rare-category problem in
    #18: drawn at random, a category holding 2% of the corpus lands 4
    comments in a 200-comment sample, and precision and recall computed on 4
    instances mean nothing. Stratifying on a machine label deliberately
    over-samples the rare categories so they can actually be validated.

    Two columns, e.g. `--stratify-by story,kw_domain`, balances both at once:
    categories roughly evenly represented, AND no single story dominating
    within a category. That second part matters on its own -- a category
    balanced by count but drawn entirely from one dominant story reproduces
    the exact keyword-audit lift distortion driven by story composition.
    Order columns from least to most granular (story before a many-valued
    label) so a thin combination degrades to "not enough of this story for
    this category" rather than silently vanishing into "unlabelled".

    Either way the machine label decides only who gets *read* -- the coder
    still sees a blind file -- so a wrong machine label costs sampling
    efficiency, never correctness.
    """
    random.seed(seed)
    columns = [c.strip() for c in stratify_by.split(",") if c.strip()] if stratify_by else None
    strata = collections.defaultdict(list)
    for row in rows:
        if columns:
            key = tuple((row.get(col) or "unlabelled").strip() or "unlabelled" for col in columns)
        else:
            key = (row.get("story", ""), row.get("platform", ""))
        strata[key].append(row)

    columns = [c.strip() for c in (stratify_by or "").split(",") if c.strip()]
    missing = [c for c in columns if rows and c not in rows[0]]
    if missing:
        # Silently bucketing every row into one stratum is not "no
        # stratification", it is a stratified design that quietly stopped
        # stratifying -- and the inclusion probabilities written below would
        # then describe a design that was never run.
        sys.exit(f"No column(s) {', '.join(repr(c) for c in missing)} in the input. Available: "
                 f"{', '.join(sorted(k for k in rows[0] if k and not k.startswith('_')))}")

    sizes = {k: len(v) for k, v in strata.items()}
    allocation = allocate(sizes, n)
    picked = []
    shortfalls = {}
    for key in sorted(strata, key=str):
        # The inclusion probability for this comment: how likely it was to be
        # drawn, given its stratum. Recorded on every row because correcting a
        # corpus-wide proportion for classifier error (design-based supervised
        # learning, or prediction-powered inference) requires the gold-label
        # sampling probability to be KNOWN and non-zero. Stratifying makes the
        # probabilities unequal, so they cannot be reconstructed afterwards --
        # and once coding has started the design cannot be changed. Without
        # this column the correction is simply unavailable, and raw classifier
        # proportions are biased with intervals that badly undercover even at
        # 80-90% accuracy.
        if video_cap_fraction:
            chosen, short = sample_with_video_cap(strata[key], allocation[key], video_cap_fraction)
            if short:
                shortfalls[key] = short
            # sample_with_video_cap already computed each row's true,
            # cap-aware probability -- see its docstring for why the flat
            # allocation/size figure below is wrong whenever the cap binds.
            for row in chosen:
                row["_inclusion_prob"] = row.pop("_video_cap_prob")
        else:
            chosen = random.sample(strata[key], allocation[key])
            prob = allocation[key] / sizes[key] if sizes[key] else 0.0
            for row in chosen:
                row["_inclusion_prob"] = prob
        for row in chosen:
            row["_stratum"] = key if isinstance(key, str) else "/".join(str(x) for x in key)
        picked += chosen
    random.shuffle(picked)

    # Deliberately blind: the coder sees the comment, not what the machines said.
    # `source` (the video URL) is not a machine label -- it is where the
    # comment came from, not a guess about it -- so carrying it here does not
    # compromise that. It also gives a coder somewhere to click through for
    # context when a comment alone is too short or ambiguous to code (a bare
    # "this" or "lol" makes no sense without the video it is replying under).
    #
    # It used to reach the coded file only incidentally, via `join` carrying
    # it back from the labelled corpus after hand-coding -- undocumented, and
    # a library function reachable without going through `join` first would
    # not have it. `corrected_ci_bootstrap` clusters the gold set by this same
    # column, so it needs to survive from here, not arrive as an afterthought.
    writer = csv.DictWriter(out, fieldnames=["comment_id", "story", "platform", "source",
                                             "body", "hand_domain", "hand_sentiment",
                                             "coder_notes", "sampling_stratum",
                                             "inclusion_prob"])
    writer.writeheader()
    for row in picked:
        writer.writerow({
            "comment_id": row.get("comment_id", ""),
            "story": row.get("story", ""),
            "platform": row.get("platform", ""),
            "source": row.get("source", ""),
            "body": row.get("body", ""),
            "hand_domain": "",
            "hand_sentiment": "",
            "coder_notes": "",
            "sampling_stratum": row.get("_stratum", ""),
            "inclusion_prob": f"{row.get('_inclusion_prob', 0):.6f}",
        })

    print(f"Wrote {len(picked)} comments from {len(strata)} strata for hand coding.",
          file=sys.stderr)
    for key in sorted(strata, key=str):
        label = key if isinstance(key, str) else "/".join(x for x in key if x)
        short = " (all of them)" if allocation[key] == len(strata[key]) else ""
        print(f"  {label or '(blank)'}: {allocation[key]} of {len(strata[key])}{short}",
              file=sys.stderr)
    # Falling short must be visible: it means the corpus could not supply the
    # sample that was asked for, which is itself worth knowing.
    if len(picked) < n:
        print(f"NOTE: asked for {n} but only {len(picked)} comments exist across these "
              f"strata. The sample is the whole of what is available, not a draw from it.",
              file=sys.stderr)
    if video_cap_fraction:
        print(f"\nVideo soft cap: no more than {video_cap_fraction:.0%} of a stratum's "
              f"allocation drawn from one video, where the corpus allows it.", file=sys.stderr)
        if shortfalls:
            for key in sorted(shortfalls, key=str):
                label = key if isinstance(key, str) else "/".join(x for x in key if x)
                print(f"  {label}: cap too tight to fill this stratum alone -- "
                      f"{shortfalls[key]} comment(s) backfilled past the cap.", file=sys.stderr)
        else:
            print("  Every stratum filled its allocation within the cap -- no backfill needed.",
                  file=sys.stderr)
    print("Have at least two people code the same file independently, then compare "
          "them to each other before comparing either to the machine.", file=sys.stderr)
    print("\n`inclusion_prob` records how likely each comment was to be drawn. Keep the column: "
          "correcting corpus proportions for classifier error needs a known sampling "
          "probability, and once coding has started the design cannot be changed.",
          file=sys.stderr)

    # Kish's effective sample size: how many EQUALLY-weighted comments this
    # unequally-weighted draw is worth, for the Horvitz-Thompson weighting
    # (1 / inclusion_prob) report._gold_tpr_fpr and validate's own gold_weight
    # apply downstream. A stratified draw over a skewed corpus can produce an
    # enormous spread in inclusion_prob -- measured on the real 400-row gold
    # set, 0.000225 to 0.667, a ~3,000x range -- which does not make the
    # weighting wrong, but does mean a handful of the lowest-probability rows
    # can end up doing most of the work in a corrected estimate or a
    # per-story accuracy ratio. This is visible here, at draw time, rather
    # than only discoverable by inspecting the column by hand afterwards.
    weights = [1.0 / prob if prob > 0 else 1.0
               for prob in (row.get("_inclusion_prob", 0) for row in picked)]
    sum_w, sum_w2 = sum(weights), sum(w * w for w in weights)
    kish_ess = (sum_w * sum_w) / sum_w2 if sum_w2 else float("nan")
    print(f"\nKish effective sample size: {kish_ess:.0f} of {len(picked)} rows nominal "
          f"({kish_ess / len(picked):.0%}) -- how many equally-weighted comments this draw's "
          f"Horvitz-Thompson weighting is actually worth. Well below 100% means a few "
          f"low-probability rows carry disproportionate weight in anything computed from "
          f"inclusion_prob downstream (corrected_ci_bootstrap, story_accuracy_ratios) -- worth "
          f"a look at which strata those rows come from before trusting a correction built on "
          f"them.", file=sys.stderr)


MACHINE_COLUMNS_HINT = (
    "The file has hand_domain but no domain_* columns, so every machine label "
    "reads as absent and the score is 0.00 by construction rather than by "
    "measurement. `slens-validate sample` writes only the nine columns a coder "
    "needs, on purpose -- showing them the machine's answer would bias the "
    "coding. Join the two back together first:\n\n"
    "    slens-validate join coded.csv --labelled labelled.csv --output scored.csv\n\n"
    "(A file whose domain_* columns are present but blank on every row is treated the "
    "same way: that is what a join that matched nothing produces. If the gold comments "
    "were left out of the labelled corpus, label the gold file itself instead.)\n"
)


def machine_prefix(rows):
    """Which set of machine labels this file carries: "domain_", "kw_", or None.

    Checked explicitly because its absence is silent and catastrophic: a coder
    who agrees with the classifier on every single row still scored recall
    0.00 across all four categories, exit 0, no warning.

    Both prefixes are accepted so a keyword-only run can be scored. That is
    the offline rehearsal path -- no API key, no spend -- and refusing it
    would mean the only way to answer "how good is our labelling" is to pay
    for a full LLM pass first. `domain_` wins when both are present, since
    that is the arm being reported.
    """
    if not rows:
        return None
    for prefix in ("domain_", "kw_"):
        if any(f"{prefix}{c}" in rows[0] for c in CATEGORIES) and has_machine_labels(rows, prefix):
            return prefix
    return None


def has_machine_labels(rows, prefix="domain_"):
    """True when at least one row has a machine label under `prefix`, not just the column.

    `join` writes the domain_* header even when no coded comment matched the labelled
    file, so a header check passes on a file whose machine columns are all blank, and
    every comparison then reads "not tagged": recall 0.00 everywhere, exit 0.
    """
    return any((row.get(f"{prefix}{c}") or "").strip() for row in rows for c in CATEGORIES)


def machine_unlabelled(row, prefix="domain_"):
    """True when a row carries no machine label at all under `prefix`.

    All four category cells blank, as `join` writes for a coded comment that was not in the
    labelled file. A classifier that ran writes an explicit 0 or 1 in every cell, so a blank
    row was never labelled: it is not a row the classifier "missed".
    """
    return not any((row.get(f"{prefix}{c}") or "").strip() for c in CATEGORIES)


def drop_unlabelled(rows, prefix="domain_", what="gold rows"):
    """Rows that carry machine labels, saying how many without any were left out.

    Scoring reads a blank machine cell as "not tagged", so a gold row `join` could not match
    would count as a classifier miss in every category. Left out instead, and counted.
    """
    kept = [r for r in rows if not machine_unlabelled(r, prefix)]
    if len(kept) != len(rows):
        print(f"Left out {len(rows) - len(kept)} of {len(rows)} {what} with no machine labels "
              f"({prefix}* blank on every category): they were never labelled, so scoring them "
              f"would count every one as a classifier miss.", file=sys.stderr)
    return kept


def drop_blank_hand(rows, column="hand_domain", what="gold rows"):
    """Rows a human has coded: a blank `column` is "not coded yet", never "none"."""
    kept = [r for r in rows if (r.get(column) or "").strip()]
    if len(kept) != len(rows):
        print(f"Left out {len(rows) - len(kept)} of {len(rows)} {what} with a blank {column} "
              f"(not coded, which is not the same as \"none\").", file=sys.stderr)
    return kept


def partial_join_error(coded_rows, labelled_rows):
    """The message for a join that leaves some CODED comments without machine labels, else None.

    `unmatched_join_error` only catches a join that matches nothing. One that matches most
    of them still writes the rest with blank domain_* columns, which then score as misses.
    A row nobody has coded yet (no hand_domain or hand_sentiment) does not count: it cannot
    be scored either way.
    """
    labelled_ids = {r.get("comment_id") for r in labelled_rows if r.get("comment_id")}
    missing = [r for r in coded_rows if r.get("comment_id") not in labelled_ids
               and ((r.get("hand_domain") or "").strip()
                    or (r.get("hand_sentiment") or "").strip())]
    if not missing:
        return None
    ids = [r.get("comment_id") or "(no id)" for r in missing]
    return (f"Refusing to join: {len(missing)} of {len(coded_rows)} coded comments are not in "
            f"the labelled file ({', '.join(ids[:5])}{' ...' if len(ids) > 5 else ''}). They "
            f"would be written with blank machine labels and then score as classifier misses "
            f"in every category.\n\n"
            f"Check you are joining against the corpus the sample was drawn from, or label the "
            f"gold file itself with the same instrument. If a partial join is what you want "
            f"(the unmatched rows are then left out of scoring, and counted), pass "
            f"--allow-partial.")


def unmatched_join_error(coded_rows, labelled_rows):
    """The message for a join in which no coded comment is in the labelled file, else None."""
    labelled_ids = {r.get("comment_id") for r in labelled_rows if r.get("comment_id")}
    if not coded_rows or any(r.get("comment_id") in labelled_ids for r in coded_rows):
        return None
    return (f"Refusing to join: none of the {len(coded_rows)} coded comments is in the labelled "
            f"file, so there are no machine labels to put beside the hand codes. Scoring the "
            f"result would report recall 0.00 for every category.\n\n"
            f"If the gold comments were left out of the corpus that was labelled (the Gemma "
            f"batches are: `slens-batches prepare --exclude` removes them), label the gold file "
            f"itself with the same instrument instead, e.g.\n\n"
            f"    slens-gemma GOLD.csv --output gold_labelled.csv --cache <the run's cache> "
            f"--fixed-examples <the examples file>\n\n"
            f"Its output keeps the hand codes beside the machine labels and goes straight to "
            f"`slens-validate score`, `slens-report --gold` and `slens-export-dashboard --scored`.")


def sentiment_bucket(value):
    """negative / neutral / positive for a 1-5 rating or the word; 0 and -1 and
    anything else stay as they are (irrelevant and uncodeable are not on the scale)."""
    text = str(value).strip().lower()
    try:
        number = float(text)
    except ValueError:
        return text
    if 1 <= number <= 2:
        return "negative"
    if number == 3:
        return "neutral"
    if 4 <= number <= 5:
        return "positive"
    return text


def cmd_combine(coder_rows_by_path, manifest_rows, master_rows, replacement_rows, handle,
                corpus_rows=None, topup_rows=None, cap_fraction=None, adjudicated_rows=None,
                sentiment_agreement="exact"):
    """Recombine every coder's completed file into one gold-standard file.

    Three things this exists to get right, none of which "just concatenate
    the coder files" does on its own:

    1. `inclusion_prob` always comes from `master_rows` (the drawn sample,
       e.g. gold_sample_balanced.csv), never from a coder's own copy of it.
       A coder's file is a snapshot taken at split time; if the sampling
       design or the corpus changes afterwards (as it did here -- the video
       cap's inclusion_prob formula was fixed after these files were
       already out), the coder's copy goes stale but coding continues
       regardless, since a coder never reads or acts on that column. Ignore
       it in every source file and re-derive it fresh from the master
       sample every time this runs, so a stale coder-file value can never
       silently survive into the combined file `score`/`corrected_ci_bootstrap`
       read.
    2. A "core" comment (coded by more than one coder, for the interrater
       check) is combined only where every coder who coded it agrees on
       both `hand_domain` and `hand_sentiment`. A disagreement is not
       silently averaged, voted on, or resolved by picking one coder over
       another -- it is adjudication, which is a human decision (see
       CLAUDE.md stage 4), so this reports it and leaves the row for a
       person to resolve rather than guessing.
    3. `replacement_rows` (comments hand-picked and coded to replace a row
       drawn from a video later found to need excluding) swap in for the
       specific row they replace, inheriting that row's `inclusion_prob`
       and `sampling_stratum` -- they were selected to serve the same
       stratum cell, not drawn by the stratified sampler themselves, so
       they have no `inclusion_prob` of their own to contribute. Matched by
       an explicit `replaces_comment_id` column on the replacement file,
       never inferred, so a mismatch fails loudly instead of silently
       replacing the wrong row.

    Optional, for a gold set drawn in more than one phase against a corpus that
    was re-frozen afterwards:

    * `adjudicated_rows` -- a person's resolution of rows the coders disagreed on
      (`comment_id`, `hand_domain`, `hand_sentiment`). Each overrides whatever
      the coders wrote for that comment, is marked `adjudicated` in
      `coder_notes`, and takes its probability like any other row.
    * `topup_rows` -- a later random draw from each story's `unclassified`
      stratum, coded by whoever coded it (no split-manifest entry needed).
    * `corpus_rows` (+ `cap_fraction`, the soft video cap the first draw used)
      -- the FINAL corpus. When given, every probability is re-derived from it
      by `recompute_inclusion_probs` instead of being copied from the sample
      files (which describe the frame that existed when each was drawn). A
      replacement row then takes its stratum's probability for its own video,
      not the replaced row's, and is flagged in `inclusion_prob_basis`.
    """
    master_by_id = {r["comment_id"]: r for r in master_rows if r.get("comment_id")}
    topup_ids = set()
    if topup_rows:
        manifest_rows = list(manifest_rows)
        for trow in topup_rows:
            cid = (trow.get("comment_id") or "").strip()
            if not cid or cid in master_by_id:
                continue
            master_by_id[cid] = dict(trow, sampling_stratum=f"{trow.get('story', '')}/"
                                                            f"{UNCLASSIFIED_STRATUM}")
            manifest_rows.append({"comment_id": cid, "assigned_to": "topup", "is_core": "0"})
            topup_ids.add(cid)

    coders_by_id = collections.defaultdict(dict)
    for path, rows in coder_rows_by_path.items():
        for row in rows:
            cid = (row.get("comment_id") or "").strip()
            if not cid:
                continue
            coders_by_id[cid][path] = row

    manifest_by_id = collections.defaultdict(list)
    for row in manifest_rows:
        manifest_by_id[row["comment_id"]].append(row)

    out_rows = []
    needs_adjudication = []
    not_yet_coded = []
    missing_from_coder_files = []

    for cid, manifest_entries in manifest_by_id.items():
        master_row = master_by_id.get(cid)
        if master_row is None:
            continue  # not in the current master sample -- a replaced row, handled below
        # Gather every coder's answer for this comment_id (one per manifest
        # entry -- one entry per coder for a core row, exactly one entry
        # for a non-core row).
        seen = []
        for path, row in coders_by_id.get(cid, {}).items():
            hd = (row.get("hand_domain") or "").strip()
            hs = (row.get("hand_sentiment") or "").strip()
            seen.append((path, hd, hs, row.get("coder_notes", "")))

        if not seen:
            missing_from_coder_files.append(cid)
            continue
        if any(not hd or not hs for _, hd, hs, _ in seen):
            not_yet_coded.append(cid)
            continue

        # Compare hand_domain as a SET of categories, not the raw string --
        # "democracy, economics" and "economics, democracy" are the same
        # judgement in different word order, and treating them as distinct
        # sent every such row to adjudication that didn't actually need it.
        # `labels_from_text` is the same substring-membership parse every
        # other consumer of hand_domain already uses (kappa, score, agree),
        # so this makes combine's notion of "agree" consistent with theirs.
        if sentiment_agreement == "bucket":
            distinct = {(frozenset(labels_from_text(hd)), sentiment_bucket(hs))
                        for _, hd, hs, _ in seen}
        else:
            distinct = {(frozenset(labels_from_text(hd)), hs) for _, hd, hs, _ in seen}
        if len(distinct) > 1:
            needs_adjudication.append((cid, seen))
            continue

        _, hand_domain, hand_sentiment, _ = seen[0]
        if sentiment_agreement == "bucket" and len({hs for _, _, hs, _ in seen}) > 1:
            # Same bucket, different numbers (say 1 and 2): one value has to be
            # written, and picking either coder's would be arbitrary, so the
            # bucket's word is used.
            hand_sentiment = sentiment_bucket(hand_sentiment)
        notes = " | ".join(n for _, _, _, n in seen if n)
        out_rows.append({
            "comment_id": cid,
            "story": master_row.get("story", ""),
            "source": master_row.get("source", ""),
            "body": master_row.get("body", ""),
            "hand_domain": hand_domain,
            "hand_sentiment": hand_sentiment,
            "coder_notes": notes,
            "sampling_stratum": master_row.get("sampling_stratum", ""),
            "inclusion_prob": master_row.get("inclusion_prob", ""),
        })

    for arow in adjudicated_rows or []:
        cid = (arow.get("comment_id") or "").strip()
        master_row = master_by_id.get(cid)
        if master_row is None:
            sys.exit(f"Adjudicated row {cid!r} is not in the master or top-up sample -- "
                     f"refusing to add a row nobody drew.")
        hand_domain = (arow.get("hand_domain") or "").strip()
        hand_sentiment = (arow.get("hand_sentiment") or "").strip()
        if not hand_domain or not hand_sentiment:
            sys.exit(f"Adjudicated row {cid!r} has a blank hand_domain or hand_sentiment -- an "
                     f"adjudication has to give both.")
        out_rows = [r for r in out_rows if r["comment_id"] != cid]
        needs_adjudication = [item for item in needs_adjudication if item[0] != cid]
        not_yet_coded = [c for c in not_yet_coded if c != cid]
        out_rows.append({
            "comment_id": cid,
            "story": master_row.get("story", ""),
            "source": master_row.get("source", ""),
            "body": master_row.get("body", ""),
            "hand_domain": hand_domain,
            "hand_sentiment": hand_sentiment,
            "coder_notes": "adjudicated",
            "sampling_stratum": master_row.get("sampling_stratum", ""),
            "inclusion_prob": master_row.get("inclusion_prob", ""),
        })

    replaced_ids = set()
    for row in replacement_rows:
        replaces = (row.get("replaces_comment_id") or "").strip()
        if not replaces:
            sys.exit(f"Replacement row {row.get('comment_id')!r} has no "
                     f"replaces_comment_id -- refusing to guess which row it stands in for.")
        original = master_by_id.get(replaces)
        if original is None:
            sys.exit(f"Replacement row {row.get('comment_id')!r} claims to replace "
                     f"{replaces!r}, which is not in the master sample.")
        hand_domain = (row.get("hand_domain") or "").strip()
        hand_sentiment = (row.get("hand_sentiment") or "").strip()
        if not hand_domain or not hand_sentiment:
            not_yet_coded.append(row.get("comment_id"))
            continue
        out_rows.append({
            "comment_id": row.get("comment_id", ""),
            "story": row.get("story") or original.get("story", ""),
            "source": row.get("source") or "",
            "body": row.get("body", ""),
            "hand_domain": hand_domain,
            "hand_sentiment": hand_sentiment,
            "coder_notes": row.get("coder_notes", ""),
            # `or`, not a get-default: a replacement file that HAS the column but
            # leaves the cell blank returns "" from row.get(), which would skip the
            # fallback and put the row in no stratum at all.
            "sampling_stratum": row.get("sampling_stratum") or original.get("sampling_stratum", ""),
            # Inherited, not recomputed: this row was hand-picked to stand in
            # for `original` inside the same stratum cell, not drawn by the
            # stratified sampler, so it has no inclusion probability of its
            # own -- it takes on the row it is replacing's.
            "inclusion_prob": original.get("inclusion_prob", ""),
        })
        replaced_ids.add(replaces)

    out_rows = [r for r in out_rows if r["comment_id"] not in replaced_ids]

    fieldnames = ["comment_id", "story", "source", "body", "hand_domain", "hand_sentiment",
                 "coder_notes", "sampling_stratum", "inclusion_prob"]
    if corpus_rows is not None:
        replacement_ids = {(r.get("comment_id") or "") for r in replacement_rows}
        phase1_counts = collections.Counter(
            r.get("sampling_stratum", "") for r in master_rows if r.get("comment_id"))
        topup_counts = collections.Counter(
            master_by_id[cid].get("story", "") for cid in topup_ids)
        out_rows, dropped = recompute_inclusion_probs(
            out_rows, corpus_rows, dict(phase1_counts), dict(topup_counts),
            cap_fraction=cap_fraction, topup_ids=topup_ids, replacement_ids=replacement_ids)
        fieldnames += ["inclusion_prob_drawn", "inclusion_prob_basis"]
        if dropped:
            print(f"WARNING: {len(dropped)} coded row(s) are not in the final corpus (or sit "
                  f"in a stratum with no phase-1 allocation, or come out at probability 0), "
                  f"so they have no usable inclusion "
                  f"probability under it -- EXCLUDED, not given a guessed one: "
                  f"{', '.join(dropped[:10])}{' ...' if len(dropped) > 10 else ''}",
                  file=sys.stderr)
        weights = [1.0 / float(r["inclusion_prob"]) for r in out_rows
                   if float(r["inclusion_prob"]) > 0]
        if weights:
            kish = sum(weights) ** 2 / sum(w * w for w in weights)
            print(f"Inclusion probabilities recomputed against the final corpus "
                  f"({len(corpus_rows)} rows): {len(out_rows)} gold rows, Kish effective "
                  f"sample size {kish:.0f}, probabilities {min(1 / w for w in weights):.5f}"
                  f"-{max(1 / w for w in weights):.5f}.", file=sys.stderr)
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)

    print(f"Combined {len(out_rows)} rows.", file=sys.stderr)
    if not_yet_coded:
        print(f"NOTE: {len(not_yet_coded)} row(s) not yet fully coded (blank hand_domain "
              f"or hand_sentiment somewhere) -- excluded, not guessed at: "
              f"{', '.join(not_yet_coded[:10])}{' ...' if len(not_yet_coded) > 10 else ''}",
              file=sys.stderr)
    if missing_from_coder_files:
        print(f"WARNING: {len(missing_from_coder_files)} comment_id(s) from the manifest were "
              f"not found in any coder file passed in: "
              f"{', '.join(missing_from_coder_files[:10])}"
              f"{' ...' if len(missing_from_coder_files) > 10 else ''}", file=sys.stderr)
    if needs_adjudication:
        print(f"\nNEEDS ADJUDICATION -- {len(needs_adjudication)} comment_id(s) where coders "
              f"disagree, left OUT of the combined file rather than resolved automatically:",
              file=sys.stderr)
        for cid, seen in needs_adjudication:
            answers = "; ".join(f"{os.path.basename(str(path))}: {hd} / {hs}"
                                for path, hd, hs, _ in seen)
            print(f"  {cid}: {answers}", file=sys.stderr)
    return out_rows, needs_adjudication, not_yet_coded, missing_from_coder_files


def clean_id(value):
    """A comment_id as a spreadsheet may have mangled it: surrounding or embedded spaces,
    non-breaking spaces, zero-width characters and BOMs removed. Case is NOT changed: YouTube
    comment ids are case-sensitive, and lower-casing would match the wrong comment."""
    import re
    return re.sub(r"[\s\u00a0\u200b\u200c\u200d\ufeff]+", "", str(value or ""))


def cmd_reweight(gold_rows, master_rows, topup_rows, corpus_rows, cap_fraction, handle):
    """Recompute inclusion probabilities on a gold file that has ALREADY been assembled.

    `combine` does this while it assembles the coders' files. A gold file put together by
    hand (rows copied across from several sheets) never went through it, so its
    `inclusion_prob` column is whatever each sheet carried: figures from the first draw,
    the top-up, or nothing. This re-derives every row's probability against the FINAL corpus
    with the same arithmetic (`recompute_inclusion_probs`).

    The gold file must hold ONE row per comment. A comment coded by several people (the
    shared core) must already be resolved to a single row, or its weight would be counted
    once per coder.

    Each row is placed by where it came from:
      * its comment_id is in `master_rows` (the original draw): phase 1, stratum from the draw;
      * its comment_id is in `topup_rows`: the top-up (unclassified stratum of its story);
      * neither: a hand-picked replacement. It needs a `sampling_stratum` of its own (the
        stratum of the row it replaced) and is flagged `+replacement_approx`.
    A coded row absent from the final corpus has no probability under it: excluded and listed.
    Returns the rows written.
    """
    master_by_id = {clean_id(r.get("comment_id")): r for r in master_rows
                    if clean_id(r.get("comment_id"))}
    topup_ids = {clean_id(r.get("comment_id")) for r in topup_rows} - {""}
    corpus_rows = [dict(r, comment_id=clean_id(r.get("comment_id"))) for r in corpus_rows]
    ids = [clean_id(r.get("comment_id")) for r in gold_rows]
    if any(not i for i in ids):
        sys.exit(f"{sum(1 for i in ids if not i)} gold row(s) have no comment_id. Fill them in "
                 f"first (scripts_scratch_fill_comment_ids.py).")
    repeated = [c for c, n in collections.Counter(ids).items() if n > 1]
    if repeated:
        sys.exit(f"{len(repeated)} comment_id(s) appear more than once in the gold file "
                 f"(e.g. {', '.join(repeated[:5])}). It must hold one row per comment: resolve "
                 f"the shared core rows to a single row each first, or their weight would be "
                 f"counted once per coder.")

    rows, replacement_ids, unplaced = [], set(), []
    for row in gold_rows:
        cid = clean_id(row.get("comment_id"))
        new = dict(row, comment_id=cid)       # the cleaned id is the one looked up below
        if cid in master_by_id:
            new["sampling_stratum"] = master_by_id[cid].get("sampling_stratum", "")
        elif cid in topup_ids:
            pass          # stratum is assigned from the story inside recompute_inclusion_probs
        else:
            if not (row.get("sampling_stratum") or "").strip():
                unplaced.append(cid)
                continue
            replacement_ids.add(cid)
        rows.append(new)
    if unplaced:
        sys.exit(f"{len(unplaced)} gold row(s) are in neither the original draw nor the top-up and "
                 f"have no sampling_stratum, so they cannot be placed in a stratum (e.g. "
                 f"{', '.join(unplaced[:5])}). A hand-picked replacement needs the stratum of "
                 f"the row it replaced.")

    phase1_counts = collections.Counter(
        r.get("sampling_stratum", "") for r in master_rows if (r.get("comment_id") or "").strip())
    topup_counts = collections.Counter(
        (r.get("story") or "").strip() for r in topup_rows if (r.get("comment_id") or "").strip())
    out_rows, dropped = recompute_inclusion_probs(
        rows, corpus_rows, dict(phase1_counts), dict(topup_counts),
        cap_fraction=cap_fraction or None, topup_ids=topup_ids, replacement_ids=replacement_ids)

    fieldnames = list(gold_rows[0].keys())
    for extra in ("inclusion_prob_drawn", "inclusion_prob_basis"):
        if extra not in fieldnames:
            fieldnames.append(extra)
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(out_rows)

    basis = collections.Counter(r["inclusion_prob_basis"] for r in out_rows)
    weights = [1.0 / float(r["inclusion_prob"]) for r in out_rows if float(r["inclusion_prob"]) > 0]
    kish = (sum(weights) ** 2 / sum(w * w for w in weights)) if weights else 0.0
    probs = [float(r["inclusion_prob"]) for r in out_rows]
    print(f"Recomputed inclusion probabilities for {len(out_rows)} of {len(gold_rows)} gold rows "
          f"against the final corpus ({len(corpus_rows):,} rows).", file=sys.stderr)
    for name, n in sorted(basis.items()):
        print(f"  {n:>5}  {name}", file=sys.stderr)
    if probs:
        print(f"Probabilities {min(probs):.5f}-{max(probs):.5f}; Kish effective sample size "
              f"{kish:.0f} of {len(out_rows)}.", file=sys.stderr)
    if dropped:
        corpus_ids = {r["comment_id"] for r in corpus_rows}
        lower_index = collections.defaultdict(set)
        for cid in corpus_ids:
            lower_index[cid.lower()].add(cid)
        gold_by_id = {clean_id(r.get("comment_id")): r for r in gold_rows}
        not_in_corpus = [c for c in dropped if c not in corpus_ids]
        no_stratum = [c for c in dropped if c in corpus_ids]
        case_only = [c for c in not_in_corpus if c.lower() in lower_index]
        print(f"WARNING: {len(dropped)} gold row(s) were EXCLUDED, not given a guessed "
              f"probability:", file=sys.stderr)
        if not_in_corpus:
            print(f"  {len(not_in_corpus)} are not in the final corpus (the comment, or its "
                  f"whole video, was removed after the gold draw; or the id is wrong): "
                  f"{', '.join(not_in_corpus[:10])}{' ...' if len(not_in_corpus) > 10 else ''}",
                  file=sys.stderr)
            if case_only:
                print(f"  {len(case_only)} of those match a corpus id only if letter case is "
                      f"ignored. YouTube ids are case-sensitive, so they were NOT matched; check "
                      f"whether the sheet changed the capitals: {', '.join(case_only[:5])}",
                      file=sys.stderr)
            by_video = collections.Counter(
                (gold_by_id[c].get("source") or "(no source column)") for c in not_in_corpus)
            if len(by_video) <= 6 or "(no source column)" not in by_video:
                top = ", ".join(f"{v}: {n}" for v, n in by_video.most_common(5))
                print(f"  by video (top): {top}", file=sys.stderr)
            by_story = collections.Counter(
                (gold_by_id[c].get("story") or "?") for c in not_in_corpus)
            print(f"  by story: {dict(by_story)}", file=sys.stderr)
        if no_stratum:
            strata = collections.Counter(
                next((r.get("sampling_stratum") for r in rows
                      if r.get("comment_id") == c), "?") for c in no_stratum)
            print(f"  {len(no_stratum)} are in the corpus but have no probability under the design: "
                  f"their stratum has no allocation in the original draw, or their video has no "
                  f"rows in that stratum of the final corpus (probability 0) (stratum names "
                  f"seen: {dict(strata)}): {', '.join(no_stratum[:10])}",
                  file=sys.stderr)
    return out_rows


def cmd_join(coded_rows, labelled_rows, handle, allow_partial=False):
    """Put the machine labels back beside the hand codes, on comment_id.

    The missing primitive. `sample` deliberately strips the machine columns so
    the coder cannot see them; nothing put them back, so `score` and
    `--scored` both compared hand codes against nothing at all.
    """
    error = unmatched_join_error(coded_rows, labelled_rows)
    if not error and not allow_partial:
        error = partial_join_error(coded_rows, labelled_rows)
    if error:
        sys.exit(error)
    by_id = {r.get("comment_id"): r for r in labelled_rows if r.get("comment_id")}
    carried = [c for c in (labelled_rows[0].keys() if labelled_rows else [])
               if c.startswith(("domain_", "kw_")) or c in ("domain", "domain_judge",
                                                            "sentiment", "source", "story")]
    out_fields = list(coded_rows[0].keys()) + [c for c in carried
                                               if c not in coded_rows[0]]
    writer = csv.DictWriter(handle, fieldnames=out_fields, extrasaction="ignore")
    writer.writeheader()
    matched = missing = 0
    altered = collections.Counter()
    for row in coded_rows:
        source = by_id.get(row.get("comment_id"))
        if source is None:
            missing += 1
            # Written through unjoined rather than dropped: a coded comment
            # vanishing silently would shrink the gold set without saying so.
            writer.writerow(row)
            continue
        matched += 1
        from_corpus = {c: source.get(c, "") for c in carried}
        # The CORPUS wins on the carried columns, the coder wins on everything
        # they were asked to fill in. It used to be the other way round
        # (`{**from_corpus, **row}`), which mattered once `source` started
        # travelling with the sample (#61): that column now sits in a
        # spreadsheet in front of two human coders, and `corrected_ci_bootstrap`
        # clusters the gold set by it. Measured on a 5-video corpus, a coder
        # clearing that column took the gold set from 5 clusters to 1 -- which
        # the min_gold_clusters guard then refuses outright. Partial damage is
        # quieter and worse. Excel also autoformats URLs, and this project has
        # been bitten by Excel's handling of these files twice already.
        for column, corpus_value in from_corpus.items():
            coded_value = (row.get(column) or "").strip()
            # Includes the blank case on purpose: a coder clearing the column
            # is the single most likely way this happens, and it is exactly
            # the one that used to collapse the gold set to one cluster.
            if column in row and coded_value != (corpus_value or "").strip():
                altered[column] += 1
        writer.writerow({**row, **from_corpus})
    print(f"Joined {matched} of {len(coded_rows)} coded comments to their machine labels.",
          file=sys.stderr)
    if altered:
        detail = ", ".join(f"{column} ({count})" for column, count in sorted(altered.items()))
        print(f"NOTE: the coded file disagreed with the corpus on {detail}. The corpus value "
              f"was used -- it is authoritative for provenance, and a coder is not asked to "
              f"edit these. Worth a look if the count is large: it may mean the file came "
              f"back reformatted.", file=sys.stderr)
    if missing:
        print(f"WARNING: {missing} coded comment_id(s) are not in the labelled corpus. They "
              f"were written through without machine labels (--allow-partial). Every scorer "
              f"leaves such rows out and counts them; none treats them as misses. Check "
              f"you are joining against the same corpus the sample was drawn from.",
              file=sys.stderr)
    return matched, missing


def gold_weight(row):
    """Horvitz-Thompson weight for one gold row: 1 / inclusion probability.

    `cmd_sample --stratify-by` deliberately over-samples rare categories and
    records `inclusion_prob` per row, precisely so the resulting precision and
    recall can be read back onto the corpus. Every scorer then counted rows
    unweighted, which makes the figures describe the *draw* rather than the
    corpus -- and a stratified draw is by construction not representative.

    Measured on the audit corpus with `--stratify-by domain`, inclusion
    probabilities spanned 0.027 to 1.000 (a 37x spread) and `social` recall
    moved from 0.89 unweighted to 0.76 weighted.

    A blank probability weighs 1, which reproduces the unweighted result for a
    simple random sample, where equal probabilities cancel anyway. That is only
    right when EVERY row is blank: `check_gold_weights` refuses a file that mixes
    blank and recorded probabilities. A recorded value that is not a number in
    (0, 1] raises. It used to become weight 1 too, so a row that should weigh
    thousands (or, at 0, could not have been drawn) silently counted as one.
    """
    raw = row.get("inclusion_prob")
    text = "" if raw is None else str(raw).strip()
    if not text:
        return 1.0
    try:
        prob = float(text)
    except ValueError:
        prob = float("nan")
    if not 0 < prob <= 1:
        raise ValueError(
            f"gold row {row.get('comment_id', '?')!r}: inclusion_prob {text!r} is not a "
            f"probability in (0, 1]. Recompute it (slens-validate reweight) or exclude the row; "
            f"it cannot be given a weight.")
    return 1.0 / prob


def check_gold_weights(rows):
    """Refuse a gold set where some rows record an inclusion probability and others
    do not. Blank means "weight 1", which is only right for an unweighted (simple
    random) set; mixed in among stratified rows weighing up to thousands, a blank
    row counts as almost nothing and nobody is told. Also validates every value
    (see `gold_weight`). Returns True when the set is weighted."""
    recorded = [bool(str(r.get("inclusion_prob") or "").strip()) for r in rows]
    if any(recorded) and not all(recorded):
        missing = [r.get("comment_id", "?") for r, has in zip(rows, recorded) if not has]
        raise ValueError(
            f"{len(missing)} of {len(rows)} gold row(s) have no inclusion_prob while the others "
            f"do: {', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}. Recompute "
            f"the probabilities for every row (slens-validate reweight) or remove these rows.")
    for row in rows:
        gold_weight(row)
    return any(recorded)


def require_gold_weights(rows, source="the gold set"):
    """`check_gold_weights` for a command line: a clear exit instead of a traceback."""
    try:
        return check_gold_weights(rows)
    except ValueError as error:
        sys.exit(f"{source}: {error}")


def cmd_score(rows):
    coded = [r for r in rows if (r.get("hand_domain") or "").strip()]
    if not coded:
        sys.exit("No hand_domain values found.")
    prefix = machine_prefix(coded)
    if prefix is None:
        sys.exit("Refusing to score: nothing here to score against.\n\n" + MACHINE_COLUMNS_HINT)
    source = "LLM" if prefix == "domain_" else "keyword classifier"
    # A hand-coded row with no machine label at all (a join that could not match it) was never
    # labelled: leave it out and say so, rather than score its blank cells as misses.
    coded = drop_unlabelled(coded, prefix)
    require_gold_weights(coded)
    print(f"Scoring against the {source} labels ({prefix}*).", file=sys.stderr)
    print(f"Scoring {len(coded)} hand-coded comments\n")
    print(f"{'domain':<14}{'precision':>11}{'recall':>9}{'F1':>7}   ({source} vs hand)")
    for category in CATEGORIES:
        tp = fp = fn = 0.0
        for row in coded:
            # A comment the classifier could not parse is not one it got
            # wrong -- it is one it did not answer. Counting it as a miss for
            # every category deflates recall by the parse-failure rate, and
            # report.corrected_estimate divides by (TPR - FPR), so the
            # corrected prevalence inflates in turn. Different denominator.
            if row.get(f"{prefix}{PARSE_FAILURE}") == "1":
                continue
            weight = gold_weight(row)
            hand = category in (row["hand_domain"] or "").lower()
            machine = row.get(f"{prefix}{category}") == "1"
            tp += weight if (hand and machine) else 0.0
            fp += weight if (machine and not hand) else 0.0
            fn += weight if (hand and not machine) else 0.0
        # Same distinction as evaluate.prf: a category in play but scored
        # nothing is F1 0.0; only a category absent from both sides is NaN.
        # (The old `if precision and recall` guard did not even do what it
        # looked like -- NaN is truthy, so it let NaN through the arithmetic.)
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        if tp + fp + fn == 0:
            f1 = float("nan")
        elif tp == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        print(f"{category:<14}{precision:>11.2f}{recall:>9.2f}{f1:>7.2f}")
    print("\nReport these numbers alongside any harm breakdown in the briefing.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["agree", "sample", "score", "join", "interrater",
                                            "combine", "selfconsistency", "reweight"])
    parser.add_argument("input_csv", nargs="?",
                        help="For `interrater`: the first coder's completed file. Not used by "
                             "`combine` -- see --coders/--manifest/--master instead.")
    parser.add_argument("--coders", nargs="+",
                        help="`combine` only: every coder's completed file (.xlsx or .csv).")
    parser.add_argument("--manifest",
                        help="`combine` only: scripts_scratch_split_gold_set.py's manifest "
                             "(comment_id, assigned_to, is_core), recording who coded what.")
    parser.add_argument("--master",
                        help="`combine` and `reweight`: the drawn gold sample (e.g. "
                             "gold_sample_balanced.csv) -- inclusion_prob always comes from "
                             "here, never from a coder's own (possibly stale) copy of it.")
    parser.add_argument("--replacements",
                        help="`combine` only, optional: a hand-coded file of comments that "
                             "replace specific drawn rows (e.g. because their video was later "
                             "excluded), each row carrying a replaces_comment_id column naming "
                             "which drawn row it stands in for.")
    parser.add_argument("--topup",
                        help="`combine` and `reweight`, optional: a later random draw of extra comments from "
                             "each story's unclassified stratum (e.g. gold_sample_topup.csv). "
                             "Needs no split-manifest entry; whoever coded it, in --coders, is used.")
    parser.add_argument("--sentiment-agreement", choices=["exact", "bucket"], default="exact",
                        help="`combine` only: when two coders rated the same core comment, "
                             "'exact' needs identical sentiment; 'bucket' accepts the same "
                             "side (1-2 negative, 3 neutral, 4-5 positive; 0 and -1 stay exact) "
                             "and writes the bucket word.")
    parser.add_argument("--adjudicated",
                        help="`combine` only, optional: a file of comment_id, hand_domain, "
                             "hand_sentiment resolving rows the coders disagreed on. Each "
                             "overrides the coders' answers for that comment.")
    parser.add_argument("--corpus",
                        help="`combine` and `reweight`: the FINAL corpus with a primary_category "
                             "column (e.g. the *_kw_primary.csv). When given, every inclusion "
                             "probability is recomputed against it, instead of copied from the "
                             "sample files, which describe the frame as it was when drawn. "
                             "Requires --video-cap-fraction (0 for none).")
    parser.add_argument("--json-output",
                        help="Write `interrater`'s figures as JSON, for the dashboard. Coder "
                             "agreement is the ceiling on every accuracy claim in the project; "
                             "if it can only be retyped from a terminal it stops being "
                             "evidence.")
    parser.add_argument("--second-input", help="`interrater` only: the second coder's completed "
                                               "file, covering the same comment_ids as input_csv.")
    parser.add_argument("--sentiment-levels", choices=["scale", "buckets"], default="scale",
                        help="`interrater` with three or more coders: how the stance is compared. "
                             "'scale' (default) compares the 1-5 ratings (ordinal alpha); "
                             "'buckets' compares only the direction, 1-2 negative / 3 neutral / "
                             "4-5 positive, as three ordered classes. The kind-of-answer figure "
                             "(stance / irrelevant / uncodeable) is reported either way.")
    parser.add_argument("--extra-inputs", nargs="+",
                        help="`interrater` only, optional: any THIRD or later coder's completed "
                             "file(s), for more than two overlapping coders (e.g. the shared "
                             "50-row core coded by all 5). Passing this switches the report from "
                             "pairwise Cohen's kappa to Fleiss' kappa across every file given "
                             "(input_csv + --second-input + --extra-inputs).")
    parser.add_argument("--exclude-ids",
                        help="`score` only: a file of comment_ids (one per line, or a CSV with a "
                             "comment_id column) to leave out of scoring -- the comments shown to "
                             "the model as worked examples.")
    parser.add_argument("--allow-partial", action="store_true",
                        help="`join` only: accept a join in which some coded comments are not in "
                             "--labelled. They are written without machine labels and every "
                             "scorer then leaves them out (counted), never scores them as "
                             "misses. Without this flag such a join is refused.")
    parser.add_argument("--no-example-exclusion", action="store_true",
                        help="`score` only: silence the warning printed when no worked-example "
                             "ids are known (no run record beside the file and no --exclude-ids). "
                             "Use it only when none of the gold comments was shown to the model "
                             "as an example.")
    parser.add_argument("--core-ids", help="`selfconsistency` only: a file naming the comment_ids "
                                           "to restrict to (one per line, or any CSV with a "
                                           "comment_id column) -- e.g. official-data/"
                                           "gold_sample_core_50.csv, so this measures the LLM's "
                                           "self-consistency on the SAME comments the human "
                                           "coder-vs-coder kappa was measured on, not a different, "
                                           "much larger set that isn't a fair comparison. Omit to "
                                           "measure over every row in input_csv instead.")
    parser.add_argument("--compare-json", help="`selfconsistency` only: a prior `slens-validate "
                                               "interrater --json-output ...` run's file, printed "
                                               "alongside this run's figures for a direct "
                                               "side-by-side instead of two outputs to reconcile "
                                               "by eye.")
    parser.add_argument("--labelled", help="`join` only: the labelled corpus to take the machine "
                                           "labels from, matched on comment_id. `sample` strips "
                                           "those columns so the coder cannot see the machine's "
                                           "answer; `join` puts them back afterwards, which is "
                                           "what `score` and --scored need.")
    parser.add_argument("--n", type=int, default=300, help="Sample size for `sample`")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify-by", help="Column(s) to stratify `sample` on instead of "
                                              "story+platform, comma-separated, e.g. kw_domain "
                                              "or story,kw_domain. A domain column over-samples "
                                              "rare categories so they can be validated (#18); "
                                              "adding story alongside it also keeps one story "
                                              "from dominating a category's examples.")
    parser.add_argument("--output", help="Write `sample`'s CSV directly to this file, with the "
                                          "same Windows-safe encoding the collectors use. If "
                                          "omitted, writes to stdout — fine in Git Bash, but "
                                          "PowerShell's '>' redirection still re-encodes as "
                                          "UTF-16, so prefer --output there.")
    parser.add_argument("--video-cap-fraction", type=float,
                        help="`sample` and `combine --corpus`: cap how much of a stratum's allocation can come "
                             "from one video, as a fraction (e.g. 0.15 = at most 15%%). A soft "
                             "cap -- if a stratum cannot reach its target within the cap, the "
                             "shortfall is backfilled past it rather than under-filling the "
                             "stratum. Guards against a story's gold sample being mostly one or "
                             "two videos' comments, which understates the between-video variance "
                             "the clustered bootstrap depends on. Omit for the old uncapped "
                             "behaviour.")
    args = parser.parse_args()

    if args.command == "reweight":
        if not (args.input_csv and args.master and args.corpus and args.output):
            parser.error("reweight needs the gold file (positional), --master (the original "
                         "draw), --corpus (the final corpus with primary_category) and --output")
        if args.video_cap_fraction is None:
            parser.error("reweight needs --video-cap-fraction stated explicitly (the first "
                         "draw's soft cap, e.g. 0.15; 0 for none)")
        corpus_rows = read(args.corpus)
        if corpus_rows and "primary_category" not in corpus_rows[0]:
            parser.error(f"{args.corpus} has no primary_category column")
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", newline="", encoding="utf-8-sig") as handle:
            cmd_reweight(read_any(args.input_csv), read_any(args.master),
                         read_any(args.topup) if args.topup else [], corpus_rows,
                         args.video_cap_fraction, handle)
        print(f"Wrote {args.output}", file=sys.stderr)
        return

    if args.command == "combine":
        if not args.coders:
            parser.error("combine needs --coders <coder file> [<coder file> ...]")
        if not args.manifest:
            parser.error("combine needs --manifest <split manifest csv>")
        if not args.master:
            parser.error("combine needs --master <the drawn gold sample, e.g. "
                         "gold_sample_balanced.csv>")
        if not args.output:
            parser.error("combine needs --output <where to write the combined file>")
        if args.topup and not args.corpus:
            parser.error("--topup needs --corpus: a top-up is a second draw, so its inclusion "
                         "probability only exists once recomputed against the final corpus")
        corpus_rows = None
        if args.corpus:
            if args.video_cap_fraction is None:
                parser.error("--corpus needs --video-cap-fraction stated explicitly (the first "
                             "draw's soft cap, e.g. 0.15; 0 for none) -- guessing it would "
                             "silently change every probability")
            corpus_rows = read(args.corpus)
            if corpus_rows and "primary_category" not in corpus_rows[0]:
                parser.error(f"{args.corpus} has no primary_category column; use the corpus "
                             f"after slens-primary-category")
        coder_rows_by_path = {path: read_any(path) for path in args.coders}
        replacement_rows = read_any(args.replacements) if args.replacements else []
        topup_rows = read_any(args.topup) if args.topup else None
        adjudicated_rows = read_any(args.adjudicated) if args.adjudicated else None
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", newline="", encoding="utf-8-sig") as handle:
            cmd_combine(coder_rows_by_path, read(args.manifest), read(args.master),
                       replacement_rows, handle, corpus_rows=corpus_rows,
                       topup_rows=topup_rows, cap_fraction=args.video_cap_fraction or None,
                       adjudicated_rows=adjudicated_rows,
                       sentiment_agreement=args.sentiment_agreement)
        print(f"Wrote {args.output}", file=sys.stderr)
        return

    if not args.input_csv:
        parser.error(f"{args.command} needs a positional input file")
    if args.command == "score":
        # The run record beside the gold file names the worked examples; --exclude-ids adds more.
        exclude_ids = example_ids_for_gold(args.input_csv, getattr(args, "exclude_ids", None),
                                           args.no_example_exclusion)
    else:
        exclude_ids = read_ids(args.exclude_ids) if getattr(args, "exclude_ids", None) else None
    # read_any, not read: `interrater`'s positional input is a coder's
    # completed file, handed out as .xlsx the same way `combine`'s --coders
    # are (see scripts_scratch_split_gold_set.py) -- read() on one raised an
    # opaque UnicodeDecodeError rather than telling the caller to convert it,
    # or worse, silently misparsing it if the encoding happened to look valid.
    # A harmless no-op for every other command here, all of which pass CSVs.
    data = read_any(args.input_csv)
    if args.command == "agree":
        cmd_agree(data)
    elif args.command == "sample":
        if args.output:
            out_dir = os.path.dirname(args.output)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.output, "w", newline="", encoding="utf-8-sig") as handle:
                cmd_sample(data, args.n, args.seed, handle, args.stratify_by, args.video_cap_fraction)
        else:
            # newline="" here matters as much as the encoding: csv.writer's own
            # "\r\n" row terminator gets doubled to "\r\r\n" by stdout's default
            # newline translation otherwise, which shows up as a blank row after
            # every real one.
            sys.stdout.reconfigure(newline="")
            cmd_sample(data, args.n, args.seed, sys.stdout, args.stratify_by, args.video_cap_fraction)
    elif args.command == "join":
        if not args.labelled:
            parser.error("join needs --labelled <the labelled corpus>")
        if not args.output:
            parser.error("join needs --output <where to write the joined file>")
        labelled = read(args.labelled)
        # Checked before the output is opened, so a refused join leaves no empty file behind.
        error = unmatched_join_error(data, labelled)
        if not error and not args.allow_partial:
            error = partial_join_error(data, labelled)
        if error:
            sys.exit(error)
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", newline="", encoding="utf-8-sig") as handle:
            cmd_join(data, labelled, handle, allow_partial=args.allow_partial)
        print(f"Wrote {args.output}", file=sys.stderr)
        # The machine labels beside the hand codes ARE the labelled file's, so its run record
        # (judge, prompts, worked examples: hashes and settings, no comment text) describes
        # them too. Carried across so `score`, `slens-report` and `slens-export-dashboard` can
        # leave the worked examples out and check the instrument, not warn that none is known.
        record = args.labelled + ".run.json"
        if os.path.exists(record) and os.path.abspath(args.output) != os.path.abspath(args.labelled):
            shutil.copyfile(record, args.output + ".run.json")
            print(f"Run record copied to {args.output}.run.json", file=sys.stderr)
    elif args.command == "interrater":
        if not args.second_input:
            parser.error("interrater needs --second-input <coder B's completed file>")
        if args.extra_inputs:
            rows_by_coder = [data, read_any(args.second_input)] + [
                read_any(path) for path in args.extra_inputs]
            paths = [args.input_csv, args.second_input] + list(args.extra_inputs)
            names = [os.path.splitext(os.path.basename(p))[0].replace("Gold Set Complete Kappa", "")
                     .strip(" _#-") or os.path.basename(p) for p in paths]
            cmd_interrater_multi(rows_by_coder, json_output=args.json_output, names=names,
                                 sentiment_levels=args.sentiment_levels)
        else:
            cmd_interrater(data, read_any(args.second_input), json_output=args.json_output)
    elif args.command == "selfconsistency":
        core_ids = read_ids(args.core_ids) if args.core_ids else None
        cmd_selfconsistency(data, core_ids=core_ids, compare_json=args.compare_json)
    else:
        cmd_score(drop_ids(data, exclude_ids, "gold rows"))


if __name__ == "__main__":
    main()
