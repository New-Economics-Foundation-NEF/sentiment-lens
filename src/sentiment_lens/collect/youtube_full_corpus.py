"""Full-corpus variant of `slens-collect`: every comment on every eligible video,
not the hybrid-selected, capped sample.

    slens-collect-full-corpus --story propublica --search "..." --output data/x.csv

This is deliberately a thin wrapper around `youtube.py`, not a second copy of
it -- three near-identical collectors already existed once (#16) and were
consolidated into one specifically so a fix in one could not silently fail to
reach the others. Duplicating the collection logic again would recreate that
exact risk. Instead this just pre-injects the flags that turn "a representative
sample of videos, capped per video" into "everything that clears Shorts and
--min-comments", ahead of whatever the caller passes -- so an explicit flag on
the command line still wins (argparse keeps the last value of a repeated flag).

What "no caps" means concretely:
  --limit 0                 no per-video comment cap (comment_limit is falsy,
                             see scrape_comments in youtube.py)
  --target-videos 999999    every video that survives filtering is drawn,
                             not a sample of it (search_youtube falls back to
                             "take all survivors" once target exceeds them)
  --min-comments 0          no thin-video filtering
  --max-shorts-seconds 0    Shorts are not excluded either

Precision in this project's own clustered-bootstrap methodology comes from
video *count*, not comment count per video -- see CLAUDE.md section 5. This
variant exists for corpora you want to slice and re-aggregate differently
later (e.g. a deeper per-video look), not because it improves that
methodology's confidence intervals. `--max-search-pages` is left at whatever
the caller sets (default 1): that widens how many candidate videos are found
in the first place, a separate decision from how much of each one is kept.
"""
import sys

from sentiment_lens.collect import youtube

FULL_CORPUS_OVERRIDES = [
    "--limit", "0",
    "--target-videos", "999999",
    "--min-comments", "0",
    "--max-shorts-seconds", "0",
]


def main():
    # Inserted before the caller's own args: argparse keeps the LAST value of
    # a repeated flag, so an explicit --limit/--target-videos/etc. on the
    # actual command line still overrides these defaults.
    sys.argv = [sys.argv[0]] + FULL_CORPUS_OVERRIDES + sys.argv[1:]
    youtube.main()


if __name__ == "__main__":
    main()
