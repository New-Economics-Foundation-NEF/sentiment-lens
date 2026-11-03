"""YouTube collector using the official Data API v3.

    export YOUTUBE_API_KEY=...
    python youtube_api.py "<video-url-or-id>" --story wealth_tax --limit 500 > out.csv

`youtube_scrape.py` (yt-dlp) still works and needs no key. This exists because
#4 concluded both collectors should move to official APIs, and because the API
gives two things scraping cannot:

* **Exact timestamps.** yt-dlp derives times from relative labels ("3 months
  ago"), so it writes `timestamp_precision = day` and the README rightly warns
  that time-series work must filter those rows out. `publishedAt` is exact, so
  rows from here carry `second` and are usable for time series.
* **A documented quota instead of a rate limit to guess at.** commentThreads
  costs 1 unit per call against a free 10,000/day, and each call returns up to
  100 comments. The whole corpus is a rounding error against that.

Collects top-level comments and their replies. `depth` is 0 and 1 respectively,
where yt-dlp leaves it blank.
"""
import argparse
import os
import re
import sys
import time

import requests

import ewl.common as common

API = "https://www.googleapis.com/youtube/v3"


def video_id(url_or_id):
    """Accept a bare id, a watch URL, a youtu.be link or a shorts link."""
    if re.fullmatch(r"[\w-]{11}", url_or_id):
        return url_or_id
    match = re.search(r"(?:v=|/shorts/|youtu\.be/|/embed/)([\w-]{11})", url_or_id)
    if not match:
        sys.exit(f"Could not find a video id in {url_or_id!r}")
    return match.group(1)


def get(path, key, **params):
    response = requests.get(f"{API}/{path}", params={"key": key, **params}, timeout=30)
    if response.status_code != 200:
        error = response.json().get("error", {})
        reason = (error.get("errors") or [{}])[0].get("reason", "")
        if reason in {"commentsDisabled", "videoNotFound"}:
            sys.exit(f"{reason} for this video — nothing to collect.")
        if reason in {"quotaExceeded", "dailyLimitExceeded"}:
            sys.exit("YouTube API quota exhausted for today. The default is 10,000 units; "
                     "each page of 100 comments costs 1.")
        sys.exit(f"YouTube API {response.status_code}: {error.get('message', response.text[:200])}")
    return response.json()


def emit(rows, item, story, video, depth, lang, dropped, drop_non_lang=False):
    """Map one API comment resource onto the shared schema."""
    snippet = item["snippet"]
    body = snippet.get("textOriginal") or snippet.get("textDisplay") or ""
    author = snippet.get("authorDisplayName", "")
    if not common.is_usable(body, author):
        return
    detected = common.detect_language(body)
    # Recorded, not acted on, unless dropping was explicitly asked for -- the
    # same contract as youtube_collect.py. Previously this collector dropped by
    # language unconditionally while the other one did not, so which collector
    # produced a corpus decided whether the known-bad filter (#23) had been
    # applied to it.
    if lang and drop_non_lang and detected and detected != lang:
        dropped[0] += 1
        return
    rows.append({
        "comment_id": item["id"],
        "platform": "youtube",
        "story": story,
        "source": f"https://www.youtube.com/watch?v={video}",
        "author_hash": common.hash_author(author),
        "body": body,
        "upvotes": snippet.get("likeCount", 0),
        "timestamp": snippet.get("publishedAt", ""),
        # Exact, unlike the yt-dlp path — see the module docstring.
        "timestamp_precision": "second",
        "depth": depth,
        "collected_at": common.now_iso(),
        "detected_lang": detected,
    })


def collect(video, story, key, limit, order, lang=None, drop_non_lang=False):
    rows, page, dropped = [], None, [0]
    while len(rows) < limit:
        payload = get("commentThreads", key, part="snippet,replies", videoId=video,
                      maxResults=100, order=order, textFormat="plainText",
                      **({"pageToken": page} if page else {}))
        for thread in payload.get("items", []):
            emit(rows, thread["snippet"]["topLevelComment"], story, video, 0, lang, dropped,
                 drop_non_lang)
            for reply in thread.get("replies", {}).get("comments", []):
                emit(rows, reply, story, video, 1, lang, dropped, drop_non_lang)
        page = payload.get("nextPageToken")
        print(f"  {len(rows)} comments", file=sys.stderr)
        if not page:
            break
        time.sleep(0.1)
    if lang and dropped[0]:
        print(f"  dropped {dropped[0]} comment(s) not detected as '{lang}'", file=sys.stderr)
    # Truncation must be visible. #5 was a silent 200-per-thread cap whose loss
    # nobody could see from the output.
    if len(rows) > limit:
        print(f"NOTE: truncated to --limit {limit}; {len(rows) - limit} more were available "
              f"and are being dropped.", file=sys.stderr)
    return rows[:limit]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("--drop-non-lang", action="store_true",
                        help="Discard comments whose detected language differs from --lang, "
                             "instead of only recording it in detected_lang. Off by default "
                             "-- see #23 for why.")
    parser.add_argument("--story", required=True, help="Story label written into every row")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--order", default="relevance", choices=["relevance", "time"],
                        help="'time' is the unbiased ordering; 'relevance' is YouTube's "
                             "engagement ranking and is not a random sample.")
    parser.add_argument("--lang", help="Keep only comments detected as this ISO 639-1 "
                                        "language code, e.g. 'en'. Requires `pip install "
                                        "langdetect`. Unset (default): no filtering. "
                                        "Comments under ~20 characters are always kept, "
                                        "since a detector cannot trust that little text.")
    args = parser.parse_args()

    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        sys.exit("Set YOUTUBE_API_KEY. Get one from the Google Cloud console: enable "
                 "YouTube Data API v3 and create an API key.")

    video = video_id(args.video)
    print(f"Collecting from {video} (story={args.story}, order={args.order}"
          f"{f', lang={args.lang}' if args.lang else ''})", file=sys.stderr)
    rows = collect(video, args.story, key, args.limit, args.order, args.lang,
                   args.drop_non_lang)

    import csv
    writer = csv.DictWriter(sys.stdout, fieldnames=common.FIELDNAMES, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    print(f"\nWrote {len(rows)} comments. Ordering was '{args.order}' — record that in the "
          f"methods section, since it determines what this sample is a sample of.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
