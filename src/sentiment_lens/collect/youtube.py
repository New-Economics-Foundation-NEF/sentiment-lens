"""Collect comments from YouTube videos into the shared CSV schema.

    python youtube_collect.py --story luigi --limit 2000 <video-url> [<video-url> ...]
    python youtube_collect.py --story luigi --search "unitedhealthcare ceo shooting"

Consolidation, per #16
-----------------------
This replaces YT_ScraperV2.py and YT_ScraperV2_RandomSample.py, which #16
found to be 288 lines duplicated to change one function (`search_youtube`).
A fix to the shared half — like the BOM/encoding bug in #15 — had to be made
twice, and the two were already starting to drift.

What survives from each, rather than picking one file and discarding the
rest:

* From YT_ScraperV2_RandomSample.py — random video selection when using
  `--search`, not YouTube's top-N by relevance alone. #16 called this "the
  best idea in the set." It has since become half of a hybrid strategy (see
  `search_youtube`'s docstring): `--top-n` videos by relevance, then random
  selection for the rest up to `--target-videos`, since a design-effect
  measurement found that pure relevance risks a biased handful of videos
  and precision comes from video *count*, which needs a real target rather
  than however many a purely-random draw happens to include. `--min-comments`
  and `--seed` come along with it.
* From YT_ScraperV2.py — `--output` with explicit Windows-safe encoding,
  `--search`, `--order`, `--lang`, and per-video error handling that keeps
  one bad URL from aborting a run.
* From youtube_api.py — reply collection. Requesting
  `part="snippet,replies"` returns up to ~5 replies per thread in the same
  response, at no extra quota cost. #16 measured this at keeping 91% of a
  video's comments when replies are skipped, i.e. dropping 9% for nothing.
  `depth` is 0 for a top-level comment, 1 for a reply — matching the Reddit
  collector's existing `depth` column, so it means the same thing on both
  platforms. `--limit` counts top-level comments only; replies come with
  them and are not counted against it, since discarding replies to stay
  under a comment budget would defeat the point of collecting them. A full
  reply thread needs a further `comments.list` call per thread, at one more
  quota unit each — not implemented here, same call #16 made: probably not
  worth it for this project's scale.

youtube_api.py itself is unchanged and still exists: it is the tool
`RUNBOOK.md`'s measured walkthrough actually used, and nothing about that
history should quietly change under it.
"""
import argparse #This is the package required to handle the commmand line interface, which is where you input search terms, how many comments you want etc
import collections #Used for the exclusion-accounting counters -- how many comments were removed at each filtering stage.
import os #manages interactions with the OS
import random #Used to pick a random sample of candidate videos, instead of always taking YouTube's top-ranked results.
import re #This Package is used for finding and matching patterns in text, rather than searching all of a specific phrase.
import sys # Package used to control several things about the system, including where the output of the code appears

#By default windows has a certain rulebook for how to translate numbers into letters, symbols etc.
#This process of "translating" is called encoding. The line below ensures that the script enocdes to a specific format that understands emojis. Without it, the script would spit out emojis incomprehesibly.
sys.stdout.reconfigure(encoding="utf-8")

from googleapiclient.discovery import build #This is importing a specific function from the discovery sub-package. This module allows you to build objects using all of Google's APIs- you invoke the function and then input the specific APIs you want to use
from googleapiclient.errors import HttpError #This imports a class that provides error outputs to do with the YT API- e.g. invalid video ID, deleted video etc

from sentiment_lens.common import FIELDNAMES, detect_language, hash_author, is_usable, now_iso #grabs the column names, hashing function, timestamping, language detection and comment suitability test function from the other file.

API_KEY = os.environ.get("YOUTUBE_API_KEY") #Safely stores the YT API key input into the terminal as an object.

VIDEO_ID_PATTERNS = [
    re.compile(r"(?:v=|/shorts/|youtu\.be/)([A-Za-z0-9_-]{11})"),
]
#This creates a template for some pattern of text that ca be used later


def extract_video_id(url_or_id):
    """Accept a full YouTube URL or a bare video ID and return the ID."""
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id): #if there is  full match for a specific pattern, return the url or id.
        return url_or_id
    for pattern in VIDEO_ID_PATTERNS: #Creates a variabel called pattern which stores list (in this case list of one) of ID patterns
        match = pattern.search(url_or_id) #Conducts a search trying to match the url to all of the patterns established in the ID patterns list
        if match:
            return match.group(1)
    return None


def get_client():
    if not API_KEY:
        sys.exit("Set the YOUTUBE_API_KEY environment variable before running this script.")
    return build("youtube", "v3", developerKey=API_KEY)


def utc_iso_from_string(published_at):
    """The API already returns UTC ISO 8601 strings, so this is a pass-through
    kept separate from sentiment_lens.common.utc_iso(), which expects epoch seconds for
    the Reddit collector instead."""
    return published_at or ""


def scrape_comments(youtube, url, story, comment_limit, order="relevance", lang=None,
                    drop_non_lang=False, exclusions=None, video_log=None,
                    search_query="", video_title=""):
    """Collect one video's comments.

    `exclusions`, if given, is a shared `collections.Counter` that this call
    adds to rather than replaces -- callers collecting multiple videos pass
    the same Counter through every call to get one running total across a
    story or a whole run, for the exclusion-accounting table. `video_log`,
    if given, is a shared list this call appends one row to, for the
    video-list appendix (video ID, comments kept, dates, search query).
    """
    if exclusions is None:
        exclusions = collections.Counter()

    video_id = extract_video_id(url)
    if not video_id:
        print(f"  could not parse a video ID from: {url}", file=sys.stderr)
        return []

    print(f"Fetching comments for: {url} (order={order})", file=sys.stderr)
    #'order' is only visible if a comment_limit cuts collection short: with every
    #comment collected, which order they arrived in changes nothing about the
    #final set of rows. Recorded and printed anyway, in case a run does end up capped.
    collected_at = now_iso()
    rows = []
    lang_dropped = 0
    total_seen = 0  # top-level threads seen; replies are not counted against --limit
    next_page_token = None
    hit_cap = False

    def add(item, depth):
        nonlocal lang_dropped
        exclusions["api_returned"] += 1
        snippet = item["snippet"]
        author = snippet.get("authorDisplayName") or ""
        body = snippet.get("textDisplay") or snippet.get("textOriginal") or ""
        if not is_usable(body, author):
            exclusions["deleted_removed_or_automod"] += 1
            return
        detected = detect_language(body)
        # Recorded on every row. Dropping only when --drop-non-lang was asked
        # for explicitly: see #23 for why the old drop-by-default removed eight
        # comments from a 313-comment sample and all eight were English.
        if lang and drop_non_lang and detected and detected != lang:
            lang_dropped += 1
            exclusions["non_english"] += 1
            return
        rows.append({
            "comment_id": item.get("id", ""),
            "platform": "youtube",
            "story": story,
            "source": url,
            "author_hash": hash_author(author),  # normalised inside hash_author
            "body": body,
            "upvotes": snippet.get("likeCount"),
            "timestamp": utc_iso_from_string(snippet.get("publishedAt")),
            "timestamp_precision": "second",  # API gives real ISO timestamps
            "depth": depth,  # 0 = top-level comment, 1 = reply
            "collected_at": collected_at,
            # Best guess, or "" when too short / undetectable / langdetect
            # absent. For inspection and downstream filtering, not acted on here.
            "detected_lang": detected,
        })

    while True: #this creates a loop that only breaks under the condition that something becomes false. If not broken, the loop runs forever.
        try:
            response = youtube.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText",
                order=order,
            ).execute()
        except HttpError as e:
            print(f"  API error on {url}: {e}", file=sys.stderr)
            break

        items = response.get("items") or []
        if not items and total_seen == 0:
            print(f"No comments found for: {url}", file=sys.stderr)

        for position, item in enumerate(items):
            total_seen += 1
            if comment_limit and total_seen > comment_limit:
                hit_cap = True
                # Everything from here to the end of this page is returned by
                # the API and never examined. Counted, because --limit is the
                # normal way this collector is run and was therefore the
                # largest exclusion the ledger said nothing about: `add()` is
                # what increments api_returned, and we never reach it.
                skipped = items[position:]
                exclusions["excluded_by_comment_limit"] += sum(
                    1 + len(it.get("replies", {}).get("comments", []))
                    for it in skipped)
                break

            add(item["snippet"]["topLevelComment"], 0)
            for reply in item.get("replies", {}).get("comments", []):
                add(reply, 1)

        if hit_cap:
            break
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    if comment_limit and hit_cap:
        print(f"  WARNING: hit the {comment_limit}-comment cap on {url}; "
              f"this video's comments are truncated", file=sys.stderr)
        exclusions["videos_truncated_by_limit"] += 1
    if lang and lang_dropped:
        print(f"  dropped {lang_dropped} comment(s) not detected as '{lang}'", file=sys.stderr)
    print(f"  kept {len(rows)} rows from {total_seen} top-level thread(s) seen "
          f"(includes inline replies)", file=sys.stderr)
    if video_log is not None:
        video_log.append({
            "video_id": video_id,
            "story": story,
            "video_title": video_title,
            "search_query": search_query,
            "comments_kept": len(rows),
            "collected_at": collected_at,
        })
    return rows


# Anchored, and the T separator is required before any time component, so
# "P1H" (H without T) and the empty "P" / "PT" are rejected rather than parsed.
# That matters because a bogus 0 reads as a 0-second video and is silently
# excluded as a Short -- the same "unknown must not look like very short" trap
# duration_seconds() exists to avoid.
ISO_DURATION = re.compile(
    r"^P(?!$)(?:(\d+)D)?(?:T(?!$)(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$")


def duration_seconds(iso):
    """Parse the ISO 8601 duration videos().list returns (e.g. "PT4M13S").

    Returns None when the value is missing or unparseable — live streams and
    premieres have no duration — so callers can treat "unknown" differently
    from "short", rather than a failed parse quietly reading as 0 seconds and
    excluding the video.
    """
    match = ISO_DURATION.match(iso or "")
    if not match or not any(match.groups()):
        # No match, or "P" / "PT" with no components at all. Both mean "cannot
        # tell", which must stay distinguishable from a real zero.
        return None
    days, hours, minutes, seconds = (int(g or 0) for g in match.groups())
    total = days * 86400 + hours * 3600 + minutes * 60 + seconds
    if total == 0 and "T" not in iso:
        # YouTube returns "P0D" for live and upcoming streams, which have no
        # duration rather than a zero-second one. Reading that as 0 would
        # exclude every live stream as a Short, for the wrong reason and
        # without saying so. "PT0S" is a genuine zero and still parses.
        return None
    return total


# videos().list accepts at most 50 IDs per call. Anything longer is not an
# error you can see: the API returns nothing useful and every video reads as 0
# comments, so the whole pool is excluded by --min-comments and the run reports
# "0 of 191 cleared" with no indication why. Pooling several --search queries
# made that reachable, so the batching is explicit rather than assumed.
MAX_IDS_PER_CALL = 50


def video_meta(youtube, video_ids):
    """{id: (comment_count, duration_seconds)}, in batches of 50 IDs.

    `statistics` and `contentDetails` come back in the same request, so knowing
    a video's length costs no extra quota over the comment count we already
    fetch. Each batch is 1 quota unit.

    A video with comments disabled, or that never returned a statistics block,
    is treated as 0 comments rather than dropped from the dict — so it is
    excluded by --min-comments like any other thin video, instead of silently
    vanishing for a different reason. Duration is None when unknown.
    """
    meta = {}
    for start in range(0, len(video_ids), MAX_IDS_PER_CALL):
        batch = video_ids[start:start + MAX_IDS_PER_CALL]
        try:
            response = youtube.videos().list(part="statistics,contentDetails",
                                             id=",".join(batch)).execute()
        except HttpError as e:
            print(f"  could not fetch video metadata: {e}", file=sys.stderr)
            continue
        for item in response.get("items", []):
            stats = item.get("statistics", {})
            details = item.get("contentDetails", {})
            meta[item["id"]] = (int(stats.get("commentCount", 0)),
                                duration_seconds(details.get("duration")))
    for video_id in video_ids:
        meta.setdefault(video_id, (0, None))
    return meta


def search_youtube(youtube, queries, story, top_n, target_videos, comment_limit, min_comments,
                   seed, order="relevance", lang=None, max_shorts_seconds=180,
                   drop_non_lang=False, exclusions=None, video_log=None,
                   max_search_pages=1):
    """Hybrid video selection: the `top_n` most relevant results, plus
    further videos chosen at random (never repeating one already chosen),
    until `target_videos` total videos are selected for this story.

    Pure relevance-ranking risks missing genuinely representative comments;
    pure random sampling risks a video-count too low to trust a proportion
    (see the design-effect finding: precision comes from the number of
    *videos*, not the number of comments -- a comment-level bootstrap
    understates the interval, because comments under one video are not
    independent observations). The `top_n` slice guarantees the highest-
    engagement conversation for the story is captured; the random top-up
    guards against `top_n` alone being a biased, YouTube-algorithm-selected
    handful of videos.

    `top_n` is taken from `search().list()`'s own relevance ordering (its
    default), applied to the candidate pool *after* Shorts and
    --min-comments have already thinned it, so the comment-count line below
    describes the pool actually being sampled from either way.
    """
    if exclusions is None:
        exclusions = collections.Counter()

    # One shared candidate pool across every --search query, deduplicated by
    # video ID and keeping first-seen order, so the relevance slice below is
    # taken over everything the story is discussed under rather than over
    # whichever query happened to be listed first. Each search PAGE costs 100
    # quota units, so this is the expensive part of a run: four queries at one
    # page each is 400 of the 10,000/day allowance. --max-search-pages
    # multiplies that, which is why it defaults to 1 and paging is stopped as
    # soon as the pool is large enough for the draw.
    if isinstance(queries, str):
        queries = [queries]
    # Which query found each video, so the appendix records how a video
    # entered the corpus rather than attributing everything to one term.
    query_of = {}
    # The title costs no extra quota -- search().list's snippet already
    # carries it -- and it is what actually lets a human tell a genuinely
    # on-topic video from one a broad query pulled in by accident, without
    # opening all thirty of them by hand.
    title_of = {}
    candidate_ids, seen_ids = [], set()
    for query in queries:
        found = 0
        page_token = None
        pages = 0
        # search().list caps a page at 50, and this used to make exactly one
        # call -- so the candidate pool was the top 50 relevance-ranked results
        # per query, full stop. With --target-videos 30 on top of Shorts and
        # --min-comments filtering, the pool routinely could not reach the
        # target, and the shortfall was reported as "only N video(s) cleared
        # --min-comments": an API paging limit presented as a fact about the
        # corpus. Page until the requested pool size is reached.
        # Stop as soon as the pool is comfortably larger than the draw needs.
        # Shorts and --min-comments filtering both cut into it, so aim well
        # above target_videos rather than exactly at it.
        enough = max(50, (target_videos or 0) * 3)
        # `pages == 0` keeps the first page of EVERY query unconditional. The
        # early stop is about not walking extra pages, not about skipping a
        # query -- a pool already filled by query 1 must not silence query 2,
        # which is the whole point of taking the relevance slice across all of
        # them rather than the first one listed.
        while pages < max_search_pages and (pages == 0 or len(candidate_ids) < enough):
            try:
                response = youtube.search().list(
                    q=query,
                    part="snippet",
                    type="video",
                    maxResults=50,  # the API's own per-page maximum
                    pageToken=page_token,
                ).execute()
            except HttpError as e:
                print(f"  search failed for '{query}': {e}", file=sys.stderr)
                break
            pages += 1
            for item in response.get("items", []):
                video_id = item.get("id", {}).get("videoId")
                if video_id and video_id not in seen_ids:
                    seen_ids.add(video_id)
                    candidate_ids.append(video_id)
                    query_of[video_id] = query
                    title_of[video_id] = item.get("snippet", {}).get("title", "")
                    found += 1
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        if len(queries) > 1 or pages > 1:
            print(f"  '{query}': {found} new candidate(s) over {pages} page(s)",
                  file=sys.stderr)

    if not candidate_ids:
        print(f"  no search results for {queries}", file=sys.stderr)
        return []
    if len(queries) > 1:
        print(f"  {len(candidate_ids)} unique candidate video(s) across "
              f"{len(queries)} queries", file=sys.stderr)

    meta = video_meta(youtube, candidate_ids)

    # Shorts first, so the comment-count line below describes the pool that is
    # actually being sampled from.
    shorts = []
    if max_shorts_seconds:
        shorts = [v for v in candidate_ids
                  if meta[v][1] is not None and meta[v][1] <= max_shorts_seconds]
        candidate_ids = [v for v in candidate_ids if v not in set(shorts)]
        if shorts:
            print(f"  excluded {len(shorts)} video(s) at or under "
                  f"{max_shorts_seconds}s as Shorts", file=sys.stderr)
            exclusions["excluded_as_shorts"] += len(shorts)
    if not candidate_ids:
        # `query` used to be read here from the loop above, so with several
        # --search terms this named only the last one.
        print(f"  every search result for {queries} was excluded as a Short",
              file=sys.stderr)
        return []

    survivors = [video_id for video_id in candidate_ids if meta[video_id][0] > min_comments]
    excluded_thin = len(candidate_ids) - len(survivors)
    if excluded_thin:
        exclusions["excluded_below_min_comments"] += excluded_thin
    print(f"  {len(survivors)} of {len(candidate_ids)} remaining search results have more "
          f"than {min_comments} comments", file=sys.stderr)

    if len(survivors) <= target_videos:
        print(f"  WARNING: only {len(survivors)} video(s) cleared --min-comments "
              f"{min_comments}, at or below the {target_videos} targeted; taking all of them",
              file=sys.stderr)
        chosen = survivors
    else:
        top = survivors[:min(top_n, target_videos)]
        remaining = [v for v in survivors if v not in set(top)]
        random.seed(seed)
        extra_needed = max(target_videos - len(top), 0)
        extra = random.sample(remaining, min(extra_needed, len(remaining)))
        chosen = top + extra
        # Videos that cleared both filters and simply were not drawn. Without
        # this the video side of the ledger cannot close even in principle:
        # 45 survivors with 30 targeted leaves 15 unaccounted for.
        exclusions["surviving_videos_not_drawn"] += len(survivors) - len(chosen)
        print(f"  picked {len(top)} video(s) by relevance and {len(extra)} more at random "
              f"(seed={seed}) for {len(chosen)} of {target_videos} targeted", file=sys.stderr)

    rows = []
    for video_id in chosen:
        url = f"https://www.youtube.com/watch?v={video_id}"
        rows += scrape_comments(youtube, url, story, comment_limit, order, lang,
                                drop_non_lang, exclusions=exclusions, video_log=video_log,
                                search_query=query_of.get(video_id, ""),
                                video_title=title_of.get(video_id, ""))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("urls", nargs="*", help="One or more video URLs (preferred)")
    parser.add_argument("--story", required=True, help="Short story label, e.g. luigi / elon")
    parser.add_argument("--search", action="append", default=[], metavar="QUERY",
                        help="Search term instead of URLs; repeatable. Each query contributes "
                             "candidates to one shared pool, deduplicated by video ID, and the "
                             "hybrid draw happens once over that pool. Repeat it when a story "
                             "is discussed under several names: the Oxfam inequality report is "
                             "covered under 'Davos', \'world's richest 1%%\' and 'wealth gap' "
                             "without ever being named, and a single query found only 5 videos "
                             "clearing --min-comments against 25 for Luigi. Five videos is too "
                             "few for a clustered interval to mean anything, so this is a "
                             "sample-size fix, not a convenience.")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Hybrid video selection: take this many videos by relevance "
                             "first (default 10), then fill up to --target-videos at random. "
                             "See the design-effect note on search_youtube() for why this "
                             "isn't pure relevance or pure random.")
    parser.add_argument("--target-videos", type=int, default=30,
                        help="Total videos to select per story when using --search (default "
                             "30): --top-n by relevance, the rest at random. Precision comes "
                             "from the number of videos collected, not the number of "
                             "comments, so this -- not a comment-count target -- is the lever "
                             "that actually narrows a reported proportion's interval.")
    parser.add_argument("--drop-non-lang", action="store_true",
                        help="Actually discard comments whose detected language differs from "
                             "--lang, instead of only recording it. Off by default and worth "
                             "leaving off: measured over 313 comments this removed 8 and all 8 "
                             "were English (see #23). The detected_lang column is written "
                             "either way, so filtering downstream stays reversible.")
    parser.add_argument("--max-shorts-seconds", type=int, default=60,
                        help="Exclude --search results at or under this many seconds as YouTube "
                             "Shorts (0 = keep everything). The Data API exposes no "
                             "authoritative is-a-Short flag, so this is a duration heuristic. "
                             "Default 60 is deliberately conservative; 180 is the current "
                             "maximum Shorts length and excludes far more, including short "
                             "clips from credible channels. Measured on 50 results for "
                             "'billionaire wealth tax': 12 videos at or under 60s, 27 at or "
                             "under 180s. Record whichever value was used.")
    parser.add_argument("--min-comments", type=int, default=0,
                        help="Only consider search results with MORE than this many comments "
                             "(soft-coded: raise it to demand a bigger comment section before "
                             "a video is eligible to be sampled).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for --search video sampling, so the same search "
                             "reproduces the same picked videos.")
    parser.add_argument("--limit", type=int, default=2000,
                        help="Max top-level comments per video (0 = all). Replies do not "
                             "count against this.")
    parser.add_argument("--max-search-pages", type=int, default=1,
                        help="How many 50-result pages of search results to walk per "
                             "--search query (default 1). One page used to be a hard "
                             "limit, so the candidate pool was the top 50 by relevance "
                             "and any shortfall was reported as if it were a property "
                             "of the corpus. Raising this widens the pool, but each "
                             "page costs 100 quota units, and because --seed draws "
                             "from the pool, a different value produces a DIFFERENT "
                             "video set for the same seed. Do not change it midway "
                             "through a collection you intend to reproduce.")
    parser.add_argument("--order", default="relevance", choices=["relevance", "time"],
                        help="Comment order within each video. Only matters if --limit "
                             "actually cuts a video's comments short: with every comment "
                             "collected (--limit 0), the final rows are the same either way. "
                             "'time' is the unbiased ordering if a cap does bite; 'relevance' "
                             "is YouTube's engagement ranking.")
    parser.add_argument("--lang", help="Keep only comments detected as this ISO 639-1 "
                                        "language code, e.g. 'en'. Requires `pip install "
                                        "langdetect`. Unset (default): no filtering. "
                                        "Comments under ~20 characters are always kept, "
                                        "since a detector cannot trust that little text.")
    parser.add_argument("--output", help="CSV file to write to. If omitted, writes to stdout "
                                          "(NOT recommended on Windows PowerShell — its '>' "
                                          "redirection re-encodes text as UTF-16, not UTF-8).")
    parser.add_argument("--video-list-output", help="Write the video-list appendix (video ID, "
                                                    "comments kept, dates, search query) to "
                                                    "this CSV -- the reproducibility record of "
                                                    "exactly which videos went into the corpus.")
    parser.add_argument("--exclusion-table-output", help="Write the exclusion-accounting table "
                                                         "(API-returned vs. removed at each "
                                                         "filtering stage) to this CSV, "
                                                         "alongside printing it to stderr.")
    args = parser.parse_args()

    if not args.urls and not args.search:
        parser.error("give at least one video URL, or --search")

    youtube = get_client()

    exclusions = collections.Counter()
    video_log = []

    comments = []
    for url in args.urls:
        comments += scrape_comments(youtube, url, args.story, args.limit, args.order,
                                    args.lang, args.drop_non_lang, exclusions=exclusions,
                                    video_log=video_log, search_query="(direct URL)")
    if args.search:
        comments += search_youtube(youtube, args.search, args.story, args.top_n,
                                   args.target_videos, args.limit, args.min_comments,
                                   args.seed, args.order, args.lang, args.max_shorts_seconds,
                                   args.drop_non_lang, exclusions=exclusions,
                                   video_log=video_log,
                                   max_search_pages=args.max_search_pages)

    import csv
    # Writing the file directly here, with an explicit UTF-8 encoding,
    # sidesteps PowerShell's '>' redirection entirely (which silently
    # re-encodes captured stdout as UTF-16 in Windows PowerShell 5.1).
    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(comments)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(comments)
    print(f"Total: {len(comments)} comments", file=sys.stderr)

    # Exclusion accounting: captures counts already tallied above during
    # collection, rather than recomputing anything -- a reader otherwise
    # cannot tell whether the corpus is what was collected or a filtered
    # remnant, and this pre-empts the obvious "did the filters do the work"
    # question. `api_returned` is every top-level-or-reply item examined;
    # everything else is a reason a comment was not kept.
    # Two units live in this ledger and must not be added together: some
    # stages count comments, others count whole videos. A single `count`
    # column silently invited a reader to sum them. The `kept` rows are what
    # let the arithmetic actually close -- without them nothing in the table
    # says how many comments survived, which is the one number the table
    # exists to justify.
    COMMENT_STAGES = {"api_returned", "deleted_removed_or_automod", "non_english",
                      "excluded_by_comment_limit"}
    VIDEO_STAGES = {"excluded_as_shorts", "excluded_below_min_comments",
                    "videos_truncated_by_limit", "surviving_videos_not_drawn"}
    exclusion_rows = [{"stage": "kept", "unit": "comments", "count": len(comments)}]
    exclusion_rows += [{"stage": stage,
                        "unit": "comments" if stage in COMMENT_STAGES
                                else "videos" if stage in VIDEO_STAGES else "unknown",
                        "count": count}
                       for stage, count in exclusions.most_common()]
    if video_log:
        exclusion_rows.append({"stage": "kept", "unit": "videos", "count": len(video_log)})

    print("\nExclusion accounting:", file=sys.stderr)
    print(f"  {'stage':<30}{'unit':<10}{'count':>8}", file=sys.stderr)
    for row in exclusion_rows:
        print(f"  {row['stage']:<30}{row['unit']:<10}{row['count']:>8}", file=sys.stderr)
    # State the identity explicitly so a reader can check it rather than
    # reverse-engineer which stages belong on which side.
    examined = exclusions.get("api_returned", 0)
    accounted = (len(comments) + exclusions.get("deleted_removed_or_automod", 0)
                 + exclusions.get("non_english", 0))
    print(f"  comments: kept + deleted + non_english = {accounted} "
          f"vs api_returned {examined}"
          f"{'  OK' if accounted == examined else '  MISMATCH'}", file=sys.stderr)
    if exclusions.get("excluded_by_comment_limit"):
        print(f"  ({exclusions['excluded_by_comment_limit']} further comments were "
              f"returned by the API but never examined, having fallen past --limit)",
              file=sys.stderr)
    if args.exclusion_table_output:
        out_dir = os.path.dirname(args.exclusion_table_output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.exclusion_table_output, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["stage", "unit", "count"])
            writer.writeheader()
            writer.writerows(exclusion_rows)

    # Video-list appendix: exactly which videos went into the corpus, for
    # reproducibility -- video ID, title, comments kept, when collected, and
    # the search query that found it ("(direct URL)" for a video named on the
    # command line rather than found via --search). The title costs no extra
    # quota (search().list's snippet already carries it) and is what makes
    # this file actually checkable by a human without opening thirty videos.
    if args.video_list_output:
        out_dir = os.path.dirname(args.video_list_output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.video_list_output, "w", newline="", encoding="utf-8-sig") as f:
            fieldnames = ["video_id", "story", "video_title", "search_query",
                         "comments_kept", "collected_at"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(video_log)
        print(f"Video list ({len(video_log)} video(s)) written to {args.video_list_output}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
