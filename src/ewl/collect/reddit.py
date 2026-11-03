"""Collect the full comment tree of one Reddit thread via the official Reddit API.

    export REDDIT_CLIENT_ID=... REDDIT_CLIENT_SECRET=...
    python reddit_scrape.py <thread-url> --story luigi > luigi_reddit.csv

Why this replaces the old.reddit.com scraper
--------------------------------------------
The previous version fetched `old.reddit.com/<thread>/.json` with a spoofed
Chrome user-agent and TLS verification disabled. Checked on 2026-08-20, that
route no longer returns data at all: old.reddit.com redirects anonymous .json
requests to a login wall (`?reason=lor2`) and www.reddit.com answers 403. Both
behaviours are user-agent independent, so there is no header to tweak. The
choice in the methodology note between "Method One: scraping" and "Method Two:
official API" is therefore already settled — Method One is not a slower or
riskier option, it is a dead one. Using the API also removes the terms-of-service
exposure the note flags, and removes the need to pretend to be a browser.

Registering an app takes a few minutes at https://www.reddit.com/prefs/apps
(choose "script"); no approval queue is involved for ordinary read-only use.

Other fixes carried over
------------------------
* Collapsed branches ("load more comments") are now followed. The old code
  silently skipped Reddit's `more` stubs, which is why every large thread in the
  pilot corpus stopped near 190-200 comments — those totals measured the page
  cap, not the volume of discussion.
* Requests are throttled and retried, so a rate limit cannot truncate a run into
  a partial file that looks complete.
* Usernames are hashed at the point of collection and deleted/removed comments
  are dropped here, rather than being cleaned by hand afterwards.
"""
import argparse
import csv
import os
import re
import sys
import time

import requests

from ewl.common import FIELDNAMES, hash_author, is_usable, now_iso, utc_iso

USER_AGENT = "script:ewl-research:1.0 (Extreme Wealth Line project; public comment collection)"
API = "https://oauth.reddit.com"
REQUEST_DELAY = 1.1  # Reddit allows 100 requests/min for OAuth clients
MAX_RETRIES = 4

THREAD_ID_RE = re.compile(r"/comments/([a-z0-9]+)", re.I)

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


def authenticate():
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    if not client_id or not client_secret:
        sys.exit(
            "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set.\n"
            "Register a 'script' app at https://www.reddit.com/prefs/apps and export both."
        )
    response = session.post(
        "https://www.reddit.com/api/v1/access_token",
        auth=(client_id, client_secret),
        data={"grant_type": "client_credentials"},
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()
    session.headers["Authorization"] = f"bearer {response.json()['access_token']}"


def api_get(path, params=None):
    """GET with backoff. Raises after MAX_RETRIES so a partial run fails loudly
    instead of writing a short file that reads as a complete one."""
    for attempt in range(MAX_RETRIES):
        time.sleep(REQUEST_DELAY)
        response = session.get(API + path, params=params, timeout=30)
        if response.status_code == 429 or response.status_code >= 500:
            wait = REQUEST_DELAY * (2 ** attempt)
            print(f"  {response.status_code} from Reddit, retrying in {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"Gave up on {path} after {MAX_RETRIES} attempts")


def parse_thing(child, comments, more_ids, meta):
    """Walk one node of the comment tree, recording comments and 'more' stubs."""
    kind = child.get("kind")
    data = child.get("data", {})

    if kind == "more":
        # Reddit hands back a stub listing the ids it did not inline. Following
        # these is the difference between a capped page and the whole thread.
        more_ids.extend(data.get("children", []))
        return

    if kind != "t1":  # t1 = comment; t3 = the post itself, which we do not want
        return

    if is_usable(data.get("body"), data.get("author")):
        comments.append({
            "comment_id": data.get("id", ""),
            "platform": "reddit",
            "story": meta["story"],
            "source": meta["source"],
            "author_hash": hash_author(data.get("author")),
            "body": data.get("body"),
            "upvotes": data.get("ups"),
            "timestamp": utc_iso(data.get("created_utc")),
            "timestamp_precision": "second",
            "depth": data.get("depth"),
            "collected_at": meta["collected_at"],
        })

    replies = data.get("replies")
    if isinstance(replies, dict):
        for grandchild in replies.get("data", {}).get("children", []):
            parse_thing(grandchild, comments, more_ids, meta)


def scrape_thread(url, story):
    match = THREAD_ID_RE.search(url)
    if not match:
        sys.exit(f"Could not find a thread id in {url}")
    thread_id = match.group(1)
    meta = {"story": story, "source": url.split("?")[0].rstrip("/"), "collected_at": now_iso()}

    print(f"Fetching thread {thread_id}", file=sys.stderr)
    # raw_json=1 stops Reddit HTML-escaping &, < and > inside comment bodies,
    # which otherwise leaks &amp; into the text VADER scores.
    listings = api_get(f"/comments/{thread_id}", params={"limit": 500, "raw_json": 1, "depth": 100})

    comments, more_ids = [], []
    for listing in listings:
        for child in listing.get("data", {}).get("children", []):
            parse_thing(child, comments, more_ids, meta)

    # Drain the "load more comments" stubs. morechildren returns a flat list
    # which can itself contain further stubs, so loop until nothing is left.
    seen = set()
    while more_ids:
        batch = [i for i in more_ids[:100] if i not in seen]
        more_ids = more_ids[100:]
        if not batch:
            continue
        seen.update(batch)
        print(f"  expanding {len(batch)} collapsed branches ({len(comments)} so far)", file=sys.stderr)
        payload = api_get("/api/morechildren", params={
            "api_type": "json", "link_id": f"t3_{thread_id}",
            "children": ",".join(batch), "raw_json": 1,
        })
        for child in payload.get("json", {}).get("data", {}).get("things", []):
            parse_thing(child, comments, more_ids, meta)

    print(f"Collected {len(comments)} usable comments from {meta['source']}", file=sys.stderr)
    return comments


def main():
    parser = argparse.ArgumentParser(description="Collect one Reddit thread via the official API.")
    parser.add_argument("url", help="Full URL of a single Reddit thread")
    parser.add_argument("--story", required=True,
                        help="Short story label, e.g. luigi / elon / oxfam / private_jet")
    args = parser.parse_args()

    authenticate()
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(scrape_thread(args.url, args.story))


if __name__ == "__main__":
    main()
