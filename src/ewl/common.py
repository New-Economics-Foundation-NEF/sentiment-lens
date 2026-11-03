"""Shared helpers for the EWL collectors and classifier.

Both collectors write the SAME columns in the SAME order, so the per-story
CSVs can be concatenated into a master file without hand-editing in Excel.
"""
import hashlib #This is the package used for hashing out the usernames.
import os #This is the package used to interact with the operating system.
from datetime import datetime, timezone #Used for creating a common date and time format later

# Read every CSV with encoding="utf-8-sig", never plain "utf-8".
#
# The collectors write with "utf-8-sig" on purpose: PowerShell's `>` redirection
# re-encodes captured stdout as UTF-16, and Excel garbles emoji without a BOM,
# so the BOM is a deliberate Windows fix and should stay. But a reader opening
# that file as plain "utf-8" keeps the BOM as part of the FIRST column name,
# which silently becomes "\ufeffcomment_id". Nothing raises. Anything reading
# `body` still works, so a run looks completely normal — while every lookup of
# `comment_id` quietly returns None.
#
# That is not hypothetical: it silently disabled classify.py's pool-overlap
# guard, which compares comment_ids to catch a corpus being labelled against a
# retrieval pool it belongs to. The guard found no overlap because it could not
# see the column, so the check that exists to catch contaminated scoring
# reported all-clear on exactly the files the collectors produce.
#
# "utf-8-sig" on read strips a BOM when present and is identical to "utf-8"
# when absent, so it is always the correct choice for reading here.

# One schema for both platforms. `depth` is blank for YouTube (no nesting
# information is available), `platform`/`story`/`collected_at` make each row
# self-describing once the files are merged.
FIELDNAMES = [
    "comment_id",
    "platform",
    "story",
    "source",
    "author_hash",
    "body",
    "upvotes",
    "timestamp",
    "timestamp_precision",
    "depth",
    "collected_at",
    "detected_lang",
]

#Creates a list of column titles for the spreadsheet later. Works for both social media platforms.

# Salt for pseudonymising usernames. Usernames are personal data under UK GDPR,
# so the raw handle never reaches disk. Set EWL_AUTHOR_SALT to a project secret
# and record it outside the repo. A collector must not fall back to a
# hardcoded default: that default would sit in the public repo, making every
# hash trivially reversible by dictionary attack — the opposite of the point.
_SALT = os.environ.get("EWL_AUTHOR_SALT")#This is searching a dictionary called environ. Environ is a dictionary that includes all environment variables held in the terminal and JUST the terminal.
#The first thing in the bracket is the key to look for that dictionary. There used to be a second argument here as a fallback default if the key was missing -
#that has been removed on purpose: see the check in hash_author() below. A default salt baked into the repo is not really a secret at all.
#Hashing 101: A hash is one way- there is no method to reverse a hash whatsoever. But if the content you are hashing is common e.g.passwords, then someone doesn't need to reverse engineer the hashing, they can simply guess.
#The salt is a variable that we set in the terminal. It is appeneded to all usernames before they are hashed. The way hashing works means that this modifier (sometimes called secret ingredient) cannot be guessed.
#But that only holds if the salt itself is actually secret. A salt with a hardcoded fallback value that ships in the public repo is not secret to anyone who can read the repo, so anyone could pre-compute the hash for any guessed username and match it straight back - the exact dictionary-guessing attack the salt exists to prevent.
#
# The check itself lives in hash_author(), not here at module load. Raising here
# would mean `import ewl_common` fails for anything that only wants FIELDNAMES —
# which is exactly what pushed merge.py into keeping its own hand-duplicated
# copy of the schema instead of importing the real one. Checking at first *use*
# gives the same guarantee (no username is ever hashed without a real salt)
# without that collateral damage.

DELETED_MARKERS = {"[deleted]", "[removed]", "[deleted by user]", ""}
#This is a set of strings that detail out the different ways YT and Reddit refer to deleted usernames.



def hash_author(author): #Function which requires the input of author name.
    """Return a stable pseudonym for a username, never the username itself."""
    if not _SALT:
        raise SystemExit(
            "EWL_AUTHOR_SALT is not set. Set it to a project secret before collecting: "
            "usernames are personal data under UK GDPR and must never be hashed with a "
            "salt that ships in the repo. Keep it outside the repo, e.g. in "
            "~/.config/ewl.env (mode 600), and source it before running a collector."
        )
    if len(_SALT) < 16:
        raise SystemExit(
            "EWL_AUTHOR_SALT is too short to resist a dictionary attack; use at least "
            "16 random characters."
        )
    #Refuses to hash anything until the salt actually exists and is long enough, rather than
    #silently collecting data with a missing or guessable salt.
    # Normalisation belongs here, not at the call sites. youtube.py stripped a
    # leading "@" and youtube_api.py did not, so the same YouTube user hashed
    # to two different pseudonyms depending on which collector ran -- and a
    # corpus merged from both counted one person as two, which is exactly the
    # independence assumption the cluster bootstrap and cap_per_author rest on.
    #
    # It has to happen BEFORE the marker test, not after. Normalising second
    # meant "[deleted] " and "@[deleted]" no longer matched DELETED_MARKERS,
    # got normalised to "[deleted]", and hashed -- collapsing every deleted
    # user in the corpus into one synthetic high-volume author, which is the
    # same error this function exists to prevent, only worse.
    author = (author or "").strip().lstrip("@").strip()
    if not author or author in DELETED_MARKERS:
        return ""
    return hashlib.sha256((_SALT + author).encode("utf-8")).hexdigest()[:16]
#If there is no author or the author matches something from the deleted markers set, then return a blank.
#Otherwise combine the author name and the salt, hash them using the rule sha256, encode it to a specific format and slice it to be 16 characters (hexadecimal)
#hashlib.sha256 is the actual hashing function. encode is the function that converts the combined name into bytes for the computer. Hexdigest takes the hashed output which is in bytes and makes it into hexadecimal.
#"" is another way of saying an empty string
def is_usable(body, author=None):
    """Drop deleted/removed comments and automod posts at collection time."""
    if body is None:
        return False
    if body.strip() in DELETED_MARKERS:
        return False
    if author and author.lower() in {"automoderator", "youtube"}: #as the default for the author is none, if there was no author for a specific comment, trying to do the function None.lower() would cause an error.

        return False
    return True
#Consider this to be similar to a test- if there is nothing in the main comment body,the comment is one of the deletd markers, or the author is an automoderator or from youtube, it shows up false.
#In terms of the if statement at the end it is biconditional. Python first checks for if the author string actually exists (left-hand side) if it is none or an empty string, it outputs false.
#The right hand side of the condition puts the string into lower case, and then checks for moderation strings.

def detect_language(body, min_chars=20):
    """Best-guess ISO 639-1 code for a comment, or "" when it cannot be told.

    Recorded rather than acted on. #23 measured the previous drop-on-detect
    behaviour over 313 comments: it removed 8, and **all 8 were English** --
    "TAX WEALTH, NOT WORK!!!" was read as Swahili with probability 1.00, and
    "It just makes absolute sense." as Estonian, also 1.00. Zero true
    positives. Median length of a dropped comment was 36 characters against
    125 for a kept one, so it behaved as a length filter wearing a language
    filter's clothes -- and it removed exactly the short, punchy comments that
    carry the clearest sentiment signal.

    Because the confidences were 1.00 on wrong answers, no threshold rescues
    it. So the detection is written to a column for inspection and the
    filtering decision is left downstream, where it is reversible and its cost
    is visible as a number instead of a silent subtraction.

    "" for a comment too short to judge, for a detection failure, or when
    langdetect is not installed -- three different reasons to have no answer,
    none of which should be mistaken for a confident foreign-language verdict.
    """
    text = (body or "").strip()
    if len(text) < min_chars:
        return ""
    try:
        from langdetect import detect, LangDetectException
        from langdetect.detector_factory import DetectorFactory
        DetectorFactory.seed = 0  # langdetect is non-deterministic without a fixed seed
    except ImportError:
        return ""
    try:
        return detect(text)
    except LangDetectException:
        return ""


def is_language(body, lang, min_chars=20):
    """True if `body` is not confidently detected as some *other* language.

    Kept for callers that still want to filter, but read the warning in
    detect_language() first: on this corpus it produced 8 false positives and
    no true ones. Unknown always passes -- silently dropping a comment the
    detector could not read would look like a language decision when it is
    really "could not tell".
    """
    if not lang:
        return True
    detected = detect_language(body, min_chars)
    return detected == "" or detected == lang


def utc_iso(epoch_seconds):
    if epoch_seconds is None:
        return ""
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()

#Captures the time the comment was posted

def now_iso():
    return datetime.now(tz=timezone.utc).isoformat()
#Captures the time the comment was scraped
