"""Your reusable functions live here.

The rule from the brief: Python logic goes in `.py` files, and the notebooks
hold the narrative. The app imports the same functions, so the text it receives
goes through exactly the same cleaning as your training data.

To use this module from a notebook in `notebooks/`:

    import sys
    sys.path.append("..")
    from src.functions import *

and from the app in `app/`, the same two lines with `sys.path.append(".")` when
you run `streamlit run app/streamlit_app.py` from the repo root.

The stubs below only show the shape. Rename them, change the arguments, delete
the ones for the other track, write your own.
"""

from pathlib import Path
import json
import re
import time

import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0      # langdetect is random by default: a fixed seed makes results reproducible

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"

RANDOM_STATE = 42
LLM_MODEL = "gpt-5.4-nano"                                    # LLM track: one place to change the model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
STEAM_REVIEWS_URL = "https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/"
PRICE_PER_M_INPUT = 0.20     # USD per 1M input tokens, gpt-5.4-nano (OpenAI pricing, checked 2026-11-10)
PRICE_PER_M_OUTPUT = 1.25    # USD per 1M output tokens; reasoning tokens are billed as output

BASELINE_PROMPT = (
    "You help PC players decide whether to buy a game by telling them how players have received it. "
    "Answer from what you know about players' opinions of the game, in 3 to 5 sentences. "
    "If you do not have reliable information to answer, say that you don't know instead of guessing."
)


# Data collection

def fetch_reviews(app_id, max_reviews=1000, sort=1, extra_filters=None, pause=1.0):
    """Download up to `max_reviews` English reviews of one Steam game, page by page.

    sort: 0 = Helpful, 1 = Recent, 2 = Updated (Valve's enum values).
    extra_filters: optional dict of extra API parameters, e.g. a date range.
    """
    collected = []
    cursor = "*"
    seen_cursors = {cursor}
    retries = 0

    while len(collected) < max_reviews:
        query = {
            "appid": app_id,
            "filter": sort,
            "languages": ["english"],
            "num_per_page": 100,
            "cursor": cursor,
            "purchase_type": 1,
        }
        if extra_filters:
            query.update(extra_filters)

        response = requests.get(STEAM_REVIEWS_URL,
                                params={"input_json": json.dumps(query)},
                                timeout=30)

        if response.status_code == 429 and retries < 3:   # rate-limited: wait and retry
            retries += 1
            print("Rate limited, waiting 60 s ...")
            time.sleep(60)
            continue
        response.raise_for_status()                         # stop on any other error

        data = response.json()
        payload = data.get("response", data)
        page = payload.get("reviews", [])
        if not page:                                        # no more reviews
            break
        collected.extend(page)

        cursor = payload.get("cursor")
        if not cursor or cursor in seen_cursors:            # Steam returned the same page again
            break
        seen_cursors.add(cursor)
        time.sleep(pause)                                   # be polite between requests

    return collected[:max_reviews]

def count_reviews(app_id, start, end, review_type):
    """Number of English reviews of one game created between two Unix timestamps.
    review_type: 1 = positive only, 2 = negative only."""
    query = {
        "appid": app_id,
        "filter": 1,
        "languages": ["english"],
        "num_per_page": 1,              # we only need the count, not the reviews
        "purchase_type": 1,
        "review_type": review_type,
        "date_range_start": start,
        "date_range_end": end,
    }
    response = requests.get(STEAM_REVIEWS_URL,
                            params={"input_json": json.dumps(query)},
                            timeout=30)
    response.raise_for_status()
    data = response.json()
    payload = data.get("response", data)
    return payload.get("total_matching")

def reviews_to_dataframe(raw_reviews, app_id, game):
    """Flatten the raw JSON reviews into one row per review. The SteamID is never copied."""
    rows = []
    for r in raw_reviews:
        author = r.get("author", {})
        hardware = r.get("hardware", {})
        rows.append({
            "review_id": r["recommendationid"],
            "app_id": app_id,
            "game": game,
            "text": r.get("review", ""),
            "voted_up": r.get("voted_up"),
            "created": pd.to_datetime(r.get("timestamp_created"), unit="s"),
            "updated": pd.to_datetime(r.get("timestamp_updated"), unit="s"),
            "playtime_at_review_h": author.get("playtime_at_review", 0) / 60,
            "playtime_forever_h": author.get("playtime_forever", 0) / 60,
            "votes_up": r.get("votes_up", 0),
            "votes_funny": r.get("votes_funny", 0),
            "steam_deck": r.get("primarily_steam_deck", False),
            "received_for_free": r.get("received_for_free", False),
            "refunded": r.get("refunded", False),
            "gpu": hardware.get("adapter_description"),
            "developer_response": r.get("developer_response"),
        })
    return pd.DataFrame(rows)

def collect_game(app_id, game, steam_release, end_date,
                 n_windows=5, per_window=200, pause=1.0):
    """Stratified sample: for each equal time window, the most helpful positive and
    negative reviews, in the window's real positive/negative proportions."""
    edges = pd.date_range(start=steam_release, end=end_date, periods=n_windows + 1, tz="UTC")
    frames = []
    for i in range(n_windows):
        start = int(edges[i].timestamp())
        end = int(edges[i + 1].timestamp()) - 1

        # 1. How many positive and negative reviews really exist in this window?
        pos_real = count_reviews(app_id, start, end, review_type=1) or 0
        neg_real = count_reviews(app_id, start, end, review_type=2) or 0
        total_real = pos_real + neg_real
        if total_real == 0:
            print(f"  window {i + 1}: no reviews, check the release date")
            continue
        real_share = pos_real / total_real

        # 2. Split the window's quota in the same proportions (never more than exist)
        n_pos = min(round(per_window * real_share), pos_real)
        n_neg = min(per_window - n_pos, neg_real)

        # 3. Fetch the most helpful reviews of each sentiment
        parts = []
        for review_type, n in [(1, n_pos), (2, n_neg)]:
            if n == 0:
                continue
            raw = fetch_reviews(
                app_id, max_reviews=n, sort=0, pause=pause,
                extra_filters={"date_range_start": start, "date_range_end": end,
                               "day_range": 0, "review_type": review_type},
            )
            parts.append(reviews_to_dataframe(raw, app_id, game))

        df = pd.concat(parts, ignore_index=True)
        df["window"] = i + 1
        df["window_reviews_total"] = total_real
        df["window_real_share"] = round(real_share, 3)
        frames.append(df)
        print(f"  window {i + 1}: {edges[i].date()} → {edges[i + 1].date()}: "
              f"{len(df)} reviews ({n_pos} positive, {n_neg} negative), "
              f"real share {real_share:.3f}")

    return pd.concat(frames, ignore_index=True).drop_duplicates("review_id")

def collect_all(games_csv, end_date, out_dir, **kwargs):
    """Collect every game in `games_csv`, saving one parquet file per game.
    Games already saved are skipped, so a failed run can simply be restarted."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    games = pd.read_csv(games_csv)

    for row in games.itertuples():
        path = out_dir / f"{row.app_id}.parquet"
        if path.exists():
            print(f"{row.game}: already collected, skipping")
            continue
        print(row.game)
        df = collect_game(row.app_id, row.game, row.steam_release, end_date, **kwargs)
        df["collected_on"] = end_date
        df.to_parquet(path, index=False)

    return pd.concat([pd.read_parquet(p) for p in sorted(out_dir.glob("*.parquet"))],
                     ignore_index=True)
# Both tracks

def load_reviews(folder=RAW / "steam_reviews"):
    """Read the review snapshot: one parquet file per game, written by collect_all()."""
    files = sorted(Path(folder).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No review files in {folder}. Run the collection first (see README).")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


BBCODE_TAG = re.compile(
    r"\[/?(?:h[1-3]|b|i|u|s|strike|spoiler|noparse|hr|list|olist|quote|code|table|tr|th|td|url)"
    r"(?:=[^\]]*)?\]",
    re.IGNORECASE,
)
LIST_ITEM = re.compile(r"\[\*\]")
URL = re.compile(r"https?://\S+")
QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def clean_text(text):
    """Remove Steam markup and links, normalize quotes and whitespace, keep paragraph breaks.
    The app calls this too, on the user's question."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n")
    text = LIST_ITEM.sub("- ", text)              # [*] list items become "- "
    text = BBCODE_TAG.sub("", text)               # drop tags, keep the text inside them
    text = URL.sub("", text)
    text = text.translate(QUOTES)                 # curly quotes become straight ones
    text = re.sub(r"([^\w\s])\1{3,}", r"\1\1\1", text)   # "......." → "...", "!!!!!!" → "!!!"
    text = re.sub(r"[ \t]+", " ", text)           # runs of spaces become one space
    text = re.sub(r" *\n *", "\n", text)          # no spaces around line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)        # at most one empty line between paragraphs
    return text.strip()

def prepare_reviews(reviews, min_words=5, min_letter_share=0.5, screen_english_share=0.3):
    """Clean every text and drop the rows the assistant cannot use.
    Returns the clean table and a log of how many rows each rule removed."""
    df = reviews.copy()
    df["text"] = df["text"].map(clean_text)
    df["n_words"] = df["text"].str.split().str.len()
    df["letter_share"] = df["text"].str.count(r"[A-Za-z]") / df["text"].str.len().replace(0, 1)
    words = df["text"].str.lower().str.findall(r"[a-z']+")
    df["english_share"] = words.map(lambda ws: sum(w in ENGLISH_STOP_WORDS for w in ws) / max(len(ws), 1))
    df["edited"] = (df["updated"] - df["created"]) > pd.Timedelta(days=1)

    log = []

    mask = df["letter_share"] < min_letter_share
    log.append(("Mostly non-Latin characters (other language, emoji, ASCII art)", int(mask.sum())))
    df = df[~mask]

    mask = df["n_words"] < min_words
    log.append((f"Fewer than {min_words} words", int(mask.sum())))
    df = df[~mask]

    suspects = (df["n_words"] >= 20) & (df["english_share"] < screen_english_share)
    df["language"] = "en"
    df.loc[suspects, "language"] = df.loc[suspects, "text"].map(detect_language)
    mask = df["language"] != "en"
    log.append(("Not English (language detected on suspects with few English function words)", int(mask.sum())))
    df = df[~mask]

    mask = df["text"].str.lower().duplicated()
    log.append(("Exact duplicate text (first copy kept)", int(mask.sum())))
    df = df[~mask]

    return df.reset_index(drop=True), pd.DataFrame(log, columns=["reason", "rows_removed"])

def detect_language(text):
    """Language code of the text's main language ("en", "es", "pt"…), or "unknown"."""
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def show_reviews(rows, n=3, chars=800):
    """Print the first `n` reviews of a table in full (up to `chars` characters), with their context."""
    for r in rows.head(n).itertuples():
        verdict = "👍" if r.voted_up else "👎"
        print(f"[{r.game}] {verdict} | {r.playtime_at_review_h:.0f} h | "
              f"{r.created:%Y-%m-%d} | window {r.window} | id {r.review_id}")
        print(r.text[:chars])
        print("-" * 80)

def find_sources(chunks, pattern, game=None, min_playtime=None, steam_deck=None, n=5):
    """Chunks matching a regex, optionally filtered as a user would. Prints the count and n random hits."""
    rows = chunks
    if pd.notna(game):
        rows = rows[rows["game"] == game]
    if pd.notna(min_playtime):
        rows = rows[rows["playtime_at_review_h"] >= min_playtime]
    if pd.notna(steam_deck):
        rows = rows[rows["steam_deck"] == steam_deck]
    hits = rows[rows["text"].str.contains(pattern, case=False, regex=True)]
    print(f"{len(hits)} matching chunks from {hits['review_id'].nunique()} reviews\n")
    for r in hits.sample(min(n, len(hits)), random_state=RANDOM_STATE).itertuples():
        verdict = "👍" if r.voted_up else "👎"
        print(f"[{r.game}] {verdict} | {r.playtime_at_review_h:.0f} h | {r.created:%Y-%m-%d} | "
              f"window {r.window} | review {r.review_id}")
        print(r.text)
        print("-" * 80)
    return hits

# Model track

def make_target(ratings):
    """Turn star ratings into your target, with the rule stated in notebook 01."""
    pass


def compare_models(models, X, y, cv, scoring):
    """Cross-validate each approach on the same folds and return a tidy table with the training time."""
    pass


# LLM track
def get_client():
    """OpenAI client, with the API key read from the .env file at the repo root."""
    load_dotenv(ROOT / ".env")
    return OpenAI()


def ask_llm(client, system_prompt, user_message, model=LLM_MODEL):
    """One call to the LLM. Returns the answer with its tokens, cost and time."""
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_message}],
    )
    usage = response.usage
    return {
        "answer": response.choices[0].message.content,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "cost_usd": (usage.prompt_tokens * PRICE_PER_M_INPUT
                     + usage.completion_tokens * PRICE_PER_M_OUTPUT) / 1_000_000,
        "seconds": round(time.perf_counter() - start, 2),
    }
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def n_tokens(text, tokenizer):
    """Number of tokens the embedding model sees in `text` (without its special tokens)."""
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def split_long_paragraph(paragraph, tokenizer, max_tokens):
    """Split a paragraph longer than `max_tokens` into sentences, then into word groups if needed."""
    pieces, current = [], ""
    for sentence in SENTENCE_END.split(paragraph):
        candidate = f"{current} {sentence}".strip()
        if n_tokens(candidate, tokenizer) <= max_tokens:
            current = candidate
            continue
        if current:
            pieces.append(current)
        if n_tokens(sentence, tokenizer) <= max_tokens:
            current = sentence
        else:                                    # one giant sentence: cut it into word groups
            group, group_tokens = [], 0
            for word in sentence.split():
                word_tokens = n_tokens(word, tokenizer)
                if group and group_tokens + word_tokens > max_tokens:
                    pieces.append(" ".join(group))
                    group, group_tokens = [], 0
                group.append(word)
                group_tokens += word_tokens
            if group:
                pieces.append(" ".join(group))
            current = ""
    if current:
        pieces.append(current)
    return pieces


def chunk_review(text, tokenizer, max_tokens=200):
    """Split one review into chunks of at most `max_tokens`, following its paragraphs."""
    if n_tokens(text, tokenizer) <= max_tokens:
        return [text]                            # most reviews: a single chunk
    pieces = []
    for paragraph in (p.strip() for p in text.split("\n")):
        if not paragraph:
            continue
        if n_tokens(paragraph, tokenizer) <= max_tokens:
            pieces.append(paragraph)
        else:
            pieces.extend(split_long_paragraph(paragraph, tokenizer, max_tokens))

    chunks, current = [], ""
    for piece in pieces:                         # pack consecutive pieces up to the limit
        candidate = f"{current}\n{piece}".strip()
        if n_tokens(candidate, tokenizer) <= max_tokens:
            current = candidate
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def make_chunks(reviews, tokenizer, max_tokens=200):
    """One row per chunk, with the metadata of its review."""
    rows = []
    for r in reviews.itertuples():
        pieces = chunk_review(r.text, tokenizer, max_tokens)
        for i, piece in enumerate(pieces):
            rows.append({
                "chunk_id": f"{r.review_id}-{i}",
                "review_id": r.review_id,
                "chunk_index": i,
                "n_chunks": len(pieces),
                "text": piece,
                "game": r.game,
                "app_id": r.app_id,
                "voted_up": r.voted_up,
                "created": r.created,
                "window": r.window,
                "playtime_at_review_h": r.playtime_at_review_h,
                "edited": r.edited,
                "votes_up": r.votes_up,
                "votes_funny": r.votes_funny,
                "steam_deck": r.steam_deck,
                "gpu": r.gpu,
            })
    return pd.DataFrame(rows)

def build_index(documents):
    """Embed the chunks and store them, with their metadata, in a Chroma collection."""
    pass


def answer_question(question, collection, k=5):
    """Retrieve, prompt and answer; return the answer, the cited ids, the retrieved ids and the tokens."""
    pass


def evaluate(questions, answer_fn):
    """Run the evaluation set and return one row per question: hit, cites, refused, cost."""
    pass
