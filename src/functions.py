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
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"

RANDOM_STATE = 42
LLM_MODEL = "gpt-5.4-nano"                                    # LLM track: one place to change the model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
STEAM_REVIEWS_URL = "https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/"


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

def load_reviews():
    """Read the raw reviews (and, for Amazon, join the product metadata you need)."""
    pass


def clean_text(text):
    """Remove HTML and extra whitespace. The app calls this too, on the user's input."""
    pass


# Model track

def make_target(ratings):
    """Turn star ratings into your target, with the rule stated in notebook 01."""
    pass


def compare_models(models, X, y, cv, scoring):
    """Cross-validate each approach on the same folds and return a tidy table with the training time."""
    pass


# LLM track

def build_index(documents):
    """Embed the chunks and store them, with their metadata, in a Chroma collection."""
    pass


def answer_question(question, collection, k=5):
    """Retrieve, prompt and answer; return the answer, the cited ids, the retrieved ids and the tokens."""
    pass


def evaluate(questions, answer_fn):
    """Run the evaluation set and return one row per question: hit, cites, refused, cost."""
    pass
