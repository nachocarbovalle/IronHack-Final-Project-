"""Download the raw data for Project 3 into data/raw/.

Usage
-----
    python download_data.py                                  # Amazon reviews, Health_and_Personal_Care
    python download_data.py amazon --category All_Beauty     # Amazon reviews, another category
    python download_data.py amazon --list                    # the categories you can choose
    python download_data.py clothing                         # Women's clothing e-commerce reviews (CC0)

Nothing in data/ is committed, so this script is how anyone reproduces your
project: they clone the repo, run it, and have the same files you had.

See data/README.md for what each dataset contains, its traps, and how to cite it.
"""

import argparse
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"

AMAZON = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw"
CLOTHING_URL = (
    "https://huggingface.co/datasets/chibifire/kaggle-womens-ecom-clothing-reviews/"
    "resolve/main/data/train-00000-of-00001.parquet"
)

# Categories small enough for a laptop: name -> (reviews, size of reviews + metadata files)
CATEGORIES = {
    "Subscription_Boxes": ("16,000 reviews", "10 MB"),
    "Magazine_Subscriptions": ("71,000 reviews", "37 MB"),
    "Gift_Cards": ("152,000 reviews", "52 MB"),
    "Digital_Music": ("130,000 reviews", "146 MB"),
    "Health_and_Personal_Care": ("494,000 reviews", "345 MB"),
    "Handmade_Products": ("664,000 reviews", "690 MB"),
    "All_Beauty": ("702,000 reviews", "540 MB"),
}
DEFAULT_CATEGORY = "Health_and_Personal_Care"

HEADERS = {"User-Agent": "Mozilla/5.0 (Ironhack DSML Project 3)"}


def fetch(url: str, dest: Path) -> None:
    """Stream `url` to `dest` in chunks, skipping the download if the file is already there."""
    if dest.exists():
        print(f"  {dest.name} already downloaded, skipping")
        return
    print(f"  downloading {dest.name} ...", end="", flush=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request) as response, open(partial, "wb") as file:
        while chunk := response.read(1 << 20):
            file.write(chunk)
    partial.rename(dest)
    print(f" {dest.stat().st_size / 1e6:.0f} MB")


def download_amazon(category: str) -> None:
    """One category of Amazon Reviews 2023: the reviews and the product metadata."""
    print(f"Amazon Reviews 2023, category {category}")
    fetch(f"{AMAZON}/review_categories/{category}.jsonl", RAW / f"{category}.jsonl")
    fetch(f"{AMAZON}/meta_categories/meta_{category}.jsonl", RAW / f"meta_{category}.jsonl")


def download_clothing() -> None:
    """Women's e-commerce clothing reviews (CC0), one parquet file."""
    print("Women's e-commerce clothing reviews")
    fetch(CLOTHING_URL, RAW / "womens_clothing_reviews.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", nargs="?", default="amazon", choices=["amazon", "clothing"])
    parser.add_argument("--category", default=DEFAULT_CATEGORY,
                        help=f"Amazon category (default: {DEFAULT_CATEGORY}); see --list")
    parser.add_argument("--list", action="store_true", help="list the Amazon categories and exit")
    args = parser.parse_args()

    if args.list:
        for name, (reviews, size) in CATEGORIES.items():
            print(f"  {name:26s} {reviews:>17s}   {size}")
        print("\nAny other category of https://amazon-reviews-2023.github.io works too, but most are several GB.")
        return

    RAW.mkdir(parents=True, exist_ok=True)
    if args.dataset == "amazon":
        download_amazon(args.category)
    else:
        download_clothing()
    print(f"\nDone. Files are in {RAW}/")


if __name__ == "__main__":
    main()
