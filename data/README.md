# The data

Two options, both downloaded by the script at the repo root. Pick one on launch day and stay with it, or bring your own (see the end of this page).

```bash
python download_data.py                                  # Amazon reviews, Health_and_Personal_Care (the default)
python download_data.py amazon --category Gift_Cards     # another Amazon category
python download_data.py amazon --list                    # the categories and their sizes
python download_data.py clothing                         # Women's clothing e-commerce reviews
```

Files land in `data/raw/`, which is gitignored, so someone cloning your repo re-runs the script rather than pulling data out of GitHub.

| | Dataset | What one row is | Rows | Licence |
|---|---|---|---|---|
| **1** | **Amazon Reviews 2023** (default) | a product review, plus a separate file describing each product | 16,000 to 700,000, depending on the category | none stated: a research dataset, cite it (below) |
| **2** | **Women's e-commerce clothing reviews** | a review of a clothing item | 23,486 | Public domain ([CC0](https://creativecommons.org/publicdomain/zero/1.0/)) |

---

## Option 1 — Amazon Reviews 2023 · *the default*

Real Amazon product reviews collected in 2023 by the McAuley Lab at UC San Diego, published on [Hugging Face](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) and described on the [dataset's website](https://amazon-reviews-2023.github.io/). The data is split by product category; the default, **Health_and_Personal_Care**, has 494,121 reviews of 60,293 products, mostly from 2016 to 2023.

| File | What it is |
|---|---|
| `<Category>.jsonl` | One review per line: `rating` (1 to 5 stars), `title`, `text`, `parent_asin` (the product id), `user_id`, `timestamp` (milliseconds since 1970), `helpful_vote`, `verified_purchase`, `images`. |
| `meta_<Category>.jsonl` | One product per line: `parent_asin`, `title`, `store`, `price`, `average_rating`, `rating_number`, `features` and `description` (lists of text), `categories`, `details`. |

Read them with pandas, and join reviews to products on **`parent_asin`** (not `asin`, which is the product variant):

```python
import pandas as pd

reviews = pd.read_json("data/raw/Health_and_Personal_Care.jsonl", lines=True)          # about 3 seconds
products = pd.read_json("data/raw/meta_Health_and_Personal_Care.jsonl", lines=True)
reviews["date"] = pd.to_datetime(reviews["timestamp"], unit="ms")
```

| Category | Reviews | Download |
|---|---|---|
| Subscription_Boxes | 16,000 | 10 MB |
| Magazine_Subscriptions | 71,000 | 37 MB |
| Gift_Cards | 152,000 | 52 MB |
| Digital_Music | 130,000 | 146 MB |
| **Health_and_Personal_Care** (default) | 494,000 | 345 MB |
| Handmade_Products | 664,000 | 690 MB |
| All_Beauty | 702,000 | 540 MB |

**The questions it supports.**

- *Model track:* is this review negative, neutral or positive, from the text alone? What rating will it give? Is a complaint about quality, delivery, price or side effects? A store could use such a model to route complaints to the right team, or to spot a product with a quality problem before its average rating drops.
- *LLM track:* what do customers say about a given product, with the reviews that say it? Which of two products has fewer complaints about a given problem? The product metadata (description, features, price) gives an LLM the facts; the reviews give it the experience.

**The catches** (numbers for Health_and_Personal_Care).

- **The ratings are very unbalanced.** 61% of reviews give 5 stars and 14% give 1; only 7.5% give 3. If you map stars to sentiment (1–2 negative, 3 neutral, 4–5 positive), 73% are positive: a model that always says *positive* has 73% accuracy and is useless. Choose a metric that sees the small classes (macro F1, per-class recall), and decide what to do with the 3-star reviews, which are often mixed.
- **Many texts are tiny.** The median review has 20 words, 10% have three words or fewer (*Great*, *Not for me*), and 249 are empty. Decide whether the title helps (it is often the most informative line).
- **HTML and duplicates.** About 8% of texts contain `<br />` line breaks and 1% contain HTML entities such as `&#34;`. 5,219 reviews are exact duplicates (same user, product and text), and 39,000 texts appear more than once. Clean before you split, so that the same text cannot be in both the training and the test set.
- **Products are not independent.** The top product has 3,192 reviews, and 217 products have at least 200. A random split puts reviews of the same product on both sides; if your model must work on *new* products, split by `parent_asin` instead.
- **Time matters.** Reviews span several years. For a realistic test, train on older reviews and test on newer ones, and say which split you chose.
- **Reviewers are people.** `user_id` is pseudonymous, but review texts can contain names, emails or health details. Do not use `user_id` as a feature, and do not send texts with personal details to an API.

**How to cite it.** The data has no stated licence. It is published for research, and its authors ask to be cited. Download it with the script (never commit or redistribute it), name the source in your README and on your data slide, and cite:

> Yupeng Hou, Jiacheng Li, Zhankui He, An Yan, Xiusi Chen and Julian McAuley. *Bridging Language and Items for Retrieval and Recommendation.* arXiv:2403.03952, 2024. Data: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), McAuley Lab, UC San Diego.

If you need a dataset with an explicit open licence (for example, to keep working on it after the bootcamp or to publish it), choose option 2.

---

## Option 2 — Women's e-commerce clothing reviews · *licence-clear*

23,486 reviews of women's clothing from an online shop, released into the public domain on [Kaggle](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) and mirrored on [Hugging Face](https://huggingface.co/datasets/chibifire/kaggle-womens-ecom-clothing-reviews), which is where the script downloads it from, without a login.

| File | What it is |
|---|---|
| `womens_clothing_reviews.parquet` | One review per row: `rating` (1 to 5), `recommended_ind` (1 if the customer recommends the item), `title`, `review_text`, `age`, `positive_feedback_count`, `clothing_id`, and the product's `division_name`, `department_name` and `class_name`. |

```python
reviews = pd.read_parquet("data/raw/womens_clothing_reviews.parquet")
```

**The questions it supports.** Will this customer recommend the item, or what rating will they give, from the text? What do customers say about fit, size or fabric in each department? Which items get the most complaints about sizing?

**The catches.**

- **Unbalanced targets.** 56% of ratings are 5 stars, and 82% of customers recommend the item.
- **Rating and recommendation disagree sometimes.** 1,189 of the 2,871 three-star reviews recommend the item, and 168 four-star reviews do not. Look at them before you choose your target.
- **Missing text.** 845 reviews have no text and 3,810 have no title.
- **It is smaller.** 23,000 reviews is plenty for TF-IDF and embeddings, and for fine-tuning a small transformer on a sample. Per item, there are fewer reviews than on Amazon, so a RAG assistant works better per department or class than per item.

**Licence.** Public domain ([CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)). No attribution is required, but name the source in your README anyway: *Women's E-Commerce Clothing Reviews, by nicapotato on Kaggle, CC0.*

---

## Bringing your own data or topic

You may, and you are graded on exactly the same [rubric](../RUBRIC.md). The conditions:

- **A clear user and question.** Who uses your product, and what do they decide with it?
- **Enough for an honest evaluation.** *Model track:* a few thousand labelled texts (or labels you can derive with a stated rule). *LLM track:* a document collection that answers your users' questions, and at least 15 test questions with the source of each answer.
- **A licence that permits educational use.** Anything marked *NonCommercial* is a grey area for a bootcamp; anything with no stated licence needs your teacher's agreement, as for option 1.
- **No personal or confidential data sent to an API.** Your company's documents, customer emails or medical records are not project data unless you are explicitly allowed to use them, and even then not with an external LLM.

Clear it with your teacher on launch day. Not on Wednesday.

## Attribution, in one place

Whatever you use, your project README must name the source and its licence or citation. For option 1 that is the citation above; for option 2 the line above; for your own data, whatever its licence requires.
