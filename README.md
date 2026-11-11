# 🎮 Ask the Players

**A RAG assistant that answers questions about PC games from Steam reviews, and shows the reviews it used.**

Nacho Carbó · Ironhack Data Science & Machine Learning bootcamp · Project 3, "AI Product: From Model to Product" · November 2026

---

## The user and the question

**The user** is a PC player deciding whether to buy a game, or wondering how it runs and plays today. Steam shows one score and thousands of unsorted reviews; finding the answer to one specific question ("Is it a good first roguelike?", "Does it still have performance problems?") means reading far too many of them.

**The decision:** buy now, wait for a patch or a sale, or skip the game.

**The product:** pick one of 23 games, ask a question, and get a short answer built only from real player reviews, with numbered references to the reviews it used, shown underneath so every claim can be checked. When the reviews cannot answer, it says so instead of guessing.

## The track

**LLM track: retrieval-augmented generation (RAG).**

| Step | Choice |
|---|---|
| Documents | Cleaned Steam reviews, split along their paragraphs into chunks of up to 200 tokens (44,477 chunks) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, normalized |
| Index | Chroma, persisted on disk |
| Retrieval | Top 5 chunks, filtered to the selected game |
| Generation | `gpt-5.4-nano`, prompted to answer **only** from the retrieved reviews, cite them, say when evidence is limited, and refuse when the reviews do not answer |

## The data

- **Source:** Valve's public Steam review API (`IUserReviewsService/GetAppReviews`, no key needed), English reviews of purchased copies, collected on **10 November 2026**.
- **23 games**, from Dying Light (2015) to 007 First Light (2026), listed in `data/games.csv`.
- **Sampling:** each game's life on Steam is split into 5 equal time windows, with 200 reviews per window, sorted by helpfulness but **stratified by the window's real share of positive and negative reviews**. Sorting by helpfulness alone made the samples 11 to 30 points more negative than reality, because complaints are long and collect helpful votes.
- **One row** is one review: its text, whether the player recommends the game, playtime at the time of the review, dates (created and last updated), votes and a few hardware flags. Steam account IDs are never stored.
- **Cleaning:** 23,000 reviews collected → **22,707 kept**. Removed: 105 mostly non-Latin, 20 with fewer than 5 words, 116 in another language (a function-word screen, confirmed with `langdetect`), 52 exact duplicates. Steam markup, links and repeated symbols are stripped.
- **Traps found:** Steam keeps only the latest version of an edited review (18.5% were edited); 13% of reviews mention another game of the corpus ("Dark Souls" is used as a genre word); the newest content is under-represented (only 16 chunks discuss a DLC released 12 days before collection).

## The evaluation, the baseline and the result

**The metric.** A confident wrong answer is the worst mistake (the player pays for something invented), an answer without a source is next, and a refusal is the cheapest. The system is evaluated on **30 questions written before it was built**, in 8 types: facts (11), changes over time (4), filtered by playtime or Steam Deck (2), comparisons (3), scarce evidence (2), recent games and content (2), hard cases (2), and **4 out-of-scope questions** the reviews cannot answer. Each has an expected answer written from the reviews and 3 to 6 verified source reviews (`data/eval_questions.csv`). Answers are graded with a rubric fixed before reading them: 1 correct, 0.5 partial, 0 wrong or refused, plus flags for refusals and hallucinations.

**The baseline** is the same LLM with no reviews, given the same "say you don't know instead of guessing" instruction.

| | RAG system | LLM alone (baseline) |
|---|---|---|
| **Answer score** (26 answerable questions) | **0.63** | 0.33 |
| Correct / partial / wrong / refused | 8 / 17 / 0 / 1 | 4 / 9 / 1 / 12 |
| Out-of-scope questions refused correctly | **4 of 4** | 3 of 4 |
| Hallucinations | **0** | 0 |
| Retrieval hit rate (top 5) | **0.85** (5 random chunks: 0.52) | — |
| Invented citations | **0 of 30** | — |

| Question type | RAG | Baseline |
|---|---|---|
| Changes over time | 0.75 | 0.00 |
| Scarce evidence (complaints) | 0.75 | 0.00 |
| Comparisons | 0.67 | 0.33 |
| Facts | 0.64 | 0.55 |
| Filtered | 0.50 | 0.25 |
| Hard cases | 0.50 | 0.50 |
| Recent games and content | 0.50 | 0.00 |

The RAG system nearly doubles the baseline's score without losing its honesty. Question by question: 14 better, 8 the same, 4 worse. The gains are largest where the LLM alone knows nothing (its knowledge stops in August 2025); on famous older games, its own knowledge sometimes gave fuller answers than 5 retrieved chunks.

**One design choice compared:** the game filter. Without it, the hit rate was the same (0.85), because every evaluation question names its game. It is kept because real users will not always name the game.

## Where it goes wrong

Of the 18 imperfect answers, **12 are retrieval failures** (the evidence was not among the 5 chunks) and **6 are generation failures** (the evidence was there, but the answer missed or misused it):
- **Minority views are crowded out:** a few dissenting reviews rarely make the top 5 among hundreds of positive ones.
- **Embeddings find the topic, not the angle or the sentiment:** "complaints about Spider-Man" retrieves positive reviews of Spider-Man.
- **Reasoning about time and evidence:** a player's anticipation ("can't wait for the DLC") presented as the DLC's reception; reviews from 2023 used as evidence that problems "still" exist.
- **Variability:** the same question, with the same retrieved reviews, can get different answers on two runs.

**Do not trust it for:** "Is there anything wrong with this game?" questions about well-liked games, specific angles (the late game, one mode), brand-new content, "is it still…?" questions, or any single answer without checking its sources. The full analysis is in `notebooks/03_evaluate.ipynb`.

## Cost

| | RAG system | LLM alone |
|---|---|---|
| Tokens per question (in / out) | 900 / 165 | 82 / 104 |
| Cost per question | $0.00039 | $0.00015 |
| Time per answer | 1.9 s (+ 16 ms retrieval) | 1.8 s |

The whole evaluation (30 questions) cost about one cent. At **1,000 questions a day**, the LLM would cost about **$11.60 a month**. The index takes 309 MB on disk; retrieval runs locally, at no cost.

## Limitations

- **A small evaluation:** 30 questions, graded by one person, one run per question. Differences of a few points are not reliable, and per-type scores rest on 2 to 11 questions.
- **The retrieval hit rate is approximate:** it uses patterns describing the expected evidence, which miss paraphrases and can match negations.
- **The evaluation set informed one decision** (the game filter comparison), so that comparison is not fully independent.
- **The data has blind spots:** English Steam reviews only (so the PC version), the newest reviews are under-represented, edited reviews show only their latest version, and bilingual reviews were removed.
- **Re-collecting the data gives different reviews:** new reviews keep arriving and helpfulness rankings change, so a rebuilt index will differ from the evaluated one.

## Next steps

1. **Improve retrieval**, which causes 12 of the 18 failures: hybrid search (keywords + embeddings) for names the embedding model does not know, retrieving more chunks and re-ranking them, and rewriting questions to capture their angle.
2. **Give the model today's date and the collection date**, so it can judge how old its evidence is.
3. **Refuse when the closest review is far from the question**, with a threshold chosen on new questions, not on the evaluation set.
4. **Allow two games at once** for comparisons.
5. **Evaluate more:** more questions, a second grader, several runs per question.

---

## How to run it

**Requirements:** Python 3.12, conda (or another virtual environment), an OpenAI API key.

### 1. Clone the repository and install the dependencies

```bash
git clone https://github.com/nachocarbovalle/IronHack-Final-Project-.git
cd IronHack-Final-Project-
conda create -n final-project python=3.12
conda activate final-project
pip install -r requirements.txt
```

### 2. Add your API key

```bash
cp .env.example .env
```

Open `.env` and set `OPENAI_API_KEY=` to your key (no quotes, no spaces). `.env` is gitignored: never commit it. For a deployment on Streamlit Community Cloud, put the key in the platform's secrets instead; the app reads it from `st.secrets` when `.env` is absent.

### 3. Collect the reviews

The raw reviews are not in the repository (`data/raw/` is gitignored). Run the collection in `notebooks/00_api_test.ipynb`: it downloads the reviews of the 23 games in `data/games.csv` from the Steam API into `data/raw/steam_reviews/`, one Parquet file per game. Games already downloaded are skipped.

### 4. Build the index

Run `notebooks/02_build.ipynb` from top to bottom. It cleans and chunks the reviews with the functions in `src/functions.py`, embeds the chunks, and saves the index in `models/chroma/` and the configuration in `models/config.json` (both gitignored build artefacts). Embedding the 44,477 chunks takes about a minute and a half on a laptop.

### 5. Run the app

From the repository root:

```bash
streamlit run app/streamlit_app.py
```

The app has four pages:
- **🏠 Home:** the project in one page.
- **🎤 Presentation:** the project's slides, with charts built from the saved results in `data/results/`.
- **🕹️ The games:** the 23 games, with their Steam banners and release dates.
- **🎮 Ask the players:** the assistant. Pick a game, ask a question, and check the numbered sources under the answer. A warning appears when the closest review is far from the question (a display heuristic, not a validated threshold).

Questions asked in the app are logged locally in `logs/questions.csv` (gitignored) for monitoring.

---

## Repository layout

```
.
├── README.md
├── RUBRIC.md                       the project's grading rubric
├── requirements.txt
├── .env.example                    copy to .env and add your key
├── .streamlit/config.toml          the app's theme
├── app/
│   └── streamlit_app.py            the app and the presentation
├── data/
│   ├── games.csv                   the 23 games: Steam app id, release date, banner URL
│   ├── eval_questions.csv          the 30-question evaluation set
│   ├── results/                    baseline and RAG answers, grades, failure analysis
│   └── raw/                        the collected reviews (gitignored)
├── models/                         Chroma index and config.json (gitignored, built by notebook 02)
├── notebooks/
│   ├── 00_api_test.ipynb           Steam API tests and review collection
│   ├── 01_framing_and_baseline.ipynb   framing, data quality, cleaning, chunking, evaluation set, baseline
│   ├── 02_build.ipynb              the index, retrieval, generation, the design choice, the saved version
│   └── 03_evaluate.ipynb           the report: final evaluation, failures, cost, conclusions
└── src/
    └── functions.py                every reusable function, imported by the notebooks and the app
```

## Data and credits

Reviews are written by Steam users and collected through Steam's public review API for an educational, non-commercial project; only the review text and aggregate fields are kept, never account identifiers. Game banners are displayed from Steam's servers and are not stored in this repository. Built with pandas, sentence-transformers, Chroma, the OpenAI API and Streamlit.