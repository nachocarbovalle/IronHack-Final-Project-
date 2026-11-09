![Ironhack logo](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Project 3 | AI Product: From Model to Product

*Data Science & Machine Learning — Week 9, the final project*

Build a small AI product that someone could actually use. You will frame a problem, build either a trained NLP model or an LLM application, evaluate it honestly against a baseline, and put it in the hands of a user through a **Streamlit app** that you demo live.

`Python` · `NLP` · `Transformers` · `Embeddings` · `LLM APIs` · `RAG` · `Agents` · `Evaluation` · `Streamlit` · `MLOps`

---

## Overview

This is the last project of the bootcamp, and the most open one. Weeks 7 and 8 gave you two ways to build with language: **train a model** (TF-IDF, embeddings, a fine-tuned transformer) or **build on an LLM** (prompts, RAG, tools). Project 3 asks you to pick one, take it all the way to a working app, and prove that it works.

- **Format** — in pairs or on your own. Working in a pair, divide the work by stage (data, core, evaluation, app), not by person-per-notebook, and run each other's code every day.
- **Duration** — launch on Friday of Week 8, 4 working days, presentations on Friday of Week 9.
- **Tools** — Python, pandas, scikit-learn, sentence-transformers, Hugging Face `transformers`, the OpenAI API, LangChain (optional), Chroma, MLflow (optional), Streamlit, GitHub.
- **Repository** — click **Use this template** at the top of this page to create your own repo.

### The goal

Answer one question for one kind of user, with an AI component they can use: *Is this review a complaint? What do customers say about the battery? Which of my products have a quality problem?* By Friday you should be able to say what your product does, for whom, how well it works compared with a simple alternative, what it costs to run, and where it should not be trusted.

> [!IMPORTANT]
> Choose your track, your data and your user on launch day, and stick with them. By Wednesday your evaluation is built around them; switching costs four days of work in two.

<br>

## Choose your track

Every project has **one AI core** and **one app**. You choose the core:

| | **Model track** | **LLM track** |
|---|---|---|
| **The core** | A model you train and evaluate: a text classifier, a rating predictor, a topic or aspect model | An LLM application: RAG over documents, an agent with tools, structured extraction at scale |
| **From the course** | Week 7: TF-IDF, embeddings, fine-tuning; Week 5: pipelines, validation | Week 7: the OpenAI API, RAG; Week 8: agents and tools |
| **The baseline** | A dummy model, then TF-IDF with a linear model | The LLM alone (no retrieval, no tools), or a keyword search |
| **The evaluation** | A held-out test set, used once; the right metric for the decision | An evaluation set of at least 15 questions, written **before** you build, with the expected answer or source |
| **Also report** | Training time and model size | The cost and the time per question |
| **The app** | Loads the saved model and predicts for new text | Calls your pipeline and shows the answer with its sources |

You may do both (for example, a sentiment model whose predictions feed an LLM summary), but one done well beats two done halfway. **The app is mandatory on both tracks.**

<br>

## Choose your data

The default dataset is **Amazon Reviews 2023**: real product reviews with a star rating, a title, the text and the product's description, from a research dataset published by the McAuley Lab at UC San Diego. Read [`data/README.md`](data/README.md) for the fields, the categories, the traps to watch for, and how to cite it.

```bash
python download_data.py                                  # Amazon reviews, Health_and_Personal_Care (the default), 345 MB
python download_data.py amazon --category Gift_Cards     # another category, listed in data/README.md
python download_data.py clothing                         # option 2: women's clothing reviews, public domain
```

The Amazon data has no stated licence: it is published for research, and you must cite it (the reference is in the data guide) and never commit it. The second option, **23,000 women's clothing reviews** released into the public domain, is smaller and licence-clear.

Files land in `data/raw/`, which is gitignored. **Do not commit data.** Anyone cloning your repo runs the same script.

Some project ideas on this data:

- **Model track:** classify reviews as negative, neutral or positive from the text alone, and route the negative ones to a support team; predict the star rating; detect which aspect a complaint is about (quality, delivery, price, side effects).
- **LLM track:** *Ask the reviews*, a RAG assistant that answers questions about a product from its reviews and cites them; a product-comparison agent with a retrieval tool and a calculator; an extractor that turns thousands of reviews into a table of complaints, then summarises it.

> [!NOTE]
> **Bringing your own data or topic?** You may, and you are graded on exactly the same rubric. You need a clear user and question, enough data for an honest evaluation (a few thousand labelled texts for the model track; a document collection and 15 test questions for the LLM track), and a licence that permits educational use. **No personal data** you are not allowed to process, and no confidential documents sent to an API. Clear it with your teacher on launch day, not on Wednesday.

<br>

## Day-by-day breakdown

Launch day (Friday of Week 8) is when you settle who you are working with, choose your track, data and user, create your repo, download the data and write your framing — so Monday starts with the data in front of you.

### Day 1 — Data and baseline

- Finish **`01_framing_and_baseline.ipynb`**: what one row (or one document) is, data quality, and the distribution of what you predict or the kinds of questions you answer.
- **Model track:** clean the text, define the target, **split once** (a fixed `random_state`, stratified), and score a dummy model and a TF-IDF baseline with cross-validation.
- **LLM track:** prepare the documents (cleaning, chunking, metadata), write your **evaluation set** of at least 15 questions with the expected source of each answer, and score the baseline: the LLM alone, or a keyword search.

> [!TIP]
> By the end of today you should be able to say your metric and your baseline score out loud. If you cannot, you are behind.

### Day 2 — Build the core

Today is the core engineering day, and the one the week turns on.

- **Model track:** compare **at least three approaches** on the same validation folds or the same validation set — for example TF-IDF with a linear model, sentence embeddings with a classifier, and a fine-tuned DistilBERT or a zero-shot model — in **`02_build.ipynb`**. Track the runs with MLflow if you can.
- **LLM track:** build the full pipeline end to end — retrieval with metadata, a grounded prompt that cites its sources and says "I don't know", or an agent with its tools — and run it on a few questions from your evaluation set.

> [!IMPORTANT]
> **Anyone without an end-to-end result on Tuesday evening has no Wednesday.** Everything after this builds on it.

### Day 3 — Evaluate and improve

- **Model track:** tune the best approach, choose the final one, **evaluate it once on the test set**, analyse the errors (confusion matrix, per-class results, examples of mistakes), and **save** it.
- **LLM track:** run the **whole evaluation set**: retrieval hit rate, whether answers cite the right sources, whether out-of-scope questions are refused, the cost per question. Make **one improvement** (chunk size, a filter, a better prompt, a stronger embedding model) and measure it on the same set.
- Write the results in **`03_evaluate.ipynb`**, your report.

### Day 4 — The app and the story

- Build the **Streamlit app** in `app/streamlit_app.py`: it loads your saved model (or builds your pipeline) once with `st.cache_resource`, takes the user's input, and shows the result. Keys go in `.streamlit/secrets.toml` or `.env`, **never in the code**.
- Check that someone else can run it from a fresh clone by following your README.
- Write your README, build your slides, and **rehearse once, timed**.
- *Optional:* deploy the app on Streamlit Community Cloud.

> [!NOTE]
> **Notebook 03 is your report.** It is read on its own, by someone who has not seen notebooks 01 and 02: the question, the evidence, what you concluded and how far to trust it.

<br>

## Deliverables

Your main deliverable is your GitHub repo, created from this template, containing:

| Item | Description |
|---|---|
| `README.md` | Your project: the user and the question, the track, the data and its source, the baseline and the final result, the cost, the limitations, and **how to run the app**. Replace this brief with your own. |
| `notebooks/01_framing_and_baseline.ipynb` | Framing, data, the split or the evaluation set, and the baseline. |
| `notebooks/02_build.ipynb` | The approaches compared (model track) or the pipeline built (LLM track). |
| `notebooks/03_evaluate.ipynb` | **The report.** The final evaluation, the error or failure analysis, the cost, and the limitations. |
| `app/streamlit_app.py` | The app you demo. |
| `src/functions.py` | Reusable functions, imported by the notebooks and the app. |
| `requirements.txt` | Everything needed to run the notebooks and the app. |
| Slides | Linked or committed in your README, so they can be read after the presentation. Any tool. |

Saved models in `models/`, your `.env` file and `.streamlit/secrets.toml` are gitignored. The README says how to recreate the model and where to put the keys.

### Minimum requirements

Your project must meet all of these to pass:

- **Framing** — a stated user, the question or decision your product supports, and a metric chosen for that decision.
- **Baseline** — a simple alternative scored with the same evaluation as your final solution.
- **Honest evaluation** — *model track:* at least 3 approaches compared on the same validation, and the final model evaluated once on a held-out test set; *LLM track:* an evaluation set of at least 15 questions written before building, with retrieval or tool use checked, citations checked, and out-of-scope questions tested.
- **Failure analysis** — where and why it goes wrong, with examples.
- **Cost** — *LLM track:* the cost per question; *model track:* the training time and the size of the model.
- **A working app** — runs from a fresh clone by following the README, and uses your saved model or your pipeline.
- **No secrets in git** — no API key anywhere in the history.

The full grading rubric is in [`RUBRIC.md`](RUBRIC.md). Read it on launch day — it tells you exactly what is being evaluated.

<br>

## Advanced features (optional)

Not required, but they will strengthen your project:

- **Experiment tracking** — every run in MLflow, the final model registered and loaded by alias in the app.
- **Both tracks** — a trained classifier that filters or routes reviews, and an LLM that summarises or answers from them.
- **An LLM as a judge** — a second model that grades each answer against the expected one, checked by hand on a sample.
- **Deployment** — the app live on Streamlit Community Cloud, with its keys in the platform's secrets.
- **Monitoring** — a page in the app that shows the distribution of predictions or questions, and flags drift.
- **Multimodal** — a photo or a voice question as input, with the cost measured.

<br>

## Coding best practices

- **Modularise.** Logic in `src/functions.py`, imported by both the notebooks and the app. The notebook is for the narrative.
- **One seed.** Define `RANDOM_STATE` once and pass it everywhere, so a second run gives the same numbers.
- **One model name.** Define the LLM and the embedding model once; log the tokens of every call.
- **Keys out of code.** `.env` locally, `st.secrets` in the app, both gitignored. Check `git status` before every commit.
- **Restart and run all.** Before you commit a notebook, run it top to bottom on a fresh kernel.
- **Commit often.** Small, frequent commits with descriptive messages.

<br>

## Presentation guidelines

| Component | Duration |
|---|---|
| Talking with slides | 7 minutes |
| Live demo of the app | 3 minutes |
| **Total** | **10 minutes** |

> [!IMPORTANT]
> **You present from your own machine by sharing your screen.** The demo is your Streamlit app, running, on an example the audience can follow. Have it open and ready before your slot, with a second example in case the first one surprises you.

### Suggested slide structure (~10 slides)

1. **Title** — project title and your name or names.
2. **The user and the problem** — who uses it, and what they decide with it.
3. **The data** — source, what one row or document is, the traps you found.
4. **The metric and the baseline** — why this metric, and the score to beat.
5. **What you built** — the approaches compared, or the pipeline, in one diagram.
6. **The result** — the final evaluation against the baseline.
7. **Where it goes wrong** — the failure analysis, with examples.
8. **Cost and limits** — what it costs to run, and where it should not be trusted.
9. **Conclusions and next steps** — would you ship it, and what would you do next?
10. **Closing** — project title, your name or names, thank you.

<br>

## Repo layout

The notebooks are laid out in sections with the intent of each one written down, for both tracks, and `src/functions.py` and the app hold a few stubs that exist only to show the shape. **This is a place to start, not a template to fill in** — rename things, add sections, delete the ones for the other track. You are graded on the work, not on how closely you followed the scaffold.

```
.
├── README.md                          this brief — replace it with your own
├── RUBRIC.md                          how you are graded
├── requirements.txt
├── download_data.py                   fetches the reviews into data/raw/
├── .env.example                       the keys you need, without their values
├── data/
│   ├── raw/                           downloaded data, gitignored
│   └── README.md                      the dataset, its fields, its traps, how to cite it
├── models/                            saved models and indexes, gitignored
├── notebooks/
│   ├── 01_framing_and_baseline.ipynb  frame, look, split or evaluation set, baseline
│   ├── 02_build.ipynb                 compare approaches, or build the pipeline
│   └── 03_evaluate.ipynb              evaluate, analyse failures, cost — the report
├── app/
│   └── streamlit_app.py               the app you demo
└── src/
    └── functions.py                   your reusable logic
```

## Getting set up

```bash
# 1. Click "Use this template" above, then clone YOUR new repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install what you need (read the note at the top of requirements.txt first)
pip install -r requirements.txt

# 3. LLM track: copy .env.example to .env and add your keys (never commit .env)
cp .env.example .env

# 4. Download the data
python download_data.py

# 5. Open the first notebook, and later run the app
jupyter lab notebooks/01_framing_and_baseline.ipynb
streamlit run app/streamlit_app.py
```

<br>

## Tips for success

1. **Choose your track, data and user on launch day.** Switching mid-week costs you time you do not have.
2. **Write the evaluation before you build.** A test set or an evaluation set decided on Monday keeps you honest on Thursday.
3. **Beat the baseline before you beat anyone else.** A fine-tuned transformer that does not beat TF-IDF, or a RAG system that does not beat the LLM alone, is a finding — say so.
4. **Start the app on Wednesday evening**, even if it only shows a title and a text box. Apps always take longer than you think.
5. **Watch the cost.** Use `gpt-5.4-nano` while you build, and log the tokens of every call.
6. **Keep keys out of git.** One leaked key and the project stops while it is revoked.
7. **Ask for help early.** Stuck for more than 30 minutes? Reach out to a classmate, your TA or your teacher.
8. **Practise the demo out loud** at least once before Friday. Time yourselves.
9. **Read [`RUBRIC.md`](RUBRIC.md).** It tells you exactly what is being evaluated.

Good luck. Build something you would trust someone else to use.
