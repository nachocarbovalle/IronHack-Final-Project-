![Ironhack logo](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Project 3 — Evaluation Rubric

*AI Product: From Model to Product · Data Science & Machine Learning*

This is the **single source of truth** for how Project 3 is graded. It is adapted from Ironhack's DSML project rubric and uses the same **0–3 scale** as the [Generic Lab Rubric](https://gist.github.com/ironhack-edu/f5cf405db1708c201ad774ee4516bc94) and the Project 1 and 2 rubrics, so labs and projects are assessed on one house scale.

Read it on launch day. It tells you exactly what is being looked at, and there are no surprises in here that the [brief](README.md) does not already ask for.

## How the scale works

| Score | Meaning |
|---|---|
| **0 — Incomplete** | Not attempted, or attempted in a way that does not meet the requirement. |
| **1 — Fair** | Attempted, with real gaps. |
| **2 — Good** | Meets the requirement competently. **This is the target.** |
| **3 — Excellent** | Exceeds it — depth, rigour or polish beyond what was asked. |

Each level is a **checklist behind a threshold**: "all of these apply", "at least two of the following apply". Read the threshold before the bullets. You are placed at the highest level whose threshold you satisfy.

**Passing is a 2 on every criterion that applies to your project.** Two or three 3s is a strong project; 3s everywhere is rare and is not what you should be aiming for on a four-day project.

**Two tracks, one rubric.** Criteria 4 and 5 have a version for each track: you are assessed on the one you chose. Everything else applies to both.

## The eleven criteria

| # | Criterion | Domain | Where it is judged from |
|---|---|---|---|
| 1 | [Problem framing](#1--problem-framing) | Business Problem Solving | The framing at the top of notebook 01, the conclusions in notebook 03 |
| 2 | [Data preparation](#2--data-preparation) | Data Preparation | Notebooks 01 and 02 |
| 3 | [Exploratory analysis](#3--exploratory-analysis) | Data Analysis | Notebook 01 |
| 4 | [Building the core](#4--building-the-core) | Machine Learning / Generative AI | Notebook 02 |
| 5 | [Evaluation](#5--evaluation) | Machine Learning / Generative AI | Notebook 03 |
| 6 | [The app](#6--the-app) | Development Tools & Workflow | `app/streamlit_app.py`, the demo |
| 7 | [Data visualisation](#7--data-visualisation) | Data visualization and communication | Notebooks, app, slides |
| 8 | [Code quality and structure](#8--code-quality-and-structure) | Coding | `src/functions.py`, the repo as a whole |
| 9 | [Git and GitHub](#9--git-and-github) | Coding | Commit history |
| 10 | [Documentation](#10--documentation) | Coding | `README.md`, docstrings, comments |
| 11 | [Presentation and demo](#11--presentation-and-demo) | Data visualization and communication | Friday, 7 + 3 minutes |

Criteria **4**, **5** and **6** are specific to this project and carry most of the week's learning. The other eight are the Ironhack DSML rubric criteria, carried over with the changes noted.

---

## 1 | Problem framing

**Learning outcome:** Turn a business problem into a product question — who uses it, what they decide with it, and how you will know it works — and choose a metric from the cost of each kind of mistake.

> Adapted from the DSML rubric's *Business problem solving*: research questions become a user, a decision, a metric and a baseline.

**0 — Incomplete** · *all of these apply*
- There is no clear statement of what the product does or for whom.
- No metric or evaluation is chosen, or it is chosen with no reason.
- No conclusions are drawn for the user.

**1 — Fair** · *at least two of the following apply*
- The task is stated, but not who would use it or what they would do with the result.
- A metric is used, but it does not fit the problem — for example accuracy on reviews that are 73% positive, with no discussion.
- There is no baseline, so the result cannot be judged.
- Conclusions restate the scores without saying what they mean for the user.

**2 — Good** · *all of these apply*
- **The product question is framed**: the user, the input, the output, and the decision it supports.
- **The metric is justified** by which mistake costs more — a missed complaint, a wrong answer, an answer with no source.
- **A baseline is scored** with the same evaluation as the final solution, and every result is read against it.
- Conclusions answer the original question, including whether the product is good enough to use, and for what.

**3 — Excellent** · *at least three of the following apply*
- The cost of each kind of mistake is made concrete — with numbers or a worked scenario — and drives the choice of metric, threshold or refusal behaviour.
- The conclusion includes a recommendation a non-technical stakeholder could act on.
- The framing is revisited at the end: what the product can and cannot be used for, and what would change the recommendation.
- The running cost is compared with the value of the decision it supports.

---

## 2 | Data preparation

**Learning outcome:** Identify data quality issues in a dataset and apply appropriate data cleaning, wrangling, and manipulation techniques to address them, without letting information from the test set or the evaluation set into the build.

**0 — Incomplete** · *all of these apply*
- No relevant data quality issues found.
- No relevant implementation of data cleaning or wrangling techniques.

**1 — Fair** · *all of these apply*
- Some data quality issues identified, but they are irrelevant, unclear or incomplete.
- Cleaning and wrangling techniques were applied, but they are inadequate for the issues identified.

**2 — Good** · *at least three of the following apply*
- Identified the relevant data quality issues, with clear explanations of their impact on the models.
- Implemented appropriate cleaning techniques, consistently applied to most identified issues.
- **Text is cleaned for its use**: HTML, empty and duplicate texts handled; for the LLM track, documents chunked with metadata (product, rating, date, id) that the answers can cite.
- **The target is defined from the data with a stated rule** (for example, stars to sentiment), or the evaluation set is written from the documents with the source of each answer.

**3 — Excellent** · *all of these apply*
- Identified most relevant data quality issues, with clear explanations of their impact.
- Implemented relevant techniques, consistently applied to **all** identified issues.
- Every column kept or dropped for a reason that can be read in the notebook, including the borderline ones.
- Personal data in the texts (names, emails, phone numbers) is checked for, and removed or masked before anything is sent to an API.

---

## 3 | Exploratory analysis

**Learning outcome:** Apply Exploratory Data Analysis techniques to understand the texts, the target or the questions, and to guide the decisions that follow.

**0 — Incomplete** · *all of these apply*
- No basic EDA techniques used to analyse the data.
- Improper use of EDA for the data types present.

**1 — Fair** · *at least two of the following apply*
- Basic EDA techniques, but incomplete or inaccurate.
- The target's distribution is not looked at.
- Charts are produced, but they are not easy to understand or not interpreted.

**2 — Good** · *at least three of the following apply*
- **The target or the questions are analysed**: the share of each class, or the kinds of questions and where their answers are.
- **The texts are analysed**: their length, their vocabulary, and how they differ between classes, products or periods.
- Findings are written in prose under the outputs, not left as bare cells.
- The EDA visibly informs a later decision — a transformation, a dropped column, a feature, a metric.

**3 — Excellent** · *all of these apply*
- Thorough univariate and bivariate analysis, using numerical measures and charts suited to each data type.
- Hard or ambiguous examples are read and shown — the 3-star reviews, the sarcastic ones, the questions with no answer in the data.
- Every modelling decision that follows can be traced back to something shown in the EDA.

---

## 4 | Building the core

**Learning outcome:** Build an NLP model or an LLM application with the tools of Weeks 7 and 8, and make design choices you can justify with evidence rather than intuition.

**Skill tags:** *Deep Learning & AI: Natural Language Processing*, *Deep Learning & AI: Generative AI*

### Model track

**0 — Incomplete** · *all of these apply*
- One model, or models with no comparison.
- No baseline.
- The test set used for training or tuning, or no separate test set at all.

**1 — Fair** · *at least two of the following apply*
- Several approaches, but scored on different splits, or on the training data.
- A vectoriser or any step that learns from the data fitted on all the data before the split.
- Settings chosen by looking at test-set scores.
- Results reported with no spread, so a small difference cannot be told from noise.

**2 — Good** · *all of these apply*
- **The data is split once**, with a fixed `random_state` and stratified, and the test set is not used until criterion 5.
- **Every step that learns from data is inside a `Pipeline`** (or fitted on the training data only, for embeddings and transformers).
- **At least three approaches compared on the same validation** — for example TF-IDF with a linear model, embeddings with a classifier, and a fine-tuned or zero-shot transformer — against the baseline, with the mean and spread, or the validation score.
- **The best approach tuned** on the training data only, and its training time reported.

**3 — Excellent** · *at least three of the following apply*
- Experiments tracked in MLflow, and the final model registered.
- Class imbalance handled deliberately (class weights, the metric, the threshold) and its effect measured.
- The simplest approach within the noise of the best is preferred, and the choice is argued, including training time, size and cost.
- A learning curve shows whether more labelled data would help.

### LLM track

**0 — Incomplete** · *all of these apply*
- A single prompt with no retrieval, tools or structure, and no comparison.
- No baseline.
- Design choices (model, chunking, prompt, tools) made with no evidence.

**1 — Fair** · *at least two of the following apply*
- The pipeline works on a few hand-picked questions only.
- No sources: answers cannot be traced to a document or a tool result.
- The prompt has no rule for questions the data cannot answer.
- Keys or personal data written in the code or the prompts.

**2 — Good** · *all of these apply*
- **A complete pipeline**: retrieval with metadata and a grounded prompt, or an agent with well-described tools, or structured extraction validated with Pydantic.
- **Answers cite their sources** (document ids, products, tool results), and the prompt says what to do when the answer is not in the data.
- **The baseline is run** on the evaluation set: the LLM alone, or a keyword search.
- **One design choice is compared with evidence**: chunk size, `k`, a metadata filter, the embedding model, the prompt, or the tools — on the evaluation set.

**3 — Excellent** · *at least three of the following apply*
- Several design choices compared on the evaluation set, with the winner justified by quality and cost.
- Guardrails: a step limit for agents, input checks, refusal of out-of-scope questions, read-only tools.
- The cost and the latency of each question are logged and reported.
- A trained model from the model track is combined with the LLM (for example, to route or filter what the LLM reads).

---

## 5 | Evaluation

**Learning outcome:** Evaluate the final solution once, on data or questions it was not built on, read the result honestly, find where and why it fails, and know what it costs.

**Skill tags:** *Statistics & Machine Learning: Model Evaluation*, *Deep Learning & AI: Generative AI*

### Model track

**0 — Incomplete** · *all of these apply*
- No evaluation on held-out data.
- Scores reported with no interpretation.

**1 — Fair** · *at least two of the following apply*
- The test set is used several times, so the reported score is optimistic.
- Only one number is reported — no confusion matrix, no per-class results.
- No examples of mistakes are read.
- The model is not saved, or cannot be reloaded.

**2 — Good** · *all of these apply*
- **The final model is evaluated once on the test set**, and the score is compared with the validation estimate and with the baseline.
- **Errors are analysed**: a confusion matrix, per-class precision and recall, and a sample of misclassified texts with a reading of why.
- **The training time and the size of the model are reported**, and the time to predict one text.
- **The model is saved** and reloaded to predict new text, as the app does.

**3 — Excellent** · *at least three of the following apply*
- The errors are segmented — by product, by text length, by period — and the segments where the model should not be trusted are named.
- Mislabelled or ambiguous examples are identified, and their effect on the score is discussed.
- The result is turned into the user's terms: how many complaints found, how many missed, how much work saved.
- Limitations are stated precisely: what products, language or time period the model holds for.

### LLM track

**0 — Incomplete** · *all of these apply*
- No evaluation set, or questions chosen after seeing the answers.
- Answers judged by eye on a few examples.

**1 — Fair** · *at least two of the following apply*
- Fewer than 15 questions, or no expected source or answer for them.
- Only the final answers are checked, not the retrieval or the tool calls.
- No out-of-scope questions.
- The cost is not reported.

**2 — Good** · *all of these apply*
- **An evaluation set of at least 15 questions, written before building**, each with the expected source or answer, including at least 3 that the data cannot answer.
- **Retrieval or tool use is evaluated separately from the answers**: a hit rate at `k`, or the share of questions with the right tools.
- **Answers are checked**: correct, citing the right source, and refusing when they should. Every failure is classified as a retrieval (or tool) failure or a generation failure.
- **The cost and the time per question are reported**, with the total for the evaluation.

**3 — Excellent** · *at least three of the following apply*
- An LLM is used as a judge, and its grades are checked against your own on a sample.
- The evaluation set covers different kinds of questions (facts, comparisons, summaries, out of scope), and the results are broken down by kind.
- Prompt injection or misuse is tested, and the result reported.
- The result is turned into the user's terms: how often they can trust an answer, and how they can tell when not to.

---

## 6 | The app

**Learning outcome:** Turn a model or an LLM pipeline into an application a user can run, with Streamlit, so that it can be trusted and reproduced by someone else.

**Skill tags:** *Development Tools & Workflow: Web Development & Project Management*, *Development Tools & Workflow: Version Control & MLOps*

**0 — Incomplete** · *all of these apply*
- No app, or an app that does not run.
- The app does not use the model or the pipeline from the notebooks.

**1 — Fair** · *at least two of the following apply*
- The app runs only on the author's machine, or needs steps that are not written down.
- The model is retrained, or the index rebuilt, every time the page reruns.
- A key is written in the code.
- Errors (an empty input, an API failure) crash the app.

**2 — Good** · *all of these apply*
- **The app runs from a fresh clone** by following the README, with `requirements.txt` and the data or model steps listed.
- **It loads the saved model (or builds the pipeline) once**, with `st.cache_resource`, and reuses the functions in `src/`.
- **The output is understandable by the user**: the prediction with its confidence, or the answer with its sources.
- **Keys are read from `st.secrets` or `.env`**, both gitignored.

**3 — Excellent** · *at least three of the following apply*
- The app is deployed on Streamlit Community Cloud (or similar), with its secrets in the platform.
- Inputs are validated, and errors produce a clear message instead of a stack trace.
- The app shows the uncertainty or the limits of the result — a low-confidence warning, a "no answer found" state, the sources.
- The app logs what users ask or predict, for monitoring.

---

## 7 | Data visualisation

**Learning outcome:** Select and use appropriate data visualisation techniques that effectively communicate insights, and create informative visualisations using Python libraries.

**0 — Incomplete** · *at least two of these apply*
- No appropriate visualisation techniques used to communicate insights.
- Visualisations lack clarity; the metrics are not properly identified, defined or plotted.
- Visual design is not informative.

**1 — Fair** · *at least two of these apply*
- Fair use of visualisation to communicate insights, but chart-type selection and design need work.
- Visualisations have some clarity but need better organisation.
- Metrics need to be more clearly defined and plotted.
- Some plots lack detail or have formatting problems — missing titles, unlabelled axes, unreadable ticks.

**2 — Good** · *at least three of these apply*
- Solid use of visualisation for both the data (EDA) and the results: at least a comparison with the baseline and an error or failure analysis chart.
- Charts are clear, well organised and support decision-making.
- Chart types are appropriate to the data, with clear labelling: title, axis labels, units.
- Each chart carries a written takeaway — what it shows, in one sentence.

**3 — Excellent** · *all of these apply*
- Exceptional use of visualisation to communicate insights effectively.
- Highly effective, with a clear and intuitive layout that conveys the key insight at a glance.
- Metrics expertly defined, measured and plotted, with great attention to detail.
- Results shown in terms a stakeholder understands — complaints found, questions answered, euros per month — not only in scores.

---

## 8 | Code quality and structure

**Learning outcome:** Write clean, modular, efficient code following best practices, and maintain a clean and logical project structure.

**0 — Incomplete** · *at least three of these apply*
- Much unused code left in the project.
- No functions.
- Naming conventions not applied, making the code hard to read.
- Many hard-coded values or global variables.
- No consistent approach to naming, structure and organisation of files and folders.

**1 — Fair** · *at least four of these apply*
- Some unused code left in the project.
- Functions are either too large or do several things at once.
- Naming conventions barely applied.
- Some hard-coded values or global variables.
- Some files and folders are organised appropriately; others need work.

**2 — Good** · *at least four of these apply*
- Little unused code left.
- **Functions are modular and reusable, and live in `src/functions.py`**, imported by both the notebooks and the app.
- Naming conventions well applied.
- Few hard-coded values; the random seed, the paths and the model names are defined once.
- **The notebooks run top to bottom on a fresh kernel**, after `download_data.py`.

**3 — Excellent** · *all of these apply*
- No unused code left.
- Functions are cleanly modular and reusable, each doing one thing.
- Naming conventions applied consistently throughout.
- No hard-coded values or magic strings; paths and configuration are defined in one place.
- All files and folders organised appropriately, and the results are reproducible: the same numbers on a second run.

---

## 9 | Git and GitHub

**Learning outcome:** Save and track changes in the source code using Git and GitHub.

**0 — Incomplete** · *all of these apply*
- Zero or one commit in total.
- Commit messages provide no useful information.

**1 — Fair** · *all of these apply*
- At least two commits made during the project.
- Commit messages are unclear and ambiguous.

**2 — Good** · *at least two of these apply*
- Several commits made during the project, but fewer than one per project day.
- Commit messages are clear and accurately describe the changes.
- Working in a pair, both people have commits in the history.
- No data files, no large model files and **no keys** committed — `data/`, `models/`, `.env` and `.streamlit/secrets.toml` stayed ignored.

**3 — Excellent** · *all of these apply*
- **At least one commit per project day**, from each person if you are working in a pair.
- Atomic commits with accurate, precise descriptions, consistently.
- Branches used for development rather than committing everything straight to `main`.

---

## 10 | Documentation

**Learning outcome:** Document the project's features, configuration and technical specifications.

**0 — Incomplete** · *all of these apply*
- No attempt to document the project in a README.
- No documentation of the code or functions.

**1 — Fair** · *at least two of these apply*
- A partially completed README.
- Functions partially documented, with incomplete docstrings.
- Few comments explaining the rationale or logic behind the code.

**2 — Good** · *all of these apply*
- **A well-structured, clear README**, written for *your* project — the user and the question, the track, the data and its source, the metric, the baseline and the final result, the cost, the limitations, and how to run the app. The template brief has been replaced, not left in place.
- Functions documented with accurate docstrings.
- Enough comments to explain the rationale, logic and main ideas.

**3 — Excellent** · *all of these apply*
- A fully comprehensive, well-structured README that someone could follow from clone to a running app without asking a question.
- Functions documented with complete docstrings — parameters, returns, and the decisions the caller has to make.
- Clear, concise comments explaining purpose and functionality, addressing *why* rather than restating *what*.

---

## 11 | Presentation and demo

**Learning outcome:** Build a presentation and perform a demo to deliver your results.

Format: **7 minutes of slides plus a 3-minute live demo**, presented from your own machine by sharing your screen. Any slide tool. The demo is your Streamlit app, running.

**0 — Incomplete** · *at least two of these apply*
- The presentation lacks clear structure and purpose, making the results hard to follow.
- The demo is poorly executed and does not communicate the results.
- No storytelling, making the presentation hard to engage with.

**1 — Fair** · *at least two of these apply*
- The presentation has some structure, but would benefit from better organisation and clearer presentation of the findings behind the conclusions.
- The demo communicates the results adequately, but could be more polished, with better pacing.
- Storytelling is present but not used effectively.

**2 — Good** · *at least three of these apply*
- Clear structure and purpose, effectively communicating the results.
- The demo is engaging, well rehearsed, and stays within the allocated time: the app is used on an example the audience can follow.
- Storytelling techniques are well incorporated and add to the audience's engagement.
- Conclusions and next steps are included.
- The metric and the baseline are explained so a non-technical listener knows whether the product is any good, and what it costs to run.

**3 — Excellent** · *at least four of these apply*
- Highly compelling, with a clear message and a well-structured narrative.
- Flawlessly executed demo, showcasing the results memorably and using the time proficiently.
- Storytelling expertly used to build a narrative that deepens the audience's understanding.
- Conclusions and next steps included, alongside the strengths and limitations of the product and recommendations for further work.
- Visualisation used in an innovative and meaningful way to support the findings.

---

## Bringing your own data or topic

A pair working on their own data or documents is assessed on **exactly these criteria** — there is no separate, easier or harder rubric.

The conditions in [the brief](README.md#choose-your-data) are what keep that fair: a clear user and question, enough data for an honest evaluation, a licence that permits educational use, and no personal or confidential data sent to an API. Without them, criteria 4 and 5 have nothing honest to assess. Clear it with your teacher on launch day.

## Attribution

The Amazon Reviews 2023 data comes from a research paper whose authors ask to be cited: the exact reference is in [`data/README.md`](data/README.md). Name the source in your README and on your data slide, and do not commit or redistribute the data. A missing attribution counts against criterion 10.

---

## A note on flexibility

Carried over from Ironhack's project rubric, and it applies here too:

> In this agnostic project rubric, it is important to acknowledge that each project may have unique characteristics and requirements. As such, it is understood that not all learning outcomes or criteria points listed in the rubric will be applicable to every project.
>
> Students should focus on relevant outcomes and criteria for self-assessment, while teachers should consider project-specific aspects for evaluation. This approach ensures a tailored assessment aligned with the unique requirements of each project.

In practice: if a criterion genuinely does not apply to your project, say so in your README and explain why, rather than leaving a gap for the reader to interpret.
