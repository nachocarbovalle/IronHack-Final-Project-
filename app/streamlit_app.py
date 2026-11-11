"""Ask the Players: a RAG assistant over Steam reviews, and the project's presentation.

Run from the repo root:
    streamlit run app/streamlit_app.py
"""
import os
import sys
import re

import pandas as pd
import streamlit as st

sys.path.append(".")                          # so `src` is importable when run from the repo root
from src.functions import ROOT, answer_question, load_pipeline, retrieve

RESULTS = ROOT / "data" / "results"
LOG = ROOT / "logs" / "questions.csv"
WEAK_EVIDENCE = 0.7                           # closest-chunk distance above which the evidence is flagged as weak

st.set_page_config(page_title="Ask the Players", page_icon="🎮", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');

h1, h2, h3 { font-family: 'Press Start 2P', monospace !important; letter-spacing: 1px; }
h1 { font-size: 1.7rem !important; line-height: 1.5 !important;
     color: #05d9e8; text-shadow: 0 0 10px #05d9e8; }
h2, h3 { font-size: 0.95rem !important; line-height: 1.7 !important; color: #ff2a6d; }

div.stButton > button, div.stLinkButton > a {
    font-family: 'Press Start 2P', monospace; font-size: 0.65rem;
    border: 2px solid #05d9e8; border-radius: 0; box-shadow: 4px 4px 0 #ff2a6d;
}
div.stButton > button:hover { box-shadow: 2px 2px 0 #ff2a6d; transform: translate(2px, 2px); }

[data-testid="stMetricValue"] { font-family: 'VT323', monospace; font-size: 2.8rem;
                                color: #05d9e8; text-shadow: 0 0 6px #05d9e8; }
[data-testid="stSidebar"] { border-right: 2px solid #ff2a6d; }
</style>
""", unsafe_allow_html=True)

# ---------- Setup, done once ----------

def load_api_key():
    """Use the key from .env when it exists, otherwise from Streamlit secrets (for deployment)."""
    if os.getenv("OPENAI_API_KEY"):
        return
    try:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass                                  # get_client() will still try the .env file


@st.cache_resource(show_spinner="Loading the review index and the models…")
def get_pipeline():
    """The saved configuration, the Chroma index, the embedding model and the LLM client, loaded once."""
    load_api_key()
    return load_pipeline()


@st.cache_data
def get_games():
    return sorted(pd.read_csv(ROOT / "data" / "games.csv")["game"])


def log_question(game, question, out):
    """Append each question to a local CSV (gitignored), for monitoring what users ask."""
    LOG.parent.mkdir(exist_ok=True)
    row = pd.DataFrame([{
        "time": pd.Timestamp.now().isoformat(timespec="seconds"),
        "game": game, "question": question,
        "min_distance": out["min_distance"],
        "n_cited": len([c for c in str(out["cited_ids"]).split(";") if c]),
        "cost_usd": out["cost_usd"], "seconds": out["seconds"],
    }])
    row.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)

# ---------- Home ----------

def home():
    st.write("")
    st.title("🎮 Ask the Players")
    st.markdown("#### Nacho Carbó's Ironhack final project · Data Science & Machine Learning bootcamp, November 2026")
    st.write("")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Why this project?")
        st.markdown("""I decided to build a RAG assistant to answer questions about videogames based on real Steam reviews because of my passion for gaming. I have been a gamer for many years and more often than not I have found myself in situations where I wanted to know more about a game before buying it. Steam reviews are a great source of information, but they can be overwhelming and difficult to navigate. By building this assistant, I wanted to create a tool that would help gamers make informed decisions about their purchases, while also showcasing my skills in data science and machine learning.
""")
        st.subheader("What it does?")
        st.markdown("""
Ask a question about one of 23 PC games, and get an answer built from **real Steam reviews**,
with the reviews it used shown underneath, so every claim can be checked.
When the reviews cannot answer, it says so instead of guessing.
""")
    with right:
        _, _, _, both, _ = load_results()
        st.subheader("In numbers")
        a, b = st.columns(2)
        a.metric("Games", 23)
        b.metric("Reviews", "22,707")
        a.metric("Answer score", f"{both['score_rag'].mean():.2f}",
                 f"+{both['score_rag'].mean() - both['score_baseline'].mean():.2f} vs LLM alone")
        b.metric("Hallucinations", 0, "in 30 answers", delta_color="off")

    st.write("")
    st.subheader("Start here")
    cards = [
        (PRESENTATION_PAGE, "🎤", "The presentation", "The problem, the data, how it was built and evaluated, and where it fails."),
        (GAMES_PAGE, "🕹️", "The games", "The 23 games the assistant knows, with their release dates on Steam."),
        (ASSISTANT_PAGE, "🎮", "Ask the players", "The live assistant: pick a game, ask a question, check the sources."),
    ]
    for column, (page, icon, title, text) in zip(st.columns(3), cards):
        with column.container(border=True):
            st.markdown(f"**{icon} {title}**")
            st.caption(text)
            st.page_link(page, label="Open", icon="➡️")

# ---------- The assistant ----------

def number_citations(answer, hits):
    """Replace '[review 123456]' with short references '[1]', '[2]'… matching the numbered sources."""
    order = list(dict.fromkeys(hits["review_id"]))         # unique review ids, in retrieval order

    def to_number(match):
        rid = match.group(1)
        return f"[{order.index(rid) + 1}]" if rid in order else match.group(0)

    return re.sub(r"\[review (\d+)\]", to_number, answer)

def show_answer(result):
    out, hits = result["out"], result["hits"]
    st.markdown(number_citations(out["answer"], hits))
    st.caption(f"{out['seconds']} s · ${out['cost_usd']:.5f} · "
               f"closest review distance {out['min_distance']} (lower is closer)")

    if out["min_distance"] > WEAK_EVIDENCE:
        st.warning("The closest reviews are far from your question, so the evidence may be weak. "
                   "Check the sources below before trusting this answer.")
    st.info("Based on 5 Steam reviews, shown below. It can miss minority opinions, "
            "and older reviews may not describe the game as it is today.")

    cited = set(str(out["cited_ids"]).split(";"))
    with st.expander(f"Sources: the {len(hits)} reviews the answer was built from", expanded=True):
        for h in hits.itertuples():
            verdict = "👍 Recommended" if h.voted_up else "👎 Not recommended"
            edited = " · edited later" if h.edited else ""
            used = " · **cited in the answer**" if h.review_id in cited else ""
            n = list(dict.fromkeys(hits["review_id"])).index(h.review_id) + 1
            st.markdown(f"**[{n}]** Review {h.review_id} · {verdict} · {h.playtime_h:.0f} h played · "
                        f"written {h.created_date}{edited}{used}")
            st.caption(h.text[:600] + ("…" if len(h.text) > 600 else ""))


def assistant():
    st.title("🎮 Ask the players")
    st.write("Questions about 23 PC games, answered from 22,707 English Steam reviews "
             "(collected on 10 November 2026), with the reviews it used.")

    try:
        config, collection, embedder, client = get_pipeline()
    except Exception as e:
        st.error("The assistant could not start. If this is a fresh clone, build the index first by "
                 f"running `notebooks/02_build.ipynb` (see the README). Details: {type(e).__name__}")
        st.stop()

    game = st.selectbox("Game", get_games(), index=None, placeholder="Choose a game")
    question = st.text_area("Your question", max_chars=300,
                            placeholder="e.g. Is it a good first game for someone new to the genre?")

    if st.button("Ask the players", type="primary"):
        if not game:
            st.warning("Choose a game first.")
        elif len(question.split()) < 3:
            st.warning("Write a question of at least a few words.")
        else:
            with st.spinner("Reading the reviews…"):
                try:
                    out = answer_question(question, collection, embedder, client,
                                          k=config["k"], game=game)
                    hits = retrieve(question, collection, embedder, k=config["k"], game=game)
                except Exception as e:
                    st.error(f"The answer could not be generated ({type(e).__name__}). "
                             "Check the API key in `.env` and your connection, then try again.")
                    st.stop()
            st.session_state["result"] = {"out": out, "hits": hits}
            log_question(game, question, out)

    if "result" in st.session_state:
        st.divider()
        show_answer(st.session_state["result"])

# ---------- The games ----------

@st.cache_data
def get_games_table():
    games = pd.read_csv(ROOT / "data" / "games.csv", parse_dates=["steam_release"])
    return games.sort_values("steam_release")


def games_page():
    st.title("🕹️ The games")
    st.write("The 23 games the assistant knows, ordered by their release on Steam.")
    games = get_games_table()
    columns = st.columns(3)
    for i, g in enumerate(games.itertuples()):
        with columns[i % 3].container(border=True):
            image = getattr(g, "header_image", None)
            if isinstance(image, str):
                st.image(image)
            st.markdown(f"**{g.game}**")
            st.caption(f"Released on Steam: {g.steam_release:%d %B %Y}")
            st.link_button("Steam page", f"https://store.steampowered.com/app/{g.app_id}")

# ---------- The presentation ----------

@st.cache_data
def load_results():
    """The saved evaluation results, so the slides show exactly the numbers of notebook 03."""
    eval_set = pd.read_csv(ROOT / "data" / "eval_questions.csv")
    rag = pd.read_csv(RESULTS / "rag_final_graded.csv")
    base = pd.read_csv(RESULTS / "baseline_llm_only_graded.csv")
    failures = pd.read_csv(RESULTS / "failure_analysis.csv")
    answerable = eval_set.loc[eval_set["answerable"], "id"]
    both = rag[["id", "type", "score"]].merge(base[["id", "score"]], on="id",
                                              suffixes=("_rag", "_baseline"))
    return rag, base, failures, both[both["id"].isin(answerable)], answerable


def heading(title, subtitle=None):
    st.title(title)
    if subtitle:
        st.subheader(subtitle)
    st.write("")


def slide_title():
    st.write("")
    st.title("🎮 Ask the Players")
    st.markdown("#### A RAG assistant that answers questions about PC games from Steam reviews, and shows its sources")
    st.write("")
    st.markdown("**Nacho Carbó** · Ironhack Data Science & Machine Learning · Final project, November 2026")


def slide_problem():
    heading("The user and the problem")
    left, right = st.columns(2)
    with left:
        st.markdown("""
**The user:** a PC player deciding whether to buy a game.

**The problem:** Steam shows one score and *thousands* of unsorted reviews.
Reading enough of them to answer a specific question takes far too long.

**The decision:** buy now, wait for a patch or a sale, or skip it.
""")
    with right:
        st.markdown("**Questions a player really asks:**")
        st.info("Is Hades a good first roguelike?")
        st.info("Does Borderlands 4 still have performance problems?")
        st.info("How does Ghost of Tsushima run on the Steam Deck?")


def slide_data():
    heading("The data", "23 games · 22,707 English Steam reviews, from Valve's official review API")
    left, right = st.columns([1, 1])
    with left:
        st.markdown("""
- **Stratified sampling:** each game's life on Steam split into 5 time windows,
  200 reviews each, in the window's **real** share of positive and negative reviews.
- **Why:** sorting by "most helpful" alone made the sample **11–30 points more negative**
  than reality: complaints are long and collect helpful votes.
- **Traps found and fixed:** reviews in other languages (detected with `langdetect`),
  copy-pasted reviews, Steam markup, reviews too long for the embedding model (split into chunks).
- **A trap kept in mind:** Steam stores only the latest version of an edited review (18.5% of them).
""")
    with right:
        bias = pd.DataFrame({
            "Real share of positive reviews": [87.4, 84.3, 80.7, 78.0, 82.1],
            "Sample sorted by helpfulness": [76.5, 54.5, 58.0, 55.0, 59.5],
        }, index=["1", "2", "3", "4", "5"])
        st.markdown("**Crimson Desert: % positive by time window**")
        st.bar_chart(bias, stack=False, x_label="Time window", y_label="% positive",
                     color=["#05d9e8", "#ff2a6d"])


def slide_metric():
    _, base, _, both, answerable = load_results()
    heading("The metric and the baseline")
    left, right = st.columns(2)
    with left:
        st.markdown("""
**Which mistake costs most?**
1. 🔴 **A confident wrong answer**: the player pays for something invented
2. 🟠 **An answer without a source**: impossible to check
3. 🟢 **A refusal**: frustrating, but honest

**30 test questions**, written *before* building the system, in 8 types
(facts, changes over time, comparisons, recent games, hard cases…),
including **4 the reviews cannot answer**. Graded with a rubric fixed in advance.
""")
    with right:
        st.markdown("**The baseline: the same LLM, with no reviews**")
        a, b = st.columns(2)
        a.metric("Score (26 answerable)", f"{both['score_baseline'].mean():.2f}")
        b.metric("Refused", f"{int(base[base['id'].isin(answerable)]['refused'].sum())} of 26")
        a.metric("Hallucinations", int(base["hallucinated"].sum()))
        b.metric("Random retrieval hit rate", "0.52")
        st.caption("Honest, but not useful: it refuses half the questions, and when it answers, it is generic.")


def slide_pipeline():
    heading("What I built")
    st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef2f7", fontname="Helvetica", fontsize=11];
  q [label="Question\\n+ selected game"];
  e [label="Embed the question\\nall-MiniLM-L6-v2"];
  r [label="Chroma index\\n44,477 review chunks\\ntop 5, same game only"];
  p [label="Prompt: answer ONLY\\nfrom these reviews,\\ncite them, or say\\n'I don't know'"];
  l [label="gpt-5.4-nano"];
  a [label="Answer with [1]…[5]\\n+ the 5 reviews"];
  q -> e -> r -> p -> l -> a;
}""")
    st.markdown("""
**One design choice compared: the game filter.** Without it, the hit rate was the same (0.85),
because every test question names its game. I kept it anyway: real users will not always name the game.
""")


def slide_result():
    rag, base, _, both, _ = load_results()
    heading("The result")
    rag_score, base_score = both["score_rag"].mean(), both["score_baseline"].mean()
    a, b, c, d = st.columns(4)
    a.metric("Answer score", f"{rag_score:.2f}", f"+{rag_score - base_score:.2f} vs LLM alone")
    b.metric("Out-of-scope refused", f"{int(rag[~rag['id'].isin(both['id'])]['score'].sum())} of 4", "+1")
    c.metric("Hallucinations", int(rag["hallucinated"].sum()), "0", delta_color="off")
    d.metric("Retrieval hit rate", "0.85", "+0.33 vs random")
    by_type = (both.groupby("type")[["score_rag", "score_baseline"]].mean()
                   .rename(columns={"score_rag": "RAG system", "score_baseline": "LLM alone"})
                   .sort_values("RAG system"))
    st.bar_chart(by_type, stack=False, horizontal=True, x_label="Score (0 to 1)", y_label="",
                 color=["#ff2a6d", "#05d9e8"])
    st.caption("Biggest gains where the LLM alone knows nothing: changes over time, recent games, specific complaints.")


def slide_failures():
    _, _, failures, both, _ = load_results()
    heading("Where it goes wrong")
    change = both["score_rag"] - both["score_baseline"]
    left, right = st.columns([1, 1])
    with left:
        st.markdown(f"**Question by question:** {(change > 0).sum()} better · "
                    f"{(change == 0).sum()} the same · {(change < 0).sum()} worse")
        counts = failures["failure"].value_counts().rename("Imperfect answers")
        st.bar_chart(counts, x_label="Cause", y_label="Answers", color="#05d9e8")
    with right:
        st.markdown("""
**Retrieval (12): the evidence was not among the 5 reviews**
- Minority complaints are crowded out by hundreds of positive reviews
- "Complaints about Spider-Man" retrieves… positive reviews of Spider-Man

**Generation (6): the evidence was there, the reasoning failed**
- *"Players mention the DLC positively, saying they can't wait for it"*:
  anticipation presented as reception
- 2023 reviews used as evidence that problems *still* exist
""")


def slide_cost():
    rag, base, _, _, _ = load_results()
    heading("Cost and limits")
    a, b, c = st.columns(3)
    a.metric("Cost per question", f"${rag['cost_usd'].mean():.5f}", f"LLM alone: ${base['cost_usd'].mean():.5f}",
             delta_color="off")
    b.metric("Time per answer", f"{rag['seconds'].mean():.1f} s", "retrieval: 16 ms", delta_color="off")
    c.metric("1,000 questions a day", f"${rag['cost_usd'].mean() * 30_000:.2f} / month")
    st.markdown("""
**Do not trust it for:**
- *"Is there anything wrong with this game?"*: minority complaints are often missing
- Specific angles (the late game, one mode), brand-new content, *"is it still…?"* questions
- Any single answer: the same question can get different answers on two runs

**It never invented a fact or a source in 30 answers, and refused all 4 questions it could not answer.**
""")


def slide_conclusions():
    heading("Conclusions and next steps")
    left, right = st.columns(2)
    with left:
        st.markdown("""
**Would I ship it? Yes, as a beta,** for players who want *evidence* before buying, not a verdict.

It nearly **doubles** the LLM's score **without losing its honesty**, for about $12 a month.

**With this warning:** *based on 5 reviews, shown below; it can miss minority opinions,
and older reviews may not describe the game today.*
""")
    with right:
        st.markdown("""
**Next, in order of impact:**
1. **Better retrieval** (12 of 18 failures): keyword + semantic search, more chunks re-ranked
2. **Tell the model today's date**, so it can judge how old its evidence is
3. **Refuse when the closest review is far**, with a threshold set on *new* questions
4. **Two games at once** for comparisons
5. **A bigger evaluation**: more questions, a second grader, several runs
""")


def slide_closing():
    st.write("")
    st.title("Thank you! 🎮")
    st.subheader("Now, let's ask the players.")
    st.write("")
    st.page_link(ASSISTANT_PAGE, label="Open the live demo", icon="🎮")
    st.write("")
    st.caption("Nacho Carbó · Ask the Players · Ironhack DSML final project")


SLIDES = [slide_title, slide_problem, slide_data, slide_metric, slide_pipeline,
          slide_result, slide_failures, slide_cost, slide_conclusions, slide_closing]


def presentation():
    if "slide" not in st.session_state:
        st.session_state.slide = 0
    i = st.session_state.slide
    SLIDES[i]()

    st.divider()
    left, middle, right = st.columns([1, 6, 1])
    if left.button("← Previous", disabled=i == 0):
        st.session_state.slide -= 1
        st.rerun()
    middle.progress((i + 1) / len(SLIDES), text=f"Slide {i + 1} of {len(SLIDES)}")
    if right.button("Next →", disabled=i == len(SLIDES) - 1):
        st.session_state.slide += 1
        st.rerun()

# ---------- Navigation ----------

HOME_PAGE = st.Page(home, title="Home", icon="🏠", default=True)
PRESENTATION_PAGE = st.Page(presentation, title="Presentation", icon="🎤", url_path="presentation")
GAMES_PAGE = st.Page(games_page, title="The games", icon="🕹️", url_path="games")
ASSISTANT_PAGE = st.Page(assistant, title="Ask the players", icon="🎮", url_path="ask")

pages = st.navigation([HOME_PAGE, PRESENTATION_PAGE, GAMES_PAGE, ASSISTANT_PAGE])

with st.sidebar:
    st.divider()
    st.markdown("**Nacho Carbó**  \nIronhack DSML · Final project")
    st.markdown("[🐙 Code on GitHub](https://github.com/nachocarbovalle/IronHack-Final-Project-)")
    st.caption("Reviews collected from Steam on 10 November 2026")

pages.run()

