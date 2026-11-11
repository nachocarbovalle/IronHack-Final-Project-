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


# ---------- The presentation (step 2) ----------

def presentation():
    st.title("🎤 Presentation")
    st.write("Coming in step 2.")


# ---------- Navigation ----------

pages = st.navigation([
    st.Page(presentation, title="Presentation", icon="🎤"),
    st.Page(assistant, title="Ask the players", icon="🎮"),
])
pages.run()

