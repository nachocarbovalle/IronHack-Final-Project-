"""The app you demo on Friday. Run it from the repo root:

    streamlit run app/streamlit_app.py

This is a skeleton: replace every TODO with your own product. Day 3 of Week 8
(Streamlit, from notebook to app) shows each piece.
"""

import sys

import streamlit as st

sys.path.append(".")                       # so `src` is importable when you run from the repo root
from src.functions import *                # noqa: E402,F401  the same functions as your notebooks

st.set_page_config(page_title="TODO: your product's name", page_icon="🛍️")
st.title("TODO: your product's name")
st.caption("TODO: one sentence for the user: what it does, and on what data.")


@st.cache_resource
def load_core():
    """Load the saved model, or build the pipeline, ONCE per session (not on every rerun).

    Model track: return joblib.load(MODELS / "your_model.joblib")
    LLM track:   return the Chroma collection and anything else the pipeline needs.
                 Read the key with st.secrets["OPENAI_API_KEY"] or from .env, never from the code.
    """
    return None


core = load_core()

text = st.text_area("TODO: what the user types", placeholder="TODO: an example input")

if st.button("TODO: the action"):
    if not text.strip():
        st.warning("Please write something first.")
    else:
        # TODO: call your functions from src/functions.py, then show the result:
        # - model track: the prediction and its confidence (st.metric, st.progress, st.bar_chart)
        # - LLM track: the answer and its sources (st.markdown, st.expander for each source)
        st.info("TODO: show the result here.")

with st.sidebar:
    st.header("About")
    st.write("TODO: the data (with its source), the model or pipeline, the evaluation result, and the limits.")
