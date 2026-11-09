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

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"

RANDOM_STATE = 42
LLM_MODEL = "gpt-5.4-nano"                                    # LLM track: one place to change the model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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
