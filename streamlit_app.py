"""
Twitter Sentiment Analysis — Streamlit Dashboard
Twitter-native dark theme with interactive NLP analysis

Loads pre-trained models from the models/ directory.
Run the notebook's Section 17 cell first to generate those files.
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import nltk
for pkg in ["vader_lexicon", "stopwords", "punkt", "punkt_tab", "wordnet"]:
    nltk.download(pkg, quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# sklearn imports are required even if not training — joblib needs them
# available in the namespace to deserialise the .pkl objects
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer   # noqa: F401
from sklearn.linear_model import LogisticRegression                             # noqa: F401
from sklearn.naive_bayes import MultinomialNB                                   # noqa: F401
from sklearn.preprocessing import LabelEncoder                                  # noqa: F401
import lightgbm as lgb                                                          # noqa: F401
import xgboost as xgb                                                           # noqa: F401
import joblib
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tweet Sentiment Analyzer",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Chirp:wght@400;700&family=DM+Sans:ital,wght@0,300;0,400;0,600;0,700;1,400&display=swap');

:root {
    --bg-primary:   #15202B;
    --bg-card:      #1E2D3D;
    --bg-elevated:  #253341;
    --blue:         #1D9BF0;
    --blue-hover:   #1A8CD8;
    --blue-glow:    rgba(29,155,240,0.15);
    --green:        #00BA7C;
    --red:          #F4212E;
    --yellow:       #FFD400;
    --muted:        #8899A6;
    --border:       rgba(255,255,255,0.08);
    --text:         #E7E9EA;
    --text-dim:     #8899A6;
    --radius:       16px;
    --radius-sm:    10px;
}

/* App background */
.stApp, .stApp > div {
    background: var(--bg-primary) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

/* Hide default streamlit decorations */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 4rem !important; max-width: 1200px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown { color: var(--text-dim) !important; }

/* Headings */
h1, h2, h3, h4 {
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
    letter-spacing: -0.3px;
}

/* Input fields */
.stTextArea textarea, .stTextInput input {
    background: var(--bg-elevated) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    transition: border-color 0.2s;
    caret-color: white !important;  /* ← add this line */
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px var(--blue-glow) !important;
}

/* Primary buttons */
.stButton > button {
    background: var(--blue) !important;
    color: white !important;
    border: none !important;
    border-radius: 9999px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 0.55rem 1.6rem !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: background 0.2s, transform 0.1s !important;
    cursor: pointer;
}
.stButton > button:hover {
    background: var(--blue-hover) !important;
    transform: translateY(-1px);
}

/* Select boxes */
.stSelectbox > div > div {
    background: var(--bg-elevated) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: var(--radius-sm) !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.2rem !important;
}
[data-testid="stMetricValue"] { color: var(--text) !important; font-size: 2rem !important; }
[data-testid="stMetricLabel"] { color: var(--text-dim) !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-dim) !important;
    font-weight: 600;
    font-size: 15px;
    border-radius: 0 !important;
    padding: 0.8rem 1.5rem !important;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    color: var(--blue) !important;
    border-bottom: 2px solid var(--blue) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 1.5rem 0 !important; }

/* File uploader */
[data-testid="stFileUploadDropzone"] {
    background: var(--bg-elevated) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-dim) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: var(--radius-sm) !important; }
.stDataFrame { background: var(--bg-card) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* Spinner */
.stSpinner > div { border-color: var(--blue) transparent transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─── REUSABLE HTML CARDS ────────────────────────────────────────────────────
def sentiment_badge(label: str) -> str:
    colors = {
        "Positive": ("#00BA7C", "#0D2B21", "↑"),
        "Negative": ("#F4212E", "#2B0D10", "↓"),
        "Neutral":  ("#1D9BF0", "#0D1E2B", "→"),
    }
    c, bg, icon = colors.get(label, ("#8899A6", "#1E2D3D", "•"))
    return f"""
    <span style="
        display:inline-flex; align-items:center; gap:6px;
        background:{bg}; color:{c};
        border:1.5px solid {c}40; border-radius:9999px;
        padding:5px 14px; font-weight:700; font-size:14px;
        letter-spacing:0.3px;
    ">{icon} {label}</span>"""


def tweet_card(text: str, sentiment: str, confidence: float, scores: dict) -> str:
    colors = {
        "Positive": ("#00BA7C", "#0D2B21", "↑"),
        "Negative": ("#F4212E", "#2B0D10", "↓"),
        "Neutral":  ("#1D9BF0", "#0D1E2B", "→"),
    }
    c, bg, icon = colors.get(sentiment, ("#8899A6", "#1E2D3D", "•"))
    badge = (
        f'<span style="display:inline-flex;align-items:center;gap:6px;'
        f'background:{bg};color:{c};border:1.5px solid {c}40;border-radius:9999px;'
        f'padding:5px 14px;font-weight:700;font-size:14px;letter-spacing:0.3px;">'
        f'{icon} {sentiment}</span>'
    )
    bar_colors = {"Positive": "#00BA7C", "Negative": "#F4212E", "Neutral": "#1D9BF0"}

    bars = ""
    for s, p in scores.items():
        pct = round(p * 100, 1)
        bar_color = bar_colors.get(s, '#8899A6')
        bars += (
            f'<div style="margin-bottom:8px;">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
            f'<span style="font-size:12px;color:#8899A6;font-weight:600;">{s.upper()}</span>'
            f'<span style="font-size:12px;color:#E7E9EA;font-weight:700;">{pct}%</span>'
            f'</div>'
            f'<div style="background:#253341;border-radius:9999px;height:6px;overflow:hidden;">'
            f'<div style="height:100%;width:{pct}%;background:{bar_color};border-radius:9999px;"></div>'
            f'</div>'
            f'</div>'
        )

    tweet_text = text[:280] + ("…" if len(text) > 280 else "")
    return (
        f'<div style="background:#1E2D3D;border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:20px 24px;margin:12px 0;">'
        f'<div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:16px;">'
        f'<div style="width:40px;height:40px;background:#253341;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0;">🐦</div>'
        f'<div style="flex:1;"><p style="margin:0;font-size:15px;line-height:1.5;color:#E7E9EA;">{tweet_text}</p></div>'
        f'</div>'
        f'<div style="display:flex;align-items:center;justify-content:space-between;padding-top:12px;border-top:1px solid rgba(255,255,255,0.06);">'
        f'<div>{badge}</div>'
        f'<span style="color:#8899A6;font-size:13px;">Confidence: <b style="color:#E7E9EA;">{confidence*100:.1f}%</b></span>'
        f'</div>'
        f'<div style="margin-top:16px;">{bars}</div>'
        f'</div>'
    )


def model_card(name: str, neg_recall: float, macro_f1: float, best: bool = False) -> str:
    border = "border:1.5px solid #1D9BF0;" if best else "border:1px solid rgba(255,255,255,0.08);"
    badge = '<span style="background:#1D9BF040;color:#1D9BF0;border-radius:9999px;padding:2px 10px;font-size:11px;font-weight:700;margin-left:8px;">BEST</span>' if best else ""
    nr_pct = neg_recall * 100
    mf_pct = macro_f1 * 100
    return f"""
    <div style="background:#1E2D3D;{border}border-radius:16px;padding:18px 20px;margin-bottom:10px;">
        <div style="display:flex;align-items:center;margin-bottom:14px;">
            <span style="font-weight:700;font-size:15px;color:#E7E9EA;">{name}</span>{badge}
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
            <div>
                <div style="font-size:11px;color:#8899A6;font-weight:600;margin-bottom:6px;">NEG RECALL</div>
                <div style="background:#253341;border-radius:9999px;height:8px;margin-bottom:4px;">
                    <div style="height:100%;width:{nr_pct}%;background:#F4212E;border-radius:9999px;"></div>
                </div>
                <div style="font-weight:700;color:#F4212E;">{neg_recall:.4f}</div>
            </div>
            <div>
                <div style="font-size:11px;color:#8899A6;font-weight:600;margin-bottom:6px;">MACRO F1</div>
                <div style="background:#253341;border-radius:9999px;height:8px;margin-bottom:4px;">
                    <div style="height:100%;width:{mf_pct}%;background:#1D9BF0;border-radius:9999px;"></div>
                </div>
                <div style="font-weight:700;color:#1D9BF0;">{macro_f1:.4f}</div>
            </div>
        </div>
    </div>"""


# ─── NLP PREPROCESSING  (mirrors notebook's preprocess_tweet exactly) ────────
# Keep negation words — they are critical for sentiment direction.
STOP_WORDS = set(stopwords.words("english")) - {"not", "no", "never", "nor", "n't"}
_lemmatizer = WordNetLemmatizer()


def preprocess_tweet(text: str) -> str:
    """
    Clean and normalise a single tweet for ML models.

    Pipeline (identical to the training notebook):
        lowercase → remove URLs → remove @mentions → expand #hashtags
        → strip non-alpha → tokenise → remove stopwords → lemmatise
        → filter short tokens (len > 2)

    Returns a space-joined string of clean tokens.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)       # URLs
    text = re.sub(r"@\w+", "", text)                    # @mentions
    text = re.sub(r"#(\w+)", r"\1", text)               # #hashtags → words
    text = re.sub(r"[^a-z\s]", " ", text)               # special chars
    tokens = word_tokenize(text)
    tokens = [
        _lemmatizer.lemmatize(t)
        for t in tokens
        if t not in STOP_WORDS and len(t) > 2
    ]
    return " ".join(tokens)


# ─── MODEL LOADING ──────────────────────────────────────────────────────────
MODELS_DIR = Path("models")

REQUIRED_FILES = {
    "lr_model.pkl":      "Logistic Regression model",
    "nb_model.pkl":      "Naive Bayes model",
    "lgb_model.pkl":     "LightGBM model",
    "xgb_model.pkl":     "XGBoost model",
    "tfidf.pkl":         "TF-IDF vectorizer",
    "cv.pkl":            "CountVectorizer",
    "label_encoder.pkl": "Label encoder",
}


def _missing_files() -> list[str]:
    return [f for f in REQUIRED_FILES if not (MODELS_DIR / f).exists()]


@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    """
    Load all pre-trained artefacts from the models/ directory.

    Expected files (generated by the notebook's Section 17 cell):
        models/lr_model.pkl       — LogisticRegression  (TF-IDF, 3-class)
        models/nb_model.pkl       — MultinomialNB        (CountVec, binary)
        models/lgb_model.pkl      — LGBMClassifier       (TF-IDF, 3-class)
        models/xgb_model.pkl      — XGBClassifier        (TF-IDF, 3-class)
        models/tfidf.pkl          — fitted TfidfVectorizer
        models/cv.pkl             — fitted CountVectorizer
        models/label_encoder.pkl  — fitted LabelEncoder

    Returns a dict with keys: lr, nb, lgb, xgb, vader, tfidf, cv, le
    """
    return {
        "lr":    joblib.load(MODELS_DIR / "lr_model.pkl"),
        "nb":    joblib.load(MODELS_DIR / "nb_model.pkl"),
        "lgb":   joblib.load(MODELS_DIR / "lgb_model.pkl"),
        "xgb":   joblib.load(MODELS_DIR / "xgb_model.pkl"),
        "tfidf": joblib.load(MODELS_DIR / "tfidf.pkl"),
        "cv":    joblib.load(MODELS_DIR / "cv.pkl"),
        "le":    joblib.load(MODELS_DIR / "label_encoder.pkl"),
        "vader": SentimentIntensityAnalyzer(),
    }


# ─── PREDICTION HELPERS ──────────────────────────────────────────────────────
def predict_single(text: str, model_name: str, models: dict) -> tuple:
    """
    Returns (predicted_label, confidence, {label: probability}) for one tweet.

    For VADER the raw pos/neg/neu scores are renormalised to sum to 1.
    For LR, LightGBM, and XGBoost the sklearn-compatible predict_proba is used
    against TF-IDF features. For NB, CountVectorizer features are used.
    """
    le = models["le"]
    classes = le.classes_   # e.g. ["Negative", "Neutral", "Positive"]

    if model_name == "VADER (Rule-Based)":
        raw = models["vader"].polarity_scores(text)
        pos, neg, neu = raw["pos"], raw["neg"], raw["neu"]
        total = pos + neg + neu + 1e-9
        prob_map = {c: 0.0 for c in classes}
        prob_map["Positive"] = pos / total
        prob_map["Negative"] = neg / total
        prob_map["Neutral"]  = neu / total
        predicted  = max(prob_map, key=prob_map.get)
        confidence = prob_map[predicted]
        return predicted, confidence, prob_map

    clean = preprocess_tweet(text)

    # Models that use TF-IDF features (multiclass, 3-class)
    TFIDF_MODELS = {
        "Logistic Regression": "lr",
        "LightGBM":            "lgb",
        "XGBoost":             "xgb",
    }

    if model_name in TFIDF_MODELS:
        clf   = models[TFIDF_MODELS[model_name]]
        X     = models["tfidf"].transform([clean])
        probs = clf.predict_proba(X)[0]
    else:  # Naive Bayes — uses CountVectorizer
        clf   = models["nb"]
        X     = models["cv"].transform([clean])
        probs = clf.predict_proba(X)[0]

    # Use the model's own .classes_ to map probs — critical for binary NB
    # (NB has 2 classes; le has 3, so using le indices would cause IndexError)
    model_classes = [le.inverse_transform([c])[0] for c in clf.classes_]
    prob_map      = {c: 0.0 for c in classes}          # start: all classes at 0
    for cls_label, p in zip(model_classes, probs):
        prob_map[cls_label] = float(p)                 # fill only known classes

    predicted  = model_classes[int(np.argmax(probs))]
    confidence = float(np.max(probs))
    return predicted, confidence, prob_map


def batch_predict(texts, model_name: str, models: dict) -> pd.DataFrame:
    rows = []
    for t in texts:
        pred, conf, scores = predict_single(str(t), model_name, models)
        rows.append({
            "tweet":      t,
            "sentiment":  pred,
            "confidence": round(conf * 100, 1),
            **{f"p_{k}": round(v, 3) for k, v in scores.items()},
        })
    return pd.DataFrame(rows)


def make_wordcloud_data(tweets, sentiments, target: str) -> list[tuple]:
    filtered = [t for t, s in zip(tweets, sentiments) if s == target]
    words = []
    for t in filtered:
        words.extend(preprocess_tweet(str(t)).split())
    return Counter(words).most_common(50)


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 20px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
            <span style="font-size:28px;">𝕏</span>
            <div>
                <div style="font-size:17px;font-weight:700;color:#E7E9EA;">Sentiment Analyzer</div>
                <div style="font-size:12px;color:#8899A6;">NLP Dashboard v2.0</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:11px;font-weight:700;color:#8899A6;letter-spacing:1px;margin-bottom:10px;">MODEL</p>', unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Choose model",
        ["Logistic Regression", "Naive Bayes", "LightGBM", "XGBoost", "VADER (Rule-Based)"],
        label_visibility="collapsed",
    )

    model_descriptions = {
        "Logistic Regression": "🔵 **Multiclass** · TF-IDF features · Balanced class weights · Neg Recall 0.757 · F1 0.706",
        "Naive Bayes":         "🟢 **Binary** · CountVectorizer · Fastest inference · Neg Recall 0.824 · F1 0.727",
        "LightGBM":            "🟠 **Multiclass** · TF-IDF features · Leaf-wise boosting · Neg Recall 0.721 · F1 0.698",
        "XGBoost":             "🔴 **Multiclass** · TF-IDF features · Level-wise boosting · Neg Recall 0.777 · F1 0.745 ⭐ Recommended",
        "VADER (Rule-Based)":  "🟡 **Rule-based** · No training needed · Lexicon-based · Neg Recall 0.389 · F1 0.481",
    }
    st.markdown(f'<p style="font-size:13px;color:#8899A6;line-height:1.5;margin-top:8px;">{model_descriptions[model_choice]}</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:11px;font-weight:700;color:#8899A6;letter-spacing:1px;margin-bottom:10px;">ABOUT</p>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:13px;color:#8899A6;line-height:1.6;">
    Analyzes Twitter sentiment about <b style="color:#E7E9EA;">Apple & Google</b> products.
    Models trained on a combined dataset of 19,000+ tweets.<br><br>
    <b style="color:#E7E9EA;">Primary metric:</b> Negative recall<br>
    <b style="color:#E7E9EA;">Secondary metric:</b> Macro F1
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <p style="font-size:11px;color:#8899A6;line-height:1.6;">
    Built by Faith Ng'endo · Allan Muchiri · Anthony Njeru · William Nyawir · Sarah Owendi
    </p>
    """, unsafe_allow_html=True)


# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #1E2D3D 0%, #15202B 60%, #0D1B26 100%);
    border: 1px solid rgba(29,155,240,0.2);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
">
    <div style="
        position:absolute; top:-40px; right:-40px;
        width:200px; height:200px;
        background: radial-gradient(circle, rgba(29,155,240,0.12) 0%, transparent 70%);
        border-radius: 50%;
    "></div>
    <div style="display:flex; align-items:center; gap:16px; position:relative;">
        <div style="
            width:52px; height:52px;
            background: linear-gradient(135deg, #1D9BF0, #1A73E8);
            border-radius:14px; display:flex; align-items:center;
            justify-content:center; font-size:26px; box-shadow:0 4px 20px rgba(29,155,240,0.4);
        ">🐦</div>
        <div>
            <h1 style="margin:0; font-size:24px; font-weight:800; color:#E7E9EA; letter-spacing:-0.5px;">
                Tweet Sentiment Analyzer
            </h1>
            <p style="margin:4px 0 0; color:#8899A6; font-size:14px;">
                NLP-powered sentiment analysis for Apple & Google product tweets
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── LOAD MODELS (with graceful error if .pkl files are missing) ──────────────
missing = _missing_files()
if missing:
    st.error(
        "**Pre-trained model files not found.**\n\n"
        "Add the **Section 17 cell** from `save_models_cell.py` to the end of your "
        "notebook and run it. That will create a `models/` folder next to this app "
        "containing the following files:\n\n"
        + "\n".join(f"- `models/{f}` — {REQUIRED_FILES[f]}" for f in missing)
        + "\n\nThen restart the Streamlit app."
    )
    st.stop()

with st.spinner("Loading models…"):
    models = load_models()


# ─── TABS ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["  🔍  Analyze Tweet", "  📊  Batch Analysis", "  🏆  Model Comparison", "  📈  Insights"]
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — SINGLE TWEET ANALYZER                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab1:
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown('<p style="font-size:13px;font-weight:700;color:#8899A6;letter-spacing:0.8px;margin-bottom:8px;">TWEET TEXT</p>', unsafe_allow_html=True)

        tweet_input = st.text_area(
            "Tweet input",
            placeholder="What's happening? Type or paste a tweet here…",
            height=140,
            label_visibility="collapsed",
        )

        analyze_btn = st.button("Analyze Sentiment →", use_container_width=True, type="primary")

    with col_result:
        if analyze_btn and tweet_input.strip():
            pred, conf, scores = predict_single(tweet_input, model_choice, models)

            st.markdown(tweet_card(tweet_input, pred, conf, scores), unsafe_allow_html=True)

            categories    = list(scores.keys())
            values        = [scores[c] * 100 for c in categories]
            values_closed = values + [values[0]]
            cats_closed   = categories + [categories[0]]

            fig = go.Figure(go.Scatterpolar(
                r=values_closed,
                theta=cats_closed,
                fill="toself",
                fillcolor="rgba(29,155,240,0.12)",
                line=dict(color="#1D9BF0", width=2),
                marker=dict(size=7, color="#1D9BF0"),
            ))
            fig.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 100],
                                   gridcolor="rgba(255,255,255,0.06)",
                                   tickcolor="rgba(0,0,0,0)",
                                   tickfont=dict(color="#8899A6", size=9)),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.06)",
                                    tickfont=dict(color="#E7E9EA", size=13, family="DM Sans")),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=40, r=40, t=20, b=20),
                height=240,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        elif analyze_btn and not tweet_input.strip():
            st.markdown("""
            <div style="background:#253341;border:1px solid rgba(255,255,255,0.08);
                        border-radius:12px;padding:16px 20px;color:#8899A6;font-size:14px;">
                ⚠️ Please enter a tweet to analyze.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#1E2D3D;border:1px dashed rgba(255,255,255,0.1);
                        border-radius:16px;padding:40px 20px;text-align:center;margin-top:8px;">
                <div style="font-size:32px;margin-bottom:12px;">🐦</div>
                <p style="color:#8899A6;font-size:14px;margin:0;line-height:1.6;">
                    Enter a tweet on the left and click<br>
                    <b style="color:#1D9BF0;">Analyze Sentiment</b> to see the result
                </p>
            </div>""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — BATCH ANALYSIS                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab2:
    col_up, col_manual = st.columns([1, 1], gap="large")

    with col_up:
        st.markdown('<p style="font-size:13px;font-weight:700;color:#8899A6;letter-spacing:0.8px;margin-bottom:8px;">UPLOAD CSV</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload CSV with tweet text",
            type=["csv"],
            label_visibility="collapsed",
        )
        st.markdown('<p style="font-size:12px;color:#8899A6;margin-top:8px;">CSV must contain a column named <b style="color:#E7E9EA;">text</b> or <b style="color:#E7E9EA;">tweet_text</b></p>', unsafe_allow_html=True)

        if uploaded:
            df_up     = pd.read_csv(uploaded)
            text_col  = next((c for c in df_up.columns
                              if c.lower() in ["text", "tweet_text", "tweet"]), df_up.columns[0])
            st.success(f"Loaded {len(df_up):,} tweets from '{text_col}'")

            if st.button("Run Batch Prediction", type="primary", use_container_width=True):
                with st.spinner("Analyzing tweets…"):
                    res_df = batch_predict(df_up[text_col].fillna(""), model_choice, models)
                st.session_state["batch_result"] = res_df

    with col_manual:
        st.markdown('<p style="font-size:13px;font-weight:700;color:#8899A6;letter-spacing:0.8px;margin-bottom:8px;">OR PASTE TWEETS</p>', unsafe_allow_html=True)
        bulk_text = st.text_area(
            "Paste tweets",
            placeholder="Paste multiple tweets, one per line…\n\nExample:\nLove my new iPhone!\nGoogle Maps is terrible today\nApple announced iOS 17",
            height=160,
            label_visibility="collapsed",
        )
        if st.button("Analyze All Tweets", type="primary", use_container_width=True):
            lines = [l.strip() for l in bulk_text.strip().splitlines() if l.strip()]
            if lines:
                with st.spinner("Analyzing…"):
                    res_df = batch_predict(lines, model_choice, models)
                st.session_state["batch_result"] = res_df

    if "batch_result" in st.session_state:
        res    = st.session_state["batch_result"]
        counts = res["sentiment"].value_counts()
        total  = len(res)
        st.markdown("---")

        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, color in [
            (c1, "Total Tweets",  total,                      None),
            (c2, "Positive",      counts.get("Positive", 0),  "#00BA7C"),
            (c3, "Negative",      counts.get("Negative", 0),  "#F4212E"),
            (c4, "Neutral",       counts.get("Neutral",  0),  "#1D9BF0"),
        ]:
            with col:
                pct = f" ({val/total*100:.0f}%)" if total > 0 and label != "Total Tweets" else ""
                st.markdown(f"""
                <div style="background:#1E2D3D;border:1px solid rgba(255,255,255,0.08);
                            border-radius:14px;padding:16px 20px;text-align:center;">
                    <div style="font-size:28px;font-weight:800;
                                color:{color if color else '#E7E9EA'};">{val}{pct}</div>
                    <div style="font-size:12px;color:#8899A6;margin-top:4px;font-weight:600;">{label.upper()}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_pie, col_hist = st.columns([1, 1])
        with col_pie:
            pie_fig = go.Figure(go.Pie(
                labels=counts.index.tolist(),
                values=counts.values.tolist(),
                hole=0.55,
                marker=dict(colors=["#00BA7C", "#F4212E", "#1D9BF0"][:len(counts)],
                            line=dict(color="#15202B", width=2)),
                textfont=dict(color="white", size=12, family="DM Sans"),
            ))
            pie_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E7E9EA", family="DM Sans"),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8899A6")),
                margin=dict(l=10, r=10, t=30, b=10), height=280,
                title=dict(text="Sentiment Distribution", font=dict(color="#E7E9EA", size=14)),
            )
            st.plotly_chart(pie_fig, use_container_width=True, config={"displayModeBar": False})

        with col_hist:
            hist_fig = px.histogram(
                res, x="confidence", color="sentiment",
                color_discrete_map={"Positive": "#00BA7C", "Negative": "#F4212E", "Neutral": "#1D9BF0"},
                nbins=20, barmode="overlay",
            )
            hist_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8899A6", family="DM Sans"),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=10, r=10, t=30, b=10), height=280,
                title=dict(text="Confidence Distribution", font=dict(color="#E7E9EA", size=14)),
                xaxis=dict(title="Confidence %", gridcolor="rgba(255,255,255,0.05)", color="#8899A6"),
                yaxis=dict(title="Count",        gridcolor="rgba(255,255,255,0.05)", color="#8899A6"),
            )
            st.plotly_chart(hist_fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<p style="font-size:13px;font-weight:700;color:#8899A6;letter-spacing:0.8px;margin-bottom:8px;">RESULTS TABLE</p>', unsafe_allow_html=True)
        display_cols = ["tweet", "sentiment", "confidence"] + [c for c in res.columns if c.startswith("p_")]
        st.dataframe(
            res[display_cols].style.apply(
                lambda x: ["background-color: rgba(0,186,124,0.08)" if v == "Positive"
                           else "background-color: rgba(244,33,46,0.08)" if v == "Negative"
                           else "background-color: rgba(29,155,240,0.08)" for v in x],
                subset=["sentiment"],
            ),
            height=300,
            use_container_width=True,
        )

        csv = res.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Results CSV", csv, "sentiment_results.csv",
                           "text/csv", use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 3 — MODEL COMPARISON                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab3:
    st.markdown('<p style="font-size:13px;font-weight:700;color:#8899A6;letter-spacing:0.8px;margin-bottom:16px;">PERFORMANCE LEADERBOARD</p>', unsafe_allow_html=True)

    results_data = [
        ("Naive Bayes (Binary)",             0.8239, 0.7269, True),
        ("XGBoost (Gradient Boosting)",       0.7772, 0.7445, False),
        ("Logistic Regression (Multiclass)", 0.7574, 0.7056, False),
        ("LightGBM (Gradient Boosting)",     0.7206, 0.6981, False),
        ("VADER (Rule-Based Baseline)",      0.3890, 0.4812, False),
    ]

    col_cards, col_chart = st.columns([1, 1.2], gap="large")

    with col_cards:
        for name, neg_r, macro_f, best in results_data:
            st.markdown(model_card(name, neg_r, macro_f, best), unsafe_allow_html=True)

    with col_chart:
        names      = [r[0].split("(")[0].strip() for r in results_data]
        neg_recalls = [r[1] for r in results_data]
        macro_f1s   = [r[2] for r in results_data]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Neg Recall (Primary)", x=names, y=neg_recalls,
            marker=dict(color="#F4212E", opacity=0.85, line=dict(color="rgba(0,0,0,0)")),
            text=[f"{v:.3f}" for v in neg_recalls],
            textposition="outside", textfont=dict(color="#E7E9EA", size=11),
        ))
        fig.add_trace(go.Bar(
            name="Macro F1 (Secondary)", x=names, y=macro_f1s,
            marker=dict(color="#1D9BF0", opacity=0.85, line=dict(color="rgba(0,0,0,0)")),
            text=[f"{v:.3f}" for v in macro_f1s],
            textposition="outside", textfont=dict(color="#E7E9EA", size=11),
        ))
        fig.add_hline(y=0.70, line_dash="dash", line_color="#F4212E",
                      opacity=0.4, annotation_text="Neg Recall target",
                      annotation_font=dict(color="#F4212E", size=10))
        fig.update_layout(
            barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8899A6", family="DM Sans"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8899A6"),
                        orientation="h", y=1.08),
            margin=dict(l=10, r=10, t=50, b=60), height=400,
            xaxis=dict(tickangle=-15, gridcolor="rgba(255,255,255,0.03)", color="#8899A6"),
            yaxis=dict(range=[0, 1.05], gridcolor="rgba(255,255,255,0.06)", color="#8899A6"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("""
        <div style="background:#1E2D3D;border-left:3px solid #1D9BF0;
                    border-radius:0 12px 12px 0;padding:14px 18px;margin-top:8px;">
            <p style="margin:0;font-size:13px;color:#8899A6;line-height:1.7;">
                <b style="color:#E7E9EA;">Key Insight:</b> Naive Bayes achieves the highest
                Negative Recall (0.82) but only on binary classification.
                <b style="color:#1D9BF0;">XGBoost</b> is the recommended production model —
                it handles all three sentiment classes while still exceeding the 0.70
                negative recall target.
            </p>
        </div>""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 4 — INSIGHTS                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab4:
    st.markdown('<p style="font-size:13px;font-weight:700;color:#8899A6;letter-spacing:0.8px;margin-bottom:16px;">DATASET INSIGHTS</p>', unsafe_allow_html=True)

    col_wc, col_stats = st.columns([1, 1], gap="large")

    with col_wc:
        # Sample tweets for word-frequency visualisation.
        # These come from the demo set; in production you could load
        # a sample of the real training data from a CSV instead.
        SAMPLE_TWEETS = [
            ("Love my new iPhone! The camera is absolutely stunning", "Positive"),
            ("Google Maps just saved my road trip, best app ever!", "Positive"),
            ("Apple's customer service is genuinely incredible", "Positive"),
            ("New iOS update is so smooth, loving every feature", "Positive"),
            ("Google Pixel camera quality is insane, way better than expected", "Positive"),
            ("Just got my MacBook and I'm in love, so fast", "Positive"),
            ("Google Assistant understood me perfectly, amazing", "Positive"),
            ("iPhone battery life improved so much, amazing!", "Positive"),
            ("Android Auto makes driving so much safer", "Positive"),
            ("Apple Watch saved someone's life, incredible technology", "Positive"),
            ("iPhone keeps crashing after the update, absolutely terrible", "Negative"),
            ("Google's privacy policies are a nightmare, I hate this", "Negative"),
            ("Apple overcharges for everything, these prices are insane", "Negative"),
            ("Lost all my contacts after iOS update, this is a disaster", "Negative"),
            ("Google Pixel camera is overrated, very disappointing", "Negative"),
            ("MacBook runs so hot, unacceptable for the price", "Negative"),
            ("Google Maps gave me wrong directions again, furious", "Negative"),
            ("App Store rejected my app for no reason, ridiculous", "Negative"),
            ("Android updates are painfully slow, Google fix this", "Negative"),
            ("My iPhone has terrible signal issues, so frustrating", "Negative"),
            ("Apple released iOS 17 today with several new features", "Neutral"),
            ("Google announced new Pixel phones at their fall event", "Neutral"),
            ("The new iPhone comes in three storage configurations", "Neutral"),
            ("Android 14 is now available for compatible devices", "Neutral"),
            ("Apple's market cap reached a new milestone this quarter", "Neutral"),
            ("Google Maps added new traffic features for commuters", "Neutral"),
            ("The iPhone 15 Pro has a titanium frame design choice", "Neutral"),
            ("Google's earnings call scheduled for Thursday afternoon", "Neutral"),
            ("Apple plans to open new retail stores in several cities", "Neutral"),
            ("Android users can now upgrade to the latest version", "Neutral"),
        ]
        sample_texts  = [t for t, _ in SAMPLE_TWEETS]
        sample_labels = [l for _, l in SAMPLE_TWEETS]

        sent_filter = st.selectbox("View top words for:", ["Positive", "Negative", "Neutral"])
        top_words   = make_wordcloud_data(sample_texts, sample_labels, sent_filter)

        if top_words:
            words, counts_wc = zip(*top_words[:20])
            color_map  = {"Positive": "#00BA7C", "Negative": "#F4212E", "Neutral": "#1D9BF0"}
            bar_color  = color_map[sent_filter]
            rgba_fade  = {
                "#00BA7C": "rgba(0,186,124,0.25)",
                "#F4212E": "rgba(244,33,46,0.25)",
                "#1D9BF0": "rgba(29,155,240,0.25)",
            }
            wc_fig = go.Figure(go.Bar(
                x=list(counts_wc), y=list(words), orientation="h",
                marker=dict(
                    color=list(counts_wc),
                    colorscale=[[0, rgba_fade.get(bar_color, "rgba(136,153,166,0.25)")], [1, bar_color]],
                    line=dict(color="rgba(0,0,0,0)"),
                ),
                text=[str(c) for c in counts_wc],
                textposition="outside",
                textfont=dict(color="#8899A6", size=10),
            ))
            wc_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8899A6", family="DM Sans"),
                margin=dict(l=10, r=40, t=10, b=10), height=420,
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)", color="#8899A6"),
                yaxis=dict(autorange="reversed", color="#E7E9EA"),
                title=dict(text=f"Top Words in {sent_filter} Tweets",
                           font=dict(color="#E7E9EA", size=14)),
            )
            st.plotly_chart(wc_fig, use_container_width=True, config={"displayModeBar": False})

    with col_stats:
        st.markdown('<p style="font-size:13px;font-weight:700;color:#8899A6;letter-spacing:0.8px;margin-bottom:12px;">DATASET STATISTICS</p>', unsafe_allow_html=True)

        for label, value, color in [
            ("Total Tweets (merged)",  "19,038",        "#1D9BF0"),
            ("Positive Tweets",        "7,961 (41.8%)", "#00BA7C"),
            ("Negative Tweets",        "5,564 (29.2%)", "#F4212E"),
            ("Neutral Tweets",         "5,357 (28.2%)", "#1D9BF0"),
            ("Uncertain / Unknown",    "156 (0.8%)",    "#8899A6"),
            ("Avg. Tweet Length",      "74 characters", "#FFD400"),
            ("Avg. Word Count",        "13.2 words",    "#FFD400"),
            ("Train / Test Split",     "80% / 20%",     "#8899A6"),
        ]:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:10px 16px;background:#1E2D3D;border-radius:10px;margin-bottom:6px;
                        border:1px solid rgba(255,255,255,0.05);">
                <span style="color:#8899A6;font-size:13px;">{label}</span>
                <span style="color:{color};font-weight:700;font-size:14px;">{value}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p style="font-size:13px;font-weight:700;color:#8899A6;letter-spacing:0.8px;margin-bottom:12px;">PREPROCESSING PIPELINE</p>', unsafe_allow_html=True)

        for num, step in [
            ("1", "Lowercase + URL/mention removal"),
            ("2", "Hashtag text extraction"),
            ("3", "Special character stripping"),
            ("4", "Tokenization (NLTK word_tokenize)"),
            ("5", "Stopword removal (keep negation)"),
            ("6", "WordNet lemmatization"),
        ]:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;
                        padding:8px 14px;background:#1E2D3D;border-radius:10px;margin-bottom:5px;">
                <span style="width:24px;height:24px;background:#1D9BF022;color:#1D9BF0;
                             border-radius:50%;display:inline-flex;align-items:center;
                             justify-content:center;font-size:11px;font-weight:800;flex-shrink:0;">{num}</span>
                <span style="color:#E7E9EA;font-size:13px;">{step}</span>
            </div>""", unsafe_allow_html=True)