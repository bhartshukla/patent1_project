"""
Mind Matrix V3 — Mental Health Score Intelligence (MHSI) System
===============================================================
A production-grade multi-phase computational pipeline for mental health
assessment based on natural language analysis.

Architecture:
  Phase 1: Context-Aware Emotion Analysis  (VADER + non-linear transform)
  Phase 2: Cognitive Distortion Detection  (CDI)
  Phase 3: Behavioral Inference            (active/passive language)
  Phase 4: MHSI Scoring Engine             (non-linear sigmoid formula)

USP Scores (independent, clearly displayed):
  • Happiness Score
  • Confidence Score
  • Satisfaction Score

Author  : Mind Matrix V3 Team
Version : 3.0.0
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import re
import math
import time
import datetime
from collections import deque

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

# ── NLTK bootstrap (download once per session) ───────────────────────────────
for _pkg in ("vader_lexicon", "punkt", "punkt_tab", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}" if "punkt" in _pkg else f"sentiment/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

# ════════════════════════════════════════════════════════════════════════════
# 1 · WORD LEXICONS
# ════════════════════════════════════════════════════════════════════════════

POSITIVE_WORDS = {
    "happy", "joy", "excited", "wonderful", "great", "amazing", "love",
    "fantastic", "good", "positive", "cheerful", "glad", "elated", "bliss",
    "content", "hopeful", "optimistic", "bright", "grateful", "thankful",
    "peaceful", "calm", "relaxed", "energetic", "motivated", "inspired",
    "proud", "successful", "thriving", "flourishing", "alive", "vibrant",
    "enthusiastic", "delighted", "pleased", "satisfied", "fulfilled",
}

NEGATIVE_WORDS = {
    "sad", "depressed", "anxious", "worried", "hopeless", "worthless",
    "miserable", "terrible", "awful", "dreadful", "painful", "suffering",
    "crying", "exhausted", "tired", "empty", "lonely", "isolated", "lost",
    "broken", "failed", "useless", "stupid", "hate", "disgusted", "angry",
    "furious", "devastated", "overwhelmed", "panic", "fear", "scared",
    "hurt", "bitter", "resentful", "betrayed", "abandoned", "rejected",
}

CONFIDENCE_WORDS = {
    "confident", "certain", "sure", "capable", "strong", "determined",
    "decisive", "bold", "courageous", "fearless", "assertive", "resilient",
    "powerful", "competent", "skilled", "able", "ready", "prepared",
    "believe", "trust", "faith", "know", "will", "can", "achieve",
    "succeed", "excel", "master", "lead", "overcome", "accomplish",
}

LOW_CONFIDENCE_WORDS = {
    "unsure", "uncertain", "doubt", "afraid", "scared", "weak", "helpless",
    "incapable", "incompetent", "worthless", "failure", "inadequate",
    "insecure", "timid", "hesitant", "confused", "lost", "overwhelmed",
    "cannot", "cant", "impossible", "never", "always", "useless",
    "terrible", "awful", "stupid", "idiot", "dumb",
}

SATISFACTION_WORDS = {
    "satisfied", "fulfilled", "content", "pleased", "happy", "accomplished",
    "achieved", "completed", "done", "successful", "proud", "worthy",
    "enough", "adequate", "sufficient", "comfortable", "settled", "stable",
    "meaningful", "purposeful", "valued", "appreciated", "respected",
    "loved", "accepted", "belong", "connected", "supported", "thriving",
}

DISSATISFACTION_WORDS = {
    "unsatisfied", "unfulfilled", "discontent", "disappointed", "frustrated",
    "failed", "incomplete", "unfinished", "unsuccessful", "regret", "missed",
    "lacking", "insufficient", "inadequate", "stuck", "stagnant", "empty",
    "meaningless", "purposeless", "unvalued", "unappreciated", "disrespected",
    "unloved", "unwanted", "rejected", "isolated", "alone", "struggling",
}

ACTIVE_WORDS = {
    "will", "going", "planning", "decided", "choosing", "taking", "doing",
    "working", "trying", "pursuing", "creating", "building", "achieving",
    "acting", "moving", "progressing", "improving", "growing", "learning",
    "starting", "beginning", "initiating", "leading", "driving", "making",
}

PASSIVE_WORDS = {
    "cannot", "wont", "giving", "giving up", "stopped", "quit", "done",
    "finished", "hopeless", "pointless", "worthless", "useless", "whatever",
    "doesnt matter", "nothing", "no point", "why bother", "exhausted",
    "drained", "empty", "numb", "disconnected",
}

# ── Cognitive Distortion Patterns ────────────────────────────────────────────
DISTORTION_PATTERNS = {
    "catastrophizing": {
        "patterns": [
            r"\b(everything|nothing|always|never|worst|terrible|disaster|ruined|"
            r"catastrophe|end of the world|hopeless|doomed|destroyed)\b"
        ],
        "weight": 0.35,
    },
    "helplessness": {
        "patterns": [
            r"\b(can'?t|cannot|unable|impossible|helpless|powerless|no way|"
            r"no hope|nothing i can do|out of control|stuck|trapped)\b"
        ],
        "weight": 0.30,
    },
    "all_or_nothing": {
        "patterns": [
            r"\b(always|never|everyone|nobody|everything|nothing|completely|"
            r"totally|absolutely|perfect|failure|either|all or nothing)\b"
        ],
        "weight": 0.25,
    },
    "mind_reading": {
        "patterns": [
            r"\b(they think|everyone thinks|people think|must think|probably"
            r" thinks|judging me|hate me|against me|criticize|laugh at)\b"
        ],
        "weight": 0.10,
    },
}

# ── Motivational & Temporal Words ────────────────────────────────────────────
MOTIVATION_WORDS = {
    "goal", "plan", "future", "improve", "grow", "better", "progress",
    "achieve", "succeed", "hope", "dream", "aspire", "ambition", "purpose",
}

TEMPORAL_POSITIVE = {
    "better", "improving", "recovering", "healing", "growing", "progressing",
    "moving forward", "getting better", "looking up", "hopeful",
}

TEMPORAL_NEGATIVE = {
    "worse", "declining", "deteriorating", "slipping", "falling", "losing",
    "going downhill", "getting worse",
}


# ════════════════════════════════════════════════════════════════════════════
# 2 · CORE COMPUTATIONAL PIPELINE
# ════════════════════════════════════════════════════════════════════════════

# Cached VADER analyser (one instance per session)
@st.cache_resource
def _get_vader() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def _tokenize(text: str) -> list[str]:
    """Lower-case word tokenise; fall back to split() on error."""
    try:
        return word_tokenize(text.lower())
    except Exception:
        return text.lower().split()


# ── Phase 1 ─────────────────────────────────────────────────────────────────

def compute_sentiment(text: str) -> dict:
    """
    Phase 1: Context-Aware Emotion Analysis.

    Returns
    -------
    dict with keys: raw_compound, transformed_S, word_sentiments
    """
    sia = _get_vader()
    scores = sia.polarity_scores(text)
    raw = scores["compound"]                         # [-1, +1]
    transformed_S = raw / (1 + abs(raw))             # non-linear squash

    # Per-word sentiment for variance computation
    words = _tokenize(text)
    word_sentiments = [
        sia.polarity_scores(w)["compound"]
        for w in words if w.isalpha()
    ]

    return {
        "raw_compound": raw,
        "transformed_S": transformed_S,
        "word_sentiments": word_sentiments,
        "pos": scores["pos"],
        "neg": scores["neg"],
        "neu": scores["neu"],
    }


def compute_variance(word_sentiments: list[float]) -> float:
    """
    Phase 1 (cont.): Emotional variance across words.
    High variance ⟹ emotionally volatile text.
    """
    if len(word_sentiments) < 2:
        return 0.0
    arr = np.array(word_sentiments)
    return float(np.var(arr))


# ── Phase 2 ─────────────────────────────────────────────────────────────────

def compute_cdi(text: str) -> dict:
    """
    Phase 2: Cognitive Distortion Detection.

    Assigns weights per distortion; applies synergy multiplier (1.4×) when
    multiple distortions co-occur.

    Returns
    -------
    dict with keys: CDI, distortions_found, raw_score
    """
    text_lower = text.lower()
    detected: dict[str, float] = {}

    for distortion, meta in DISTORTION_PATTERNS.items():
        hit_count = 0
        for pattern in meta["patterns"]:
            matches = re.findall(pattern, text_lower)
            hit_count += len(matches)
        if hit_count > 0:
            # Score proportional to hit density, capped at weight
            density = min(hit_count / max(len(text.split()), 1) * 10, 1.0)
            detected[distortion] = meta["weight"] * density

    raw_score = sum(detected.values())

    # Synergy multiplier when 2+ distortions co-occur
    if len(detected) >= 2:
        raw_score *= 1.4

    CDI = float(np.clip(raw_score, 0.0, 1.0))

    return {
        "CDI": CDI,
        "distortions_found": list(detected.keys()),
        "raw_score": raw_score,
        "distortion_weights": detected,
    }


# ── Phase 3 ─────────────────────────────────────────────────────────────────

def behavioral_score(text: str) -> dict:
    """
    Phase 3: Behavioral Inference.

    Detects active vs passive language and motivation level.

    Returns
    -------
    dict with keys: B ([-1,+1]), motivation, active_count, passive_count
    """
    words = set(_tokenize(text))

    active_count  = len(words & ACTIVE_WORDS)
    passive_count = len(words & PASSIVE_WORDS)
    motive_count  = len(words & MOTIVATION_WORDS)

    total = active_count + passive_count
    if total == 0:
        B = 0.0
    else:
        B = (active_count - passive_count) / total

    motivation = "High" if motive_count >= 3 else ("Moderate" if motive_count >= 1 else "Low")

    return {
        "B": float(np.clip(B, -1.0, 1.0)),
        "motivation": motivation,
        "active_count": active_count,
        "passive_count": passive_count,
        "motive_count": motive_count,
    }


# ── USP Scores ───────────────────────────────────────────────────────────────

def compute_happiness_score(text: str, sentiment_boost: float) -> float:
    """
    USP Score 1: Happiness Score.
    = positive_word_count - negative_word_count + sentiment_boost (scaled)
    Returns a value in [0, 10].
    """
    words = set(_tokenize(text))
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    raw = (pos - neg) + (sentiment_boost * 3)          # sentiment boost scaled
    return float(np.clip((raw + 5) / 10 * 10, 0, 10))  # normalise to [0,10]


def compute_confidence_score(text: str) -> float:
    """
    USP Score 2: Confidence Score.
    = confidence_words - low_confidence_words
    Returns a value in [0, 10].
    """
    words = set(_tokenize(text))
    conf    = len(words & CONFIDENCE_WORDS)
    low_conf = len(words & LOW_CONFIDENCE_WORDS)
    raw = conf - low_conf
    return float(np.clip((raw + 5) / 10 * 10, 0, 10))


def compute_satisfaction_score(text: str) -> float:
    """
    USP Score 3: Satisfaction Score.
    = satisfaction_words - dissatisfaction_words
    Returns a value in [0, 10].
    """
    words = set(_tokenize(text))
    sat    = len(words & SATISFACTION_WORDS)
    dissat = len(words & DISSATISFACTION_WORDS)
    raw = sat - dissat
    return float(np.clip((raw + 5) / 10 * 10, 0, 10))


# ── Temporal & Fulfillment Helpers ───────────────────────────────────────────

def _compute_temporal_trend(text: str, history: list[float]) -> float:
    """
    Temporal trend T in [-1, +1].
    Considers both lexical cues and historic score delta.
    """
    words = set(_tokenize(text))
    t_pos = len(words & TEMPORAL_POSITIVE)
    t_neg = len(words & TEMPORAL_NEGATIVE)
    lexical_T = (t_pos - t_neg) / max(t_pos + t_neg, 1) if (t_pos + t_neg) else 0.0

    if len(history) >= 2:
        delta = history[-1] - history[-2]
        hist_T = np.tanh(delta)
    else:
        hist_T = 0.0

    return float(np.clip(0.6 * lexical_T + 0.4 * hist_T, -1.0, 1.0))


def _compute_anomaly(score: float, history: list[float]) -> float:
    """
    Anomaly score A via z-score (0 if insufficient history).
    Capped at [0, 1].
    """
    if len(history) < 3:
        return 0.0
    arr = np.array(history)
    mean, std = arr.mean(), arr.std()
    if std < 1e-6:
        return 0.0
    z = abs((score - mean) / std)
    return float(np.clip(z / 3.0, 0.0, 1.0))   # normalise: z=3 ⟹ A=1


# ── Phase 4 ─────────────────────────────────────────────────────────────────

def compute_mhsi(
    S: float,
    V: float,
    CDI: float,
    happiness: float,
    satisfaction: float,
    T: float,
    B: float,
    A: float,
) -> float:
    """
    Phase 4: MHSI Scoring Engine.

    Formula (per specification):
      MHSI = sigmoid( a1·S − a2·V² + a3·log(1+C) + a4·tanh(F) + a5·T + a6·B ) · (1−A)

    Parameters
    ----------
    S  : transformed sentiment      ∈ [-1, +1]
    V  : emotional variance         ≥ 0
    CDI: cognitive distortion index ∈ [0, 1]
    happiness, satisfaction         ∈ [0, 10]   (scaled to [0,1] internally)
    T  : temporal trend             ∈ [-1, +1]
    B  : behavioral score           ∈ [-1, +1]
    A  : anomaly score              ∈ [0, 1]

    Returns
    -------
    MHSI ∈ [0, 10]
    """
    # Calibrated weights (tuned for the [0,10] target range)
    a1, a2, a3, a4, a5, a6 = 2.5, 1.8, 1.2, 1.5, 0.8, 1.0

    C = 1.0 - CDI                                      # cognitive clarity
    F = (happiness / 10 + satisfaction / 10) / 2       # fulfillment ∈ [0, 1]

    linear_combo = (
        a1 * S
        - a2 * (V ** 2)
        + a3 * math.log(1 + C)
        + a4 * math.tanh(F)
        + a5 * T
        + a6 * B
    )

    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    raw_mhsi = sigmoid(linear_combo) * (1 - A)
    return float(np.clip(raw_mhsi * 10, 0, 10))


# ── Master analyse() ─────────────────────────────────────────────────────────

def analyze(text: str, score_history: list[float]) -> dict:
    """
    Full pipeline: runs all 4 phases and returns a comprehensive result dict.
    """
    # Phase 1
    sent = compute_sentiment(text)
    S    = sent["transformed_S"]
    V    = compute_variance(sent["word_sentiments"])

    # Phase 2
    cdi_result = compute_cdi(text)
    CDI        = cdi_result["CDI"]

    # Phase 3
    beh = behavioral_score(text)
    B   = beh["B"]

    # USP Scores (independent)
    happiness    = compute_happiness_score(text, sent["raw_compound"])
    confidence   = compute_confidence_score(text)
    satisfaction = compute_satisfaction_score(text)

    # Temporal & anomaly (require history)
    T = _compute_temporal_trend(text, score_history)
    # Placeholder MHSI for anomaly calc against history (A=0 first pass)
    prelim = compute_mhsi(S, V, CDI, happiness, satisfaction, T, B, A=0.0)
    A      = _compute_anomaly(prelim, score_history)

    # Final MHSI
    mhsi = compute_mhsi(S, V, CDI, happiness, satisfaction, T, B, A)

    # Risk classification
    if mhsi >= 7.0:
        risk = "Good"
        risk_color = "#22c55e"
        risk_icon  = "✅"
    elif mhsi >= 4.0:
        risk = "Moderate"
        risk_color = "#f59e0b"
        risk_icon  = "⚠️"
    else:
        risk = "High Risk"
        risk_color = "#ef4444"
        risk_icon  = "🚨"

    # Recommendations
    recommendations = _build_recommendations(risk, cdi_result["distortions_found"], beh)

    return {
        # USP
        "happiness":    happiness,
        "confidence":   confidence,
        "satisfaction": satisfaction,
        # MHSI
        "mhsi":  mhsi,
        "risk":  risk,
        "risk_color":  risk_color,
        "risk_icon":   risk_icon,
        # Internals
        "S": S, "V": V, "CDI": CDI, "B": B, "T": T, "A": A,
        "sentiment": sent,
        "cdi_result": cdi_result,
        "behavioral": beh,
        "recommendations": recommendations,
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "text_preview": text[:80] + ("…" if len(text) > 80 else ""),
    }


# ── Recommendations ──────────────────────────────────────────────────────────

def _build_recommendations(
    risk: str, distortions: list[str], beh: dict
) -> list[str]:
    recs: list[str] = []

    if risk == "High Risk":
        recs += [
            "🆘 Consider speaking with a licensed mental health professional.",
            "📞 Reach out to a trusted friend, family member, or counsellor today.",
            "🧘 Practice grounding exercises: 5-4-3-2-1 sensory technique.",
        ]
    elif risk == "Moderate":
        recs += [
            "💬 Journaling daily can help clarify recurring thought patterns.",
            "🏃 Regular physical activity (30 min/day) significantly boosts mood.",
            "🌱 Try mindfulness or breathing exercises for 10 minutes each morning.",
        ]
    else:
        recs += [
            "🌟 You're doing well — keep nurturing your positive habits.",
            "🎯 Set one meaningful goal this week to maintain momentum.",
        ]

    if "catastrophizing" in distortions:
        recs.append("🔍 Practice cognitive reframing: challenge 'worst-case' assumptions.")
    if "helplessness" in distortions:
        recs.append("💪 List three small actions you *can* take today — agency builds over time.")
    if "all_or_nothing" in distortions:
        recs.append("🎨 Embrace nuance — most situations exist on a spectrum, not in absolutes.")

    if beh["motivation"] == "Low":
        recs.append("🕯️ Break large tasks into micro-steps; celebrate tiny wins.")
    if beh["passive_count"] > beh["active_count"]:
        recs.append("🚀 Shift language from 'I can't' to 'I'm learning to' — it rewires mindset.")

    return recs[:5]   # max 5 recommendations


# ════════════════════════════════════════════════════════════════════════════
# 3 · STREAMLIT UI
# ════════════════════════════════════════════════════════════════════════════

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mind Matrix V3",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  /* ── Background ── */
  .stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #0f1629 50%, #0d0d1a 100%);
    color: #e2e8f0;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a1f 0%, #111827 100%);
    border-right: 1px solid rgba(139,92,246,0.2);
  }

  /* ── Title ── */
  .mm-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a855f7, #38bdf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    margin-bottom: 0;
  }
  .mm-subtitle {
    color: #94a3b8;
    font-size: 0.9rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0;
  }

  /* ── Cards ── */
  .usp-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    backdrop-filter: blur(8px);
    transition: transform 0.2s, border-color 0.2s;
  }
  .usp-card:hover { transform: translateY(-3px); border-color: rgba(168,85,247,0.4); }

  .usp-label {
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.5rem;
  }
  .usp-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
  }
  .usp-icon { font-size: 1.4rem; margin-bottom: 0.4rem; }

  /* ── MHSI Hero ── */
  .mhsi-hero {
    background: linear-gradient(135deg, rgba(139,92,246,0.15) 0%, rgba(56,189,248,0.10) 100%);
    border: 1px solid rgba(139,92,246,0.35);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
  }
  .mhsi-hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(139,92,246,0.08) 0%, transparent 60%);
    pointer-events: none;
  }
  .mhsi-score {
    font-family: 'Space Mono', monospace;
    font-size: 4.5rem;
    font-weight: 700;
    line-height: 1;
  }
  .mhsi-label {
    font-size: 0.8rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.6rem;
  }

  /* ── Risk badge ── */
  .risk-badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 50px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    margin-top: 0.6rem;
  }

  /* ── Section headers ── */
  .section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #6366f1;
    border-bottom: 1px solid rgba(99,102,241,0.25);
    padding-bottom: 0.4rem;
    margin: 1.6rem 0 1rem 0;
  }

  /* ── Insight pills ── */
  .insight-pill {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 50px;
    padding: 0.25rem 0.8rem;
    font-size: 0.75rem;
    color: #a5b4fc;
    margin: 0.2rem;
  }

  /* ── History row ── */
  .hist-row {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.4rem;
    font-size: 0.82rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  /* ── Recommendation ── */
  .rec-item {
    background: rgba(255,255,255,0.03);
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 0.65rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
    color: #cbd5e1;
  }

  /* ── Progress bar override ── */
  .stProgress > div > div > div { background-color: #8b5cf6 !important; }

  /* ── Textarea ── */
  textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(139,92,246,0.3) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
  }
  textarea:focus { border-color: #8b5cf6 !important; box-shadow: 0 0 0 2px rgba(139,92,246,0.2) !important; }

  /* ── Button ── */
  .stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    transition: opacity 0.2s, transform 0.2s !important;
    width: 100%;
  }
  .stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 2rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state bootstrap ───────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history: list[dict] = [] # type: ignore
if "score_history" not in st.session_state:
    st.session_state.score_history: list[float] = [] # type: ignore


# ════════════════════════════════════════════════════════════════════════════
# 4 · CHART BUILDERS
# ════════════════════════════════════════════════════════════════════════════

_DARK_PAPER   = "rgba(0,0,0,0)"
_DARK_PLOT    = "rgba(255,255,255,0.03)"
_FONT_COLOR   = "#94a3b8"
_GRID_COLOR   = "rgba(255,255,255,0.07)"

_BASE_LAYOUT = dict(
    paper_bgcolor=_DARK_PAPER,
    plot_bgcolor=_DARK_PLOT,
    font=dict(family="DM Sans", color=_FONT_COLOR, size=12),
    margin=dict(l=20, r=20, t=30, b=20),
)


def _radar_chart(result: dict) -> go.Figure:
    """Radar chart showing six internal dimensions."""
    cats = ["Happiness", "Confidence", "Satisfaction",
            "Clarity", "Behaviour", "Sentiment"]

    # Normalise each to [0, 10]
    clarity  = (1 - result["CDI"]) * 10
    behav    = (result["B"] + 1) / 2 * 10
    sentim   = (result["S"] + 1) / 2 * 10

    vals = [
        result["happiness"],
        result["confidence"],
        result["satisfaction"],
        clarity,
        behav,
        sentim,
    ]
    vals_closed = vals + [vals[0]]
    cats_closed = cats + [cats[0]]

    fig = go.Figure(go.Scatterpolar(
        r=vals_closed,
        theta=cats_closed,
        fill="toself",
        fillcolor="rgba(139,92,246,0.15)",
        line=dict(color="#8b5cf6", width=2),
        marker=dict(size=6, color="#a78bfa"),
    ))
    fig.update_layout(
        **_BASE_LAYOUT,
        polar=dict(
            bgcolor="rgba(255,255,255,0.03)",
            radialaxis=dict(
                visible=True, range=[0, 10],
                tickfont=dict(size=9, color=_FONT_COLOR),
                gridcolor=_GRID_COLOR,
                linecolor=_GRID_COLOR,
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#c4b5fd"),
                gridcolor=_GRID_COLOR,
                linecolor=_GRID_COLOR,
            ),
        ),
        showlegend=False,
        height=340,
    )
    return fig


def _history_chart(score_history: list[float]) -> go.Figure:
    """Line chart of historical MHSI scores."""
    x = list(range(1, len(score_history) + 1))
    fig = go.Figure()

    # Fill area
    fig.add_trace(go.Scatter(
        x=x, y=score_history,
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.10)",
        line=dict(color="#6366f1", width=2.5),
        mode="lines+markers",
        marker=dict(size=7, color="#818cf8",
                    line=dict(color="#1e1b4b", width=2)),
        name="MHSI",
    ))

    # Risk zone bands
    fig.add_hrect(y0=7, y1=10, fillcolor="rgba(34,197,94,0.07)",
                  line_width=0, annotation_text="Good", annotation_position="right")
    fig.add_hrect(y0=4, y1=7,  fillcolor="rgba(245,158,11,0.07)",
                  line_width=0, annotation_text="Moderate", annotation_position="right")
    fig.add_hrect(y0=0, y1=4,  fillcolor="rgba(239,68,68,0.07)",
                  line_width=0, annotation_text="High Risk", annotation_position="right")

    fig.update_layout(
        **_BASE_LAYOUT,
        yaxis=dict(range=[0, 10.2], gridcolor=_GRID_COLOR,
                   title="MHSI", title_font_color=_FONT_COLOR),
        xaxis=dict(title="Session #", title_font_color=_FONT_COLOR,
                   gridcolor=_GRID_COLOR, tickmode="linear"),
        height=280,
        showlegend=False,
    )
    return fig


def _distortion_bar(cdi_result: dict) -> go.Figure:
    """Horizontal bar for distortion weights."""
    dw = cdi_result["distortion_weights"]
    if not dw:
        labels = ["No distortions detected"]
        values = [0.0]
        colors = ["rgba(34,197,94,0.6)"]
    else:
        labels = [d.replace("_", " ").title() for d in dw]
        values = list(dw.values())
        colors = ["#f87171", "#fb923c", "#facc15", "#a78bfa"][:len(labels)]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        marker_line_color="rgba(0,0,0,0)",
    ))
    fig.update_layout(
        **_BASE_LAYOUT,
        xaxis=dict(range=[0, 0.55], gridcolor=_GRID_COLOR,
                   title="Weight", title_font_color=_FONT_COLOR),
        yaxis=dict(autorange="reversed"),
        height=max(160, 60 * len(labels)),
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
# 5 · SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
      <div style='font-size:2.5rem'>🧠</div>
      <div style='font-family:Space Mono,monospace; font-size:1.1rem;
                  background:linear-gradient(90deg,#a855f7,#38bdf8);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                  background-clip:text; font-weight:700'>Mind Matrix V3</div>
      <div style='color:#475569; font-size:0.72rem; letter-spacing:0.15em;
                  text-transform:uppercase; margin-top:2px'>MHSI Engine · v3.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Pipeline Weights")
    a1_disp = st.slider("Sentiment (a1)",   0.5, 5.0, 2.5, 0.1)
    a2_disp = st.slider("Variance Pen (a2)", 0.5, 4.0, 1.8, 0.1)
    a3_disp = st.slider("Clarity (a3)",     0.5, 3.0, 1.2, 0.1)
    a4_disp = st.slider("Fulfillment (a4)", 0.5, 3.0, 1.5, 0.1)
    a5_disp = st.slider("Temporal (a5)",    0.1, 2.0, 0.8, 0.1)
    a6_disp = st.slider("Behaviour (a6)",   0.1, 2.0, 1.0, 0.1)

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    <div style='color:#64748b; font-size:0.82rem; line-height:1.6'>
    <b style='color:#8b5cf6'>Phase 1</b> — Emotion Analysis (VADER + non-linear)<br>
    <b style='color:#38bdf8'>Phase 2</b> — Cognitive Distortion Detection (CDI)<br>
    <b style='color:#34d399'>Phase 3</b> — Behavioral Inference<br>
    <b style='color:#f59e0b'>Phase 4</b> — MHSI Sigmoid Formula<br><br>
    <b>USP Scores</b>: Happiness · Confidence · Satisfaction — computed independently and fused into MHSI.
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown("---")
        if st.button("🗑️  Clear Session History"):
            st.session_state.history.clear()
            st.session_state.score_history.clear()
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# 6 · MAIN LAYOUT
# ════════════════════════════════════════════════════════════════════════════

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<p class='mm-title'>Mind Matrix V3</p>
<p class='mm-subtitle'>Mental Health Score Intelligence · Multi-Phase Computational Pipeline</p>
""", unsafe_allow_html=True)

# ── Input section ─────────────────────────────────────────────────────────────
col_input, col_action = st.columns([5, 1], gap="medium")

with col_input:
    user_text = st.text_area(
        label="💬 How are you feeling today?",
        placeholder=(
            "Describe your thoughts, emotions, or experiences freely...\n"
            "e.g. 'I've been feeling really overwhelmed lately, like I can't "
            "catch a break. Nothing seems to go right and I'm exhausted.'"
        ),
        height=130,
        key="user_text",
        label_visibility="visible",
    )

with col_action:
    st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
    analyze_btn = st.button("🔬  Analyse", use_container_width=True)
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    example_btn = st.button("💡  Example", use_container_width=True)

# Example injection
EXAMPLE_TEXTS = [
    "I feel completely hopeless. Nothing ever works out and I'm always failing. I can't do anything right.",
    "Today was great! I completed a big project at work and my team gave me wonderful feedback. Feeling proud!",
    "I'm a bit anxious about the presentation tomorrow but I've prepared well and I think I can handle it.",
    "Everything feels pointless lately. I can never seem to catch up, no matter how hard I try. I'm exhausted.",
    "I had a really peaceful morning. I went for a run, made a healthy breakfast, and felt motivated to tackle the day.",
]

if example_btn:
    import random
    st.session_state["injected_text"] = random.choice(EXAMPLE_TEXTS)
    st.rerun()

if "injected_text" in st.session_state:
    user_text = st.session_state.pop("injected_text")


# ── Analysis execution ────────────────────────────────────────────────────────
result = None

if analyze_btn and user_text and user_text.strip():
    with st.spinner("Running multi-phase pipeline…"):
        time.sleep(0.3)   # brief UX pause
        result = analyze(user_text.strip(), st.session_state.score_history)

    # Persist to session history
    st.session_state.score_history.append(result["mhsi"])
    st.session_state.history.insert(0, result)

    # Use last result if available (rerun not needed; result is fresh)
elif st.session_state.history:
    result = st.session_state.history[0]   # show most recent on page load

elif analyze_btn and not user_text.strip():
    st.warning("Please enter some text before analysing.")


# ── Results display ───────────────────────────────────────────────────────────
if result:

    st.markdown("<div class='section-header'>◆ USP SCORES — INDEPENDENT COMPUTATION</div>",
                unsafe_allow_html=True)

    # ── USP Cards row ───────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3, gap="medium")

    def _usp_color(score: float) -> str:
        if score >= 7: return "#22c55e"
        if score >= 4: return "#f59e0b"
        return "#ef4444"

    with c1:
        col = _usp_color(result["happiness"])
        st.markdown(f"""
        <div class='usp-card'>
          <div class='usp-icon'>😊</div>
          <div class='usp-label'>Happiness Score</div>
          <div class='usp-value' style='color:{col}'>{result["happiness"]:.1f}</div>
          <div style='font-size:0.72rem;color:#475569;margin-top:0.4rem'>/ 10</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(result["happiness"] / 10)

    with c2:
        col = _usp_color(result["confidence"])
        st.markdown(f"""
        <div class='usp-card'>
          <div class='usp-icon'>💪</div>
          <div class='usp-label'>Confidence Score</div>
          <div class='usp-value' style='color:{col}'>{result["confidence"]:.1f}</div>
          <div style='font-size:0.72rem;color:#475569;margin-top:0.4rem'>/ 10</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(result["confidence"] / 10)

    with c3:
        col = _usp_color(result["satisfaction"])
        st.markdown(f"""
        <div class='usp-card'>
          <div class='usp-icon'>🎯</div>
          <div class='usp-label'>Satisfaction Score</div>
          <div class='usp-value' style='color:{col}'>{result["satisfaction"]:.1f}</div>
          <div style='font-size:0.72rem;color:#475569;margin-top:0.4rem'>/ 10</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(result["satisfaction"] / 10)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── MHSI Hero + Radar ────────────────────────────────────────────────────
    mhsi_col, radar_col = st.columns([2, 3], gap="large")

    with mhsi_col:
        mhsi_col_hex = result["risk_color"]
        badge_bg = {
            "Good": "rgba(34,197,94,0.15)",
            "Moderate": "rgba(245,158,11,0.15)",
            "High Risk": "rgba(239,68,68,0.15)",
        }[result["risk"]]

        st.markdown(f"""
        <div class='mhsi-hero'>
          <div class='mhsi-label'>◆ MENTAL HEALTH SCORE INDEX</div>
          <div class='mhsi-score' style='color:{mhsi_col_hex}'>{result["mhsi"]:.2f}</div>
          <div style='color:#475569;font-size:0.8rem;margin:0.3rem 0'>out of 10.00</div>
          <div class='risk-badge' style='background:{badge_bg};color:{mhsi_col_hex};border:1px solid {mhsi_col_hex}40'>
            {result["risk_icon"]} {result["risk"]}
          </div>
          <div style='margin-top:1.4rem'>
        """, unsafe_allow_html=True)

        st.progress(result["mhsi"] / 10)

        st.markdown("""
        <div style='display:flex;justify-content:space-between;font-size:0.7rem;color:#475569;margin-top:-0.3rem'>
          <span>0 · High Risk</span><span>5 · Moderate</span><span>10 · Good</span>
        </div>
        </div></div>
        """, unsafe_allow_html=True)

        # Internal metrics
        st.markdown("""
        <div style='display:flex;flex-wrap:wrap;gap:0.3rem;margin-top:1rem'>
        """, unsafe_allow_html=True)
        pill_data = [
            ("S", f"{result['S']:.2f}", "Sentiment"),
            ("V", f"{result['V']:.3f}", "Variance"),
            ("CDI", f"{result['CDI']:.2f}", "Distortion"),
            ("B", f"{result['B']:.2f}", "Behaviour"),
            ("T", f"{result['T']:.2f}", "Temporal"),
            ("A", f"{result['A']:.2f}", "Anomaly"),
        ]
        pills_html = "".join(
            f"<span class='insight-pill' title='{tip}'><b>{k}</b> {v}</span>"
            for k, v, tip in pill_data
        )
        st.markdown(pills_html + "</div>", unsafe_allow_html=True)

    with radar_col:
        st.markdown("<div class='section-header'>◆ MULTI-DIMENSIONAL PROFILE</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(_radar_chart(result), use_container_width=True, config={"displayModeBar": False})

    # ── Cognitive Distortions + Behavioral ──────────────────────────────────
    dist_col, beh_col = st.columns(2, gap="large")

    with dist_col:
        st.markdown("<div class='section-header'>◆ COGNITIVE DISTORTION ANALYSIS</div>",
                    unsafe_allow_html=True)
        cdi = result["cdi_result"]
        st.plotly_chart(_distortion_bar(cdi), use_container_width=True,
                        config={"displayModeBar": False})

        if cdi["distortions_found"]:
            st.markdown(
                f"**CDI:** `{cdi['CDI']:.3f}` &nbsp;|&nbsp; "
                f"**Patterns:** {', '.join(d.replace('_',' ').title() for d in cdi['distortions_found'])}",
                unsafe_allow_html=True
            )
            if len(cdi["distortions_found"]) >= 2:
                st.markdown("⚡ *Synergy multiplier (1.4×) applied — multiple distortions co-occur.*",
                            unsafe_allow_html=True)
        else:
            st.markdown("✅ *No significant cognitive distortions detected.*")

    with beh_col:
        st.markdown("<div class='section-header'>◆ BEHAVIORAL INFERENCE</div>",
                    unsafe_allow_html=True)
        beh = result["behavioral"]

        # Active vs Passive donut
        a_cnt, p_cnt = beh["active_count"], beh["passive_count"]
        neutral = max(0, 10 - a_cnt - p_cnt)
        fig_beh = go.Figure(go.Pie(
            labels=["Active", "Passive", "Neutral"],
            values=[max(a_cnt, 0.01), max(p_cnt, 0.01), max(neutral, 0.01)],
            hole=0.55,
            marker_colors=["#22c55e", "#ef4444", "#334155"],
            textfont=dict(size=11, color="white"),
        ))
        fig_beh.update_layout(
            **_BASE_LAYOUT,
            showlegend=True,
            legend=dict(font=dict(color=_FONT_COLOR, size=11)),
            height=220,
            annotations=[dict(
                text=f"B={beh['B']:+.2f}",
                x=0.5, y=0.5,
                font=dict(size=16, color="#a78bfa", family="Space Mono"),
                showarrow=False,
            )],
        )
        st.plotly_chart(fig_beh, use_container_width=True, config={"displayModeBar": False})

        mot_color = {"High": "#22c55e", "Moderate": "#f59e0b", "Low": "#ef4444"}[beh["motivation"]]
        st.markdown(
            f"**Motivation Level:** "
            f"<span style='color:{mot_color};font-weight:700'>{beh['motivation']}</span> &nbsp;|&nbsp; "
            f"Motivational cues: `{beh['motive_count']}`",
            unsafe_allow_html=True
        )

    # ── Recommendations ──────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>◆ PERSONALISED RECOMMENDATIONS</div>",
                unsafe_allow_html=True)
    for rec in result["recommendations"]:
        st.markdown(f"<div class='rec-item'>{rec}</div>", unsafe_allow_html=True)

    # ── Session History ───────────────────────────────────────────────────────
    if len(st.session_state.history) > 1:
        st.markdown("<div class='section-header'>◆ SESSION HISTORY · MHSI TREND</div>",
                    unsafe_allow_html=True)
        hist_chart, hist_list = st.columns([3, 2], gap="large")

        with hist_chart:
            st.plotly_chart(
                _history_chart(st.session_state.score_history),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        with hist_list:
            for i, h in enumerate(st.session_state.history[:8]):
                risk_colors = {"Good": "#22c55e", "Moderate": "#f59e0b", "High Risk": "#ef4444"}
                rc = risk_colors.get(h["risk"], "#94a3b8")
                st.markdown(f"""
                <div class='hist-row'>
                  <div>
                    <span style='color:#64748b;font-size:0.72rem'>#{len(st.session_state.history)-i}</span>
                    &nbsp;
                    <span style='color:#94a3b8'>{h['text_preview']}</span>
                  </div>
                  <div>
                    <span style='color:{rc};font-family:Space Mono,monospace;font-weight:700'>
                      {h['mhsi']:.1f}
                    </span>
                    <span style='color:#475569;font-size:0.7rem'>&nbsp;{h['timestamp']}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Sentiment detail ─────────────────────────────────────────────────────
    with st.expander("🔬 Raw Sentiment Detail", expanded=False):
        sent = result["sentiment"]
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("VADER Compound", f"{sent['raw_compound']:.3f}")
        sc2.metric("Transformed S",  f"{sent['transformed_S']:.3f}")
        sc3.metric("Positive",       f"{sent['pos']:.2%}")
        sc4.metric("Negative",       f"{sent['neg']:.2%}")

else:
    # ── Welcome placeholder ──────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding:3rem 2rem; color:#334155'>
      <div style='font-size:4rem; margin-bottom:1rem'>🧠</div>
      <div style='font-family:Space Mono,monospace; font-size:1.1rem; color:#6366f1; margin-bottom:0.5rem'>
        Awaiting your input
      </div>
      <div style='color:#475569; font-size:0.9rem; max-width:480px; margin:0 auto; line-height:1.7'>
        Type how you're feeling in the text box above and click <b style='color:#8b5cf6'>Analyse</b>.
        The multi-phase MHSI pipeline will process your text through four computational stages
        and return your <b>Happiness</b>, <b>Confidence</b>, and <b>Satisfaction</b> scores,
        fused into a final <b>Mental Health Score Index</b>.
      </div>
      <div style='margin-top:2rem; display:flex; justify-content:center; gap:1.5rem; flex-wrap:wrap'>
        <div style='background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.3);
                    border-radius:12px;padding:1rem 1.5rem;font-size:0.82rem;color:#a5b4fc'>
          😊 Happiness Score
        </div>
        <div style='background:rgba(56,189,248,0.1);border:1px solid rgba(56,189,248,0.3);
                    border-radius:12px;padding:1rem 1.5rem;font-size:0.82rem;color:#7dd3fc'>
          💪 Confidence Score
        </div>
        <div style='background:rgba(52,211,153,0.1);border:1px solid rgba(52,211,153,0.3);
                    border-radius:12px;padding:1rem 1.5rem;font-size:0.82rem;color:#6ee7b7'>
          🎯 Satisfaction Score
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:2rem 0 0.8rem'>
<div style='text-align:center;color:#334155;font-size:0.75rem;letter-spacing:0.1em'>
  MIND MATRIX V3 · MHSI ENGINE · For research and wellness tracking purposes only.
  Not a substitute for professional mental health advice.
</div>
""", unsafe_allow_html=True)