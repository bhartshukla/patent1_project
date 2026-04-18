"""
╔══════════════════════════════════════════════════════════════════╗
║           MIND MATRIX V3 — Mental Health Assessment System       ║
║           Production-Grade Multi-Phase Computational Pipeline    ║
╚══════════════════════════════════════════════════════════════════╝

Architecture:
  Phase 1 → Context-Aware Emotion Analysis (VADER + Non-linear transform)
  Phase 2 → Cognitive Distortion Detection (CDI)
  Phase 3 → Behavioral Inference Engine
  Phase 4 → MHSI Scoring Engine (sigmoid formula)

USP Scores (always computed independently):
  • Happiness Score
  • Confidence Score
  • Satisfaction Score
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import deque
import re
import math

# ─── NLTK Bootstrap ────────────────────────────────────────────────────────────
import nltk
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    from nltk.tokenize import word_tokenize
except Exception:
    def word_tokenize(text):
        return re.findall(r"\b\w+\b", text.lower())

# ═══════════════════════════════════════════════════════════════════════════════
#  LEXICON DICTIONARIES
# ═══════════════════════════════════════════════════════════════════════════════

POSITIVE_WORDS = {
    "happy", "joy", "love", "excited", "wonderful", "great", "amazing", "fantastic",
    "excellent", "good", "nice", "pleasure", "delighted", "cheerful", "glad", "thrilled",
    "elated", "content", "blessed", "grateful", "positive", "hopeful", "optimistic",
    "peaceful", "calm", "energetic", "motivated", "inspired", "proud", "fulfilled",
    "joyful", "blissful", "ecstatic", "radiant", "vibrant", "alive", "free", "strong"
}

NEGATIVE_WORDS = {
    "sad", "depressed", "anxious", "afraid", "terrible", "horrible", "awful", "bad",
    "miserable", "unhappy", "upset", "angry", "frustrated", "lonely", "hopeless",
    "worthless", "guilty", "ashamed", "scared", "panicking", "overwhelmed", "empty",
    "numb", "tired", "exhausted", "broken", "lost", "stuck", "helpless", "hate",
    "pain", "suffering", "despair", "fear", "worried", "nervous", "stressed", "dark"
}

CONFIDENCE_WORDS = {
    "confident", "sure", "certain", "believe", "capable", "strong", "determined",
    "decisive", "assertive", "bold", "courageous", "resilient", "able", "skilled",
    "accomplished", "competent", "successful", "winning", "powerful", "ready",
    "unstoppable", "motivated", "driven", "focused", "clear", "firm", "committed"
}

LOW_CONFIDENCE_WORDS = {
    "doubt", "unsure", "uncertain", "insecure", "weak", "incapable", "fail",
    "failure", "can't", "cannot", "won't", "unable", "incompetent", "useless",
    "pointless", "impossible", "never", "always", "worthless", "stupid", "dumb",
    "pathetic", "loser", "quit", "give up", "hopeless", "powerless", "defeated"
}

SATISFACTION_WORDS = {
    "satisfied", "fulfilled", "accomplished", "achieved", "complete", "enough",
    "sufficient", "content", "pleased", "proud", "gratified", "rewarded", "happy",
    "grateful", "thankful", "appreciate", "meaningful", "purposeful", "valuable",
    "worthwhile", "deserving", "earned", "success", "progress", "growth", "better"
}

DISSATISFACTION_WORDS = {
    "dissatisfied", "unfulfilled", "incomplete", "insufficient", "lacking", "missing",
    "disappointed", "frustrated", "regret", "failed", "missed", "wasted", "empty",
    "pointless", "meaningless", "stuck", "stagnant", "behind", "worse", "regretful",
    "wish", "should have", "could have", "would have", "never", "nothing", "useless"
}

# Cognitive Distortion Patterns
COGNITIVE_DISTORTIONS = {
    "catastrophizing": {
        "patterns": [
            r"\bworst\b", r"\bterrible\b", r"\bdisaster\b", r"\bruined\b", r"\bdestroyed\b",
            r"\bnever recover\b", r"\bend of\b", r"\bno way out\b", r"\bpermanent\b",
            r"\bforever\b.*\bworse\b", r"\bnothing will\b", r"\beverything is over\b"
        ],
        "weight": 0.35
    },
    "helplessness": {
        "patterns": [
            r"\bcan't do anything\b", r"\bno control\b", r"\bhelpless\b", r"\bpowerless\b",
            r"\bnothing i can do\b", r"\bimpossible\b", r"\bno hope\b", r"\bgive up\b",
            r"\bwhat's the point\b", r"\buseless\b", r"\bdoesn't matter\b", r"\bno use\b"
        ],
        "weight": 0.30
    },
    "all_or_nothing": {
        "patterns": [
            r"\balways\b", r"\bnever\b", r"\beveryone\b", r"\bno one\b", r"\bnothing\b",
            r"\beverything\b", r"\bcompletely\b", r"\bperfect\b.*\bor\b", r"\btotal failure\b",
            r"\ball bad\b", r"\bentirely\b", r"\babsolutely\b.*\bnot\b"
        ],
        "weight": 0.25
    },
    "mind_reading": {
        "patterns": [
            r"\bthey think\b", r"\beveryone thinks\b", r"\bthey hate\b", r"\bthey don't like\b",
            r"\bi know they\b", r"\bpeople think\b", r"\bthey judge\b", r"\blaughing at me\b"
        ],
        "weight": 0.10
    }
}

# Active vs Passive behavioral markers
ACTIVE_MARKERS = {
    "will", "going to", "plan", "decided", "action", "doing", "working", "trying",
    "improving", "started", "beginning", "creating", "building", "achieving", "moving",
    "forward", "change", "commit", "pursue", "accomplish", "take charge", "step"
}

PASSIVE_MARKERS = {
    "can't", "won't", "unable", "helpless", "stuck", "waiting", "hoping someone",
    "if only", "wish", "maybe someday", "nothing i can do", "what's the point",
    "whatever", "doesn't matter", "give up", "too late", "no choice", "forced"
}


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: CONTEXT-AWARE EMOTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_sentiment(text: str) -> tuple[float, float]:
    """
    Phase 1: Compute transformed sentiment score using VADER.
    Applies non-linear transformation: S = raw_sentiment / (1 + |raw_sentiment|)

    Returns:
        S (float): Transformed sentiment in (-1, 1)
        raw (float): Raw compound VADER score
    """
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    raw = scores["compound"]  # Range: -1 to +1

    # Non-linear transformation (compresses extremes, preserves sign)
    S = raw / (1 + abs(raw))
    return S, raw


def compute_variance(text: str) -> float:
    """
    Phase 1: Compute emotional variance across words in the text.
    Uses VADER per-word scoring to measure emotional volatility.

    Returns:
        V (float): Emotional variance (higher = more volatile)
    """
    sia = SentimentIntensityAnalyzer()
    words = word_tokenize(text.lower())
    word_scores = []

    for word in words:
        score = sia.polarity_scores(word)["compound"]
        if score != 0:
            word_scores.append(score)

    if len(word_scores) < 2:
        return 0.0

    V = float(np.var(word_scores))
    return min(V, 1.0)  # Cap at 1 to prevent formula dominance


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: COGNITIVE DISTORTION DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_cdi(text: str) -> tuple[float, dict]:
    """
    Phase 2: Compute Cognitive Distortion Index (CDI).
    Detects distortion patterns, applies weights, and synergy multiplier.

    Returns:
        CDI (float): Cognitive distortion index in [0, 1]
        details (dict): Per-distortion detection results
    """
    text_lower = text.lower()
    detected = {}
    total_score = 0.0
    distortion_count = 0

    for distortion_name, config in COGNITIVE_DISTORTIONS.items():
        matched = []
        for pattern in config["patterns"]:
            if re.search(pattern, text_lower):
                matched.append(pattern)

        if matched:
            distortion_count += 1
            weighted = config["weight"] * min(len(matched) / 3.0, 1.0)
            total_score += weighted
            detected[distortion_name] = {
                "detected": True,
                "matches": len(matched),
                "contribution": round(weighted, 3)
            }
        else:
            detected[distortion_name] = {"detected": False, "matches": 0, "contribution": 0.0}

    # Synergy multiplier: if multiple distortions co-occur, amplify by 1.4×
    if distortion_count >= 2:
        total_score *= 1.4

    CDI = min(total_score, 1.0)
    return CDI, detected


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3: BEHAVIORAL INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def behavioral_score(text: str) -> tuple[float, str]:
    """
    Phase 3: Infer behavioral orientation from active vs passive language.

    Returns:
        B (float): Behavioral score in [-1, +1]
        motivation (str): Inferred motivation level
    """
    text_lower = text.lower()

    active_count = sum(1 for marker in ACTIVE_MARKERS if marker in text_lower)
    passive_count = sum(1 for marker in PASSIVE_MARKERS if marker in text_lower)

    total = active_count + passive_count
    if total == 0:
        B = 0.0
    else:
        B = (active_count - passive_count) / total

    # Classify motivation
    if B > 0.5:
        motivation = "High Motivation"
    elif B > 0.0:
        motivation = "Moderate Motivation"
    elif B == 0.0:
        motivation = "Neutral"
    elif B > -0.5:
        motivation = "Low Motivation"
    else:
        motivation = "Very Low Motivation"

    return B, motivation


# ═══════════════════════════════════════════════════════════════════════════════
#  USP SCORES: HAPPINESS, CONFIDENCE, SATISFACTION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_happiness_score(text: str, sentiment_raw: float) -> float:
    """
    USP Score 1: Happiness Score
    = positive_words - negative_words + sentiment_boost

    Returns:
        Happiness score normalized to [0, 10]
    """
    words = set(word_tokenize(text.lower()))
    pos_count = len(words & POSITIVE_WORDS)
    neg_count = len(words & NEGATIVE_WORDS)

    # Sentiment boost: scale raw VADER compound to [-2, +2] contribution
    sentiment_boost = sentiment_raw * 2.0

    raw_score = pos_count - neg_count + sentiment_boost

    # Normalize to [0, 10]: assume range of raw is [-10, 10]
    normalized = (raw_score + 10) / 20 * 10
    return round(float(np.clip(normalized, 0, 10)), 2)


def compute_confidence_score(text: str) -> float:
    """
    USP Score 2: Confidence Score
    = confidence_words - low_confidence_words

    Returns:
        Confidence score normalized to [0, 10]
    """
    words = set(word_tokenize(text.lower()))
    conf_count = len(words & CONFIDENCE_WORDS)
    low_conf_count = len(words & LOW_CONFIDENCE_WORDS)

    raw_score = conf_count - low_conf_count

    # Normalize to [0, 10]
    normalized = (raw_score + 10) / 20 * 10
    return round(float(np.clip(normalized, 0, 10)), 2)


def compute_satisfaction_score(text: str) -> float:
    """
    USP Score 3: Satisfaction Score
    = satisfaction_words - dissatisfaction_words

    Returns:
        Satisfaction score normalized to [0, 10]
    """
    words = set(word_tokenize(text.lower()))
    sat_count = len(words & SATISFACTION_WORDS)
    dissat_count = len(words & DISSATISFACTION_WORDS)

    raw_score = sat_count - dissat_count

    # Normalize to [0, 10]
    normalized = (raw_score + 10) / 20 * 10
    return round(float(np.clip(normalized, 0, 10)), 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  ANOMALY DETECTION (Z-Score Based)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_anomaly_score(current_mhsi: float, history: list) -> float:
    """
    Compute anomaly score using z-score of current MHSI vs session history.

    Returns:
        A (float): Anomaly score in [0, 1]; higher = more anomalous
    """
    if len(history) < 3:
        return 0.0

    arr = np.array(history)
    mean, std = arr.mean(), arr.std()

    if std < 1e-6:
        return 0.0

    z = abs((current_mhsi - mean) / std)

    # Map z-score to [0, 1] using sigmoid
    A = 1 / (1 + math.exp(-0.5 * (z - 2)))
    return round(float(np.clip(A, 0, 0.8)), 3)  # Cap at 0.8 to not dominate MHSI


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 4: MHSI SCORING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mhsi(
    S: float,
    V: float,
    CDI: float,
    happiness: float,
    satisfaction: float,
    T: float,
    B: float,
    A: float,
    weights: dict = None
) -> float:
    """
    Phase 4: Mental Health Status Index (MHSI) — Core Formula.

    MHSI = sigmoid(a1*S - a2*(V²) + a3*log(1+C) + a4*tanh(F) + a5*T + a6*B) * (1 - A)

    Where:
        S  = transformed sentiment
        V  = emotional variance
        C  = cognitive clarity = 1 - CDI
        F  = fulfillment = (happiness + satisfaction) / 20  → [-1, 1]
        T  = temporal trend (from session history)
        B  = behavioral score
        A  = anomaly score

    Returns:
        MHSI (float): Scaled to [0, 10]
    """
    if weights is None:
        weights = {
            "a1": 2.5,   # Sentiment weight
            "a2": 1.5,   # Variance penalty
            "a3": 1.8,   # Cognitive clarity bonus
            "a4": 2.0,   # Fulfillment contribution
            "a5": 1.2,   # Temporal trend
            "a6": 1.0    # Behavioral score
        }

    C = 1.0 - CDI                          # Cognitive clarity
    F = (happiness + satisfaction) / 20.0  # Fulfillment, normalized to [0,1]
    F = F * 2.0 - 1.0                      # Shift to [-1, 1]

    # Core linear combination inside sigmoid
    z = (
        weights["a1"] * S
        - weights["a2"] * (V ** 2)
        + weights["a3"] * math.log(1.0 + max(C, 0.0))
        + weights["a4"] * math.tanh(F)
        + weights["a5"] * T
        + weights["a6"] * B
    )

    # Sigmoid maps z → (0, 1)
    sigmoid_val = 1.0 / (1.0 + math.exp(-z))

    # Apply anomaly penalty
    raw_mhsi = sigmoid_val * (1.0 - A)

    # Scale to [0, 10]
    mhsi = raw_mhsi * 10.0
    return round(float(np.clip(mhsi, 0, 10)), 2)


def compute_temporal_trend(history: list) -> float:
    """
    Compute temporal trend T from MHSI history.
    Positive trend = improving; Negative = declining.

    Returns:
        T (float): Trend signal in [-1, +1]
    """
    if len(history) < 2:
        return 0.0

    recent = history[-min(5, len(history)):]
    diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
    avg_diff = np.mean(diffs)

    # Normalize to [-1, 1]
    T = math.tanh(avg_diff / 3.0)
    return round(float(T), 3)


# ═══════════════════════════════════════════════════════════════════════════════
#  MASTER ANALYZE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def analyze(text: str, mhsi_history: list) -> dict:
    """
    Master analysis pipeline — runs all 4 phases and returns full result dict.

    Args:
        text         : User's input text
        mhsi_history : List of past MHSI scores in this session

    Returns:
        result (dict): All computed scores, labels, and metadata
    """
    if not text.strip():
        return None

    # ── Phase 1: Emotion Analysis ──────────────────────────────────────────────
    S, sentiment_raw = compute_sentiment(text)
    V = compute_variance(text)

    # ── Phase 2: Cognitive Distortion Detection ────────────────────────────────
    CDI, cdi_details = compute_cdi(text)

    # ── Phase 3: Behavioral Inference ─────────────────────────────────────────
    B, motivation = behavioral_score(text)

    # ── USP Scores (independent computation) ──────────────────────────────────
    happiness    = compute_happiness_score(text, sentiment_raw)
    confidence   = compute_confidence_score(text)
    satisfaction = compute_satisfaction_score(text)

    # ── Temporal Trend ────────────────────────────────────────────────────────
    T = compute_temporal_trend(mhsi_history)

    # ── Anomaly Score ─────────────────────────────────────────────────────────
    A = compute_anomaly_score(0.0, mhsi_history)  # Pre-compute without current MHSI

    # ── Phase 4: MHSI ─────────────────────────────────────────────────────────
    mhsi = compute_mhsi(S, V, CDI, happiness, satisfaction, T, B, A)

    # Recompute anomaly against actual history + current
    A_final = compute_anomaly_score(mhsi, mhsi_history + [mhsi])

    # ── Risk Classification ───────────────────────────────────────────────────
    if mhsi >= 7.0:
        risk_level = "Good"
        risk_color = "#00d4aa"
        risk_emoji = "🟢"
        recommendations = [
            "Your mental wellness indicators are strong — keep nurturing your routines.",
            "Continue practicing gratitude and positive social connections.",
            "Consider journaling your wins to reinforce positive patterns.",
            "Share your positivity — mentoring others can deepen your own fulfillment.",
        ]
    elif mhsi >= 4.0:
        risk_level = "Moderate"
        risk_color = "#f0a500"
        risk_emoji = "🟡"
        recommendations = [
            "Some stress signals detected — consider a mindfulness or breathing exercise.",
            "Try a 10-minute walk or light physical activity to reset your mood.",
            "Reach out to a trusted friend or family member today.",
            "Limit news and social media if they're increasing your anxiety.",
            "Sleep hygiene can significantly impact mood — aim for 7–8 hours.",
        ]
    else:
        risk_level = "High Risk"
        risk_color = "#ff4d6d"
        risk_emoji = "🔴"
        recommendations = [
            "⚠️ Please consider speaking with a mental health professional.",
            "Reach out to a crisis helpline if you're feeling overwhelmed or unsafe.",
            "You don't have to face this alone — connecting with someone you trust can help.",
            "Try grounding techniques: name 5 things you can see, 4 you can touch.",
            "Avoid making major life decisions when distress levels are high.",
            "Small step: drink a glass of water, step outside briefly, or breathe slowly.",
        ]

    return {
        # Phase scores
        "S": round(S, 3),
        "V": round(V, 3),
        "CDI": round(CDI, 3),
        "B": round(B, 3),
        "T": round(T, 3),
        "A": round(A_final, 3),
        "cognitive_clarity": round(1.0 - CDI, 3),
        "motivation": motivation,
        "sentiment_raw": round(sentiment_raw, 3),
        "cdi_details": cdi_details,

        # USP Scores
        "happiness": happiness,
        "confidence": confidence,
        "satisfaction": satisfaction,

        # Final Score
        "mhsi": mhsi,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "risk_emoji": risk_emoji,
        "recommendations": recommendations,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

def setup_page():
    """Configure Streamlit page settings and inject custom CSS."""
    st.set_page_config(
        page_title="Mind Matrix V3",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    /* ── Google Fonts ─────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=JetBrains+Mono:wght@400;600&display=swap');

    /* ── Global Reset & Variables ─────────────────────────────────────────── */
    :root {
        --bg-primary: #0a0e1a;
        --bg-card: #111827;
        --bg-card-hover: #1a2235;
        --border: rgba(255,255,255,0.07);
        --border-glow: rgba(0,212,170,0.3);
        --text-primary: #e8eaf6;
        --text-secondary: #8892a4;
        --text-muted: #4a5568;
        --accent-teal: #00d4aa;
        --accent-purple: #8b5cf6;
        --accent-amber: #f0a500;
        --accent-rose: #ff4d6d;
        --accent-blue: #3b82f6;
        --font-display: 'DM Serif Display', serif;
        --font-body: 'DM Sans', sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
        --radius: 16px;
        --radius-sm: 10px;
        --shadow-card: 0 4px 24px rgba(0,0,0,0.4);
        --shadow-glow: 0 0 40px rgba(0,212,170,0.12);
    }

    /* ── App Background ───────────────────────────────────────────────────── */
    .stApp {
        background: var(--bg-primary);
        background-image:
            radial-gradient(ellipse at 20% 10%, rgba(139,92,246,0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 80%, rgba(0,212,170,0.06) 0%, transparent 50%);
        font-family: var(--font-body);
        color: var(--text-primary);
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    .block-container { padding: 1.5rem 2rem 2rem; max-width: 1400px; }

    /* ── Typography ───────────────────────────────────────────────────────── */
    h1, h2, h3 { font-family: var(--font-display) !important; }

    /* ── Hero Header ──────────────────────────────────────────────────────── */
    .hero-header {
        text-align: center;
        padding: 2.5rem 0 1.5rem;
        position: relative;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(0,212,170,0.1);
        border: 1px solid rgba(0,212,170,0.3);
        color: var(--accent-teal);
        font-family: var(--font-mono);
        font-size: 0.68rem;
        letter-spacing: 0.15em;
        padding: 4px 14px;
        border-radius: 100px;
        margin-bottom: 1rem;
        text-transform: uppercase;
    }
    .hero-title {
        font-family: var(--font-display) !important;
        font-size: clamp(2.2rem, 5vw, 3.5rem);
        color: var(--text-primary);
        margin: 0;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    .hero-title span {
        background: linear-gradient(135deg, var(--accent-teal), var(--accent-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        color: var(--text-secondary);
        font-size: 1rem;
        margin: 0.75rem 0 0;
        font-weight: 300;
        letter-spacing: 0.02em;
    }

    /* ── Cards ────────────────────────────────────────────────────────────── */
    .card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.5rem;
        box-shadow: var(--shadow-card);
        transition: border-color 0.3s, box-shadow 0.3s;
        position: relative;
        overflow: hidden;
    }
    .card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent-teal), transparent);
        opacity: 0;
        transition: opacity 0.3s;
    }
    .card:hover { border-color: var(--border-glow); box-shadow: var(--shadow-glow); }
    .card:hover::before { opacity: 1; }

    /* ── USP Score Cards ──────────────────────────────────────────────────── */
    .usp-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.4rem 1.6rem;
        box-shadow: var(--shadow-card);
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .usp-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-glow); }
    .usp-icon {
        font-size: 1.8rem;
        display: block;
        margin-bottom: 0.4rem;
    }
    .usp-label {
        font-family: var(--font-mono);
        font-size: 0.65rem;
        letter-spacing: 0.12em;
        color: var(--text-secondary);
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    .usp-value {
        font-family: var(--font-display);
        font-size: 2.4rem;
        line-height: 1;
        margin-bottom: 0.3rem;
    }
    .usp-bar {
        height: 4px;
        border-radius: 100px;
        margin-top: 0.75rem;
        background: rgba(255,255,255,0.06);
        overflow: hidden;
    }
    .usp-bar-fill {
        height: 100%;
        border-radius: 100px;
        transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
    }

    /* ── MHSI Hero Score ──────────────────────────────────────────────────── */
    .mhsi-hero {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 2rem;
        text-align: center;
        box-shadow: var(--shadow-card);
        position: relative;
        overflow: hidden;
    }
    .mhsi-hero::after {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(ellipse at 50% 0%, rgba(0,212,170,0.06) 0%, transparent 70%);
        pointer-events: none;
    }
    .mhsi-label {
        font-family: var(--font-mono);
        font-size: 0.65rem;
        letter-spacing: 0.2em;
        color: var(--text-secondary);
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .mhsi-score {
        font-family: var(--font-display);
        font-size: 5rem;
        line-height: 1;
        font-style: italic;
        margin: 0;
    }
    .mhsi-outof {
        font-family: var(--font-mono);
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-top: 0.2rem;
    }
    .risk-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 100px;
        font-family: var(--font-mono);
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: 1rem;
    }

    /* ── Phase Metric Pills ───────────────────────────────────────────────── */
    .metric-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(255,255,255,0.04);
        border: 1px solid var(--border);
        border-radius: 100px;
        padding: 0.4rem 0.9rem;
        font-family: var(--font-mono);
        font-size: 0.75rem;
        color: var(--text-secondary);
        margin: 0.2rem;
    }
    .metric-pill .val {
        color: var(--text-primary);
        font-weight: 600;
    }

    /* ── Distortion Tags ──────────────────────────────────────────────────── */
    .distortion-tag {
        display: inline-block;
        background: rgba(255,77,109,0.12);
        border: 1px solid rgba(255,77,109,0.25);
        color: #ff8fa3;
        font-family: var(--font-mono);
        font-size: 0.65rem;
        padding: 3px 10px;
        border-radius: 100px;
        margin: 2px;
        text-transform: capitalize;
    }
    .no-distortion {
        color: var(--accent-teal);
        font-family: var(--font-mono);
        font-size: 0.8rem;
    }

    /* ── Recommendation Cards ─────────────────────────────────────────────── */
    .rec-item {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 0.85rem 1rem;
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }
    .rec-item .rec-icon { flex-shrink: 0; font-size: 1rem; }

    /* ── Section Title ────────────────────────────────────────────────────── */
    .section-title {
        font-family: var(--font-display);
        font-size: 1.25rem;
        color: var(--text-primary);
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-title::after {
        content: '';
        flex: 1;
        height: 1px;
        background: var(--border);
        margin-left: 0.5rem;
    }

    /* ── History Sparkline Container ──────────────────────────────────────── */
    .history-empty {
        text-align: center;
        padding: 2rem;
        color: var(--text-muted);
        font-family: var(--font-mono);
        font-size: 0.8rem;
        letter-spacing: 0.1em;
    }

    /* ── Sidebar ──────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: var(--bg-card) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

    /* ── Input Textarea ───────────────────────────────────────────────────── */
    .stTextArea textarea {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-body) !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
        padding: 1rem !important;
        transition: border-color 0.2s !important;
    }
    .stTextArea textarea:focus {
        border-color: rgba(0,212,170,0.4) !important;
        box-shadow: 0 0 0 3px rgba(0,212,170,0.08) !important;
    }

    /* ── Buttons ──────────────────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-teal), #00b4d8) !important;
        color: #0a0e1a !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-family: var(--font-body) !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.65rem 2rem !important;
        letter-spacing: 0.02em !important;
        transition: opacity 0.2s, transform 0.2s !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        opacity: 0.9 !important;
        transform: translateY(-1px) !important;
    }

    /* Clear button variant */
    .stButton.clear > button {
        background: rgba(255,255,255,0.06) !important;
        color: var(--text-secondary) !important;
    }

    /* ── Plotly Chart Containers ──────────────────────────────────────────── */
    .js-plotly-plot { border-radius: var(--radius) !important; }

    /* ── Scrollbar ────────────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
    </style>
    """, unsafe_allow_html=True)


def render_hero():
    """Render the hero header section."""
    st.markdown("""
    <div class="hero-header">
        <div class="hero-badge">Multi-Phase Computational Pipeline · v3.0</div>
        <h1 class="hero-title">Mind <span>Matrix</span> V3</h1>
        <p class="hero-subtitle">
            Context-aware mental health assessment · Cognitive distortion detection · Behavioral inference
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_usp_cards(happiness: float, confidence: float, satisfaction: float):
    """Render the three USP score cards (Happiness, Confidence, Satisfaction)."""
    cards = [
        {
            "icon": "😊",
            "label": "Happiness Score",
            "value": happiness,
            "color": "#f59e0b",
            "gradient": "linear-gradient(90deg, #f59e0b, #f97316)"
        },
        {
            "icon": "💪",
            "label": "Confidence Score",
            "value": confidence,
            "color": "#8b5cf6",
            "gradient": "linear-gradient(90deg, #8b5cf6, #6366f1)"
        },
        {
            "icon": "✨",
            "label": "Satisfaction Score",
            "value": satisfaction,
            "color": "#00d4aa",
            "gradient": "linear-gradient(90deg, #00d4aa, #0ea5e9)"
        }
    ]

    cols = st.columns(3)
    for col, card in zip(cols, cards):
        pct = card["value"] / 10 * 100
        with col:
            st.markdown(f"""
            <div class="usp-card">
                <span class="usp-icon">{card["icon"]}</span>
                <div class="usp-label">{card["label"]}</div>
                <div class="usp-value" style="color:{card["color"]}">{card["value"]}</div>
                <div style="font-family:var(--font-mono);font-size:0.7rem;color:var(--text-muted);">/ 10.0</div>
                <div class="usp-bar">
                    <div class="usp-bar-fill" style="width:{pct}%;background:{card["gradient"]};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_mhsi_card(result: dict):
    """Render the main MHSI score hero card."""
    mhsi = result["mhsi"]
    risk_level = result["risk_level"]
    risk_color = result["risk_color"]
    risk_emoji = result["risk_emoji"]

    st.markdown(f"""
    <div class="mhsi-hero">
        <div class="mhsi-label">Mental Health Status Index</div>
        <div class="mhsi-score" style="color:{risk_color};">{mhsi}</div>
        <div class="mhsi-outof">out of 10.00</div>
        <div>
            <span class="risk-badge" style="background:rgba({','.join(str(int(risk_color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.15);
                color:{risk_color};border:1px solid {risk_color}40;">
                {risk_emoji} {risk_level}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_phase_metrics(result: dict):
    """Render computed phase metrics as compact pills."""
    metrics = [
        ("S", result["S"], "Sentiment"),
        ("V", result["V"], "Variance"),
        ("CDI", result["CDI"], "Distortion Index"),
        ("C", result["cognitive_clarity"], "Cognitive Clarity"),
        ("B", result["B"], "Behavioral"),
        ("T", result["T"], "Temporal Trend"),
        ("A", result["A"], "Anomaly"),
    ]

    pills_html = "".join(
        f'<span class="metric-pill">{label} <span class="val">{val:+.3f}</span>'
        f'<span style="color:var(--text-muted);font-size:0.6rem">({name})</span></span>'
        for label, val, name in metrics
    )

    st.markdown(f"""
    <div class="card" style="padding:1.2rem 1.5rem;">
        <div class="section-title" style="font-size:1rem;margin-bottom:0.75rem;">
            📐 Phase Computation Metrics
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:0.1rem;">{pills_html}</div>
        <div style="margin-top:0.8rem;font-family:var(--font-mono);font-size:0.7rem;color:var(--text-muted);">
            Motivation: <span style="color:var(--text-secondary)">{result["motivation"]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_cdi_breakdown(cdi_details: dict):
    """Render cognitive distortion breakdown."""
    detected = [k.replace("_", " ").title() for k, v in cdi_details.items() if v["detected"]]

    if detected:
        tags = "".join(f'<span class="distortion-tag">{d}</span>' for d in detected)
        content = tags
    else:
        content = '<span class="no-distortion">✓ No significant cognitive distortions detected</span>'

    st.markdown(f"""
    <div class="card" style="padding:1.2rem 1.5rem;">
        <div class="section-title" style="font-size:1rem;margin-bottom:0.6rem;">
            🧩 Cognitive Distortion Scan
        </div>
        <div>{content}</div>
    </div>
    """, unsafe_allow_html=True)


def render_recommendations(recommendations: list, risk_color: str):
    """Render personalized recommendations."""
    icons = ["💡", "🌿", "🤝", "📱", "😴", "🌊"]
    items_html = "".join(
        f'<div class="rec-item"><span class="rec-icon">{icons[i % len(icons)]}</span><span>{rec}</span></div>'
        for i, rec in enumerate(recommendations)
    )

    st.markdown(f"""
    <div class="card" style="border-left:3px solid {risk_color}40;">
        <div class="section-title" style="font-size:1.05rem;">
            🎯 Personalized Recommendations
        </div>
        {items_html}
    </div>
    """, unsafe_allow_html=True)


def render_radar_chart(result: dict):
    """Render radar chart with key dimensions."""
    categories = [
        "Happiness", "Confidence", "Satisfaction",
        "Cognitive\nClarity", "Behavioral\nEnergy", "Emotional\nStability"
    ]

    # Normalize all values to 0-10
    cog_clarity_norm = result["cognitive_clarity"] * 10
    behavioral_norm  = (result["B"] + 1) / 2 * 10  # [-1,1] → [0,10]
    stability_norm   = (1 - result["V"]) * 10       # Lower variance = more stable

    values = [
        result["happiness"],
        result["confidence"],
        result["satisfaction"],
        cog_clarity_norm,
        behavioral_norm,
        stability_norm
    ]
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure()

    # Fill area
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(0,212,170,0.10)",
        line=dict(color="#00d4aa", width=2),
        mode="lines+markers",
        marker=dict(size=6, color="#00d4aa", symbol="circle"),
        name="Current Assessment",
        hovertemplate="<b>%{theta}</b><br>Score: %{r:.1f}/10<extra></extra>"
    ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 10],
                tickfont=dict(size=9, color="#4a5568", family="JetBrains Mono"),
                gridcolor="rgba(255,255,255,0.06)",
                linecolor="rgba(255,255,255,0.08)",
                tickvals=[2, 4, 6, 8, 10],
                ticktext=["2", "4", "6", "8", "10"],
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#8892a4", family="DM Sans"),
                gridcolor="rgba(255,255,255,0.05)",
                linecolor="rgba(255,255,255,0.08)",
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#8892a4"),
        showlegend=False,
        margin=dict(t=30, b=30, l=60, r=60),
        height=340,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_history_chart(mhsi_history: list):
    """Render session MHSI history as a line chart."""
    if len(mhsi_history) < 2:
        st.markdown("""
        <div class="history-empty">
            NO HISTORY YET · SUBMIT MORE ENTRIES TO SEE TREND
        </div>
        """, unsafe_allow_html=True)
        return

    x = list(range(1, len(mhsi_history) + 1))
    y = mhsi_history

    # Color zones
    fig = go.Figure()

    # Zone fills
    fig.add_hrect(y0=7, y1=10, fillcolor="rgba(0,212,170,0.04)", line_width=0)
    fig.add_hrect(y0=4, y1=7,  fillcolor="rgba(240,165,0,0.04)",  line_width=0)
    fig.add_hrect(y0=0, y1=4,  fillcolor="rgba(255,77,109,0.04)", line_width=0)

    # Line
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines+markers",
        line=dict(color="#00d4aa", width=2.5, shape="spline", smoothing=0.8),
        marker=dict(
            size=8, color=y, colorscale=[[0, "#ff4d6d"], [0.4, "#f0a500"], [0.7, "#00d4aa"], [1, "#3b82f6"]],
            line=dict(color="#0a0e1a", width=2), cmin=0, cmax=10
        ),
        fill="tozeroy",
        fillcolor="rgba(0,212,170,0.05)",
        hovertemplate="Entry %{x}<br>MHSI: <b>%{y:.2f}</b><extra></extra>"
    ))

    # Threshold lines
    for yval, color, label in [(7, "#00d4aa", "Good"), (4, "#f0a500", "Moderate")]:
        fig.add_hline(y=yval, line=dict(color=color, width=1, dash="dot"),
                      annotation_text=label,
                      annotation_font=dict(size=9, color=color, family="JetBrains Mono"),
                      annotation_position="right")

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=True, gridcolor="rgba(255,255,255,0.04)",
            tickfont=dict(size=9, color="#4a5568", family="JetBrains Mono"),
            linecolor="rgba(255,255,255,0.06)", title="Session Entry",
            title_font=dict(size=10, color="#4a5568")
        ),
        yaxis=dict(
            range=[0, 10.5], showgrid=True, gridcolor="rgba(255,255,255,0.04)",
            tickfont=dict(size=9, color="#4a5568", family="JetBrains Mono"),
            linecolor="rgba(255,255,255,0.06)", title="MHSI Score",
            title_font=dict(size=10, color="#4a5568")
        ),
        font=dict(family="DM Sans", color="#8892a4"),
        showlegend=False,
        margin=dict(t=15, b=40, l=50, r=80),
        height=260,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_sidebar(mhsi_history: list, usp_history: dict):
    """Render sidebar with session stats and history."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1rem 0 1.5rem;">
            <div style="font-size:2rem;margin-bottom:0.5rem;">🧠</div>
            <div style="font-family:'DM Serif Display',serif;font-size:1.2rem;color:#e8eaf6;">
                Session Overview
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                        color:#4a5568;letter-spacing:0.1em;margin-top:0.3rem;">
                MIND MATRIX V3
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Session stats
        if mhsi_history:
            avg_mhsi = round(np.mean(mhsi_history), 2)
            max_mhsi = round(max(mhsi_history), 2)
            min_mhsi = round(min(mhsi_history), 2)

            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                        border-radius:12px;padding:1rem;margin-bottom:1rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                            letter-spacing:0.12em;color:#4a5568;text-transform:uppercase;
                            margin-bottom:0.75rem;">Session Stats</div>
                <div style="display:flex;justify-content:space-between;margin-bottom:0.5rem;">
                    <span style="font-size:0.8rem;color:#8892a4;">Entries</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;
                                color:#e8eaf6;">{len(mhsi_history)}</span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-bottom:0.5rem;">
                    <span style="font-size:0.8rem;color:#8892a4;">Avg MHSI</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;
                                color:#00d4aa;">{avg_mhsi}</span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-bottom:0.5rem;">
                    <span style="font-size:0.8rem;color:#8892a4;">Peak</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;
                                color:#3b82f6;">{max_mhsi}</span>
                </div>
                <div style="display:flex;justify-content:space-between;">
                    <span style="font-size:0.8rem;color:#8892a4;">Lowest</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;
                                color:#ff4d6d;">{min_mhsi}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # USP averages
            if usp_history["happiness"]:
                avg_h = round(np.mean(usp_history["happiness"]), 1)
                avg_c = round(np.mean(usp_history["confidence"]), 1)
                avg_s = round(np.mean(usp_history["satisfaction"]), 1)

                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                            border-radius:12px;padding:1rem;margin-bottom:1rem;">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                                letter-spacing:0.12em;color:#4a5568;text-transform:uppercase;
                                margin-bottom:0.75rem;">Avg USP Scores</div>
                    <div style="margin-bottom:0.5rem;">
                        <span style="font-size:0.75rem;color:#8892a4;">😊 Happiness</span>
                        <div style="height:3px;background:rgba(255,255,255,0.06);border-radius:2px;margin-top:3px;">
                            <div style="width:{avg_h*10}%;height:100%;background:linear-gradient(90deg,#f59e0b,#f97316);border-radius:2px;"></div>
                        </div>
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#f59e0b;">{avg_h}/10</span>
                    </div>
                    <div style="margin-bottom:0.5rem;">
                        <span style="font-size:0.75rem;color:#8892a4;">💪 Confidence</span>
                        <div style="height:3px;background:rgba(255,255,255,0.06);border-radius:2px;margin-top:3px;">
                            <div style="width:{avg_c*10}%;height:100%;background:linear-gradient(90deg,#8b5cf6,#6366f1);border-radius:2px;"></div>
                        </div>
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#8b5cf6;">{avg_c}/10</span>
                    </div>
                    <div>
                        <span style="font-size:0.75rem;color:#8892a4;">✨ Satisfaction</span>
                        <div style="height:3px;background:rgba(255,255,255,0.06);border-radius:2px;margin-top:3px;">
                            <div style="width:{avg_s*10}%;height:100%;background:linear-gradient(90deg,#00d4aa,#0ea5e9);border-radius:2px;"></div>
                        </div>
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#00d4aa;">{avg_s}/10</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="text-align:center;padding:2rem 0;color:#4a5568;
                        font-family:'JetBrains Mono',monospace;font-size:0.75rem;
                        letter-spacing:0.1em;">
                SUBMIT AN ENTRY<br>TO SEE STATS
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;
                    color:#4a5568;letter-spacing:0.08em;line-height:1.8;padding:0.5rem 0;">
            MHSI = sigmoid(<br>
            &nbsp;&nbsp;a₁S − a₂V² + a₃log(1+C)<br>
            &nbsp;&nbsp;+ a₄tanh(F) + a₅T + a₆B<br>
            ) × (1 − A)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.7rem;color:#4a5568;text-align:center;padding:0.5rem 0;
                    font-family:'JetBrains Mono',monospace;letter-spacing:0.05em;">
            ⚠️ For informational use only.<br>Not a substitute for professional care.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    setup_page()

    # ── Session State Init ──────────────────────────────────────────────────
    if "mhsi_history" not in st.session_state:
        st.session_state.mhsi_history = []
    if "usp_history" not in st.session_state:
        st.session_state.usp_history = {"happiness": [], "confidence": [], "satisfaction": []}
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    # ── Sidebar ─────────────────────────────────────────────────────────────
    render_sidebar(st.session_state.mhsi_history, st.session_state.usp_history)

    # ── Hero Header ──────────────────────────────────────────────────────────
    render_hero()

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Input Section ────────────────────────────────────────────────────────
    with st.container():
        st.markdown("""
        <div class="section-title">💬 How are you feeling today?</div>
        """, unsafe_allow_html=True)

        user_input = st.text_area(
            label="",
            placeholder=(
                "Describe your thoughts, feelings, or current state of mind...\n\n"
                "Example: 'I've been feeling overwhelmed lately with work, but I'm trying to stay positive "
                "and keep pushing forward. Some days feel impossible but I believe things can improve.'"
            ),
            height=130,
            key="user_input",
            label_visibility="collapsed"
        )

        col_btn1, col_btn2, col_spacer = st.columns([2, 1, 4])
        with col_btn1:
            analyze_clicked = st.button("🧠 Analyze Now", key="analyze_btn")
        with col_btn2:
            clear_clicked = st.button("✕ Clear", key="clear_btn")

    if clear_clicked:
        st.session_state.mhsi_history = []
        st.session_state.usp_history = {"happiness": [], "confidence": [], "satisfaction": []}
        st.session_state.last_result = None
        st.rerun()

    # ── Run Analysis ──────────────────────────────────────────────────────────
    if analyze_clicked:
        if not user_input or not user_input.strip():
            st.warning("⚠️ Please enter some text before analyzing.")
        elif len(user_input.strip()) < 10:
            st.warning("⚠️ Please enter at least a sentence for meaningful analysis.")
        else:
            with st.spinner("Running multi-phase computational pipeline..."):
                result = analyze(user_input, st.session_state.mhsi_history)

            if result:
                # Update session history
                st.session_state.mhsi_history.append(result["mhsi"])
                st.session_state.usp_history["happiness"].append(result["happiness"])
                st.session_state.usp_history["confidence"].append(result["confidence"])
                st.session_state.usp_history["satisfaction"].append(result["satisfaction"])
                st.session_state.last_result = result

    # ── Display Results ───────────────────────────────────────────────────────
    result = st.session_state.last_result

    if result:
        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

        # ── USP Scores (THE CORE USP) ─────────────────────────────────────────
        st.markdown("""
        <div class="section-title">⭐ Core Wellness Scores (USP)</div>
        """, unsafe_allow_html=True)
        render_usp_cards(result["happiness"], result["confidence"], result["satisfaction"])

        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

        # ── MHSI + Radar ──────────────────────────────────────────────────────
        col_mhsi, col_radar = st.columns([1, 1.4])

        with col_mhsi:
            st.markdown("""
            <div class="section-title">🎯 MHSI Score</div>
            """, unsafe_allow_html=True)
            render_mhsi_card(result)

            st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
            render_phase_metrics(result)

            st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
            render_cdi_breakdown(result["cdi_details"])

        with col_radar:
            st.markdown("""
            <div class="section-title">📊 Dimensional Profile</div>
            """, unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="card" style="padding:1rem;">', unsafe_allow_html=True)
                render_radar_chart(result)
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

        # ── Recommendations ───────────────────────────────────────────────────
        render_recommendations(result["recommendations"], result["risk_color"])

        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

        # ── Session History Chart ─────────────────────────────────────────────
        st.markdown("""
        <div class="section-title">📈 Session Trend</div>
        """, unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="card" style="padding:1rem 1.5rem;">', unsafe_allow_html=True)
            render_history_chart(st.session_state.mhsi_history)
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        # ── Empty state ───────────────────────────────────────────────────────
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;
                    border:1px dashed rgba(255,255,255,0.07);
                    border-radius:16px;margin-top:1rem;">
            <div style="font-size:3rem;margin-bottom:1rem;">🧠</div>
            <div style="font-family:'DM Serif Display',serif;font-size:1.5rem;
                        color:#e8eaf6;margin-bottom:0.5rem;">
                Ready to Analyze
            </div>
            <div style="color:#4a5568;font-size:0.9rem;max-width:400px;margin:0 auto;line-height:1.6;">
                Enter how you're feeling in the text box above and click
                <strong style="color:#00d4aa">Analyze Now</strong> to run the
                multi-phase mental health pipeline.
            </div>
            <div style="margin-top:1.5rem;display:flex;justify-content:center;gap:1.5rem;
                        font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#4a5568;">
                <span>😊 Happiness</span>
                <span>💪 Confidence</span>
                <span>✨ Satisfaction</span>
                <span>🎯 MHSI</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()