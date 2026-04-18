"""
Mind Matrix V3 — Mental Health Score Intelligence (MHSI) System
===============================================================
Merged production build combining the original Mind Matrix UI/UX features
(question rotation, word lexicons, session management, favicon, form layout)
with the V3 multi-phase computational pipeline.

Architecture:
  Phase 1 · Context-Aware Emotion Analysis   (VADER + non-linear transform)
  Phase 2 · Cognitive Distortion Detection   (CDI with synergy multiplier)
  Phase 3 · Behavioral Inference             (active/passive + motivation)
  Phase 4 · MHSI Scoring Engine              (non-linear sigmoid formula)

USP Scores (independent, clearly displayed):
  Happiness Score    = positive_words - negative_words + sentiment_boost
  Confidence Score   = confidence_words - low_confidence_words
  Satisfaction Score = satisfaction_words - dissatisfaction_words

Original features preserved:
  65-question pool with 10-question random rotation
  Original expanded word lexicons
  favicon.png support
  Form submit / Clear / New Questions buttons
  Personalized recommendations + mental health resource links

Author  : Mind Matrix V3 Team
Version : 3.0.0
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import re
import math
import time
import random
import datetime
from collections import Counter

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

# ── NLTK bootstrap (downloads on first run) ──────────────────────────────────
for _pkg in ("vader_lexicon", "punkt", "punkt_tab"):
    try:
        nltk.data.find(
            f"tokenizers/{_pkg}" if "punkt" in _pkg else f"sentiment/{_pkg}"
        )
    except LookupError:
        nltk.download(_pkg, quiet=True)


# ════════════════════════════════════════════════════════════════════════════
# 1 · WORD LEXICONS
# ════════════════════════════════════════════════════════════════════════════

positive_words = {
    "happy", "joyful", "excited", "great", "wonderful", "amazing", "cheerful",
    "delighted", "fantastic", "thrilled", "blissful", "optimistic", "enthusiastic",
    "content", "peaceful", "radiant", "upbeat", "lively", "exhilarated", "bright",
    "hopeful", "motivated", "inspired", "glowing", "blessed", "prosperous",
    "adventurous", "brilliant", "ecstatic", "vivacious", "elated", "satisfied",
    "vibrant", "accomplished", "energetic", "loving", "passionate", "grateful",
    "confident", "victorious", "playful", "serene", "encouraging", "spirited",
    "flourishing", "thriving", "empowered", "free", "affectionate", "charming",
    "fulfilled", "wholesome", "resilient", "warmhearted", "remarkable",
    "rejuvenated", "impressive", "successful", "lighthearted", "sunny", "bubbly",
    "dreamy", "tranquil", "jubilant", "fearless", "dynamic", "secure",
    "harmonious", "incredible", "astonishing", "soothing", "magnetic",
    "compassionate", "marvelous", "heartwarming", "enriched", "sparkling",
    "amusing", "courageous", "relieved", "enlightened", "productive", "nurturing",
    "triumphant", "gentle", "embracing", "heroic", "pioneering",
    "good", "joy", "love", "positive", "calm",
}

negative_words = {
    "sad", "angry", "frustrated", "depressed", "worried", "miserable", "fearful",
    "hopeless", "irritated", "anxious", "disappointed", "nervous", "discouraged",
    "lonely", "exhausted", "rejected", "insecure", "bitter", "uneasy", "restless",
    "shattered", "vulnerable", "overwhelmed", "unwanted", "worthless", "drained",
    "devastated", "pessimistic", "moody", "tormented", "disturbed", "sorrowful",
    "weary", "neglected", "upset", "gloomy", "distressed", "heartbroken",
    "abandoned", "lost", "suffering", "ashamed", "fatigued", "skeptical",
    "distrustful", "unfocused", "weak", "remorseful", "doubting", "resentful",
    "furious", "infuriated", "withdrawn", "unmotivated", "powerless", "unhappy",
    "agonized", "isolated", "unworthy", "hesitant", "embarrassed", "dreadful",
    "spiteful", "irate", "victimized", "enraged", "displeased", "panicked",
    "burdened", "disheartened", "excluded", "trapped", "restrained", "suppressed",
    "misunderstood", "detached", "regretful", "betrayed", "traumatized",
    "reluctant", "broken",
    "hate", "terrible", "awful", "fear", "scared", "empty", "hurt", "numb",
}

confidence_words = {
    "able", "confident", "certain", "assured", "capable", "determined", "fearless",
    "assertive", "strong", "courageous", "bold", "empowered", "daring", "tenacious",
    "independent", "ambitious", "motivated", "secure", "competent", "unstoppable",
    "solid", "wise", "visionary", "efficient", "passionate", "winner", "firm",
    "victorious", "disciplined", "consistent", "energetic", "unshakable", "mindful",
    "resilient", "composed", "resourceful", "influential", "proactive", "charismatic",
    "logical", "masterful", "convincing", "heroic", "skillful", "expert",
    "outstanding", "remarkable", "powerful", "esteemed", "relentless", "undeterred",
    "pragmatic", "focused", "intellectual", "smart", "sophisticated", "talented",
    "proficient", "inspired", "strategic", "balanced", "dedicated", "innovative",
    "keen", "sharp", "unyielding", "thoughtful", "steadfast", "grounded",
    "accomplished", "unwavering", "enlightened",
    "believe", "trust", "know", "will", "can", "achieve", "succeed", "excel",
    "master", "lead", "overcome",
}

low_confidence_words = {
    "unable", "doubt", "uncertain", "nervous", "hesitant", "fearful", "anxious",
    "reluctant", "unsure", "worried", "insecure", "apprehensive", "timid",
    "discouraged", "doubtful", "intimidated", "uneasy", "shy", "fragile",
    "lacking", "unsteady", "skeptical", "disoriented", "incapable", "exhausted",
    "confused", "defeated", "powerless", "self-conscious", "faltering",
    "unmotivated", "weak", "struggling", "frightened", "burdened", "helpless",
    "repressed", "ineffective", "troubled", "downcast", "lost", "passive",
    "vulnerable", "indecisive", "wavering", "shaken", "inhibited", "stammering",
    "unready", "flustered", "overwhelmed", "withdrawn", "unprepared", "stressed",
    "unconfident", "dreading", "floundering", "overthinking", "frozen",
    "overcautious", "pessimistic", "wary", "afraid", "mistrusting", "panicky",
    "numb", "trembling", "quivering",
    "cannot", "impossible", "never", "stupid", "failure", "inadequate",
}

satisfaction_words = {
    "satisfied", "content", "fulfilled", "happy", "pleased", "gratified",
    "delighted", "joyful", "cheerful", "blissful", "serene", "comfortable",
    "relaxed", "ecstatic", "thrilled", "elated", "excited", "euphoric",
    "thankful", "grateful", "appreciative", "secure", "reassured", "optimistic",
    "hopeful", "enthusiastic", "encouraged", "positive", "relieved", "prosperous",
    "harmonious", "balanced", "accomplished", "triumphant", "victorious", "proud",
    "glad", "overjoyed", "radiant", "uplifted", "inspired", "motivated",
    "peaceful", "satiated", "blessed", "merry", "jubilant", "animated", "cheery",
    "buoyant", "lighthearted", "exhilarated", "rejuvenated", "flourishing",
    "thriving", "sunny", "heartened", "wholesome", "loving", "joyous", "vivacious",
    "upbeat", "bubbly", "exultant", "enlightened", "giddy", "warmhearted",
    "spirited", "appreciated", "valued", "cherished", "embraced", "rewarded",
    "celebrated", "exalted", "acclaimed", "commended", "respected", "admired",
    "acknowledged", "treasured", "nurtured", "sympathetic", "caring", "generous",
    "considerate", "compassionate", "harmonized", "steady", "grounded", "trusting",
    "free", "easygoing", "contented", "sanguine", "eased", "soulful", "smiling",
    "meaningful", "purposeful", "belong", "connected", "supported",
}

dissatisfaction_words = {
    "frustrated", "disappointed", "unsatisfied", "unhappy", "annoyed", "angry",
    "dissatisfied", "bitter", "resentful", "miserable", "displeased", "upset",
    "irritated", "aggravated", "unfulfilled", "distressed", "vexed",
    "discontented", "unimpressed", "disheartened", "disgusted", "unappreciated",
    "ignored", "neglected", "overlooked", "pessimistic", "hopeless", "defeated",
    "melancholy", "lonely", "alienated", "abandoned", "isolated", "regretful",
    "ashamed", "embarrassed", "guilty", "inadequate", "inferior", "insulted",
    "offended", "humiliated", "demeaned", "belittled", "despised", "loathed",
    "disregarded", "discarded", "dejected", "rejected", "underwhelmed",
    "troubled", "anxious", "nervous", "overwhelmed", "weary", "fatigued",
    "exhausted", "drained", "uncertain", "apprehensive", "weak", "unstable",
    "doubtful", "discouraged", "demotivated", "demoralized", "lost", "aimless",
    "purposeless", "stressed", "strained", "overburdened", "conflicted",
    "misunderstood", "disoriented", "detached", "sorrowful", "disturbed",
    "pained", "aching", "tormented", "agonized", "shattered", "heartbroken",
    "devastated", "oppressed", "powerless", "restricted", "constrained",
    "betrayed", "cheated", "deceived", "fooled",
    "meaningless", "empty", "stagnant", "stuck",
}

# ── Pipeline-specific word sets ───────────────────────────────────────────────
ACTIVE_WORDS = {
    "will", "going", "planning", "decided", "choosing", "taking", "doing",
    "working", "trying", "pursuing", "creating", "building", "achieving",
    "acting", "moving", "progressing", "improving", "growing", "learning",
    "starting", "beginning", "initiating", "leading", "driving", "making",
}

PASSIVE_WORDS = {
    "cannot", "stopped", "quit", "hopeless", "pointless", "worthless",
    "useless", "whatever", "nothing", "exhausted", "drained", "empty", "numb",
    "disconnected", "finished", "done",
}

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
    "getting worse",
}

DISTORTION_PATTERNS = {
    "catastrophizing": {
        "patterns": [
            r"\b(everything|nothing|always|never|worst|terrible|disaster|ruined|"
            r"catastrophe|hopeless|doomed|destroyed|end of the world)\b"
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
            r"\b(they think|everyone thinks|people think|must think|probably thinks|"
            r"judging me|hate me|against me|criticize|laugh at)\b"
        ],
        "weight": 0.10,
    },
}

# ── Questions pool (original 65 preserved) ────────────────────────────────────
questions = [
    "How do you feel about your daily routine today?",
    "Do you feel confident in your decisions today?",
    "What made you happy today?",
    "Did you feel a sense of achievement today?",
    "Did you engage in any activities that made you feel fulfilled today?",
    "How did social interactions affect your mood today?",
    "Did you face any unexpected difficulties today?",
    "Did you take time for yourself today?",
    "What moment today made you feel genuinely joyful?",
    "Describe an interaction that brought you happiness today.",
    "What small thing made you smile or laugh today?",
    "When did you feel most at peace today?",
    "When did you feel most self-assured today?",
    "What decision today made you proud of yourself?",
    "How did you overcome self-doubt today?",
    "What challenge did you handle better than expected?",
    "What activity today energized your mood?",
    "Who or what inspired positive emotions in you today?",
    "Did you experience gratitude today? What for?",
    "What personal achievement made you happy today?",
    "What part of your routine felt most rewarding today?",
    "Did you meet your personal expectations today? How?",
    "What made you think, 'This was time well spent' today?",
    "How meaningful did your activities feel today?",
    "Did you take a risk today? How did it feel?",
    "What accomplishment made you feel capable today?",
    "How did you advocate for yourself today?",
    "What situation made you feel in control today?",
    "How did you handle stress today?",
    "Do you feel satisfied with your progress today?",
    "How often did you feel sad or down today?",
    "What progress (big or small) satisfied you today?",
    "How content are you with your relationships today?",
    "Did you feel financially secure today? Why/why not?",
    "What creative or intellectual output satisfied you?",
    "Did you feel supported by your family and friends today?",
    "What activities made you feel confident today?",
    "Do you think you were emotionally strong today?",
    "How would you describe your overall emotional state today?",
    "What hobby or interest lifted your spirits today?",
    "Did nature or surroundings boost your happiness today? How?",
    "What made you feel optimistic about the future today?",
    "What was the best part of your day today?",
    "How much energy did you have today?",
    "Did you accomplish what you planned for today?",
    "How did you handle any negative emotions today?",
    "What motivated you the most today?",
    "How well did you sleep last night, and did it affect your mood today?",
    "Did you have a moment of relaxation today?",
    "How often did you feel anxious or nervous today?",
    "Did you experience any moments of self-doubt today?",
    "When did you assert your opinions/boundaries confidently?",
    "What skill or talent did you use effectively today?",
    "How did you grow your self-trust today?",
    "What compliment or feedback boosted your confidence?",
    "How would you rate your overall mood today?",
    "Did you feel productive today?",
    "How much support did you receive from others today?",
    "What was the most enjoyable thing you did today?",
    "How well did you manage your stress today?",
    "What would you like to improve about your mindset today?",
    "How fulfilled do you feel about today's accomplishments?",
    "What task gave you a sense of completion today?",
    "Are your daily efforts aligning with long-term goals?",
    "How satisfied are you with your work-rest balance today?",
    "What was the biggest challenge you faced today?",
]


# ════════════════════════════════════════════════════════════════════════════
# 2 · CORE COMPUTATIONAL PIPELINE
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _get_vader() -> SentimentIntensityAnalyzer:
    """Cache VADER analyser — one instance per Streamlit session."""
    return SentimentIntensityAnalyzer()


def _tokenize(text: str) -> list:
    """Lower-case tokenise with fallback to simple split."""
    try:
        return word_tokenize(text.lower())
    except Exception:
        return text.lower().split()


# ── Phase 1: Emotion Analysis ────────────────────────────────────────────────

def compute_sentiment(text: str) -> dict:
    """
    Phase 1: Context-Aware Emotion Analysis.

    - VADER compound score → raw ∈ [-1, +1]
    - Non-linear transform: S = raw / (1 + |raw|)
    - Per-word sentiments collected for variance computation.
    """
    sia    = _get_vader()
    scores = sia.polarity_scores(text)
    raw    = scores["compound"]
    S      = raw / (1 + abs(raw))   # squash to open interval (-1, +1)

    words = _tokenize(text)
    word_sentiments = [
        sia.polarity_scores(w)["compound"]
        for w in words if w.isalpha()
    ]
    return {
        "raw_compound":    raw,
        "transformed_S":   S,
        "word_sentiments": word_sentiments,
        "pos": scores["pos"],
        "neg": scores["neg"],
        "neu": scores["neu"],
    }


def compute_variance(word_sentiments: list) -> float:
    """
    Phase 1 (cont.): Emotional variance across individual word sentiments.
    High variance → emotionally volatile or mixed text.
    """
    if len(word_sentiments) < 2:
        return 0.0
    return float(np.var(np.array(word_sentiments)))


# ── Phase 2: Cognitive Distortion Detection ───────────────────────────────────

def compute_cdi(text: str) -> dict:
    """
    Phase 2: Cognitive Distortion Detection.

    Detects four distortion types; applies 1.4x synergy multiplier
    when 2 or more co-occur. CDI ∈ [0, 1].
    """
    text_lower = text.lower()
    detected   = {}

    for name, meta in DISTORTION_PATTERNS.items():
        hits = sum(len(re.findall(p, text_lower)) for p in meta["patterns"])
        if hits > 0:
            density       = min(hits / max(len(text.split()), 1) * 10, 1.0)
            detected[name] = meta["weight"] * density

    raw = sum(detected.values())
    if len(detected) >= 2:
        raw *= 1.4   # synergy multiplier

    return {
        "CDI":                float(np.clip(raw, 0.0, 1.0)),
        "distortions_found":  list(detected.keys()),
        "distortion_weights": detected,
    }


# ── Phase 3: Behavioral Inference ────────────────────────────────────────────

def behavioral_score(text: str) -> dict:
    """
    Phase 3: Behavioral Inference.
    Measures active vs passive language; infers motivation level.
    B ∈ [-1, +1].
    """
    words        = set(_tokenize(text))
    active_count  = len(words & ACTIVE_WORDS)
    passive_count = len(words & PASSIVE_WORDS)
    motive_count  = len(words & MOTIVATION_WORDS)

    total = active_count + passive_count
    B     = (active_count - passive_count) / total if total else 0.0

    motivation = (
        "High"     if motive_count >= 3 else
        "Moderate" if motive_count >= 1 else
        "Low"
    )
    return {
        "B":             float(np.clip(B, -1.0, 1.0)),
        "motivation":    motivation,
        "active_count":  active_count,
        "passive_count": passive_count,
        "motive_count":  motive_count,
    }


# ── USP Scores (independently computed) ──────────────────────────────────────

def compute_happiness_score(text: str, sentiment_boost: float) -> float:
    """
    USP Score 1: Happiness Score.
    = positive_words - negative_words + sentiment_boost
    Normalised to [0, 10].
    """
    wc  = Counter(_tokenize(text))
    pos = sum(wc[w] for w in positive_words    if w in wc)
    neg = sum(wc[w] for w in negative_words    if w in wc)
    raw = (pos - neg) + (sentiment_boost * 3)
    return float(np.clip((raw + 5) / 10 * 10, 0, 10))


def compute_confidence_score(text: str) -> float:
    """
    USP Score 2: Confidence Score.
    = confidence_words - low_confidence_words
    Normalised to [0, 10].
    """
    wc       = Counter(_tokenize(text))
    conf     = sum(wc[w] for w in confidence_words     if w in wc)
    low_conf = sum(wc[w] for w in low_confidence_words if w in wc)
    raw      = conf - low_conf
    return float(np.clip((raw + 5) / 10 * 10, 0, 10))


def compute_satisfaction_score(text: str) -> float:
    """
    USP Score 3: Satisfaction Score.
    = satisfaction_words - dissatisfaction_words
    Normalised to [0, 10].
    """
    wc     = Counter(_tokenize(text))
    sat    = sum(wc[w] for w in satisfaction_words    if w in wc)
    dissat = sum(wc[w] for w in dissatisfaction_words if w in wc)
    raw    = sat - dissat
    return float(np.clip((raw + 5) / 10 * 10, 0, 10))


# ── Temporal trend & anomaly helpers ─────────────────────────────────────────

def _temporal_trend(text: str, history: list) -> float:
    """T ∈ [-1,+1]: blend of lexical temporal cues and historical score delta."""
    words = set(_tokenize(text))
    t_pos = len(words & TEMPORAL_POSITIVE)
    t_neg = len(words & TEMPORAL_NEGATIVE)
    lex_T = (t_pos - t_neg) / max(t_pos + t_neg, 1) if (t_pos + t_neg) else 0.0
    hist_T = float(np.tanh(history[-1] - history[-2])) if len(history) >= 2 else 0.0
    return float(np.clip(0.6 * lex_T + 0.4 * hist_T, -1.0, 1.0))


def _anomaly_score(score: float, history: list) -> float:
    """A ∈ [0,1]: z-score anomaly detection (requires >= 3 historical points)."""
    if len(history) < 3:
        return 0.0
    arr = np.array(history)
    std = arr.std()
    if std < 1e-6:
        return 0.0
    z = abs((score - arr.mean()) / std)
    return float(np.clip(z / 3.0, 0.0, 1.0))


# ── Phase 4: MHSI Scoring Engine ─────────────────────────────────────────────

def compute_mhsi(
    S: float,
    V: float,
    CDI: float,
    happiness: float,
    satisfaction: float,
    T: float,
    B: float,
    A: float,
    weights: tuple = (2.5, 1.8, 1.2, 1.5, 0.8, 1.0),
) -> float:
    """
    Phase 4: MHSI Scoring Engine.

    Formula:
      MHSI = sigmoid( a1*S - a2*V^2 + a3*log(1+C) + a4*tanh(F) + a5*T + a6*B ) * (1-A)

    Where:
      C = 1 - CDI              (cognitive clarity)
      F = (H/10 + Sat/10) / 2  (fulfilment, normalised)

    Returns MHSI ∈ [0, 10].
    """
    a1, a2, a3, a4, a5, a6 = weights
    C  = 1.0 - CDI
    F  = (happiness / 10.0 + satisfaction / 10.0) / 2.0

    lc = (
        a1 * S
        - a2 * (V ** 2)
        + a3 * math.log(1 + C)
        + a4 * math.tanh(F)
        + a5 * T
        + a6 * B
    )

    sigmoid = lambda x: 1.0 / (1.0 + math.exp(-x))
    return float(np.clip(sigmoid(lc) * (1.0 - A) * 10.0, 0.0, 10.0))


# ── Master analyze() ──────────────────────────────────────────────────────────

def analyze(text: str, score_history: list, weights: tuple = (2.5, 1.8, 1.2, 1.5, 0.8, 1.0)) -> dict:
    """
    Runs the complete 4-phase pipeline and returns a unified result dict.
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

    # Temporal trend (uses existing history)
    T = _temporal_trend(text, score_history)

    # Preliminary MHSI for anomaly detection (A=0 first pass)
    prelim = compute_mhsi(S, V, CDI, happiness, satisfaction, T, B, A=0.0, weights=weights)
    A      = _anomaly_score(prelim, score_history)

    # Final MHSI
    mhsi = compute_mhsi(S, V, CDI, happiness, satisfaction, T, B, A, weights=weights)

    # Risk classification
    if mhsi >= 7.0:
        risk, risk_color, risk_icon = "Good",      "#22c55e", "✅"
    elif mhsi >= 4.0:
        risk, risk_color, risk_icon = "Moderate",  "#f59e0b", "⚠️"
    else:
        risk, risk_color, risk_icon = "High Risk", "#ef4444", "🚨"

    return {
        "happiness":    happiness,
        "confidence":   confidence,
        "satisfaction": satisfaction,
        "mhsi":        mhsi,
        "risk":        risk,
        "risk_color":  risk_color,
        "risk_icon":   risk_icon,
        "S": S, "V": V, "CDI": CDI, "B": B, "T": T, "A": A,
        "sentiment":   sent,
        "cdi_result":  cdi_result,
        "behavioral":  beh,
        "timestamp":   datetime.datetime.now().strftime("%H:%M:%S"),
        "text_preview": text[:80] + ("..." if len(text) > 80 else ""),
    }


# ════════════════════════════════════════════════════════════════════════════
# 3 · CHART BUILDERS
# ════════════════════════════════════════════════════════════════════════════

_TR   = "rgba(0,0,0,0)"
_PBG  = "rgba(255,255,255,0.02)"
_FC   = "#94a3b8"
_GC   = "rgba(255,255,255,0.07)"
_BASE = dict(
    paper_bgcolor=_TR, plot_bgcolor=_PBG,
    font=dict(family="DM Sans, sans-serif", color=_FC, size=12),
    margin=dict(l=20, r=20, t=24, b=16),
)


def _radar_chart(result: dict) -> go.Figure:
    cats = ["Happiness", "Confidence", "Satisfaction", "Clarity", "Behaviour", "Sentiment"]
    vals = [
        result["happiness"],
        result["confidence"],
        result["satisfaction"],
        (1 - result["CDI"]) * 10,
        (result["B"] + 1) / 2 * 10,
        (result["S"] + 1) / 2 * 10,
    ]
    vc = vals + [vals[0]]
    cc = cats + [cats[0]]
    fig = go.Figure(go.Scatterpolar(
        r=vc, theta=cc, fill="toself",
        fillcolor="rgba(139,92,246,0.15)",
        line=dict(color="#8b5cf6", width=2),
        marker=dict(size=6, color="#a78bfa"),
    ))
    fig.update_layout(
        **_BASE, showlegend=False, height=320,
        polar=dict(
            bgcolor="rgba(255,255,255,0.02)",
            radialaxis=dict(visible=True, range=[0, 10],
                            tickfont=dict(size=9), gridcolor=_GC),
            angularaxis=dict(tickfont=dict(size=11, color="#c4b5fd"), gridcolor=_GC),
        ),
    )
    return fig


def _history_chart(score_history: list) -> go.Figure:
    x = list(range(1, len(score_history) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=score_history,
        fill="tozeroy", fillcolor="rgba(99,102,241,0.10)",
        line=dict(color="#6366f1", width=2.5),
        mode="lines+markers",
        marker=dict(size=7, color="#818cf8", line=dict(color="#1e1b4b", width=2)),
    ))
    for y0, y1, col, lbl in [
        (7, 10, "rgba(34,197,94,0.06)",  "Good"),
        (4,  7, "rgba(245,158,11,0.06)", "Moderate"),
        (0,  4, "rgba(239,68,68,0.06)",  "High Risk"),
    ]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=col, line_width=0,
                      annotation_text=lbl, annotation_position="right")
    fig.update_layout(
        **_BASE, showlegend=False, height=240,
        yaxis=dict(range=[0, 10.2], gridcolor=_GC, title="MHSI"),
        xaxis=dict(title="Session #", gridcolor=_GC, tickmode="linear"),
    )
    return fig


def _distortion_bar(cdi_result: dict) -> go.Figure:
    dw = cdi_result["distortion_weights"]
    if dw:
        labels = [d.replace("_", " ").title() for d in dw]
        values = list(dw.values())
        colors = ["#f87171", "#fb923c", "#facc15", "#a78bfa"][:len(labels)]
    else:
        labels, values, colors = ["No distortions detected"], [0.01], ["#22c55e"]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors, marker_line_color="rgba(0,0,0,0)",
    ))
    fig.update_layout(
        **_BASE,
        xaxis=dict(range=[0, 0.55], gridcolor=_GC),
        yaxis=dict(autorange="reversed"),
        height=max(130, 52 * len(labels)),
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
# 4 · STREAMLIT APP
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Mind Matrix V3",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(135deg, #0d0d1a 0%, #0f1629 55%, #0d0d1a 100%); color: #e2e8f0; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0a0a1f 0%,#111827 100%); border-right:1px solid rgba(139,92,246,0.2); }

/* Header */
.mm-title { font-family:'Space Mono',monospace; font-size:2rem; font-weight:700;
  background:linear-gradient(90deg,#a855f7,#38bdf8,#34d399);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  background-clip:text; letter-spacing:-0.02em; margin-bottom:0; }
.mm-sub { color:#64748b; font-size:0.78rem; letter-spacing:0.15em; text-transform:uppercase; }

/* Section labels */
.sec-label { font-family:'Space Mono',monospace; font-size:0.68rem; letter-spacing:0.18em;
  text-transform:uppercase; color:#6366f1; border-bottom:1px solid rgba(99,102,241,0.25);
  padding-bottom:0.3rem; margin:1.2rem 0 0.7rem; }

/* USP cards */
.usp-card { background:linear-gradient(135deg,rgba(255,255,255,0.05) 0%,rgba(255,255,255,0.02) 100%);
  border:1px solid rgba(255,255,255,0.09); border-radius:16px; padding:1.2rem 1.3rem;
  text-align:center; transition:transform 0.2s,border-color 0.2s; }
.usp-card:hover { transform:translateY(-3px); border-color:rgba(168,85,247,0.4); }
.usp-icon { font-size:1.2rem; margin-bottom:0.3rem; }
.usp-lbl { font-size:0.66rem; letter-spacing:0.16em; text-transform:uppercase; color:#475569; margin-bottom:0.35rem; }
.usp-val { font-family:'Space Mono',monospace; font-size:1.9rem; font-weight:700; line-height:1; }

/* MHSI hero */
.mhsi-hero { background:linear-gradient(135deg,rgba(139,92,246,0.14) 0%,rgba(56,189,248,0.09) 100%);
  border:1px solid rgba(139,92,246,0.35); border-radius:20px; padding:1.6rem 1.4rem;
  text-align:center; position:relative; overflow:hidden; }
.mhsi-hero::before { content:''; position:absolute; top:-50%; left:-50%; width:200%; height:200%;
  background:radial-gradient(circle,rgba(139,92,246,0.07) 0%,transparent 60%); pointer-events:none; }
.mhsi-lbl { font-size:0.66rem; letter-spacing:0.2em; text-transform:uppercase; color:#64748b; margin-bottom:0.4rem; }
.mhsi-score { font-family:'Space Mono',monospace; font-size:3.8rem; font-weight:700; line-height:1; }
.risk-badge { display:inline-block; padding:0.32rem 1rem; border-radius:50px;
  font-size:0.8rem; font-weight:600; letter-spacing:0.05em; margin-top:0.45rem; }

/* Pills */
.pill { display:inline-block; background:rgba(99,102,241,0.13); border:1px solid rgba(99,102,241,0.28);
  border-radius:50px; padding:0.18rem 0.65rem; font-size:0.7rem; color:#a5b4fc; margin:0.12rem; }

/* Recommendations */
.rec-item { background:rgba(255,255,255,0.03); border-left:3px solid #6366f1;
  border-radius:0 8px 8px 0; padding:0.55rem 0.9rem; margin-bottom:0.4rem;
  font-size:0.86rem; color:#cbd5e1; }

/* History row */
.hist-row { background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06);
  border-radius:8px; padding:0.5rem 0.85rem; margin-bottom:0.3rem; font-size:0.78rem;
  display:flex; justify-content:space-between; align-items:center; }

/* Question items */
.q-item { margin:5px 0; color:#cbd5e1; font-size:0.86rem; }

/* Inputs */
textarea { background:rgba(255,255,255,0.04) !important; border:1px solid rgba(139,92,246,0.3) !important;
  border-radius:12px !important; color:#e2e8f0 !important; font-size:15px !important; }
textarea:focus { border-color:#8b5cf6 !important; }

/* Buttons */
.stButton > button { background:linear-gradient(135deg,#7c3aed,#4f46e5) !important;
  color:white !important; border:none !important; border-radius:10px !important;
  padding:0.52rem 1.2rem !important; font-weight:600 !important; letter-spacing:0.04em !important;
  width:100% !important; transition:opacity 0.2s,transform 0.2s !important; }
.stButton > button:hover { opacity:0.87 !important; transform:translateY(-1px) !important; }

/* Progress */
.stProgress > div > div > div { background-color:#8b5cf6 !important; }
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding-top:1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "questions_pool"  not in st.session_state:
    st.session_state.questions_pool  = random.sample(questions, 10)
if "score_history"   not in st.session_state:
    st.session_state.score_history   = []
if "history"         not in st.session_state:
    st.session_state.history         = []
if "input_text"      not in st.session_state:
    st.session_state.input_text      = ""
if "result"          not in st.session_state:
    st.session_state.result          = None


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:0.8rem 0 1.2rem'>
      <div style='font-size:2rem'>🧠</div>
      <div style='font-family:Space Mono,monospace;font-size:0.95rem;font-weight:700;
        background:linear-gradient(90deg,#a855f7,#38bdf8);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text'>
        Mind Matrix V3</div>
      <div style='color:#475569;font-size:0.65rem;letter-spacing:0.14em;text-transform:uppercase;margin-top:2px'>
        MHSI Engine · v3.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Pipeline Weights")
    a1 = st.slider("Sentiment (a1)",    0.5, 5.0, 2.5, 0.1)
    a2 = st.slider("Variance pen (a2)", 0.5, 4.0, 1.8, 0.1)
    a3 = st.slider("Clarity (a3)",      0.5, 3.0, 1.2, 0.1)
    a4 = st.slider("Fulfillment (a4)",  0.5, 3.0, 1.5, 0.1)
    a5 = st.slider("Temporal (a5)",     0.1, 2.0, 0.8, 0.1)
    a6 = st.slider("Behaviour (a6)",    0.1, 2.0, 1.0, 0.1)

    st.markdown("---")
    st.markdown("### 📖 Pipeline Phases")
    st.markdown("""
    <div style='color:#64748b;font-size:0.78rem;line-height:1.75'>
      <b style='color:#8b5cf6'>Phase 1</b> — VADER + non-linear transform<br>
      <b style='color:#38bdf8'>Phase 2</b> — Cognitive Distortion (CDI)<br>
      <b style='color:#34d399'>Phase 3</b> — Behavioral Inference<br>
      <b style='color:#f59e0b'>Phase 4</b> — Sigmoid MHSI Formula<br><br>
      <b>USP</b>: Happiness · Confidence · Satisfaction — computed independently.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📚 Mental Health Resources")
    st.markdown("""
    <div style='font-size:0.78rem;color:#64748b;line-height:1.9'>
      📞 Crisis: <b style='color:#a5b4fc'>1-800-273-8255</b><br>
      🌐 <a href='https://www.mhanational.org' style='color:#6366f1'>Mental Health America</a><br>
      🧘 <a href='https://www.mindful.org'     style='color:#6366f1'>Mindfulness Exercises</a><br>
      🧠 <a href='https://www.psychologytools.com' style='color:#6366f1'>CBT Tools</a>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown("---")
        if st.button("🗑️  Clear History"):
            st.session_state.history.clear()
            st.session_state.score_history.clear()
            st.session_state.result = None
            st.rerun()


# ── HEADER ───────────────────────────────────────────────────────────────────
hcol_icon, hcol_txt = st.columns([1, 9], gap="small")
with hcol_icon:
    try:
        st.image("favicon.png", width=68)
    except Exception:
        st.markdown("<div style='font-size:2.8rem;margin-top:0.25rem'>🧠</div>",
                    unsafe_allow_html=True)
with hcol_txt:
    st.markdown("<p class='mm-title'>Mind Matrix V3</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='mm-sub'>Your Personal Mental Wellness Intelligence Platform · MHSI Engine</p>",
        unsafe_allow_html=True,
    )

st.markdown(
    "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:0.5rem 0 0.9rem'>",
    unsafe_allow_html=True,
)


# ── QUESTIONS ─────────────────────────────────────────────────────────────────
with st.expander("📋 Today's Assessment Questions", expanded=True):
    qc1, qc2 = st.columns(2)
    for i, q in enumerate(st.session_state.questions_pool):
        (qc1 if i % 2 == 0 else qc2).markdown(
            f"<div class='q-item'>🔹 {q}</div>", unsafe_allow_html=True
        )


# ── INPUT FORM ────────────────────────────────────────────────────────────────
with st.form("input_form", clear_on_submit=False):
    user_input = st.text_area(
        "✍️ Write your responses here (combine answers in one paragraph):",
        height=175,
        value=st.session_state.input_text,
        placeholder=(
            "Reflect on the questions above and write freely. "
            "e.g. 'Today I felt a bit anxious about work but managed to finish my project "
            "and felt really proud of myself. My energy was good overall...'"
        ),
    )
    fc1, fc2, fc3 = st.columns([3, 2, 2], gap="small")
    with fc1: analyze_btn = st.form_submit_button("🔬  Analyze Responses")
    with fc2: clear_btn   = st.form_submit_button("🧹  Clear Input")
    with fc3: new_btn     = st.form_submit_button("🔄  New Questions")

    if clear_btn:
        st.session_state.input_text = ""
        st.rerun()

    if new_btn:
        st.session_state.questions_pool = random.sample(questions, 10)
        st.session_state.input_text     = ""
        st.session_state.result         = None
        st.rerun()


# ── ANALYSIS EXECUTION ────────────────────────────────────────────────────────
if analyze_btn:
    if user_input and user_input.strip():
        st.session_state.input_text = user_input
        weights_tuple = (a1, a2, a3, a4, a5, a6)

        with st.spinner("🧠 Running multi-phase MHSI pipeline..."):
            pb = st.progress(0)
            for pct in range(100):
                time.sleep(0.013)
                pb.progress(pct + 1)

            result = analyze(
                user_input.strip(),
                st.session_state.score_history,
                weights=weights_tuple,
            )

        st.session_state.score_history.append(result["mhsi"])
        st.session_state.history.insert(0, result)
        st.session_state.result = result
        st.success("✅ Analysis complete!")
    else:
        st.warning("⚠️ Please write your responses before analysing.")


# ── RESULTS ───────────────────────────────────────────────────────────────────
result = st.session_state.result

if result:

    # ── USP Score Cards ──────────────────────────────────────────────────────
    st.markdown(
        "<div class='sec-label'>◆ USP SCORES — INDEPENDENTLY COMPUTED</div>",
        unsafe_allow_html=True,
    )
    uc1, uc2, uc3 = st.columns(3, gap="medium")

    def _clr(s: float) -> str:
        return "#22c55e" if s >= 7 else ("#f59e0b" if s >= 4 else "#ef4444")

    with uc1:
        c = _clr(result["happiness"])
        st.markdown(f"""
        <div class='usp-card'><div class='usp-icon'>😊</div>
          <div class='usp-lbl'>Happiness Score</div>
          <div class='usp-val' style='color:{c}'>{result["happiness"]:.1f}</div>
          <div style='font-size:0.65rem;color:#334155;margin-top:0.25rem'>/ 10</div>
        </div>""", unsafe_allow_html=True)
        st.progress(result["happiness"] / 10)

    with uc2:
        c = _clr(result["confidence"])
        st.markdown(f"""
        <div class='usp-card'><div class='usp-icon'>💪</div>
          <div class='usp-lbl'>Confidence Score</div>
          <div class='usp-val' style='color:{c}'>{result["confidence"]:.1f}</div>
          <div style='font-size:0.65rem;color:#334155;margin-top:0.25rem'>/ 10</div>
        </div>""", unsafe_allow_html=True)
        st.progress(result["confidence"] / 10)

    with uc3:
        c = _clr(result["satisfaction"])
        st.markdown(f"""
        <div class='usp-card'><div class='usp-icon'>🎯</div>
          <div class='usp-lbl'>Satisfaction Score</div>
          <div class='usp-val' style='color:{c}'>{result["satisfaction"]:.1f}</div>
          <div style='font-size:0.65rem;color:#334155;margin-top:0.25rem'>/ 10</div>
        </div>""", unsafe_allow_html=True)
        st.progress(result["satisfaction"] / 10)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── MHSI Hero + Radar ────────────────────────────────────────────────────
    hero_col, radar_col = st.columns([2, 3], gap="large")

    with hero_col:
        rc  = result["risk_color"]
        bbg = {
            "Good":      "rgba(34,197,94,0.13)",
            "Moderate":  "rgba(245,158,11,0.13)",
            "High Risk": "rgba(239,68,68,0.13)",
        }[result["risk"]]

        st.markdown(f"""
        <div class='mhsi-hero'>
          <div class='mhsi-lbl'>◆ MENTAL HEALTH SCORE INDEX</div>
          <div class='mhsi-score' style='color:{rc}'>{result["mhsi"]:.2f}</div>
          <div style='color:#475569;font-size:0.76rem;margin:0.2rem 0'>out of 10.00</div>
          <div class='risk-badge' style='background:{bbg};color:{rc};border:1px solid {rc}40'>
            {result["risk_icon"]} {result["risk"]}
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:0.65rem'>", unsafe_allow_html=True)
        st.progress(result["mhsi"] / 10)
        st.markdown("""
        <div style='display:flex;justify-content:space-between;font-size:0.64rem;
                    color:#334155;margin-top:-0.18rem'>
          <span>0 · High Risk</span><span>5 · Moderate</span><span>10 · Good</span>
        </div></div>""", unsafe_allow_html=True)

        pills = [
            ("S",   f"{result['S']:.2f}",   "Transformed Sentiment"),
            ("V",   f"{result['V']:.3f}",   "Emotional Variance"),
            ("CDI", f"{result['CDI']:.2f}", "Cognitive Distortion Index"),
            ("B",   f"{result['B']:.2f}",   "Behavioral Score"),
            ("T",   f"{result['T']:.2f}",   "Temporal Trend"),
            ("A",   f"{result['A']:.2f}",   "Anomaly Score"),
        ]
        st.markdown(
            "<div style='display:flex;flex-wrap:wrap;gap:0.18rem;margin-top:0.75rem'>" +
            "".join(f"<span class='pill' title='{tip}'><b>{k}</b> {v}</span>"
                    for k, v, tip in pills) +
            "</div>",
            unsafe_allow_html=True,
        )

    with radar_col:
        st.markdown(
            "<div class='sec-label'>◆ MULTI-DIMENSIONAL WELLNESS PROFILE</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(_radar_chart(result), use_container_width=True,
                        config={"displayModeBar": False})

    # ── Cognitive Distortions + Behavioral ──────────────────────────────────
    dc, bc = st.columns(2, gap="large")

    with dc:
        st.markdown(
            "<div class='sec-label'>◆ PHASE 2 · COGNITIVE DISTORTION ANALYSIS</div>",
            unsafe_allow_html=True,
        )
        cdi = result["cdi_result"]
        st.plotly_chart(_distortion_bar(cdi), use_container_width=True,
                        config={"displayModeBar": False})
        if cdi["distortions_found"]:
            patterns = ", ".join(d.replace("_", " ").title() for d in cdi["distortions_found"])
            st.markdown(f"**CDI:** `{cdi['CDI']:.3f}` &nbsp;|&nbsp; **Detected:** {patterns}",
                        unsafe_allow_html=True)
            if len(cdi["distortions_found"]) >= 2:
                st.markdown("⚡ *Synergy multiplier (1.4×) applied.*")
        else:
            st.markdown("✅ *No significant cognitive distortions detected.*")

    with bc:
        st.markdown(
            "<div class='sec-label'>◆ PHASE 3 · BEHAVIORAL INFERENCE</div>",
            unsafe_allow_html=True,
        )
        beh = result["behavioral"]
        a_c, p_c  = beh["active_count"], beh["passive_count"]
        neutral    = max(0, 10 - a_c - p_c)

        fig_beh = go.Figure(go.Pie(
            labels=["Active", "Passive", "Neutral"],
            values=[max(a_c, 0.01), max(p_c, 0.01), max(neutral, 0.01)],
            hole=0.55,
            marker_colors=["#22c55e", "#ef4444", "#334155"],
            textfont=dict(size=11, color="white"),
        ))
        fig_beh.update_layout(
            **_BASE, height=205, showlegend=True,
            legend=dict(font=dict(color=_FC, size=11)),
            annotations=[dict(
                text=f"B={beh['B']:+.2f}", x=0.5, y=0.5,
                font=dict(size=14, color="#a78bfa", family="Space Mono"),
                showarrow=False,
            )],
        )
        st.plotly_chart(fig_beh, use_container_width=True, config={"displayModeBar": False})

        mot_c = {"High": "#22c55e", "Moderate": "#f59e0b", "Low": "#ef4444"}[beh["motivation"]]
        st.markdown(
            f"**Motivation:** <span style='color:{mot_c};font-weight:700'>{beh['motivation']}</span>"
            f" &nbsp;|&nbsp; Motivational cues: `{beh['motive_count']}`",
            unsafe_allow_html=True,
        )

    # ── Recommendations ──────────────────────────────────────────────────────
    st.markdown(
        "<div class='sec-label'>◆ PERSONALISED RECOMMENDATIONS</div>",
        unsafe_allow_html=True,
    )

    recs = []
    risk = result["risk"]
    cdi_det = result["cdi_result"]["distortions_found"]
    beh     = result["behavioral"]

    if risk == "Good":
        recs += [
            "🌟 Keep nurturing your positive habits — consistency compounds over time.",
            "📓 Continue journaling and gratitude practice daily.",
            "🎯 Set one meaningful growth goal this week to sustain momentum.",
            "🤝 Share your positivity — it uplifts others and reinforces your own wellbeing.",
        ]
    elif risk == "Moderate":
        recs += [
            "🧘 Practice 10 min of mindfulness or breathing exercises daily.",
            "🏃 Light physical activity (30 min/day) significantly boosts mood.",
            "📓 Journaling can help identify and break recurring thought patterns.",
            "👥 Connect with a trusted friend or family member this week.",
        ]
    else:
        recs += [
            "🆘 Consider speaking with a licensed mental health professional.",
            "📞 Reach out to a trusted person — you don't have to face this alone.",
            "🧘 Practice the 5-4-3-2-1 grounding technique when feeling overwhelmed.",
            "💊 Establish a simple self-care routine: sleep, water, short walks.",
        ]

    if "catastrophizing" in cdi_det:
        recs.append("🔍 Challenge 'worst-case' thinking: ask 'What is the realistic outcome?'")
    if "helplessness" in cdi_det:
        recs.append("💪 List 3 small actions you *can* take today — agency builds gradually.")
    if "all_or_nothing" in cdi_det:
        recs.append("🎨 Embrace nuance — most situations exist on a spectrum, not in absolutes.")
    if beh["motivation"] == "Low":
        recs.append("🕯️ Break large tasks into micro-steps and celebrate tiny wins.")
    if beh["passive_count"] > beh["active_count"]:
        recs.append("🚀 Reframe 'I can't' as 'I'm learning to' — language shapes mindset.")

    for rec in recs[:5]:
        st.markdown(f"<div class='rec-item'>{rec}</div>", unsafe_allow_html=True)

    # ── Session History ───────────────────────────────────────────────────────
    if len(st.session_state.history) > 1:
        st.markdown(
            "<div class='sec-label'>◆ SESSION HISTORY · MHSI TREND</div>",
            unsafe_allow_html=True,
        )
        tc, lc = st.columns([3, 2], gap="large")

        with tc:
            st.plotly_chart(_history_chart(st.session_state.score_history),
                            use_container_width=True, config={"displayModeBar": False})

        with lc:
            rc_map = {"Good": "#22c55e", "Moderate": "#f59e0b", "High Risk": "#ef4444"}
            for i, h in enumerate(st.session_state.history[:8]):
                hrc = rc_map.get(h["risk"], "#94a3b8")
                n   = len(st.session_state.history) - i
                st.markdown(f"""
                <div class='hist-row'>
                  <div style='flex:1;overflow:hidden;white-space:nowrap;text-overflow:ellipsis'>
                    <span style='color:#475569;font-size:0.66rem'>#{n}</span>
                    &nbsp;<span style='color:#94a3b8'>{h['text_preview']}</span>
                  </div>
                  <div style='margin-left:0.5rem;white-space:nowrap'>
                    <span style='color:{hrc};font-family:Space Mono,monospace;font-weight:700'>
                      {h['mhsi']:.1f}</span>
                    <span style='color:#334155;font-size:0.65rem'>&nbsp;{h['timestamp']}</span>
                  </div>
                </div>""", unsafe_allow_html=True)

    # ── Raw sentiment expander ────────────────────────────────────────────────
    with st.expander("🔬 Phase 1 · Raw Sentiment Detail", expanded=False):
        sent = result["sentiment"]
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("VADER Compound", f"{sent['raw_compound']:.3f}")
        sc2.metric("Transformed S",  f"{sent['transformed_S']:.3f}")
        sc3.metric("Positive",       f"{sent['pos']:.1%}")
        sc4.metric("Negative",       f"{sent['neg']:.1%}")

else:
    # ── Welcome placeholder ───────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center;padding:2.2rem 1.5rem;color:#334155'>
      <div style='font-size:3.2rem;margin-bottom:0.7rem'>🧠</div>
      <div style='font-family:Space Mono,monospace;font-size:0.95rem;color:#6366f1;margin-bottom:0.5rem'>
        Awaiting your responses</div>
      <div style='color:#475569;font-size:0.86rem;max-width:500px;margin:0 auto;line-height:1.8'>
        Reflect on the questions above, write freely in the text box, then click
        <b style='color:#8b5cf6'>Analyze Responses</b>. The 4-phase MHSI pipeline
        will independently compute your <b>Happiness</b>, <b>Confidence</b>, and
        <b>Satisfaction</b> scores and fuse them into a final
        <b>Mental Health Score Index</b>.
      </div>
      <div style='margin-top:1.6rem;display:flex;justify-content:center;gap:1rem;flex-wrap:wrap'>
        <div style='background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.3);
          border-radius:12px;padding:0.8rem 1.2rem;font-size:0.78rem;color:#a5b4fc'>😊 Happiness Score</div>
        <div style='background:rgba(56,189,248,0.1);border:1px solid rgba(56,189,248,0.3);
          border-radius:12px;padding:0.8rem 1.2rem;font-size:0.78rem;color:#7dd3fc'>💪 Confidence Score</div>
        <div style='background:rgba(52,211,153,0.1);border:1px solid rgba(52,211,153,0.3);
          border-radius:12px;padding:0.8rem 1.2rem;font-size:0.78rem;color:#6ee7b7'>🎯 Satisfaction Score</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:1.6rem 0 0.5rem'>
<div style='text-align:center;color:#1e293b;font-size:0.7rem;letter-spacing:0.1em'>
  MIND MATRIX V3 · MHSI ENGINE · For research &amp; wellness tracking only.
  Not a substitute for professional mental health advice. · Crisis: 1-800-273-8255
</div>
""", unsafe_allow_html=True)