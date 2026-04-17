import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import time
from collections import Counter
import random

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Word sets (keep the same as original)
positive_words = {"happy", "joyful", "excited", "great", "wonderful", "amazing", "cheerful", "delighted", "fantastic", "thrilled", "blissful", "optimistic", "enthusiastic", "content", "peaceful", "radiant", "upbeat", "lively", "exhilarated", "bright", "hopeful", "motivated", "inspired", "glowing", "blessed", "prosperous", "adventurous", "brilliant", "ecstatic", "vivacious", "elated", "satisfied", "vibrant", "accomplished", "energetic", "loving", "passionate", "grateful", "confident", "victorious", "playful", "serene", "encouraging", "spirited", "flourishing", "thriving", "empowered", "free", "affectionate", "charming", "fulfilled", "wholesome", "resilient", "warmhearted", "remarkable", "rejuvenated", "impressive", "successful", "lighthearted", "sunny", "bubbly", "fun-loving", "dreamy", "tranquil", "jubilant", "fearless", "dynamic", "secure", "harmonious", "incredible", "astonishing", "soothing", "magnetic", "compassionate", "marvelous", "heartwarming", "enriched", "sparkling", "amusing", "courageous", "relieved", "enlightened", "productive", "nurturing", "triumphant", "gentle", "embracing", "heroic", "pioneering"}

negative_words = {"sad", "angry", "frustrated", "depressed", "worried", "miserable", "fearful", "hopeless", "irritated", "anxious", "disappointed", "nervous", "discouraged", "lonely", "exhausted", "rejected", "insecure", "bitter", "uneasy", "restless", "shattered", "vulnerable", "overwhelmed", "unwanted", "worthless", "drained", "devastated", "grief-stricken", "pessimistic", "moody", "tormented", "disturbed", "sorrowful", "weary", "neglected", "upset", "gloomy", "distressed", "heartbroken", "abandoned", "lost", "suffering", "ashamed", "fatigued", "skeptical", "distrustful", "unfocused", "weak", "remorseful", "doubting", "grief-ridden", "resentful", "furious", "infuriated", "withdrawn", "unmotivated", "powerless", "unhappy", "agonized", "isolated", "unworthy", "hesitant", "embarrassed", "dreadful", "spiteful", "guilt-ridden", "irate", "victimized", "enraged", "displeased", "panicked", "burdened", "disheartened", "excluded", "trapped", "restrained", "grief-filled", "suppressed", "misunderstood", "detached", "regretful", "self-doubtful", "pessimistic", "betrayed", "traumatized", "reluctant", "distrustful", "broken"}

confidence_words = {"able", "confident", "certain", "assured", "capable", "determined", "fearless", "assertive", "strong", "courageous", "bold", "self-reliant", "empowered", "daring", "self-sufficient", "tenacious", "independent", "ambitious", "motivated", "secure", "competent", "unstoppable", "undaunted", "solid", "wise", "visionary", "efficient", "passionate", "winner", "firm", "radiant", "victorious", "disciplined", "consistent", "energetic", "unshakable", "mindful", "resilient", "composed", "resourceful", "influential", "proactive", "charismatic", "logical", "masterful", "convincing", "heroic", "skillful", "expert", "outstanding", "remarkable", "powerful", "esteemed", "mindful", "ambitious", "relentless", "undeterred", "pragmatic", "focused", "intellectual", "self-sustained", "smart", "sophisticated", "talented", "proficient", "inspired", "strategic", "balanced", "dedicated", "innovative", "well-prepared", "keen", "sharp", "unyielding", "thoughtful", "steadfast", "grounded", "accomplished", "unwavering", "enlightened"}

low_confidence_words = {"unable", "doubt", "uncertain", "nervous", "hesitant", "fearful", "anxious", "reluctant", "unsure", "worried", "insecure", "apprehensive", "timid", "discouraged", "doubtful", "intimidated", "uneasy", "shy", "fragile", "lacking", "unsteady", "skeptical", "disoriented", "incapable", "exhausted", "confused", "defeated", "powerless", "self-conscious", "faltering", "unmotivated", "weak", "struggling", "frightened", "burdened", "helpless", "repressed", "ineffective", "troubled", "downcast", "lost", "passive", "vulnerable", "indecisive", "hesitant", "discouraged", "indifferent", "worried", "nervous", "wavering", "shaken", "inhibited", "stammering", "unready", "uncertain", "worried", "flustered", "overwhelmed", "withdrawn", "unprepared", "stressed", "self-doubting", "unconfident", "dreading", "distrustful", "inhibited", "floundering", "hesitating", "timid", "overthinking", "frozen", "overcautious", "inexpressive", "pessimistic", "wary", "afraid", "unsure", "guilt-ridden", "second-guessing", "doubt-ridden", "dithering", "wavering", "mistrusting", "fear-stricken", "panicky", "numb", "self-conscious", "unassertive", "trembling", "unsteady", "quivering", "frightened", "overwhelmed", "lacking", "hesitating"}

satisfaction_words = {
    "satisfied", "content", "fulfilled", "happy", "pleased", "gratified", "delighted", "joyful", "cheerful", "blissful",
    "serene", "comfortable", "relaxed", "ecstatic", "thrilled", "elated", "excited", "euphoric", "thankful", "grateful",
    "appreciative", "secure", "reassured", "optimistic", "hopeful", "enthusiastic", "encouraged", "positive", "relieved",
    "prosperous", "harmonious", "balanced", "accomplished", "triumphant", "victorious", "proud", "glad", "overjoyed",
    "radiant", "uplifted", "inspired", "motivated", "peaceful", "satiated", "blessed", "merry", "jubilant", "animated",
    "cheery", "buoyant", "lighthearted", "exhilarated", "rejuvenated", "flourishing", "thriving", "sunny", "heartened",
    "wholesome", "loving", "joyous", "vivacious", "upbeat", "bubbly", "exultant", "enlightened", "giddy", "warmhearted",
    "spirited", "appreciated", "valued", "cherished", "embraced", "rewarded", "celebrated", "exalted", "acclaimed",
    "commended", "respected", "admired", "acknowledged", "treasured", "nurtured", "sympathetic", "caring", "fulfilled",
    "generous", "considerate", "compassionate", "harmonized", "steady", "grounded", "trusting", "secure", "assured",
    "free", "easygoing", "contented", "prosperous", "sanguine", "eased", "soulful", "gratified", "smiling"
}

dissatisfaction_words = {
    "frustrated", "disappointed", "unsatisfied", "unhappy", "annoyed", "angry", "dissatisfied", "bitter", "resentful",
    "miserable", "displeased", "upset", "irritated", "aggravated", "unfulfilled", "distressed", "vexed", "discontented",
    "unimpressed", "disheartened", "disgusted", "unappreciated", "ignored", "neglected", "overlooked", "pessimistic",
    "hopeless", "defeated", "melancholy", "lonely", "alienated", "abandoned", "isolated", "regretful", "ashamed",
    "embarrassed", "guilty", "inadequate", "inferior", "insulted", "offended", "humiliated", "demeaned", "belittled",
    "despised", "loathed", "disregarded", "discarded", "dejected", "rejected", "underwhelmed", "troubled", "anxious",
    "nervous", "overwhelmed", "weary", "fatigued", "exhausted", "burnt out", "drained", "hesitant", "uncertain",
    "apprehensive", "fearful", "hesitating", "weak", "unstable", "doubtful", "discouraged", "demotivated", "demoralized",
    "hopeless", "lost", "aimless", "purposeless", "stressed", "strained", "overburdened", "conflicted", "misunderstood",
    "disoriented", "detached", "resentful", "sorrowful", "disturbed", "pained", "aching", "tormented", "agonized",
    "shattered", "heartbroken", "devastated", "tormented", "oppressed", "powerless", "restricted", "constrained",
    "enslaved", "manipulated", "betrayed", "used", "cheated", "deceived", "lied to", "scammed", "fooled", "tricked"
}

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
    "What would you like to improve about your mindset today?"
    "How fulfilled do you feel about today's accomplishments?",
    "What task gave you a sense of completion today?",
    "Are your daily efforts aligning with long-term goals?",
    "How satisfied are you with your work-rest balance today?",
    "What was the biggest challenge you faced today?",
]

def analyze_text(input_text):
    """Analyze user response for mental health assessment."""
    sentiment_score = sia.polarity_scores(input_text)
    
    word_counts = Counter(input_text.lower().split())

    positive_count = sum(word_counts[word] for word in positive_words if word in word_counts)
    negative_count = sum(word_counts[word] for word in negative_words if word in word_counts)
    confidence_count = sum(word_counts[word] for word in confidence_words if word in word_counts)
    low_confidence_count = sum(word_counts[word] for word in low_confidence_words if word in word_counts)
    satisfaction_count = sum(word_counts[word] for word in satisfaction_words if word in word_counts)
    dissatisfaction_count = sum(word_counts[word] for word in dissatisfaction_words if word in word_counts)

    # Compute scores
    happiness_score = positive_count - negative_count + sentiment_score["compound"] * 10
    confidence_score = confidence_count - low_confidence_count
    satisfaction_score = satisfaction_count - dissatisfaction_count

    # Final mental health score
    total_score = (happiness_score * 0.4) + (confidence_score * 0.3) + (satisfaction_score * 0.3)
    mental_health_score = round(max(0, min(10, total_score)))

    condition = "Good" if mental_health_score >= 7 else "Neutral" if 4 <= mental_health_score < 7 else "Bad"

    return happiness_score, confidence_score, satisfaction_score, mental_health_score, condition

def main():
    st.set_page_config(
        page_title="Mind Matrix",
        layout="wide",
        page_icon="favicon.png",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS styling
    st.markdown("""
    <style>
        .main {background-color: #f8f9fa;}
        .stTextArea textarea {font-size: 16px !important;}
        .header {padding: 2rem 0; border-bottom: 2px solid #e0e0e0;}
        .score-card {border-radius: 10px; padding: 1.5rem; margin: 1rem 0; color: white;}
        .good {background-color: #2e7d32 !important; border-left: 5px solid #43a047;}
        .neutral {background-color: #f9a825 !important; border-left: 5px solid #ffb300;}
        .bad {background-color: #c62828 !important; border-left: 5px solid #e53935;}
        .recommendation {padding: 1rem; border-radius: 8px; margin: 1rem 0;}
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'questions' not in st.session_state:
        st.session_state.questions = random.sample(questions, 10)
    if 'answered' not in st.session_state:
        st.session_state.answered = False
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""

    # Header Section
    with st.container():
        st.markdown("<div class='header'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image("favicon.png", width=80)
        with col2:
            st.title("Mind Matrix (your personal Assistance)")
            st.markdown("**Your Personal Mental Wellness Assessment Tool**")
        st.markdown("</div>", unsafe_allow_html=True)

    # Questions Section
    with st.expander("📋 Today's Assessment Questions", expanded=True):
        for i, question in enumerate(st.session_state.questions):
            st.markdown(f"<div style='margin: 8px 0;'>🔹 {question}</div>", unsafe_allow_html=True)

   # User Input Section
    with st.form("input_form"):
        user_input = st.text_area(
            "✍️ Write your responses here (combine answers in one paragraph):",
            height=200,
            value=st.session_state.input_text,
            key="user_input"
        )
        
        # Modified Action Buttons
        col1, col2, col3 = st.columns([2,1,1])
        with col1:
            analyze_btn = st.form_submit_button("🔍 Analyze Responses")
        with col2:
            clear_btn = st.form_submit_button("🧹 Clear Input")
        with col3:
            new_btn = st.form_submit_button("🔄 New Analysis")

      # Clear Input: Reset answers without changing questions
        if clear_btn:
            st.session_state.answers = [""] * len(st.session_state.questions)  # Clear answers
            st.rerun()
            
        if new_btn:
            st.session_state.questions = random.sample(questions, 10)
            st.session_state.input_text = ""
            st.session_state.answered = False
            st.rerun()
            
       
    # Analysis and Results
    if analyze_btn and user_input.strip():
        st.session_state.answered = True
        st.session_state.input_text = user_input
        
        with st.spinner("🧠 Analyzing your responses. This takes just a moment..."):
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.02)
                progress_bar.progress(percent_complete + 1)
            
            happiness, confidence, satisfaction, score, condition = analyze_text(user_input)

        st.success("✅ Analysis Complete!")
        st.markdown("---")
        
        # Score Cards
        st.subheader("📊 Your Assessment Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="score-card good">
                <h3>😊 Happiness</h3>
                <h2>{happiness}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="score-card good">
                <h3>💪 Confidence</h3>
                <h2>{confidence}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="score-card good">
                <h3>👍 Satisfaction</h3>
                <h2>{satisfaction}</h2>
            </div>
            """, unsafe_allow_html=True)

        # Overall Score
        score_class = "good" if condition == "Good" else "neutral" if condition == "Neutral" else "bad"
        st.markdown(f"""
        <div class="score-card {score_class}">
            <h2>Overall Mental Health Score: {score}/10</h2>
            <h3>Current Status: {condition}</h3>
        </div>
        """, unsafe_allow_html=True)

        # Personalized Suggestions
        st.subheader("📌 Personalized Recommendations")
        if condition == "Good":
            st.markdown("""
            <div class="recommendation good">
                <h3>🌟 Maintain Your Positive Momentum</h3>
                <ul>
                    <li>Continue journaling and gratitude practice</li>
                    <li>Share your positivity with others</li>
                    <li>Maintain balanced sleep and nutrition</li>
                    <li>Explore new hobbies and challenges</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        elif condition == "Neutral":
            st.markdown("""
            <div class="recommendation neutral">
                <h3>🔄 Boost Your Well-being</h3>
                <ul>
                    <li>Practice daily mindfulness exercises</li>
                    <li>Connect with friends/family weekly</li>
                    <li>Set small achievable daily goals</li>
                    <li>Try light physical activity daily</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="recommendation bad">
                <h3>🤝 Supportive Strategies</h3>
                <ul>
                    <li>Practice 5-4-3-2-1 grounding technique daily</li>
                    <li>Try progressive muscle relaxation</li>
                    <li>Consider professional consultation</li>
                    <li>Establish a self-care routine</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Additional Resources
        st.markdown("---")
        st.subheader("📚 Helpful Resources")
        with st.expander("Mental Health Resources"):
            st.markdown("""
            - **Crisis Hotline**: 1-800-273-TALK (8255)
            - [Mental Health America Resources](https://www.mhanational.org)
            - [Mindfulness Exercises](https://www.mindful.org)
            - [Cognitive Behavioral Therapy Tools](https://www.psychologytools.com)
            """)

if __name__ == "__main__":
    main()


    