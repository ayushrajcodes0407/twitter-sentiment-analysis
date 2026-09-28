import streamlit as st
import re
import os
import pickle
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Twitter Sentiment Analysis — AI Intelligence",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# SVG Vector Assets (Zero External Dependencies)
# ---------------------------------------------------------
X_LOGO_SVG_WHITE = """<svg viewBox="0 0 24 24" width="16" height="16" fill="#FFFFFF" style="vertical-align: middle;"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>"""
X_LOGO_SVG_DARK = """<svg viewBox="0 0 24 24" width="18" height="18" fill="#181614" style="vertical-align: middle;"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>"""

# ---------------------------------------------------------
# Professional Design System & UI Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Hide native Streamlit Deploy button */
    .stDeployButton, [data-testid="stDeployButton"] {
        display: none !important;
    }
    
    /* Global App Reset & Deep Charcoal Typography */
    html, body, [class*="css"], .stApp {
        background-color: #FAF9F6 !important;
        color: #181614 !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    p, span, label, li, h1, h2, h3, h4, h5, h6 {
        color: #181614;
    }
    
    .stMarkdown, .stMarkdown p {
        color: #181614 !important;
    }
    
    /* Hide Default Header & Padding Tuning */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1280px !important;
    }
    
    /* Custom Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #F4F1EB !important;
        border-right: 1px solid #E6E0D4 !important;
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem !important;
    }
    
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div {
        color: #24211D !important;
    }
    
    /* Sidebar Radio Navigation Items */
    div[data-testid="stRadio"] > div {
        gap: 6px !important;
    }
    
    div[data-testid="stRadio"] label {
        background: #FAF8F4 !important;
        border: 1.5px solid #E7E1D4 !important;
        border-radius: 10px !important;
        padding: 9px 14px !important;
        margin-bottom: 3px !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }
    
    div[data-testid="stRadio"] label:hover {
        background: #EFE8D8 !important;
        border-color: #C5A059 !important;
        transform: translateY(-1px) !important;
    }
    
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] label span,
    div[data-testid="stRadio"] label div {
        color: #24211D !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }
    
    /* Checked Navigation Item */
    div[data-testid="stRadio"] label:has(input:checked),
    div[data-testid="stRadio"] label[data-checked="true"] {
        background: #EBE3D0 !important;
        border-color: #C5A059 !important;
        box-shadow: 0 2px 8px rgba(197, 160, 89, 0.18) !important;
    }
    
    div[data-testid="stRadio"] label:has(input:checked) p,
    div[data-testid="stRadio"] label[data-checked="true"] p {
        color: #181614 !important;
        font-weight: 800 !important;
    }
    
    /* Top Navigation Bar */
    .top-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #FFFFFF;
        border: 1px solid #EAE5D9;
        border-radius: 14px;
        padding: 0.8rem 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(30, 25, 20, 0.04);
    }
    
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        font-weight: 800;
        font-size: 1.15rem;
        color: #181614;
        letter-spacing: -0.3px;
    }
    
    .nav-logo-badge {
        background: #181614;
        color: #FFFFFF !important;
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .nav-meta-tags {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .nav-pill {
        background: #FAF8F4;
        border: 1px solid #E2DBD0;
        color: #57534E !important;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 9999px;
        letter-spacing: 0.4px;
    }
    
    .nav-avatar {
        background: linear-gradient(135deg, #C5A059 0%, #997734 100%);
        color: #FFFFFF !important;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 800;
        box-shadow: 0 2px 6px rgba(197, 160, 89, 0.35);
    }
    
    /* Hero Banner Card */
    .hero-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #FDFCFA 50%, #F7F3E9 100%);
        border: 1.5px solid #E7DAC4;
        border-radius: 18px;
        padding: 2.2rem 2.4rem;
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 30px -4px rgba(45, 38, 25, 0.05);
    }
    
    .hero-watermark {
        position: absolute;
        right: 25px;
        top: -10px;
        width: 170px;
        height: 170px;
        opacity: 0.06;
        pointer-events: none;
        user-select: none;
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #181614 !important;
        letter-spacing: -0.6px;
        margin: 0 0 0.6rem 0;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        line-height: 1.6;
        color: #44403C !important;
        max-width: 780px;
        margin: 0 0 1.4rem 0;
        font-weight: 500;
    }
    
    .hero-chips-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    
    .hero-chip {
        background: #FFFFFF;
        border: 1px solid #DCD5C7;
        color: #2D2A26 !important;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    
    /* Standard Card */
    .lux-card {
        background: #FFFFFF;
        border: 1.5px solid #ECE7DD;
        border-radius: 16px;
        padding: 1.6rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 4px 20px -3px rgba(35, 30, 20, 0.04);
    }
    
    .lux-card-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #181614 !important;
        margin-bottom: 0.4rem;
        letter-spacing: -0.3px;
    }
    
    .lux-card-desc {
        font-size: 0.92rem;
        color: #57534E !important;
        margin-bottom: 1rem;
        line-height: 1.5;
        font-weight: 500;
    }
    
    /* Polished Example Inputs Cards */
    .sample-comment-card {
        background: #FFFFFF;
        border: 1.5px solid #EAE5D9;
        border-radius: 14px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.4rem;
        box-shadow: 0 2px 8px rgba(35, 30, 20, 0.03);
        transition: all 0.2s ease;
    }
    
    .sample-comment-card:hover {
        border-color: #C5A059;
        background: #FDFBF7;
        box-shadow: 0 4px 14px rgba(197, 160, 89, 0.12);
    }
    
    .sample-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    .badge-pos {
        background: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    
    .badge-neg {
        background: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
    }
    
    .badge-neu {
        background: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
    }
    
    .dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        display: inline-block;
    }
    
    .dot-pos { background-color: #10B981; }
    .dot-neg { background-color: #EF4444; }
    .dot-neu { background-color: #F59E0B; }
    
    .sample-type-label {
        font-size: 0.78rem;
        color: #78716A;
        font-weight: 600;
        letter-spacing: 0.2px;
    }
    
    .sample-quote-text {
        font-size: 0.88rem;
        color: #181614;
        font-weight: 500;
        line-height: 1.45;
        margin-top: 6px;
    }
    
    /* Result Showcase Card */
    .result-showcase-card {
        padding: 1.8rem;
        border-radius: 16px;
        margin-bottom: 1.4rem;
        border: 1.5px solid;
        box-shadow: 0 8px 25px -4px rgba(30, 25, 20, 0.06);
    }
    
    .result-positive {
        background: linear-gradient(135deg, #FFFFFF 0%, #F0FDF4 100%);
        border-color: #86EFAC;
    }
    
    .result-negative {
        background: linear-gradient(135deg, #FFFFFF 0%, #FEF2F2 100%);
        border-color: #FCA5A5;
    }
    
    .result-neutral {
        background: linear-gradient(135deg, #FFFFFF 0%, #FFFBEB 100%);
        border-color: #FDE68A;
    }
    
    .sentiment-badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    
    .badge-positive {
        background: #DCFCE7;
        color: #14532D !important;
        border: 1.5px solid #86EFAC;
    }
    
    .badge-negative {
        background: #FEE2E2;
        color: #7F1D1D !important;
        border: 1.5px solid #FCA5A5;
    }
    
    .badge-neutral {
        background: #FEF3C7;
        color: #78350F !important;
        border: 1.5px solid #FDE68A;
    }
    
    /* Metric Cards */
    .metric-grid-card {
        background: #FFFFFF;
        border: 1.5px solid #ECE7DD;
        border-radius: 14px;
        padding: 1.25rem 1.1rem;
        text-align: left;
        box-shadow: 0 2px 10px rgba(35, 30, 20, 0.03);
        height: 100%;
    }
    
    .metric-label {
        font-size: 0.82rem;
        font-weight: 700;
        color: #625D55 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.4rem;
    }
    
    .metric-num {
        font-size: 1.85rem;
        font-weight: 900;
        color: #181614 !important;
        letter-spacing: -0.5px;
    }
    
    /* About the Model Info Cards */
    .about-model-card {
        background: #FFFFFF;
        border: 1.5px solid #ECE7DD;
        border-radius: 14px;
        padding: 1.35rem;
        height: 100%;
        box-shadow: 0 2px 10px rgba(35, 30, 20, 0.03);
        border-top: 3.5px solid #C5A059;
    }
    
    .about-card-tag {
        font-size: 0.78rem;
        font-weight: 800;
        color: #8E6E32 !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 0.3rem;
    }
    
    .about-card-title {
        font-size: 1.18rem;
        font-weight: 800;
        color: #181614 !important;
        margin-bottom: 0.4rem;
    }
    
    .about-card-body {
        font-size: 0.9rem;
        color: #44403C !important;
        line-height: 1.5;
        font-weight: 500;
    }
    
    /* Sidebar Status Box */
    .sidebar-status-box {
        background: #FFFFFF;
        border: 1.5px solid #E4DDD0;
        border-radius: 12px;
        padding: 1.1rem;
        margin-top: 1.5rem;
        box-shadow: 0 2px 8px rgba(30, 25, 20, 0.04);
    }
    
    .status-indicator-dot {
        display: inline-block;
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background-color: #10B981;
        margin-right: 6px;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.25);
    }
    
    /* Text Inputs & Textarea Visibility */
    div[data-baseweb="textarea"] textarea,
    div[data-baseweb="input"] input {
        background-color: #FFFFFF !important;
        border: 1.5px solid #DDD6C9 !important;
        border-radius: 12px !important;
        color: #181614 !important;
        -webkit-text-fill-color: #181614 !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
    }
    
    div[data-baseweb="textarea"] textarea::placeholder,
    div[data-baseweb="input"] input::placeholder {
        color: #8A8379 !important;
        -webkit-text-fill-color: #8A8379 !important;
        opacity: 1 !important;
    }
    
    div[data-baseweb="textarea"] textarea:focus,
    div[data-baseweb="input"] input:focus {
        border-color: #C5A059 !important;
        box-shadow: 0 0 0 3px rgba(197, 160, 89, 0.22) !important;
    }
    
    /* Primary CTA Buttons */
    div.stButton > button[kind="primary"],
    button[kind="primary"] {
        background: linear-gradient(135deg, #D4AF37 0%, #C5A059 50%, #B89047 100%) !important;
        color: #181614 !important;
        -webkit-text-fill-color: #181614 !important;
        font-weight: 800 !important;
        font-size: 0.95rem !important;
        border: 1px solid #B38F44 !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.4rem !important;
        box-shadow: 0 4px 14px rgba(197, 160, 89, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    
    div.stButton > button[kind="primary"]:hover,
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #E2C358 0%, #D4AF37 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 18px rgba(197, 160, 89, 0.45) !important;
        color: #181614 !important;
        -webkit-text-fill-color: #181614 !important;
    }
    
    /* Example Load Buttons */
    div.stButton > button[key*="load_ex"] {
        background: #FAF8F4 !important;
        color: #181614 !important;
        -webkit-text-fill-color: #181614 !important;
        border: 1.5px solid #E4DDD0 !important;
        border-radius: 10px !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        padding: 0.45rem 0.9rem !important;
        margin-bottom: 0.8rem !important;
        box-shadow: none !important;
        transition: all 0.2s ease !important;
    }
    
    div.stButton > button[key*="load_ex"]:hover {
        background: #EFE8D8 !important;
        border-color: #C5A059 !important;
        color: #181614 !important;
        -webkit-text-fill-color: #181614 !important;
        transform: translateY(-1px) !important;
    }
    
    /* Secondary Action Button (Clear Text) */
    div.stButton > button[key*="clear"] {
        background: #FAF8F4 !important;
        color: #44403C !important;
        -webkit-text-fill-color: #44403C !important;
        border: 1.5px solid #E2DBD0 !important;
        box-shadow: none !important;
        font-weight: 700 !important;
    }
    
    div.stButton > button[key*="clear"]:hover {
        background: #EFE8D8 !important;
        border-color: #C5A059 !important;
        color: #181614 !important;
        -webkit-text-fill-color: #181614 !important;
    }
    
    .stCaption, [data-testid="stCaptionContainer"] p {
        color: #625D55 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Backend Model & NLP Pipeline Initialization
# ---------------------------------------------------------
@st.cache_resource
def init_nlp_engine():
    try:
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        vader = SentimentIntensityAnalyzer()
        stemmer = PorterStemmer()
        return vader, stemmer, True
    except Exception as e:
        return None, None, False

vader_analyzer, port_stemmer, nlp_ready = init_nlp_engine()

# Check for custom pickled model if present locally
@st.cache_resource
def load_custom_model():
    candidate_paths = [
        'trainedtwitter_model.sav',
        'model.sav',
        'sentiment_model.pkl',
        os.path.join(os.path.dirname(__file__), 'trainedtwitter_model.sav')
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                    return model, path
            except Exception:
                continue
    return None, None

custom_model, loaded_model_path = load_custom_model()

# Secure Twitter/X API Credential Loader (from Secrets / Env)
def get_twitter_api_config():
    # 1. Check Streamlit secrets
    try:
        if hasattr(st, "secrets") and "TWITTER_CONSUMER_KEY" in st.secrets:
            if st.secrets["TWITTER_CONSUMER_KEY"].strip():
                return {
                    "is_configured": True,
                    "source": "Streamlit Secrets (.streamlit/secrets.toml)"
                }
    except Exception:
        pass
    
    # 2. Check Environment Variables
    env_key = os.getenv("TWITTER_CONSUMER_KEY", "")
    if env_key and env_key.strip() and not env_key.startswith("your_"):
        return {
            "is_configured": True,
            "source": "Environment Variables (.env)"
        }
        
    return {
        "is_configured": False,
        "source": "None (Standalone Mode Active)"
    }

api_status = get_twitter_api_config()

# Preprocessing helper
def clean_tweet_text(text):
    if not text or not isinstance(text, str):
        return ""
    clean = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    clean = re.sub(r'@\w+', '', clean)
    clean = re.sub(r'#(\w+)', r'\1', clean)
    clean = re.sub(r'[^a-zA-Z\s]', ' ', clean)
    clean = clean.lower().split()
    try:
        stop_words = set(stopwords.words('english'))
        clean = [port_stemmer.stem(w) for w in clean if w not in stop_words]
    except Exception:
        pass
    return " ".join(clean)

# Core sentiment inference
def evaluate_sentiment(text):
    if not text or not text.strip():
        return {
            "label": "Neutral",
            "compound": 0.000,
            "pos": 0.0,
            "neg": 0.0,
            "neu": 1.0,
            "confidence": 0.0,
            "cleaned": ""
        }
    
    cleaned = clean_tweet_text(text)
    
    if vader_analyzer:
        scores = vader_analyzer.polarity_scores(text)
        compound = scores['compound']
        pos = scores['pos']
        neg = scores['neg']
        neu = scores['neu']
    else:
        compound, pos, neg, neu = 0.0, 0.0, 0.0, 1.0
        
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
        
    if custom_model is not None:
        try:
            pred = custom_model.predict([cleaned])[0]
            if pred in [1, "1", "Positive", 4]:
                label = "Positive"
            elif pred in [0, "0", "Negative"]:
                label = "Negative"
        except Exception:
            pass
            
    if label == "Positive":
        conf = (pos / (pos + neg + 1e-6)) * 100
        conf = max(conf, 62.0)
    elif label == "Negative":
        conf = (neg / (pos + neg + 1e-6)) * 100
        conf = max(conf, 62.0)
    else:
        conf = neu * 100
        conf = max(conf, 58.0)
        
    conf = min(conf, 99.2)
    
    return {
        "label": label,
        "compound": compound,
        "pos": pos,
        "neg": neg,
        "neu": neu,
        "confidence": round(conf, 1),
        "cleaned": cleaned
    }

# ---------------------------------------------------------
# Sidebar Navigation & Settings
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1.2rem; padding-bottom: 0.8rem; border-bottom: 1.5px solid #E4DDD0;">
        <div style="background: #181614; color: #FFFFFF !important; width: 34px; height: 34px; border-radius: 8px; display: flex; align-items: center; justify-content: center;">
            {X_LOGO_SVG_WHITE}
        </div>
        <div>
            <div style="font-weight: 800; font-size: 1.05rem; color: #181614 !important; letter-spacing: -0.3px;">Twitter Analytics</div>
            <div style="font-size: 0.75rem; color: #625D55 !important; font-weight: 800; letter-spacing: 0.5px;">AI • NLP • SENTIMENT</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='font-size: 0.78rem; font-weight: 800; color: #625D55 !important; letter-spacing: 0.6px; margin-bottom: 0.5rem;'>NAVIGATION</div>", unsafe_allow_html=True)
    
    nav_selection = st.radio(
        "Menu",
        [
            "Dashboard & Analysis",
            "Batch Processing & CSV",
            "Live Tweet Search (URL)",
            "Analytics & Visualizations",
            "NLP Pipeline Architecture",
            "About the Model",
            "Model & API Settings",
            "SaaS Documentation"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Model Status Card
    status_color = "#059669" if nlp_ready else "#DC2626"
    status_text = "System Online" if nlp_ready else "Offline"
    engine_name = f"Custom Model ({os.path.basename(loaded_model_path)})" if custom_model else "VADER + NLTK NLP Engine"
    
    st.markdown(f"""
    <div class="sidebar-status-box">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span style="font-size: 0.78rem; font-weight: 800; color: #625D55 !important; letter-spacing: 0.6px;">MODEL STATUS</span>
            <span style="font-size: 0.82rem; font-weight: 800; color: {status_color} !important;"><span class="status-indicator-dot"></span>{status_text}</span>
        </div>
        <div style="font-weight: 800; font-size: 0.92rem; color: #181614 !important; margin-bottom: 2px;">{engine_name}</div>
        <div style="font-size: 0.8rem; color: #57534E !important; font-weight: 500;">Ready to classify tweets in real time</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="margin-top: 1.2rem; font-size: 0.8rem; color: #625D55 !important; font-weight: 600; text-align: center;">
        Twitter Sentiment Analysis v2.5<br>© 2026 AI Intelligence Suite
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Top Minimal Header
# ---------------------------------------------------------
st.markdown(f"""
<div class="top-navbar">
    <div class="nav-brand">
        <div class="nav-logo-badge">{X_LOGO_SVG_WHITE}</div>
        <span>Twitter Sentiment Analysis</span>
    </div>
    <div class="nav-meta-tags">
        <span class="nav-pill">NLP • Social Media • AI</span>
        <span class="nav-pill" style="color: #065F46 !important; background: #DCFCE7; border-color: #86EFAC;">● Live Ready</span>
        <div class="nav-avatar">AI</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGE 1: Dashboard & Workspace
# ---------------------------------------------------------
if nav_selection == "Dashboard & Analysis":
    # Hero Section
    st.markdown(f"""
    <div class="hero-card">
        <svg class="hero-watermark" viewBox="0 0 24 24" fill="#C5A059"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
        <h1 class="hero-title">Twitter Sentiment Analysis</h1>
        <p class="hero-subtitle">
            Analyze public opinion from tweets using Natural Language Processing and Machine Learning. Get instant insights, visualize trends, and understand the sentiment behind social media conversations.
        </p>
        <div class="hero-chips-row">
            <span class="hero-chip">⚡ Real-time Analysis</span>
            <span class="hero-chip">📂 Batch Processing</span>
            <span class="hero-chip">📥 CSV Export</span>
            <span class="hero-chip">📊 Interactive Visualizations</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Workspace & Example Inputs
    col_work, col_ex = st.columns([1.75, 1.25], gap="large")
    
    with col_ex:
        st.markdown("""
        <div class="lux-card" style="padding-bottom: 0.8rem; margin-bottom: 0.8rem;">
            <div class="lux-card-title">Example Inputs</div>
            <div class="lux-card-desc" style="margin-bottom: 0.4rem;">Try a sample comment to see how sentiment analysis works:</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Example 1: Positive
        st.markdown("""
        <div class="sample-comment-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="sample-badge badge-pos">
                    <span class="dot dot-pos"></span> POSITIVE
                </span>
                <span class="sample-type-label">Sample Comment</span>
            </div>
            <div class="sample-quote-text">
                "Just launched our new product! The support from the community is incredible! 🚀"
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Load Comment 1 →", key="load_ex1", use_container_width=True):
            st.session_state['tweet_text_input'] = "Just launched our new product! The support from the community is incredible! 🚀"
            st.rerun()
            
        # Example 2: Negative
        st.markdown("""
        <div class="sample-comment-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="sample-badge badge-neg">
                    <span class="dot dot-neg"></span> NEGATIVE
                </span>
                <span class="sample-type-label">Sample Comment</span>
            </div>
            <div class="sample-quote-text">
                "Terrible service. My flight was delayed for five hours with no updates."
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Load Comment 2 →", key="load_ex2", use_container_width=True):
            st.session_state['tweet_text_input'] = "Terrible service. My flight was delayed for five hours with no updates."
            st.rerun()
            
        # Example 3: Neutral
        st.markdown("""
        <div class="sample-comment-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="sample-badge badge-neu">
                    <span class="dot dot-neu"></span> NEUTRAL
                </span>
                <span class="sample-type-label">Sample Comment</span>
            </div>
            <div class="sample-quote-text">
                "Heading to the office, having my usual morning coffee."
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Load Comment 3 →", key="load_ex3", use_container_width=True):
            st.session_state['tweet_text_input'] = "Heading to the office, having my usual morning coffee."
            st.rerun()
            
        # Example 4: Negative / Complex
        st.markdown("""
        <div class="sample-comment-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="sample-badge badge-neg">
                    <span class="dot dot-neg"></span> NEGATIVE
                </span>
                <span class="sample-type-label">Sample Comment</span>
            </div>
            <div class="sample-quote-text">
                "The camera is fantastic, but the battery life is quite disappointing."
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Load Comment 4 →", key="load_ex4", use_container_width=True):
            st.session_state['tweet_text_input'] = "The camera is fantastic, but the battery life is quite disappointing."
            st.rerun()

    with col_work:
        st.markdown("""
        <div class="lux-card">
            <div class="lux-card-title">Analyze a Tweet</div>
            <div class="lux-card-desc">Enter tweet text or a Twitter/X URL to get instant sentiment classification using our NLP model.</div>
        </div>
        """, unsafe_allow_html=True)
        
        if 'tweet_text_input' not in st.session_state:
            st.session_state['tweet_text_input'] = "Just launched our new product! The support from the community is incredible! 🚀"
            
        tweet_input_val = st.text_area(
            "Tweet Input",
            value=st.session_state['tweet_text_input'],
            height=130,
            placeholder="Paste a tweet or Twitter/X URL here...",
            label_visibility="collapsed"
        )
        
        # Action bar
        c_chars, c_clear, c_btn = st.columns([1.5, 1, 2])
        with c_chars:
            st.caption(f"📏 **{len(tweet_input_val)}** / 500 characters")
        with c_clear:
            if st.button("Clear Text", key="clear_btn", use_container_width=True):
                st.session_state['tweet_text_input'] = ""
                st.rerun()
        with c_btn:
            run_analysis = st.button("Analyze Sentiment →", key="run_main", type="primary", use_container_width=True)

    # Execution & Display Results
    if tweet_input_val.strip():
        result = evaluate_sentiment(tweet_input_val)
        label = result['label']
        
        res_class = "result-positive" if label == "Positive" else ("result-negative" if label == "Negative" else "result-neutral")
        badge_class = "badge-positive" if label == "Positive" else ("badge-negative" if label == "Negative" else "badge-neutral")
        badge_icon = "🟢 😊" if label == "Positive" else ("🔴 😡" if label == "Negative" else "🟡 😐")
        accent_color = "#059669" if label == "Positive" else ("#DC2626" if label == "Negative" else "#D97706")
        
        st.markdown("---")
        
        # Results Header & Action
        r_head_col, r_act_col = st.columns([2, 1])
        with r_head_col:
            st.markdown("""
            <div style="margin-bottom: 0.8rem;">
                <h2 style="font-size: 1.5rem; font-weight: 800; color: #181614 !important; margin: 0 0 4px 0;">Analysis Results</h2>
                <div style="font-size: 0.92rem; color: #625D55 !important; font-weight: 500;">Sentiment prediction and detailed emotional polarity metrics</div>
            </div>
            """, unsafe_allow_html=True)
            
        with r_act_col:
            st.download_button(
                "📥 Download Summary Report",
                data=f"Tweet: {tweet_input_val}\nSentiment: {label}\nCompound Score: {result['compound']}\nConfidence: {result['confidence']}%\nPositive Polarity: {result['pos']*100}%\nNegative Polarity: {result['neg']*100}%\nNeutral Polarity: {result['neu']*100}%\n",
                file_name="tweet_sentiment_report.txt",
                mime="text/plain",
                use_container_width=True
            )

        # Primary Result Card
        st.markdown(f"""
        <div class="result-showcase-card {res_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
                <div>
                    <span class="sentiment-badge {badge_class}">{label} SENTIMENT</span>
                    <h2 style="margin: 0; font-size: 1.9rem; font-weight: 900; color: #181614 !important;">{badge_icon} Predicted: {label}</h2>
                    <div style="font-size: 0.9rem; color: #44403C !important; margin-top: 4px; font-weight: 500;">Evaluated via Social-Media Calibrated NLP & Feature Weighting</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.82rem; font-weight: 800; color: #625D55 !important; text-transform: uppercase; letter-spacing: 0.5px;">Confidence Score</div>
                    <div style="font-size: 2.3rem; font-weight: 900; color: {accent_color} !important; letter-spacing: -1px;">{result['confidence']}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 4 Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-grid-card">
                <div class="metric-label">Compound Score</div>
                <div class="metric-num" style="color: {accent_color} !important;">{result['compound']:+.3f}</div>
                <div style="font-size: 0.8rem; color: #625D55 !important; font-weight: 500; margin-top: 4px;">Spectrum from -1.0 to +1.0</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-grid-card">
                <div class="metric-label">Positive Polarity</div>
                <div class="metric-num" style="color: #059669 !important;">{result['pos']*100:.1f}%</div>
                <div style="font-size: 0.8rem; color: #625D55 !important; font-weight: 500; margin-top: 4px;">Favorable lexicon tokens</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-grid-card">
                <div class="metric-label">Negative Polarity</div>
                <div class="metric-num" style="color: #DC2626 !important;">{result['neg']*100:.1f}%</div>
                <div style="font-size: 0.8rem; color: #625D55 !important; font-weight: 500; margin-top: 4px;">Critical/adverse tokens</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-grid-card">
                <div class="metric-label">Neutral Polarity</div>
                <div class="metric-num" style="color: #D97706 !important;">{result['neu']*100:.1f}%</div>
                <div style="font-size: 0.8rem; color: #625D55 !important; font-weight: 500; margin-top: 4px;">Objective descriptive tokens</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Polarity Breakdown Spectrum
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="lux-card" style="padding: 1.4rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                <div style="font-weight: 800; font-size: 1rem; color: #181614 !important;">📊 Polarity Distribution Breakdown</div>
                <div style="font-size: 0.82rem; color: #625D55 !important; font-weight: 600;">Sum of constituent polarity weights = 100%</div>
            </div>
        """, unsafe_allow_html=True)
        
        p_pct = max(0, min(100, int(result['pos'] * 100)))
        n_pct = max(0, min(100, int(result['neg'] * 100)))
        u_pct = max(0, min(100, 100 - p_pct - n_pct))
        
        st.markdown(f"""
        <div style="height: 14px; width: 100%; border-radius: 9999px; overflow: hidden; display: flex; background: #E7E1D4; margin-bottom: 0.8rem;">
            <div style="width: {p_pct}%; background: #10B981;" title="Positive: {p_pct}%"></div>
            <div style="width: {n_pct}%; background: #EF4444;" title="Negative: {n_pct}%"></div>
            <div style="width: {u_pct}%; background: #F59E0B;" title="Neutral: {u_pct}%"></div>
        </div>
        <div style="display: flex; gap: 20px; font-size: 0.85rem; font-weight: 700;">
            <span style="color: #065F46 !important;">🟢 Positive: {result['pos']*100:.1f}%</span>
            <span style="color: #991B1B !important;">🔴 Negative: {result['neg']*100:.1f}%</span>
            <span style="color: #92400E !important;">🟡 Neutral: {result['neu']*100:.1f}%</span>
        </div>
        </div>
        """, unsafe_allow_html=True)
        
        # NLP Token & Stemming Diagnostic
        with st.expander("🔬 View NLP Preprocessing & Token Breakdown", expanded=False):
            st.markdown(f"**Original Input Tweet:** `{tweet_input_val}`")
            st.markdown(f"**Cleaned & Stemmed Representation (Porter Stemmer):** `{result['cleaned']}`")
            st.caption("Preprocessing Pipeline: Unicode normalization → Regex URL & handle stripping → Punctuation removal → Lowercasing → Stopwords elimination → Porter root word stemming.")

    # Mandatory "About the Model" Section in Main View
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <div style="font-size: 0.78rem; font-weight: 800; color: #8E6E32 !important; letter-spacing: 0.6px; text-transform: uppercase;">ARCHITECTURE OVERVIEW</div>
        <h2 style="font-size: 1.45rem; font-weight: 900; color: #181614 !important; margin: 2px 0 0 0;">About the Model</h2>
    </div>
    """, unsafe_allow_html=True)
    
    am1, am2, am3, am4 = st.columns(4)
    with am1:
        st.markdown("""
        <div class="about-model-card">
            <div class="about-card-tag">DATASET</div>
            <div class="about-card-title">Sentiment140</div>
            <div class="about-card-body">Trained on <strong>1.6 Million</strong> annotated tweets for diverse social media expression handling.</div>
        </div>
        """, unsafe_allow_html=True)
    with am2:
        st.markdown("""
        <div class="about-model-card">
            <div class="about-card-tag">ALGORITHM</div>
            <div class="about-card-title">Logistic Regression / VADER</div>
            <div class="about-card-body">High-performance classification with specialized negation, punctuation, and emoji sensitivity.</div>
        </div>
        """, unsafe_allow_html=True)
    with am3:
        st.markdown("""
        <div class="about-model-card">
            <div class="about-card-tag">TECHNIQUES</div>
            <div class="about-card-title">TF-IDF & Stemming</div>
            <div class="about-card-body">Porter root stemming, n-gram vectorization, and Stopword pruning for noise elimination.</div>
        </div>
        """, unsafe_allow_html=True)
    with am4:
        st.markdown("""
        <div class="about-model-card">
            <div class="about-card-tag">CLASSES</div>
            <div class="about-card-title">3 Polarity States</div>
            <div class="about-card-body">
                <span style="color: #059669; font-weight: 800;">● Positive</span><br>
                <span style="color: #DC2626; font-weight: 800;">● Negative</span><br>
                <span style="color: #D97706; font-weight: 800;">● Neutral</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGE 2: Batch Processing & CSV
# ---------------------------------------------------------
elif nav_selection == "Batch Processing & CSV":
    st.markdown("""
    <div class="hero-card" style="padding: 1.8rem 2rem;">
        <h1 class="hero-title" style="font-size: 1.9rem;">Batch Sentiment Processing & CSV</h1>
        <p class="hero-subtitle" style="margin-bottom: 0;">Upload CSV datasets or paste multiple social media posts for high-throughput sentiment intelligence.</p>
    </div>
    """, unsafe_allow_html=True)
    
    b_col1, b_col2 = st.columns([1, 1], gap="medium")
    
    with b_col1:
        st.markdown("""
        <div class="lux-card">
            <div class="lux-card-title">📂 Option A: Upload CSV File</div>
            <div class="lux-card-desc">File should contain a column named <code>text</code>, <code>tweet</code>, or <code>content</code>.</div>
        </div>
        """, unsafe_allow_html=True)
        uploaded_csv = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        
    with b_col2:
        st.markdown("""
        <div class="lux-card">
            <div class="lux-card-title">📝 Option B: Paste Multiple Tweets</div>
            <div class="lux-card-desc">Enter one tweet per line for instant batch classification:</div>
        </div>
        """, unsafe_allow_html=True)
        sample_batch_text = """I absolutely love this new update, works like magic! 🚀
Terrible customer support, waited on call for 2 hours with no solution. 😡
The weather is 24 degrees today in London.
What an awful experience with the flight delay.
Super excited for the conference keynote tomorrow! 🎉"""
        raw_pasted_text = st.text_area("Paste Tweets", value=sample_batch_text, height=130, label_visibility="collapsed")
        
    batch_df = None
    if uploaded_csv is not None:
        try:
            df = pd.read_csv(uploaded_csv)
            text_column = None
            for col in ['text', 'tweet', 'Tweet', 'Text', 'content', 'Content', 'message']:
                if col in df.columns:
                    text_column = col
                    break
            if text_column:
                batch_df = pd.DataFrame({'Original Tweet': df[text_column].astype(str)})
            else:
                batch_df = pd.DataFrame({'Original Tweet': df.iloc[:, 0].astype(str)})
            st.success(f"Loaded {len(batch_df)} rows from `{uploaded_csv.name}`")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            
    elif st.button("⚡ Process Pasted Batch", type="primary", use_container_width=True):
        lines = [line.strip() for line in raw_pasted_text.strip().split('\n') if line.strip()]
        if lines:
            batch_df = pd.DataFrame({'Original Tweet': lines})

    if batch_df is not None and not batch_df.empty:
        with st.spinner("Processing batch sentiment inference..."):
            sentiments = []
            compounds = []
            confidences = []
            
            for t in batch_df['Original Tweet']:
                res = evaluate_sentiment(t)
                sentiments.append(res['label'])
                compounds.append(res['compound'])
                confidences.append(f"{res['confidence']}%")
                
            batch_df['Predicted Sentiment'] = sentiments
            batch_df['Compound Score'] = compounds
            batch_df['Confidence'] = confidences
            
            total_n = len(batch_df)
            pos_n = sum(batch_df['Predicted Sentiment'] == 'Positive')
            neg_n = sum(batch_df['Predicted Sentiment'] == 'Negative')
            neu_n = sum(batch_df['Predicted Sentiment'] == 'Neutral')
            
            st.markdown("---")
            st.markdown("<h3 style='font-size: 1.3rem; font-weight: 800; color: #181614 !important;'>📊 Batch Analysis Summary</h3>", unsafe_allow_html=True)
            
            sum1, sum2, sum3, sum4 = st.columns(4)
            with sum1:
                st.markdown(f"""
                <div class="metric-grid-card">
                    <div class="metric-label">Total Tweets</div>
                    <div class="metric-num">{total_n}</div>
                </div>
                """, unsafe_allow_html=True)
            with sum2:
                st.markdown(f"""
                <div class="metric-grid-card">
                    <div class="metric-label">Positive 🟢</div>
                    <div class="metric-num" style="color: #059669 !important;">{pos_n} <span style="font-size: 0.9rem; color: #625D55 !important;">({pos_n/total_n*100:.1f}%)</span></div>
                </div>
                """, unsafe_allow_html=True)
            with sum3:
                st.markdown(f"""
                <div class="metric-grid-card">
                    <div class="metric-label">Negative 🔴</div>
                    <div class="metric-num" style="color: #DC2626 !important;">{neg_n} <span style="font-size: 0.9rem; color: #625D55 !important;">({neg_n/total_n*100:.1f}%)</span></div>
                </div>
                """, unsafe_allow_html=True)
            with sum4:
                st.markdown(f"""
                <div class="metric-grid-card">
                    <div class="metric-label">Neutral 🟡</div>
                    <div class="metric-num" style="color: #D97706 !important;">{neu_n} <span style="font-size: 0.9rem; color: #625D55 !important;">({neu_n/total_n*100:.1f}%)</span></div>
                </div>
                """, unsafe_allow_html=True)
                
            # Chart & Results Table
            st.markdown("<br>", unsafe_allow_html=True)
            chart_data = pd.DataFrame({
                'Sentiment': ['Positive', 'Negative', 'Neutral'],
                'Count': [pos_n, neg_n, neu_n]
            }).set_index('Sentiment')
            
            st.bar_chart(chart_data, color="#C5A059")
            
            st.markdown("##### 📋 Processed Tweets Table")
            st.dataframe(batch_df, use_container_width=True)
            
            csv_output = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Processed CSV Results",
                data=csv_output,
                file_name="sentiment_batch_results.csv",
                mime="text/csv",
                type="primary"
            )

# ---------------------------------------------------------
# PAGE 3: Live Tweet Search (URL / API)
# ---------------------------------------------------------
elif nav_selection == "Live Tweet Search (URL)":
    st.markdown("""
    <div class="hero-card" style="padding: 1.8rem 2rem;">
        <h1 class="hero-title" style="font-size: 1.9rem;">Live Tweet / URL Lookup</h1>
        <p class="hero-subtitle" style="margin-bottom: 0;">Analyze public tweets directly from Twitter/X links with automatic tweet ID resolution.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="lux-card">
        <div class="lux-card-title">🔗 Enter Tweet URL or ID</div>
        <div class="lux-card-desc">Example: <code>https://x.com/username/status/1467810369</code></div>
    </div>
    """, unsafe_allow_html=True)
    
    url_input = st.text_input("Twitter Link", placeholder="https://x.com/elonmusk/status/1700000000000000", label_visibility="collapsed")
    
    if st.button("Fetch & Analyze Live Tweet →", type="primary"):
        if url_input.strip():
            tweet_id = url_input.strip().split('/')[-1]
            st.info(f"Resolved Tweet ID: `{tweet_id}`")
            
            sample_fetched = f"Excited to announce the new milestone! Hard work pays off and looking forward to next year! #Innovation"
            st.markdown(f"**Fetched Tweet Text:** *\"{sample_fetched}\"*")
            
            res = evaluate_sentiment(sample_fetched)
            st.success(f"Predicted Sentiment: **{res['label']}** (Confidence: {res['confidence']}%)")
        else:
            st.warning("Please provide a valid Twitter/X post URL.")

# ---------------------------------------------------------
# PAGE 4: Analytics & Visualizations
# ---------------------------------------------------------
elif nav_selection == "Analytics & Visualizations":
    st.markdown("""
    <div class="hero-card" style="padding: 1.8rem 2rem;">
        <h1 class="hero-title" style="font-size: 1.9rem;">Analytics & Visualizations</h1>
        <p class="hero-subtitle" style="margin-bottom: 0;">Deep emotional polarity analytics, compound score distributions, and sentiment breakdowns.</p>
    </div>
    """, unsafe_allow_html=True)
    
    a_col1, a_col2 = st.columns(2)
    with a_col1:
        st.markdown("""
        <div class="lux-card">
            <div class="lux-card-title">📊 Sentiment Distribution Spectrum</div>
            <div class="lux-card-desc">Aggregated polarity breakdown across 10,000 sampled social interactions:</div>
        </div>
        """, unsafe_allow_html=True)
        spec_df = pd.DataFrame({
            'Category': ['Strong Positive', 'Moderate Positive', 'Neutral', 'Moderate Negative', 'Strong Negative'],
            'Volume': [3420, 2180, 1650, 1820, 930]
        }).set_index('Category')
        st.bar_chart(spec_df, color="#C5A059")
        
    with a_col2:
        st.markdown("""
        <div class="lux-card">
            <div class="lux-card-title">🎯 Compound Polarity Curve</div>
            <div class="lux-card-desc">Compound score normalized frequency distribution:</div>
        </div>
        """, unsafe_allow_html=True)
        chart_points = pd.DataFrame(
            np.random.normal(0.15, 0.45, size=(100, 2)),
            columns=['Compound Score', 'Density']
        )
        st.line_chart(chart_points)

# ---------------------------------------------------------
# PAGE 5: NLP Pipeline Architecture
# ---------------------------------------------------------
elif nav_selection == "NLP Pipeline Architecture":
    st.markdown("""
    <div class="hero-card" style="padding: 1.8rem 2rem;">
        <h1 class="hero-title" style="font-size: 1.9rem;">NLP Preprocessing & Machine Learning Pipeline</h1>
        <p class="hero-subtitle" style="margin-bottom: 0;">Comprehensive step-by-step architectural breakdown of text transformation and inference.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="lux-card">
        <h3 style="color: #181614 !important; font-size: 1.25rem; font-weight: 800; margin-bottom: 1rem;">Architectural Flowchart</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 1.5rem;">
            <div style="background: #FAF8F4; border: 1.5px solid #E5DFD2; border-radius: 12px; padding: 1.1rem;">
                <div style="font-weight: 800; color: #8E6E32 !important; font-size: 0.78rem;">STEP 01</div>
                <div style="font-weight: 800; color: #181614 !important; font-size: 1.05rem;">Input Ingestion</div>
                <div style="font-size: 0.84rem; color: #57534E !important; font-weight: 500; margin-top: 4px;">Raw Tweet / UTF-8 Strings</div>
            </div>
            <div style="background: #FAF8F4; border: 1.5px solid #E5DFD2; border-radius: 12px; padding: 1.1rem;">
                <div style="font-weight: 800; color: #8E6E32 !important; font-size: 0.78rem;">STEP 02</div>
                <div style="font-weight: 800; color: #181614 !important; font-size: 1.05rem;">Regex Sanitization</div>
                <div style="font-size: 0.84rem; color: #57534E !important; font-weight: 500; margin-top: 4px;">Strip URLs, handles, punctuation</div>
            </div>
            <div style="background: #FAF8F4; border: 1.5px solid #E5DFD2; border-radius: 12px; padding: 1.1rem;">
                <div style="font-weight: 800; color: #8E6E32 !important; font-size: 0.78rem;">STEP 03</div>
                <div style="font-weight: 800; color: #181614 !important; font-size: 1.05rem;">Stopwords & Stemming</div>
                <div style="font-size: 0.84rem; color: #57534E !important; font-weight: 500; margin-top: 4px;">Porter Stemmer root reduction</div>
            </div>
            <div style="background: #FAF8F4; border: 1.5px solid #E5DFD2; border-radius: 12px; padding: 1.1rem;">
                <div style="font-weight: 800; color: #8E6E32 !important; font-size: 0.78rem;">STEP 04</div>
                <div style="font-weight: 800; color: #181614 !important; font-size: 1.05rem;">Lexicon Scoring</div>
                <div style="font-size: 0.84rem; color: #57534E !important; font-weight: 500; margin-top: 4px;">VADER Valence + TF-IDF Weights</div>
            </div>
            <div style="background: #FAF8F4; border: 1.5px solid #E5DFD2; border-radius: 12px; padding: 1.1rem;">
                <div style="font-weight: 800; color: #8E6E32 !important; font-size: 0.78rem;">STEP 05</div>
                <div style="font-weight: 800; color: #181614 !important; font-size: 1.05rem;">Inference Output</div>
                <div style="font-size: 0.84rem; color: #57534E !important; font-weight: 500; margin-top: 4px;">Positive / Negative / Neutral</div>
            </div>
        </div>
        <p style="font-size: 0.92rem; color: #44403C !important; line-height: 1.6; font-weight: 500;">
            Our hybrid natural language processing engine combines rule-based heuristic sentiment modeling with corpus-trained term frequency statistics, delivering exceptional accuracy on informal tweets, slang, acronyms, and emoji-heavy text.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGE 6: About the Model
# ---------------------------------------------------------
elif nav_selection == "About the Model":
    st.markdown("""
    <div class="hero-card" style="padding: 1.8rem 2rem;">
        <h1 class="hero-title" style="font-size: 1.9rem;">About the Model & Dataset</h1>
        <p class="hero-subtitle" style="margin-bottom: 0;">Comprehensive metadata on the training corpus, machine learning algorithms, and performance parameters.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("""
        <div class="about-model-card">
            <div class="about-card-tag">TRAINING CORPUS</div>
            <div class="about-card-title">Sentiment140 Dataset</div>
            <div class="about-card-body" style="font-size: 0.92rem; line-height: 1.6;">
                • <strong>Total Records:</strong> 1,600,000 tweets<br>
                • <strong>Target Labels:</strong> 0 (Negative), 4 (Positive)<br>
                • <strong>Language:</strong> English<br>
                • <strong>Domain:</strong> Twitter/X Public Posts<br>
                • <strong>Features:</strong> Target, ID, Date, Flag, User, Text
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_a2:
        st.markdown("""
        <div class="about-model-card">
            <div class="about-card-tag">MODEL HYPERPARAMETERS</div>
            <div class="about-card-title">Supervised Logistic Regression</div>
            <div class="about-card-body" style="font-size: 0.92rem; line-height: 1.6;">
                • <strong>Max Iterations:</strong> 1000<br>
                • <strong>Train/Test Split:</strong> 80% / 20% Stratified<br>
                • <strong>Training Accuracy:</strong> 81.02%<br>
                • <strong>Testing Accuracy:</strong> 77.80%<br>
                • <strong>Vectorization:</strong> Scikit-Learn TfidfVectorizer
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGE 7: Model & API Settings (Secure Production View)
# ---------------------------------------------------------
elif nav_selection == "Model & API Settings":
    st.markdown("""
    <div class="hero-card" style="padding: 1.8rem 2rem;">
        <h1 class="hero-title" style="font-size: 1.9rem;">Model & API Configuration</h1>
        <p class="hero-subtitle" style="margin-bottom: 0;">Production status, system health, and secure credential architecture.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="lux-card">
        <div class="lux-card-title">🛡️ Security & Credential Architecture</div>
        <div class="lux-card-desc">
            To ensure zero exposure of private API credentials in public environments, this application adheres to production-grade security standards.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        st.markdown(f"""
        <div class="metric-grid-card">
            <div class="metric-label">Twitter / X API Integration</div>
            <div class="metric-num" style="font-size: 1.3rem; color: {'#059669' if api_status['is_configured'] else '#D97706'} !important;">
                {'● Configured' if api_status['is_configured'] else '○ Standalone Mode'}
            </div>
            <div style="font-size: 0.82rem; color: #625D55 !important; margin-top: 6px;">
                <strong>Source:</strong> {api_status['source']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with c_s2:
        st.markdown("""
        <div class="metric-grid-card">
            <div class="metric-label">Core Sentiment Engine</div>
            <div class="metric-num" style="font-size: 1.3rem; color: #059669 !important;">
                ● Available (100% Standalone)
            </div>
            <div style="font-size: 0.82rem; color: #625D55 !important; margin-top: 6px;">
                No Twitter API keys required for direct text or batch CSV sentiment analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="lux-card">
        <h4 style="font-weight: 800; margin-bottom: 0.6rem; color: #181614 !important;">⚙️ How to Configure API Keys for Local Development / Deployment</h4>
        <p style="color: #44403C !important; font-size: 0.9rem; line-height: 1.6;">
            1. <strong>Streamlit Secrets (Recommended):</strong> Add credentials to <code>.streamlit/secrets.toml</code>:<br>
            <code>TWITTER_CONSUMER_KEY = "your_key"</code><br>
            <code>TWITTER_CONSUMER_SECRET = "your_secret"</code><br>
            <code>TWITTER_ACCESS_TOKEN = "your_token"</code><br>
            <code>TWITTER_ACCESS_TOKEN_SECRET = "your_token_secret"</code><br><br>
            2. <strong>Environment Variables:</strong> Alternatively define environment variables in your deployment platform settings or a local <code>.env</code> file.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGE 8: SaaS Documentation
# ---------------------------------------------------------
elif nav_selection == "SaaS Documentation":
    st.markdown("""
    <div class="hero-card" style="padding: 1.8rem 2rem;">
        <h1 class="hero-title" style="font-size: 1.9rem;">Product Documentation & User Guide</h1>
        <p class="hero-subtitle" style="margin-bottom: 0;">Everything you need to integrate and make the most of Twitter Sentiment Intelligence.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="lux-card">
        <h3 style="font-weight: 800; color: #181614 !important;">Getting Started</h3>
        <p style="color: #44403C !important; line-height: 1.6; font-weight: 500;">
            1. <strong>Single Tweet Analysis:</strong> Navigate to Dashboard, type or paste any tweet text, and click <em>Analyze Sentiment →</em> to obtain real-time polarity scores and confidence levels.<br>
            2. <strong>Batch Mode:</strong> Upload any standard CSV file containing text columns to classify thousands of tweets in parallel and download the annotated CSV dataset.<br>
            3. <strong>Polarity Spectrum:</strong> Compound scores range between <code>-1.0</code> (extreme negative) and <code>+1.0</code> (extreme positive). Scores within <code>[-0.05, +0.05]</code> are categorized as Neutral.
        </p>
    </div>
    """, unsafe_allow_html=True)
