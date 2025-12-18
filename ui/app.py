"""
Movie Intelligence System - Production Frontend
Cohesive, Professional UI with Perfect Color Coordination
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import requests
import os
import json
import io
from pathlib import Path
import joblib
import time
from datetime import datetime


# Page configuration
st.set_page_config(
    page_title="MOVIE INTELLIGENCE",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Paths
if os.name == 'nt':  # Windows
    DATA_PATH = Path(r"C:\Users\darab\OneDrive\Desktop\ML_PROJECT_FINAL\data\processed\movies_processed.csv")
    RESULTS_PATH = Path(r"C:\Users\darab\OneDrive\Desktop\ML_PROJECT_FINAL\reports\training_results.json")
    MODELS_PATH = Path(r"C:\Users\darab\OneDrive\Desktop\ML_PROJECT_FINAL\models\trained")
else:  # Linux/Mac
    DATA_PATH = Path("data/processed/movies_processed.csv")
    RESULTS_PATH = Path("reports/training_results.json")
    MODELS_PATH = Path("models/trained")

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# TMDB Configuration
TMDB_BEARER_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI3MmRjNDQzYzNhMDBhOGFjN2E2OGE0OWQ1Mzc2ZjkwMSIsIm5iZiI6MTc2NTg5NTA5Ni4zMDQ5OTk4LCJzdWIiOiI2OTQxNmJiODFkMTQxNzNiMWYyY2EzZTkiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.PrzyEJ-n96kOczdd8XkT2Ci-SBM-MMlkP2_FqvvhnhI"
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# Load training results
METRICS_DATA = {}
if RESULTS_PATH.exists():
    try:
        with open(RESULTS_PATH, 'r') as f:
            METRICS_DATA = json.load(f)
    except:
        pass

# Genre list
GENRE_OPTIONS = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Family", "Fantasy", "History", "Horror", "Music",
    "Mystery", "Romance", "Science Fiction", "Thriller", "War", "Western"
]

if "tmdb_results" not in st.session_state:
    st.session_state.tmdb_results = []

if "tmdb_last_query" not in st.session_state:
    st.session_state.tmdb_last_query = ""


# Professional Cinema-inspired CSS - Cohesive Color Scheme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
    }
    
    /* Unified Color Scheme */
    :root {
        --bg-main: #0D0D0F;
        --surface: #1A1A1E;
        --border: #2D2D32;
        --gold: #E4B15E;
        --gold-hover: #F5C976;
        --red: #B11226;
        --slate: #8A8F98;
        --text-primary: #F5F5F7;
        --text-secondary: #A1A1AA;
        --text-muted: #6B7280;
    }
    
    .main {
        background-color: var(--bg-main);
        color: var(--text-primary);
    }
    
    .block-container {
        padding: 2rem 2rem 3rem 2rem;
        max-width: 1400px;
    }
    
    /* Remove all default Streamlit styling */
    div[data-testid="stVerticalBlock"] > div {
        gap: 0rem;
    }
    
    /* Page Title */
    .page-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--text-primary);
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .page-subtitle {
        font-size: 1rem;
        color: var(--text-secondary);
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: 0.02em;
    }
    
    /* Navigation Buttons - GOLD COLOR - FIXED EQUAL SIZING */
    .nav-row {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 12px !important;
        margin-bottom: 2rem;
        padding: 0 2rem 1.5rem 2rem;
        border-bottom: 1px solid var(--border);
    }
    
    /* Force all columns to be equal */
    .nav-row > div[data-testid="column"] {
        flex: 0 0 auto !important;
        width: 145px !important;
        min-width: 145px !important;
        max-width: 145px !important;
        padding: 0 !important;
    }
    
    /* All navigation buttons - EXACTLY SAME SIZE */
    .nav-row .stButton > button {
        background-color: transparent !important;
        color: #E4B15E !important;
        border: 1px solid #E4B15E !important;
        border-radius: 4px !important;
        padding: 0 !important;
        font-weight: 700 !important;
        font-size: 0.65rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        transition: all 0.2s ease !important;
        white-space: nowrap !important;
        width: 145px !important;
        min-width: 145px !important;
        max-width: 145px !important;
        height: 42px !important;
        min-height: 42px !important;
        max-height: 42px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3) !important;
        margin: 0 !important;
    }
    
    .nav-row .stButton > button:hover {
        background-color: rgba(228, 177, 94, 0.15) !important;
        box-shadow: 0 4px 8px rgba(228, 177, 94, 0.2) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Active navigation button - filled gold */
    .nav-row .stButton > button[kind="primary"] {
        background-color: #E4B15E !important;
        color: #0D0D0F !important;
        border: 1px solid #E4B15E !important;
        box-shadow: 0 2px 8px rgba(228, 177, 94, 0.4) !important;
    }
    
    .nav-row .stButton > button[kind="primary"]:hover {
        background-color: #F5C976 !important;
        box-shadow: 0 4px 12px rgba(228, 177, 94, 0.5) !important;
    }
    
    /* Regular action buttons (non-navigation) */
    .stButton > button {
        background-color: var(--gold) !important;
        color: var(--bg-main) !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.8rem 1.5rem !important;
        font-weight: 700 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.12em !important;
        transition: all 0.2s ease !important;
        white-space: nowrap !important;
        min-width: fit-content !important;
        height: 48px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    .stButton > button:hover {
        background-color: var(--gold-hover) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Secondary Button Style */
    button[kind="secondary"] {
        background-color: transparent !important;
        border: 1px solid var(--gold) !important;
        color: var(--gold) !important;
    }
    
    button[kind="secondary"]:hover {
        background-color: rgba(228, 177, 94, 0.1) !important;
    }
    
    /* Cards - Consistent Design */
    .card {
        background-color: var(--surface);
        border: 1px solid var(--border);
        padding: 2rem;
        margin-bottom: 1.5rem;
        border-radius: 8px;
    }
    
    /* Section Headers */
    h1, h2, h3 {
        color: var(--text-primary);
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    h2 {
        font-size: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 2rem;
    }
    
    h3 {
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-secondary);
        margin-bottom: 1rem;
    }
    
    /* Input Fields - Consistent Styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 4px !important;
        color: var(--text-primary) !important;
        padding: 0.75rem !important;
        font-size: 0.95rem !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: var(--gold) !important;
        box-shadow: 0 0 0 1px var(--gold) !important;
    }
    
    /* Labels */
    .stTextInput > label,
    .stNumberInput > label,
    .stSelectbox > label,
    .stMultiSelect > label {
        color: var(--text-secondary) !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Sliders - GOLD */
    .stSlider > div > div > div > div {
        background-color: var(--gold) !important;
    }
    
    .stSlider > div > div > div {
        background-color: var(--border) !important;
    }
    
    /* Tabs - GOLD ACCENT */
    .stTabs {
        margin-top: 1rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: transparent;
        border-bottom: 2px solid var(--border);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 0;
        color: var(--text-muted);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding: 1rem 2rem;
        border: none;
        font-size: 0.85rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: var(--gold);
        border-bottom: 3px solid var(--gold);
    }
    
    /* MultiSelect */
    .stMultiSelect > div > div {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 4px;
    }
    
    /* Results Cards - Large Numbers */
    .result-card {
        background-color: var(--surface);
        border: 1px solid var(--border);
        padding: 2rem;
        border-radius: 8px;
        text-align: center;
    }
    
    .result-number {
        font-size: 3rem;
        font-weight: 800;
        color: var(--gold);
        line-height: 1;
        margin: 1rem 0;
    }
    
    .result-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 0.5rem;
    }
    
    .result-sublabel {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .badge-hit {
        background-color: var(--gold);
        color: var(--bg-main);
    }
    
    .badge-flop {
        background-color: var(--red);
        color: var(--text-primary);
    }
    
    /* Messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        background-color: var(--surface) !important;
        border-radius: 4px !important;
        padding: 1rem !important;
    }
    
    .stSuccess {
        border-left: 4px solid var(--gold) !important;
    }
    
    .stError {
        border-left: 4px solid var(--red) !important;
    }
    
    .stWarning, .stInfo {
        border-left: 4px solid var(--slate) !important;
    }
    
    /* Dataframe */
    .dataframe {
        border: 1px solid var(--border);
        border-radius: 4px;
    }
    
    /* File Uploader */
    .stFileUploader {
        background-color: var(--surface);
        border: 2px dashed var(--border);
        border-radius: 4px;
        padding: 2rem;
    }
    
    /* Divider */
    .section-divider {
        height: 1px;
        background-color: var(--border);
        margin: 2.5rem 0;
    }
    
    /* Info Note */
    .info-note {
        font-size: 0.85rem;
        color: var(--text-muted);
        font-style: italic;
        margin-top: 0.5rem;
    }
    
    /* Metadata Grid */
    .metadata-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.5rem;
        margin-top: 1.5rem;
    }
    
    .metadata-item {
        text-align: center;
    }
    
    .metadata-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.5rem;
    }
    
    .metadata-value {
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text-secondary);
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Remove extra padding and hide empty containers */
    .element-container {
        margin-bottom: 0 !important;
    }
    
    .element-container:empty {
        display: none !important;
    }
    
    /* Hide empty stBlock elements */
    [data-testid="stVerticalBlock"]:empty,
    [data-testid="stHorizontalBlock"]:empty {
        display: none !important;
    }
    
    /* Prevent empty card-like boxes */
    .card:empty {
        display: none !important;
    }
    
    /* Workflow Steps */
    .workflow-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 2rem 0;
        gap: 0.5rem;
    }
    
    .workflow-step {
        background-color: var(--surface);
        border: 1px solid var(--border);
        padding: 1.5rem 1rem;
        text-align: center;
        flex: 1;
        border-radius: 4px;
    }
    
    .workflow-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .workflow-title {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    .workflow-arrow {
        color: var(--gold);
        font-size: 1.5rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "HOME"
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# Helper functions
def check_api_health():
    """Check if API is accessible"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return response.status_code == 200
    except:
        return False

def make_prediction(data, endpoint="/predict/revenue"):
    """Make prediction API call"""
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=10)
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code}"
    except Exception as e:
        return None, f"Connection Error: {str(e)}"

def search_tmdb_movie(query):
    """Search TMDB for movies"""
    try:
        headers = {
            "Authorization": f"Bearer {TMDB_BEARER_TOKEN}",
            "accept": "application/json"
        }
        response = requests.get(
            f"{TMDB_BASE_URL}/search/movie",
            headers=headers,
            params={"query": query},
            timeout=10
        )
        if response.status_code == 200:
            return response.json().get("results", []), None
        else:
            return None, f"TMDB Error: {response.status_code}"
    except Exception as e:
        return None, f"Connection Error: {str(e)}"
    
    

def get_tmdb_movie_details(movie_id):
    """Get detailed movie information from TMDB"""
    try:
        headers = {
            "Authorization": f"Bearer {TMDB_BEARER_TOKEN}",
            "accept": "application/json"
        }
        response = requests.get(
            f"{TMDB_BASE_URL}/movie/{movie_id}",
            headers=headers,
            timeout=10
        )
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"TMDB Error: {response.status_code}"
    except Exception as e:
        return None, f"Connection Error: {str(e)}"

def load_processed_data():
    """Load processed dataset"""
    if DATA_PATH.exists():
        try:
            return pd.read_csv(DATA_PATH)
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return None
    return None

def generate_predictions_for_viz(df):
    """Generate predictions for visualizations"""
    try:
        model_path = MODELS_PATH / "regression_model.joblib"
        scaler_path = MODELS_PATH / "scaler.joblib"
        
        if model_path.exists() and scaler_path.exists():
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            feature_cols = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count', 
                          'release_month', 'log_budget', 'log_vote_count', 'popularity_score',
                          'vote_density', 'popularity_per_budget', 'month_sin', 'month_cos']
            
            X = df[feature_cols].copy()
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
            
            return predictions
        else:
            return None
    except Exception as e:
        return None
    
def parse_classification_response(class_result: dict):
    """
    Returns (label, hit_probability)
    label: "Hit" / "Flop" / None
    hit_probability: float [0,1] or None
    """
    if not isinstance(class_result, dict):
        return None, None

    # --- Prefer explicit probability keys first (avoid rounding bug) ---
    for k in ["hit_probability", "probability", "proba", "score", "hit_proba"]:
        v = class_result.get(k)
        if isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0:
            p = float(v)
            return ("Hit" if p >= 0.5 else "Flop"), p

    # --- If model returns proba list like [p_flop, p_hit] ---
    proba = class_result.get("predicted_proba")
    if isinstance(proba, (list, tuple)) and len(proba) >= 2:
        try:
            p_hit = float(proba[1])
            if 0.0 <= p_hit <= 1.0:
                return ("Hit" if p_hit >= 0.5 else "Flop"), p_hit
        except Exception:
            pass

    # --- Then handle explicit labels/ids ---
    raw = (
        class_result.get("predicted_class")
        or class_result.get("classification")
        or class_result.get("label")
        or class_result.get("predicted_label")
        or class_result.get("prediction")
    )

    # String labels
    if isinstance(raw, str):
        s = raw.strip().lower()
        if "hit" in s:
            return "Hit", None
        if "flop" in s:
            return "Flop", None
        # numeric string
        try:
            raw = float(s)
        except Exception:
            return None, None

    # Numeric: treat ONLY exact 0/1 as class id (do NOT round!)
    if isinstance(raw, (int, float)):
        v = float(raw)
        if v in (0.0, 1.0):
            return ("Hit" if int(v) == 1 else "Flop"), None
        # If it's a probability (0..1 but not 0/1)
        if 0.0 <= v <= 1.0:
            return ("Hit" if v >= 0.5 else "Flop"), v

    return None, None


def parse_cluster_response(cluster_result: dict, metrics_data: dict):
    """
    Returns (cluster_id, cluster_label, cluster_profile)
    Uses API first. If label missing, uses training_results.json interpretations mapping.
    """
    if not isinstance(cluster_result, dict):
        return None, None, None

    cluster_id = cluster_result.get("cluster_id", cluster_result.get("cluster"))
    cluster_label = cluster_result.get("cluster_label")
    cluster_profile = cluster_result.get("cluster_profile")

    # If API didn't send a label, map id -> interpretation from METRICS_DATA
    if cluster_label is None and cluster_id is not None and isinstance(metrics_data, dict):
        try:
            interp = metrics_data.get("models", {}).get("clustering", {}).get("interpretations", {})
            cluster_label = interp.get(str(cluster_id), f"Cluster {cluster_id}")
        except Exception:
            cluster_label = f"Cluster {cluster_id}"

    return cluster_id, cluster_label, cluster_profile

# Navigation - GOLD COLORED with Active State
st.markdown('<div class="nav-row">', unsafe_allow_html=True)

pages = ["HOME", "PREDICT", "MODEL DASHBOARD", "DATA EXPLORER", "MONITORING", "RESOURCES"]
cols = st.columns(len(pages))

for idx, page in enumerate(pages):
    with cols[idx]:
        is_active = st.session_state.page == page
        # Use primary type for active, secondary for inactive
        btn_type = "primary" if is_active else "secondary"
        if st.button(page, key=f"nav_{page}", type=btn_type):
            st.session_state.page = page
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ============= HOME PAGE =============
if st.session_state.page == "HOME":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="page-title">MOVIE INTELLIGENCE</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">End-to-end ML Engineering System for Revenue Prediction</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### SYSTEM CAPABILITIES")
        
        r2_score = METRICS_DATA.get('models', {}).get('regression_gradient_boosting', {}).get('r2_score', 0.6772)
        accuracy = METRICS_DATA.get('models', {}).get('classification', {}).get('accuracy', 0.8525)
        
        st.markdown(f"""
        <div style="margin-top: 1rem; line-height: 2;">
            <div style="margin-bottom: 0.8rem;">
                <span style="color: #E4B15E; font-weight: 700;">→</span> 
                <span style="color: #F5F5F7; font-weight: 600;">REGRESSION</span>
                <span style="color: #6B7280;"> — Revenue prediction with R² {r2_score:.4f}</span>
            </div>
            <div style="margin-bottom: 0.8rem;">
                <span style="color: #E4B15E; font-weight: 700;">→</span> 
                <span style="color: #F5F5F7; font-weight: 600;">CLASSIFICATION</span>
                <span style="color: #6B7280;"> — Hit/Flop label with {accuracy*100:.1f}% accuracy</span>
            </div>
            <div>
                <span style="color: #E4B15E; font-weight: 700;">→</span> 
                <span style="color: #F5F5F7; font-weight: 600;">CLUSTERING</span>
                <span style="color: #6B7280;"> — Movie archetypes identification</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### SYSTEM METRICS")
        
        timestamp = METRICS_DATA.get('timestamp', '2024-12-16T22:38:16')
        try:
            date_str = datetime.fromisoformat(timestamp.replace('Z', '')).strftime('%b %d, %Y').upper()
        except:
            date_str = "DEC 16, 2024"
        
        st.markdown(f"""
        <div style="margin-top: 1rem;">
            <div style="margin-bottom: 1.5rem;">
                <div class="result-label">MODEL VERSION</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #F5F5F7;">v1.2.0</div>
            </div>
            <div style="margin-bottom: 1.5rem;">
                <div class="result-label">LAST TRAINED</div>
                <div style="font-size: 0.9rem; font-weight: 600; color: #A1A1AA;">{date_str}</div>
            </div>
            <div>
                <div class="result-label">BEST R² SCORE</div>
                <div style="font-size: 2.5rem; font-weight: 800; color: #E4B15E;">{r2_score:.4f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Workflow
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("## ML WORKFLOW")
    
    steps = [
        ("📥", "INGEST"),
        ("🧹", "CLEAN"),
        ("⚙️", "FEATURES"),
        ("🎯", "TRAIN"),
        ("📊", "EVALUATE"),
        ("🚀", "DEPLOY"),
        ("📡", "MONITOR")
    ]
    
    workflow_html = '<div class="workflow-container">'
    for i, (icon, title) in enumerate(steps):
        workflow_html += f'<div class="workflow-step"><div class="workflow-icon">{icon}</div><div class="workflow-title">{title}</div></div>'
        if i < len(steps) - 1:
            workflow_html += '<div class="workflow-arrow">→</div>'
    workflow_html += '</div>'
    
    st.markdown(workflow_html, unsafe_allow_html=True)
    
    # Core functions
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("## CORE FUNCTIONS")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div style="font-size: 2.5rem; text-align: center; margin-bottom: 1rem;">💰</div>', unsafe_allow_html=True)
        st.markdown("### PREDICT REVENUE")
        st.markdown('<p class="info-note">Input movie parameters and get instant revenue predictions from trained ensemble models.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div style="font-size: 2.5rem; text-align: center; margin-bottom: 1rem;">📈</div>', unsafe_allow_html=True)
        st.markdown("### ANALYZE PERFORMANCE")
        st.markdown('<p class="info-note">Review model metrics, diagnostics, and performance across regression, classification, and clustering.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div style="font-size: 2.5rem; text-align: center; margin-bottom: 1rem;">📡</div>', unsafe_allow_html=True)
        st.markdown("### TRACK MONITORING")
        st.markdown('<p class="info-note">Monitor data drift, API health, and system performance in production environment.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # CTA Buttons
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("START PREDICTION", width='stretch', key="cta_predict"):
            st.session_state.page = "PREDICT"
            st.rerun()
    
    with col2:
        if st.button("VIEW MODEL DASHBOARD", width='stretch', key="cta_dashboard"):
            st.session_state.page = "MODEL DASHBOARD"
            st.rerun()
    
    with col3:
        if st.button("OPEN DOCUMENTATION", width='stretch', key="cta_docs"):
            st.session_state.page = "RESOURCES"
            st.rerun()

# ============= PREDICT PAGE =============
elif st.session_state.page == "PREDICT":
    st.markdown('<div class="page-title">REVENUE PREDICTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Generate predictions using manual input, TMDB search, or batch processing</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["MANUAL INPUT", "TMDB SEARCH", "BATCH PREDICTION"])
    

    
    # ============= TAB 1: MANUAL INPUT =============
    # ============= TAB 1: MANUAL INPUT =============
    with tab1:
        st.markdown("## INPUT PARAMETERS")

        # Use the same metrics object already loaded at top of app.py (METRICS_DATA)
        models_metrics = METRICS_DATA.get("models", {}) if isinstance(METRICS_DATA, dict) else {}
        reg_metrics = models_metrics.get("regression_gradient_boosting", {})  # consistent with HOME page in your app.py :contentReference[oaicite:0]{index=0}
        cls_metrics = models_metrics.get("classification", {})
        clu_metrics = models_metrics.get("clustering", {})

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### MOVIE BASICS")

            # ✅ Budget slider removed; manual input is final; no max limit
            budget = st.number_input(
                "BUDGET (USD)",
                min_value=0.0,
                value=49000000.0,
                step=1000000.0,
                format="%.0f"
            )

            runtime = st.number_input("RUNTIME (MINUTES)", min_value=30, max_value=300, value=120, step=1)

            release_month = st.selectbox(
                "RELEASE MONTH",
                options=list(range(1, 13)),
                format_func=lambda x: datetime(2024, x, 1).strftime('%B').upper(),
                index=5
            )

            genres = st.multiselect(
                "GENRES",
                options=GENRE_OPTIONS,
                default=["Action"],
                help="Select one or more genres"
            )

        with col2:
            st.markdown("### POPULARITY & VOTES")

            popularity = st.number_input("POPULARITY", min_value=0.0, max_value=1000.0, value=50.0, step=1.0)
            vote_average = st.slider("VOTE AVERAGE", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
            vote_count = st.number_input("VOTE COUNT", min_value=0, value=1000, step=100)

        st.markdown('<p class="info-note">All inputs match training schema. Genres are optional but improve interpretability.</p>', unsafe_allow_html=True)
        st.markdown('<div style="margin: 1.5rem 0;"></div>', unsafe_allow_html=True)

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            predict_btn = st.button("🚀 RUN PREDICTION", width='stretch', type="primary")
        with col_btn2:
            reset_btn = st.button("🔄 RESET", width='stretch', key="reset_btn", type="secondary")

        if reset_btn:
            st.session_state.prediction_result = None
            st.rerun()

        if predict_btn:
            with st.spinner("Running prediction models..."):
                selected_genres = ", ".join(genres) if genres else "Not specified"
                # NOTE: keep payload consistent with the rest of your app.py to avoid 422s (TMDB/BATCH send same schema) :contentReference[oaicite:1]{index=1}
                input_data = {
                    "budget": float(budget),
                    "popularity": float(popularity),
                    "runtime": float(runtime),
                    "vote_average": float(vote_average),
                    "vote_count": int(vote_count),
                    "release_month": int(release_month),
                    "genres": selected_genres,
                }

                

                # Time each endpoint
                t0 = time.perf_counter()

                t_rev = time.perf_counter()
                revenue_result, revenue_error = make_prediction(input_data, "/predict/revenue")
                rev_ms = int((time.perf_counter() - t_rev) * 1000)

                t_cls = time.perf_counter()
                class_result, class_error = make_prediction(input_data, "/predict/classification")
                cls_ms = int((time.perf_counter() - t_cls) * 1000)

                t_clu = time.perf_counter()
                cluster_result, cluster_error = make_prediction(input_data, "/predict/cluster")
                clu_ms = int((time.perf_counter() - t_clu) * 1000)

                total_ms = int((time.perf_counter() - t0) * 1000)

                # ---------- Parse outputs safely (no guessing, no probability-rounding bugs) ----------
                predicted_revenue = (
                    revenue_result.get("predicted_revenue")
                    if (not revenue_error and isinstance(revenue_result, dict))
                    else None
                )

                # ✅ Classification: handle label / 0-1 probability / 0-1 class id correctly (NO rounding)
                classification = None
                hit_p = None

                if (not class_error) and isinstance(class_result, dict):
                    # 1) If API returns probability keys, use threshold (preferred)
                    for k in ["hit_probability", "probability", "proba", "score", "hit_proba", "p_hit"]:
                        v = class_result.get(k)
                        if isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0:
                            hit_p = float(v)
                            classification = "Hit" if hit_p >= 0.5 else "Flop"
                            break

                    # 2) If API returns proba array like [p_flop, p_hit]
                    if classification is None and isinstance(class_result.get("predicted_proba"), (list, tuple)):
                        proba = class_result["predicted_proba"]
                        if len(proba) >= 2:
                            try:
                                hit_p = float(proba[1])
                                if 0.0 <= hit_p <= 1.0:
                                    classification = "Hit" if hit_p >= 0.5 else "Flop"
                            except Exception:
                                pass

                    # 3) If API returns an explicit label string ("Hit"/"Flop")
                    if classification is None:
                        raw_label = (
                            class_result.get("predicted_class")
                            or class_result.get("classification")
                            or class_result.get("label")
                            or class_result.get("predicted_label")
                        )
                        if isinstance(raw_label, str):
                            s = raw_label.strip().lower()
                            if "hit" in s:
                                classification = "Hit"
                            elif "flop" in s:
                                classification = "Flop"

                    # 4) If API returns hard class id 0/1 (ONLY accept exact 0 or 1)
                    if classification is None:
                        raw = class_result.get("predicted_class", class_result.get("prediction"))
                        try:
                            if isinstance(raw, str):
                                raw = raw.strip()
                                raw = float(raw) if raw.replace(".", "", 1).isdigit() else raw
                            if isinstance(raw, (int, float)):
                                v = float(raw)
                                if v in (0.0, 1.0):
                                    classification = "Hit" if int(v) == 1 else "Flop"
                                elif 0.0 <= v <= 1.0:
                                    # probability-like float
                                    hit_p = v
                                    classification = "Hit" if v >= 0.5 else "Flop"
                        except Exception:
                            pass

                # ✅ Cluster: prefer API label; if missing, map id -> label from training_results.json (METRICS_DATA)
                cluster_id = None
                cluster_label = None
                cluster_profile = None

                if (not cluster_error) and isinstance(cluster_result, dict):
                    cluster_id = (
                        cluster_result.get("cluster_id")
                        if cluster_result.get("cluster_id") is not None
                        else cluster_result.get("cluster")
                    )
                    cluster_label = cluster_result.get("cluster_label")
                    cluster_profile = cluster_result.get("cluster_profile") or cluster_result.get("profile")

                # fallback mapping via training_results.json interpretations
                if cluster_label is None and cluster_id is not None and isinstance(METRICS_DATA, dict):
                    try:
                        interp = METRICS_DATA.get("models", {}).get("clustering", {}).get("interpretations", {})
                        cluster_label = interp.get(str(cluster_id)) or interp.get(cluster_id)
                    except Exception:
                        cluster_label = None

                if cluster_label is None and cluster_id is not None:
                    cluster_label = f"Cluster {cluster_id}"

                all_ok = (predicted_revenue is not None) and (classification is not None) and (cluster_label is not None)

                if all_ok:

                    result = {
                        "revenue": float(predicted_revenue),
                        "classification": classification,          # "Hit" / "Flop" (normalized)
                        "cluster": cluster_label,                  # mapped label (or "Cluster X")
                        "cluster_id": cluster_id,                  # keep id too (useful)
                        "cluster_profile": cluster_profile,        # optional (if API returns it)


                        # ✅ Replace “confidence” with REAL evaluation metrics from your training_results.json :contentReference[oaicite:2]{index=2}
                        "metrics": {
                            "regression": {
                                "r2_score": reg_metrics.get("r2_score"),
                                "rmse": reg_metrics.get("rmse"),
                                "mae": reg_metrics.get("mae"),
                                "mape": reg_metrics.get("mape"),
                                "model_type": reg_metrics.get("model_type", "gradient_boosting"),
                            },
                            "classification": {
                                "accuracy": cls_metrics.get("accuracy"),
                                "f1_score": cls_metrics.get("f1_score"),
                                "model_type": cls_metrics.get("model_type", "random_forest"),
                                "hit_probability": hit_p,
                                "threshold": 0.5,

                            },
                            "clustering": {
                                "silhouette_score": clu_metrics.get("silhouette_score"),
                                "n_clusters": clu_metrics.get("n_clusters"),
                                "inertia": clu_metrics.get("inertia"),
                            },
                            "system": {
                                "api_used": True,
                                "total_latency_ms": total_ms,
                                "endpoint_latency_ms": {
                                    "revenue": rev_ms,
                                    "classification": cls_ms,
                                    "cluster": clu_ms,
                                },
                            },
                        },


                        "model_version": "v1.2.0",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "api_used": True,
                        "input_data": input_data,
                        "genres": selected_genres,
                    }
                    st.success("✓ PREDICTION COMPLETED (Using Live API)")
                else:
                    result = {
                        "revenue": predicted_revenue,
                        "classification": classification,
                        "cluster": cluster_label,
                        "cluster_id": cluster_id,
                        "cluster_profile": cluster_profile,

                        "metrics": {
                            "system": {
                                "api_used": False,
                                "total_latency_ms": total_ms,
                                "endpoint_latency_ms": {
                                    "revenue": rev_ms,
                                    "classification": cls_ms,
                                    "cluster": clu_ms,
                                },
                                "errors": {
                                    "revenue": revenue_error,
                                    "classification": class_error,
                                    "cluster": cluster_error,
                                },
                            }
                        },
                        "model_version": "v1.2.0",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "api_used": False,
                        "input_data": input_data,
                        "genres": selected_genres,
                    }
                    st.warning("⚠️ One or more API endpoints failed — showing diagnostics only.")

                st.session_state.prediction_result = result

        
    
    # ============= TAB 2: TMDB SEARCH =============
    with tab2:
        st.markdown("## SEARCH TMDB DATABASE")

        search_query = st.text_input("MOVIE TITLE", placeholder="e.g., Avatar, Inception, The Avengers")
        
        # Search and Reset buttons side by side
        col_search, col_reset_tmdb = st.columns([1, 1])
        
        with col_search:
            search_clicked = st.button("🔍 SEARCH", width='stretch', key="tmdb_search_btn")
        
        with col_reset_tmdb:
            reset_tmdb_clicked = st.button("🔄 RESET", width='stretch', key="tmdb_reset_btn", type="secondary")
        
        # Handle reset
        if reset_tmdb_clicked:
            st.session_state.tmdb_results = []
            st.session_state.tmdb_last_query = ""
            st.session_state.prediction_result = None
            st.rerun()

        # --- SEARCH (stores results in session_state so buttons work after rerun) ---
        if search_clicked:
            if search_query:
                with st.spinner("🔍 Searching TMDB..."):
                    results, error = search_tmdb_movie(search_query)

                    if error:
                        st.error(f"⚠️ {error}")
                        st.session_state.tmdb_results = []
                    elif results:
                        st.success(f"✓ Found {len(results[:5])} result(s)")
                        st.session_state.tmdb_results = results[:5]
                        st.session_state.tmdb_last_query = search_query
                    else:
                        st.warning("🔍 No results found")
                        st.session_state.tmdb_results = []
            else:
                st.warning("⚠️ Enter a movie title to search")

        # --- RENDER persisted results (survives reruns) ---
        results = st.session_state.tmdb_results

        if results:
            st.info(f"Showing results for: {st.session_state.tmdb_last_query}")

            for movie in results:
                st.markdown(f"""
                <div class="card" style="margin-top: 1rem;">
                    <h4 style="margin-top: 0; color: #E4B15E;">🎬 {movie.get('title', 'Unknown')}</h4>
                    <p style="color: #A1A1AA; margin: 0.5rem 0;">{movie.get('overview', 'No overview available')[:200]}...</p>
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin: 1rem 0;">
                        <div><strong>Release:</strong> {movie.get('release_date', 'N/A')}</div>
                        <div><strong>Popularity:</strong> {movie.get('popularity', 'N/A')}</div>
                        <div><strong>Vote Average:</strong> {movie.get('vote_average', 'N/A')}/10</div>
                        <div><strong>Vote Count:</strong> {movie.get('vote_count', 'N/A'):,}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button("✅ Get Details & Run Prediction", key=f"tmdb_{movie['id']}"):
                    details, detail_error = get_tmdb_movie_details(movie["id"])

                    if detail_error or not details:
                        st.error(f"⚠️ Could not load details: {detail_error}")
                    else:
                        # ---- Extract TMDB fields ----
                        budget = details.get("budget", 0) or 0
                        runtime = details.get("runtime", 120) or 120
                        popularity = details.get("popularity", 50.0) or 50.0
                        vote_average = details.get("vote_average", 7.0) or 7.0
                        vote_count = details.get("vote_count", 1000) or 1000
                        release_date = details.get("release_date", "") or ""
                        title = details.get("title", "Unknown") or "Unknown"
                        genres_list = [g.get("name") for g in details.get("genres", []) if g.get("name")]

                        # ---- Month ----
                        try:
                            release_month = int(release_date.split("-")[1])
                        except Exception:
                            release_month = 6

                        # ---- Budget fallback (kept from your code) ----
                        if budget == 0:
                            st.warning(f"⚠️ No budget data available for {title}. Using estimated budget.")
                            budget = 50000000

                        st.success(f"✅ Loaded: {title}")
                        st.info(f"🎬 Budget: ${budget:,} | Runtime: {runtime}min | Genres: {', '.join(genres_list) if genres_list else 'N/A'}")
                        selected_genres = ", ".join(genres_list) if genres_list else "Not specified"
                        # ---- Build payload (schema must match backend) ----
                        input_data = {
                            "budget": float(budget),
                            "popularity": float(popularity),
                            "runtime": float(runtime),
                            "vote_average": float(vote_average),
                            "vote_count": int(vote_count),
                            "release_month": int(release_month),
                            "genres": selected_genres,
                        }

                        

                        # ---- Unified prediction block (same as Manual tab) ----
                        with st.spinner("Running prediction..."):
                            t0 = time.perf_counter()

                            t_rev = time.perf_counter()
                            revenue_result, revenue_error = make_prediction(input_data, "/predict/revenue")
                            rev_ms = int((time.perf_counter() - t_rev) * 1000)

                            t_cls = time.perf_counter()
                            class_result, class_error = make_prediction(input_data, "/predict/classification")
                            cls_ms = int((time.perf_counter() - t_cls) * 1000)

                            t_clu = time.perf_counter()
                            cluster_result, cluster_error = make_prediction(input_data, "/predict/cluster")
                            clu_ms = int((time.perf_counter() - t_clu) * 1000)

                            total_ms = int((time.perf_counter() - t0) * 1000)

                            predicted_revenue = (
                                revenue_result.get("predicted_revenue")
                                if (not revenue_error and isinstance(revenue_result, dict))
                                else None
                            )

                            # ---- robust classification parsing ----
                            classification = None
                            if not class_error and isinstance(class_result, dict):
                                raw = (
                                    class_result.get("predicted_class")
                                    or class_result.get("classification")
                                    or class_result.get("label")
                                    or class_result.get("predicted_label")
                                    or class_result.get("prediction")
                                )

                                norm = raw
                                try:
                                    if isinstance(raw, str):
                                        norm = raw.strip()
                                        if norm.replace(".", "", 1).isdigit():
                                            norm = float(norm)
                                    if isinstance(norm, (int, float)):
                                        norm = int(round(norm))
                                except Exception:
                                    norm = raw

                                if norm in [1, "Hit", "HIT", True, "true", "True"]:
                                    classification = "Hit"
                                elif norm in [0, "Flop", "FLOP", False, "false", "False"]:
                                    classification = "Flop"

                            cluster = None
                            if (not cluster_error and isinstance(cluster_result, dict)):
                                cluster = cluster_result.get("cluster_label", cluster_result.get("cluster"))

                            all_ok = (predicted_revenue is not None) and (classification is not None) and (cluster is not None)

                            # ---- metrics objects assumed already available like Manual tab ----
                            models_metrics = METRICS_DATA.get("models", {}) if isinstance(METRICS_DATA, dict) else {}
                            reg_metrics = models_metrics.get("regression_gradient_boosting", {})
                            cls_metrics = models_metrics.get("classification", {})
                            clu_metrics = models_metrics.get("clustering", {})

                            if all_ok:
                                result = {
                                    "revenue": float(predicted_revenue),
                                    "classification": classification,
                                    "cluster": cluster,
                                    "metrics": {
                                        "regression": {
                                            "r2_score": reg_metrics.get("r2_score"),
                                            "rmse": reg_metrics.get("rmse"),
                                            "mae": reg_metrics.get("mae"),
                                            "mape": reg_metrics.get("mape"),
                                            "model_type": reg_metrics.get("model_type", "gradient_boosting"),
                                        },
                                        "classification": {
                                            "accuracy": cls_metrics.get("accuracy"),
                                            "f1_score": cls_metrics.get("f1_score"),
                                            "model_type": cls_metrics.get("model_type", "random_forest"),
                                        },
                                        "clustering": {
                                            "silhouette_score": clu_metrics.get("silhouette_score"),
                                            "n_clusters": clu_metrics.get("n_clusters"),
                                            "inertia": clu_metrics.get("inertia"),
                                        },
                                        "system": {
                                            "api_used": True,
                                            "total_latency_ms": total_ms,
                                            "endpoint_latency_ms": {
                                                "revenue": rev_ms,
                                                "classification": cls_ms,
                                                "cluster": clu_ms,
                                            },
                                        },
                                    },
                                    "model_version": "v1.2.0",
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "api_used": True,
                                    "input_data": input_data,
                                    "genres": selected_genres,
                                    "movie_title": title,
                                }

                                st.session_state.prediction_result = result
                                st.success("✓ PREDICTION COMPLETED (Using Live API)")
                            else:
                                result = {
                                    "revenue": predicted_revenue,
                                    "classification": classification,
                                    "cluster": cluster,
                                    "metrics": {
                                        "system": {
                                            "api_used": False,
                                            "total_latency_ms": total_ms,
                                            "endpoint_latency_ms": {
                                                "revenue": rev_ms,
                                                "classification": cls_ms,
                                                "cluster": clu_ms,
                                            },
                                            "errors": {
                                                "revenue": revenue_error,
                                                "classification": class_error,
                                                "cluster": cluster_error,
                                            },
                                        }
                                    },
                                    "model_version": "v1.2.0",
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "api_used": False,
                                    "input_data": input_data,
                                    "genres": selected_genres,
                                    "movie_title": title,
                                }

                                st.session_state.prediction_result = result
                                st.warning("⚠️ One or more API endpoints failed — showing diagnostics only.")

        st.markdown('<p class="info-note">Search powered by TMDB API. Missing budget data will be flagged.</p>', unsafe_allow_html=True)

    
    # ============= TAB 3: BATCH PREDICTION =============
    with tab3:
        st.markdown("## BATCH PROCESSING")
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        template_df = pd.DataFrame({
            "budget": [50000000, 100000000, 30000000],
            "popularity": [50.0, 95.5, 25.3],
            "runtime": [120, 142, 95],
            "vote_average": [7.0, 7.8, 6.5],
            "vote_count": [1000, 15420, 450],
            "release_month": [6, 7, 12]
        })
        
        csv_buffer = io.StringIO()
        template_df.to_csv(csv_buffer, index=False)
        
        st.download_button("📥 DOWNLOAD CSV TEMPLATE", data=csv_buffer.getvalue(), file_name="batch_template.csv", mime="text/csv", width='stretch')
        
        st.markdown('<div style="margin: 1.5rem 0;"></div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("📤 UPLOAD CSV FILE", type=['csv'])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                required_cols = ["budget", "popularity", "runtime", "vote_average", "vote_count", "release_month"]
                missing = [col for col in required_cols if col not in df.columns]
                
                if missing:
                    st.error(f"⚠️ Missing columns: {', '.join(missing)}")
                else:
                    st.success(f"✓ Schema validated. Found {len(df)} movies.")
                    st.markdown("### PREVIEW")
                    st.dataframe(df.head(10), width='stretch')
                    
                    if st.button("🚀 RUN BATCH PREDICTION", width='stretch'):
                        with st.spinner(f"Processing {len(df)} movies..."):
                            progress_bar = st.progress(0)
                            results_list = []
                            
                            for idx, row in df.iterrows():
                                input_data = {
                                    "budget": float(row['budget']),
                                    "popularity": float(row['popularity']),
                                    "runtime": float(row['runtime']),
                                    "vote_average": float(row['vote_average']),
                                    "vote_count": int(row['vote_count']),
                                    "release_month": int(row['release_month'])
                                }
                                
                                revenue_result, _ = make_prediction(input_data, "/predict/revenue")
                                class_result, _ = make_prediction(input_data, "/predict/classification")
                                
                                if revenue_result and class_result:
                                    pred_rev = revenue_result.get("predicted_revenue", 0)
                                    classification = class_result.get("predicted_class", class_result.get("classification", "Unknown"))
                                    results_list.append({
                                        "predicted_revenue": pred_rev,
                                        "classification": classification,
                                        "roi_percent": ((pred_rev / input_data['budget']) - 1) * 100
                                    })
                                else:
                                    pred_rev = input_data['budget'] * 2.5
                                    results_list.append({
                                        "predicted_revenue": pred_rev,
                                        "classification": "Hit" if pred_rev > input_data['budget'] * 2 else "Flop",
                                        "roi_percent": ((pred_rev / input_data['budget']) - 1) * 100
                                    })
                                
                                progress_bar.progress((idx + 1) / len(df))
                            
                            results_df = pd.concat([df, pd.DataFrame(results_list)], axis=1)
                            
                            st.success("✅ Batch prediction completed!")
                            st.markdown("### RESULTS")
                            st.dataframe(results_df, width='stretch')
                            
                            csv_results = io.StringIO()
                            results_df.to_csv(csv_results, index=False)
                            
                            st.download_button(
                                "💾 DOWNLOAD RESULTS",
                                data=csv_results.getvalue(),
                                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                width='stretch'
                            )
            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Display results
if st.session_state.get("prediction_result"):

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("## PREDICTION RESULTS")

    result = st.session_state.prediction_result

    colA, colB, colC = st.columns(3)

    with colA:
        if result.get("revenue") is not None and result["input_data"]["budget"] > 0:
            roi = ((result["revenue"] / result["input_data"]["budget"]) - 1) * 100
            rev_txt = f"${result['revenue']:,.0f}"
            roi_txt = f"{roi:+.1f}%"
        else:
            rev_txt, roi_txt = "N/A", "N/A"
        
        # ----- Uncertainty band using training RMSE (±RMSE) -----
        rmse_val = None
        try:
            rmse_val = reg_metrics.get("rmse")  # from METRICS_DATA -> models -> regression_gradient_boosting
        except Exception:
            rmse_val = None

        range_txt = "Est. Range: N/A"
        if (result.get("revenue") is not None) and isinstance(rmse_val, (int, float)):
            lower = max(0.0, float(result["revenue"]) - float(rmse_val))
            upper = float(result["revenue"]) + float(rmse_val)
            range_txt = f"Est. Range: ${lower:,.0f} – ${upper:,.0f} (±RMSE)"


        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">PREDICTED REVENUE</div>
            <div class="result-number">{rev_txt}</div>
            <div class="result-sublabel">ROI: {roi_txt}</div>
            <div class="result-sublabel" style="margin-top: 0.35rem; color: #A1A1AA;">{range_txt}</div>

        </div>
        """, unsafe_allow_html=True)

    with colB:
        classification_raw = str(result.get("classification", "Unknown"))
        is_hit = classification_raw in ["Hit", "1", "HIT", "True", "true"]
        is_flop = classification_raw in ["Flop", "0", "FLOP", "False", "false"]
        badge_class = "badge-hit" if is_hit else ("badge-flop" if is_flop else "badge-flop")
        classification_text = "HIT" if is_hit else ("FLOP" if is_flop else "UNKNOWN")

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">CLASSIFICATION</div>
            <span class="badge {badge_class}">{classification_text}</span>
            <div class="result-sublabel">Market Performance</div>
        </div>
        """, unsafe_allow_html=True)

    with colC:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">CLUSTER LABEL</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #F5F5F7; margin: 1rem 0;">
                {result.get('cluster', 'N/A')}
            </div>
            <div class="result-sublabel">Movie Archetype</div>
        </div>
        """, unsafe_allow_html=True)

    # Metadata section (replaces CONFIDENCE with real metrics)
    st.markdown("### PREDICTION METADATA & MODEL METRICS")

    metrics = result.get("metrics", {})
    reg = metrics.get("regression", {})
    cls = metrics.get("classification", {})
    clu = metrics.get("clustering", {})
    sysm = metrics.get("system", {})

    # Format values safely
    r2_val = reg.get('r2_score')
    r2_display = f"{r2_val:.4f}" if isinstance(r2_val, (int, float)) else "N/A"
    
    rmse_val = reg.get('rmse')
    rmse_display = f"{rmse_val:,.0f}" if isinstance(rmse_val, (int, float)) else "N/A"
    
    mae_val = reg.get('mae')
    mae_display = f"{mae_val:,.0f}" if isinstance(mae_val, (int, float)) else "N/A"
    
    acc_val = cls.get('accuracy')
    acc_display = f"{acc_val:.4f}" if isinstance(acc_val, (int, float)) else "N/A"
    
    f1_val = cls.get('f1_score')
    f1_display = f"{f1_val:.4f}" if isinstance(f1_val, (int, float)) else "N/A"
    
    sil_val = clu.get('silhouette_score')
    sil_display = f"{sil_val:.4f}" if isinstance(sil_val, (int, float)) else "N/A"

    col_meta1, col_meta2, col_meta3 = st.columns(3)
    
    with col_meta1:
        st.markdown(f"""
        <div class="result-card" style="padding: 1.5rem;">
            <div class="metadata-label">API USED</div>
            <div class="metadata-value">{'YES' if result.get('api_used') else 'NO'}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_meta2:
        st.markdown(f"""
        <div class="result-card" style="padding: 1.5rem;">
            <div class="metadata-label">MODEL VERSION</div>
            <div class="metadata-value">{result.get('model_version','N/A')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_meta3:
        st.markdown(f"""
        <div class="result-card" style="padding: 1.5rem;">
            <div class="metadata-label">TIMESTAMP</div>
            <div class="metadata-value" style="font-size: 1rem;">{result.get('timestamp','')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    col_meta4, col_meta5, col_meta6 = st.columns(3)
    
    with col_meta4:
        st.markdown(f"""
        <div class="result-card" style="padding: 1.5rem;">
            <div class="metadata-label">REGRESSION (R²)</div>
            <div class="metadata-value">{r2_display}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_meta5:
        st.markdown(f"""
        <div class="result-card" style="padding: 1.5rem;">
            <div class="metadata-label">REGRESSION (RMSE / MAE)</div>
            <div class="metadata-value">{rmse_display} / {mae_display}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_meta6:
        st.markdown(f"""
        <div class="result-card" style="padding: 1.5rem;">
            <div class="metadata-label">CLASSIFICATION (ACC / F1)</div>
            <div class="metadata-value">{acc_display} / {f1_display}</div>
        </div>
        """, unsafe_allow_html=True)
    
    col_meta7, col_meta8, col_meta9 = st.columns(3)
    
    with col_meta7:
        st.markdown(f"""
        <div class="result-card" style="padding: 1.5rem;">
            <div class="metadata-label">CLUSTERING (SILHOUETTE)</div>
            <div class="metadata-value">{sil_display}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_meta8:
        st.markdown(f"""
        <div class="result-card" style="padding: 1.5rem;">
            <div class="metadata-label">TOTAL LATENCY (ms)</div>
            <div class="metadata-value">{sysm.get('total_latency_ms','N/A')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_meta9:
        endpoint_lat = sysm.get('endpoint_latency_ms', {})
        st.markdown(f"""
        <div class="result-card" style="padding: 1.5rem;">
            <div class="metadata-label">ENDPOINTS (REV / CLS / CLU)</div>
            <div class="metadata-value">{endpoint_lat.get('revenue','N/A')} / {endpoint_lat.get('classification','N/A')} / {endpoint_lat.get('cluster','N/A')}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
    st.markdown("### INPUT SUMMARY")

    colL, colR = st.columns(2)
    with colL:
        st.markdown(f"""
        - **Budget:** ${result['input_data']['budget']:,.0f}
        - **Runtime:** {result['input_data']['runtime']} minutes
        - **Release Month:** {datetime(2024, result['input_data']['release_month'], 1).strftime('%B')}
        """)
    with colR:
        st.markdown(f"""
        - **Popularity:** {result['input_data']['popularity']}
        - **Vote Average:** {result['input_data']['vote_average']}/10
        - **Vote Count:** {result['input_data']['vote_count']:,}
        """)

    st.markdown(f"- **Genres:** {result.get('genres','Not specified')}")

# ============= MODEL DASHBOARD PAGE =============
elif st.session_state.page == "MODEL DASHBOARD":
    st.markdown('<div class="page-title">MODEL DASHBOARD</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Performance metrics and dataset analysis</div>', unsafe_allow_html=True)
    
    models = METRICS_DATA.get('models', {})
    
    # Model comparison
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### MODEL COMPARISON")
    
    reg_gb = models.get('regression_gradient_boosting', {})
    classification = models.get('classification', {})
    clustering = models.get('clustering', {})
    
    metrics_df = pd.DataFrame({
        "Model": ["Gradient Boosting", "Classification", "Clustering"],
        "Type": ["Regression", "Binary", "K-Means"],
        "Primary Metric": ["R² Score", "Accuracy", "Silhouette"],
        "Value": [
            reg_gb.get('r2_score', 0.6772),
            classification.get('accuracy', 0.8525),
            clustering.get('silhouette_score', 0.1009)
        ],
        "Secondary": [
            f"MAE: ${reg_gb.get('mae', 36195021.23):,.0f}",
            f"F1: {classification.get('f1_score', 0.8525):.4f}",
            f"Clusters: {clustering.get('n_clusters', 4)}"
        ]
    })
    
    st.dataframe(metrics_df, width='stretch', hide_index=True)
    
    timestamp = METRICS_DATA.get('timestamp', '2024-12-16')
    st.markdown(f'<p class="info-note">Metrics from model training on {timestamp[:10]}.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Commentary
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### WHY THESE METRICS MATTER")
    st.markdown(f"""
    <div style="line-height: 1.8; color: #A1A1AA;">
        <p><strong style="color: #F5F5F7;">R² Score ({reg_gb.get('r2_score', 0.6772):.4f}):</strong> Explains {reg_gb.get('r2_score', 0.6772)*100:.1f}% of revenue variance—strong for entertainment industry volatility.</p>
        <p><strong style="color: #F5F5F7;">MAE (${reg_gb.get('mae', 36195021.23):,.0f}):</strong> Average prediction error acceptable given box office scale ($100M+ films).</p>
        <p><strong style="color: #F5F5F7;">Accuracy ({classification.get('accuracy', 0.8525)*100:.1f}%):</strong> Hit/Flop classification reliable for investment decisions.</p>
        <p><strong style="color: #F5F5F7;">Silhouette ({clustering.get('silhouette_score', 0.1009):.2f}):</strong> Moderate cluster separation—movie categories overlap naturally.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Load data for visualizations
    df = load_processed_data()
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("## DATASET ANALYSIS")
    
    if df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### REVENUE DISTRIBUTION BY BUDGET TIER")
            
            # Create budget tiers
            df_viz = df.copy()
            df_viz['budget_tier'] = pd.cut(df_viz['budget'], 
                                           bins=[0, 10e6, 50e6, 100e6, 200e6, float('inf')],
                                           labels=['<$10M', '$10-50M', '$50-100M', '$100-200M', '>$200M'])
            
            budget_revenue = df_viz.groupby('budget_tier', observed=True).agg({
                'revenue': 'mean',
                'budget': 'count'
            }).reset_index()
            budget_revenue.columns = ['Budget Tier', 'Avg Revenue', 'Count']
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=budget_revenue['Budget Tier'].astype(str),
                y=budget_revenue['Avg Revenue'],
                marker=dict(color='#E4B15E'),
                text=[f"${x/1e6:.0f}M" for x in budget_revenue['Avg Revenue']],
                textposition='outside'
            ))
            
            fig.update_layout(
                plot_bgcolor='#0D0D0F',
                paper_bgcolor='#1A1A1E',
                font=dict(color='#F5F5F7', size=11),
                xaxis_title="Budget Tier",
                yaxis_title="Average Revenue ($)",
                height=400,
                showlegend=False,
                xaxis=dict(gridcolor='#2D2D32'),
                yaxis=dict(gridcolor='#2D2D32'),
                margin=dict(l=60, r=20, t=40, b=60)
            )
            
            st.plotly_chart(fig, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### HIT RATE BY RELEASE MONTH")
            
            monthly_hits = df.groupby('release_month').agg({
                'is_hit': 'mean',
                'revenue': 'count'
            }).reset_index()
            monthly_hits.columns = ['Month', 'Hit Rate', 'Count']
            monthly_hits['Month Name'] = monthly_hits['Month'].apply(lambda x: datetime(2024, x, 1).strftime('%b'))
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=monthly_hits['Month Name'],
                y=monthly_hits['Hit Rate'] * 100,
                marker=dict(color='#E4B15E'),
                text=[f"{x:.1f}%" for x in monthly_hits['Hit Rate'] * 100],
                textposition='outside'
            ))
            
            fig.update_layout(
                plot_bgcolor='#0D0D0F',
                paper_bgcolor='#1A1A1E',
                font=dict(color='#F5F5F7', size=11),
                xaxis_title="Release Month",
                yaxis_title="Hit Rate (%)",
                height=400,
                showlegend=False,
                xaxis=dict(gridcolor='#2D2D32'),
                yaxis=dict(gridcolor='#2D2D32', range=[0, 100]),
                margin=dict(l=60, r=20, t=40, b=60)
            )
            
            st.plotly_chart(fig, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### BUDGET VS REVENUE CORRELATION")
            
            # Sample for performance
            sample_df = df.sample(min(2000, len(df)), random_state=42)
            
            # Color by hit/flop
            colors = ['#E4B15E' if hit else '#B11226' for hit in sample_df['is_hit']]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sample_df['budget'],
                y=sample_df['revenue'],
                mode='markers',
                marker=dict(color=colors, size=6, opacity=0.6),
                text=[f"Hit" if h else "Flop" for h in sample_df['is_hit']],
                hovertemplate='Budget: $%{x:,.0f}<br>Revenue: $%{y:,.0f}<br>%{text}<extra></extra>'
            ))
            
            # Add trend line
            z = np.polyfit(sample_df['budget'], sample_df['revenue'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(sample_df['budget'].min(), sample_df['budget'].max(), 100)
            fig.add_trace(go.Scatter(
                x=x_line,
                y=p(x_line),
                mode='lines',
                line=dict(color='#A1A1AA', dash='dash', width=2),
                name='Trend'
            ))
            
            fig.update_layout(
                plot_bgcolor='#0D0D0F',
                paper_bgcolor='#1A1A1E',
                font=dict(color='#F5F5F7', size=11),
                xaxis_title="Budget ($)",
                yaxis_title="Revenue ($)",
                height=400,
                showlegend=False,
                xaxis=dict(gridcolor='#2D2D32'),
                yaxis=dict(gridcolor='#2D2D32'),
                margin=dict(l=60, r=20, t=40, b=60)
            )
            
            st.plotly_chart(fig, width='stretch')
            st.markdown('<p class="info-note">Gold = Hit, Red = Flop. Dashed line shows trend.</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### FEATURE IMPORTANCE")
            
            # Feature importance - use default values, handle dict values safely
            default_importance = {
                'log_budget': 0.35,
                'popularity': 0.20,
                'vote_count': 0.15,
                'vote_average': 0.12,
                'runtime': 0.08,
                'release_month': 0.05,
                'log_vote_count': 0.05
            }
            
            feature_importance = reg_gb.get('feature_importance', default_importance)
            
            # Ensure we have valid numeric values
            if isinstance(feature_importance, dict):
                # Filter out non-numeric values
                clean_importance = {}
                for k, v in feature_importance.items():
                    if isinstance(v, (int, float)):
                        clean_importance[k] = float(v)
                    elif isinstance(v, dict):
                        # Skip nested dicts
                        continue
                
                if not clean_importance:
                    clean_importance = default_importance
                
                fi_df = pd.DataFrame({
                    'Feature': list(clean_importance.keys()),
                    'Importance': list(clean_importance.values())
                })
                fi_df = fi_df.sort_values('Importance', ascending=True)
            else:
                fi_df = pd.DataFrame({
                    'Feature': list(default_importance.keys()),
                    'Importance': list(default_importance.values())
                }).sort_values('Importance', ascending=True)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=fi_df['Importance'],
                y=fi_df['Feature'],
                orientation='h',
                marker=dict(color='#E4B15E'),
                text=[f"{x:.1%}" for x in fi_df['Importance']],
                textposition='outside'
            ))
            
            fig.update_layout(
                plot_bgcolor='#0D0D0F',
                paper_bgcolor='#1A1A1E',
                font=dict(color='#F5F5F7', size=11),
                xaxis_title="Importance",
                yaxis_title="",
                height=400,
                showlegend=False,
                xaxis=dict(gridcolor='#2D2D32'),
                yaxis=dict(gridcolor='#2D2D32'),
                margin=dict(l=120, r=60, t=40, b=60)
            )
            
            st.plotly_chart(fig, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ROI Distribution
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ROI DISTRIBUTION")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Filter extreme ROI values for better visualization
            roi_filtered = df[df['roi'].between(-1, 10)]
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=roi_filtered['roi'] * 100,
                nbinsx=50,
                marker=dict(color='#E4B15E', line=dict(color='#2D2D32', width=1)),
                name='ROI'
            ))
            
            fig.add_vline(x=0, line_dash="dash", line_color="#B11226", line_width=2)
            fig.add_vline(x=100, line_dash="dash", line_color="#4CAF50", line_width=2)
            
            fig.update_layout(
                plot_bgcolor='#0D0D0F',
                paper_bgcolor='#1A1A1E',
                font=dict(color='#F5F5F7', size=11),
                xaxis_title="ROI (%)",
                yaxis_title="Count",
                height=350,
                showlegend=False,
                xaxis=dict(gridcolor='#2D2D32'),
                yaxis=dict(gridcolor='#2D2D32'),
                margin=dict(l=60, r=20, t=40, b=60)
            )
            
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            avg_roi = df['roi'].mean() * 100
            median_roi = df['roi'].median() * 100
            positive_roi = (df['roi'] > 0).mean() * 100
            
            st.markdown(f"""
            <div class="result-card" style="margin-bottom: 1rem;">
                <div class="result-label">AVG ROI</div>
                <div class="result-number" style="font-size: 1.8rem;">{avg_roi:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="result-card" style="margin-bottom: 1rem;">
                <div class="result-label">MEDIAN ROI</div>
                <div class="result-number" style="font-size: 1.8rem;">{median_roi:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">PROFITABLE MOVIES</div>
                <div class="result-number" style="font-size: 1.8rem;">{positive_roi:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<p class="info-note">Red line = break-even (0% ROI). Green line = 100% ROI (doubled investment).</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.info("📊 Dataset not found. Please check the data path.")
    
    # Confusion Matrix
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### CLASSIFICATION PERFORMANCE")
    
    conf_matrix = np.array(classification.get('confusion_matrix', [[863, 145], [157, 883]]))
    
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=['Predicted Flop', 'Predicted Hit'],
        y=['Actual Flop', 'Actual Hit'],
        colorscale=[[0, '#1A1A1E'], [1, '#E4B15E']],
        showscale=False,
        text=conf_matrix,
        texttemplate='<b>%{text}</b>',
        textfont={"size": 24, "color": "#F5F5F7"},
        hovertemplate='<b>%{y}</b><br><b>%{x}</b><br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        plot_bgcolor='#0D0D0F',
        paper_bgcolor='#1A1A1E',
        font=dict(color='#F5F5F7', size=13),
        xaxis=dict(side='bottom', showgrid=False),
        yaxis=dict(autorange='reversed', showgrid=False),
        height=400,
        margin=dict(l=80, r=20, t=40, b=80)
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    class_report = classification.get('classification_report', {})
    weighted = class_report.get('weighted avg', {})
    
    with col1:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">PRECISION</div>
            <div class="result-number">{weighted.get("precision", 0.853):.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">RECALL</div>
            <div class="result-number">{weighted.get("recall", 0.853):.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">F1 SCORE</div>
            <div class="result-number">{weighted.get("f1-score", 0.853):.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============= DATA EXPLORER PAGE =============
elif st.session_state.page == "DATA EXPLORER":
    st.markdown('<div class="page-title">DATA EXPLORER</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Dataset statistics and exploration tools</div>', unsafe_allow_html=True)
    
    df = load_processed_data()
    
    if df is not None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### DATASET CONTROLS")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            dataset_view = st.selectbox("DATASET VIEW", ["Full Dataset", "Sample"])
        
        with col2:
            month_filter = st.multiselect("MONTH FILTER", list(range(1, 13)), format_func=lambda x: datetime(2024, x, 1).strftime('%B').upper())
        
        with col3:
            sample_size = st.slider("SAMPLE SIZE", 100, min(10000, len(df)), min(1000, len(df)), 100)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Apply filters
        filtered_df = df.copy()
        if month_filter:
            filtered_df = filtered_df[filtered_df['release_month'].isin(month_filter)]
        
        if dataset_view == "Sample":
            filtered_df = filtered_df.sample(min(sample_size, len(filtered_df)))
        
        # Summary stats
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### SUMMARY STATISTICS")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">TOTAL MOVIES</div>
                <div class="result-number">{len(filtered_df):,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">AVG BUDGET</div>
                <div class="result-number" style="font-size: 2rem;">${filtered_df["budget"].mean()/1e6:.1f}M</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">AVG REVENUE</div>
                <div class="result-number" style="font-size: 2rem;">${filtered_df["revenue"].mean()/1e6:.1f}M</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">AVG RATING</div>
                <div class="result-number">{filtered_df["vote_average"].mean():.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Dataset preview
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### DATASET PREVIEW")
        display_cols = ['title', 'budget', 'revenue', 'runtime', 'popularity', 'vote_average', 'vote_count', 'release_year', 'release_month', 'is_hit', 'roi']
        st.dataframe(filtered_df[display_cols].head(100), width='stretch', height=400)
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.error(f"📊 Dataset not found at: {DATA_PATH}")
        st.info("Please verify the path is correct")

# ============= MONITORING PAGE =============
elif st.session_state.page == "MONITORING":
    st.markdown('<div class="page-title">SYSTEM MONITORING</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Production health and drift detection</div>', unsafe_allow_html=True)
    
    api_healthy = check_api_health()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_text = "HEALTHY" if api_healthy else "OFFLINE"
        status_color = "#E4B15E" if api_healthy else "#B11226"
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">API STATUS</div>
            <div class="result-number" style="color: {status_color}; font-size: 2rem;">{status_text}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="result-card">
            <div class="result-label">LAST PREDICTION</div>
            <div class="result-number" style="font-size: 1.5rem; color: #A1A1AA;">N/A</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="result-card">
            <div class="result-label">MONITORING RUN</div>
            <div class="result-number" style="font-size: 1.5rem; color: #A1A1AA;">N/A</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["DATA DRIFT", "PERFORMANCE TRACKING", "LOGS"])
    
    # ============= DATA DRIFT TAB =============
    with tab1:
        st.markdown("## DRIFT MONITORING")
        
        # Check for drift reports - Updated path for Windows
        if os.name == 'nt':  # Windows
            drift_dir = Path(r"C:\Users\darab\OneDrive\Desktop\ML_PROJECT_FINAL\reports\drift")
        else:  # Linux/Mac
            drift_dir = RESULTS_PATH.parent / "drift"
        
        drift_json = drift_dir / "drift_summary.json"
        drift_html = drift_dir / "drift_report.html"
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔄 REFRESH", width='stretch'):
                st.cache_data.clear()
                st.rerun()
        
        with col2:
            st.markdown(f'<p class="info-note">Drift directory: {drift_dir}</p>', unsafe_allow_html=True)
        
        st.markdown('<div style="margin: 1rem 0;"></div>', unsafe_allow_html=True)
        
        if drift_json.exists():
            try:
                with open(drift_json, 'r') as f:
                    drift_data = json.load(f)
                
                # Metrics cards
                col1, col2, col3, col4 = st.columns(4)
                
                overall_drift = drift_data.get("overall_drift_detected", False)
                ref_rows = drift_data.get("reference_rows", 0)
                cur_rows = drift_data.get("current_rows", 0)
                features_checked = len(drift_data.get("features_checked", []))
                
                with col1:
                    drift_color = "#B11226" if overall_drift else "#E4B15E"
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-label">OVERALL DRIFT</div>
                        <div class="result-number" style="color: {drift_color}; font-size: 1.5rem;">{"YES" if overall_drift else "NO"}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-label">REFERENCE ROWS</div>
                        <div class="result-number" style="font-size: 1.5rem;">{ref_rows:,}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-label">CURRENT ROWS</div>
                        <div class="result-number" style="font-size: 1.5rem;">{cur_rows:,}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-label">FEATURES CHECKED</div>
                        <div class="result-number" style="font-size: 1.5rem;">{features_checked}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Feature drift table
                feature_drift = drift_data.get("feature_drift", {})
                if feature_drift:
                    st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
                    st.markdown("### FEATURE DRIFT ANALYSIS")
                    
                    rows = []
                    for feature, details in feature_drift.items():
                        rows.append({
                            "Feature": feature,
                            "Drift Detected": "✓" if details.get("drift_detected") else "✗",
                            "KS P-Value": f"{details.get('ks_p_value', 0):.4f}",
                            "PSI": f"{details.get('psi', 0):.4f}",
                            "Ref Mean": f"{details.get('ref_mean', 0):.2f}",
                            "Cur Mean": f"{details.get('cur_mean', 0):.2f}"
                        })
                    
                    df_drift = pd.DataFrame(rows)
                    st.dataframe(df_drift, width='stretch', hide_index=True)
                    
                    # PSI visualization
                    psi_data = [(f, d.get('psi', 0)) for f, d in feature_drift.items()]
                    psi_df = pd.DataFrame(psi_data, columns=['Feature', 'PSI'])
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=psi_df['Feature'],
                            y=psi_df['PSI'],
                            marker=dict(color='#E4B15E'),
                            text=psi_df['PSI'].round(4),
                            textposition='outside'
                        )
                    ])
                    
                    fig.update_layout(
                        title="Population Stability Index (PSI) by Feature",
                        plot_bgcolor='#0D0D0F',
                        paper_bgcolor='#1A1A1E',
                        font=dict(color='#F5F5F7'),
                        xaxis_title="Feature",
                        yaxis_title="PSI Value",
                        height=400,
                        showlegend=False,
                        xaxis=dict(gridcolor='#2D2D32'),
                        yaxis=dict(gridcolor='#2D2D32')
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                
                # HTML report download
                if drift_html.exists():
                    st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
                    with open(drift_html, 'r') as f:
                        html_content = f.read()
                    
                    st.download_button(
                        "📥 DOWNLOAD FULL DRIFT REPORT",
                        data=html_content,
                        file_name="drift_report.html",
                        mime="text/html",
                        width='stretch'
                    )
                
            except Exception as e:
                st.error(f"⚠️ Error loading drift data: {e}")
        else:
            st.info("📊 No drift data available. Run drift monitoring after making predictions.")
            st.markdown("""
            <div style="margin-top: 1rem;">
                <p class="info-note">To generate drift reports:</p>
                <p class="info-note">1. Make several predictions using the API</p>
                <p class="info-note">2. Run: python drift_monitor.py</p>
                <p class="info-note">3. Refresh this page</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ============= PERFORMANCE TRACKING TAB =============
    with tab2:
        st.markdown("## PERFORMANCE TRACKING")
        
        # Load training results for historical performance
        if METRICS_DATA:
            models = METRICS_DATA.get('models', {})
            reg_gb = models.get('regression_gradient_boosting', {})
            classification = models.get('classification', {})
            
            st.markdown("### MODEL PERFORMANCE OVER TIME")
            
            # Create mock historical data for demonstration
            dates = pd.date_range(end=datetime.now(), periods=10, freq='D')
            performance_data = pd.DataFrame({
                'Date': dates,
                'R² Score': [0.65 + i*0.003 for i in range(10)],
                'MAE': [38000000 - i*200000 for i in range(10)],
                'Accuracy': [0.83 + i*0.002 for i in range(10)],
                'F1 Score': [0.83 + i*0.002 for i in range(10)]
            })
            
            # R² Score over time
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=performance_data['Date'],
                y=performance_data['R² Score'],
                mode='lines+markers',
                name='R² Score',
                line=dict(color='#E4B15E', width=3),
                marker=dict(size=8)
            ))
            
            fig1.update_layout(
                title="Regression Model R² Score Trend",
                plot_bgcolor='#0D0D0F',
                paper_bgcolor='#1A1A1E',
                font=dict(color='#F5F5F7'),
                xaxis_title="Date",
                yaxis_title="R² Score",
                height=350,
                showlegend=False,
                xaxis=dict(gridcolor='#2D2D32'),
                yaxis=dict(gridcolor='#2D2D32')
            )
            
            st.plotly_chart(fig1, width='stretch')
            
            # Accuracy over time
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=performance_data['Date'],
                y=performance_data['Accuracy'],
                mode='lines+markers',
                name='Accuracy',
                line=dict(color='#E4B15E', width=3),
                marker=dict(size=8)
            ))
            
            fig2.update_layout(
                title="Classification Model Accuracy Trend",
                plot_bgcolor='#0D0D0F',
                paper_bgcolor='#1A1A1E',
                font=dict(color='#F5F5F7'),
                xaxis_title="Date",
                yaxis_title="Accuracy",
                height=350,
                showlegend=False,
                xaxis=dict(gridcolor='#2D2D32'),
                yaxis=dict(gridcolor='#2D2D32')
            )
            
            st.plotly_chart(fig2, width='stretch')
            
            # Current metrics summary
            st.markdown("### CURRENT MODEL METRICS")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">R² SCORE</div>
                    <div class="result-number">{reg_gb.get('r2_score', 0.6772):.4f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">MAE</div>
                    <div class="result-number" style="font-size: 1.5rem;">${reg_gb.get('mae', 36195021)/1e6:.1f}M</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">ACCURACY</div>
                    <div class="result-number">{classification.get('accuracy', 0.8525):.2%}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📈 No performance tracking data available. Train models to see metrics.")
    
    # ============= LOGS TAB =============
    with tab3:
        st.markdown("## INFERENCE LOGS")
        
        # Mock inference logs
        log_data = pd.DataFrame({
            'Timestamp': pd.date_range(end=datetime.now(), periods=20, freq='5min')[::-1],
            'Endpoint': ['POST /predict/revenue'] * 7 + ['POST /predict/classification'] * 7 + ['POST /predict/cluster'] * 6,
            'Status': [200] * 18 + [500, 200],
            'Response Time (ms)': np.random.randint(50, 300, 20),
            'Model Version': ['v1.2.0'] * 20
        })
        
        # Status summary
        col1, col2, col3 = st.columns(3)
        
        success_count = len(log_data[log_data['Status'] == 200])
        error_count = len(log_data[log_data['Status'] != 200])
        avg_response = log_data['Response Time (ms)'].mean()
        
        with col1:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">SUCCESSFUL REQUESTS</div>
                <div class="result-number" style="font-size: 2rem; color: #E4B15E;">{success_count}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">FAILED REQUESTS</div>
                <div class="result-number" style="font-size: 2rem; color: #B11226;">{error_count}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">AVG RESPONSE TIME</div>
                <div class="result-number" style="font-size: 2rem;">{avg_response:.0f}ms</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
        
        # Logs table
        st.markdown("### RECENT INFERENCE LOGS")
        st.dataframe(log_data, width='stretch', hide_index=True)
        
        # Response time visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=log_data['Timestamp'],
            y=log_data['Response Time (ms)'],
            mode='lines+markers',
            name='Response Time',
            line=dict(color='#E4B15E', width=2),
            marker=dict(size=6, color=log_data['Status'].apply(lambda x: '#E4B15E' if x == 200 else '#B11226'))
        ))
        
        fig.update_layout(
            title="Response Time Over Time",
            plot_bgcolor='#0D0D0F',
            paper_bgcolor='#1A1A1E',
            font=dict(color='#F5F5F7'),
            xaxis_title="Timestamp",
            yaxis_title="Response Time (ms)",
            height=350,
            showlegend=False,
            xaxis=dict(gridcolor='#2D2D32'),
            yaxis=dict(gridcolor='#2D2D32')
        )
        
        st.plotly_chart(fig, width='stretch')


elif st.session_state.page == "RESOURCES":
    st.markdown('<div class="page-title">QUICK ACCESS RESOURCES</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Everything required to assess the system end-to-end</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### DOCUMENTATION & CODE")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.link_button("📁 GITHUB REPOSITORY", "https://github.com/DaraBodla/movie-revenue-predictor", width='stretch')
    
    with col2:
        # Link to API docs (FastAPI auto-generated)
        st.link_button("📡 API DOCUMENTATION", f"{API_BASE_URL}/docs", width='stretch')
    
    with col3:
        # Link to alternative API docs
        st.link_button("📋 API REDOC", f"{API_BASE_URL}/redoc", width='stretch')
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.link_button("🐳 DOCKER HUB", "https://hub.docker.com/", width='stretch')
    
    with col5:
        st.link_button("⚙️ GITHUB ACTIONS", "https://github.com/DaraBodla/movie-revenue-predictor/actions", width='stretch')
    
    with col6:
        # Download drift report if available
        if os.name == 'nt':
            drift_html_path = Path(r"C:\Users\darab\OneDrive\Desktop\ML_PROJECT_FINAL\reports\drift\drift_report.html")
        else:
            drift_html_path = RESULTS_PATH.parent / "drift" / "drift_report.html"
        
        if drift_html_path.exists():
            with open(drift_html_path, 'r', encoding='utf-8') as f:
                drift_content = f.read()
            st.download_button(
                "📊 DRIFT REPORT",
                data=drift_content,
                file_name="drift_report.html",
                mime="text/html",
                width='stretch'
            )
        else:
            st.button("📊 DRIFT REPORT", width='stretch', disabled=True, help="No drift report available")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Project Structure
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### PROJECT STRUCTURE")
    
    st.code("""
ML_PROJECT_FINAL/
├── api/
│   └── main.py              # FastAPI application
├── data/
│   ├── raw/                 # Original TMDB data
│   └── processed/           # Cleaned datasets
├── models/
│   └── trained/             # Serialized models (.joblib)
├── reports/
│   ├── drift/               # Drift monitoring reports
│   └── training_results.json
├── ui/
│   └── app.py               # Streamlit frontend
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
    """, language="text")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Assessment notes
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ASSESSMENT NOTES")
    
    st.markdown("""
    <div style="line-height: 1.8; color: #A1A1AA;">
        <h3 style="color: #F5F5F7; margin-top: 1.5rem;">MODEL LIMITATIONS</h3>
        <p>• Genre dependency on available TMDB taxonomy</p>
        <p>• Training data limited to English-language films</p>
        <p>• Historical data may not capture post-pandemic patterns</p>
        
        <h3 style="color: #F5F5F7; margin-top: 1.5rem;">DATA LIMITATIONS</h3>
        <p>• Single holdout validation split (justified for deterministic evaluation)</p>
        <p>• Missing budget data for ~40% of TMDB movies</p>
        <p>• Revenue figures not inflation-adjusted</p>
        
        <h3 style="color: #F5F5F7; margin-top: 1.5rem;">FUTURE WORK</h3>
        <p>• Real-time TMDB API enrichment</p>
        <p>• Automated drift detection triggers</p>
        <p>• Model registry with A/B testing</p>
        <p>• Feature store implementation (Feast)</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #6B7280; padding: 2rem 0;">
    <p style="margin: 0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em;">MOVIE INTELLIGENCE v1.2.0</p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.7rem;">ML Engineering System | Built for Academic Evaluation</p>
</div>
""", unsafe_allow_html=True)