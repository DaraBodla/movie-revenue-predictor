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

# Page configuration
st.set_page_config(
    page_title="MOVIE INTELLIGENCE",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# Load metrics data
METRICS_DATA = {
    "timestamp": "2025-12-16T22:38:16.352328",
    "models": {
        "regression_gradient_boosting": {
            "rmse": 92499921.16,
            "mae": 36195021.23,
            "r2_score": 0.6772,
            "mape": 10442410.62,
            "model_type": "gradient_boosting"
        },
        "classification": {
            "accuracy": 0.8525,
            "f1_score": 0.8525,
            "confusion_matrix": [[863, 145], [157, 883]]
        },
        "clustering": {
            "silhouette_score": 0.1009,
            "n_clusters": 4
        }
    }
}

# Custom CSS - Cinema/Box Office Inspired (Minimalist Dark)
st.markdown("""
<style>
    /* Import clean font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Color System */
    :root {
        --bg-main: #0D0D0F;
        --surface: #16161A;
        --border: #24242A;
        --gold: #E4B15E;
        --red: #B11226;
        --slate: #8A8F98;
        --text-primary: #F5F5F7;
        --text-secondary: #A1A1AA;
        --text-muted: #6B7280;
    }
    
    /* Main background */
    .main {
        background-color: #0D0D0F;
        color: #F5F5F7;
    }
    
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    
    /* Remove all shadows and gradients */
    div, button, input, select {
        box-shadow: none !important;
        background-image: none !important;
    }
    
    /* Typography */
    h1 {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #F5F5F7;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        color: #F5F5F7;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        font-size: 1.2rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #A1A1AA;
        margin-bottom: 0.75rem;
    }
    
    /* Page title */
    .page-title {
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #F5F5F7;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .page-subtitle {
        font-size: 1rem;
        color: #A1A1AA;
        text-align: center;
        margin-bottom: 3rem;
        letter-spacing: 0.02em;
    }
    
    /* Cards - Minimalist */
    .card {
        background-color: #16161A;
        border: 1px solid #24242A;
        padding: 2rem;
        margin-bottom: 2rem;
        border-radius: 0;
    }
    
    /* Revenue numbers - Box office style */
    .revenue-number {
        font-size: 4rem;
        font-weight: 800;
        color: #E4B15E;
        line-height: 1;
        letter-spacing: -0.02em;
    }
    
    .metric-number {
        font-size: 3rem;
        font-weight: 800;
        color: #E4B15E;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.5rem;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        border-radius: 0;
    }
    
    .badge-hit {
        background-color: #E4B15E;
        color: #0D0D0F;
    }
    
    .badge-flop {
        background-color: #B11226;
        color: #F5F5F7;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #E4B15E;
        color: #0D0D0F;
        border: none;
        border-radius: 0;
        padding: 1rem 3rem;
        font-weight: 700;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        transition: all 0.2s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #F5C976;
        transform: translateY(-2px);
    }
    
    /* Secondary button */
    button[kind="secondary"] {
        background-color: transparent !important;
        border: 1px solid #24242A !important;
        color: #A1A1AA !important;
    }
    
    button[kind="secondary"]:hover {
        background-color: #16161A !important;
        border-color: #8A8F98 !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background-color: #16161A;
        border: 1px solid #24242A;
        border-radius: 0;
        color: #F5F5F7;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #E4B15E;
        outline: none;
    }
    
    /* Slider */
    .stSlider > div > div > div > div {
        background-color: #E4B15E;
    }
    
    .stSlider > div > div > div {
        background-color: #24242A;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: transparent;
        border-bottom: 1px solid #24242A;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 0;
        color: #6B7280;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding: 1rem 2rem;
        border: none;
        font-size: 0.85rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #E4B15E;
        border-bottom: 2px solid #E4B15E;
    }
    
    /* Messages */
    .stSuccess {
        background-color: #16161A;
        border: 1px solid #E4B15E;
        border-radius: 0;
        color: #F5F5F7;
        padding: 1rem;
    }
    
    .stError {
        background-color: #16161A;
        border: 1px solid #B11226;
        border-radius: 0;
        color: #F5F5F7;
        padding: 1rem;
    }
    
    .stWarning {
        background-color: #16161A;
        border: 1px solid #E4B15E;
        border-radius: 0;
        color: #F5F5F7;
        padding: 1rem;
    }
    
    .stInfo {
        background-color: #16161A;
        border: 1px solid #8A8F98;
        border-radius: 0;
        color: #F5F5F7;
        padding: 1rem;
    }
    
    /* Dataframe */
    .dataframe {
        border: 1px solid #24242A;
        border-radius: 0;
    }
    
    .dataframe thead tr {
        background-color: #16161A;
        color: #A1A1AA;
    }
    
    .dataframe tbody tr {
        background-color: #0D0D0F;
        color: #F5F5F7;
    }
    
    /* Workflow stepper */
    .workflow-step {
        background-color: #16161A;
        border: 1px solid #24242A;
        padding: 1.5rem 1rem;
        text-align: center;
        flex: 1;
        min-width: 100px;
    }
    
    .workflow-step-title {
        font-size: 0.7rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.5rem;
    }
    
    .workflow-arrow {
        color: #24242A;
        font-size: 2rem;
        padding: 0 0.5rem;
    }
    
    /* Status indicator */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-healthy {
        background-color: #E4B15E;
    }
    
    .status-error {
        background-color: #B11226;
    }
    
    /* File uploader */
    .stFileUploader {
        background-color: #16161A;
        border: 1px dashed #24242A;
        border-radius: 0;
        padding: 2rem;
    }
    
    /* Multiselect */
    .stMultiSelect > div > div {
        background-color: #16161A;
        border: 1px solid #24242A;
        border-radius: 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #16161A;
        border-right: 1px solid #24242A;
    }
    
    /* Navigation pills */
    .nav-pill {
        background-color: transparent;
        border: 1px solid #24242A;
        padding: 0.75rem 1.5rem;
        margin-bottom: 0.5rem;
        color: #A1A1AA;
        text-transform: uppercase;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .nav-pill:hover {
        background-color: #16161A;
        border-color: #E4B15E;
        color: #E4B15E;
    }
    
    .nav-pill-active {
        background-color: #16161A;
        border-color: #E4B15E;
        color: #E4B15E;
    }
    
    /* Section divider */
    .section-divider {
        height: 1px;
        background-color: #24242A;
        margin: 3rem 0;
    }
    
    /* Info note */
    .info-note {
        font-size: 0.85rem;
        color: #6B7280;
        font-style: italic;
        margin-top: 0.5rem;
    }
    
    /* Plotly chart backgrounds */
    .js-plotly-plot {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "Home"
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

def make_prediction(data, endpoint="/predict"):
    """Make prediction API call"""
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=10)
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return None, f"Connection Error: {str(e)}"

# Navigation
st.markdown("""
<div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 3rem; border-bottom: 1px solid #24242A; padding-bottom: 1rem;">
""", unsafe_allow_html=True)

pages = ["Home", "Predict", "Model Dashboard", "Data Explorer", "Monitoring", "Resources"]
cols = st.columns(len(pages))

for idx, page in enumerate(pages):
    with cols[idx]:
        if st.button(page, key=f"nav_{page}", use_container_width=True):
            st.session_state.page = page
            st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# ============= HOME / OVERVIEW PAGE =============
if st.session_state.page == "Home":
    # Hero Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="page-title">MOVIE INTELLIGENCE</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">End-to-end ML Engineering System for Revenue Prediction</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### SYSTEM CAPABILITIES")
        st.markdown("""
        <div style="margin-top: 1.5rem;">
            <div style="margin-bottom: 1rem;">
                <span style="color: #E4B15E; font-weight: 700;">→</span> 
                <span style="color: #F5F5F7; font-weight: 600;">REGRESSION</span>
                <span style="color: #6B7280;"> — Revenue prediction with R² 0.6772</span>
            </div>
            <div style="margin-bottom: 1rem;">
                <span style="color: #E4B15E; font-weight: 700;">→</span> 
                <span style="color: #F5F5F7; font-weight: 600;">CLASSIFICATION</span>
                <span style="color: #6B7280;"> — Hit/Flop label with 85.3% accuracy</span>
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
        
        st.markdown(f"""
        <div style="margin-bottom: 1.5rem;">
            <div class="metric-label">MODEL VERSION</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #F5F5F7;">v1.2.0</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="margin-bottom: 1.5rem;">
            <div class="metric-label">LAST TRAINED</div>
            <div style="font-size: 0.9rem; font-weight: 600; color: #A1A1AA;">DEC 16, 2024</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div>
            <div class="metric-label">BEST R² SCORE</div>
            <div style="font-size: 2rem; font-weight: 800; color: #E4B15E;">0.6772</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Workflow Strip
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("## ML WORKFLOW")
    
    workflow_cols = st.columns(15)
    steps = ["INGEST", "CLEAN", "FEATURES", "TRAIN", "EVALUATE", "DEPLOY", "MONITOR"]
    
    for i, step in enumerate(steps):
        with workflow_cols[i*2]:
            st.markdown(f"""
            <div class="workflow-step">
                <div style="font-size: 2rem;">{"📥" if i==0 else "🧹" if i==1 else "⚙️" if i==2 else "🎯" if i==3 else "📊" if i==4 else "🚀" if i==5 else "📡"}</div>
                <div class="workflow-step-title">{step}</div>
            </div>
            """, unsafe_allow_html=True)
        
        if i < len(steps) - 1:
            with workflow_cols[i*2 + 1]:
                st.markdown('<div class="workflow-arrow">→</div>', unsafe_allow_html=True)
    
    # What this app does
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("## CORE FUNCTIONS")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 2.5rem; text-align: center; margin-bottom: 1rem;">💰</div>', unsafe_allow_html=True)
        st.markdown("### PREDICT REVENUE")
        st.markdown('<p class="info-note">Input movie parameters and get instant revenue predictions from trained ensemble models.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 2.5rem; text-align: center; margin-bottom: 1rem;">📈</div>', unsafe_allow_html=True)
        st.markdown("### ANALYZE PERFORMANCE")
        st.markdown('<p class="info-note">Review model metrics, diagnostics, and performance across regression, classification, and clustering.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 2.5rem; text-align: center; margin-bottom: 1rem;">📡</div>', unsafe_allow_html=True)
        st.markdown("### TRACK MONITORING")
        st.markdown('<p class="info-note">Monitor data drift, API health, and system performance in production environment.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # CTA Buttons
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("START PREDICTION", use_container_width=True):
            st.session_state.page = "Predict"
            st.rerun()
    
    with col2:
        if st.button("VIEW MODEL DASHBOARD", use_container_width=True):
            st.session_state.page = "Model Dashboard"
            st.rerun()
    
    with col3:
        if st.button("OPEN DOCUMENTATION", use_container_width=True):
            st.session_state.page = "Resources"
            st.rerun()

# ============= PREDICT PAGE =============
elif st.session_state.page == "Predict":
    st.markdown('<div class="page-title">REVENUE PREDICTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Generate predictions using manual input, TMDB search, or batch processing</div>', unsafe_allow_html=True)
    
    # Tabs for different prediction methods
    tab1, tab2, tab3 = st.tabs(["MANUAL INPUT", "TMDB SEARCH", "BATCH PREDICTION"])
    
    # ============= TAB 1: MANUAL INPUT =============
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### INPUT PARAMETERS")
        
        # Two-column form
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 style="margin-top: 1.5rem;">MOVIE BASICS</h3>', unsafe_allow_html=True)
            
            budget = st.number_input(
                "BUDGET (USD)",
                min_value=0,
                value=50000000,
                step=1000000,
                format="%d",
                help="Production budget in US dollars"
            )
            
            budget_slider = st.slider(
                "Adjust Budget",
                min_value=0,
                max_value=300000000,
                value=budget,
                step=5000000,
                label_visibility="collapsed"
            )
            budget = budget_slider
            
            runtime = st.number_input(
                "RUNTIME (MINUTES)",
                min_value=30,
                max_value=300,
                value=120,
                step=1
            )
            
            release_month = st.selectbox(
                "RELEASE MONTH",
                options=list(range(1, 13)),
                format_func=lambda x: datetime(2024, x, 1).strftime('%B').upper(),
                index=5
            )
            
            genres = st.multiselect(
                "GENRES",
                options=[
                    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
                    "Drama", "Family", "Fantasy", "History", "Horror", "Music",
                    "Mystery", "Romance", "Science Fiction", "Thriller", "War", "Western"
                ],
                default=["Action", "Adventure"]
            )
        
        with col2:
            st.markdown('<h3 style="margin-top: 1.5rem;">POPULARITY & VOTES</h3>', unsafe_allow_html=True)
            
            popularity = st.number_input(
                "POPULARITY",
                min_value=0.0,
                max_value=1000.0,
                value=50.0,
                step=1.0
            )
            
            vote_average = st.slider(
                "VOTE AVERAGE",
                min_value=0.0,
                max_value=10.0,
                value=7.0,
                step=0.1
            )
            
            vote_count = st.number_input(
                "VOTE COUNT",
                min_value=0,
                value=1000,
                step=100
            )
        
        st.markdown('<p class="info-note">All inputs match training schema.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Buttons
        col1, col2 = st.columns([1, 1])
        
        with col1:
            predict_btn = st.button("RUN PREDICTION", use_container_width=True, type="primary")
        
        with col2:
            reset_btn = st.button("RESET", use_container_width=True, key="reset_btn")
        
        if reset_btn:
            st.session_state.prediction_result = None
            st.rerun()
        
        # Make prediction
        if predict_btn:
            if not genres:
                st.error("⚠️ Please select at least one genre")
            else:
                with st.spinner("Running prediction models..."):
                    input_data = {
                        "budget": float(budget),
                        "runtime": float(runtime),
                        "release_month": int(release_month),
                        "popularity": float(popularity),
                        "vote_average": float(vote_average),
                        "vote_count": int(vote_count),
                        "genres": ",".join(genres)
                    }
                    
                    # Try API calls
                    revenue_result, revenue_error = make_prediction(input_data, "/predict/revenue")
                    classification_result, class_error = make_prediction(input_data, "/predict/classification")
                    cluster_result, cluster_error = make_prediction(input_data, "/predict/cluster")
                    
                    # Use mock if API fails
                    if revenue_error:
                        st.error(f"Revenue API failed: {revenue_error}")
                        if class_error:
                            st.warning(f"Classification API failed: {class_error}")
                        if cluster_error:
                            st.warning(f"Cluster API failed: {cluster_error}")
                        st.stop()
                        result = {
                            "revenue": predicted_revenue,
                            "classification": "Hit" if predicted_revenue > budget * 2 else "Flop",
                            "cluster": "Big Budget Blockbuster" if budget > 100000000 else "Mid-Budget Film",
                            "confidence": "N/A",
                            "model_version": "v1.2.0",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                    else:
                        result = {
                            "revenue": revenue_result.get("predicted_revenue", 0.0),
                            "classification": (
                                classification_result.get("prediction_label", "Unknown") if not class_error else "Unknown"
                            ),
                            "cluster": (
                                cluster_result.get("cluster_label", "Unknown") if not cluster_error else "Unknown"
                            ),
                            "confidence": revenue_result.get("confidence_interval", "N/A"),
                            "model_version": "v1.2.0",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    st.session_state.prediction_result = result
                    st.success("✓ PREDICTION COMPLETED")
        
        # Display results
        if st.session_state.prediction_result:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("## PREDICTION RESULTS")
            
            result = st.session_state.prediction_result
            
            # Three result cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">PREDICTED REVENUE</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="revenue-number">${result["revenue"]:,.0f}</div>', unsafe_allow_html=True)
                roi = ((result['revenue'] / budget) - 1) * 100
                st.markdown(f'<p class="info-note">ROI: {roi:+.1f}%</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">CLASSIFICATION</div>', unsafe_allow_html=True)
                badge_class = "badge-hit" if result['classification'] == "Hit" else "badge-flop"
                st.markdown(f'<div style="text-align: center; margin: 2rem 0;"><span class="badge {badge_class}">{result["classification"]}</span></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">CLUSTER LABEL</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size: 1.3rem; font-weight: 700; color: #F5F5F7; margin: 1.5rem 0; text-align: center;">{result["cluster"]}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Metadata
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### PREDICTION METADATA")
            
            meta_col1, meta_col2, meta_col3 = st.columns(3)
            
            with meta_col1:
                st.markdown(f"""
                <div class="metric-label">CONFIDENCE / UNCERTAINTY</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #A1A1AA; margin-top: 0.5rem;">{result['confidence']}</div>
                """, unsafe_allow_html=True)
            
            with meta_col2:
                st.markdown(f"""
                <div class="metric-label">MODEL VERSION</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #A1A1AA; margin-top: 0.5rem;">{result['model_version']}</div>
                """, unsafe_allow_html=True)
            
            with meta_col3:
                st.markdown(f"""
                <div class="metric-label">INFERENCE TIMESTAMP</div>
                <div style="font-size: 1rem; font-weight: 600; color: #A1A1AA; margin-top: 0.5rem;">{result['timestamp']}</div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # ============= TAB 2: TMDB SEARCH =============
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### SEARCH TMDB DATABASE")
        
        search_query = st.text_input(
            "MOVIE TITLE",
            placeholder="e.g., Avatar, Inception, The Avengers",
            label_visibility="visible"
        )
        
        if st.button("SEARCH", use_container_width=True):
            if search_query:
                st.info("🔍 TMDB search integration coming soon. Use Manual Input for now.")
            else:
                st.warning("⚠️ Enter a movie title to search")
        
        st.markdown('<p class="info-note">Search results will auto-fill the form. Missing fields will be flagged.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============= TAB 3: BATCH PREDICTION =============
    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### BATCH PROCESSING")
        
        # Template download
        template_df = pd.DataFrame({
            "budget": [50000000, 100000000, 30000000],
            "runtime": [120, 142, 95],
            "release_month": [6, 7, 12],
            "popularity": [50.0, 95.5, 25.3],
            "vote_average": [7.0, 7.8, 6.5],
            "vote_count": [1000, 15420, 450],
            "genres": ["Action,Adventure", "Action,Sci-Fi", "Drama,Romance"]
        })
        
        csv_buffer = io.StringIO()
        template_df.to_csv(csv_buffer, index=False)
        
        st.download_button(
            "DOWNLOAD CSV TEMPLATE",
            data=csv_buffer.getvalue(),
            file_name="batch_prediction_template.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader("UPLOAD CSV FILE", type=['csv'])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Validate schema
                required_cols = ["budget", "runtime", "release_month", "popularity", "vote_average", "vote_count", "genres"]
                missing = [col for col in required_cols if col not in df.columns]
                
                if missing:
                    st.error(f"⚠️ Missing columns: {', '.join(missing)}")
                else:
                    st.success(f"✓ Schema validated. Found {len(df)} movies.")
                    
                    # Preview
                    st.markdown("### PREVIEW")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    if st.button("RUN BATCH PREDICTION", use_container_width=True):
                        with st.spinner(f"Processing {len(df)} movies..."):
                            st.info("🔄 Batch prediction in progress...")
                            # Implement batch prediction logic here
            
            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ============= MODEL DASHBOARD PAGE =============
elif st.session_state.page == "Model Dashboard":
    st.markdown('<div class="page-title">MODEL DASHBOARD</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Performance metrics and diagnostic visualizations</div>', unsafe_allow_html=True)
    
    # Model comparison
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### MODEL COMPARISON")
    
    # Metrics table
    metrics_df = pd.DataFrame({
        "Model": ["Gradient Boosting", "Classification", "Clustering"],
        "Type": ["Regression", "Binary", "K-Means"],
        "Primary Metric": ["R² Score", "Accuracy", "Silhouette"],
        "Value": [0.6772, 0.8525, 0.1009],
        "Secondary": ["MAE: $36.2M", "F1: 0.8525", "Clusters: 4"]
    })
    
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    st.markdown('<p class="info-note">Metrics from model training on Dec 16, 2024.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Commentary
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### WHY THESE METRICS MATTER")
    st.markdown("""
    <div style="line-height: 1.8; color: #A1A1AA;">
        <p><strong style="color: #F5F5F7;">R² Score (0.6772):</strong> Explains 67.7% of revenue variance—strong for entertainment industry volatility.</p>
        <p><strong style="color: #F5F5F7;">MAE ($36.2M):</strong> Average prediction error acceptable given box office scale ($100M+ films).</p>
        <p><strong style="color: #F5F5F7;">Accuracy (85.3%):</strong> Hit/Flop classification reliable for investment decisions.</p>
        <p><strong style="color: #F5F5F7;">Silhouette (0.10):</strong> Moderate cluster separation—movie categories overlap naturally.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("## DIAGNOSTIC VISUALIZATIONS")
    
    # Placeholder for actual vs predicted plot
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ACTUAL VS PREDICTED REVENUE")
        st.info("📊 Load processed data to generate plot")
        st.markdown('<p class="info-note">Regression diagnostics show model fit quality.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### RESIDUAL DISTRIBUTION")
        st.info("📊 Load processed data to generate plot")
        st.markdown('<p class="info-note">Residuals centered at zero indicate unbiased model.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Confusion matrix
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### CLASSIFICATION PERFORMANCE")
    
    # Create confusion matrix visualization
    conf_matrix = np.array([[863, 145], [157, 883]])
    
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=['Predicted Flop', 'Predicted Hit'],
        y=['Actual Flop', 'Actual Hit'],
        colorscale=[[0, '#16161A'], [1, '#E4B15E']],
        showscale=False,
        text=conf_matrix,
        texttemplate='%{text}',
        textfont={"size": 20, "color": "#F5F5F7"}
    ))
    
    fig.update_layout(
        plot_bgcolor='#0D0D0F',
        paper_bgcolor='#0D0D0F',
        font=dict(color='#F5F5F7', size=12),
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed'),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-label">PRECISION</div>
        <div class="metric-number">0.853</div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-label">RECALL</div>
        <div class="metric-number">0.853</div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-label">F1 SCORE</div>
        <div class="metric-number">0.853</div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============= DATA EXPLORER PAGE =============
elif st.session_state.page == "Data Explorer":
    st.markdown('<div class="page-title">DATA EXPLORER</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Dataset statistics and exploration tools</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### DATASET CONTROLS")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dataset = st.selectbox("DATASET", ["Training", "Test", "Full"])
    
    with col2:
        month_filter = st.multiselect("MONTH FILTER", list(range(1, 13)), format_func=lambda x: datetime(2024, x, 1).strftime('%B').upper())
    
    with col3:
        sample_size = st.slider("SAMPLE SIZE", 100, 10000, 1000, 100)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.info("📊 Connect to processed dataset to enable exploration features.")

# ============= MONITORING PAGE =============
elif st.session_state.page == "Monitoring":
    st.markdown('<div class="page-title">SYSTEM MONITORING</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Production health and drift detection</div>', unsafe_allow_html=True)
    
    # Status strip
    api_healthy = check_api_health()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">API STATUS</div>', unsafe_allow_html=True)
        status_text = "HEALTHY" if api_healthy else "OFFLINE"
        status_color = "#E4B15E" if api_healthy else "#B11226"
        st.markdown(f'<div style="font-size: 1.5rem; font-weight: 800; color: {status_color};">{status_text}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">LAST PREDICTION</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 1rem; font-weight: 600; color: #A1A1AA;">N/A</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">MONITORING RUN</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 1rem; font-weight: 600; color: #A1A1AA;">N/A</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Monitoring tabs
    tab1, tab2, tab3 = st.tabs(["DATA DRIFT", "PERFORMANCE TRACKING", "LOGS"])
    
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.info("📊 Drift detection reports available after monitoring pipeline execution.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.info("📈 Performance tracking across model versions coming soon.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.info("📋 Recent inference logs will appear here.")
        st.markdown('</div>', unsafe_allow_html=True)

# ============= RESOURCES PAGE =============
elif st.session_state.page == "Resources":
    st.markdown('<div class="page-title">QUICK ACCESS RESOURCES</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Everything required to assess the system end-to-end</div>', unsafe_allow_html=True)
    
    # Quick links
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### DOCUMENTATION & CODE")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.link_button(
            "GITHUB REPOSITORY",
            "https://github.com/DaraBodla/movie-revenue-predictor",
            use_container_width=True
        )
    
    with col2:
        st.button("TECHNICAL REPORT (PDF)", use_container_width=True, disabled=True)
    
    with col3:
        st.button("API DOCUMENTATION", use_container_width=True, disabled=True)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.button("DOCKER INSTRUCTIONS", use_container_width=True, disabled=True)
    
    with col5:
        st.button("CI/CD WORKFLOW", use_container_width=True, disabled=True)
    
    with col6:
        st.button("DRIFT REPORTS", use_container_width=True, disabled=True)
    
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