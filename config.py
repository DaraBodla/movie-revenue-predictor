"""
Configuration file for Movie Intelligence System
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "trained"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data files
RAW_DATA_PATH = RAW_DATA_DIR / "movies_clean.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "movies_processed.csv"

# Model paths
REGRESSION_MODEL_PATH = MODELS_DIR / "regression_model.joblib"
CLASSIFICATION_MODEL_PATH = MODELS_DIR / "classification_model.joblib"
CLUSTERING_MODEL_PATH = MODELS_DIR / "clustering_model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.joblib"
LABEL_ENCODERS_PATH = MODELS_DIR / "label_encoders.joblib"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature engineering
GENRE_TOP_N = 10  # Top N genres to include as features
MIN_BUDGET = 1000  # Minimum budget threshold

# Model hyperparameters
LINEAR_REGRESSION_PARAMS = {}

RANDOM_FOREST_REGRESSION_PARAMS = {
    'n_estimators': 100,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

GRADIENT_BOOSTING_REGRESSION_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 5,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE
}

RANDOM_FOREST_CLASSIFICATION_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'class_weight': 'balanced'
}

KMEANS_PARAMS = {
    'n_clusters': 4,
    'random_state': RANDOM_STATE,
    'n_init': 10
}

# API settings
API_TITLE = "Movie Release Performance Intelligence System"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
End-to-end MLOps platform for predicting movie performance.

Features:
- Revenue prediction (Regression)
- Hit/Flop classification
- Seasonal trend analysis
- Movie clustering
- Batch predictions
"""

# Monitoring
ENABLE_LOGGING = True
LOG_FILE = PROJECT_ROOT / "logs" / "app.log"
