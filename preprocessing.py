"""
Data preprocessing and feature engineering for movie dataset
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from typing import Tuple, Dict, List
import config


class MovieDataPreprocessor:
    """Handles all data preprocessing and feature engineering"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.top_genres = None
        # Learned during training (fit=True) so we can use in inference
        self.release_month_counts_ = None
        
    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """Load the movie dataset"""
        if filepath is None:
            filepath = config.RAW_DATA_PATH
        
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records from {filepath}")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare the dataset.

        Expected minimal columns for this repository's `movies_clean.csv`:
        budget, popularity, runtime, vote_average, vote_count, release_month, revenue
        """
        df = df.copy()

        required = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count', 'revenue']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}. "
                             f"Found columns: {list(df.columns)}")

        # Ensure release_month exists (either provided directly or derived from release_date)
        if 'release_month' not in df.columns:
            if 'release_date' in df.columns:
                # Derive month from ISO date string / datetime
                parsed = pd.to_datetime(df['release_date'], errors='coerce')
                df['release_month'] = parsed.dt.month
            else:
                raise ValueError("Dataset must include 'release_month' or 'release_date' so release month can be used as a feature.")
        
        # Basic missing-value handling (lightweight, deterministic)
        # Note: We keep this simple to avoid leaking information from target.
        numeric_impute_mean = ['runtime', 'vote_average']
        numeric_impute_median = ['budget', 'popularity', 'vote_count']
        for col in numeric_impute_mean:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].mean())
        for col in numeric_impute_median:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())

        # Outlier capping (IQR * 3) for stability
        def _cap_iqr(series: pd.Series) -> pd.Series:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                return series
            lower = q1 - 3 * iqr
            upper = q3 + 3 * iqr
            return series.clip(lower=lower, upper=upper)

        for col in ['budget', 'popularity', 'vote_count', 'runtime']:
            if col in df.columns:
                df[col] = _cap_iqr(df[col])

        # Basic validity filters
        df = df[df['budget'] >= config.MIN_BUDGET]
        df = df[df['revenue'] > 0]
        df = df[df['runtime'] > 0]
        df = df[df['vote_count'] >= 0]

        # Drop obvious duplicates / NaNs
        df = df.dropna(subset=required).drop_duplicates()

        return df
    def engineer_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Create engineered features (inference-safe).

        This project uses the cleaned numeric dataset `movies_clean.csv` which contains:
        budget, popularity, runtime, vote_average, vote_count, release_month, revenue (target).

        IMPORTANT:
        - Never use `revenue` to create input features for inference (target leakage).
        - Do not assume time/categorical columns exist (e.g., release_date, genres, language).
        """
        df = df.copy()

        # Basic numeric safety (avoid division by zero)
        df['budget_safe'] = df['budget'].clip(lower=1)
        df['runtime_safe'] = df['runtime'].clip(lower=1)
        df['vote_count_safe'] = df['vote_count'].clip(lower=0)

        # Release month (must exist by clean_data() contract)
        if 'release_month' not in df.columns:
            raise ValueError("Missing 'release_month' after cleaning.")
        df['release_month'] = pd.to_numeric(df['release_month'], errors='coerce')
        if df['release_month'].isna().any():
            raise ValueError("release_month contains non-numeric values that could not be parsed.")
        df['release_month'] = df['release_month'].astype(int)
        # clip to valid month range
        df['release_month'] = df['release_month'].clip(lower=1, upper=12)
        # cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['release_month'] / 12.0)
        df['month_cos'] = np.cos(2 * np.pi * df['release_month'] / 12.0)

        # Competition indicator: how many movies are in this release month (learned from training data)
        if fit:
            self.release_month_counts_ = df['release_month'].value_counts().to_dict()
        month_counts = self.release_month_counts_ or {}
        df['release_competition'] = df['release_month'].map(month_counts).fillna(0).astype(float)

        # Inference-safe engineered features (derived ONLY from predictors)
        df['log_budget'] = np.log1p(df['budget_safe'])
        df['log_vote_count'] = np.log1p(df['vote_count_safe'])
        df['popularity_score'] = df['popularity'] * df['vote_average']
        df['vote_density'] = df['vote_count_safe'] / df['runtime_safe']
        df['popularity_per_budget'] = df['popularity'] / df['budget_safe']

        # Additional high-impact features
        df['budget_per_minute'] = df['budget_safe'] / (df['runtime_safe'] + 1.0)
        df['hype_score'] = df['popularity'] * df['vote_count_safe']

        # Seasonal buckets (based on release_month)
        df['is_summer_release'] = df['release_month'].isin([5,6,7,8]).astype(int)
        df['is_holiday_release'] = df['release_month'].isin([11,12]).astype(int)

        # Optional text-derived feature if title exists (training data may include it)
        if 'title' in df.columns:
            df['is_sequel'] = df['title'].astype(str).str.contains(r'\d|II|III|IV', regex=True).astype(int)
        else:
            df['is_sequel'] = 0

        return df
    def _get_season(self, month: int) -> str:
        """Convert month to season"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def _create_genre_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Create one-hot encoded features for top genres"""
        # Extract all genres
        all_genres = []
        for genres_str in df['genres'].dropna():
            if isinstance(genres_str, str):
                genres = [g.strip() for g in genres_str.split(',')]
                all_genres.extend(genres)
        
        # Get top N genres
        if fit:
            genre_counts = pd.Series(all_genres).value_counts()
            self.top_genres = genre_counts.head(config.GENRE_TOP_N).index.tolist()
        
        # Create binary features for top genres
        genre_features = pd.DataFrame()
        for genre in self.top_genres:
            genre_features[f'genre_{genre.lower().replace(" ", "_")}'] = df['genres'].apply(
                lambda x: 1 if isinstance(x, str) and genre in x else 0
            )
        
        return genre_features
    
    def create_target_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create classification labels for 'hit' vs 'flop'.

        Since the cleaned dataset does not include a release year/month, we use a
        dataset-wide median revenue as the decision boundary.
        """
        df = df.copy()

        if 'revenue' not in df.columns:
            raise ValueError("Column 'revenue' is required to create labels.")

        median_revenue = df['revenue'].median()
        df['is_hit'] = (df['revenue'] > median_revenue).astype(int)

        # ROI is meaningful without leaking into features (kept for reporting/analysis only)
        df['roi'] = (df['revenue'] - df['budget']) / (df['budget'] + 1)

        return df
    def prepare_features(self, df: pd.DataFrame, target_col: str | None = 'revenue',
                        fit: bool = True) -> Tuple[pd.DataFrame, pd.Series | None]:
        """Prepare feature matrix X and (optionally) target y.

        - For training: pass target_col='revenue' (default) and fit=True.
        - For inference: pass target_col=None and fit=False.
        """
        df = df.copy()

        # Base predictors from the dataset
        base_features = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count', 'release_month']

        # Engineered features created by engineer_features()
        engineered = ['log_budget', 'log_vote_count', 'popularity_score', 'vote_density', 'popularity_per_budget', 'month_sin', 'month_cos']

        feature_cols = base_features + engineered

        missing_features = [c for c in feature_cols if c not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required feature columns after feature engineering: {missing_features}")

        X = df[feature_cols].copy()
        y = None
        if target_col is not None:
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in dataset.")
            y = df[target_col].copy()

        # Standardize features
        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.feature_names = feature_cols
        else:
            if self.scaler is None:
                raise ValueError("Scaler is not fitted. Train the model or load a fitted preprocessor first.")
            X_scaled = self.scaler.transform(X)

        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        return X_scaled, y
    def save_preprocessor(self):
        """Save preprocessing objects"""
        joblib.dump(self.scaler, config.SCALER_PATH)
        joblib.dump(self.label_encoders, config.LABEL_ENCODERS_PATH)
        joblib.dump({
            'feature_names': self.feature_names,
            'top_genres': self.top_genres
        }, config.FEATURE_NAMES_PATH)
        print(f"Preprocessor saved to {config.MODELS_DIR}")
    
    def load_preprocessor(self):
        """Load preprocessing objects"""
        self.scaler = joblib.load(config.SCALER_PATH)
        self.label_encoders = joblib.load(config.LABEL_ENCODERS_PATH)
        feature_data = joblib.load(config.FEATURE_NAMES_PATH)
        self.feature_names = feature_data['feature_names']
        self.top_genres = feature_data['top_genres']
        print("Preprocessor loaded successfully")
    
    def get_time_series_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a time-series friendly view of the data.

        The current cleaned dataset does not include temporal columns. If you want
        time-series analysis, you must provide a dataset with a real date/month field.
        """
        raise ValueError("Time-series analysis requires temporal columns (e.g., release_month/release_date), "
                         "but the current dataset does not include them.")



def prepare_train_test_split(X, y, test_size=None, random_state=None):
    """
    Split feature matrix X and target y into train/test.
    Keeps pandas indices intact (important for later steps that map back to df).
    """
    if test_size is None:
        test_size = getattr(config, "TEST_SIZE", 0.2)
    if random_state is None:
        random_state = getattr(config, "RANDOM_STATE", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test
