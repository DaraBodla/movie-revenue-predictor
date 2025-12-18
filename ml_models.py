"""
Machine Learning Models for Movie Intelligence System
Implements: Regression, Classification, Clustering, Time-Series, Dimensionality Reduction
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    classification_report, confusion_matrix, f1_score, accuracy_score,
    silhouette_score,roc_auc_score, precision_recall_curve

)
import joblib
from typing import Dict, Tuple, Any
import config


class MovieRevenuePredictor:
    """Regression models for revenue prediction"""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = self._init_model()
        self.metrics = {}
        
    def _init_model(self):
        """Initialize the regression model"""
        if self.model_type == 'linear':
            return LinearRegression(**config.LINEAR_REGRESSION_PARAMS)
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(**config.RANDOM_FOREST_REGRESSION_PARAMS)
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(**config.GRADIENT_BOOSTING_REGRESSION_PARAMS)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the regression model"""
        print(f"\nTraining {self.model_type} regression model...")
        self.model.fit(X_train, y_train)
        print("Training completed!")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100
        
        self.metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape,
            'model_type': self.model_type
        }
        
        print(f"\n{self.model_type.upper()} Regression Metrics:")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"MAE: ${mae:,.2f}")
        print(f"R² Score: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        return self.metrics
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """Get feature importance (for tree-based models)"""
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        return None
    
    def save(self, filepath: str = None):
        """Save the trained model"""
        if filepath is None:
            filepath = config.REGRESSION_MODEL_PATH
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str = None):
        """Load a trained model"""
        if filepath is None:
            filepath = config.REGRESSION_MODEL_PATH
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")


from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score
)

import config


class MovieClassifier:
    """Classification model for hit/flop prediction with tunable threshold"""

    def __init__(self, model_type: str = "xgboost", threshold: float = 0.5, calibrate: bool = False):
        self.model_type = model_type
        self.threshold = float(threshold)
        self.calibrate = bool(calibrate)

        self.model = None
        self.base_model = None  # kept for feature importance when wrapped
        self.metrics = {}

        self.model = self._init_model()

    def _init_model(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.calibration import CalibratedClassifierCV

        if self.model_type == "random_forest":
            self.base_model = RandomForestClassifier(**config.RANDOM_FOREST_CLASSIFICATION_PARAMS)
            return self.base_model

        if self.model_type == "random_forest_calibrated":
            self.base_model = RandomForestClassifier(**config.RANDOM_FOREST_CLASSIFICATION_PARAMS)
            return CalibratedClassifierCV(self.base_model, method="sigmoid", cv=3)

        if self.model_type in ("xgboost", "xgboost_calibrated"):
            try:
                from xgboost import XGBClassifier
            except Exception as e:
                raise ImportError(
                    "XGBoost is not installed. Run: pip install xgboost"
                ) from e

            # Reasonable defaults if you don't have config params
            xgb_params = getattr(config, "XGBOOST_CLASSIFICATION_PARAMS", None) or {
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.0,
                "random_state": getattr(config, "RANDOM_STATE", 42),
                "eval_metric": "logloss",
                "n_jobs": -1,
            }

            self.base_model = XGBClassifier(**xgb_params)

            if self.model_type == "xgboost_calibrated":
                return CalibratedClassifierCV(self.base_model, method="sigmoid", cv=3)

            return self.base_model

        raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise ValueError("Model does not support predict_proba")

    def predict(self, X):
        """
        IMPORTANT:
        - We DO NOT rely on model.predict() because that locks you to threshold=0.5.
        - We use tuned self.threshold on predicted probabilities.
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def set_threshold(self, threshold: float):
        self.threshold = float(threshold)

    def evaluate(self, X_test, y_test):
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

        y_pred = self.predict(X_test)

        # ROC-AUC uses raw probabilities
        proba = None
        auc = None
        try:
            proba = self.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, proba))
        except Exception:
            pass

        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))

        self.metrics = {
            "accuracy": acc,
            "f1_score": f1,
            "roc_auc": auc,
            "threshold": self.threshold,
            "model_type": self.model_type,
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        return self.metrics

    def get_feature_importance(self, feature_names: list):
        import pandas as pd

        model_for_importance = self.base_model if self.base_model is not None else self.model

        # XGBoost + RF have feature_importances_
        if hasattr(model_for_importance, "feature_importances_"):
            importances = model_for_importance.feature_importances_
            return (
                pd.DataFrame({"feature": feature_names, "importance": importances})
                .sort_values("importance", ascending=False)
            )

        return pd.DataFrame({"feature": feature_names, "importance": [None] * len(feature_names)})

    def save(self, path: str = None):
        import joblib
        if path is None:
            path = config.CLASSIFICATION_MODEL_PATH

        joblib.dump(
            {
                "model": self.model,
                "base_model": self.base_model,
                "model_type": self.model_type,
                "threshold": self.threshold,
            },
            path,
        )

    def load(self, path: str = None):
        import joblib
        if path is None:
            path = config.CLASSIFICATION_MODEL_PATH

        obj = joblib.load(path)

        if isinstance(obj, dict) and "model" in obj:
            self.model = obj["model"]
            self.base_model = obj.get("base_model", None)
            self.model_type = obj.get("model_type", "unknown")
            self.threshold = float(obj.get("threshold", 0.5))
        else:
            # backward compat: raw estimator
            self.model = obj
            self.base_model = obj
            self.model_type = "unknown"
            self.threshold = 0.5


class MovieClusterer:
    """Clustering model for movie segmentation"""
    
    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.model = KMeans(**{**config.KMEANS_PARAMS, 'n_clusters': n_clusters})
        self.metrics = {}
        self.cluster_profiles = None
        
    def train(self, X_train: pd.DataFrame) -> None:
        """Train the clustering model"""
        print(f"\nTraining K-Means clustering model (k={self.n_clusters})...")
        self.model.fit(X_train)
        print("Training completed!")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Assign cluster labels"""
        return self.model.predict(X)
    
    def evaluate(self, X: pd.DataFrame) -> Dict[str, float]:
        """Evaluate clustering quality"""
        labels = self.predict(X)
        
        silhouette = silhouette_score(X, labels)
        inertia = self.model.inertia_
        
        print(f"\nClustering Metrics:")
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Inertia: {inertia:,.2f}")
        
        self.metrics = {
            'silhouette_score': silhouette,
            'inertia': inertia,
            'n_clusters': self.n_clusters
        }
        
        return self.metrics
    
    def create_cluster_profiles(self, X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Create interpretable cluster profiles"""
        labels = self.predict(X)
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = labels
        
        # Aggregate statistics per cluster
        profile = df_with_clusters.groupby('cluster').agg({
            'budget': ['mean', 'median'],
            'revenue': ['mean', 'median'],
            'vote_average': 'mean',
            'runtime': 'mean'
        }).round(2)
        
        profile.columns = ['_'.join(col).strip() for col in profile.columns.values]
        profile['count'] = df_with_clusters.groupby('cluster').size()
        
        # Calculate ROI per cluster
        df_with_clusters['roi'] = (df_with_clusters['revenue'] - df_with_clusters['budget']) / (df_with_clusters['budget'] + 1)
        profile['avg_roi'] = df_with_clusters.groupby('cluster')['roi'].mean().round(2)
        
        self.cluster_profiles = profile
        
        print("\nCluster Profiles:")
        print(profile)
        
        return profile
    
    def interpret_clusters(self) -> Dict[int, str]:
        """Provide interpretable cluster names"""
        if self.cluster_profiles is None:
            return {}
        
        interpretations = {}
        for cluster_id in range(self.n_clusters):
            if cluster_id not in self.cluster_profiles.index:
                continue
                
            profile = self.cluster_profiles.loc[cluster_id]
            
            budget = profile['budget_mean']
            revenue = profile['revenue_mean']
            roi = profile['avg_roi']
            
            if budget < 10e6 and revenue < 50e6:
                label = "Low-Budget Indies"
            elif budget < 50e6 and roi > 2:
                label = "Profitable Mid-Budget"
            elif budget > 100e6 and revenue > 500e6:
                label = "Blockbusters"
            elif roi < 1:
                label = "Underperformers"
            else:
                label = f"Cluster {cluster_id}"
            
            interpretations[cluster_id] = label
        
        return interpretations
    
    def save(self, filepath: str = None):
        """Save the trained model"""
        if filepath is None:
            filepath = config.CLUSTERING_MODEL_PATH
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str = None):
        """Load a trained model"""
        if filepath is None:
            filepath = config.CLUSTERING_MODEL_PATH
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")


class TimeSeriesAnalyzer:
    """Time-series analysis for seasonal trends"""
    
    def __init__(self):
        self.monthly_trends = None
        self.seasonal_trends = None
        self.yearly_trends = None
        
    def analyze(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Perform comprehensive time-series analysis"""
        print("\nPerforming time-series analysis...")
        
        # Guard: the cleaned dataset may not include temporal columns
        required_cols = ['release_month']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"Time-series analysis skipped: missing columns {missing}.")
            return {}

        # Monthly trends
        self.monthly_trends = df.groupby('release_month').agg({
            'revenue': ['mean', 'median', 'count'],
            'budget': 'mean',
            'vote_average': 'mean'
        }).round(2)
        
        self.monthly_trends.columns = ['_'.join(col).strip() for col in self.monthly_trends.columns.values]
        self.monthly_trends = self.monthly_trends.sort_values('revenue_mean', ascending=False)
        
        # Seasonal trends
        season_map = {1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring',
                     6: 'summer', 7: 'summer', 8: 'summer', 9: 'fall', 10: 'fall',
                     11: 'fall', 12: 'winter'}
        
        df['season'] = df['release_month'].map(season_map)
        self.seasonal_trends = df.groupby('season').agg({
            'revenue': ['mean', 'median', 'count'],
            'vote_average': 'mean'
        }).round(2)
        
        self.seasonal_trends.columns = ['_'.join(col).strip() for col in self.seasonal_trends.columns.values]
        
        # Yearly trends
        self.yearly_trends = df.groupby('release_year').agg({
            'revenue': ['mean', 'median', 'sum', 'count'],
            'budget': 'mean'
        }).round(2)
        
        self.yearly_trends.columns = ['_'.join(col).strip() for col in self.yearly_trends.columns.values]
        
        print("\nTop 5 Months by Average Revenue:")
        print(self.monthly_trends.head())
        
        print("\nSeasonal Performance:")
        print(self.seasonal_trends)
        
        return {
            'monthly': self.monthly_trends,
            'seasonal': self.seasonal_trends,
            'yearly': self.yearly_trends
        }
    
    def get_best_release_months(self, top_n: int = 3) -> list:
        """Get the best months to release a movie"""
        return self.monthly_trends.head(top_n).index.tolist()
    
    def forecast_trend(self, periods: int = 12) -> Dict:
        """Simple trend forecast (linear extrapolation)"""
        # This is a simplified forecast - in production, use ARIMA/Prophet
        if self.yearly_trends is None:
            return {}
        
        recent_years = self.yearly_trends.tail(5)
        avg_growth = recent_years['revenue_mean'].pct_change().mean()
        
        return {
            'average_growth_rate': avg_growth,
            'trend': 'increasing' if avg_growth > 0 else 'decreasing'
        }


class DimensionalityReducer:
    """PCA for visualization and dimensionality reduction"""
    
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.model = PCA(n_components=n_components, random_state=config.RANDOM_STATE)
        self.explained_variance = None
        
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit PCA and transform data"""
        print(f"\nPerforming PCA (n_components={self.n_components})...")
        X_reduced = self.model.fit_transform(X)
        self.explained_variance = self.model.explained_variance_ratio_
        
        print(f"Explained variance: {self.explained_variance}")
        print(f"Total variance explained: {sum(self.explained_variance):.2%}")
        
        return X_reduced
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data"""
        return self.model.transform(X)
    
    def get_component_loadings(self, feature_names: list) -> pd.DataFrame:
        """Get feature loadings for each component"""
        loadings = pd.DataFrame(
            self.model.components_.T,
            columns=[f'PC{i+1}' for i in range(self.n_components)],
            index=feature_names
        )
        return loadings
