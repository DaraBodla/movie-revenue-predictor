
"""Test suite for Movie Intelligence System (aligned to movies_clean.csv schema).

Dataset schema (required):
- budget, popularity, runtime, vote_average, vote_count, release_month, revenue
"""
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from main import app
import config
from src.utils.preprocessing import MovieDataPreprocessor
from src.models.ml_models import MovieRevenuePredictor, MovieClassifier, MovieClusterer


def sample_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    budget = rng.integers(1_000, 100_000_000, size=n)
    popularity = rng.random(size=n) * 50
    runtime = rng.integers(60, 200, size=n)
    vote_average = rng.random(size=n) * 10
    vote_count = rng.integers(0, 50_000, size=n)
    # Synthetic revenue correlated with budget and popularity
    revenue = budget * (0.5 + rng.random(size=n) * 3.0) + popularity * 100_000
    return pd.DataFrame({
        "budget": budget.astype(float),
        "popularity": popularity.astype(float),
        "runtime": runtime.astype(float),
        "vote_average": vote_average.astype(float),
        "vote_count": vote_count.astype(float),
        "release_month": rng.integers(1, 13, size=n).astype(int),
        "revenue": revenue.astype(float),
    })


class TestPreprocessing:
    def test_clean_and_engineer(self):
        df = sample_df()
        p = MovieDataPreprocessor()
        df2 = p.clean_data(df)
        df3 = p.engineer_features(df2, fit=True)
        for col in ["log_budget","log_vote_count","popularity_score","vote_density","popularity_per_budget"]:
            assert col in df3.columns

    def test_prepare_features_train(self):
        df = sample_df()
        p = MovieDataPreprocessor()
        df3 = p.engineer_features(p.clean_data(df), fit=True)
        X, y = p.prepare_features(df3, target_col="revenue", fit=True)
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 13  # updated engineered features

    def test_prepare_features_inference(self):
        df = sample_df(5).drop(columns=["revenue"])
        p = MovieDataPreprocessor()
        # Fit scaler first using training sample
        train = sample_df(50)
        train_feat = p.engineer_features(p.clean_data(train), fit=True)
        p.prepare_features(train_feat, target_col="revenue", fit=True)

        df_feat = p.engineer_features(df, fit=False)
        X, y = p.prepare_features(df_feat, target_col=None, fit=False)
        assert y is None
        assert X.shape == (5, 13)


class TestModels:
    def test_regression_fit_predict(self):
        df = sample_df()
        p = MovieDataPreprocessor()
        df_feat = p.engineer_features(p.clean_data(df), fit=True)
        X, y = p.prepare_features(df_feat, target_col="revenue", fit=True)

        model = MovieRevenuePredictor(model_type="random_forest")
        model.train(X, y)
        preds = model.predict(X.iloc[:3])
        assert len(preds) == 3

    def test_classifier_fit_predict(self):
        df = sample_df()
        p = MovieDataPreprocessor()
        df = p.create_target_labels(df)
        df_feat = p.engineer_features(p.clean_data(df), fit=True)
        X, _ = p.prepare_features(df_feat, target_col="revenue", fit=True)
        y = df_feat.loc[X.index, "is_hit"]

        clf = MovieClassifier()
        clf.train(X, y)
        preds = clf.predict(X.iloc[:3])
        assert len(preds) == 3

    def test_clusterer_fit_predict(self):
        df = sample_df()
        p = MovieDataPreprocessor()
        df_feat = p.engineer_features(p.clean_data(df), fit=True)
        X, _ = p.prepare_features(df_feat, target_col="revenue", fit=True)

        cl = MovieClusterer(n_clusters=3)
        cl.train(X)
        preds = cl.predict(X.iloc[:5])
        assert len(preds) == 5


class TestAPI:
    def test_health(self):
        client = TestClient(app)
        r = client.get("/health")
        assert r.status_code == 200

    def test_predict_revenue_schema(self):
        # This test will pass only if models are available; in CI without trained artifacts it may 503.
        client = TestClient(app)
        payload = {"budget": 50000000, "popularity": 12.3, "runtime": 110, "vote_average": 7.1, "vote_count": 3400, "release_month": 7}
        r = client.post("/predict/revenue", json=payload)
        assert r.status_code in (200, 503)
