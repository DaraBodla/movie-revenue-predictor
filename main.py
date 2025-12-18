"""
FastAPI Application for Movie Intelligence System
Provides endpoints for all ML tasks
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.security import APIKeyHeader
from src.utils.tmdb_client import tmdb_search_movie, tmdb_movie_details
from src.utils.tmdb_mapper import map_tmdb_details_to_model_input



from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import io
import os
import json
import config
from src.utils.logging_config import setup_logging
from src.monitoring.live_log import log_live_input

logger = setup_logging()


API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

CLASSIFICATION_THRESHOLD = float(os.getenv("CLASSIFICATION_THRESHOLD","0.50"))

def verify_api_key(request: Request, api_key: str = Depends(API_KEY_HEADER)):
    """Optional API key check.

    If API_KEY env var is set, requests must send X-API-Key.
    """
    expected = os.getenv("API_KEY")
    if expected and api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

from src.utils.preprocessing import MovieDataPreprocessor
from src.models.ml_models import (
    MovieRevenuePredictor, MovieClassifier, MovieClusterer, TimeSeriesAnalyzer
)


# Pydantic models for request/response
class MovieInput(BaseModel):
    """Single movie input for revenue prediction.

    This schema is aligned to the uploaded `movies_clean.csv` predictors.
    """
    budget: float = Field(..., description="Movie budget in USD", gt=0)
    popularity: float = Field(..., description="Popularity score", ge=0)
    runtime: float = Field(..., description="Runtime in minutes", gt=0)
    vote_average: float = Field(..., description="Average rating (0-10)", ge=0, le=10)
    vote_count: float = Field(..., description="Number of votes", ge=0)
    release_month: int = Field(..., description="Release month (1-12)", ge=1, le=12)
    genres: Optional[str] = None

class RevenuePredictionResponse(BaseModel):
    """Revenue prediction response"""
    predicted_revenue: float
    predicted_revenue_formatted: str
    confidence_interval: Optional[Dict[str, float]] = None
    model_used: str


class ClassificationResponse(BaseModel):
    """Classification prediction response"""
    prediction: str # "Hit" or "Flop" for ui
    predicted_class: str
    is_hit: bool
    hit_probability: float
    flop_probability: float
    prediction_label: str


class ClusterResponse(BaseModel):
    """Clustering response"""
    cluster_id: int
    cluster_label: str
    cluster_profile: Dict


class TimeSeriesResponse(BaseModel):
    """Time-series analysis response"""
    monthly_trends: Dict
    best_release_months: List[int]
    seasonal_trends: Dict


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    models_loaded: Dict[str, bool]


# Initialize FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=config.API_DESCRIPTION
)



# Optional rate limiting (enabled if slowapi is installed)
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address)
except Exception:  # slowapi not installed / disabled
    limiter = None

if limiter:
    # Attach limiter to app in startup below
    pass


# Global model storage
class ModelRegistry:
    """Singleton for model management"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.preprocessor = None
            cls._instance.regression_model = None
            cls._instance.classification_model = None
            cls._instance.clustering_model = None
            cls._instance._load_models()
        return cls._instance
    
    def _load_models(self):
        """Load all trained models"""
        try:
            # Load preprocessor
            self.preprocessor = MovieDataPreprocessor()
            self.preprocessor.load_preprocessor()
            
            # Load regression model
            self.regression_model = MovieRevenuePredictor()
            self.regression_model.load()
            
            # Load classification model
            self.classification_model = MovieClassifier()
            self.classification_model.load()
            
            # Load clustering model
            self.clustering_model = MovieClusterer()
            self.clustering_model.load()
            
            print("All models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise


# Initialize model registry
try:
    models = ModelRegistry()
except Exception as e:
    print(f"Warning: Could not load models on startup: {e}")
    models = None



def _maybe_limit(rule: str):
    """Apply slowapi rate limit if enabled; otherwise no-op decorator."""
    if limiter:
        return limiter.limit(rule)
    def _decorator(fn):
        return fn
    return _decorator




@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check"""
    return {
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "regression": models.regression_model is not None if models else False,
            "classification": models.classification_model is not None if models else False,
            "clustering": models.clustering_model is not None if models else False
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint.

    Always returns HTTP 200 so Docker/K8s health checks can work even before
    models are trained. The response body indicates whether models are loaded.
    """
    if models is None:
        return {
            "status": "degraded",
            "timestamp": datetime.now().isoformat(),
            "models_loaded": {
                "regression": False,
                "classification": False,
                "clustering": False
            }
        }

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "regression": models.regression_model is not None,
            "classification": models.classification_model is not None,
            "clustering": models.clustering_model is not None
        }
    }


@app.get("/live/tmdb/search")
async def live_tmdb_search(q: str, year: int | None = None, api_key: str = Depends(verify_api_key)):
    results = await tmdb_search_movie(q, year=year)
    return [
        {
            "id": r.get("id"),
            "title": r.get("title"),
            "release_date": r.get("release_date"),
            "popularity": r.get("popularity"),
            "vote_average": r.get("vote_average"),
            "vote_count": r.get("vote_count"),
            "genres": r.get("genres"),
        }
        for r in results
    ]

@app.get("/live/tmdb/{movie_id}/inputs")
async def live_tmdb_inputs(movie_id: int, api_key: str = Depends(verify_api_key)):
    details = await tmdb_movie_details(movie_id)
    return map_tmdb_details_to_model_input(details)


@app.post("/predict/revenue", response_model=RevenuePredictionResponse)
@_maybe_limit("100/minute")
async def predict_revenue(
    request: Request,
    movie: MovieInput,
    api_key: str = Depends(verify_api_key)
):
    """Predict movie revenue for a single movie"""
    log_live_input(movie.dict())
    
    if models is None or models.regression_model is None:
        raise HTTPException(status_code=503, detail="Regression model not loaded")
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([movie.dict()])
        
        # Preprocess
        input_data = models.preprocessor.engineer_features(input_data, fit=False)
        X, _ = models.preprocessor.prepare_features(input_data, target_col=None, fit=False)
        
        # Predict
        prediction = models.regression_model.predict(X)[0]
        
        # Format response
        return {
            "predicted_revenue": float(prediction),
            "predicted_revenue_formatted": f"${prediction:,.2f}",
            "model_used": "random_forest"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/classification", response_model=ClassificationResponse)
async def predict_classification(movie: MovieInput):
    """Predict if movie will be a hit or flop"""
    log_live_input(movie.dict())

    if models is None or models.classification_model is None:
        raise HTTPException(status_code=503, detail="Classification model not loaded")

    try:
        input_data = pd.DataFrame([movie.dict()])

        input_data = models.preprocessor.engineer_features(input_data, fit=False)
        X, _ = models.preprocessor.prepare_features(input_data, target_col=None, fit=False)

        proba = models.classification_model.predict_proba(X)[0]
        p_hit = float(proba[1])
        p_flop = float(proba[0])

        is_hit = p_hit >= CLASSIFICATION_THRESHOLD
        label = "Hit" if is_hit else "Flop"

        return {
            "prediction": label,
            "predicted_class": label,# ✅ add this (UI-friendly)
            "is_hit": bool(is_hit),
            "hit_probability": p_hit,
            "flop_probability": p_flop,
            "prediction_label": label,
            "threshold_used": CLASSIFICATION_THRESHOLD,# keep your old key too
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")



@app.post("/predict/cluster", response_model=ClusterResponse)
async def predict_cluster(movie: MovieInput):
    log_live_input(movie.dict())
    """Assign movie to a performance cluster"""
    if models is None or models.clustering_model is None:
        raise HTTPException(status_code=503, detail="Clustering model not loaded")
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([movie.dict()])
        
        # Preprocess
        input_data = models.preprocessor.engineer_features(input_data, fit=False)
        X, _ = models.preprocessor.prepare_features(input_data, target_col=None, fit=False)
        
        # Predict cluster
        cluster_id = int(models.clustering_model.predict(X)[0])
        
        # Get cluster interpretation
        # ✅ Load cluster interpretations + profiles from training_results.json (single source of truth)
        results_file = config.REPORTS_DIR / "training_results.json"

        interpretations = {}
        profiles = {}

        if results_file.exists():
            with open(results_file, "r", encoding="utf-8") as f:
                tr = json.load(f)
            clustering = tr.get("models", {}).get("clustering", {})
            interpretations = clustering.get("interpretations", {}) or {}
            profiles = clustering.get("cluster_profiles", {}) or {}

        # label from file (keys are usually strings)
        cluster_label = interpretations.get(str(cluster_id), f"Cluster {cluster_id}")

        # build profile for this cluster id (profiles stored as metric-> {cluster_id: value})
        cluster_profile = {"cluster_id": cluster_id}
        try:
            for metric_name, mapping in (profiles or {}).items():
                if isinstance(mapping, dict):
                    cluster_profile[metric_name] = mapping.get(str(cluster_id), mapping.get(cluster_id))
        except Exception:
            pass

        return {
            "cluster_id": cluster_id,
            "cluster_label": cluster_label,
            "cluster_profile": cluster_profile
        }

        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """Batch prediction from CSV file"""
    if models is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Read uploaded CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_cols = ['budget','popularity','runtime','vote_average','vote_count','release_month']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Preprocess
        df = models.preprocessor.engineer_features(df, fit=False)
        X, _ = models.preprocessor.prepare_features(df, target_col=None, fit=False)
        
        # Predictions
        revenue_predictions = models.regression_model.predict(X)
        classification_predictions = models.classification_model.predict(X)
        cluster_predictions = models.clustering_model.predict(X)
        
        # Combine results
        results_df = df[['title']].copy() if 'title' in df.columns else pd.DataFrame(index=df.index)
        results_df['predicted_revenue'] = revenue_predictions
        results_df['is_hit'] = classification_predictions
        results_df['cluster_id'] = cluster_predictions
        
        return JSONResponse(content={
            "total_predictions": len(results_df),
            "predictions": results_df.to_dict('records')
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/analysis/timeseries", response_model=TimeSeriesResponse)
async def get_timeseries_analysis():
    """Get time-series analysis results"""
    try:
        # Load processed data
        if not config.PROCESSED_DATA_PATH.exists():
            raise HTTPException(status_code=404, detail="Processed data not found. Run training first.")
        
        df = pd.read_csv(config.PROCESSED_DATA_PATH)
        
        # Perform analysis
        analyzer = TimeSeriesAnalyzer()
        trends = analyzer.analyze(df)
        best_months = analyzer.get_best_release_months(top_n=3)
        
        return {
            "monthly_trends": trends['monthly'].to_dict(),
            "best_release_months": best_months,
            "seasonal_trends": trends['seasonal'].to_dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")
    
DRIFT_DIR = os.path.join("reports", "drift")
DRIFT_HTML = os.path.join(DRIFT_DIR, "drift_report.html")
DRIFT_JSON = os.path.join(DRIFT_DIR, "drift_summary.json")

@app.get("/drift/report", response_class=HTMLResponse)
def get_drift_report():
    if not os.path.exists(DRIFT_HTML):
        return HTMLResponse("<h3>Drift report not found. Run drift_monitor.py first.</h3>", status_code=404)
    return FileResponse(DRIFT_HTML)

@app.get("/drift/summary")
def get_drift_summary():
    if not os.path.exists(DRIFT_JSON):
        return {"detail": "Drift summary not found. Run drift_monitor.py first."}
    with open(DRIFT_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/models/info")
async def get_model_info():
    """Get information about loaded models"""
    if models is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Load training results
        results_file = config.REPORTS_DIR / 'training_results.json'
        if results_file.exists():
            import json
            with open(results_file, 'r') as f:
                training_results = json.load(f)
        else:
            training_results = {}
        
        return {
            "model_info": {
                "regression": {
                    "type": "Random Forest",
                    "features": len(models.preprocessor.feature_names) if models.preprocessor.feature_names else 0,
                    "metrics": training_results.get('models', {}).get('regression_random_forest', {})
                },
                "classification": {
                    "type": "Random Forest",
                    "features": len(models.preprocessor.feature_names) if models.preprocessor.feature_names else 0,
                    "metrics": training_results.get('models', {}).get('classification', {})
                },
                "clustering": {
                    "type": "K-Means",
                    "n_clusters": 4,
                    "metrics": training_results.get('models', {}).get('clustering', {})
                }
            },
            "training_timestamp": training_results.get('timestamp', 'unknown')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Info retrieval error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

@app.get("/health/detailed")
async def detailed_health():
    """Detailed health check: reports API + model loading status."""
    status = {"timestamp": datetime.utcnow().isoformat(), "status": "healthy", "checks": {}}
    loaded = (models is not None and models.regression_model is not None and models.classification_model is not None and models.clustering_model is not None)
    status["checks"]["models_loaded"] = {"status": "up" if loaded else "not_loaded"}
    if not loaded:
        status["status"] = "degraded"

    return status
