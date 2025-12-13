"""
Prefect Orchestration Pipeline for Movie Intelligence System
Automates: Data ingestion → Validation → Training → Deployment
"""
from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Tuple
import time

import config
from src.utils.preprocessing import MovieDataPreprocessor, prepare_train_test_split
from src.models.ml_models import (
    MovieRevenuePredictor, MovieClassifier, MovieClusterer,
    TimeSeriesAnalyzer, DimensionalityReducer
)


@task(name="ingest_data", retries=2, retry_delay_seconds=30)
def ingest_data() -> pd.DataFrame:
    """Task 1: Ingest raw data"""
    print("📥 Ingesting data from source...")
    preprocessor = MovieDataPreprocessor()
    df = preprocessor.load_data()
    print(f"✅ Data ingested: {len(df)} records")
    return df


@task(name="validate_data", retries=1)
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Task 2: Validate data quality"""
    print("🔍 Validating data quality...")
    
    # Check for required columns
    required_columns = ['budget', 'revenue', 'runtime', 'genres', 'release_month', 'release_year']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check data types
    assert df['budget'].dtype in [int, float], "Budget must be numeric"
    assert df['revenue'].dtype in [int, float], "Revenue must be numeric"
    
    # Check for minimum data requirements
    assert len(df) > 100, f"Insufficient data: {len(df)} records (minimum 100 required)"
    
    # Check for extreme outliers
    assert df['budget'].max() < 1e10, "Budget values too large"
    assert df['revenue'].max() < 5e10, "Revenue values too large"
    
    print(f"✅ Data validation passed")
    return df


@task(name="clean_data")
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Task 3: Clean and prepare data"""
    print("🧹 Cleaning data...")
    preprocessor = MovieDataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    print(f"✅ Data cleaned: {len(df_clean)} records")
    return df_clean


@task(name="engineer_features")
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Task 4: Feature engineering"""
    print("🔧 Engineering features...")
    preprocessor = MovieDataPreprocessor()
    df_featured = preprocessor.engineer_features(df, fit=True)
    df_featured = preprocessor.create_target_labels(df_featured)
    
    # Save preprocessor
    preprocessor.save_preprocessor()
    
    # Save processed data
    df_featured.to_csv(config.PROCESSED_DATA_PATH, index=False)
    
    print(f"✅ Features engineered: {len(df_featured.columns)} columns")
    return df_featured


@task(name="prepare_features")
def prepare_features(df: pd.DataFrame) -> Tuple:
    """Task 5: Prepare feature matrices"""
    print("📊 Preparing feature matrices...")
    preprocessor = MovieDataPreprocessor()
    preprocessor.load_preprocessor()
    
    X, y_revenue = preprocessor.prepare_features(df, target_col='revenue', fit=False)
    X_train, X_test, y_train, y_test = prepare_train_test_split(X, y_revenue)
    
    print(f"✅ Features prepared - Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, df


@task(name="train_regression_models")
def train_regression_models(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                           y_train: pd.Series, y_test: pd.Series) -> Dict:
    """Task 6: Train regression models"""
    print("🎯 Training regression models...")
    
    results = {}
    model_types = ['linear', 'random_forest', 'gradient_boosting']
    
    for model_type in model_types:
        print(f"\nTraining {model_type} model...")
        predictor = MovieRevenuePredictor(model_type=model_type)
        predictor.train(X_train, y_train)
        metrics = predictor.evaluate(X_test, y_test)
        
        # Save best model (Random Forest)
        if model_type == 'random_forest':
            predictor.save()
        
        results[f'regression_{model_type}'] = metrics
    
    print("✅ Regression models trained")
    return results


@task(name="train_classification_model")
def train_classification_model(X_train: pd.DataFrame, X_test: pd.DataFrame, df: pd.DataFrame) -> Dict:
    """Task 7: Train classification model"""
    print("🎯 Training classification model...")
    
    y_class = df.loc[X_train.index.union(X_test.index), 'is_hit']
    y_train_class = y_class.loc[X_train.index]
    y_test_class = y_class.loc[X_test.index]
    
    classifier = MovieClassifier(model_type='random_forest')
    classifier.train(X_train, y_train_class)
    metrics = classifier.evaluate(X_test, y_test_class)
    classifier.save()
    
    print("✅ Classification model trained")
    return {'classification': metrics}


@task(name="train_clustering_model")
def train_clustering_model(X_train: pd.DataFrame, df: pd.DataFrame) -> Dict:
    """Task 8: Train clustering model"""
    print("🎯 Training clustering model...")
    
    clusterer = MovieClusterer(n_clusters=4)
    clusterer.train(X_train)
    metrics = clusterer.evaluate(X_train)
    
    df_train = df.loc[X_train.index]
    profiles = clusterer.create_cluster_profiles(X_train, df_train)
    clusterer.save()
    
    metrics['cluster_profiles'] = profiles.to_dict()
    
    print("✅ Clustering model trained")
    return {'clustering': metrics}


@task(name="run_timeseries_analysis")
def run_timeseries_analysis(df: pd.DataFrame) -> Dict:
    """Task 9: Time-series analysis"""
    print("📈 Running time-series analysis...")
    
    analyzer = TimeSeriesAnalyzer()
    trends = analyzer.analyze(df)
    best_months = analyzer.get_best_release_months(top_n=3)
    forecast = analyzer.forecast_trend()
    
    results = {
        'timeseries': {
            'monthly_trends': trends['monthly'].to_dict(),
            'seasonal_trends': trends['seasonal'].to_dict(),
            'best_release_months': best_months,
            'forecast': forecast
        }
    }
    
    print("✅ Time-series analysis complete")
    return results


@task(name="run_pca")
def run_pca(X_train: pd.DataFrame, feature_names: list) -> Dict:
    """Task 10: Dimensionality reduction"""
    print("🔬 Running PCA...")
    
    pca = DimensionalityReducer(n_components=2)
    X_reduced = pca.fit_transform(X_train)
    loadings = pca.get_component_loadings(feature_names)
    
    results = {
        'pca': {
            'explained_variance': pca.explained_variance.tolist(),
            'total_variance_explained': float(sum(pca.explained_variance)),
            'top_pc1_features': loadings['PC1'].abs().sort_values(ascending=False).head(10).to_dict()
        }
    }
    
    print("✅ PCA complete")
    return results


@task(name="save_results")
def save_results(all_results: Dict) -> str:
    """Task 11: Save all results"""
    print("💾 Saving results...")
    
    results_file = config.REPORTS_DIR / 'training_results.json'
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"✅ Results saved to {results_file}")
    return str(results_file)


@task(name="validate_models")
def validate_models() -> bool:
    """Task 12: Validate trained models exist"""
    print("✓ Validating models...")
    
    model_paths = [
        config.REGRESSION_MODEL_PATH,
        config.CLASSIFICATION_MODEL_PATH,
        config.CLUSTERING_MODEL_PATH,
        config.SCALER_PATH
    ]
    
    all_exist = all(path.exists() for path in model_paths)
    
    if all_exist:
        print("✅ All models validated")
    else:
        missing = [str(p) for p in model_paths if not p.exists()]
        raise ValueError(f"Missing models: {missing}")
    
    return all_exist


@task(name="generate_report")
def generate_report(results_path: str) -> str:
    """Task 13: Generate summary report"""
    print("📄 Generating report...")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    report_lines = [
        "=" * 80,
        "MOVIE INTELLIGENCE SYSTEM - TRAINING REPORT",
        "=" * 80,
        f"\nTimestamp: {results.get('timestamp', 'N/A')}",
        "\n" + "=" * 80,
        "MODEL PERFORMANCE SUMMARY",
        "=" * 80
    ]
    
    # Regression
    if 'regression_random_forest' in results.get('models', {}):
        rf = results['models']['regression_random_forest']
        report_lines.extend([
            "\n[REGRESSION - Random Forest]",
            f"  RMSE: ${rf.get('rmse', 0):,.2f}",
            f"  R² Score: {rf.get('r2_score', 0):.4f}",
            f"  MAE: ${rf.get('mae', 0):,.2f}"
        ])
    
    # Classification
    if 'classification' in results.get('models', {}):
        clf = results['models']['classification']
        report_lines.extend([
            "\n[CLASSIFICATION]",
            f"  Accuracy: {clf.get('accuracy', 0):.4f}",
            f"  F1 Score: {clf.get('f1_score', 0):.4f}"
        ])
    
    # Clustering
    if 'clustering' in results.get('models', {}):
        cluster = results['models']['clustering']
        report_lines.extend([
            "\n[CLUSTERING]",
            f"  Silhouette Score: {cluster.get('silhouette_score', 0):.4f}",
            f"  Clusters: {cluster.get('n_clusters', 0)}"
        ])
    
    report_lines.append("\n" + "=" * 80)
    
    report = "\n".join(report_lines)
    
    # Save report
    report_file = config.REPORTS_DIR / 'training_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\n✅ Report saved to {report_file}")
    
    return str(report_file)


@flow(name="movie-intelligence-ml-pipeline", log_prints=True)
def ml_training_pipeline():
    """
    Complete ML Training Pipeline with Prefect Orchestration
    
    Flow:
    1. Data Ingestion
    2. Data Validation
    3. Data Cleaning
    4. Feature Engineering
    5. Feature Preparation
    6. Model Training (Regression, Classification, Clustering)
    7. Time-Series Analysis
    8. PCA
    9. Results Saving
    10. Model Validation
    11. Report Generation
    """
    
    print("\n" + "="*80)
    print("🚀 STARTING MOVIE INTELLIGENCE ML PIPELINE")
    print("="*80 + "\n")
    
    # Pipeline execution
    df_raw = ingest_data()
    df_validated = validate_data(df_raw)
    df_clean = clean_data(df_validated)
    df_featured = engineer_features(df_clean)
    
    X_train, X_test, y_train, y_test, df = prepare_features(df_featured)
    
    # Train models in parallel
    regression_results = train_regression_models(X_train, X_test, y_train, y_test)
    classification_results = train_classification_model(X_train, X_test, df)
    clustering_results = train_clustering_model(X_train, df)
    timeseries_results = run_timeseries_analysis(df)
    
    # Load preprocessor to get feature names for PCA
    preprocessor = MovieDataPreprocessor()
    preprocessor.load_preprocessor()
    pca_results = run_pca(X_train, preprocessor.feature_names)
    
    # Combine all results
    from datetime import datetime
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'models': {
            **regression_results,
            **classification_results,
            **clustering_results,
            **timeseries_results,
            **pca_results
        }
    }
    
    # Save and validate
    results_path = save_results(all_results)
    validate_models()
    report_path = generate_report(results_path)
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")
    
    return {
        'status': 'success',
        'results_path': results_path,
        'report_path': report_path
    }


if __name__ == "__main__":
    # Run the pipeline
    result = ml_training_pipeline()
    print(f"\nPipeline Result: {result}")
