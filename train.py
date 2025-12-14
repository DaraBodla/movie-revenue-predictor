"""
Complete ML Training Pipeline
Trains all models: Regression, Classification, Clustering, Time-Series Analysis
"""
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path

import config
from src.utils.preprocessing import MovieDataPreprocessor, prepare_train_test_split
from src.validation.data_validation import DataValidator

from src.models.ml_models import (
    MovieRevenuePredictor, MovieClassifier, MovieClusterer,
    TimeSeriesAnalyzer, DimensionalityReducer
)


class MLPipeline:
    """Complete ML training and evaluation pipeline"""
    
    def __init__(self):
        self.preprocessor = MovieDataPreprocessor()
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'metrics': {}
        }
        
    def run_complete_pipeline(self):
        """Execute the complete ML pipeline"""
        print("="*80)
        print("MOVIE INTELLIGENCE SYSTEM - ML TRAINING PIPELINE")
        print("="*80)
        
        # Step 1: Data Loading
        print("\n[STEP 1] Loading data...")
        df = self.preprocessor.load_data()
        
        # Step 2: Data Cleaning
        print("\n[STEP 2] Cleaning data...")
        df = self.preprocessor.clean_data(df)

        # Option 1: Auto-drop runtime outliers before validation
        # Keep only feature-film runtimes (matches your validation intent)
        before = len(df)
        df = df[df["runtime"].between(30, 300)].copy()
        df.reset_index(drop=True, inplace=True)
        dropped = before - len(df)
        print(f"[STEP 2a] Dropped {dropped} rows due to runtime outliers (kept 30–300).")

        
        # Step 2b: Data Validation
        print("\n[STEP 2b] Validating data...")
        validator = DataValidator()
        result = validator.validate_raw_data(df)
        if not result.success:
            raise ValueError(f"Data validation failed: {result.details}")

        # Step 3: Feature Engineering
        print("\n[STEP 3] Engineering features...")
        df = self.preprocessor.engineer_features(df, fit=True)
        
        # Step 4: Create Target Labels
        print("\n[STEP 4] Creating target labels...")
        df = self.preprocessor.create_target_labels(df)
        
        # Save processed data
        df.to_csv(config.PROCESSED_DATA_PATH, index=False)
        print(f"Processed data saved to {config.PROCESSED_DATA_PATH}")
        
        # Step 5: Prepare Features
        print("\n[STEP 5] Preparing feature matrices...")
        X, y_revenue = self.preprocessor.prepare_features(df, target_col='revenue', fit=True)
        
        # Save preprocessor
        self.preprocessor.save_preprocessor()
        
        # Split data
        X_train, X_test, y_train, y_test = prepare_train_test_split(X, y_revenue)
        
        # Step 6: Train Regression Models
        print("\n" + "="*80)
        print("[STEP 6] TRAINING REGRESSION MODELS")
        print("="*80)
        self._train_regression_models(X_train, X_test, y_train, y_test)
        
        # Step 7: Train Classification Model
        print("\n" + "="*80)
        print("[STEP 7] TRAINING CLASSIFICATION MODEL")
        print("="*80)
        self._train_classification_model(X_train, X_test, df)
        
        # Step 8: Train Clustering Model
        print("\n" + "="*80)
        print("[STEP 8] TRAINING CLUSTERING MODEL")
        print("="*80)
        self._train_clustering_model(X_train, df)
        
        # Step 9: Time-Series Analysis
        print("\n" + "="*80)
        print("[STEP 9] TIME-SERIES ANALYSIS")
        print("="*80)
        self._perform_timeseries_analysis(df)
        
        # Step 10: Dimensionality Reduction
        print("\n" + "="*80)
        print("[STEP 10] DIMENSIONALITY REDUCTION (PCA)")
        print("="*80)
        self._perform_pca(X_train)
        
        # Step 11: Save Results
        self._save_results()
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return self.results
    
    def _train_regression_models(self, X_train, X_test, y_train, y_test):
        """Train and compare multiple regression models"""
        model_types = ['linear', 'random_forest', 'gradient_boosting']
        
        for model_type in model_types:
            print(f"\n--- {model_type.upper()} MODEL ---")
            
            # Train model
            predictor = MovieRevenuePredictor(model_type=model_type)
            predictor.train(X_train, y_train)
            
            # Evaluate
            metrics = predictor.evaluate(X_test, y_test)
            
            # Feature importance (for tree-based models)
            if model_type in ['random_forest', 'gradient_boosting']:
                importance = predictor.get_feature_importance(self.preprocessor.feature_names)
                print(f"\nTop 10 Important Features:")
                print(importance.head(10))
                metrics['feature_importance'] = importance.head(20).to_dict()
            
            # Save model (save best performing one)
            if model_type == 'random_forest':  # Default best model
                predictor.save()
            
            # Store results
            self.results['models'][f'regression_{model_type}'] = metrics
    
    def _train_classification_model(self, X_train, X_test, df):
        """Train classification model for hit/flop prediction"""
        # Prepare classification target
        y_class = df.loc[X_train.index.union(X_test.index), 'is_hit']
        y_train_class = y_class.loc[X_train.index]
        y_test_class = y_class.loc[X_test.index]
        
        # Train classifier
        classifier = MovieClassifier(model_type='random_forest')
        classifier.train(X_train, y_train_class)
        
        # Evaluate
        metrics = classifier.evaluate(X_test, y_test_class)
        
        # Feature importance
        importance = classifier.get_feature_importance(self.preprocessor.feature_names)
        print(f"\nTop 10 Important Features for Classification:")
        print(importance.head(10))
        metrics['feature_importance'] = importance.head(20).to_dict()
        
        # Save model
        classifier.save()
        
        # Store results
        self.results['models']['classification'] = metrics
    
    def _train_clustering_model(self, X_train, df):
        """Train clustering model for movie segmentation"""
        # Train clusterer
        clusterer = MovieClusterer(n_clusters=4)
        clusterer.train(X_train)
        
        # Evaluate
        metrics = clusterer.evaluate(X_train)
        
        # Create cluster profiles
        df_train = df.loc[X_train.index]
        profiles = clusterer.create_cluster_profiles(X_train, df_train)
        
        # Interpret clusters
        interpretations = clusterer.interpret_clusters()
        print("\nCluster Interpretations:")
        for cluster_id, label in interpretations.items():
            print(f"Cluster {cluster_id}: {label}")
        
        metrics['cluster_profiles'] = profiles.to_dict()
        metrics['interpretations'] = interpretations
        
        # Save model
        clusterer.save()
        
        # Store results
        self.results['models']['clustering'] = metrics
    
    def _perform_timeseries_analysis(self, df):
        """Perform time-series analysis"""
        analyzer = TimeSeriesAnalyzer()
        trends = analyzer.analyze(df)
        
        # Get best release months
        best_months = analyzer.get_best_release_months(top_n=3)
        print(f"\nBest Release Months: {best_months}")
        
        # Forecast trend
        forecast = analyzer.forecast_trend()
        print(f"\nTrend Forecast: {forecast}")
        
        # Store results
        self.results['models']['timeseries'] = {
            'monthly_trends': trends['monthly'].to_dict(),
            'seasonal_trends': trends['seasonal'].to_dict(),
            'best_release_months': best_months,
            'forecast': forecast
        }
    
    def _perform_pca(self, X_train):
        """Perform PCA for dimensionality reduction"""
        pca = DimensionalityReducer(n_components=2)
        X_reduced = pca.fit_transform(X_train)
        
        # Get component loadings
        loadings = pca.get_component_loadings(self.preprocessor.feature_names)
        print("\nTop features in PC1:")
        print(loadings['PC1'].abs().sort_values(ascending=False).head(5))
        
        # Store results
        self.results['models']['pca'] = {
            'explained_variance': pca.explained_variance.tolist(),
            'total_variance_explained': float(sum(pca.explained_variance)),
            'top_pc1_features': loadings['PC1'].abs().sort_values(ascending=False).head(10).to_dict()
        }
    
    def _save_results(self):
        """Save pipeline results"""
        results_file = config.REPORTS_DIR / 'training_results.json'
        
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            return obj
        
        # Save results
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=convert_types)
        
        print(f"\nResults saved to {results_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        
        if 'regression_random_forest' in self.results['models']:
            rf_metrics = self.results['models']['regression_random_forest']
            print(f"\nREGRESSION (Random Forest):")
            print(f"  RMSE: ${rf_metrics['rmse']:,.2f}")
            print(f"  R² Score: {rf_metrics['r2_score']:.4f}")
        
        if 'classification' in self.results['models']:
            class_metrics = self.results['models']['classification']
            print(f"\nCLASSIFICATION:")
            print(f"  Accuracy: {class_metrics['accuracy']:.4f}")
            print(f"  F1 Score: {class_metrics['f1_score']:.4f}")
        
        if 'clustering' in self.results['models']:
            cluster_metrics = self.results['models']['clustering']
            print(f"\nCLUSTERING:")
            print(f"  Silhouette Score: {cluster_metrics['silhouette_score']:.4f}")
            print(f"  Number of Clusters: {cluster_metrics['n_clusters']}")


def main():
    """Main execution function"""
    pipeline = MLPipeline()
    results = pipeline.run_complete_pipeline()
    return results


if __name__ == "__main__":
    main()
