#!/usr/bin/env python3
"""
Hospital Readmission Prediction Pipeline
========================================

A comprehensive machine learning pipeline for predicting hospital readmissions
with XGBoost, Random Forest, Logistic Regression, and Neural Network models.
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
import xgboost as xgb

class HospitalReadmissionPipeline:
    """
    Complete ML pipeline for hospital readmission prediction
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.feature_importance = {}
        self.cv_results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        Path('model_outputs').mkdir(exist_ok=True)
    
    def load_data(self, data_path):
        """Load and prepare the dataset"""
        print("Loading dataset...")
        df = pd.read_csv(data_path)
        print(f"Dataset shape: {df.shape}")
        
        # Separate features and target
        X = df.drop(['readmitted_30_days', 'encounter_id', 'patient_nbr', 'readmitted'], axis=1)
        y = df['readmitted_30_days']
        
        # Clean feature names for XGBoost compatibility
        X.columns = X.columns.str.replace('[', '_', regex=False)
        X.columns = X.columns.str.replace(']', '_', regex=False)
        X.columns = X.columns.str.replace('<', '_', regex=False)
        X.columns = X.columns.str.replace('>', '_', regex=False)
        X.columns = X.columns.str.replace('(', '_', regex=False)
        X.columns = X.columns.str.replace(')', '_', regex=False)
        X.columns = X.columns.str.replace('/', '_', regex=False)
        
        print(f"Features shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def split_data(self, X, y):
        """Split data into train/validation/test sets (70/15/15)"""
        print("Splitting data...")
        
        # First split: 70% train, 30% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Second split: 15% validation, 15% test from the 30% temp
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"Train set shape: {X_train.shape}")
        print(f"Validation set shape: {X_val.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def prepare_features(self, X_train, X_val, X_test):
        """Prepare features with scaling for neural network"""
        print("Preparing features...")
        
        # Standard scaling for neural network
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def initialize_models(self):
        """Initialize models with exact specifications"""
        print("Initializing models...")
        
        self.models = {
            'XGBoost': xgb.XGBClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                eval_metric='logloss'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'Logistic Regression': LogisticRegression(
                C=0.1,
                penalty='l2',
                solver='liblinear',
                max_iter=1000,
                random_state=42
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(200, 100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=1000,
                random_state=42
            )
        }
        
        print(f"Initialized {len(self.models)} models")
    
    def train_models(self, X_train, y_train, X_train_scaled):
        """Train all models"""
        print("Training models...")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            if name == 'Neural Network':
                # Use scaled features for neural network
                model.fit(X_train_scaled, y_train)
            else:
                # Use original features for tree-based and logistic regression
                model.fit(X_train, y_train)
            
            print(f"{name} training completed")
    
    def evaluate_models(self, X_train, X_val, X_test, y_train, y_val, y_test,
                       X_train_scaled, X_val_scaled, X_test_scaled):
        """Evaluate all models and compute metrics"""
        print("Evaluating models...")
        
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            
            # Choose appropriate features
            if name == 'Neural Network':
                X_train_eval = X_train_scaled
                X_val_eval = X_val_scaled
                X_test_eval = X_test_scaled
            else:
                X_train_eval = X_train
                X_val_eval = X_val
                X_test_eval = X_test
            
            # Predictions
            y_train_pred = model.predict(X_train_eval)
            y_val_pred = model.predict(X_val_eval)
            y_test_pred = model.predict(X_test_eval)
            
            # Probabilities for ROC-AUC
            y_train_prob = model.predict_proba(X_train_eval)[:, 1]
            y_val_prob = model.predict_proba(X_val_eval)[:, 1]
            y_test_prob = model.predict_proba(X_test_eval)[:, 1]
            
            # Compute metrics
            self.results[name] = {
                'train_accuracy': accuracy_score(y_train, y_train_pred),
                'val_accuracy': accuracy_score(y_val, y_val_pred),
                'test_accuracy': accuracy_score(y_test, y_test_pred),
                'train_precision': precision_score(y_train, y_train_pred, average='weighted'),
                'val_precision': precision_score(y_val, y_val_pred, average='weighted'),
                'test_precision': precision_score(y_test, y_test_pred, average='weighted'),
                'train_recall': recall_score(y_train, y_train_pred, average='weighted'),
                'val_recall': recall_score(y_val, y_val_pred, average='weighted'),
                'test_recall': recall_score(y_test, y_test_pred, average='weighted'),
                'train_f1': f1_score(y_train, y_train_pred, average='weighted'),
                'val_f1': f1_score(y_val, y_val_pred, average='weighted'),
                'test_f1': f1_score(y_test, y_test_pred, average='weighted'),
                'train_roc_auc': roc_auc_score(y_train, y_train_prob),
                'val_roc_auc': roc_auc_score(y_val, y_val_prob),
                'test_roc_auc': roc_auc_score(y_test, y_test_prob)
            }
            
            print(f"{name} - Test Accuracy: {self.results[name]['test_accuracy']:.4f}")
    
    def perform_cross_validation(self, X, y, X_scaled):
        """Perform 5-fold stratified cross-validation"""
        print("Performing 5-fold stratified cross-validation...")
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"CV for {name}...")
            
            # Choose appropriate features
            X_eval = X_scaled if name == 'Neural Network' else X
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X_eval, y, cv=cv, scoring='accuracy')
            cv_precision = cross_val_score(model, X_eval, y, cv=cv, scoring='precision_weighted')
            cv_recall = cross_val_score(model, X_eval, y, cv=cv, scoring='recall_weighted')
            cv_f1 = cross_val_score(model, X_eval, y, cv=cv, scoring='f1_weighted')
            cv_roc_auc = cross_val_score(model, X_eval, y, cv=cv, scoring='roc_auc')
            
            self.cv_results[name] = {
                'accuracy_mean': cv_scores.mean(),
                'accuracy_std': cv_scores.std(),
                'precision_mean': cv_precision.mean(),
                'precision_std': cv_precision.std(),
                'recall_mean': cv_recall.mean(),
                'recall_std': cv_recall.std(),
                'f1_mean': cv_f1.mean(),
                'f1_std': cv_f1.std(),
                'roc_auc_mean': cv_roc_auc.mean(),
                'roc_auc_std': cv_roc_auc.std()
            }
            
            print(f"{name} - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    def analyze_feature_importance(self, X_train, feature_names):
        """Analyze feature importance for tree-based models"""
        print("Analyzing feature importance...")
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importance = model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                self.feature_importance[name] = feature_importance_df
                print(f"{name} - Top 5 features:")
                print(feature_importance_df.head())
                print()
            
            elif hasattr(model, 'coef_'):
                # Logistic Regression
                importance = np.abs(model.coef_[0])
                feature_importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                self.feature_importance[name] = feature_importance_df
                print(f"{name} - Top 5 features:")
                print(feature_importance_df.head())
                print()
    
    def save_models(self):
        """Save trained models using joblib"""
        print("Saving models...")
        
        for name, model in self.models.items():
            filename = f"model_outputs/{name.replace(' ', '_').lower()}_{self.timestamp}.pkl"
            joblib.dump(model, filename)
            print(f"Saved {name} to {filename}")
        
        # Save scaler
        scaler_filename = f"model_outputs/scaler_{self.timestamp}.pkl"
        joblib.dump(self.scalers['standard'], scaler_filename)
        print(f"Saved scaler to {scaler_filename}")
    
    def save_results(self):
        """Save all results to files"""
        print("Saving results...")
        
        # Performance metrics
        metrics_df = pd.DataFrame(self.results).T
        metrics_filename = f"model_outputs/performance_metrics_{self.timestamp}.csv"
        metrics_df.to_csv(metrics_filename)
        print(f"Saved performance metrics to {metrics_filename}")
        
        # Cross-validation results
        cv_df = pd.DataFrame(self.cv_results).T
        cv_filename = f"model_outputs/cv_results_{self.timestamp}.csv"
        cv_df.to_csv(cv_filename)
        print(f"Saved CV results to {cv_filename}")
        
        # Feature importance
        for name, importance_df in self.feature_importance.items():
            importance_filename = f"model_outputs/feature_importance_{name.replace(' ', '_').lower()}_{self.timestamp}.csv"
            importance_df.to_csv(importance_filename, index=False)
            print(f"Saved {name} feature importance to {importance_filename}")
        
        # Complete results as JSON
        complete_results = {
            'performance_metrics': self.results,
            'cv_results': self.cv_results,
            'timestamp': self.timestamp
        }
        
        json_filename = f"model_outputs/complete_results_{self.timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(complete_results, f, indent=2)
        print(f"Saved complete results to {json_filename}")
    
    def generate_comparison_report(self):
        """Generate a comprehensive model comparison report"""
        print("Generating model comparison report...")
        
        report = f"""
Hospital Readmission Prediction - Model Comparison Report
========================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

TARGET ACCURACY: 85%

MODEL SPECIFICATIONS:
- XGBoost: n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0
- Random Forest: n_estimators=300, max_depth=15, min_samples_split=5, min_samples_leaf=2, max_features='sqrt'
- Logistic Regression: C=0.1, penalty='l2', solver='liblinear', max_iter=1000
- Neural Network: hidden_layer_sizes=(200, 100, 50), activation='relu', solver='adam', alpha=0.001

DATA SPLIT: 70% train, 15% validation, 15% test
CROSS-VALIDATION: 5-fold stratified

TEST SET PERFORMANCE:
==================
"""
        
        # Sort models by test accuracy
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)
        
        for name, metrics in sorted_models:
            accuracy_status = "✓ MEETS TARGET" if metrics['test_accuracy'] >= 0.85 else "✗ BELOW TARGET"
            report += f"""
{name}:
  Test Accuracy:  {metrics['test_accuracy']:.4f} {accuracy_status}
  Test Precision: {metrics['test_precision']:.4f}
  Test Recall:    {metrics['test_recall']:.4f}
  Test F1-Score:  {metrics['test_f1']:.4f}
  Test ROC-AUC:   {metrics['test_roc_auc']:.4f}
"""
        
        report += f"""

CROSS-VALIDATION RESULTS:
======================
"""
        
        # Sort CV results by accuracy
        sorted_cv = sorted(self.cv_results.items(), key=lambda x: x[1]['accuracy_mean'], reverse=True)
        
        for name, cv_metrics in sorted_cv:
            report += f"""
{name}:
  CV Accuracy:  {cv_metrics['accuracy_mean']:.4f} (+/- {cv_metrics['accuracy_std'] * 2:.4f})
  CV Precision: {cv_metrics['precision_mean']:.4f} (+/- {cv_metrics['precision_std'] * 2:.4f})
  CV Recall:    {cv_metrics['recall_mean']:.4f} (+/- {cv_metrics['recall_std'] * 2:.4f})
  CV F1-Score:  {cv_metrics['f1_mean']:.4f} (+/- {cv_metrics['f1_std'] * 2:.4f})
  CV ROC-AUC:   {cv_metrics['roc_auc_mean']:.4f} (+/- {cv_metrics['roc_auc_std'] * 2:.4f})
"""
        
        report += f"""

FEATURE IMPORTANCE (Top 10):
=========================
"""
        
        for name, importance_df in self.feature_importance.items():
            report += f"""
{name}:
{importance_df.head(10).to_string(index=False)}

"""
        
        report += f"""
RECOMMENDATIONS:
===============
1. Best performing model: {sorted_models[0][0]} (Test Accuracy: {sorted_models[0][1]['test_accuracy']:.4f})
2. Most stable model (CV): {sorted_cv[0][0]} (CV Accuracy: {sorted_cv[0][1]['accuracy_mean']:.4f})
3. Models meeting 85% target: {len([m for m in sorted_models if m[1]['test_accuracy'] >= 0.85])} out of 4

FILES GENERATED:
===============
- Trained models: model_outputs/*_{self.timestamp}.pkl
- Performance metrics: model_outputs/performance_metrics_{self.timestamp}.csv
- CV results: model_outputs/cv_results_{self.timestamp}.csv
- Feature importance: model_outputs/feature_importance_*_{self.timestamp}.csv
- Complete results: model_outputs/complete_results_{self.timestamp}.json
"""
        
        # Save report
        report_filename = f"model_outputs/model_comparison_report_{self.timestamp}.txt"
        with open(report_filename, 'w') as f:
            f.write(report)
        
        print(f"Saved comparison report to {report_filename}")
        print(report)
    
    def run_pipeline(self, data_path):
        """Run the complete pipeline"""
        print("="*80)
        print("HOSPITAL READMISSION PREDICTION PIPELINE")
        print("="*80)
        
        # Load data
        X, y = self.load_data(data_path)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Prepare features
        X_train_scaled, X_val_scaled, X_test_scaled = self.prepare_features(X_train, X_val, X_test)
        
        # Initialize models
        self.initialize_models()
        
        # Train models
        self.train_models(X_train, y_train, X_train_scaled)
        
        # Evaluate models
        self.evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test,
                           X_train_scaled, X_val_scaled, X_test_scaled)
        
        # Cross-validation
        X_full = pd.concat([X_train, X_val, X_test])
        y_full = pd.concat([y_train, y_val, y_test])
        X_full_scaled = self.scalers['standard'].fit_transform(X_full)
        
        self.perform_cross_validation(X_full, y_full, X_full_scaled)
        
        # Feature importance
        self.analyze_feature_importance(X_train, X.columns)
        
        # Save everything
        self.save_models()
        self.save_results()
        self.generate_comparison_report()
        
        print("="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)

def main():
    """Main function to run the pipeline"""
    # Initialize pipeline
    pipeline = HospitalReadmissionPipeline()
    
    # Run with the diabetes dataset
    data_path = "processed_data/diabetes_processed_20250715_124353.csv"
    pipeline.run_pipeline(data_path)

if __name__ == "__main__":
    main()