#!/usr/bin/env python3
"""
Hospital Readmission Prediction Service
=======================================

A real-time prediction service for hospital readmission risk assessment
with ensemble modeling, confidence intervals, and explainability features.
"""

import joblib
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Validation and statistics
from pydantic import BaseModel, Field, validator
from scipy import stats
from sklearn.metrics import accuracy_score

class PatientData(BaseModel):
    """Pydantic model for patient data validation"""
    
    # Demographics
    age_numeric: float = Field(..., ge=0, le=150, description="Patient age in years")
    race_caucasian: float = Field(0, ge=0, le=1, description="Race: Caucasian")
    race_asian: float = Field(0, ge=0, le=1, description="Race: Asian")
    race_african_american: float = Field(0, ge=0, le=1, description="Race: African American")
    race_hispanic: float = Field(0, ge=0, le=1, description="Race: Hispanic")
    race_other: float = Field(0, ge=0, le=1, description="Race: Other")
    
    # Age categories
    age__10_20_: float = Field(0, ge=0, le=1, description="Age 10-20")
    age__20_30_: float = Field(0, ge=0, le=1, description="Age 20-30")
    age__30_40_: float = Field(0, ge=0, le=1, description="Age 30-40")
    age__40_50_: float = Field(0, ge=0, le=1, description="Age 40-50")
    age__50_60_: float = Field(0, ge=0, le=1, description="Age 50-60")
    age__60_70_: float = Field(0, ge=0, le=1, description="Age 60-70")
    age__70_80_: float = Field(0, ge=0, le=1, description="Age 70-80")
    age__80_90_: float = Field(0, ge=0, le=1, description="Age 80-90")
    age__90_100_: float = Field(0, ge=0, le=1, description="Age 90-100")
    
    # Weight categories
    weight__50_75_: float = Field(0, ge=0, le=1, description="Weight 50-75")
    weight__75_100_: float = Field(0, ge=0, le=1, description="Weight 75-100")
    weight__100_125_: float = Field(0, ge=0, le=1, description="Weight 100-125")
    weight__125_150_: float = Field(0, ge=0, le=1, description="Weight 125-150")
    weight__150_200_: float = Field(0, ge=0, le=1, description="Weight 150-200")
    weight__greater_200_: float = Field(0, ge=0, le=1, description="Weight >200")
    
    # Clinical metrics
    time_in_hospital: float = Field(..., ge=0, le=50, description="Time in hospital (days)")
    num_lab_procedures: float = Field(..., ge=0, le=200, description="Number of lab procedures")
    num_procedures: float = Field(..., ge=0, le=50, description="Number of procedures")
    num_medications: float = Field(..., ge=0, le=100, description="Number of medications")
    number_outpatient: float = Field(..., ge=0, le=100, description="Number of outpatient visits")
    number_emergency: float = Field(..., ge=0, le=100, description="Number of emergency visits")
    number_inpatient: float = Field(..., ge=0, le=100, description="Number of inpatient visits")
    number_diagnoses: float = Field(..., ge=0, le=20, description="Number of diagnoses")
    
    # Glucose and A1C levels
    max_glu_serum_norm: float = Field(0, ge=0, le=1, description="Max glucose serum normal")
    max_glu_serum__300_: float = Field(0, ge=0, le=1, description="Max glucose serum >300")
    max_glu_serum__200_: float = Field(0, ge=0, le=1, description="Max glucose serum >200")
    
    a1cresult_norm: float = Field(0, ge=0, le=1, description="A1C result normal")
    a1cresult__8_: float = Field(0, ge=0, le=1, description="A1C result >8")
    a1cresult__7_: float = Field(0, ge=0, le=1, description="A1C result >7")
    
    # Medications
    insulin_no: float = Field(0, ge=0, le=1, description="Insulin: No")
    insulin_steady: float = Field(0, ge=0, le=1, description="Insulin: Steady")
    insulin_up: float = Field(0, ge=0, le=1, description="Insulin: Up")
    insulin_down: float = Field(0, ge=0, le=1, description="Insulin: Down")
    
    diabetesmed_yes: float = Field(0, ge=0, le=1, description="Diabetes medication: Yes")
    diabetesmed_no: float = Field(0, ge=0, le=1, description="Diabetes medication: No")
    
    # Specific medications
    glipizide_no: float = Field(0, ge=0, le=1, description="Glipizide: No")
    glipizide_steady: float = Field(0, ge=0, le=1, description="Glipizide: Steady")
    glipizide_up: float = Field(0, ge=0, le=1, description="Glipizide: Up")
    glipizide_down: float = Field(0, ge=0, le=1, description="Glipizide: Down")
    
    # Change and discharge
    change_no: float = Field(0, ge=0, le=1, description="Change: No")
    change_ch: float = Field(0, ge=0, le=1, description="Change: Yes")
    
    # Payer codes
    payer_code_mc: float = Field(0, ge=0, le=1, description="Payer code: MC")
    payer_code_md: float = Field(0, ge=0, le=1, description="Payer code: MD")
    payer_code_hm: float = Field(0, ge=0, le=1, description="Payer code: HM")
    payer_code_un: float = Field(0, ge=0, le=1, description="Payer code: UN")
    payer_code_bc: float = Field(0, ge=0, le=1, description="Payer code: BC")
    payer_code_sp: float = Field(0, ge=0, le=1, description="Payer code: SP")
    payer_code_cp: float = Field(0, ge=0, le=1, description="Payer code: CP")
    payer_code_si: float = Field(0, ge=0, le=1, description="Payer code: SI")
    payer_code_dm: float = Field(0, ge=0, le=1, description="Payer code: DM")
    payer_code_cm: float = Field(0, ge=0, le=1, description="Payer code: CM")
    payer_code_ch: float = Field(0, ge=0, le=1, description="Payer code: CH")
    payer_code_po: float = Field(0, ge=0, le=1, description="Payer code: PO")
    payer_code_wc: float = Field(0, ge=0, le=1, description="Payer code: WC")
    payer_code_ot: float = Field(0, ge=0, le=1, description="Payer code: OT")
    payer_code_fr: float = Field(0, ge=0, le=1, description="Payer code: FR")
    payer_code_mp: float = Field(0, ge=0, le=1, description="Payer code: MP")
    payer_code_oa: float = Field(0, ge=0, le=1, description="Payer code: OA")
    
    # Medical specialties
    medical_specialty_emergency_trauma: float = Field(0, ge=0, le=1, description="Medical specialty: Emergency/Trauma")
    medical_specialty_internal_medicine: float = Field(0, ge=0, le=1, description="Medical specialty: Internal Medicine")
    medical_specialty_family_general_practice: float = Field(0, ge=0, le=1, description="Medical specialty: Family/General Practice")
    medical_specialty_cardiology: float = Field(0, ge=0, le=1, description="Medical specialty: Cardiology")
    medical_specialty_surgery_general: float = Field(0, ge=0, le=1, description="Medical specialty: Surgery-General")
    medical_specialty_orthopedics: float = Field(0, ge=0, le=1, description="Medical specialty: Orthopedics")
    medical_specialty_gastroenterology: float = Field(0, ge=0, le=1, description="Medical specialty: Gastroenterology")
    medical_specialty_nephrology: float = Field(0, ge=0, le=1, description="Medical specialty: Nephrology")
    medical_specialty_orthopedics_reconstructive: float = Field(0, ge=0, le=1, description="Medical specialty: Orthopedics-Reconstructive")
    medical_specialty_psychiatry: float = Field(0, ge=0, le=1, description="Medical specialty: Psychiatry")
    medical_specialty_pulmonology: float = Field(0, ge=0, le=1, description="Medical specialty: Pulmonology")
    medical_specialty_surgery_cardiovascular_thoracic: float = Field(0, ge=0, le=1, description="Medical specialty: Surgery-Cardiovascular/Thoracic")
    medical_specialty_surgery_vascular: float = Field(0, ge=0, le=1, description="Medical specialty: Surgery-Vascular")
    medical_specialty_obstetricsandgynecology: float = Field(0, ge=0, le=1, description="Medical specialty: Obstetrics&Gynecology")
    medical_specialty_urology: float = Field(0, ge=0, le=1, description="Medical specialty: Urology")
    medical_specialty_surgery_neuro: float = Field(0, ge=0, le=1, description="Medical specialty: Surgery-Neuro")
    medical_specialty_oncology: float = Field(0, ge=0, le=1, description="Medical specialty: Oncology")
    medical_specialty_otolaryngology: float = Field(0, ge=0, le=1, description="Medical specialty: Otolaryngology")
    medical_specialty_endocrinology: float = Field(0, ge=0, le=1, description="Medical specialty: Endocrinology")
    medical_specialty_missing: float = Field(0, ge=0, le=1, description="Medical specialty: Missing")
    
    # Diagnoses
    diag_3_428: float = Field(0, ge=0, le=1, description="Diagnosis 3: 428")
    diag_3_250_02: float = Field(0, ge=0, le=1, description="Diagnosis 3: 250.02")
    diag_3_276: float = Field(0, ge=0, le=1, description="Diagnosis 3: 276")
    diag_3_401: float = Field(0, ge=0, le=1, description="Diagnosis 3: 401")
    diag_3_496: float = Field(0, ge=0, le=1, description="Diagnosis 3: 496")
    diag_3_250_0: float = Field(0, ge=0, le=1, description="Diagnosis 3: 250.0")
    diag_3_427: float = Field(0, ge=0, le=1, description="Diagnosis 3: 427")
    diag_3_599: float = Field(0, ge=0, le=1, description="Diagnosis 3: 599")
    diag_3_584: float = Field(0, ge=0, le=1, description="Diagnosis 3: 584")
    diag_3_250_01: float = Field(0, ge=0, le=1, description="Diagnosis 3: 250.01")
    diag_3_272: float = Field(0, ge=0, le=1, description="Diagnosis 3: 272")
    diag_3_486: float = Field(0, ge=0, le=1, description="Diagnosis 3: 486")
    diag_3_518: float = Field(0, ge=0, le=1, description="Diagnosis 3: 518")
    diag_3_414: float = Field(0, ge=0, le=1, description="Diagnosis 3: 414")
    diag_3_250_03: float = Field(0, ge=0, le=1, description="Diagnosis 3: 250.03")
    diag_3_V45: float = Field(0, ge=0, le=1, description="Diagnosis 3: V45")
    diag_3_285: float = Field(0, ge=0, le=1, description="Diagnosis 3: 285")
    diag_3_other: float = Field(0, ge=0, le=1, description="Diagnosis 3: Other")
    
    @validator('*', pre=True)
    def validate_float(cls, v):
        """Convert input to float"""
        if v is None:
            return 0.0
        return float(v)
    
    class Config:
        extra = "forbid"


class PredictionResult(BaseModel):
    """Pydantic model for prediction results"""
    
    risk_score: float = Field(..., ge=0, le=1, description="Risk score (0-1)")
    risk_score_percent: float = Field(..., ge=0, le=100, description="Risk score as percentage")
    confidence_lower: float = Field(..., ge=0, le=1, description="Lower confidence interval")
    confidence_upper: float = Field(..., ge=0, le=1, description="Upper confidence interval")
    risk_level: str = Field(..., description="Risk level (Low/Medium/High)")
    model_used: str = Field(..., description="Model used for prediction")
    prediction_time_ms: float = Field(..., description="Prediction time in milliseconds")
    feature_importance: Dict[str, float] = Field(..., description="Feature importance scores")
    recommendations: List[str] = Field(..., description="Clinical recommendations")
    model_confidence: float = Field(..., ge=0, le=1, description="Model confidence score")
    
    class Config:
        extra = "forbid"


class HospitalReadmissionPredictionService:
    """
    Real-time hospital readmission prediction service with ensemble modeling
    and explainability features.
    """
    
    def __init__(self, model_path: str = "model_outputs", model_timestamp: str = "20250715_150347"):
        """
        Initialize the prediction service
        
        Args:
            model_path: Path to the model directory
            model_timestamp: Timestamp of the model version to load
        """
        self.model_path = Path(model_path)
        self.model_timestamp = model_timestamp
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_metadata = {}
        self.feature_names = None
        
        # Risk level thresholds
        self.risk_thresholds = {
            'low': 0.30,
            'medium': 0.70,
            'high': 1.00
        }
        
        # Load models and metadata
        self._load_models()
        self._load_feature_importance()
        self._load_metadata()
        
        print(f"Loaded {len(self.models)} models for prediction service")
    
    def _load_models(self):
        """Load trained models and scalers"""
        try:
            # Load individual models
            model_files = {
                'XGBoost': f'xgboost_{self.model_timestamp}.pkl',
                'Random Forest': f'random_forest_{self.model_timestamp}.pkl',
                'Logistic Regression': f'logistic_regression_{self.model_timestamp}.pkl',
                'Neural Network': f'neural_network_{self.model_timestamp}.pkl'
            }
            
            for model_name, filename in model_files.items():
                model_file = self.model_path / filename
                if model_file.exists():
                    self.models[model_name] = joblib.load(model_file)
                    print(f"Loaded {model_name} model")
                else:
                    print(f"Warning: {model_name} model file not found: {filename}")
            
            # Load scaler
            scaler_file = self.model_path / f'scaler_{self.model_timestamp}.pkl'
            if scaler_file.exists():
                self.scalers['standard'] = joblib.load(scaler_file)
                print("Loaded standard scaler")
            else:
                print("Warning: Scaler file not found")
                
        except Exception as e:
            raise RuntimeError(f"Error loading models: {str(e)}")
    
    def _load_feature_importance(self):
        """Load feature importance data"""
        try:
            importance_files = {
                'XGBoost': f'feature_importance_xgboost_{self.model_timestamp}.csv',
                'Random Forest': f'feature_importance_random_forest_{self.model_timestamp}.csv',
                'Logistic Regression': f'feature_importance_logistic_regression_{self.model_timestamp}.csv'
            }
            
            for model_name, filename in importance_files.items():
                importance_file = self.model_path / filename
                if importance_file.exists():
                    df = pd.read_csv(importance_file)
                    self.feature_importance[model_name] = dict(zip(df['feature'], df['importance']))
                    if self.feature_names is None:
                        self.feature_names = df['feature'].tolist()
                    print(f"Loaded feature importance for {model_name}")
                
        except Exception as e:
            print(f"Warning: Error loading feature importance: {str(e)}")
    
    def _load_metadata(self):
        """Load model metadata and performance metrics"""
        try:
            metadata_file = self.model_path / f'complete_results_{self.model_timestamp}.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.model_metadata = json.load(f)
                print("Loaded model metadata")
            else:
                print("Warning: Model metadata file not found")
                
        except Exception as e:
            print(f"Warning: Error loading metadata: {str(e)}")
    
    def _classify_risk_level(self, risk_score: float) -> str:
        """Classify risk level based on risk score"""
        if risk_score <= self.risk_thresholds['low']:
            return "Low"
        elif risk_score <= self.risk_thresholds['medium']:
            return "Medium"
        else:
            return "High"
    
    def _calculate_confidence_interval(self, risk_score: float, model_name: str) -> Tuple[float, float]:
        """Calculate confidence interval for prediction"""
        try:
            # Get model performance metrics
            if model_name in self.model_metadata.get('cv_results', {}):
                cv_results = self.model_metadata['cv_results'][model_name]
                accuracy_std = cv_results.get('accuracy_std', 0.05)
            else:
                accuracy_std = 0.05  # Default standard deviation
            
            # Calculate confidence interval using normal distribution
            confidence_level = 0.95
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            margin_of_error = z_score * accuracy_std
            
            lower_bound = max(0, risk_score - margin_of_error)
            upper_bound = min(1, risk_score + margin_of_error)
            
            return lower_bound, upper_bound
            
        except Exception as e:
            print(f"Warning: Error calculating confidence interval: {str(e)}")
            return max(0, risk_score - 0.1), min(1, risk_score + 0.1)
    
    def _get_feature_contribution(self, patient_data: Dict[str, float], model_name: str) -> Dict[str, float]:
        """Calculate feature contribution for explainability"""
        try:
            if model_name not in self.feature_importance:
                return {}
            
            importance_scores = self.feature_importance[model_name]
            
            # Calculate weighted contribution based on feature values and importance
            contributions = {}
            for feature, value in patient_data.items():
                if feature in importance_scores:
                    # Weight by both feature value and importance
                    contribution = value * importance_scores[feature]
                    contributions[feature] = contribution
            
            # Sort by contribution magnitude
            sorted_contributions = dict(sorted(contributions.items(), 
                                             key=lambda x: abs(x[1]), 
                                             reverse=True))
            
            # Return top 10 features
            return dict(list(sorted_contributions.items())[:10])
            
        except Exception as e:
            print(f"Warning: Error calculating feature contribution: {str(e)}")
            return {}
    
    def _generate_recommendations(self, risk_score: float, risk_level: str, 
                                feature_contributions: Dict[str, float]) -> List[str]:
        """Generate clinical recommendations based on prediction"""
        recommendations = []
        
        # Risk level specific recommendations
        if risk_level == "High":
            recommendations.append("Schedule immediate follow-up within 7 days")
            recommendations.append("Consider care coordination with case management")
            recommendations.append("Implement enhanced discharge planning")
            recommendations.append("Assess medication adherence and provide education")
        elif risk_level == "Medium":
            recommendations.append("Schedule follow-up within 14 days")
            recommendations.append("Provide detailed discharge instructions")
            recommendations.append("Consider telehealth monitoring")
        else:
            recommendations.append("Standard discharge planning")
            recommendations.append("Routine follow-up as per protocol")
        
        # Feature-specific recommendations
        top_features = list(feature_contributions.keys())[:3]
        
        if 'diabetesmed_yes' in top_features:
            recommendations.append("Diabetes management education and monitoring")
        
        if 'time_in_hospital' in top_features:
            recommendations.append("Review factors contributing to extended stay")
        
        if 'num_lab_procedures' in top_features:
            recommendations.append("Optimize laboratory monitoring protocols")
        
        if 'age__70_80_' in top_features or 'age__80_90_' in top_features:
            recommendations.append("Geriatric-specific care considerations")
        
        if 'number_emergency' in top_features:
            recommendations.append("Address factors leading to emergency visits")
        
        return recommendations
    
    def _prepare_features(self, patient_data: PatientData) -> np.ndarray:
        """Prepare features for model prediction"""
        # Convert to dictionary and then to array
        data_dict = patient_data.dict()
        
        # Ensure all expected features are present
        if self.feature_names:
            feature_array = np.array([data_dict.get(feature, 0.0) for feature in self.feature_names])
        else:
            # Fallback to all features from patient data
            feature_array = np.array(list(data_dict.values()))
        
        return feature_array.reshape(1, -1)
    
    def predict_single(self, patient_data: Union[Dict, PatientData], 
                      model_name: str = "XGBoost") -> PredictionResult:
        """
        Make prediction for a single patient
        
        Args:
            patient_data: Patient data as dictionary or PatientData object
            model_name: Name of the model to use for prediction
            
        Returns:
            PredictionResult object with prediction details
        """
        start_time = time.time()
        
        try:
            # Validate input data
            if isinstance(patient_data, dict):
                patient_data = PatientData(**patient_data)
            
            # Check if model exists
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not available. Available models: {list(self.models.keys())}")
            
            # Prepare features
            features = self._prepare_features(patient_data)
            
            # Scale features if needed (for Neural Network)
            if model_name == "Neural Network" and 'standard' in self.scalers:
                features_scaled = self.scalers['standard'].transform(features)
                prediction_features = features_scaled
            else:
                prediction_features = features
            
            # Make prediction
            model = self.models[model_name]
            risk_score = model.predict_proba(prediction_features)[0][1]  # Probability of readmission
            
            # Calculate confidence interval
            confidence_lower, confidence_upper = self._calculate_confidence_interval(risk_score, model_name)
            
            # Classify risk level
            risk_level = self._classify_risk_level(risk_score)
            
            # Get feature contributions
            feature_contributions = self._get_feature_contribution(patient_data.dict(), model_name)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(risk_score, risk_level, feature_contributions)
            
            # Calculate prediction time
            prediction_time_ms = (time.time() - start_time) * 1000
            
            # Model confidence based on feature importance coverage
            model_confidence = min(1.0, len(feature_contributions) / 10)  # Normalized by top 10 features
            
            return PredictionResult(
                risk_score=risk_score,
                risk_score_percent=risk_score * 100,
                confidence_lower=confidence_lower,
                confidence_upper=confidence_upper,
                risk_level=risk_level,
                model_used=model_name,
                prediction_time_ms=prediction_time_ms,
                feature_importance=feature_contributions,
                recommendations=recommendations,
                model_confidence=model_confidence
            )
            
        except Exception as e:
            raise RuntimeError(f"Error making prediction: {str(e)}")
    
    def predict_ensemble(self, patient_data: Union[Dict, PatientData], 
                        models: Optional[List[str]] = None) -> PredictionResult:
        """
        Make ensemble prediction using multiple models
        
        Args:
            patient_data: Patient data as dictionary or PatientData object
            models: List of model names to use for ensemble. If None, uses all available models
            
        Returns:
            PredictionResult object with ensemble prediction
        """
        start_time = time.time()
        
        try:
            # Validate input data
            if isinstance(patient_data, dict):
                patient_data = PatientData(**patient_data)
            
            # Select models for ensemble
            if models is None:
                models = list(self.models.keys())
            
            # Validate all models are available
            for model_name in models:
                if model_name not in self.models:
                    raise ValueError(f"Model '{model_name}' not available. Available models: {list(self.models.keys())}")
            
            # Prepare features
            features = self._prepare_features(patient_data)
            
            # Get predictions from all models
            predictions = []
            confidence_intervals = []
            feature_contributions_all = {}
            
            for model_name in models:
                # Scale features if needed
                if model_name == "Neural Network" and 'standard' in self.scalers:
                    features_scaled = self.scalers['standard'].transform(features)
                    prediction_features = features_scaled
                else:
                    prediction_features = features
                
                # Make prediction
                model = self.models[model_name]
                risk_score = model.predict_proba(prediction_features)[0][1]
                predictions.append(risk_score)
                
                # Calculate confidence interval
                ci_lower, ci_upper = self._calculate_confidence_interval(risk_score, model_name)
                confidence_intervals.append((ci_lower, ci_upper))
                
                # Get feature contributions
                contributions = self._get_feature_contribution(patient_data.dict(), model_name)
                feature_contributions_all[model_name] = contributions
            
            # Calculate ensemble prediction (weighted average)
            # Weight by model performance if available
            weights = []
            for model_name in models:
                if model_name in self.model_metadata.get('performance_metrics', {}):
                    # Use test ROC AUC as weight
                    roc_auc = self.model_metadata['performance_metrics'][model_name].get('test_roc_auc', 0.5)
                    weights.append(max(0.1, roc_auc))  # Minimum weight of 0.1
                else:
                    weights.append(1.0)  # Default weight
            
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Ensemble prediction
            ensemble_risk_score = sum(pred * weight for pred, weight in zip(predictions, weights))
            
            # Ensemble confidence interval
            ensemble_ci_lower = sum(ci[0] * weight for ci, weight in zip(confidence_intervals, weights))
            ensemble_ci_upper = sum(ci[1] * weight for ci, weight in zip(confidence_intervals, weights))
            
            # Aggregate feature contributions
            ensemble_contributions = {}
            for model_name, contributions in feature_contributions_all.items():
                model_weight = weights[models.index(model_name)]
                for feature, contribution in contributions.items():
                    if feature in ensemble_contributions:
                        ensemble_contributions[feature] += contribution * model_weight
                    else:
                        ensemble_contributions[feature] = contribution * model_weight
            
            # Sort ensemble contributions
            ensemble_contributions = dict(sorted(ensemble_contributions.items(), 
                                               key=lambda x: abs(x[1]), 
                                               reverse=True))
            
            # Classify risk level
            risk_level = self._classify_risk_level(ensemble_risk_score)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(ensemble_risk_score, risk_level, ensemble_contributions)
            
            # Calculate prediction time
            prediction_time_ms = (time.time() - start_time) * 1000
            
            # Model confidence based on agreement between models
            prediction_std = np.std(predictions)
            model_confidence = max(0.1, 1.0 - prediction_std)  # Higher confidence with lower std
            
            return PredictionResult(
                risk_score=ensemble_risk_score,
                risk_score_percent=ensemble_risk_score * 100,
                confidence_lower=ensemble_ci_lower,
                confidence_upper=ensemble_ci_upper,
                risk_level=risk_level,
                model_used=f"Ensemble({', '.join(models)})",
                prediction_time_ms=prediction_time_ms,
                feature_importance=dict(list(ensemble_contributions.items())[:10]),
                recommendations=recommendations,
                model_confidence=model_confidence
            )
            
        except Exception as e:
            raise RuntimeError(f"Error making ensemble prediction: {str(e)}")
    
    def predict_batch(self, patient_data_list: List[Union[Dict, PatientData]], 
                     model_name: str = "XGBoost") -> List[PredictionResult]:
        """
        Make predictions for a batch of patients
        
        Args:
            patient_data_list: List of patient data
            model_name: Name of the model to use for prediction
            
        Returns:
            List of PredictionResult objects
        """
        results = []
        
        for patient_data in patient_data_list:
            try:
                result = self.predict_single(patient_data, model_name)
                results.append(result)
            except Exception as e:
                print(f"Error predicting for patient: {str(e)}")
                # Continue with other patients
                continue
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            'available_models': list(self.models.keys()),
            'model_timestamp': self.model_timestamp,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'risk_thresholds': self.risk_thresholds
        }
        
        if self.model_metadata:
            info['performance_metrics'] = self.model_metadata.get('performance_metrics', {})
            info['cv_results'] = self.model_metadata.get('cv_results', {})
        
        return info
    
    def validate_patient_data(self, patient_data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate patient data format
        
        Args:
            patient_data: Patient data dictionary
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            PatientData(**patient_data)
            return True, []
        except Exception as e:
            return False, [str(e)]


# Example usage and testing
if __name__ == "__main__":
    # Initialize the service
    service = HospitalReadmissionPredictionService()
    
    # Example patient data
    example_patient = {
        'age_numeric': 65.0,
        'race_caucasian': 1.0,
        'age__60_70_': 1.0,
        'weight__100_125_': 1.0,
        'time_in_hospital': 3.0,
        'num_lab_procedures': 45.0,
        'num_procedures': 2.0,
        'num_medications': 15.0,
        'number_outpatient': 1.0,
        'number_emergency': 0.0,
        'number_inpatient': 0.0,
        'number_diagnoses': 8.0,
        'diabetesmed_yes': 1.0,
        'insulin_no': 1.0,
        'glipizide_up': 1.0,
        'max_glu_serum_norm': 1.0,
        'a1cresult_norm': 1.0,
        'change_no': 1.0,
        'payer_code_mc': 1.0,
        'medical_specialty_emergency_trauma': 1.0,
        'diag_3_428': 1.0
    }
    
    # Test single prediction
    print("Testing single prediction...")
    result = service.predict_single(example_patient, "XGBoost")
    print(f"Risk Score: {result.risk_score_percent:.1f}%")
    print(f"Risk Level: {result.risk_level}")
    print(f"Prediction Time: {result.prediction_time_ms:.2f}ms")
    print(f"Top Features: {list(result.feature_importance.keys())[:5]}")
    
    # Test ensemble prediction
    print("\nTesting ensemble prediction...")
    ensemble_result = service.predict_ensemble(example_patient)
    print(f"Ensemble Risk Score: {ensemble_result.risk_score_percent:.1f}%")
    print(f"Ensemble Risk Level: {ensemble_result.risk_level}")
    print(f"Ensemble Prediction Time: {ensemble_result.prediction_time_ms:.2f}ms")
    
    # Get model info
    print("\nModel Information:")
    model_info = service.get_model_info()
    print(f"Available Models: {model_info['available_models']}")
    print(f"Feature Count: {model_info['feature_count']}")