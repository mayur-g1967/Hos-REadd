#!/usr/bin/env python3
"""
Hospital Readmission Prediction Service - Example Usage
=======================================================

This script demonstrates how to use the HospitalReadmissionPredictionService
with various features including single predictions, ensemble predictions,
batch processing, and performance testing.
"""

import json
import time
from typing import Dict, List
from hospital_readmission_prediction_service import (
    HospitalReadmissionPredictionService,
    PatientData,
    PredictionResult
)

def create_sample_patients() -> List[Dict]:
    """Create sample patient data for testing"""
    
    # High-risk patient (elderly, diabetic, multiple comorbidities)
    high_risk_patient = {
        'age_numeric': 78.0,
        'race_caucasian': 1.0,
        'age__70_80_': 1.0,
        'weight__100_125_': 1.0,
        'time_in_hospital': 7.0,
        'num_lab_procedures': 65.0,
        'num_procedures': 4.0,
        'num_medications': 25.0,
        'number_outpatient': 3.0,
        'number_emergency': 2.0,
        'number_inpatient': 1.0,
        'number_diagnoses': 12.0,
        'diabetesmed_yes': 1.0,
        'insulin_up': 1.0,
        'glipizide_up': 1.0,
        'max_glu_serum__300_': 1.0,
        'a1cresult__8_': 1.0,
        'change_ch': 1.0,
        'payer_code_mc': 1.0,
        'medical_specialty_emergency_trauma': 1.0,
        'diag_3_428': 1.0,  # Heart failure
        'diag_3_250_02': 1.0  # Diabetes
    }
    
    # Medium-risk patient (middle-aged, some complications)
    medium_risk_patient = {
        'age_numeric': 55.0,
        'race_caucasian': 1.0,
        'age__50_60_': 1.0,
        'weight__75_100_': 1.0,
        'time_in_hospital': 4.0,
        'num_lab_procedures': 35.0,
        'num_procedures': 2.0,
        'num_medications': 12.0,
        'number_outpatient': 1.0,
        'number_emergency': 0.0,
        'number_inpatient': 0.0,
        'number_diagnoses': 6.0,
        'diabetesmed_yes': 1.0,
        'insulin_no': 1.0,
        'glipizide_steady': 1.0,
        'max_glu_serum_norm': 1.0,
        'a1cresult_norm': 1.0,
        'change_no': 1.0,
        'payer_code_bc': 1.0,
        'medical_specialty_internal_medicine': 1.0,
        'diag_3_250_0': 1.0  # Diabetes
    }
    
    # Low-risk patient (young, minimal complications)
    low_risk_patient = {
        'age_numeric': 35.0,
        'race_caucasian': 1.0,
        'age__30_40_': 1.0,
        'weight__75_100_': 1.0,
        'time_in_hospital': 2.0,
        'num_lab_procedures': 15.0,
        'num_procedures': 1.0,
        'num_medications': 5.0,
        'number_outpatient': 0.0,
        'number_emergency': 0.0,
        'number_inpatient': 0.0,
        'number_diagnoses': 3.0,
        'diabetesmed_no': 1.0,
        'insulin_no': 1.0,
        'glipizide_no': 1.0,
        'max_glu_serum_norm': 1.0,
        'a1cresult_norm': 1.0,
        'change_no': 1.0,
        'payer_code_bc': 1.0,
        'medical_specialty_family_general_practice': 1.0,
        'diag_3_other': 1.0
    }
    
    return [
        ("High Risk", high_risk_patient),
        ("Medium Risk", medium_risk_patient),
        ("Low Risk", low_risk_patient)
    ]

def demonstrate_single_predictions(service: HospitalReadmissionPredictionService):
    """Demonstrate single patient predictions with different models"""
    
    print("="*80)
    print("SINGLE PATIENT PREDICTIONS")
    print("="*80)
    
    patients = create_sample_patients()
    
    for patient_type, patient_data in patients:
        print(f"\n{patient_type} Patient Analysis:")
        print("-" * 50)
        
        # Test all available models
        for model_name in service.get_model_info()['available_models']:
            try:
                result = service.predict_single(patient_data, model_name)
                
                print(f"\n{model_name} Model:")
                print(f"  Risk Score: {result.risk_score_percent:.1f}%")
                print(f"  Risk Level: {result.risk_level}")
                print(f"  Confidence: {result.confidence_lower:.2f} - {result.confidence_upper:.2f}")
                print(f"  Prediction Time: {result.prediction_time_ms:.2f}ms")
                print(f"  Model Confidence: {result.model_confidence:.2f}")
                
                # Show top 3 contributing features
                top_features = list(result.feature_importance.items())[:3]
                print(f"  Top Features: {[f'{k}: {v:.3f}' for k, v in top_features]}")
                
            except Exception as e:
                print(f"  Error with {model_name}: {str(e)}")

def demonstrate_ensemble_predictions(service: HospitalReadmissionPredictionService):
    """Demonstrate ensemble predictions"""
    
    print("\n" + "="*80)
    print("ENSEMBLE PREDICTIONS")
    print("="*80)
    
    patients = create_sample_patients()
    
    for patient_type, patient_data in patients:
        print(f"\n{patient_type} Patient - Ensemble Analysis:")
        print("-" * 50)
        
        try:
            # Full ensemble (all models)
            ensemble_result = service.predict_ensemble(patient_data)
            
            print(f"Ensemble Result:")
            print(f"  Risk Score: {ensemble_result.risk_score_percent:.1f}%")
            print(f"  Risk Level: {ensemble_result.risk_level}")
            print(f"  Confidence Interval: {ensemble_result.confidence_lower:.2f} - {ensemble_result.confidence_upper:.2f}")
            print(f"  Model Used: {ensemble_result.model_used}")
            print(f"  Prediction Time: {ensemble_result.prediction_time_ms:.2f}ms")
            print(f"  Model Confidence: {ensemble_result.model_confidence:.2f}")
            
            # Show top 5 contributing features
            top_features = list(ensemble_result.feature_importance.items())[:5]
            print(f"  Top Features:")
            for feature, importance in top_features:
                print(f"    {feature}: {importance:.3f}")
            
            # Show recommendations
            print(f"  Recommendations:")
            for i, rec in enumerate(ensemble_result.recommendations[:3], 1):
                print(f"    {i}. {rec}")
            
        except Exception as e:
            print(f"  Error with ensemble: {str(e)}")

def demonstrate_batch_predictions(service: HospitalReadmissionPredictionService):
    """Demonstrate batch predictions"""
    
    print("\n" + "="*80)
    print("BATCH PREDICTIONS")
    print("="*80)
    
    # Create batch of patients
    batch_data = [patient_data for _, patient_data in create_sample_patients()]
    
    print(f"Processing batch of {len(batch_data)} patients...")
    
    start_time = time.time()
    batch_results = service.predict_batch(batch_data, "XGBoost")
    batch_time = time.time() - start_time
    
    print(f"Batch processing completed in {batch_time:.3f} seconds")
    print(f"Average time per patient: {(batch_time / len(batch_results)):.3f} seconds")
    
    print("\nBatch Results Summary:")
    print("-" * 30)
    
    risk_levels = {'Low': 0, 'Medium': 0, 'High': 0}
    total_time = 0
    
    for i, result in enumerate(batch_results):
        risk_levels[result.risk_level] += 1
        total_time += result.prediction_time_ms
        
        print(f"Patient {i+1}: {result.risk_score_percent:.1f}% ({result.risk_level})")
    
    print(f"\nRisk Distribution:")
    for level, count in risk_levels.items():
        print(f"  {level}: {count} patients")
    
    print(f"Average prediction time: {total_time / len(batch_results):.2f}ms")

def performance_test(service: HospitalReadmissionPredictionService):
    """Test prediction performance and speed"""
    
    print("\n" + "="*80)
    print("PERFORMANCE TEST")
    print("="*80)
    
    # Create test patient
    test_patient = create_sample_patients()[0][1]  # Use high-risk patient
    
    # Test single prediction speed
    print("Testing single prediction speed...")
    
    times = []
    for i in range(100):
        start = time.time()
        result = service.predict_single(test_patient, "XGBoost")
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"100 predictions statistics:")
    print(f"  Average time: {avg_time:.2f}ms")
    print(f"  Min time: {min_time:.2f}ms")
    print(f"  Max time: {max_time:.2f}ms")
    print(f"  Target (<200ms): {'✓ PASSED' if avg_time < 200 else '✗ FAILED'}")
    
    # Test ensemble prediction speed
    print("\nTesting ensemble prediction speed...")
    
    start = time.time()
    ensemble_result = service.predict_ensemble(test_patient)
    end = time.time()
    ensemble_time = (end - start) * 1000
    
    print(f"Ensemble prediction time: {ensemble_time:.2f}ms")
    print(f"Target (<200ms): {'✓ PASSED' if ensemble_time < 200 else '✗ FAILED'}")

def validate_data_formats(service: HospitalReadmissionPredictionService):
    """Test data validation functionality"""
    
    print("\n" + "="*80)
    print("DATA VALIDATION TEST")
    print("="*80)
    
    # Test valid data
    valid_patient = create_sample_patients()[0][1]
    is_valid, errors = service.validate_patient_data(valid_patient)
    print(f"Valid patient data: {'✓ PASSED' if is_valid else '✗ FAILED'}")
    if errors:
        print(f"  Errors: {errors}")
    
    # Test invalid data
    invalid_cases = [
        ("Missing required field", {'age_numeric': 50.0}),  # Missing time_in_hospital
        ("Invalid age", {'age_numeric': 200.0, 'time_in_hospital': 3.0}),  # Age too high
        ("Invalid time", {'age_numeric': 50.0, 'time_in_hospital': -1.0}),  # Negative time
        ("Invalid binary", {'age_numeric': 50.0, 'time_in_hospital': 3.0, 'race_caucasian': 2.0})  # Binary > 1
    ]
    
    for case_name, invalid_data in invalid_cases:
        is_valid, errors = service.validate_patient_data(invalid_data)
        print(f"{case_name}: {'✗ FAILED' if not is_valid else '✓ UNEXPECTED PASS'}")
        if errors:
            print(f"  Errors: {errors[:2]}...")  # Show first 2 errors

def display_model_information(service: HospitalReadmissionPredictionService):
    """Display comprehensive model information"""
    
    print("\n" + "="*80)
    print("MODEL INFORMATION")
    print("="*80)
    
    model_info = service.get_model_info()
    
    print(f"Available Models: {model_info['available_models']}")
    print(f"Model Version: {model_info['model_timestamp']}")
    print(f"Feature Count: {model_info['feature_count']}")
    print(f"Risk Thresholds: {model_info['risk_thresholds']}")
    
    # Performance metrics
    if 'performance_metrics' in model_info:
        print(f"\nModel Performance (Test Set):")
        print("-" * 40)
        
        for model_name, metrics in model_info['performance_metrics'].items():
            print(f"\n{model_name}:")
            print(f"  Accuracy: {metrics.get('test_accuracy', 0):.3f}")
            print(f"  Precision: {metrics.get('test_precision', 0):.3f}")
            print(f"  Recall: {metrics.get('test_recall', 0):.3f}")
            print(f"  F1 Score: {metrics.get('test_f1', 0):.3f}")
            print(f"  ROC AUC: {metrics.get('test_roc_auc', 0):.3f}")
    
    # Cross-validation results
    if 'cv_results' in model_info:
        print(f"\nCross-Validation Results:")
        print("-" * 40)
        
        for model_name, cv_metrics in model_info['cv_results'].items():
            print(f"\n{model_name}:")
            print(f"  CV Accuracy: {cv_metrics.get('accuracy_mean', 0):.3f} ± {cv_metrics.get('accuracy_std', 0):.3f}")
            print(f"  CV ROC AUC: {cv_metrics.get('roc_auc_mean', 0):.3f} ± {cv_metrics.get('roc_auc_std', 0):.3f}")

def main():
    """Main function to run all demonstrations"""
    
    print("Hospital Readmission Prediction Service - Comprehensive Demo")
    print("=" * 80)
    
    try:
        # Initialize the service
        print("Initializing prediction service...")
        service = HospitalReadmissionPredictionService()
        
        # Display model information
        display_model_information(service)
        
        # Run demonstrations
        demonstrate_single_predictions(service)
        demonstrate_ensemble_predictions(service)
        demonstrate_batch_predictions(service)
        
        # Performance and validation tests
        performance_test(service)
        validate_data_formats(service)
        
        print("\n" + "="*80)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        print(f"\nError during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()