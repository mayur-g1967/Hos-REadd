# Hospital Readmission Prediction Service Documentation

## Overview

The Hospital Readmission Prediction Service is a real-time machine learning service designed to assess the risk of hospital readmission for patients. It provides predictions using multiple ML models including XGBoost, Random Forest, Logistic Regression, and Neural Networks, with ensemble capabilities for improved accuracy.

## Features

### Core Features
- **Multi-Model Support**: XGBoost, Random Forest, Logistic Regression, Neural Network
- **Ensemble Predictions**: Combine multiple models for improved accuracy
- **Real-time Processing**: <200ms response time for single predictions
- **Batch Processing**: Process multiple patients simultaneously
- **Confidence Intervals**: Statistical confidence bounds for predictions
- **Risk Classification**: Automatic Low/Medium/High risk level assignment
- **Feature Importance**: Explainable AI with feature contribution analysis
- **Data Validation**: Comprehensive input validation using Pydantic
- **Model Versioning**: Support for different model versions

### Risk Level Classification
- **Low Risk**: 0-30% readmission probability
- **Medium Risk**: 31-70% readmission probability
- **High Risk**: 71-100% readmission probability

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
xgboost>=1.6.0
joblib>=1.2.0
matplotlib>=3.5.0
seaborn>=0.11.0
pydantic>=2.0.0
scipy>=1.9.0
fastapi>=0.104.0
uvicorn>=0.24.0
```

## Usage

### 1. Basic Python Usage

```python
from hospital_readmission_prediction_service import HospitalReadmissionPredictionService

# Initialize the service
service = HospitalReadmissionPredictionService()

# Example patient data
patient_data = {
    'age_numeric': 65.0,
    'race_caucasian': 1.0,
    'time_in_hospital': 4.0,
    'num_lab_procedures': 45.0,
    'num_medications': 12.0,
    'diabetesmed_yes': 1.0,
    # ... include all required features
}

# Single prediction
result = service.predict_single(patient_data, model_name="XGBoost")
print(f"Risk Score: {result.risk_score_percent:.1f}%")
print(f"Risk Level: {result.risk_level}")

# Ensemble prediction
ensemble_result = service.predict_ensemble(patient_data)
print(f"Ensemble Risk Score: {ensemble_result.risk_score_percent:.1f}%")
```

### 2. REST API Usage

#### Start the API Server
```bash
python prediction_api.py --host 0.0.0.0 --port 8000
```

#### API Documentation
- Interactive API docs: `http://localhost:8000/docs`
- ReDoc documentation: `http://localhost:8000/redoc`

#### API Endpoints

##### Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

##### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_data": {
      "age_numeric": 65.0,
      "race_caucasian": 1.0,
      "time_in_hospital": 4.0,
      "num_lab_procedures": 45.0,
      "num_medications": 12.0,
      "diabetesmed_yes": 1.0
    },
    "model_name": "XGBoost"
  }'
```

##### Ensemble Prediction
```bash
curl -X POST "http://localhost:8000/predict/ensemble" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_data": {
      "age_numeric": 65.0,
      "race_caucasian": 1.0,
      "time_in_hospital": 4.0,
      "num_lab_procedures": 45.0,
      "num_medications": 12.0,
      "diabetesmed_yes": 1.0
    },
    "models": ["XGBoost", "Random Forest"]
  }'
```

##### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_data_list": [
      {"age_numeric": 65.0, "time_in_hospital": 4.0, ...},
      {"age_numeric": 45.0, "time_in_hospital": 2.0, ...}
    ],
    "model_name": "XGBoost"
  }'
```

### 3. Example Usage Script

Run the comprehensive example:
```bash
python prediction_service_example.py
```

This will demonstrate:
- Single patient predictions with all models
- Ensemble predictions
- Batch processing
- Performance testing
- Data validation

## Data Format

### Required Features (55 total)

#### Demographics
- `age_numeric`: Patient age (0-150)
- `race_caucasian`, `race_asian`, `race_african_american`, `race_hispanic`, `race_other`: Race indicators (0-1)

#### Age Categories
- `age__10_20_`, `age__20_30_`, `age__30_40_`, `age__40_50_`, `age__50_60_`, `age__60_70_`, `age__70_80_`, `age__80_90_`, `age__90_100_`: Age category indicators (0-1)

#### Weight Categories
- `weight__50_75_`, `weight__75_100_`, `weight__100_125_`, `weight__125_150_`, `weight__150_200_`, `weight__greater_200_`: Weight category indicators (0-1)

#### Clinical Metrics
- `time_in_hospital`: Length of stay in days (0-50)
- `num_lab_procedures`: Number of lab procedures (0-200)
- `num_procedures`: Number of procedures (0-50)
- `num_medications`: Number of medications (0-100)
- `number_outpatient`: Outpatient visits (0-100)
- `number_emergency`: Emergency visits (0-100)
- `number_inpatient`: Inpatient visits (0-100)
- `number_diagnoses`: Number of diagnoses (0-20)

#### Lab Results
- `max_glu_serum_norm`, `max_glu_serum__300_`, `max_glu_serum__200_`: Glucose levels (0-1)
- `a1cresult_norm`, `a1cresult__8_`, `a1cresult__7_`: A1C results (0-1)

#### Medications
- `insulin_no`, `insulin_steady`, `insulin_up`, `insulin_down`: Insulin status (0-1)
- `diabetesmed_yes`, `diabetesmed_no`: Diabetes medication status (0-1)
- `glipizide_no`, `glipizide_steady`, `glipizide_up`, `glipizide_down`: Glipizide status (0-1)

#### Other Clinical
- `change_no`, `change_ch`: Medication changes (0-1)
- `payer_code_*`: Insurance payer codes (0-1)
- `medical_specialty_*`: Medical specialties (0-1)
- `diag_3_*`: Diagnosis codes (0-1)

## Output Format

### PredictionResult Object
```python
{
    "risk_score": 0.75,                    # Risk score (0-1)
    "risk_score_percent": 75.0,            # Risk score as percentage
    "confidence_lower": 0.68,              # Lower confidence interval
    "confidence_upper": 0.82,              # Upper confidence interval
    "risk_level": "High",                  # Risk level (Low/Medium/High)
    "model_used": "XGBoost",               # Model used for prediction
    "prediction_time_ms": 45.2,            # Prediction time in milliseconds
    "feature_importance": {                 # Top contributing features
        "diabetesmed_yes": 0.023,
        "age__70_80_": 0.019,
        "time_in_hospital": 0.016
    },
    "recommendations": [                    # Clinical recommendations
        "Schedule immediate follow-up within 7 days",
        "Consider care coordination with case management",
        "Diabetes management education and monitoring"
    ],
    "model_confidence": 0.89               # Model confidence score
}
```

## Performance Characteristics

### Speed Requirements
- Single prediction: <200ms
- Batch processing: <50ms per patient
- Ensemble prediction: <300ms

### Accuracy Metrics
Based on test set performance:

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| XGBoost | 88.7% | 90.0% | 88.7% | 84.1% | 66.3% |
| Random Forest | 87.8% | 77.1% | 87.8% | 82.1% | 62.6% |
| Logistic Regression | 87.8% | 77.1% | 87.8% | 82.1% | 66.5% |
| Neural Network | 87.0% | 83.1% | 87.0% | 84.1% | 77.7% |

### Model Recommendations
- **XGBoost**: Best overall performance, recommended for most use cases
- **Neural Network**: Highest ROC AUC, good for probability calibration
- **Ensemble**: Combines strengths of all models, most robust predictions

## Error Handling

### Common Errors
1. **Missing Features**: Ensure all required features are provided
2. **Invalid Values**: Check data types and ranges
3. **Model Loading**: Verify model files exist in `model_outputs/`
4. **Memory Issues**: For large batches, process in smaller chunks

### Validation
The service includes comprehensive validation:
- Data type validation
- Range validation
- Required field validation
- Schema validation using Pydantic

## Integration Guidelines

### Web Application Integration
1. Use the REST API for web-based applications
2. Implement proper error handling
3. Cache predictions when appropriate
4. Monitor API performance and availability

### Batch Processing
1. Process patients in batches of 10-100 for optimal performance
2. Implement retry logic for failed predictions
3. Consider async processing for large datasets

### Real-time Integration
1. Use single prediction endpoint for real-time scoring
2. Implement connection pooling for high-volume applications
3. Monitor response times and scale as needed

## Clinical Recommendations

### Risk Level Actions
- **High Risk (71-100%)**:
  - Schedule immediate follow-up within 7 days
  - Implement enhanced discharge planning
  - Consider care coordination with case management
  - Assess medication adherence

- **Medium Risk (31-70%)**:
  - Schedule follow-up within 14 days
  - Provide detailed discharge instructions
  - Consider telehealth monitoring

- **Low Risk (0-30%)**:
  - Standard discharge planning
  - Routine follow-up as per protocol

### Feature-Specific Interventions
- **Diabetes Management**: Enhanced education and monitoring
- **Extended Stay**: Review contributing factors
- **Multiple Lab Procedures**: Optimize monitoring protocols
- **Elderly Patients**: Geriatric-specific care considerations

## Monitoring and Maintenance

### Performance Monitoring
- Track prediction response times
- Monitor model accuracy over time
- Alert on service failures

### Model Updates
- Retrain models quarterly with new data
- Validate model performance before deployment
- Maintain model versioning

### Data Quality
- Monitor input data quality
- Implement data drift detection
- Regular validation of feature distributions

## Troubleshooting

### Common Issues
1. **Service Won't Start**: Check model files and dependencies
2. **Slow Predictions**: Monitor system resources and optimize batch sizes
3. **Inaccurate Predictions**: Verify input data quality and feature encoding
4. **Memory Issues**: Reduce batch sizes or increase system memory

### Debugging
1. Enable debug logging
2. Check input data validation
3. Verify model loading
4. Monitor system resources

## Support and Contact

For technical support or questions:
- Review API documentation at `/docs`
- Check error logs for detailed error messages
- Validate input data format using the `/validate` endpoint
- Monitor service health using the `/health` endpoint

## Version History

- **v1.0.0**: Initial release with multi-model support and REST API
- Model timestamp: 20250715_150347
- Feature count: 78 features
- Supported models: XGBoost, Random Forest, Logistic Regression, Neural Network