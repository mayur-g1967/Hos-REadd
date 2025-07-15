#!/usr/bin/env python3
"""
Hospital Readmission Prediction API
===================================

A REST API wrapper for the hospital readmission prediction service
using FastAPI for easy integration with web applications.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import uvicorn
import json
import time
from datetime import datetime

from hospital_readmission_prediction_service import (
    HospitalReadmissionPredictionService,
    PatientData,
    PredictionResult
)

# Initialize FastAPI app
app = FastAPI(
    title="Hospital Readmission Prediction API",
    description="Real-time hospital readmission risk assessment API with ML models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction service
prediction_service = None

# API Models
class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    patient_data: Dict[str, float] = Field(..., description="Patient data dictionary")
    model_name: Optional[str] = Field("XGBoost", description="Model to use for prediction")

class EnsemblePredictionRequest(BaseModel):
    """Request model for ensemble prediction"""
    patient_data: Dict[str, float] = Field(..., description="Patient data dictionary")
    models: Optional[List[str]] = Field(None, description="List of models to use in ensemble")

class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction"""
    patient_data_list: List[Dict[str, float]] = Field(..., description="List of patient data dictionaries")
    model_name: Optional[str] = Field("XGBoost", description="Model to use for prediction")

class APIResponse(BaseModel):
    """Standard API response model"""
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Response message")
    data: Optional[Any] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message if any")
    timestamp: str = Field(..., description="Response timestamp")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    models_loaded: int = Field(..., description="Number of models loaded")
    timestamp: str = Field(..., description="Response timestamp")

# Initialize service on startup
@app.on_event("startup")
async def startup_event():
    """Initialize prediction service on startup"""
    global prediction_service
    try:
        prediction_service = HospitalReadmissionPredictionService()
        print("Prediction service initialized successfully")
    except Exception as e:
        print(f"Error initializing prediction service: {str(e)}")
        raise

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service not initialized"
        )
    
    model_info = prediction_service.get_model_info()
    
    return HealthResponse(
        status="healthy",
        models_loaded=len(model_info['available_models']),
        timestamp=datetime.now().isoformat()
    )

@app.get("/models", response_model=APIResponse)
async def get_model_info():
    """Get information about available models"""
    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service not initialized"
        )
    
    try:
        model_info = prediction_service.get_model_info()
        
        return APIResponse(
            success=True,
            message="Model information retrieved successfully",
            data=model_info,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving model information: {str(e)}"
        )

@app.post("/predict", response_model=APIResponse)
async def predict_single(request: PredictionRequest):
    """Make prediction for a single patient"""
    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service not initialized"
        )
    
    try:
        # Validate patient data
        is_valid, errors = prediction_service.validate_patient_data(request.patient_data)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid patient data: {'; '.join(errors)}"
            )
        
        # Make prediction
        result = prediction_service.predict_single(
            patient_data=request.patient_data,
            model_name=request.model_name
        )
        
        return APIResponse(
            success=True,
            message="Prediction completed successfully",
            data=result.dict(),
            timestamp=datetime.now().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making prediction: {str(e)}"
        )

@app.post("/predict/ensemble", response_model=APIResponse)
async def predict_ensemble(request: EnsemblePredictionRequest):
    """Make ensemble prediction using multiple models"""
    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service not initialized"
        )
    
    try:
        # Validate patient data
        is_valid, errors = prediction_service.validate_patient_data(request.patient_data)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid patient data: {'; '.join(errors)}"
            )
        
        # Make ensemble prediction
        result = prediction_service.predict_ensemble(
            patient_data=request.patient_data,
            models=request.models
        )
        
        return APIResponse(
            success=True,
            message="Ensemble prediction completed successfully",
            data=result.dict(),
            timestamp=datetime.now().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making ensemble prediction: {str(e)}"
        )

@app.post("/predict/batch", response_model=APIResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make predictions for a batch of patients"""
    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service not initialized"
        )
    
    try:
        # Validate batch size
        if len(request.patient_data_list) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch size exceeds maximum limit of 100 patients"
            )
        
        # Validate each patient data
        for i, patient_data in enumerate(request.patient_data_list):
            is_valid, errors = prediction_service.validate_patient_data(patient_data)
            if not is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid patient data at index {i}: {'; '.join(errors)}"
                )
        
        # Make batch predictions
        results = prediction_service.predict_batch(
            patient_data_list=request.patient_data_list,
            model_name=request.model_name
        )
        
        # Convert results to dictionaries
        results_dict = [result.dict() for result in results]
        
        return APIResponse(
            success=True,
            message=f"Batch prediction completed successfully for {len(results)} patients",
            data={
                "predictions": results_dict,
                "total_patients": len(results),
                "successful_predictions": len(results)
            },
            timestamp=datetime.now().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making batch prediction: {str(e)}"
        )

@app.post("/validate", response_model=APIResponse)
async def validate_patient_data(patient_data: Dict[str, float]):
    """Validate patient data format"""
    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service not initialized"
        )
    
    try:
        is_valid, errors = prediction_service.validate_patient_data(patient_data)
        
        return APIResponse(
            success=True,
            message="Data validation completed",
            data={
                "is_valid": is_valid,
                "errors": errors
            },
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error validating patient data: {str(e)}"
        )

@app.get("/schema", response_model=APIResponse)
async def get_patient_schema():
    """Get the patient data schema"""
    try:
        schema = PatientData.schema()
        
        return APIResponse(
            success=True,
            message="Patient data schema retrieved successfully",
            data=schema,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving schema: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Hospital Readmission Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return APIResponse(
        success=False,
        message="Request failed",
        error=exc.detail,
        timestamp=datetime.now().isoformat()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    return APIResponse(
        success=False,
        message="Internal server error",
        error=str(exc),
        timestamp=datetime.now().isoformat()
    )

# Helper function to run the server
def run_server(host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
    """Run the API server"""
    uvicorn.run(
        "prediction_api:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hospital Readmission Prediction API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    print(f"Starting Hospital Readmission Prediction API server on {args.host}:{args.port}")
    print(f"API documentation available at: http://{args.host}:{args.port}/docs")
    
    run_server(host=args.host, port=args.port, debug=args.debug)