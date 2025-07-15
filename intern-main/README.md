# Diabetes 130-US Hospitals Dataset Processor

A comprehensive Python script to process the Kaggle "Diabetes 130-US hospitals" dataset for hospital readmission prediction with ML-ready output and Supabase compatibility.

## Features

- **Data Loading & Exploration**: Load and analyze 101,766+ patient records with 55 features
- **Data Cleaning**: Handle missing values, duplicates, and data quality issues
- **Target Creation**: Create binary target variable for 30-day readmission prediction
- **Missing Value Handling**: Multiple imputation strategies (mode, median, KNN)
- **Outlier Detection**: IQR-based outlier detection and capping
- **Feature Engineering**: Medical-specific feature creation and transformation
- **Categorical Encoding**: One-hot and label encoding for ML compatibility
- **Feature Importance**: Random Forest and mutual information analysis
- **Export Capabilities**: CSV, JSON, and Parquet formats for database import
- **Validation Reporting**: Comprehensive data quality and processing reports

## Dataset Requirements

The script is designed to work with the "Diabetes 130-US hospitals for years 1999-2008" dataset from Kaggle:
- **Records**: 101,766 patient encounters
- **Features**: 55 attributes including demographics, medical, and administrative data
- **Target**: Hospital readmission within 30 days

### Feature Categories Processed

1. **Demographics**: age, gender, race, weight
2. **Medical**: diagnoses (diag_1, diag_2, diag_3), medical_specialty
3. **Medications**: 23 diabetes medications including metformin, insulin, etc.
4. **Administrative**: admission_type_id, discharge_disposition_id, admission_source_id, payer_code
5. **Clinical**: time_in_hospital, procedures, lab tests, prior visits

## Installation

1. **Clone or download the script files**:
   ```bash
   # Download the files
   # diabetes_data_processor.py
   # requirements.txt
   # README.md
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**:
   - Visit [Kaggle Diabetes Dataset](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)
   - Download `diabetic_data.csv`

## Usage

### Command Line Interface

```bash
# Basic usage
python diabetes_data_processor.py --data_path diabetic_data.csv

# Specify output directory
python diabetes_data_processor.py --data_path diabetic_data.csv --output_dir my_output

# Choose export formats
python diabetes_data_processor.py --data_path diabetic_data.csv --formats csv json parquet
```

### Python API Usage

```python
from diabetes_data_processor import DiabetesDataProcessor

# Initialize processor
processor = DiabetesDataProcessor(output_dir="processed_data")

# Run full pipeline
results = processor.run_full_pipeline("diabetic_data.csv")

# Access processed data
processed_data = processor.processed_data
feature_importance = processor.feature_importance

# Individual steps (if needed)
processor.load_data("diabetic_data.csv")
processor.explore_data()
processor.clean_data()
processor.create_target_variable()
processor.handle_missing_values()
processor.handle_outliers()
processor.feature_engineering()
processor.encode_categorical_features()
importance_df = processor.calculate_feature_importance()
exported_files = processor.export_data(['csv', 'json'])
report_path = processor.generate_validation_report()
```

## Output Files

The script generates several output files in the specified directory:

### Data Files
- `diabetes_processed_YYYYMMDD_HHMMSS.csv` - Cleaned dataset in CSV format
- `diabetes_processed_YYYYMMDD_HHMMSS.json` - JSON format for API/database import
- `diabetes_processed_YYYYMMDD_HHMMSS.parquet` - Efficient binary format

### Analysis Files
- `feature_importance_YYYYMMDD_HHMMSS.csv` - Feature importance scores
- `validation_report_YYYYMMDD_HHMMSS.json` - Comprehensive processing report
- `processing_log_YYYYMMDD_HHMMSS.log` - Detailed processing log

## Data Processing Pipeline

### 1. Data Loading & Exploration
- Load CSV with automatic shape and column detection
- Generate summary statistics and missing value analysis
- Analyze target variable distribution

### 2. Data Cleaning
- Remove duplicate records
- Handle missing values (replace '?' with NaN)
- Remove death/hospice discharge cases
- Filter out unknown gender records
- Handle duplicate encounters

### 3. Target Variable Creation
- Create binary target: `readmitted_30_days`
- 1 = readmitted within 30 days (`<30`)
- 0 = not readmitted or readmitted after 30 days

### 4. Missing Value Handling
- **Categorical**: Mode imputation
- **Numerical (< 20% missing)**: Median imputation
- **Numerical (≥ 20% missing)**: KNN imputation

### 5. Outlier Detection & Handling
- IQR-based detection (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
- Outlier capping instead of removal
- Preserve data integrity

### 6. Feature Engineering
- **Age Groups**: Convert age ranges to numeric and create categories
- **Medication Changes**: Count total medications changed
- **Diagnosis Complexity**: Count number of diagnoses
- **Service Utilization**: Sum of prior visits
- **High-Risk Indicator**: Based on hospital stay and procedures
- **Lab Efficiency**: Lab procedures per day

### 7. Categorical Encoding
- **Low cardinality (≤10 unique)**: One-hot encoding
- **High cardinality (>10 unique)**: Label encoding
- Supabase-compatible column names

### 8. Feature Importance Analysis
- Random Forest feature importance
- Mutual information scores
- Combined normalized scoring
- Top feature identification

## Supabase Integration

The processed data is optimized for Supabase database import:

- Column names are lowercase with underscores
- JSON format with proper data types
- No special characters in column names
- Optimized data types for database storage

### Database Schema Suggestions

```sql
-- Create table for processed diabetes data
CREATE TABLE diabetes_patients (
  id SERIAL PRIMARY KEY,
  encounter_id INTEGER UNIQUE,
  patient_nbr INTEGER,
  
  -- Demographics
  age_numeric INTEGER,
  gender VARCHAR(10),
  race VARCHAR(50),
  
  -- Clinical indicators
  readmitted_30_days INTEGER, -- Target variable
  time_in_hospital INTEGER,
  num_procedures INTEGER,
  num_medications INTEGER,
  
  -- Engineered features
  high_risk_patient INTEGER,
  total_medications_changed INTEGER,
  diagnosis_count INTEGER,
  
  -- Add other columns as needed
  created_at TIMESTAMP DEFAULT NOW()
);
```

## Data Quality Metrics

The script generates a comprehensive data quality score (0-100) based on:

1. **Completeness**: Percentage of non-missing values
2. **Target Balance**: Ratio of positive to negative cases
3. **Data Size**: Adequacy of sample size (target: 50k+ records)

## Feature Importance Analysis

The script provides two types of feature importance:

1. **Random Forest Importance**: Tree-based feature selection
2. **Mutual Information**: Statistical dependency measures

Features are ranked by a combined normalized score for optimal ML model performance.

## Error Handling & Logging

- Comprehensive error handling with informative messages
- Detailed logging to both console and file
- Progress tracking throughout the pipeline
- Validation checks at each processing step

## Performance Considerations

- Memory-efficient processing for large datasets
- Parallel processing where applicable (Random Forest)
- Optimized data types for reduced memory usage
- Chunked processing for very large files (if needed)

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning preprocessing and analysis
- **pathlib**: Modern path handling
- **logging**: Process tracking and debugging

## Troubleshooting

### Common Issues

1. **Memory Error**: Reduce dataset size or use chunked processing
2. **Missing Columns**: Ensure dataset matches expected format
3. **Import Errors**: Install all requirements with `pip install -r requirements.txt`
4. **File Path Issues**: Use absolute paths or ensure correct working directory

### Support

For issues or questions:
1. Check the processing log file for detailed error information
2. Verify dataset format matches expected structure
3. Ensure all dependencies are installed correctly

## License

This script is provided as-is for educational and research purposes. Please comply with Kaggle's dataset license terms when using the diabetes dataset.