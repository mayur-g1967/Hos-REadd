"""
Diabetes 130-US Hospitals Dataset Processor
==========================================

A comprehensive script to process the Kaggle "Diabetes 130-US hospitals" dataset
for hospital readmission prediction with ML-ready output and Supabase compatibility.

Features:
- Load and clean 101,766+ patient records with 55 features
- Handle missing values and outliers
- Create binary target for 30-day readmission prediction
- Feature engineering for medical data
- Export to CSV/JSON for database import
- Generate data validation report and feature importance analysis

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import json
import warnings
from typing import Dict, List, Tuple, Any
from pathlib import Path
import logging
from datetime import datetime

# ML libraries
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class DiabetesDataProcessor:
    """
    Comprehensive processor for Diabetes 130-US hospitals dataset.
    """
    
    def __init__(self, data_path: str = None, output_dir: str = "processed_data"):
        """
        Initialize the processor.
        
        Args:
            data_path: Path to the diabetes dataset CSV file
            output_dir: Directory to save processed outputs
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize data containers
        self.raw_data = None
        self.processed_data = None
        self.feature_importance = None
        self.validation_report = {}
        
        # Setup logging
        self._setup_logging()
        
        # Define feature categories
        self.feature_categories = {
            'demographics': ['age', 'gender', 'race', 'weight'],
            'medical': ['diag_1', 'diag_2', 'diag_3', 'medical_specialty'],
            'medications': ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                          'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                          'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                          'miglitol', 'troglitazone', 'tolazamide', 'examide',
                          'citoglipton', 'insulin', 'glyburide-metformin',
                          'glipizide-metformin', 'glimepiride-pioglitazone',
                          'metformin-rosiglitazone', 'metformin-pioglitazone'],
            'administrative': ['admission_type_id', 'discharge_disposition_id',
                             'admission_source_id', 'payer_code'],
            'clinical': ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                        'num_medications', 'number_outpatient', 'number_emergency',
                        'number_inpatient', 'number_diagnoses', 'max_glu_serum',
                        'A1Cresult', 'change', 'diabetesMed']
        }
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load the diabetes dataset from CSV.
        
        Args:
            data_path: Path to the dataset file
            
        Returns:
            Loaded DataFrame
        """
        if data_path:
            self.data_path = data_path
            
        if not self.data_path:
            raise ValueError("No data path provided. Please specify the path to the diabetes dataset.")
            
        self.logger.info(f"Loading data from {self.data_path}")
        
        try:
            self.raw_data = pd.read_csv(self.data_path)
            self.logger.info(f"Data loaded successfully. Shape: {self.raw_data.shape}")
            
            # Basic data info
            self.validation_report['original_shape'] = self.raw_data.shape
            self.validation_report['original_columns'] = list(self.raw_data.columns)
            
            return self.raw_data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def explore_data(self) -> Dict[str, Any]:
        """
        Perform initial data exploration and generate summary statistics.
        
        Returns:
            Dictionary containing exploration results
        """
        if self.raw_data is None:
            raise ValueError("Data not loaded. Please load data first.")
            
        self.logger.info("Performing data exploration...")
        
        exploration = {
            'shape': self.raw_data.shape,
            'columns': list(self.raw_data.columns),
            'dtypes': self.raw_data.dtypes.to_dict(),
            'missing_values': self.raw_data.isnull().sum().to_dict(),
            'missing_percentage': (self.raw_data.isnull().sum() / len(self.raw_data) * 100).to_dict(),
            'unique_values': {col: self.raw_data[col].nunique() for col in self.raw_data.columns},
            'memory_usage': self.raw_data.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        # Target variable analysis
        if 'readmitted' in self.raw_data.columns:
            exploration['target_distribution'] = self.raw_data['readmitted'].value_counts().to_dict()
            
        self.validation_report['exploration'] = exploration
        self.logger.info(f"Data exploration completed. Dataset has {exploration['shape'][0]} rows and {exploration['shape'][1]} columns.")
        
        return exploration
        
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values, duplicates, and data quality issues.
        
        Returns:
            Cleaned DataFrame
        """
        if self.raw_data is None:
            raise ValueError("Data not loaded. Please load data first.")
            
        self.logger.info("Starting data cleaning...")
        df = self.raw_data.copy()
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        self.logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        # Handle missing values indicated by '?' 
        df = df.replace('?', np.nan)
        
        # Remove records with missing patient identifiers
        if 'patient_nbr' in df.columns:
            df = df.dropna(subset=['patient_nbr'])
            
        # Handle specific data quality issues
        
        # 1. Weight: Remove records where weight is unknown or invalid
        if 'weight' in df.columns:
            weight_missing = df['weight'].isnull().sum()
            df = df.dropna(subset=['weight'])
            self.logger.info(f"Removed {weight_missing} records with missing weight")
            
        # 2. Discharge disposition: Remove records indicating patient death
        if 'discharge_disposition_id' in df.columns:
            # Typically IDs 11, 13, 14, 19, 20, 21 indicate death/hospice
            death_ids = [11, 13, 14, 19, 20, 21]
            initial_count = len(df)
            df = df[~df['discharge_disposition_id'].isin(death_ids)]
            removed_death = initial_count - len(df)
            self.logger.info(f"Removed {removed_death} records with death/hospice discharge")
            
        # 3. Gender: Remove records with unknown gender
        if 'gender' in df.columns:
            df = df[df['gender'].isin(['Male', 'Female'])]
            
        # 4. Handle encounter_id duplicates (keep first occurrence)
        if 'encounter_id' in df.columns:
            initial_encounters = len(df)
            df = df.drop_duplicates(subset=['encounter_id'], keep='first')
            encounter_dupes = initial_encounters - len(df)
            self.logger.info(f"Removed {encounter_dupes} duplicate encounters")
            
        # Update validation report
        self.validation_report['cleaning'] = {
            'duplicates_removed': duplicates_removed,
            'records_after_cleaning': len(df),
            'cleaning_steps': [
                'Removed duplicates',
                'Handled missing values (? -> NaN)',
                'Removed death/hospice cases',
                'Removed unknown gender records',
                'Removed duplicate encounters'
            ]
        }
        
        self.processed_data = df
        self.logger.info(f"Data cleaning completed. Final dataset shape: {df.shape}")
        
        return df
        
    def create_target_variable(self) -> pd.DataFrame:
        """
        Create binary target variable for 30-day readmission prediction.
        
        Returns:
            DataFrame with processed target variable
        """
        if self.processed_data is None:
            raise ValueError("Data not processed. Please clean data first.")
            
        self.logger.info("Creating target variable for 30-day readmission...")
        df = self.processed_data.copy()
        
        if 'readmitted' not in df.columns:
            raise ValueError("'readmitted' column not found in dataset")
            
        # Create binary target: 1 if readmitted within 30 days, 0 otherwise
        df['readmitted_30_days'] = (df['readmitted'] == '<30').astype(int)
        
        # Track target distribution
        target_dist = df['readmitted_30_days'].value_counts()
        self.logger.info(f"Target distribution - Not readmitted: {target_dist[0]}, Readmitted within 30 days: {target_dist[1]}")
        
        self.validation_report['target_creation'] = {
            'target_distribution': target_dist.to_dict(),
            'readmission_rate': target_dist[1] / len(df) * 100
        }
        
        self.processed_data = df
        return df
        
    def handle_missing_values(self) -> pd.DataFrame:
        """
        Handle missing values using appropriate strategies for different feature types.
        
        Returns:
            DataFrame with imputed missing values
        """
        if self.processed_data is None:
            raise ValueError("Data not processed. Please process data first.")
            
        self.logger.info("Handling missing values...")
        df = self.processed_data.copy()
        
        # Track missing values before imputation
        missing_before = df.isnull().sum()
        
        # Strategy 1: Mode imputation for categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col].fillna(mode_value, inplace=True)
                
        # Strategy 2: Median imputation for numerical variables with few missing values
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().any():
                missing_pct = df[col].isnull().sum() / len(df) * 100
                if missing_pct < 20:  # Use median for < 20% missing
                    df[col].fillna(df[col].median(), inplace=True)
                else:  # Use KNN imputation for higher missing percentages
                    imputer = KNNImputer(n_neighbors=5)
                    df[col] = imputer.fit_transform(df[[col]]).ravel()
                    
        # Track missing values after imputation
        missing_after = df.isnull().sum()
        
        self.validation_report['missing_value_handling'] = {
            'missing_before': missing_before.to_dict(),
            'missing_after': missing_after.to_dict(),
            'imputation_strategies': {
                'categorical': 'Mode imputation',
                'numerical_low_missing': 'Median imputation (<20% missing)',
                'numerical_high_missing': 'KNN imputation (>=20% missing)'
            }
        }
        
        self.processed_data = df
        self.logger.info("Missing value handling completed")
        
        return df
        
    def handle_outliers(self) -> pd.DataFrame:
        """
        Detect and handle outliers in numerical features.
        
        Returns:
            DataFrame with outliers handled
        """
        if self.processed_data is None:
            raise ValueError("Data not processed. Please process data first.")
            
        self.logger.info("Handling outliers...")
        df = self.processed_data.copy()
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col not in ['readmitted_30_days', 'patient_nbr', 'encounter_id']]
        
        outlier_info = {}
        
        for col in numerical_cols:
            # Use IQR method for outlier detection
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_info[col] = {
                'outliers_count': outliers,
                'outlier_percentage': outliers / len(df) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            # Cap outliers instead of removing them (to preserve data)
            df[col] = np.clip(df[col], lower_bound, upper_bound)
            
        self.validation_report['outlier_handling'] = outlier_info
        self.logger.info(f"Outlier handling completed for {len(numerical_cols)} numerical columns")
        
        self.processed_data = df
        return df
        
    def feature_engineering(self) -> pd.DataFrame:
        """
        Perform feature engineering specific to medical data.
        
        Returns:
            DataFrame with engineered features
        """
        if self.processed_data is None:
            raise ValueError("Data not processed. Please process data first.")
            
        self.logger.info("Performing feature engineering...")
        df = self.processed_data.copy()
        
        # 1. Age group binning
        if 'age' in df.columns:
            # Convert age ranges to numerical midpoints then create groups
            age_mapping = {
                '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
                '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
                '[80-90)': 85, '[90-100)': 95
            }
            df['age_numeric'] = df['age'].map(age_mapping)
            
            # Create age groups
            df['age_group'] = pd.cut(df['age_numeric'], 
                                   bins=[0, 30, 50, 70, 100], 
                                   labels=['Young', 'Middle', 'Senior', 'Elderly'])
            
        # 2. Total medication count
        medication_cols = [col for col in self.feature_categories['medications'] if col in df.columns]
        if medication_cols:
            # Count medications that are not 'No' or 'Steady'
            df['total_medications_changed'] = 0
            for med_col in medication_cols:
                if med_col in df.columns:
                    df['total_medications_changed'] += (
                        (df[med_col] != 'No') & (df[med_col] != 'Steady')
                    ).astype(int)
                    
        # 3. Diagnosis complexity score
        diag_cols = ['diag_1', 'diag_2', 'diag_3']
        available_diag_cols = [col for col in diag_cols if col in df.columns]
        if available_diag_cols:
            df['diagnosis_count'] = df[available_diag_cols].notna().sum(axis=1)
            
        # 4. Service utilization score
        utilization_cols = ['number_outpatient', 'number_emergency', 'number_inpatient']
        available_util_cols = [col for col in utilization_cols if col in df.columns]
        if available_util_cols:
            df['total_prior_visits'] = df[available_util_cols].sum(axis=1)
            
        # 5. High-risk patient indicator
        if 'time_in_hospital' in df.columns and 'num_procedures' in df.columns:
            df['high_risk_patient'] = (
                (df['time_in_hospital'] > df['time_in_hospital'].median()) &
                (df['num_procedures'] > df['num_procedures'].median())
            ).astype(int)
            
        # 6. Medication change indicator
        if 'change' in df.columns:
            df['medication_changed'] = (df['change'] == 'Ch').astype(int)
            
        # 7. Diabetes medication indicator
        if 'diabetesMed' in df.columns:
            df['on_diabetes_med'] = (df['diabetesMed'] == 'Yes').astype(int)
            
        # 8. Lab procedures per day
        if 'num_lab_procedures' in df.columns and 'time_in_hospital' in df.columns:
            df['lab_procedures_per_day'] = df['num_lab_procedures'] / (df['time_in_hospital'] + 1)
            
        engineered_features = [
            'age_numeric', 'age_group', 'total_medications_changed', 'diagnosis_count',
            'total_prior_visits', 'high_risk_patient', 'medication_changed',
            'on_diabetes_med', 'lab_procedures_per_day'
        ]
        
        actual_engineered = [feat for feat in engineered_features if feat in df.columns]
        
        self.validation_report['feature_engineering'] = {
            'engineered_features': actual_engineered,
            'total_features_before': len(self.processed_data.columns),
            'total_features_after': len(df.columns)
        }
        
        self.processed_data = df
        self.logger.info(f"Feature engineering completed. Added {len(actual_engineered)} new features")
        
        return df
        
    def encode_categorical_features(self) -> pd.DataFrame:
        """
        Encode categorical features for ML compatibility.
        
        Returns:
            DataFrame with encoded categorical features
        """
        if self.processed_data is None:
            raise ValueError("Data not processed. Please process data first.")
            
        self.logger.info("Encoding categorical features...")
        df = self.processed_data.copy()
        
        # Get categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target-related columns from encoding
        categorical_cols = [col for col in categorical_cols if col not in ['readmitted']]
        
        encoding_info = {}
        
        for col in categorical_cols:
            unique_values = df[col].nunique()
            
            if unique_values <= 10:  # One-hot encode for low cardinality
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df.drop(col, axis=1, inplace=True)
                
                encoding_info[col] = {
                    'method': 'one_hot',
                    'unique_values': unique_values,
                    'new_columns': list(dummies.columns)
                }
                
            else:  # Label encode for high cardinality
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                df.drop(col, axis=1, inplace=True)
                
                encoding_info[col] = {
                    'method': 'label_encoding',
                    'unique_values': unique_values,
                    'new_columns': [f'{col}_encoded']
                }
                
        self.validation_report['categorical_encoding'] = encoding_info
        self.processed_data = df
        self.logger.info(f"Categorical encoding completed for {len(categorical_cols)} columns")
        
        return df
        
    def calculate_feature_importance(self) -> pd.DataFrame:
        """
        Calculate feature importance using Random Forest and mutual information.
        
        Returns:
            DataFrame with feature importance scores
        """
        if self.processed_data is None:
            raise ValueError("Data not processed. Please process data first.")
            
        if 'readmitted_30_days' not in self.processed_data.columns:
            raise ValueError("Target variable not found. Please create target variable first.")
            
        self.logger.info("Calculating feature importance...")
        df = self.processed_data.copy()
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in 
                       ['readmitted_30_days', 'readmitted', 'patient_nbr', 'encounter_id']]
        
        X = df[feature_cols]
        y = df['readmitted_30_days']
        
        # Handle any remaining non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rf_importance = rf.feature_importances_
        
        # Mutual information importance
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'rf_importance': rf_importance,
            'mutual_info_score': mi_scores
        })
        
        # Normalize scores
        importance_df['rf_importance_norm'] = importance_df['rf_importance'] / importance_df['rf_importance'].max()
        importance_df['mi_score_norm'] = importance_df['mutual_info_score'] / importance_df['mutual_info_score'].max()
        
        # Combined score
        importance_df['combined_score'] = (importance_df['rf_importance_norm'] + importance_df['mi_score_norm']) / 2
        
        # Sort by combined score
        importance_df = importance_df.sort_values('combined_score', ascending=False)
        
        self.feature_importance = importance_df
        self.validation_report['feature_importance'] = {
            'top_10_features': importance_df.head(10)['feature'].tolist(),
            'total_features_analyzed': len(importance_df)
        }
        
        self.logger.info("Feature importance calculation completed")
        
        return importance_df
        
    def export_data(self, formats: List[str] = ['csv', 'json']) -> Dict[str, str]:
        """
        Export processed data in specified formats for Supabase compatibility.
        
        Args:
            formats: List of export formats ('csv', 'json', 'parquet')
            
        Returns:
            Dictionary with file paths of exported data
        """
        if self.processed_data is None:
            raise ValueError("Data not processed. Please process data first.")
            
        self.logger.info(f"Exporting data in formats: {formats}")
        
        # Prepare final dataset
        df = self.processed_data.copy()
        
        # Ensure all columns are Supabase compatible (no special characters)
        df.columns = [col.replace('-', '_').replace(' ', '_').lower() for col in df.columns]
        
        exported_files = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # CSV export
        if 'csv' in formats:
            csv_path = self.output_dir / f"diabetes_processed_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            exported_files['csv'] = str(csv_path)
            self.logger.info(f"Data exported to CSV: {csv_path}")
            
        # JSON export
        if 'json' in formats:
            json_path = self.output_dir / f"diabetes_processed_{timestamp}.json"
            
            # Convert to JSON-serializable format
            df_json = df.copy()
            
            # Handle numpy types
            for col in df_json.columns:
                if df_json[col].dtype == 'object':
                    df_json[col] = df_json[col].astype(str)
                elif np.issubdtype(df_json[col].dtype, np.integer):
                    df_json[col] = df_json[col].astype(int)
                elif np.issubdtype(df_json[col].dtype, np.floating):
                    df_json[col] = df_json[col].astype(float)
                    
            # Export as JSON
            df_json.to_json(json_path, orient='records', indent=2)
            exported_files['json'] = str(json_path)
            self.logger.info(f"Data exported to JSON: {json_path}")
            
        # Parquet export (efficient for large datasets)
        if 'parquet' in formats:
            parquet_path = self.output_dir / f"diabetes_processed_{timestamp}.parquet"
            df.to_parquet(parquet_path, index=False)
            exported_files['parquet'] = str(parquet_path)
            self.logger.info(f"Data exported to Parquet: {parquet_path}")
            
        return exported_files
        
    def generate_validation_report(self) -> str:
        """
        Generate a comprehensive data validation report.
        
        Returns:
            Path to the generated report
        """
        self.logger.info("Generating validation report...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"validation_report_{timestamp}.json"
        
        # Add final dataset statistics
        if self.processed_data is not None:
            self.validation_report['final_dataset'] = {
                'shape': self.processed_data.shape,
                'columns': list(self.processed_data.columns),
                'memory_usage_mb': self.processed_data.memory_usage(deep=True).sum() / 1024**2,
                'target_distribution': self.processed_data['readmitted_30_days'].value_counts().to_dict() if 'readmitted_30_days' in self.processed_data.columns else None
            }
            
        # Add feature importance summary
        if self.feature_importance is not None:
            self.validation_report['feature_importance_summary'] = {
                'top_5_features': self.feature_importance.head(5)[['feature', 'combined_score']].to_dict('records')
            }
            
        # Add processing summary
        self.validation_report['processing_summary'] = {
            'timestamp': timestamp,
            'total_processing_steps': len([k for k in self.validation_report.keys() if k not in ['processing_summary']]),
            'data_quality_score': self._calculate_data_quality_score()
        }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(self.validation_report, f, indent=2, default=str)
            
        self.logger.info(f"Validation report generated: {report_path}")
        return str(report_path)
        
    def _calculate_data_quality_score(self) -> float:
        """Calculate overall data quality score (0-100)."""
        if self.processed_data is None:
            return 0.0
            
        # Factors for data quality
        factors = []
        
        # 1. Completeness (no missing values)
        missing_rate = self.processed_data.isnull().sum().sum() / (self.processed_data.shape[0] * self.processed_data.shape[1])
        completeness_score = (1 - missing_rate) * 100
        factors.append(completeness_score)
        
        # 2. Target balance (not too imbalanced)
        if 'readmitted_30_days' in self.processed_data.columns:
            target_dist = self.processed_data['readmitted_30_days'].value_counts()
            balance_ratio = min(target_dist) / max(target_dist)
            balance_score = min(balance_ratio * 100, 100)  # Cap at 100
            factors.append(balance_score)
            
        # 3. Data size adequacy
        size_score = min(len(self.processed_data) / 50000 * 100, 100)  # Target 50k+ records
        factors.append(size_score)
        
        return np.mean(factors) if factors else 0.0
        
    def run_full_pipeline(self, data_path: str = None) -> Dict[str, Any]:
        """
        Run the complete data processing pipeline.
        
        Args:
            data_path: Path to the dataset file
            
        Returns:
            Dictionary with pipeline results and file paths
        """
        try:
            self.logger.info("Starting full data processing pipeline...")
            
            # Step 1: Load data
            if data_path:
                self.load_data(data_path)
            elif not self.raw_data:
                raise ValueError("No data loaded. Please provide data_path or load data first.")
                
            # Step 2: Explore data
            exploration = self.explore_data()
            
            # Step 3: Clean data
            self.clean_data()
            
            # Step 4: Create target variable
            self.create_target_variable()
            
            # Step 5: Handle missing values
            self.handle_missing_values()
            
            # Step 6: Handle outliers
            self.handle_outliers()
            
            # Step 7: Feature engineering
            self.feature_engineering()
            
            # Step 8: Encode categorical features
            self.encode_categorical_features()
            
            # Step 9: Calculate feature importance
            importance_df = self.calculate_feature_importance()
            
            # Step 10: Export data
            exported_files = self.export_data(['csv', 'json'])
            
            # Step 11: Generate validation report
            report_path = self.generate_validation_report()
            
            # Save feature importance
            importance_path = self.output_dir / f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            importance_df.to_csv(importance_path, index=False)
            
            results = {
                'status': 'success',
                'processed_data_shape': self.processed_data.shape,
                'exported_files': exported_files,
                'feature_importance_file': str(importance_path),
                'validation_report': report_path,
                'data_quality_score': self.validation_report['processing_summary']['data_quality_score']
            }
            
            self.logger.info("Full pipeline completed successfully!")
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    """
    Main function to run the diabetes data processor using a hardcoded path.
    """
    # ✅ Your dataset path (modify if you later use a different file)
    data_path = r"C:\Users\Mayur\Downloads\intern-main\intern-main\demo_output\diabetes_processed_20250715_060328.csv"
    
    # Output directory where processed files will be saved
    output_dir = "processed_data"
    
    # Export formats (choose any: 'csv', 'json', 'parquet')
    formats = ['csv', 'json']

    # Initialize processor
    processor = DiabetesDataProcessor(output_dir=output_dir)

    try:
        # Run the full pipeline with the hardcoded data path
        results = processor.run_full_pipeline(data_path)

        print("\n" + "="*60)
        print("PROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Processed dataset shape: {results['processed_data_shape']}")
        print(f"Data quality score: {results['data_quality_score']:.1f}/100")
        print(f"\nExported files:")
        for format_type, file_path in results['exported_files'].items():
            print(f"  {format_type.upper()}: {file_path}")
        print(f"\nFeature importance: {results['feature_importance_file']}")
        print(f"Validation report: {results['validation_report']}")
        print("\n" + "="*60)

    except Exception as e:
        print(f"ERROR: {str(e)}")
        print("\nPlease check the data path and ensure the dataset is in the correct format.")


if __name__ == "__main__":
    main()