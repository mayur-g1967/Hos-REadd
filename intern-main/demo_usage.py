"""
Demo Usage Script for Diabetes Data Processor
===========================================

This script demonstrates how to use the DiabetesDataProcessor class
and shows its key features. It includes examples of both CLI and API usage.

Run this script to see example outputs and understand the processor capabilities.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from diabetes_data_processor import DiabetesDataProcessor

def create_sample_data(n_samples=1000):
    """
    Create a sample diabetes dataset for demonstration purposes.
    This simulates the structure of the real Kaggle diabetes dataset.
    
    Args:
        n_samples: Number of sample records to create
        
    Returns:
        DataFrame with sample data
    """
    np.random.seed(42)  # For reproducible results
    
    # Age groups as in the real dataset
    age_groups = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', 
                  '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
    
    # Sample data generation
    data = {
        'encounter_id': range(1, n_samples + 1),
        'patient_nbr': np.random.randint(1, 50000, n_samples),
        
        # Demographics
        'race': np.random.choice(['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other'], n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'age': np.random.choice(age_groups, n_samples),
        'weight': np.random.choice(['[0-25)', '[25-50)', '[50-75)', '[75-100)', '[100-125)', '?'], 
                                   n_samples, p=[0.1, 0.2, 0.3, 0.25, 0.1, 0.05]),
        
        # Medical
        'admission_type_id': np.random.randint(1, 9, n_samples),
        'discharge_disposition_id': np.random.randint(1, 30, n_samples),
        'admission_source_id': np.random.randint(1, 26, n_samples),
        'time_in_hospital': np.random.randint(1, 15, n_samples),
        'payer_code': np.random.choice(['MC', 'MD', 'HM', 'SP', 'BC', '?'], n_samples),
        'medical_specialty': np.random.choice(['InternalMedicine', 'Cardiology', 'Surgery', 
                                              'Emergency/Trauma', '?'], n_samples),
        
        # Diagnoses
        'diag_1': np.random.choice(['250.01', '250.02', '250.03', '428.0', '414.01', '?'], n_samples),
        'diag_2': np.random.choice(['250.01', '250.02', '250.03', '428.0', '414.01', '?'], n_samples),
        'diag_3': np.random.choice(['250.01', '250.02', '250.03', '428.0', '414.01', '?'], n_samples),
        
        # Clinical metrics
        'num_lab_procedures': np.random.randint(0, 100, n_samples),
        'num_procedures': np.random.randint(0, 10, n_samples),
        'num_medications': np.random.randint(1, 30, n_samples),
        'number_outpatient': np.random.randint(0, 20, n_samples),
        'number_emergency': np.random.randint(0, 10, n_samples),
        'number_inpatient': np.random.randint(0, 15, n_samples),
        'number_diagnoses': np.random.randint(1, 16, n_samples),
        
        # Lab results
        'max_glu_serum': np.random.choice(['None', 'Norm', '>200', '>300'], n_samples),
        'A1Cresult': np.random.choice(['None', 'Norm', '>7', '>8'], n_samples),
        
        # Medications (sample subset)
        'metformin': np.random.choice(['No', 'Steady', 'Up', 'Down'], n_samples),
        'insulin': np.random.choice(['No', 'Steady', 'Up', 'Down'], n_samples),
        'glipizide': np.random.choice(['No', 'Steady', 'Up', 'Down'], n_samples),
        'glyburide': np.random.choice(['No', 'Steady', 'Up', 'Down'], n_samples),
        
        # Treatment indicators
        'change': np.random.choice(['No', 'Ch'], n_samples),
        'diabetesMed': np.random.choice(['Yes', 'No'], n_samples),
        
        # Target variable
        'readmitted': np.random.choice(['NO', '<30', '>30'], n_samples, p=[0.7, 0.1, 0.2])
    }
    
    return pd.DataFrame(data)

def demonstrate_api_usage():
    """
    Demonstrate how to use the DiabetesDataProcessor API.
    """
    print("="*60)
    print("DIABETES DATA PROCESSOR - API DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    print("\n1. Creating sample diabetes dataset...")
    sample_data = create_sample_data(1000)
    
    # Save sample data
    sample_file = "sample_diabetes_data.csv"
    sample_data.to_csv(sample_file, index=False)
    print(f"   Sample data saved to: {sample_file}")
    print(f"   Dataset shape: {sample_data.shape}")
    
    # Initialize processor
    print("\n2. Initializing DiabetesDataProcessor...")
    processor = DiabetesDataProcessor(output_dir="demo_output")
    
    try:
        # Run full pipeline
        print("\n3. Running full processing pipeline...")
        results = processor.run_full_pipeline(sample_file)
        
        # Display results
        print("\n4. PROCESSING RESULTS:")
        print("-" * 40)
        print(f"Status: {results['status']}")
        print(f"Processed shape: {results['processed_data_shape']}")
        print(f"Data quality score: {results['data_quality_score']:.1f}/100")
        
        print(f"\nExported files:")
        for format_type, file_path in results['exported_files'].items():
            print(f"  {format_type.upper()}: {file_path}")
            
        print(f"\nAnalysis files:")
        print(f"  Feature importance: {results['feature_importance_file']}")
        print(f"  Validation report: {results['validation_report']}")
        
        # Show top features
        if processor.feature_importance is not None:
            print(f"\nTop 5 most important features:")
            top_features = processor.feature_importance.head(5)
            for idx, row in top_features.iterrows():
                print(f"  {row['feature']}: {row['combined_score']:.3f}")
                
        # Show data quality insights
        print(f"\nData Quality Insights:")
        if 'final_dataset' in processor.validation_report:
            final_info = processor.validation_report['final_dataset']
            print(f"  Final dataset shape: {final_info['shape']}")
            print(f"  Memory usage: {final_info['memory_usage_mb']:.2f} MB")
            
            if final_info['target_distribution']:
                target_dist = final_info['target_distribution']
                total = sum(target_dist.values())
                readmission_rate = target_dist.get(1, 0) / total * 100
                print(f"  30-day readmission rate: {readmission_rate:.1f}%")
        
        print("\n5. SUCCESS: Demo completed successfully!")
        
    except Exception as e:
        print(f"\nERROR during processing: {str(e)}")
        print("This is expected if dependencies are missing.")
        
    finally:
        # Cleanup
        if os.path.exists(sample_file):
            os.remove(sample_file)
            print(f"\nCleaned up sample file: {sample_file}")

def demonstrate_individual_steps():
    """
    Demonstrate individual processing steps.
    """
    print("\n" + "="*60)
    print("INDIVIDUAL PROCESSING STEPS DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    sample_data = create_sample_data(500)
    sample_file = "sample_data_steps.csv"
    sample_data.to_csv(sample_file, index=False)
    
    try:
        processor = DiabetesDataProcessor(output_dir="demo_steps_output")
        
        # Step 1: Load data
        print("\nStep 1: Loading data...")
        processor.load_data(sample_file)
        print(f"  Loaded shape: {processor.raw_data.shape}")
        
        # Step 2: Explore data
        print("\nStep 2: Data exploration...")
        exploration = processor.explore_data()
        print(f"  Missing values found: {sum(exploration['missing_values'].values())}")
        print(f"  Memory usage: {exploration['memory_usage']:.2f} MB")
        
        # Step 3: Clean data
        print("\nStep 3: Data cleaning...")
        processor.clean_data()
        print(f"  Cleaned shape: {processor.processed_data.shape}")
        
        # Step 4: Create target variable
        print("\nStep 4: Creating target variable...")
        processor.create_target_variable()
        target_dist = processor.processed_data['readmitted_30_days'].value_counts()
        print(f"  Target distribution: {target_dist.to_dict()}")
        
        # Step 5: Handle missing values
        print("\nStep 5: Handling missing values...")
        processor.handle_missing_values()
        remaining_missing = processor.processed_data.isnull().sum().sum()
        print(f"  Remaining missing values: {remaining_missing}")
        
        # Step 6: Feature engineering
        print("\nStep 6: Feature engineering...")
        original_cols = len(processor.processed_data.columns)
        processor.feature_engineering()
        new_cols = len(processor.processed_data.columns)
        print(f"  Added {new_cols - original_cols} new features")
        
        # Step 7: Encode categorical features
        print("\nStep 7: Encoding categorical features...")
        processor.encode_categorical_features()
        final_cols = len(processor.processed_data.columns)
        print(f"  Final feature count: {final_cols}")
        
        # Step 8: Feature importance
        print("\nStep 8: Calculating feature importance...")
        importance_df = processor.calculate_feature_importance()
        print(f"  Analyzed {len(importance_df)} features")
        print(f"  Top feature: {importance_df.iloc[0]['feature']}")
        
        print("\nIndividual steps demo completed successfully!")
        
    except Exception as e:
        print(f"\nError in individual steps: {str(e)}")
        
    finally:
        if os.path.exists(sample_file):
            os.remove(sample_file)

def show_usage_examples():
    """
    Show various usage examples.
    """
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    
    print("\n1. Command Line Usage:")
    print("   python diabetes_data_processor.py --data_path diabetic_data.csv")
    print("   python diabetes_data_processor.py --data_path data.csv --output_dir results")
    print("   python diabetes_data_processor.py --data_path data.csv --formats csv json")
    
    print("\n2. Python API Usage:")
    print("""
   from diabetes_data_processor import DiabetesDataProcessor
   
   # Quick processing
   processor = DiabetesDataProcessor()
   results = processor.run_full_pipeline("diabetic_data.csv")
   
   # Step-by-step processing
   processor.load_data("diabetic_data.csv")
   processor.clean_data()
   processor.create_target_variable()
   processor.feature_engineering()
   exported_files = processor.export_data(['csv', 'json'])
   """)
   
    print("\n3. Advanced Usage:")
    print("""
   # Custom output directory
   processor = DiabetesDataProcessor(output_dir="my_results")
   
   # Access intermediate results
   cleaned_data = processor.processed_data
   feature_importance = processor.feature_importance
   quality_score = processor._calculate_data_quality_score()
   
   # Export specific formats
   files = processor.export_data(['parquet', 'json'])
   """)

def main():
    """
    Main demonstration function.
    """
    print("DIABETES DATA PROCESSOR - COMPREHENSIVE DEMO")
    print("=" * 60)
    print("This demo shows the capabilities of the DiabetesDataProcessor")
    print("without requiring the actual Kaggle dataset.")
    print("=" * 60)
    
    try:
        # Show usage examples
        show_usage_examples()
        
        # Demonstrate API usage with sample data
        demonstrate_api_usage()
        
        # Demonstrate individual steps
        demonstrate_individual_steps()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nTo use with real data:")
        print("1. Download the Kaggle diabetes dataset")
        print("2. Run: python diabetes_data_processor.py --data_path diabetic_data.csv")
        print("3. Check the 'processed_data' directory for results")
        
    except ImportError as e:
        print(f"\nImport Error: {e}")
        print("\nPlease install required dependencies:")
        print("pip install -r requirements.txt")
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Please check your environment setup.")

if __name__ == "__main__":
    main()