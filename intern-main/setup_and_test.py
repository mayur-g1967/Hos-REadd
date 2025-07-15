
"""
Setup and Test Script for Diabetes Data Processor
================================================

This script:
1. Checks Python version compatibility
2. Installs required dependencies
3. Tests the diabetes data processor functionality
4. Provides setup guidance

Run this script first to ensure everything is working correctly.
"""

import sys
import subprocess
import importlib
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"❌ Python {version.major}.{version.minor} detected.")
        print("   Minimum required: Python 3.7+")
        print("   Please upgrade Python and try again.")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def install_dependencies():
    """Install required dependencies from requirements.txt."""
    print("\nInstalling dependencies...")
    
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt file not found!")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("   Try running manually: pip install -r requirements.txt")
        return False

def check_imports():
    """Check if all required modules can be imported."""
    print("\nChecking imports...")
    
    required_modules = [
        "pandas", "numpy", "sklearn", "pathlib", "json", "logging"
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    print("✅ All imports successful")
    return True

def test_basic_functionality():
    """Test basic functionality of the diabetes processor."""
    print("\nTesting basic functionality...")
    
    try:
        # Test if we can import the main module
        from diabetes_data_processor import DiabetesDataProcessor
        print("✅ Successfully imported DiabetesDataProcessor")
        
        # Test initialization
        processor = DiabetesDataProcessor(output_dir="test_output")
        print("✅ Successfully initialized processor")
        
        # Test if output directory is created
        if Path("test_output").exists():
            print("✅ Output directory created successfully")
        
        # Clean up test directory
        import shutil
        if Path("test_output").exists():
            shutil.rmtree("test_output")
            
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def run_demo():
    """Run the demo script to show functionality."""
    print("\nRunning demonstration...")
    
    try:
        from demo_usage import create_sample_data
        
        # Create a small sample
        sample_data = create_sample_data(100)
        print(f"✅ Created sample data with shape: {sample_data.shape}")
        
        # Test if we can process it
        from diabetes_data_processor import DiabetesDataProcessor
        processor = DiabetesDataProcessor(output_dir="test_demo")
        
        # Save sample data
        sample_file = "test_sample.csv"
        sample_data.to_csv(sample_file, index=False)
        
        # Try basic processing steps
        processor.load_data(sample_file)
        exploration = processor.explore_data()
        print(f"✅ Data exploration completed - found {len(exploration['columns'])} columns")
        
        processor.clean_data()
        print(f"✅ Data cleaning completed - shape: {processor.processed_data.shape}")
        
        # Clean up
        import shutil
        if Path("test_demo").exists():
            shutil.rmtree("test_demo")
        if Path(sample_file).exists():
            os.remove(sample_file)
            
        print("✅ Demo test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Demo test failed: {e}")
        return False

def print_usage_instructions():
    """Print usage instructions."""
    print("\n" + "="*60)
    print("SETUP COMPLETED - USAGE INSTRUCTIONS")
    print("="*60)
    
    print("\n📁 Files created:")
    print("   diabetes_data_processor.py  - Main processing script")
    print("   demo_usage.py              - Demonstration script")
    print("   requirements.txt           - Dependencies")
    print("   README.md                  - Documentation")
    print("   setup_and_test.py          - This setup script")
    
    print("\n🚀 Quick Start:")
    print("   1. Download Kaggle diabetes dataset:")
    print("      https://www.kaggle.com/datasets/brandao/diabetes")
    print("   2. Save as 'diabetic_data.csv'")
    print("   3. Run: python diabetes_data_processor.py --data_path diabetic_data.csv")
    
    print("\n🧪 Test with sample data:")
    print("   python demo_usage.py")
    
    print("\n📊 Command line options:")
    print("   --data_path PATH     Path to your diabetes dataset")
    print("   --output_dir DIR     Directory for output files (default: processed_data)")
    print("   --formats LIST       Export formats: csv, json, parquet")
    
    print("\n💡 Python API usage:")
    print("   from diabetes_data_processor import DiabetesDataProcessor")
    print("   processor = DiabetesDataProcessor()")
    print("   results = processor.run_full_pipeline('diabetic_data.csv')")
    
    print("\n📋 Output files will include:")
    print("   - Cleaned dataset (CSV/JSON/Parquet)")
    print("   - Feature importance analysis")
    print("   - Data validation report")
    print("   - Processing logs")

def main():
    """Main setup and test function."""
    print("DIABETES DATA PROCESSOR - SETUP & TEST")
    print("="*60)
    print("Setting up and testing the diabetes data processing environment...")
    
    success = True
    
    # Step 1: Check Python version
    if not check_python_version():
        success = False
    
    # Step 2: Install dependencies
    if success and not install_dependencies():
        success = False
    
    # Step 3: Check imports
    if success and not check_imports():
        success = False
    
    # Step 4: Test basic functionality
    if success and not test_basic_functionality():
        success = False
    
    # Step 5: Run demo
    if success and not run_demo():
        success = False
    
    # Final result
    print("\n" + "="*60)
    if success:
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("✅ All tests passed - Ready to process diabetes data")
        print_usage_instructions()
    else:
        print("❌ SETUP FAILED")
        print("Please check the error messages above and resolve issues.")
        print("\nCommon solutions:")
        print("1. Upgrade Python to 3.7+")
        print("2. Run: pip install --upgrade pip")
        print("3. Run: pip install -r requirements.txt")
        print("4. Check internet connection for package downloads")
    
    print("="*60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)