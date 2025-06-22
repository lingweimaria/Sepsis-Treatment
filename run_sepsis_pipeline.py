#!/usr/bin/env python3
"""
Complete Sepsis Pipeline Runner
Runs data preprocessing and training for sepsis patients
"""

import os
import sys
from pathlib import Path
import argparse

def run_preprocessing():
    """Run sepsis data preprocessing"""
    print("=" * 60)
    print("STEP 1: Running Sepsis Data Preprocessing")
    print("=" * 60)
    
    from sepsis_data_preprocessing import SepsisICUDataPreprocessor
    
    try:
        # Auto-detect data directory
        possible_data_dirs = [Path('data'), Path('代码/data'), Path('./data')]
        data_dir = None
        for potential_dir in possible_data_dirs:
            if potential_dir.exists():
                data_dir = potential_dir
                break
        
        preprocessor = SepsisICUDataPreprocessor(data_dir=data_dir)
        results = preprocessor.run_sepsis_preprocessing_pipeline()
        
        if results:
            print("✅ Preprocessing completed successfully!")
            return True
        else:
            print("❌ Preprocessing failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        return False

def run_training(epochs=30, lr=1e-4):
    """Run sepsis model training"""
    print("=" * 60)
    print("STEP 2: Running Sepsis Model Training")
    print("=" * 60)
    print(f"Note: Using fixed training parameters (epochs=15, lr=1e-4) from fixed_sepsis_training.py")
    print("=" * 60)
    
    try:
        # Import here to avoid issues if preprocessing hasn't been run
        from fixed_sepsis_training import main as train_main
        import sys
        
        # The fixed_sepsis_training module doesn't use command line arguments
        # It uses hardcoded parameters, so we just call the main function directly
        train_main()
        
        print("✅ Training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_data_availability():
    """Check if required data files are available"""
    print("Checking data availability...")
    
    # Try different possible data directory locations
    possible_data_dirs = [Path('data'), Path('代码/data'), Path('./data')]
    data_dir = None
    
    for potential_dir in possible_data_dirs:
        if potential_dir.exists():
            data_dir = potential_dir
            break
    
    if data_dir is None:
        print("❌ No data directory found!")
        return False
    
    print(f"Using data directory: {data_dir.absolute()}")
    required_files = [
        'top 2000 diseases_mimic3.csv',
        'admission time_mimic3.csv',
        'static variables(demographics)_mimic3.csv',
        'time series variables(vital signs)_mimic3.csv',
        'gcs components.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not (data_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required data files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    else:
        print("✅ All required data files found!")
        return True

def print_progress_summary():
    """Print summary of what will be done"""
    print("🏥 ICU Sepsis Treatment Recommendation System")
    print("=" * 60)
    print("This pipeline will:")
    print("1. 🔍 Identify sepsis patients from disease data")
    print("2. 📊 Filter and preprocess clinical data for sepsis patients only")
    print("3. 🧮 Calculate SOFA scores for reward shaping")
    print("4. 🔄 Create sequences limited to 7 days to reduce training time")
    print("5. 🤖 Train RL agent with enhanced SOFA rewards")
    print("6. 📈 Generate training curves and save model")
    print()
    print("Expected benefits:")
    print("✅ Smaller dataset = Faster training")
    print("✅ Disease-specific focus = Better performance")
    print("✅ Enhanced SOFA rewards = Clinical relevance")
    print("=" * 60)

def main():
    """Main pipeline runner"""
    parser = argparse.ArgumentParser(description='Run Complete Sepsis Pipeline')
    parser.add_argument('--skip-preprocessing', action='store_true', 
                       help='Skip preprocessing if already done')
    parser.add_argument('--preprocessing-only', action='store_true',
                       help='Run only preprocessing')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    print_progress_summary()
    
    # Check data availability
    if not check_data_availability():
        print("\n❌ Cannot proceed without required data files!")
        print("Please ensure all data files are in the 'data/' directory.")
        return
    
    success = True
    
    # Run preprocessing
    if not args.skip_preprocessing:
        success = run_preprocessing()
        if not success:
            print("\n❌ Pipeline failed at preprocessing stage!")
            return
    else:
        print("⏭️  Skipping preprocessing (as requested)")
    
    # Stop here if preprocessing only
    if args.preprocessing_only:
        print("\n✅ Preprocessing completed! Use --skip-preprocessing for training.")
        return
    
    # Check if preprocessed data exists
    preprocessed_dir = Path('preprocessed_sepsis_data')
    if not preprocessed_dir.exists():
        print("\n❌ Preprocessed data not found! Run preprocessing first.")
        return
    
    # Run training
    success = run_training(epochs=args.epochs, lr=args.lr)
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 SEPSIS PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Generated files:")
        print("📁 preprocessed_sepsis_data/ - Processed sepsis data")
        print("📁 models/ - Trained models")
        print("📊 sepsis_training_curves.png - Training visualization")
        print("\nNext steps for your presentation:")
        print("1. 📊 Analyze the training curves")
        print("2. 🔍 Compare with baseline (original SOFA rewards)")
        print("3. 📝 Prepare results for June 26 presentation")
        print("=" * 60)
    else:
        print("\n❌ Pipeline failed at training stage!")

if __name__ == "__main__":
    main() 