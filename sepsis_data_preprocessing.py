#!/usr/bin/env python3
"""
Sepsis-Focused Data Preprocessing Pipeline for ICU Treatment Recommendation
This script focuses only on sepsis patients to reduce data size and training time
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class SepsisICUDataPreprocessor:
    """
    Data preprocessor focused on sepsis patients in ICU
    """
    
    def __init__(self, data_dir=None, processed_dir='processed_data'):
        """
        Initialize preprocessor with local data directories
        
        Args:
            data_dir (str): Directory containing raw CSV files
            processed_dir (str): Directory containing processed CSV files
        """
        # Auto-detect data directory if not specified
        if data_dir is None:
            possible_data_dirs = [Path('data'), Path('代码/data'), Path('./data')]
            for potential_dir in possible_data_dirs:
                if potential_dir.exists():
                    data_dir = potential_dir
                    break
            if data_dir is None:
                data_dir = Path('data')  # fallback
        
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = {}
        
        # Sepsis-related ICD codes for filtering
        self.sepsis_icd_codes = [
            '038',    # Septicemia
            '995.92', # Severe sepsis
            '785.52', # Septic shock
            '038.0',  # Streptococcal septicemia
            '038.1',  # Staphylococcal septicemia
            '038.2',  # Pneumococcal septicemia
            '038.3',  # Septicemia due to anaerobes
            '038.4',  # Septicemia due to other gram-negative organisms
            '038.8',  # Other specified septicemias
            '038.9',  # Unspecified septicemia
            '999.31', # Infection due to central venous catheter
            '999.32', # Bloodstream infection due to central venous catheter
        ]
        
        # Key vital signs and lab values for sepsis monitoring
        self.key_features = [
            'temperature', 'heart_rate', 'respiratory_rate', 
            'arterial_blood_pressure_systolic', 'arterial_blood_pressure_diastolic',
            'o2_saturation_pulseoxymetry', 'glucose_(serum)', 'lactate',
            'creatinine', 'gcs_-_eye_opening', 'gcs_-_motor_response', 
            'gcs_-_verbal_response', 'foley'
        ]
        
        print(f"Sepsis-focused ICU Data Preprocessor initialized")
        print(f"Raw data dir: {self.data_dir.absolute()}")
        print(f"Processed data dir: {self.processed_dir.absolute()}")
    
    def identify_sepsis_patients(self, diseases_df):
        """
        Identify patients with sepsis diagnosis
        
        Args:
            diseases_df (pd.DataFrame): Diseases dataset with ICD codes
            
        Returns:
            set: Set of hadm_ids for sepsis patients
        """
        print("Identifying sepsis patients...")
        
        # Check different possible column names for ICD codes
        icd_column = None
        for col in ['icd9_code', 'icd_code', 'diagnosis_code', 'code']:
            if col in diseases_df.columns:
                icd_column = col
                break
        
        if icd_column is None:
            # Try to find any column that might contain ICD codes
            text_columns = diseases_df.select_dtypes(include=['object']).columns
            if len(text_columns) > 0:
                icd_column = text_columns[0]
                print(f"Using column '{icd_column}' as ICD code column")
            else:
                print("Warning: Could not identify ICD code column")
                return set()
        
        sepsis_patients = set()
        
        # Convert ICD codes to string for comparison
        diseases_df[icd_column] = diseases_df[icd_column].astype(str)
        
        for code in self.sepsis_icd_codes:
            # Match exact codes and codes that start with the pattern
            mask = diseases_df[icd_column].str.startswith(code, na=False)
            sepsis_hadm_ids = diseases_df[mask]['hadm_id'].unique()
            sepsis_patients.update(sepsis_hadm_ids)
            print(f"Found {len(sepsis_hadm_ids)} patients with ICD code {code}")
        
        print(f"Total sepsis patients identified: {len(sepsis_patients)}")
        return sepsis_patients
    
    def load_and_filter_data(self):
        """Load data and filter for sepsis patients only"""
        print("Loading data and filtering for sepsis patients...")
        
        datasets = {}
        
        # File mapping
        file_mapping = {
            'diseases': 'top 2000 diseases_mimic3.csv',
            'admission_time': 'admission time_mimic3.csv',
            'demographics': 'static variables(demographics)_mimic3.csv',
            'weight_height': 'static variables(weight_height)_mimic3.csv',
            'vital_signs': 'time series variables(vital signs)_mimic3.csv',
            'output': 'time series variables(output)_mimic3.csv',
            'gcs': 'gcs components.csv',
            'medications': 'top 1000 medications_mimic3.csv'
        }
        
        # Load diseases data first to identify sepsis patients
        diseases_path = self.data_dir / file_mapping['diseases']
        if diseases_path.exists():
            print("Loading diseases data...")
            diseases_df = pd.read_csv(diseases_path)
            sepsis_patients = self.identify_sepsis_patients(diseases_df)
            
            if len(sepsis_patients) == 0:
                print("No sepsis patients found! Using all patients...")
                sepsis_patients = None
        else:
            print("Diseases file not found! Using all patients...")
            sepsis_patients = None
        
        # Load other datasets and filter for sepsis patients
        for name, filename in file_mapping.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                try:
                    print(f"Loading {name}...")
                    df = pd.read_csv(filepath)
                    
                    # Filter for sepsis patients if we have them
                    if sepsis_patients is not None and 'hadm_id' in df.columns:
                        original_size = len(df)
                        df = df[df['hadm_id'].isin(sepsis_patients)]
                        print(f"Filtered {name}: {original_size} -> {len(df)} rows")
                    
                    datasets[name] = df
                    
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        return datasets, sepsis_patients
    
    def process_time_series_data(self, datasets):
        """Process and merge time series data"""
        print("Processing time series data...")
        
        # Start with admission times to establish patient episodes
        if 'admission_time' in datasets:
            admissions = datasets['admission_time'].copy()
            admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'])
            admissions.columns = [col.lower() for col in admissions.columns]
        else:
            print("No admission time data found!")
            return None
        
        # Process vital signs
        if 'vital_signs' in datasets:
            vital_signs = datasets['vital_signs'].copy()
            vital_signs['charttime'] = pd.to_datetime(vital_signs['charttime'])
            
            # Merge with admission times
            vital_signs = vital_signs.merge(
                admissions[['hadm_id', 'admittime']], 
                on='hadm_id', 
                how='left'
            )
            
            # Calculate hours from admission
            vital_signs['hours_from_admit'] = (
                vital_signs['charttime'] - vital_signs['admittime']
            ).dt.total_seconds() / 3600
            
            # Create daily time steps (24-hour bins)
            vital_signs['time_step'] = (vital_signs['hours_from_admit'] // 24 + 1).astype(int)
            
            # Filter for first 7 days to reduce data size
            vital_signs = vital_signs[vital_signs['time_step'] <= 7]
            
            # Aggregate to daily averages
            vital_signs_agg = vital_signs.groupby(
                ['subject_id', 'hadm_id', 'time_step', 'label'],
                as_index=False
            )['valuenum'].mean()
            
            # Pivot to wide format
            vital_signs_wide = vital_signs_agg.pivot(
                index=['subject_id', 'hadm_id', 'time_step'],
                columns='label',
                values='valuenum'
            ).reset_index()
            
            # Clean column names
            vital_signs_wide.columns = [col.replace(' ', '_').lower() for col in vital_signs_wide.columns]
            
            print(f"Processed vital signs: {vital_signs_wide.shape}")
            
        else:
            print("No vital signs data found!")
            return None
        
        # Process GCS data
        if 'gcs' in datasets:
            gcs_data = datasets['gcs'].copy()
            gcs_data['charttime'] = pd.to_datetime(gcs_data['charttime'])
            
            # Merge with admission times
            gcs_data = gcs_data.merge(
                admissions[['hadm_id', 'admittime']], 
                on='hadm_id', 
                how='left'
            )
            
            # Calculate hours from admission and time steps
            gcs_data['hours_from_admit'] = (
                gcs_data['charttime'] - gcs_data['admittime']
            ).dt.total_seconds() / 3600
            gcs_data['time_step'] = (gcs_data['hours_from_admit'] // 24 + 1).astype(int)
            
            # Filter for first 7 days
            gcs_data = gcs_data[gcs_data['time_step'] <= 7]
            
            # Aggregate GCS components
            gcs_agg = gcs_data.groupby(
                ['subject_id', 'hadm_id', 'time_step', 'label'],
                as_index=False
            )['valuenum'].mean()
            
            # Pivot GCS data
            gcs_wide = gcs_agg.pivot(
                index=['subject_id', 'hadm_id', 'time_step'],
                columns='label',
                values='valuenum'
            ).reset_index()
            
            # Clean column names
            gcs_wide.columns = [col.replace(' ', '_').replace('-', '_').lower() for col in gcs_wide.columns]
            
            # Merge with vital signs
            combined_data = vital_signs_wide.merge(
                gcs_wide, 
                on=['subject_id', 'hadm_id', 'time_step'], 
                how='outer'
            )
            
            print(f"Added GCS data. Combined shape: {combined_data.shape}")
            
        else:
            combined_data = vital_signs_wide
            print("No GCS data found, using only vital signs")
        
        # Add static demographics
        if 'demographics' in datasets:
            demographics = datasets['demographics'].copy()
            combined_data = combined_data.merge(
                demographics, 
                on=['subject_id', 'hadm_id'], 
                how='left'
            )
            print(f"Added demographics. Final shape: {combined_data.shape}")
        
        return combined_data
    
    def calculate_sepsis_sofa_score(self, data):
        """Calculate SOFA score with focus on sepsis-relevant components"""
        print("Calculating SOFA scores...")
        
        sofa_scores = pd.Series(0, index=data.index)
        
        # CNS (Glasgow Coma Scale) - very important for sepsis
        if all(col in data.columns for col in ['gcs___eye_opening', 'gcs___motor_response', 'gcs___verbal_response']):
            gcs_total = (data['gcs___eye_opening'].fillna(4) + 
                        data['gcs___motor_response'].fillna(6) + 
                        data['gcs___verbal_response'].fillna(5))
            
            sofa_scores += np.where(gcs_total < 6, 4,
                                   np.where(gcs_total < 10, 3,
                                           np.where(gcs_total < 13, 2,
                                                   np.where(gcs_total < 15, 1, 0))))
        
        # Cardiovascular (Blood pressure) - critical for septic shock
        if 'arterial_blood_pressure_systolic' in data.columns and 'arterial_blood_pressure_diastolic' in data.columns:
            map_values = (data['arterial_blood_pressure_systolic'] + 2 * data['arterial_blood_pressure_diastolic']) / 3
            sofa_scores += np.where(map_values < 70, 1, 0)
        
        # Respiratory (simplified - based on oxygen saturation)
        if 'o2_saturation_pulseoxymetry' in data.columns:
            spo2 = data['o2_saturation_pulseoxymetry']
            sofa_scores += np.where(spo2 < 85, 4,
                                   np.where(spo2 < 90, 3,
                                           np.where(spo2 < 95, 2,
                                                   np.where(spo2 < 98, 1, 0))))
        
        # Renal (Creatinine and urine output)
        if 'creatinine' in data.columns:
            creatinine = data['creatinine']
            sofa_scores += np.where(creatinine > 5, 4,
                                   np.where(creatinine > 3.5, 3,
                                           np.where(creatinine > 2, 2,
                                                   np.where(creatinine > 1.2, 1, 0))))
        
        data['sofa_score'] = sofa_scores
        return data
    
    def prepare_sequences(self, data, max_sequence_length=7):
        """Prepare sequences for RL training"""
        print("Preparing sequences for training...")
        
        # Focus on key features for sepsis
        feature_cols = []
        for feature in self.key_features:
            if feature in data.columns:
                feature_cols.append(feature)
        
        # Add demographic features
        demo_cols = ['age', 'gender_m', 'gender_f'] if any(col in data.columns for col in ['age', 'gender_m', 'gender_f']) else []
        feature_cols.extend([col for col in demo_cols if col in data.columns])
        
        # Add SOFA score
        if 'sofa_score' in data.columns:
            feature_cols.append('sofa_score')
        
        print(f"Selected {len(feature_cols)} features: {feature_cols}")
        
        # Group by patient and create sequences
        sequences = []
        patients = data.groupby(['subject_id', 'hadm_id'])
        
        for (subject_id, hadm_id), patient_data in patients:
            # Sort by time step
            patient_data = patient_data.sort_values('time_step')
            
            # Extract features
            patient_features = patient_data[feature_cols].values
            
            # Handle missing values with forward fill then backward fill
            patient_df = pd.DataFrame(patient_features, columns=feature_cols)
            patient_df = patient_df.ffill().bfill()
            patient_features = patient_df.values
            
            # Create sequence (pad or truncate to max_sequence_length)
            if len(patient_features) > max_sequence_length:
                patient_features = patient_features[:max_sequence_length]
            elif len(patient_features) < max_sequence_length:
                # Pad with the last observation
                padding = np.tile(patient_features[-1], (max_sequence_length - len(patient_features), 1))
                patient_features = np.vstack([patient_features, padding])
            
            sequences.append({
                'subject_id': subject_id,
                'hadm_id': hadm_id,
                'features': patient_features,
                'sequence_length': min(len(patient_data), max_sequence_length)
            })
        
        print(f"Created {len(sequences)} patient sequences")
        return sequences, feature_cols
    
    def normalize_sequences(self, sequences, feature_cols):
        """Normalize feature sequences"""
        print("Normalizing sequences...")
        
        # Collect all feature values for fitting scaler
        all_features = []
        for seq in sequences:
            all_features.append(seq['features'])
        
        all_features = np.vstack(all_features)
        
        # Fit scaler
        scaler = StandardScaler()
        scaler.fit(all_features)
        
        # Normalize sequences
        normalized_sequences = []
        for seq in sequences:
            normalized_features = scaler.transform(seq['features'])
            normalized_sequences.append({
                'subject_id': seq['subject_id'],
                'hadm_id': seq['hadm_id'],
                'features': normalized_features,
                'sequence_length': seq['sequence_length']
            })
        
        self.scaler = scaler
        return normalized_sequences
    
    def split_sequences(self, sequences, train_ratio=0.7, val_ratio=0.15):
        """Split sequences into train/val/test sets"""
        print("Splitting sequences...")
        
        np.random.shuffle(sequences)
        
        n_total = len(sequences)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_sequences = sequences[:n_train]
        val_sequences = sequences[n_train:n_train + n_val]
        test_sequences = sequences[n_train + n_val:]
        
        print(f"Train: {len(train_sequences)}, Val: {len(val_sequences)}, Test: {len(test_sequences)}")
        
        return {
            'train': train_sequences,
            'val': val_sequences,
            'test': test_sequences
        }
    
    def save_processed_data(self, splits, feature_cols, output_dir='preprocessed_sepsis_data'):
        """Save processed sepsis data"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"Saving processed data to {output_dir}...")
        
        # Save splits
        for split_name, sequences in splits.items():
            # Convert to arrays for saving
            features_list = [seq['features'] for seq in sequences]
            
            if features_list:
                features_array = np.stack(features_list)
                
                # Save as numpy arrays
                np.save(output_dir / f'{split_name}_features.npy', features_array)
                
                # Save metadata
                metadata = {
                    'subject_ids': [seq['subject_id'] for seq in sequences],
                    'hadm_ids': [seq['hadm_id'] for seq in sequences],
                    'sequence_lengths': [seq['sequence_length'] for seq in sequences]
                }
                
                metadata_df = pd.DataFrame(metadata)
                metadata_df.to_csv(output_dir / f'{split_name}_metadata.csv', index=False)
        
        # Save feature names and scaler
        feature_info = {
            'feature_names': feature_cols,
            'n_features': len(feature_cols)
        }
        
        pd.DataFrame([feature_info]).to_csv(output_dir / 'feature_info.csv', index=False)
        
        # Save scaler
        import pickle
        with open(output_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print("Data saved successfully!")
    
    def run_sepsis_preprocessing_pipeline(self):
        """Run the complete sepsis preprocessing pipeline"""
        print("=" * 60)
        print("Starting Sepsis-Focused ICU Data Preprocessing Pipeline")
        print("=" * 60)
        
        # Step 1: Load and filter data
        datasets, sepsis_patients = self.load_and_filter_data()
        
        if not datasets:
            print("No data loaded! Please check your data directory.")
            return None
        
        # Step 2: Process time series data
        combined_data = self.process_time_series_data(datasets)
        
        if combined_data is None or len(combined_data) == 0:
            print("No time series data processed!")
            return None
        
        print(f"Combined data shape: {combined_data.shape}")
        
        # Step 3: Calculate SOFA scores
        combined_data = self.calculate_sepsis_sofa_score(combined_data)
        
        # Step 4: Prepare sequences
        sequences, feature_cols = self.prepare_sequences(combined_data)
        
        if not sequences:
            print("No sequences created!")
            return None
        
        # Step 5: Normalize sequences
        normalized_sequences = self.normalize_sequences(sequences, feature_cols)
        
        # Step 6: Split data
        splits = self.split_sequences(normalized_sequences)
        
        # Step 7: Save processed data
        self.save_processed_data(splits, feature_cols)
        
        print("=" * 60)
        print("Sepsis preprocessing pipeline completed successfully!")
        print(f"Total sepsis patients processed: {len(sepsis_patients) if sepsis_patients else 'Unknown'}")
        print(f"Total sequences created: {len(sequences)}")
        print(f"Feature dimensions: {len(feature_cols)}")
        print("=" * 60)
        
        return splits, feature_cols

def main():
    """Main function to run the sepsis preprocessing pipeline"""
    preprocessor = SepsisICUDataPreprocessor()
    
    try:
        results = preprocessor.run_sepsis_preprocessing_pipeline()
        
        if results:
            splits, feature_cols = results
            print("\nPreprocessing completed successfully!")
            print("You can now use the processed data for training.")
            
        else:
            print("\nPreprocessing failed. Please check your data and try again.")
            
    except Exception as e:
        print(f"\nError during preprocessing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 