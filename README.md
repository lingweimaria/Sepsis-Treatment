# Sepsis Treatment Reinforcement Learning System

## Overview

This project implements a reinforcement learning system for optimizing sepsis treatment decisions using MIMIC-III data. The system uses TD3 (Twin Delayed Deep Deterministic Policy Gradient) and DDPG algorithms to learn optimal drug prescription policies for sepsis patients.

## System Architecture

```
Raw Data → Data Preprocessing → Feature Engineering → Drug Mapping → Model Training → Evaluation
    ↓              ↓                    ↓              ↓               ↓            ↓
MIMIC-III     Cleaned Data        Merged Dataset    Action Space    TD3/DDPG    Performance
                                                                                   Analysis
```

## Data Processing Pipeline

### 1. Raw Data Sources

The system uses the following MIMIC-III datasets:

#### Core Clinical Data
- **`final_df.csv`**: Patient demographics, vital signs, lab values, and static features
- **`sofa.csv`**: Sequential Organ Failure Assessment (SOFA) scores over time

#### Medical Records
- **`ADMISSIONS.csv`**: Hospital admission records
- **`PATIENTS.csv`**: Patient demographic information
- **`CHARTEVENTS.csv`**: Vital signs and clinical measurements
- **`LABEVENTS.csv`**: Laboratory test results
- **`PRESCRIPTIONS.csv`**: Medication prescriptions
- **`top 1000 medications_mimic3.csv`**: Top 1000 medications with standardized names

#### Dictionary Files
- **`D_ITEMS.csv`**: Item definitions for chart events
- **`D_LABITEMS.csv`**: Laboratory item definitions
- **`D_ICD_DIAGNOSES.csv`**: ICD diagnosis code definitions

### 2. Data Preprocessing

#### Step 1: Prescription Data Cleaning (`clean_prescription_data.py`)

**Purpose**: Clean temporal anomalies in prescription data

**Process**:
1. **Temporal Filtering**:
   - Remove records with years < 2100 (data entry errors)
   - Remove records with years > 2200 (future dates)
   - Remove records with negative duration (ENDDATE < STARTDATE)

2. **Data Validation**:
   - Ensure valid HADM_ID references
   - Validate drug name consistency
   - Generate cleaning statistics

**Output**: `data/processed/cleaned_prescriptions.csv`

#### Step 2: Dataset Merging (`merge_datasets.py`)

**Purpose**: Combine clinical data with SOFA scores

**Process**:
1. **Data Loading**:
   - Load `sofa.csv` (temporal SOFA scores)
   - Load `final_df.csv` (clinical features)

2. **Feature Separation**:
   - **Static Features**: `subject_id`, `gender`, `age`, `weight_kg`, `height_cm`, `hospital_expire_flag`, `admittime`
   - **Temporal Features**: All remaining clinical measurements

3. **Feature Engineering**:
   - Binary encoding: `gender` (M=1, F=0)
   - Remove irrelevant features: `religion`, `language`, `marital_status`, `ethnicity`
   - Preserve `admittime` for temporal alignment

4. **Data Merging**:
   - Merge on `hadm_id` and `time_step`
   - Handle missing values through forward filling
   - Ensure temporal consistency

**Output**: `data/processed/merged_dataset.csv`

### 3. Drug Dictionary Mapping

#### Drug Standardization Process

**Purpose**: Map raw drug names to standardized action indices

**Files**:
- **`improved_drug_map.json`**: Maps drug names to indices
- **`improved_drug_idx.json`**: Bidirectional mapping with metadata

**Drug Categories** (Top 100 drugs):
1. **Cardiovascular**: METOPROLOL, FUROSEMIDE, HYDRALAZINE, AMIODARONE
2. **Analgesics**: MORPHINE, HYDROMORPHONE, OXYCODONE, FENTANYL
3. **Antibiotics**: VANCOMYCIN, LEVOFLOXACIN, PIPERACILLIN, CEFTRIAXONE
4. **Electrolytes**: SODIUM, POTASSIUM, MAGNESIUM, CALCIUM
5. **Fluids**: D5W, NS (Normal Saline), LR (Lactated Ringer's)
6. **Sedatives**: PROPOFOL, LORAZEPAM, MIDAZOLAM

**Mapping Structure**:
```json
{
  "drug_to_index": {
    "SODIUM": 0,
    "POTASSIUM": 1,
    "INSULIN": 2,
    ...
  },
  "num_drugs": 100,
  "other_index": -1
}
```

### 4. Final Dataset Creation (`create_training_dataset.py`)

**Purpose**: Generate the complete training dataset with actions

**Process**:

1. **Data Integration**:
   - Load merged clinical data
   - Load cleaned prescription data
   - Load drug mapping dictionaries

2. **SOFA Component Removal**:
   - Remove individual SOFA components: `sofa_resp`, `sofa_coag`, `sofa_liver`, `sofa_cardiovascular`, `sofa_cns`, `sofa_renal`
   - Retain overall `sofa` score for reward calculation

3. **Temporal Window Generation**:
   - Create 24-hour time windows based on `admittime + time_step`
   - Each `time_step` represents 24 hours from admission

4. **Action Space Construction**:
   - Map prescriptions to 24-hour time windows
   - Create binary action vectors (100 dimensions)
   - Handle overlapping prescriptions within windows

5. **Data Quality Assurance**:
   - Filter patients with sufficient temporal data
   - Ensure action-state alignment
   - Generate metadata for model training

**Output**: 
- `data/processed/training_dataset.csv`
- `data/processed/training_dataset_metadata.json`

## Model Architecture

### 1. Environment Design

#### State Space
- **Dimensions**: ~50-100 clinical features
- **Features**: 
  - Vital signs (heart rate, blood pressure, temperature)
  - Laboratory values (creatinine, bilirubin, platelet count)
  - Demographics (age, gender, weight)
  - SOFA score (disease severity indicator)

#### Action Space
- **Dimensions**: 100 drug categories
- **Type**: Continuous (0-1) representing prescription probability
- **Interpretation**: Values > 0.5 indicate drug administration

#### Reward Function
```python
reward = sofa_reward + vital_signs_reward + sparsity_penalty + terminal_reward
```

**Components**:
1. **SOFA Reward**: `clip(prev_sofa - current_sofa, -2, 2)`
2. **Vital Signs Reward**: 
   - +0.25 if 90 ≤ SBP ≤ 140
   - +0.25 if 60 ≤ HR ≤ 110
3. **Sparsity Penalty**: `-0.1 × mean(action)` (discourage over-prescription)
4. **Terminal Reward**: +15.0 (survival) or -15.0 (death)

### 2. Neural Network Architecture

#### LSTM Actor Network
```python
class EnhancedLSTMActorNetwork:
    - LSTM(state_dim, hidden_dim=128, num_layers=2, dropout=0.2)
    - LayerNorm + ReLU + Linear(hidden_dim)
    - LayerNorm + Sigmoid(action_dim)
```

#### LSTM Critic Network
```python
class EnhancedLSTMCriticNetwork:
    - LSTM(state_dim, hidden_dim=128, num_layers=1)
    - State Branch: Linear(hidden_dim, hidden_dim//2)
    - Action Branch: Linear(action_dim, hidden_dim//2)
    - Q-Value Output: Linear(hidden_dim, 1)
```

### 3. Training Algorithms

#### TD3 (Twin Delayed Deep Deterministic Policy Gradient)

**Key Features**:
- **Twin Critics**: Reduces overestimation bias
- **Delayed Policy Updates**: Updates actor every 2 critic updates
- **Target Noise**: Adds noise to target actions

**Hyperparameters**:
- Learning Rate: 3e-4 (actor), 3e-4 (critic)
- Discount Factor (γ): 0.99
- Soft Update Rate (τ): 0.005
- Target Noise: 0.2, Noise Clip: 0.5

**Update Equations**:
```python
# Critic Update
target_q = min(Q1_target, Q2_target)
target_values = rewards + γ * (1 - dones) * target_q
critic_loss = MSE(Q(s,a), target_values)

# Actor Update (delayed)
actor_loss = -Q1(s, π(s)).mean()
```

#### DDPG (Baseline Comparison)

**Key Features**:
- Single critic network
- Continuous action spaces
- Experience replay

**Hyperparameters**:
- Learning Rate: 1e-3 (actor), 1e-3 (critic)
- Soft Update Rate (τ): 0.001
- Exploration Noise: 0.1

## Usage Instructions

### 1. Environment Setup

```bash
# Clone repository
git clone <repository_url>
cd sepsis_rl_system

# Install dependencies
pip install pandas numpy torch scikit-learn matplotlib seaborn
```

### 2. Data Preparation

```bash
# Step 1: Clean prescription data
python scripts/clean_prescription_data.py

# Step 2: Merge datasets
python scripts/merge_datasets.py

# Step 3: Create training dataset
python scripts/create_training_dataset.py
```

### 3. Model Training

```bash
# Train TD3 model with comparison to DDPG
python scripts/training/td.py

# Evaluate existing model similarity
python scripts/training/td.py --evaluate-similarity [model_path]

# Test mortality evaluation
python scripts/training/td.py --test-mortality

# Generate visualization
python scripts/training/td.py --test-viz
```

### 4. Model Evaluation

The system automatically generates comprehensive evaluation including:

1. **Mortality Rate Comparison**:
   - Physician baseline vs. AI agent performance
   - Training progression analysis

2. **Agent-Physician Similarity**:
   - Cosine similarity of action patterns
   - Drug usage frequency correlation
   - Mean squared error of decisions

3. **Case Study Analysis**:
   - Individual patient trajectory visualization
   - Treatment outcome tracking

## Results and Visualization

### Performance Metrics

1. **Clinical Outcomes**:
   - Mortality rate reduction
   - SOFA score improvement
   - Treatment response time

2. **Policy Analysis**:
   - Action similarity with physician decisions
   - Drug prescription patterns
   - Temporal attention mechanisms

### Visualization Components

The system generates a comprehensive 2×2 visualization grid:

1. **Learning Curves**: TD3 vs DDPG training progress
2. **Mortality Trends**: Performance improvement over training
3. **Attention Heatmap**: Temporal dependencies in decision-making
4. **Case Study**: Individual patient treatment trajectory

## File Structure

```
sepsis_rl_system/
├── data/
│   ├── raw/                    # Original MIMIC-III data
│   └── processed/              # Processed datasets
├── scripts/
│   ├── clean_prescription_data.py
│   ├── merge_datasets.py
│   ├── create_training_dataset.py
│   ├── improved_drug_map.json
│   ├── improved_drug_idx.json
│   └── training/
│       └── td.py              # Main training script
├── models/                     # Saved model checkpoints
└── README.md
```

## Technical Details

### Computational Requirements

- **GPU**: Recommended for faster training (CUDA support)
- **Memory**: 8GB+ RAM for full dataset processing
- **Storage**: 5GB+ for MIMIC-III data and processed files
- **Training Time**: 2-4 hours on modern GPU
