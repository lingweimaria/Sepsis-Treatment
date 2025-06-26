# 败血症数据处理流水线总结报告
生成时间: 2025-06-23 22:30:00

## 数据文件统计
### raw data
- time series variables(vital signs)_mimic3.csv: 833.9 MB
- filtered_static_ts.csv: 17.3 MB
- top 1000 medications_mimic3.csv: 628.3 MB
- static variables(weight_height)_mimic3.csv: 0.4 MB
- static variables(demographics)_mimic3.csv: 2.4 MB
- time series variables(output)_mimic3.csv: 55.9 MB
- top 2000 diseases_mimic3.csv: 10.9 MB
- admission time_mimic3.csv: 1.8 MB
- gcs components.csv: 100.9 MB
### processed data
#### final training data
- test_metadata.csv: 447 行, 3 列
- test_features.npy: shape=(447, 7, 12)
- train_metadata.csv: 2083 行, 3 列
- train_features.npy: shape=(2083, 7, 12)
- val_metadata.csv: 446 行, 3 列
- val_features.npy: shape=(446, 7, 12)
- feature_info.csv: 1 行, 2 列
#### SOFA score
- enhanced_sofa_dataset.csv: 96510 行, 29 列
#### expert decision data
- expert_decisions.csv: 5115 行, 16 列
- expert_decisions.pkl: 完整专家决策数据
- expert_decisions_stats.json: 统计报告
## preprocessing scripts
- sepsis_data_preprocessing.py
- enhanced_sofa_calculator.py
- expert_decision_extractor.py
## 模型维度信息
### 状态空间 (State Space)
- **维度**: 12
- **特征列表**:
  - temperature (体温)
  - heart_rate (心率)  
  - respiratory_rate (呼吸频率)
  - arterial_blood_pressure_systolic (收缩压)
  - arterial_blood_pressure_diastolic (舒张压)
  - o2_saturation_pulseoxymetry (血氧饱和度)
  - glucose_(serum) (血糖)
  - gcs_-_eye_opening (格拉斯哥昏迷量表-睁眼)
  - gcs_-_motor_response (格拉斯哥昏迷量表-运动)
  - gcs_-_verbal_response (格拉斯哥昏迷量表-语言)
  - age (年龄)
  - sofa_score (SOFA评分)

### 动作空间 (Action Space)
- **总定义维度**: 20 (Action ID: 0-19)
- **实际使用维度**: 13 (脓毒症相关药物)
- **序列长度**: 7天 (7个时间步)

#### 实际使用的动作分布
| Action ID | 动作类型 | 记录数 | 占比 |
|-----------|----------|-------|------|
| 0 | observation (观察) | 1967 | 38.5% |
| 11 | electrolyte (电解质) | 1321 | 25.8% |
| 10 | glucose (葡萄糖/液体) | 404 | 7.9% |
| 17 | analgesic (镇痛) | 236 | 4.6% |
| 4 | beta_blocker (β阻滞剂) | 206 | 4.0% |
| 18 | sedative (镇静) | 179 | 3.5% |
| 13 | antibiotic (抗生素) | 165 | 3.2% |
| 3 | diuretic (利尿剂) | 157 | 3.1% |
| 9 | insulin (胰岛素) | 153 | 3.0% |
| 19 | anticoagulant (抗凝) | 148 | 2.9% |
| 2 | antihypertensive (降压) | 84 | 1.6% |
| 7 | steroid (类固醇) | 51 | 1.0% |
| 1 | vasopressor (血管加压) | 44 | 0.9% |

#### 未使用的动作 (7个)
- Action 5: oxygen_therapy (氧疗)
- Action 6: bronchodilator (支气管扩张剂)
- Action 8: ventilation (机械通气)
- Action 12: nutrition (营养支持)
- Action 14: antifungal (抗真菌)
- Action 15: antiviral (抗病毒)
- Action 16: antiseptic (防腐剂)

### 专家决策统计
- **总决策记录**: 5,115条
- **No.of patients involved**: 95个
- **Average number of decisions per patient**: 53.8条
- **Drug Screening Effectiveness**: 从原始4,126,347条记录中筛选出2,456,302条脓毒症相关记录 (59.5%)

## 数据处理流程
1. 原始MIMIC-III数据导入
2. 脓毒症患者筛选 (基于ICD-9编码)
3. 脓毒症相关药物筛选 (59.5%数据保留率)
4. 基础数据预处理和清洗
5. SOFA评分计算
6. 专家决策数据生成 (用于混合RL+SL训练)
7. 特征工程和序列构建
8. 训练/验证/测试数据分割 (70%/15%/15%)
9. 数据验证和质量检查