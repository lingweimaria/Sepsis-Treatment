# Sepsis DRL Data Pipeline

## 目录结构

```
data_pipeline/
├── raw_data/                    # 原始数据
│   └── mimic3_original/        # MIMIC-III原始数据文件
├── processed_data/             # 处理后数据
│   ├── complete_sofa_data/     # 完整SOFA评分数据
│   ├── intermediate/           # 中间处理数据
│   └── final/                  # 最终训练数据
├── preprocessing_scripts/      # 数据处理脚本
├── utils/                      # 工具函数
├── configs/                    # 配置文件
└── README.md                   # 本文件
```

## 数据流程

### 1. 原始数据 (raw_data/mimic3_original/)
- `time series variables(vital signs)_mimic3.csv` - 生命体征时间序列
- `static variables(demographics)_mimic3.csv` - 人口统计学静态变量
- `static variables(weight_height)_mimic3.csv` - 身高体重数据
- `top 1000 medications_mimic3.csv` - 药物数据
- `time series variables(output)_mimic3.csv` - 输出量数据
- `filtered_static_ts.csv` - 过滤后的静态时间序列

### 2. 中间处理数据 (processed_data/intermediate/)
- `vital_signs_processed.csv` - 处理后生命体征
- `demographics_processed.csv` - 处理后人口统计学数据
- `weight_height_processed.csv` - 处理后身高体重
- `medications_processed.csv` - 处理后药物数据
- `gcs_processed.csv` - GCS评分数据

### 3. SOFA评分数据 (processed_data/complete_sofa_data/)
- `enhanced_sofa_dataset.csv` - 增强SOFA数据集
- `sofa_data_report.json` - SOFA数据报告

### 4. 最终训练数据 (processed_data/final/)
- `train_features.npy` - 训练特征
- `val_features.npy` - 验证特征  
- `test_features.npy` - 测试特征
- `train_metadata.csv` - 训练元数据
- `val_metadata.csv` - 验证元数据
- `test_metadata.csv` - 测试元数据
- `feature_info.csv` - 特征信息

## 处理脚本说明

### 核心处理脚本
1. **sepsis_data_preprocessing.py** - 败血症数据预处理主脚本
2. **enhanced_sofa_calculator.py** - 增强SOFA评分计算器


## SOFA评分计算

SOFA (Sequential Organ Failure Assessment) 评分包含6个器官系统：

1. **呼吸系统** - SpO2/FiO2比值 (0-4分)
2. **心血管系统** - 平均动脉压 + 血管加压药 (0-4分)
3. **中枢神经系统** - GCS评分 (0-4分)
4. **肾脏系统** - 肌酐 + 尿量 (0-4分)
5. **凝血系统** - 血小板计数 (0-4分) [数据缺失]
6. **肝脏系统** - 胆红素 (0-4分) [数据缺失]

### 可用指标
- ✅ GCS评分 (眼部、言语、运动反应)
- ✅ 生命体征 (血压、心率、呼吸频率、体温、SpO2)
- ✅ 实验室指标 (肌酐、血糖、乳酸、pH值)
- ✅ 血管加压药使用情况
- ✅ 尿量数据
- ❌ 血小板计数 (需要LABEVENTS表)
- ❌ 胆红素 (需要LABEVENTS表)

## 使用方法

### 1. 运行完整数据处理流水线
```bash
python run_complete_pipeline.py
```

### 2. 运行模型训练 (包含可视化)
```bash
python fixed_sepsis_training.py
```

### 3. 单独运行SOFA计算
```bash
python preprocessing_scripts/enhanced_sofa_calculator.py
```

### 4. 运行败血症预处理
```bash
python preprocessing_scripts/sepsis_data_preprocessing.py
```

## 数据质量

- **训练序列**: ~1000-5000个有效序列
- **验证序列**: ~200-1000个有效序列
- **特征维度**: 通常12-15个特征
- **序列长度**: 固定为7个时间步
- **数据完整性**: 约60-80% (取决于具体指标)

## 强化学习集成

处理后的数据直接用于强化学习训练：
- **状态空间**: 多维生命体征 + SOFA评分
- **动作空间**: 20种治疗动作
- **奖励函数**: 基于SOFA改善、生命体征稳定性等临床指标

### 模型训练功能 (fixed_sepsis_training.py)
- **LSTM智能体**: 增强的败血症LSTM智能体，包含注意力机制
- **临床奖励计算**: 基于医学指标的奖励函数
- **医学安全约束**: 防止危险治疗组合
- **训练可视化**: 自动生成训练曲线图表
- **模型保存**: 自动保存最佳模型和检查点

### 生成的文件
训练完成后会在以下位置生成文件：
- `models/enhanced_best_sepsis_agent.pth` - 最佳模型
- `plots/` - 训练可视化图表
  - `total_loss.png` - 总损失曲线
  - `clinical_rewards.png` - 临床奖励曲线
  - `actor_critic_loss.png` - Actor-Critic损失对比
  - `safety_violations.png` - 安全违规情况
  - `gradient_norm.png` - 梯度范数
  - `learning_rate.png` - 学习率调度
- `enhanced_training_report.txt` - 训练报告

## 注意事项

1. 确保MIMIC-III数据访问权限
2. 处理过程中会自动处理缺失值
3. SOFA评分计算可能不完整 (缺少血小板和胆红素)
4. 建议在处理前备份原始数据