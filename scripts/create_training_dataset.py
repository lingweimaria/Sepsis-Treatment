#!/usr/bin/env python3
"""
创建最终训练数据集
将药物action数据与merged dataset合并
"""

import pandas as pd
import numpy as np
import json
# from tqdm import tqdm  # 移除tqdm依赖
import os

# 文件路径
MERGED_DATA_CSV = "/Users/lulingwei/Desktop/TUM/sem_2/reinforcement learning/代码/sepsis_rl_system/data/processed/merged_dataset.csv"
PRES_CSV = "/Users/lulingwei/Desktop/TUM/sem_2/reinforcement learning/代码/sepsis_rl_system/data/processed/cleaned_prescriptions.csv"
DRUG_MAP_JSON = "/Users/lulingwei/Desktop/TUM/sem_2/reinforcement learning/代码/sepsis_rl_system/scripts/improved_drug_map.json"
DRUG_IDX_JSON = "/Users/lulingwei/Desktop/TUM/sem_2/reinforcement learning/代码/sepsis_rl_system/scripts/improved_drug_idx.json"
OUTPUT_CSV = "/Users/lulingwei/Desktop/TUM/sem_2/reinforcement learning/代码/sepsis_rl_system/data/processed/training_dataset.csv"

print("=" * 60)
print("创建最终训练数据集")
print("=" * 60)

# 步骤1：读取数据
print("\n📥 步骤1：读取数据")
print("读取merged dataset...")
state_data = pd.read_csv(MERGED_DATA_CSV)
print(f"State data shape: {state_data.shape}")

# ==================== 新增代码开始 ====================
# 定义所有要删除的SOFA分项列
sofa_components_to_delete = [
    'sofa_resp', 
    'sofa_coag', 
    'sofa_liver', 
    'sofa_cardiovascular', 
    'sofa_cns', 
    'sofa_renal'
]

# 找出数据中实际存在的、需要删除的列
cols_to_drop = [col for col in sofa_components_to_delete if col in state_data.columns]

if cols_to_drop:
    # 删除这些列
    state_data.drop(columns=cols_to_drop, inplace=True)
    print(f"✅ 已成功删除以下SOFA分项列: {cols_to_drop}")
    print(f"State data shape (删除后): {state_data.shape}")
else:
    print("ℹ️ 未找到任何SOFA分项列，无需删除。")

# ==================== 新增代码结束 ====================

print("读取处方数据...")
pres_data = pd.read_csv(PRES_CSV, 
                       usecols=['HADM_ID', 'STARTDATE', 'ENDDATE', 'DRUG'],
                       dtype={'HADM_ID': 'int32'},
                       parse_dates=['STARTDATE', 'ENDDATE'])
print(f"Prescription data shape: {pres_data.shape}")

print("读取改进的药物映射...")
with open(DRUG_MAP_JSON, 'r') as f:
    drug_to_idx = json.load(f)
    
with open(DRUG_IDX_JSON, 'r') as f:
    drug_idx_info = json.load(f)

print(f"Drug mapping: {len(drug_to_idx)} drugs")
print(f"Drug categories: {drug_idx_info['num_drugs']} categories")

# 步骤2：药物数据清洗和映射
print("\n🧹 步骤2：药物数据清洗和映射")

# 直接映射到药物索引
pres_data['drug_idx'] = pres_data['DRUG'].map(drug_to_idx)

# 丢弃未映射的药物（drug_idx == -1）
pres_data = pres_data[pres_data['drug_idx'] != -1]
print(f"处理后prescription data shape: {pres_data.shape}")

# 获取所有可能的药物索引
all_drug_indices = list(range(drug_idx_info['num_drugs']))
print(f"药物动作维度: {len(all_drug_indices)}")

# 步骤3：生成时间窗口
print("\n🗓️ 步骤3：生成时间窗口")

# 为每个住院记录生成24小时时间窗口
# 现在使用admittime来计算实际的时间窗口
def get_time_windows(df):
    """为每个state记录生成时间窗口（基于admittime + time_step）"""
    windows = []
    for _, row in df.iterrows():
        # 将admittime转换为datetime对象
        admit_time = pd.to_datetime(row['admittime'])
        # 每个time_step代表24小时，计算实际时间窗口
        window_start = admit_time + pd.Timedelta(hours=24 * row['time_step'])
        window_end = admit_time + pd.Timedelta(hours=24 * (row['time_step'] + 1))
        
        windows.append({
            'hadm_id': row['hadm_id'],
            'time_step': row['time_step'],
            'admittime': admit_time,
            'window_start': window_start,
            'window_end': window_end
        })
    return pd.DataFrame(windows)

print("生成时间窗口...")
time_windows = get_time_windows(state_data[['hadm_id', 'time_step', 'admittime']])
print(f"时间窗口数量: {len(time_windows)}")

# 步骤4：计算药物使用情况
print("\n💊 步骤4：计算药物使用情况")

# 合并处方数据和时间窗口
pres_windows = pres_data.merge(time_windows, left_on='HADM_ID', right_on='hadm_id')

# 判断药物是否在时间窗口内使用
def is_drug_active(row):
    """判断药物是否在时间窗口内活跃（基于实际时间对齐）"""
    # 检查药物的使用时间是否与时间窗口重叠
    drug_start = pd.to_datetime(row['STARTDATE'])
    drug_end = pd.to_datetime(row['ENDDATE'])
    window_start = pd.to_datetime(row['window_start'])
    window_end = pd.to_datetime(row['window_end'])
    
    # 判断两个时间段是否有重叠
    # 重叠条件：药物开始时间 < 窗口结束时间 AND 药物结束时间 > 窗口开始时间
    return (drug_start < window_end) and (drug_end > window_start)

print("计算药物使用情况...")
pres_windows['is_active'] = pres_windows.apply(is_drug_active, axis=1)

# 筛选活跃药物
active_drugs = pres_windows[pres_windows['is_active']]
print(f"活跃药物记录数: {len(active_drugs)}")

# 步骤5：构建药物动作矩阵
print("\n📊 步骤5：构建药物动作矩阵")

# 聚合每个时间步的药物使用情况
drug_usage = active_drugs.groupby(['hadm_id', 'time_step', 'drug_idx']).size().reset_index(name='count')

# 转换为二进制矩阵（使用或不使用）
drug_usage['used'] = 1

# 创建pivot表
drug_matrix = drug_usage.pivot_table(
    index=['hadm_id', 'time_step'],
    columns='drug_idx',
    values='used',
    fill_value=0
).reset_index()

# 确保所有药物索引都存在
for idx in all_drug_indices:
    if idx not in drug_matrix.columns:
        drug_matrix[idx] = 0

# 重新排序列，确保列名是整数而非字符串
action_columns = ['hadm_id', 'time_step'] + sorted(all_drug_indices)
drug_matrix = drug_matrix[action_columns]

# 将数字列名转换为整数
print("转换药物动作列名从字符串到整数...")
rename_dict = {}
for col in drug_matrix.columns:
    if col not in ['hadm_id', 'time_step']:
        rename_dict[col] = int(col)

drug_matrix = drug_matrix.rename(columns=rename_dict)
print(f"重命名后的药物动作列前10个: {list(drug_matrix.columns[2:12])}")

print(f"药物动作矩阵 shape: {drug_matrix.shape}")

# 步骤6：合并所有数据
print("\n🔗 步骤6：合并所有数据")

# 合并state数据和drug数据
final_dataset = state_data.merge(
    drug_matrix, 
    on=['hadm_id', 'time_step'], 
    how='left'
)

# 填充缺失的药物动作为0
action_idx_cols = [col for col in sorted(all_drug_indices) if col in final_dataset.columns]
final_dataset[action_idx_cols] = final_dataset[action_idx_cols].fillna(0).astype('int8')

print(f"药物动作列数量: {len(action_idx_cols)}")
print(f"药物动作列类型: {type(action_idx_cols[0]) if action_idx_cols else 'None'}")
print(f"药物动作列前10个: {action_idx_cols[:10]}")

# 不添加"无药物"标志列，避免维度不匹配
# final_dataset['no_drug_action'] = (final_dataset[action_idx_cols].sum(axis=1) == 0).astype('int8')

print(f"最终数据集 shape: {final_dataset.shape}")
# 计算无药物动作记录数
no_drug_count = (final_dataset[action_idx_cols].sum(axis=1) == 0).sum()
print(f"无药物动作记录数: {no_drug_count}")

# 步骤7：数据质量检查
print("\n🔍 步骤7：数据质量检查")

print("数据质量统计:")
print(f"总记录数: {len(final_dataset)}")
print(f"唯一患者数: {final_dataset['hadm_id'].nunique()}")
print(f"时间步范围: {final_dataset['time_step'].min()} - {final_dataset['time_step'].max()}")

# 检查药物使用分布
drug_usage_stats = final_dataset[action_idx_cols].sum().sort_values(ascending=False)
print(f"最常用的5种药物索引: {drug_usage_stats.head().to_dict()}")

# 步骤8：保存结果
print("\n💾 步骤8：保存结果")

# 创建输出目录
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# 保存最终数据集
final_dataset.to_csv(OUTPUT_CSV, index=False)
print(f"✅ 训练数据集已保存到: {OUTPUT_CSV}")

# 保存元数据
metadata = {
    'dataset_shape': final_dataset.shape,
    'num_patients': int(final_dataset['hadm_id'].nunique()),
    'num_drug_actions': len(action_idx_cols),
    'time_step_range': [int(final_dataset['time_step'].min()), int(final_dataset['time_step'].max())],
    'drug_usage_stats': drug_usage_stats.head(10).to_dict(),
    'drug_mapping': drug_idx_info,
    'column_info': {
        'state_features': len(state_data.columns) - 2,  # 减去hadm_id和time_step
        'drug_actions': len(action_idx_cols),
        'total_columns': len(final_dataset.columns)
    },
    'action_column_info': {
        'action_columns': action_idx_cols,
        'action_column_type': str(type(action_idx_cols[0])) if action_idx_cols else 'None',
        'no_drug_records': int(no_drug_count)
    }
}

metadata_file = OUTPUT_CSV.replace('.csv', '_metadata.json')
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ 元数据已保存到: {metadata_file}")

print("\n" + "=" * 60)
print("训练数据集创建完成！")
print("=" * 60)