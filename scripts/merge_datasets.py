#!/usr/bin/env python3
"""
数据集合并脚本
将 sofa.csv 和 final_df.csv 进行合并
"""

import pandas as pd
import numpy as np

# 文件路径
SOFA_CSV = "/Users/lulingwei/Desktop/TUM/sem_2/reinforcement learning/代码/sepsis_rl_system/data/raw/sofa.csv"
FINAL_DF_CSV = "/Users/lulingwei/Desktop/TUM/sem_2/reinforcement learning/代码/sepsis_rl_system/data/raw/final_df.csv"
OUTPUT_CSV = "/Users/lulingwei/Desktop/TUM/sem_2/reinforcement learning/代码/sepsis_rl_system/data/processed/merged_dataset.csv"

print("=" * 60)
print("开始数据集合并流程")
print("=" * 60)

# 步骤1：读取数据
print("\n📥 步骤1：读取数据")
print("正在读取 sofa.csv...")
sofa_df = pd.read_csv(SOFA_CSV)
print(f"sofa.csv 形状: {sofa_df.shape}")
print(f"sofa.csv 列名: {list(sofa_df.columns)}")

print("\n正在读取 final_df.csv...")
final_df = pd.read_csv(FINAL_DF_CSV, parse_dates=['admittime'])
print(f"final_df.csv 形状: {final_df.shape}")
print(f"final_df.csv 列名: {list(final_df.columns)}")

# 步骤2：数据清理
print("\n🧹 步骤2：数据清理")

# 2.1 清理 sofa 数据
print("清理 sofa 数据...")
sofa_clean = sofa_df.copy()
# 确保关键列的数据类型正确
sofa_clean['hadm_id'] = sofa_clean['hadm_id'].astype('int64')
sofa_clean['time_step'] = sofa_clean['time_step'].astype('int64')
print(f"sofa 清理后形状: {sofa_clean.shape}")

# 2.2 清理 final_df 数据
print("清理 final_df 数据...")
final_clean = final_df.copy()
# 确保关键列的数据类型正确
final_clean['hadm_id'] = final_clean['hadm_id'].astype('int64')
final_clean['time_step'] = final_clean['time_step'].astype('int64')
print(f"final_df 清理后形状: {final_clean.shape}")

# 步骤3：分离静态数据和时序数据
print("\n🔄 步骤3：分离静态数据和时序数据")

# 3.1 识别静态列（在final_df中，对于同一个hadm_id，这些列的值应该是相同的）
# 只保留与治疗相关的静态特征，包含admittime用于时间对齐
static_columns = [
    'subject_id', 'gender', 'age', 'weight_kg', 
    'height_cm', 'hospital_expire_flag', 'admittime'
]

# 3.2 提取静态数据
print("提取静态数据...")
static_data = final_clean[['hadm_id'] + static_columns].drop_duplicates(subset=['hadm_id'])
print(f"静态数据形状: {static_data.shape}")
print(f"静态数据列名: {list(static_data.columns)}")

# 3.3 特征清理和编码
print("\n🔧 步骤3.3：特征清理和编码")

# 删除与治疗无关的特征 (保留admittime用于时间对齐)
columns_to_remove = ['religion', 'language', 'marital_status', 'ethnicity']
print(f"删除与治疗无关的特征: {columns_to_remove}")
print("保留admittime字段用于后续时间对齐")
existing_cols_to_remove = [col for col in columns_to_remove if col in final_clean.columns]
if existing_cols_to_remove:
    final_clean = final_clean.drop(columns=existing_cols_to_remove)
    print(f"已删除列: {existing_cols_to_remove}")
    print(f"final_clean清理后形状: {final_clean.shape}")
else:
    print("未找到需要删除的列")

print("对gender进行二进制编码...")
print(f"Gender编码前分布: {static_data['gender'].value_counts().to_dict()}")
# 对gender进行二进制编码 (M=1, F=0)
static_data['gender'] = static_data['gender'].map({'M': 1, 'F': 0})
print(f"Gender编码后分布: {static_data['gender'].value_counts().to_dict()}")

# 检查是否有未成功编码的数据
gender_missing = static_data['gender'].isnull().sum()
if gender_missing > 0:
    print(f"⚠️  警告: 有 {gender_missing} 个gender值未能成功编码")

print("✅ 特征清理和编码完成")

# 3.4 识别时序列
final_temporal_columns = [col for col in final_clean.columns if col not in static_columns]
sofa_temporal_columns = [col for col in sofa_clean.columns]

print(f"final_df 时序列数量: {len(final_temporal_columns)}")
print(f"sofa 时序列数量: {len(sofa_temporal_columns)}")

# 步骤4：准备时序数据合并
print("\n🔗 步骤4：准备时序数据合并")

# 4.1 从final_df中提取时序数据
final_temporal = final_clean[final_temporal_columns].copy()
print(f"final_df 时序数据形状: {final_temporal.shape}")

# 4.2 sofa数据已经是时序数据
sofa_temporal = sofa_clean.copy()
print(f"sofa 时序数据形状: {sofa_temporal.shape}")

# 步骤5：合并时序数据
print("\n🔄 步骤5：合并时序数据")

# 检查重叠的列（除了hadm_id和time_step）
common_cols = set(final_temporal.columns) & set(sofa_temporal.columns)
merge_keys = {'hadm_id', 'time_step'}
overlap_cols = common_cols - merge_keys
if overlap_cols:
    print(f"⚠️  发现重叠列: {overlap_cols}")
    # 对重叠列添加后缀以区分
    sofa_temporal = sofa_temporal.rename(columns={col: f"{col}_sofa" for col in overlap_cols})

# 执行左连接（以final_df为主）
print("执行时序数据合并...")
merged_temporal = final_temporal.merge(
    sofa_temporal, 
    on=['hadm_id', 'time_step'], 
    how='left',
    suffixes=('', '_sofa')
)
print(f"合并后时序数据形状: {merged_temporal.shape}")

# 步骤6：与静态数据连接
print("\n🔗 步骤6：与静态数据连接")

# 将合并后的时序数据与静态数据进行左连接
final_merged = merged_temporal.merge(
    static_data,
    on='hadm_id',
    how='left'
)
print(f"最终合并数据形状: {final_merged.shape}")

# 步骤7：数据质量检查
print("\n🔍 步骤7：数据质量检查")

print("检查合并后的数据质量...")
print(f"总行数: {len(final_merged)}")
print(f"唯一hadm_id数量: {final_merged['hadm_id'].nunique()}")
print(f"time_step范围: {final_merged['time_step'].min()} 到 {final_merged['time_step'].max()}")

# 检查缺失值
missing_counts = final_merged.isnull().sum()
high_missing = missing_counts[missing_counts > len(final_merged) * 0.5]
if not high_missing.empty:
    print(f"⚠️  高缺失率列 (>50%): {high_missing.to_dict()}")

# 步骤8：保存结果
print("\n💾 步骤8：保存结果")

# 创建输出目录
import os
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# 保存合并后的数据
final_merged.to_csv(OUTPUT_CSV, index=False)
print(f"✅ 数据已保存到: {OUTPUT_CSV}")

# 保存数据摘要
summary_file = OUTPUT_CSV.replace('.csv', '_summary.txt')
with open(summary_file, 'w') as f:
    f.write("数据合并摘要\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"最终数据形状: {final_merged.shape}\n")
    f.write(f"列数: {len(final_merged.columns)}\n")
    f.write(f"行数: {len(final_merged)}\n")
    f.write(f"唯一hadm_id数量: {final_merged['hadm_id'].nunique()}\n")
    f.write(f"time_step范围: {final_merged['time_step'].min()} 到 {final_merged['time_step'].max()}\n\n")
    f.write("特征清理说明:\n")
    f.write("- 已删除与治疗无关的特征: religion, language, marital_status, ethnicity\n")
    f.write("- 已对gender进行二进制编码 (M=1, F=0)\n")
    f.write("- 保留的静态特征: subject_id, gender, age, weight_kg, height_cm, hospital_expire_flag, admittime\n")
    f.write("- admittime字段保留用于后续药物数据时间对齐\n\n")
    f.write("列名列表:\n")
    for i, col in enumerate(final_merged.columns, 1):
        f.write(f"{i:3d}. {col}\n")
    f.write("\n缺失值统计:\n")
    for col, count in missing_counts.items():
        if count > 0:
            f.write(f"{col}: {count} ({count/len(final_merged)*100:.1f}%)\n")

print(f"✅ 数据摘要已保存到: {summary_file}")

print("\n" + "=" * 60)
print("数据合并完成！")
print("=" * 60)