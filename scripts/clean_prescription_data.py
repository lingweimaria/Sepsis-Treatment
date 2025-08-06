#!/usr/bin/env python3
"""
清洗处方数据中的时间异常
"""

import pandas as pd
import numpy as np
import json

def clean_prescription_data():
    """清洗处方数据的时间异常"""
    
    print("=" * 60)
    print("处方数据时间清洗")
    print("=" * 60)
    
    # 读取原始数据
    print("\n📥 读取原始数据...")
    pres_data = pd.read_csv('data/raw/top 1000 medications_mimic3.csv', 
                           usecols=['HADM_ID', 'STARTDATE', 'ENDDATE', 'DRUG'],
                           dtype={'HADM_ID': 'int32'},
                           parse_dates=['STARTDATE', 'ENDDATE'])
    
    print(f"原始记录数: {len(pres_data):,}")
    
    # 记录清洗统计
    cleaning_stats = {
        'original_count': len(pres_data),
        'removed_counts': {}
    }
    
    # 1. 移除异常年份数据
    print("\n🧹 步骤1: 移除异常年份数据")
    
    # 移除2100年前的数据
    early_records = pres_data[pres_data['STARTDATE'].dt.year < 2100]
    cleaning_stats['removed_counts']['early_records'] = len(early_records)
    pres_data = pres_data[pres_data['STARTDATE'].dt.year >= 2100]
    print(f"移除2100年前的记录: {len(early_records):,}")
    
    # 移除2200年后的数据
    late_records = pres_data[pres_data['STARTDATE'].dt.year > 2200]
    cleaning_stats['removed_counts']['late_records'] = len(late_records)
    pres_data = pres_data[pres_data['STARTDATE'].dt.year <= 2200]
    print(f"移除2200年后的记录: {len(late_records):,}")
    
    # 2. 移除负持续时间记录
    print("\n🧹 步骤2: 移除负持续时间记录")
    negative_duration = pres_data[pres_data['ENDDATE'] < pres_data['STARTDATE']]
    cleaning_stats['removed_counts']['negative_duration'] = len(negative_duration)
    pres_data = pres_data[pres_data['ENDDATE'] >= pres_data['STARTDATE']]
    print(f"移除负持续时间记录: {len(negative_duration):,}")
    
    # 3. 移除极长持续时间记录 (>1年)
    print("\n🧹 步骤3: 移除极长持续时间记录")
    pres_data['duration_days'] = (pres_data['ENDDATE'] - pres_data['STARTDATE']).dt.days
    extreme_duration = pres_data[pres_data['duration_days'] > 365]
    cleaning_stats['removed_counts']['extreme_duration'] = len(extreme_duration)
    pres_data = pres_data[pres_data['duration_days'] <= 365]
    print(f"移除超过1年的记录: {len(extreme_duration):,}")
    
    # 4. 移除极短持续时间记录 (0小时)
    print("\n🧹 步骤4: 移除极短持续时间记录")
    zero_duration = pres_data[pres_data['duration_days'] == 0]
    cleaning_stats['removed_counts']['zero_duration'] = len(zero_duration)
    pres_data = pres_data[pres_data['duration_days'] > 0]
    print(f"移除0持续时间记录: {len(zero_duration):,}")
    
    # 移除临时计算列
    pres_data = pres_data.drop(columns=['duration_days'])
    
    # 5. 清洗结果统计
    print("\n📊 清洗结果统计:")
    cleaning_stats['cleaned_count'] = len(pres_data)
    cleaning_stats['total_removed'] = sum(cleaning_stats['removed_counts'].values())
    cleaning_stats['retention_rate'] = len(pres_data) / cleaning_stats['original_count']
    
    print(f"原始记录: {cleaning_stats['original_count']:,}")
    print(f"清洗后记录: {cleaning_stats['cleaned_count']:,}")
    print(f"移除记录: {cleaning_stats['total_removed']:,}")
    print(f"保留率: {cleaning_stats['retention_rate']*100:.1f}%")
    
    # 详细移除统计
    print("\n移除记录详情:")
    for category, count in cleaning_stats['removed_counts'].items():
        print(f"  {category}: {count:,}")
    
    # 6. 验证清洗后数据质量
    print("\n✅ 验证清洗后数据质量:")
    pres_data['duration_hours'] = (pres_data['ENDDATE'] - pres_data['STARTDATE']).dt.total_seconds() / 3600
    
    print(f"时间范围: {pres_data['STARTDATE'].min()} 到 {pres_data['ENDDATE'].max()}")
    print(f"持续时间统计:")
    print(f"  平均: {pres_data['duration_hours'].mean():.1f}小时")
    print(f"  中位数: {pres_data['duration_hours'].median():.1f}小时")
    print(f"  最短: {pres_data['duration_hours'].min():.1f}小时")
    print(f"  最长: {pres_data['duration_hours'].max():.1f}小时")
    
    # 检查是否还有异常
    print(f"\n异常检查:")
    print(f"  负持续时间: {(pres_data['duration_hours'] < 0).sum()}")
    print(f"  超过1年: {(pres_data['duration_hours'] > 365*24).sum()}")
    print(f"  年份范围: {pres_data['STARTDATE'].dt.year.min()} - {pres_data['STARTDATE'].dt.year.max()}")
    
    # 7. 保存清洗后的数据
    print("\n💾 保存清洗后的数据...")
    output_path = 'data/processed/cleaned_prescriptions.csv'
    pres_data.to_csv(output_path, index=False)
    print(f"清洗后数据已保存到: {output_path}")
    
    # 保存清洗统计
    stats_path = 'data/processed/prescription_cleaning_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(cleaning_stats, f, indent=2)
    print(f"清洗统计已保存到: {stats_path}")
    
    print("\n" + "=" * 60)
    print("处方数据清洗完成！")
    print("=" * 60)
    
    return pres_data, cleaning_stats

if __name__ == "__main__":
    import os
    os.chdir('/Users/lulingwei/Desktop/TUM/sem_2/reinforcement learning/代码/sepsis_rl_system')
    
    cleaned_data, stats = clean_prescription_data()
    
    print("\n建议:")
    print("1. 使用清洗后的数据重新生成训练数据集")
    print("2. 清洗后的数据时间范围更合理，应该能减少过滤率")
    print("3. 可以考虑在create_training_dataset.py中使用cleaned_prescriptions.csv")