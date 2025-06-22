#!/usr/bin/env python3
"""
MIMIC-III专家决策提取器
从真实医疗数据中提取医生的治疗决策，用于监督学习
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import json
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class MIMICExpertDecisionExtractor:
    """从MIMIC-III数据提取专家决策"""
    
    def __init__(self, raw_data_dir, output_dir):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建专家数据目录
        self.expert_data_dir = self.output_dir / "expert_data"
        self.expert_data_dir.mkdir(exist_ok=True)
        
        # 药物到动作的映射
        self.drug_action_mapping = self._create_drug_action_mapping()
        
        # 数据缓存
        self.medications_cache = None
        self.vital_signs_cache = None
        self.demographics_cache = None
        
    def _create_drug_action_mapping(self):
        """创建药物到治疗动作的映射"""
        return {
            # 0: 观察/维持治疗
            'observation': 0,
            
            # 1-4: 心血管系统药物
            'vasopressor': 1,       # 血管加压药
            'antihypertensive': 2,  # 降压药  
            'diuretic': 3,          # 利尿剂
            'beta_blocker': 4,      # β受体阻滞剂
            
            # 5-8: 呼吸系统
            'oxygen_therapy': 5,    # 氧疗
            'bronchodilator': 6,    # 支气管扩张剂
            'steroid': 7,           # 类固醇
            'ventilation': 8,       # 机械通气
            
            # 9-12: 代谢支持
            'insulin': 9,           # 胰岛素
            'glucose': 10,          # 葡萄糖
            'electrolyte': 11,      # 电解质补充
            'nutrition': 12,        # 营养支持
            
            # 13-16: 感染控制
            'antibiotic': 13,       # 抗生素
            'antifungal': 14,       # 抗真菌
            'antiviral': 15,        # 抗病毒
            'antiseptic': 16,       # 防腐剂
            
            # 17-19: 其他治疗
            'analgesic': 17,        # 镇痛剂
            'sedative': 18,         # 镇静剂
            'anticoagulant': 19     # 抗凝剂
        }
    
    def _map_drug_to_action(self, drug_name):
        """将药物名称映射到动作ID"""
        drug_name = drug_name.lower() if drug_name else ""
        
        # 血管加压药
        if any(word in drug_name for word in ['norepinephrine', 'epinephrine', 'dopamine', 'vasopressin', 'phenylephrine']):
            return self.drug_action_mapping['vasopressor']
        
        # 降压药
        elif any(word in drug_name for word in ['lisinopril', 'amlodipine', 'losartan', 'hydralazine', 'nifedipine']):
            return self.drug_action_mapping['antihypertensive']
        
        # 利尿剂
        elif any(word in drug_name for word in ['furosemide', 'hydrochlorothiazide', 'spironolactone', 'torsemide']):
            return self.drug_action_mapping['diuretic']
        
        # β受体阻滞剂
        elif any(word in drug_name for word in ['metoprolol', 'propranolol', 'atenolol', 'carvedilol', 'esmolol']):
            return self.drug_action_mapping['beta_blocker']
        
        # 胰岛素
        elif any(word in drug_name for word in ['insulin']):
            return self.drug_action_mapping['insulin']
        
        # 葡萄糖/糖水
        elif any(word in drug_name for word in ['dextrose', 'd5w', 'glucose', '5% dextrose']):
            return self.drug_action_mapping['glucose']
        
        # 电解质
        elif any(word in drug_name for word in ['potassium', 'magnesium', 'calcium', 'phosphate', 'sodium chloride']):
            return self.drug_action_mapping['electrolyte']
        
        # 抗生素
        elif any(word in drug_name for word in ['vancomycin', 'piperacillin', 'ceftriaxone', 'levofloxacin', 'meropenem']):
            return self.drug_action_mapping['antibiotic']
        
        # 镇痛剂
        elif any(word in drug_name for word in ['morphine', 'fentanyl', 'acetaminophen', 'oxycodone']):
            return self.drug_action_mapping['analgesic']
        
        # 镇静剂
        elif any(word in drug_name for word in ['propofol', 'midazolam', 'lorazepam', 'dexmedetomidine']):
            return self.drug_action_mapping['sedative']
        
        # 抗凝剂
        elif any(word in drug_name for word in ['heparin', 'warfarin', 'enoxaparin']):
            return self.drug_action_mapping['anticoagulant']
        
        # 类固醇
        elif any(word in drug_name for word in ['hydrocortisone', 'methylprednisolone', 'prednisone']):
            return self.drug_action_mapping['steroid']
        
        # 默认为观察
        else:
            return self.drug_action_mapping['observation']
    
    def load_raw_data(self):
        """加载原始MIMIC-III数据"""
        print("📊 加载MIMIC-III原始数据...")
        
        # 加载药物数据
        print("  - 加载药物数据...")
        medications_file = self.raw_data_dir / "top 1000 medications_mimic3.csv"
        self.medications_cache = pd.read_csv(medications_file)
        print(f"    药物记录: {len(self.medications_cache):,} 条")
        
        # 加载生命体征数据
        print("  - 加载生命体征数据...")
        vital_signs_file = self.raw_data_dir / "time series variables(vital signs)_mimic3.csv"
        # 由于文件很大，只读取一部分用于示例
        vital_signs_chunk = pd.read_csv(vital_signs_file, nrows=500000)  # 读取50万行作为示例
        self.vital_signs_cache = vital_signs_chunk
        print(f"    生命体征记录: {len(self.vital_signs_cache):,} 条 (示例数据)")
        
        # 加载人口统计学数据
        print("  - 加载人口统计学数据...")
        demographics_file = self.raw_data_dir / "static variables(demographics)_mimic3.csv"
        self.demographics_cache = pd.read_csv(demographics_file)
        print(f"    人口统计学记录: {len(self.demographics_cache):,} 条")
        
        print("✅ 数据加载完成!")
    
    def extract_expert_decisions(self, max_patients=100, time_window_hours=6):
        """提取专家决策数据"""
        print(f"🔍 开始提取专家决策 (最多{max_patients}个患者, 时间窗口{time_window_hours}小时)...")
        
        if self.medications_cache is None:
            self.load_raw_data()
        
        expert_decisions = []
        processed_patients = 0
        
        # 预处理数据
        self._preprocess_data()
        
        # 获取有重叠数据的患者
        medication_patients = set(self.medications_cache['SUBJECT_ID'].unique())
        vital_signs_patients = set(self.vital_signs_cache['subject_id'].unique())
        common_patients = list(medication_patients & vital_signs_patients)
        
        print(f"  - 找到 {len(common_patients)} 个有完整数据的患者")
        
        # 限制处理的患者数量
        if max_patients:
            common_patients = common_patients[:max_patients]
        
        for patient_id in common_patients:
            if processed_patients >= max_patients:
                break
                
            try:
                patient_decisions = self._extract_patient_decisions(
                    patient_id, time_window_hours
                )
                expert_decisions.extend(patient_decisions)
                processed_patients += 1
                
                if processed_patients % 10 == 0:
                    print(f"    已处理 {processed_patients}/{max_patients} 个患者, "
                          f"提取决策 {len(expert_decisions)} 个")
                    
            except Exception as e:
                print(f"    ⚠️ 处理患者 {patient_id} 时出错: {e}")
                continue
        
        print(f"✅ 专家决策提取完成! 总计 {len(expert_decisions)} 个决策")
        return expert_decisions
    
    def _preprocess_data(self):
        """预处理数据"""
        # 转换时间格式
        self.medications_cache['STARTDATE'] = pd.to_datetime(
            self.medications_cache['STARTDATE'], errors='coerce'
        )
        self.vital_signs_cache['charttime'] = pd.to_datetime(
            self.vital_signs_cache['charttime'], errors='coerce'
        )
        
        # 移除无效时间数据
        self.medications_cache = self.medications_cache.dropna(subset=['STARTDATE'])
        self.vital_signs_cache = self.vital_signs_cache.dropna(subset=['charttime'])
        
        # 排序
        self.medications_cache = self.medications_cache.sort_values(['SUBJECT_ID', 'STARTDATE'])
        self.vital_signs_cache = self.vital_signs_cache.sort_values(['subject_id', 'charttime'])
    
    def _extract_patient_decisions(self, patient_id, time_window_hours):
        """提取单个患者的专家决策"""
        patient_decisions = []
        
        # 获取患者的药物和生命体征数据
        patient_meds = self.medications_cache[
            self.medications_cache['SUBJECT_ID'] == patient_id
        ].copy()
        
        patient_vitals = self.vital_signs_cache[
            self.vital_signs_cache['subject_id'] == patient_id
        ].copy()
        
        if len(patient_meds) == 0 or len(patient_vitals) == 0:
            return patient_decisions
        
        # 为每个用药决策找到对应的状态
        for _, med_record in patient_meds.iterrows():
            drug_time = med_record['STARTDATE']
            drug_name = med_record['DRUG']
            
            # 跳过无效药物记录
            if pd.isna(drug_time) or not drug_name:
                continue
            
            # 映射药物到动作
            action = self._map_drug_to_action(drug_name)
            
            # 获取用药前的生命体征状态
            time_cutoff = drug_time - timedelta(hours=time_window_hours)
            relevant_vitals = patient_vitals[
                (patient_vitals['charttime'] >= time_cutoff) & 
                (patient_vitals['charttime'] <= drug_time)
            ]
            
            if len(relevant_vitals) == 0:
                continue
            
            # 提取状态特征
            state_features = self._extract_state_features(relevant_vitals)
            
            if state_features is not None:
                patient_decisions.append({
                    'patient_id': patient_id,
                    'timestamp': drug_time,
                    'drug_name': drug_name,
                    'action': action,
                    'state_features': state_features,
                    'vital_signs_count': len(relevant_vitals)
                })
        
        return patient_decisions
    
    def _extract_state_features(self, vitals_data):
        """从生命体征数据提取状态特征"""
        try:
            # 重要的生命体征标签
            important_labels = [
                'Temperature F', 'Temperature C',
                'Heart Rate', 'Respiratory Rate',
                'Arterial Blood Pressure systolic',
                'Arterial Blood Pressure diastolic', 
                'O2 saturation pulseoxymetry',
                'GCS - Eye Opening',
                'GCS - Motor Response', 
                'GCS - Verbal Response'
            ]
            
            features = {}
            
            for label in important_labels:
                label_data = vitals_data[vitals_data['label'] == label]
                if len(label_data) > 0:
                    # 使用最新的值
                    latest_value = label_data.iloc[-1]['valuenum']
                    if pd.notna(latest_value):
                        features[label.lower().replace(' ', '_').replace('-', '_')] = latest_value
            
            # 检查是否有足够的特征
            if len(features) < 5:  # 至少需要5个有效特征
                return None
            
            # 填充缺失特征的默认值
            feature_defaults = {
                'temperature_f': 98.6,
                'temperature_c': 37.0,
                'heart_rate': 80,
                'respiratory_rate': 16,
                'arterial_blood_pressure_systolic': 120,
                'arterial_blood_pressure_diastolic': 80,
                'o2_saturation_pulseoxymetry': 98,
                'gcs___eye_opening': 4,
                'gcs___motor_response': 6,
                'gcs___verbal_response': 5
            }
            
            # 使用华氏度转摄氏度
            if 'temperature_f' in features and 'temperature_c' not in features:
                features['temperature_c'] = (features['temperature_f'] - 32) * 5/9
            elif 'temperature_c' not in features and 'temperature_f' not in features:
                features['temperature_c'] = feature_defaults['temperature_c']
            
            # 填充其他缺失值
            for key, default_val in feature_defaults.items():
                if key not in features and key != 'temperature_f':
                    features[key] = default_val
            
            # 转换为向量格式 (与当前模型输入一致)
            feature_vector = [
                features.get('temperature_c', 37.0),
                features.get('heart_rate', 80),
                features.get('respiratory_rate', 16),
                features.get('arterial_blood_pressure_systolic', 120),
                features.get('arterial_blood_pressure_diastolic', 80),
                features.get('o2_saturation_pulseoxymetry', 98),
                0,  # 占位符 (glucose等其他特征)
                features.get('gcs___eye_opening', 4),
                features.get('gcs___motor_response', 6),
                features.get('gcs___verbal_response', 5),
                35,  # age占位符
                0   # sofa_score占位符
            ]
            
            return np.array(feature_vector)
            
        except Exception as e:
            print(f"      特征提取错误: {e}")
            return None
    
    def save_expert_decisions(self, expert_decisions):
        """保存专家决策数据"""
        print("💾 保存专家决策数据...")
        
        if not expert_decisions:
            print("❌ 没有专家决策数据可保存")
            return
        
        # 转换为DataFrame
        decisions_df = []
        for decision in expert_decisions:
            row = {
                'patient_id': decision['patient_id'],
                'timestamp': decision['timestamp'],
                'drug_name': decision['drug_name'],
                'expert_action': decision['action'],
                'vital_signs_count': decision['vital_signs_count']
            }
            
            # 添加状态特征
            state_features = decision['state_features']
            feature_names = [
                'temperature', 'heart_rate', 'respiratory_rate',
                'bp_systolic', 'bp_diastolic', 'spo2', 'glucose',
                'gcs_eye', 'gcs_motor', 'gcs_verbal', 'age', 'sofa_score'
            ]
            
            for i, feature_name in enumerate(feature_names):
                if i < len(state_features):
                    row[feature_name] = state_features[i]
                else:
                    row[feature_name] = 0
            
            decisions_df.append(row)
        
        df = pd.DataFrame(decisions_df)
        
        # 保存到CSV
        csv_path = self.expert_data_dir / "expert_decisions.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ 专家决策已保存到: {csv_path}")
        
        # 保存为pickle格式
        pickle_path = self.expert_data_dir / "expert_decisions.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(expert_decisions, f)
        print(f"✅ 专家决策已保存到: {pickle_path}")
        
        # 生成统计报告
        self._generate_statistics_report(df)
        
        return csv_path
    
    def _generate_statistics_report(self, decisions_df):
        """生成专家决策统计报告"""
        print("📊 生成统计报告...")
        
        stats = {
            'total_decisions': len(decisions_df),
            'unique_patients': decisions_df['patient_id'].nunique(),
            'action_distribution': decisions_df['expert_action'].value_counts().to_dict(),
            'drug_distribution': decisions_df['drug_name'].value_counts().head(20).to_dict(),
            'feature_statistics': {}
        }
        
        # 特征统计
        feature_cols = ['temperature', 'heart_rate', 'respiratory_rate', 
                       'bp_systolic', 'bp_diastolic', 'spo2']
        
        for col in feature_cols:
            if col in decisions_df.columns:
                stats['feature_statistics'][col] = {
                    'mean': float(decisions_df[col].mean()),
                    'std': float(decisions_df[col].std()),
                    'min': float(decisions_df[col].min()),
                    'max': float(decisions_df[col].max())
                }
        
        # 保存统计报告
        stats_path = self.expert_data_dir / "expert_decisions_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 统计报告已保存到: {stats_path}")
        
        # 打印简要统计
        print("\n📈 专家决策统计摘要:")
        print(f"  - 总决策数: {stats['total_decisions']:,}")
        print(f"  - 患者数: {stats['unique_patients']:,}")
        print(f"  - 动作分布:")
        for action, count in sorted(stats['action_distribution'].items()):
            percentage = count / stats['total_decisions'] * 100
            print(f"    动作 {action}: {count:,} ({percentage:.1f}%)")
    
    def run_extraction(self, max_patients=100, time_window_hours=6):
        """运行完整的专家决策提取流程"""
        print("🚀 开始MIMIC-III专家决策提取流程...")
        print(f"参数: 最大患者数={max_patients}, 时间窗口={time_window_hours}小时")
        
        try:
            # 1. 加载数据
            self.load_raw_data()
            
            # 2. 提取专家决策
            expert_decisions = self.extract_expert_decisions(
                max_patients=max_patients,
                time_window_hours=time_window_hours
            )
            
            # 3. 保存结果
            if expert_decisions:
                csv_path = self.save_expert_decisions(expert_decisions)
                print(f"\n🎉 专家决策提取完成!")
                print(f"📁 输出文件: {csv_path}")
                return csv_path
            else:
                print("❌ 未提取到任何专家决策")
                return None
                
        except Exception as e:
            print(f"❌ 专家决策提取失败: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """主函数"""
    # 设置路径
    current_dir = Path(__file__).parent
    raw_data_dir = current_dir.parent / "raw_data" / "mimic3_original"
    output_dir = current_dir.parent / "processed_data"
    
    # 检查原始数据目录
    if not raw_data_dir.exists():
        print(f"❌ 原始数据目录不存在: {raw_data_dir}")
        return
    
    # 创建提取器
    extractor = MIMICExpertDecisionExtractor(raw_data_dir, output_dir)
    
    # 运行提取
    result_path = extractor.run_extraction(
        max_patients=100,      # 处理100个患者作为示例
        time_window_hours=6    # 6小时的时间窗口
    )
    
    if result_path:
        print(f"\n✅ 专家决策数据已准备就绪!")
        print(f"📂 数据文件: {result_path}")
        print("🔄 现在可以在训练中使用真实专家数据了!")
    else:
        print("\n❌ 专家决策提取失败")

if __name__ == "__main__":
    main()