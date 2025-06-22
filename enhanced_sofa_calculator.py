#!/usr/bin/env python3
"""
增强版SOFA评分计算器
集成完整MIMIC-III数据源，包括血管活性药物、尿量输出等
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class EnhancedSOFACalculator:
    """增强版SOFA评分计算器"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        
        # 血管活性药物映射
        self.vasopressor_mapping = {
            'Norepinephrine': {'weight': 1.0, 'category': 'high'},
            'Epinephrine': {'weight': 1.0, 'category': 'high'}, 
            'Dopamine': {'weight': 0.8, 'category': 'medium'},
            'Dobutamine': {'weight': 0.6, 'category': 'medium'},
            'Vasopressin': {'weight': 1.0, 'category': 'high'},
            'Phenylephrine': {'weight': 0.7, 'category': 'medium'}
        }
        
        # 缓存数据
        self._vasopressor_data = None
        self._urine_data = None
        self._lab_data = None
    
    def load_enhanced_data_sources(self):
        """加载增强的数据源"""
        logger.info("🔄 加载增强数据源...")
        
        try:
            # 加载血管活性药物数据
            self._load_vasopressor_data()
            
            # 加载尿量数据
            self._load_urine_data()
            
            # 加载实验室数据
            self._load_lab_data()
            
            logger.info("✅ 所有增强数据源加载完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 加载增强数据源失败: {e}")
            return False
    
    def _load_vasopressor_data(self):
        """加载血管活性药物数据"""
        logger.info("💊 加载血管活性药物数据...")
        
        meds_file = self.data_dir / 'top 1000 medications_mimic3.csv'
        if not meds_file.exists():
            logger.warning("药物数据文件不存在")
            return
        
        # 分块读取以节省内存
        vasopressor_records = []
        chunk_size = 100000
        
        for chunk in pd.read_csv(meds_file, chunksize=chunk_size):
            for drug_name, drug_info in self.vasopressor_mapping.items():
                drug_chunk = chunk[
                    chunk['DRUG'].str.contains(drug_name, case=False, na=False)
                ].copy()
                
                if len(drug_chunk) > 0:
                    drug_chunk['vasopressor_type'] = drug_name
                    drug_chunk['vasopressor_weight'] = drug_info['weight']
                    drug_chunk['vasopressor_category'] = drug_info['category']
                    vasopressor_records.append(drug_chunk)
        
        if vasopressor_records:
            self._vasopressor_data = pd.concat(vasopressor_records, ignore_index=True)
            logger.info(f"✅ 加载了 {len(self._vasopressor_data)} 条血管活性药物记录")
        else:
            logger.warning("❌ 未找到血管活性药物数据")
    
    def _load_urine_data(self):
        """加载尿量数据"""
        logger.info("🚰 加载尿量数据...")
        
        output_file = self.data_dir / 'time series variables(output)_mimic3.csv'
        if not output_file.exists():
            logger.warning("输出数据文件不存在")
            return
        
        # 分块读取尿量数据
        urine_records = []
        chunk_size = 50000
        
        for chunk in pd.read_csv(output_file, chunksize=chunk_size):
            urine_chunk = chunk[chunk['label'] == 'Foley'].copy()
            if len(urine_chunk) > 0:
                urine_records.append(urine_chunk)
        
        if urine_records:
            self._urine_data = pd.concat(urine_records, ignore_index=True)
            logger.info(f"✅ 加载了 {len(self._urine_data)} 条尿量记录")
        else:
            logger.warning("❌ 未找到尿量数据")
    
    def _load_lab_data(self):
        """加载实验室数据"""
        logger.info("🔬 加载实验室数据...")
        
        vitals_file = self.data_dir / 'time series variables(vital signs)_mimic3.csv'
        if not vitals_file.exists():
            logger.warning("生命体征数据文件不存在")
            return
        
        # 提取实验室相关指标
        lab_records = []
        chunk_size = 100000
        lab_labels = ['Glucose (serum)', 'PH (Arterial)', 'Lactate']
        
        for chunk in pd.read_csv(vitals_file, chunksize=chunk_size):
            lab_chunk = chunk[chunk['label'].isin(lab_labels)].copy()
            if len(lab_chunk) > 0:
                lab_records.append(lab_chunk)
        
        if lab_records:
            self._lab_data = pd.concat(lab_records, ignore_index=True)
            logger.info(f"✅ 加载了 {len(self._lab_data)} 条实验室记录")
        else:
            logger.warning("❌ 未找到实验室数据")
    
    def calculate_enhanced_respiratory_sofa(self, data: Dict) -> int:
        """计算增强呼吸系统SOFA评分"""
        
        # 获取SpO2和FiO2
        spo2 = data.get('o2_saturation_pulseoxymetry', 100)
        fio2 = data.get('inspired_o2_fraction', 21)  # 默认21%
        
        # 确保FiO2在正确范围内
        if fio2 <= 1.0:
            fio2 = fio2 * 100  # 转换为百分比
        
        # 计算SpO2/FiO2比值（替代PaO2/FiO2）
        if fio2 > 0:
            spo2_fio2_ratio = spo2 / (fio2 / 100)
        else:
            return 0
        
        # SOFA呼吸系统评分
        if spo2_fio2_ratio >= 400:
            return 0
        elif spo2_fio2_ratio >= 300:
            return 1
        elif spo2_fio2_ratio >= 200:
            return 2
        elif spo2_fio2_ratio >= 100:
            return 3
        else:
            return 4
    
    def calculate_enhanced_cardiovascular_sofa(self, data: Dict, patient_id: int, timestamp: str) -> int:
        """计算增强心血管系统SOFA评分（包含血管活性药物）"""
        
        # 计算平均动脉压
        systolic = data.get('arterial_blood_pressure_systolic', 120)
        diastolic = data.get('arterial_blood_pressure_diastolic', 80)
        mean_arterial_pressure = (systolic + 2 * diastolic) / 3
        
        # 基础MAP评分
        if mean_arterial_pressure >= 70:
            base_score = 0
        else:
            base_score = 1
        
        # 检查血管活性药物使用
        vasopressor_score = self._get_vasopressor_score(patient_id, timestamp)
        
        # 综合评分
        total_score = max(base_score, vasopressor_score)
        
        return min(total_score, 4)  # 最大4分
    
    def _get_vasopressor_score(self, patient_id: int, timestamp: str) -> int:
        """获取血管活性药物评分"""
        
        if self._vasopressor_data is None:
            return 0
        
        # 查找该患者在指定时间窗口内的血管活性药物使用
        patient_vasopressors = self._vasopressor_data[
            self._vasopressor_data['SUBJECT_ID'] == patient_id
        ]
        
        if len(patient_vasopressors) == 0:
            return 0
        
        # 计算血管活性药物权重
        total_weight = 0
        unique_drugs = set()
        
        for _, row in patient_vasopressors.iterrows():
            drug_type = row['vasopressor_type']
            weight = row['vasopressor_weight']
            category = row['vasopressor_category']
            
            if drug_type not in unique_drugs:
                unique_drugs.add(drug_type)
                total_weight += weight
        
        # 根据药物使用情况评分
        if total_weight == 0:
            return 0
        elif total_weight <= 0.6:
            return 1  # 低剂量
        elif total_weight <= 1.0:
            return 2  # 中等剂量
        elif total_weight <= 2.0:
            return 3  # 高剂量
        else:
            return 4  # 极高剂量或多种药物
    
    def calculate_enhanced_renal_sofa(self, data: Dict, patient_id: int, timestamp: str) -> int:
        """计算增强肾脏系统SOFA评分（包含尿量）"""
        
        # 肌酐评分
        creatinine = data.get('creatinine', 1.0)
        creatinine_score = self._get_creatinine_score(creatinine)
        
        # 尿量评分
        urine_score = self._get_urine_output_score(patient_id, timestamp)
        
        # 取较高分数
        return max(creatinine_score, urine_score)
    
    def _get_creatinine_score(self, creatinine: float) -> int:
        """根据肌酐值计算评分"""
        if creatinine < 1.2:
            return 0
        elif creatinine < 2.0:
            return 1
        elif creatinine < 3.5:
            return 2
        elif creatinine < 5.0:
            return 3
        else:
            return 4
    
    def _get_urine_output_score(self, patient_id: int, timestamp: str) -> int:
        """根据尿量计算评分"""
        
        if self._urine_data is None:
            return 0
        
        # 查找该患者的尿量数据
        patient_urine = self._urine_data[
            self._urine_data['subject_id'] == patient_id
        ]
        
        if len(patient_urine) == 0:
            return 0
        
        # 计算24小时尿量（简化处理）
        try:
            total_urine_24h = patient_urine['valuenum'].sum()
            
            if total_urine_24h < 200:
                return 4
            elif total_urine_24h < 500:
                return 3
            else:
                return 0
        except:
            return 0
    
    def calculate_complete_sofa_scores(self, observations: np.ndarray, patient_ids: np.ndarray, 
                                     timestamps: np.ndarray) -> Dict[str, np.ndarray]:
        """计算完整的SOFA评分"""
        
        logger.info("🧮 计算增强SOFA评分...")
        
        # 确保数据已加载
        if not self.load_enhanced_data_sources():
            logger.warning("⚠️ 使用基础SOFA计算（无增强数据）")
        
        n_samples = len(observations)
        
        # 初始化评分数组
        scores = {
            'respiratory': np.zeros(n_samples),
            'coagulation': np.zeros(n_samples),  # 暂时为0（缺少血小板数据）
            'liver': np.zeros(n_samples),        # 暂时为0（缺少胆红素数据）
            'cardiovascular': np.zeros(n_samples),
            'cns': np.zeros(n_samples),
            'renal': np.zeros(n_samples),
            'total': np.zeros(n_samples)
        }
        
        # 特征名称映射
        feature_names = [
            'arterial_blood_pressure_diastolic', 'arterial_blood_pressure_systolic',
            'creatinine', 'foley', 'gcs_-_eye_opening', 'gcs_-_motor_response',
            'gcs_-_verbal_response', 'glucose_(serum)', 'heart_rate',
            'inspired_o2_fraction', 'lactate', 'o2_saturation_pulseoxymetry',
            'ph_(arterial)', 'respiratory_rate', 'temperature', 'gender', 'age',
            'religion', 'language', 'marital_status', 'ethnicity', 'weight_kg', 'height_cm'
        ]
        
        for i in range(n_samples):
            # 构建数据字典
            data_dict = {}
            if len(observations[i]) >= len(feature_names):
                for j, feature in enumerate(feature_names):
                    data_dict[feature] = observations[i][j]
            
            patient_id = patient_ids[i] if len(patient_ids) > i else 0
            timestamp = timestamps[i] if len(timestamps) > i else ""
            
            # 计算各系统SOFA评分
            scores['respiratory'][i] = self.calculate_enhanced_respiratory_sofa(data_dict)
            scores['cardiovascular'][i] = self.calculate_enhanced_cardiovascular_sofa(
                data_dict, patient_id, timestamp)
            scores['cns'][i] = self._calculate_cns_sofa(data_dict)
            scores['renal'][i] = self.calculate_enhanced_renal_sofa(
                data_dict, patient_id, timestamp)
            
            # 计算总分
            scores['total'][i] = (scores['respiratory'][i] + 
                                scores['coagulation'][i] + 
                                scores['liver'][i] + 
                                scores['cardiovascular'][i] + 
                                scores['cns'][i] + 
                                scores['renal'][i])
        
        logger.info(f"✅ 完成 {n_samples} 个样本的增强SOFA评分计算")
        
        return scores
    
    def _calculate_cns_sofa(self, data: Dict) -> int:
        """计算中枢神经系统SOFA评分"""
        
        gcs_eye = data.get('gcs_-_eye_opening', 4)
        gcs_verbal = data.get('gcs_-_verbal_response', 5) 
        gcs_motor = data.get('gcs_-_motor_response', 6)
        
        gcs_total = gcs_eye + gcs_verbal + gcs_motor
        
        if gcs_total >= 15:
            return 0
        elif gcs_total >= 13:
            return 1
        elif gcs_total >= 10:
            return 2
        elif gcs_total >= 6:
            return 3
        else:
            return 4
    
    def calculate_sofa_rewards(self, observations: np.ndarray, patient_ids: np.ndarray = None,
                             timestamps: np.ndarray = None, terminal_reward: float = 15, 
                             intermediate_reward: float = 1) -> np.ndarray:
        """计算基于增强SOFA评分的奖励"""
        
        logger.info("🎯 计算增强SOFA奖励...")
        
        if patient_ids is None:
            patient_ids = np.arange(len(observations))
        if timestamps is None:
            timestamps = np.arange(len(observations))
        
        # 计算SOFA评分
        sofa_scores = self.calculate_complete_sofa_scores(observations, patient_ids, timestamps)
        total_scores = sofa_scores['total']
        
        # 计算奖励
        rewards = np.zeros(len(observations))
        
        for i in range(len(observations)):
            current_score = total_scores[i]
            
            # 中间奖励：基于SOFA评分变化
            if i > 0:
                prev_score = total_scores[i-1]
                score_change = prev_score - current_score  # 分数降低为正奖励
                rewards[i] = score_change * intermediate_reward
            
            # 终端奖励：基于最终结果
            if i == len(observations) - 1:  # 最后一个时间步
                if current_score <= 6:  # 低风险
                    rewards[i] += terminal_reward
                elif current_score >= 15:  # 高风险
                    rewards[i] -= terminal_reward
        
        logger.info(f"✅ 计算完成，奖励范围: [{rewards.min():.3f}, {rewards.max():.3f}]")
        
        return rewards

def main():
    """测试增强SOFA计算器"""
    
    print("🧮 测试增强SOFA计算器...")
    
    calculator = EnhancedSOFACalculator()
    
    # 创建测试数据
    n_samples = 1000
    n_features = 23
    test_observations = np.random.randn(n_samples, n_features)
    test_patient_ids = np.random.randint(1000, 2000, n_samples)
    test_timestamps = np.arange(n_samples)
    
    # 计算SOFA评分
    scores = calculator.calculate_complete_sofa_scores(
        test_observations, test_patient_ids, test_timestamps)
    
    # 计算奖励
    rewards = calculator.calculate_sofa_rewards(
        test_observations, test_patient_ids, test_timestamps)
    
    print(f"\n📊 结果统计:")
    print(f"总SOFA评分: 平均 {scores['total'].mean():.2f}, 标准差 {scores['total'].std():.2f}")
    print(f"奖励: 平均 {rewards.mean():.3f}, 范围 [{rewards.min():.3f}, {rewards.max():.3f}]")
    
    # 各系统评分统计
    print(f"\n🏥 各系统SOFA评分:")
    for system in ['respiratory', 'cardiovascular', 'cns', 'renal']:
        mean_score = scores[system].mean()
        print(f"  {system}: {mean_score:.2f}")

if __name__ == "__main__":
    main() 