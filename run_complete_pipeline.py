#!/usr/bin/env python3
"""
完整的败血症数据处理流水线
整合所有数据处理步骤，从原始MIMIC-III数据到最终训练数据
"""

import os
import sys
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime

class SepsisDataPipeline:
    """败血症数据处理流水线"""
    
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = Path(__file__).parent
        else:
            self.base_dir = Path(base_dir)
        
        self.raw_data_dir = self.base_dir / "raw_data" / "mimic3_original"
        self.processed_data_dir = self.base_dir / "processed_data"
        self.scripts_dir = self.base_dir / "preprocessing_scripts"
        
        self.log_file = self.base_dir / f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
    def log(self, message):
        """记录日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def check_prerequisites(self):
        """检查先决条件"""
        self.log("🔍 检查先决条件...")
        
        # 检查目录结构
        required_dirs = [
            self.raw_data_dir,
            self.processed_data_dir,
            self.scripts_dir
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                self.log(f"❌ 缺少目录: {dir_path}")
                return False
            else:
                self.log(f"✅ 目录存在: {dir_path}")
        
        # 检查原始数据文件
        required_files = [
            "time series variables(vital signs)_mimic3.csv",
            "static variables(demographics)_mimic3.csv",
            "static variables(weight_height)_mimic3.csv",
            "top 1000 medications_mimic3.csv",
            "time series variables(output)_mimic3.csv"
        ]
        
        for file_name in required_files:
            file_path = self.raw_data_dir / file_name
            if not file_path.exists():
                self.log(f"❌ 缺少原始数据文件: {file_name}")
                return False
            else:
                file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                self.log(f"✅ 数据文件存在: {file_name} ({file_size:.1f} MB)")
        
        # 检查处理脚本
        required_scripts = [
            "sepsis_data_preprocessing.py",
            "enhanced_sofa_calculator.py"
        ]
        
        for script_name in required_scripts:
            script_path = self.scripts_dir / script_name
            if not script_path.exists():
                self.log(f"❌ 缺少处理脚本: {script_name}")
                return False
            else:
                self.log(f"✅ 脚本存在: {script_name}")
        
        self.log("✅ 所有先决条件检查通过")
        return True
    
    def run_basic_preprocessing(self):
        """运行基础数据预处理"""
        self.log("🔄 开始基础数据预处理...")
        
        try:
            script_path = self.scripts_dir / "sepsis_data_preprocessing.py"
            
            # 运行预处理脚本
            result = subprocess.run([
                sys.executable, str(script_path)
            ], capture_output=True, text=True, cwd=str(self.base_dir.parent))
            
            if result.returncode == 0:
                self.log("✅ 基础数据预处理完成")
                self.log(f"输出: {result.stdout}")
                return True
            else:
                self.log(f"❌ 基础数据预处理失败: {result.stderr}")
                return False
                
        except Exception as e:
            self.log(f"❌ 基础数据预处理异常: {e}")
            return False
    
    def run_sofa_calculation(self):
        """运行SOFA评分计算"""
        self.log("🔄 开始SOFA评分计算...")
        
        try:
            script_path = self.scripts_dir / "enhanced_sofa_calculator.py"
            
            # 运行SOFA计算脚本
            result = subprocess.run([
                sys.executable, str(script_path)
            ], capture_output=True, text=True, cwd=str(self.base_dir.parent))
            
            if result.returncode == 0:
                self.log("✅ SOFA评分计算完成")
                self.log(f"输出: {result.stdout}")
                return True
            else:
                self.log(f"❌ SOFA评分计算失败: {result.stderr}")
                return False
                
        except Exception as e:
            self.log(f"❌ SOFA评分计算异常: {e}")
            return False
    
    def generate_expert_data(self):
        """生成专家决策数据用于监督学习"""
        self.log("🔄 生成专家决策数据...")
        
        try:
            # 检查是否已有训练数据
            final_data_dir = self.processed_data_dir / "final"
            train_features_path = final_data_dir / "train_features.npy"
            
            if not train_features_path.exists():
                self.log("❌ 训练数据不存在，无法生成专家数据")
                return False
            
            # 加载训练数据
            train_features = np.load(train_features_path)
            self.log(f"✅ 加载训练数据: {train_features.shape}")
            
            # 生成专家决策数据
            expert_data = []
            for i, sequence in enumerate(train_features):
                # 重塑序列数据
                if len(sequence.shape) == 1:
                    # 假设序列长度为7
                    sequence = sequence.reshape(7, -1)
                
                # 获取最后一个时间步的状态
                final_state = sequence[-1]
                
                # 基于医学规则生成专家决策
                expert_action = self.generate_expert_action(final_state)
                
                # 组合状态和动作
                expert_sample = {
                    'state_features': final_state.tolist(),
                    'expert_action': expert_action,
                    'sequence_id': i
                }
                expert_data.append(expert_sample)
            
            # 保存专家数据
            expert_data_dir = self.processed_data_dir / "expert_data"
            expert_data_dir.mkdir(exist_ok=True)
            
            expert_df = pd.DataFrame(expert_data)
            expert_data_path = expert_data_dir / "expert_decisions.csv"
            expert_df.to_csv(expert_data_path, index=False)
            
            self.log(f"✅ 专家数据已保存: {expert_data_path}")
            self.log(f"   专家样本数量: {len(expert_data)}")
            
            return True
            
        except Exception as e:
            self.log(f"❌ 生成专家数据异常: {e}")
            return False
    
    def generate_expert_action(self, state_features):
        """基于医学规则生成专家动作"""
        # 假设状态特征的顺序：[temp, hr, resp_rate, bp_sys, bp_dias, spo2, ...]
        if len(state_features) < 6:
            return 0  # 默认观察动作
        
        temp = state_features[0]      # 体温
        hr = state_features[1]        # 心率
        bp_sys = state_features[3]    # 收缩压
        bp_dias = state_features[4]   # 舒张压
        spo2 = state_features[5]      # 氧饱和度
        
        # 专家决策规则
        if temp > 38.5:
            return 1  # 降温治疗
        elif hr > 100:
            return 2  # 心率控制
        elif bp_sys < 90:
            return 3  # 升压治疗
        elif bp_sys > 140:
            return 4  # 降压治疗
        elif spo2 < 92:
            return 5  # 氧疗
        else:
            return 0  # 观察/维持治疗

    def validate_output_data(self):
        """验证输出数据"""
        self.log("🔍 验证输出数据...")
        
        validation_results = {}
        
        # 检查最终训练数据
        final_data_dir = self.processed_data_dir / "final"
        if final_data_dir.exists():
            required_files = [
                "train_features.npy",
                "val_features.npy", 
                "train_metadata.csv",
                "val_metadata.csv",
                "feature_info.csv"
            ]
            
            for file_name in required_files:
                file_path = final_data_dir / file_name
                if file_path.exists():
                    if file_name.endswith('.npy'):
                        try:
                            data = np.load(file_path)
                            validation_results[file_name] = {
                                'exists': True,
                                'shape': data.shape,
                                'dtype': str(data.dtype)
                            }
                            self.log(f"✅ {file_name}: shape={data.shape}, dtype={data.dtype}")
                        except Exception as e:
                            validation_results[file_name] = {'exists': True, 'error': str(e)}
                            self.log(f"❌ {file_name}: 读取错误 - {e}")
                    elif file_name.endswith('.csv'):
                        try:
                            data = pd.read_csv(file_path)
                            validation_results[file_name] = {
                                'exists': True,
                                'shape': data.shape,
                                'columns': list(data.columns)
                            }
                            self.log(f"✅ {file_name}: shape={data.shape}")
                        except Exception as e:
                            validation_results[file_name] = {'exists': True, 'error': str(e)}
                            self.log(f"❌ {file_name}: 读取错误 - {e}")
                else:
                    validation_results[file_name] = {'exists': False}
                    self.log(f"❌ 缺少文件: {file_name}")
        
        # 检查SOFA数据
        sofa_data_dir = self.processed_data_dir / "complete_sofa_data"
        if sofa_data_dir.exists():
            sofa_dataset_path = sofa_data_dir / "enhanced_sofa_dataset.csv"
            if sofa_dataset_path.exists():
                try:
                    sofa_data = pd.read_csv(sofa_dataset_path)
                    validation_results['enhanced_sofa_dataset.csv'] = {
                        'exists': True,
                        'shape': sofa_data.shape,
                        'columns': list(sofa_data.columns)
                    }
                    self.log(f"✅ SOFA数据集: shape={sofa_data.shape}")
                    self.log(f"   特征: {list(sofa_data.columns)}")
                except Exception as e:
                    validation_results['enhanced_sofa_dataset.csv'] = {'exists': True, 'error': str(e)}
                    self.log(f"❌ SOFA数据集读取错误: {e}")
            else:
                validation_results['enhanced_sofa_dataset.csv'] = {'exists': False}
                self.log("❌ 缺少SOFA数据集文件")
        
        # 检查专家数据
        expert_data_dir = self.processed_data_dir / "expert_data"
        if expert_data_dir.exists():
            expert_decisions_path = expert_data_dir / "expert_decisions.csv"
            if expert_decisions_path.exists():
                try:
                    expert_data = pd.read_csv(expert_decisions_path)
                    validation_results['expert_decisions.csv'] = {
                        'exists': True,
                        'shape': expert_data.shape,
                        'columns': list(expert_data.columns),
                        'action_distribution': expert_data['expert_action'].value_counts().to_dict()
                    }
                    self.log(f"✅ 专家决策数据: shape={expert_data.shape}")
                    self.log(f"   动作分布: {expert_data['expert_action'].value_counts().to_dict()}")
                except Exception as e:
                    validation_results['expert_decisions.csv'] = {'exists': True, 'error': str(e)}
                    self.log(f"❌ 专家数据读取错误: {e}")
            else:
                validation_results['expert_decisions.csv'] = {'exists': False}
                self.log("❌ 缺少专家决策数据文件")
        
        # 保存验证结果
        validation_report_path = self.base_dir / "data_validation_report.json"
        with open(validation_report_path, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2, ensure_ascii=False)
        
        self.log(f"📄 验证报告已保存: {validation_report_path}")
        
        # 检查是否有关键文件缺失
        critical_files = ["train_features.npy", "enhanced_sofa_dataset.csv"]
        missing_critical = [f for f in critical_files if not validation_results.get(f, {}).get('exists', False)]
        
        # 专家数据是可选的，但会发出警告
        if not validation_results.get('expert_decisions.csv', {}).get('exists', False):
            self.log("⚠️ 缺少专家决策数据，混合RL+SL训练将使用模拟专家数据")
        
        if missing_critical:
            self.log(f"❌ 缺少关键文件: {missing_critical}")
            return False
        else:
            self.log("✅ 所有关键文件验证通过")
            return True
    
    def generate_summary_report(self):
        """生成总结报告"""
        self.log("📊 生成总结报告...")
        
        report_content = []
        report_content.append("# 败血症数据处理流水线总结报告")
        report_content.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("")
        
        # 数据文件统计
        report_content.append("## 数据文件统计")
        
        # 原始数据
        report_content.append("### 原始数据")
        if self.raw_data_dir.exists():
            for file_path in self.raw_data_dir.glob("*.csv"):
                size_mb = file_path.stat().st_size / (1024 * 1024)
                report_content.append(f"- {file_path.name}: {size_mb:.1f} MB")
        
        # 处理后数据
        report_content.append("### 处理后数据")
        
        # 最终训练数据
        final_data_dir = self.processed_data_dir / "final"
        if final_data_dir.exists():
            report_content.append("#### 最终训练数据")
            for file_path in final_data_dir.glob("*"):
                if file_path.is_file():
                    if file_path.suffix == '.npy':
                        try:
                            data = np.load(file_path)
                            report_content.append(f"- {file_path.name}: shape={data.shape}")
                        except:
                            report_content.append(f"- {file_path.name}: 读取失败")
                    elif file_path.suffix == '.csv':
                        try:
                            data = pd.read_csv(file_path)
                            report_content.append(f"- {file_path.name}: {data.shape[0]} 行, {data.shape[1]} 列")
                        except:
                            report_content.append(f"- {file_path.name}: 读取失败")
        
        # SOFA数据
        sofa_data_dir = self.processed_data_dir / "complete_sofa_data"
        if sofa_data_dir.exists():
            report_content.append("#### SOFA评分数据")
            sofa_dataset_path = sofa_data_dir / "enhanced_sofa_dataset.csv"
            if sofa_dataset_path.exists():
                try:
                    sofa_data = pd.read_csv(sofa_dataset_path)
                    report_content.append(f"- enhanced_sofa_dataset.csv: {sofa_data.shape[0]} 行, {sofa_data.shape[1]} 列")
                    
                    # SOFA评分统计
                    if 'sofa_score' in sofa_data.columns:
                        sofa_stats = sofa_data['sofa_score'].describe()
                        report_content.append(f"  - SOFA评分范围: {sofa_stats['min']:.1f} - {sofa_stats['max']:.1f}")
                        report_content.append(f"  - SOFA评分均值: {sofa_stats['mean']:.1f}")
                except:
                    report_content.append("- enhanced_sofa_dataset.csv: 读取失败")
        
        # 专家决策数据
        expert_data_dir = self.processed_data_dir / "expert_data"
        if expert_data_dir.exists():
            report_content.append("#### 专家决策数据")
            expert_decisions_path = expert_data_dir / "expert_decisions.csv"
            if expert_decisions_path.exists():
                try:
                    expert_data = pd.read_csv(expert_decisions_path)
                    report_content.append(f"- expert_decisions.csv: {expert_data.shape[0]} 行, {expert_data.shape[1]} 列")
                    
                    # 专家动作分布统计
                    if 'expert_action' in expert_data.columns:
                        action_counts = expert_data['expert_action'].value_counts()
                        report_content.append("  - 动作分布:")
                        for action, count in action_counts.items():
                            percentage = (count / len(expert_data)) * 100
                            report_content.append(f"    - 动作{action}: {count} ({percentage:.1f}%)")
                except:
                    report_content.append("- expert_decisions.csv: 读取失败")
        
        # 处理脚本
        report_content.append("## 使用的处理脚本")
        if self.scripts_dir.exists():
            for script_path in self.scripts_dir.glob("*.py"):
                report_content.append(f"- {script_path.name}")
        
        # 数据流程
        report_content.append("## 数据处理流程")
        report_content.append("1. 原始MIMIC-III数据导入")
        report_content.append("2. 基础数据预处理和清洗")
        report_content.append("3. SOFA评分计算")
        report_content.append("4. 专家决策数据生成 (用于混合RL+SL训练)")
        report_content.append("5. 特征工程和序列构建")
        report_content.append("6. 训练/验证/测试数据分割")
        report_content.append("7. 数据验证和质量检查")
        
        # 保存报告
        report_path = self.base_dir / "pipeline_summary_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        self.log(f"📄 总结报告已保存: {report_path}")
    
    def run_complete_pipeline(self):
        """运行完整流水线"""
        self.log("🚀 开始运行完整败血症数据处理流水线")
        
        # 1. 检查先决条件
        if not self.check_prerequisites():
            self.log("❌ 先决条件检查失败，流水线停止")
            return False
        
        # 2. 运行基础预处理
        if not self.run_basic_preprocessing():
            self.log("❌ 基础预处理失败，流水线停止")
            return False
        
        # 3. 运行SOFA计算
        if not self.run_sofa_calculation():
            self.log("❌ SOFA计算失败，流水线停止")
            return False
        
        # 4. 生成专家决策数据
        if not self.generate_expert_data():
            self.log("⚠️ 专家数据生成失败，但流水线继续")
        
        # 5. 验证输出数据
        if not self.validate_output_data():
            self.log("❌ 数据验证失败，但流水线继续")
        
        # 6. 生成总结报告
        self.generate_summary_report()
        
        self.log("🎉 完整数据处理流水线执行完成!")
        self.log(f"📄 详细日志: {self.log_file}")
        
        return True

def main():
    """主函数"""
    # 获取当前脚本所在目录
    current_dir = Path(__file__).parent
    
    # 创建流水线实例
    pipeline = SepsisDataPipeline(current_dir)
    
    # 运行完整流水线
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n✅ 数据处理流水线成功完成!")
        print(f"📁 数据目录: {pipeline.processed_data_dir}")
        print(f"📄 日志文件: {pipeline.log_file}")
    else:
        print("\n❌ 数据处理流水线执行失败!")
        print(f"📄 查看详细日志: {pipeline.log_file}")
    
    return success

if __name__ == "__main__":
    main()