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
        
        # 保存验证结果
        validation_report_path = self.base_dir / "data_validation_report.json"
        with open(validation_report_path, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2, ensure_ascii=False)
        
        self.log(f"📄 验证报告已保存: {validation_report_path}")
        
        # 检查是否有关键文件缺失
        critical_files = ["train_features.npy", "enhanced_sofa_dataset.csv"]
        missing_critical = [f for f in critical_files if not validation_results.get(f, {}).get('exists', False)]
        
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
        report_content.append("4. 特征工程和序列构建")
        report_content.append("5. 训练/验证/测试数据分割")
        report_content.append("6. 数据验证和质量检查")
        
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
        
        # 4. 验证输出数据
        if not self.validate_output_data():
            self.log("❌ 数据验证失败，但流水线继续")
        
        # 5. 生成总结报告
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