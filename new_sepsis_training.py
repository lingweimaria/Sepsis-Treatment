#!/usr/bin/env python3

import os
# 解决macOS上的OpenMP冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import warnings
import json
import math # 新增：用于正弦位置编码

warnings.filterwarnings('ignore')

class SafetyLayer(nn.Module):
    """增强的医学安全层 - 统一处理安全约束"""
    
    def __init__(self, num_actions=20):
        super().__init__()
        self.num_actions = num_actions
        
        # 基于真实临床指南的危险组合
        self.dangerous_combinations = [
            [1, 2],   # 血管加压药 + 降压药 (相互抵消)
            [3, 11],  # 利尿剂 + 大量电解质 (电解质紊乱)
            [17, 18], # 镇痛剂 + 镇静剂 (过度镇静)
        ]
        
        # 需要谨慎使用的高风险动作
        self.high_risk_actions = [1, 17, 18, 19]  # 血管加压药、镇痛、镇静、抗凝
        
    def apply_constraints(self, logits, latest_state):
        """应用医学安全约束"""
        constrained_logits = logits.clone()
        action_probs = torch.softmax(logits, dim=-1)
        
        # 约束1：防止危险药物组合
        for dangerous_combo in self.dangerous_combinations:
            combo_prob = torch.prod(action_probs[:, dangerous_combo], dim=1)
            high_risk_mask = combo_prob > 0.05
            if high_risk_mask.any():
                for action_idx in dangerous_combo:
                    constrained_logits[high_risk_mask, action_idx] -= 1.5
        
        # 约束2：基于生理状态的智能约束
        if latest_state is not None:
            # 低血压约束 - 禁止降压药，鼓励血管加压药
            bp_systolic = latest_state[:, 3]
            low_bp_mask = bp_systolic < 85
            if low_bp_mask.any():
                constrained_logits[low_bp_mask, 2] -= 5.0  # 禁止降压药
                constrained_logits[low_bp_mask, 1] += 0.5  # 轻微鼓励血管加压药
            
            # 高血压约束
            high_bp_mask = bp_systolic > 160
            if high_bp_mask.any():
                constrained_logits[high_bp_mask, 1] -= 3.0  # 限制血管加压药
            
            # 心动过速约束
            heart_rate = latest_state[:, 1]
            high_hr_mask = heart_rate > 130
            if high_hr_mask.any():
                constrained_logits[high_hr_mask, 4] += 0.3  # 鼓励β受体阻滞剂
            
            # 低氧约束
            spo2 = latest_state[:, 5]
            low_o2_mask = spo2 < 90
            if low_o2_mask.any():
                constrained_logits[low_o2_mask, 5] += 0.5  # 鼓励氧疗
        
        return constrained_logits

# ✨ 新增：固定的正弦/余弦位置编码模块
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return x

class EnhancedSepsisLSTMAgent(nn.Module):
    """增强的败血症LSTM智能体 - 包含医学约束和监督学习模仿"""
    
    def __init__(self, input_dim, hidden_dim=128, num_actions=20, sequence_length=7, expert_data_path=None):
        super(EnhancedSepsisLSTMAgent, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.sequence_length = sequence_length
        
        # 多层LSTM用于更好的时序建模
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=False
        )
        
        # 注意力机制用于关注重要的时间步
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer normalization for stability
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # ✨ 修正：使用固定的正弦位置编码
        self.pos_encoder = SinusoidalPositionalEncoding(hidden_dim, sequence_length)
        
        # ✨ 改进的Actor网络 - 简化架构，更稳定
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, 128),  # temporal + medical
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        
        # ✨ 改进的Critic网络 - 简化架构，更稳定
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # ✨ 改进的医学特征提取器 - 输出维度与hidden_dim一致
        self.medical_feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # ✨ 专家模仿网络 - 简化架构
        self.expert_imitation_net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        
        # ✨ 使用softmax归一化的融合权重
        self.rl_weight = nn.Parameter(torch.tensor(1.0))
        self.sl_weight = nn.Parameter(torch.tensor(1.0))
        
        # ✨ 统一的安全层
        self.safety_layer = SafetyLayer(num_actions)
        
        # debug flag
        self._printed = False
        
        # 保守的权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data, gain=0.5)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data, gain=0.5)
                elif 'bias' in name:
                    param.data.fill_(0.01)
    
    def _normalize_weights(self):
        """✨ 使用softmax归一化权重，避免除零"""
        w = torch.softmax(torch.stack([self.rl_weight, self.sl_weight]), dim=0)
        return w[0], w[1]
    
    def forward(self, sequences, sequence_lengths=None, use_expert_imitation=True):
        """前向传播 - 增加专家模仿功能"""
        batch_size, T, _ = sequences.shape
        
        # LSTM处理时序数据
        lstm_output, (h_n, c_n) = self.lstm(sequences)
        
        # Layer normalization
        lstm_output = self.layer_norm1(lstm_output)
        
        # ✨ 修正：应用固定的位置编码
        lstm_output_with_pos = self.pos_encoder(lstm_output)
        
        # ✨ 注意力机制 - 显式提取权重
        attended_output, attention_weights = self.attention(
            lstm_output_with_pos, lstm_output_with_pos, lstm_output_with_pos,
            need_weights=True, average_attn_weights=False  # 保留 [B, heads, T, T]
        )
        
        # 结合LSTM和注意力输出 (修正：残差连接应使用添加了位置编码的输入)
        combined_output = lstm_output_with_pos + attended_output
        combined_output = self.layer_norm2(combined_output)
        
        # ✨ 取最后一个有效时间步
        if sequence_lengths is not None:
            # 将列表转换为张量
            seq_lengths_tensor = torch.tensor(sequence_lengths, device=combined_output.device)
            idx = (seq_lengths_tensor - 1).clamp(0, T-1)
            temporal_context = combined_output[torch.arange(batch_size), idx, :]
        else:
            temporal_context = combined_output[:, -1, :]
        
        # ✨ 医学特征提取（使用最新的医学数据）
        med_feat_raw = sequences[:, -1, :]  # 取最新观测
        medical_features = self.medical_feature_extractor(med_feat_raw)
        
        # ✨ 融合时序和医学特征
        fused_features = torch.cat([temporal_context, medical_features], dim=1)   # [B, 2H]
        
        # ✨ Actor输出
        rl_action_logits = self.actor(fused_features)
        expert_action_logits = self.expert_imitation_net(fused_features)
        
        # ✨ 使用改进的权重融合
        if use_expert_imitation:
            w_rl, w_sl = self._normalize_weights()
            fused_action_logits = w_rl * rl_action_logits + w_sl * expert_action_logits
        else:
            fused_action_logits = rl_action_logits
        
        # ✨ 使用统一的安全层
        constrained_action_logits = self.safety_layer.apply_constraints(fused_action_logits, med_feat_raw)
        
        # ✨ Critic输出
        state_values = self.critic(fused_features)
        
        # ✨ debug 打印一次形状
        if not self._printed:
            print("✨ Enhanced Model Debug Info:")
            print(f"   - attention_weights shape: {attention_weights.shape}")   # 期望 [B, heads, T, T]
            print(f"   - fused_features shape: {fused_features.shape}")
            self._printed = True
        
        # ✨ 返回更有意义的输出，包括融合后的logits
        return {
            'constrained_logits': constrained_action_logits,
            'fused_logits': fused_action_logits,  # 新增：融合后但未约束的logits
            'state_values': state_values,
            'temporal_features': combined_output,
            'attention_weights': attention_weights,
            'rl_logits': rl_action_logits,
            'expert_logits': expert_action_logits,
            'temporal_context': temporal_context  # 新增：注意力加权的上下文
        }

class ClinicalRewardCalculator:
    """临床相关的奖励计算器 - 基于医学指标的奖励"""
    
    def __init__(self):
        # 重新设计的临床指标权重 - 更加注重SOFA评分和生命体征改善
        self.vital_signs_weight = 0.4  # 增加生命体征权重
        self.sofa_weight = 0.5         # 增加SOFA评分权重
        self.stability_weight = 0.15   # 适当降低稳定性权重
        self.safety_weight = 0.05      # 降低安全权重，避免过度保守
        
        # 正常值范围 (基于临床标准) - 扩大一些范围以增加敏感度
        self.normal_ranges = {
            'temperature': (35.5, 38.0),   # 稍微扩大温度范围
            'heart_rate': (50, 110),       # 扩大心率范围
            'respiratory_rate': (10, 24),  # 扩大呼吸频率范围
            'arterial_blood_pressure_systolic': (85, 150),
            'arterial_blood_pressure_diastolic': (55, 95),
            'o2_saturation_pulseoxymetry': (92, 100),  # 稍微降低氧饱和度下限
            'glucose_(serum)': (60, 200),  # 扩大血糖范围
            'gcs_-_eye_opening': (3, 4),
            'gcs_-_motor_response': (4, 6),  # 扩大运动反应范围
            'gcs_-_verbal_response': (3, 5)  # 扩大言语反应范围
        }
        
    def calculate_vital_signs_reward(self, features):
        """基于生命体征的奖励计算 - 改进版"""
        batch_size = features.size(0)
        rewards = torch.zeros(batch_size)
        
        # 特征名称映射到索引
        feature_names = ['temperature', 'heart_rate', 'respiratory_rate', 
                        'arterial_blood_pressure_systolic', 'arterial_blood_pressure_diastolic',
                        'o2_saturation_pulseoxymetry', 'glucose_(serum)', 'gcs_-_eye_opening',
                        'gcs_-_motor_response', 'gcs_-_verbal_response', 'age', 'sofa_score']
        
        # 计算时序改善奖励 - 关注生命体征的改善趋势
        if features.size(1) >= 2:  # 至少有2个时间步
            current_values = features[:, -1, :10]  # 当前时间步的生命体征
            previous_values = features[:, -2, :10]  # 前一时间步的生命体征
            
            improvement_reward = 0.0
            for i, feature_name in enumerate(feature_names[:10]):
                if feature_name in self.normal_ranges:
                    normal_min, normal_max = self.normal_ranges[feature_name]
                    normal_center = (normal_min + normal_max) / 2
                    
                    # 计算当前和之前距离正常中心的距离
                    current_distance = torch.abs(current_values[:, i] - normal_center)
                    previous_distance = torch.abs(previous_values[:, i] - normal_center)
                    
                    # 改善奖励 - 距离正常值更近获得正奖励
                    improvement = previous_distance - current_distance
                    normalized_improvement = improvement / (normal_max - normal_min)
                    improvement_reward += normalized_improvement * 0.3  # 每个特征最多贡献0.3奖励
            
            rewards += improvement_reward
        
        # 当前状态的正常性奖励 - 使用更敏感的计算方式
        current_state_reward = 0.0
        for i, feature_name in enumerate(feature_names[:10]):
            if feature_name in self.normal_ranges:
                normal_min, normal_max = self.normal_ranges[feature_name]
                values = features[:, -1, i]  # 当前时间步的值
                
                # 使用分段线性函数而非指数函数，增加敏感度
                in_range_mask = (values >= normal_min) & (values <= normal_max)
                below_range_mask = values < normal_min
                above_range_mask = values > normal_max
                
                # 在正常范围内：满分
                feature_reward = torch.zeros_like(values)
                feature_reward[in_range_mask] = 1.0
                
                # 在范围外：线性递减，而非指数递减
                range_width = normal_max - normal_min
                feature_reward[below_range_mask] = torch.clamp(
                    1.0 - (normal_min - values[below_range_mask]) / range_width, 
                    min=0.0, max=1.0
                )
                feature_reward[above_range_mask] = torch.clamp(
                    1.0 - (values[above_range_mask] - normal_max) / range_width, 
                    min=0.0, max=1.0
                )
                
                current_state_reward += feature_reward * 0.15  # 每个特征最多贡献0.15奖励
        
        rewards += current_state_reward
        return rewards
    
    def calculate_sofa_reward(self, features):
        """基于SOFA评分的奖励 - 改进版，更注重改善趋势"""
        current_sofa = features[:, -1, -1]  # 当前SOFA评分
        
        # 基础SOFA奖励 - 使用更敏感的分段函数
        base_sofa_reward = torch.zeros_like(current_sofa)
        
        # 优秀状态 (SOFA 0-6): 高奖励
        excellent_mask = current_sofa <= 6
        base_sofa_reward[excellent_mask] = 2.0 - current_sofa[excellent_mask] / 6.0
        
        # 中等状态 (SOFA 7-12): 中等奖励
        moderate_mask = (current_sofa > 6) & (current_sofa <= 12)
        base_sofa_reward[moderate_mask] = 1.0 - (current_sofa[moderate_mask] - 6) / 12.0
        
        # 严重状态 (SOFA 13-18): 低奖励
        severe_mask = (current_sofa > 12) & (current_sofa <= 18)
        base_sofa_reward[severe_mask] = 0.5 - (current_sofa[severe_mask] - 12) / 12.0
        
        # 危重状态 (SOFA >18): 极低奖励
        critical_mask = current_sofa > 18
        base_sofa_reward[critical_mask] = 0.1
        
        # SOFA改善奖励 - 关注评分的改善
        improvement_reward = torch.zeros_like(current_sofa)
        if features.size(1) >= 2:  # 至少有2个时间步
            previous_sofa = features[:, -2, -1]
            sofa_improvement = previous_sofa - current_sofa  # 评分降低是好事
            
            # 改善奖励：每降低1分SOFA评分获得0.5奖励
            improvement_reward = torch.clamp(sofa_improvement * 0.5, min=-1.0, max=2.0)
        
        total_sofa_reward = base_sofa_reward + improvement_reward
        return torch.clamp(total_sofa_reward, min=0.0, max=3.0)
    
    def calculate_stability_reward(self, features):
        """基于生命体征稳定性的奖励"""
        if features.size(1) < 2:  # 序列长度小于2，无法计算稳定性
            return torch.zeros(features.size(0))
        
        # 计算生命体征的变化率
        vital_signs = features[:, :, :10]  # 前10个特征是生命体征
        changes = torch.abs(vital_signs[:, 1:] - vital_signs[:, :-1])
        
        # 稳定性奖励：变化越小，奖励越高
        stability = torch.exp(-changes.mean(dim=(1, 2)))
        return stability
    
    def calculate_safety_constraint_reward(self, action_logits):
        """基于安全约束的奖励"""
        action_probs = torch.softmax(action_logits, dim=-1)
        
        # 鼓励适度的动作选择（避免极端治疗）
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1)
        target_entropy = np.log(action_logits.size(-1)) * 0.6  # 目标熵
        
        # 熵接近目标值时奖励最高
        entropy_reward = torch.exp(-torch.abs(entropy - target_entropy))
        return entropy_reward
    
    def calculate_reward(self, state_features, action_logits, episode_step=0):
        """
        综合临床奖励计算 - **修正版**: 移除了不稳定的随机探索奖励
        """
        # 各项奖励计算
        vital_reward = self.calculate_vital_signs_reward(state_features)
        sofa_reward = self.calculate_sofa_reward(state_features)
        stability_reward = self.calculate_stability_reward(state_features)
        safety_reward = self.calculate_safety_constraint_reward(action_logits)
        
        # 加权组合 - 使用新的权重
        total_reward = (
            self.vital_signs_weight * vital_reward +
            self.sofa_weight * sofa_reward +
            self.stability_weight * stability_reward +
            self.safety_weight * safety_reward
        )
        
        # 改进的生存时间奖励 - 更具动态性
        survival_bonus = torch.full_like(total_reward, min(episode_step * 0.02, 0.3))
        
        # **修正**: 移除了随机探索奖励
        # exploration_bonus = torch.randn_like(total_reward) * 0.05
        
        final_reward = total_reward + survival_bonus
        
        # 扩大奖励范围，移除过于严格的限制
        return torch.clamp(final_reward, min=-1.0, max=5.0)

class ExpertDataLoader:
    """专家数据加载器 - 用于监督学习"""
    
    def __init__(self, expert_data_path=None):
        self.expert_data_path = expert_data_path
        self.expert_actions = None
        self.expert_states = None
        
    def load_expert_data(self):
        """
        加载专家决策数据 - ✨ 修正版: 增强了解析CSV中字符串列表的鲁棒性
        """
        if self.expert_data_path is None or not Path(self.expert_data_path).exists():
            print(f"⚠️ 专家数据文件未在 '{self.expert_data_path}' 找到。将生成模拟数据。")
            self.generate_simulated_expert_data()
            return

        print(f"✅ 找到专家数据文件: {self.expert_data_path}")
        try:
            expert_df = pd.read_csv(self.expert_data_path)
            
            self.expert_actions = expert_df['expert_action'].values.astype(np.int64)
            
            states_list = []
            # 遍历 'state_features' 列中的每一个字符串
            for state_str in expert_df['state_features']:
                try:
                    # 1. 将 'nan' 替换为 'null'，使其成为有效的JSON格式
                    # 2. 移除多余的引号，以防万一
                    cleaned_str = state_str.replace('nan', 'null').strip('"')
                    
                    # 3. 使用json加载字符串为Python列表
                    parsed_list = json.loads(cleaned_str)
                    
                    # 4. 将列表中的 'null' 替换为 0.0，并确保所有元素都是浮点数
                    state_row = [float(x) if x is not None else 0.0 for x in parsed_list]
                    states_list.append(state_row)
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"   - ⚠️ 解析行失败: '{state_str[:50]}...'. 错误: {e}. 使用零向量填充。")
                    # 如果某一行解析失败，用一个零向量填充以保持数据对齐
                    # 这里的维度需要与成功解析的行一致，我们先用一个占位符
                    states_list.append(None) 
            
            # 确定特征维度 (从第一个成功解析的行)
            feature_dim = 0
            for row in states_list:
                if row is not None:
                    feature_dim = len(row)
                    break
            
            if feature_dim == 0:
                raise ValueError("未能从专家数据中确定特征维度。")

            # 填充失败的行
            final_states_list = [row if row is not None else [0.0] * feature_dim for row in states_list]

            self.expert_states = np.array(final_states_list, dtype=np.float32)
            
            print(f"✅ 成功加载并解析专家数据: {len(self.expert_actions)} 个样本")
            print(f"   - 专家状态形状: {self.expert_states.shape}")
            print(f"   - 专家动作形状: {self.expert_actions.shape}")

        except Exception as e:
            print(f"❌ 加载专家数据时发生严重错误: {e}。将使用模拟数据。")
            import traceback
            traceback.print_exc()
            self.generate_simulated_expert_data()
    
    def generate_simulated_expert_data(self, num_samples=1000):
        """生成模拟专家决策数据"""
        # 基于医学规则生成模拟专家决策
        np.random.seed(42)
        
        # 模拟状态特征 (temperature, heart_rate, bp_sys, bp_dias, spo2, ...)
        states = []
        actions = []
        
        for _ in range(num_samples):
            # 生成患者状态
            temp = np.random.normal(37.0, 1.5)  # 体温
            hr = np.random.normal(80, 20)       # 心率
            bp_sys = np.random.normal(120, 25)  # 收缩压
            bp_dias = np.random.normal(80, 15)  # 舒张压
            spo2 = np.random.normal(96, 4)      # 氧饱和度
            
            # 其他特征
            other_features = np.random.normal(0, 1, 7)  # 其他12-5=7个特征
            state = np.array([temp, hr, 0, bp_sys, bp_dias, spo2] + list(other_features))
            
            # 基于状态的专家决策规则
            action = self.expert_decision_rule(temp, hr, bp_sys, bp_dias, spo2)
            
            states.append(state)
            actions.append(action)
        
        self.expert_states = np.array(states, dtype=np.float32)
        self.expert_actions = np.array(actions, dtype=np.int64)
        print(f"✅ 生成模拟专家数据: {len(self.expert_actions)} 个样本")
        print(f"   模拟状态形状: {self.expert_states.shape}, 数据类型: {self.expert_states.dtype}")
        print(f"   模拟动作形状: {self.expert_actions.shape}, 数据类型: {self.expert_actions.dtype}")
    
    def expert_decision_rule(self, temp, hr, bp_sys, bp_dias, spo2):
        """基于医学知识的专家决策规则"""
        # 发热 -> 降温药物
        if temp > 38.5:
            return 1  # 降温药物
        
        # 心动过速 -> β受体阻滞剂
        elif hr > 100:
            return 2  # β受体阻滞剂
        
        # 低血压 -> 血管加压药
        elif bp_sys < 90:
            return 3  # 血管加压药
        
        # 高血压 -> 降压药
        elif bp_sys > 140:
            return 4  # 降压药
        
        # 低氧 -> 氧疗
        elif spo2 < 92:
            return 5  # 氧疗
        
        # 正常情况 -> 观察
        else:
            return 0  # 观察/维持治疗
    
    def get_expert_batch(self, batch_size=32):
        """获取专家数据批次"""
        if self.expert_states is None:
            self.load_expert_data()
        
        indices = np.random.choice(len(self.expert_actions), batch_size, replace=True)
        
        # 确保返回的数据类型正确
        batch_states = self.expert_states[indices].astype(np.float32)
        batch_actions = self.expert_actions[indices].astype(np.int64)
        
        return batch_states, batch_actions

class EnhancedTrainer:
    """增强的训练器 - 包含医学安全约束和监督学习"""
    
    def __init__(self, agent, reward_calculator, lr=5e-5, device='cpu', expert_data_path=None):
        """初始化训练器"""
        self.agent = agent.to(device)
        self.reward_calculator = reward_calculator
        self.device = device
        # ✨ 使用模型内置的安全层
        # self.safety_constraints = self.agent.safety_layer
        
        # 专家数据加载器 - 优先尝试真实数据
        if expert_data_path is None:
            # 尝试加载真实专家数据
            real_expert_path = "data_pipeline/processed_data/expert_data/expert_decisions.csv"
            if Path(real_expert_path).exists():
                print(f"👩‍⚕️ 找到真实专家数据: {real_expert_path}")
                expert_data_path = real_expert_path
            else:
                print("⚠️ 未找到真实专家数据，将使用模拟数据")
        
        self.expert_loader = ExpertDataLoader(expert_data_path)
        self.expert_loader.load_expert_data()
        
        # 更稳定的优化器设置
        self.optimizer = optim.AdamW(
            self.agent.parameters(), 
            lr=lr, 
            weight_decay=1e-3,  # 增加正则化
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.8, patience=3
        )
        
        self.huber_loss = nn.SmoothL1Loss()  # 更稳定的损失函数
        self.global_step = 0
        
        # 增强的历史记录
        self.history = {
            'train_total_loss': [],
            'train_actor_loss': [],
            'train_critic_loss': [],
            'train_expert_loss': [],
            'train_reward': [],
            'val_loss': [],
            'val_reward': [],
            'learning_rate': [],
            'gradient_norm': [],
            'safety_violations': [],
            'rl_sl_weights': [],
            'val_clinical_metrics': [] # 新增：保存验证时的临床指标
        }
    
    def prepare_batch(self, sequences, batch_size=8):
        """准备训练批次"""
        np.random.shuffle(sequences)
        
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            
            # 堆叠特征
            features = torch.tensor([seq['features'] for seq in batch_sequences], 
                                   dtype=torch.float32, device=self.device)
            
            # 序列长度
            seq_lengths = [seq['sequence_length'] for seq in batch_sequences]
            
            # 检查并处理异常值
            features = torch.clamp(features, min=-10, max=10)
            features = torch.where(torch.isnan(features), torch.zeros_like(features), features)
            
            yield features, seq_lengths
    
    def train_epoch(self, train_sequences, epoch):
        """
        增强的训练epoch - **修正版**: 使用标准的Actor-Critic损失函数
        """
        self.agent.train()
        epoch_actor_losses = []
        epoch_critic_losses = []
        epoch_expert_losses = []
        epoch_rewards = []
        epoch_safety_violations = []
        epoch_gradient_norms = []
        
        for batch_idx, (features, seq_lengths) in enumerate(self.prepare_batch(train_sequences)):
            
            self.optimizer.zero_grad()
            
            # 前向传播 - 修复：处理新的字典返回格式
            outputs = self.agent(features, seq_lengths)
            action_logits = outputs['constrained_logits']
            state_values = outputs['state_values']
            expert_action_logits = outputs['expert_logits']
            
            # ✨ 计算奖励 - 使用已约束的logits
            rewards = self.reward_calculator.calculate_reward(
                features, action_logits, episode_step=self.global_step
            ).to(self.device)
            
            # --- Critic Loss 计算 ---
            state_values = state_values.squeeze()
            # 使用Huber损失以增加稳定性
            critic_loss = self.huber_loss(state_values, rewards)
            
            # --- Actor Loss 计算 ---
            # 计算Advantage
            advantages = (rewards - state_values).detach() # 使用.detach()来阻止梯度流向Critic
            
            # 标准化Advantage
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 计算动作分布和熵
            action_dist = torch.distributions.Categorical(logits=action_logits)
            log_probs = action_dist.log_prob(action_dist.sample())
            entropy_loss = -0.01 * action_dist.entropy().mean() # 熵正则化，鼓励探索
            
            # **修正**: 标准的Actor损失 (策略梯度损失)
            actor_loss = -(log_probs * advantages).mean()
            
            # --- 专家模仿损失 (Supervised Learning) ---
            expert_loss = self.calculate_expert_imitation_loss(
                features, expert_action_logits, batch_idx
            )
            
            # --- 安全约束损失 ---
            safety_violation = self.calculate_safety_violations(torch.softmax(action_logits, dim=-1))
            safety_loss = 0.1 * safety_violation
            
            # --- 总损失 ---
            # **修正**: 平衡各项损失，移除不必要的clamp
            total_loss = critic_loss + actor_loss + entropy_loss + 0.5 * expert_loss + safety_loss
            
            # 检查异常损失值
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"异常损失检测 epoch {epoch}, batch {batch_idx}: "
                      f"total={total_loss.item():.6f}, actor={actor_loss.item():.6f}, "
                      f"critic={critic_loss.item():.6f}. 跳过...")
                continue
            
            # 反向传播
            total_loss.backward()
            
            # **修正**: 使用更合理的梯度裁剪值
            grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
            epoch_gradient_norms.append(grad_norm.item())
            
            # 更新参数
            self.optimizer.step()
            
            # 记录指标
            epoch_actor_losses.append(actor_loss.item())
            epoch_critic_losses.append(critic_loss.item())
            epoch_expert_losses.append(expert_loss.item())
            epoch_rewards.append(rewards.mean().item())
            epoch_safety_violations.append(safety_violation.item())
            
            self.global_step += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: "
                      f"Total Loss = {total_loss.item():.4f}, "
                      f"Actor Loss = {actor_loss.item():.4f}, "
                      f"Critic Loss = {critic_loss.item():.4f}, "
                      f"Expert Loss = {expert_loss.item():.4f}, "
                      f"Avg Reward = {rewards.mean().item():.4f}")
        
        # 计算平均指标
        avg_actor_loss = np.mean(epoch_actor_losses) if epoch_actor_losses else 0
        avg_critic_loss = np.mean(epoch_critic_losses) if epoch_critic_losses else 0
        avg_expert_loss = np.mean(epoch_expert_losses) if epoch_expert_losses else 0
        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0
        avg_safety_violations = np.mean(epoch_safety_violations) if epoch_safety_violations else 0
        avg_grad_norm = np.mean(epoch_gradient_norms) if epoch_gradient_norms else 0
        
        total_avg_loss = avg_critic_loss + avg_actor_loss + 0.5 * avg_expert_loss + 0.1 * avg_safety_violations
        
        # 更新历史记录
        self.history['train_actor_loss'].append(avg_actor_loss)
        self.history['train_critic_loss'].append(avg_critic_loss)
        self.history['train_expert_loss'].append(avg_expert_loss)
        self.history['train_total_loss'].append(total_avg_loss)
        self.history['train_reward'].append(avg_reward)
        self.history['safety_violations'].append(avg_safety_violations)
        self.history['gradient_norm'].append(avg_grad_norm)
        self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
        self.history['rl_sl_weights'].append([self.agent.rl_weight.item(), self.agent.sl_weight.item()])
        
        return total_avg_loss, avg_reward
    
    def calculate_safety_violations(self, action_probs):
        """✨ 计算安全违规情况 - 使用模型内置安全层的规则"""
        violations = torch.zeros(action_probs.size(0), device=self.device)
        
        # ✨ 使用模型内置安全层的危险组合规则
        dangerous_combinations = [
            [1, 2],   # 血管加压药 + 降压药 (相互抵消)
            [3, 11],  # 利尿剂 + 大量电解质 (电解质紊乱)
            [17, 18], # 镇痛剂 + 镇静剂 (过度镇静)
        ]
        high_risk_actions = [1, 17, 18, 19]  # 血管加压药、镇痛、镇静、抗凝
        
        # 检查危险动作组合 - 更严格
        for dangerous_combo in dangerous_combinations:
            combo_prob = torch.prod(action_probs[:, dangerous_combo], dim=1)
            violations += combo_prob * 2.0  # 加重危险组合惩罚
        
        # 检查高风险动作
        high_risk_probs = action_probs[:, high_risk_actions]
        violations += high_risk_probs.sum(dim=1) * 1.0  # 提高高风险动作惩罚
        
        # 检查动作分布集中度 - 过于集中也不安全
        max_prob = torch.max(action_probs, dim=1)[0]
        violations += torch.clamp(max_prob - 0.8, min=0) * 0.5  # 单一动作概率过高
        
        return violations.mean()
    
    def calculate_expert_imitation_loss(self, current_features, expert_logits, batch_idx):
        """计算专家模仿损失"""
        batch_size = current_features.size(0)
        
        # 获取专家数据批次
        expert_states, expert_actions = self.expert_loader.get_expert_batch(batch_size)
        
        # 安全地转换为张量 - 处理object数组问题
        try:
            # 确保expert_states是数值数组
            if expert_states.dtype == np.object_:
                # 如果是object数组，尝试转换为float数组
                expert_states_list = []
                for state in expert_states:
                    if isinstance(state, (list, np.ndarray)):
                        expert_states_list.append(np.array(state, dtype=np.float32))
                    else:
                        # 如果是单个值，创建固定长度的数组
                        expert_states_list.append(np.zeros(current_features.size(-1), dtype=np.float32))
                expert_states = np.array(expert_states_list, dtype=np.float32)
            
            # 确保形状匹配
            if expert_states.shape[1] != current_features.size(-1):
                # 调整形状以匹配当前特征维度
                target_dim = current_features.size(-1)
                if expert_states.shape[1] < target_dim:
                    # 填充zeros
                    padding = np.zeros((expert_states.shape[0], target_dim - expert_states.shape[1]))
                    expert_states = np.concatenate([expert_states, padding], axis=1)
                else:
                    # 截断
                    expert_states = expert_states[:, :target_dim]
            
            expert_states = torch.tensor(expert_states, dtype=torch.float32, device=self.device)
            expert_actions = torch.tensor(expert_actions, dtype=torch.long, device=self.device)
            
            # 计算交叉熵损失
            expert_loss = nn.CrossEntropyLoss()(expert_logits, expert_actions)
            
        except Exception as e:
            print(f"⚠️ 专家数据转换错误: {e}, 使用零损失")
            # 返回零损失以避免训练中断
            expert_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return expert_loss
    
    def validate(self, val_sequences):
        """增强的模型验证 - 包含临床指标"""
        self.agent.eval()
        val_losses = []
        val_rewards = []
        clinical_metrics = {
            'sofa_scores': [],
            'mortality_predictions': [],
            'treatment_effectiveness': [],
            'safety_scores': []
        }
        
        with torch.no_grad():
            for features, seq_lengths in self.prepare_batch(val_sequences):
                # 修复：处理新的字典返回格式
                outputs = self.agent(features, seq_lengths, use_expert_imitation=False)
                action_logits = outputs['constrained_logits']
                state_values = outputs['state_values']
                
                # ✨ 安全约束已在模型内部处理
                action_probs = torch.softmax(action_logits, dim=-1)
                
                rewards = self.reward_calculator.calculate_reward(
                    features, action_logits, episode_step=self.global_step
                ).to(self.device)
                
                state_values_squeezed = state_values.squeeze()
                
                critic_loss = self.huber_loss(state_values_squeezed, rewards)
                
                total_loss = critic_loss.item()
                
                if not (np.isnan(total_loss) or np.isinf(total_loss)):
                    val_losses.append(total_loss)
                    val_rewards.append(rewards.mean().item())
                    
                    # 收集临床指标
                    self.collect_clinical_metrics(features, action_probs, state_values, clinical_metrics)
        
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        avg_val_reward = np.mean(val_rewards) if val_rewards else 0
        
        self.history['val_loss'].append(avg_val_loss)
        self.history['val_reward'].append(avg_val_reward)
        
        # 计算临床指标
        clinical_summary = self.summarize_clinical_metrics(clinical_metrics)
        
        # 新增：将计算出的临床指标保存到history中
        self.history['val_clinical_metrics'].append(clinical_summary)
        
        return avg_val_loss, avg_val_reward, clinical_summary
    
    def collect_clinical_metrics(self, features, action_probs, state_values, metrics):
        """收集临床指标"""
        # SOFA评分变化
        sofa_scores = features[:, -1, -1]  # 最后一个特征是SOFA评分
        metrics['sofa_scores'].extend(sofa_scores.cpu().numpy())
        
        # 死亡风险预测（基于状态值）
        mortality_risk = torch.sigmoid(-state_values.squeeze())  # 状态值越低，死亡风险越高
        metrics['mortality_predictions'].extend(mortality_risk.cpu().numpy())
        
        # 治疗有效性（基于动作选择的集中度）
        action_entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1)
        treatment_effectiveness = 1.0 / (1.0 + action_entropy)  # 熵越低，治疗越集中
        metrics['treatment_effectiveness'].extend(treatment_effectiveness.cpu().numpy())
        
        # 安全评分
        safety_scores = self.calculate_safety_scores(action_probs)
        metrics['safety_scores'].extend(safety_scores.cpu().numpy())
    
    def calculate_safety_scores(self, action_probs):
        """✨ 计算安全评分 - 使用内置安全层规则"""
        safety_scores = torch.ones(action_probs.size(0), device=self.device)
        
        # ✨ 检查危险动作组合
        dangerous_combinations = [
            [1, 2],   # 血管加压药 + 降压药 (相互抵消)
            [3, 11],  # 利尿剂 + 大量电解质 (电解质紊乱)
            [17, 18], # 镇痛剂 + 镇静剂 (过度镇静)
        ]
        
        for dangerous_combo in dangerous_combinations:
            combo_prob = torch.prod(action_probs[:, dangerous_combo], dim=1)
            safety_scores -= combo_prob  # 危险组合概率越高，安全评分越低
        
        return torch.clamp(safety_scores, min=0.0, max=1.0)
    
    def summarize_clinical_metrics(self, metrics):
        """汇总临床指标"""
        summary = {}
        
        if metrics['sofa_scores']:
            sofa_scores = np.array(metrics['sofa_scores'])
            summary['avg_sofa_score'] = np.mean(sofa_scores)
            summary['sofa_improvement'] = max(0, 15 - np.mean(sofa_scores)) / 15  # 标准化改善指标
        
        if metrics['mortality_predictions']:
            mortality_risk = np.array(metrics['mortality_predictions'])
            summary['mortality_risk'] = np.mean(mortality_risk)
        
        if metrics['treatment_effectiveness']:
            effectiveness = np.array(metrics['treatment_effectiveness'])
            summary['treatment_effectiveness'] = np.mean(effectiveness)
        
        if metrics['safety_scores']:
            safety = np.array(metrics['safety_scores'])
            summary['safety_score'] = np.mean(safety)
        
        return summary
    
    def train(self, train_sequences, val_sequences, epochs=30):
        """增强的训练模型"""
        print("🚀 开始增强版败血症训练...")
        print(f"训练序列: {len(train_sequences)}")
        print(f"验证序列: {len(val_sequences)}")
        print(f"训练轮数: {epochs}")
        
        best_val_reward = -float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_reward = self.train_epoch(train_sequences, epoch)
            
            # 验证
            val_loss, val_reward, val_metrics = self.validate(val_sequences)
            
            # 更新学习率
            self.scheduler.step(val_reward)
            
            print(f"\n📊 Epoch {epoch + 1}/{epochs}:")
            print(f"  Train - Loss: {train_loss:.4f}, Reward: {train_reward:.4f}")
            print(f"  Val - Loss: {val_loss:.4f}, Reward: {val_reward:.4f}")
            print(f"  Safety Violations: {self.history['safety_violations'][-1]:.4f}")
            print(f"  Gradient Norm: {self.history['gradient_norm'][-1]:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # 输出临床指标
            if val_metrics:
                print(f"  Clinical Metrics - Mortality Risk: {val_metrics.get('mortality_risk', 0):.4f}, "
                      f"SOFA Improvement: {val_metrics.get('sofa_improvement', 0):.4f}")
            
            # 早停和模型保存
            if val_reward > best_val_reward:
                best_val_reward = val_reward
                patience_counter = 0
                torch.save(self.agent.state_dict(), 'models/enhanced_best_sepsis_agent.pth')
                print("  ✅ 新的最佳模型已保存!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                checkpoint_path = f'models/enhanced_checkpoint_epoch_{epoch+1}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.agent.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_reward': best_val_reward,
                    'history': self.history
                }, checkpoint_path)
                print(f"  💾 检查点已保存: {checkpoint_path}")
        
        # 新增：在训练结束后保存完整的历史记录
        self.save_full_history()
        
        print("\n🎉 训练完成!")
        return self.history
    
    def save_training_report(self, filename='enhanced_training_report.txt'):
        """保存训练报告"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("Enhanced Sepsis DRL Training Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Epochs: {len(self.history['train_reward'])}\n")
            f.write(f"Best Training Reward: {max(self.history['train_reward']):.4f}\n")
            f.write(f"Best Validation Reward: {max(self.history['val_reward']):.4f}\n")
            
            if self.history['safety_violations']:
                f.write(f"Final Safety Violations: {self.history['safety_violations'][-1]:.4f}\n")
            if self.history['gradient_norm']:
                f.write(f"Final Gradient Norm: {self.history['gradient_norm'][-1]:.4f}\n")
            if self.history['learning_rate']:
                f.write(f"Final Learning Rate: {self.history['learning_rate'][-1]:.2e}\n\n")
            
            f.write("Training Improvements:\n")
            f.write("- Enhanced LSTM architecture with attention mechanism\n")
            f.write("- Clinical reward function based on medical indicators\n")
            f.write("- Medical safety constraints to prevent dangerous combinations\n")
            f.write("- Improved training stability with gradient clipping and regularization\n")
            f.write("- Comprehensive clinical metrics evaluation\n")
        
        print(f"Training report saved to {filename}")

    def save_full_history(self, filename='training_history.json'):
        """新增：保存完整的训练历史到JSON文件"""
        # 将numpy类型转换为Python原生类型以便JSON序列化
        history_to_save = {}
        for key, value in self.history.items():
            if isinstance(value, list) and len(value) > 0:
                # 处理包含numpy数值或张量的列表
                if isinstance(value[0], (np.generic, np.ndarray)):
                    history_to_save[key] = [item.item() if hasattr(item, 'item') else item for item in value]
                elif isinstance(value[0], torch.Tensor):
                     history_to_save[key] = [item.item() for item in value]
                elif isinstance(value[0], dict): # 处理字典列表
                    history_to_save[key] = [{k: (v.item() if hasattr(v, 'item') else v) for k, v in item.items()} for item in value]
                else:
                    history_to_save[key] = value
            else:
                history_to_save[key] = value
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(history_to_save, f, indent=4)
        print(f"  💾 完整的训练历史已保存至: {filename}")
    
    def plot_training_history(self):
        """Generate individual training curve plots for better visualization and saving"""
        # Create plots directory if it doesn't exist
        Path('plots').mkdir(exist_ok=True)
        
        # Plot 1: Total Loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_total_loss'], label='Train Total Loss', color='blue', linewidth=2)
        plt.plot(self.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/total_loss.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 2: Rewards
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_reward'], label='Train Reward', color='purple', linewidth=2)
        plt.plot(self.history['val_reward'], label='Validation Reward', color='brown', linewidth=2)
        plt.title('Clinical Rewards Over Training', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/clinical_rewards.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 3: Actor vs Critic vs Expert Loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_actor_loss'], label='Actor Loss (RL)', color='green', linewidth=2)
        plt.plot(self.history['train_critic_loss'], label='Critic Loss', color='orange', linewidth=2)
        if self.history['train_expert_loss']:
            plt.plot(self.history['train_expert_loss'], label='Expert Loss (SL)', color='blue', linewidth=2)
        plt.title('Actor vs Critic vs Expert Loss Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/actor_critic_expert_loss.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 4: Safety Violations
        if self.history['safety_violations']:
            plt.figure(figsize=(10, 6))
            plt.plot(self.history['safety_violations'], label='Safety Violations', color='red', linewidth=2)
            plt.title('Safety Violations During Training', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Violation Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('plots/safety_violations.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Plot 5: Gradient Norm
        if self.history['gradient_norm']:
            plt.figure(figsize=(10, 6))
            plt.plot(self.history['gradient_norm'], label='Gradient Norm', color='cyan', linewidth=2)
            plt.title('Gradient Norm During Training', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Norm')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('plots/gradient_norm.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Plot 6: Learning Rate Schedule
        if self.history['learning_rate']:
            plt.figure(figsize=(10, 6))
            plt.plot(self.history['learning_rate'], label='Learning Rate', color='magenta', linewidth=2)
            plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('plots/learning_rate.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Plot 7: RL vs SL Weight Evolution
        if self.history['rl_sl_weights']:
            plt.figure(figsize=(10, 6))
            rl_weights = [w[0] for w in self.history['rl_sl_weights']]
            sl_weights = [w[1] for w in self.history['rl_sl_weights']]
            plt.plot(rl_weights, label='RL Weight', color='red', linewidth=2)
            plt.plot(sl_weights, label='SL Weight', color='blue', linewidth=2)
            plt.title('RL vs SL Weight Evolution During Training', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Weight')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('plots/rl_sl_weights.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print("📊 Individual plots saved to 'plots/' directory:")

def load_sepsis_data(data_dir='preprocessed_sepsis_data'):
    """加载预处理的败血症数据"""
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Preprocessed data directory {data_dir} not found!")
    
    # 加载特征信息
    feature_info = pd.read_csv(data_dir / 'feature_info.csv')
    feature_names = eval(feature_info['feature_names'].iloc[0])
    
    print(f"✅ 加载特征信息: {len(feature_names)} 个特征")
    
    # 加载训练和验证数据
    train_features = np.load(data_dir / 'train_features.npy')
    val_features = np.load(data_dir / 'val_features.npy')
    train_metadata = pd.read_csv(data_dir / 'train_metadata.csv')
    val_metadata = pd.read_csv(data_dir / 'val_metadata.csv')
    
    print(f"✅ 加载训练数据: {train_features.shape[0]} 个序列")
    print(f"✅ 加载验证数据: {val_features.shape[0]} 个序列")
    
    # 转换为序列格式
    def create_sequences(features, metadata):
        sequences = []
        for i in range(len(features)):
            # 修正: 这里的序列长度应该是动态的，但当前数据预处理是固定的
            # 如果您的数据是变长的，需要在这里进行相应修改
            sequence_length = 7  # 假设固定序列长度
            sequences.append({
                'features': features[i].reshape(sequence_length, -1),
                'sequence_length': sequence_length
            })
        return sequences
    
    train_sequences = create_sequences(train_features, train_metadata)
    val_sequences = create_sequences(val_features, val_metadata)
    
    # 数据验证
    valid_train_sequences = []
    for seq in train_sequences:
        if (seq['features'].shape[0] > 0 and 
            not np.isnan(seq['features']).all() and
            seq['sequence_length'] > 0):
            valid_train_sequences.append(seq)
    
    valid_val_sequences = []
    for seq in val_sequences:
        if (seq['features'].shape[0] > 0 and 
            not np.isnan(seq['features']).all() and
            seq['sequence_length'] > 0):
            valid_val_sequences.append(seq)
    
    print(f"✅ 有效训练序列: {len(valid_train_sequences)}")
    print(f"✅ 有效验证序列: {len(valid_val_sequences)}")
    
    input_dim = valid_train_sequences[0]['features'].shape[1] if valid_train_sequences else len(feature_names)
    
    return valid_train_sequences, valid_val_sequences, input_dim, feature_names

def main():
    """主函数 - 增强版"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    # 创建models目录
    Path('models').mkdir(exist_ok=True)
    
    try:
        # 加载数据
        print("📊 加载数据...")
        train_sequences, val_sequences, input_dim, feature_names = load_sepsis_data()
        
        # 创建模型
        print("🤖 创建模型...")
        agent = EnhancedSepsisLSTMAgent(
            input_dim=input_dim,
            hidden_dim=128,
            num_actions=20,
            sequence_length=7
        )
        
        # 创建奖励计算器
        reward_calculator = ClinicalRewardCalculator()
        
        # 创建训练器 - 更稳定的学习率
        trainer = EnhancedTrainer(
            agent=agent,
            reward_calculator=reward_calculator,
            lr=3e-4,  # 恢复正常学习率
            device=device,
            expert_data_path=None  # 自动使用真实专家数据
        )
        
        # 训练模型
        print("🏋️ 开始训练...")
        history = trainer.train(train_sequences, val_sequences, epochs=25)
        
        # 绘制训练曲线
        print("📈 绘制训练曲线...")
        trainer.plot_training_history()
        
        # 保存训练报告
        print("📄 生成训练报告...")
        trainer.save_training_report()
        
        print("🎉 Enhanced training completed!")
        print("📁 Generated files:")
        print("  - models/enhanced_best_sepsis_agent.pth")
        print("  - enhanced_training_report.txt")
        print("  - training_history.json (新增)") # 新增
        print("  - models/enhanced_checkpoint_epoch_*.pth")
        print("  - plots/total_loss.png")
        print("  - plots/clinical_rewards.png")
        print("  - plots/actor_critic_expert_loss.png")
        print("  - plots/rl_sl_weights.png")
        print("  - plots/safety_violations.png")
        print("  - plots/gradient_norm.png")
        print("  - plots/learning_rate.png")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
