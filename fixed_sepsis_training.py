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
warnings.filterwarnings('ignore')

class EnhancedSepsisLSTMAgent(nn.Module):
    """增强的败血症LSTM智能体 - 包含医学约束"""
    
    def __init__(self, input_dim, hidden_dim=128, num_actions=20, sequence_length=7):
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
        
        # 医学特征提取器
        self.medical_feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim//2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim//2),
            nn.Dropout(0.1)
        )
        
        # 增强的Actor网络
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, num_actions)
        )
        
        # 增强的Critic网络
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, 1)
        )
        
        # 医学约束层
        self.medical_constraint = nn.Linear(num_actions, num_actions)
        
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
    
    def forward(self, sequences, sequence_lengths=None):
        """前向传播"""
        batch_size = sequences.size(0)
        
        # LSTM处理时序数据
        lstm_output, (h_n, c_n) = self.lstm(sequences)
        
        # Layer normalization
        lstm_output = self.layer_norm1(lstm_output)
        
        # 注意力机制
        attended_output, attention_weights = self.attention(
            lstm_output, lstm_output, lstm_output
        )
        
        # 结合LSTM和注意力输出
        combined_output = lstm_output + attended_output
        combined_output = self.layer_norm2(combined_output)
        
        # 使用最后一个时间步的输出
        if sequence_lengths is not None:
            last_outputs = []
            for i, length in enumerate(sequence_lengths):
                last_outputs.append(combined_output[i, length-1, :])
            temporal_context = torch.stack(last_outputs)
        else:
            temporal_context = combined_output[:, -1, :]
        
        # 医学特征提取（使用最新的医学数据）
        current_medical_state = sequences[:, -1, :]
        medical_features = self.medical_feature_extractor(current_medical_state)
        
        # 融合时序和医学特征
        combined_features = torch.cat([temporal_context, medical_features], dim=1)
        
        # Actor和Critic输出
        raw_action_logits = self.actor(combined_features)
        
        # 应用医学约束
        constrained_action_logits = self.medical_constraint(raw_action_logits)
        
        state_values = self.critic(combined_features)
        
        return constrained_action_logits, state_values, combined_output, attention_weights

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
        综合临床奖励计算 - 改进版，允许更大的奖励范围和动态性
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
        
        # 添加随机性奖励以增加探索
        exploration_bonus = torch.randn_like(total_reward) * 0.05
        
        final_reward = total_reward + survival_bonus + exploration_bonus
        
        # 扩大奖励范围，移除过于严格的限制
        return torch.clamp(final_reward, min=-1.0, max=5.0)

class MedicalSafetyConstraints:
    """医学安全约束类"""
    
    def __init__(self, num_actions=20):
        self.num_actions = num_actions
        
        # 定义危险动作组合（示例）
        self.dangerous_combinations = [
            [0, 1],   # 示例：某些药物组合
            [5, 10],  # 示例：高剑量组合
        ]
        
        # 最大允许剑量
        self.max_dosage_actions = [15, 16, 17, 18, 19]  # 示例：高剂量动作
        
    def apply_constraints(self, action_logits, medical_state=None):
        """应用医学安全约束"""
        constrained_logits = action_logits.clone()
        
        # 约束1：防止危险药物组合
        action_probs = torch.softmax(action_logits, dim=-1)
        
        for dangerous_combo in self.dangerous_combinations:
            combo_prob = torch.prod(action_probs[:, dangerous_combo], dim=1)
            if combo_prob.max() > 0.1:  # 如果危险组合概率过高
                for action_idx in dangerous_combo:
                    constrained_logits[:, action_idx] -= 2.0  # 降低概率
        
        # 约束2：基于当前医学状态的约束
        if medical_state is not None:
            # 如果血压过低，限制降压药物
            bp_systolic = medical_state[:, 3]  # 收缩压索引
            low_bp_mask = bp_systolic < 90
            if low_bp_mask.any():
                constrained_logits[low_bp_mask, 10:15] -= 3.0  # 限制降压类动作
            
            # 如果心率过高，限制兴奋剂
            heart_rate = medical_state[:, 1]
            high_hr_mask = heart_rate > 120
            if high_hr_mask.any():
                constrained_logits[high_hr_mask, 5:10] -= 3.0  # 限制兴奋剂类动作
        
        return constrained_logits

class EnhancedTrainer:
    """增强的训练器 - 包含医学安全约束"""
    
    def __init__(self, agent, reward_calculator, lr=5e-5, device='cpu'):
        """初始化训练器"""
        self.agent = agent.to(device)
        self.reward_calculator = reward_calculator
        self.device = device
        self.safety_constraints = MedicalSafetyConstraints()
        
        # 更稳定的优化器设置
        self.optimizer = optim.AdamW(
            self.agent.parameters(), 
            lr=lr, 
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.8, patience=3, verbose=True
        )
        
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.SmoothL1Loss()  # 更稳定的损失函数
        self.global_step = 0
        
        # 增强的历史记录
        self.history = {
            'train_total_loss': [],
            'train_actor_loss': [],
            'train_critic_loss': [],
            'train_reward': [],
            'val_loss': [],
            'val_reward': [],
            'learning_rate': [],
            'gradient_norm': [],
            'safety_violations': []
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
        """增强的训练epoch - 包含安全约束和改进的损失计算"""
        self.agent.train()
        epoch_actor_losses = []
        epoch_critic_losses = []
        epoch_rewards = []
        epoch_safety_violations = []
        epoch_gradient_norms = []
        
        for batch_idx, (features, seq_lengths) in enumerate(self.prepare_batch(train_sequences)):
            
            self.optimizer.zero_grad()
            
            # 前向传播
            action_logits, state_values, _, attention_weights = self.agent(features, seq_lengths)
            
            # 应用医学安全约束
            current_medical_state = features[:, -1, :]
            constrained_action_logits = self.safety_constraints.apply_constraints(
                action_logits, current_medical_state
            )
            
            # 计算动作概率分布
            action_probs = torch.softmax(constrained_action_logits, dim=-1)
            
            # 计算奖励
            rewards = self.reward_calculator.calculate_reward(
                features, constrained_action_logits, episode_step=self.global_step
            ).to(self.device)
            
            # 更稳定的Critic损失计算
            state_values_squeezed = state_values.squeeze()
            if state_values_squeezed.dim() == 0:
                state_values_squeezed = state_values_squeezed.unsqueeze(0)
            
            # 使用Huber损失更稳定
            critic_loss = self.huber_loss(state_values_squeezed, rewards)
            
            # 计算advantages并标准化
            advantages = rewards - state_values_squeezed.detach()
            if advantages.std() > 1e-6:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 简化的Actor损失计算 - 标准策略梯度
            action_dist = torch.distributions.Categorical(action_probs)
            sampled_actions = action_dist.sample()
            
            # 计算选中动作的对数概率
            log_probs = action_dist.log_prob(sampled_actions)
            
            # 标准策略梯度损失 (REINFORCE)
            policy_loss = -torch.mean(log_probs * advantages.detach())
            
            # 熵正则化
            entropy = action_dist.entropy().mean()
            entropy_bonus = 0.02 * entropy
            
            # 注意力正则化（鼓励关注重要时间步）
            attention_penalty = 0.001 * torch.var(attention_weights).mean()
            
            # 最终Actor损失
            actor_loss = policy_loss - entropy_bonus + attention_penalty
            
            # 安全约束损失
            safety_violation = self.calculate_safety_violations(action_probs)
            safety_loss = 0.1 * safety_violation
            
            # 总损失
            total_loss = critic_loss + actor_loss + safety_loss
            
            # 检查NaN/Inf
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"NaN/Inf detected at epoch {epoch}, batch {batch_idx}. Skipping...")
                continue
            
            # 反向传播
            total_loss.backward()
            
            # 计算梯度范数
            grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=0.5)
            epoch_gradient_norms.append(grad_norm.item())
            
            # 更新参数
            self.optimizer.step()
            
            # 记录指标
            epoch_actor_losses.append(actor_loss.item())
            epoch_critic_losses.append(critic_loss.item())
            epoch_rewards.append(rewards.mean().item())
            epoch_safety_violations.append(safety_violation.item())
            
            self.global_step += 1
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: "
                      f"Actor Loss = {actor_loss.item():.6f}, "
                      f"Critic Loss = {critic_loss.item():.6f}, "
                      f"Avg Reward = {rewards.mean().item():.4f}, "
                      f"Entropy = {entropy.item():.4f}, "
                      f"Safety Violations = {safety_violation.item():.4f}")
        
        # 计算平均指标
        avg_actor_loss = np.mean(epoch_actor_losses) if epoch_actor_losses else 0
        avg_critic_loss = np.mean(epoch_critic_losses) if epoch_critic_losses else 0
        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0
        avg_safety_violations = np.mean(epoch_safety_violations) if epoch_safety_violations else 0
        avg_grad_norm = np.mean(epoch_gradient_norms) if epoch_gradient_norms else 0
        
        total_avg_loss = avg_actor_loss + avg_critic_loss
        
        # 更新历史记录
        self.history['train_actor_loss'].append(avg_actor_loss)
        self.history['train_critic_loss'].append(avg_critic_loss)
        self.history['train_total_loss'].append(total_avg_loss)
        self.history['train_reward'].append(avg_reward)
        self.history['safety_violations'].append(avg_safety_violations)
        self.history['gradient_norm'].append(avg_grad_norm)
        self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
        
        return total_avg_loss, avg_reward
    
    def calculate_safety_violations(self, action_probs):
        """计算安全违规情况"""
        violations = torch.zeros(action_probs.size(0), device=self.device)
        
        # 检查危险动作组合
        for dangerous_combo in self.safety_constraints.dangerous_combinations:
            combo_prob = torch.prod(action_probs[:, dangerous_combo], dim=1)
            violations += combo_prob  # 危险组合的概率作为违规指标
        
        # 检查高剂量动作
        high_dose_probs = action_probs[:, self.safety_constraints.max_dosage_actions]
        violations += high_dose_probs.sum(dim=1) * 0.5  # 高剂量动作总概率
        
        return violations.mean()
    
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
                action_logits, state_values, _, attention_weights = self.agent(features, seq_lengths)
                
                # 应用安全约束
                current_medical_state = features[:, -1, :]
                constrained_action_logits = self.safety_constraints.apply_constraints(
                    action_logits, current_medical_state
                )
                
                action_probs = torch.softmax(constrained_action_logits, dim=-1)
                
                rewards = self.reward_calculator.calculate_reward(
                    features, constrained_action_logits, episode_step=self.global_step
                ).to(self.device)
                
                state_values_squeezed = state_values.squeeze()
                if state_values_squeezed.dim() == 0:
                    state_values_squeezed = state_values_squeezed.unsqueeze(0)
                
                critic_loss = self.huber_loss(state_values_squeezed, rewards)
                
                # 计算Actor损失
                advantages = rewards - state_values_squeezed
                action_dist = torch.distributions.Categorical(action_probs)
                sampled_actions = action_dist.sample()
                log_probs = action_dist.log_prob(sampled_actions)
                actor_loss = -torch.mean(log_probs * advantages)
                
                total_loss = critic_loss.item() + actor_loss.item()
                
                if not (torch.isnan(torch.tensor(total_loss)) or torch.isinf(torch.tensor(total_loss))):
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
        """计算安全评分"""
        safety_scores = torch.ones(action_probs.size(0), device=self.device)
        
        # 检查危险动作组合
        for dangerous_combo in self.safety_constraints.dangerous_combinations:
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
            print(f"  Train - Loss: {train_loss:.6f}, Reward: {train_reward:.4f}")
            print(f"  Val - Loss: {val_loss:.6f}, Reward: {val_reward:.4f}")
            print(f"  Safety Violations: {self.history['safety_violations'][-1]:.4f}")
            print(f"  Gradient Norm: {self.history['gradient_norm'][-1]:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # 输出临床指标
            if val_metrics:
                print(f"  Clinical Metrics - Mortality Risk: {val_metrics['mortality_risk']:.4f}, "
                      f"SOFA Improvement: {val_metrics['sofa_improvement']:.4f}")
            
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
        
        # Plot 3: Actor vs Critic Loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_actor_loss'], label='Actor Loss', color='green', linewidth=2)
        plt.plot(self.history['train_critic_loss'], label='Critic Loss', color='orange', linewidth=2)
        plt.title('Actor vs Critic Loss Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/actor_critic_loss.png', dpi=300, bbox_inches='tight')
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
        
        # 创建训练器
        trainer = EnhancedTrainer(
            agent=agent,
            reward_calculator=reward_calculator,
            lr=1e-4,  # 提高学习率以适应新的奖励函数
            device=device
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
        print("  - models/enhanced_checkpoint_epoch_*.pth")
        print("  - plots/total_loss.png")
        print("  - plots/clinical_rewards.png")
        print("  - plots/actor_critic_loss.png")
        print("  - plots/safety_violations.png")
        print("  - plots/gradient_norm.png")
        print("  - plots/learning_rate.png")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 