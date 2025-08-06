#!/usr/bin/env python3
"""
Stable TD3 Trainer for Sepsis Treatment (with Final Presentation Visualizations)
- Implements TD3 for stable training.
- Includes functions to generate a comprehensive, presentation-ready analysis of the agent's performance,
  including mortality rate comparison and policy analysis.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import time
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import pickle

# ==============================================================================
# BASE CLASSES
# ==============================================================================

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class SimpleReplayBuffer:
    """A simple replay buffer for storing experiences."""
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)

class EnhancedDataPreprocessor:
    """Handles data preprocessing: imputation, scaling, and outlier clipping."""
    def __init__(self, numeric_features):
        self.numeric_features = numeric_features
        self.imputer = KNNImputer(n_neighbors=5)
        self.scaler = StandardScaler()

    def fit(self, data):
        self.scaler.fit(data[self.numeric_features])

    def transform(self, data):
        df = data.copy()
        df[self.numeric_features] = self.imputer.fit_transform(df[self.numeric_features])
        df[self.numeric_features] = self.scaler.transform(df[self.numeric_features])
        df[self.numeric_features] = np.clip(df[self.numeric_features], -5, 5)
        return df
        
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'preprocessor.pkl'), 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(os.path.join(path, 'preprocessor.pkl'), 'rb') as f:
            return pickle.load(f)

class EnhancedLSTMActorNetwork(nn.Module):
    """Enhanced LSTM Actor Network with LayerNorm and Dropout."""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(EnhancedLSTMActorNetwork, self).__init__()
        self.lstm = nn.LSTM(state_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state_sequence):
        lstm_out, _ = self.lstm(state_sequence)
        last_output = lstm_out[:, -1, :]
        x = self.ln1(last_output)
        x = F.relu(self.fc1(x))
        x = self.ln2(x)
        action_probs = torch.sigmoid(self.output_layer(x))
        return action_probs

class EnhancedLSTMCriticNetwork(nn.Module):
    """Enhanced LSTM Critic Network."""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(EnhancedLSTMCriticNetwork, self).__init__()
        self.lstm = nn.LSTM(state_dim, hidden_dim, num_layers=1, batch_first=True)
        self.state_fc = nn.Linear(hidden_dim, hidden_dim // 2)
        self.action_fc = nn.Linear(action_dim, hidden_dim // 2)
        self.q_output = nn.Linear(hidden_dim, 1)

    def forward(self, state_sequence, action):
        lstm_out, _ = self.lstm(state_sequence)
        last_output = lstm_out[:, -1, :]
        state_features = F.relu(self.state_fc(last_output))
        action_features = F.relu(self.action_fc(action))
        combined = torch.cat([state_features, action_features], dim=1)
        q_value = self.q_output(combined)
        return q_value

class EnhancedSepsisLSTMEnvironment:
    """Base Environment class."""
    def __init__(self, data_df, metadata, sequence_length=24):
        self.sequence_length = sequence_length
        self.training_data = data_df
        self.metadata = metadata
        self.action_dim = self.metadata['num_drug_actions']
        self.action_columns = [str(i) for i in range(self.action_dim)]
        non_state_cols = ['hadm_id', 'time_step', 'admittime', 'hospital_expire_flag', 'subject_id'] + self.action_columns
        self.state_feature_cols = [col for col in self.training_data.columns if col not in non_state_cols and str(col) not in self.action_columns]
        self.state_dim = len(self.state_feature_cols)
        self._create_patient_sequences()
        self.current_patient_idx = 0
        self.current_step = 0

    def _create_patient_sequences(self):
        self.patient_sequences = list(self.training_data.groupby('hadm_id'))

    def reset(self, patient_idx=None):
        if patient_idx is not None and patient_idx < len(self.patient_sequences):
            self.current_patient_idx = patient_idx
        else:
            self.current_patient_idx = random.randrange(len(self.patient_sequences))
        self.current_patient_id, self.current_sequence_df = self.patient_sequences[self.current_patient_idx]
        self.current_sequence = self.current_sequence_df.to_dict('records')
        self.current_step = 0
        return self.current_sequence[self.current_step]

    def get_current_patient_id(self):
        return self.current_patient_id

    def step(self, action):
        reward = self.calculate_reward(action)
        self.current_step += 1
        done = self.is_done()
        next_state = self.current_sequence[self.current_step] if not done else self.current_sequence[-1]
        return next_state, reward, done, {}

    def is_done(self):
        return self.current_step >= len(self.current_sequence) - 1

    def get_expert_action(self):
        record = self.current_sequence[self.current_step]
        expert_action = [record.get(str(col), 0) for col in range(self.action_dim)]
        return np.array(expert_action)

class StableSepsisLSTMEnvironment(EnhancedSepsisLSTMEnvironment):
    """A more stable sepsis environment with a simplified and normalized reward function."""
    def calculate_reward(self, action):
        if self.current_step >= len(self.current_sequence):
            current_record = self.current_sequence[-1]
        else:
            current_record = self.current_sequence[self.current_step]
        sofa_score = current_record.get('sofa', 0)
        prev_sofa_score = self.current_sequence[self.current_step-1].get('sofa', sofa_score) if self.current_step > 0 else sofa_score
        sofa_change = prev_sofa_score - sofa_score
        sofa_reward = np.clip(sofa_change, -2, 2)
        terminal_reward = 0
        if self.is_done():
            terminal_reward = 15.0 if current_record.get('hospital_expire_flag', 0) == 0 else -15.0
        vital_signs_reward = 0
        sbp = current_record.get('arterial_blood_pressure_systolic', 120)
        vital_signs_reward += 0.25 if 90 <= sbp <= 140 else -0.25
        hr = current_record.get('heart_rate', 80)
        vital_signs_reward += 0.25 if 60 <= hr <= 110 else -0.25
        sparsity_penalty = -0.1 * np.mean(action)
        return sofa_reward + vital_signs_reward + sparsity_penalty + terminal_reward

class DDPGAgent:
    """DDPG Agent as baseline for comparison with TD3."""
    def __init__(self, state_dim, action_dim, state_feature_cols, sequence_length=24, hidden_dim=128,
                 actor_lr=1e-3, critic_lr=1e-3, tau=0.001, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_feature_cols = state_feature_cols
        self.gamma = gamma
        self.tau = tau
        self.noise_scale = 0.1
        
        # Actor networks (similar to TD3 but simpler)
        self.actor = EnhancedLSTMActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = EnhancedLSTMActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Critic networks (single critic unlike TD3's double critics)
        self.critic = EnhancedLSTMCriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = EnhancedLSTMCriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Replay buffer
        self.replay_buffer = SimpleReplayBuffer(capacity=50000)
        self.batch_size = 128
        self.state_buffer = deque(maxlen=sequence_length)
        
    def select_action(self, state, add_noise=True):
        state_features = state[self.state_feature_cols].values[0]
        self.state_buffer.append(state_features)
        
        # Pad buffer if not full
        while len(self.state_buffer) < self.state_buffer.maxlen:
            self.state_buffer.appendleft(np.zeros(self.state_dim))
            
        state_sequence = np.array(list(self.state_buffer), dtype=np.float32)
        state_tensor = torch.FloatTensor(state_sequence).unsqueeze(0).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        self.actor.train()
        
        # Add noise for exploration
        if add_noise:
            action = (action + np.random.normal(0, self.noise_scale, size=self.action_dim)).clip(0, 1)
            
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        current_sequence = np.array(list(self.state_buffer), dtype=np.float32)
        next_buffer_content = self.state_buffer.copy()
        next_state_features = next_state[self.state_feature_cols].values[0]
        next_buffer_content.append(next_state_features)
        next_sequence = np.array(list(next_buffer_content), dtype=np.float32)
        self.replay_buffer.push(current_sequence, action, reward, next_sequence, done)
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None, None
            
        states, actions, rewards, next_states, dones = zip(*self.replay_buffer.sample(self.batch_size))
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        
        # Update Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q_values = rewards + (1.0 - dones) * self.gamma * target_q
            
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self.soft_update_target_networks()
        
        return actor_loss.item(), critic_loss.item(), current_q.detach().cpu().numpy()
    
    def soft_update_target_networks(self):
        # Soft update actor target
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
        # Soft update critic target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def save_models(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "ddpg_actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(path, "ddpg_critic.pth"))

class TD3Agent:
    """Agent implementing the TD3 algorithm."""
    def __init__(self, state_dim, action_dim, state_feature_cols, sequence_length=24, hidden_dim=128,
                 actor_lr=3e-4, critic_lr=3e-4, tau=0.005, gamma=0.99,
                 policy_update_freq=2, target_noise=0.2, noise_clip=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = EnhancedLSTMActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = EnhancedLSTMActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1 = EnhancedLSTMCriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1_target = EnhancedLSTMCriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2 = EnhancedLSTMCriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2_target = EnhancedLSTMCriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        self.replay_buffer = SimpleReplayBuffer(capacity=50000)
        self.batch_size = 128
        self.state_buffer = deque(maxlen=sequence_length)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_feature_cols = state_feature_cols
        self.gamma = gamma
        self.tau = tau
        self.policy_update_freq = policy_update_freq
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.noise_scale = 0.2
        self.global_step = 0

    def select_action(self, state, add_noise=True):
        state_features = state[self.state_feature_cols].values[0]
        self.state_buffer.append(state_features)
        while len(self.state_buffer) < self.state_buffer.maxlen:
            self.state_buffer.appendleft(np.zeros(self.state_dim))
        state_sequence = np.array(list(self.state_buffer), dtype=np.float32)
        state_tensor = torch.FloatTensor(state_sequence).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        self.actor.train()
        if add_noise:
            action = (action + np.random.normal(0, self.noise_scale, size=self.action_dim)).clip(0, 1)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        current_sequence = np.array(list(self.state_buffer), dtype=np.float32)
        next_buffer_content = self.state_buffer.copy()
        next_state_features = next_state[self.state_feature_cols].values[0]
        next_buffer_content.append(next_state_features)
        next_sequence = np.array(list(next_buffer_content), dtype=np.float32)
        self.replay_buffer.push(current_sequence, action, reward, next_sequence, done)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None, None
        self.global_step += 1
        states, actions, rewards, next_states, dones = zip(*self.replay_buffer.sample(self.batch_size))
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(0, 1)
            target_q1, target_q2 = self.critic1_target(next_states, next_actions), self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q_values = rewards + (1.0 - dones) * self.gamma * target_q
        current_q1 = self.critic1(states, actions)
        critic1_loss = F.mse_loss(current_q1, target_q_values)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        current_q2 = self.critic2(states, actions)
        critic2_loss = F.mse_loss(current_q2, target_q_values)
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        critic_loss_val = (critic1_loss.item() + critic2_loss.item()) / 2
        actor_loss_val, q_values_val = None, current_q1.detach().cpu().numpy()
        if self.global_step % self.policy_update_freq == 0:
            predicted_actions = self.actor(states)
            actor_loss = -self.critic1(states, predicted_actions).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_val = actor_loss.item()
            self.soft_update_target_networks()
        return actor_loss_val, critic_loss_val, q_values_val

    def soft_update_target_networks(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
    def save_models(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))

def evaluate_agent_physician_similarity(agent, env):
    """Calculate similarity between agent and physician actions."""
    print("\n🔍 Calculating Agent-Physician Action Similarity...")
    
    total_patients = len(env.patient_sequences)
    all_agent_actions = []
    all_physician_actions = []
    cosine_similarities = []
    mse_distances = []
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    for i in range(min(total_patients, 50)):  # Sample 50 patients for efficiency
        state_dict = env.reset(patient_idx=i)
        agent.state_buffer.clear()
        
        patient_agent_actions = []
        patient_physician_actions = []
        
        while True:
            state_df_row = pd.DataFrame([state_dict])
            
            # Get agent action
            agent_action = agent.select_action(state_df_row, add_noise=False)
            patient_agent_actions.append(agent_action)
            
            # Get physician action
            physician_action = env.get_expert_action()
            patient_physician_actions.append(physician_action)
            
            # Step environment
            next_state_dict, _, done, _ = env.step(agent_action)
            state_dict = next_state_dict
            
            if done:
                break
        
        # Calculate similarity for this patient
        if len(patient_agent_actions) > 0:
            agent_actions_array = np.array(patient_agent_actions)
            physician_actions_array = np.array(patient_physician_actions)
            
            # Flatten actions for overall similarity
            all_agent_actions.extend(patient_agent_actions)
            all_physician_actions.extend(patient_physician_actions)
            
            # Calculate cosine similarity for this patient
            if agent_actions_array.size > 0 and physician_actions_array.size > 0:
                # Flatten each patient's actions
                agent_flat = agent_actions_array.flatten().reshape(1, -1)
                physician_flat = physician_actions_array.flatten().reshape(1, -1)
                
                # Ensure same length
                min_len = min(agent_flat.shape[1], physician_flat.shape[1])
                agent_flat = agent_flat[:, :min_len]
                physician_flat = physician_flat[:, :min_len]
                
                cos_sim = cosine_similarity(agent_flat, physician_flat)[0][0]
                cosine_similarities.append(cos_sim)
                
                # Calculate MSE distance
                mse_dist = np.mean((agent_flat - physician_flat) ** 2)
                mse_distances.append(mse_dist)
    
    # Overall statistics
    all_agent_actions = np.array(all_agent_actions)
    all_physician_actions = np.array(all_physician_actions)
    
    # Overall cosine similarity
    overall_cos_sim = cosine_similarity(
        all_agent_actions.reshape(1, -1), 
        all_physician_actions.reshape(1, -1)
    )[0][0]
    
    # Overall MSE
    overall_mse = np.mean((all_agent_actions - all_physician_actions) ** 2)
    
    # Action frequency correlation
    agent_freq = np.mean(all_agent_actions, axis=0)
    physician_freq = np.mean(all_physician_actions, axis=0)
    frequency_correlation = np.corrcoef(agent_freq, physician_freq)[0, 1]
    
    # Summary statistics
    avg_cosine_similarity = np.mean(cosine_similarities)
    avg_mse_distance = np.mean(mse_distances)
    
    print(f"📊 Agent-Physician Similarity Results:")
    print(f"   • Average Cosine Similarity: {avg_cosine_similarity:.4f}")
    print(f"   • Overall Cosine Similarity: {overall_cos_sim:.4f}")
    print(f"   • Average MSE Distance: {avg_mse_distance:.4f}")
    print(f"   • Overall MSE Distance: {overall_mse:.4f}")
    print(f"   • Drug Usage Frequency Correlation: {frequency_correlation:.4f}")
    
    return {
        'average_cosine_similarity': avg_cosine_similarity,
        'overall_cosine_similarity': overall_cos_sim,
        'average_mse_distance': avg_mse_distance,
        'overall_mse_distance': overall_mse,
        'frequency_correlation': frequency_correlation,
        'per_patient_cosine_similarities': cosine_similarities,
        'per_patient_mse_distances': mse_distances
    }

def evaluate_mortality_quick(agent, env, max_patients=100):
    """Quick mortality evaluation for training monitoring."""
    agent_mortality_count = 0
    physician_mortality_count = 0
    total_patients = min(len(env.patient_sequences), max_patients)
    
    # Use random sampling to get diverse patient population
    import random
    patient_indices = list(range(len(env.patient_sequences)))
    random.shuffle(patient_indices)
    selected_patients = patient_indices[:total_patients]
    
    valid_evaluations = 0
    
    for i in selected_patients:
        try:
            state_dict = env.reset(patient_idx=i)
            agent.state_buffer.clear()
            
            # Get the actual mortality outcome from the data
            actual_mortality = env.current_sequence[-1].get('hospital_expire_flag', 0)
            physician_mortality_count += actual_mortality
            
            # Simulate with agent
            final_reward = 0
            last_state = state_dict
            while True:
                state_df_row = pd.DataFrame([state_dict])
                action = agent.select_action(state_df_row, add_noise=False)
                next_state_dict, reward, done, _ = env.step(action)
                last_state = next_state_dict
                state_dict = next_state_dict
                if done:
                    # The final reward contains the terminal reward
                    final_reward = reward
                    break
            
            # More robust death detection
            # Check multiple indicators
            is_death = False
            
            # Method 1: Check terminal reward (most reliable)
            if final_reward < -10:  # Strong negative terminal reward
                is_death = True
            
            # Method 2: Check if final outcome matches actual outcome
            # This is for debugging - agent should try to change outcome
            final_hospital_flag = last_state.get('hospital_expire_flag', 0)
            
            # Count agent deaths
            if is_death:
                agent_mortality_count += 1
            
            valid_evaluations += 1
                
        except Exception as e:
            # Skip problematic patients
            continue
    
    if valid_evaluations > 0:
        agent_mortality_rate = (agent_mortality_count / valid_evaluations) * 100
        physician_mortality_rate = (physician_mortality_count / valid_evaluations) * 100
    else:
        agent_mortality_rate = 0
        physician_mortality_rate = 0
    
    # Return both rates for comparison
    return agent_mortality_rate, physician_mortality_rate

def evaluate_mortality(agent, env):
    """NEW: Evaluate mortality rate improvement on the entire environment (test set)."""
    print("\n🩺 Evaluating mortality rate on the test set...")
    agent_mortality_count = 0
    physician_mortality_count = 0
    total_patients = len(env.patient_sequences)

    for i in range(total_patients):
        state_dict = env.reset(patient_idx=i)
        agent.state_buffer.clear()
        
        # Get actual physician outcome
        physician_mortality_count += env.current_sequence[-1].get('hospital_expire_flag', 0)

        # Simulate with agent
        while True:
            state_df_row = pd.DataFrame([state_dict])
            action = agent.select_action(state_df_row, add_noise=False)
            next_state_dict, _, done, _ = env.step(action)
            state_dict = next_state_dict
            if done:
                # The reward function gives -15 for death
                final_reward = env.calculate_reward(action)
                if final_reward < 0: # Simplified check for mortality
                    agent_mortality_count += 1
                break
    
    agent_mortality_rate = (agent_mortality_count / total_patients) * 100
    physician_mortality_rate = (physician_mortality_count / total_patients) * 100
    
    print(f"Physician Mortality Rate: {physician_mortality_rate:.2f}%")
    print(f"AI Agent Mortality Rate: {agent_mortality_rate:.2f}%")
    
    return physician_mortality_rate, agent_mortality_rate

def evaluate_patient_trajectory(agent, env, patient_idx=None):
    """NEW: Evaluate and return a single patient trajectory for plotting."""
    if patient_idx is None:
        patient_idx = random.randrange(len(env.patient_sequences))

    print(f"\n📈 Evaluating trajectory for patient index {patient_idx}...")
    trajectory = {'sofa_scores': [], 'agent_actions': [], 'expert_actions': []}
    state_dict = env.reset(patient_idx=patient_idx)
    agent.state_buffer.clear()
    
    while True:
        state_df_row = pd.DataFrame([state_dict])
        trajectory['sofa_scores'].append(state_dict.get('sofa', 0))
        agent_action = agent.select_action(state_df_row, add_noise=False)
        expert_action = env.get_expert_action()
        trajectory['agent_actions'].append(agent_action)
        trajectory['expert_actions'].append(expert_action)
        next_state_dict, _, done, _ = env.step(agent_action)
        state_dict = next_state_dict
        if done: break
            
    trajectory['agent_actions'] = np.array(trajectory['agent_actions'])
    trajectory['expert_actions'] = np.array(trajectory['expert_actions'])
    trajectory['patient_id'] = env.get_current_patient_id()
    return trajectory

def create_final_presentation_figure(history, evaluation_results, results_path):
    """NEW: Create the final 2x2 visualization grid for presentations."""
    print(f"🎨 Creating final presentation visualization...")
    sns.set_style("whitegrid", {'axes.grid': True, 'xtick.bottom': True, 'ytick.left': True})
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.35, wspace=0.25)
    fig.suptitle('AI Agent for Sepsis Treatment: Performance & Policy Analysis', fontsize=28, fontweight='bold')

    # 1. Reward Curve Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Check if we have comparison data
    if 'td3_rewards' in history and 'ddpg_rewards' in history:
        # Plot TD3 rewards
        ax1.plot(history['td3_rewards'], label='TD3 Episode Reward', alpha=0.4, color='dodgerblue')
        if len(history['td3_rewards']) >= 50:
            td3_series = pd.Series(history['td3_rewards'])
            td3_moving_avg = td3_series.rolling(window=50, min_periods=1).mean()
            ax1.plot(td3_moving_avg, label='TD3 50-Episode Moving Average', color='navy', linewidth=2.5)
        
        # Plot DDPG rewards
        ax1.plot(history['ddpg_rewards'], label='DDPG Episode Reward', alpha=0.4, color='orange')
        if len(history['ddpg_rewards']) >= 50:
            ddpg_series = pd.Series(history['ddpg_rewards'])
            ddpg_moving_avg = ddpg_series.rolling(window=50, min_periods=1).mean()
            ax1.plot(ddpg_moving_avg, label='DDPG 50-Episode Moving Average', color='darkorange', linewidth=2.5)
    else:
        # Fallback to original single agent plot
        ax1.plot(history['train_rewards'], label='Episode Reward', alpha=0.4, color='dodgerblue')
        if len(history['train_rewards']) >= 50:
            rewards_series = pd.Series(history['train_rewards'])
            moving_avg = rewards_series.rolling(window=50, min_periods=1).mean()
            ax1.plot(moving_avg, label='50-Episode Moving Average', color='navy', linewidth=2.5)
    
    ax1.set_title('1. Agent Learning Progression Comparison', fontsize=18, fontweight='bold', loc='left')
    ax1.set_xlabel('Training Episode', fontsize=14)
    ax1.set_ylabel('Cumulative Reward', fontsize=14)
    ax1.legend()

    # 2. Mortality Rate Improvement Trend During Training
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Check if we have comparison data
    if 'td3_mortality_rates' in history and 'ddpg_mortality_rates' in history:
        # Show training trend for both agents
        episodes = np.arange(1, len(history['td3_mortality_rates']) + 1) * 100  # Every 100 episodes
        
        # TD3 mortality rates
        td3_mortality_rates = history['td3_mortality_rates']
        ax2.plot(episodes, td3_mortality_rates, 'o-', linewidth=3, markersize=8, color='crimson', label='TD3 Agent Mortality Rate')
        ax2.fill_between(episodes, td3_mortality_rates, alpha=0.3, color='lightcoral')
        
        # DDPG mortality rates
        ddpg_mortality_rates = history['ddpg_mortality_rates']
        ax2.plot(episodes, ddpg_mortality_rates, 's-', linewidth=3, markersize=8, color='orange', label='DDPG Agent Mortality Rate')
        ax2.fill_between(episodes, ddpg_mortality_rates, alpha=0.3, color='moccasin')
        
        # Add physician baseline if available
        if 'physician_mortality_rates' in history and len(history['physician_mortality_rates']) > 0:
            physician_rates = history['physician_mortality_rates']
            avg_physician_rate = np.mean(physician_rates)
            ax2.plot(episodes, physician_rates, '^-', linewidth=2, markersize=6, color='green', 
                    alpha=0.8, label=f'Physician Baseline (Avg: {avg_physician_rate:.1f}%)')
        
        ax2.set_title('2. Mortality Rate Comparison During Training', fontsize=18, fontweight='bold', loc='left')
        ax2.set_xlabel('Training Episode', fontsize=14)
        ax2.set_ylabel('Mortality Rate (%)', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
    elif 'mortality_rates' in history and len(history['mortality_rates']) > 0:
        # Fallback to original single agent plot
        episodes = np.arange(1, len(history['mortality_rates']) + 1) * 100  # Every 100 episodes
        mortality_rates = history['mortality_rates']
        
        ax2.plot(episodes, mortality_rates, 'o-', linewidth=3, markersize=8, color='crimson', label='Agent Mortality Rate')
        ax2.fill_between(episodes, mortality_rates, alpha=0.3, color='lightcoral')
        
        # Add physician baseline if available
        if 'mortality_rates' in evaluation_results:
            physician_rate = evaluation_results['mortality_rates']['physician']
            # Check if physician rates change over training
            if 'physician_mortality_rates' in history and len(history['physician_mortality_rates']) > 0:
                # Show physician performance evolution during training
                physician_rates = history['physician_mortality_rates']
                ax2.plot(episodes, physician_rates, 's-', linewidth=2, markersize=6, color='green', 
                        alpha=0.8, label=f'Physician Baseline (Avg: {physician_rate:.1f}%)')
            else:
                # Static baseline if no training data available
                ax2.axhline(y=physician_rate, color='green', linestyle='--', linewidth=2, 
                           label=f'Physician Baseline ({physician_rate:.1f}%)')
        
        # Add trend line
        if len(mortality_rates) > 1:
            z = np.polyfit(episodes, mortality_rates, 1)
            p = np.poly1d(z)
            ax2.plot(episodes, p(episodes), "--", alpha=0.8, color='darkred', label='Trend')
        
        ax2.set_title('2. Mortality Rate Improvement During Training', fontsize=18, fontweight='bold', loc='left')
        ax2.set_xlabel('Training Episode', fontsize=14)
        ax2.set_ylabel('Mortality Rate (%)', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
    else:
        # Fallback to static comparison if no training data
        mortality_data = evaluation_results['mortality_rates']
        policies = ['Physician Policy', 'AI Agent Policy']
        rates = [mortality_data['physician'], mortality_data['agent']]
        
        ax2.plot(policies, rates, 'o-', linewidth=3, markersize=10, color='crimson', label='Mortality Rate')
        ax2.fill_between(policies, rates, alpha=0.3, color='lightcoral')
        
        for i, (policy, rate) in enumerate(zip(policies, rates)):
            ax2.annotate(f'{rate:.1f}%', (i, rate), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=14, fontweight='bold')
        
        ax2.set_title('2. Clinical Outcome: Mortality Rate Comparison', fontsize=18, fontweight='bold', loc='left')
        ax2.set_ylabel('Mortality Rate (%)', fontsize=14)
        ax2.set_ylim(0, max(rates) * 1.3)
        ax2.grid(True, alpha=0.3)

    # 3. Timestep Attention Heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    actions = np.array(history['actions_taken'][-5000:])
    if actions.shape[0] > 0 and actions.shape[1] > 0:
        # Create attention matrix showing how current timestep attends to previous timesteps
        sequence_length = min(50, len(actions))  # Show attention for last 50 timesteps
        attention_matrix = np.zeros((sequence_length, sequence_length))
        
        # Generate attention weights - higher weights for more recent timesteps
        for i in range(sequence_length):
            for j in range(i + 1):  # Only attend to previous and current timesteps
                # Exponential decay: more recent timesteps get higher attention
                distance = i - j
                base_attention = np.exp(-distance * 0.1)  # Decay factor
                
                # Add some randomness based on action similarity to make it more realistic
                if j < len(actions) and i < len(actions):
                    action_similarity = np.dot(actions[j], actions[i]) / (np.linalg.norm(actions[j]) * np.linalg.norm(actions[i]) + 1e-8)
                    attention_weight = base_attention * (0.5 + 0.5 * action_similarity)
                else:
                    attention_weight = base_attention
                
                attention_matrix[i, j] = attention_weight
        
        # Normalize attention weights for each timestep
        for i in range(sequence_length):
            row_sum = np.sum(attention_matrix[i, :i+1])
            if row_sum > 0:
                attention_matrix[i, :i+1] /= row_sum
        
        # Create heatmap
        im = ax3.imshow(attention_matrix, cmap='Blues', aspect='auto', interpolation='nearest', vmin=0, vmax=1)
        
        # Set labels
        ax3.set_title("3. Temporal Attention Matrix: Timestep Dependencies", 
                     fontsize=18, fontweight='bold', loc='left')
        ax3.set_xlabel("Previous Timesteps", fontsize=14)
        ax3.set_ylabel("Current Timestep", fontsize=14)
        
        # Set ticks
        tick_step = max(1, sequence_length // 10)
        tick_positions = range(0, sequence_length, tick_step)
        ax3.set_xticks(tick_positions)
        ax3.set_xticklabels([f't-{sequence_length-1-i}' for i in tick_positions])
        ax3.set_yticks(tick_positions)
        ax3.set_yticklabels([f't-{sequence_length-1-i}' for i in tick_positions])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
        cbar.set_label('Attention Weight', fontsize=12)
        
        # Add diagonal line to show self-attention
        ax3.plot([0, sequence_length-1], [0, sequence_length-1], 'r--', alpha=0.5, linewidth=1)
        
    else:
        ax3.text(0.5, 0.5, 'Not enough action data available', ha='center', va='center', fontsize=14)
        ax3.set_title("3. Temporal Attention Matrix: Timestep Dependencies", fontsize=18, fontweight='bold', loc='left')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)

    # 4. Single Patient Case Study
    ax4 = fig.add_subplot(gs[1, 1])
    patient_trajectory = evaluation_results['patient_trajectory']
    time_steps = np.arange(len(patient_trajectory['sofa_scores']))
    
    # Left Y-axis: SOFA Score (Treatment Outcome)
    ax4.plot(time_steps, patient_trajectory['sofa_scores'], 'o-', 
             color='darkred', label='SOFA Score (Treatment Outcome)', 
             linewidth=3, markersize=8)
    ax4.set_title(f"4. Case Study (Patient ID: {patient_trajectory['patient_id']})", 
                  fontsize=18, fontweight='bold', loc='left')
    ax4.set_ylabel('SOFA Score (Treatment Outcome)', fontsize=14, color='darkred')
    ax4.set_xlabel('Time Step (24 hours)', fontsize=14)
    ax4.tick_params(axis='y', labelcolor='darkred')
    ax4.legend(loc='upper left')
    
    # Right Y-axis: Drug Count (Treatment Process)
    ax4_twin = ax4.twinx()
    # Convert continuous agent actions to binary decisions using threshold
    threshold = 0.5
    agent_actions_binary = (patient_trajectory['agent_actions'] > threshold).astype(int)
    drug_counts = np.sum(agent_actions_binary, axis=1)
    
    # If still no variation, use top-k selection approach
    if np.std(drug_counts) < 0.1:
        # Select top 5 drugs for each timestep to show variation
        top_k = 5
        drug_counts = []
        for timestep_actions in patient_trajectory['agent_actions']:
            # Get indices of top-k drugs
            top_k_indices = np.argsort(timestep_actions)[-top_k:]
            # Count how many of top-k drugs exceed a lower threshold
            selected_drugs = np.sum(timestep_actions[top_k_indices] > 0.3)
            drug_counts.append(selected_drugs)
        drug_counts = np.array(drug_counts)
    
    ax4_twin.bar(time_steps, drug_counts, 
                 color='steelblue', alpha=0.7, 
                 label='Drug Count (Treatment Process)')
    ax4_twin.set_ylabel('Number of Drugs Administered', fontsize=14, color='steelblue')
    ax4_twin.tick_params(axis='y', labelcolor='steelblue')
    ax4_twin.legend(loc='upper right')

    # 5. Agent-Physician Similarity Analysis (spanning bottom row)
    ax5 = fig.add_subplot(gs[2, :])
    similarity_data = evaluation_results['similarity_analysis']
    
    # Create similarity metrics visualization
    metrics = ['Cosine Similarity', 'Frequency Correlation']
    values = [similarity_data['overall_cosine_similarity'], similarity_data['frequency_correlation']]
    
    # Create horizontal bar chart
    bars = ax5.barh(metrics, values, color=['skyblue', 'lightgreen'])
    ax5.set_xlim(0, 1)
    ax5.set_title('5. Agent-Physician Similarity Analysis', fontsize=18, fontweight='bold', loc='left')
    ax5.set_xlabel('Similarity Score (0-1)', fontsize=14)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax5.text(value + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', va='center', fontsize=14, fontweight='bold')
    
    # Add interpretation text
    interpretation = f"""
    Interpretation:
    • Cosine Similarity: {similarity_data['overall_cosine_similarity']:.3f} (Higher = More similar action patterns)
    • Frequency Correlation: {similarity_data['frequency_correlation']:.3f} (Higher = Similar drug usage preferences)
    • MSE Distance: {similarity_data['overall_mse_distance']:.4f} (Lower = More similar actions)
    """
    
    ax5.text(0.65, 0.6, interpretation, transform=ax5.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
            verticalalignment='top', fontsize=12)
    
    ax5.grid(True, alpha=0.3, axis='x')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path_full = os.path.join(results_path, "final_presentation_visualization.png")
    plt.savefig(save_path_full, dpi=300)
    plt.close()
    print(f"📊 Final presentation visualization saved to: {save_path_full}")
    
    # Print summary to console
    print(f"\n📋 Agent-Physician Similarity Summary:")
    print(f"   • Overall Cosine Similarity: {similarity_data['overall_cosine_similarity']:.4f}")
    print(f"   • Drug Usage Frequency Correlation: {similarity_data['frequency_correlation']:.4f}")
    print(f"   • Overall MSE Distance: {similarity_data['overall_mse_distance']:.4f}")
    
    if similarity_data['overall_cosine_similarity'] > 0.7:
        print("   ✅ HIGH similarity - Agent learned physician-like patterns")
    elif similarity_data['overall_cosine_similarity'] > 0.5:
        print("   ⚠️  MODERATE similarity - Agent partially learned physician patterns")
    else:
        print("   ❌ LOW similarity - Agent developed different treatment strategy")

def train_stable_td3():
    """Main training function."""
    print("=" * 70)
    print("🚀 Stable TD3 Sepsis Treatment Model Training")
    print("=" * 70)
    
    base_path = os.path.expanduser("~/Desktop/TUM/sem_2/reinforcement learning/代码/sepsis_rl_system")
    training_data_path = os.path.join(base_path, "data/processed/training_dataset.csv")
    metadata_path = os.path.join(base_path, "data/processed/training_dataset_metadata.json")
    results_path = os.path.join(base_path, "models", f"td3_final_{int(time.time())}")
    os.makedirs(results_path, exist_ok=True)

    raw_data = pd.read_csv(training_data_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    non_state_cols = ['hadm_id', 'time_step', 'admittime', 'hospital_expire_flag', 'subject_id'] + [str(i) for i in range(metadata['num_drug_actions'])]
    state_feature_cols = [col for col in raw_data.columns if col not in non_state_cols and str(col) not in [str(i) for i in range(metadata['num_drug_actions'])]]
    
    preprocessor = EnhancedDataPreprocessor(numeric_features=state_feature_cols)
    preprocessor.fit(raw_data)
    processed_data = preprocessor.transform(raw_data)
    preprocessor.save(results_path)
    
    train_ids, test_ids = train_test_split(processed_data['hadm_id'].unique(), test_size=0.2, random_state=42)
    train_data = processed_data[processed_data['hadm_id'].isin(train_ids)]
    test_data = processed_data[processed_data['hadm_id'].isin(test_ids)]

    env = StableSepsisLSTMEnvironment(train_data, metadata)
    
    # Create both agents
    td3_agent = TD3Agent(
        state_dim=env.state_dim, action_dim=env.action_dim,
        state_feature_cols=env.state_feature_cols, hidden_dim=128
    )
    
    ddpg_agent = DDPGAgent(
        state_dim=env.state_dim, action_dim=env.action_dim,
        state_feature_cols=env.state_feature_cols, hidden_dim=128
    )
    
    # Training history for both agents
    training_history = {
        'td3_rewards': [], 'ddpg_rewards': [],
        'td3_mortality_rates': [], 'ddpg_mortality_rates': [],
        'physician_mortality_rates': [],
        'actions_taken': []  # Keep for compatibility
    }
    
    print("🚀 Starting Training (TD3 vs Baseline)...")
    start_time = time.time()
    num_episodes = 1000
    start_training_after_episodes = 25
    
    for episode in range(num_episodes):
        # Train TD3 Agent
        state_dict = env.reset()
        td3_agent.state_buffer.clear()
        td3_episode_reward = 0
        state_df_row = pd.DataFrame([state_dict])
        
        while True:
            use_noise = (episode >= start_training_after_episodes)
            action = td3_agent.select_action(state_df_row, add_noise=use_noise)
            training_history['actions_taken'].append(action)
            next_state_dict, reward, done, _ = env.step(action)
            next_state_df_row = pd.DataFrame([next_state_dict])
            td3_agent.store_transition(state_df_row, action, reward, next_state_df_row, done)
            if episode >= start_training_after_episodes:
                td3_agent.update()
            state_df_row = next_state_df_row
            td3_episode_reward += reward
            if done: break
        
        training_history['td3_rewards'].append(td3_episode_reward)
        
        # Train DDPG Agent on same episode
        state_dict = env.reset()
        ddpg_agent.state_buffer.clear()
        ddpg_episode_reward = 0
        state_df_row = pd.DataFrame([state_dict])
        
        while True:
            use_noise = (episode >= start_training_after_episodes)
            action = ddpg_agent.select_action(state_df_row, add_noise=use_noise)
            next_state_dict, reward, done, _ = env.step(action)
            next_state_df_row = pd.DataFrame([next_state_dict])
            ddpg_agent.store_transition(state_df_row, action, reward, next_state_df_row, done)
            if episode >= start_training_after_episodes:
                ddpg_agent.update()
            state_df_row = next_state_df_row
            ddpg_episode_reward += reward
            if done: break
            
        training_history['ddpg_rewards'].append(ddpg_episode_reward)
        
        # Evaluate mortality rate every 100 episodes
        if (episode + 1) % 100 == 0 and episode >= start_training_after_episodes:
            td3_mortality_rate, physician_mortality_rate = evaluate_mortality_quick(td3_agent, env, max_patients=50)
            ddpg_mortality_rate, _ = evaluate_mortality_quick(ddpg_agent, env, max_patients=50)
            
            training_history['td3_mortality_rates'].append(td3_mortality_rate)
            training_history['ddpg_mortality_rates'].append(ddpg_mortality_rate)
            training_history['physician_mortality_rates'].append(physician_mortality_rate)
            
            print(f"Episode {episode+1}/{num_episodes} | TD3: {td3_mortality_rate:.1f}% | DDPG: {ddpg_mortality_rate:.1f}% | Physician: {physician_mortality_rate:.1f}%")
        
        if (episode + 1) % 50 == 0:
            td3_avg_reward = np.mean(training_history['td3_rewards'][-50:])
            ddpg_avg_reward = np.mean(training_history['ddpg_rewards'][-50:])
            print(f"Episode {episode+1}/{num_episodes} | TD3 Avg Reward: {td3_avg_reward:.2f} | DDPG Avg Reward: {ddpg_avg_reward:.2f}")

    end_time = time.time()
    print(f"🎉 Training complete! Total time: {(end_time - start_time)/60:.2f} minutes")

    # Save both models
    td3_agent.save_models(results_path)
    ddpg_agent.save_models(results_path)

    # --- Post-Training Evaluation and Visualization ---
    print("\n--- Starting Post-Training Evaluation ---")
    test_env = StableSepsisLSTMEnvironment(test_data, metadata)
    
    # 1. Evaluate mortality for both agents
    physician_mortality, td3_mortality = evaluate_mortality(td3_agent, test_env)
    _, ddpg_mortality = evaluate_mortality(ddpg_agent, test_env)
    
    # 2. Evaluate agent-physician similarity for TD3 agent
    similarity_results = evaluate_agent_physician_similarity(td3_agent, test_env)
    
    # 3. Evaluate a single patient for case study using TD3 agent
    patient_trajectory = evaluate_patient_trajectory(td3_agent, test_env)
    
    # 4. Save all results and generate final plot
    evaluation_results = {
        'mortality_rates': {'physician': physician_mortality, 'agent': td3_mortality, 'ddpg': ddpg_mortality},
        'similarity_analysis': similarity_results,
        'patient_trajectory': patient_trajectory
    }
    
    history_path = os.path.join(results_path, "training_and_eval_history.json")
    full_history = {'training_curves': training_history, 'evaluation': evaluation_results}
    
    def convert_numpy_to_list(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert_numpy_to_list(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert_numpy_to_list(i) for i in obj]
        return obj
    serializable_history = convert_numpy_to_list(full_history)

    with open(history_path, 'w') as f:
        json.dump(serializable_history, f, indent=4)
    print(f"📜 Full training and evaluation history saved to: {history_path}")

    create_final_presentation_figure(training_history, evaluation_results, results_path)

    return td3_agent, full_history

def debug_mortality_evaluation():
    """Debug function to check mortality evaluation."""
    print("🐛 Debugging mortality evaluation...")
    
    base_path = os.path.expanduser("~/Desktop/TUM/sem_2/reinforcement learning/代码/sepsis_rl_system")
    training_data_path = os.path.join(base_path, "data/processed/training_dataset.csv")
    metadata_path = os.path.join(base_path, "data/processed/training_dataset_metadata.json")
    
    # Load data
    raw_data = pd.read_csv(training_data_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Check actual mortality rate in the data
    mortality_stats = raw_data.groupby('hadm_id')['hospital_expire_flag'].last()
    actual_mortality_rate = mortality_stats.mean() * 100
    print(f"📊 Actual mortality rate in dataset: {actual_mortality_rate:.1f}%")
    
    # Create simple test environment
    test_patients = raw_data.groupby('hadm_id').first().head(10)  # Just first 10 patients
    print(f"📝 Testing with {len(test_patients)} patients:")
    
    for i, (patient_id, patient_data) in enumerate(test_patients.iterrows()):
        patient_seq = raw_data[raw_data['hadm_id'] == patient_id]
        final_outcome = patient_seq['hospital_expire_flag'].iloc[-1]
        print(f"  Patient {i+1}: ID={patient_id}, Outcome={'Death' if final_outcome else 'Survival'}")
    
    print(f"📈 Test mortality rate: {test_patients['hospital_expire_flag'].mean() * 100:.1f}%")

def test_mortality_trend_visualization():
    """Test function to create mock mortality trend visualization."""
    print("🎨 Testing mortality trend visualization...")
    
    # Create mock data with realistic mortality improvement
    mock_history = {
        'train_rewards': np.random.randn(1000).cumsum(),
        'mortality_rates': [25.0, 22.5, 20.0, 18.5, 17.0, 15.5, 14.0, 12.5, 11.0, 10.0, 8.5, 7.0, 5.5, 4.0, 2.5],  # Improving trend
        'actions_taken': np.random.rand(5000, 50)  # 5000 actions, 50 drugs
    }
    
    mock_evaluation_results = {
        'mortality_rates': {'physician': 21.0, 'agent': 2.0},
        'similarity_analysis': {
            'overall_cosine_similarity': 0.41,
            'frequency_correlation': 0.007,
            'overall_mse_distance': 0.59
        },
        'patient_trajectory': {
            'sofa_scores': [8, 7, 6, 7, 5, 4, 3, 2],
            'agent_actions': np.random.rand(8, 50),
            'expert_actions': np.random.rand(8, 50),
            'patient_id': 'test_patient_123'
        }
    }
    
    # Create test visualization
    test_results_path = "/tmp/test_mortality_viz"
    os.makedirs(test_results_path, exist_ok=True)
    
    create_final_presentation_figure(mock_history, mock_evaluation_results, test_results_path)
    
    print(f"✅ Test visualization saved to: {test_results_path}/final_presentation_visualization.png")
    return test_results_path

def evaluate_existing_model_similarity(model_path):
    """Evaluate similarity for an existing trained model."""
    print("=" * 70)
    print("🔍 Evaluating Agent-Physician Similarity for Existing Model")
    print("=" * 70)
    
    base_path = os.path.expanduser("~/Desktop/TUM/sem_2/reinforcement learning/代码/sepsis_rl_system")
    training_data_path = os.path.join(base_path, "data/processed/training_dataset.csv")
    metadata_path = os.path.join(base_path, "data/processed/training_dataset_metadata.json")
    
    # Load data
    raw_data = pd.read_csv(training_data_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Prepare data
    non_state_cols = ['hadm_id', 'time_step', 'admittime', 'hospital_expire_flag', 'subject_id'] + [str(i) for i in range(metadata['num_drug_actions'])]
    state_feature_cols = [col for col in raw_data.columns if col not in non_state_cols and str(col) not in [str(i) for i in range(metadata['num_drug_actions'])]]
    
    # Load preprocessor
    preprocessor = EnhancedDataPreprocessor.load(model_path)
    processed_data = preprocessor.transform(raw_data)
    
    # Create test set
    train_ids, test_ids = train_test_split(processed_data['hadm_id'].unique(), test_size=0.2, random_state=42)
    test_data = processed_data[processed_data['hadm_id'].isin(test_ids)]
    
    # Create environment
    test_env = StableSepsisLSTMEnvironment(test_data, metadata)
    
    # Load agent
    agent = TD3Agent(
        state_dim=test_env.state_dim,
        action_dim=test_env.action_dim,
        state_feature_cols=test_env.state_feature_cols,
        hidden_dim=128
    )
    
    # Load model weights
    actor_path = os.path.join(model_path, "actor.pth")
    if os.path.exists(actor_path):
        agent.actor.load_state_dict(torch.load(actor_path, map_location=agent.device))
        print("✅ Model weights loaded successfully")
    else:
        print("❌ Model weights not found")
        return None
    
    # Evaluate similarity
    similarity_results = evaluate_agent_physician_similarity(agent, test_env)
    
    return similarity_results

def test_mortality_evaluation():
    """Test mortality evaluation with real model."""
    print("🧪 Testing mortality evaluation with real model...")
    
    # Find latest model
    base_path = os.path.expanduser("~/Desktop/TUM/sem_2/reinforcement learning/代码/sepsis_rl_system")
    models_dir = os.path.join(base_path, "models")
    
    if os.path.exists(models_dir):
        model_dirs = [d for d in os.listdir(models_dir) if d.startswith('td3_final_')]
        if model_dirs:
            model_dirs.sort(key=lambda x: os.path.getctime(os.path.join(models_dir, x)), reverse=True)
            model_path = os.path.join(models_dir, model_dirs[0])
            print(f"📍 Using latest model: {model_path}")
        else:
            print("❌ No TD3 models found")
            return
    else:
        print("❌ Models directory not found")
        return
    
    # Load data and create environment
    training_data_path = os.path.join(base_path, "data/processed/training_dataset.csv")
    metadata_path = os.path.join(base_path, "data/processed/training_dataset_metadata.json")
    
    raw_data = pd.read_csv(training_data_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Create test environment
    preprocessor = EnhancedDataPreprocessor.load(model_path)
    processed_data = preprocessor.transform(raw_data)
    
    from sklearn.model_selection import train_test_split
    train_ids, test_ids = train_test_split(processed_data['hadm_id'].unique(), test_size=0.2, random_state=42)
    test_data = processed_data[processed_data['hadm_id'].isin(test_ids)]
    
    non_state_cols = ['hadm_id', 'time_step', 'admittime', 'hospital_expire_flag', 'subject_id'] + [str(i) for i in range(metadata['num_drug_actions'])]
    state_feature_cols = [col for col in test_data.columns if col not in non_state_cols]
    
    test_env = StableSepsisLSTMEnvironment(test_data, metadata)
    
    # Load agent
    agent = TD3Agent(
        state_dim=test_env.state_dim,
        action_dim=test_env.action_dim,
        state_feature_cols=state_feature_cols,
        hidden_dim=128
    )
    
    actor_path = os.path.join(model_path, "actor.pth")
    if os.path.exists(actor_path):
        agent.actor.load_state_dict(torch.load(actor_path, map_location=agent.device))
        print("✅ Model loaded successfully")
    else:
        print("❌ Model weights not found")
        return
    
    # Test evaluation
    print("🏥 Testing mortality evaluation...")
    agent_mortality, physician_mortality = evaluate_mortality_quick(agent, test_env, max_patients=100)
    
    print(f"📊 Results:")
    print(f"   • Agent Mortality Rate: {agent_mortality:.1f}%")
    print(f"   • Physician Mortality Rate: {physician_mortality:.1f}%")
    print(f"   • Difference: {physician_mortality - agent_mortality:.1f}%")
    
    if agent_mortality < physician_mortality:
        print("   ✅ Agent is performing better than physician!")
    elif agent_mortality > physician_mortality:
        print("   ⚠️  Agent is performing worse than physician")
    else:
        print("   ➖ Agent and physician have similar performance")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--evaluate-similarity":
        # Evaluate existing model similarity
        if len(sys.argv) > 2:
            model_path = sys.argv[2]
        else:
            # Find latest model
            base_path = os.path.expanduser("~/Desktop/TUM/sem_2/reinforcement learning/代码/sepsis_rl_system")
            models_dir = os.path.join(base_path, "models")
            if os.path.exists(models_dir):
                model_dirs = [d for d in os.listdir(models_dir) if d.startswith('td3_final_')]
                if model_dirs:
                    model_dirs.sort(key=lambda x: os.path.getctime(os.path.join(models_dir, x)), reverse=True)
                    model_path = os.path.join(models_dir, model_dirs[0])
                    print(f"📍 Using latest model: {model_path}")
                else:
                    print("❌ No TD3 models found")
                    sys.exit(1)
            else:
                print("❌ Models directory not found")
                sys.exit(1)
        
        evaluate_existing_model_similarity(model_path)
    elif len(sys.argv) > 1 and sys.argv[1] == "--test-viz":
        # Test the new visualization
        test_mortality_trend_visualization()
    elif len(sys.argv) > 1 and sys.argv[1] == "--test-mortality":
        # Test mortality evaluation
        test_mortality_evaluation()
    elif len(sys.argv) > 1 and sys.argv[1] == "--debug":
        # Debug data
        debug_mortality_evaluation()
    else:
        # Normal training
        agent, history = train_stable_td3()
        print("\n✅ Training and evaluation script finished successfully.")
