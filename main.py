import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Dict, Tuple, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lib.constants import SimulationConfig, HIDDEN_DIM, LOOKBACK_WINDOW, GRU_HIDDEN_DIM, GRU_HIDDEN_LAYERS, GRU_OUTPUT_SIZE, MAX_ACTION
from models.gru import GRUNet
from models.actor import Actor
from models.critic import Critic
from replay_buffer.replay import ReplayBuffer
from RL_env.trading_env import TradingEnvironment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    def __init__(self, config: SimulationConfig, mode: str = "reg", model_path: str = None):
        self.config = config
        self.mode = mode
        self.device = device
        
        # Determine GRU configuration based on mode
        # Input size is 1 (S) because we use S_history
        input_size = 1 
        hidden_size = GRU_HIDDEN_DIM
        num_layers = GRU_HIDDEN_LAYERS
        
        if mode == "reg":
            output_size = 1
            head_type = "reg"
            self.gru_output_dim = 1
        elif mode == "prob":
            output_size = len(config.theta_values)
            head_type = "prob"
            self.gru_output_dim = output_size
        elif mode == "hid":
            output_size = None  # Return hidden state
            head_type = "hid"
            self.gru_output_dim = hidden_size
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Initialize GRU
        self.gru = GRUNet(input_size, hidden_size, num_layers, output_size=output_size, head_type=head_type).to(device)
        
        # Load pre-trained weights for reg/prob modes
        if mode in ["reg", "prob"]:
            if model_path and os.path.exists(model_path):
                print(f"Loading GRU model from {model_path}")
                self.gru.load_state_dict(torch.load(model_path, map_location=device))
            else:
                print(f"Warning: Model path {model_path} not found for mode {mode}. Using random weights.")
            
            # Freeze GRU for reg/prob
            for param in self.gru.parameters():
                param.requires_grad = False
            self.gru.eval()
        else:
            # For hid mode, we train the GRU
            self.gru.train()

        # State dimension for Actor/Critic
        # S_norm (1) + I_norm (1) + GRU_output (gru_output_dim)
        self.state_dim = 2 + self.gru_output_dim
        
        # Initialize Actor
        self.actor = Actor(self.state_dim, action_dim=1).to(device)
        self.actor_target = Actor(self.state_dim, action_dim=1).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Initialize Critic
        self.critic = Critic(self.state_dim, action_dim=1).to(device)
        self.critic_target = Critic(self.state_dim, action_dim=1).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        # For hid mode, include GRU parameters in Actor optimizer
        if mode == "hid":
            self.actor_optimizer = optim.Adam(
                list(self.actor.parameters()) + list(self.gru.parameters()),
                lr=config.lr
            )
        else:
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr)
            
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.lr)
        
        self.buffer = ReplayBuffer(capacity=100000)
        self.epsilon = 1.0
        self.epsilon_decay = config.n_episodes // 2  # Decay over half episodes roughly
        self.epsilon_min = 0.01

    def _normalize(self, S: float, I: float) -> Tuple[float, float]:
        S_norm = (S - self.config.mu_inv) / 0.5  # Approximation based on sigma
        I_norm = 2 * (I - self.config.I_min) / (self.config.I_max - self.config.I_min) - 1
        return S_norm, I_norm

    def _build_state(self, obs: Dict) -> torch.Tensor:
        S_norm, I_norm = self._normalize(obs['S'], obs['I'])
        
        # Encode history with GRU
        # obs['S_history'] is numpy array, convert to tensor
        # Shape: (seq_len,) -> (1, seq_len, 1)
        history = torch.FloatTensor(obs['S_history']).unsqueeze(0).unsqueeze(-1).to(device)
        
        if self.mode in ["reg", "prob"]:
            with torch.no_grad():
                gru_out = self.gru(history).squeeze(0) # (output_size,)
        else:
            # For hid mode, we need gradients
            gru_out = self.gru(history).squeeze(0) # (hidden_size,)
            
        # Build complete state
        state = torch.cat([
            torch.tensor([S_norm, I_norm], dtype=torch.float32, device=device),
            gru_out
        ])
        return state

    def select_action(self, obs: Dict, explore: bool = True) -> float:
        state = self._build_state(obs).unsqueeze(0) # Batch dim
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0, 0]
        self.actor.train()
        
        if explore:
            noise = np.random.normal(0, self.epsilon)
            action = np.clip(action + noise, -1.0, 1.0) # Action is normalized in [-1, 1]
            
        return action

    def update(self, batch_size: int):
        if len(self.buffer) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        # Process batch states
        # Note: This effectively runs GRU forward pass for each item in batch again.
        # Ideally we might store the GRU output in buffer for reg/prob, but for hid we MUST run it here to backprop.
        # To keep it simple and consistent, we rebuild state from observations.
        # However, ReplayBuffer stores 'state' which are Dicts from env.
        
        state_tensors = torch.stack([self._build_state(s) for s in states])
        next_state_tensors = torch.stack([self._build_state(s) for s in next_states])
        
        actions = torch.FloatTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        # Update Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_state_tensors)
            target_Q = self.critic_target(next_state_tensors, next_actions)
            target_Q = rewards + self.config.gamma * (1 - dones) * target_Q
            
        current_Q = self.critic(state_tensors.detach(), actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor
        # For hid mode, this also updates GRU
        actor_loss = -self.critic(state_tensors, self.actor(state_tensors)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
            
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

    def decay_epsilon(self, episode: int):
        self.epsilon = max(self.epsilon_min, 1.0 / (1 + episode / self.epsilon_decay))
        
    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'gru': self.gru.state_dict()
        }, path)
        
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.gru.load_state_dict(checkpoint['gru'])

def train(mode: str, episodes: int = 1000):
    config = SimulationConfig()
    
    # Path to pre-trained GRU models
    model_path = None
    if mode == "reg":
        model_path = os.path.join("models", "checkpoints", "best_gru_model_reg.pth")
    elif mode == "prob":
        model_path = os.path.join("models", "checkpoints", "best_gru_model_prob.pth")
        
    agent = DDPGAgent(config, mode=mode, model_path=model_path)
    env = TradingEnvironment(config)
    
    print(f"Starting training in {mode} mode for {episodes} episodes...")
    
    rewards_history = []
    
    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done = env.step(action)
            
            agent.buffer.push(obs, action, reward, next_obs, done)
            agent.update(config.batch_size)
            
            obs = next_obs
            episode_reward += reward
            
        agent.decay_epsilon(episode)
        rewards_history.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode+1}/{episodes}, Reward: {episode_reward:.2f}, Avg Reward (10): {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
            
        if (episode + 1) % 100 == 0:
            save_path = f"models/checkpoints/ddpg_{mode}_ep{episode+1}.pth"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            agent.save(save_path)
            
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="reg", choices=["reg", "prob", "hid"], help="DDPG mode: reg, prob, or hid")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to train")
    args = parser.parse_args()
    
    train(args.mode, args.episodes)
