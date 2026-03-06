"""
Réplication de l'étude DDPG pour le trading avec régimes Markoviens
Implémente: hid-DDPG, prob-DDPG, reg-DDPG
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, List, Dict

# Configuration GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 1. PARAMÈTRES DE SIMULATION (Table 1)
# =============================================================================
class SimulationConfig:
    # Inventaire
    I_max = 10
    I_min = -10
    
    # Trading
    transaction_cost = 0.05  # lambda
    dt = 0.2  # Delta t
    n_steps = 2000  # pas par épisode
    mu_inv = 1.0  # moyenne invariante
    
    # Régimes theta
    theta_values = [0.9, 1.0, 1.1]
    kappa_values = [3.0, 7.0]
    sigma_values = [0.1, 0.3]
    
    # Matrices de taux de transition
    A_theta = np.array([
        [-0.1, 0.05, 0.05],
        [0.05, -0.1, 0.05],
        [0.05, 0.05, -0.1]
    ])
    A_kappa = np.array([
        [-0.1, 0.1],
        [0.1, -0.1]
    ])
    A_sigma = np.array([
        [-0.1, 0.1],
        [0.1, -0.1]
    ])
    
    # Entraînement
    n_episodes = 10000
    batch_size = 512
    lr = 0.001
    tau = 0.001  # soft update
    gamma = 0.99  # facteur d'actualisation
    
    # GRU
    lookback_window = 10
    
    # Test
    n_test_episodes = 500

# =============================================================================
# 2. GÉNÉRATION DES DONNÉES - Processus OU avec régimes Markoviens
# =============================================================================
class MarkovChain:
    """Chaîne de Markov à temps continu"""
    def __init__(self, states: List[float], rate_matrix: np.ndarray, dt: float):
        self.states = np.array(states)
        self.rate_matrix = rate_matrix
        self.dt = dt
        self.n_states = len(states)
        self.current_idx = np.random.randint(self.n_states)
        
        # Matrice de transition pour dt
        self.trans_matrix = np.eye(self.n_states) + rate_matrix * dt
        
    def step(self) -> float:
        probs = self.trans_matrix[self.current_idx]
        probs = np.clip(probs, 0, 1)
        probs /= probs.sum()
        self.current_idx = np.random.choice(self.n_states, p=probs)
        return self.states[self.current_idx]
    
    def reset(self):
        self.current_idx = np.random.randint(self.n_states)
        return self.states[self.current_idx]

class OUProcess:
    """Processus Ornstein-Uhlenbeck avec régimes Markoviens"""
    def __init__(self, config: SimulationConfig, case: int = 3):
        self.config = config
        self.case = case
        self.dt = config.dt
        
        # Initialiser les chaînes de Markov selon le cas
        self.theta_mc = MarkovChain(config.theta_values, config.A_theta, self.dt)
        
        if case >= 2:
            self.kappa_mc = MarkovChain(config.kappa_values, config.A_kappa, self.dt)
        else:
            self.kappa_mc = None
            self.kappa_fixed = 5.0
            
        if case >= 3:
            self.sigma_mc = MarkovChain(config.sigma_values, config.A_sigma, self.dt)
        else:
            self.sigma_mc = None
            self.sigma_fixed = 0.2
            
        self.S = config.mu_inv
        
    def step(self) -> Tuple[float, float, float, float]:
        theta = self.theta_mc.step()
        kappa = self.kappa_mc.step() if self.kappa_mc else self.kappa_fixed
        sigma = self.sigma_mc.step() if self.sigma_mc else self.sigma_fixed
        
        dW = np.random.normal(0, np.sqrt(self.dt))
        dS = kappa * (theta - self.S) * self.dt + sigma * dW
        self.S += dS
        
        return self.S, theta, kappa, sigma
    
    def reset(self) -> float:
        self.theta_mc.reset()
        if self.kappa_mc:
            self.kappa_mc.reset()
        if self.sigma_mc:
            self.sigma_mc.reset()
        self.S = self.config.mu_inv
        return self.S

# =============================================================================
# 3. ENVIRONNEMENT DE TRADING
# =============================================================================
class TradingEnvironment:
    def __init__(self, config: SimulationConfig, case: int = 3):
        self.config = config
        self.ou_process = OUProcess(config, case)
        self.lookback = config.lookback_window
        
        self.S_history = deque(maxlen=self.lookback)
        self.I = 0  # Inventaire
        self.t = 0
        
    def reset(self) -> Dict:
        self.ou_process.reset()
        self.I = np.random.uniform(self.config.I_min, self.config.I_max)
        self.t = 0
        
        # Remplir l'historique
        self.S_history.clear()
        S = self.ou_process.S
        for _ in range(self.lookback):
            S, _, _, _ = self.ou_process.step()
            self.S_history.append(S)
            
        return self._get_state()
    
    def _get_state(self) -> Dict:
        return {
            'S': self.ou_process.S,
            'I': self.I,
            'S_history': np.array(self.S_history),
            't': self.t
        }
    
    def step(self, action: float) -> Tuple[Dict, float, bool]:
        """
        Action: changement d'inventaire delta_I dans [-1, 1] (normalisé)
        """
        # Dénormaliser l'action
        delta_I = action * (self.config.I_max - self.config.I_min) / 2
        
        # Appliquer les contraintes d'inventaire
        new_I = np.clip(self.I + delta_I, self.config.I_min, self.config.I_max)
        actual_delta = new_I - self.I
        
        # Avancer le processus
        S_old = self.ou_process.S
        S_new, theta, kappa, sigma = self.ou_process.step()
        self.S_history.append(S_new)
        
        # Calculer la récompense
        # r_t = I_t * (S_{t+1} - S_t) - lambda * |delta_I|
        pnl = self.I * (S_new - S_old)
        cost = self.config.transaction_cost * abs(actual_delta)
        reward = pnl - cost
        
        # Mettre à jour l'état
        self.I = new_I
        self.t += 1
        
        done = self.t >= self.config.n_steps
        
        return self._get_state(), reward, done

# =============================================================================
# 4. RÉSEAUX DE NEURONES
# =============================================================================
class GRUEncoder(nn.Module):
    """Encodeur GRU pour l'historique du signal"""
    def __init__(self, input_size: int = 1, hidden_size: int = 10, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_size = hidden_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 1)
        _, h_n = self.gru(x)
        return h_n[-1]  # (batch, hidden_size)

class Actor(nn.Module):
    """Réseau Actor pour DDPG"""
    def __init__(self, state_dim: int, hidden_dim: int = 20, n_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Tanh())  # Action dans [-1, 1]
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

class Critic(nn.Module):
    """Réseau Critic (Q-function) pour DDPG"""
    def __init__(self, state_dim: int, hidden_dim: int = 20, n_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(state_dim + 1, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.network(x)

# =============================================================================
# 5. AGENTS DDPG
# =============================================================================
class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class HidDDPG:
    """hid-DDPG: Approche une étape avec état caché GRU"""
    def __init__(self, config: SimulationConfig):
        self.config = config
        
        # GRU: 1 couche, 10 nœuds cachés
        self.gru = GRUEncoder(input_size=1, hidden_size=10, num_layers=1).to(device)
        
        # État: (S_t, I_t, o_t) -> dim = 2 + 10 = 12
        state_dim = 12
        
        # Actor-Critic: 4 couches, 20 nœuds cachés
        self.actor = Actor(state_dim, hidden_dim=20, n_layers=4).to(device)
        self.actor_target = Actor(state_dim, hidden_dim=20, n_layers=4).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, hidden_dim=20, n_layers=4).to(device)
        self.critic_target = Critic(state_dim, hidden_dim=20, n_layers=4).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimiseurs
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.gru.parameters()),
            lr=config.lr
        )
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.lr)
        
        self.buffer = ReplayBuffer()
        self.epsilon = 1.0
        self.epsilon_decay = 100
        self.epsilon_min = 0.01
        
    def _normalize(self, S: float, I: float) -> Tuple[float, float]:
        S_norm = (S - self.config.mu_inv) / 0.5  # Approximation
        I_norm = 2 * (I - self.config.I_min) / (self.config.I_max - self.config.I_min) - 1
        return S_norm, I_norm
    
    def _build_state(self, obs: Dict) -> torch.Tensor:
        S_norm, I_norm = self._normalize(obs['S'], obs['I'])
        
        # Encoder l'historique avec GRU
        history = torch.FloatTensor(obs['S_history']).unsqueeze(0).unsqueeze(-1).to(device)
        o_t = self.gru(history).squeeze(0)
        
        # Construire l'état complet (tout en float32)
        state = torch.cat([
            torch.tensor([S_norm, I_norm], dtype=torch.float32, device=device),
            o_t
        ])
        return state
    
    def select_action(self, obs: Dict, explore: bool = True) -> float:
        state = self._build_state(obs).unsqueeze(0)
        
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0, 0]
            
        if explore:
            noise = np.random.normal(0, self.epsilon)
            action = np.clip(action + noise, -1, 1)
            
        return action
    
    def update(self, batch_size: int):
        if len(self.buffer) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        # Convertir en tensors
        state_tensors = torch.stack([self._build_state(s) for s in states])
        next_state_tensors = torch.stack([self._build_state(s) for s in next_states])
        actions = torch.FloatTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        # Update Critic (Double DQN style)
        with torch.no_grad():
            next_actions = self.actor_target(next_state_tensors)
            target_Q = self.critic_target(next_state_tensors, next_actions)
            target_Q = rewards + self.config.gamma * (1 - dones) * target_Q
            
        current_Q = self.critic(state_tensors, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor (5 itérations)
        for _ in range(5):
            actor_loss = -self.critic(state_tensors, self.actor(state_tensors)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        
        # Soft update des réseaux cibles
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
            
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
    
    def decay_epsilon(self, episode: int):
        self.epsilon = max(self.epsilon_min, 1.0 / (1 + episode / self.epsilon_decay))

class ProbDDPG(HidDDPG):
    """prob-DDPG: Approche deux étapes avec probabilités postérieures"""
    def __init__(self, config: SimulationConfig):
        self.config = config
        
        # GRU: 5 couches, 20 nœuds cachés
        self.gru = GRUEncoder(input_size=1, hidden_size=20, num_layers=5).to(device)
        
        # Classifieur pour probabilités postérieures (3 régimes theta)
        self.classifier = nn.Sequential(
            nn.Linear(20, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softmax(dim=-1)
        ).to(device)
        
        # État: (S_t, I_t, phi_1, phi_2, phi_3) -> dim = 5
        state_dim = 5
        
        # Actor-Critic: 5 couches, 64 nœuds cachés
        self.actor = Actor(state_dim, hidden_dim=64, n_layers=5).to(device)
        self.actor_target = Actor(state_dim, hidden_dim=64, n_layers=5).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, hidden_dim=64, n_layers=5).to(device)
        self.critic_target = Critic(state_dim, hidden_dim=64, n_layers=5).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimiseurs
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.gru.parameters()) + list(self.classifier.parameters()),
            lr=config.lr
        )
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.lr)
        
        self.buffer = ReplayBuffer()
        self.epsilon = 1.0
        self.epsilon_decay = 100
        self.epsilon_min = 0.01
        
    def _build_state(self, obs: Dict) -> torch.Tensor:
        S_norm, I_norm = self._normalize(obs['S'], obs['I'])
        
        # Encoder l'historique avec GRU
        history = torch.FloatTensor(obs['S_history']).unsqueeze(0).unsqueeze(-1).to(device)
        h = self.gru(history)
        
        # Calculer les probabilités postérieures
        probs = self.classifier(h).squeeze(0)
        
        # Construire l'état complet
        state = torch.cat([
            torch.tensor([S_norm, I_norm], device=device),
            probs
        ])
        return state

class RegDDPG(HidDDPG):
    """reg-DDPG: Approche deux étapes avec prédiction du signal"""
    def __init__(self, config: SimulationConfig):
        self.config = config
        
        # GRU: 5 couches, 20 nœuds cachés
        self.gru = GRUEncoder(input_size=1, hidden_size=20, num_layers=5).to(device)
        
        # Prédicteur pour S_{t+1}
        self.predictor = nn.Sequential(
            nn.Linear(20, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(device)
        
        # État: (S_t, I_t, S_pred) -> dim = 3
        state_dim = 3
        
        # Actor-Critic: 5 couches, 64 nœuds cachés
        self.actor = Actor(state_dim, hidden_dim=64, n_layers=5).to(device)
        self.actor_target = Actor(state_dim, hidden_dim=64, n_layers=5).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, hidden_dim=64, n_layers=5).to(device)
        self.critic_target = Critic(state_dim, hidden_dim=64, n_layers=5).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimiseurs
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.gru.parameters()) + list(self.predictor.parameters()),
            lr=config.lr
        )
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.lr)
        
        self.buffer = ReplayBuffer()
        self.epsilon = 1.0
        self.epsilon_decay = 100
        self.epsilon_min = 0.01
        
    def _build_state(self, obs: Dict) -> torch.Tensor:
        S_norm, I_norm = self._normalize(obs['S'], obs['I'])
        
        # Encoder l'historique avec GRU
        history = torch.FloatTensor(obs['S_history']).unsqueeze(0).unsqueeze(-1).to(device)
        h = self.gru(history)
        
        # Prédire S_{t+1}
        S_pred = self.predictor(h).squeeze()
        
        # Construire l'état complet
        state = torch.tensor([S_norm, I_norm, S_pred.item()], device=device)
        return state

# =============================================================================
# 6. ENTRAÎNEMENT ET TEST
# =============================================================================
def train_agent(agent, env: TradingEnvironment, config: SimulationConfig, verbose: bool = True):
    """Entraîner un agent DDPG"""
    rewards_history = []
    
    for episode in range(config.n_episodes):
        obs = env.reset()
        episode_reward = 0
        
        while True:
            action = agent.select_action(obs, explore=True)
            next_obs, reward, done = env.step(action)
            
            agent.buffer.push(obs, action, reward, next_obs, done)
            agent.update(config.batch_size)
            
            episode_reward += reward
            obs = next_obs
            
            if done:
                break
        
        agent.decay_epsilon(episode)
        rewards_history.append(episode_reward)
        
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode+1}/{config.n_episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return rewards_history

def test_agent(agent, env: TradingEnvironment, config: SimulationConfig) -> Tuple[float, float]:
    """Tester un agent sur M épisodes"""
    rewards = []
    
    for _ in range(config.n_test_episodes):
        obs = env.reset()
        obs['S'] = 1.0  # S_0 = 1
        obs['I'] = 0.0  # I_0 = 0
        episode_reward = 0
        
        while True:
            action = agent.select_action(obs, explore=False)
            next_obs, reward, done = env.step(action)
            episode_reward += reward
            obs = next_obs
            
            if done:
                break
        
        rewards.append(episode_reward)
    
    return np.mean(rewards), np.std(rewards)

# =============================================================================
# 7. PROGRAMME PRINCIPAL
# =============================================================================
def main():
    print("=" * 60)
    print("RÉPLICATION ÉTUDE DDPG - TRADING AVEC RÉGIMES MARKOVIENS")
    print("=" * 60)
    
    config = SimulationConfig()
    
    # Tester les trois cas de complexité
    for case in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"CAS {case}: ", end="")
        if case == 1:
            print("theta_t Markov Chain uniquement")
        elif case == 2:
            print("theta_t et kappa_t Markov Chains")
        else:
            print("theta_t, kappa_t et sigma_t Markov Chains")
        print("=" * 60)
        
        env = TradingEnvironment(config, case=case)
        
        # Tester les trois algorithmes
        algorithms = {
            'hid-DDPG': HidDDPG(config),
            'prob-DDPG': ProbDDPG(config),
            'reg-DDPG': RegDDPG(config)
        }
        
        results = {}
        
        for name, agent in algorithms.items():
            print(f"\n--- Entraînement {name} ---")
            train_agent(agent, env, config, verbose=True)
            
            print(f"--- Test {name} ---")
            mean_reward, std_reward = test_agent(agent, env, config)
            results[name] = (mean_reward, std_reward)
            print(f"{name}: Récompense moyenne = {mean_reward:.2f} ± {std_reward:.2f}")
        
        # Résumé du cas
        print(f"\n--- Résumé Cas {case} ---")
        for name, (mean, std) in results.items():
            print(f"{name}: {mean:.2f} ± {std:.2f}")

if __name__ == "__main__":
    # Pour un test rapide, réduire les paramètres
    # SimulationConfig.n_episodes = 100
    # SimulationConfig.n_test_episodes = 50
    main()