from models.gru import GRUNet
from lib.constants import SimulationConfig

def __init__(self, config: SimulationConfig):
    self.config = config
    
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUNet(input_size, hidden_size, num_layers, output_size=output_size)
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    # État: (S_t, I_t, o_t) -> dim = 2 + 10 = 12
    state_dim = 3
    
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
