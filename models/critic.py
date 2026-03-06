import torch
import torch.nn as nn
from lib.constants import HIDDEN_LAYERS

class Critic(nn.Module):
    """
    DDPG Critic Network.
    Architecture defined in lib/constants.py
    """
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        layers = []
        # Input layer takes state and action concatenated
        input_dim = state_dim + action_dim
        
        # Hidden layers
        for hidden_dim in HIDDEN_LAYERS:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
            
        # Output layer (Q-value, single scalar)
        layers.append(nn.Linear(input_dim, 1))
        
        self.net = nn.Sequential(*layers)

    def forward(self, state, action):
        """
        Args:
            state (torch.Tensor): State tensor.
            action (torch.Tensor): Action tensor.
        Returns:
            torch.Tensor: Q-value estimate.
        """
        return self.net(torch.cat([state, action], 1))
