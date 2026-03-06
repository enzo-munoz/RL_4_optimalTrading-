import torch
import torch.nn as nn
from lib.constants import HIDDEN_LAYERS, MAX_ACTION

class Actor(nn.Module):
    """
    DDPG Actor Network.
    Architecture defined in lib/constants.py
    """
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        # Hidden layers
        for hidden_dim in HIDDEN_LAYERS:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())
        
        self.net = nn.Sequential(*layers)
        self.max_action = MAX_ACTION

    def forward(self, state):
        """
        Args:
            state (torch.Tensor): State tensor.
        Returns:
            torch.Tensor: Action tensor in range [-max_action, max_action].
        """
        return self.max_action * self.net(state)
