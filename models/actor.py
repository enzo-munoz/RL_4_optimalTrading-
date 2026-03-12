import torch
import torch.nn as nn
from lib.constants import HIDDEN_LAYERS, MAX_ACTION


class Actor(nn.Module):
    """
    DDPG Actor Network.

    Architecture is per-case (passed via hidden_layers) — defaults to HIDDEN_LAYERS
    from constants (Case 1: 5 layers × 16 nodes).
    Output is scaled to [-MAX_ACTION, MAX_ACTION] via Tanh.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_layers=None):
        super(Actor, self).__init__()
        layers_cfg = hidden_layers if hidden_layers is not None else HIDDEN_LAYERS

        layers = []
        d = state_dim
        for h in layers_cfg:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, action_dim), nn.Tanh()]

        self.net        = nn.Sequential(*layers)
        self.max_action = MAX_ACTION

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)
