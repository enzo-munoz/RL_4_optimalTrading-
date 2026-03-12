import torch
import torch.nn as nn
from lib.constants import HIDDEN_LAYERS


class Critic(nn.Module):
    """
    DDPG Critic Network.

    Architecture is per-case (passed via hidden_layers) — defaults to HIDDEN_LAYERS
    from constants (Case 1: 5 layers × 16 nodes).
    Input is (state, action) concatenated; output is a scalar Q-value.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_layers=None):
        super(Critic, self).__init__()
        layers_cfg = hidden_layers if hidden_layers is not None else HIDDEN_LAYERS

        layers = []
        d = state_dim + action_dim
        for h in layers_cfg:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1)]

        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=1))
