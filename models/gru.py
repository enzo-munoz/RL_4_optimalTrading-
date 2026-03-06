import torch
import torch.nn as nn

import torch.nn.functional as F

class GRUNet(nn.Module):
    """
    Modular GRU Network that can serve as:
    1. Feature Extractor (returns hidden state) - for hid-DDPG
    2. Classifier (returns probabilities) - for prob-DDPG
    3. Regressor (returns value) - for reg-DDPG
    """
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.0, output_size=None, head_type="reg"):
        """
        Args:
            input_size (int): Dimension of the input features (e.g. window size * features).
            hidden_size (int): Dimension of the GRU hidden state.
            num_layers (int): Number of GRU layers.
            dropout (float): Dropout probability.
            output_size (int, optional): Dimension of the output layer. If None, returns hidden state.
            head_type (str, optional): 'reg', 'hid' or 'prob'. Defines the output head architecture.
        """
        super(GRUNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.head_type = head_type
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.output_layer = None
        
        if output_size is not None:
            if head_type == "prob":
                # For prob: additional fully connected layers with SiLU activation
                # Architecture: Hidden -> Linear -> SiLU -> Linear -> SiLU -> Linear -> Softmax
                fc_hidden_dim = 64  # Arbitrary hidden dimension for the head
                self.output_layer = nn.Sequential(
                    nn.Linear(hidden_size, fc_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(fc_hidden_dim, fc_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(fc_hidden_dim, output_size),
                    nn.Softmax(dim=1)
                )
            elif head_type in ["reg", "hid"]:
                # For reg/hid: Linear -> LeakyReLU
                # The GRU is meant to produce an estimate of the next signal
                self.output_layer = nn.Sequential(
                    nn.Linear(hidden_size, output_size),
                    nn.LeakyReLU()
                )
            else:
                # Default linear layer if type is unknown or simple
                self.output_layer = nn.Linear(hidden_size, output_size)
            
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)
        Returns:
            torch.Tensor: Output tensor. 
                          If output_size is None, returns last hidden state: (batch_size, hidden_size).
                          If output_size is set, returns (batch_size, output_size).
        """
        # x shape: (batch_size, seq_len, input_size)
        out, h_n = self.gru(x)
        
        # Take the last hidden state from the last layer
        # h_n shape: (num_layers, batch_size, hidden_size)
        # We want (batch_size, hidden_size)
        last_hidden = h_n[-1]
        
        if self.output_layer is not None:
            return self.output_layer(last_hidden)
        
        return last_hidden
        
