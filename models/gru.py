import torch
import torch.nn as nn

# GRU linear head configuration (Table 2)
GRU_FC_HIDDEN = 64   # hidden nodes in each linear layer of the head
GRU_FC_LAYERS = 5    # total linear layers in the head


class GRUNet(nn.Module):
    """
    Modular GRU network with a fixed-architecture MLP head.

    GRU component  — 5 recurrent layers, 20 hidden units  (Table 2: d_l=5, d_h=20)
    Linear head    — 5 linear layers,    64 hidden units  (Table 2: l=5, d=64)

    Head variants:
      'prob' : SiLU activations → Softmax   (classifies θ regime)
      'reg'  : LeakyReLU activations        (predicts S_{t+1})
      'hid'  : same as 'reg'
    """

    def __init__(
        self,
        input_size:  int,
        hidden_size: int,
        num_layers:  int  = 5,
        dropout:     float = 0.0,
        output_size: int  = None,
        head_type:   str  = "reg",
    ):
        super(GRUNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.head_type   = head_type

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_layer = None
        if output_size is not None:
            self.output_layer = self._build_head(hidden_size, output_size, head_type)

    @staticmethod
    def _build_head(in_dim: int, out_dim: int, head_type: str) -> nn.Sequential:
        """Build a GRU_FC_LAYERS-layer MLP head from in_dim → out_dim."""
        layers = []
        d = in_dim
        if head_type == "prob":
            for _ in range(GRU_FC_LAYERS - 1):
                layers += [nn.Linear(d, GRU_FC_HIDDEN), nn.SiLU()]
                d = GRU_FC_HIDDEN
            layers += [nn.Linear(d, out_dim), nn.Softmax(dim=1)]
        else:  # reg / hid / default
            for _ in range(GRU_FC_LAYERS - 1):
                layers += [nn.Linear(d, GRU_FC_HIDDEN), nn.LeakyReLU()]
                d = GRU_FC_HIDDEN
            layers += [nn.Linear(d, out_dim)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            (batch_size, output_size)  if head present
            (batch_size, hidden_size)  otherwise
        """
        _, h_n = self.gru(x)
        last_hidden = h_n[-1]   # (batch_size, hidden_size)

        if self.output_layer is not None:
            return self.output_layer(last_hidden)
        return last_hidden
