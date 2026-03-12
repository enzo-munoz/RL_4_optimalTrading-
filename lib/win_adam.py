import torch
from torch.optim import Optimizer


class WinAdam(Optimizer):
    """
    Windowed Adam (W-ADAM) optimizer.

    Instead of Adam's infinite exponentially-weighted gradient history,
    W-ADAM uses a sliding window of the last `window` gradient estimates
    to compute first and second moment approximations.

    Used for GRU pre-training (Table 2). The window size matches the
    lookback window W of the corresponding mode (10 for prob, 50 for reg).
    """

    def __init__(self, params, lr: float = 1e-3, window: int = 10, eps: float = 1e-8):
        defaults = dict(lr=lr, window=window, eps=eps)
        super(WinAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            W   = group['window']
            lr  = group['lr']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad  = p.grad.data.clone()
                state = self.state[p]

                if len(state) == 0:
                    state['grad_window']    = []
                    state['grad_sq_window'] = []

                state['grad_window'].append(grad)
                state['grad_sq_window'].append(grad.pow(2))

                if len(state['grad_window']) > W:
                    state['grad_window'].pop(0)
                    state['grad_sq_window'].pop(0)

                m = torch.stack(state['grad_window']).mean(0)
                v = torch.stack(state['grad_sq_window']).mean(0)

                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-lr)

        return loss
