from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only
import torch


class SimpleLogger(Logger):
    """
    SimpleLogger is a simplistic logger to avoid using tensorboard, csvlogger, or something else to keep track of losses

    """

    def __init__(self):
        super().__init__()
        self.metrics = {}

    @property
    def name(self):
        return "SimpleLogger"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "MajeureData"

    def log_metrics(self, metrics, step=None) -> None:
        """Record metrics"""

        def _handle_value_item(value):
            if isinstance(value, torch.Tensor):
                return value.item()
            return value

        if step is None:
            step = len(self.metrics)

        metrics_sanitized = {k: _handle_value_item(v) for k, v in metrics.items()}
        for k, v in metrics_sanitized.items():
            try:
                self.metrics[k].append(v)
            except KeyError:
                self.metrics[k] = []
                self.metrics[k].append(v)
