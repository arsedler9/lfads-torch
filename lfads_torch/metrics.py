import torch
from torchmetrics import Metric


def r2_score(preds, targets):
    if preds.ndim > 2:
        preds = preds.reshape(-1, preds.shape[-1])
    if targets.ndim > 2:
        targets = targets.reshape(-1, targets.shape[-1])
    target_mean = torch.mean(targets, dim=0)
    ss_tot = torch.sum((targets - target_mean) ** 2, dim=0)
    ss_res = torch.sum((targets - preds) ** 2, dim=0)
    return torch.mean(1 - ss_res / ss_tot)


class ExpSmoothedMetric(Metric):
    """Averages within epochs and exponentially smooths between epochs."""

    def __init__(self, coef=0.9, **kwargs):
        super().__init__(**kwargs)
        self.coef = coef
        # Automatically reset by torchmetrics after `compute` is called
        self.add_state("value", default=torch.tensor(0.0))
        self.add_state("count", default=torch.tensor(0))
        # `prev` must be immune to `reset`
        self.prev = torch.tensor(float("nan"))

    def update(self, value):
        self.value += value
        self.count += 1

    def compute(self):
        curr = self.value / self.count
        if torch.isnan(self.prev):
            self.prev = curr
        smth = self.coef * self.prev + (1 - self.coef) * curr
        self.prev = smth
        return smth
