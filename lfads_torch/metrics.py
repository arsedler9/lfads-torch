import torch


def r2_score(preds, targets):
    if preds.ndim > 2:
        preds = preds.reshape(-1, preds.shape[-1])
    if targets.ndim > 2:
        targets = targets.reshape(-1, targets.shape[-1])
    target_mean = torch.mean(targets, dim=0)
    ss_tot = torch.sum((targets - target_mean) ** 2, dim=0)
    ss_res = torch.sum((targets - preds) ** 2, dim=0)
    return torch.mean(1 - ss_res / ss_tot)
