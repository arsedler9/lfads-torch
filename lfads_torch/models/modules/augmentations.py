import torch
from torch import nn


class DataAugmentation(nn.Module):
    def __init__(self, transforms=[]):
        super().__init__()
        self.transforms = nn.Sequential(*transforms)

    @torch.no_grad()
    def forward(self, data):
        return self.transforms(data)


class TemporalShift(nn.Module):
    def __init__(self, seed=None, std=3.0, max_shift=6):
        super().__init__()
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)
        self.std = std
        self.max = max_shift

    def forward(self, data):
        batch_size, n_steps, n_channels = data.shape
        # Sample a shift for each channel and each sample
        mean = torch.zeros(batch_size, 1, n_channels)
        shifts = torch.normal(mean, self.std, generator=self.rng)
        shifts = shifts.to(data.device)
        # Clamp the shifts within the range
        shifts = torch.clamp(shifts.round().long(), -self.max, self.max)
        # Create indices and shift them based on the sampled values
        indices = torch.arange(n_steps, device=data.device)[None, :, None]
        indices = indices.repeat(batch_size, 1, n_channels)
        # TODO: Fix rollover bug (zeros?)
        shifted_indices = (indices - shifts) % n_steps
        # Apply the shifts to the data
        shifted_data = torch.gather(data, dim=1, index=shifted_indices)
        return shifted_data
