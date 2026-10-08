"""Ordinary fully connected PyTorch networks with physical-coordinate normalization."""
from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """Pointwise MLP. bounds maps each physical coordinate to [-1, 1]."""

    def __init__(self, input_dim, output_dim=1, hidden_layers=(32, 32, 32),
                 activation="tanh", bounds=None, seed=0):
        super().__init__()
        activations = {"tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "silu": nn.SiLU,
                       "softplus": nn.Softplus, "relu": nn.ReLU}
        if activation not in activations:
            raise ValueError(f"Unknown activation {activation!r}: {sorted(activations)}")
        widths = (input_dim, *hidden_layers, output_dim)
        if any(not isinstance(w, int) or w < 1 for w in widths):
            raise ValueError("Network widths must be positive integers")
        self.input_dim, self.output_dim = input_dim, output_dim
        self.spec = dict(input_dim=input_dim, output_dim=output_dim,
                         hidden_layers=list(hidden_layers), activation=activation, seed=seed,
                         bounds=None if bounds is None else torch.as_tensor(bounds).tolist())
        b = torch.as_tensor(bounds if bounds is not None else [[0., 1.]] * input_dim,
                            dtype=torch.float64)
        if b.shape != (input_dim, 2) or not torch.isfinite(b).all() or (b[:, 1] <= b[:, 0]).any():
            raise ValueError("bounds must be finite (input_dim, 2) increasing intervals")
        self.normalize = bounds is not None
        self.spec["bounds"] = b.tolist() if bounds is not None else None
        self.register_buffer("lower", b[:, 0].clone())
        self.register_buffer("upper", b[:, 1].clone())
        layers = []
        # Do not perturb the caller's random generator merely by constructing a model.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for i, (a, b_width) in enumerate(zip(widths[:-1], widths[1:])):
                layer = nn.Linear(a, b_width)
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
                layers.append(layer)
                if i < len(widths) - 2:
                    layers.append(activations[activation]())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Input must have shape (N, {self.input_dim})")
        if self.normalize:
            lower, upper = self.lower.to(x), self.upper.to(x)
            x = 2 * (x - lower) / (upper - lower) - 1
        return self.layers(x)
