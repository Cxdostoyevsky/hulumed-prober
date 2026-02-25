# Copyright 2024 Hulu-Med. Probe head for binary classification at intermediate model layers.

import torch
import torch.nn as nn
from typing import Optional


class ProbeHead(nn.Module):
    """二分类探针：支持 Linear 或 2 层 MLP"""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        probe_type: str = "mlp",
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.probe_type = probe_type

        if probe_type == "linear":
            self.classifier = nn.Linear(input_dim, num_classes)
        elif probe_type == "mlp":
            hdim = hidden_dim if hidden_dim is not None else max(input_dim // 2, 64)
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hdim),
                nn.GELU(),
                nn.Linear(hdim, num_classes),
            )
        else:
            raise ValueError(f"Unknown probe_type: {probe_type}. Use 'linear' or 'mlp'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
