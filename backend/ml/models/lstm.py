"""
LSTMClassifier — single responsibility: model architecture only.

No training logic, no data loading, no MLflow.
Outputs raw logits; callers apply sigmoid / BCEWithLogitsLoss.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    Single- or multi-layer LSTM followed by a linear projection.

    Output: raw logits (batch,) — use BCEWithLogitsLoss during training,
    torch.sigmoid() during inference.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Inter-layer dropout (applied to final hidden state before FC)
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len=1, input_size)
        _, (h_n, _) = self.lstm(x)
        return self.fc(self.drop(h_n[-1])).squeeze(1)  # raw logits (batch,)
