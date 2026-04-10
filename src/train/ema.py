"""Exponential Moving Average of model parameters.

Usage::

    ema = EMAWeights(model, decay=0.999)
    for batch in train_loader:
        loss = train_step(model, batch)
        loss.backward()
        optimizer.step()
        ema.update()

    # Validation with EMA weights
    ema.apply()
    val_metrics = validate(model)
    ema.restore()

Only float parameters are tracked. Non-float buffers (e.g. token-type ids)
are left untouched.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["EMAWeights"]


class EMAWeights:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        self.model = model
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad and p.is_floating_point():
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self) -> None:
        for name, p in self.model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(
                p.detach(), alpha=1.0 - self.decay,
            )

    def apply(self) -> None:
        """Swap model parameters with EMA shadow values."""
        self.backup = {}
        for name, p in self.model.named_parameters():
            if name not in self.shadow:
                continue
            self.backup[name] = p.detach().clone()
            p.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Restore the pre-``apply`` training parameters."""
        for name, p in self.model.named_parameters():
            if name not in self.backup:
                continue
            p.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for k, v in state.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)
