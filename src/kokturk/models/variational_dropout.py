"""Variational dropout and DropConnect for recurrent networks.

Gal & Ghahramani (2016) "A Theoretically Grounded Application of Dropout in
Recurrent Neural Networks" and Merity et al. (2017) "Regularizing and
Optimizing LSTM Language Models" (AWD-LSTM).

Standard ``nn.Dropout`` samples a fresh mask at every timestep which
destroys the long-range signal carried by the GRU hidden state — critical
for Turkish vowel harmony, where the suffix at position 8 depends on the
root vowel at position 1. :class:`VariationalDropout` samples one mask per
sequence and broadcasts it across all timesteps.

:class:`WeightDropout` is DropConnect applied to specified weight matrices
(e.g. ``weight_hh_l0``) of an RNN module, recomputed on every forward pass
when training. At eval time the original weights are restored.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["VariationalDropout", "WeightDropout"]


class VariationalDropout(nn.Module):
    """Locked dropout: same mask for all timesteps of a sequence.

    Input shape: ``(batch, seq_len, features)`` — mask of shape
    ``(batch, 1, features)`` is broadcast along the time axis. When
    ``training is False`` or ``p == 0`` the layer is an identity.

    Args:
        p: drop probability in ``[0, 1)``.
    """

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"p must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        if x.dim() != 3:
            # Fall back: treat (B, F) as a 1-timestep sequence.
            mask = x.new_empty(x.size(0), x.size(-1)).bernoulli_(1 - self.p)
            return x * mask / (1 - self.p)
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p)
        return x * mask / (1 - self.p)


class WeightDropout(nn.Module):
    """DropConnect over specified weight parameters of an RNN module.

    On each forward pass during training, every listed weight is dropped
    to a fresh mask before being copied back onto the wrapped module. At
    eval time (``self.training is False``) the full weights are used.

    Args:
        module: the wrapped RNN (``nn.GRU`` / ``nn.LSTM``).
        weights: weight names to drop, e.g. ``["weight_hh_l0"]``.
        dropout: drop probability.

    Notes:
        The original weights are stored as ``<name>_raw`` so that the
        module ``state_dict`` still round-trips. The wrapped module's
        ``flatten_parameters()`` is swallowed because DropConnect breaks
        cuDNN's contiguous weight assumption.
    """

    def __init__(
        self,
        module: nn.Module,
        weights: list[str],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.module = module
        self.weights = list(weights)
        self.dropout = dropout
        self._setup()

    def _setup(self) -> None:
        for name in self.weights:
            w = getattr(self.module, name)
            # Stash raw weight under a different name so we can recompute.
            del self.module._parameters[name]
            self.module.register_parameter(name + "_raw", nn.Parameter(w.data))
        # Silence cuDNN's flatten — its weight layout breaks under DropConnect.
        self.module.flatten_parameters = lambda *a, **k: None  # type: ignore[assignment]

    def _apply_dropout(self) -> None:
        for name in self.weights:
            raw = getattr(self.module, name + "_raw")
            if self.training and self.dropout > 0.0:
                w = F.dropout(raw, p=self.dropout, training=True)
            else:
                w = raw
            # setattr avoids parameter registration (w is a view of raw).
            setattr(self.module, name, w)

    def forward(self, *args, **kwargs):
        self._apply_dropout()
        return self.module(*args, **kwargs)
