"""Trainer variants for supported weight-update objectives."""

from __future__ import annotations

import torch
from transformers import Trainer


class WeightedSFTTrainer(Trainer):
    """Completion-only SFT trainer with per-token loss weights."""

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        loss_weights = inputs.pop("loss_weights", None)
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        if loss_weights is None:
            loss = outputs.loss
        else:
            loss = weighted_causal_lm_loss(logits, labels, loss_weights)

        return (loss, outputs) if return_outputs else loss


def weighted_causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor, loss_weights: torch.Tensor) -> torch.Tensor:
    """Compute weighted next-token cross entropy for causal LM outputs."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_weights = loss_weights[..., 1:].contiguous().to(shift_logits.device)
    shift_labels = shift_labels.to(shift_logits.device)

    losses = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)

    valid = shift_labels.ne(-100)
    weights = shift_weights.masked_fill(valid.logical_not(), 0.0)
    denominator = weights.sum().clamp_min(1.0)
    return (losses * weights).sum() / denominator
