"""
Attention kernel dispatcher.

This module is the stable interface between model code and low-level attention
kernels. It must remain safe on CPU/macOS while leaving a clean path for CUDA
Triton kernels on Colab, Kaggle, or RunPod.
"""

from __future__ import annotations

import importlib.util
import math
from enum import StrEnum

import torch
from torch import Tensor


class AttentionBackend(StrEnum):
    """Supported attention backend choices."""

    AUTO = "auto"
    TORCH = "torch"
    SDPA = "sdpa"
    TRITON = "triton"


def attention_kernel(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None = None,
    causal: bool = False,
    dropout_p: float = 0.0,
    backend: str | AttentionBackend = AttentionBackend.AUTO,
) -> Tensor:
    """
    Compute attention with the requested backend.

    Args:
        query/key/value: tensors shaped [batch, heads, seq_len, head_dim].
        attention_mask: optional bool/int mask with shape [batch, seq],
            [batch, query_seq, key_seq], or [batch, heads, query_seq, key_seq].
            True/1 means keep, False/0 means mask out.
        causal: whether to apply a lower-triangular causal mask.
        dropout_p: attention dropout probability. Backend may reject dropout
            when unsupported.
        backend: "auto", "torch", "sdpa", or "triton".

    Returns:
        Tensor shaped [batch, heads, query_seq, head_dim].
    """
    validate_qkv(query, key, value)
    selected_backend = resolve_attention_backend(backend, query, attention_mask, dropout_p)

    if selected_backend == AttentionBackend.TORCH:
        return torch_attention(query, key, value, attention_mask, causal, dropout_p)
    if selected_backend == AttentionBackend.SDPA:
        return sdpa_attention(query, key, value, attention_mask, causal, dropout_p)
    if selected_backend == AttentionBackend.TRITON:
        return triton_attention(query, key, value, attention_mask, causal, dropout_p)

    raise ValueError(f"Unsupported attention backend: {selected_backend}")


def resolve_attention_backend(
    backend: str | AttentionBackend,
    query: Tensor,
    attention_mask: Tensor | None = None,
    dropout_p: float = 0.0,
) -> AttentionBackend:
    """Resolve an attention backend for this runtime and tensor context."""
    requested = AttentionBackend(backend)
    if requested != AttentionBackend.AUTO:
        if requested == AttentionBackend.TRITON:
            validate_triton_request(query, attention_mask, dropout_p)
        return requested

    if can_use_triton(query, attention_mask, dropout_p):
        return AttentionBackend.TRITON
    if can_use_sdpa(query):
        return AttentionBackend.SDPA
    return AttentionBackend.TORCH


def torch_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None = None,
    causal: bool = False,
    dropout_p: float = 0.0,
) -> Tensor:
    """Reference PyTorch matmul attention fallback."""
    scale = 1.0 / math.sqrt(query.size(-1))
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    scores = apply_attention_masks(scores, attention_mask, causal)
    weights = torch.softmax(scores, dim=-1)
    if dropout_p:
        weights = torch.nn.functional.dropout(weights, p=dropout_p, training=True)
    return torch.matmul(weights, value)


def sdpa_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None = None,
    causal: bool = False,
    dropout_p: float = 0.0,
) -> Tensor:
    """PyTorch scaled_dot_product_attention backend."""
    attn_mask = sdpa_attention_mask(attention_mask, query, key, causal)
    is_causal = causal and attn_mask is None
    return torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )


def triton_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None = None,
    causal: bool = False,
    dropout_p: float = 0.0,
) -> Tensor:
    """Dispatch to the optional Triton implementation."""
    validate_triton_request(query, attention_mask, dropout_p)
    try:
        from src.kernels.triton_attention import triton_attention_forward
    except ModuleNotFoundError as exc:
        raise RuntimeError("Triton attention backend is not implemented yet.") from exc

    return triton_attention_forward(query, key, value, causal=causal)


def can_use_sdpa(query: Tensor) -> bool:
    return hasattr(torch.nn.functional, "scaled_dot_product_attention") and query.dim() == 4


def can_use_triton(query: Tensor, attention_mask: Tensor | None, dropout_p: float) -> bool:
    if not query.is_cuda:
        return False
    if not triton_is_available():
        return False
    if query.dtype not in {torch.float16, torch.bfloat16}:
        return False
    if attention_mask is not None:
        return False
    if dropout_p:
        return False
    return True


def validate_triton_request(query: Tensor, attention_mask: Tensor | None, dropout_p: float) -> None:
    if not query.is_cuda:
        raise RuntimeError("Triton attention requires CUDA tensors.")
    if not triton_is_available():
        raise RuntimeError("Triton attention requested, but the triton package is not installed.")
    if query.dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError("Triton attention currently supports float16/bfloat16 tensors only.")
    if attention_mask is not None:
        raise RuntimeError("Triton attention mask support is not implemented yet.")
    if dropout_p:
        raise RuntimeError("Triton attention dropout support is not implemented yet.")


def triton_is_available() -> bool:
    return importlib.util.find_spec("triton") is not None


def validate_qkv(query: Tensor, key: Tensor, value: Tensor) -> None:
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query, key, and value must have shape [batch, heads, seq_len, head_dim].")
    if query.shape[:2] != key.shape[:2] or key.shape[:2] != value.shape[:2]:
        raise ValueError("query, key, and value must have matching batch and head dimensions.")
    if key.size(-2) != value.size(-2):
        raise ValueError("key and value must have matching sequence lengths.")
    if query.size(-1) != key.size(-1):
        raise ValueError("query and key must have matching head dimensions.")


def apply_attention_masks(scores: Tensor, attention_mask: Tensor | None, causal: bool) -> Tensor:
    if attention_mask is not None:
        scores = scores.masked_fill(expand_score_mask(attention_mask, scores).logical_not(), torch.finfo(scores.dtype).min)
    if causal:
        query_len = scores.size(-2)
        key_len = scores.size(-1)
        causal_mask = torch.ones(query_len, key_len, device=scores.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(causal_mask.logical_not(), torch.finfo(scores.dtype).min)
    return scores


def expand_score_mask(attention_mask: Tensor, scores: Tensor) -> Tensor:
    if attention_mask.dtype != torch.bool:
        attention_mask = attention_mask != 0
    if attention_mask.dim() == 2:
        return attention_mask[:, None, None, :].expand_as(scores)
    if attention_mask.dim() == 3:
        return attention_mask[:, None, :, :].expand_as(scores)
    if attention_mask.dim() == 4:
        return attention_mask.expand_as(scores)
    raise ValueError("attention_mask must have 2, 3, or 4 dimensions.")


def sdpa_attention_mask(
    attention_mask: Tensor | None,
    query: Tensor,
    key: Tensor,
    causal: bool,
) -> Tensor | None:
    query_len = query.size(-2)
    key_len = key.size(-2)
    mask = None

    if attention_mask is not None:
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask != 0
        if attention_mask.dim() == 2:
            mask = attention_mask[:, None, None, :].expand(query.size(0), query.size(1), query_len, key_len)
        elif attention_mask.dim() == 3:
            mask = attention_mask[:, None, :, :].expand(query.size(0), query.size(1), query_len, key_len)
        elif attention_mask.dim() == 4:
            mask = attention_mask.expand(query.size(0), query.size(1), query_len, key_len)
        else:
            raise ValueError("attention_mask must have 2, 3, or 4 dimensions.")

    if causal:
        causal_mask = torch.ones(query_len, key_len, device=query.device, dtype=torch.bool).tril()
        mask = causal_mask if mask is None else mask & causal_mask

    return mask
