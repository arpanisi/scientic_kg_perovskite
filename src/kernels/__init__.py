"""Optional attention kernel dispatchers."""

from src.kernels.attention import (
    AttentionBackend,
    attention_kernel,
    resolve_attention_backend,
    triton_is_available,
)

__all__ = [
    "AttentionBackend",
    "attention_kernel",
    "resolve_attention_backend",
    "triton_is_available",
]
