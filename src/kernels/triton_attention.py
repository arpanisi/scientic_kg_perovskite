"""
Dense Triton attention forward kernel.

Scope for this first kernel:
  - CUDA only
  - float16/bfloat16 Q/K/V
  - forward pass only
  - causal and non-causal
  - no dropout
  - no arbitrary attention masks

The public dispatcher in `src.kernels.attention` enforces those constraints
before calling `triton_attention_forward`.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - exercised only without optional dependency.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _dense_attention_forward_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        output_ptr,
        num_heads: tl.constexpr,
        query_len: tl.constexpr,
        key_len: tl.constexpr,
        head_dim: tl.constexpr,
        query_stride_b: tl.constexpr,
        query_stride_h: tl.constexpr,
        query_stride_m: tl.constexpr,
        query_stride_d: tl.constexpr,
        key_stride_b: tl.constexpr,
        key_stride_h: tl.constexpr,
        key_stride_n: tl.constexpr,
        key_stride_d: tl.constexpr,
        value_stride_b: tl.constexpr,
        value_stride_h: tl.constexpr,
        value_stride_n: tl.constexpr,
        value_stride_d: tl.constexpr,
        output_stride_b: tl.constexpr,
        output_stride_h: tl.constexpr,
        output_stride_m: tl.constexpr,
        output_stride_d: tl.constexpr,
        scale: tl.constexpr,
        causal: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ) -> None:
        batch_head_id = tl.program_id(0)
        block_m_id = tl.program_id(1)

        batch_id = batch_head_id // num_heads
        head_id = batch_head_id - batch_id * num_heads
        query_batch_offset = batch_id * query_stride_b + head_id * query_stride_h
        key_batch_offset = batch_id * key_stride_b + head_id * key_stride_h
        value_batch_offset = batch_id * value_stride_b + head_id * value_stride_h
        output_batch_offset = batch_id * output_stride_b + head_id * output_stride_h

        offs_m = block_m_id * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        dim_mask = offs_d < head_dim

        query = tl.load(
            query_ptr + query_batch_offset + offs_m[:, None] * query_stride_m + offs_d[None, :] * query_stride_d,
            mask=(offs_m[:, None] < query_len) & dim_mask[None, :],
            other=0.0,
        )

        m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        for start_n in range(0, key_len, BLOCK_N):
            cols = start_n + offs_n
            key = tl.load(
                key_ptr + key_batch_offset + cols[None, :] * key_stride_n + offs_d[:, None] * key_stride_d,
                mask=(cols[None, :] < key_len) & dim_mask[:, None],
                other=0.0,
            )
            value = tl.load(
                value_ptr + value_batch_offset + cols[:, None] * value_stride_n + offs_d[None, :] * value_stride_d,
                mask=(cols[:, None] < key_len) & dim_mask[None, :],
                other=0.0,
            )

            scores = tl.dot(query, key) * scale
            scores = tl.where(cols[None, :] < key_len, scores, -float("inf"))
            if causal:
                scores = tl.where(cols[None, :] <= offs_m[:, None], scores, -float("inf"))

            m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_ij[:, None])
            alpha = tl.exp(m_i - m_ij)
            l_ij = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(value.dtype), value)
            m_i = m_ij
            l_i = l_ij

        output = acc / l_i[:, None]
        tl.store(
            output_ptr + output_batch_offset + offs_m[:, None] * output_stride_m + offs_d[None, :] * output_stride_d,
            output,
            mask=(offs_m[:, None] < query_len) & dim_mask[None, :],
        )


def triton_attention_forward(query: Tensor, key: Tensor, value: Tensor, causal: bool = False) -> Tensor:
    """Run dense Triton attention forward."""
    if triton is None:
        raise RuntimeError("Triton attention requested, but the triton package is not installed.")
    if not query.is_cuda:
        raise RuntimeError("Triton attention requires CUDA tensors.")
    if query.dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError("Triton attention currently supports float16/bfloat16 tensors only.")
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query, key, and value must have shape [batch, heads, seq_len, head_dim].")

    batch_size, num_heads, query_len, head_dim = query.shape
    key_len = key.size(-2)
    if key.shape[:2] != (batch_size, num_heads) or value.shape[:2] != (batch_size, num_heads):
        raise ValueError("query, key, and value must share batch/head dimensions.")
    if value.size(-2) != key_len:
        raise ValueError("key and value must share sequence length.")
    if key.size(-1) != head_dim or value.size(-1) != head_dim:
        raise ValueError("query, key, and value must share head_dim.")
    if head_dim > 128:
        raise ValueError("This Triton attention kernel supports head_dim <= 128.")

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    output = torch.empty_like(query)

    block_m = 16
    block_n = 32
    block_d = triton.next_power_of_2(head_dim)
    grid = (batch_size * num_heads, triton.cdiv(query_len, block_m))
    scale = 1.0 / math.sqrt(head_dim)

    _dense_attention_forward_kernel[grid](
        query,
        key,
        value,
        output,
        num_heads,
        query_len,
        key_len,
        head_dim,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        scale,
        causal,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return output
