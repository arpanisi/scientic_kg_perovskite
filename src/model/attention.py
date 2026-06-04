"""
Attention modules for sequence models.

All modules expect tensors shaped as:
    query/key/value: [batch, heads, seq_len, head_dim]

The implementations are intentionally low-level attention kernels rather than
complete transformer blocks. Projection layers, residual connections, and MLPs
belong in the model/block code that calls these modules.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from src.kernels import attention_kernel


ATTENTION_TYPES = {
    "regular": "RegularAttention",  # Vaswani et al. 2017
    "fast": "FastAttention",  # Katharopoulos et al. 2020
    "sparse": "SparseAttention",  # Beltagy 2020 + Zaheer 2020 (BigBird)
    "flash": "FlashAttention",  # Dao et al. 2022/2023
    "custom": "CustomAttention",         # CrystaLLM-π + Solar-GECO synthesis, 2025
}


class TransformerAttention(nn.Module):
    """
    Parent multi-head attention module.

    This class owns the common transformer plumbing: Q/K/V projections, head
    reshaping, attention implementation dispatch, output projection, and dropout.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention_type: str = "regular",
        dropout: float = 0.0,
        causal: bool = False,
        sparse_window_size: int = 128,
        sparse_num_global_tokens: int = 4,
        sparse_num_random_tokens: int = 16,
        custom_property_dim: int = 1,
        custom_num_device_layers: int = 5,
        kernel_backend: str = "auto",
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        self.output_dropout = nn.Dropout(dropout)
        self.attention = build_attention(
            attention_type=attention_type,
            dropout=dropout,
            causal=causal,
            sparse_window_size=sparse_window_size,
            sparse_num_global_tokens=sparse_num_global_tokens,
            sparse_num_random_tokens=sparse_num_random_tokens,
            embed_dim=embed_dim,
            num_heads=num_heads,
            custom_property_dim=custom_property_dim,
            custom_num_device_layers=custom_num_device_layers,
            kernel_backend=kernel_backend,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        key_value_states: Tensor | None = None,
        layer_ids: Tensor | None = None,
        target_property: Tensor | None = None,
    ) -> Tensor:
        source_states = hidden_states if key_value_states is None else key_value_states

        query = self.split_heads(self.query_projection(hidden_states))
        key = self.split_heads(self.key_projection(source_states))
        value = self.split_heads(self.value_projection(source_states))

        if isinstance(self.attention, CustomAttention):
            attended = self.attention(
                query,
                key,
                value,
                attention_mask=attention_mask,
                layer_ids=layer_ids,
                target_property=target_property,
            )
        else:
            attended = self.attention(query, key, value, attention_mask=attention_mask)
        merged = self.merge_heads(attended)
        return self.output_dropout(self.output_projection(merged))

    def split_heads(self, tensor: Tensor) -> Tensor:
        batch_size, seq_len, _ = tensor.shape
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2).contiguous()

    def merge_heads(self, tensor: Tensor) -> Tensor:
        batch_size, _, seq_len, _ = tensor.shape
        tensor = tensor.transpose(1, 2).contiguous()
        return tensor.view(batch_size, seq_len, self.embed_dim)


def build_attention(
    attention_type: str,
    dropout: float = 0.0,
    causal: bool = False,
    sparse_window_size: int = 128,
    sparse_num_global_tokens: int = 4,
    sparse_num_random_tokens: int = 16,
    embed_dim: int | None = None,
    num_heads: int | None = None,
    custom_property_dim: int = 1,
    custom_num_device_layers: int = 5,
    kernel_backend: str = "auto",
) -> nn.Module:
    if attention_type == "regular":
        return RegularAttention(dropout=dropout, causal=causal, backend=kernel_backend)
    if attention_type == "fast":
        if causal:
            raise ValueError("FastAttention does not support causal masking in this implementation.")
        return FastAttention(dropout=dropout)
    if attention_type == "sparse":
        return SparseAttention(
            window_size=sparse_window_size,
            num_global_tokens=sparse_num_global_tokens,
            num_random_tokens=sparse_num_random_tokens,
            dropout=dropout,
            causal=causal,
        )
    if attention_type == "flash":
        return FlashAttention(dropout=dropout, causal=causal)
    if attention_type == "custom":
        if embed_dim is None or num_heads is None:
            raise ValueError("CustomAttention requires embed_dim and num_heads.")
        return CustomAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            property_dim=custom_property_dim,
            num_device_layers=custom_num_device_layers,
            dropout=dropout,
        )
    raise ValueError(f"Unknown attention_type '{attention_type}'. Expected one of: {sorted(ATTENTION_TYPES)}")


class RegularAttention(nn.Module):
    """Scaled dot-product attention."""

    def __init__(self, dropout: float = 0.0, causal: bool = False, backend: str = "auto") -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.causal = causal
        self.backend = backend

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        dropout_p = self.dropout.p if self.training else 0.0
        return attention_kernel(
            query,
            key,
            value,
            attention_mask=attention_mask,
            causal=self.causal,
            dropout_p=dropout_p,
            backend=self.backend,
        )


class FlashAttention(nn.Module):
    """
    FlashAttention wrapper using PyTorch scaled_dot_product_attention.

    FlashAttention was introduced by Dao et al. 2022 and extended in
    FlashAttention-2 by Dao 2023. This wrapper explicitly requests PyTorch's
    Flash SDPA backend when CUDA is available; on non-CUDA devices it falls back
    to the standard SDPA implementation so CPU smoke tests still work.
    """

    def __init__(self, dropout: float = 0.0, causal: bool = False) -> None:
        super().__init__()
        self.dropout = dropout
        self.causal = causal

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        attn_mask = sdpa_attention_mask(attention_mask, query, key, self.causal)
        dropout_p = self.dropout if self.training else 0.0
        is_causal = self.causal and attn_mask is None

        if query.is_cuda:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=False,
            ):
                return torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                )

        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )


class FastAttention(nn.Module):
    """
    Linear attention using an ELU+1 positive feature map.

    This trades exact softmax attention for O(sequence length) complexity. It is
    useful for long contexts, but it is an approximation and should be benchmarked
    against RegularAttention for the target task.
    """

    def __init__(self, dropout: float = 0.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.eps = eps

    def feature_map(self, tensor: Tensor) -> Tensor:
        return torch.nn.functional.elu(tensor) + 1.0

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        if attention_mask is not None:
            key = key.masked_fill(expand_key_padding_mask(attention_mask, key).logical_not(), 0.0)
            value = value.masked_fill(expand_key_padding_mask(attention_mask, value).logical_not(), 0.0)

        query_features = self.feature_map(query)
        key_features = self.feature_map(key)
        key_features = self.dropout(key_features)

        key_value = torch.einsum("bhld,bhle->bhde", key_features, value)
        normalizer = torch.einsum("bhld,bhd->bhl", query_features, key_features.sum(dim=-2))
        output = torch.einsum("bhld,bhde->bhle", query_features, key_value)
        return output / normalizer.unsqueeze(-1).clamp_min(self.eps)


class SparseAttention(nn.Module):
    """
    BigBird-style sparse attention mask.

    The mask combines three sparse components:
      1. Local/sliding-window attention from Longformer (Beltagy et al. 2020).
      2. Global tokens from Longformer/BigBird (Beltagy 2020; Zaheer 2020).
      3. Random attention links from BigBird (Zaheer et al. 2020).

    This keeps the class as one sparse-attention implementation while making the
    sparse pattern more expressive than a pure local window.
    """

    def __init__(
        self,
        window_size: int = 128,
        num_global_tokens: int = 4,
        num_random_tokens: int = 16,
        dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        if num_global_tokens < 0:
            raise ValueError("num_global_tokens cannot be negative.")
        if num_random_tokens < 0:
            raise ValueError("num_random_tokens cannot be negative.")
        self.window_size = window_size
        self.num_global_tokens = num_global_tokens
        self.num_random_tokens = num_random_tokens
        self.dropout = nn.Dropout(dropout)
        self.causal = causal

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        scale = 1.0 / math.sqrt(query.size(-1))
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        scores = scores.masked_fill(self.bigbird_mask(scores).logical_not(), torch.finfo(scores.dtype).min)
        scores = apply_attention_masks(scores, attention_mask, self.causal)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return torch.matmul(weights, value)

    def bigbird_mask(self, scores: Tensor) -> Tensor:
        query_len = scores.size(-2)
        key_len = scores.size(-1)
        mask = self.local_mask(query_len, key_len, scores.device)
        mask = mask | self.global_mask(query_len, key_len, scores.device)
        if self.num_random_tokens:
            mask = mask | self.random_mask(query_len, key_len, scores.device)
        return mask

    def local_mask(self, query_len: int, key_len: int, device: torch.device) -> Tensor:
        query_positions = torch.arange(query_len, device=device).unsqueeze(-1)
        key_positions = torch.arange(key_len, device=device).unsqueeze(0)
        return (query_positions - key_positions).abs() <= self.window_size

    def global_mask(self, query_len: int, key_len: int, device: torch.device) -> Tensor:
        global_count = min(self.num_global_tokens, query_len, key_len)
        mask = torch.zeros(query_len, key_len, device=device, dtype=torch.bool)
        if global_count:
            mask[:global_count, :] = True
            mask[:, :global_count] = True
        return mask

    def random_mask(self, query_len: int, key_len: int, device: torch.device) -> Tensor:
        random_count = min(self.num_random_tokens, key_len)
        if random_count == 0:
            return torch.zeros(query_len, key_len, device=device, dtype=torch.bool)

        query_positions = torch.arange(query_len, device=device).unsqueeze(-1)
        offsets = torch.arange(1, random_count + 1, device=device).unsqueeze(0)
        # Deterministic pseudo-random links keep the mask reproducible without
        # storing per-batch random indices. Large coprime-ish strides spread links.
        random_positions = (query_positions * 1103515245 + offsets * 12345) % key_len
        mask = torch.zeros(query_len, key_len, device=device, dtype=torch.bool)
        mask.scatter_(dim=-1, index=random_positions, value=True)
        return mask


class CustomAttention(nn.Module):
    """
    Property-Conditioned Hierarchical Co-Attention for Perovskite Inverse Design.

    Synthesizes two recent architecture ideas:
    - CrystaLLM-pi (Antunes et al., 2025, arXiv:2511.21299):
      PKV residual attention injects a continuous target property, such as PCE,
      directly into K and V projections without sequence tokenization.
    - Solar-GECO (2025, arXiv:2511.19263):
      Co-attention uses self-attention within device layers and cross-attention
      across layers such as ETL, perovskite, HTL, and back contact.

    The combination gives:
    - Property conditioning: target PCE steers what the model attends to.
    - Hierarchy awareness: layer identity determines cross-attention routing.
    - No discrete tokenization of PCE: injected as a continuous embedding.

    Args:
        embed_dim: model embedding dimension.
        num_heads: number of attention heads.
        property_dim: dimension of continuous property embedding.
        num_device_layers: number of device stack layer identities.
        dropout: attention dropout.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        property_dim: int = 1,
        num_device_layers: int = 5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # PKV residual injection: project a continuous target property into K/V
        # space and add it as a residual to preserve the base representation.
        self.property_to_key = nn.Linear(property_dim, embed_dim)
        self.property_to_value = nn.Linear(property_dim, embed_dim)

        # Co-attention: first attend within the current hierarchy-aware stream,
        # then let tokens attend over the self-attended layer summaries.
        self.self_attn = RegularAttention(dropout=dropout)
        self.cross_attn = RegularAttention(dropout=dropout)

        # Device-layer identity embedding: substrate, ETL, perovskite, HTL,
        # backcontact, etc. IDs are supplied by the caller.
        self.layer_embedding = nn.Embedding(num_device_layers, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
        layer_ids: Tensor | None = None,
        target_property: Tensor | None = None,
    ) -> Tensor:
        if target_property is not None:
            property_key = self.split_projected(self.property_to_key(target_property))
            property_value = self.split_projected(self.property_to_value(target_property))
            key = key + property_key
            value = value + property_value

        if layer_ids is not None:
            layer_bias = self.split_projected(self.layer_embedding(layer_ids))
            query = query + layer_bias
            key = key + layer_bias

        self_out = self.self_attn(query, key, value, attention_mask=attention_mask)
        cross_out = self.cross_attn(query, self_out, self_out, attention_mask=attention_mask)
        return self.dropout(self_out + cross_out)

    def split_projected(self, tensor: Tensor) -> Tensor:
        if tensor.dim() == 2:
            batch_size, _ = tensor.shape
            tensor = tensor.view(batch_size, self.num_heads, self.head_dim)
            return tensor.unsqueeze(2)
        if tensor.dim() == 3:
            batch_size, seq_len, _ = tensor.shape
            tensor = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
            return tensor.transpose(1, 2).contiguous()
        raise ValueError("Projected tensor must have shape [batch, embed_dim] or [batch, seq_len, embed_dim].")


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


def expand_key_padding_mask(attention_mask: Tensor, tensor: Tensor) -> Tensor:
    if attention_mask.dtype != torch.bool:
        attention_mask = attention_mask != 0
    if attention_mask.dim() == 2:
        return attention_mask[:, None, :, None].expand_as(tensor)
    raise ValueError("FastAttention supports 2D key padding masks shaped [batch, seq_len].")
