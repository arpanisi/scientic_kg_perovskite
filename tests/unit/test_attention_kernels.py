import pytest
import torch

from src.kernels import AttentionBackend, attention_kernel, resolve_attention_backend


def qkv(batch=2, heads=2, seq_len=6, head_dim=8):
    torch.manual_seed(7)
    query = torch.randn(batch, heads, seq_len, head_dim)
    key = torch.randn(batch, heads, seq_len, head_dim)
    value = torch.randn(batch, heads, seq_len, head_dim)
    return query, key, value


def test_auto_backend_resolves_to_cpu_safe_backend():
    query, _, _ = qkv()

    backend = resolve_attention_backend("auto", query)

    assert backend in {AttentionBackend.SDPA, AttentionBackend.TORCH}


@pytest.mark.parametrize("backend", ["torch", "sdpa", "auto"])
def test_attention_kernel_returns_expected_shape_on_cpu(backend):
    query, key, value = qkv()

    output = attention_kernel(query, key, value, backend=backend)

    assert output.shape == query.shape
    assert torch.isfinite(output).all()


def test_attention_kernel_supports_key_padding_mask():
    query, key, value = qkv()
    attention_mask = torch.ones(query.size(0), query.size(-2), dtype=torch.bool)
    attention_mask[:, -2:] = False

    torch_output = attention_kernel(query, key, value, attention_mask=attention_mask, backend="torch")
    sdpa_output = attention_kernel(query, key, value, attention_mask=attention_mask, backend="sdpa")

    assert torch_output.shape == query.shape
    assert sdpa_output.shape == query.shape
    torch.testing.assert_close(sdpa_output, torch_output, rtol=1e-5, atol=1e-5)


def test_attention_kernel_supports_causal_mask():
    query, key, value = qkv()

    torch_output = attention_kernel(query, key, value, causal=True, backend="torch")
    sdpa_output = attention_kernel(query, key, value, causal=True, backend="sdpa")

    assert torch_output.shape == query.shape
    torch.testing.assert_close(sdpa_output, torch_output, rtol=1e-5, atol=1e-5)


def test_attention_kernel_supports_combined_padding_and_causal_mask():
    query, key, value = qkv()
    attention_mask = torch.ones(query.size(0), query.size(-2), dtype=torch.bool)
    attention_mask[:, -1] = False

    torch_output = attention_kernel(query, key, value, attention_mask=attention_mask, causal=True, backend="torch")
    sdpa_output = attention_kernel(query, key, value, attention_mask=attention_mask, causal=True, backend="sdpa")

    torch.testing.assert_close(sdpa_output, torch_output, rtol=1e-5, atol=1e-5)


def test_triton_backend_request_fails_clearly_without_cuda():
    query, key, value = qkv()

    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        attention_kernel(query, key, value, backend="triton")


def test_invalid_qkv_shape_raises():
    query, key, value = qkv()

    with pytest.raises(ValueError, match="must have shape"):
        attention_kernel(query[0], key, value, backend="torch")
