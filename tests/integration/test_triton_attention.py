import pytest
import torch

from src.kernels import attention_kernel, triton_is_available


pytestmark = pytest.mark.integration


def require_cuda_triton():
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton attention integration tests.")
    if not triton_is_available():
        pytest.skip("Triton package is not available.")


def make_qkv(seq_len: int, head_dim: int, dtype: torch.dtype):
    torch.manual_seed(11)
    device = torch.device("cuda")
    query = torch.randn(2, 4, seq_len, head_dim, device=device, dtype=dtype)
    key = torch.randn(2, 4, seq_len, head_dim, device=device, dtype=dtype)
    value = torch.randn(2, 4, seq_len, head_dim, device=device, dtype=dtype)
    return query, key, value


@pytest.mark.parametrize("seq_len", [16, 32, 128])
@pytest.mark.parametrize("head_dim", [32, 64])
@pytest.mark.parametrize("causal", [False, True])
def test_triton_dense_attention_matches_torch_reference(seq_len, head_dim, causal):
    require_cuda_triton()
    query, key, value = make_qkv(seq_len, head_dim, torch.float16)

    triton_output = attention_kernel(query, key, value, causal=causal, backend="triton")
    reference_output = attention_kernel(query, key, value, causal=causal, backend="sdpa")

    assert triton_output.shape == reference_output.shape
    assert torch.isfinite(triton_output).all()
    torch.testing.assert_close(triton_output, reference_output, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_dense_attention_dtype_smoke(dtype):
    require_cuda_triton()
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16.")

    query, key, value = make_qkv(seq_len=32, head_dim=64, dtype=dtype)

    output = attention_kernel(query, key, value, backend="triton")

    assert output.shape == query.shape
    assert output.dtype == dtype
    assert torch.isfinite(output).all()
