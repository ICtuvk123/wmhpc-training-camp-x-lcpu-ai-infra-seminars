"""V1a recurrence-depth ladder against the unchanged production baseline."""
import math

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("V1a validation requires CUDA-enabled PyTorch", allow_module_level=True)

import flash_kda


@pytest.mark.parametrize("chunks", [1, 2, 4, 8, 16, 128])
def test_v1a_recurrence_ladder(monkeypatch, chunks):
    torch.manual_seed(2026 + chunks)
    batch, heads, dim, tokens = 1, 2, 128, chunks * 16
    shape = (batch, tokens, heads, dim)
    q, k, v, g = [torch.randn(shape, dtype=torch.bfloat16, device="cuda") for _ in range(4)]
    beta = torch.randn(shape[:-1], dtype=torch.bfloat16, device="cuda")
    initial = torch.randn((batch, heads, dim, dim), dtype=torch.bfloat16, device="cuda")
    initial_copy = initial.clone()
    alog = torch.rand(heads, dtype=torch.float32, device="cuda")
    bias = torch.rand(heads, dim, dtype=torch.float32, device="cuda")
    results = []
    for implementation in ("baseline", "v1a"):
        monkeypatch.setenv("FLASH_KDA_K2_IMPL", implementation)
        output = torch.full_like(q, float("nan"))
        final = torch.full_like(initial, float("nan"))
        flash_kda.fwd(q, k, v, g, beta, 1 / math.sqrt(dim), output,
                      A_log=alog, dt_bias=bias, lower_bound=-5.0,
                      initial_state=initial, final_state=final)
        torch.cuda.synchronize()
        assert torch.isfinite(output).all()
        assert torch.isfinite(final).all()
        assert torch.equal(initial, initial_copy)
        results.append((output, final))
    assert torch.equal(results[0][0], results[1][0]), f"output mismatch at {chunks} chunks"
    assert torch.equal(results[0][1], results[1][1]), f"state mismatch at {chunks} chunks"
