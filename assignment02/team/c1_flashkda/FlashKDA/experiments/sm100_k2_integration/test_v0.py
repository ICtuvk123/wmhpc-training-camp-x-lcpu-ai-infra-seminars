"""GPU-only production baseline/V0 equivalence; keeps the exact-match contract."""
import math

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("V0 validation requires a CUDA-enabled PyTorch and SM100/SM103", allow_module_level=True)
if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
    pytest.skip("V0 requires SM100/SM103", allow_module_level=True)

import flash_kda


def compare(monkeypatch, batch, tokens, heads, has_in, has_out, dtype, lengths=None):
    torch.manual_seed(42)
    shape = (batch, tokens, heads, 128)
    q, k, v, g = [torch.randn(shape, dtype=torch.bfloat16, device="cuda") for _ in range(4)]
    beta = torch.randn(shape[:-1], dtype=torch.bfloat16, device="cuda")
    alog = torch.rand(heads, device="cuda")
    bias = torch.rand(heads, 128, device="cuda")
    seq = None if lengths is None else torch.tensor(
        [0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int64, device="cuda")
    n = batch if lengths is None else len(lengths)
    initial = torch.randn((n, heads, 128, 128), dtype=dtype, device="cuda") if has_in else None
    original = initial.clone() if initial is not None else None
    results = []
    for impl in ("baseline", "sm100_v0"):
        monkeypatch.setenv("FLASH_KDA_K2_IMPL", impl)
        output = torch.full_like(q, float("nan"))
        final = torch.full((n, heads, 128, 128), float("nan"), dtype=dtype, device="cuda") if has_out else None
        flash_kda.fwd(q, k, v, g, beta, 1 / math.sqrt(128), output,
                      A_log=alog, dt_bias=bias, lower_bound=-5.0,
                      initial_state=initial, final_state=final, cu_seqlens=seq)
        torch.cuda.synchronize()
        assert torch.isfinite(output).all()
        if final is not None:
            assert torch.isfinite(final).all()
        if original is not None:
            assert torch.equal(initial, original), "initial state was mutated"
        results.append((output, final))
    assert torch.equal(results[0][0], results[1][0]), "V0 output differs from production baseline"
    if has_out:
        assert torch.equal(results[0][1], results[1][1]), "V0 final state differs from production baseline"


@pytest.mark.parametrize("has_in,has_out", [(True, True), (True, False), (False, True), (False, False)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("case", ["one_chunk", "tail", "batched", "varlen"])
def test_state_contract(monkeypatch, has_in, has_out, dtype, case):
    if case == "varlen":
        # Empty sequences exercise initialization/final store without Phase 6.
        compare(monkeypatch, 1, 66, 2, has_in, has_out, dtype, [0, 16, 17, 0, 33])
    else:
        batch, tokens = {"one_chunk": (1, 16), "tail": (1, 37), "batched": (3, 65)}[case]
        compare(monkeypatch, batch, tokens, 2, has_in, has_out, dtype)


@pytest.mark.parametrize("batch,tokens", [(4, 2048), (1, 8192), (8, 1024)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_frozen_workload(monkeypatch, batch, tokens, dtype):
    compare(monkeypatch, batch, tokens, 64, True, True, dtype)
