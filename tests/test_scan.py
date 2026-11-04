"""Mathematical scan parity tests; these arrays are not research data."""

import pytest
import torch

from drvim.model.scan import selective_scan


@pytest.mark.parametrize("length", [7, 257, 513])
def test_parallel_matches_recurrence_and_gradients(length: int) -> None:
    torch.manual_seed(17)
    batch, channels, state_dim = 2, 3, 4
    inputs = [torch.randn(batch, length, channels, dtype=torch.float64),
              torch.rand(batch, length, channels, dtype=torch.float64) * 0.1,
              -torch.arange(1, state_dim + 1, dtype=torch.float64).repeat(channels, 1),
              torch.randn(batch, length, state_dim, dtype=torch.float64),
              torch.randn(batch, length, state_dim, dtype=torch.float64),
              torch.randn(channels, dtype=torch.float64)]
    reference = [tensor.clone().requires_grad_() for tensor in inputs]
    parallel = [tensor.clone().requires_grad_() for tensor in inputs]
    target = torch.randn(batch, length, channels, dtype=torch.float64)
    ref_y = selective_scan(*reference, backend="reference")
    par_y = selective_scan(*parallel, backend="parallel")
    torch.testing.assert_close(par_y, ref_y, atol=1e-6, rtol=1e-6)
    ref_grad = torch.autograd.grad((ref_y * target).sum(), reference)
    par_grad = torch.autograd.grad((par_y * target).sum(), parallel)
    for observed, expected in zip(par_grad, ref_grad):
        torch.testing.assert_close(observed, expected, atol=1e-5, rtol=1e-5)
