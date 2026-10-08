"""Tests for the temporal decay kernel.

The kernel is the whole method, so these are deliberately thorough: an error
here would not crash anything, it would quietly produce a model that tracks
drift wrongly.
"""

from __future__ import annotations

import math

import pytest
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel

from dbo_torch.kernels import TemporalDecayKernel


@pytest.fixture(autouse=True)
def _double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


@pytest.fixture
def times():
    return torch.tensor([[0.0], [1.0], [2.0], [5.0], [11.0]])


def test_matches_closed_form(times):
    kernel = TemporalDecayKernel()
    kernel.alpha = 0.85
    K = kernel(times, times).to_dense().detach()

    for i in range(times.shape[0]):
        for j in range(times.shape[0]):
            lag = abs(times[i, 0] - times[j, 0])
            assert K[i, j].item() == pytest.approx(0.85**lag.item(), abs=1e-12)


def test_diagonal_is_unity(times):
    """Zero lag means no decay, whatever alpha is."""
    for alpha in (0.01, 0.5, 0.999, 1.0):
        kernel = TemporalDecayKernel()
        kernel.alpha = alpha
        diag = kernel(times, times, diag=True).detach()
        assert diag.shape == (times.shape[0],)
        torch.testing.assert_close(diag, torch.ones_like(diag))


def test_diag_agrees_with_dense_diagonal(times):
    kernel = TemporalDecayKernel()
    kernel.alpha = 0.7
    dense = kernel(times, times).to_dense().detach().diagonal()
    fast = kernel(times, times, diag=True).detach()
    torch.testing.assert_close(dense, fast)


def test_alpha_one_is_the_stationary_limit(times):
    """At alpha = 1 the temporal factor must vanish exactly, not approximately.

    This is what makes the stationary BO baseline a genuine special case of the
    same code path rather than a separate implementation.
    """
    kernel = TemporalDecayKernel()
    kernel.alpha = 1.0
    K = kernel(times, times).to_dense().detach()
    torch.testing.assert_close(K, torch.ones_like(K))


def test_symmetric_and_positive_semidefinite(times):
    kernel = TemporalDecayKernel()
    kernel.alpha = 0.6
    K = kernel(times, times).to_dense().detach()

    torch.testing.assert_close(K, K.T)
    eigenvalues = torch.linalg.eigvalsh(K + 1e-10 * torch.eye(K.shape[0]))
    assert eigenvalues.min() > -1e-8


def test_decays_monotonically_with_lag():
    kernel = TemporalDecayKernel()
    kernel.alpha = 0.9
    origin = torch.zeros(1, 1)
    lags = torch.arange(0.0, 20.0).unsqueeze(-1)

    covariances = kernel(origin, lags).to_dense().detach().squeeze()
    assert torch.all(covariances[1:] < covariances[:-1])
    assert covariances[-1] < covariances[0]


@pytest.mark.parametrize("alpha", [0.001, 0.25, 0.5, 0.9, 0.99, 1.0])
@pytest.mark.parametrize("parameterization", ["decay", "direct"])
def test_alpha_roundtrips(alpha, parameterization):
    """Setting alpha then reading it back must be the identity.

    Both parameterisations pass through a transform and its inverse, and
    'decay' additionally passes through a clamp, so this is not free.
    """
    kernel = TemporalDecayKernel(parameterization=parameterization)
    kernel.alpha = alpha
    assert kernel.alpha.item() == pytest.approx(alpha, abs=1e-9)


def test_parameterizations_agree_on_covariance(times):
    """Different search geometry, same function."""
    a, b = (TemporalDecayKernel(parameterization=p) for p in ("decay", "direct"))
    a.alpha, b.alpha = 0.77, 0.77
    torch.testing.assert_close(
        a(times, times).to_dense().detach(), b(times, times).to_dense().detach()
    )


def test_alpha_is_differentiable(times):
    kernel = TemporalDecayKernel()
    kernel.alpha = 0.8
    loss = kernel(times, times).to_dense().sum()
    (grad,) = torch.autograd.grad(loss, kernel.raw_alpha)

    assert torch.isfinite(grad).all()
    assert grad.abs().item() > 0


def test_shapes():
    kernel = TemporalDecayKernel()
    x = torch.rand(7, 1)
    y = torch.rand(4, 1)

    assert kernel(x, x).to_dense().shape == (7, 7)
    assert kernel(x, y).to_dense().shape == (7, 4)
    assert kernel(x, x, diag=True).shape == (7,)


def test_batch_shapes():
    kernel = TemporalDecayKernel(batch_shape=torch.Size([3]))
    x = torch.rand(3, 5, 1)

    assert kernel(x, x).to_dense().shape == (3, 5, 5)
    assert kernel(x, x, diag=True).shape == (3, 5)


def test_rejects_multidimensional_input():
    """Guards against forgetting active_dims, which would otherwise silently
    treat a control parameter as if it were time."""
    kernel = TemporalDecayKernel()
    with pytest.raises(RuntimeError, match="one input dimension"):
        kernel(torch.rand(4, 2), torch.rand(4, 2)).to_dense()


def test_active_dims_selects_the_time_column():
    """In the composed kernel, the spatial factor must ignore time and the
    temporal factor must ignore the control parameters."""
    d = 2
    spatial = ScaleKernel(RBFKernel(ard_num_dims=d, active_dims=[0, 1]))
    temporal = TemporalDecayKernel(active_dims=[2])
    spatial.base_kernel.lengthscale = torch.tensor([[1.0, 1.0]])
    spatial.outputscale = torch.tensor(1.0)
    temporal.alpha = torch.tensor(0.5)
    composed = spatial * temporal

    # Identical control parameters, three time units apart.
    a = torch.tensor([[1.0, 2.0, 0.0]])
    b = torch.tensor([[1.0, 2.0, 3.0]])

    value = composed(a, b).to_dense().detach().item()
    assert value == pytest.approx(0.5**3, abs=1e-12)


def test_invalid_arguments():
    with pytest.raises(ValueError, match="parameterization"):
        TemporalDecayKernel(parameterization="nope")

    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="initial_alpha"):
            TemporalDecayKernel(initial_alpha=bad)


@pytest.mark.parametrize("parameterization", ["decay", "direct"])
def test_initial_alpha_of_exactly_one_is_rejected(parameterization):
    """alpha = 1 is reachable only in a limit where the gradient vanishes, so a
    fit started there could never move; it must be refused, not frozen."""
    with pytest.raises(ValueError, match="stationary=True"):
        TemporalDecayKernel(parameterization=parameterization, initial_alpha=1.0)


def test_large_lag_does_not_underflow_to_nan():
    """Long studies produce large lags; alpha ** 1000 must be 0, not NaN."""
    kernel = TemporalDecayKernel()
    kernel.alpha = 0.5
    K = kernel(torch.tensor([[0.0]]), torch.tensor([[2000.0]])).to_dense().detach()

    assert torch.isfinite(K).all()
    assert K.item() == pytest.approx(0.0, abs=1e-12)


def test_initial_alpha_default_matches_reference():
    """The reference implementation starts at a decay rate of 0.01."""
    assert TemporalDecayKernel().alpha.item() == pytest.approx(0.99)


def test_repr_reports_alpha():
    kernel = TemporalDecayKernel()
    kernel.alpha = 0.42
    assert "0.42" in repr(kernel)


def test_alpha_prior_registers_without_crashing():
    """gpytorch >= 1.4 rejects bound zero-arg closures; the prior must register
    with module-taking closures and round-trip through the parameterisation."""
    from gpytorch.priors import GammaPrior

    kernel = TemporalDecayKernel(alpha_prior=GammaPrior(2.0, 2.0))
    assert any("alpha_prior" in name for name, *_ in kernel.named_priors())


@pytest.mark.parametrize("alpha", [1.0, 1e-9])
def test_direct_parameterization_finite_raw_at_boundaries(alpha):
    """The Interval constraint's inverse transform is +/-inf exactly at its
    boundaries; a non-finite raw parameter would poison gradient fitting."""
    kernel = TemporalDecayKernel(parameterization="direct")
    kernel.alpha = alpha
    assert torch.isfinite(kernel.raw_alpha).all()
    assert kernel.alpha.item() == pytest.approx(alpha, abs=1e-8)


def test_batched_setter_accepts_batch_shaped_value():
    kernel = TemporalDecayKernel(batch_shape=torch.Size([3]))
    kernel.alpha = torch.tensor([0.5, 0.7, 0.9])
    values = kernel.alpha.detach().flatten().tolist()
    assert values == pytest.approx([0.5, 0.7, 0.9])


def test_fractional_and_unordered_times():
    """Nothing requires time to be integer or sorted."""
    kernel = TemporalDecayKernel()
    kernel.alpha = 0.8
    t = torch.tensor([[3.5], [0.25], [9.0]])
    K = kernel(t, t).to_dense().detach()

    torch.testing.assert_close(K, K.T)
    expected = 0.8 ** abs(3.5 - 0.25)
    assert K[0, 1].item() == pytest.approx(expected, abs=1e-12)
    assert not math.isnan(K.sum().item())
