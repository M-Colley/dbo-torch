"""Tests for the DBO Gaussian process model.

The important test here is recovery: given data generated with a known drift
rate, marginal likelihood fitting should find roughly that drift rate. Without
it, everything else could pass while the model learned nothing.
"""

from __future__ import annotations

import pytest
import torch

from dbo_torch.model import (
    DBOModelConfig,
    build_model,
    fit_model,
    get_alpha,
    posterior_mean_std,
)


@pytest.fixture(autouse=True)
def _double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def drifting_data(n=40, drift=True, noise=0.05, seed=0):
    """Cost surface whose optimum slides from +5 to 0 over the run."""
    gen = torch.Generator().manual_seed(seed)
    t = torch.arange(1.0, n + 1).unsqueeze(-1)
    u = torch.rand(n, 1, generator=gen) * 14.0 - 5.0
    optimum = 5.0 * (1.0 - (t - 1) / (n - 1)) if drift else torch.full_like(t, 2.5)
    y = (u - optimum).abs() + noise * torch.randn(n, 1, generator=gen)
    return torch.cat([u, t], dim=-1), y


def test_builds_and_fits():
    X, Y = drifting_data()
    model = fit_model(build_model(X, Y))
    assert 0.0 < get_alpha(model) <= 1.0


def test_recovers_drift():
    """On drifting data alpha should settle below 1, and on stationary data it
    should stay close to 1. This is the model doing its actual job."""
    X_drift, Y_drift = drifting_data(drift=True, seed=1)
    X_flat, Y_flat = drifting_data(drift=False, seed=1)

    alpha_drift = get_alpha(fit_model(build_model(X_drift, Y_drift)))
    alpha_flat = get_alpha(fit_model(build_model(X_flat, Y_flat)))

    assert alpha_drift < alpha_flat
    assert alpha_drift < 1.0


def test_drifting_model_predicts_the_current_optimum():
    """A DBO model at the final time should prefer the current optimum (0) to
    the one that was best at the start (5). A stationary model cannot."""
    X, Y = drifting_data(n=50, seed=2)
    t_now = float(X[:, -1].max())

    dbo = fit_model(build_model(X, Y))
    probe = torch.tensor([[0.0, t_now], [5.0, t_now]])
    mean_dbo, _ = posterior_mean_std(dbo, probe)

    assert mean_dbo[0] < mean_dbo[1]


def test_stationary_mode_pins_alpha():
    X, Y = drifting_data()
    model = fit_model(build_model(X, Y, DBOModelConfig(stationary=True)))
    assert get_alpha(model) == 1.0


def test_stationary_mode_cannot_track_drift():
    """The BO baseline should fail to distinguish current from stale optima —
    that failure is what the method exists to fix."""
    X, Y = drifting_data(n=50, seed=2)
    t_now = float(X[:, -1].max())

    bo = fit_model(build_model(X, Y, DBOModelConfig(stationary=True)))
    probe = torch.tensor([[0.0, t_now], [5.0, t_now]])
    mean_bo, _ = posterior_mean_std(bo, probe)

    dbo = fit_model(build_model(X, Y))
    mean_dbo, _ = posterior_mean_std(dbo, probe)

    gap_bo = (mean_bo[1] - mean_bo[0]).item()
    gap_dbo = (mean_dbo[1] - mean_dbo[0]).item()
    assert gap_dbo > gap_bo


@pytest.mark.parametrize("kernel", ["rbf", "matern52"])
def test_spatial_kernel_choices(kernel):
    X, Y = drifting_data()
    model = fit_model(build_model(X, Y, DBOModelConfig(spatial_kernel=kernel)))
    assert 0.0 < get_alpha(model) <= 1.0


@pytest.mark.parametrize("parameterization", ["decay", "direct"])
def test_alpha_parameterizations_both_fit(parameterization):
    X, Y = drifting_data(seed=3)
    model = fit_model(
        build_model(X, Y, DBOModelConfig(alpha_parameterization=parameterization))
    )
    assert 0.0 < get_alpha(model) < 1.0


def test_matlab_compatible_preset_disables_transforms():
    config = DBOModelConfig.matlab_compatible()
    assert config.normalize_inputs is False
    assert config.standardize_outcome is False
    assert config.spatial_kernel == "rbf"


def test_multidimensional_inputs():
    gen = torch.Generator().manual_seed(4)
    n = 30
    t = torch.arange(1.0, n + 1).unsqueeze(-1)
    u = torch.rand(n, 3, generator=gen) * torch.tensor([14.0, 10.0, 20.0]) - 5.0
    y = (u[:, :1] - 5.0 * (1 - (t - 1) / (n - 1))).abs() + 0.05 * u[:, 1:2]

    model = fit_model(build_model(torch.cat([u, t], dim=-1), y))
    lengthscales = model.covar_module.kernels[0].base_kernel.lengthscale
    assert lengthscales.shape[-1] == 3  # one per control parameter, not per column


def test_posterior_shapes():
    X, Y = drifting_data(n=20)
    model = fit_model(build_model(X, Y))
    probe = torch.tensor([[1.0, 20.0], [2.0, 20.0], [3.0, 20.0]])

    mean, sd = posterior_mean_std(model, probe)
    assert mean.shape == (3,)
    assert sd.shape == (3,)
    assert torch.all(sd > 0)


def test_uncertainty_grows_away_from_data():
    X, Y = drifting_data(n=25, seed=5)
    model = fit_model(build_model(X, Y))
    t_now = float(X[:, -1].max())

    near = X[0, 0].item()
    _, sd = posterior_mean_std(model, torch.tensor([[near, t_now], [500.0, t_now]]))
    assert sd[1] > sd[0]


def test_rejects_malformed_inputs():
    X, Y = drifting_data(n=10)

    with pytest.raises(ValueError, match="2-dimensional"):
        build_model(X.flatten(), Y)

    with pytest.raises(ValueError, match="at least two columns"):
        build_model(X[:, :1], Y)

    with pytest.raises(ValueError, match=r"shape \(n, 1\)"):
        build_model(X, Y.flatten())

    with pytest.raises(ValueError, match="spatial_kernel"):
        build_model(X, Y, DBOModelConfig(spatial_kernel="linear"))


def test_fit_survives_degenerate_data():
    """Repeated identical inputs make the covariance singular. A study must not
    die at that iteration; fitting should recover or warn, not raise."""
    X = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
    Y = torch.zeros(4, 1)

    model = fit_model(build_model(X, Y))
    assert 0.0 < get_alpha(model) <= 1.0


def test_fit_model_recovery_loop_raises_noise_floor(monkeypatch):
    """Force the first attempts to fail and check the retry machinery: the
    noise floor must actually grow and fitting must eventually succeed."""
    from linear_operator.utils.errors import NotPSDError

    import dbo_torch.model as model_module

    X, Y = drifting_data(n=20, seed=6)
    model = build_model(X, Y)
    floor_before = float(model.likelihood.noise_covar.raw_noise_constraint.lower_bound)

    real_fit = model_module.fit_gpytorch_mll
    calls = {"n": 0}

    def flaky(mll, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise NotPSDError("synthetic failure")
        return real_fit(mll, **kwargs)

    monkeypatch.setattr(model_module, "fit_gpytorch_mll", flaky)
    model = fit_model(model, noise_growth=2.0)

    assert calls["n"] == 3
    floor_after = float(model.likelihood.noise_covar.raw_noise_constraint.lower_bound)
    assert floor_after > floor_before
    assert 0.0 < get_alpha(model) <= 1.0


def test_fit_model_warns_and_returns_when_every_attempt_fails(monkeypatch):
    """Total fitting failure must degrade to a warning, never kill a study."""
    from linear_operator.utils.errors import NotPSDError

    import dbo_torch.model as model_module

    X, Y = drifting_data(n=10, seed=7)
    model = build_model(X, Y)

    def always_fails(mll, **kwargs):
        raise NotPSDError("always")

    monkeypatch.setattr(model_module, "fit_gpytorch_mll", always_fails)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        model = fit_model(model, max_attempts=3)

    assert 0.0 < get_alpha(model) <= 1.0


def test_get_alpha_finds_kernel_in_foreign_model():
    """A model not built by build_model but containing a TemporalDecayKernel
    should still report its alpha instead of silently claiming stationarity."""
    from botorch.models import SingleTaskGP
    from gpytorch.kernels import RBFKernel, ScaleKernel

    from dbo_torch.kernels import TemporalDecayKernel

    X, Y = drifting_data(n=10)
    temporal = TemporalDecayKernel(active_dims=[1])
    temporal.alpha = 0.42
    covar = ScaleKernel(RBFKernel(ard_num_dims=1, active_dims=[0])) * temporal
    foreign = SingleTaskGP(train_X=X, train_Y=Y, covar_module=covar)

    assert get_alpha(foreign) == pytest.approx(0.42)


# -- normalisation, reference starting point, bookkeeping ---------------


def test_normalisation_maps_the_domain_not_the_data():
    """With bounds given, the domain maps onto the unit cube whatever has been
    observed, so a lengthscale means the same thing at every iteration."""
    X = torch.tensor([[5.0, 1.0], [7.0, 2.0], [3.0, 3.0]])
    Y = torch.tensor([[1.0], [2.0], [0.5]])
    model = build_model(X, Y, bounds=torch.tensor([[-5.0], [9.0]]))

    tf = model.input_transform
    lo = tf.offset[..., 0].item()
    width = tf.coefficient[..., 0].item()
    assert (lo, lo + width) == pytest.approx((-5.0, 9.0))


def test_bounds_shape_is_checked():
    X, Y = drifting_data(n=10)
    with pytest.raises(ValueError, match="bounds must have shape"):
        build_model(X, Y, bounds=torch.tensor([[-5.0, 0.0], [9.0, 1.0]]))


def test_time_column_does_not_trigger_the_input_scaling_warning():
    """Time is unnormalised by design; BoTorch must not warn about it on every
    model build."""
    import warnings

    from botorch.exceptions.warnings import InputDataWarning

    X, Y = drifting_data(n=40)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_model(X, Y, bounds=torch.tensor([[-5.0], [9.0]]))
    assert not [w for w in caught if issubclass(w.category, InputDataWarning)]


def test_matlab_compatible_starts_where_the_reference_does():
    """Lengthscale of half the domain width, signal and noise SD of
    std(Y)/sqrt(2), and a noise floor of 1% of std(Y)."""
    X, Y = drifting_data(n=12)
    model = build_model(
        X, Y, DBOModelConfig.matlab_compatible(), bounds=torch.tensor([[-5.0], [9.0]])
    )
    sd = float(Y.std())

    lengthscale = model.covar_module.kernels[0].base_kernel.lengthscale
    outputscale = model.covar_module.kernels[0].outputscale
    floor = model.likelihood.noise_covar.raw_noise_constraint.lower_bound

    assert lengthscale.item() == pytest.approx(7.0)
    assert outputscale.item() == pytest.approx(sd**2 / 2)
    assert model.likelihood.noise.item() == pytest.approx(sd**2 / 2)
    assert floor.item() == pytest.approx((0.01 * sd) ** 2)


def test_reference_init_in_normalised_coordinates():
    X, Y = drifting_data(n=12)
    config = DBOModelConfig(reference_init=True)
    model = build_model(X, Y, config, bounds=torch.tensor([[-5.0], [9.0]]))
    lengthscale = model.covar_module.kernels[0].base_kernel.lengthscale
    assert lengthscale.item() == pytest.approx(0.5)


def test_relative_noise_floor_has_an_absolute_minimum():
    X, _ = drifting_data(n=6)
    Y = torch.full((6, 1), 3.0)  # constant data: zero spread
    config = DBOModelConfig(standardize_outcome=False, noise_floor_fraction=0.01)
    floor = build_model(X, Y, config).likelihood.noise_covar.raw_noise_constraint.lower_bound
    assert floor.item() == pytest.approx(1e-12)


def test_initial_alpha_of_one_is_rejected_by_the_model():
    X, Y = drifting_data(n=10)
    with pytest.raises(ValueError, match="stationary=True"):
        build_model(X, Y, DBOModelConfig(initial_alpha=1.0))


def test_multistart_keeps_the_best_fit():
    """With several starts, the kept fit must be at least as likely as each
    start fitted on its own."""
    from dbo_torch.model import _total_log_likelihood

    X, Y = drifting_data(n=15, seed=8)
    bounds = torch.tensor([[-5.0], [9.0]])

    singles = [
        _total_log_likelihood(
            fit_model(build_model(X, Y, bounds=bounds), lengthscale_starts=(factor,))
        )
        for factor in (1.0, 0.25)
    ]
    best = fit_model(build_model(X, Y, bounds=bounds), lengthscale_starts=(1.0, 0.25))
    assert _total_log_likelihood(best) >= max(singles) - 1e-6


def test_lengthscale_starts_must_be_positive_factors():
    X, Y = drifting_data(n=10)
    for bad in ((), (1.0, 0.0)):
        with pytest.raises(ValueError, match="lengthscale_starts"):
            fit_model(build_model(X, Y), lengthscale_starts=bad)


def test_matlab_compatible_fits_once_as_the_reference_does():
    assert DBOModelConfig.matlab_compatible().lengthscale_starts == (1.0,)


def test_alpha_appears_once_in_the_state_dict():
    X, Y = drifting_data(n=10)
    keys = [k for k in build_model(X, Y).state_dict() if k.endswith("raw_alpha")]
    assert keys == ["covar_module.kernels.1.raw_alpha"]
