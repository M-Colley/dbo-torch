"""Tests for the DynamicBO ask/tell loop and validation iterations."""

from __future__ import annotations

import json

import pytest
import torch

from dbo_torch import DBOConfig, DBOModelConfig, DynamicBO, as_stationary

N = 30


@pytest.fixture(autouse=True)
def _double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def drifting_objective(noise: float = 0.0, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    state = {"i": 0}

    def objective(x):
        state["i"] += 1
        ideal = 5.0 * (1.0 - (state["i"] - 1) / (N - 1))
        value = abs(x[0] - ideal)
        if noise:
            value += noise * float(torch.randn(1, generator=gen))
        return value

    return objective


def fast_config(**overrides) -> DBOConfig:
    """Small acquisition budget, so the suite stays quick."""
    defaults = dict(
        seed_points=[[5.0], [7.0], [3.0]],
        seed=0,
        num_restarts=2,
        raw_samples=32,
    )
    defaults.update(overrides)
    return DBOConfig(**defaults)


def make_optimizer(**overrides) -> DynamicBO:
    return DynamicBO(bounds=[(-5.0, 9.0)], config=fast_config(**overrides))


# -- construction -------------------------------------------------------


def test_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="hi > lo"):
        DynamicBO(bounds=[(9.0, -5.0)])


def test_rejects_wrong_length_seed_points():
    with pytest.raises(ValueError, match="seed point"):
        DynamicBO(bounds=[(0.0, 1.0), (0.0, 1.0)], config=DBOConfig(seed_points=[[0.5]]))


# -- seeding ------------------------------------------------------------


def test_seed_points_are_used_in_order():
    opt = make_optimizer()
    for expected in ([5.0], [7.0], [3.0]):
        x = opt.suggest()
        assert x == pytest.approx(expected)
        opt.observe(x, 1.0)


def test_random_seeding_when_no_seed_points_given():
    opt = make_optimizer(seed_points=None, num_seed_points=4)
    for _ in range(4):
        x = opt.suggest()
        assert -5.0 <= x[0] <= 9.0
        opt.observe(x, 1.0)
    assert opt.num_observations == 4


# -- bookkeeping --------------------------------------------------------


def test_iteration_and_time_track_observations():
    opt = make_optimizer()
    objective = drifting_objective()

    for expected in range(1, 6):
        assert opt.next_iteration == expected
        x = opt.suggest()
        observation = opt.observe(x, objective(x))
        assert observation.iteration == expected
        assert observation.time == float(expected)

    assert opt.num_observations == 5


def test_observe_validates_input():
    opt = make_optimizer()
    with pytest.raises(ValueError, match="Expected 1 input"):
        opt.observe([1.0, 2.0], 0.5)
    with pytest.raises(ValueError, match="finite"):
        opt.observe([1.0], float("nan"))


def test_alpha_is_none_before_fitting():
    assert make_optimizer().alpha is None


# -- suggestions --------------------------------------------------------


def test_suggestions_respect_bounds():
    opt = make_optimizer()
    opt.run(drifting_objective(noise=0.05), 12)
    for observation in opt.observations:
        assert -5.0 <= observation.x[0] <= 9.0


def test_run_produces_the_requested_number_of_iterations():
    opt = make_optimizer()
    observations = opt.run(drifting_objective(), 10)
    assert len(observations) == 10
    assert opt.num_observations == 10


# -- validation iterations ---------------------------------------------


def test_validation_iterations_are_scheduled():
    opt = make_optimizer(validation_every=5)
    opt.run(drifting_objective(noise=0.05), 15)

    flagged = [o.iteration for o in opt.observations if o.is_validation]
    assert flagged == [5, 10, 15]


def test_no_validation_when_disabled():
    opt = make_optimizer(validation_every=None)
    opt.run(drifting_objective(), 8)
    assert not any(o.is_validation for o in opt.observations)


def test_validation_records_a_prediction():
    """Validation iterations exist to measure model accuracy, which requires
    the prediction to be captured before the measurement is taken."""
    opt = make_optimizer(validation_every=4)
    opt.run(drifting_objective(noise=0.05), 12)

    errors = opt.prediction_error()
    assert errors
    for entry in errors:
        assert entry["predicted"] is not None
        assert entry["error"] >= 0


def test_suggest_validation_returns_a_point_in_bounds():
    opt = make_optimizer()
    opt.run(drifting_objective(noise=0.05), 8)
    x = opt.suggest_validation()
    assert len(x) == 1
    assert -5.0 <= x[0] <= 9.0


def test_visited_only_validation_returns_an_evaluated_point():
    opt = make_optimizer(validation_visited_only=True)
    opt.run(drifting_objective(noise=0.05), 8)

    x = opt.suggest_validation()
    visited = [o.x for o in opt.observations]
    assert any(x == pytest.approx(v) for v in visited)


def test_validation_before_any_model_falls_back_safely():
    """Called too early, this must return something usable rather than raise."""
    opt = make_optimizer()
    x = opt.suggest_validation()
    assert -5.0 <= x[0] <= 9.0


# -- behaviour ----------------------------------------------------------


def test_tracks_a_drifting_optimum():
    """The optimum falls from 5 to 0; late suggestions should sit lower than
    early ones. This is the end-to-end statement of what DBO is for."""
    opt = make_optimizer(validation_every=None)
    opt.run(drifting_objective(noise=0.05), N)

    early = [o.x[0] for o in opt.observations[3:9]]
    late = [o.x[0] for o in opt.observations[-6:]]
    assert sum(late) / len(late) < sum(early) / len(early)


def test_stationary_baseline_pins_alpha():
    opt = DynamicBO(bounds=[(-5.0, 9.0)], config=as_stationary(fast_config()))
    opt.run(drifting_objective(noise=0.05), 8)
    assert opt.alpha == 1.0


def test_as_stationary_leaves_the_original_config_untouched():
    config = fast_config()
    as_stationary(config)
    assert config.model.stationary is False


# -- reporting ----------------------------------------------------------


def test_best_observed_returns_the_lowest_cost():
    opt = make_optimizer()
    opt.run(drifting_objective(), 10)
    best = opt.best_observed()
    assert best.y == min(o.y for o in opt.observations)


def test_history_is_json_serialisable():
    opt = make_optimizer(validation_every=5)
    opt.run(drifting_objective(noise=0.05), 10)
    json.dumps(opt.history())


def test_save_writes_a_readable_run(tmp_path):
    opt = make_optimizer(validation_every=5)
    opt.run(drifting_objective(noise=0.05), 10)

    path = opt.save(tmp_path / "nested" / "run.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert len(payload["observations"]) == 10
    assert payload["bounds"] == [[-5.0, 9.0]]
    assert 0.0 < payload["alpha"] <= 1.0


def test_repr_is_informative():
    opt = make_optimizer()
    assert "n=0" in repr(opt)
    opt.run(drifting_objective(), 5)
    assert "n=5" in repr(opt)


# -- pending / prediction bookkeeping -----------------------------------


def test_prediction_attached_only_to_the_suggested_input():
    """The recorded prediction belongs to the suggested point; observing a
    different input must not inherit it."""
    opt = make_optimizer(validation_every=None)
    opt.run(drifting_objective(noise=0.05), 5)

    x = opt.suggest()
    mismatched = opt.observe([x[0] + 1.0], 1.0)
    assert mismatched.predicted_y is None
    assert mismatched.is_validation is False

    x = opt.suggest()
    matched = opt.observe(x, 1.0)
    assert matched.predicted_y is not None


def test_float32_roundtrip_still_matches_pending():
    """Unity narrows every value to float32; the pending match must survive it."""
    import struct

    opt = make_optimizer(validation_every=None)
    opt.run(drifting_objective(noise=0.05), 5)

    x = opt.suggest()
    narrowed = [struct.unpack("<f", struct.pack("<f", v))[0] for v in x]
    obs = opt.observe(narrowed, 1.0)
    assert obs.predicted_y is not None


def test_direct_suggest_validation_records_prediction():
    """suggest_validation called directly (validation_every=None flow) must
    record the prediction and the validation flag, same as the scheduled path."""
    opt = make_optimizer(validation_every=None)
    opt.run(drifting_objective(noise=0.05), 6)

    x = opt.suggest_validation()
    obs = opt.observe(x, 0.5)
    assert obs.is_validation is True
    assert obs.predicted_y is not None
    assert opt.prediction_error()


# -- seeding vs validation ----------------------------------------------


def test_validation_never_displaces_seed_points():
    """With more seeds than the validation period, every seed must still be
    applied in order and validation must wait for the seed budget."""
    seeds = [[5.0], [7.0], [3.0], [6.0], [4.0], [2.0]]
    opt = make_optimizer(seed_points=seeds, validation_every=5)
    opt.run(drifting_objective(noise=0.05), 12)

    applied = [o.x for o in opt.observations[:6]]
    for got, expected in zip(applied, seeds, strict=True):
        assert got == pytest.approx(expected)

    flagged = [o.iteration for o in opt.observations if o.is_validation]
    assert flagged == [10]


# -- model lifecycle ----------------------------------------------------


def test_refit_every_preserves_fitted_hyperparameters():
    """Iterations that skip the refit must carry the fitted hyperparameters
    forward, not silently rebuild the GP at factory defaults."""
    opt = make_optimizer(refit_every=3, validation_every=None)
    objective = drifting_objective(noise=0.05)

    opt.run(objective, 3)
    model = opt._ensure_model()  # n=3 -> refit
    fitted = model.covar_module.kernels[0].base_kernel.lengthscale.detach().clone()

    x = opt.suggest()
    opt.observe(x, objective(x))  # n=4 -> stale, but no refit due
    model = opt._ensure_model()
    carried = model.covar_module.kernels[0].base_kernel.lengthscale.detach()

    torch.testing.assert_close(carried, fitted)
    # And it is genuinely a fitted value, not the softplus(0) factory default.
    default = torch.nn.functional.softplus(torch.tensor(0.0, dtype=carried.dtype))
    assert not torch.allclose(carried, default.expand_as(carried))


# -- configuration ------------------------------------------------------


def test_acquisition_time_offset_is_applied():
    opt = make_optimizer(acquisition_time_offset=1.0)
    opt.run(drifting_objective(noise=0.05), 6)
    assert opt._acquisition_time() == pytest.approx(opt.num_observations + 1.0)


def test_acquisition_time_offset_reaches_the_acquisition_search(monkeypatch):
    """The offset must shift the time the acquisition is actually pinned to,
    not just the helper's arithmetic."""
    import dbo_torch._base as base_module

    real = base_module.optimize_acqf
    pinned = []

    def spy(*args, **kwargs):
        pinned.append(dict(kwargs["fixed_features"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(base_module, "optimize_acqf", spy)

    opt = make_optimizer(acquisition_time_offset=1.0, validation_every=None)
    opt.run(drifting_objective(noise=0.05), 6)

    assert pinned
    # 3 seeds, so acquisition runs when 3, 4, 5 observations exist; with an
    # offset of 1 the pinned time is n_observations + 1. The incumbent search
    # and the exploration guard search at the same pinned time, so compare
    # distinct values.
    times = sorted({p[opt.dim] for p in pinned})
    assert times == pytest.approx([4.0, 5.0, 6.0])


def test_continuous_validation_search_stays_in_bounds():
    opt = make_optimizer(validation_visited_only=False)
    opt.run(drifting_objective(noise=0.05), 8)
    x = opt.suggest_validation()
    assert len(x) == 1
    assert -5.0 <= x[0] <= 9.0


def test_multidimensional_optimisation():
    opt = DynamicBO(
        bounds=[(-5.0, 9.0), (10.0, 25.0)],
        config=DBOConfig(num_seed_points=3, seed=0, num_restarts=2, raw_samples=32),
    )
    opt.run(lambda x: abs(x[0]) + 0.1 * abs(x[1] - 18.0), 10)

    assert opt.dim == 2
    for observation in opt.observations:
        assert -5.0 <= observation.x[0] <= 9.0
        assert 10.0 <= observation.x[1] <= 25.0


def test_matlab_compatible_model_runs():
    opt = DynamicBO(
        bounds=[(-5.0, 9.0)],
        config=fast_config(model=DBOModelConfig.matlab_compatible()),
    )
    opt.run(drifting_objective(noise=0.05), 10)
    assert 0.0 < opt.alpha <= 1.0


# -- randomness ---------------------------------------------------------


def test_suggest_leaves_the_global_rng_untouched():
    state = torch.random.get_rng_state()
    opt = make_optimizer(seed_points=None)
    opt.run(drifting_objective(noise=0.05), 5)
    assert torch.equal(torch.random.get_rng_state(), state)


def test_other_code_cannot_perturb_a_run():
    """A second optimiser built and stepped mid-run (a BO arm beside a DBO
    arm, say), or unrelated draws from the global RNG, must not change the
    first run's suggestions."""

    def run(interleave: bool) -> list[list[float]]:
        opt = make_optimizer(seed_points=None)
        objective = drifting_objective(noise=0.05)
        xs = []
        for _ in range(5):
            if interleave:
                make_optimizer(seed=123, seed_points=None).suggest()
                torch.rand(7)
            x = opt.suggest()
            xs.append(x)
            opt.observe(x, objective(x))
        return xs

    assert run(interleave=True) == run(interleave=False)


# -- predictions --------------------------------------------------------


@pytest.mark.parametrize("offset", [0.0, 1.0])
def test_prediction_is_made_when_the_point_will_be_measured(offset):
    """The recorded prediction must refer to the time the measurement is
    taken, whatever time the acquisition function scored candidates at."""
    from dbo_torch.model import posterior_mean_std

    opt = make_optimizer(validation_every=None, acquisition_time_offset=offset)
    opt.run(drifting_objective(noise=0.05), 5)

    x = opt.suggest()
    model = opt._ensure_model()
    point = torch.tensor([x + [float(opt.next_iteration)]])
    mu, sd = posterior_mean_std(model, point)

    assert opt._pending["predicted_y"] == pytest.approx(mu.item())
    assert opt._pending["predicted_sd"] == pytest.approx(sd.item())
    assert opt.observe(x, 1.0).time == float(opt.num_observations)


def test_optimiser_normalises_against_its_domain():
    opt = make_optimizer()
    opt.run(drifting_objective(), 4)
    tf = opt._ensure_model().input_transform
    lo = tf.offset[..., 0].item()
    width = tf.coefficient[..., 0].item()
    assert (lo, lo + width) == pytest.approx((-5.0, 9.0))


# -- search quality -----------------------------------------------------


def _grid_at(t: float) -> torch.Tensor:
    grid = torch.linspace(-5.0, 9.0, 2801).unsqueeze(-1)
    return torch.cat([grid, torch.full_like(grid, t)], dim=-1)


def test_incumbent_finds_the_minimum_of_the_posterior_mean():
    from dbo_torch.model import posterior_mean_std

    opt = make_optimizer(num_restarts=8, raw_samples=128, validation_every=None)
    opt.run(drifting_objective(noise=0.05), 10)
    model = opt._ensure_model()
    t = opt._acquisition_time()

    mu, _ = posterior_mean_std(model, _grid_at(t))
    assert opt._incumbent(model, t) <= float(mu.min()) + 1e-4


def test_continuous_validation_minimises_the_upper_bound():
    from dbo_torch.model import posterior_mean_std
    from dbo_torch.optimizer import _z_score

    opt = make_optimizer(validation_visited_only=False, num_restarts=8, raw_samples=128)
    opt.run(drifting_objective(noise=0.05), 10)
    model = opt._ensure_model()
    t = opt._acquisition_time()
    k = _z_score(1.0 - opt.config.validation_confidence)

    def bound(X):
        mu, sd = posterior_mean_std(model, X)
        return mu + k * sd

    x = opt.suggest_validation()
    chosen = bound(torch.tensor([x + [t]]))
    assert float(chosen) <= float(bound(_grid_at(t)).min()) + 1e-4


def test_incumbent_is_computed_once_per_suggestion(monkeypatch):
    """The over-exploitation guard re-searches EI; the incumbent is a property
    of the fitted model and must not be recomputed for every re-search."""
    opt = make_optimizer(validation_every=None, max_exploit_iterations=3)
    opt.run(drifting_objective(noise=0.05), 4)

    calls = {"incumbent": 0, "ei": 0}
    real_incumbent, real_ei = DynamicBO._incumbent, DynamicBO._argmax_ei

    def incumbent(self, *a, **k):
        calls["incumbent"] += 1
        return real_incumbent(self, *a, **k)

    def ei(self, *a, **k):
        calls["ei"] += 1
        return real_ei(self, *a, **k)

    monkeypatch.setattr(DynamicBO, "_incumbent", incumbent)
    monkeypatch.setattr(DynamicBO, "_argmax_ei", ei)
    monkeypatch.setattr(DynamicBO, "_exploiting_too_much", lambda *a: True)

    opt.suggest()
    assert calls == {"incumbent": 1, "ei": 4}


# -- warm start ---------------------------------------------------------


def test_warm_start_refits_from_the_previous_fit(monkeypatch):
    import dbo_torch._base as base_module

    starts = []
    real_fit = base_module.fit_model

    def spy(model, *args, **kwargs):
        kernel = model.covar_module.kernels[0].base_kernel
        starts.append(kernel.lengthscale.detach().clone())
        return real_fit(model, *args, **kwargs)

    monkeypatch.setattr(base_module, "fit_model", spy)

    opt = make_optimizer(warm_start=True, validation_every=None)
    objective = drifting_objective(noise=0.05)
    opt.run(objective, 4)  # the fourth suggestion fits cold, at n = 3
    fitted = opt._model.covar_module.kernels[0].base_kernel.lengthscale.detach().clone()

    opt.suggest()  # refit at n = 4 starts from the n = 3 fit
    torch.testing.assert_close(starts[-1], fitted)


# -- save and resume ----------------------------------------------------


def _deterministic_cost(x: list[float], iteration: int) -> float:
    return abs(x[0] - 5.0 * (1.0 - (iteration - 1) / (N - 1)))


def test_resumed_run_continues_exactly(tmp_path):
    """A study that crashes and resumes from its last save must produce the
    same suggestions as one that never stopped, including an outstanding
    suggestion made before the crash."""
    a = make_optimizer(validation_every=4)
    for _ in range(6):
        x = a.suggest()
        a.observe(x, _deterministic_cost(x, a.next_iteration))
    outstanding = a.suggest()

    b = DynamicBO.load(a.save(tmp_path / "run.json"))
    assert b.history() == a.history()
    assert b.config == a.config

    for opt in (a, b):
        opt.observe(outstanding, _deterministic_cost(outstanding, opt.next_iteration))
    assert b.observations[-1] == a.observations[-1]

    for _ in range(3):
        xa, xb = a.suggest(), b.suggest()
        assert xb == pytest.approx(xa, abs=1e-9)
        a.observe(xa, _deterministic_cost(xa, a.next_iteration))
        b.observe(xb, _deterministic_cost(xb, b.next_iteration))
    assert b.alpha == pytest.approx(a.alpha)


def test_save_records_the_full_configuration_and_versions(tmp_path):
    opt = make_optimizer(validation_every=5, refit_every=2)
    opt.run(drifting_objective(noise=0.05), 4)
    path = opt.save(tmp_path / "run.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["kind"] == "DynamicBO"
    assert payload["config"]["refit_every"] == 2
    assert payload["config"]["seed"] == 0
    assert payload["config"]["dtype"] == "float64"
    assert payload["config"]["model"]["stationary"] is False
    assert payload["versions"]["botorch"]
    assert not list(tmp_path.glob("*.tmp"))


def test_load_reads_files_from_the_first_save_format(tmp_path):
    old = {
        "bounds": [[-5.0, 9.0]],
        "alpha": 1.0,
        "config": {
            "exploration_ratio": 0.1,
            "validation_every": 10,
            "validation_confidence": 0.01,
            "acquisition_time_offset": 0.0,
            "stationary": True,
        },
        "observations": [
            {
                "iteration": 1, "x": [5.0], "y": 1.0, "time": 1.0,
                "is_validation": False, "predicted_y": None, "predicted_sd": None,
            },
        ],
    }
    path = tmp_path / "old.json"
    path.write_text(json.dumps(old), encoding="utf-8")

    opt = DynamicBO.load(path)
    assert opt.config.model.stationary is True
    assert opt.config.validation_every == 10
    assert opt.num_observations == 1


def test_load_refuses_a_multi_objective_run(tmp_path):
    from dbo_torch import DynamicMOBO

    mo = DynamicMOBO(bounds=[(-5.0, 9.0)], ref_point=[10.0, 10.0])
    path = mo.save(tmp_path / "mo.json")
    with pytest.raises(ValueError, match="DynamicMOBO run"):
        DynamicBO.load(path)
