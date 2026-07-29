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
    import dbo_torch.optimizer as optimizer_module

    real = optimizer_module.optimize_acqf
    pinned = []

    def spy(*args, **kwargs):
        pinned.append(dict(kwargs["fixed_features"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(optimizer_module, "optimize_acqf", spy)

    opt = make_optimizer(acquisition_time_offset=1.0, validation_every=None)
    opt.run(drifting_objective(noise=0.05), 6)

    assert pinned
    # 3 seeds, so acquisition runs when 3, 4, 5 observations exist; with an
    # offset of 1 the pinned time is n_observations + 1. The exploration guard
    # may re-search at the same pinned time, so compare distinct values.
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
