"""Tests for the DynamicMOBO ask/tell loop, per-objective drift, and reporting."""

from __future__ import annotations

import json

import pytest
import torch

from dbo_torch import DynamicMOBO, MODBOConfig, as_stationary_mo

N = 16
BOUNDS = [(-5.0, 5.0)]
REF_POINT = [40.0, 40.0]


@pytest.fixture(autouse=True)
def _double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def drifting_biobjective(noise: float = 0.0, seed: int = 0, n: int = N):
    """A bi-objective problem with a genuine, drifting trade-off.

    f1(x, t) = (x - a(t))^2 with a(t) drifting from 4 to 0 across the run;
    f2(x)    = (x + 2)^2, stationary.
    The Pareto set at time t is the interval between a(t) and -2, which drifts.
    """
    gen = torch.Generator().manual_seed(seed)
    state = {"i": 0}

    def objective(x):
        state["i"] += 1
        a = 4.0 * (1.0 - (state["i"] - 1) / (n - 1))
        f1 = (x[0] - a) ** 2
        f2 = (x[0] + 2.0) ** 2
        if noise:
            f1 += noise * float(torch.randn(1, generator=gen))
            f2 += noise * float(torch.randn(1, generator=gen))
        return [f1, f2]

    return objective


def fast_config(**overrides) -> MODBOConfig:
    """Small acquisition budget, so the suite stays quick."""
    defaults = dict(
        seed_points=[[0.0], [3.0], [-3.0]],
        seed=0,
        num_restarts=2,
        raw_samples=32,
        mc_samples=16,
    )
    defaults.update(overrides)
    return MODBOConfig(**defaults)


def make_optimizer(ref_point=None, **overrides) -> DynamicMOBO:
    return DynamicMOBO(
        bounds=BOUNDS,
        ref_point=REF_POINT if ref_point is None else ref_point,
        config=fast_config(**overrides),
    )


@pytest.fixture(scope="module")
def drift_run() -> DynamicMOBO:
    """One completed run on the drifting problem, shared by read-only tests."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        opt = make_optimizer()
        opt.run(drifting_biobjective(noise=0.05), N)
    finally:
        torch.set_default_dtype(previous)
    return opt


# -- construction -------------------------------------------------------


def test_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="hi > lo"):
        DynamicMOBO(bounds=[(5.0, -5.0)], ref_point=REF_POINT)


def test_rejects_single_objective_ref_point():
    with pytest.raises(ValueError, match="at least two objectives"):
        DynamicMOBO(bounds=BOUNDS, ref_point=[40.0])


def test_rejects_non_finite_ref_point():
    with pytest.raises(ValueError, match="finite"):
        DynamicMOBO(bounds=BOUNDS, ref_point=[40.0, float("inf")])


def test_rejects_wrong_length_seed_points():
    with pytest.raises(ValueError, match="seed point"):
        DynamicMOBO(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            ref_point=REF_POINT,
            config=MODBOConfig(seed_points=[[0.5]]),
        )


# -- bookkeeping --------------------------------------------------------


def test_observe_validates_input():
    opt = make_optimizer()
    with pytest.raises(ValueError, match="Expected 1 input"):
        opt.observe([1.0, 2.0], [0.5, 0.5])
    with pytest.raises(ValueError, match="Expected 2 objective"):
        opt.observe([1.0], [0.5])
    with pytest.raises(ValueError, match="Expected 2 objective"):
        opt.observe([1.0], [0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="finite"):
        opt.observe([1.0], [0.5, float("nan")])


def test_alphas_are_none_before_fitting():
    assert make_optimizer().alphas is None


def test_pareto_front_is_empty_before_observations():
    assert make_optimizer().pareto_front() == []
    assert make_optimizer().hypervolume_trace() == []


def test_validation_before_any_model_falls_back_safely():
    opt = make_optimizer()
    x = opt.suggest_validation()
    assert -5.0 <= x[0] <= 5.0


# -- the point of the design -------------------------------------------


def test_per_objective_alpha_recovery(drift_run):
    """f1 drifts and f2 is stationary, so the fitted drift rates must differ:
    alpha for f1 below alpha for f2, and alpha for f2 close to 1. This is the
    test that one temporal kernel per objective delivers what it promises."""
    alphas = drift_run.alphas
    assert len(alphas) == 2
    assert all(0.0 < a <= 1.0 for a in alphas)
    assert alphas[0] < alphas[1]
    assert alphas[1] > 0.95


def test_tracks_the_drifting_pareto_set(drift_run):
    """The Pareto set's upper end falls from 4 to 0; late suggestions should
    sit lower than early ones."""
    xs = [o.x[0] for o in drift_run.observations]
    early = xs[3:8]
    late = xs[-5:]
    assert sum(late) / len(late) < sum(early) / len(early)
    assert all(-5.0 <= x <= 5.0 for x in xs)


def test_current_time_front_differs_from_raw(drift_run):
    """Under drift the raw observed front is anchored by stale measurements;
    the drift-adjusted front re-scores every design at t = now. Late in the
    run the two must disagree, in membership or in values."""
    raw = {o["iteration"]: o["y"] for o in drift_run.pareto_front(at_current_time=False)}
    now = {o["iteration"]: o["y"] for o in drift_run.pareto_front(at_current_time=True)}
    assert raw and now

    if set(raw) != set(now):
        return
    gaps = [
        abs(raw[i][k] - now[i][k]) for i in raw for k in range(2)
    ]
    assert max(gaps) > 0.5


def test_pareto_fronts_are_mutually_non_dominated(drift_run):
    for at_now in (False, True):
        front = [e["y"] for e in drift_run.pareto_front(at_current_time=at_now)]
        for i, a in enumerate(front):
            for j, b in enumerate(front):
                if i != j:
                    assert not (a[0] <= b[0] and a[1] <= b[1] and a != b)


# -- validation iterations ---------------------------------------------


def test_validation_iterations_fire_and_record_predictions():
    opt = make_optimizer(validation_every=4)
    opt.run(drifting_biobjective(noise=0.05, n=12), 12)

    flagged = [o.iteration for o in opt.observations if o.is_validation]
    assert flagged == [4, 8, 12]

    for obs in opt.observations:
        visited_before = [p.x for p in opt.observations if p.iteration < obs.iteration]
        if obs.is_validation:
            assert -5.0 <= obs.x[0] <= 5.0
            # Visited-only: a validation point re-tests an evaluated input.
            assert any(obs.x == pytest.approx(v) for v in visited_before)
            # Per-objective predictions captured before the measurement.
            assert len(obs.predicted_y) == 2
            assert len(obs.predicted_sd) == 2
            assert all(sd >= 0 for sd in obs.predicted_sd)

    errors = opt.prediction_error()
    assert [e["iteration"] for e in errors] == [4, 8, 12]
    assert all(len(e["error"]) == 2 for e in errors)


# -- reporting ----------------------------------------------------------


def test_hypervolume_trace_is_non_decreasing_and_positive(drift_run):
    """The observed front only gains points, so its hypervolume under a fixed
    reference point can never shrink."""
    trace = drift_run.hypervolume_trace()
    assert len(trace) == N
    assert all(b >= a - 1e-9 for a, b in zip(trace, trace[1:], strict=False))
    assert trace[-1] > 0


def test_history_is_json_serialisable(drift_run):
    json.dumps(drift_run.history())


def test_save_writes_a_readable_run(drift_run, tmp_path):
    path = drift_run.save(tmp_path / "nested" / "run.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert len(payload["observations"]) == N
    assert payload["bounds"] == [[-5.0, 5.0]]
    assert payload["ref_point"] == REF_POINT
    assert len(payload["alphas"]) == 2
    assert all(0.0 < a <= 1.0 for a in payload["alphas"])
    assert len(payload["hypervolume_trace"]) == N


def test_repr_is_informative(drift_run):
    assert "n=0" in repr(make_optimizer())
    assert f"n={N}" in repr(drift_run)
    assert "m=2" in repr(drift_run)


def test_acquisition_iterations_record_predictions(drift_run):
    acquisition = [
        o for o in drift_run.observations if not o.is_validation and o.iteration > 3
    ]
    assert acquisition
    for obs in acquisition:
        assert len(obs.predicted_y) == 2
        assert len(obs.predicted_sd) == 2


# -- stationary baseline ------------------------------------------------


def test_stationary_baseline_pins_every_alpha():
    opt = DynamicMOBO(
        bounds=BOUNDS, ref_point=REF_POINT, config=as_stationary_mo(fast_config())
    )
    opt.run(drifting_biobjective(noise=0.05, n=8), 8)
    assert opt.alphas == [1.0, 1.0]


def test_as_stationary_mo_leaves_the_original_config_untouched():
    config = fast_config()
    as_stationary_mo(config)
    assert config.model.stationary is False


# -- regressions shared with the single-objective optimiser -------------


def test_validation_never_displaces_seed_points():
    """With the validation period shorter than the seed budget, every seed
    must still be applied in order and validation must wait for them."""
    seeds = [[0.0], [3.0], [-3.0]]
    opt = make_optimizer(seed_points=seeds, validation_every=2)
    opt.run(drifting_biobjective(noise=0.05), 5)

    applied = [o.x for o in opt.observations[:3]]
    for got, expected in zip(applied, seeds, strict=True):
        assert got == pytest.approx(expected)
    assert [o.iteration for o in opt.observations if o.is_validation] == [4]


def test_refit_every_preserves_fitted_hyperparameters():
    """Iterations that skip the refit must carry every objective's fitted
    hyperparameters forward, not rebuild at factory defaults."""
    opt = make_optimizer(refit_every=3)
    objective = drifting_biobjective(noise=0.05)

    opt.run(objective, 3)
    opt._ensure_models()  # n = 3: refit
    fitted = [m.covar_module.kernels[0].base_kernel.lengthscale.detach().clone()
              for m in opt._models]
    fitted_alphas = opt.alphas

    x = opt.suggest()
    opt.observe(x, objective(x))  # n = 4: stale, no refit due
    opt._ensure_models()
    carried = [m.covar_module.kernels[0].base_kernel.lengthscale.detach()
               for m in opt._models]

    for got, expected in zip(carried, fitted, strict=True):
        torch.testing.assert_close(got, expected)
    assert opt.alphas == pytest.approx(fitted_alphas)


def test_direct_suggest_validation_records_its_own_prediction():
    """Called directly after an ordinary suggestion, suggest_validation must
    record its own flag and prediction, not leave the earlier suggestion's
    pending state to be attached to the validation point."""
    opt = make_optimizer()
    objective = drifting_biobjective(noise=0.05)
    opt.run(objective, 6)

    opt.suggest()
    x = opt.suggest_validation()
    expected = list(opt._pending["predicted_y"])
    obs = opt.observe(x, objective(x))

    assert obs.is_validation is True
    assert obs.predicted_y == pytest.approx(expected)
    assert opt.prediction_error()


def test_prediction_attached_only_to_the_suggested_input():
    opt = make_optimizer()
    opt.run(drifting_biobjective(noise=0.05), 5)

    x = opt.suggest()
    mismatched = opt.observe([x[0] + 1.0], [1.0, 1.0])
    assert mismatched.predicted_y is None
    assert mismatched.is_validation is False

    x = opt.suggest()
    assert opt.observe(x, [1.0, 1.0]).predicted_y is not None


def test_suggest_leaves_the_global_rng_untouched():
    state = torch.random.get_rng_state()
    opt = make_optimizer(seed_points=None)
    opt.run(drifting_biobjective(noise=0.05), 5)
    assert torch.equal(torch.random.get_rng_state(), state)


# -- multi-objective specifics ------------------------------------------


def test_acquisition_baseline_has_no_duplicate_inputs(monkeypatch):
    """Validation re-tests visited inputs, so once the time column is
    overwritten the raw baseline is guaranteed to contain duplicates."""
    import dbo_torch.mo_optimizer as mo_module

    real = mo_module.qLogNoisyExpectedHypervolumeImprovement
    baselines = []

    def spy(*args, **kwargs):
        baselines.append(kwargs["X_baseline"].clone())
        return real(*args, **kwargs)

    monkeypatch.setattr(mo_module, "qLogNoisyExpectedHypervolumeImprovement", spy)

    opt = make_optimizer(validation_every=4)
    opt.run(drifting_biobjective(noise=0.05), 6)

    duplicated = [o.x for o in opt.observations]
    assert len({tuple(x) for x in duplicated}) < len(duplicated)
    last = baselines[-1]
    assert torch.unique(last, dim=0).size(0) == last.size(0)


def test_validation_scores_by_upper_bound_and_half_means_posterior_mean():
    """validation_confidence = 0.5 gives a zero multiplier, so the chosen point
    is the best hypervolume contributor by posterior mean alone."""
    opt = make_optimizer(validation_confidence=0.5)
    opt.run(drifting_biobjective(noise=0.05), 8)

    model = opt._ensure_models()
    probe = opt._visited_at(opt._acquisition_time())
    means = opt._posterior_means(model, probe)
    expected = probe[opt._best_contributor(-means), :1].tolist()

    assert opt.suggest_validation() == pytest.approx(expected)


def test_resumed_run_continues_exactly(tmp_path):
    def cost(x, iteration):
        a = 4.0 * (1.0 - (iteration - 1) / (N - 1))
        return [(x[0] - a) ** 2, (x[0] + 2.0) ** 2]

    a = make_optimizer(validation_every=4)
    for _ in range(5):
        x = a.suggest()
        a.observe(x, cost(x, a.next_iteration))

    b = DynamicMOBO.load(a.save(tmp_path / "run.json"))
    assert b.history() == a.history()
    assert b.ref_point == a.ref_point

    for _ in range(2):
        xa, xb = a.suggest(), b.suggest()
        assert xb == pytest.approx(xa, abs=1e-9)
        a.observe(xa, cost(xa, a.next_iteration))
        b.observe(xb, cost(xb, b.next_iteration))
