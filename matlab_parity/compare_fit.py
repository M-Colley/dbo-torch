"""Tier 3 parity check: does fitting start where MATLAB's does, and end as well?

Reads the fits emitted by ``fit_reference.m``, which reproduces how the
reference implementation fits its GP, and makes three checks per case:

0. **Starting point** (exact). ``DBOModelConfig.matlab_compatible()`` must
   start from the same lengthscales, signal SD and noise floor as MATLAB.
1. **Likelihood agreement** (tight). The log marginal likelihood MATLAB
   reports at its optimum is re-evaluated here at the same hyperparameters,
   mean included. This is deterministic linear algebra.
2. **Optimum quality** (one-sided). The same data are fitted here from the
   same starting point; the resulting likelihood must be no worse than
   MATLAB's. The hyperparameters need not agree — the surface is multi-modal
   and the two optimisers differ — but a port that lands in worse optima fails.

    matlab -batch "fit_reference"
    python compare_fit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from botorch.models import SingleTaskGP
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dbo_torch.kernels import TemporalDecayKernel  # noqa: E402
from dbo_torch.model import DBOModelConfig, build_model, fit_model, get_alpha  # noqa: E402

DEFAULT_REF = Path(__file__).parent / "out" / "matlab_fit.json"

START_TOL = 1e-10  # relative; the starting point is plain arithmetic
LL_TOL = 1e-8  # relative; one Cholesky solve apart
QUALITY_TOL = 1e-3  # nats; Python may not land more than this below MATLAB


def log_likelihood(model) -> float:
    """Total (not per-observation) exact log marginal likelihood."""
    model.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    with torch.no_grad():
        value = mll(model(*model.train_inputs), model.train_targets)
    return float(value) * model.train_targets.numel()


def pinned_model(X, Y, case) -> SingleTaskGP:
    """MATLAB's fitted GP rebuilt with every hyperparameter pinned."""
    d = X.shape[-1] - 1
    spatial = ScaleKernel(RBFKernel(ard_num_dims=d, active_dims=list(range(d))))
    temporal = TemporalDecayKernel(active_dims=[d])
    spatial.base_kernel.lengthscale = torch.as_tensor(
        np.atleast_1d(case["lengthscale"]), dtype=torch.float64
    ).reshape(1, -1)
    spatial.outputscale = torch.tensor(float(case["sigmaF"]) ** 2)
    temporal.alpha = torch.tensor(float(case["alpha"]))

    likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-12))
    likelihood.noise = torch.tensor(float(case["sigma"]) ** 2)
    mean = ConstantMean()
    mean.constant = torch.tensor(float(np.atleast_1d(case["beta"])[0]))

    return SingleTaskGP(
        train_X=X,
        train_Y=Y,
        mean_module=mean,
        covar_module=spatial * temporal,
        likelihood=likelihood,
        outcome_transform=None,
        input_transform=None,
    )


def relative_gap(a: float, b: float) -> float:
    return abs(a - b) / max(1.0, abs(b))


def main(ref_path: Path = DEFAULT_REF) -> int:
    torch.set_default_dtype(torch.float64)

    if not ref_path.exists():
        print(f"Reference file not found: {ref_path}")
        print('Generate it first with:  matlab -batch "fit_reference"')
        return 2

    payload = json.loads(ref_path.read_text(encoding="utf-8"))
    cases = payload["cases"]
    if isinstance(cases, dict):
        cases = [cases]

    print(f"MATLAB {payload.get('matlab_version', '?')}")
    print(f"{len(cases)} case(s) from {ref_path.name}\n")
    header = (
        f"{'case':<14} {'start':>8} {'LL matlab':>11} {'d(LL)':>9} "
        f"{'LL python':>11} {'py - ml':>9} {'alpha ml':>9} {'alpha py':>9}  result"
    )
    print(header)
    print("-" * len(header))

    failures = 0
    for case in cases:
        X = torch.tensor(np.atleast_2d(np.array(case["X"], dtype=float)))
        Y = torch.tensor(np.array(case["Y"], dtype=float).reshape(-1, 1))
        bounds = torch.tensor(
            np.stack([np.atleast_1d(case["LB"]), np.atleast_1d(case["UB"])]).astype(float)
        )

        # 0. Starting point.
        start = build_model(X, Y, DBOModelConfig.matlab_compatible(), bounds=bounds)
        theta0 = np.atleast_1d(np.array(case["theta0"], dtype=float))
        ours = np.concatenate([
            start.covar_module.kernels[0].base_kernel.lengthscale.detach().reshape(-1).numpy(),
            [start.covar_module.kernels[0].outputscale.item() ** 0.5],
            [1.0 - start.covar_module.kernels[1].alpha.item()],
        ])
        start_gap = float(np.max(np.abs(ours - np.exp(theta0)) / np.exp(theta0)))
        floor = start.likelihood.noise_covar.raw_noise_constraint.lower_bound.item() ** 0.5
        floor_gap = relative_gap(floor, float(case["sigma_lower_bound0"]))
        start_ok = max(start_gap, floor_gap) < START_TOL

        # 1. Likelihood at MATLAB's optimum.
        ll_matlab = float(case["log_likelihood"])
        ll_at_matlab = log_likelihood(pinned_model(X, Y, case))
        ll_gap = relative_gap(ll_at_matlab, ll_matlab)
        ll_ok = ll_gap < LL_TOL

        # 2. Python's own fit from the same start.
        fitted = fit_model(start)
        ll_python = log_likelihood(fitted)
        quality_ok = ll_python >= ll_matlab - QUALITY_TOL

        ok = start_ok and ll_ok and quality_ok
        failures += not ok
        print(
            f"{case['name']:<14} {max(start_gap, floor_gap):8.1e} {ll_matlab:11.4f} "
            f"{ll_gap:9.1e} {ll_python:11.4f} {ll_python - ll_matlab:+9.4f} "
            f"{float(case['alpha']):9.5f} {get_alpha(fitted):9.5f}  "
            f"{'match' if ok else 'MISMATCH'}"
        )
        if int(case.get("attempts", 1)) > 1:
            print(f"  (MATLAB needed {case['attempts']} attempts; its floor was raised)")

    print("-" * len(header))
    print(
        "start: worst relative gap in lengthscales, sigmaF, decay and noise floor; "
        "d(LL): relative gap at MATLAB's optimum; py - ml: Python's fitted LL "
        "minus MATLAB's (must be >= "
        f"-{QUALITY_TOL:g})."
    )

    if failures:
        print(f"\n{failures} case(s) failed.")
        return 1

    print(
        "\nSame starting point, same likelihood, and Python's optimum is at least "
        "as good as MATLAB's in every case."
    )
    return 0


if __name__ == "__main__":
    ref = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REF
    raise SystemExit(main(ref))
