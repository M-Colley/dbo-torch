"""Tier 2 parity check: does the Python GP posterior match MATLAB's `fitrgp`?

Rebuilds the exact models emitted by ``gp_reference.m`` — same data, same
pinned hyperparameters, same zero mean function — and compares posterior means
and standard deviations. Nothing is fitted on either side, so this is a
comparison of linear algebra, not of optimisers, and the two should agree to
near machine precision.

Test points are evaluated both at the latest time (where acquisition and
validation-iteration selection operate) and back at time 1 (where the temporal
kernel does the work that separates DBO from stationary BO). A bug in how time
enters the model shows up in the second set and not the first.

    matlab -batch "gp_reference"
    python compare_gp.py
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
from gpytorch.means import ZeroMean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dbo_torch.kernels import TemporalDecayKernel  # noqa: E402

DEFAULT_REF = Path(__file__).parent / "out" / "matlab_gp.json"

# Looser than the Tier 1 kernel tolerance: this involves a Cholesky solve, so
# conditioning costs a few digits. Still far tighter than any difference that
# could come from a modelling discrepancy.
TOL = 1e-8


def build_pinned_gp(X, Y, lengthscale, sigma_f, alpha, sigma):
    """A DBO GP with every hyperparameter pinned and a zero mean function."""
    d = X.shape[-1] - 1

    spatial = ScaleKernel(RBFKernel(ard_num_dims=d, active_dims=list(range(d))))
    temporal = TemporalDecayKernel(active_dims=[d])

    spatial.base_kernel.lengthscale = torch.as_tensor(
        lengthscale, dtype=torch.float64
    ).reshape(1, -1)
    spatial.outputscale = torch.tensor(sigma_f**2, dtype=torch.float64)
    temporal.alpha = torch.tensor(alpha, dtype=torch.float64)

    likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-12))
    likelihood.noise = torch.tensor(sigma**2, dtype=torch.float64)

    model = SingleTaskGP(
        train_X=X,
        train_Y=Y,
        mean_module=ZeroMean(),
        covar_module=spatial * temporal,
        likelihood=likelihood,
        # MATLAB was told Standardize=false, so no transforms here either.
        outcome_transform=None,
        input_transform=None,
    )
    model.eval()
    return model


def posterior(model, Xs, observation_noise: bool):
    with torch.no_grad():
        post = model.posterior(Xs, observation_noise=observation_noise)
        mu = post.mean.squeeze(-1).numpy()
        sd = post.variance.clamp_min(0).sqrt().squeeze(-1).numpy()
    return mu, sd


def main(ref_path: Path = DEFAULT_REF) -> int:
    torch.set_default_dtype(torch.float64)

    if not ref_path.exists():
        print(f"Reference file not found: {ref_path}")
        print('Generate it first with:  matlab -batch "gp_reference"')
        return 2

    payload = json.loads(ref_path.read_text(encoding="utf-8"))
    cases = payload["cases"]
    if isinstance(cases, dict):
        cases = [cases]

    print(f"MATLAB {payload.get('matlab_version', '?')}")
    print(f"{len(cases)} case(s) from {ref_path.name}\n")
    header = (
        f"{'case':<16} {'alpha':>6} {'d(mean)':>11} {'d(sd_lat)':>11} "
        f"{'d(sd_resp)':>11} {'d(mean@t=1)':>12}  result"
    )
    print(header)
    print("-" * len(header))

    worst = 0.0
    failures = 0

    for case in cases:
        X = torch.tensor(np.atleast_2d(np.array(case["X"], dtype=float)))
        Y = torch.tensor(np.array(case["Y"], dtype=float).reshape(-1, 1))
        Xs = torch.tensor(np.atleast_2d(np.array(case["Xs"], dtype=float)))
        XsPast = torch.tensor(np.atleast_2d(np.array(case["XsPast"], dtype=float)))

        model = build_pinned_gp(
            X, Y,
            np.array(case["lengthscale"], dtype=float).reshape(-1),
            float(case["sigmaF"]),
            float(case["alpha"]),
            float(case["sigma"]),
        )

        mu, sd_latent = posterior(model, Xs, observation_noise=False)
        _, sd_resp = posterior(model, Xs, observation_noise=True)
        mu_past, _ = posterior(model, XsPast, observation_noise=False)

        ref = {k: np.array(case[k], dtype=float).reshape(-1) for k in
               ("mu", "sd_latent", "sd_response", "mu_past")}

        d_mu = float(np.max(np.abs(mu - ref["mu"])))
        d_lat = float(np.max(np.abs(sd_latent - ref["sd_latent"])))
        d_res = float(np.max(np.abs(sd_resp - ref["sd_response"])))
        d_past = float(np.max(np.abs(mu_past - ref["mu_past"])))

        worst = max(worst, d_mu, d_lat, d_res, d_past)
        ok = max(d_mu, d_lat, d_res, d_past) < TOL
        failures += not ok

        print(
            f"{case['name']:<16} {case['alpha']:>6.2f} {d_mu:11.3e} {d_lat:11.3e} "
            f"{d_res:11.3e} {d_past:12.3e}  {'match' if ok else 'MISMATCH'}"
        )

    print("-" * len(header))
    print(f"worst disagreement: {worst:.3e}   (tolerance {TOL:.0e})")

    if failures:
        print(f"\n{failures} case(s) disagree.")
        return 1

    print("\nPosteriors agree. The Python GP is the same model MATLAB fits.")
    return 0


if __name__ == "__main__":
    ref = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REF
    raise SystemExit(main(ref))
