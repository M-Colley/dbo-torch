"""Tier 1 parity check: does the Python covariance match MATLAB's, exactly?

Loads covariance matrices produced by ``kernel_reference.m`` at fixed
hyperparameters and rebuilds them with the GPyTorch kernel from this package.
No fitting is involved on either side, so the two must agree to machine
precision. A failure here is a real bug in the kernel — a wrong sign, a
transposed index, a missing factor of a half — and not the kind of harmless
divergence you get from two different optimisers.

This check needs only base MATLAB. The Statistics and Machine Learning Toolbox
is required for Tiers 2 and 3, not for this one.

    matlab -batch "kernel_reference"
    python compare_kernel.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dbo_torch.kernels import TemporalDecayKernel  # noqa: E402

DEFAULT_REF = Path(__file__).parent / "out" / "matlab_kernel.json"

# Threshold for "these are the same matrix". Both sides do the same handful of
# floating-point operations in double precision, so anything above this is a
# structural difference, not rounding.
TOL = 1e-12


def build_kernel(d: int, lengthscale, sigma_f: float, alpha: float):
    """Assemble the DBO covariance with hyperparameters pinned to given values."""
    spatial = ScaleKernel(RBFKernel(ard_num_dims=d, active_dims=list(range(d))))
    temporal = TemporalDecayKernel(active_dims=[d])

    ls = torch.as_tensor(lengthscale, dtype=torch.float64).reshape(1, -1)
    if ls.numel() == 1 and d > 1:
        ls = ls.expand(1, d).contiguous()

    spatial.base_kernel.lengthscale = ls
    spatial.outputscale = torch.tensor(sigma_f**2, dtype=torch.float64)
    temporal.alpha = torch.tensor(alpha, dtype=torch.float64)

    return (spatial * temporal).double()


def main(ref_path: Path = DEFAULT_REF) -> int:
    torch.set_default_dtype(torch.float64)

    if not ref_path.exists():
        print(f"Reference file not found: {ref_path}")
        print('Generate it first with:  matlab -batch "kernel_reference"')
        return 2

    payload = json.loads(ref_path.read_text(encoding="utf-8"))
    cases = payload["cases"]
    if isinstance(cases, dict):  # a single case is not wrapped in a list
        cases = [cases]

    print(f"MATLAB {payload.get('matlab_version', '?')}")
    print(f"{len(cases)} case(s) from {ref_path.name}\n")
    print(f"{'case':<22} {'shape':>9} {'max|dK|':>12} {'max|dKrect|':>13}  result")
    print("-" * 68)

    worst = 0.0
    failures = 0

    for case in cases:
        X = torch.tensor(np.atleast_2d(np.array(case["X"], dtype=float)))
        Xb = torch.tensor(np.atleast_2d(np.array(case["Xb"], dtype=float)))
        K_ref = np.atleast_2d(np.array(case["K"], dtype=float))
        Krect_ref = np.atleast_2d(np.array(case["Krect"], dtype=float))

        d = X.shape[-1] - 1
        kernel = build_kernel(d, case["lengthscale"], case["sigmaF"], case["alpha"])

        with torch.no_grad():
            K = kernel(X, X).to_dense().numpy()
            Krect = kernel(X, Xb).to_dense().numpy()

        dK = float(np.max(np.abs(K - K_ref)))
        dR = float(np.max(np.abs(Krect - Krect_ref)))
        worst = max(worst, dK, dR)

        ok = dK < TOL and dR < TOL
        failures += not ok
        print(
            f"{case['name']:<22} {str(K.shape):>9} {dK:12.3e} {dR:13.3e}  "
            f"{'match' if ok else 'MISMATCH'}"
        )

    print("-" * 68)
    print(f"worst disagreement: {worst:.3e}   (tolerance {TOL:.0e})")

    if failures:
        print(f"\n{failures} case(s) disagree. The kernels are not the same function.")
        return 1

    print("\nAll cases agree to machine precision.")
    return 0


if __name__ == "__main__":
    ref = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REF
    raise SystemExit(main(ref))
