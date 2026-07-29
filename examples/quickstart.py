"""Smallest useful DBO run: a drifting optimum, DBO against stationary BO.

    python examples/quickstart.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dbo_torch import DBOConfig, DynamicBO, as_stationary  # noqa: E402

N = 40


def participant(seed: int):
    """Ideal input slides from 5 to 0 while the optimiser works."""
    gen = torch.Generator().manual_seed(seed)
    state = {"i": 0}

    def cost(x: list[float]) -> float:
        state["i"] += 1
        ideal = 5.0 * (1.0 - (state["i"] - 1) / (N - 1))
        return abs(x[0] - ideal) + 0.2 * float(torch.randn(1, generator=gen))

    return cost


def main() -> int:
    torch.set_default_dtype(torch.float64)

    config = DBOConfig(
        seed_points=[[5.0], [7.0], [3.0]],
        validation_every=10,
        seed=0,
        num_restarts=5,
        raw_samples=128,
    )

    for label, cfg in (("DBO", config), ("BO ", as_stationary(config))):
        opt = DynamicBO(bounds=[(-5.0, 9.0)], config=cfg)
        opt.run(participant(7), N)

        final = [o for o in opt.observations if o.is_validation][-1]
        print(
            f"{label}  alpha={opt.alpha:.4f}  "
            f"final validation: input={final.x[0]:+.2f} cost={final.y:.2f}"
        )

    print("\nThe true optimum is 0.00 by the final iteration.")
    print("DBO should sit closer to it; BO stays anchored to where the optimum was.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
