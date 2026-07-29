"""Reproduce the qualitative finding of the RA-L validation study, in simulation.

The study drove a hip exoskeleton's torque amplitude to hit a target hip
extension angle, while secretly ramping treadmill speed so that the torque
needed to hit that target fell steadily toward zero. Neither optimiser was told
the speed was changing. Every tenth iteration was a *validation iteration*: the
optimiser applied its current best estimate rather than an exploratory point,
so the two could be compared without their different exploration policies
confounding the result.

The reported outcome was that DBO and BO were indistinguishable early on and
diverged late, with DBO reaching lower cost and lower applied torque by
iterations 70 and 80.

This script reproduces that structure against a simulated participant. It is
not a reproduction of the human data — it checks that the optimiser behaves as
described when the objective drifts underneath it.

    python examples/replicate_ral.py --seeds 5
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dbo_torch import DBOConfig, DynamicBO, as_stationary  # noqa: E402

N_ITERATIONS = 80
VALIDATION_EVERY = 10
TORQUE_BOUNDS = [(-5.0, 9.0)]          # Nm, flexion to extension
SEED_TORQUES = [[5.0], [7.0], [3.0]]   # the study's first three iterations


def make_participant(seed: int, noise: float = 0.35):
    """A simulated participant whose ideal torque falls from 5 Nm to 0 Nm.

    Cost is the absolute gap between target and achieved hip extension, which
    is what the study minimised. That absolute value puts a kink at the
    optimum — the cost function is not smooth there — which is exactly the
    awkwardness the paper flags as limiting GP accuracy. Keeping it here means
    the simulation exercises the same difficulty.
    """
    gen = torch.Generator().manual_seed(seed)
    state = {"i": 0}

    def objective(x: list[float]) -> float:
        state["i"] += 1
        progress = (state["i"] - 1) / (N_ITERATIONS - 1)
        ideal = 5.0 * (1.0 - progress)
        measured = abs(x[0] - ideal)
        return measured + noise * float(torch.randn(1, generator=gen))

    return objective


def run_one(
    seed: int, stationary: bool
) -> tuple[dict[int, tuple[float, float]], float | None]:
    cfg = DBOConfig(
        seed_points=SEED_TORQUES,
        validation_every=VALIDATION_EVERY,
        exploration_ratio=0.1,
        seed=seed,
        num_restarts=5,
        raw_samples=128,
    )
    if stationary:
        cfg = as_stationary(cfg)

    opt = DynamicBO(bounds=TORQUE_BOUNDS, config=cfg)
    opt.run(make_participant(1000 + seed), N_ITERATIONS)

    return {
        o.iteration: (o.y, o.x[0])
        for o in opt.observations
        if o.is_validation
    }, opt.alpha


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)

    results = {"DBO": {}, "BO": {}}
    alphas = []
    started = time.time()

    for seed in range(args.seeds):
        for name, stationary in (("DBO", False), ("BO", True)):
            trace, alpha = run_one(seed, stationary)
            if name == "DBO":
                alphas.append(alpha)
            for iteration, value in trace.items():
                results[name].setdefault(iteration, []).append(value)
        print(
            f"  seed {seed + 1}/{args.seeds} done "
            f"({time.time() - started:.0f}s elapsed)",
            flush=True,
        )

    print(f"\nSimulated participants: {args.seeds}")
    print(f"Mean fitted alpha (DBO): {statistics.mean(alphas):.4f}\n")

    print(f"{'iter':>5} | {'DBO cost':>10} {'BO cost':>10} | "
          f"{'DBO torque':>11} {'BO torque':>10} | {'ideal':>6}")
    print("-" * 68)

    for iteration in sorted(results["DBO"]):
        dbo, bo = results["DBO"][iteration], results["BO"][iteration]
        ideal = 5.0 * (1.0 - (iteration - 1) / (N_ITERATIONS - 1))
        print(
            f"{iteration:5d} | "
            f"{statistics.mean(c for c, _ in dbo):10.3f} "
            f"{statistics.mean(c for c, _ in bo):10.3f} | "
            f"{statistics.mean(t for _, t in dbo):11.3f} "
            f"{statistics.mean(t for _, t in bo):10.3f} | "
            f"{ideal:6.2f}"
        )

    late = [70, 80]
    d_late = statistics.mean(
        c for i in late for c, _ in results["DBO"].get(i, [])
    )
    b_late = statistics.mean(
        c for i in late for c, _ in results["BO"].get(i, [])
    )
    print("-" * 68)
    print(
        f"\nLate-stage mean cost (iterations 70 and 80): "
        f"DBO {d_late:.3f} vs BO {b_late:.3f}"
    )
    print(
        "DBO lower, as reported."
        if d_late < b_late
        else "DBO not lower here — expected with few seeds; raise --seeds."
    )

    if args.out:
        args.out.write_text(
            "\n".join(
                f"{i},{statistics.mean(c for c, _ in results['DBO'][i])},"
                f"{statistics.mean(c for c, _ in results['BO'][i])}"
                for i in sorted(results["DBO"])
            ),
            encoding="utf-8",
        )
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
