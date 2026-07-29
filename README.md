# dbo-torch

Dynamic Bayesian Optimization on [BoTorch](https://botorch.org/) — Bayesian
optimisation for objectives that **drift while you are optimising them**.

Standard BO assumes the system responds the same way to the same input today as
it did an hour ago. Human-in-the-loop problems break that assumption routinely:
people adapt, learn, fatigue, and habituate. When the objective moves, a
stationary GP explains the mismatch as noise, widens its error bars, and keeps
recommending an input that stopped being optimal.

DBO adds one thing — the covariance between two observations decays with how
far apart in time they were taken:

$$k\big((u,t),(u',t')\big) = k_u(u,u') \cdot \alpha^{|t-t'|}, \qquad \alpha \in (0,1]$$

`α` is fitted from the data by marginal likelihood alongside the length scales
and signal variance, so the optimiser **infers how fast the system is drifting**
rather than being told. At `α = 1` the temporal factor is identically one and
the method reduces exactly to stationary BO, which makes the baseline a
configuration flag rather than a separate implementation.

---

## Install

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.10, PyTorch ≥ 2.0, BoTorch ≥ 0.11.

## Use

```python
from dbo_torch import DynamicBO, DBOConfig

opt = DynamicBO(
    bounds=[(-5.0, 9.0)],                        # one control parameter
    config=DBOConfig(
        seed_points=[[5.0], [7.0], [3.0]],       # fixed first iterations
        validation_every=10,                     # test the best estimate every 10
    ),
)

for _ in range(80):
    x = opt.suggest()                            # what to try next
    cost = run_one_trial(x)                      # your experiment
    opt.observe(x, cost)

print(f"fitted drift rate: alpha = {opt.alpha:.4f}")
```

`DynamicBO` **minimises** cost. Time is measured in iterations: the first
observation sits at `t = 1`, the second at `t = 2`, and so on.

### The stationary baseline

To compare against ordinary BO on identical data, pin `α = 1`:

```python
from dbo_torch import as_stationary

baseline = DynamicBO(bounds=[(-5.0, 9.0)], config=as_stationary(cfg))
```

Same code path, same acquisition, same everything else — the only difference is
whether the model is allowed to discount the past.

### Validation iterations

Comparing optimisers on the cost they happen to incur is unfair: an optimiser
that explores more will look worse without being worse. A validation iteration
applies the optimiser's **current best estimate** instead of an exploratory
point, which removes that confound.

```python
DBOConfig(validation_every=10)
```

Every tenth iteration then applies the input minimising an upper confidence
bound on cost, and records what the model *predicted* before the measurement
arrives:

```python
opt.prediction_error()
# [{'iteration': 10, 'predicted': 1.42, 'measured': 1.55, 'error': 0.13}, ...]
```

That error is a direct measure of model quality, independent of exploration
luck. It is the quantity that most clearly separates DBO from BO as a run
progresses.

---

## Does it work?

Three checks ship with the repo, in increasing order of looseness.

**The covariance matches MATLAB to machine precision.** `matlab_parity/`
generates reference matrices from MATLAB and compares:

```
case                       shape      max|dK|   max|dKrect|  result
1d_integer_time           (8, 8)    8.882e-16     8.882e-16  match
1d_alpha_one              (8, 8)    8.882e-16     8.882e-16  match
1d_fast_decay             (8, 8)    4.857e-17     4.857e-17  match
3d_ard                    (6, 6)    6.661e-16     6.661e-16  match
1d_fractional_time        (5, 5)    4.441e-16     4.441e-16  match
```

**The GP posterior matches MATLAB's `fitrgp`.** Same data, same pinned
hyperparameters, worst disagreement `1.6e-14` across posterior mean, latent
standard deviation, and response standard deviation — checked both at the
current time and back at `t = 1`, where the temporal kernel does its work.

**The published behaviour reproduces.** `examples/replicate_ral.py` runs the
RA-L protocol against a simulated participant whose ideal input falls from
5 Nm to 0 Nm over 80 iterations:

```
 iter |   DBO cost    BO cost |  DBO torque  BO torque |  ideal
   10 |      0.653      0.673 |       4.833      4.853 |   4.43
   20 |      0.393      0.754 |       4.447      4.808 |   3.80
   ...
   70 |      1.500      2.415 |       1.888      2.803 |   0.63
   80 |      1.365      2.308 |       1.159      2.102 |   0.00
```

Indistinguishable early, separating late — the reported pattern. The paper
gives final applied torque as 1.2 ± 0.33 Nm for DBO against 2.6 ± 0.25 Nm for
BO; this simulation gives 1.16 against 2.10.

See [matlab_parity/SETUP.md](matlab_parity/SETUP.md) to run the MATLAB side,
including which toolbox you need and how to install the reference
implementation without overwriting your MATLAB.

---

## Configuration

Everything lives in `DBOConfig` and `DBOModelConfig`. The settings that change
results most:

| Setting | Default | What it does |
|---|---|---|
| `validation_every` | `None` | Apply the best estimate every N iterations. |
| `validation_visited_only` | `True` | Restrict the best estimate to already-tried inputs. Matches the reference implementation; set `False` to search the full domain, which tracks fast drift better. |
| `exploration_ratio` | `0.1` | Re-searches with inflated variance when the acquisition function collapses onto a point the model is already sure about. `0` disables. |
| `acquisition_time_offset` | `0.0` | `0` scores candidates at the current time, as the reference does. `1` scores them at the time they will actually be evaluated, removing a one-step lag. |
| `model.stationary` | `False` | Pin `α = 1` for the BO baseline. |
| `model.alpha_parameterization` | `"decay"` | `"decay"` fits `1 − α` in log space, reproducing the reference. `"direct"` fits `α` under an interval constraint and behaves better when drift is fast. |
| `model.spatial_kernel` | `"rbf"` | Squared exponential with ARD, as the reference DBO kernel uses. `"matern52"` also available. |

`DBOModelConfig.matlab_compatible()` presets the transforms MATLAB does not
apply, for parity runs.

---

## Multi-objective: `DynamicMOBO`

The multi-objective extension gives **each objective its own GP with its own
drift rate**: discomfort can habituate quickly while task time drifts slowly,
and a single global `α` cannot express that. Acquisition is
qLogNEHVI with the baseline points' time coordinates overwritten to *now*, so
the Pareto front being improved is the currently predicted one rather than the
stale observed one — the multi-objective analogue of DBO's posterior-based
incumbent.

```python
from dbo_torch import DynamicMOBO, MODBOConfig

opt = DynamicMOBO(
    bounds=[(-5.0, 5.0)],
    ref_point=[25.0, 25.0],          # worst acceptable value per objective
    config=MODBOConfig(validation_every=10),
)
for _ in range(40):
    x = opt.suggest()
    opt.observe(x, [measure_discomfort(x), measure_time(x)])

print(opt.alphas)                    # one fitted drift rate per objective
front = opt.pareto_front(at_current_time=True)   # the drift-adjusted front
```

Everything minimises, matching `DynamicBO`. On a benchmark with one drifting
and one stationary objective, the fitted rates separate exactly as they
should: `alphas = [0.996, 0.99999992]`. Under drift,
`pareto_front(at_current_time=True)` and the raw observed front genuinely
differ — the raw front retains designs whose measured values are no longer
attainable, and the drift-adjusted one drops them.

## Unity

This repository is the pure Python library. The Unity integration lives in
[Bayesian Optimization for Unity](https://github.com/Pascal-Jansen/Bayesian-Optimization-for-Unity),
where DBO is a selectable backend alongside BoTorch, CABOP and MetaTAF — see
`docs/dbo-backend.md` there. That project vendors a snapshot of this package
under `Assets/StreamingAssets/BOData/BayesianOptimization/dbo_torch/`.

---

## When to reach for this

DBO helps when the objective genuinely moves during a session — motor
adaptation, learning effects, fatigue, habituation, or an experimental
manipulation that shifts the optimum on purpose. The cost is one extra
hyperparameter and a GP that discounts its own history.

It does **not** help when the objective is stationary and merely noisy. There,
`α` fits close to 1 and you have paid for nothing; ordinary BO is the right
tool. If you are unsure, run both — that is what `as_stationary` is for — and
look at the fitted `α`.

A caveat worth stating: `α` is a single global drift rate. If different regions
of your input space drift at different rates, or the drift is endogenous
(driven by the exposure the optimiser itself produces, rather than by time),
this model is misspecified in a way that a fitted `α` will not reveal. It will
still usually beat a stationary model, but the fit is doing something cruder
than it appears.

---

## Citation

Cite the method papers, not this repository alone. The authors' stated
preference is that the primary citation is the computational paper:

```bibtex
@article{kim2025dbo,
  author  = {Kim, GilHwan and Sergi, Fabrizio},
  title   = {Dynamic {Bayesian} optimization for non-stationary systems},
  journal = {Computer Methods in Biomechanics and Biomedical Engineering},
  year    = {2025},
  doi     = {10.1080/10255842.2025.2595150}
}

@article{kim2026validation,
  author  = {Kim, GilHwan and Sergi, Fabrizio},
  title   = {Validation of Dynamic {Bayesian} Optimization for a Non-Stationary
             Human-in-the-Loop Optimization Problem},
  journal = {IEEE Robotics and Automation Letters},
  volume  = {11}, number = {5}, pages = {5733--5740}, year = {2026},
  doi     = {10.1109/LRA.2026.3665072}
}
```

## Licence and provenance

BSD 3-Clause. This is an independent implementation written from the published
method; it contains no MathWorks code. That distinction matters, because the
reference implementation is a patched copy of a proprietary MATLAB toolbox file
and cannot be redistributed. [PROVENANCE.md](PROVENANCE.md) sets out exactly
what is derived from what, and why this repository is safe to publish.
