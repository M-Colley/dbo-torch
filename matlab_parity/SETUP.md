# Running the MATLAB reference for comparison

You only need this if you want to check the Python implementation against the
original MATLAB one. `dbo-torch` itself has no MATLAB dependency.

## 1. Install the Statistics and Machine Learning Toolbox

The reference DBO code is a patched copy of `BayesianOptimization.m` from that
toolbox, and it calls `fitrgp` and `bayesopt`, which both live there. Base
MATLAB is not enough.

Check what you have:

```matlab
exist('fitrgp'), exist('bayesopt')
```

Two zeros means the toolbox is missing. Two 2s means you are ready.

To install it into an existing MATLAB:

**From inside MATLAB** — Home tab → Add-Ons → Get Add-Ons → search
"Statistics and Machine Learning Toolbox" → Install. Sign in with the
MathWorks Account attached to your licence.

**From the installer**, if the Add-On route refuses (it sometimes does for
older releases) — download the R2019b installer from
<https://www.mathworks.com/downloads> under "Download previous releases", run
`setup.exe`, sign in, select your licence, and **point it at your existing
installation folder** (`C:\Program Files\MATLAB\R2019b`). Tick only
*Statistics and Machine Learning Toolbox*. It adds to the existing install
rather than replacing it.

Nothing else is required. The Optimization Toolbox is not needed, and the
Parallel Computing Toolbox must **not** be engaged — the reference DBO code
only works with the parallel option off.

## 2. Install the reference DBO code without damaging your MATLAB

The File Exchange instructions tell you to overwrite
`matlabroot/toolbox/stats/bayesoptim/BayesianOptimization.m` as administrator.
That works, but it silently changes the behaviour of `bayesopt` for every
project on the machine, and an installer repair will revert it without warning.

Prefer shadowing the file instead. MATLAB resolves classes by path order, so a
copy earlier on the path wins, and nothing in your installation is touched:

```matlab
% Put the downloaded, unmodified DBO BayesianOptimization.m in its own folder,
% e.g. C:\Users\Mark\Desktop\DBO\matlab_dbo\
addpath('C:\Users\Mark\Desktop\DBO\matlab_dbo', '-begin');
clear classes                     %#ok<CLCLS>  - forces the class to reload
which BayesianOptimization -all   % the DBO copy should be listed first
```

Run `rmpath` to go back to stock MATLAB. Do not save this path permanently;
add it at the top of the parity script instead, so it is always explicit which
version you are running.

If you do choose to overwrite the toolbox file, back the original up first:

```matlab
src = fullfile(matlabroot,'toolbox','stats','bayesoptim','BayesianOptimization.m');
copyfile(src, [src '.orig']);
```

**Do not commit the DBO `BayesianOptimization.m` to this repository.** It is
MathWorks-copyrighted and is listed in `.gitignore`. See `PROVENANCE.md`.

## 3. Switching between DBO and BO

The reference implementation selects the mode by commenting a line in the
custom kernel — the line that overrides `alpha` with a constant `1`. With
`alpha` free, you get DBO; pinned at 1, the temporal factor vanishes and you
get stock BO.

In this package the equivalent is a flag, so both can run in one session:

```python
DBOConfig(model=DBOModelConfig(stationary=False))  # DBO
DBOConfig(model=DBOModelConfig(stationary=True))   # BO baseline
# or: as_stationary(cfg)
```

## 4. Run the comparison

Three tiers, each a MATLAB generator paired with a Python checker. Tier 1
(`kernel_reference.m`) needs only base MATLAB; Tiers 2 (`gp_reference.m`) and
3 (`fit_reference.m`) need the Statistics and Machine Learning Toolbox from
step 1.

```bash
matlab -batch "cd('matlab_parity'); kernel_reference; gp_reference; fit_reference"
python matlab_parity/compare_kernel.py
python matlab_parity/compare_gp.py
python matlab_parity/compare_fit.py
```

If `matlab` is not on your PATH (a default Windows install does not add it),
use the full executable path, e.g.
`"C:\Program Files\MATLAB\R2019b\bin\matlab.exe" -batch ...`.

The generators write `out/matlab_kernel.json`, `out/matlab_gp.json` and
`out/matlab_fit.json`; each compare script accepts an optional path to the
JSON and defaults to those locations. `compare_kernel.py` rebuilds the
covariance matrices with the GPyTorch kernel at pinned hyperparameters;
`compare_gp.py` rebuilds the full GP posteriors against MATLAB's `fitrgp`,
probing both at the current time and back at t = 1, where the temporal kernel
does the work that distinguishes DBO from stationary BO. `compare_fit.py`
checks fitting: `fit_reference.m` fits five data sets the way the reference
implementation does, and the Python side must start from the same point,
reproduce MATLAB's likelihood at MATLAB's optimum, and fit to an optimum at
least as good.

## What agreement to expect

Tiers 1 and 2 are deterministic linear algebra with pinned hyperparameters —
no fitting, no acquisition, no randomness — so the two sides must agree to
near machine precision. Anything worse than the tolerances below is a
structural difference, not rounding:

| Quantity | Expected agreement |
|---|---|
| Kernel matrix for fixed hyperparameters | ~1e-12 (pure arithmetic) |
| Posterior mean and sd for fixed hyperparameters | ~1e-8 (one Cholesky solve apart) |
| Fitting starting point (`matlab_compatible()`) | ~1e-10 (pure arithmetic) |
| Log likelihood at MATLAB's fitted hyperparameters | ~1e-8 relative |
| Python's fitted log likelihood | no worse than MATLAB's, within 1e-3 |

Tier 3 deliberately does not require the fitted hyperparameters to agree.
MATLAB and BoTorch use different optimisers over a genuinely multi-modal
likelihood surface, and a check that demanded equal optima would mix optimiser
variance into a correctness statement. Instead it splits into a deterministic
part (same start, same likelihood function) and a one-sided quality part
(Python's optimum is at least as good), either of which a real porting error
would break. In practice the optima coincide: on the shipped cases the fitted
likelihoods agree to four decimals and the fitted `alpha` to five.

Still out of scope: per-iteration selected inputs and full optimisation
trajectories, which depend on acquisition optimisers and random restarts.
Behavioural reproduction is demonstrated separately by
`examples/replicate_ral.py`.
