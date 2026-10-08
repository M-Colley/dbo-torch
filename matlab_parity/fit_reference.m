function fit_reference(outfile)
%FIT_REFERENCE  Emit hyperparameters fitted by fitrgp the way reference DBO fits them.
%
%   Tier 3 of the parity harness. Tiers 1 and 2 pin every hyperparameter;
%   this tier lets fitrgp fit them, from the starting point and under the
%   settings the reference implementation uses, and exports the result for
%   compare_fit.py. That script makes checks of different strength:
%
%     0. Starting point (exact). Python's matlab_compatible() preset must start
%        from the same lengthscales, signal SD and noise floor as below.
%     1. Likelihood agreement (tight). The log marginal likelihood fitrgp
%        reports at its optimum is re-evaluated in Python at the same
%        hyperparameters. That is deterministic linear algebra.
%     2. Optimum quality (one-sided). Python fits the same data from the same
%        starting point; its likelihood must be no worse than MATLAB's. The
%        hyperparameters themselves need not agree - the surface is
%        multi-modal - but a port that lands in worse optima fails.
%
%   The fit mirrors the reference: separable kernel with
%   theta = log([l_1..l_D; sigmaF; 1 - alpha]); initial theta from half the
%   domain width, std(Y)/sqrt(2) and a decay rate of 0.01; fitrgp's defaults
%   otherwise (constant basis, no standardisation); exact fitting; a noise
%   floor of 1% of std(Y), at least 1e-6; and, on failure, the floor doubled
%   and the starting decay rate raised by half, up to ten times.
%
%   Requires the Statistics and Machine Learning Toolbox.
%
%   Usage:
%       fit_reference                     % writes out/matlab_fit.json

if nargin < 1 || isempty(outfile)
    outfile = fullfile(fileparts(mfilename('fullpath')), 'out', 'matlab_fit.json');
end
outdir = fileparts(outfile);
if ~isempty(outdir) && ~exist(outdir, 'dir')
    mkdir(outdir);
end

if exist('fitrgp', 'file') ~= 2
    error('fit_reference:missingToolbox', ...
        ['fitrgp not found. Install the Statistics and Machine Learning ' ...
         'Toolbox - see SETUP.md.']);
end

% Data are generated here and exported, so the two sides never need to agree
% on a random number generator.
rng(0, 'twister');
cases = {};

% --- The RA-L shape at three study lengths: optimum drifts from 5 to 0 ----
for n = [10, 25, 40]
    T = (1:n)';
    U = -5 + 14*rand(n, 1);
    Y = abs(U - (5 - 5*(T-1)/(n-1))) + 0.2*randn(n, 1);
    cases{end+1} = struct('name', sprintf('1d_drift_n%d', n), 'X', [U, T], ...
                          'Y', Y, 'LB', -5, 'UB', 9); %#ok<AGROW>
end

% --- No drift: alpha should fit close to 1 --------------------------------
n = 25;
T = (1:n)';
U = -5 + 14*rand(n, 1);
Y = abs(U - 2.5) + 0.2*randn(n, 1);
cases{end+1} = struct('name', '1d_flat_n25', 'X', [U, T], 'Y', Y, ...
                      'LB', -5, 'UB', 9);

% --- Three parameters on very different scales, with ARD -----------------
n = 30;
T = (1:n)';
LB = [-5, 10, 50];
UB = [9, 25, 80];
U = LB + (UB - LB) .* rand(n, 3);
Y = abs(U(:,1) - (5 - 5*(T-1)/(n-1))) + 0.05*abs(U(:,2) - 18) + 0.2*randn(n, 1);
cases{end+1} = struct('name', '3d_drift_n30', 'X', [U, T], 'Y', Y, ...
                      'LB', LB, 'UB', UB);

results = cell(1, numel(cases));

for c = 1:numel(cases)
    s = cases{c};
    D = size(s.X, 2) - 1;
    sdY = std(s.Y);

    sigmaF0 = sdY / sqrt(2);
    if isnan(sigmaF0) || sigmaF0 == 0
        sigmaF0 = 1;
    end
    theta0 = log([(s.UB(:) - s.LB(:)) / 2; sigmaF0; 0.01]);
    sigmaLB0 = max(1e-6, max(1e-8, 0.01*sdY));

    [gpr, attempts, sigmaLB] = fit_like_reference(s.X, s.Y, theta0, sigmaLB0);

    theta = gpr.KernelInformation.KernelParameters;
    params = exp(theta);
    alpha = 1 - params(D+2);
    if alpha <= 0
        alpha = 1e-9;
    end

    results{c} = struct( ...
        'name', s.name, 'X', s.X, 'Y', s.Y, 'LB', s.LB, 'UB', s.UB, ...
        'theta0', theta0.', 'sigma_lower_bound0', sigmaLB0, ...
        'attempts', attempts, 'sigma_lower_bound', sigmaLB, ...
        'theta', theta.', 'lengthscale', params(1:D).', 'sigmaF', params(D+1), ...
        'alpha', alpha, 'sigma', gpr.Sigma, 'beta', gpr.Beta, ...
        'log_likelihood', gpr.LogLikelihood);
end

payload = struct('generator', 'fit_reference.m', 'matlab_version', version, ...
                 'cases', {results});

fid = fopen(outfile, 'w');
if fid < 0
    error('fit_reference:cannotWrite', 'Could not open %s for writing.', outfile);
end
fprintf(fid, '%s', jsonencode(payload));
fclose(fid);

fprintf('Wrote %d fitted cases to %s\n', numel(results), outfile);
end


function [gpr, attempts, sigmaLB] = fit_like_reference(X, Y, theta0, sigmaLB)
%FIT_LIKE_REFERENCE  Exact fit with the reference's recovery on failure.
attempts = 0;
while true
    attempts = attempts + 1;
    try
        gpr = fitrgp(X, Y, ...
            'KernelFunction',   @dbo_kernel_fn, ...
            'KernelParameters', theta0, ...
            'SigmaLowerBound',  sigmaLB, ...
            'FitMethod',        'exact');
        return
    catch err
        if attempts > 10
            rethrow(err);
        end
        theta0(end) = log(exp(theta0(end)) * 1.5);
        sigmaLB = 2 * sigmaLB;
    end
end
end


function K = dbo_kernel_fn(XM, XN, theta)
%DBO_KERNEL_FN  Separable DBO covariance in the form fitrgp expects.
%
%   theta = [log(l_1); ...; log(l_D); log(sigmaF); log(1 - alpha)]
%   The final column of XM and XN carries time.

D      = size(XM,2) - 1;
params = exp(theta);
ls     = params(1:D);
sigmaF = params(D+1);
decay  = params(D+2);

alpha = 1 - decay;
if alpha > 1
    alpha = 1;
elseif alpha <= 0
    alpha = 1e-9;
end

sq = zeros(size(XM,1), size(XN,1));
for r = 1:D
    d  = (XM(:,r) - XN(:,r).') / ls(r);
    sq = sq + d.^2;
end

K = (sigmaF^2) * exp(-0.5*sq) .* (alpha .^ abs(XM(:,end) - XN(:,end).'));
end
