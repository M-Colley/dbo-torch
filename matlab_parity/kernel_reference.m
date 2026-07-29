function kernel_reference(outfile)
%KERNEL_REFERENCE  Emit DBO covariance matrices for cross-checking against Python.
%
%   Computes the separable Dynamic Bayesian Optimization covariance
%
%       k((u,t),(u',t')) = sigmaF^2 * exp(-0.5 * sum_r ((u_r - u'_r)/l_r)^2)
%                          * alpha^|t - t'|
%
%   over a set of fixed hyperparameter settings and writes the results to
%   JSON. This is Tier 1 of the parity harness: it is pure arithmetic with no
%   fitting, so Python and MATLAB should agree to machine precision. Any
%   disagreement here is a genuine bug in the kernel, not an artefact of
%   different optimisers.
%
%   Deliberately uses only base MATLAB - no pdist2, no fitrgp - so it runs
%   without the Statistics and Machine Learning Toolbox.
%
%   Usage:
%       kernel_reference                       % writes out/matlab_kernel.json
%       kernel_reference('somewhere/k.json')

if nargin < 1 || isempty(outfile)
    outfile = fullfile(fileparts(mfilename('fullpath')), 'out', 'matlab_kernel.json');
end
outdir = fileparts(outfile);
if ~isempty(outdir) && ~exist(outdir, 'dir')
    mkdir(outdir);
end

cases = {};

% --- Case 1: one control parameter, integer iteration times ------------
% The shape of the RA-L experiment: a single torque amplitude, sampled once
% per iteration.
rng(0);
U1 = [5; 7; 3; -1; 0.5; 8.25; -4; 2];
T1 = (1:numel(U1))';
cases{end+1} = struct( ...
    'name',        '1d_integer_time', ...
    'X',           [U1, T1], ...
    'lengthscale', 2.5, ...
    'sigmaF',      1.75, ...
    'alpha',       0.97);

% --- Case 2: alpha = 1, i.e. the stationary limit ----------------------
% The temporal factor must vanish exactly, reducing to ordinary SE-ARD.
cases{end+1} = struct( ...
    'name',        '1d_alpha_one', ...
    'X',           [U1, T1], ...
    'lengthscale', 2.5, ...
    'sigmaF',      1.75, ...
    'alpha',       1.0);

% --- Case 3: strong decay ---------------------------------------------
% Exercises the regime where distant observations are almost fully
% discounted, where an exponent sign error would be unmissable.
cases{end+1} = struct( ...
    'name',        '1d_fast_decay', ...
    'X',           [U1, T1], ...
    'lengthscale', 1.0, ...
    'sigmaF',      0.5, ...
    'alpha',       0.55);

% --- Case 4: three control parameters with ARD -------------------------
% Matches the paper's stated future direction of optimising torque
% amplitude, duration and peak timing together. Distinct lengthscales per
% dimension catch an ARD indexing mistake.
U4 = [ 5.0, 18.0, 65.0;
       7.0, 15.0, 60.0;
       3.0, 20.0, 70.0;
      -1.0, 18.0, 55.0;
       0.5, 22.0, 68.0;
       8.0, 16.0, 62.0];
T4 = (1:size(U4,1))';
cases{end+1} = struct( ...
    'name',        '3d_ard', ...
    'X',           [U4, T4], ...
    'lengthscale', [2.5, 6.0, 12.0], ...
    'sigmaF',      1.25, ...
    'alpha',       0.9);

% --- Case 5: non-uniform, non-integer times ----------------------------
% Time need not be the iteration index; confirm fractional lags work.
U5 = [5; 7; 3; -1; 0.5];
T5 = [0.0; 0.5; 2.25; 7.0; 7.125];
cases{end+1} = struct( ...
    'name',        '1d_fractional_time', ...
    'X',           [U5, T5], ...
    'lengthscale', 3.0, ...
    'sigmaF',      2.0, ...
    'alpha',       0.8);

% --- Evaluate ----------------------------------------------------------
results = cell(1, numel(cases));
for c = 1:numel(cases)
    s = cases{c};
    K = dbo_kernel(s.X, s.X, s.lengthscale, s.sigmaF, s.alpha);

    % A rectangular block too, to catch shape or transpose errors that a
    % symmetric matrix would hide.
    Xb = s.X(1:2:end, :);
    Krect = dbo_kernel(s.X, Xb, s.lengthscale, s.sigmaF, s.alpha);

    results{c} = struct( ...
        'name',        s.name, ...
        'X',           s.X, ...
        'Xb',          Xb, ...
        'lengthscale', s.lengthscale, ...
        'sigmaF',      s.sigmaF, ...
        'alpha',       s.alpha, ...
        'K',           K, ...
        'Krect',       Krect);
end

payload = struct('generator', 'kernel_reference.m', ...
                 'matlab_version', version, ...
                 'cases', {results});

fid = fopen(outfile, 'w');
if fid < 0
    error('kernel_reference:cannotWrite', 'Could not open %s for writing.', outfile);
end
fprintf(fid, '%s', jsonencode(payload));
fclose(fid);

fprintf('Wrote %d kernel cases to %s\n', numel(results), outfile);
end


function K = dbo_kernel(XM, XN, lengthscale, sigmaF, alpha)
%DBO_KERNEL  Separable spatial-times-temporal covariance.
%
%   The last column of XM and XN is the time coordinate; all preceding
%   columns are control parameters.

D = size(XM, 2) - 1;
if isscalar(lengthscale)
    lengthscale = repmat(lengthscale, 1, D);
end
if numel(lengthscale) ~= D
    error('dbo_kernel:badLengthscale', ...
          'Expected %d lengthscales, got %d.', D, numel(lengthscale));
end

% Squared exponential with automatic relevance determination, accumulated
% dimension by dimension. Written with explicit broadcasting rather than
% pdist2 so this runs without the Statistics Toolbox.
sq = zeros(size(XM,1), size(XN,1));
for r = 1:D
    d = (XM(:,r) - XN(:,r).') / lengthscale(r);
    sq = sq + d.^2;
end
Kspatial = (sigmaF^2) * exp(-0.5 * sq);

% Temporal decay. alpha^|dt|, evaluated directly.
lag = abs(XM(:,end) - XN(:,end).');
Ktemporal = alpha .^ lag;

K = Kspatial .* Ktemporal;
end
