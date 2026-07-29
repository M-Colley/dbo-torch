function gp_reference(outfile)
%GP_REFERENCE  Emit GP posteriors from MATLAB's fitrgp using the DBO kernel.
%
%   Tier 2 of the parity harness. Fits nothing: hyperparameters are pinned, so
%   MATLAB and Python are solving identical linear algebra and should agree to
%   near machine precision. This isolates the GP machinery (Cholesky solve,
%   mean function, noise handling) from the optimiser, which is stochastic and
%   cannot be compared this tightly.
%
%   Requires the Statistics and Machine Learning Toolbox.
%
%   Note on the noise convention, which is the usual source of confusion when
%   comparing GP libraries: MATLAB's predict returns the standard deviation of
%   the *response*, which includes observation noise. BoTorch's posterior
%   variance is that of the *latent function* unless observation noise is
%   requested. Both are exported here so the Python side can check each
%   against the right quantity.
%
%   Usage:
%       gp_reference                      % writes out/matlab_gp.json

if nargin < 1 || isempty(outfile)
    outfile = fullfile(fileparts(mfilename('fullpath')), 'out', 'matlab_gp.json');
end
outdir = fileparts(outfile);
if ~isempty(outdir) && ~exist(outdir, 'dir')
    mkdir(outdir);
end

if exist('fitrgp', 'file') ~= 2
    error('gp_reference:missingToolbox', ...
        ['fitrgp not found. Install the Statistics and Machine Learning ' ...
         'Toolbox - see SETUP.md.']);
end

cases = {};

% --- Case 1: the RA-L shape - one torque parameter, drifting optimum ---
U = [5; 7; 3; -1; 0.5; 8.25; -4; 2; 6; 1.5];
T = (1:numel(U))';
optTrace = 5 - 5*(T-1)/(numel(T)-1);
Y = abs(U - optTrace);                       % noiseless, so the test is exact
cases{end+1} = struct('name','1d_drift', 'X',[U,T], 'Y',Y, ...
                      'lengthscale',2.5, 'sigmaF',1.75, 'alpha',0.95, 'sigma',0.10);

% --- Case 2: same data, near-stationary -------------------------------
cases{end+1} = struct('name','1d_alpha_099', 'X',[U,T], 'Y',Y, ...
                      'lengthscale',2.5, 'sigmaF',1.75, 'alpha',0.99, 'sigma',0.10);

% --- Case 3: fast drift, larger noise ---------------------------------
cases{end+1} = struct('name','1d_fast_decay', 'X',[U,T], 'Y',Y, ...
                      'lengthscale',1.5, 'sigmaF',1.0, 'alpha',0.70, 'sigma',0.25);

% --- Case 4: three parameters with ARD --------------------------------
U4 = [ 5.0, 18.0, 65.0;
       7.0, 15.0, 60.0;
       3.0, 20.0, 70.0;
      -1.0, 18.0, 55.0;
       0.5, 22.0, 68.0;
       8.0, 16.0, 62.0;
       2.0, 19.0, 64.0];
T4 = (1:size(U4,1))';
Y4 = abs(U4(:,1) - (5 - 5*(T4-1)/(size(U4,1)-1))) + 0.01*U4(:,2);
cases{end+1} = struct('name','3d_ard', 'X',[U4,T4], 'Y',Y4, ...
                      'lengthscale',[2.5,6.0,12.0], 'sigmaF',1.25, ...
                      'alpha',0.90, 'sigma',0.05);

results = cell(1, numel(cases));

for c = 1:numel(cases)
    s = cases{c};
    D = size(s.X,2) - 1;

    ls = s.lengthscale(:);
    if isscalar(ls), ls = repmat(ls, D, 1); end

    % Parameter vector handed to the custom kernel. Log scale throughout, with
    % the decay rate (1 - alpha) fitted rather than alpha itself - the
    % parameterisation the reference DBO implementation uses.
    theta = log([ls; s.sigmaF; max(1 - s.alpha, realmin)]);

    gpr = fitrgp(s.X, s.Y, ...
        'KernelFunction',  @dbo_kernel_fn, ...
        'KernelParameters', theta, ...
        'BasisFunction',   'none', ...     % zero mean, so Python needs ZeroMean
        'Sigma',           s.sigma, ...
        'FitMethod',       'none', ...     % pinned - no optimisation
        'PredictMethod',   'exact', ...
        'Standardize',     false);

    % Test grid at the latest time, which is where the acquisition function
    % and the validation-iteration selection both operate.
    tNow = max(s.X(:,end));
    grid1 = linspace(-5, 9, 25)';
    if D == 1
        Xs = [grid1, tNow*ones(size(grid1))];
    else
        Xs = [grid1, 18*ones(size(grid1)), 64*ones(size(grid1)), ...
              tNow*ones(size(grid1))];
    end

    % Also probe backwards in time, where the temporal kernel does the work
    % that distinguishes DBO from stationary BO.
    XsPast = Xs;  XsPast(:,end) = 1;

    [mu,      sdResponse]     = predict(gpr, Xs);
    [muPast,  sdResponsePast] = predict(gpr, XsPast);

    % Latent-function standard deviation, i.e. response variance with the
    % observation noise removed. This is what BoTorch reports by default.
    sdLatent     = sqrt(max(sdResponse.^2     - s.sigma^2, 0));
    sdLatentPast = sqrt(max(sdResponsePast.^2 - s.sigma^2, 0));

    results{c} = struct( ...
        'name',s.name, 'X',s.X, 'Y',s.Y, 'Xs',Xs, 'XsPast',XsPast, ...
        'lengthscale',ls.', 'sigmaF',s.sigmaF, 'alpha',s.alpha, 'sigma',s.sigma, ...
        'mu',mu, 'sd_response',sdResponse, 'sd_latent',sdLatent, ...
        'mu_past',muPast, 'sd_response_past',sdResponsePast, ...
        'sd_latent_past',sdLatentPast);
    % Note: no log-likelihood is exported. With FitMethod='none', fitrgp
    % leaves gpr.LogLikelihood empty, so the field would always be null.
end

payload = struct('generator','gp_reference.m', 'matlab_version',version, ...
                 'cases',{results});

fid = fopen(outfile,'w');
if fid < 0
    error('gp_reference:cannotWrite','Could not open %s for writing.', outfile);
end
fprintf(fid,'%s', jsonencode(payload));
fclose(fid);

fprintf('Wrote %d GP cases to %s\n', numel(results), outfile);
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
