function CONTROLS = SBOR_Controls

%% Define parameters which influence the underlying operation of the 
%% SparseBayes inference algorithm

% TOLERANCES
% 
% Any Q^2-S "relevance factor" less than this is considered to be zero
% 
CONTROLS.ZeroFactor			= 1e-9;
%
% If the change in log-alpha for the best re-estimation is less than this,
% we consider termination
% 
CONTROLS.MinDeltaLogAlpha	= 1e-3;
%
% In the Gaussian case, we also require a beta update to change the value
% of log-beta (inverse noise variance) less than this to terminate
% 
CONTROLS.MinDeltaLogBeta	= 1e-6;

% ADD/DELETE
% 
% - preferring addition where possible will probably make the algorithm a
% little slower and perhaps less "greedy"
% 
% - preferring deletion may make the model a little more sparse and the
% algorithm may run slightly quicker
% 
% Note: both these can be set to 'true' at the same time, in which case
% both take equal priority over re-estimation.
% 
CONTROLS.PriorityAddition	= false;
CONTROLS.PriorityDeletion	= true;

% (GAUSSIAN) NOISE
%
% When to update the noise estimate
%
% The number of iterations from the start for which we update it every
% iteration (to get in the right ball-park to begin with)
% 
CONTROLS.BetaUpdateStart		= 10;
%
% After the above, we only regularly update it after 
% a given number of iterations
% 
CONTROLS.BetaUpdateFrequency	= 10;
%
% Prevent zero-noise estimate (perfect fit) problem
% -	effectively says the noise variance estimate is clamped to be no
%	lower than variance-of-targets / BetaMaxFactor.
% 
CONTROLS.BetaMaxFactor			= 1e6;

% POSTERIORMODE
%
% How many alpha updates to do in between each full posterior mode
% computation in the non-Gaussian case
% 
% In principle, this should be set to one (to update the posterior every
% iteration) but it may be more efficient to do several alpha updates before
% re-finding the posterior mode.
% 
CONTROLS.PosteriorModeFrequency	= 1;

% REDUNDANT BASIS
%
% Check for basis vector alignment/correlation redundancy
% 
CONTROLS.BasisAlignmentTest		= true;
%
ALIGNMENT_ZERO					= 1e-5;
%
% If BasisAlignmentTest is true, any basis vector with inner product more
% than MAX_ALIGNMENT with any existing model vector will not be added
% 
CONTROLS.AlignmentMax			= 1 - ALIGNMENT_ZERO;

CONTROLS.learning_rate = 1;

CONTROLS.STEP_MIN = 2^-8;

% Threshold update
% 1: naive
% 2: linear search based on eq. 17 and eq.18
CONTROLS.update_threshold = 1;
