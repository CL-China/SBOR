
function OPTIONS = SBOR_Options(varargin)

% Ensure arguments are supplied in pairs
% 
if rem(nargin,2)
  error('Arguments to SB2_UserOptions should be (property, value) pairs')
end
% Any options specified?
numSettings	= nargin/2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set defaults
%
% Assume we will infer the noise in the Gaussian case
% 
OPTIONS.fixedNoise		= false;
%
% Fixed noise
%
OPTIONS.Noise = 1;
%
% Option to allow subset of the basis (e.g. bias) to be unregularised
% 
OPTIONS.freeBasis		= [];
%
% Option to set max iterations to run for
%
OPTIONS.iterations		= 100;
%
% Option to set max time to run for
%
OPTIONS.time			= 10000; % seconds
%
% Option to the kernel parameter
%
OPTIONS.theta           = 0.3;

OPTIONS.goodKernel     = 1e-5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parse string/variable pairs

for n=1:numSettings
  property_	= varargin{(n-1)*2+1};
  value		= varargin{(n-1)*2+2};
  switch upper(property_)
      %
      case 'GOODKERNEL',
          OPTIONS.goodKernel = value;
      case 'THETA',
          OPTIONS.theta = value;
    %
   case 'FIXEDNOISE',
    OPTIONS.fixedNoise	= value;
    %
   case 'FREEBASIS',
    OPTIONS.freeBasis	= value;
    %
   case 'ITERATIONS',
    OPTIONS.iterations	= value;
	%
   case 'TIME',
    OPTIONS.time		= timeInSeconds(value);
    %
   case 'MONITOR',
    OPTIONS.monitor		= value;
    %
   case 'DIAGNOSTICLEVEL',
    OPTIONS.diagnosticLevel	= value;
    MAX_LEVEL	= 4;
    if ischar(value)
      switch upper(value)
       case {'ZERO','NONE'},
	OPTIONS.diagnosticLevel	= 0;
       case 'LOW',
	OPTIONS.diagnosticLevel	= 1;
       case 'MEDIUM',
	OPTIONS.diagnosticLevel	= 2;
       case 'HIGH',
	OPTIONS.diagnosticLevel	= 3;
       case 'ULTRA',
	OPTIONS.diagnosticLevel	= 4;
       otherwise,
	error('Unrecognised textual diagnostic level: ''%s''\n', value)
      end
    elseif isnumeric(value)
      if value>=0 & value<=MAX_LEVEL
	OPTIONS.diagnosticLevel		= value;  
      else
	error(['Supplied level should be integer in [0,%d], '...
	       'or one of ZERO/LOW/MEDIUM/HIGH/ULTRA'], MAX_LEVEL);
	
      end
    end
    %
   case 'DIAGNOSTICFILE',
    OPTIONS.diagnosticFile_	= value;  
    OPTIONS.diagnosticFID	= -1; % It will be opened later
    %
   case 'CALLBACK',
    OPTIONS.callback		= true;
    OPTIONS.callbackFunc	= value;
    if exist(OPTIONS.callbackFunc)~=2
      error('Callback function ''%s'' does not appear to exist\n', value)
    end
    %
   case 'CALLBACKDATA',
    OPTIONS.callbackData	= value;
   otherwise,
    error('Unrecognised user option: ''%s''\n', property_)
    %
  end
end

%%
%% Support function: parse time specification
%%
function s = timeInSeconds(value_)
%
[v_ r_]				= strtok(value_,' ');
v					= str2num(v_);
r_(isspace(r_))		= [];
switch upper(r_)
 case {'SECONDS', 'SECOND'}
  s	= v;
 case {'MINUTES', 'MINUTE'}
  s = 60*v;
 case {'HOURS', 'HOUR'}
  s = 3600*v;
 otherwise,
  error('Badly formed time string: ''%s''', value_)
end
% Returns time in seconds
% 