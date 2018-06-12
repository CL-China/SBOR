function [w,used,by,logMarginalLog,sigma,e] = ISBOR_EQ(train_x,train_y,OPTIONS)
%   This code is based on Tipping's SB2.

if ~exist('OPTIONS','var')
    OPTIONS = SBOR_Options;
end


CONTROLS = SBOR_Controls;
%   Initialization returns one relevant vector candidates and the scaled kernel matrix.
[K,k_nz,scale,used,w,alpha,sigma,by,y_n] = initialization(train_x,train_y,OPTIONS);
[N,M_full] = size(K);
M  = length(used);
T = range(train_y)+1;

%  Pre-loop process
[w,Sigma,S,Q,s,q,Factor,logML,sigma,delta,H,delta_eq] = FullEstimate(K,...
    k_nz,train_y,used,alpha,w,sigma,by,OPTIONS,CONTROLS);




%   number of different estimations
addCount    = 0;
deleteCount = 0;
updateCount = 0;


maxLogSize	= OPTIONS.iterations + CONTROLS.BetaUpdateStart + ...
    ceil(OPTIONS.iterations/CONTROLS.BetaUpdateFrequency);
logMarginalLog = zeros(maxLogSize,1);
count = 0;
e = zeros(zeros,OPTIONS.iterations);
% ACTION CODES
%
% Assign an integer code to the basic action types
%
ACTION_REESTIMATE  = 0;
ACTION_ADD         = 1;
ACTION_DELETE      = -1;
%
% Some extra types
%
ACTION_TERMINATE = 10;
%

%
ACTION_ALIGNMENT_SKIP = 12;

Aligned_in = [];
Aligned_out = [];
alignDeferCount = 0;

g_b = zeros(T,1); % gradient for b & detla, g_b(0) is auxillary. 
%%   main loop
i = 0;


LAST_ITERATION = false;


while (~LAST_ITERATION)
    i = i+1;
    % number of useful data
    
    
    %% DECISION PHASE
    %%
    %% Assess all potential actions
    %%
    %
    % Compute change in likelihood for all possible updates
    %
    DeltaML       = zeros(M_full,1);
    %
    Action        = ACTION_REESTIMATE*ones(M_full,1); % Default
    %
    % 'Relevance Factor' (Q^S-S) values for basis functions in model
    %
    usedFactor    = Factor(used);
    
    %
    % RE-ESTIMATION: must be a POSITIVE 'factor' and already IN the model
    %
    iu        = usedFactor>CONTROLS.ZeroFactor;
    index     = used(iu);
    NewAlpha  = s(index).^2 ./ Factor(index);
    Delta     = (1./NewAlpha - 1./alpha(iu)); % Temp vector
    %
    % Quick computation of change in log-likelihood given all re-estimations
    %
    DeltaML(index)    = (Delta.*(Q(index).^2) ./ ...
        (Delta.*S(index) + 1) - ...
        log(abs(1 + S(index).*Delta)))/2; % original  log(1 + S(index).*Delta))
%     if ~isempty(find((1+S(index).*Delta) < 0,1))
%         fprintf('Illegal value in DeltalML at %d',i);
%     end
%       DeltaML(index) = -1;

    iu          = ~iu;  % iu = usedFactor <= CONTROLS.ZeroFactor
    index       = used(iu);
    anyToDelete  = ~isempty(setdiff(index,OPTIONS.freeBasis)) && M>1;
    %
    if anyToDelete
        %
        % Quick computation of change in log-likelihood given all deletions
        %
        DeltaML(index)  = -(q(index).^2 ./ (s(index) + alpha(iu)) - ...
            log(abs(1 - s(index) ./ alpha(iu))))/2; %original  log(1 - s(index) ./ alpha(iu)))
        Action(index)   = ACTION_DELETE;
        % Note: if M==1, DeltaML will be left as zero, which is fine
    end
    
    %
    % ADDITION: must be a POSITIVE factor and OUT of the model
    %
    % Find ALL good factors ...
    GoodFactor        = Factor>CONTROLS.ZeroFactor;
    % ... then mask out those already in model
    GoodFactor(used)  = 0;
    % ... and then mask out any that are aligned with those in the model
    if CONTROLS.BasisAlignmentTest
        GoodFactor(Aligned_out) = 0;
    end
    %
    index         = find(GoodFactor);
    anyToAdd      = ~isempty(index);
    if anyToAdd
        %
        % Quick computation of change in log-likelihood given all additions
        %
        qout            = Q(index).^2 ./ S(index);
        DeltaML(index)  = (qout - 1 - log(abs(qout)))/2; %log(qout))
        Action(index)   = ACTION_ADD;
    end
    
    
  
  % If we prefer ADD or DELETE actions over RE-ESTIMATION
%   % 
%   if (anyToAdd && CONTROLS.PriorityAddition) || ...
%       (anyToDelete && CONTROLS.PriorityDeletion)
%     % We won't perform re-estimation this iteration, which we achieve by
%     % zero-ing out the delta
%     DeltaML(Action==ACTION_REESTIMATE)	= 0;
%     % Furthermore, we should enforce ADD if preferred and DELETE is not
%     % - and vice-versa
%     if (anyToAdd && CONTROLS.PriorityAddition && ~CONTROLS.PriorityDeletion)
%       DeltaML(Action==ACTION_DELETE)	= 0;
%     end
%     if (anyToDelete && CONTROLS.PriorityDeletion && ~CONTROLS.PriorityAddition)
%       DeltaML(Action==ACTION_ADD)		= 0;
%     end
%   end
  
    
    
    %   Choose the proper candidates which max the marginal
    [deltaLogMarginal, nu]    = max(DeltaML);
    selectedAction        = Action(nu);
    anyWorthwhileAction   = deltaLogMarginal>0;
    
    %
    %   We need to note if basis nu is already in the model, and if so,
    %   find its interior index, denoted by "j"
    %
    if selectedAction==ACTION_REESTIMATE || selectedAction==ACTION_DELETE
        j = find(used==nu);
    end
    
    %
    % Get the individual basis vector for update and compute its optimal alpha,
    % according to equation (20): alpha = S_out^2 / (Q_out^2 - S_out)
    
    k       = K(:,nu);
    newAlpha  = s(nu)^2 / Factor(nu);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % TERMINATION CONDITIONS
    %
    % Propose to terminate if:
    %
    % 1.  there is no worthwhile (likelihood-increasing) action, OR
    %
    % 2a. the best action is an ACTION_REESTIMATE but this would only lead to
    %     an infinitesimal alpha change, AND
    % 2b. at the same time there are no potential awaiting deletions
    %
%     if ~anyWorthwhileAction || ...
%             (selectedAction==ACTION_REESTIMATE && ...
%             abs(log(newAlpha) - log(alpha(j)))<CONTROLS.MinDeltaLogAlpha && ...
%             ~anyToDelete) || ...
%         i == OPTIONS.iterations
%     if i== OPTIONS.iterations

    if i==OPTIONS.iterations|| i>10 &&abs(logMarginalLog(count)-logMarginalLog(count-1)) < 1e-3/M_full*abs(logMarginalLog(count-1))
           
        selectedAction  = ACTION_TERMINATE;        
        fprintf('iteration stops at %d\n delta of log-marginal is %f. \n '...
                , i, abs(logMarginalLog(count)-logMarginalLog(count-1)) )
        
    end
    
%     
    % ALIGNMENT CHECKS
    %
    % If we're checking "alignment", we may have further processing to do
    % on addition and deletion
    %
    if CONTROLS.BasisAlignmentTest
        %
        % Addition - rule out addition (from now onwards) if the new basis
        % vector is aligned too closely to one or more already in the model
        %
        if selectedAction==ACTION_ADD
            % Basic test for correlated basis vectors
            % (note, k and columns of PHI are normalised)
            %
            p             = k'*k_nz;
            findAligned   = find(p>=CONTROLS.AlignmentMax);
            numAligned    = length(findAligned);
            if numAligned>0
                % The added basis function is effectively indistinguishable from
                % one present already
                selectedAction  = ACTION_ALIGNMENT_SKIP;
                alignDeferCount = alignDeferCount+1;
               	act_			= 'alignment-deferred addition';
                % Make a note so we don't try this next time
                % May be more than one in the model, which we need to note was
                % the cause of function 'nu' being rejected
                Aligned_out = [Aligned_out ; nu*ones(numAligned,1)];
                Aligned_in  = [Aligned_in ; used(findAligned)];
            end
        end
        %
        % Deletion: reinstate any previously deferred basis functions
        % resulting from this basis function
        %
        if selectedAction==ACTION_DELETE
            findAligned   = find(Aligned_in==nu);
            numAligned    = length(findAligned);
            if numAligned>0
                reinstated                  = Aligned_out(findAligned);
                Aligned_in(findAligned)     = [];
                Aligned_out(findAligned)    = [];
            end
        end
    end

    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% ACTION PHASE
    %%
    %% Implement above decision
    %%
    
    % We'll want to note if we've made a change which necessitates later
    % updating of the statistics
    %
    UPDATE_REQUIRED = false;
    %
    switch selectedAction
        %
        case ACTION_REESTIMATE,
            
            alpha_old = alpha(j);
            alpha(j) = newAlpha;
            s_j = Sigma(:,j);
            deltaInv = 1/(newAlpha - alpha_old);
            kappa = 1/(Sigma(j,j) + deltaInv);
            tmp = kappa*s_j;
            SigmaNew = Sigma - tmp * s_j';
            deltaW = -w(j)*tmp; % not sure about this part. maybe I need a +/-
            w = w + deltaW;
            Sigma = SigmaNew;
            
            updateCount = updateCount + 1;
            act_        = 're-estimation';
            UPDATE_REQUIRED = true;
            
        case ACTION_ADD,
            
            B_Phi       = (k.*H);
            tmp         = ((B_Phi'*k_nz)*Sigma)';
            %
            alpha       = [alpha ; newAlpha];
            k_nz        = [k_nz k];
            %
            s_ii        = 1/(newAlpha+S(nu));
            s_i         = -s_ii*tmp;
            TAU         = -s_i*tmp';
            SigmaNEW    = [Sigma+TAU s_i ; s_i' s_ii];
            mu_i        = s_ii*Q(nu);
            deltaMu     = [-mu_i*tmp ; mu_i]; % not sure about this part. maybe I need a +/-
            w           = [w ; 0] + deltaMu;
            Sigma       = SigmaNEW;
            used        = [used; nu];
            addCount    = addCount+1;
            act_        = 'addition';
            %
            UPDATE_REQUIRED = true;
        case ACTION_DELETE,
            k_nz(:,j)    = [];
            alpha(j)    = [];
            %
            s_jj            = Sigma(j,j);
            s_j             = Sigma(:,j);
            tmp             = s_j/s_jj;
            SigmaNEW        = Sigma - tmp*s_j';
            SigmaNEW(j,:)   = [];
            SigmaNEW(:,j)   = [];
            deltaMu         = -w(j)*tmp; % not sure about the sign +/-
            w              = w +deltaMu;
            w(j)           = [];
            Sigma=SigmaNEW;
            used(j)     = [];
            
            %%
            deleteCount = deleteCount+1;
            act_        = 'deletion';
            %
            UPDATE_REQUIRED = true;
           
    end

    M = length(used);
    if UPDATE_REQUIRED
        if  ~rem(i+1,CONTROLS.BetaUpdateFrequency)
            OPTIONS.fixedNoise = false;
        end
        [w,Sigma,S,Q,s,q,Factor,logML,sigma,delta,H,delta_eq] = FullEstimate(K,...
            k_nz,train_y,used,alpha,w,sigma,by,OPTIONS,CONTROLS);
        OPTIONS.fixedNoise = true;
%         fprintf('%f   ', logML)
        count = count + 1;
        logMarginalLog(count) = logML;
    
        % update the threshold
        % #1 linear searche
        y = k_nz *w;
%         
%         for j = 2 : T
%             y1 = (y(train_y==j-1));
%             y2 = (y(train_y==j));
%             y_max = max(y1); 
%             y_min = min(y2);
%             by(j) = mean([y1(y1>y_min);y2(y2<y_max)]);
%             if isnan(by(j))
%                 by(j) = (y_n(j-1)*y_max + y_n(j)*y_min)/(y_n(j-1)+y_n(j));
%             end
%         end
        
        
%         e(count) = predict(train_x,train_x,train_y,w./scale(used)',used,by,OPTIONS.theta,sigma);

    %     
        % #2 based on eq 17&18 
%         we actually need r-1 by, and r-2 delta
        if  ~rem(i+1,10)
        delta_g = -sum(delta);    
        for j = 2 : T
            delta_g = delta_g - sum(delta(train_y <= j-1)) +...
                sum(delta_eq(train_y == j-1));
            g_b(j) = g_b(j-1) + 0.1*CONTROLS.learning_rate/N * delta_g ;
            
        end
        by(2:T) = by(2:T) + g_b(2:T);
        end
        
    %     
    %     if ~rem(i,10)
    %         fprintf('%d>>> log-maginal: %f\n',i,logML);
    %     end
    end
    if(selectedAction == ACTION_TERMINATE)
       LAST_ITERATION=true;
    end
end % end of main loop

fprintf('Re-estemate: %d \n Add: %d \n Delete: %d \n Aligned: %d\n Noise: %d \n',...
    updateCount,addCount,deleteCount,alignDeferCount, sigma)
% post process
w = w./scale(used)';
end




%  Initialization for the ordinal regression
function [K,PHI,Kscale,used,w,alpha,sigma,by,y_n] = initialization(train_x,train_y,OPTIONS)
theta = OPTIONS.theta;
[N,~] = size(train_x);
% y_idx = [1;find(diff(train_y))+1;N];
T = range(train_y)+1;
y_n = zeros(T,1);
for i = 1:T
    y_n(i) = length(find(train_y==i));
end

K = kernel(train_x,train_x,theta);
K (K<OPTIONS.goodKernel) = 0;
Kscale = sqrt(sum(K.^2));

K = K./repmat(Kscale,N,1);

%% Choose the initial candidates

K_big = K*(train_y-mean(T));
idx = [];
for i = 1:T 
    tmp_idx = find(train_y==i);
    [~,bar] = sort(K_big(tmp_idx),'descend');
    idx = [idx;tmp_idx(bar(1:ceil(y_n(i)/50)))];
end
used = idx;
PHI = K(:,used);
phi = PHI(used,:);
t = train_y(used);

% sigma = 1;
% 
% 
% p		= diag(PHI'*PHI)./sigma;
% q		= (PHI'*train_y)./sigma;
% alpha =  p.^2./(q.^2-p);
% 
% U   = chol(PHI'*PHI./sigma + diag(alpha));
% Ui	= inv(U);
% SIGMA	= Ui * Ui';
%   % Posterior mean Mu
% w	= (SIGMA * (PHI'*train_y))./sigma;
% y = phi*w;

% 
% % using linear regression to initialize

w = (phi+0.1*eye(length(t)))\t;
% w = ones(length(t),1)./N;
%   init hyper parameters
y = phi*w;
alpha = ones(length(w),1)./N.^2;
% alpha = 0.5./w.^2;
% 
% if OPTIONS.fixedNoise
%     sigma = OPTIONS.Noise;
% else
    sigma = std(y);
% end
% sigma = OPTIONS.Noise;
by = zeros(T+1,1);
by(1) = -exp(100);
by(T+1) = exp(100);

for j = 2 : T
    y1 = (y(t==j-1));
    y2 = (y(t==j));
    y_max = max(y1); y_min = min(y2);
    by(j) = mean([y1(y1>y_min);y2(y2<y_max)] );
    if isnan(by(j))
        by(j) = 0.5*(y_max+y_min);
    end
end
% df = diff(by);
% if ~isempty(find(df<0,1))
%     tmp = [0:T-2];
%     tdelta = (by(T) - by(2))/(T-1);
%     by(2:T) = by(2)+tdelta*tmp;
% end
end
