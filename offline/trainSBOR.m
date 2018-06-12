function [w,idx,by,MAE,alpha,sigma] = trainSBOR(train_x,train_y,theta,ker)
    its = 500;
    learning_rate = 0.3;
    its_check = its;
    MAE = zeros(its/its_check,1);

    [N] = size(train_x,1); T = range(train_y)+1; 
    
    %  parameters
    f = ones(N,1);
    y_idx = [1;find(diff(train_y))+1;N];
    err = zeros(N,1);

%   using least-square to heuristic initialize 
    K = kernel(train_x,train_x,theta,ker);   
    M_full = size(K,2);
    L = zeros(M_full,M_full);
    w = (K'*K + 0.5*eye(M_full))\(K'*train_y);
%     w = ones(M_full,1)/M_full;
%     alpha = .5./(w+1e-4).^2;
    alpha = ones(M_full,1)/M_full.^2;
    y = K*w;
    sigma = 1;
%     sigma = std(y);

    used = logical(ones(M_full,1));
    by = zeros(T+1,1);
    by(1) = -exp(100);
    by(T+1) = exp(100);
    db = ones(T+2,1); % delta of threshold
    for j = 2 : T
        y1 = (y(y_idx(j-1):y_idx(j)-1));
        y2 = (y(y_idx(j):y_idx(j+1)-1));
        y_max = max(y1); y_min = min(y2);
        by(j) = mean([y1(y1>y_min);y2(y2<y_max)] );
        if isnan(by(j))
            by(j) = 0.5*(y_max+y_min);
        end
    end

%    Laplace matrix
%     L = K(:,1:end-1)'*K(:,1:end-1);
%     L = diag(sum(L))-L;
%     L = [L zeros(M_full-1,1);zeros(1,M_full)];
%   pre-compute 
    z = [(by(train_y+1) -y)/sigma, (by(train_y)-y)/sigma];
    N_z = normpdf(z);
    Psi_z = normcdf(z);
    delta = 1/sigma*(N_z(:,1)-N_z(:,2))./(Psi_z(:,1)-Psi_z(:,2));
   
%   main loop
    for i = 1:its 
        w_old = w;
        k_nz = K(:,used);
        err(i) =  0.5*alpha(used)'*(w(used).^2)-sum(log(Psi_z(:,1)-Psi_z(:,2)));
%% MAP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       check gradient
% checkgrad('posterior', w(used), 0.001, k_nz,train_y,alpha(used),by,sigma)
% for j = 1:sum(find(used==1))
% [e(j)]=checkgrad('dposterior', w(used), 0.001, k_nz,train_y,...
% alpha(used),by,j);
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        zN_z = z.*N_z;
        H = delta.^2 + 1/sigma.^2*(zN_z(:,1)-zN_z(:,2))./(Psi_z(:,1)-Psi_z(:,2));
        AL = diag(alpha(used))+ L(used,used);
        Hessian = AL+ 0.001*eye(sum(used)) + k_nz'.*repmat(H',length(w(used)),1)*k_nz;
        U = chol(Hessian);
        Ui = inv(U);
        Sigma = Ui*Ui';
        
%         g = learning_rate*Sigma*(k_nz'*delta + alpha(used).*w(used));
        g = learning_rate*Sigma*(k_nz'*delta + AL*w(used));
        w(used) = w(used) - g;% w(used) =  -Sigma*k_nz'*O*t_hat;
        y = k_nz * w(used);
                
%         [w(used),Sigma,y,Ui,~] = OR_MAP(w(used),k_nz,...
%             train_y,alpha(used),L(used,used),sigma,by,learning_rate);
%    
%      
%% Hyper parameter
%       TODO try different error         
        sigma = norm(train_y-1 - (T-1)*sigmoid(y-mean(y)))^2/(N - sum(used) + alpha(used)'*sum(Ui.^2,2));
        sigma = sqrt(sigma);

        gamma = 1- alpha(used).*sum(Ui.^2,2);
        alpha(used) = gamma./w(used).^2;
        used = (abs(w)>1e-3);
        
%       parameters for b, delta & next round
        z = [(by(train_y+1) -y)/sigma, (by(train_y)-y)/sigma];
        N_z = normpdf(z);
        Psi_z = normcdf(z);
        delta = 1/sigma*(N_z(:,1)-N_z(:,2))./(Psi_z(:,1)-Psi_z(:,2));
        
% %       threshold
%         % First way to opt by 
%         y_tile(1) = sum(y(train_y==1))/y_num(1);
% %       by(2) = y_tile(1) + sigma/2;
%         by(2) = by(2) + 1/N*sum(delta);
% %       for delta
%         for j = 3:(T-1)
%             y_tile(j) = sum(y(train_y==j))/y_num(j);
%             by(j+1) = 2 * y_tile(j) -by(j);
%         end
        % second 
        % TODO method in paper

% %       Third
        for j = 2 : T
            y1 = (y(y_idx(j-1):y_idx(j)-1));
            y2 = (y(y_idx(j):y_idx(j+1)-1));        
            y_max = max(y1); y_min = min(y2);
            by(j) = mean([y1(y1>y_min);y2(y2<y_max)] );
            if isnan(by(j))
                by(j) = 0.5*(y_max+y_min);
            end
        end
       if rem(i,its_check)==0
            y_t = K(:,used)*w(used);
            b_t = ones(N,1)*by';
            t = sign(y_t*ones(1,size(b_t,2))-b_t);
            t = sum(t,2);
            t = (t-min(t))/2+1;
            MAE(i/its_check) = norm(t-train_y,1)/N;
        end
       if max(abs(w_old(used)-w(used)))< 0.001
           fprintf('Iteration break at: %d',i)
           break
       end
    end
    idx = find(used==true);
    w = w(idx);
    alpha = alpha(idx);
end

