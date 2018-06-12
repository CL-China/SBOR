function [w,idx,by,alpha,sigma] = train(train_x,train_y,theta)
    warning off
    its = 10;
    learning_rate = 0.3;
    
    [N,M] = size(train_x); T = range(train_y)+1; 
    
    %  parameters
    used = logical(ones(N,1));
    f = ones(N,1);
    y_idx = [1;find(diff(train_y))+1;N];
    err = zeros(N,1);

%   using least-square to heuristic initialize 
    K = kernel(train_x,train_x,theta);   
    w = (K'*K+0.5*K)\(K'*train_y);
%     w = ones(N,1)/N;
    alpha = .5./w.^2;
    y = K*w;
       
    sigma = std(y);
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
%     db(3:T-1) = diff(by(2:T-1));
%     ymax = max(y); ymin = min(y);
%     by(2) =  ymin + (ymax-ymin)/T; % b
%     for i = 3:T; by(i) = (ymax-ymin)/T + by(i-1);end

%   pre-compute 
    z = [(by(train_y+1) -y)/sigma, (by(train_y)-y)/sigma];
    N_z = normpdf(z);
    Psi_z = normcdf(z);
    delta = 1/sigma*(N_z(:,1)-N_z(:,2))./(Psi_z(:,1)-Psi_z(:,2));

%   main loop
    for i = 1:its 
        k_nz = K(:,used);
        err(i) =  0.5*alpha(used)'*(w(used).^2)-sum(log(Psi_z(:,1)-Psi_z(:,2)));
%% MAP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       check gradient
% checkgrad('posterior', w(used), 0.001, k_nz,train_y,alpha(used),by,sigma)
% for j = 1:sum(find(used==1)); 
% [e(j)]=checkgrad('dposterior', w(used), 0.001, k_nz,train_y,...
% alpha(used),by,j);end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        zN_z = z.*N_z;
        od = delta.^2 + 1/sigma.^2*(zN_z(:,1)-zN_z(:,2))./(Psi_z(:,1)-Psi_z(:,2));
        O = diag(od);% O is S
%         t_hat = y + O*delta;
        H = (diag(alpha(used)) + k_nz'*O*k_nz);
        U = chol(H);
        Ui = inv(U);
        Sigma = Ui*Ui';
        % w(used) =  -Sigma*k_nz'*O*t_hat;
        g = learning_rate*Sigma*(k_nz'*delta + alpha(used).*w(used));
%         lambda = 1;
%         while lambda > 2^-8
%             w_new = w(used) - lambda * g;
%             y_new = k_nz * w_new;
%             z = [(by(train_y+1) -y_new)/sigma, (by(train_y)-y_new)/sigma];
%             Psi_z = normcdf(z);
%             err_new =  -sum(log(Psi_z(:,1)-Psi_z(:,2))) + 0.5 * alpha(used)'*(w_new.^2);
%             if err_new < err(i)
%                 lambda = 0;
%                 w(used) = w_new;
%             else
%                 lambda = lambda/2;
%             end
%         end
        w(used) = w(used) - g;
        y = k_nz * w(used);
                

%% Hyper parameter
        % alpha & sigma
%         Q = k_nz'*(train_y - y);
%         S = (od'*k_nz.^2)' - sum((k_nz'*O*k_nz*Ui).^2,2);
%         
%         s = (alpha(used).*S)./(alpha(used)-S);
%         q = (alpha(used).*Q)./(alpha(used)-S);
%         
%         sigma = norm(train_y - y)^2/(N - sum(used) + alpha(used)'*sum(Ui.^2,2));
%         sigma = sqrt(sigma);
% 
%         f(used) = q.^2-s;
%         alpha(used) = s.^2./f(used);
        gamma = 1- alpha(used).*diag(Sigma);
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
                
    end
    idx = find(used==true);
    w = w(idx);
    alpha = alpha(idx);
end

