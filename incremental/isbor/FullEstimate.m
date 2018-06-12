function [w,Sigma,S,Q,s,q,f,logML,sigma,delta,H,delta_eq] = ...
    	FullEstimate(K,k_nz,train_y,used,alpha,w,sigma,by,OPTIONS,CONTROLS)
 % 		w alpha are the in-using parameters w = w(used) and alpha = alpha(used);
 % 		used is the indicator has length of useful sample
        [N,M] = size(k_nz);
        T = range(train_y)+1;
    	[w,Sigma,y,delta,logpost,U,Ui,H,t_hat,z] = OR_MAP(w,k_nz,train_y,...
    		alpha,sigma,by,CONTROLS);
%  		minus log marginal likelihood
    	logML = logpost + 0.5*sum(log(alpha)) - 0.5*sum(log(diag(U)));

        
% 		compute Q S
%         e = (train_y - T*(sigmoid(y-mean(y)))); % train_y - T*(sigmoid(y-mean(y)))
%     	Q_1 = K'*delta;
        Q = K'*delta;% based on eq, we should use K'*delta
        KHk_nz = K'*(H*ones(1,M).*k_nz);
        S = (K.^2)'*H - sum((KHk_nz*Ui).^2,2);
		
        q = Q;
        s = S;	

		s(used) = (alpha.*S(used))./(alpha-S(used));
        q(used) = (alpha.*Q(used))./(alpha-S(used));
        
        
        N_z = normpdf(z);
        Psi_z = normcdf(z);
        delta_eq =  N_z(:,2)./(Psi_z(:,1)-Psi_z(:,2));
        
        if ~OPTIONS.fixedNoise
            sigma = norm(train_y-1 - (T-1)*sigmoid(y-mean(y)))^2 ...
            /(N - length(used) + alpha'*sum(Ui.^2,2)); % t_hat - y
%             sigma = norm(t_hat-y)^2/(N - length(used) + alpha'*sum(Ui.^2,2));
            sigma = sqrt(sigma);
        end
%         Gamma = 1- alpha'*sum(Ui.^2,2);
        f = q.^2-s;