function [w,Sigma,y,delta,logpost,U,Ui,H,t_hat,z] = OR_MAP(w,k_nz,...
    train_y,alpha,sigma,by,CONTROLS)
	learning_rate = CONTROLS.learning_rate;
    itsMax = 10;
	STEP_MIN = CONTROLS.STEP_MIN;

    for i = 1:itsMax % loop to find the minima
        y = k_nz*w;
        z = [(by(train_y+1) -y)/sigma, (by(train_y)-y)/sigma];
        N_z = normpdf(z);
        Psi_z = normcdf(z);
    % 	gradient of the log likelihood
        delta = -1/sigma*(N_z(:,1)-N_z(:,2))./(Psi_z(:,1)-Psi_z(:,2));
    % 	second derivation of minus log posterior
        zN_z = z.*N_z;
        H = delta.^2 + 1/sigma.^2*(zN_z(:,1)-zN_z(:,2))./(Psi_z(:,1)-Psi_z(:,2)); 
        t_hat = y + H.*delta;
        Hessian = (diag(alpha)+ 0.0001*eye(length(alpha)) + k_nz'.*repmat(H',length(w),1)*k_nz); % minus Hessian
        U = chol(Hessian);
        Ui = inv(U);
        Sigma = Ui*Ui';
        g = learning_rate*Sigma*(k_nz'*delta - alpha.*w);

        loglikelihood = sum(log(Psi_z(:,1)-Psi_z(:,2)));
        logpost = .5*alpha'*(w.^2) - loglikelihood;

    %     w1 = k_nz'*delta./alpha;w2 = Sigma*k_nz'*(H.*t_hat);
        Lambda = 1;
        while(Lambda >= STEP_MIN && i< itsMax) %	newton-rapshon method
            w_new = w + Lambda*g;
            y_new = k_nz*w_new;
            z_new = [(by(train_y+1) -y_new)/sigma, (by(train_y)-y_new)/sigma];
            Psi_z = normcdf(z_new);
            loglikelihood_new = sum(log(Psi_z(:,1)-Psi_z(:,2)));
            logpost_new = .5*alpha'*(w_new.^2) - loglikelihood_new;
            if logpost_new < logpost
                Lambda = 0;
                w = w_new;
%                 y = y_new;
%                 logpost = -logpost_new;
%                 loglikelihood = loglikelihood_new;
%                 z = z_new;            
%                 N_z = normpdf(z);
%                 delta = -1/sigma*(N_z(:,1)-N_z(:,2))./(Psi_z(:,1)-Psi_z(:,2));
%                 zN_z = z.*N_z;
%                 H = delta.^2 + 1/sigma.^2*(zN_z(:,1)-zN_z(:,2))./(Psi_z(:,1)-Psi_z(:,2));
%                 t_hat = y + H.*delta;
%                 Hessian = (diag(alpha) + k_nz'.*repmat(H',length(w),1)*k_nz);
%                 U = chol(Hessian);
%                 Ui = inv(U);
%                 Sigma = Ui*Ui';
            else
                Lambda = Lambda/2;
            end
        end
        if Lambda
            break;
        end
    end
    logpost = -logpost;
   
end
