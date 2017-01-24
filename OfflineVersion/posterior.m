function [f,df] = posterior(w,k,train_y,alpha,by,sigma)

    y = k*w;
    z = [(by(train_y+1) -y)/sigma, (by(train_y)-y)/sigma];
    N_z = normpdf(z);
    Psi_z = normcdf(z);
    delta = 1/sigma*(N_z(:,1)-N_z(:,2))./(Psi_z(:,1)-Psi_z(:,2));
    df = k'*delta + alpha.*w;

    f = 0.5*alpha'*(w.^2) - sum(log(Psi_z(:,1)-Psi_z(:,2)));
end
