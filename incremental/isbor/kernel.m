function K = kernel(x1,x2,eta)
% x1 and x2 are N by M matrix, N is the number of samples.
    d = dist(x1,x2);
    K = exp(-eta*d);
end

function d = dist(x,y)
    nx = size(x,1);
    ny = size(y,1);
    d = sum(x.^2,2)*ones(1,ny) + ones(nx,1)*sum(y.^2,2)' -2*(x*y');
end