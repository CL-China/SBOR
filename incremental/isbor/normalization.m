function [x,m,s] = normalization(x,m,s)
[N] = size(x,1);
if nargin == 1
    m = mean(x);
    s = std(x);
    s(s<0.001) = 1;
    
end
x = (x - repmat(m,N,1))./(repmat(s,N,1));
