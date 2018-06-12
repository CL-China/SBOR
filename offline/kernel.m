function K = kernel(x1,x2,eta,kernel_)
% x1 and x2 are N by M matrix, N is the number of samples.
    switch upper(kernel_(1:3))
        case 'GAU',
           d = dist(x1,x2);
           K = exp(-eta*d); 
        case 'LIN',
            K = x1;
            K = [K,ones(size(x1,1),1)];
        case 'LLY',
            K = eta*dist(x1,x2);
        case 'POL',
            p = str2num(kernel_(end));
            d = dist(x1,x2);            
            K = (1+eta*d).^p;
    end

    
%     K = x1;
end

function d = dist(x,y)
    nx = size(x,1);
    ny = size(y,1);
    d = sum(x.^2,2)*ones(1,ny) + ones(nx,1)*sum(y.^2,2)' -2*(x*y');
end