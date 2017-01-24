function [e,t,y] = predict(train_x,test_x,test_y,w,idx,by,eta,ker)

if strcmp(upper(ker(1:3)),'LIN')
    if idx(end) > size(test_x,2)
        k = test_x(:,idx(1:end-1));
        k = [k,ones(size(test_x,1),1)];
    else
        k = test_x(:,idx);
    end
else
    k = kernel(test_x,train_x(idx,:),eta,ker);
end
y = k*w;


by = ones(size(test_x,1),1)*by';
t = sign(y*ones(1,size(by,2))-by);
t = sum(t,2);
t = (t-min(t))/2+1;
e = norm(t-test_y,1)/length(test_y);

% t1 = normcdf(by - y*ones(1,size(by,2)));
% p = diff(t1');
% [~,t1] = max(p);
% e1 = norm(t1'-test_y,1);

end