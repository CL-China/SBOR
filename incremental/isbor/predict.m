function [MAE,t,y,MZE] = predict(train_x,test_x,test_y,w,idx,by,eta,sigma)


k = kernel(test_x,train_x(idx,:),eta);
y = k*w;


by = ones(size(test_x,1),1)*by';
t = sign(y*ones(1,size(by,2))-by);
t = sum(t,2);
t = (t-min(t))/2+1;
MAE = norm(t-test_y,1)/length(test_y);
MZE = sum(t~=test_y)/length(test_y);
% t1 = normcdf(by - y*ones(1,size(by,2))/sigma);
% p = diff(t1');
% [~,t1] = max(p);
% e1 = norm(t1'-test_y,1)/length(test_y);

end