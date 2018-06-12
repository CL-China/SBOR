clear;

I = 1;
e = zeros(I,1);
e2 = e;
name = ['newthyroid','tae'];
ker = 'gauss';
name = 'toy';
dic = 'ordinal-regression';
% dic = 'discretized-regression/5bins';
for i = 1:I
% data = load(['../data/train_toy.',num2str(i)]);

data = load(['../datasets/',name,'/matlab/train_',name,'.',num2str(i-1)]);

train_x = data(:,1:end-1);
train_y = data(:,end);


% test = load(['../data/test_toy.',num2str(i)]);
test = load(['../datasets/',name,'/matlab/test_',name,'.',num2str(i-1)]);

test_x = test(:,1:end-1);
test_y = test(:,end);

tmp = normalization([train_x;test_x]);
train_x = tmp(1:length(train_x),:);
test_x = tmp(length(train_x)+1:end,:);
    

eta = 1;

[w,idx,by,MAE] = trainSBOR(train_x,train_y,eta,ker);



[e2(i),~,~] = predict(train_x,train_x,train_y,w,idx,by,eta,ker);
[e(i),t,y] = predict(train_x,test_x,test_y,w,idx,by,eta,ker);

fprintf('train error rate: %f \n',e2(i));
fprintf('test error rate: %f \n',e(i));
end
fprintf('mean train error rate: %f \n', mean(e2));
fprintf('mean test error rate: %f \n', mean(e));

%% drawing the result
X = test_x;
box	= 1.1*[min(X(:,1)) max(X(:,1)) min(X(:,2)) max(X(:,2))];

gsteps		= 50;
range1		= box(1):(box(2)-box(1))/(gsteps-1):box(2);
range2		= box(3):(box(4)-box(3))/(gsteps-1):box(4);
[grid1, grid2]	= meshgrid(range1,range2);
Xgrid		= [grid1(:) grid2(:)];
%
% 
if strcmp(upper(ker(1:3)),'LIN')
    if idx(end) > size(Xgrid,2)
        PHI = Xgrid(:,idx(1:end-1));
        PHI = [PHI,ones(size(Xgrid,1),1)];
    else
        PHI = Xgrid(:,idx);
    end
else
    PHI	= kernel(Xgrid,train_x(idx,:),eta,ker);
end
y_grid	= -PHI*w;

by = ones(size(y_grid,1),1)*by';
p_grid = diff(normcdf(y_grid*ones(1,size(by,2))+by)')';

color = ['y' 'm' 'c' 'r' 'g' 'b' 'w' 'k'];

figure; hold on; 
for i = 1:max(train_y); 
    plot(train_x(train_y==i,1),train_x(train_y==i,2),'*','Color',color(i));
end

for i = 1:max(train_y); 
    plot(test_x(test_y==i,1),test_x(test_y==i,2),'.','Color',color(i));
end

[~,bar] = max(p_grid,[],2);

M = range(train_y)+1;
for i = 1:M
   if max(p_grid(:,i)) < 0.5
       contour(range1,range2,reshape(p_grid(:,i),size(grid1)),...
        [max(p_grid(:,i))*0.9,max(p_grid(:,i))*0.9],'--','Color',color(i),'LineWidth',3);
   else
    contour(range1,range2,reshape(p_grid(:,i),size(grid1)),...
        [0.5,0.5],'--','Color',color(i),'LineWidth',3);
   end
end

legend();


