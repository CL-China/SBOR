clear;clc;

addpath('isbor')


I = 1; 
e = zeros(I,1);
e2 = e;
K = 10;

theta_list = [0.1:.05:1,1.1:.2:1.8,2:0.5:5,6:10];
kernel_ = 'gauss';
name = 'new_toy1000';
dddd = './datasets/new_toy';
% dddd = '../datasets';

for j = 1:I
    d=1;
    data = load([dddd,'/',name,'/matlab/train_',name,'.',num2str(j-1)]);
    test = load([dddd,'/',name,'/matlab/test_',name,'.',num2str(j-1)]);

%     data = load(['./datasets/ordinal-regression/',name,'/matlab/train_',name,'.',num2str(j-1)]);
%     test = load(['./datasets/ordinal-regression/',name,'/matlab/test_',name,'.',num2str(j-1)]);
%     data = data(1:5:end,:);test = test(1:5:end,:);
%     data(data(:,end)>5,:) = [];
%     test(test(:,end)>5,:) = [];
    train_x = data(:,1:end-1);
    train_y = data(:,end);
    test_x = test(:,1:end-1);
    test_y = test(:,end);
% 
    tmp = normalization([train_x;test_x]);
    train_x = tmp(1:length(train_x),:);
    test_x = tmp(length(train_x)+1:end,:);
%     
    
    basisWidth = 0.05;
    dimension = size(train_x,2);
    basisWidth	= basisWidth^(1/dimension);
    OPTIONS = SBOR_Options('THETA',0.1,...
        'fixedNoise',true,...
        'ITERATIONS', 50);
    tic;
    [w,idx,by,logMarginal,sigma,e1] = ISBOR(train_x,train_y,OPTIONS);
    toc
    [e2(j),t1,y1] = predict(train_x,train_x,train_y,w,idx,by,OPTIONS.theta,sigma);
    fprintf('train error rate: %f \n',e2(j));
    [e(j),t,y,MZE] = predict(train_x,test_x,test_y,w,idx,by,OPTIONS.theta,sigma);
    fprintf('test error rate: %f \n',e(j));
    
    
    %% drawing the result
    if d ==1
        eta = OPTIONS.theta;
%         ker = OPTIONS.kernel;
        X = test_x;
        box	= 1.1*[min(X(:,1)) max(X(:,1)) min(X(:,2)) max(X(:,2))];
        
        gsteps		= 50;
        range1		= box(1):(box(2)-box(1))/(gsteps-1):box(2);
        range2		= box(3):(box(4)-box(3))/(gsteps-1):box(4);
        [grid1 grid2]	= meshgrid(range1,range2);
        Xgrid		= [grid1(:) grid2(:)];
        %
        %
        PHI		= kernel(Xgrid,train_x(idx,:),eta);
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
    end
end
fprintf('mean train error rate: %f \n', mean(e2));
fprintf('mean test error rate: %f \n', mean(e));
