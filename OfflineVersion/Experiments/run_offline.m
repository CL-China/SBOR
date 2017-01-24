function run_offline(name)
fprintf(['Cross Validation ',name])

thetaList = [0.001,0.01,0.1,1,10,100,1000];
thetaLength = length(thetaList);
predir = './datasets/';
d = dir([predir,name,'/matlab']);
d(1:2) = [];
I = length(d)/2;
rv = 1000*ones(I,1);
MAE = rv;
thetaIdx = MAE;
MAEList = 1000*ones(I,thetaLength);

ker = 'gauss';
runingTime = 0;
for epoch = 1:I
    train = load([predir,name,'/matlab/',d(epoch+I/2).name]);
    test = load([predir,name,'/matlab/',d(epoch).name]);
    train_x = train(:,1:end-1);
    train_y = train(:,end);
    train_x = normalization(train_x);
    test_x = train_x(1:5:end,:);
    test_y = train_y(1:5:end,:);
    train_x(1:5:end,:) = [];
    train_y(1:5:end,:) = [];
    
    for i_theta = 1:thetaLength
        eta = thetaList(i_theta);
        try
            [w,idx,by] = trainSBOR(train_x,train_y,eta,ker);
            MAEList(epoch,i_theta) = predict(train_x,test_x,test_y,w,idx,by,eta,ker);
        catch
            fprintf('error in CV %d, %d ',epoch,i_theta);
        end
    end
    [~,thetaIdx(epoch)] = min(MAEList(epoch,:));
    
    train_x = train(:,1:end-1);
    train_y = train(:,end);
    test_x = test(:,1:end-1);
    test_y = test(:,end);
    [train_x,m,s] = normalization(train_x);
    test_x = normalization(test_x,m,s);
    
    tic;
    try
        [w,idx,by] = trainSBOR(train_x,train_y,thetaList(thetaIdx(epoch)),ker);
        rv(epoch) = length(w);
        MAE(epoch) = predict(train_x,test_x,test_y,w,idx,by,thetaList(thetaIdx(epoch)),ker);
    catch
    end
    runingTime = runingTime+toc;
end
meanMAE = mean(MAE);
rvMean = mean(rv);
save(''-mat7-binary', '['./results/',name,'.mat'],...
    'MAEList','thetaIdx','MAE','rvMean','meanMAE','runingTime');
fprintf('\n')
end