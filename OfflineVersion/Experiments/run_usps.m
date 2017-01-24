clear;
warning('off');
load usps.mat
runtime = zeros(10,1);
ker = 'gauss';
thetaList = [0.001,0.01,0.1,1];
thetaIdx = [3;3;3;2;2;2;3;2;2;2];
e = 100*ones(10,1);
for i =1:10
    x = feature(1:i*700,:);
    t = label(1:i*700);
    
    eta = thetaList(thetaIdx(i));
    tic;
    try 
        [w,idx,by] = trainSBOR(x,t,eta,ker);
        e(i) = predict(x,feature,label,w,idx,by,eta,ker);
    catch
        fprintf('no results in iteration %d', i);
    end
    runtime(i) = toc;
    fprintf('finish %d',i)
	rv(i) = length(w);
end
rvMean = mean(rv);
save('-mat7-binary','./results/usps','runtime','e','rv','rvMean')
