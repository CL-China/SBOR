clear;

I =5;

MAP = zeros(I,1);
precision = zeros(I,16);
NDCG = precision;
n_w = zeros(I,1);
ker = 'linear';
% name = 'MQ2008_raw';
% name = 'MQ2007';
name = 'OHSUMED';
for i = 1:I
load(['./LETOR/',name,'/PCA/',name,'_',num2str(i),'.mat']); d = 2;

eta = 0.3;

% train_x = [train_x;vali_x];
% train_y = [train_y;vali_y];
[w,idx,by,alpha,sigma] = train(train_x,train_y,eta,ker);
[e(i),t,y] = predict(train_x,test_x,test_y,w,idx,by,eta,ker);
n_w(i) = length(w);

% save(['os',num2str(i)],'train_x','train_y','test_x','test_y','w','idx')

  write_out(y,i,'test',name);
  f = fopen([name, 'test.fold',num2str(i),'.metric']);
  l = fgetl(f);
  while l ~= -1
    tmp = sscanf(l,'%s:');
    l = strrep(l,tmp,'');
    n = sscanf(l,'%f');
    switch upper(tmp(1:3))
        case 'PRE',
            precision(i,:) = n';
        case 'MAP',
            MAP(i) = n;
        case 'NDC',
            NDCG(i,:) = n';
    end
    l = fgetl(f);
    if strcmp(l,'')
        l = fgetl(f);
    end
  end
  fclose(f);
end
MAP(I+1) = mean(MAP);
n_w(I+1) = mean(n_w);
precision(I+1,:) = mean(precision);
NDCG(I+1,:) = mean(NDCG);
fprintf('NDCG: '); fprintf('%f ',NDCG(I+1,1:10))
fprintf('\nMAP: %f \n', MAP(I+1))

save([name,datestr(now)],'NDCG','precision','MAP','n_w','e')





