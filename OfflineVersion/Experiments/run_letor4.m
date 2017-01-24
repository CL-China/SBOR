clear;

I =5;

MAP = zeros(I,1);
precision = zeros(I,10);
NDCG = zeros(I,11);
n_w = zeros(I,1);
ker = 'linear';
name = 'MQ2008_raw';
name = 'MQ2007';
% name = 'OHSUMED';
for i = 1:I
load(['./LETOR/',name,'/',name,num2str(i),'.mat']); d = 2;

eta = 0.3;

% train_x = [train_x;vali_x];
% train_y = [train_y;vali_y];
[w,idx,by,alpha,sigma] = train(train_x,train_y,eta,ker);
[~,t,y] = predict(train_x,test_x,test_y,w,idx,by,eta,ker);
n_w(i) = length(w);

% save(['os',num2str(i)],'train_x','train_y','test_x','test_y','w','idx')

  write_out4(y,i,'test',name);
  f = fopen([name, 'test.fold',num2str(i),'.metric']);
  l = fgetl(f);
  i_tmp = 1;
  while l ~= -1
    if strcmp(l(1:3),'qid')
        l = fgetl(f);
        continue
    end 
    tmp = sscanf(l,'%s.');
    l = strrep(l,tmp,'');
    n = sscanf(l,'%f');
    if i_tmp ==1
        precision(i,:) = n(1:10)';
        MAP(i) = n(end);
        i_tmp = i_tmp+1;
    else
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

save([name,datestr(now)],'NDCG','precision','MAP','n_w')





