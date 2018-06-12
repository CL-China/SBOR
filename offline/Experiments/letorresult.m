name = 'OHSUMED';

I =5;

MAP = zeros(I,1);
precision = zeros(I,16);
NDCG = zeros(I,16);

for i =1:I
    y = load([name,num2str(i)]);
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

save([name,'lambdamart'],'NDCG','precision','MAP','n_w')