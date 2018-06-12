d = dir('./datasets');
d(1:3) = [];

parfor i=length(d)
    run_offline(d(i).name);
end
