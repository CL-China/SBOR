function write_out(output,i,name,data)
  output = output + 1e-10*randn(length(output),1); % Break ties at random
  fname = [data name '.fold' num2str(i)];
  save(fname,'output','-ascii');
  % Either copy the evaluation script in the current directory or
  % change the line below with the correct path 
  system(['perl Eval-Score-3.0.pl ./LETOR/' data '/Fold' num2str(i) '/'...
      name '.txt '  fname ' ' fname '.metric 0']);