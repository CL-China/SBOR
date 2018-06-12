angle = linspace(-pi, pi)';
xbase = [0.7*sin(angle) + rand(length(angle), 1)/10, 0.7*cos(angle) + rand(length(angle), 1)/10;
    sin(angle) + rand(length(angle), 1)/10, cos(angle) + rand(length(angle), 1)/10;
    1.2*sin(angle) + rand(length(angle), 1)/10, 1.2*cos(angle) + rand(length(angle), 1)/10];
plot(xbase(:,1), xbase(:,2), '*')