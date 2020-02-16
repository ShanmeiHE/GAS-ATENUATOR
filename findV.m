function [speed] = findV(KE)
% E  is energy in Joule
% this function finds the velocity  in relativistic regime

c = 3*10^8;
m = 9.109*10^(-31);
gamma = 1 + KE/(m*c^2);
speed = c*sqrt(1-1./gamma.^2);

end

