function [particle_density] = particle_density0(dr,dz,dphi,position,cellnumxy,sum_cell,VC,FN)

%% parameter of cell (all given in the main.m)
% dr
% dz
% dphi
% cellnumxy
% sum_cell
% VC                    volume of the cell

%% position of particle
% r
% z
% phi

%% FN  statistical weight



%% number the particles
index_z = round(position(:,2)/dz+0.5);
index_r = round(position(:,1)/dr+0.5);
index_phi = round(position(:,3)./dphi(index_r)+0.5);
a = index_r~=1;
sum_cell0 = [0;sum_cell];
INDEX = (index_z-1)*cellnumxy+sum_cell0(index_r).*a+index_phi;

%% density statistics
particle_density = ind2vec(INDEX');
particle_density = sum(particle_density,2);
particle_density = FN*particle_density./VC;
particle_density = full(particle_density);

end
