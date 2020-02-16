function [particle_density] = particle_density0(dr,dz,dphi,r,phi,z,cellnumxy,sum_cell,VC,FN)

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
tic
INDEX=zeros(length(r(:,1)),1);

index_z = round(z/dz+0.5);
index_r = round(r/dr+0.5);
index_phi = round(phi./dphi(index_r)+0.5);
a = find(index_r~=1);
INDEX(a) = (index_z(a)-1)*cellnumxy+sum_cell(index_r(a)-1)+index_phi(a);
a = find(index_r==1);
INDEX(a) = (index_z(a)-1)*cellnumxy+0+index_phi(a);

%% density statistics
particle_density = ind2vec(INDEX');
particle_density = sum(particle_density,2);
particle_density = FN*particle_density./VC;
particle_density = full(particle_density);
toc
end
