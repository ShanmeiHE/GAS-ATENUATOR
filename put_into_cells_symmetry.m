function [INDEX] = put_into_cells_symmetry(rmax,phimax,zmax,r,phi,z,Rp,Lp,PHIp,cellnumxy,sum_cell)
%put N_total mol. into cells, and sign the mol. with INDEX according to the
%cell where it locates
dz = Lp/zmax;
dr = Rp/rmax;
dphi = PHIp*ones(length(phimax),1)./phimax; 


INDEX=zeros(length(r),1,'gpuArray');
index_z = ceil(z/dz);
index_r = ceil(r/dr);
index_phi = ceil(phi./dphi(index_r));
a = find(index_r~=1);
INDEX(a) = (index_z(a)-1)*cellnumxy+sum_cell(index_r(a)-1)'+index_phi(a);
a = find(index_r==1);
INDEX(a) = (index_z(a)-1)*cellnumxy+0+index_phi(a);


end

