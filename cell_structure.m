function [cellnum,dphi,phi_divide,sum_cell,cellr,cellphi] = cell_structure( PHIp,Rp,rmax,delta)
%calculate the num of cells of one plane sector, and dphi of each r
%sum_cell is the sum of cell for r<ri


dphi=zeros(rmax,1);
phi_divide=zeros(rmax,1);
sum_cell=zeros(rmax,1);
i = 1:rmax;
l=i/rmax*Rp*PHIp;
phi_divide=ceil(l/delta);
dphi = PHIp.*ones(1,rmax)./phi_divide;
sum_cell=[];
cellr = [];
cellphi =[];

for k = 1:rmax
    sum_cell(k) = sum(phi_divide(1:k));
    cellr = [cellr;k*delta*ones(phi_divide(k),1)];
    l = 1:phi_divide(k);
    cellphi = [cellphi;PHIp/phi_divide(k)*l'];
end
cellr = gpuArray(cellr);
cellphi = gpuArray(cellphi);
cellnum=sum_cell(end);



