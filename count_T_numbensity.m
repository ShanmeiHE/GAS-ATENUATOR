function [T_distribution,numdensity_distribution] = count_T_numbensity(r,z,Rp,Lp,n_r,n_z,v,m,kb)
%count temperature and num density in each counting cell, n_r is the cut
%number of Rp, n_z of Lp
%   此处显示详细说明
dr=Rp/n_r;
dz=Lp/n_z;
T_distribution=zeros(n_r,n_z);
n_distribution=zeros(n_r,n_z);
for i=1:n_r
    for j=1:n_z
        a=find(r<i*dr&r>=(i-1)*dr&z<j*dz&z>=(j-1)*dz);
        numdensity_distribution(i,j)=length(a);
        T_distribution(i,j)=m/3/kb*mean(v(a).^2);
    end
end
end

