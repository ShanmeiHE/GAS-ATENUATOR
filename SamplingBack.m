function [r,z,phi,vx,vy,vz] = SamplingBack(amplification,r,z,phi,vx,vy,vz)
n = length(r)
n/amplification^3
a = round(unidrnd(n,round(n/amplification^3),1));
r = r(a);
z = z(a);
phi = phi(a);
vx = vx(a);
vy = vy(a);
vz = vz(a);
