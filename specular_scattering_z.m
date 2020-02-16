function [vxnew,vynew,vznew,xnew,ynew,znew] = specular_scattering_z(vx,vy,vz,x,y,z,Lp,dt)
% z>Lp&z<0 operates separately!!
%   此处显示详细说明
% vv=sqrt(vx.^2+vy.^2+vz.^2)
n=length(vx); %the num of mol. to be considered
x0=x-vx.*dt; %initial position
y0=y-vy.*dt;
z0=z-vz.*dt;
vr=zeros(n,1);
if length(find(z<Lp&z>0))~=0
    disp('error');
    return;
end
if z(1)>Lp  %indicates that z exceeds the upmost boundary
t1=(z-Lp)./(z-z0).*dt; %t1 is the time period between dt-ti, ti is the time of collision
else
    t1=(0-z)./(z0-z).*dt;
end

xi=x-vx.*t1; %position when colliding with wall
yi=y-vy.*t1;
zi=z-vz.*t1;

vznew=-vz; 
vxnew=vx;
vynew=vy;
%  vnew=sqrt(vxnew.^2+vynew.^2+vznew.^2)

%calculate actual positions after collision
xnew=xi+vxnew.*t1;
ynew=yi+vynew.*t1;
znew=zi+vznew.*t1;

end

