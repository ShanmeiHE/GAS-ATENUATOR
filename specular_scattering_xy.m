function [vxnew,vynew,vznew,xnew,ynew,znew] = specular_scattering_xy(vx,vy,vz,x,y,z,Rp,dt)
%UNTITLED5 此处显示有关此函数的摘要
%   此处显示详细说明
n=length(vx); %the num of mol. to be considered
x0=x-vx.*dt; %initial position
y0=y-vy.*dt;
z0=z-vz.*dt;
r0=sqrt(x0.^2+y0.^2);
r=sqrt(x.^2+y.^2);
vr=zeros(n,1);
if r<Rp
    disp('error');
    return;
end
t1=((r-Rp)./(r-r0)).*dt; %t1 is the time period between dt-ti, ti is the time of collision

xi=x-vx.*t1; %position when colliding with wall
yi=y-vy.*t1;
zi=z-vz.*t1;


%calculate the degree of site of collision
theta=atan2(yi,xi);



vr=vx.*cos(theta)+vy.*sin(theta);
vtheta=-vx.*sin(theta)+vy.*cos(theta);
vznew=vz;
vr=-vr;

%calculate vx,vy according to vr,vtheta
vxnew=vr.*cos(theta)-vtheta.*sin(theta);
vynew=vr.*sin(theta)+vtheta.*cos(theta);


%calculate actual positions after collision
xnew=xi+vxnew.*t1;
ynew=yi+vynew.*t1;
znew=zi+vznew.*t1;
end

