function [vxnew,vynew,vznew,xnew,ynew,znew] = diffuse_scattering_z(vx,vy,vz,x,y,z,Lp,dt,Tw,m )
%calculate the velocity and position after collision with wall in z plane,
%based on diffuse scattering
%the input position x/y/z are positions disregarding collision, i.e. the abs of z
%has already exceeded Lp; z>Lp&z<0 operates separately!!
%   此处显示详细说明
R=1.3806E-23/m;
n=length(vx); %the num of mol. to be considered
x0=x-vx.*dt; %initial position
y0=y-vy.*dt;
z0=z-vz.*dt;

if z(1)>Lp  %indicates that z exceeds the upmost boundary
t1=(z-Lp)./(z-z0).*dt; %t1 is the time period between dt-ti, ti is the time of collision
else
    t1=(0-z)./(z0-z).*dt;
end
xi=x-vx.*t1; %position when colliding with wall
yi=y-vy.*t1;
zi=z-vz.*t1;


%produce vr,vtheta,vz according to diffuse scattering
sigma=sqrt(R*Tw);
if z(1)>Lp  %indicates that z exceeds the upmost boundary
vznew=-rayl(n,sigma); %add signal -, because the mol. is sure to go back to the middle away from boundary
else
    vznew=rayl(n,sigma);
end
vxnew=normrnd(0,sigma,[n,1]);
vynew=normrnd(0,sigma,[n,1]);


%calculate actual positions after collision
xnew=xi+vxnew.*t1;
ynew=yi+vynew.*t1;
znew=zi+vznew.*t1;

subout1=find(znew>Lp); %the subscript of mol. that excess the boundary
%znew(subout1)
if length(subout1)>0
    %length(subout1)
    [vxnew(subout1),vynew(subout1),vznew(subout1),xnew(subout1),ynew(subout1),znew(subout1)] ...
 = specular_scattering_z(vx(subout1),vy(subout1),vz(subout1),x(subout1),y(subout1),z(subout1),Lp,dt);
end

subout2=find(znew<0); %the subscript of mol. that excess the boundary
if length(subout2)>0
    %length(subout2)
    [vxnew(subout2),vynew(subout2),vznew(subout2),xnew(subout2),ynew(subout2),znew(subout2)] ...
 = specular_scattering_z(vx(subout2),vy(subout2),vz(subout2),x(subout2),y(subout2),z(subout2),Lp,dt);
else
    return
end














%8.5 to test NaN in x/y/z, real reason is that z/r is infinity( but the
%reason for the infinity has not been discoverd)

% R=1.3806E-23/6.6359E-26;
% n=length(vx); %the num of mol. to be considered
% x0=x-vx.*dt; %initial position
% y0=y-vy.*dt;
% z0=z-vz.*dt;
% if length(find(z<Lp&z>0))~=0
%     disp('error');
%     return;
% end
% if z(1)>Lp  %indicates that z exceeds the upmost boundary
% t1=(z-Lp)./(z-z0).*dt; %t1 is the time period between dt-ti, ti is the time of collision
% else
%     t1=(0-z)./(z0-z).*dt;
% end
%  if sum(t1>1e5)>0
%         subt1=find(t1>1e5);
%     disp('t1 infinity');
%     disp('z z0'); 
%     z(subt1)
%     z0(subt1)
%     
% end
% if sum(isnan(t1))>0
%     disp('isnan z t1!');
% end
% 
% xi=x-vx.*t1; %position when colliding with wall
% yi=y-vy.*t1;
% zi=z-vz.*t1;
% 
% 
% %produce vr,vtheta,vz according to diffuse scattering
% sigma=sqrt(R*Tw);
% if z(1)>Lp  %indicates that z exceeds the upmost boundary
% vznew=-rayl(n,sigma); %add signal -, because the mol. is sure to go back to the middle away from boundary
% else
%     vznew=rayl(n,sigma);
% end
% vxnew=normrnd(0,sigma,[n,1]);
% vynew=normrnd(0,sigma,[n,1]);
% 
% if sum(isnan(vxnew))>0||sum(isnan(vynew))>0||sum(isnan(vznew))>0
%     disp('isnan z vxyz!');
% end
% %calculate actual positions after collision
% xnew=xi+vxnew.*t1;
% ynew=yi+vynew.*t1;
% znew=zi+vznew.*t1;
% if sum(isnan(xnew))>0||sum(isnan(ynew))>0||sum(isnan(znew))>0
%     disp('isnan z xyz!');
% end
% end
