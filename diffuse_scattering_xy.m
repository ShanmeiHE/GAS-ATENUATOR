function [vxnew,vynew,vznew,xnew,ynew,znew] = diffuse_scattering_xy(vx,vy,vz,x,y,z,Rp,dt,Tw,m )
%calculate the velocity and position after collision with wall in xy plane,
%based on diffuse scattering
%the input position x/y/z are positions disregarding collision, i.e. the r
%has already exceeded Rp
%   此处显示详细说明
R=1.3806E-23/m;
n=length(vx); %the num of mol. to be considered
x0=x-vx.*dt; %initial position
y0=y-vy.*dt;
z0=z-vz.*dt;
r0=sqrt(x0.^2+y0.^2);
r=sqrt(x.^2+y.^2);
if r<Rp
    disp('error');
    return;
end
t1=((r-Rp)./(r-r0)).*dt; %t1 is the time period between dt-ti, ti is the time of collision

xi=x-vx.*t1; %position when colliding with wall
yi=y-vy.*t1;
zi=z-vz.*t1;
%  rr=sqrt(xi.^2+yi.^2)

%calculate the degree of site of collision
theta=atan2(yi,xi);


%produce vr,vtheta,vz according to diffuse scattering
sigma=sqrt(R*Tw);
vr=-rayl(n,sigma); %add signal -, because the mol. is sure to go back to the middle away from boundary 
vtheta=normrnd(0,sigma,[n,1]);
vznew=normrnd(0,sigma,[n,1]);


%calculate vx,vy according to vr,vtheta
vxnew=vr.*cos(theta)-vtheta.*sin(theta);
vynew=vr.*sin(theta)+vtheta.*cos(theta);
%vnew=sqrt(vxnew.^2+vynew.^2+vznew.^2);

%calculate actual positions after collision
xnew=xi+vxnew.*t1;
ynew=yi+vynew.*t1;
znew=zi+vznew.*t1;
rnew=sqrt(xnew.^2+ynew.^2);
%rnew=sqrt(xnew.^2+ynew.^2)
subout=find(rnew>Rp); %the subscript of mol. that excess the boundary
if length(subout)>0
    [vxnew(subout),vynew(subout),vznew(subout),xnew(subout),ynew(subout),znew(subout)] ...
 = specular_scattering_xy(vx(subout),vy(subout),vz(subout),x(subout),y(subout),z(subout),Rp,dt);
end
end








%8.5 to test NaN in x/y/z, real reason is that z/r is infinity( but the
%reason for the infinity has not been discoverd)

% R=1.3806E-23/6.6359E-26;
% n=length(vx); %the num of mol. to be considered
% x0=x-vx.*dt; %initial position
% y0=y-vy.*dt;
% z0=z-vz.*dt;
% r0=sqrt(x0.^2+y0.^2);
% r=sqrt(x.^2+y.^2);
% if r<Rp
%     disp('error');
%     return;
% end
% t1=((r-Rp)./(r-r0)).*dt; %t1 is the time period between dt-ti, ti is the time of collision
% if sum(t1>1e5)>0
%     subt1=find(t1>1e5);
%     disp('t1 infinity');
%     disp('r r0'); 
%     r(subt1)
%     r0(subt1)
% end
% if sum(isnan(t1))>0
%     disp('isnandiffuset1!');
% end
% xi=x-vx.*t1; %position when colliding with wall
% yi=y-vy.*t1;
% zi=z-vz.*t1;
% % rr=sqrt(xi.^2+yi.^2)
% if sum(isnan(xi))>0||sum(isnan(yi))>0||sum(isnan(zi))>0
%     disp('isnan diffuse xi!');
% end
% %calculate the degree of site of collision
% theta=atan2(yi,xi);
% if sum(isnan(theta))>0
%     disp('isnantheta!');
%     sub=find(isnan(theta));
%     yi(sub)
%     xi(sub)
% end
% 
% %produce vr,vtheta,vz according to diffuse scattering
% sigma=sqrt(R*Tw);
% vr=-rayl(n,sigma); %add signal -, because the mol. is sure to go back to the middle away from boundary 
% vtheta=normrnd(0,sigma,[n,1]);
% vznew=normrnd(0,sigma,[n,1]);
% 
% 
% %calculate vx,vy according to vr,vtheta
% vxnew=vr.*cos(theta)-vtheta.*sin(theta);
% vynew=vr.*sin(theta)+vtheta.*cos(theta);
% %vnew=sqrt(vxnew.^2+vynew.^2+vznew.^2);
% if sum(isnan(vxnew))>0||sum(isnan(vynew))>0||sum(isnan(vznew))>0
%     disp('isnandiffuse v xyz!');
% end
% %calculate actual positions after collision
% xnew=xi+vxnew.*t1;
% ynew=yi+vynew.*t1;
% znew=zi+vznew.*t1;
% if sum(isnan(xnew))>0||sum(isnan(ynew))>0||sum(isnan(znew))>0
%     disp('isnandiffuse xyz!');
%     if sum(isnan(xnew))>0
%     subx=find(isnan(xnew));
%     disp('xnew vx\n'); 
%     xnew(subx) 
%     vxnew(subx)
%     disp('xi\n'); xi(subx)
%     disp('t1\n'); t1(subx) 
%     end
%     if sum(isnan(xnew))>0
%     suby=find(isnan(ynew));
%    disp('ynew vy\n'); ynew(suby) 
%    vynew(suby)
%    disp('yi\n'); yi(suby)
%    disp('t1\n'); t1(suby) 
%     end
%     if sum(isnan(znew))>0
%     subz=find(isnan(znew));
%    disp('znew vz\n'); znew(subz) 
%    vznew(subz) 
%    disp('zi\n'); zi(subz) 
%    disp('t1\n'); t1(subz) 
%     end
%     %disp('t1\n'); t1
% end
% %rnew=sqrt(xnew.^2+ynew.^2)
% end
