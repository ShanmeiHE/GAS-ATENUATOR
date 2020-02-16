function [ vxn,vyn,vzn ] = rotate0( vx,vy,vz,theta,phi )
%rotate from initial v theta&phi
%refers to 1.131 from guangdianxiaoying & text by myself
vxn = zeros(length(vx),1,'gpuArray');
vyn = zeros(length(vx),1,'gpuArray');
vzn = zeros(length(vx),1,'gpuArray');

v=sqrt(vx.^2+vy.^2+vz.^2);
a = find(v.^2-vz.^2>1E-5);
vxn(a) = vx(a).*cos(theta(a))+sin(theta(a))./sqrt(v(a).^2-vz(a).^2).*(vx(a).*vz(a).*cos(phi(a))-v(a).*vy(a).*sin(phi(a)));
vyn(a) = vy(a).*cos(theta(a))+sin(theta(a))./sqrt(v(a).^2-vz(a).^2).*(vy(a).*vz(a).*cos(phi(a))+v(a).*vx(a).*sin(phi(a)));
vzn(a) = vz(a).*cos(theta(a))-sqrt(v(a).^2-vz(a).^2).*sin(theta(a)).*cos(phi(a));

b = find(v.^2-vz.^2<1E-5);
vxn(b) = v(b).*sin(theta(b)).*cos(phi(b));
vyn(b) = v(b).*sin(theta(b)).*sin(phi(b));
vzn(b) = v(b).*cos(theta(b));


