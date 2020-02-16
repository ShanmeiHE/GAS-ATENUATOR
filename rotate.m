function [ vxn,vyn,vzn ] = rotate( vx,vy,vz,theta,phi )
%rotate from initial v theta&phi
%refers to 1.131 from guangdianxiaoying & text by myself
v=sqrt(vx^2+vy^2+vz^2);
if v^2-vz^2>1E-5
vxn=vx*cos(theta)+sin(theta)/sqrt(v^2-vz^2)*(vx*vz*cos(phi)-v*vy*sin(phi));
vyn=vy*cos(theta)+sin(theta)/sqrt(v^2-vz^2)*(vy*vz*cos(phi)+v*vx*sin(phi));
vzn=vz*cos(theta)-sqrt(v^2-vz^2)*sin(theta)*cos(phi);
else
    vxn=v*sin(theta)*cos(phi);
    vyn=v*sin(theta)*sin(phi);
    vzn=v*cos(theta);
end

