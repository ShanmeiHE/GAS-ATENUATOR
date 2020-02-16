function [xn,yn,zn,rn,phin] = free_motion( vx,vy,vz,x,y,z,dt )
%go on with new v during the time period of dt
xn=x+vx*dt;
    yn=y+vy*dt;
    zn=z+vz*dt;
    rn=sqrt(xn.^2+yn.^2);
    phin=atan2(yn,xn);

end

