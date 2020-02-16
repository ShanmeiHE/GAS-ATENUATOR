function [xn,yn,phin,vxn,vyn] = boundary_phi_0(r,phi,phimax,vx,vy)
%boundary condition at phi=0
if isempty(find(phi>0))==0
    disp('boundary phi 0 error\n');
    return;
end
rxy=sqrt(x.^2+y.^2);
phin=phi+phimax;
xn=rxy.*cos(phin);
yn=rxy.*sin(phin);
vxn=vx*cos(phimax)-vy*sin(phimax);
vyn=vx*sin(phimax)+vy*cos(phimax);

end


