function [xn,yn,phin,vxn,vyn] = boundary_phi_phimax(x,y,phi,phimax,vx,vy)
%boundary condition at phi=0
if isempty(find(phi<phimax))==0
    disp('boundary phi max error\n');
    return;
end
rxy=sqrt(x.^2+y.^2);
phin=phi-phimax;
xn=rxy.*cos(phin);
yn=rxy.*sin(phin);
vxn=vx*cos(phimax)+vy*sin(phimax);
vyn=-vx*sin(phimax)+vy*cos(phimax);
end


