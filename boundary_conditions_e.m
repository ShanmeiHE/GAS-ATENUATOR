function [xn,yn,zn,phin,vxn,vyn,vzn] = boundary_conditions_e( x,y,z,phi,vx,vy,vz,phimax)
%periodic boundary conditions for electrons

%rebound at phi=0 or phi=phimax (periodical boundary condition)
while(1)
    aphi0=find(phi<0);
    
    if length(aphi0)>0
        
        [x(aphi0),y(aphi0),phi(aphi0),vx(aphi0),vy(aphi0)] = boundary_phi_0(x(aphi0),y(aphi0),phi(aphi0),phimax,vx(aphi0),vy(aphi0));
        
    end
    if isempty(find(phi<0))
        break;
    end
end

while(1)
    aphimax=find(phi>phimax);
    if length(aphimax)>0
        
        [x(aphimax),y(aphimax),phi(aphimax),vx(aphimax),vy(aphimax)] = boundary_phi_phimax(x(aphimax),y(aphimax),phi(aphimax),phimax,vx(aphimax),vy(aphimax));
        
    end
    if isempty(find(phi>phimax))
        break;
    end
end

    
    xn=x;
    yn=y;
    zn=z;
    phin=phi;
    vxn=vx;
    vyn=vy;
    vzn=vz;
    
end

