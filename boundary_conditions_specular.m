function [xn,yn,zn,rn,phin,vxn,vyn,vzn,vn,N_boundary] = boundary_conditions_specular( x,y,z,r,phi,vx,vy,vz,v,dt,Rp,Lp,phimax,T0,m)
%rebound at boundary
N_boundary=0;
%firstly rebound at r=Rp

ar=find(r>Rp);
if length(ar)>0
    N_boundary=N_boundary+length(ar);
    %fprintf('vi=%f\n',v(ar(1)));
    [vx(ar),vy(ar),vz(ar),x(ar),y(ar),z(ar)] = specular_scattering_xy(vx(ar),vy(ar),vz(ar) ...
  ,x(ar),y(ar),z(ar),Rp,dt);  
    r(ar)=sqrt(x(ar).^2+y(ar).^2);
    phi(ar)=atan2(y(ar),x(ar));
    v(ar)=sqrt(vx(ar).^2+vy(ar).^2+vz(ar).^2);
end

%secondly rebound at phi=0 or phi=phimax (periodical boundary condition)
while(1)
    aphi0=find(phi<0);
    
    if length(aphi0)>0
        N_boundary=N_boundary+length(aphi0);
        [x(aphi0),y(aphi0),phi(aphi0),vx(aphi0),vy(aphi0)] = boundary_phi_0(x(aphi0),y(aphi0),phi(aphi0),phimax,vx(aphi0),vy(aphi0));
        
    end
    if isempty(find(phi<0))
        break;
    end
end

while(1)
    aphimax=find(phi>phimax);
    if length(aphimax)>0
        N_boundary=N_boundary+length(aphimax);
        [x(aphimax),y(aphimax),phi(aphimax),vx(aphimax),vy(aphimax)] = boundary_phi_phimax(x(aphimax),y(aphimax),phi(aphimax),phimax,vx(aphimax),vy(aphimax));
        
    end
    if isempty(find(phi>phimax))
        break;
    end
end

%thirdly rebound at z=0 or z=Lp
azLp=find(z>Lp);
    if length(azLp)>0
    N_boundary=N_boundary+length(azLp);
    %fprintf('vi=%f\n',v(a(1)));
    [vx(azLp),vy(azLp),vz(azLp),x(azLp),y(azLp),z(azLp)] = specular_scattering_z(vx(azLp),vy(azLp),vz(azLp) ...
    ,x(azLp),y(azLp),z(azLp),Lp,dt);  
    r(azLp)=sqrt(x(azLp).^2+y(azLp).^2);
    phi(azLp)=atan2(y(azLp),x(azLp));
    v(azLp)=sqrt(vx(azLp).^2+vy(azLp).^2+vz(azLp).^2);
    %fprintf('vf=%f\n',v(a(1)));
    end
    
az0=find(z<0);
    if length(az0)>0
    N_boundary=N_boundary+length(az0);
    [vx(az0),vy(az0),vz(az0),x(az0),y(az0),z(az0)] = specular_scattering_z(vx(az0),vy(az0),vz(az0) ...
    ,x(az0),y(az0),z(az0),Lp,dt);  
    r(az0)=sqrt(x(az0).^2+y(az0).^2);
    phi(az0)=atan2(y(az0),x(az0));
    v(az0)=sqrt(vx(az0).^2+vy(az0).^2+vz(az0).^2);
    end
    
    xn=x;
    yn=y;
    zn=z;
    rn=r;
    phin=phi;
    vxn=vx;
    vyn=vy;
    vzn=vz;
    vn=v;
    
end

