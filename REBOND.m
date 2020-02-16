function [particle_position,particle_velocity,N_boundary] = REBOND( particle_position,particle_r,particle_phi,particle_velocity,Lp,PHIp)
%rebound at boundary

N_boundary=0;

    aphi0_total = gpuArray([]);
    n = 0;
    while(1)
        aphi0=find(particle_phi<0);
        n = n+1;
        if n == 1
            aphi0_first = aphi0;
        end
        if length(aphi0)>0
            N_boundary=N_boundary+length(aphi0);
            particle_phi(aphi0)=particle_phi(aphi0)+PHIp;
            particle_velocity_1 = particle_velocity(aphi0,1)*cos(PHIp)-particle_velocity(aphi0,2)*sin(PHIp);
            particle_velocity(aphi0,2) = particle_velocity(aphi0,1)*sin(PHIp)+particle_velocity(aphi0,2)*cos(PHIp);
            particle_velocity(aphi0,1) = particle_velocity_1;
        else
            break;
        end
    end
    
    particle_position(aphi0_first,1) = particle_r(aphi0_first).*cos(particle_phi(aphi0_first));
    particle_position(aphi0_first,2) = particle_r(aphi0_first).*sin(particle_phi(aphi0_first));
    
    
    
    aphimax_total = gpuArray([]);
    n = 0;
    while(1)
        aphimax=find(particle_phi>PHIp);
        n = n+1;
        if n == 1
            aphimax_first = aphimax;
        end
        if length(aphimax)>0
            N_boundary=N_boundary+length(aphimax);
            particle_phi(aphimax)=particle_phi(aphimax)-PHIp;
            particle_velocity_1 = particle_velocity(aphimax,1)*cos(PHIp)+particle_velocity(aphimax,2)*sin(PHIp);
            particle_velocity(aphimax,2) = -particle_velocity(aphimax,1)*sin(PHIp)+particle_velocity(aphimax,2)*cos(PHIp);
            particle_velocity(aphimax,1) = particle_velocity_1;
        else
            break;
        end
    end
    particle_position(aphimax_first,1) = particle_r(aphimax_first).*cos(particle_phi(aphimax_first));
    particle_position(aphimax_first,2) = particle_r(aphimax_first).*sin(particle_phi(aphimax_first));

    %thirdly rebound at z=0 or z=Lp
    azLp=find(particle_position(:,3)>Lp);
    if length(azLp)>0
        N_boundary=N_boundary+length(azLp);
        %fprintf('vi=%f\n',v(a(1)));
        particle_position(azLp,3) = 2*Lp-particle_position(azLp,3);
        particle_velocity(azLp,3) = -particle_velocity(azLp,3);
        %fprintf('vf=%f\n',v(a(1)));
    end

    az0=find(particle_position(:,3)<0);
    if length(az0)>0
        N_boundary=N_boundary+length(az0);
        particle_position(az0,3) = -particle_position(az0,3);
        particle_velocity(az0,3) = -particle_velocity(az0,3);
    end
    
   
    
end

