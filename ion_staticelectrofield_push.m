function [ion1_position,ion1_velocity,ion2_position,ion2_velocity,RHO,rho_e,PHI,T_e,gridE_r] = ion_staticelectrofield_push(ion1_position,ion1_velocity,ion2_position,ion2_velocity,delta,delta_t,r,z,phi,W,T_e,PHI,RHO)

k_B=1.38064853E-23;  
sigm_iona=10^(-15);
m_ion=4.65E-26;
c=3*10^8;
epsilon=8.854187817*10^(-12);
mu=4*pi*10^(-7);
e=1.6*10^(-19);
delta_r = delta;
delta_z = delta;

gridr = 0:delta:r;
gridz = 0:delta_z:z;
len=length(gridr)*length(gridz);
[R,Z] = meshgrid(gridr,gridz);
R = R';
Z = Z';


    ion1_position_half = ion1_position + 1/2*delta_t*ion1_velocity;
    ion1_position_r = sqrt(ion1_position_half(:,1).^2+ion1_position_half(:,2).^2);

    ion2_position_half = ion2_position + 1/2*delta_t*ion2_velocity;
    ion2_position_r = sqrt(ion2_position_half(:,1).^2+ion2_position_half(:,2).^2);
    
    
    
    
    ion1_position_r = sqrt(ion1_position_half(:,1).^2+ion1_position_half(:,2).^2);

    
    ion2_position_r = sqrt(ion2_position_half(:,1).^2+ion2_position_half(:,2).^2);

    
    %gridj = zeros(length(gridr),length(gridz),2,'gpuArray');
    
    rho = [W*e*ones(length(ion1_position_r),1);W*2*e*ones(length(ion2_position_r),1)];
    rho_i = zeros(length(gridr),length(gridz),2);

    if length(rho)>0

        numberz = length(gridr);
        %a = zeros(length(J(:,1)),1,'gpuArray');
        a1 = zeros(length(ion1_position_r),1);
        ar1 = round(ion1_position_r/delta_r);
        az1 = round(ion1_position_half(:,3)/delta_z);
        a1 = ar1 + az1*numberz + 1;
        a1(a1<=0) = 1;
        a1(a1>=len+1) = len;

        
        
        a2 = zeros(length(ion2_position_r),1);
        ar2 = round(ion2_position_r/delta_r);
        az2 = round(ion2_position_half(:,3)/delta_z);
        a2 = ar2 + az2*numberz + 1;
        a2(a2<=0) = 1;
        a2(a2>=len+1) = len;

        a = [a1;a2];
        
        
        cell_e = ind2vec(a');
        rho_i = cell_e * rho;
        rho_i = sum(rho_i,2);
        grid_rho_i = zeros(length(gridr),length(gridz),1);
        grid_rho_i(1:length(rho_i)) = rho_i;
        grid_rho_i = grid_rho_i./(1/2*phi*((R+delta_r).^2-R.^2)*delta_z);        
    end
    
    n_e = length(ion1_position_half(:,1))+ 2*length(ion2_position_half(:,1));
    Energy_qPHI = trapz(gridz,trapz(gridr,RHO.*PHI.*R))
    Kinetic_ion = W*1/2*m_ion*sum(sum(ion1_velocity.^2)+sum(ion2_velocity.^2))
    Kinetic_e = W*k_B*T_e*n_e
    total_energy = Energy_qPHI + Kinetic_ion + Kinetic_e

    
    
    [gridE,rho_e,RHO,PHI,T_e ] = Poisson(grid_rho_i,r,z,T_e,delta,Energy_qPHI+Kinetic_e,n_e,W);
    

    ion1_position_r = sqrt(ion1_position_half(:,1).^2+ion1_position_half(:,2).^2);
    ion1_velocity_r = sqrt(ion1_velocity(:,1).^2+ion1_velocity(:,2).^2);
    sin_ion1_theta = ion1_position_half(:,2)./ion1_position_r;
    cos_ion1_theta = ion1_position_half(:,1)./ion1_position_r;
    ion1_velocity_r = ion1_velocity(:,1).*cos_ion1_theta +  ion1_velocity(:,2).*sin_ion1_theta;
    
    ion2_position_r = sqrt(ion2_position_half(:,1).^2+ion2_position_half(:,2).^2);
    ion2_velocity_r = sqrt(ion2_velocity(:,1).^2+ion2_velocity(:,2).^2);
    sin_ion2_theta = ion2_position_half(:,2)./ion2_position_r;
    cos_ion2_theta = ion2_position_half(:,1)./ion2_position_r;
    ion2_velocity_r = ion2_velocity(:,1).*cos_ion2_theta +  ion2_velocity(:,2).*sin_ion2_theta;


    ion_number = length(ion1_position_half(:,1))+ length(ion2_position_half(:,1));
    if ion_number > 0

        ar1 = round(ion1_position_r/delta_r);
        az1 = round(ion1_position_half(:,3)/delta_z);
        a1 = ar1 + az1*numberz + 1;
        a1(a1<=0) = 1;
        a1(a1>=len+1) = len;
        
        ar2 = round(ion2_position_r/delta_r);
        az2 = round(ion2_position_half(:,3)/delta_z);
        a2 = ar2 + az2*numberz + 1;
        a2(a2<=0) = 1;
        a2(a2>=len+1) = len;
        
        gridE_r = gridE(:,:,1);
        gridE_z = gridE(:,:,2);
        

        ion1_velocity_r;
        ion1_velocity_z = ion1_velocity(:,3);
        ion2_velocity_z = ion2_velocity(:,3);

        F_r1 = e*gridE_r(a1);
        F_z1 = e*gridE_z(a1);
        F_r2 = 2*e*gridE_r(a2);
        F_z2 = 2*e*gridE_z(a2);
        
        
        
        %fprintf('F_r\n');

        %F1 = zeros(length(F_r1),3,'gpuArray');
        %F2 = zeros(length(F_r2),3,'gpuArray');
        F1 = zeros(length(F_r1),3);
        F2 = zeros(length(F_r2),3);
        F1(:,1) = F_r1.*cos_ion1_theta;
        F1(:,2) = F_r1.*sin_ion1_theta;
        F1(:,3) = F_z1;
        
        F2(:,1) = F_r2.*cos_ion2_theta;
        F2(:,2) = F_r2.*sin_ion2_theta;
        F2(:,3) = F_z2;


        ion1_velocity = ion1_velocity + F1/m_ion*delta_t;
        ion2_velocity = ion2_velocity + F2/m_ion*delta_t;
        %mean(F2/m_ion*delta_t)
        
        
        ion1_position = ion1_position_half + 1/2*delta_t*ion1_velocity;
        ion2_position = ion2_position_half + 1/2*delta_t*ion2_velocity;
        
        n_e = length(ion1_position_half(:,1))+ 2*length(ion2_position_half(:,1));
        Energy_qPHI = trapz(gridz,trapz(gridr,RHO.*PHI.*R));
        Kinetic_ion = W*1/2*m_ion*sum(sum(ion1_velocity.^2)+sum(ion2_velocity.^2));
        
        
        

    end
    