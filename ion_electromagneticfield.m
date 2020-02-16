function [ion1_position,ion1_velocity,ion2_position,ion2_velocity,gridE_r,gridE_z,gridH_theta] = ion_electromagneticfield(ion1_position,ion1_velocity,ion2_position,ion2_velocity,gridE_r,gridE_z,gridH_theta,delta,delta_t,r,z,phi,W)


sigma=10^(-15);
me=9.1096*10^(-31);
c=3*10^8;
epsilon=8.854187817*10^(-12);
mu=4*pi*10^(-7);
e=1.6*10^(-19);
delta_r = delta;
delta_z = delta;

H_theta_gridr = delta_r/2:delta_r:r-delta_r/2;
H_theta_gridz = delta_z/2:delta_z:r-delta_z/2;

H_theta_lenr = length(H_theta_gridr);
H_theta_lenz = length(H_theta_gridz);
H_theta_len = length(H_theta_gridr)*length(H_theta_gridz);

[H_theta_R,H_theta_Z] = meshgrid(H_theta_gridr,H_theta_gridz);
H_theta_R = H_theta_R';
H_theta_Z = H_theta_Z';



E_r_gridr = delta_r/2:delta_r:r-delta_r/2;
E_r_gridz = 0:delta_z:r;

E_r_lenr = length(E_r_gridr);
E_r_lenz = length(E_r_gridz);
E_r_len = length(E_r_gridr)*length(E_r_gridz);

[E_r_R,E_r_Z] = meshgrid(E_r_gridr,E_r_gridz);
E_r_R = E_r_R';
E_r_Z = E_r_Z';



E_z_gridr = 0:delta_r:r;
E_z_gridz = delta_z/2:delta_z:r-delta_z/2;

E_z_lenr = length(E_z_gridr);
E_z_lenz = length(E_z_gridz);
E_z_len = length(E_z_gridr)*length(E_z_gridz);

[E_z_R,E_z_Z] = meshgrid(E_z_gridr,E_z_gridz);
E_z_R = E_z_R';
E_z_Z = E_z_Z';





ion1_position_half = ion1_position + 1/2*delta_t*ion1_velocity;
ion1_position_r = sqrt(ion1_position_half(:,1).^2+ion1_position_half(:,2).^2);

ion2_position_half = ion2_position + 1/2*delta_t*ion2_velocity;
ion2_position_r = sqrt(ion2_position_half(:,1).^2+ion2_position_half(:,2).^2);

    

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
    
    
    if length(J)>0

        numberz = length(gridr);
        %a = zeros(length(J(:,1)),1,'gpuArray');
        a = zeros(length(particle_position_r),1);
        ar = round(particle_position_r/delta_r-0.5);
        az = round(particle_position_half(:,3)/delta_z);
        a = ar + az*numberz + 1;
        
        Jr = J(:,1);
        Jz = J(:,2);

        a = [a;a2];
        
        
        vr = [particle_velocity_r;2*ion2_velocity_r];
        vz = [particle_velocity(:,3);2*ion2_velocity(:,3)];
        cell_e = ind2vec(a');
        jr = cell_e * vr;
        jz = cell_e * vz;
        jr = sum(jr,2);
        jz = sum(jz,2);
        gridjr = zeros(length(gridr),length(gridz),1);
        gridjz = zeros(length(gridr),length(gridz),1);
        gridjr(1:length(jr)) = jr;
        gridjz(1:length(jz)) = jz;
        gridj(:,:,1) = W*e*gridjr./(phi*((R+delta_r).^2-R.^2)*delta_z);
        gridj(:,:,2) = W*e*gridjz./(phi*((R+delta_r).^2-R.^2)*delta_z);
        
    end
    
    %{
    energy_of_electric_field = 0;
    energy_of_magnetic_field = 0;
    J_E = 0;
    for i = 1:(length(gridE(1,:,1))-1)
        energy_of_electric_field = energy_of_electric_field + sum(gridE(i,:,1).^2+gridE(i,:,2).^2) *(i^2-(i-1)^2)*phi*delta_z*delta_r^2;
        energy_of_magnetic_field = energy_of_magnetic_field + sum(gridB_theta(i,:).^2) *(i^2-(i-1)^2)*phi*delta_z*delta_r^2;
        J_E  = J_E + sum(gridE(i,:,1).*gridj(i,:,1)+gridE(i,:,2).*gridj(i,:,2)) *(i^2-(i-1)^2)*phi*delta_z*delta_r^2;
    end
    energy_of_electric_field_before = 1/2*epsilon*energy_of_electric_field;
    energy_of_magnetic_field_before = 1/2/mu*energy_of_magnetic_field;
    gridEbefore = gridE;

    %}
    for repeat=1:2
        delta_t_repeat =delta_t/2;
        [gradient_Er_pz,~] = gradient(gridE(:,:,1),delta_z,delta_r);
        [~,gradient_Ez_pr] = gradient(gridE(:,:,2),delta_z,delta_r);
        curlE_theta = gradient_Er_pz - gradient_Ez_pr;

        gridB_theta(:,:) = gridB_theta(:,:)-curlE_theta*delta_t_repeat;

        [gradient_Btheta_pr,gradient_Btheta_pz] = gradient(gridB_theta,delta_r,delta_z);
        curlB_r = -gradient_Btheta_pz;
        curlB_z = 1./(R+delta_r).*gridB_theta + gradient_Btheta_pr ;

        gridE(:,:,1) = gridE(:,:,1) + delta_t_repeat/epsilon*(curlB_r*mu-gridj(:,:,1));
        gridE(:,:,2) = gridE(:,:,2) + delta_t_repeat/epsilon*(curlB_z*mu-gridj(:,:,2));       
    end
%{
    energy_of_electric_field = 0;
    energy_of_magnetic_field = 0;
    J_E = 0;
    E_Ept = 0;
    
    
    for i = 1:(length(gridE(1,:,1))-1)
        energy_of_electric_field = energy_of_electric_field + sum(gridE(i,:,1).^2+gridE(i,:,2).^2) *(i^2-(i-1)^2)*phi*delta_z*delta_r^2;
        energy_of_magnetic_field = energy_of_magnetic_field + sum(gridB_theta(i,:).^2) *(i^2-(i-1)^2)*phi*delta_z*delta_r^2;
        J_E  = J_E + sum(gridE(i,:,1).*gridj(i,:,1)+gridE(i,:,2).*gridj(i,:,2)) *(i^2-(i-1)^2)*phi*delta_z*delta_r^2;
        E_Ept = E_Ept + sum(gridE(i,:,1).*(gridE(i,:,1)-gridEbefore(i,:,1))+gridE(i,:,2).*(gridE(i,:,2)-gridEbefore(i,:,2))) *(i^2-(i-1)^2)*phi*delta_z*delta_r^2/delta_t;        
    end
    
    energy_of_electric_field = 1/2*epsilon*energy_of_electric_field;
    energy_of_magnetic_field = 1/2/mu*energy_of_magnetic_field;
    %}
    %{
    fprintf('deltaE\n')
    (energy_of_electric_field - energy_of_electric_field_before)
    fprintf('E*dE/dt')
    E_Ept*epsilon*delta_t
    fprintf('J_E')
    J_E*delta_t
    %}
    

    ion_number = length(particle_position_half(:,1))+ length(ion2_position_half(:,1));
    if ion_number > 0

        ar = round(particle_position_r/delta_r-0.5);
        az = round(ion1_position_half(:,3)/delta_z);
        a = ar + az*numberz + 1;
        
        ar2 = round(ion2_position_r/delta_r-0.5);
        az2 = round(ion2_position_half(:,3)/delta_z);
        a2 = ar2 + az2*numberz + 1;
        
        gridE_r = gridE(:,:,1);
        gridE_z = gridE(:,:,2);
        

        ion1_velocity_r;
        ion1_velocity_z = ion1_velocity(:,3);
        ion2_velocity_z = ion2_velocity(:,3);

        F_r1 = e*gridE_r(a) + e*(-ion1_velocity_z.*gridB_theta(a));
        F_z1 = e*gridE_z(a) + e*(ion1_velocity_r.*gridB_theta(a));
        F_r2 = 2*e*gridE_r(a2) + 2*e*(-ion2_velocity_z.*gridB_theta(a2));
        F_z2 = 2*e*gridE_z(a2) + 2*e*(ion2_velocity_r.*gridB_theta(a2));
        
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
%{
        W*(sum(sum(F1.*ion1_velocity))+sum(sum(F2.*ion2_velocity)));
        J_E;
        nonrelative_kinectic_before = W*1/2*m_ion*(sum(sum(ion1_velocity.^2))+sum(sum(ion2_velocity.^2)));
%}
        ion1_velocity = ion1_velocity + F1/m_ion*delta_t;
        ion2_velocity = ion2_velocity + F2/m_ion*delta_t;
        
%{
        nonrelative_kinectic = W*1/2*m_ion*(sum(sum(ion1_velocity.^2))+sum(sum(ion2_velocity.^2)));
        delta_nonrelative_kinectic = nonrelative_kinectic - nonrelative_kinectic_before;
%}
        

        ion1_position = ion1_position_half + 1/2*delta_t*ion1_velocity;
        ion2_position = ion2_position_half + 1/2*delta_t*ion2_velocity;
        
        
        gridE(:,length(gridE(:,1,:)),1) = zeros(length(gridE(:,1,1)),1);
        gridE(:,1,1) = zeros(length(gridE(:,1,1)),1);
        gridE(length(gridE(1,:,:)),:,2) = zeros(length(gridE(1,:,1)),1);
        gridE(1,:,1) = zeros(length(gridE(1,:,1)),1);
    end
    %{
    energy_before = nonrelative_kinectic_before + energy_of_electric_field_before + energy_of_magnetic_field_before;
    
    energy_after = nonrelative_kinectic + energy_of_electric_field + energy_of_magnetic_field;
    %}
    E_field_after = gridE;
    B_field_after = gridB_theta;
    