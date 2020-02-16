function [electron_position,electron_velocity,E_field_after,B_field_after,energy_total] = electricmagneticfield(electron_position_before,electron_velocity_before,E_field_before,B_field_before,delta,delta_t,r,z,phi,W,duandian)


sigma=10^(-15);
me=9.1096*10^(-31);
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
electron_position = electron_position_before;
electron_velocity = electron_velocity_before;
gridE = E_field_before;
gridB_theta = B_field_before;

neutralize = find(sqrt(electron_position(:,1).^2+electron_position(:,2).^2)>r & electron_position(:,3)>z & electron_position(:,3)<0);
electron_position(neutralize,:)=[];
electron_position(neutralize,:)=[];


electron_position_r = sqrt(electron_position(:,1).^2+electron_position(:,2).^2);
electron_velocity_r = sqrt(electron_velocity(:,1).^2+electron_velocity(:,2).^2);
sin_electron_theta = electron_position(:,2)./electron_position_r;
cos_electron_theta = electron_position(:,1)./electron_position_r;
electron_velocity_r = electron_velocity(:,1).*cos_electron_theta +  electron_velocity(:,2).*sin_electron_theta;
    J(:,1) = -W*e*electron_velocity_r;
    J(:,2) = -W*e*electron_velocity(:,3);
    gridj = zeros(length(gridr),length(gridz),2);
    gridv = zeros(length(gridr),length(gridz),2);
    if length(J)>0  
        numberz = length(gridr);
        a = zeros(length(J(:,1)),1);
        ar = round(electron_position_r/delta_r-0.5);
        az = round(electron_position(:,3)/delta_z);
        a = ar + az*numberz + 1;
        Jr = J(:,1);
        Jz = J(:,2);
        vr = electron_velocity_r;
        vz = electron_velocity(:,3);
        gridjr = zeros(length(gridr),length(gridz),1);
        gridjz = zeros(length(gridr),length(gridz),1);
        gridvr = zeros(length(gridr),length(gridz),1);
        gridvz = zeros(length(gridr),length(gridz),1);
        b = unique(a);
        jrr = zeros(length(b),1);
        jzz = zeros(length(b),1);
        vrr = zeros(length(b),1);
        vzz = zeros(length(b),1);
        parfor j=1:length(b)
            jrr(j) = sum(Jr(a==b(j)));
            jzz(j) = sum(Jz(a==b(j)));
            vrr(j) = sum(vr(a==b(j)));
            vzz(j) = sum(vz(a==b(j)));
        end
        gridjr(b) = jrr./(phi*((R(b)+delta_r).^2-R(b).^2)*delta_r);
        gridjz(b) = jzz./(phi*((R(b)+delta_r).^2-R(b).^2)*delta_z);
        gridvr(b) = vrr;
        gridvz(b) = vzz;
        gridj(:,:,1) = gridjr;
        gridj(:,:,2) = gridjz;
        gridv(:,:,1) = gridvr;
        gridv(:,:,2) = gridvz;
    end

    energy_of_electric_field = 0;
    energy_of_magnetic_field = 0;
    J_E = 0;
    for i = 1:(length(gridE(1,:,1))-1)
        energy_of_electric_field = energy_of_electric_field + sum(gridE(i,:,1).^2+gridE(i,:,2).^2) *(i^2-(i-1)^2)*phi*delta_z*delta_r^2;
        energy_of_magnetic_field = energy_of_magnetic_field + sum(gridB_theta(i,:).^2) *(i^2-(i-1)^2)*phi*delta_z*delta_r^2;
        J_E  = J_E + sum(gridE(i,:,1).*gridj(i,:,1)+gridE(i,:,2).*gridj(i,:,2)) *(i^2-(i-1)^2)*phi*delta_z*delta_r^2;
    end
    Ebefore = 1/2*epsilon*energy_of_electric_field
    gridEbefore = gridE;

    for repeat=1:2
        delta_t_repeat =delta_t/2;
        [~,gradient_Er_pz] = gradient(gridE(:,:,1),delta_r,delta_z);
        [gradient_Ez_pr,~] = gradient(gridE(:,:,2),delta_r,delta_z);
        curlE_theta = gradient_Er_pz - gradient_Ez_pr;

        gridB_theta(:,:) = gridB_theta(:,:)-curlE_theta*delta_t_repeat;

        [gradient_Btheta_pr,gradient_Btheta_pz] = gradient(gridB_theta,delta_r,delta_z);
        curlB_r = -gradient_Btheta_pz;
        curlB_z = 1./(R+delta_r).*gridB_theta + gradient_Btheta_pr ;

        
        %gridE(:,:,1) = exp(-delta_t_repeat/epsilon*sigma)*gridE(:,:,1) + 1/sigma*(1-exp(-delta_t_repeat/epsilon*sigma))*(curlB_r*mu-gridj(:,:,1));
        %gridE(:,:,2) = exp(-delta_t_repeat/epsilon*sigma)*gridE(:,:,2) + 1/sigma*(1-exp(-delta_t_repeat/epsilon*sigma))*(curlB_z*mu-gridj(:,:,2));
        %delta_t_repeat/epsilon
        gridE(:,:,1) = gridE(:,:,1) + delta_t_repeat/epsilon*(curlB_r*mu-gridj(:,:,1));
        gridE(:,:,2) = gridE(:,:,2) + delta_t_repeat/epsilon*(curlB_z*mu-gridj(:,:,2));
        
        if repeat == 1
            gridE_half = gridE;
            gridB_theta_half = gridB_theta;
        end
        
       
    end
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
    energy_of_magnetic_field = 1/2*mu*energy_of_magnetic_field;
    electron_number = length(electron_position(:,1));
    fprintf('deltaE\n')
    (energy_of_electric_field - Ebefore)
    fprintf('E*dE/dt')
    E_Ept*epsilon*delta_t
    fprintf('J_E')
    J_E*delta_t
    if electron_number > 0
        %{
        ax1 = zeros(electron_number,1);
        ay1 = zeros(electron_number,1);
        az1 = zeros(electron_number,1);
        %}
        ar = round(electron_position_r/delta_r-0.5);
        az = round(electron_position(:,3)/delta_z);
        a = ar + az*numberz + 1;
        gridE_r = gridE_half(:,:,1);
        gridE_z = gridE_half(:,:,2);
        
        gridE_r_before = gridEbefore(:,:,1);
        gridE_z_before = gridEbefore(:,:,2);
        electron_velocity_r;
        electron_velocity_z = electron_velocity(:,3);
        F_r = -e*gridE_r(a) - e*(-electron_velocity_z.*gridB_theta_half(a));
        F_z = -e*gridE_z(a) - e*(electron_velocity_r.*gridB_theta(a));
        %fprintf('F_r\n');

        
        F(:,1) = F_r.*cos_electron_theta;
        F(:,2) = F_r.*sin_electron_theta;
        

        
        F(:,3) = F_z;
        
        %{
        W*sum(sum(F.*electron_velocity))
        W*sum(F_r.*vr+F_z.*vz)
        J_E
        %}
        electron_velocity_before = electron_velocity;
        electron_position = electron_position + delta_t*electron_velocity;
        gamma = 1./sqrt( 1-(electron_velocity(:,1).^2+electron_velocity(:,2).^2+electron_velocity(:,3).^2)/c^2 );
        %E_nonrelative_before = W*1/2*me*sum(sum(electron_velocity.^2));
        particle_kinectic_before = W*sum((gamma-1).*me.*c^2);
        nonrelative_kinectic_before = W*1/2*me*sum(sum(electron_velocity.^2));
        fprintf('Fs');
        W*sum(sum(electron_velocity.*F))*delta_t
        
        electron_u(:,1) = gamma.*electron_velocity(:,1);
        electron_u(:,2) = gamma.*electron_velocity(:,2);
        electron_u(:,3) = gamma.*electron_velocity(:,3);
%{
        A = (F(:,1).^2+F(:,2).^2+F(:,3).^2)/me^2*delta_t^2;
        B = (electron_velocity(:,1).*F(:,1)+electron_velocity(:,2).*F(:,2)+electron_velocity(:,3).*F(:,3))/me*delta_t;
        C = A./B;
        factor = 1./(2*C).*(-1+sqrt(1+4*C));
        imag(sum(factor))
        a = find(imag(factor)~=0);
        C(a)
        A(a)
        B(a)
        factor(a)
        %}
        electron_u = electron_u + F/me*delta_t;


        gamma = sqrt(1+(electron_u(:,1).^2+electron_u(:,2).^2+electron_u(:,3).^2)/c^2);
        
        
        particle_kinectic = W*sum((gamma-1)*me*c^2);
        delta_kinectic = particle_kinectic-particle_kinectic_before
        
        
        electron_velocity(:,1) = electron_u(:,1)./gamma;
        electron_velocity(:,2) = electron_u(:,2)./gamma;
        electron_velocity(:,3) = electron_u(:,3)./gamma;
        nonrelative_kinectic = W*1/2*me*sum(sum(electron_velocity.^2));
        delta_nonrelative_kinectic = nonrelative_kinectic - nonrelative_kinectic_before;
        %E_nonrelative = W*1/2*me*sum(sum(electron_velocity.^2));
        %delta_kinectic_nonrelative = E_nonrelative - E_nonrelative_before

        
    electron_position_r = sqrt(electron_position(:,1).^2+electron_position(:,2).^2);
    a = find(electron_position_r>r);
    electron_position(a,:) = [];
    electron_velocity(a,:) = [];
    [electron_position,electron_velocity,N_boundary] = REBOND( electron_position,electron_velocity,delta_t,r,z,phi);
    end
    energy_total = energy_of_electric_field + energy_of_magnetic_field + particle_kinectic

    energy_of_electric_field
    energy_of_magnetic_field
    particle_kinectic
    if mod(duandian,500) == 1
        1;
    end
    
    E_field_after = gridE;
    B_field_after = gridB_theta;
    