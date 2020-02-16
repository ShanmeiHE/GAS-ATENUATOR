function [electron_position,electron_velocity,ion1_position,ion1_velocity,ion2_position,ion2_velocity,gridE_r,gridE_z,gridH_theta,energy_total,energy_of_electric_field,energy_of_magnetic_field,particle_kinetic] = electricmagneticfield2(electron_position,electron_velocity,ion1_position,ion1_velocity,ion2_position,ion2_velocity,gridE_r,gridE_z,gridH_theta,delta,delta_t,r,z,phi,W,duandian)


sigma=10^(-15);
m_e = 9.1096*10^(-31);
m_ion1 = 4.65E-26;
m_ion2 = 4.65E-26;
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



electron_position_half = electron_position + 1/2*delta_t*electron_velocity;
electron_position_r = sqrt(electron_position_half(:,1).^2+electron_position_half(:,2).^2);
a = find(electron_position_r>r);
electron_position_half(a,:) = [];
electron_velocity(a,:) = [];
[electron_position_half,electron_velocity,N_boundary] = REBOND( electron_position_half,electron_velocity,delta_t,r,z,phi);

ion1_position_half = ion1_position + 1/2*delta_t*ion1_velocity;
ion1_position_r = sqrt(ion1_position_half(:,1).^2+ion1_position_half(:,2).^2);
a = find(ion1_position_r>r);
ion1_position_half(a,:) = [];
ion1_velocity(a,:) = [];
[ion1_position_half,ion1_velocity,N_boundary] = REBOND( ion1_position_half,ion1_velocity,delta_t,r,z,phi);

ion2_position_half = ion2_position + 1/2*delta_t*ion2_velocity;
ion2_position_r = sqrt(ion2_position_half(:,1).^2+ion2_position_half(:,2).^2);
a = find(ion2_position_r>r);
ion2_position_half(a,:) = [];
ion2_velocity(a,:) = [];
[ion2_position_half,ion2_velocity,N_boundary] = REBOND( ion2_position_half,ion2_velocity,delta_t,r,z,phi);
    
    particle_velocity = [electron_velocity;ion1_velocity;ion2_velocity];
    particle_position_half = [electron_position_half;ion1_position_half;ion2_position_half];
    particle_position_r = sqrt(particle_position_half(:,1).^2+particle_position_half(:,2).^2);
    sin_particle_theta = particle_position_half(:,2)./particle_position_r;
    cos_particle_theta = particle_position_half(:,1)./particle_position_r;
    particle_velocity_r = particle_velocity(:,1).*cos_particle_theta +  particle_velocity(:,2).*sin_particle_theta;
    
    electron_number = length(electron_position_half(:,1));
    ion1_number = length(ion1_position_half(:,1));
    ion2_number = length(ion2_position_half(:,1));
    particle_charge = [-e*ones(electron_number,1);e*ones(ion1_number,1);2*e*ones(ion2_number,1)];
    particle_mass = [m_e*ones(electron_number,1);m_ion1*ones(ion1_number,1);m_ion2*ones(ion2_number,1)];


    if length(particle_velocity_r)>0
        
        %% jr
        numberz = length(E_r_gridr);
        %r:0.5~1.5 …… 1
        %z:0~1     …… 1
        
        ar = round(particle_position_r/delta_r);
        az = round(particle_position_half(:,3)/delta_z+0.5);
 
        hx = particle_position_r/delta_r - ar +1/2;
        hy = particle_position_half(:,3)/delta_z - az+1;
        
        Eromega1 = (1-hx).*(1-hy);
        Eromega2 = hx.*(1-hy);
        Eromega3 = (1-hx).*hy;
        Eromega4 = hx.*hy;
        
        ar1 = ar     + (az-1)*numberz;
        ar1(ar1<0.9) = 1;
        ar1(ar1>E_r_len+0.1) = E_r_len;
        ar2 = (ar+1) + (az-1)*numberz;
        ar2(ar2<0.9) = 1;
        ar2(ar2>E_r_len+0.1) = E_r_len;
        ar3 = ar     + az*numberz;
        ar3(ar3<0.9) = 1;
        ar3(ar3>E_r_len+0.1) = E_r_len;
        ar4 = (ar+1) + az*numberz;
        ar4(ar4<0.9) = 1;
        ar4(ar4>E_r_len+0.1) = E_r_len;
        
        ar0 = find(ar<0.9);
        arn = find(ar>E_r_lenr-0.1);
        
        Eromega1(ar0) = 0;
        Eromega3(ar0) = 0;
        Eromega2(ar0) = (1-hy(ar0));
        Eromega4(ar0) = hy(ar0);

        Eromega1(arn) = 1-hy(arn);
        Eromega3(arn) = hy(arn);
        Eromega2(arn) = 0;
        Eromega4(arn) = 0;
        
        
        charge_vr = particle_charge.*particle_velocity_r;     
        a = [ar1;ar2;ar3;ar4];
        a(a<0) = 1;
        a(a>E_r_len) = E_z_len;
        Eromega = [Eromega1;Eromega2;Eromega3;Eromega4];
        
        nod_e = ind2vec(a');
        
        jr = nod_e * (Eromega.*[charge_vr;charge_vr;charge_vr;charge_vr]);
        jr = sum(jr,2);
        gridj_r = zeros(E_r_lenr,E_r_lenz);
        gridj_r(1:length(jr)) = jr;
        gridj_r = W*gridj_r./(phi*((E_r_R+delta_r/2).^2-(E_r_R-delta_r/2).^2)*delta_z);
        
        %% jz
        numberz = length(E_z_gridr);
        %r:0~1     …… 1
        %z:0.5~1.5 …… 1
        ar = round(particle_position_r/delta_r+0.5);
        az = round(particle_position_half(:,3)/delta_z);
 
        hx = particle_position_r/delta_r - ar+1;
        hy = particle_position_half(:,3)/delta_z - az +1/2;
        
        Ezomega1 = (1-hx).*(1-hy);
        Ezomega2 = hx.*(1-hy);
        Ezomega3 = (1-hx).*hy;
        Ezomega4 = hx.*hy;
        
        az1 = ar     + (az-1)*numberz;
        az1(az1<0.9) = 1;
        az1(az1>E_z_len+0.1) = E_z_len;
        az2 = (ar+1) + (az-1)*numberz;
        az2(az2<0.9) = 1;
        az2(az2>E_z_len+0.1) = E_z_len;
        az3 = ar     + az*numberz;
        az3(az3<0.9) = 1;
        az3(az3>E_z_len+0.1) = E_z_len;
        az4 = (ar+1) + az*numberz;
        az4(az4<0.9) = 1;
        az4(az4>E_z_len+0.1) = E_z_len;
        
        az0 = find(az<0.9);
        azn = find(az>E_z_lenz-0.1);
        
        Ezomega1(az0) = 0;
        Ezomega3(az0) = 1-hx(az0);
        Ezomega2(az0) = 0;
        Ezomega4(az0) = hx(az0);
        
        Ezomega1(azn) = 1-hx(azn);
        Ezomega3(azn) = 0;
        Ezomega2(azn) = hx(azn);
        Ezomega4(azn) = 0;
        
        charge_vz = particle_charge.*particle_velocity(:,3);
        a = [az1;az2;az3;az4];
        Ezomega = [Ezomega1;Ezomega2;Ezomega3;Ezomega4];
        
        nod_e = ind2vec(a');
        jz = nod_e * (Ezomega.*[charge_vz;charge_vz;charge_vz;charge_vz]);
        jz = sum(jz,2);
        gridj_z = zeros(E_z_lenr,E_z_lenz);
        gridj_z(1:length(jz)) = jz;
        gridj_z(1,:) = W*gridj_z(1,:)./(phi*(E_z_R(1,:)+delta_r/2).^2*delta_z);
        gridj_z(2:(E_z_lenr-1),:) = W*gridj_z(2:(E_z_lenr-1),:)./(phi*((E_z_R(2:(E_z_lenr-1),:)+delta_r/2).^2-(E_z_R(2:(E_z_lenr-1),:)-delta_r/2).^2)*delta_z);
        gridj_z(E_z_lenr,:) = W*gridj_z(E_z_lenr,:)./(phi*(E_z_R(E_z_lenr,:).^2-(E_z_R(E_z_lenr,:)-delta_r/2).^2)*delta_z);

    end
    repeat = 2;
    delta_t_repeat = delta_t/repeat;
    
    energy_of_electric_field_before = 1/2*epsilon*(trapz(E_r_gridz,trapz(E_r_gridr,gridE_r.^2.*E_r_R))+trapz(E_z_gridz,trapz(E_z_gridr,gridE_z.^2.*E_z_R)));
    energy_of_magnetic_field_before = 1/2*mu*trapz(H_theta_gridz,trapz(H_theta_gridr,gridH_theta.^2.*H_theta_R));
    
    
	gridH_theta = gridH_theta - delta_t_repeat/2/(mu*delta_r)*( gridE_z(2:E_z_lenr,:)-gridE_z(1:(E_z_lenr-1),:) ) + delta_t_repeat/2/(mu*delta_z)*( gridE_r(:,2:E_r_lenz)-gridE_r(:,1:(E_r_lenz-1)) );
    for timee = 1:repeat       
        gridH_theta = gridH_theta + delta_t_repeat/(mu*delta_r)*( gridE_z(2:E_z_lenr,:)-gridE_z(1:(E_z_lenr-1),:) ) - delta_t_repeat/(mu*delta_z)*( gridE_r(:,2:E_r_lenz)-gridE_r(:,1:(E_r_lenz-1)) );
        %sum(sum(delta_t_repeat/(mu*delta_r)*( gridE_z(2:E_z_lenr,:)-gridE_z(1:(E_z_lenr-1),:) ) - delta_t_repeat/(mu*delta_z)*( gridE_r(:,2:E_r_lenz)-gridE_r(:,1:(E_r_lenz-1)) )))
        gridE_r(:,2:(E_r_lenz-1)) = gridE_r(:,2:(E_r_lenz-1)) - delta_t_repeat/epsilon*gridj_r(:,2:(E_r_lenz-1)) - delta_t_repeat/(epsilon*delta_z)*( gridH_theta(:,2:H_theta_lenz)-gridH_theta(:,1:(H_theta_lenz-1)) );
        %sum(sum(delta_t_repeat/(epsilon*delta_z)*( gridH_theta(:,2:H_theta_lenz)-gridH_theta(:,1:(H_theta_lenz-1)) )))
        gridE_z(2:(E_z_lenr-1),:) = gridE_z(2:(E_z_lenr-1),:) - delta_t_repeat/epsilon*gridj_z(2:(E_z_lenr-1),:) + delta_t_repeat/epsilon*(1./(2*E_z_R(2:(E_z_lenr-1),:))+1/delta_r).*gridH_theta(2:H_theta_lenr,:) + delta_t_repeat/epsilon*(1./(2*E_z_R(2:(E_z_lenr-1),:))-1/delta_r).*gridH_theta(1:(H_theta_lenr-1),:) ;
        %sum(sum(delta_t_repeat/epsilon*(1./(2*E_z_R(2:(E_z_lenr-1),:)+1/delta_r)).*gridH_theta(2:H_theta_lenr,:) + delta_t_repeat/epsilon*(1./(2*E_z_R(2:(E_z_lenr-1),:)-1/delta_r)).*gridH_theta(1:(H_theta_lenr-1),:) ))
        gridE_z(1,:) = gridE_z(1,:) + 4*delta_t_repeat/(epsilon*delta_r)*gridH_theta(1,:);        
    end
    gridH_theta = gridH_theta + delta_t_repeat/2/(mu*delta_r)*( gridE_z(2:E_z_lenr,:)-gridE_z(1:(E_z_lenr-1),:) ) - delta_t_repeat/2/(mu*delta_z)*( gridE_r(:,2:E_r_lenz)-gridE_r(:,1:(E_r_lenz-1)) );
    
    energy_of_electric_field = 1/2*epsilon*(trapz(E_r_gridz,trapz(E_r_gridr,gridE_r.^2.*E_r_R))+trapz(E_z_gridz,trapz(E_z_gridr,gridE_z.^2.*E_z_R)));
    energy_of_magnetic_field = 1/2*mu*trapz(H_theta_gridz,trapz(H_theta_gridr,gridH_theta.^2.*H_theta_R));
    J_E = delta_t*(trapz(E_r_gridz,trapz(E_r_gridr,gridE_r.*gridj_r.*E_r_R))+trapz(E_z_gridz,trapz(E_z_gridr,gridE_z.*gridj_z.*E_z_R)));
    RHO = trapz(E_r_gridr,gridE_r);
    
    
    delta_E_energy = energy_of_electric_field - energy_of_electric_field_before;
    delta_B_energy = energy_of_magnetic_field - energy_of_magnetic_field_before;
    
    particle_number = length(particle_position_half(:,1));
    if particle_number > 0
        %% H_theta 
        numberz = length(H_theta_gridr);
        %r:0.5~1.5 …… 1
        %z:0.5~1.5 …… 1
        ar = round(particle_position_r/delta_r);
        az = round(particle_position_half(:,3)/delta_z);
 
        hx = particle_position_r/delta_r - ar+1/2;
        hy = particle_position_half(:,3)/delta_z - az +1/2;
        
        Homega1 = (1-hx).*(1-hy);
        Homega2 = hx.*(1-hy);
        Homega3 = (1-hx).*hy;
        Homega4 = hx.*hy;
        
        ah1 = ar     + (az-1)*numberz;
        ah1(ah1<0.9) = 1;
        ah1(ah1>H_theta_len+0.1) = H_theta_len;
        ah2 = (ar+1) + (az-1)*numberz;
        ah2(ah2<0.9) = 1;
        ah2(ah2>H_theta_len+0.1) = H_theta_len;
        ah3 = ar     + az*numberz;
        ah3(ah3<0.9) = 1;
        ah3(ah3>H_theta_len+0.1) = H_theta_len;
        ah4 = (ar+1) + az*numberz;
        ah4(ah4<0.9) = 1;
        ah4(ah4>H_theta_len+0.1) = H_theta_len;
        
        ah0 = find(ar<0.9);
        ahn = find(ar>E_z_lenz-0.1);

        Homega1(ah0) = 0;
        Homega3(ah0) = 0;
        Homega2(ah0) = (1-hy(ah0));
        Homega4(ah0) = hy(ah0);

        Homega1(ahn) = (1-hy(ahn));
        Homega3(ahn) = hy(ahn);
        Homega2(ahn) = 0;
        Homega4(ahn) = 0;

        
        ah0 = find(az<0.9);
        ahn = find(az>E_r_lenr-0.1);
        
        Homega1(ah0) = 0;
        Homega3(ah0) = 1-hx(ah0);
        Homega2(ah0) = 0;
        Homega4(ah0) = hx(ah0);

        Homega1(ahn) = 1-hx(ahn);
        Homega3(ahn) = 0;
        Homega2(ahn) = hx(ahn);
        Homega4(ahn) = 0;
        
        
        particle_velocity_r;
        particle_velocity_z = particle_velocity(:,3);
        Er = Eromega1.*gridE_r(ar1) + Eromega2.*gridE_r(ar2) + Eromega3.*gridE_r(ar3) + Eromega4.*gridE_r(ar4);
        Ez = Ezomega1.*gridE_z(az1) + Ezomega2.*gridE_r(az2) + Ezomega3.*gridE_r(az3) + Ezomega4.*gridE_r(az4);
        Btheta = (Homega1.*gridH_theta(ah1)+Homega2.*gridH_theta(ah2)+Homega3.*gridH_theta(ah3)+Homega4.*gridH_theta(ah4))*mu;
        F_r = particle_charge.*(Er + (-particle_velocity_z.*Btheta));
        F_z = particle_charge.*(Ez + (particle_velocity_r.*Btheta));
        %fprintf('F_r\n');
        %mean(mean(F_r))

        
        F(:,1) = F_r.*cos_particle_theta;
        F(:,2) = F_r.*sin_particle_theta;
       
        F(:,3) = F_z;
        
        particle_velocity_before = particle_velocity;
        
        gamma = 1./sqrt( 1-(particle_velocity(:,1).^2+particle_velocity(:,2).^2+particle_velocity(:,3).^2)/c^2 );
        
        particle_kinetic_before = W*sum((gamma-1).*particle_mass*c^2);
        
        particle_u(:,1) = gamma.*particle_velocity(:,1);
        particle_u(:,2) = gamma.*particle_velocity(:,2);
        particle_u(:,3) = gamma.*particle_velocity(:,3);

        particle_u = particle_u + F./particle_mass *delta_t;


        gamma = sqrt(1+(particle_u(:,1).^2+particle_u(:,2).^2+particle_u(:,3).^2)/c^2);
        
        
        particle_kinetic = W*sum((gamma-1).*particle_mass*c^2);
        delta_kinectic = particle_kinetic-particle_kinetic_before;
        energy_total = energy_of_electric_field + energy_of_magnetic_field + particle_kinetic;
                
        
        particle_velocity(:,1) = particle_u(:,1)./gamma;
        particle_velocity(:,2) = particle_u(:,2)./gamma;
        particle_velocity(:,3) = particle_u(:,3)./gamma;

        particle_position = particle_position_half + 1/2*delta_t*particle_velocity;
        
        electron_position = particle_position(1:electron_number,:);
        electron_velocity = particle_velocity(1:electron_number,:);
        ion1_position = particle_position((electron_number+1):(electron_number+ion1_number),:);
        ion1_velocity = particle_velocity((electron_number+1):(electron_number+ion1_number),:);
        ion2_position = particle_position((electron_number+ion1_number+1):(electron_number+ion1_number+ion2_number),:);
        ion2_velocity = particle_velocity((electron_number+ion1_number+1):(electron_number+ion1_number+ion2_number),:);

                
        electron_position_r = sqrt(electron_position(:,1).^2+electron_position(:,2).^2);
        a = find(electron_position_r>r);
        electron_position(a,:) = [];
        electron_velocity(a,:) = [];
        [electron_position,electron_velocity,N_boundary] = REBOND( electron_position,electron_velocity,delta_t,r,z,phi);
        
        ion1_position_r = sqrt(ion1_position(:,1).^2+ion1_position(:,2).^2);
        a = find(ion1_position_r>r);
        ion1_position(a,:) = [];
        ion1_velocity(a,:) = [];
        [ion1_position,ion1_velocity,N_boundary] = REBOND( ion1_position,ion1_velocity,delta_t,r,z,phi);

        ion2_position_r = sqrt(ion2_position(:,1).^2+ion2_position(:,2).^2);
        a = find(ion2_position_r>r);
        ion2_position(a,:) = [];
        ion2_velocity(a,:) = [];
        [ion2_position,ion2_velocity,N_boundary] = REBOND( ion2_position,ion2_velocity,delta_t,r,z,phi);
    end
    
    if mod(duandian,10000) == 1
        1;
    end
    
