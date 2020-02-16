function [electron_position,electron_velocity,ion1_position,ion1_velocity,ion2_position,ion2_velocity,gridE_r,gridE_z,gridH_theta,gridE_r_PML0,gridE_z_PML0,gridD_r_PML0,gridD_z_PML0,gridH_theta_PML0,gridE_r_PMLn,gridE_z_PMLn,gridD_r_PMLn,gridD_z_PMLn,gridH_theta_PMLn,energy_total,energy_of_electric_field,energy_of_magnetic_field,particle_kinetic] = electricmagneticfield3(electron_position,electron_velocity,ion1_position,ion1_velocity,ion2_position,ion2_velocity,gridE_r,gridE_z,gridH_theta,gridE_r_PML0,gridE_z_PML0,gridD_r_PML0,gridD_z_PML0,gridH_theta_PML0,gridE_r_PMLn,gridE_z_PMLn,gridD_r_PMLn,gridD_z_PMLn,gridH_theta_PMLn,delta,delta_t,r,z,phi,W,duandian)


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
H_theta_gridz = delta_z/2:delta_z:z-delta_z/2;

H_theta_lenr = length(H_theta_gridr);
H_theta_lenz = length(H_theta_gridz);
H_theta_len = length(H_theta_gridr)*length(H_theta_gridz);

[H_theta_R,H_theta_Z] = meshgrid(H_theta_gridr,H_theta_gridz);
H_theta_R = H_theta_R';
H_theta_Z = H_theta_Z';



E_r_gridr = delta_r/2:delta_r:r-delta_r/2;
E_r_gridz = 0:delta_z:z;

E_r_lenr = length(E_r_gridr);
E_r_lenz = length(E_r_gridz);
E_r_len = length(E_r_gridr)*length(E_r_gridz);

[E_r_R,E_r_Z] = meshgrid(E_r_gridr,E_r_gridz);
E_r_R = E_r_R';
E_r_Z = E_r_Z';



E_z_gridr = 0:delta_r:r;
E_z_gridz = delta_z/2:delta_z:z-delta_z/2;

E_z_lenr = length(E_z_gridr);
E_z_lenz = length(E_z_gridz);
E_z_len = length(E_z_gridr)*length(E_z_gridz);

[E_z_R,E_z_Z] = meshgrid(E_z_gridr,E_z_gridz);
E_z_R = E_z_R';
E_z_Z = E_z_Z';


PML_length = delta_z*10;
hole_radius = delta_r*50;
%% PML z<0
H_theta_gridr_PML0 = delta_r/2:delta_r:hole_radius+delta_r/2;
H_theta_gridz_PML0 = delta_z/2-PML_length:delta_z:-delta_z/2;

H_theta_lenr_PML0 = length(H_theta_gridr_PML0);
H_theta_lenz_PML0 = length(H_theta_gridz_PML0);




E_r_gridr_PML0 = delta_r/2:delta_r:hole_radius+delta_r/2;
E_r_gridz_PML0 = -PML_length:delta_z:-delta_z;

E_r_lenr_PML0 = length(E_r_gridr_PML0);
E_r_lenz_PML0 = length(E_r_gridz_PML0);





E_z_gridr_PML0 = 0:delta_r:hole_radius;
E_z_gridz_PML0 = delta_z/2-PML_length:delta_z:-delta_z/2;

E_z_lenr_PML0 = length(E_z_gridr_PML0);
E_z_lenz_PML0 = length(E_z_gridz_PML0);
%% PML z>n
H_theta_gridr_PMLn = delta_r/2:delta_r:hole_radius+delta_r/2;
H_theta_gridz_PMLn = z+delta_z/2:delta_z:z-delta_z/2+PML_length;

H_theta_lenr_PMLn = length(H_theta_gridr_PMLn);
H_theta_lenz_PMLn = length(H_theta_gridz_PMLn);




E_r_gridr_PMLn = delta_r/2:delta_r:hole_radius+delta_r/2;
E_r_gridz_PMLn = z+delta_z:delta_z:z+PML_length;

E_r_lenr_PMLn = length(E_r_gridr_PMLn);
E_r_lenz_PMLn = length(E_r_gridz_PMLn);





E_z_gridr_PMLn = 0:delta_r:hole_radius;
E_z_gridz_PMLn = z+delta_z/2:delta_z:z-delta_z/2+PML_length;

E_z_lenr_PMLn = length(E_z_gridr_PMLn);
E_z_lenz_PMLn = length(E_z_gridz_PMLn);






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
        
        numberz = length(H_theta_gridr);
        %r:0~1 …… 1
        %z:0~1 …… 1
        
        ar = round(particle_position_r/delta_r+0.5);
        az = round(particle_position_half(:,3)/delta_z+0.5);
        a = (ar-1) + (az-1)*numberz + 1;
        nod_e = ind2vec(a');
        
        jr = nod_e * (particle_charge.*particle_velocity_r);
        jr = sum(jr,2);
        gridjr = zeros(H_theta_lenr,H_theta_lenz);
        gridjr(1:length(jr)) = jr;
        gridjr = W*gridjr./(phi*((H_theta_R+delta_r/2).^2-(H_theta_R-delta_r/2).^2)*delta_z);
        
        jz = nod_e * (particle_charge.*particle_velocity(:,3));
        jz = sum(jz,2);
        gridjz = zeros(H_theta_lenr,H_theta_lenz);
        gridjz(1:length(jz)) = jz;
        gridjz = W*gridjz./(phi*((H_theta_R+delta_r/2).^2-(H_theta_R-delta_r/2).^2)*delta_z);
        
        
    end
    
    repeat = 2;
    delta_t_repeat = delta_t/repeat;
    
    energy_of_electric_field_before = 1/2*epsilon*(trapz(E_r_gridz,trapz(E_r_gridr,gridE_r.^2.*E_r_R))+trapz(E_z_gridz,trapz(E_z_gridr,gridE_z.^2.*E_z_R)));
    energy_of_magnetic_field_before = 1/2*mu*trapz(H_theta_gridz,trapz(H_theta_gridr,gridH_theta.^2.*H_theta_R));
    
    
	gridH_theta = gridH_theta - delta_t_repeat/2/(mu*delta_r)*( gridE_z(2:E_z_lenr,:)-gridE_z(1:(E_z_lenr-1),:) ) + delta_t_repeat/2/(mu*delta_z)*( gridE_r(:,2:E_r_lenz)-gridE_r(:,1:(E_r_lenz-1)) );
    for timee = 1:repeat       
        gridE_r1 = gridE_r(:,2);
        gridE_rend = gridE_r(:,E_r_lenz-1);
        gridH_theta = gridH_theta + delta_t_repeat/(mu*delta_r)*( gridE_z(2:E_z_lenr,:)-gridE_z(1:(E_z_lenr-1),:) ) - delta_t_repeat/(mu*delta_z)*( gridE_r(:,2:E_r_lenz)-gridE_r(:,1:(E_r_lenz-1)) );
        %sum(sum(delta_t_repeat/(mu*delta_r)*( gridE_z(2:E_z_lenr,:)-gridE_z(1:(E_z_lenr-1),:) ) - delta_t_repeat/(mu*delta_z)*( gridE_r(:,2:E_r_lenz)-gridE_r(:,1:(E_r_lenz-1)) )))
        gridE_r(:,2:(E_r_lenz-1)) = gridE_r(:,2:(E_r_lenz-1)) - delta_t_repeat/epsilon*(gridjr(:,1:(H_theta_lenz-1))+gridjr(:,2:H_theta_lenz))/2 - delta_t_repeat/(epsilon*delta_z)*( gridH_theta(:,2:H_theta_lenz)-gridH_theta(:,1:(H_theta_lenz-1)) );
        %sum(sum(delta_t_repeat/(epsilon*delta_z)*( gridH_theta(:,2:H_theta_lenz)-gridH_theta(:,1:(H_theta_lenz-1)) )))
        gridE_z(2:(E_z_lenr-1),:) = gridE_z(2:(E_z_lenr-1),:) - delta_t_repeat/epsilon*(gridjz(1:(H_theta_lenr-1),:)+gridjz(2:H_theta_lenr,:))/2 + delta_t_repeat/epsilon*(1./(2*E_z_R(2:(E_z_lenr-1),:))+1/delta_r).*gridH_theta(2:H_theta_lenr,:) + delta_t_repeat/epsilon*(1./(2*E_z_R(2:(E_z_lenr-1),:))-1/delta_r).*gridH_theta(1:(H_theta_lenr-1),:) ;
        %sum(sum(delta_t_repeat/epsilon*(1./(2*E_z_R(2:(E_z_lenr-1),:)+1/delta_r)).*gridH_theta(2:H_theta_lenr,:) + delta_t_repeat/epsilon*(1./(2*E_z_R(2:(E_z_lenr-1),:)-1/delta_r)).*gridH_theta(1:(H_theta_lenr-1),:) ))
        gridE_z(1,:) = gridE_z(1,:) + delta_t_repeat/epsilon*gridjz(1,:) + 4*delta_t_repeat/(epsilon*delta_r)*gridH_theta(1,:);
        gridE_r(:,1) = gridE_r1;
        gridE_r(:,E_r_lenz) = gridE_rend;
        %{
        gridD_r_PML0_after(:,1:(E_r_lenz_PML0-1)) =  (2*epsilon-sigma_z*delta_t_repeat)./(2*epsilon+sigma_z*delta_t_repeat).*gridD_z_PML0(:,1:(E_r_lenz_PML0-1)) - (2*epsilon*delta_t_repeat)./(2*epsilon+sigma_z*delta_t_repeat*dz).*(gridH_theta_PML0(:,2:H_theta_lenz_PML0)-gridH_theta_PML0(:,1:(H_theta_lenz_PML0-1)));
        gridD_r_PMLn_after(:,2:E_r_lenz_PMLn) =  (2*epsilon-sigma_z*delta_t_repeat)./(2*epsilon+sigma_z*delta_t_repeat).*gridD_z_PMLn(:,1:(E_r_lenz_PML0-1)) - (2*epsilon*delta_t_repeat)./(2*epsilon+sigma_z*delta_t_repeat*dz).*(gridH_theta_PMLn(:,2:E_r_lenz_PMLn)-gridH_theta_PMLn(:,1:(E_r_lenz_PMLn-1)));
        gridE_r_PML0(:,1:(E_r_lenz_PML0-1)) = gridE_r_PML0(:,1:(E_r_lenz_PML0-1)) + r./(epsilon*overline_r).*( (2*epsilon+sigma_r*delta_t_repeat)/(2*epsilon).*gridD_r_PML0_after(:,1:(E_r_lenz_PML0-1)) - (2*epsilon-sigma_r*delta_t_repeat)/(2*epsilon).*gridD_r_PML0(:,1:(E_r_lenz_PML0-1)) );
        gridE_r_PMLn(:,2:E_r_lenz_PMLn) = gridE_r_PMLn(:,2:E_r_lenz_PMLn) + r./(epsilon*overline_r).*( (2*epsilon+sigma_r*delta_t_repeat)/(2*epsilon).*gridD_r_PMLn_after(:,2:E_r_lenz_PMLn) - (2*epsilon-sigma_r*delta_t_repeat)/(2*epsilon).*gridD_r_PMLn(:,2:E_r_lenz_PMLn) );
        
        gridD_z_PML0_after(2:(E_z_lenr_PML0-1),:) = (2*epsilon-sigma_r*delta_t_repeat)./(2*epsilon+sigma_r*dt).*gridD_z_PML0(2:(E_z_lenr_PML0-1),:) + (2*epsilon*delta_t_repeat)./(2*epsilon+sigma_r*delta_t_repeat).*( (1./(2*E_z_R(2:(E_z_lenr_PML0-1),:))+1/delta_r).*gridH_theta_PML0(2:H_theta_lenr_PML0,:) + (1./(2*E_z_R(2:(E_z_lenr_PML0-1),:))-1/delta_r).*gridH_theta_PML0(1:(H_theta_lenr_PML0-1),:) );
        gridD_z_PMLn_after(2:(E_z_lenr_PML0-1),:) = (2*epsilon-sigma_r*delta_t_repeat)./(2*epsilon+sigma_r*dt).*gridD_z_PMLn(2:(E_z_lenr_PMLn-1),:) + (2*epsilon*delta_t_repeat)./(2*epsilon+sigma_r*delta_t_repeat).*( (1./(2*E_z_R(2:(E_z_lenr_PMLn-1),:))+1/delta_r).*gridH_theta_PMLn(2:H_theta_lenr_PMLn,:) + (1./(2*E_z_R(2:(E_z_lenr_PMLn-1),:))-1/delta_r).*gridH_theta_PMLn(1:(H_theta_lenr_PMLn-1),:) );
        gridE_z_PML0(2:(E_z_lenr_PML0-1),:) = gridE_z_PML0(2:(E_z_lenr_PML0-1),:) + r./(epsilon*overline_r).*( (2*epsilon + sigma_z*sigma_r*delta_t_repeat)/(2*epsilon).*gridD_z_PML0_after(2:(E_z_lenr_PML0-1),:) - (2*epsilon - sigma_z*sigma_r*delta_t_repeat)/(2*epsilon).*gridD_z_PML0(2:(E_z_lenr_PML0-1),:) );
        gridE_z_PMLn(2:(E_z_lenr_PMLn-1),:) = gridE_z_PMLn(2:(E_z_lenr_PMLn-1),:) + r./(epsilon*overline_r).*( (2*epsilon + sigma_z*sigma_r*delta_t_repeat)/(2*epsilon).*gridD_z_PMLn_after(2:(E_z_lenr_PMLn-1),:) - (2*epsilon - sigma_z*sigma_r*delta_t_repeat)/(2*epsilon).*gridD_z_PMLn(2:(E_z_lenr_PMLn-1),:) );
        %}
        
    end
    gridH_theta = gridH_theta + delta_t_repeat/2/(mu*delta_r)*( gridE_z(2:E_z_lenr,:)-gridE_z(1:(E_z_lenr-1),:) ) - delta_t_repeat/2/(mu*delta_z)*( gridE_r(:,2:E_r_lenz)-gridE_r(:,1:(E_r_lenz-1)) );
    
    energy_of_electric_field = 1/2*epsilon*(trapz(E_r_gridz,trapz(E_r_gridr,gridE_r.^2.*E_r_R))+trapz(E_z_gridz,trapz(E_z_gridr,gridE_z.^2.*E_z_R)));
    energy_of_magnetic_field = 1/2*mu*trapz(H_theta_gridz,trapz(H_theta_gridr,gridH_theta.^2.*H_theta_R));
    %J_E = delta_t*(trapz(E_r_gridz(2:(E_r_lenz-1)),trapz(E_r_gridr,gridE_r(:,2:(E_r_lenz-1)).*(gridj(:,1:(H_theta_lenz-1))+gridj(:,2:H_theta_lenz))/2.*E_r_R(:,2:(E_r_lenz-1))))+trapz(E_z_gridz,trapz(E_z_gridr(1:(E_z_lenr-1)),gridE_z(1:(E_z_lenr-1),:).*(gridj(1:(H_theta_lenr-1),:)+gridj(2:H_theta_lenr,:))/2.*E_z_R(1:(E_z_lenr-1),:))));
    %RHO = trapz(E_r_gridr,gridE_r);
    
    
    delta_E_energy = energy_of_electric_field - energy_of_electric_field_before;
    delta_B_energy = energy_of_magnetic_field - energy_of_magnetic_field_before;
    
    particle_number = length(particle_position_half(:,1));
    if particle_number > 0
        
        %particle_velocity_r;
        particle_velocity_z = particle_velocity(:,3);
        
        E_r_gridr_interp = [0,E_r_gridr,r];
        E_r_gridz_interp = E_r_gridz;
        E_r_lenr_interp = length(E_r_gridr_interp);
        E_r_lenz_interp = length(E_r_gridz_interp);
        gridE_r_interp = zeros(E_r_lenr_interp,E_r_lenz_interp);
        gridE_r_interp(2:(E_r_lenr_interp-1),:) = gridE_r;
        gridE_r_interp(1,:) = gridE_r(1,:);
        Er=interp2(E_r_gridz_interp,E_r_gridr_interp,gridE_r_interp,particle_position_half(:,3),particle_position_r);
        
        E_z_gridr_interp = E_z_gridr;
        E_z_gridz_interp = [0,E_z_gridz,z];
        E_z_lenr_interp = length(E_z_gridr_interp);
        E_z_lenz_interp = length(E_z_gridz_interp);
        gridE_z_interp = zeros(E_z_lenr_interp,E_z_lenz_interp);
        gridE_z_interp(:,2:(E_z_lenz_interp-1)) = gridE_z;
        Ez=interp2(E_z_gridz_interp,E_z_gridr_interp,gridE_z_interp,particle_position_half(:,3),particle_position_r);
        
        H_theta_gridr_interp = [0,H_theta_gridr,r];
        H_theta_gridz_interp = [0,H_theta_gridz,z];
        H_theta_lenr_interp = length(H_theta_gridr_interp);
        H_theta_lenz_interp = length(H_theta_gridz_interp);
        gridH_theta_interp = zeros(H_theta_lenr_interp,H_theta_lenz_interp);
        gridH_theta_interp(2:(H_theta_lenr_interp-1),2:(H_theta_lenz_interp-1)) = gridH_theta;
        gridH_theta_interp(1,2:(H_theta_lenz_interp-1)) = gridH_theta(1,:);
        gridH_theta_interp(H_theta_lenr_interp,2:(H_theta_lenz_interp-1)) = gridH_theta(H_theta_lenr,:);
        gridH_theta_interp(2:(H_theta_lenr_interp-1),1) = gridH_theta(:,1);
        gridH_theta_interp(2:(H_theta_lenr_interp-1),H_theta_lenz_interp) = gridH_theta(:,H_theta_lenz);
        gridH_theta_interp(1,1) = gridH_theta(1,1);
        gridH_theta_interp(1,H_theta_lenz_interp) = gridH_theta(1,H_theta_lenz);
        gridH_theta_interp(H_theta_lenr_interp,1) = gridH_theta(H_theta_lenr,1);
        gridH_theta_interp(H_theta_lenr_interp,H_theta_lenz_interp) = gridH_theta(H_theta_lenr,H_theta_lenz);
        Htheta=interp2(H_theta_gridz_interp,H_theta_gridr_interp,gridH_theta_interp,particle_position_half(:,3),particle_position_r);
        

        
        F_r = particle_charge.*(Er + (-particle_velocity_z.*Htheta*mu));
        F_z = particle_charge.*(Ez + (particle_velocity_r.*Htheta*mu));
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
    
