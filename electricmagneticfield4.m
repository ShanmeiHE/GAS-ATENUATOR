function [electron_position,electron_velocity,ion1_position,ion1_velocity,ion2_position,ion2_velocity,gridE_r,gridE_z,gridH_phi,energy_total,energy_of_electric_field,energy_of_magnetic_field,particle_kinetic] = electricmagneticfield4(electron_position0,electron_velocity,ion1_position0,ion1_velocity,ion2_position0,ion2_velocity,gridE_r,gridE_z,gridH_phi,delta,dt,Rp,Lp,PHIp,FN,duandian,cycle_time_for_electromagneticfield)


sigma= 10^(-15);
m_e = 9.1096*10^(-31);
m_ion1 = 4.65E-26;
m_ion2 = 4.65E-26;
c = 3*10^8;
epsilon = 8.854187817*10^(-12);
mu = 4*pi*10^(-7);
e = 1.6*10^(-19);
delta_r = delta;
delta_z = delta;


j_gridr = gpuArray.linspace(0,Rp,Rp/delta_r+1);
j_gridz = gpuArray.linspace(0,Lp,Lp/delta_z+1);

[j_R,j_Z] = meshgrid(j_gridr,j_gridz);
j_R = j_R';
j_Z = j_Z';



H_phi_gridr = gpuArray.linspace(delta_r/2,Rp-delta_r/2,Rp/delta_r);
H_phi_gridz = gpuArray.linspace(delta_z/2,Lp-delta_z/2,Lp/delta_z);

H_phi_lenr = length(H_phi_gridr);
H_phi_lenz = length(H_phi_gridz);
H_phi_len = length(H_phi_gridr)*length(H_phi_gridz);

[H_phi_R,H_phi_Z] = meshgrid(H_phi_gridr,H_phi_gridz);
H_phi_R = H_phi_R';
H_phi_Z = H_phi_Z';



E_r_gridr = gpuArray.linspace(delta_r/2,Rp-delta_r/2,Rp/delta_r);
E_r_gridz = gpuArray.linspace(0,Lp,Lp/delta_z+1);

E_r_lenr = length(E_r_gridr);
E_r_lenz = length(E_r_gridz);
E_r_len = length(E_r_gridr)*length(E_r_gridz);

[E_r_R,E_r_Z] = meshgrid(E_r_gridr,E_r_gridz);
E_r_R = E_r_R';
E_r_Z = E_r_Z';



E_z_gridr = gpuArray.linspace(0,Rp,Rp/delta_r+1);
E_z_gridz = gpuArray.linspace(delta_z/2,Lp-delta_z/2,Lp/delta_z);


E_z_lenr = length(E_z_gridr);
E_z_lenz = length(E_z_gridz);
E_z_len = length(E_z_gridr)*length(E_z_gridz);

[E_z_R,E_z_Z] = meshgrid(E_z_gridr,E_z_gridz);
E_z_R = E_z_R';
E_z_Z = E_z_Z';



electron_number = length(electron_position0(:,1));
ion1_number = length(ion1_position0(:,1));
ion2_number = length(ion2_position0(:,1));
particle_charge = [-e*ones(electron_number,1);e*ones(ion1_number,1);2*e*ones(ion2_number,1)];
particle_mass = [m_e*ones(electron_number,1);m_ion1*ones(ion1_number,1);m_ion2*ones(ion2_number,1)];

    
particle_velocity0 = [electron_velocity;ion1_velocity;ion2_velocity];
particle_position0 = [electron_position0;ion1_position0;ion2_position0];

particle_position(:,1) = particle_position0(:,1) .* cos( particle_position0(:,2) );
particle_position(:,2) = particle_position0(:,1) .* sin( particle_position0(:,2) );
particle_position(:,3) = particle_position0(:,3);






for timee=1:cycle_time_for_electromagneticfield
    % Load the kernel
    
    cudaFilename1 = 'EM2.cu';
    ptxFilename1 = 'EM2.ptx';
    kernel1 = parallel.gpu.CUDAKernel( ptxFilename1, cudaFilename1 );
    kernel1.ThreadBlockSize = [1024,1,1];
    kernel1.GridSize = [600,1,1];
    
    cudaFilename2 = 'EM3.cu';
    ptxFilename2 = 'EM3.ptx';
    kernel2 = parallel.gpu.CUDAKernel( ptxFilename2, cudaFilename2 );
    kernel2.ThreadBlockSize = [101,1,1];
    kernel2.GridSize = [1001,1,1];
    
    cudaFilename2_1 = 'EM3_1.cu';
    ptxFilename2_1 = 'EM3_1.ptx';
    kernel2_1 = parallel.gpu.CUDAKernel( ptxFilename2_1, cudaFilename2_1 );
    kernel2_1.ThreadBlockSize = [100,1,1];
    kernel2_1.GridSize = [1000,1,1];

    cudaFilename3 = 'EM4.cu';
    ptxFilename3 = 'EM4.ptx';
    kernel3 = parallel.gpu.CUDAKernel( ptxFilename3, cudaFilename3 );
    kernel3.ThreadBlockSize = [1024,1,1];
    kernel3.GridSize = [600,1,1];
    
    cudaFilename5 = 'EM5.cu';
    ptxFilename5 = 'EM5.ptx';
    kernel5 = parallel.gpu.CUDAKernel( ptxFilename5, cudaFilename5 );
    kernel5.ThreadBlockSize = [1024,1,1];
    kernel5.GridSize = [600,1,1];
    
    x = particle_position0(:,1) .* cos( particle_position0(:,2) );
    y = particle_position0(:,1) .* sin( particle_position0(:,2) );
    z = particle_position0(:,3);
    vx = particle_velocity0(:,1);
    vy = particle_velocity0(:,2);
    vz = particle_velocity0(:,3);
    particle_number = length(particle_position(:,1));
    r = zeros(particle_number, 1, 'gpuArray');
    phi = zeros(particle_number, 1, 'gpuArray');
    parDelete = gpuArray(zeros(particle_number, 1, 'int32'));
    a = gpuArray(zeros(particle_number, 1, 'int32'));
    vr = zeros(particle_number, 1, 'gpuArray');
    kinetic = zeros(particle_number, 1, 'gpuArray');
    Er=zeros(size(gridE_r),'gpuArray');
    Ez=zeros(size(gridE_z),'gpuArray');
    Hphi=zeros(size(gridH_phi),'gpuArray');
    F1 = zeros(particle_number, 1,'gpuArray');
    F2 = zeros(particle_number, 1,'gpuArray');
    F3 = zeros(particle_number, 1,'gpuArray');
    ar = zeros(particle_number, 4, 'gpuArray');
    br = zeros(particle_number, 4, 'gpuArray');
    az = zeros(particle_number, 4, 'gpuArray');
    bz = zeros(particle_number, 4, 'gpuArray');
    energy_of_electric_field = zeros(100,1,'gpuArray');
    energy_of_magnetic_field = zeros(100,1,'gpuArray');
    kinetic_energy = zeros(100,1,'gpuArray');
    total_energy = zeros(100,1,'gpuArray');
    ratio = 5;
    ddt = ratio*dt;
    tic
    for k=1:200
        k

    [x,y,z,vx,vy,vz,r,phi,vr,a,parDelete] = feval( kernel1,x,y,z,vx,vy,vz,r,phi,vr,a,parDelete,particle_number,Rp, Lp, PHIp, delta_r, delta_z, ddt );
    [~,~,ar, br, az, bz] = feval(kernel5, r, z, ar, br, az, bz, particle_number, 100, 1000, delta_r, delta_z);

    ar = double(gather(ar));
    br = double(gather(br));
    az = double(gather(az));
    bz = double(gather(bz));
    vr = gather(vr);
    vz = gather(vz);
    gridjr = zeros(E_r_len,4);
    gridjz = zeros(E_z_len,4);
    for i = 1:4
        for n=1:particle_number
            gridjr(ar(n,i),i) = gridjr(ar(n,i),i) + vr(n)*particle_charge(n)*br(n,i);
            gridjz(az(n,i),i) = gridjz(az(n,i),i) + vz(n)*particle_charge(n)*bz(n,i);
        end
    end
    gridjr = sum(gridjr,2);
    gridjz = sum(gridjz,2);
    gridjr = FN*gpuArray(reshape(gridjr,size(gridE_r)));
    gridjr = gridjr./(PHIp*((E_r_R+delta_r/2).^2-(E_r_R-delta_r/2).^2)/2*delta_z);

    gridjz = FN*gpuArray(reshape(gridjz,size(gridE_z)));
    gridjz(1,:) = gridjz(1,:)./(PHIp*((E_z_R(1,:)+delta_r/2).^2)/2*delta_z);
    gridjz(2:H_phi_lenr+1,:) = gridjz(2:H_phi_lenr+1,:)./(PHIp*((E_z_R(2:H_phi_lenr+1,:)+delta_r/2).^2-(E_z_R(2:H_phi_lenr+1,:)-delta_r/2).^2)/2*delta_z);

    [Er,Ez,Hphi] = feval(kernel2_1,Er,Ez,Hphi,mu,delta_r,delta_z,-dt/4);
    for i = 1:(2*ratio)
        [Er,Ez,Hphi] = feval(kernel2_1,Er,Ez,Hphi,mu,delta_r,delta_z,dt/2);
        [~,~,~,Er,Ez] = feval(kernel2,Er,Ez,Hphi,Er,Ez,gridjr,gridjz,mu,epsilon,delta_r,delta_z,dt/2);
    end
    [Er,Ez,Hphi] = feval(kernel2_1,Er,Ez,Hphi,mu,delta_r,delta_z,dt/4);
    
    energy_of_electric_field(k) = PHIp*1/2*epsilon*(trapz(E_r_gridz,trapz(E_r_gridr,Er.^2.*E_r_R))+trapz(E_z_gridz,trapz(E_z_gridr,Ez.^2.*E_z_R)));
    energy_of_magnetic_field(k) = PHIp*1/2*mu*trapz(H_phi_gridz,trapz(H_phi_gridr,Hphi.^2.*H_phi_R));
    
    
    [x, y, z, vx, vy, vz, ~, ~, ~, ~, ~, ~, ~, ~,kinetic] = feval(kernel3, x, y, z, vx, vy, vz, r, phi, vr, Er, Ez, Hphi, particle_charge, particle_mass, kinetic, particle_number, 100, 1000, mu, c, delta_r, delta_z, ddt);
    
    kinetic_energy(k) = sum(kinetic)*FN;
    energy_of_electric_field(k)+energy_of_magnetic_field(k)+kinetic_energy(k)
    total_energy(k) = energy_of_electric_field(k)+energy_of_magnetic_field(k)+kinetic_energy(k);
    end
    toc
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    tic
    
    particle_position(:,1) = particle_position0(:,1) .* cos( particle_position0(:,2) );
    particle_position(:,2) = particle_position0(:,1) .* sin( particle_position0(:,2) );
    particle_position(:,3) = particle_position0(:,3);
    particle_velocity = particle_velocity0;
    gridE_r=zeros(size(gridE_r),'gpuArray');
    gridE_z=zeros(size(gridE_z),'gpuArray');
    gridH_phi=zeros(size(gridH_phi),'gpuArray');
    
    gridE_r0=zeros(size(gridE_r),'gpuArray');
    gridE_z0=zeros(size(gridE_z),'gpuArray');
    gridH_phi0=zeros(size(gridH_phi),'gpuArray');
    
    for kkk=1:100
        kkk
    particle_position_half = particle_position + 1/2*dt*particle_velocity;
    particle_r = sqrt(particle_position_half(:,1).^2+particle_position_half(:,2).^2);
    particle_phi = atan2(particle_position_half(:,2),particle_position_half(:,1));
    a = find(particle_r>Rp);

    particle_position_half(a,:) = [];
    particle_velocity(a,:) = [];

    [particle_position_half,particle_velocity,N_boundary] = REBOND( particle_position_half,particle_r,particle_phi,particle_velocity,Lp,PHIp);

    particle_r = sqrt(particle_position_half(:,1).^2+particle_position_half(:,2).^2);
    sin_particle_phi = particle_position_half(:,2)./particle_r;
    cos_particle_phi = particle_position_half(:,1)./particle_r;
    particle_velocity_r = particle_velocity(:,1).*cos_particle_phi +  particle_velocity(:,2).*sin_particle_phi;
    
    ar = zeros(particle_number, 4, 'gpuArray');
    br = zeros(particle_number, 4, 'gpuArray');
    az = zeros(particle_number, 4, 'gpuArray');
    bz = zeros(particle_number, 4, 'gpuArray');
    [~,~,ar, br, az, bz] = feval(kernel5, particle_r, particle_position_half(:,3), ar, br, az, bz, particle_number, 100, 1000, delta_r, delta_z);
  
    pChar = [particle_charge;particle_charge;particle_charge;particle_charge];
    pVeloR = [gather(particle_velocity_r);gather(particle_velocity_r);gather(particle_velocity_r);gather(particle_velocity_r)];
    pVeloZ = [double(gather(particle_velocity(:,3)));double(gather(particle_velocity(:,3)));double(gather(particle_velocity(:,3)));double(gather(particle_velocity(:,3)))];
    a = reshape(ar,particle_number*4,1);
    b = reshape(br,particle_number*4,1);
    a = gather(a);
    nod_e = ind2vec(a');
    jr = nod_e * (  pChar.*pVeloR.*b );
    gridjrr = zeros(size(gridE_r),'gpuArray');
    gridjrr(1:length(jr)) = FN*jr;
    gridjr = gridjrr./(PHIp*((E_r_R+delta_r/2).^2-(E_r_R-delta_r/2).^2)/2*delta_z);

    
    a = reshape(az,particle_number*4,1);
    b = reshape(bz,particle_number*4,1);
    a = gather(a);
    nod_e = ind2vec(a');
    jz = nod_e * (  pChar.*pVeloZ.*b );
    gridjzz = zeros(size(gridE_z),'gpuArray');
    gridjzz(1:length(jz)) = FN*jz;
    gridjz = zeros(size(gridE_z),'gpuArray');
    gridjz(1,:) = gridjzz(1,:)./(PHIp*((E_z_R(1,:)+delta_r/2).^2)/2*delta_z);
    gridjz(2:H_phi_lenr+1,:) = gridjzz(2:H_phi_lenr+1,:)./(PHIp*((E_z_R(2:H_phi_lenr+1,:)+delta_r/2).^2-(E_z_R(2:H_phi_lenr+1,:)-delta_r/2).^2)/2*delta_z);
    
    
    %{
    trapz(E_r_gridz,trapz(E_r_gridr,E_r_R.*(gridjr.*gridE_r)))*PHIp
    sum(sum(gridjr.*gridE_r.*(PHIp*((E_r_R+delta_r/2).^2-(E_r_R-delta_r/2).^2)/2*delta_z)))
    sum(sum(gridE_r.*gridE_r.*(PHIp*((E_r_R+delta_r/2).^2-(E_r_R-delta_r/2).^2)/2*delta_z)))
    EEr = [Er;Er;Er;Er];
    sum(sum(gridjrr.*gridE_r))
    sum((  particle_charge.*particle_velocity_r ).*Er)*FN
	sum((  pChar.*pVeloR.*b ).*EEr)*FN
    
    EEz = [Ez;Ez;Ez;Ez];
    trapz(E_z_gridz,trapz(E_z_gridr,E_z_R.*(gridjz.*gridE_z)))*PHIp
    sum(sum(gridjzz.*gridE_z))
    sum((  particle_charge.*particle_velocity(:,3) ).*Ez)*FN
    sum((  pChar.*pVeloZ.*b ).*EEz)*FN
    %}
   
    
    repeat = 2;

    dt_repeat = dt/repeat;
    
    energy_of_electric_field_before = PHIp*1/2*epsilon*(trapz(E_r_gridz,trapz(E_r_gridr,gridE_r.^2.*E_r_R))+trapz(E_z_gridz,trapz(E_z_gridr,gridE_z.^2.*E_z_R)));
    energy_of_magnetic_field_before = PHIp*1/2*mu*trapz(H_phi_gridz,trapz(H_phi_gridr,gridH_phi.^2.*H_phi_R));
    
    gridH_phi = gridH_phi - dt_repeat/2/(mu*delta_r)*( gridE_z(2:E_z_lenr,:)-gridE_z(1:(E_z_lenr-1),:) ) + dt_repeat/2/(mu*delta_z)*( gridE_r(:,2:E_r_lenz)-gridE_r(:,1:(E_r_lenz-1)) );
    for timee = 1:repeat       
        gridE_r1 = gridE_r(:,2);
        gridE_rend = gridE_r(:,E_r_lenz-1);
        gridH_phi = gridH_phi + dt_repeat/(mu*delta_r)*( gridE_z(2:E_z_lenr,:)-gridE_z(1:(E_z_lenr-1),:) ) - dt_repeat/(mu*delta_z)*( gridE_r(:,2:E_r_lenz)-gridE_r(:,1:(E_r_lenz-1)) );
        gridE_r(:,2:(E_r_lenz-1)) = gridE_r(:,2:(E_r_lenz-1)) - dt_repeat/epsilon*gridjr(:,2:H_phi_lenz) - dt_repeat/(epsilon*delta_z)*( gridH_phi(:,2:H_phi_lenz)-gridH_phi(:,1:(H_phi_lenz-1)) );
        gridE_z(2:(E_z_lenr-1),:) = gridE_z(2:(E_z_lenr-1),:) - dt_repeat/epsilon*gridjz(2:H_phi_lenr,:) + dt_repeat/epsilon*(1./(2*E_z_R(2:(E_z_lenr-1),:))+1/delta_r).*gridH_phi(2:H_phi_lenr,:) + dt_repeat/epsilon*(1./(2*E_z_R(2:(E_z_lenr-1),:))-1/delta_r).*gridH_phi(1:(H_phi_lenr-1),:) ;
        gridE_z(1,:) = gridE_z(1,:) + dt_repeat/epsilon*gridjz(1,:) + 4*dt_repeat/(epsilon*delta_r)*gridH_phi(1,:);
        gridE_r(:,1) = gridE_r1;
        gridE_r(:,E_r_lenz) = gridE_rend;
    end
    gridH_phi = gridH_phi + dt_repeat/2/(mu*delta_r)*( gridE_z(2:E_z_lenr,:)-gridE_z(1:(E_z_lenr-1),:) ) - dt_repeat/2/(mu*delta_z)*( gridE_r(:,2:E_r_lenz)-gridE_r(:,1:(E_r_lenz-1)) );
    
    energy_of_electric_field = PHIp*1/2*epsilon*(trapz(E_r_gridz,trapz(E_r_gridr,gridE_r.^2.*E_r_R))+trapz(E_z_gridz,trapz(E_z_gridr,gridE_z.^2.*E_z_R)));
    energy_of_magnetic_field = PHIp*1/2*mu*trapz(H_phi_gridz,trapz(H_phi_gridr,gridH_phi.^2.*H_phi_R));
    J_E = PHIp*dt*(trapz(E_r_gridz,trapz(E_r_gridr,gridE_r.*gridjr.*E_r_R))+trapz(E_z_gridz,trapz(E_z_gridr,gridE_z.*gridjz.*E_z_R)));
    
    

    
    delta_E_energy = energy_of_electric_field - energy_of_electric_field_before
    delta_B_energy = energy_of_magnetic_field - energy_of_magnetic_field_before
    particle_number = length(particle_position_half(:,1));
    
    
     gridE_r_interp = [gridE_r(1,:);gridE_r];
        gridE_z_interp = [gridE_z(:,1),gridE_z];
        gridH_phi_interp = [gridH_phi(1,:);gridH_phi];
        gridH_phi_interp = [gridH_phi_interp(:,1),gridH_phi_interp];
        
        
    %particle_velocity_r;

        
        E_r_gridr_interp = [0,E_r_gridr,Rp];
        E_r_gridz_interp = E_r_gridz;
        E_r_lenr_interp = length(E_r_gridr_interp);
        E_r_lenz_interp = length(E_r_gridz_interp);
        gridE_r_interp = zeros(E_r_lenr_interp,E_r_lenz_interp,'gpuArray');
        gridE_r_interp(2:(E_r_lenr_interp-1),:) = gridE_r;
        gridE_r_interp(1,:) = gridE_r(1,:);
        Er=interp2(E_r_gridz_interp,E_r_gridr_interp,gridE_r_interp,particle_position_half(:,3),particle_r);
        
        E_z_gridr_interp = E_z_gridr;
        E_z_gridz_interp = [0,E_z_gridz,Lp];
        E_z_lenr_interp = length(E_z_gridr_interp);
        E_z_lenz_interp = length(E_z_gridz_interp);
        gridE_z_interp = zeros(E_z_lenr_interp,E_z_lenz_interp,'gpuArray');
        gridE_z_interp(:,2:(E_z_lenz_interp-1)) = gridE_z;
        Ez=interp2(E_z_gridz_interp,E_z_gridr_interp,gridE_z_interp,particle_position_half(:,3),particle_r);
        
        H_phi_gridr_interp = [0,H_phi_gridr,Rp];
        H_phi_gridz_interp = [0,H_phi_gridz,Lp];
        H_phi_lenr_interp = length(H_phi_gridr_interp);
        H_phi_lenz_interp = length(H_phi_gridz_interp);
        gridH_phi_interp = zeros(H_phi_lenr_interp,H_phi_lenz_interp,'gpuArray');
        gridH_phi_interp(2:(H_phi_lenr_interp-1),2:(H_phi_lenz_interp-1)) = gridH_phi;
        gridH_phi_interp(1,2:(H_phi_lenz_interp-1)) = gridH_phi(1,:);
        gridH_phi_interp(H_phi_lenr_interp,2:(H_phi_lenz_interp-1)) = gridH_phi(H_phi_lenr,:);
        gridH_phi_interp(2:(H_phi_lenr_interp-1),1) = gridH_phi(:,1);
        gridH_phi_interp(2:(H_phi_lenr_interp-1),H_phi_lenz_interp) = gridH_phi(:,H_phi_lenz);
        gridH_phi_interp(1,1) = gridH_phi(1,1);
        gridH_phi_interp(1,H_phi_lenz_interp) = gridH_phi(1,H_phi_lenz);
        gridH_phi_interp(H_phi_lenr_interp,1) = gridH_phi(H_phi_lenr,1);
        gridH_phi_interp(H_phi_lenr_interp,H_phi_lenz_interp) = gridH_phi(H_phi_lenr,H_phi_lenz);
        Hphi=interp2(H_phi_gridz_interp,H_phi_gridr_interp,gridH_phi_interp,particle_position_half(:,3),particle_r);
        

        
        F_r = particle_charge.*(Er + (-particle_velocity(:,3).*Hphi*mu));
        F_z = particle_charge.*(Ez + (particle_velocity_r.*Hphi*mu));

        
        F(:,1) = F_r.*cos_particle_phi;
        F(:,2) = F_r.*sin_particle_phi;
       
        F(:,3) = F_z;
        
        particle_velocity_before = particle_velocity;
        
        gamma = 1./sqrt( 1-(particle_velocity(:,1).^2+particle_velocity(:,2).^2+particle_velocity(:,3).^2)/c^2 );
        
        particle_kinetic_before = FN*sum((gamma-1).*particle_mass*c^2);
        
        particle_u(:,1) = gamma.*particle_velocity(:,1);
        particle_u(:,2) = gamma.*particle_velocity(:,2);
        particle_u(:,3) = gamma.*particle_velocity(:,3);

        particle_u = particle_u + F./particle_mass *dt;


        gamma = sqrt(1+(particle_u(:,1).^2+particle_u(:,2).^2+particle_u(:,3).^2)/c^2);
        
        
        particle_kinetic = FN*sum((gamma-1).*particle_mass*c^2);
        delta_kinectic = particle_kinetic-particle_kinetic_before
        energy_of_electric_field
        energy_of_magnetic_field;
        particle_kinetic;
        energy_total = energy_of_electric_field + energy_of_magnetic_field + particle_kinetic
                
        
        particle_velocity(:,1) = particle_u(:,1)./gamma;
        particle_velocity(:,2) = particle_u(:,2)./gamma;
        particle_velocity(:,3) = particle_u(:,3)./gamma;

        particle_position = particle_position_half + 1/2*dt*particle_velocity;
        
        
    %end
        
        
    
    
    
    
    
  
        
        
        
        
        
        
        
        
        
        
        electron_position = particle_position(1:electron_number,:);
        electron_velocity = particle_velocity(1:electron_number,:);
        ion1_position = particle_position((electron_number+1):(electron_number+ion1_number),:);
        ion1_velocity = particle_velocity((electron_number+1):(electron_number+ion1_number),:);
        ion2_position = particle_position((electron_number+ion1_number+1):(electron_number+ion1_number+ion2_number),:);
        ion2_velocity = particle_velocity((electron_number+ion1_number+1):(electron_number+ion1_number+ion2_number),:);

                
        electron_r = sqrt(electron_position(:,1).^2+electron_position(:,2).^2);
        electron_phi = atan2(electron_position(:,2),electron_position(:,1));
        a = find(electron_r>Rp);
        electron_position(a,:) = [];
        electron_velocity(a,:) = [];
        [electron_position_half,electron_velocity,N_boundary] = REBOND( electron_position_half,electron_r,electron_phi,electron_velocity,Lp,PHIp);
        
        ion1_r = sqrt(ion1_position(:,1).^2+ion1_position(:,2).^2);
        ion1_phi = atan2(ion1_position(:,2),ion1_position(:,1));
        a = find(ion1_r>Rp);
        ion1_position(a,:) = [];
        ion1_velocity(a,:) = [];
        [ion1_position_half,ion1_velocity,N_boundary] = REBOND( ion1_position_half,ion1_r,ion1_phi,ion1_velocity,Lp,PHIp);

        ion2_r = sqrt(ion2_position(:,1).^2+ion2_position(:,2).^2);
        ion2_phi = atan2(ion2_position(:,2),ion2_position(:,1));
        a = find(ion2_r>Rp);
        ion2_position(a,:) = [];
        ion2_velocity(a,:) = [];
        [ion2_position_half,ion2_velocity,N_boundary] = REBOND( ion2_position_half,ion2_r,ion2_phi,ion2_velocity,Lp,PHIp);
    
    toc
    
    if mod(duandian,10000) == 1
        1;
    end
    
    
    end
end
