function [electron_position,electron_velocity,ion1_position,ion1_velocity,ion2_position,ion2_velocity,Er,Ez,Hphi,electron_kinetic] = electricmagneticfield5(electron_position0,electron_velocity,ion1_position0,ion1_velocity,ion2_position0,ion2_velocity,Er,Ez,Hphi,delta,dt,Rp,Lp,PHIp,FN,cycle_time_for_electromagneticfield)


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



    cudaFilename1 = 'EM2.cu';
    ptxFilename1 = 'EM2.ptx';
    kernel1 = parallel.gpu.CUDAKernel( ptxFilename1, cudaFilename1 );
    kernel1.ThreadBlockSize = [1024,1,1];
    kernel1.GridSize = [600,1,1];
    
    cudaFilename2 = 'EM3.cu';
    ptxFilename2 = 'EM3.ptx';
    kernel2 = parallel.gpu.CUDAKernel( ptxFilename2, cudaFilename2 );
    kernel2.ThreadBlockSize = [101,1,1];
    kernel2.GridSize = [501,1,1];%z
    
    cudaFilename2_1 = 'EM3_1.cu';
    ptxFilename2_1 = 'EM3_1.ptx';
    kernel2_1 = parallel.gpu.CUDAKernel( ptxFilename2_1, cudaFilename2_1 );
    kernel2_1.ThreadBlockSize = [100,1,1];
    kernel2_1.GridSize = [500,1,1];%z

    cudaFilename3 = 'EM4.cu';
    ptxFilename3 = 'EM4.ptx';
    kernel3 = parallel.gpu.CUDAKernel( ptxFilename3, cudaFilename3 );
    kernel3.ThreadBlockSize = [1024,1,1];
    kernel3.GridSize = [600,1,1];
    
    cudaFilename5 = 'EM6.cu';
    ptxFilename5 = 'EM6.ptx';
    kernel5 = parallel.gpu.CUDAKernel( ptxFilename5, cudaFilename5 );
    kernel5.ThreadBlockSize = [1024,1,1];
    kernel5.GridSize = [600,1,1];
    
    x = particle_position0(:,1) .* cos( particle_position0(:,2) );
    y = particle_position0(:,1) .* sin( particle_position0(:,2) );
    z = particle_position0(:,3);
    vx = particle_velocity0(:,1);
    vy = particle_velocity0(:,2);
    vz = particle_velocity0(:,3);
    particle_number = length(x);
    r = zeros(particle_number, 1, 'gpuArray');
    phi = zeros(particle_number, 1, 'gpuArray');
    parDelete = gpuArray(zeros(particle_number, 1, 'int32'));
    a = gpuArray(zeros(particle_number, 1, 'int32'));
    vr = zeros(particle_number, 1, 'gpuArray');
    kinetic = zeros(particle_number, 1, 'gpuArray');
    a = zeros(particle_number, 4, 'gpuArray');
    b = zeros(particle_number, 4, 'gpuArray');
    energy_of_electric_field = zeros(100,1,'gpuArray');
    energy_of_magnetic_field = zeros(100,1,'gpuArray');
    kinetic_energy = zeros(100,1,'gpuArray');
    total_energy = zeros(100,1,'gpuArray');
    ratio = 5;
    ddt = ratio*dt;
    particle_number_initial = particle_number;
    tic
for k=1:(cycle_time_for_electromagneticfield/ratio)
        

    [x,y,z,vx,vy,vz,r,phi,vr,parDelete] = feval( kernel1,x,y,z,vx,vy,vz,r,phi,vr,parDelete,particle_number,Rp, Lp, PHIp, delta_r, delta_z, ddt );

                                                                  %z
    [~,~,a, b] = feval(kernel5, r, z, a, b, parDelete, particle_number, 100, 500, delta_r, delta_z);

    a = double(gather(a));
    b = double(gather(b));
    vr = gather(vr);
    vz = gather(vz);
    gridjr = zeros(H_phi_len,4);
    gridjz = zeros(H_phi_len,4);
    
    for i = 1:4
        for n=1:particle_number
            gridjr(a(n,i),i) = gridjr(a(n,i),i) + vr(n)*particle_charge(n)*b(n,i);
            gridjz(a(n,i),i) = gridjz(a(n,i),i) + vz(n)*particle_charge(n)*b(n,i);
        end
    end
    gridjr = sum(gridjr,2);
    gridjz = sum(gridjz,2);

    gridjr = FN*gpuArray(reshape(gridjr,H_phi_lenr,H_phi_lenz));
    gridjr = gridjr./(PHIp/2*((H_phi_R+delta_r/2).^2-(H_phi_R-delta_r/2).^2)*delta_z);
    
    gridjz = FN*gpuArray(reshape(gridjz,H_phi_lenr,H_phi_lenz));
    gridjz = gridjz./(PHIp/2*((H_phi_R+delta_r/2).^2-(H_phi_R-delta_r/2).^2)*delta_z);
    
    [Er,Ez,Hphi] = feval(kernel2_1,Er,Ez,Hphi,mu,delta_r,delta_z,-dt/4);
    for i = 1:(2*ratio)
        [Er,Ez,Hphi] = feval(kernel2_1,Er,Ez,Hphi,mu,delta_r,delta_z,dt/2);
        [~,~,~,Er,Ez] = feval(kernel2,Er,Ez,Hphi,Er,Ez,gridjr,gridjz,mu,epsilon,delta_r,delta_z,dt/2);
    end
    [Er,Ez,Hphi] = feval(kernel2_1,Er,Ez,Hphi,mu,delta_r,delta_z,dt/4);
    
    energy_of_electric_field = PHIp*1/2*epsilon*(trapz(E_r_gridz,trapz(E_r_gridr,Er.^2.*E_r_R))+trapz(E_z_gridz,trapz(E_z_gridr,Ez.^2.*E_z_R)));
    energy_of_magnetic_field = PHIp*1/2*mu*trapz(H_phi_gridz,trapz(H_phi_gridr,Hphi.^2.*H_phi_R));
                                                                                                                                                                             %z
    [x, y, z, vx, vy, vz, ~, ~, ~, ~, ~, ~, ~, ~,kinetic,parDelete] = feval(kernel3, x, y, z, vx, vy, vz, r, phi, vr, Er, Ez, Hphi, particle_charge, particle_mass, kinetic, parDelete, particle_number, Rp, Lp, PHIp, 100, 500, mu, c, delta_r, delta_z, ddt);

    kinetic_energy = sum(kinetic)*FN;
    energy_of_electric_field+energy_of_magnetic_field+kinetic_energy;
    total_energy = energy_of_electric_field+energy_of_magnetic_field+kinetic_energy;
    if mod(k,50)==1
        total_energy
    end
    

end
    toc
parDelete = logical(parDelete);    
electron_number = electron_number - (particle_number_initial -  particle_number);
particle_position = [x,y,z];
particle_velocity = [vx,vy,vz];
electron_kinetic = kinetic(1:electron_number,:)/(1000*e);% (keV)
electron_position = particle_position(1:electron_number,:);
electron_velocity = particle_velocity(1:electron_number,:);
electron_delete = parDelete(1:electron_number);
electron_position(electron_delete,:) = [];
electron_velocity(electron_delete,:) = [];
electron_kinetic(electron_delete,:) = [];
ion1_position = particle_position((electron_number+1):(electron_number+ion1_number),:);
ion1_velocity = particle_velocity((electron_number+1):(electron_number+ion1_number),:);
ion1_delete = parDelete((electron_number+1):(electron_number+ion1_number));
electron_position(ion1_delete,:) = [];
electron_velocity(ion1_delete,:) = [];
ion2_position = particle_position((electron_number+ion1_number+1):(electron_number+ion1_number+ion2_number),:);
ion2_velocity = particle_velocity((electron_number+ion1_number+1):(electron_number+ion1_number+ion2_number),:);
ion2_delete = parDelete((electron_number+ion1_number+1):(electron_number+ion1_number+ion2_number));
electron_position(ion2_delete,:) = [];
electron_velocity(ion2_delete,:) = [];














