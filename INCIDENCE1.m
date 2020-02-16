function [No_photon,E_position,E_velocity, ion2_position, ion2_velocity ,particle_num_density_in_every_cell,temperature_in_every_cell] = INCIDENCE1(zmax,m_a, FN, Ex, No_photon_initial,particle_num_density_in_every_cell,temperature_in_every_cell,cellnum,cellnumxy,p,sum_cell,dr,dphi,dz,cellr,cellphi,every_cell_volume)
% This function gives the position and velocity of photoelectroons
% and position of ions
% considers photoelectric effect and subsequent atomic relaxation

% input:
% laser_r is raduis of laser beam
% m_a is atomic mass number, denote the gas used
% Ex is energy of photons
% E_ray is energy of the incident pulse
% position is an n*4 matrix FNhere the columns denote x,y,z��r
% velocity is an n*4 matrix FNhere the fourth colume is speed

% output:
% No_photon is number of photons left
% E theta and E chi are the angle of velocity
% Ion_position is the position of ions. ions are assumed to be stationary
% Fifth column of Ion_position is the charge of ions

% set up constants
m_e = 9.11*10^(-31);
c = 3*10^8;
kb = 1.38064853E-23;
mass = 2*m_a*1.6726*10^(-27);

% find energy of shells and cross-sections
[E_Bind, N_shell ,E_shell ] = findEShell (m_a,Ex); % in keV
dE = (Ex-E_shell)*1.6*10^(-16);% in Joule
PE_velocity = findV(dE);
sigma_pe = findSigmaPE (Ex,m_a);% cross section for photoelectric effect in m^2
sigma_Cp = findSigmaCP( Ex);% cross section for compton scattering in m^2



par_density = gpuArray(particle_num_density_in_every_cell);
par_density = reshape(par_density,cellnumxy,zmax);
particle_density = par_density(1:length(p),:);
A=1;
cell_No_photon = [];
cell_No_photon = p * No_photon_initial;
position_z = [];
velocity = [];
sum_cell = [0,sum_cell];


position_r = [];
position_phi = [];
velocity=[];
energy_sampling = zeros(length(p),zmax,'gpuArray');
particle_number_sampling = zeros(length(p),zmax,'gpuArray');
for k=1:zmax

    l=1:length(p);
    % find the index of the particles excited
    dN_PE = [];
    
    dN_PE = particle_density(:,k) .*cell_No_photon * sigma_pe * dz * A;
    cell_No_photon = cell_No_photon-dN_PE;
    dN_PE_sub = dN_PE/FN;
    dN_PE_sub = round(dN_PE_sub);  
    
    position_z =[position_z; (k-1) * dz * A + dz * A * gpuArray.rand(sum(dN_PE_sub),1)];
    
    every_energy_sampling = zeros(length(p),1,'gpuArray');
    every_particle_number_sampling = zeros(length(p),1,'gpuArray');
    parfor m = 1:length(p)
        position_r = [position_r; (cellr(m) + dr * gpuArray.rand(dN_PE_sub(m),1) -dr)];
        position_phi = [position_phi; (cellphi(m) + dphi(round(cellr(m)/dr)) * gpuArray.rand(dN_PE_sub(m),1) -dphi(round(cellr(m)/dr)))];
        T0 = temperature_in_every_cell((k-1)*cellnumxy+m);
        sig = sqrt(kb*T0/mass);
        velocity_in_cell = sig*gpuArray.randn(dN_PE_sub(m),3);
        velocity = [velocity; velocity_in_cell];
        position_r_in_cell = cellr(m) + dr * gpuArray.rand(dN_PE_sub(m),1) -dr;
        position_phi_in_cell = cellphi(m) + dphi(round(cellr(m)/dr)) * gpuArray.rand(dN_PE_sub(m),1) -dphi(round(cellr(m)/dr));
        %velocity = [velocity ;velocity_in_cell ];
        % change the number density and temperature in the cell 
        every_energy_sampling(m) = FN*0.5*mass*sum(sum(velocity_in_cell.^2,2));
        every_particle_number_sampling(m) = FN*length(velocity_in_cell(:,1));
    end
    energy_sampling(:,k) = every_energy_sampling;
    particle_number_sampling(:,k) = every_particle_number_sampling;
end         

No_photon = sum(cell_No_photon);
excited_molecule_number = length(position_r);

particle_number = particle_num_density_in_every_cell(1:length(p),:).*(every_cell_volume(1:length(p))*ones(1,zmax,'gpuArray'));
energy = 1.5*kb* particle_number .* temperature_in_every_cell(1:length(p),:);
energy = energy - energy_sampling;
particle_number =particle_num_density_in_every_cell(1:length(p),:).*(every_cell_volume(1:length(p))*ones(1,zmax,'gpuArray')) - particle_number_sampling ;
particle_num_density_in_every_cell(1:length(p),:) = particle_number./(every_cell_volume(1:length(p))*ones(1,zmax,'gpuArray'));
temperature_in_every_cell(1:length(p),:) = energy./(1.5*kb*particle_number);


%% find position of ions and electrons
E_position = [position_r,position_phi,position_z];
ion2_position = [position_r,position_phi,position_z];


%% find direction of photoelectron
E_theta = findPEangle(dE,excited_molecule_number);
E_chi = 2*pi*gpuArray.rand(excited_molecule_number,1);% assume unpolarized
E_theta = gpuArray(E_theta);

% find velocity of electron in cartesian coordinate
E_velocity_x = PE_velocity.*sin(E_theta).*cos(E_chi);
E_velocity_y = PE_velocity.*sin(E_theta).*sin(E_chi);
E_velocity_z = PE_velocity.*cos(E_theta);


% use recoil 

photon_mtm = Ex *1.6*10^(-16)/c;

e_mtm_x = m_e.* E_velocity_x;
e_mtm_y = m_e.* E_velocity_y;
e_mtm_z = m_e.* E_velocity_z;
ion_mtm_x = mass.*velocity(:,1);
ion_mtm_y = mass.*velocity(:,2);
ion_mtm_z = mass.*velocity(:,3);

% find energy deposited to ions
% atomic relaxation process through auger effect
if E_shell == E_Bind(N_shell)
    %Charge = 1;
    velocity(:,3) = (photon_mtm + ion_mtm_z - e_mtm_z)./mass;
    velocity(:,1) = (ion_mtm_x - e_mtm_x)./mass;
    velocity(:,2) = (ion_mtm_y - e_mtm_y)./mass;
else
    E_Aug = 0.383;
    Charge = 2;
    
    % production of Auger electron
    AE_position = E_position;
    AE = E_Aug*1.6*10^(-16); % taken from data booklet
    AE_velocity = findV(AE);
    % assume isotropic distribution
    AE_theta = pi * gpuArray.rand(excited_molecule_number,1);
    AE_chi = 2*pi* gpuArray.rand(excited_molecule_number,1);
    AE_velocity_x = AE_velocity.*sin(AE_theta).*cos(AE_chi);
    AE_velocity_y = AE_velocity.*sin(AE_theta).*sin(AE_chi);
    AE_velocity_z = AE_velocity.*cos(AE_theta);
    E_velocity_x = [E_velocity_x;AE_velocity_x];
    E_velocity_y = [E_velocity_y;AE_velocity_y];
    E_velocity_z = [E_velocity_z;AE_velocity_z];  
    E_position = [E_position;AE_position];
    %E_theta = [E_theta;AE_theta];
    %E_chi = [E_chi;AE_chi];
    AE_mtm_x = m_e.* AE_velocity_x;
    AE_mtm_y = m_e.* AE_velocity_y;
    AE_mtm_z = m_e.* AE_velocity_z;
    velocity(:,3) = (photon_mtm + ion_mtm_z - e_mtm_z - AE_mtm_z)./mass;
    velocity(:,1) = (ion_mtm_x - e_mtm_x -AE_mtm_x)./mass;
    velocity(:,2) = (ion_mtm_y - e_mtm_y -AE_mtm_y)./mass;
end
ion2_velocity = velocity;

E_velocity = [E_velocity_x,E_velocity_y,E_velocity_z];







