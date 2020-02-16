                                                                                                                                                                                                                                                                                                                                                                                                                                                                      %(* ::Package:: *)
profile on
%% update 8.25 
% input parameters:real physical parameters % unit is SI
kb=1.38064853E-23;  
T0=300; 
Rp=1E-2; % radius of the attenuator
Lp=0.05;  % length of the attenuator
PHIp=0.1; % the degree phi 's region is from 0 to phimax
V=0.5*Rp^2*PHIp*Lp; % volume
NA=6.02E23;
p=10;%10 Pa, the pressure
diaref=4.17E-10;sigmaref=pi*diaref^2;Tref=273; % parameters for VHS/VSS
d=3.7E-10; % diameter of N2
m=4.65E-26;  % mass of one N2 
w=0.74; % exponent of viscosity on temperature
zeta=2;
N_real=p*V/kb/T0; % num of real mol.
num_density=N_real/V;
lambda=1/(sqrt(2)*pi*num_density*d^2); % lambda of initial mol.
m_e=9.11E-31;
delta = 1*10^(-4);

% parameters for laser
laser_r=0.003;
Ex=1;  % energy of each photon :1keV
m_a=14; % N2
No_photon_initial = 1.25E13*PHIp/2/pi;
% I0=1.6E-4; % energy of one pulse

% different process dt
dt=0.2*1E-6; % iteration time for molecule transport


% input parameters:simulation parameters
SCmax=15.0*sigmaref*sqrt(kb*T0/m); % max of cr*sigma
alpha_=1; % parameter in VSS alpha=1 is VHS
A = 100;
dr = delta;
dz = A*delta; % step of z
rmax = round(Rp/dr);  % num to divide r
zmax = round(Lp/dz); % num to divide z

dphi=zeros(rmax,1);

[cellnumxy,dphi,phimax,sum_cell,cellr,cellphi] = cell_structure( PHIp,Rp,rmax,delta);


cellnum=cellnumxy*zmax;


N_total=floor(int64(A*5E5*cellnum)); % number of simulated mol.
N_total0=N_total;
FN=double (N_real)/double(N_total); % the number of real mol. represented by simulated mol.

particle_num_density_in_every_cell = num_density * ones(cellnumxy,zmax,'gpuArray');
temperature_in_every_cell = T0 * ones(cellnumxy,zmax,'gpuArray');
every_cell_volume=calculate_VC( Rp,Lp,rmax,phimax,zmax,dphi,cellnumxy,sum_cell,cellnum );



%%Initialnize

position = zeros(0,3,'gpuArray');
velocity = zeros(0,3,'gpuArray');
electron_position = zeros(0,3,'gpuArray');
electron_velocity = zeros(0,3,'gpuArray');
ion1_position = zeros(0,3,'gpuArray');
ion1_velocity = zeros(0,3,'gpuArray');
ion2_position = zeros(0,3,'gpuArray');
ion2_velocity = zeros(0,3,'gpuArray');



% parameters for electricmagnetic field
delta_r = delta;
delta_z = delta;


H_theta_gridr = delta_r/2:delta_r:Rp-delta_r/2;
H_theta_gridz = delta_z/2:delta_z:Lp-delta_z/2;
H_theta_lenr = length(H_theta_gridr);
H_theta_lenz = length(H_theta_gridz);
H_theta_len = length(H_theta_gridr)*length(H_theta_gridz);

[H_theta_R,H_theta_Z] = meshgrid(H_theta_gridr,H_theta_gridz);
H_theta_R = gpuArray(H_theta_R');
H_theta_Z = gpuArray(H_theta_Z');

gridH_theta = zeros(H_theta_lenr,H_theta_lenz,'gpuArray');



E_r_gridr = delta_r/2:delta_r:Rp-delta_r/2;
E_r_gridz = 0:delta_z:Lp;

E_r_lenr = length(E_r_gridr);
E_r_lenz = length(E_r_gridz);
E_r_len = length(E_r_gridr)*length(E_r_gridz);

[E_r_R,E_r_Z] = meshgrid(E_r_gridr,E_r_gridz);
E_r_R = gpuArray(E_r_R');
E_r_Z = gpuArray(E_r_Z');

gridE_r = zeros(E_r_lenr,E_r_lenz,'gpuArray');



E_z_gridr = 0:delta_r:Rp;
E_z_gridz = delta_z/2:delta_z:Lp-delta_z/2;

E_z_lenr = length(E_z_gridr);
E_z_lenz = length(E_z_gridz);
E_z_len = length(E_z_gridr)*length(E_z_gridz);

[E_z_R,E_z_Z] = meshgrid(E_z_gridr,E_z_gridz);
E_z_R = gpuArray(E_z_R');
E_z_Z = gpuArray(E_z_Z');

gridE_z = zeros(E_z_lenr,E_z_lenz,'gpuArray');


%% start time loop


for TIME=1:100
    TIME
    %% laser input
    % calculation
    [p] = GUASS(delta_r,laser_r,sum_cell);
    fprintf('INCIDENCE time:\n')

    tic
     [No_photon,electron_position,electron_velocity, ion2_position, ion2_velocity ,particle_num_density_in_every_cell,temperature_in_every_cell] = INCIDENCE1(zmax,m_a, FN, Ex, No_photon_initial,particle_num_density_in_every_cell,temperature_in_every_cell,cellnum,cellnumxy,p,sum_cell,dr,dphi,dz,cellr,cellphi,every_cell_volume);
    toc
    No_photon_initial
    No_photon

    % plot:electron number density in radial direction
    index_r = round(electron_position(:,1)/dr+0.5);
    index_r = gather(index_r);
    particle_density = ind2vec(index_r');
    particle_density = sum(particle_density,2);
    particle_density = full(particle_density);
    i = 1:30;
    VC = PHIp*((i*dr).^2-((i-1)*dr).^2)*Lp;
    particle_density = FN*particle_density./VC';
    figure
    plot(1:30,particle_density);



    %% electron transport    
    [ gridE_r,gridE_z,gridH_theta,electron_position,electron_velocity,ion1_position,ion1_velocity,ion2_position,ion2_velocity,particle_num_density_in_every_cell,temperature_in_every_cell] = electron_transport_field3( position,velocity,every_cell_volume,FN,electron_position,electron_velocity,ion1_position,ion1_velocity,ion2_position,ion2_velocity,m,m_e,rmax,phimax,zmax,Rp,Lp,PHIp,cellnumxy,sum_cell,gridE_r,gridE_z,gridH_theta,delta,dr,dz,particle_num_density_in_every_cell,temperature_in_every_cell );
    
     %% plot
    electron_position_r_abs = sqrt(x_e.^2+y_e.^2);
    electron_cos_theta = x_e./electron_position_r_abs;
    electron_sin_theta = y_e./electron_position_r_abs;
    rho_electron=zeros(length(gridr),1);
    parfor i=1:length(gridr)
        rho_electron(i) =  FN*length (find((electron_position_r_abs<i*delta_r)&(electron_position_r_abs>(i-1)*delta_r)))/(phimax*((i*delta_r)^2-((i-1)*delta_r)^2)*Lp);
    end
    figure
    plot(gridr,rho_electron);
    figure
    
    
    
    %% change the statistical weight
    
    [r,z,PHIp,vx,vy,vz] = SamplingBack(amplification,r,z,PHIp,vx,vy,vz);
    FN = FN*amplification^3;
    x=r.*cos(PHIp);
    y=r.*sin(PHIp);
    figure
    scatter3(x,y,z,1,'filled')

    end
    
    


% movie(fmat2)
