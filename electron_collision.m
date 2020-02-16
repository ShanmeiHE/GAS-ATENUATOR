function [electron_position_after,electron_velocity_after,ion1_position,ion1_velocity] = electron_collision(electron_position,electron_velocity,ion1_position,ion1_velocity, t_el,t_sel_1,t_del,t_sdel,t_exc,electron_kinetic,sum_cell,cellnumxy,particle_num_density_in_every_cell,temperature_in_every_cell,PHIp,phimax,dr,dz,sig_elastic,sig_ionization,sig_excitation,dt_transport)

%% cuda set
cudaFilename = 'electron_transport.cu';
ptxFilename = 'electron_transport.ptx';
kernel = parallel.gpu.CUDAKernel( ptxFilename, cudaFilename );
kernel.ThreadBlockSize = [600,1,1];
kernel.GridSize = [600,1,1];

%% elastic data
% Pre-set table 
sqr_a0 = 2.8002852E-21; % unit: m^2
% change the unit of TDC for energy < 50eV
% and change to Nitrogen atom so can combine with the other table without confusion  
t_sel_1(:,2) = t_sel_1(:,2) .* 10^-20 / sqr_a0 /2 ; 
t_sel_1(:,1) = t_sel_1(:,1)./1000; % change the energy unit to keV 
t_sel_1 = t_sel_1(1:(length(t_sel_1)-1),1:2);
% Combine the table for TCS
t_el = t_el';
t_el = [t_sel_1;t_el];

% Combine the table for DCS
[l,w] = size(t_del);
t_del_1 = t_del(1:l,2:w); 
t_del = [t_sdel,t_del_1]; 

% plot data points of DCS
[len,width] = size(t_del);
DCS = t_del(2:len,2:width);
V_interp = log(DCS);
E = t_del(1,2:width);
xInterp = log(E);
yInterp = t_el(:,2);

%% Excitation data
kExc = t_exc(:,1);
Aexc = t_exc(:,2);
omegaExc = t_exc(:,3);
vExc = t_exc(:,4);
gammaExc = t_exc(:,5);
lenExc = length(kExc);

%% Initialize
parNum = length(electron_position(:,1));
secondaryParticle = gpuArray(zeros( parNum , 1, 'logical'));
secondary_vx = zeros( parNum,1,'gpuArray' );
secondary_vy = zeros( parNum,1,'gpuArray' );
secondary_vz = zeros( parNum,1,'gpuArray' );
particleSamplingOut = zeros( parNum,1,'gpuArray' );
ion1_vx = zeros( parNum,1,'gpuArray' );
ion1_vy = zeros( parNum,1,'gpuArray' );
ion1_vz = zeros( parNum,1,'gpuArray' );
sum_cell0 =[0,sum_cell];
dphi = PHIp*ones(length(phimax),1)./phimax; 

%% cuda calculation
tic
[x_e,y_e,z_e,vx_e,vy_e,vz_e,~,~,~,~,~,~,~,particle_num_density_in_every_cell,temperature_in_every_cell,~,~,~,~,~,~,secondaryParticle,secondary_vx,secondary_vy,secondary_vz,particleSamplingOut,ion1_vx,ion1_vy,ion1_vz,~]=feval(kernel,electron_position(:,1),electron_position(:,2),electron_position(:,3),electron_velocity(:,1),electron_velocity(:,2),electron_velocity(:,3),sig_elastic,sig_ionization,sig_excitation,electron_kinetic,xInterp,yInterp,V_interp,particle_num_density_in_every_cell,temperature_in_every_cell,dphi,kExc,Aexc,omegaExc,vExc,gammaExc,secondaryParticle,secondary_vx,secondary_vy,secondary_vz,particleSamplingOut,ion1_vx,ion1_vy,ion1_vz,sum_cell0,lenExc,cellnumxy,dr,dz,dt_transport,parNum);
toc

%% update electron and ion's position and velocity

ion1_position_add = [x_e(secondaryParticle),y_e(secondaryParticle),z_e(secondaryParticle)];
ion1_position = [ion1_position;ion1_position_add];
ion1_velocity_add = [ion1_vx(secondaryParticle),ion1_vy(secondaryParticle),ion1_vz(secondaryParticle)];
ion1_velocity = [ion1_velocity;ion1_velocity_add];

x_e = [x_e;x_e(secondaryParticle)];
y_e = [y_e;y_e(secondaryParticle)];
z_e = [z_e;z_e(secondaryParticle)];
vx_e = [vx_e;secondary_vx(secondaryParticle)];
vy_e = [vy_e;secondary_vy(secondaryParticle)];
vz_e = [vz_e;secondary_vz(secondaryParticle)];
electron_position_after = [x_e,y_e,z_e];
electron_velocity_after = [vx_e,vy_e,vz_e];











