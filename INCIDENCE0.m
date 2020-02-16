function [No_photon,E_position,E_velocity, E_theta, E_chi, Ion_position, Ion_velocity,charge ] = INCIDENCE0(laser_r,z,kmax,m_a,W, Ex, No_photon_initial,position,velocity,phimax,N_total,p,delta_r)
% This function gives the position and velocity of photoelectroons
% and position of ions
% considers photoelectric effect and subsequent atomic relaxation

% input:
% laser_r is raduis of laser beam
% m_a is atomic mass number, denote the gas used
% Ex is energy of photons
% E_ray is energy of the incident pulse
% position is an n*4 matrix where the columns denote x,y,z��r
% velocity is an n*4 matrix where the fourth colume is speed

% output:
% No_photon is number of photons left
% E theta and E chi are the angle of velocity
% Ion_position is the position of ions. ions are assumed to be stationary
% Fifth column of Ion_position is the charge of ions

% set up constants
charge=zeros(N_total,1);
m_e = 9.11*10^(-31);
c = 3*10^8;

% set up matrixes
delta_z = z/kmax;
V_lasercell = phimax/2/pi*laser_r^2 * pi * delta_z;
N_laser = zeros(kmax,1); % number of particles in each laser cell
n_laser = zeros(kmax,1); % number density of each laser cell
dN_PE = zeros(kmax,1);
dN_Comp = zeros(kmax,1);
PExcited_index = [];
CExcited_index = [];

% find energy of shells and cross-sections
[E_Bind, N_shell ,E_shell ] = findEShell (m_a,Ex); % in keV
dE = (Ex-E_shell)*1.6*10^(-16);% in Joule
PE_velocity = findV(dE);
sigma_pe = findSigmaPE (Ex,m_a);% cross section for photoelectric effect in m^2
sigma_Cp = findSigmaCP( Ex);% cross section for compton scattering in m^2

% determine if compton scattering is needed
if m_a == 14 && Ex >= 10
    Comp = 1;
    CE_position_x = [];
    CE_position_y = [];
    CE_position_z = [];
else
    Comp = 0;
end

position_x = position (:,1);
position_y = position (:,2);
position_z = position (:,3);
position_r = position (:,4);
No_photon = No_photon_initial;

for k=1:kmax
    %%%%%%%%%%%%%%%%做了改动%%%%%%%%%%%%%%%%%%%
    %%k变成k-1，k+1变成k
    b=find(position_r<laser_r & position_z>=delta_z*(k-1) & position_z<delta_z*k);
    %%%%%%%%%%%%%%%%做了改动%%%%%%%%%%%%%%%%%%%
    % find number of particles in the same cell
    N_laser(k) = length(b) ;
    % find number density in each cell
    n_laser(k) = N_laser(k) * W / V_lasercell; % unweighted
    
    % find the index of the particles excited
    dN_PE(k) = n_laser(k)*sigma_pe*No_photon*delta_z;
    %No_photon = No_photon-dN_PE(k);（对的）
    No_photon = No_photon-1000*dN_PE(k);%(错的)
    %%%%%%%%%%%%%%%%修改部分%%%%%%%%%%%%%%%%%%
    %% 变成了高斯分�?(为了�?要函数多传进来了�?个矩阵p和delta_r)
    %dN_PE(k)= round(dN_PE(k));
    dN_PE(k)= dN_PE(k)/W;
    dN_PE_sub = round(dN_PE(k)*p);
    
    cc = -ones(max(dN_PE_sub),length(p));
    for sub = 1:length(p)
        b_sub = find(position_r(b)<sub*delta_r & position_r(b)>(sub-1)*delta_r);
        MIN = round(min(length(b_sub)/11,dN_PE_sub(sub))-0.5);
        cc(1:MIN,sub) = b(b_sub(11:11:(11*MIN)));
    end
    cc = reshape(cc,[max(dN_PE_sub)*length(p),1]);
    cc(cc<0) = [];
    Excited_PE = cc;
    %Excited_PE = b (1:dN_PE(k));
    %%%%%%%%%%%%%%%%修改部分%%%%%%%%%%%%%%%%%%
    charge(Excited_PE)=2;
    PExcited_index = [PExcited_index;Excited_PE];
end


%% find position of ions and electrons
E_position = position(PExcited_index,:);
Ion_position = position(PExcited_index,:);



%% find direction of photoelectron
E_theta = findPEangle (dE, length(PExcited_index));
E_chi = 2*pi*rand(length(PExcited_index),1);% assume unpolarized

% find velocity of electron in cartesian coordinate
E_velocity_x = PE_velocity.*sin(E_theta).*cos(E_chi);
E_velocity_y = PE_velocity.*sin(E_theta).*sin(E_chi);
E_velocity_z = PE_velocity.*cos(E_theta);


% use recoil 

photon_mtm = Ex *1.6*10^(-16)/c;
m = 2*m_a*1.6726*10^(-27);

e_mtm_x = m_e.* E_velocity_x;
e_mtm_y = m_e.* E_velocity_y;
e_mtm_z = m_e.* E_velocity_z;
ion_mtm_x = m.*velocity(PExcited_index,1);
ion_mtm_y = m.*velocity(PExcited_index,2);
ion_mtm_z = m.*velocity(PExcited_index,3);

% find energy deposited to ions
% atomic relaxation process through auger effect
if E_shell == E_Bind(N_shell)
    Charge = 1;
    velocity(PExcited_index,3) = (photon_mtm + ion_mtm_z - e_mtm_z)./m;
    velocity(PExcited_index,1) = (ion_mtm_x - e_mtm_x)./m;
    velocity(PExcited_index,2) = (ion_mtm_y - e_mtm_y)./m;
    velocity(PExcited_index,4) = sqrt(velocity(PExcited_index,1).^2+velocity(PExcited_index,2).^2+velocity(PExcited_index,3).^2);
else
    E_Aug = 0.383;
    Charge = 2;
    
    % production of Auger electron
    AE_position = position(PExcited_index,:);
    AE = E_Aug*1.6*10^(-16); % taken from data booklet
    AE_velocity = findV(AE);
    % assume isotropic distribution
    AE_theta = pi * rand(length(PExcited_index),1);
    AE_chi = 2*pi*rand(length(PExcited_index),1);
    AE_velocity_x = AE_velocity.*sin(AE_theta).*cos(AE_chi);
    AE_velocity_y = AE_velocity.*sin(AE_theta).*sin(AE_chi);
    AE_velocity_z = AE_velocity.*cos(AE_theta);
    E_velocity_x = [E_velocity_x;AE_velocity_x];
    E_velocity_y = [E_velocity_y;AE_velocity_y];
    E_velocity_z = [E_velocity_z;AE_velocity_z];
    E_position = [E_position;AE_position];
    E_theta = [E_theta;AE_theta];
    E_chi = [E_chi;AE_chi];
    AE_mtm_x = m_e.* AE_velocity_x;
    AE_mtm_y = m_e.* AE_velocity_y;
    AE_mtm_z = m_e.* AE_velocity_z;
    velocity(PExcited_index,3) = (photon_mtm + ion_mtm_z - e_mtm_z - AE_mtm_z)./m;
    velocity(PExcited_index,1) = (ion_mtm_x - e_mtm_x -AE_mtm_x)./m;
    velocity(PExcited_index,2) = (ion_mtm_y - e_mtm_y -AE_mtm_y)./m;
    velocity(PExcited_index,4) = sqrt(velocity(PExcited_index,1).^2+velocity(PExcited_index,2).^2+velocity(PExcited_index,3).^2);
end
Ion_velocity = velocity(PExcited_index,:);
E_speed = sqrt(E_velocity_x.^2 + E_velocity_y.^2 + E_velocity_z.^2);
E_velocity = [E_velocity_x,E_velocity_y,E_velocity_z,E_speed];



%record the charge of ions
Ion_position (:,5) = Charge;



end

