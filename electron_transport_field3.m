function [ gridE_r,gridE_z,gridH_phi,electron_position,electron_velocity,ion1_position,ion1_velocity,ion2_position,ion2_velocity,particle_num_density_in_every_cell,temperature_in_every_cell] = electron_transport_field3( position,velocity,every_cell_volume,FN,electron_position,electron_velocity,ion1_position,ion1_velocity,ion2_position,ion2_velocity,m,m_e,rmax,phimax,zmax,Rp,Lp,PHIp,cellnumxy,sum_cell,gridE_r,gridE_z,gridH_phi,delta,dr,dz,particle_num_density_in_every_cell,temperature_in_every_cell );

c=3E8;
%occupy=ones(n_e,1);   %denote whether the electron exists
%% Read table for elastic scattering 
% Find total cross section: 
% For energy > 50 eV
T_el = readtable("TotalElasticCS_Nitrogen.csv");
t_el = gpuArray(table2array(T_el));
% For 1 eV < energy < 50 eV
T_sel = readtable("TotalElasticCS_smalleV_Nitrogen.csv");
t_sel = gpuArray(table2array(T_sel));
T_sel_1 = readtable("TotalElasticCS_smalleV_Nitrogen_1.csv");
t_sel_1 = table2array(T_sel_1);


% Find differential cross section: 
% For energy > 50 eV
T_del = readtable("ElasticCS_Nitrogen.csv");
t_del = table2array(T_del);
% For 1eV < energy < 50 eV
T_sdel = readtable("ElasticDCS_smalleV_Nitrogen.csv");
t_sdel = table2array(T_sdel);

%% Read table for total cross section of impact ionization
T_ion = readtable("TotalIonCS_Nitrogen.csv");
t_ion = table2array(T_ion);

%% Read table of parameters of impact excitation
% Fitting parameters for Nitrogen:
T_exc = readtable("ParameterExcCS_Nitrogen.csv");
t_exc = table2array(T_exc);

%% Read table of parameters of impact excitation
% Fitting parameters for Nitrogen:
T_exc = readtable("ParameterExcCS_Nitrogen.csv");
t_exc = table2array(T_exc);


%% parameter
kb = 1.38064853E-23;  
m_ion = 4.65E-26;
e = 1.6*10^(-19);
time=0;

dt_transport=1E-10;
dt_electromagneticfield = 1*10^(-13);


cycle_time_for_electromagneticfield = round(dt_transport/dt_electromagneticfield);
cycle_time_for_electron_transport = 100;


threshold = 15.6*10^(-3); % Ionization threshold, unit : eV


kk = 0;
tt = 0;
iii = 0;
tic
for time = 1:cycle_time_for_electron_transport
    fprintf('t=%d\n',time*dt_transport);
    n_e = length(electron_position(:,1))
    %% motion in electricmagnetic field
        
    fprintf('EM field:\n');        
    
    [electron_position,electron_velocity,ion1_position,ion1_velocity,ion2_position,ion2_velocity,gridE_r,gridE_z,gridH_phi,electron_kinetic] = electricmagneticfield6(electron_position,electron_velocity,ion1_position,ion1_velocity,ion2_position,ion2_velocity,gridE_r,gridE_z,gridH_phi,delta,dt_electromagneticfield,Rp,Lp,PHIp,FN,cycle_time_for_electromagneticfield);
    if mod(time,50)==49
        profile viewer
    end
            
            %historam(sqrt(electron_velocity(:,1).^2+electron_velocity(:,2).^2+electron_velocity(:,3).^2));
            %histogram(1/2*m_e/e*(electron_velocity(:,1).^2+electron_velocity(:,2).^2+electron_velocity(:,3).^2));
            %kk = kk+1;
            %histogram(1/2*m_ion/e*(ion2_velocity(:,1).^2+ion2_velocity(:,2).^2+ion2_velocity(:,3).^2));
            %fmat(kk,:) = getframe;
%{
H_phi_gridr = gpuArray.linspace(delta/2,Rp-delta/2,Rp/delta);
H_phi_gridz = gpuArray.linspace(delta/2,Lp-delta/2,Lp/delta);

H_phi_lenr = length(H_phi_gridr);
H_phi_lenz = length(H_phi_gridz);
H_phi_len = length(H_phi_gridr)*length(H_phi_gridz);

[H_phi_R,H_phi_Z] = meshgrid(H_phi_gridr,H_phi_gridz);
H_phi_R = H_phi_R';
H_phi_Z = H_phi_Z';
                    figure                              %z
                    quiver(H_phi_R,H_phi_Z,gridE_r(:,2:501),gridE_z(1:100,:));
                    figure                                 %z
                    surf(H_phi_R,H_phi_Z,sqrt(gridE_r(:,1:500).^2+gridE_z(1:100,:).^2))
                    shading interp
                    colorbar
                    colormap(jet);
                    figure
                    surf(H_phi_R,H_phi_Z,gridH_phi)
                    shading interp
                    colorbar
                    colormap(jet);
                    figure                                  %z
                    plot(H_phi_R(:,1),sum(sqrt(gridE_r(:,1:500).^2+gridE_z(1:100,:).^2)')/500)
                  %}  

    %% sampling the collision
    length(electron_position(:,1))
    
    [sig_elastic,sig_ionization,sig_excitation]=find_CS0( t_el, t_sel, t_ion, t_exc, electron_kinetic );
    [electron_position,electron_velocity,ion1_position,ion1_velocity] = electron_collision(electron_position,electron_velocity,ion1_position,ion1_velocity, t_el,t_sel_1,t_del,t_sdel,t_exc,electron_kinetic,sum_cell,cellnumxy,particle_num_density_in_every_cell,temperature_in_every_cell,PHIp,phimax,dr,dz,sig_elastic,sig_ionization,sig_excitation,dt_transport);
    
    length(electron_position(:,1))
    if mod(time,6)==5
        1;
    end

end

